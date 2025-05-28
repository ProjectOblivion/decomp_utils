import re
import json
import decomp_utils.yaml_ext as yaml
from pathlib import Path
from collections import deque, defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
from .symbols import Symbol, get_symbol_offset, add_symbols
from .asm_compare import group_by_hash, get_buckets, find_matches, group_results
from .helpers import get_logger
from box import Box

__all__ = [
    "find_segments",
    "find_symbols",
    "rename_symbols",
    "get_default",
    "parse_psp_stage_init",
    "parse_psx_header",
    "parse_init_room_entities",
    "parse_export_table",
    "parse_entity_table",
]


# Todo: Move all regex patterns to a common function
# Todo: Review all str.find() instances for slicing vs start and end as parameters
# Todo: Review glob usage
def find_segments(ovl_config):
    logger = get_logger()
    # Todo: Move this data structure to a more dynamic implementation
    known_files = [
        Box(
            name="e_red_door",
            start="EntityIsNearPlayer",
            end=f"{ovl_config.name.upper}_EntityRedDoor",
            default_box=True,
        ),
        Box(
            name="st_update",
            start="Random",
            end="UpdateStageEntities",
            default_box=True,
        ),
        Box(
            name="st_collision",
            start="HitDetection",
            end="EntityDamageDisplay",
            default_box=True,
        ),
        Box(
            name="create_entity",
            start="CreateEntityFromLayout",
            end="CreateEntityFromEntity",
            default_box=True,
        ),
        Box(
            name="st_init", start="GetLangAt", end="func_psp_09254120", default_box=True
        ),
        Box(
            name="st_common",
            start="DestroyEntity",
            end="ReplaceBreakableWithItemDrop",
            default_box=True,
        ),
        Box(
            name="blit_char",
            start=f'{"func_psp_0923C2F8" if ovl_config.version == "pspeu" else "BlitChar"}',
            end="BlitChar",
            default_box=True,
        ),
        Box(
            name="e_misc",
            start="CheckColliderOffsets",
            end="PlaySfxPositional",
            default_box=True,
        ),
        Box(
            name="e_misc_2",
            start="EntityHeartDrop",
            end="EntityMessageBox",
            default_box=True,
        ),
        Box(
            name="e_stage_name",
            start=f'{"func_psp_0923C0C0" if ovl_config.version == "pspeu" else "StageNamePopupHelper"}',
            end="EntityStageNamePopup",
            default_box=True,
        ),
        Box(
            name="e_particles",
            start=f'{"func_psp_0923AD68" if ovl_config.version == "pspeu" else "EntitySoulStealOrb"}',
            end="EntityEnemyBlood",
            default_box=True,
        ),
        Box(
            name="e_collect",
            start="PrizeDropFall",
            end="EntityRelicOrb",
            default_box=True,
        ),
        Box(
            name="e_room_fg",
            start="EntityRoomForeground",
            end="EntityRoomForeground",
            default_box=True,
        ),
        Box(
            name="e_popup",
            start="BottomCornerText",
            end="BottomCornerText",
            default_box=True,
        ),
        Box(
            name="prim_helpers",
            start="UnkPrimHelper",
            end="PrimDecreaseBrightness",
            default_box=True,
        ),
        Box(
            name="e_axe_knight",
            start="AxeKnightUnkFunc1",
            end="EntityAxeKnightThrowingAxe",
            default_box=True,
        ),
        Box(
            name="e_skeleton",
            start="SkeletonAttackCheck",
            end="UnusedSkeletonEntity",
            default_box=True,
        ),
        Box(
            name="e_fire_warg",
            start="func_801CC5A4",
            end="EntityFireWargDeathBeams",
            default_box=True,
        ),
        Box(
            name="e_warg",
            start="func_801CF438",
            end="EntityWargExplosionPuffTransparent",
            default_box=True,
        ),
        Box(
            name="st_debug",
            start=f"{ovl_config.name.upper()}_EntityBackgroundBlock",
            end=f"{ovl_config.name.upper()}_EntityLockCamera",
            default_box=True,
        ),
        Box(
            name="e_venus_weed",
            start="SetupPrimsForEntitySpriteParts",
            end="EntityVenusWeedSpike",
            default_box=True,
        ),
        Box(
            name="water_effects",
            start="func_801C4144",
            end="EntityWaterDrop",
            default_box=True,
        ),
        Box(
            name="e_breakable",
            start="EntityBreakable",
            end="EntityBreakable",
            default_box=True,
        ),
        Box(
            name="e_jewel_sword_puzzle",
            start="EntityMermanRockLeftSide",
            end="EntityFallingRock2",
            default_box=True,
        ),
        Box(
            name="e_castle_door",
            start="EntityCastleDoor",
            end="EntityCastleDoor",
            default_box=True,
        ),
        Box(
            name="e_background_bushes_trees",
            start="EntityBackgroundBushes",
            end="EntityBackgroundTrees",
            default_box=True,
        ),
        Box(
            name="e_sky_entities",
            start="EntityLightningThunder",
            end="EntityLightningCloud",
            default_box=True,
        ),
        Box(
            name="e_trapdoor",
            start="EntityTrapDoor",
            end="EntityTrapDoor",
            default_box=True,
        ),
        Box(
            name="entrance_weights",
            start="UpdateWeightChains",
            end="EntityPathBlockTallWeight",
            default_box=True,
        ),
        Box(
            name="e_heartroom",
            start="EntityHeartRoomSwitch",
            end="EntityHeartRoomGoldDoor",
            default_box=True,
        ),
        Box(
            name="e_cavern_door",
            start="DoorCascadePhysics",
            end="EntityCavernDoor",
            default_box=True,
        ),
        Box(
            name="e_stairway",
            start="EntityStairwayPiece",
            end="EntityFallingRock",
            default_box=True,
        ),
    ]

    segments = []

    # Todo: Combine include patterns and move these to a more global space
    include_asm_pattern = re.compile(r'INCLUDE_ASM\("([A-Za-z0-9/_]+)",\s?(\w+)\);')
    include_rodata_pattern = re.compile(r'INCLUDE_RODATA\("[A-Za-z0-9/_]+",\s?(\w+)\);')
    rodata_pattern = re.compile(
        rf"glabel (?:jtbl|D)_{ovl_config.version}_[0-9A-F]{{8}}\n\s+/\*\s([0-9A-F]{{1,5}})\s"
    )
    known_starts = {x.start: x for x in known_files}

    src_text = ovl_config.first_src_file.read_text()

    segment_meta = None
    functions = deque()
    for match in include_asm_pattern.finditer(src_text):
        asm_dir, current_function = match.groups()

        if current_function in known_starts:
            if segment_meta:
                if len(functions) == 1:
                    segment_meta.name = f'{ovl_config.segment_prefix}{re.sub(r"([A-Za-z])([A-Z][a-z])", r"\1_\2", functions[0]).lower().replace("entity", "e")}'
                segment_meta.end = functions[-1]
                logger.debug(
                    f"Found text segment for {segment_meta.name} at 0x{segment_meta.offset.str}"
                )
                segments.append(segment_meta)
                functions.clear()
                segment_meta = None

            segment_meta = known_starts[current_function]
            if ovl_config.version == "pspeu":
                segment_meta.name = f"{ovl_config.segment_prefix}{segment_meta.name}"
            segment_meta.asm_dir = asm_dir
        elif not segment_meta:
            segment_meta = Box(
                start=current_function, end=None, asm_dir=asm_dir, default_box=True
            )

        if segment_meta and not segment_meta.offset:
            if offset := get_symbol_offset(ovl_config, current_function):
                segment_meta.offset.int = offset
                segment_meta.offset.str = f"{segment_meta.offset.int:X}"
            else:
                asm_path = (
                    Path("asm") / ovl_config.version / asm_dir / f"{current_function}.s"
                )
                asm_text = asm_path.read_text()
                if first_offset := re.search(
                    rf"glabel {current_function}\s+/\*\s([0-9A-F]{{1,5}})\s", asm_text
                ):
                    segment_meta.offset.str = first_offset.group(1)
                    segment_meta.offset.int = int(segment_meta.offset.str, 16)
        if not segment_meta.name and segment_meta.offset:
            segment_meta.name = (
                f"{ovl_config.segment_prefix}unk_{segment_meta.offset.str}"
            )

        if segment_meta and current_function == segment_meta.end:
            logger.debug(
                f"Found text segment for {segment_meta.name} at 0x{segment_meta.offset.str}"
            )
            segments.append(segment_meta)
            functions.clear()
            segment_meta = None
        else:
            functions.append(current_function)

    if segment_meta and segment_meta not in segments:
        # Todo: Handle this without duplicating the code from the loop, if possible
        if len(functions) == 1:
            segment_meta.name = f'{ovl_config.segment_prefix}{re.sub(r"([A-Za-z])([A-Z][a-z])", r"\1_\2", functions[0]).lower().replace("entity", "e")}'
        logger.debug(
            f"Found text segment for {segment_meta.name} at 0x{segment_meta.offset.str}"
        )
        segments.append(segment_meta)

    segments = tuple(segments)

    rodata_subsegments = [
        subseg
        for subseg in ovl_config.subsegments
        if len(subseg) >= 2
        and "rodata" in subseg[1]
        and ovl_config.first_src_file.stem not in subseg[2]
    ]

    file_header = get_default("file_header").format(
        ovl_header_path=(
            f"../{ovl_config.name}/{ovl_config.name}.h"
            if ovl_config.platform == "psp"
            else f"{ovl_config.name}.h"
        )
    )
    for segment in segments:
        # Todo: Decide if these need to be more flexible about the existence of the leading space
        first_function_index = src_text.find(f" {segment.start});")
        last_function_index = src_text.find(f" {segment.end});")
        segment_start = src_text[:first_function_index].rfind("INCLUDE_ASM")
        if (segment_end := src_text[last_function_index:].find("INCLUDE_ASM")) == -1:
            segment_end = len(src_text)
        else:
            segment_end += last_function_index
        segment_text = (
            src_text[segment_start:segment_end]
            .replace(
                f"{ovl_config.nonmatchings_path}/{ovl_config.segment_prefix}{ovl_config.first_src_file.stem}",
                f"{ovl_config.nonmatchings_path}/{segment.name}",
            )
            .rstrip("\n")
        )

        # Extract rodata symbols from INCLUDE_RODATA macros
        for rodata_symbol in include_rodata_pattern.findall(segment_text):
            rodata_offset = get_symbol_offset(ovl_config, rodata_symbol)
            rodata_subsegments.append(
                Box(offset=rodata_offset, type=".rodata", name=segment.name)
            )

        # Extract rodata offsets from assembly files referenced in INCLUDE_ASM macros
        asm_files = [
            ovl_config.asm_path.joinpath(
                ovl_config.nonmatchings_path,
                ovl_config.segment_prefix,
                ovl_config.first_src_file.stem,
                match.group(2),
            ).with_suffix(".s")
            for match in include_asm_pattern.finditer(segment_text)
        ]

        for asm_file in asm_files:
            asm_text = asm_file.read_text()
            rodata_start = asm_text.find(".section .rodata")
            text_start = asm_text.find(".section .text")
            if rodata_start != -1 and text_start > rodata_start:
                rodata_text = (
                    asm_text[rodata_start:text_start]
                    if text_start > rodata_start
                    else asm_text[rodata_start:]
                )
                for rodata_offset in rodata_pattern.findall(rodata_text):
                    rodata_subsegments.append(
                        Box(
                            offset=int(rodata_offset, 16),
                            type=".rodata",
                            name=segment.name,
                        )
                    )

        ovl_config.src_path.joinpath(segment.name).with_suffix(".c").write_text(
            file_header + segment_text + "\n"
        )

    rodata_by_segment = {}
    for rodata_subsegment in rodata_subsegments:
        if (
            rodata_subsegment.name not in rodata_by_segment
            or rodata_subsegment.offset < rodata_by_segment[rodata_subsegment.name][0]
        ):
            rodata_by_segment[rodata_subsegment.name] = [
                rodata_subsegment.offset,
                rodata_subsegment.type,
                rodata_subsegment.name,
            ]

    ovl_config.first_src_file.unlink()

    # Todo: Add these segments as comments.  bss segments can't be split until their files are fully imported.
    """first_bss_index = next(i for i,subseg in enumerate(ovl_config.subsegments) if "bss" in subseg or "sbss" in subseg)
    bss_subsegs = [ovl_config.subsegments[first_bss_index]] if ovl_config.subsegments[first_bss_index][0] != create_entity_bss_start else []
    bss_subsegs.extend([yaml.FlowSegment([create_entity_bss_start, ".bss" if ovl_config.platform == "psp" else ".sbss", f"{ovl_config.name}_psp/create_entity" if ovl_config.version == "pspeu" else "create_entity"]), yaml.FlowSegment([create_entity_bss_end, "bss"])])
    ovl_config.subsegments[first_bss_index:first_bss_index+1] = bss_subsegs"""

    return segments, tuple(rodata_by_segment.values())


def find_symbols(parsed, version, ovl_name, threshold=0.95):
    # Todo: Segments by op hash
    ref_funcs_by_op_hash = group_by_hash(parsed.ref_files, "op")
    check_funcs_by_op_hash = group_by_hash(parsed.check_files, "op")
    ref_ops_by_op_hash = {k: v[0].ops.parsed for k, v in ref_funcs_by_op_hash.items()}
    check_ops_by_op_hash = {
        k: v[0].ops.parsed for k, v in check_funcs_by_op_hash.items()
    }

    buckets = get_buckets(
        (ref_ops_by_op_hash, check_ops_by_op_hash), num_buckets=20, tolerance=0.1
    )

    args = (
        {
            "ref_ops_by_op_hash": bucket[0],
            "check_ops_by_op_hash": bucket[1],
            "threshold": threshold,
        }
        for bucket in buckets
    )
    with ProcessPoolExecutor() as executor:
        results = executor.map(find_matches, args)

    matches = set()
    for ref_op_hash, results in group_results(results).items():
        ref_paths = tuple(x.path for x in ref_funcs_by_op_hash[ref_op_hash])
        check_op_hash, score, _ = results[0]
        check_paths = tuple(x.path for x in check_funcs_by_op_hash[check_op_hash])
        if ref_paths and check_paths:
            ref_names = tuple(
                re.sub(r"^[A-Z0-9]{3,4}_(\w+)", rf"{ovl_name.upper()}_\1", func.stem)
                for func in ref_paths
            )
            check_names = tuple(func.stem for func in check_paths)
            matches.add((ref_paths, ref_names, check_paths, check_names))
    matches = tuple(
        Box(
            {
                "ref": {
                    "paths": ref_paths,
                    "names": {
                        "all": tuple(set(ref_names)),
                        "no_defaults": tuple(
                            {
                                name
                                for name in ref_names
                                if not name.startswith(f"func_{version}")
                            }
                        ),
                    },
                    "counts": {
                        "all": Counter(ref_names).most_common(),
                        "no_defaults": Counter(
                            tuple(
                                name
                                for name in ref_names
                                if not name.startswith(f"func_{version}")
                            )
                        ).most_common(),
                    },
                },
                "check": {"paths": check_paths, "names": check_names},
                "score": score,
            }
        )
        for ref_paths, ref_names, check_paths, check_names in matches
    )
    return matches


def rename_symbols(ovl_config, matches):
    logger = get_logger()
    known_pairs = (
        Box(first="func_801CC5A4", last="func_801CF438"),
        Box(first="func_801CC90C", last="func_801CF6D8"),
        Box(first="EntityIsNearPlayer", last="MagicallySealedDoorIsNearPlayer"),
        Box(first="GetAnglePointToEntityShifted", last="GetAnglePointToEntity"),
        Box(
            first="CreateEntityWhenInVerticalRange",
            last="CreateEntityWhenInHorizontalRange",
        ),
        Box(first="FindFirstEntityToTheRight", last="FindFirstEntityAbove"),
        Box(first="FindFirstEntityToTheLeft", last="FindFirstEntityBelow"),
        Box(first="CreateEntitiesToTheRight", last="CreateEntitiesAbove"),
        Box(first="CreateEntitiesToTheLeft", last="CreateEntitiesBelow"),
    )
    symbols = defaultdict(list)
    for match in matches:
        for pair in known_pairs:
            if (
                len(match.ref.names.no_defaults) == 2
                and len(match.check.names) <= 2
                and pair.first in match.ref.names.no_defaults
                and pair.last in match.ref.names.no_defaults
            ):
                offset = min(
                    tuple(int(x.split("_")[-1], 16) for x in match.check.names)
                )
                symbols[pair.first].append(Symbol(pair.first, offset))
                if len(match.check.names) == 2:
                    offset = max(
                        tuple(int(x.split("_")[-1], 16) for x in match.check.names)
                    )
                    symbols[pair.last].append(Symbol(pair.last, offset))
                break
        else:
            if len(match.check.names) == 1 and match.check.names[0].startswith(
                f"func_{ovl_config.version}_"
            ):
                if (
                    len(match.ref.counts.no_defaults) > 1
                    and match.ref.counts.no_defaults[0][1]
                    == match.ref.counts.no_defaults[1][1]
                ):
                    logger.warning(
                        f"Ambiguous match for {match.ref.counts.no_defaults}"
                    )
                offset = int(match.check.names[0].split("_")[-1], 16)
                if match.ref.names.no_defaults:
                    symbols[match.ref.counts.no_defaults[0][0]].append(
                        Symbol(match.ref.counts.no_defaults[0][0], offset)
                    )
            # This is only a stopgap until MagicallySealedDoorIsNearPlayer is decompiled for psp
            elif (
                ovl_config.version == "pspeu"
                and len(match.ref.names.no_defaults) == 1
                and len(match.check.names) == 2
                and "EntityIsNearPlayer" in match.ref.names.no_defaults
            ):
                offset = min(
                    tuple(int(x.split("_")[-1], 16) for x in match.check.names)
                )
                symbols["EntityIsNearPlayer"].append(
                    Symbol("EntityIsNearPlayer", offset)
                )
                offset = max(
                    tuple(int(x.split("_")[-1], 16) for x in match.check.names)
                )
                symbols["MagicallySealedDoorIsNearPlayer"].append(
                    Symbol("MagicallySealedDoorIsNearPlayer", offset)
                )
            elif match.ref.names.no_defaults != match.check.names:
                # Todo: Convert to proper log entry format
                logger.warning(
                    json.dumps(
                        {
                            "ref_funcs": match.ref.counts.no_defaults,
                            "check_funcs": match.check.names,
                            "score": match.score,
                        }
                    )
                )

    if not symbols:
        logger.warning("\nNo new symbols found\n")
        # Todo: Where should this continue on if no new symbols are found?
        exit()
    # Todo: Figure out a better way to handle multiple functions mapping to multiple functions with the same name
    add_symbols(
        ovl_config,
        tuple(
            sorted(syms, key=lambda x: x.address, reverse=True)[0]
            for syms in symbols.values()
        ),
    )
    return len(symbols)


def get_default(filename):
    match filename:
        case "ovl.h":
            return """// SPDX-License-Identifier: AGPL-3.0-or-later
#include "stage.h"

#define OVL_EXPORT(x) {ovl_name}_##x
 
typedef enum EntityIDs {{
    /* 0x00 */ E_NONE,
}} EntityIDs;
"""
        case "file_header":
            return """// SPDX-License-Identifier: AGPL-3.0-or-later
#include "{ovl_header_path}"

"""
        case "header.c":
            return """// SPDX-License-Identifier: AGPL-3.0-or-later
#include "{ovl_header_path}"

extern RoomHeader OVL_EXPORT(rooms)[];
extern s16** OVL_EXPORT(spriteBanks)[];
extern u_long* OVL_EXPORT(cluts)[];
extern LayoutEntity* OVL_EXPORT(pStObjLayoutHorizontal)[];
extern RoomDef OVL_EXPORT(rooms_layers)[];
extern u_long* OVL_EXPORT(gfxBanks)[];
void UpdateStageEntities();

Overlay OVL_EXPORT(Overlay) = {
    .Update = Update,
    .HitDetection = HitDetection,
    .UpdateRoomPosition = UpdateRoomPosition,
    .InitRoomEntities = InitRoomEntities,
    .rooms = OVL_EXPORT(rooms),
    .spriteBanks = OVL_EXPORT(spriteBanks),
    .cluts = OVL_EXPORT(cluts),
    .objLayoutHorizontal = OVL_EXPORT(pStObjLayoutHorizontal),
    .tileLayers = OVL_EXPORT(rooms_layers),
    .gfxBanks = OVL_EXPORT(gfxBanks),
    .UpdateStageEntities = UpdateStageEntities,
// RBO5,TOP,BO6
//    .unk2C = ,
//    .unk30 = ,
// RBO6,RNO4,RBO1,NZ1,RNZ0,RCEN,BO7,RNZ1,RTOP
//    .unk34 = ,
//    .unk38 = ,
//    .StageEndCutScene = ,
};

// #include "gen/sprite_banks.h"
// #include "gen/palette_def.h"
// #include "gen/layers.h"
// #include "gen/graphics_banks.h"
"""


def parse_psp_stage_init(asm_path):
    stage_init_file, export_table_symbol, entity_table_symbol = None, None, None

    for file in asm_path.rglob("*.s"):
        file_text = file.read_text()
        # Todo: Clean up the condition checks
        if (
            " 1D09043C " in file_text
            and " 38F78424 " in file_text
            and " E127240E " in file_text
            and " C708023C " in file_text
            and " 30BC43AC " in file_text
        ):
            match = re.search(
                r"""
            \s+/\*\s[A-F0-9]{1,5}(?:\s[A-F0-9]{8}){2}\s\*/\s+lui\s+\$v1,\s+%hi\((?P<entity>[A-Za-z0-9_]+)\)\n
            .*\n
            \s+/\*\s[A-F0-9]{1,5}\s[A-F0-9]{8}\sC708023C\s\*/.*\n
            \s+/\*\s[A-F0-9]{1,5}\s[A-F0-9]{8}\s30BC43AC\s\*/.*\n
            (?:.*\n)+
            \s+/\*\s[A-F0-9]{1,5}\s[A-F0-9]{8}\s1D09043C\s\*/.*\n
            \s+/\*\s[A-F0-9]{1,5}\s[A-F0-9]{8}\s38F78424\s\*/.*\n
            \s+/\*\s[A-F0-9]{1,5}(?:\s[A-F0-9]{8}){2}\s\*/\s+lui\s+\$a1,\s+%hi\((?P<export>[A-Za-z0-9_]+)\)\n
            (?:.*\n){2}
            \s+/\*\s[A-F0-9]{1,5}\s[A-F0-9]{8}\sE127240E\s\*/.*\n
            """,
                file_text,
                re.VERBOSE,
            )
            if match:
                return file.stem, match.group("export"), match.group("entity")
        # I'm pretty sure all of these are found in the same file, but keeping this here just in case
        else:
            if (
                " 1D09043C " in file_text
                and " 38F78424 " in file_text
                and " E127240E " in file_text
            ):
                match = re.search(
                    r"""
                    \s+/\*\s[A-F0-9]{1,5}\s[A-F0-9]{8}\s1D09043C\s\*/.*\n
                    \s+/\*\s[A-F0-9]{1,5}\s[A-F0-9]{8}\s38F78424\s\*/.*\n
                    \s+/\*\s[A-F0-9]{1,5}(?:\s[A-F0-9]{8}){2}\s\*/\s+lui\s+\$a1,\s+%hi\((?P<export>[A-Za-z0-9_]+)\)\n
                    (?:.*\n){2}
                    \s+/\*\s[A-F0-9]{1,5}\s[A-F0-9]{8}\sE127240E\s\*/.*\n
                    """,
                    file_text,
                    re.VERBOSE,
                )
                if match:
                    stage_init_file = file.stem
                    export_table_symbol = match.group("export")
            if " C708023C " in file_text and " 30BC43AC " in file_text:
                match = re.search(
                    r"""
                    \s+/\*\s[A-F0-9]{1,5}(?:\s[A-F0-9]{8}){2}\s\*/\s+lui\s+\$v1,\s+%hi\((?P<entity>[A-Za-z0-9_]+)\)\n
                    .*\n
                    \s+/\*\s[A-F0-9]{1,5}\s[A-F0-9]{8}\sC708023C\s\*/.*\n
                    \s+/\*\s[A-F0-9]{1,5}\s[A-F0-9]{8}\s30BC43AC\s\*/.*\n
                    """,
                    file_text,
                    re.VERBOSE,
                )
                if match:
                    entity_table_symbol = match.group("entity")
            if stage_init_file and export_table_symbol and entity_table_symbol:
                return file.stem, export_table_symbol, entity_table_symbol
    else:
        return stage_init_file, export_table_symbol, entity_table_symbol


def parse_psx_header(ovl_name, data_file_text):
    # Account for both Abbreviated and full headers
    # Account for difference in stage headers vs other headers
    psx_header = [
        "Update",
        "HitDetection",
        "UpdateRoomPosition",
        "InitRoomEntities",
        f"{ovl_name.upper()}_rooms",
        f"{ovl_name.upper()}_spriteBanks",
        f"{ovl_name.upper()}_cluts",
        f"{ovl_name.upper()}_pStObjLayoutHorizontal",
        f"{ovl_name.upper()}_rooms_layers",
        f"{ovl_name.upper()}_gfxBanks",
        "UpdateStageEntities",
#        "unk2C",
#        "unk30",
#        "unk34",
#        "unk38",
#        "StageEndCutScene",
    ]

    header_start = data_file_text.find("glabel ")
    header_end = data_file_text.find(".size ")
    header = data_file_text[header_start:header_end]
    matches = re.findall(
        r"/\*\s[0-9A-F]{1,5}\s[0-9A-F]{8}\s([0-9A-F]{8})\s\*/\s+\.word\s+\w+", header
    )
    if matches:
        if len(matches) > 7:
            pStObjLayoutHorizontal_address = int.from_bytes(
                bytes.fromhex(matches[7]), "little"
            )
        else:
            pStObjLayoutHorizontal_address = None
        symbols = tuple(
            Symbol(name, int.from_bytes(bytes.fromhex(address), "little"))
            for name, address in zip(psx_header, matches)
        )
        return pStObjLayoutHorizontal_address, symbols
    else:
        return None


def parse_init_room_entities(ovl_name, platform, init_room_entities_path):
    symbol_pattern = re.compile(
        r"\s+/\*\s[0-9A-F]{1,5}\s[0-9A-F]{8}\s[0-9A-F]{8}\s\*/\s+[a-z]{1,5}[ \t]*\$\w+,\s%hi\(D_(?:\w+_)?([A-F0-9]{8})\)\s*"
    )
    init_room_entities_map = {
        f"{ovl_name.upper()}_pStObjLayoutHorizontal": 14 if platform == "psp" else 9,
        f"{ovl_name.upper()}_pStObjLayoutVertical": 22 if platform == "psp" else 12,
        "g_LayoutObjHorizontal": 18 if platform == "psp" else 17,
        "g_LayoutObjVertical": 26 if platform == "psp" else 19,
        "g_LayoutObjPosHorizontal": 121 if platform == "psp" else 81,
        "g_LayoutObjPosVertical": 123 if platform == "psp" else 83,
    }
    lines = init_room_entities_path.read_text().splitlines()
    symbols = tuple(
        Symbol(name, int(symbol_pattern.fullmatch(lines[i]).group(1), 16))
        for name, i in init_room_entities_map.items()
        if "(D_" in lines[i]
    )
    create_entity_bss_address = min(
        x.address for x in symbols if x.name.startswith("g_Layout")
    )

    return symbols, create_entity_bss_address


def parse_export_table(ovl_type, export_table_symbol, data_file_text):
    match ovl_type:
        case "stage":
            export_table = [
                "Update",
                "HitDetection",
                "UpdateRoomPosition",
                "InitRoomEntities",
                "g_Rooms",
                "g_SpriteBanks",
                "g_Cluts",
                "g_pStObjLayoutHorizontal",
                "g_TileLayers",
                "g_EntityGfxs",
                "UpdateStageEntities",
                "g_SpriteBank1",
                "g_SpriteBank2",
                "unk34",
                "unk38",
                "unk3C",
            ]
        # Todo: Handle servant export
        case "servant":
            # Todo: Better names
            # This was all commented out originally
            export_table = [
                "Init",
                "Update",
                "Unk08",
                "Unk0C",
                "Unk10",
                "Unk14",
                "Unk18",
                "Unk1C",
                "Unk20",
                "Unk24",
                "Unk28",
                "Unk2C",
                "Unk30",
                "Unk34",
                "Unk38",
                "Unk3C",
            ]
        # Todo: Handle weapon export
        case "weapon":
            export_table = [
                "EntityWeaponAttack",
                "func_ptr_80170004",
                "func_ptr_80170008",
                "func_ptr_8017000C",
                "func_ptr_80170010",
                "func_ptr_80170014",
                "GetWeaponId",
                "LoadWeaponPalette",
                "EntityWeaponShieldSpell",
                "func_ptr_80170024",
                "func_ptr_80170028",
                "WeaponUnused2C",
                "WeaponUnused30",
                "WeaponUnused34",
                "WeaponUnused38",
                "WeaponUnused3C",
            ]
        case _:
            return None

    export_table_start = data_file_text.find(f"glabel {export_table_symbol}")
    export_table_end = data_file_text.find(f".size {export_table_symbol}")
    if export_table_start != -1:
        matches = re.finditer(
            r"/\*\s[0-9A-F]{1,5}\s[0-9A-F]{8}\s(?P<address>[0-9A-F]{8})\s\*/\s+\.word\s+func_\w+",
            data_file_text[export_table_start:export_table_end],
        )

    if matches:
        return tuple(
            Symbol(
                name, int.from_bytes(bytes.fromhex(match.group("address")), "little")
            )
            for name, match in zip(export_table, matches)
        )
    else:
        return None


def parse_entity_table(ovl_name, entity_table_symbol, data_file_text):
    entity_table = [
        "EntityUnkBreakable",
        "EntityExplosion",
        "EntityPrizeDrop",
        "EntityDamageDisplay",
        f"{ovl_name.upper()}_EntityRedDoor",
        "EntityIntenseExplosion",
        "EntitySoulStealOrb",
        "EntityRoomForeground",
        "EntityStageNamePopup",
        "EntityEquipItemDrop",
        "EntityRelicOrb",
        "EntityHeartDrop",
        "EntityEnemyBlood",
        "EntityMessageBox",
        "EntityDummy",
        "EntityDummy",
        f"{ovl_name.upper()}_EntityBackgroundBlock",
        f"{ovl_name.upper()}_EntityLockCamera",
        "EntityUnkId13",
        "EntityExplosionVariants",
        "EntityGreyPuff",
    ]

    entity_table_start = data_file_text.find(f"glabel {entity_table_symbol}")
    entity_table_end = data_file_text.find(f".size {entity_table_symbol}")
    if entity_table_start != -1:
        parsed_entity_table = data_file_text[entity_table_start:entity_table_end]
        for line in parsed_entity_table.splitlines():
            if " func" in line or " Entity" in line:
                entity_table_address = int(line.split()[2], 16)
                break
            else:
                entity_table_address = None

        matches = re.findall(
            r"/\*\s[0-9A-F]{1,5}\s[0-9A-F]{8}\s([0-9A-F]{8})\s\*/\s+\.word\s+func_\w+",
            parsed_entity_table,
        )

    if matches:
        # Do not rename to EntityDummy if the two addresses don't match
        if len(matches) > 14 and matches[14] != matches[15]:
            entity_table[14:15] = [f"EntityUnk{matches[14]}", f"EntityUnk{matches[15]}"]
        symbols = tuple(
            Symbol(name, int.from_bytes(bytes.fromhex(address), "little"))
            for name, address in zip(entity_table, matches)
        )
    else:
        symbols = None

    return entity_table_address, symbols
