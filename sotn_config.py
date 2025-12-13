#!/usr/bin/env python3

import re
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, deque, defaultdict
from pathlib import Path
from types import SimpleNamespace
from mako.template import Template
from enum import Enum
from sotn_utils.regex import RE_PATTERNS
from sotn_utils.helpers import get_logger, splat_split, shell, add_symbols, Symbol, get_symbol_address
from sotn_utils.asm_compare import group_by_hash, get_buckets, find_matches, group_results
import sotn_utils.yaml_ext as yaml

"""
Code for handling the creation of Castlevania SOTN Splat configs
"""
# Todo: Review accuracy of naming for offset and address variables
# Todo: Add symbols closer to where the address is gathered
# Todo: Extract and import BackgroundBlockInit data
# Todo: Extract and import RedDoorTiles data
# Todo: Add g_eRedDoorUV data to e_red_door
# TODO: Add error handling to functions
# TODO: Add SrcAsmPair tools/dups/src/main.rs
# TODO: Parse and add palettes to enum
"""
        SrcAsmPair {
            asm_dir: String::from("../../asm/us/st/are/matchings/"),
            src_dir: String::from("../../src/st/are/"),
            overlay_name: String::from("ARE"),
            include_asm: get_all_include_asm("../../src/st/are/"),
            path_matcher: "st/are".to_string(),
        },
"""
# TODO: Add to tools/progress.py
# progress["stare"] = DecompProgressStats("stare", "st/are")
# TODO: Identity symbol conflicts during extraction

__all__ = [
    "get_known_starts",
    "find_segments",
    "find_symbols",
    "rename_symbols",
    "parse_psp_ovl_load",
    "parse_ovl_header",
    "create_header_c",
    "parse_init_room_entities",
    "parse_entity_updates",
    "parse_e_inits",
    "create_e_init_c",
    "sort_subsegments",
    "cross_reference_e_init_c",
    "create_ovl_include",
    "find_psx_entity_updates",
    "add_initial_symbols",
    "rename_similar_functions",
]

logger = get_logger()

class EnemyDefs(Enum):
    BlueAxeKnight = 0x006
    SwordLord = 0x009
    Skelerang = 0x00B
    BloodyZombie = 0x00D
    FlyingZombieHalf1 = 0x00E
    FlyingZombieHalf2 = 0x00F
    Diplocephalus = 0x010
    OwlKnight = 0x014
    Owl = 0x016
    LesserDemon = 0x017
    MermanLvl2 = 0x01B
    MermanLvl3 = 0x01D
    Gorgon = 0x01F
    ArmorLord = 0x022
    BlackPanther = 0x025
    DarkOctopus = 0x026
    FleaMan = 0x028
    FleaArmor = 0x029
    WhiteDragon = 0x02B 
    BoneArk = 0x02D
    BoneArkSkeleton = 0x02E
    BoneArkProjectile = 0x2F
    FleaRider = 0x030
    Marionette = 0x031
    OlroxLvl25 = 0x032
    OlroxLvl0 = 0x037
    Wereskeleton = 0x03D
    Bat = 0x040
    LargeSlime = 0x041
    Slime = 0x042
    PhantomSkull = 0x043
    FlailGuard = 0x044
    BloodSkeleton = 0x046
    HellfireBeast = 0x047
    Skeleton = 0x04B
    DiscusLordLvl22 = 0x04D
    DiscusLordLvl0 = 0x04E
    FireDemon = 0x04F
    SpittleBone = 0x051
    SkeletonApe = 0x053
    StoneRose = 0x055
    Ectoplasm = 0x058
    BonePillarLvl1 = 0x05A
    SpearGuard = 0x05D
    PlateLord = 0x061
    FrozenShade = 0x063
    BoneMusket = 0x066
    DodoBird = 0x068
    BoneScimitar = 0x069
    Toad = 0x06A
    Frog = 0x06B
    BoneArcher = 0x06C
    Zombie = 0x06E
    GraveKeeper = 0x06F
    Tombstone = 0x071
    BlueRaven = 0x072
    BlackCrow = 0x073
    JackOBones = 0x074
    BoneHalberd = 0x076
    Yorick = 0x078
    Skull = 0x079
    BladeMaster = 0x07A
    BladeSoldier = 0x07C
    NovaSkeleton = 0x07E
    WingedGuard = 0x080
    SpectralSwordNO2 = 0x081
    Poltergeist = 0x082
    Lossoth = 0x083
    ValhallaKnight = 0x085
    SpectralSwordDAI = 0x088
    SpectralSwordPuppetSword = 0x089
    SpectralSwordRDAI = 0x08A
    Spear = 0x08B
    Shield = 0x08C
    Orobourous = 0x08D
    Oruburos = 0x08E
    OruburosRider = 0x08F
    DragonRider1 = 0x090
    DragonRider2 = 0x091
    Dhuron = 0x092
    FireWarg = 0x094
    WargRider = 0x097
    CaveTroll = 0x099
    Ghost = 0x09C
    Thornweed = 0x09D
    CorpseweedUnused = 0x09E
    Corpseweed = 0x09F
    VenusWeedRoot = 0x0A1
    VenusWeedFlower = 0x0A2
    BombKnight = 0x0A5
    RockKnight = 0x0A7
    DraculaLvl0 = 0x0A9
    GreaterDemon = 0x0AC
    Warg = 0x0AF
    Slinger = 0x0B2
    CornerGuard = 0x0B4
    Bitterfly = 0x0B6
    BonePillarSkull = 0x0B7
    BonePillarFireBreath = 0xB8
    BonePillarSpikedBall = 0x0B9
    Hammer = 0x0BA
    Gurkha = 0x0BC
    Blade = 0x0BE
    OuijaTable = 0x0C1
    SniperofGoth = 0x0C3
    GalamothLvl50 = 0x0C6
    GalamothLvl0 = 0x0C7
    Minotaurus = 0x0CB
    WerewolfARE = 0x0CE
    Paranthropus = 0x0D3
    Mudman = 0x0D6
    GhostDancer = 0x0D8
    FrozenHalf = 0x0D9
    SalemWitch = 0x0DD
    Azaghal = 0x0E0
    Gremlin = 0x0E1
    HuntingGirl = 0x0E3
    VandalSword = 0x0E4
    Salome = 0x0E5
    Ctulhu = 0x0E9
    Malachi = 0x0EC
    Harpy = 0x0EF
    Slogra = 0x0F3
    GreenAxeKnight = 0x0F6
    Spellbook = 0x0F7
    MagicTomeLvl8 = 0x0F9
    MagicTomeLvl12 = 0x0FB
    Doppleganger10 = 0x0FD
    Gaibon = 0x0FE
    SkullLord = 0x105
    Lion = 0x106
    Tinman = 0x108
    AkmodanII = 0x10B
    Cloakedknight = 0x10F
    DarkwingBat = 0x111
    Fishhead = 0x115
    Karasuman = 0x118
    Imp = 0x11C
    Balloonpod = 0x11D
    Scylla = 0x11F
    Scyllawyrm = 0x126
    Granfaloon1 = 0x127
    Granfaloon2 = 0x128
    Hippogryph = 0x12C
    MedusaHead1 = 0x12F
    MedusaHead2 = 0x130
    Archer = 0x131
    RichterBelmont = 0x133
    Scarecrow = 0x142
    Schmoo = 0x143
    Beezelbub = 0x144
    FakeTrevor = 0x148
    FakeGrant = 0x14E
    FakeSypha = 0x151
    Succubus = 0x156
    KillerFish = 0x15E
    Shaft = 0x15F
    Death1 = 0x164
    Death2 = 0x169
    Cerberos = 0x16B
    Medusa = 0x16E
    TheCreature = 0x172
    Doppleganger40 = 0x174
    DraculaLvl98 = 0x17B
    StoneSkull = 0x180
    Minotaur = 0x182
    WerewolfRARE = 0x185
    BlueVenusWeed1 = 0x188
    BlueVenusWeed2 = 0x189
    Guardian = 0x18C

def create_ovl_include(entity_updates, ovl_name, ovl_type, ovl_include_path):
    entity_funcs = []
    if entity_updates:
        for i, func in enumerate([symbol.name for symbol in entity_updates]):
            if func == "EntityDummy":
                entity_funcs.append((func, f"E_DUMMY_{i+1:X}"))
            elif func.startswith("Entity") or func.startswith("OVL_EXPORT(Entity"):
                entity_funcs.append(
                    (
                        func,
                        RE_PATTERNS.camel_case.sub(
                            r"\1_\2", func.replace("OVL_EXPORT(", "").replace(")", "")
                        )
                        .upper()
                        .replace("ENTITY", "E"),
                    )
                )
            elif func == "0x00000000":
                entity_funcs.append((func, f"NULL"))
            else:
                entity_funcs.append((func, f"E_UNK_{i+1:X}"))

    template = Template((Path(__file__).parent / "templates" / "ovl.h.mako").read_text())
    ovl_header_text = template.render(
        ovl_name=ovl_name,
        ovl_type=ovl_type,
        entity_updates=entity_funcs,
    )

    if not ovl_include_path.exists():
        ovl_include_path.parent.mkdir(parents=True, exist_ok=True)
        ovl_include_path.write_text(ovl_header_text)
    elif entity_funcs and "Entities" not in ovl_include_path.read_text():
        ovl_include_path.write_text(ovl_header_text)


def find_psx_entity_updates(first_data_text, pStObjLayoutHorizontal_address = None):
    # TODO: Find a less complicated way to handle this
    # we know that the entity table is always after the ovl header
    end_of_header = first_data_text.find(".size")
    # use the address of pStObjLayoutHorizontal if it was parsed from the header to reduce the amount of data we're searching through
    if pStObjLayoutHorizontal_address:
        start_index = first_data_text.find(
            f"{pStObjLayoutHorizontal_address:08X}", end_of_header
        )
    else:
        logger.warning("No address found for pStObjLayoutHorizontal, starting at end of header")
        start_index = end_of_header

    # the first entity referenced after the ovl header, which should be the first element of the entity table
    first_entity_index = first_data_text.find(
        " func_", start_index
    )
    # the last glabel before the first function pointer should be the entity table symbol
    entity_updates_index = first_data_text.rfind(
        "glabel", start_index, first_entity_index
    )
    # this is just a convoluted way of extracting the entity table symbol name
    # get the second word of the first line, which should be the entity table symbol name
    return {"name": first_data_text[entity_updates_index:first_entity_index].splitlines()[0].split()[1]}

# TODO: Use mipsmatch to supplement segments.yaml
# TODO: mipsmatch scan some.yaml another.yaml evenmore.yaml import.bin for the bin you're importing
def get_known_starts(
    ovl_name, version, segments_path=Path(__file__).parent / "segments.yaml"
):
    segments_config = yaml.safe_load(segments_path.read_text())
    known_segments = []
    # Todo: Simplify this logic
    for label, values in segments_config.items():
        if not values or "functions" not in values:
            continue

        if "ovl" not in values or ovl_name in values["ovl"]:
            if "start" not in values:
                starts = [values["functions"][0]]
            elif isinstance(values["start"], str):
                starts = [values["start"]]
            elif isinstance(values["start"], list):
                starts = values["start"]
            else:
                continue

            if "end" in values and isinstance(values["end"], str):
                end = values["end"]
            else:
                end = ""

            functions = {v.replace("${prefix}", ovl_name.upper()) for v in values.get("functions", [])}
            known_segments.extend(
                SimpleNamespace(
                    name=values.get("name", label).replace("${prefix}", ovl_name.upper()),
                    start=start.replace("${prefix}", ovl_name.upper()),
                    end=end.replace("${prefix}", ovl_name.upper()),
                    allow=set(starts) | functions
                )
                for start in starts
            )
    # TODO: Check if this is an issue for multiple segments with the same start
    return {x.start: x for x in known_segments}

def find_segments(ovl_config, file_header):
    # Todo: Add dynamic segment detection
    segments = []
    rodata_pattern = re.compile(
        rf"glabel (?:jtbl|D)_{ovl_config.version}" + r"_[0-9A-F]{8}\n\s+/\*\s(?P<offset>[0-9A-F]{1,5})\s"
    )
    known_starts = get_known_starts(ovl_config.name, ovl_config.version)
    src_text = ovl_config.first_src_file.read_text()

    segment_meta = None
    functions = deque()
    matches = RE_PATTERNS.include_asm.findall(src_text)
    for i, match in enumerate(matches):
        asm_dir, current_function = match
        current_function_parts = current_function.split("_")
        if current_function.startswith(f"func_{ovl_config.version}_"):
            current_function_stem = "_".join(current_function_parts[:3])
        elif current_function_parts[0] == "func":
            current_function_stem = "_".join(current_function_parts[:2])
        elif current_function_parts[0] == "GetLang":
            current_function_stem = current_function_parts[0]
        else:
            current_function_stem = current_function

        in_known_segment = bool(segment_meta and (segment_meta.end or (segment_meta.allow and current_function_stem in segment_meta.allow)))

        if (
            current_function_parts[0] == "GetLang" and matches[i + 1][1] in known_starts
        ) or (
            current_function_parts[0] != "GetLang"
            and current_function_stem in known_starts
            and not in_known_segment
            and (
                not segment_meta
                or not segment_meta.name
                or not segment_meta.name.endswith(known_starts[current_function_stem].name)
            )
        ):
            if segment_meta:
                if not segment_meta.name and len(functions) == 1:
                    segment_meta.name = f"{ovl_config.segment_prefix}{RE_PATTERNS.camel_case.sub(r'\1_\2', functions[0]).lower().replace('entity', 'e')}"
                if not functions:
                    logger.error(f"Found start function {current_function} that isn't allowed for {segment_meta.name}, this is likely an error in segments.yaml")
                segment_meta.end = functions[-1]
                logger.debug(
                    f"Found text segment for {segment_meta.name} at 0x{segment_meta.offset.str}"
                )
                segments.append(segment_meta)
                functions.clear()
                segment_meta = None

            if current_function_parts[0] == "GetLang":
                segment_meta = known_starts[matches[i + 1][1]]
                segment_meta.start = current_function
            elif current_function_stem not in known_starts:
                for num in range(len(current_function_parts), 0, -1):
                    if "_".join(current_function_parts[:num]) in known_starts:
                        segment_meta = known_starts["_".join(current_function_parts[:num])]
                        segment_meta.start = current_function
            else:
                segment_meta = known_starts[current_function_stem]
                segment_meta.start = current_function
            segment_meta.offset = None
            if ovl_config.version == "pspeu":
                segment_meta.name = f"{ovl_config.segment_prefix}{segment_meta.name}"
            segment_meta.asm_dir = asm_dir
        elif not segment_meta:
            segment_meta = SimpleNamespace(
                name=None,
                start=current_function,
                end=None,
                asm_dir=asm_dir,
                offset=None,
                allow=None,
            )

        if segment_meta and not segment_meta.offset:
            address = get_symbol_address(ovl_config.ld_script_path.with_suffix(".map"), current_function)
            if address:
                offset = address - ovl_config.vram + ovl_config.start
                segment_meta.offset = SimpleNamespace(int=offset)
                segment_meta.offset.str = f"{segment_meta.offset.int:X}"
            else:
                asm_path = (
                    Path("asm") / ovl_config.version / asm_dir / f"{current_function}.s"
                )
                asm_text = asm_path.read_text()
                if first_offset := re.search(
                    rf"glabel {current_function}" + r"\s+/\*\s(?P<offset>[0-9A-F]{1,5})\s",
                    asm_text,
                ):
                    segment_meta.offset = SimpleNamespace(str=first_offset.group(1))
                    segment_meta.offset.int = int(segment_meta.offset.str, 16)
        if not segment_meta.name and segment_meta.offset:
            segment_meta.name = (
                f"{ovl_config.segment_prefix}unk_{segment_meta.offset.str}"
            )

        if segment_meta and current_function_stem == segment_meta.end:
            logger.debug(
                f"Found text segment for {segment_meta.name} at 0x{segment_meta.offset.str}"
            )
            segments.append(segment_meta)
            functions.clear()
            segment_meta = None
        elif segment_meta and segment_meta.allow and current_function_stem not in segment_meta.allow:
            logger.debug(
                f"Found text segment for {segment_meta.name} at 0x{segment_meta.offset.str}"
            )
            if not functions:
                logger.error(f"Found start function {current_function} that isn't allowed for {segment_meta.name}, this is likely an error in segments.yaml")
            segment_meta.end = functions[-1]
            segments.append(segment_meta)
            functions.clear()
            segment_meta = SimpleNamespace(
                name=None,
                start=current_function,
                end=None,
                asm_dir=asm_dir,
                offset=None,
                allow=None,
            )
            address = get_symbol_address(ovl_config.ld_script_path.with_suffix(".map"), current_function)
            if address:
                offset = address - ovl_config.vram + ovl_config.start
                segment_meta.offset = SimpleNamespace(int=offset)
                segment_meta.offset.str = f"{segment_meta.offset.int:X}"
            else:
                asm_path = (
                    Path("asm") / ovl_config.version / asm_dir / f"{current_function}.s"
                )
                asm_text = asm_path.read_text()
                if first_offset := re.search(
                    rf"glabel {current_function}" + r"\s+/\*\s(?P<offset>[0-9A-F]{1,5})\s",
                    asm_text,
                ):
                    segment_meta.offset = SimpleNamespace(str=first_offset.group(1))
                    segment_meta.offset.int = int(segment_meta.offset.str, 16)
            functions.append(current_function)
        else:
            functions.append(current_function)

    if segment_meta and segment_meta not in segments:
        # Todo: Handle this without duplicating the code from the loop, if possible
        if not segment_meta.name and len(functions) == 1:
            # Todo: Only change name if it isn't a defined segment
            segment_meta.name = f'{ovl_config.segment_prefix}{RE_PATTERNS.camel_case.sub(r"\1_\2", functions[0]).lower().replace("entity", "e")}'
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
        for rodata_symbol in RE_PATTERNS.include_rodata.findall(segment_text):
            rodata_address = get_symbol_address(ovl_config.ld_script_path.with_suffix(".map"), rodata_symbol)
            if rodata_address:
                rodata_offset = rodata_address - ovl_config.vram + ovl_config.start
                rodata_subsegments.append(
                    SimpleNamespace(
                        offset=rodata_offset, type=".rodata", name=segment.name
                    )
                )

        # Extract rodata offsets from assembly files referenced in INCLUDE_ASM macros
        asm_files = [
            ovl_config.asm_path.joinpath(
                ovl_config.nonmatchings_path,
                ovl_config.segment_prefix,
                ovl_config.first_src_file.stem,
                match.group(2),
            ).with_suffix(".s")
            for match in RE_PATTERNS.include_asm.finditer(segment_text)
        ]

        for asm_file in asm_files:
            asm_text = asm_file.read_text()
            rodata_start = asm_text.find(".section .rodata")
            text_start = asm_text.find(".section .text")
            if rodata_start != -1 and (text_start > rodata_start or text_start == -1):
                rodata_text = (
                    asm_text[rodata_start:text_start]
                    if text_start > rodata_start
                    else asm_text[rodata_start:]
                )
                for rodata_offset in rodata_pattern.findall(rodata_text):
                    rodata_subsegments.append(
                        SimpleNamespace(
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

    # Todo: Add ability to postprocess selective segments into comments.  bss segments can't be split until their files are fully imported.
    """first_bss_index = next(i for i,subseg in enumerate(ovl_config.subsegments) if "bss" in subseg or "sbss" in subseg)
    bss_subsegs = [ovl_config.subsegments[first_bss_index]] if ovl_config.subsegments[first_bss_index][0] != create_entity_bss_start else []
    bss_subsegs.extend([yaml.FlowSegment([create_entity_bss_start, ".bss" if ovl_config.platform == "psp" else ".sbss", f"{ovl_config.name}_psp/create_entity" if ovl_config.version == "pspeu" else "create_entity"]), yaml.FlowSegment([create_entity_bss_end, "bss"])])
    ovl_config.subsegments[first_bss_index:first_bss_index+1] = bss_subsegs"""

    first_text_index = next(
        i for i, subseg in enumerate(ovl_config.subsegments) if "c" in subseg
    )
    text_subsegs = [
        yaml.FlowSegment([segment.offset.int, "c", segment.name])
        for segment in segments
    ]
    rodata_subsegs = [yaml.FlowSegment(x) for x in tuple(rodata_by_segment.values())]
    ovl_config.subsegments[first_text_index : first_text_index + 1] = text_subsegs
    first_rodata_index = next(
        (
            i
            for i, subseg in enumerate(ovl_config.subsegments)
            if ".rodata" in subseg
        ),
        None,
    )
    if first_rodata_index:
        ovl_config.subsegments[first_rodata_index : first_rodata_index + 1] = (
            rodata_subsegs
        )

    return sort_subsegments(ovl_config.subsegments)


def find_symbols(parsed_check_files, parsed_ref_files, version, ovl_name, threshold=0.95):
    # Todo: Segments by op hash
    check_funcs_by_op_hash = group_by_hash(parsed_check_files, "op")
    check_ops_by_op_hash = {
        k: v[0].ops.parsed for k, v in check_funcs_by_op_hash.items()
    }

    ref_funcs_by_op_hash = group_by_hash(parsed_ref_files, "op")
    ref_ops_by_op_hash = {k: v[0].ops.parsed for k, v in ref_funcs_by_op_hash.items()}

    buckets = get_buckets(
        (ref_ops_by_op_hash, check_ops_by_op_hash), num_buckets=20, tolerance=0.1
    )

    kwargs = (
        {
            "ref_ops_by_op_hash": bucket[0],
            "check_ops_by_op_hash": bucket[1],
            "threshold": threshold,
        }
        for bucket in buckets
    )
    # Todo: Move this to a distinct concurrency file
    with ProcessPoolExecutor() as executor:
        results = executor.map(find_matches, kwargs)

    matches = set()
    for ref_op_hash, results in group_results(results).items():
        ref_paths = tuple(x.path for x in ref_funcs_by_op_hash[ref_op_hash])
        check_op_hash, score, _ = results[0]
        check_paths = tuple(x.path for x in check_funcs_by_op_hash[check_op_hash])
        if ref_paths and check_paths:
            ref_names = tuple(
                sorted((RE_PATTERNS.symbol_ovl_name_prefix.sub(f"{ovl_name.upper()}_", f"{func.stem}_from_{func.parts[3].replace('_psp', '')}" if func.stem.startswith(f"func_{version}") and "_from_" not in func.stem else func.stem) for func in ref_paths),
                       key=lambda x: (0, x) if re.match(r"^[A-Z0-9]{3,4}", x) else (1, x) if not x.startswith("func_") else (2, x) if re.match(r"func_[0-9A-F]{8}", x) else (3, x) if x.startswith("func_us_") else (4,x))
            )
            ref_names = tuple(x.split("_")[0] if not x.startswith("func") and re.match(r"[A-Za-z0-9]+_[0-9A-F]{8}", x) else x for x in ref_names)
            check_names = tuple(func.stem for func in check_paths)
            matches.add((ref_paths, ref_names, check_paths, check_names))
    # TODO: review this logic to remove no_defaults
    matches = tuple(
        SimpleNamespace(
            ref=SimpleNamespace(
                paths=ref_paths,
                names=SimpleNamespace(
                    all=tuple(set(ref_names)),
                    no_defaults=tuple(
                        {
                            name
                            for name in ref_names
                            if not name.startswith(f"func_{version}")
                        }
                    ),
                ),
                counts=SimpleNamespace(
                    all=Counter(ref_names).most_common(),
                    no_defaults=Counter(
                        tuple(
                            name
                            for name in ref_names
                            if not name.startswith(f"func_{version}")
                        )
                    ).most_common(),
                ),
            ),
            check=SimpleNamespace(paths=check_paths, names=check_names),
            score=score,
        )
        for ref_paths, ref_names, check_paths, check_names in matches
    )
    return matches


def rename_symbols(ovl_config, matches):
    # TODO: move to yaml file
    known_pairs = [
        SimpleNamespace(first="func_801CC5A4", last="func_801CF438"),
        SimpleNamespace(first="func_801CC90C", last="func_801CF6D8"),
        SimpleNamespace(
            first="GetAnglePointToEntityShifted", last="GetAnglePointToEntity"
        ),
        SimpleNamespace(
            first="CreateEntityWhenInVerticalRange",
            last="CreateEntityWhenInHorizontalRange",
        ),
        SimpleNamespace(first="FindFirstEntityToTheRight", last="FindFirstEntityAbove"),
        SimpleNamespace(first="FindFirstEntityToTheLeft", last="FindFirstEntityBelow"),
        SimpleNamespace(first="CreateEntitiesToTheRight", last="CreateEntitiesAbove"),
        SimpleNamespace(first="CreateEntitiesToTheLeft", last="CreateEntitiesBelow"),
    ]
    # e_red_door typically comes before e_sealed door, but us rno2 and bo0 have e_sealed_door first
    if ovl_config.version == "us" and ovl_config.name in ["rno2", "bo0"]:
        known_pairs.append(SimpleNamespace(first="SealedDoorIsNearPlayer", last="EntityIsNearPlayer"))
    else:
        known_pairs.append(SimpleNamespace(first="EntityIsNearPlayer", last="SealedDoorIsNearPlayer"))

    symbols = defaultdict(list)
    unhandled, ambiguous = [], []
    # TODO: Review this logic to remove no_defaults
    for match in matches:
        for pair in known_pairs:
            if (
                len(match.ref.names.all) <= 2
                and len(match.check.names) <= 2
                and pair.first in match.ref.names.all
                and (
                    pair.last in match.ref.names.all
                    or len(match.ref.names.all) == 1
                )
            ):
                offset = min(
                    tuple(int(x.split("_")[-1], 16) for x in match.check.names)
                )
                symbols[pair.first].append(Symbol(pair.first, offset))
                if (
                    len(match.check.names) == 2
                    and pair.last in match.ref.names.all
                ):
                    offset = max(
                        tuple(int(x.split("_")[-1], 16) for x in match.check.names)
                    )
                    symbols[pair.last].append(Symbol(pair.last, offset))
                break
        else:
            new_name = match.ref.counts.all[0][0]
            if "unused" in new_name.lower():
                for name in match.check.names:
                    address = int(name.split("_")[-1], 16)
                    name = name.replace(f"func_{ovl_config.version}_", f"{ovl_config.name.upper()}_Unused")
                    symbols[name].append(
                        Symbol(name, address)
                    )
            elif len(match.check.names) == 1 and match.check.names[0].startswith(
                f"func_{ovl_config.version}_"
            ):
                if match.ref.names.all:
                    address = int(match.check.names[0].split("_")[-1], 16)
                    symbols[new_name].append(
                        Symbol(f"{ovl_config.name.upper()}_Unused_" if "unused" in new_name.lower() else new_name, address)
                    )
                if (
                    len(match.ref.counts.all) > 1
                    and match.ref.counts.all[0][1]
                    == match.ref.counts.all[1][1]
                ):
                    logger.info(f"Ambiguous match: {match.check.names[0]} renamed to {new_name} with a score of {match.score}, all matches were {', '.join([x[0] for x in match.ref.counts.all])}")
                    ambiguous.append(SimpleNamespace(old_names=[match.check.names[0]], new_names=[new_name], score=match.score, all_matches=[x[0] for x in match.ref.counts.all]))
            elif len(match.check.names) == 1 and match.check.names[0] == new_name:
                continue
            elif len(match.check.names) > 1 and len(match.ref.counts.all) == 1:
                for name in match.check.names:
                    address = int(name.split("_")[-1], 16)
                    name = name.replace(f"func_{ovl_config.version}_", f"{new_name}_")
                    symbols[name].append(
                        Symbol(name, address)
                    )
            elif match.ref.names.all != match.check.names:
                logger.info(f"Unhandled naming condition: Target name {match.ref.counts.all} for {ovl_config.name} function(s) {match.check.names} with score {match.score}")
                unhandled.append(SimpleNamespace(old_names=match.check.names, new_names=[f"{x[0]} ({x[1]})" for x in match.ref.counts.all], score=match.score, all_matches=[x[0] for x in match.ref.counts.all]))

    if symbols:
        # Todo: Figure out a better way to handle multiple functions mapping to multiple functions with the same name
        add_symbols(ovl_config.ovl_symbol_addrs_path, 
            tuple(
                sorted(syms, key=lambda x: x.address, reverse=True)[0]
                for syms in symbols.values()
            ),
            ovl_config.name,
            ovl_config.vram,
            ovl_config.symbol_name_format.replace("$VRAM", ""),
            ovl_config.src_path_full,
            ovl_config.symexport_path
        )
        return len(symbols), ambiguous, unhandled
    else:
        logger.warning("\nNo new symbols found\n")
        return 0

# Validate logic and move to sotn-decomp
def parse_psp_ovl_load(ovl_name, path_prefix, asm_path):
    first_address_pattern = re.compile(r"\s+/\*\s+[A-F0-9]{1,5}\s+([A-F0-9]{8})\s")
    ovl_load_name, ovl_load_symbol, ovl_header_name, entity_updates_name = None, None, None, None
    for file in (
        dirpath / f
        for dirpath, _, filenames in asm_path.walk()
        for f in filenames
        if ".data.s" not in f
    ):
        file_text = file.read_text()
        # Todo: Clean up the condition checks
        if (
            " 1D09043C " in file_text
            and " 38F78424 " in file_text
            and " E127240E " in file_text
            and " C708023C " in file_text
            and " 30BC43AC " in file_text
        ):
            if match := RE_PATTERNS.psp_ovl_header_entity_table_pattern.search(file_text):
                if ovl_load_address := first_address_pattern.search(file_text):
                    ovl_load_symbol = Symbol(f"{ovl_name.upper()}_Load", int(ovl_load_address.group(1), 16))
                ovl_load_name = file.stem
                ovl_header_name = match.group("header")
                entity_updates_name = match.group("entity")

    # build symexport lines, but only write if needed
    template = Template(
        (Path(__file__).parent / "templates" / "symexport.txt.mako").read_text()
    )
    symexport_text = template.render(
        ovl_name=ovl_name,
        path_prefix = f"{path_prefix}_" if path_prefix else "",
        ovl_load_name=ovl_load_name,
    )

    return ovl_load_symbol, ovl_header_name, {"name": entity_updates_name}, symexport_text


def parse_ovl_header(data_file_text, name, platform, header_symbol=None):
    ovl_header = [
        "Update",
        "HitDetection",
        "UpdateRoomPosition",
        "InitRoomEntities",
        f"{name.upper()}_rooms",
        f"{name.upper()}_spriteBanks",
        f"{name.upper()}_cluts",
        (
            "g_pStObjLayoutHorizontal"
            if platform == "psp"
            else f"{name.upper()}_pStObjLayoutHorizontal"
        ),
        f"{name.upper()}_rooms_layers",
        f"{name.upper()}_gfxBanks",
        "UpdateStageEntities",
        "unk2C", # g_SpriteBank1
        "unk30", # g_SpriteBank2
        "unk34",
        "unk38",
        "StageEndCutScene",
    ]
    header_start = (
        data_file_text.find(f"glabel {header_symbol}")
        if header_symbol
        else data_file_text.find("glabel ")
    )
    header_end = (
        data_file_text.find(f".size {header_symbol}")
        if header_symbol
        else data_file_text.find(".size ")
    )
    if header_start != -1:
        header = data_file_text[header_start:header_end]
        header_address = int(header.splitlines()[0].split("_")[-1], 16)
    else:
        return {}, None
    # Todo: Should this be findall or finditer?
    matches = RE_PATTERNS.symbol_line_pattern.findall(header)
    if matches:
        if len(matches) > 7:
            pStObjLayoutHorizontal_address = int.from_bytes(
                bytes.fromhex(matches[7][0]), "little"
            )
        else:
            pStObjLayoutHorizontal_address = None

        # Todo: Ensure this is doing a 1 for 1 line replacement, whether func, d_ or null
        # Todo: Make the address parsing more straight forward, instead of capturing both address and name
        header_items = tuple(
            Symbol(
                address[1] if name.startswith("unk") or (not address[1].startswith("func_") and not address[1].startswith("D_") and not address[1].startswith("g_")) else "NULL" if address[0] == "0x00000000" else name, int.from_bytes(bytes.fromhex(address[0]), "little")
            )
            # Todo: Does this need the filtering, or should it just overwrite the existing regardless?
            for name, address in zip(ovl_header, matches)
        )
        return {"address": header_address, "size_bytes": len(header_items) * 4, "symbols": tuple(symbol for symbol in header_items if symbol.address), "items": header_items}, pStObjLayoutHorizontal_address
    else:
        return {}, None

def create_header_c(header_items, ovl_name, ovl_type, version, header_path):
    header_syms = [f"{symbol.name.replace(f'{ovl_name.upper()}_', 'OVL_EXPORT(')})" if f"{ovl_name.upper()}_" in symbol.name else "NULL" if not symbol.address else symbol.name for symbol in header_items]
    common_syms = ["NULL", "Update", "HitDetection", "UpdateRoomPosition", "InitRoomEntities", "OVL_EXPORT(rooms)", "OVL_EXPORT(spriteBanks)", "OVL_EXPORT(cluts)", "OVL_EXPORT(pStObjLayoutHorizontal)", "g_pStObjLayoutHorizontal", "OVL_EXPORT(rooms_layers)", "OVL_EXPORT(gfxBanks)", "UpdateStageEntities"]
    template = Template(
        (Path(__file__).parent / "templates" / "header.c.mako").read_text()
    )
    new_header = template.render(
        ovl_include_path=f"{ovl_name}.h",
        ovl_type=ovl_type,
        header_syms=header_syms,
        common_syms=common_syms,
    )
    if header_path.is_file():
        existing_header = header_path.read_text()
        if new_header != existing_header:
            new_lines = new_header.rstrip("\n").splitlines()
            license = new_lines[0]
            include = new_lines[1]
            existing_lines = existing_header.rstrip("\n").splitlines()
            existing_lines = existing_lines[2:]
            ifdef = f"#ifdef VERSION_{'PSP' if version=='pspeu' else version.upper()}"
            new_header = f"{license}\n{include}\n{ifdef}\n{"\n".join(new_lines[2:])}\n#else\n{'\n'.join(existing_lines)}\n#endif\n"

    header_path.write_text(new_header)

def sort_subsegments(subsegments):
    # the offset is used as a key to intentionally overwrite duplicate offsets, leaving only the longest segment
    deduped_subsegments = {subsegment[0]:subsegment for subsegment in sorted(subsegments, key=lambda x: (x[0], len(x)))}
    # sort again to ensure that they're still sorted by offset after dedupe
    sorted_subsegments = sorted([subsegment for subsegment in deduped_subsegments.values()], key=lambda x: x[0])

    new_subsegments = []
    next_offset = -1
    for subsegment in sorted_subsegments:
        if next_offset == -1 or subsegment[0] == next_offset:
            new_subsegments.append(subsegment)
            next_offset = -1
        elif subsegment[0] > next_offset:
            new_subsegments.append([next_offset, "data"])
            new_subsegments.append(subsegment)
            next_offset = -1

        if len(subsegment) == 4:
            next_offset = subsegment[0] + subsegment.pop()

    return [yaml.FlowSegment(x) for x in new_subsegments]

# TODO: Validate logic and move to sotn-decomp
def parse_init_room_entities(ovl_name, platform, init_room_entities_path, vram_start):
    init_room_entities_map = {
        f"{ovl_name.upper()}_pStObjLayoutHorizontal": 14 if platform == "psp" else 9,
        f"{ovl_name.upper()}_pStObjLayoutVertical": 22 if platform == "psp" else 12,
        "g_LayoutObjHorizontal": 18 if platform == "psp" else 17,
        "g_LayoutObjVertical": 26 if platform == "psp" else 19,
        "g_LayoutObjPosHorizontal": (
            138
            if platform == "psp" and ovl_name == "rnz0"
            else 121 if platform == "psp" else 81
        ),
        "g_LayoutObjPosVertical": (
            140
            if platform == "psp" and ovl_name == "rnz0"
            else 123 if platform == "psp" else 83
        ),
    }
    lines = init_room_entities_path.read_text().splitlines()
    init_room_entities_symbols = {
        Symbol(
            name,
            int(
                RE_PATTERNS.init_room_entities_symbol_pattern.fullmatch(lines[i]).group(
                    "address"
                ),
                16,
            ),
        )
        for name, i in init_room_entities_map.items()
        if "(D_" in lines[i]
    }

    create_entity_bss_start = min(
        x.address for x in init_room_entities_symbols if x.name.startswith("g_Layout")
    ) - vram_start

    return init_room_entities_symbols, create_entity_bss_start


def parse_entity_updates(data_file_text, ovl_name, entity_updates_symbol):
    parsed_entity_updates = None
    known_entity_updates = [
        f"{ovl_name.upper()}_EntityBreakable",
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
    entity_updates_start = data_file_text.find(f"glabel {entity_updates_symbol}")
    entity_updates_end = data_file_text.find(f".size {entity_updates_symbol}")
    if entity_updates_start != -1:
        entity_updates_lines = data_file_text[
            entity_updates_start:entity_updates_end
        ].splitlines()

        first_e_init_start = data_file_text.find("glabel ", entity_updates_end)
        first_e_init_end = data_file_text.find("\n", first_e_init_start)
        first_e_init = data_file_text[first_e_init_start:first_e_init_end].split()[1]

        # if the last item is a null address, then it is padding
        if entity_updates_lines[-1].endswith("0x00000000"):
            entity_updates_lines.pop()

        table_start, entity_updates_address = next(((i, int(line.split()[2], 16)) for i, line in enumerate(entity_updates_lines) if " func" in line or " Entity" in line), (len(entity_updates_lines) - 1, None))
        entity_updates_lines = entity_updates_lines[table_start:]
        if matches := RE_PATTERNS.symbol_line_pattern.findall("\n".join(entity_updates_lines)):
            entity_dummy_address = Counter([x[0] for x in matches]).most_common(1)[0][0]
            entity_dummy_address = int.from_bytes(bytes.fromhex(entity_dummy_address), "little")
            known_symbols = tuple(
                Symbol(
                   address[1] if name == "skip" else name, int.from_bytes(bytes.fromhex(address[0]), "little")
                )
                for name, address in zip(known_entity_updates, matches)
            )
            parsed_entity_updates = known_symbols + tuple(Symbol(
                    name.split()[-1], int.from_bytes(bytes.fromhex(address[0]), "little")
                )
                for name, address in zip(entity_updates_lines[len(known_symbols):], matches[len(known_symbols):]))
            parsed_entity_updates = tuple(Symbol("EntityDummy" if symbol.address == entity_dummy_address else symbol.name, symbol.address) for symbol in parsed_entity_updates)
            symbols = tuple(symbol for symbol in parsed_entity_updates if symbol.name.split("_")[-1] != f"{symbol.address:08X}")
        else:
            symbols = tuple()
    # TODO: Why the weird + () ?
    return {"address": entity_updates_address, "first_e_init": first_e_init, "items": parsed_entity_updates, "symbols": symbols + ()}

def cross_reference_e_init_c(check_entity_updates, check_e_inits, ref_e_init_path, ovl_name, map_path):
    if ref_e_init_path.is_file():
        symbols = []
        file_text = ref_e_init_path.read_text()
        e_init_pattern = re.compile(r"""
        \nEInit\s+(?P<name>(?:OVL_EXPORT\()?\w+\)?)\s*=\s*\{(?:\s*|\n?)
        (?P<animSet>(?:ANIMSET_(?:OVL|DRA)\()?(?:0x)?[0-9A-Fa-f]{1,4}\)?)\s*
        ,\s*(?P<animCurFrame>(?:0x)?[0-9A-Fa-f]{1,4})\s*
        ,\s*(?P<unk5A>(?:0x)?[0-9A-Fa-f]{1,4})\s*
        ,\s*(?P<palette>(?:0x)?[0-9A-Fa-f]{1,4}|PAL_[A-Z0-9_]+)\s*
        ,\s*(?P<enemyID>(?:0x)?[0-9A-Fa-f]{1,4})\};
        """, re.VERBOSE)

        if check_entity_updates:
            entity_updates_start = file_text.find("EntityUpdates")
            entity_updates_end = file_text.find("};", entity_updates_start)
            ref_entity_updates = [item.strip().replace("OVL_EXPORT(", f"{ovl_name.upper()}_").rstrip(",)") for item in file_text[entity_updates_start:entity_updates_end].splitlines()[1:] if item]
            if len(check_entity_updates) == len(ref_entity_updates):
                symbols.extend(Symbol(to_name, from_symbol.address) for from_symbol, to_name in zip(check_entity_updates, ref_entity_updates))

        if check_e_inits:
            ref_e_inits = []
            e_init_idx = file_text.find("EInit")
            while e_init_idx != -1:
                if e_init := e_init_pattern.match(file_text[e_init_idx - 1:]):
                    name = e_init.group("name").replace("OVL_EXPORT(", f"{ovl_name.upper()}_").rstrip(")")
                    animSet = e_init.group("animSet")
                    animCurFrame = e_init.group("animCurFrame")
                    animCurFrame = int(animCurFrame, 16 if "0x" in animCurFrame else 10)
                    unk5A = e_init.group("unk5A")
                    unk5A = int(unk5A, 16 if "0x" in unk5A else 10)
                    palette = e_init.group("palette")
                    palette = palette if "PAL_" in palette else int(palette, 16) if "0x" in palette else int(palette)
                    enemyID = e_init.group("enemyID")
                    ref_e_inits.append((name, animSet, animCurFrame, unk5A, palette, enemyID))
                e_init_idx = file_text.find("EInit", e_init_idx + 1)

            not_matched = 0
            for i, ref_e_init in enumerate(ref_e_inits):
                if i - not_matched < len(check_e_inits):
                    if ref_e_init[1:] != check_e_inits[i - not_matched][1:]:
                        not_matched += 1
                    else:
                        symbols.append(Symbol(ref_e_init[0], get_symbol_address(map_path, check_e_inits[i - not_matched][0])))

        return symbols, len(ref_e_inits) == len(check_e_inits) + not_matched
    return [], False

def create_e_init_c(entity_updates, e_inits, ovl_name, e_init_c_path):
    if entity_updates:
        entity_funcs = [
            (
                f"{symbol.name.replace(f'{ovl_name.upper()}_','OVL_EXPORT(')})"
                if f"{ovl_name.upper()}_" in symbol.name
                else symbol.name
            )
            for symbol in entity_updates
        ]

        template = Template(
            (Path(__file__).parent / "templates" / "e_init.c.mako").read_text()
        )
        output = template.render(
            ovl_name=ovl_name,
            entity_funcs=entity_funcs,
            e_inits=e_inits,
        )
        e_init_c_path.write_text(output)
        return True
    else:
        return False

def parse_e_inits(data_file_text, first_e_init, ovl_name, platform):
    e_init_pattern = re.compile(r"""
    glabel\s+(?P<name>\w+)\n
    \s+/\*\s+(?P<offset>[0-9A-Fa-f]+)\s+(?P<address>[0-9A-Fa-f]{8})\s+[0-9A-Fa-f]{8}\s+\*/\s+\.word\s+0x(?P<animCurFrame>[0-9A-Fa-f]{4})(?P<animSet>[0-9A-Fa-f]{4})\n
    \s+/\*\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}\s+[0-9A-Fa-f]{8}\s+\*/\s+\.word\s+0x(?P<palette>[0-9A-Fa-f]{4})(?P<unk5A>[0-9A-Fa-f]{4})\n
    \s+/\*\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}\s+[0-9A-Fa-f]{8}\s+\*/\s+\.word\s+0x0000(?P<enemyID>[0-9A-Fa-f]{4})\n
    """ + (r"""
    \s+/\*\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}\s+00000000\s+\*/\s+\.word\s+0x00000000\n
    """ if platform == "psp" else "") + r"""
    (?P<size>\.size\s+(?P=name),\s+\.\s+-\s+(?P=name)\n?)?
    """, re.VERBOSE)
    unused_e_init_pattern = r"""
    \s+/\*\s+(?P<offset>[0-9A-Fa-f]+)\s+(?P<address>[0-9A-Fa-f]{8})\s+[0-9A-Fa-f]{8}\s+\*/\s+\.word\s+0x(?P<animCurFrame>[0-9A-Fa-f]{4})(?P<animSet>[0-9A-Fa-f]{4})\n
    \s+/\*\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}\s+[0-9A-Fa-f]{8}\s+\*/\s+\.word\s+0x(?P<palette>[0-9A-Fa-f]{4})(?P<unk5A>[0-9A-Fa-f]{4})\n
    \s+/\*\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}\s+[0-9A-Fa-f]{8}\s+\*/\s+\.word\s+0x0000(?P<enemyID>[0-9A-Fa-f]{4})\n
    """
    # when e_init[3] is referenced in code
    split_e_init_pattern = re.compile(r"""
    glabel\s+(?P<name>\w+)\n
    \s+/\*\s+(?P<offset>[0-9A-Fa-f]+)\s+(?P<address>[0-9A-Fa-f]{8})\s+(?P<raw_val>[0-9A-Fa-f]{8})\s+\*/\s+\.word\s+0x(?P<animCurFrame>[0-9A-Fa-f]{4})(?P<animSet>[0-9A-Fa-f]{4})\n
    \s+/\*\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}\s+\*/\s+\.short\s+0x(?P<unk5A>[0-9A-Fa-f]{4})\n
    \.size\s+(?P=name),\s+\.\s+-\s+(?P=name)\n
    \n
    glabel\s+(?P<pal_sym>\w+)\n
    \s+/\*\s+(?P<pal_offset>[0-9A-Fa-f]+)\s+(?P<pal_address>[0-9A-Fa-f]{8})\s+\*/\s+\.short\s+0x(?P<palette>[0-9A-Fa-f]{4})\n
    \s+/\*\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}\s+[0-9A-Fa-f]{8}\s+\*/\s+\.word\s+0x0000(?P<enemyID>[0-9A-Fa-f]{4})\n
    """ + (r"""
    \s+/\*\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}\s+00000000\s+\*/\s+\.word\s+0x00000000\n
    """ if platform == "psp" else "") + r"""
    (?P<size>\.size\s+(?P=pal_sym),\s+\.\s+-\s+(?P=pal_sym)\n?)
    """, re.VERBOSE)
    if platform == "psx":
        # when e_init[3] and e_init[5] are referenced in code in us
        short_e_init_pattern = re.compile(r"""
        glabel\s+(?P<name>\w+)\n
        \s+/\*\s+(?P<offset>[0-9A-Fa-f]+)\s+(?P<address>[0-9A-Fa-f]{8})\s+[0-9A-Fa-f]{8}\s+\*/\s+\.word\s+0x(?P<animCurFrame>[0-9A-Fa-f]{4})(?P<animSet>[0-9A-Fa-f]{4})\n
        \s+/\*\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}\s+\*/\s+\.short\s+0x(?P<unk5A>[0-9A-Fa-f]{4})\n
        \.size\s+(?P=name),\s+\.\s+-\s+(?P=name)\n
        \n
        glabel\s+(?P<pal_sym>\w+)\n
        \s+/\*\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}\s+\*/\s+\.short\s+0x(?P<palette>[0-9A-Fa-f]{4})\n
        \s+/\*\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}\s+\*/\s+\.short\s+0x(?P<enemyID>[0-9A-Fa-f]{4})\n
        \.size\s+(?P=pal_sym),\s+\.\s+-\s+(?P=pal_sym)\n
        \n
        glabel\s+(?P<unk_sym>\w+)\n
        \s+/\*\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}\s+\*/\s+\.short\s+0x0000\n
        (?P<size>\.size\s+(?P=unk_sym),\s+\.\s+-\s+(?P=unk_sym)\n?)
        """, re.VERBOSE)
    if platform == "psp":
        short_e_init_pattern = re.compile(r"""
        glabel\s+(?P<name>\w+)\n
        \s+/\*\s+(?P<offset>[0-9A-Fa-f]+)\s+(?P<address>[0-9A-Fa-f]{8})\s+\*/\s+\.short\s+0x(?P<animSet>[0-9A-Fa-f]{4})\n
        \s+/\*\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}\s+\*/\s+\.short\s+0x(?P<animCurFrame>[0-9A-Fa-f]{4})\n
        \s+/\*\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}\s+\*/\s+\.short\s+0x(?P<unk5A>[0-9A-Fa-f]{4})\n
        \s+/\*\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}\s+\*/\s+\.short\s+0x(?P<palette>[0-9A-Fa-f]{4})\n
        \s+/\*\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}\s+\*/\s+\.short\s+0x(?P<enemyID>[0-9A-Fa-f]{4})\n
        \s+/\*\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}\s+\*/\s+\.short\s+0x0000\n
        \s+/\*\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}\s+\*/\s+\.short\s+0x0000\n
        \s+/\*\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}\s+\*/\s+\.short\s+0x0000\n
        (?P<size>\.size\s+(?P=name),\s+\.\s+-\s+(?P=name)\n?)
        """, re.VERBOSE)

    known_e_inits = [
        f"{ovl_name.upper()}_EInitBreakable",
        "g_EInitObtainable",
        "g_EInitParticle",
        "g_EInitSpawner",
        "g_EInitInteractable",
        "g_EInitUnkId13",
        "g_EInitLockCamera",
        "g_EInitCommon",
        "g_EInitDamageNum",
    ]

    text = data_file_text[data_file_text.find(f"glabel {first_e_init}"):]
    parsed_e_inits = []
    while not parsed_e_inits or matches:
        matches = re.match(e_init_pattern, text) or re.match(split_e_init_pattern, text) or re.match(short_e_init_pattern, text)
        if platform != "psp" and matches and not matches.groupdict().get("size"):
            size_name = matches.groupdict().get("name")
            while not matches.groupdict().get("size"):
                address = int(matches.group("address"), 16)
                name = matches.groupdict().get("name") or f"g_EInitUnused{address:08X}"
                animSet = int(matches.group("animSet"), 16)
                parsed_e_inits.append((
                    Symbol(name, address),
                    f"ANIMSET_{'OVL' if animSet & 0x8000 else 'DRA'}({animSet & ~0x8000})",
                    int(matches.group("animCurFrame"), 16),
                    int(matches.group("unk5A"), 16),
                    int(matches.group("palette"), 16),
                    int(matches.group("enemyID"), 16),
                    ))
                if matches.groupdict().get("size"):
                    break
                text = text[matches.end():]
                unused_last_line_pattern = rf"(?P<size>\.size\s+{size_name},\s+\.\s+-\s+{size_name}\n?)?"
                matches = re.match(unused_e_init_pattern+unused_last_line_pattern, text, re.VERBOSE)

        if matches:
            address = int(matches.group("address"), 16)
            name = matches.groupdict().get("name") or f"g_EInitUnused{address:08X}"
            animSet = int(matches.group("animSet"), 16)
            parsed_e_inits.append((
                Symbol(name, address),
                f"ANIMSET_{'OVL' if animSet & 0x8000 else 'DRA'}({animSet & ~0x8000})",
                int(matches.group("animCurFrame"), 16),
                int(matches.group("unk5A"), 16),
                int(matches.group("palette"), 16),
                int(matches.group("enemyID"), 16),
                ))
            text = text[matches.end() + 1:]

    EnemyDefsVals = [x.value for x in EnemyDefs]

    symbols = [Symbol(name, e_init[0].address) for name, e_init in zip(known_e_inits, parsed_e_inits) if platform != "psp"]
    added_names = []
    for e_init in parsed_e_inits[len(symbols):]:
        if e_init[5] in EnemyDefsVals:
            name = f"g_EInit{EnemyDefs(e_init[5])}".replace("EnemyDefs.", "")
            if name in added_names:
                symbols.append(Symbol(f"{name}{e_init[0].address:X}", e_init[0].address))
            else:
                symbols.append(Symbol(name, e_init[0].address))
            added_names.append(name)
        else:
            symbols.append(Symbol(e_init[0].name, e_init[0].address))
    
    e_inits = [(symbol.name if platform != "psp" else e_init[0].name, e_init[1], e_init[2], e_init[3], e_init[4], f"0x{e_init[5]:03X}") for symbol, e_init in zip(symbols, parsed_e_inits)]
    next_offset = re.match(r"glabel\s+\w+\n\s+/\*\s+(?P<offset>[0-9A-Fa-f]+)\s+", text)
    return e_inits, int(next_offset.group("offset"), 16) if next_offset else None, [x for x in symbols if not x.name.startswith("D_")]

def add_initial_symbols(ovl_config, ovl_header_name, parsed_symbols = [], entity_updates = {}, spinner=SimpleNamespace(message="")):
    subsegments = ovl_config.subsegments.copy()

### group change ###
    spinner.message = f"finding the first data file"
    first_data_offset = next(subseg[0] for subseg in subsegments if "data" in subseg)
    first_data_path = ovl_config.asm_path / "data" / f"{first_data_offset:X}.data.s"
    if first_data_path.exists():
        first_data_text = first_data_path.read_text()
### group change ###
        spinner.message = f"parsing the overlay header for symbols"
        ovl_header, pStObjLayoutHorizontal_address = (
            parse_ovl_header(
                first_data_text,
                ovl_config.name,
                ovl_config.platform,
                ovl_header_name,
            )
        )
        if ovl_header.get("items"):
            spinner.message = f"creating {ovl_config.name}/header.c"
            create_header_c(ovl_header.get("items"), ovl_config.name, ovl_config.ovl_type, ovl_config.version, ovl_config.src_path_full.parent / ovl_config.name / "header.c")
            spinner.message = f"adding header subsegment"
            header_offset = ovl_header["address"] - ovl_config.vram + ovl_config.start
            header_subseg = [header_offset, ".data", f"{ovl_config.name}/header" if ovl_config.platform == "psp" else "header", ovl_header.get("size_bytes", 0)]
            subsegments.append(header_subseg)
        # TODO: Add data segments for follow-on header symbols
        if ovl_config.platform == "psx":
### group change ###
            spinner.message = f"finding the entity table"
            entity_updates = find_psx_entity_updates(first_data_text, pStObjLayoutHorizontal_address)
    else:
        first_data_text = None
        ovl_header, pStObjLayoutHorizontal_address = {}, None
        entity_updates = {}


### group change ###
    spinner.message="gathering initial symbols"
    if entity_updates and first_data_text:
### group change ###
        spinner.message = "parsing EntityUpdates"
        entity_updates.update(parse_entity_updates(first_data_text, ovl_config.name, entity_updates.get("name")))
### group change ###
        spinner.message = "parsing EInits"
        e_inits, next_offset, symbols = parse_e_inits(first_data_text, entity_updates.get("first_e_init"), ovl_config.name, ovl_config.platform)
        parsed_symbols.extend(symbols)
        e_init_c_path = ovl_config.src_path_full.with_name(ovl_config.name) / "e_init.c"

### group change ###
        if ovl_config.version == "us":
            spinner.message = "creating e_init.c"
            e_init_success = create_e_init_c(entity_updates.get("items"), e_inits, ovl_config.name, e_init_c_path)
        else:
            spinner.message = "cross-referencing e_init.c"
            e_init_symbols, e_init_success = cross_reference_e_init_c(entity_updates.get("items"), e_inits, e_init_c_path, ovl_config.name, ovl_config.ld_script_path.with_suffix(".map"))
            parsed_symbols.extend(e_init_symbols)

        entity_updates_offset = entity_updates.get("address", 0) - ovl_config.vram + ovl_config.start
        if entity_updates_offset > 0:
            e_init_subseg = [entity_updates_offset, f"{'.' if e_init_success else ''}data", f"{ovl_config.name}/e_init" if ovl_config.platform == "psp" else "e_init", next_offset - entity_updates_offset]
            subsegments.append(e_init_subseg)
        ovl_include_path = ovl_config.src_path_full.parent / ovl_config.name / f"{ovl_config.name}.h"
        create_ovl_include(entity_updates.get("items"), ovl_config.name, ovl_config.ovl_type, ovl_include_path)


    if ovl_header.get("symbols") or entity_updates.get("symbols"):
        parsed_symbols.extend((
            symbol
            for symbols in (
                ovl_header.get("symbols"),
                entity_updates.get("symbols"),
            )
            if symbols is not None
            for symbol in symbols
        ))
    if entity_updates.get("address"):
        parsed_symbols.append(Symbol(f"{ovl_config.name.upper()}_EntityUpdates", entity_updates.get("address")))
    if ovl_header.get("address"):
        parsed_symbols.append(Symbol(f"{ovl_config.name.upper()}_Overlay", ovl_header.get("address")))

    if parsed_symbols:
### group change ###
        spinner.message = f"adding {len(parsed_symbols)} parsed symbols and splitting again"
        add_symbols(
            ovl_config.ovl_symbol_addrs_path,
            parsed_symbols,
            ovl_config.name,
            ovl_config.vram,
            ovl_config.symbol_name_format.replace("$VRAM", ""),
            ovl_config.src_path_full,
            ovl_config.symexport_path
        )
        shell(f"git clean -fdx {ovl_config.asm_path}")
        splat_split(ovl_config.config_path)

    return subsegments

def rename_similar_functions(ovl_config, parsed_check_files, parsed_ref_files, spinner=SimpleNamespace(message="")):
    matches = find_symbols(
        parsed_check_files, parsed_ref_files, ovl_config.version, ovl_config.name, threshold=0.95
    )
### group change ###
    spinner.message = f"renaming symbols found from {len(matches)} similar functions"
    num_symbols, ambiguous_renames, unhandled_renames = rename_symbols(ovl_config, matches)

    if num_symbols:
### group change ###
        spinner.message=f"renamed {num_symbols} symbols from {len(matches)} similar functions"
        shell(f"git clean -fdx {ovl_config.asm_path}")
        splat_split(ovl_config.config_path, ovl_config.disassemble_all)

    return ambiguous_renames, unhandled_renames
