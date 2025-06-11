#!/usr/bin/env python3

import argparse
import os
import re
import shutil
import decomp_utils
import hashlib
from decomp_utils import RE_TEMPLATES, RE_PATTERNS
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, deque, defaultdict
from pathlib import Path
from types import SimpleNamespace
from mako.template import Template
import multiprocessing

"""
Handles many tasks for adding an overlay:
- Extracts the data necessary to generate an initial config
- Parses known data tables (psx header, entity table, psp export table)
- Compares newly extracted functions to existing asm files to identify functions
- Adds and renames identified symbols
- Cross references asm with existing asm and matches symbols from identical instruction sequences
- Creates the ovl.h file, header.c, and e_init.c files
- Splits source files into segments based on function boundaries
- Extracts and assigns .rodata segments to the correct files
- Parses and logs suggestions for segment splits from splat output
- Validates builds by comparing SHA1 hashes and updates check files

Example usage: python3 tools/decomp_utils/generate_sotn_config.py lib --version=us

Additonal notes:
- If a segment has only one function, it is named as that function in snake case.  If the function name starts with Entity, it replaces it with 'e'.
    For example: A segment with the only function being EntityShuttingWindow would be named as e_shutting_window
"""

# Todo: Allow matches to default functions, but only if there isn't a named match
# Todo: Review accuracy of naming for offset and address variables
# Todo: Handle merging psp ovl.h file with existing psx ovl.h file
# Todo: Collect warnings and display in summary upon completion
# Todo: Add symbols closer to where the address is gathered
# Todo: Add einit common symbols
# Todo: Add EInits to e_init.c
# Todo: Extract and import BackgroundBlockInit data
# Todo: Extract and import RedDoorTiles data
# Todo: Add g_eRedDoorUV data to e_red_door
# Todo: Add // clang-format off if INCLUDE_ASM line longer than 80 characters


def get_known_starts(
    ovl_name, version, segments_path=Path("tools/decomp_utils/segments.yaml")
):
    segments_config = decomp_utils.yaml.safe_load(segments_path.read_text())

    known_segments = []
    # Todo: Simplify this logic
    for name, boundaries in segments_config.items():
        if isinstance(boundaries["start"], str):
            starts = [boundaries["start"]]
        elif isinstance(boundaries["start"], list):
            starts = boundaries["start"]
        else:
            continue

        if "end" not in boundaries:
            end = starts[0]
        elif isinstance(boundaries["end"], str):
            end = boundaries["end"]
        else:
            continue

        known_segments.extend(
            SimpleNamespace(
                name=name.replace("${prefix}", ovl_name.upper()),
                start=start.replace("${prefix}", ovl_name.upper()),
                end=end.replace("${prefix}", ovl_name.upper()),
            )
            for start in starts
        )
    return {x.start: x for x in known_segments}


def find_segments(ovl_config):
    # Todo: Add dynamic segment detection
    segments = []
    rodata_pattern = re.compile(
        RE_TEMPLATES.rodata_offset.substitute(version=ovl_config.version)
    )
    known_starts = get_known_starts(ovl_config.name, ovl_config.version)
    src_text = ovl_config.first_src_file.read_text()

    segment_meta = None
    functions = deque()
    matches = RE_PATTERNS.include_asm.findall(src_text)
    for i, match in enumerate(matches):
        asm_dir, current_function = match
        if (
            current_function in known_starts
            and (
                not segment_meta
                or not segment_meta.name.endswith(known_starts[current_function].name)
            )
        ) or (
            (current_function == "GetLang" or current_function.startswith("GetLang_"))
            and matches[i + 1][1] in known_starts
        ):
            if segment_meta:
                if len(functions) == 1:
                    segment_meta.name = f'{ovl_config.segment_prefix}{RE_PATTERNS.camel_case.sub(r"\1_\2", functions[0]).lower().replace("entity", "e")}'
                segment_meta.end = functions[-1]
                logger.debug(
                    f"Found text segment for {segment_meta.name} at 0x{segment_meta.offset.str}"
                )
                segments.append(segment_meta)
                functions.clear()
                segment_meta = None

            if current_function == "GetLang" or current_function.startswith("GetLang_"):
                segment_meta = known_starts[matches[i + 1][1]]
                segment_meta.start = current_function
            else:
                segment_meta = known_starts[current_function]
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
            )

        if segment_meta and not segment_meta.offset:
            if offset := decomp_utils.get_symbol_offset(ovl_config, current_function):
                segment_meta.offset = SimpleNamespace(int=offset)
                segment_meta.offset.str = f"{segment_meta.offset.int:X}"
            else:
                asm_path = (
                    Path("asm") / ovl_config.version / asm_dir / f"{current_function}.s"
                )
                asm_text = asm_path.read_text()
                if first_offset := re.search(
                    RE_TEMPLATES.asm_symbol_offset.substitute(
                        symbol_name=current_function
                    ),
                    asm_text,
                ):
                    segment_meta.offset = SimpleNamespace(str=first_offset.group(1))
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

    ovl_header_path = (
        f"../{ovl_config.name}/{ovl_config.name}.h"
        if ovl_config.platform == "psp"
        else f"{ovl_config.name}.h"
    )
    file_header = f'// SPDX-License-Identifier: AGPL-3.0-or-later\n#include "{ovl_header_path}"\n\n'
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
            rodata_offset = decomp_utils.get_symbol_offset(ovl_config, rodata_symbol)
            rodata_subsegments.append(
                SimpleNamespace(offset=rodata_offset, type=".rodata", name=segment.name)
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
            if rodata_start != -1 and text_start > rodata_start:
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

    return segments, tuple(rodata_by_segment.values())


def find_symbols(parsed, version, ovl_name, threshold=0.95):
    # Todo: Segments by op hash
    ref_funcs_by_op_hash = decomp_utils.group_by_hash(parsed.ref_files, "op")
    check_funcs_by_op_hash = decomp_utils.group_by_hash(parsed.check_files, "op")
    ref_ops_by_op_hash = {k: v[0].ops.parsed for k, v in ref_funcs_by_op_hash.items()}
    check_ops_by_op_hash = {
        k: v[0].ops.parsed for k, v in check_funcs_by_op_hash.items()
    }

    buckets = decomp_utils.get_buckets(
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
        results = executor.map(decomp_utils.find_matches, kwargs)

    matches = set()
    for ref_op_hash, results in decomp_utils.group_results(results).items():
        ref_paths = tuple(x.path for x in ref_funcs_by_op_hash[ref_op_hash])
        check_op_hash, score, _ = results[0]
        check_paths = tuple(x.path for x in check_funcs_by_op_hash[check_op_hash])
        if ref_paths and check_paths:
            ref_names = tuple(
                RE_PATTERNS.symbol_ovl_name_prefix.sub(ovl_name.upper(), func.stem)
                for func in ref_paths
            )
            check_names = tuple(func.stem for func in check_paths)
            matches.add((ref_paths, ref_names, check_paths, check_names))
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
    known_pairs = (
        SimpleNamespace(first="func_801CC5A4", last="func_801CF438"),
        SimpleNamespace(first="func_801CC90C", last="func_801CF6D8"),
        SimpleNamespace(first="EntityIsNearPlayer", last="SealedDoorIsNearPlayer"),
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
    )
    symbols = defaultdict(list)
    for match in matches:
        for pair in known_pairs:
            if (
                len(match.ref.names.no_defaults) <= 2
                and len(match.check.names) <= 2
                and pair.first in match.ref.names.no_defaults
                and (
                    pair.last in match.ref.names.no_defaults
                    or len(match.ref.names.no_defaults) == 1
                )
            ):
                offset = min(
                    tuple(int(x.split("_")[-1], 16) for x in match.check.names)
                )
                symbols[pair.first].append(decomp_utils.Symbol(pair.first, offset))
                if (
                    len(match.check.names) == 2
                    and pair.last in match.ref.names.no_defaults
                ):
                    offset = max(
                        tuple(int(x.split("_")[-1], 16) for x in match.check.names)
                    )
                    symbols[pair.last].append(decomp_utils.Symbol(pair.last, offset))
                break
        else:
            if len(match.check.names) == 1 and match.check.names[0].startswith(
                f"func_{ovl_config.version}_"
            ):
                offset = int(match.check.names[0].split("_")[-1], 16)
                if match.ref.names.no_defaults:
                    symbols[match.ref.counts.no_defaults[0][0]].append(
                        decomp_utils.Symbol(match.ref.counts.no_defaults[0][0], offset)
                    )
                if (
                    len(match.ref.counts.no_defaults) > 1
                    and match.ref.counts.no_defaults[0][1]
                    == match.ref.counts.no_defaults[1][1]
                ):
                    logger.warning(
                        f"Ambiguous match: {match.check.names[0]} renamed to {match.ref.counts.no_defaults[0][0]}, all matches were {[x[0] for x in match.ref.counts.no_defaults]} with a score of {match.score}"
                    )

            elif match.ref.counts.no_defaults[0][0] == "GetLang":
                for name in match.check.names:
                    name = name.replace(f"func_{ovl_config.version}_", "GetLang_")
                    symbols[name].append(
                        decomp_utils.Symbol(name, int(name.split("_")[-1], 16))
                    )
            elif match.ref.names.no_defaults != match.check.names:
                logger.warning(
                    f"Found unhandled naming condition: Target name {match.ref.counts.no_defaults} for {ovl_config.name} function(s) {match.check.names} with score {match.score}"
                )

    if not symbols:
        logger.warning("\nNo new symbols found\n")
        # Todo: Where should this continue on if no new symbols are found?
        exit()
    # Todo: Figure out a better way to handle multiple functions mapping to multiple functions with the same name
    decomp_utils.add_symbols(
        ovl_config,
        tuple(
            sorted(syms, key=lambda x: x.address, reverse=True)[0]
            for syms in symbols.values()
        ),
    )
    return len(symbols)


def parse_psp_stage_init(asm_path):
    stage_init_name, header_symbol, entity_table_symbol = None, None, None
    first_address_pattern = re.compile(r"\s+/\*\s+[A-F0-9]{1,5}\s+([A-F0-9]{8})\s")
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
            match = RE_PATTERNS.psp_entity_export_table_pattern.search(file_text)
            if match:
                stage_init_address = first_address_pattern.search(file_text)
                stage_init_address = (
                    int(stage_init_address.group(1), 16) if stage_init_address else None
                )
                return (
                    (file.stem, stage_init_address),
                    match.group("export"),
                    match.group("entity"),
                )
        # I'm pretty sure all of these are found in the same file, but keeping this here just in case
        else:
            if (
                " 1D09043C " in file_text
                and " 38F78424 " in file_text
                and " E127240E " in file_text
            ):
                match = RE_PATTERNS.psp_export_table_pattern.search(file_text)
                if match:
                    stage_init_name = file.stem
                    stage_init_address = first_address_pattern.search(file_text)
                    stage_init_address = (
                        int(stage_init_address.group(1), 16)
                        if stage_init_address
                        else None
                    )
                    header_symbol = match.group("export")
            if " C708023C " in file_text and " 30BC43AC " in file_text:
                match = RE_PATTERNS.psp_entity_table_pattern.search(file_text)
                if match:
                    entity_table_symbol = match.group("entity")
            if stage_init_name and header_symbol and entity_table_symbol:
                return (
                    (stage_init_name, stage_init_address),
                    header_symbol,
                    entity_table_symbol,
                )
    else:
        return (
            (stage_init_name, stage_init_address) if stage_init_name else None,
            header_symbol,
            entity_table_symbol,
        )


def parse_ovl_header(data_file_text, name, platform, ovl_type, header_symbol=None):
    # Account for both Abbreviated and full headers
    # Account for difference in stage headers vs other headers
    match ovl_type:
        case "stage" | "boss":
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
                "g_SpriteBank1",
                "g_SpriteBank2",
                # "unk34",
                # "unk38",
                # "unk3C",
            ]
        case "weapon":
            ovl_header = [
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
        return None
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
        symbols = tuple(
            decomp_utils.Symbol(
                name, int.from_bytes(bytes.fromhex(address[0]), "little")
            )
            # Todo: Does this need the filtering, or should it just overwrite the existing regardless?
            for name, address in zip(ovl_header, matches)
            if address[1].startswith("func_")
            or address[1].startswith("D_")
            or address[1].startswith("g_")
        )
        return header_address, symbols, pStObjLayoutHorizontal_address
    else:
        return None


def parse_init_room_entities(ovl_name, platform, init_room_entities_path):
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
    symbols = tuple(
        decomp_utils.Symbol(
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
    )
    create_entity_bss_address = min(
        x.address for x in symbols if x.name.startswith("g_Layout")
    )

    return symbols, create_entity_bss_address


def parse_entity_table(data_file_text, ovl_name, entity_table_symbol):
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
        parsed_entity_table = data_file_text[
            entity_table_start:entity_table_end
        ].splitlines()
        for i, line in enumerate(parsed_entity_table):
            if " func" in line or " Entity" in line:
                entity_table_address = int(line.split()[2], 16)
                break
            else:
                entity_table_address = None

        parsed_entity_table = "\n".join(parsed_entity_table[i:])
        matches = RE_PATTERNS.symbol_line_pattern.findall(parsed_entity_table)

    if matches:
        # Do not rename to EntityDummy if the two addresses don't match
        if len(matches) > 14 and matches[14] != matches[15]:
            entity_table[14:15] = [
                f"EntityUnk{matches[14][0]}",
                f"EntityUnk{matches[15][0]}",
            ]
        symbols = tuple(
            decomp_utils.Symbol(
                name, int.from_bytes(bytes.fromhex(address[0]), "little")
            )
            for name, address in zip(entity_table, matches)
        )

    else:
        symbols = tuple()

    return entity_table_address, symbols + ()


def create_extra_files(data_file_text, ovl_config):
    ovl_header_path = (
        f"../{ovl_config.name}/{ovl_config.name}.h"
        if ovl_config.platform == "psp"
        else f"{ovl_config.name}.h"
    )
    entity_table_start = data_file_text.find(
        f"glabel {ovl_config.name.upper()}_EntityUpdates"
    )
    entity_table_end = data_file_text.find(
        f".size {ovl_config.name.upper()}_EntityUpdates"
    )
    if entity_table_start != -1:
        parsed_entity_table = data_file_text[
            entity_table_start:entity_table_end
        ].splitlines()[1:]
        entity_funcs = [
            (
                f'{line.split()[-1].replace(f"{ovl_config.name.upper()}_", "OVL_EXPORT(")})'
                if f"{ovl_config.name.upper()}_" in line
                else line.split()[-1]
            )
            for line in parsed_entity_table
        ]
        e_inits = []
        for i, func in enumerate(entity_funcs):
            if func == "EntityDummy":
                e_inits.append((func, f"E_DUMMY_{i+1:X}"))
            elif func.startswith("Entity") or func.startswith("OVL_EXPORT(Entity"):
                e_inits.append(
                    (
                        func,
                        RE_PATTERNS.camel_case.sub(
                            r"\1_\2", func.replace("OVL_EXPORT(", "").replace(")", "")
                        )
                        .upper()
                        .replace("ENTITY", "E"),
                    )
                )
            else:
                e_inits.append((func, f"E_UNK_{i+1:X}"))

        template = Template(
            Path("tools/decomp_utils/templates/e_init.c.mako").read_text()
        )
        output = template.render(
            ovl_name=ovl_config.name,
            entity_funcs=entity_funcs,
        )
        ovl_config.src_path_full.joinpath("e_init.c").write_text(output)

        template = Template(Path("tools/decomp_utils/templates/ovl.h.mako").read_text())
        output = template.render(
            ovl_name=ovl_config.name,
            e_inits=e_inits,
        )
        ovl_config.src_path_full.joinpath(ovl_config.name).with_suffix(".h").write_text(
            output
        )

    header_start = data_file_text.find(f"glabel {ovl_config.name.upper()}_Overlay")
    header_end = data_file_text.find(f".size {ovl_config.name.upper()}_Overlay")
    if header_start != -1:
        header_syms = []
        for line in data_file_text[header_start:header_end].splitlines()[1:]:
            if f"{ovl_config.name.upper()}_" in line:
                header_syms.append(
                    f'{line.split()[-1].replace(f"{ovl_config.name.upper()}_", "OVL_EXPORT(")})'
                )
            else:
                name = line.split()[-1]
                header_syms.append("NULL" if name == "0x00000000" else name)

        template = Template(
            Path("tools/decomp_utils/templates/header.c.mako").read_text()
        )
        output = template.render(
            ovl_header_path=ovl_header_path,
            header_syms=header_syms,
        )
        ovl_config.src_path_full.joinpath("header.c").write_text(output)


def ovl_sort(name):
    # Todo: This should ikely be simplified
    game = "dra ric maria "
    stage = "are cat cen chi dai dre lib mad no0 no1 no2 no3 no4 np3 nz0 nz1 sel st0 top wrp "
    r_stage = (
        "rare rcat rcen rchi rdai rlib rno0 rno1 rno2 rno3 rno4 rnz0 rnz1 rtop rwrp "
    )
    boss = "bo0 bo1 bo2 bo3 bo4 bo5 bo6 bo7 mar rbo0 rbo1 rbo2 rbo3 rbo4 rbo5 rbo6 rbo7 rbo8 "
    servant = "tt_000 tt_001 tt_002 tt_003 tt_004 tt_005 tt_006 "

    name = Path(name).stem.lower()
    basename = name.replace("f_", "")
    if basename == "main":
        group = 0
    elif basename in game and basename != "mar":
        group = 1
    elif basename in stage:
        group = 2
    elif basename in r_stage:
        group = 3
    elif basename in boss:
        group = 4
    elif basename in boss and basename.startswith("r"):
        group = 5
    elif name in servant:
        group = 6
    elif "weapon" in name or "w0_" in name or "w1_" in name:
        group = 7
    else:
        group = 8

    return (group, basename, name.startswith("f_"))


def clean_artifacts(ovl_config):
    ovl_config.config_path.unlink(missing_ok=True)
    ovl_config.ovl_symbol_addrs_path.unlink(missing_ok=True)
    if ovl_config.symexport_path:
        ovl_config.symexport_path.unlink(missing_ok=True)
    if ovl_config.asm_path.exists():
        decomp_utils.shell(f"git clean -fdx {ovl_config.asm_path}")
    if (path := ovl_config.build_path / ovl_config.src_path_full).exists():
        shutil.rmtree(path)
    ovl_config.ld_script_path.unlink(missing_ok=True)
    ovl_config.ld_script_path.with_suffix(".elf").unlink(missing_ok=True)
    ovl_config.ld_script_path.with_suffix(".map").unlink(missing_ok=True)
    ovl_config.config_path.unlink(missing_ok=True)
    ovl_config.ovl_symbol_addrs_path.unlink(missing_ok=True)
    ovl_config.build_path.joinpath(f"{ovl_config.target_path.name}").unlink(
        missing_ok=True
    )
    if ovl_config.version != "hd" and ovl_config.src_path_full.exists():
        shutil.rmtree(ovl_config.src_path_full)


def main(args):
    logger.info("Starting...")
    with decomp_utils.Spinner(message="generating config"):
        ovl_config = decomp_utils.SotnOverlayConfig(args.overlay, args.version)
        if ovl_config.config_path.exists() and not args.force:
            logger.error(
                f"A configuration for {ovl_config.name} already exists.  Use the -f/--force option to remove all existing overlay artifacts and recreate the configuration."
            )
            raise SystemExit
        else:
            ovl_config.write_config()
            with decomp_utils.Spinner(message="ensuring no overlay artifacts exist"):
                clean_artifacts(ovl_config)

    with decomp_utils.Spinner(message="creating initial files") as spinner:
        # Todo: Create "extraction init" function
        ovl_config.write_config()
        for symbol_path in ovl_config.symbol_addrs_path:
            symbol_path.touch(exist_ok=True)
        header_path = (
            ovl_config.src_path_full.with_name(ovl_config.name) / f"{ovl_config.name}.h"
        )
        ovl_header_text = f"""// SPDX-License-Identifier: AGPL-3.0-or-later
#include "stage.h"

#define OVL_EXPORT(x) {ovl_config.name.upper()}_##x
"""
        if not header_path.exists():
            header_path.parent.mkdir(parents=True, exist_ok=True)
            header_path.write_text(ovl_header_text)

        spinner.message = f"adding sha1 hashes to check file"
        check_file_path = Path(f"config/check.{ovl_config.version}.sha")
        check_file_lines = check_file_path.read_text().splitlines()
        new_lines = check_file_lines.copy()
        bin_line = f"{ovl_config.sha1}  build/{ovl_config.version}/{ovl_config.target_path.name}"
        if bin_line not in new_lines:
            new_lines.append(bin_line)
        fbin_path = ovl_config.target_path.with_name(
            f"{"f" if ovl_config.platform == "psp" else "F"}_{ovl_config.target_path.name}"
        )
        if fbin_path.exists():
            fbin_sha1 = hashlib.sha1(fbin_path.read_bytes()).hexdigest()
            fbin_line = f"{fbin_sha1}  build/{ovl_config.version}/{fbin_path.name}"
            if fbin_line not in new_lines:
                new_lines.append(fbin_line)
        if new_lines != check_file_lines:
            # Todo: Order the sha1 lines correctly
            sorted_lines = sorted(new_lines, key=lambda x: ovl_sort(x.split()[-1]))
            check_file_path.write_text(f"{"\n".join(sorted_lines)}\n")

        decomp_utils.shell(f"git add {check_file_path}")

        spinner.message = f"performing initial split with {ovl_config.config_path}"
        decomp_utils.splat_split(ovl_config.config_path, ovl_config.disassemble_all)
        src_text = ovl_config.first_src_file.read_text()
        adjusted_text = src_text.replace(f'("asm/{args.version}/', '("')
        ovl_config.first_src_file.write_text(adjusted_text)
        decomp_utils.build(build=False, version=ovl_config.version)

    with decomp_utils.Spinner(message=f"gathering initial symbols") as spinner:
        # Cleanup and split to functions as needed
        header_symbol, header_address, header_symbols = None, None, None
        parsed_symbols = ()
        entity_table_symbol, entity_table_address, entity_table_symbols = (
            None,
            None,
            None,
        )
        first_data_index = next(
            i for i, subseg in enumerate(ovl_config.subsegments) if "data" in subseg
        )
        first_data_path = (
            ovl_config.asm_path
            / "data"
            / f"{ovl_config.subsegments[first_data_index][0]:X}.data.s"
        )
        first_data_text = (
            first_data_path.read_text() if first_data_path.exists() else None
        )

        if ovl_config.platform == "psp":
            spinner.message = f"parsing the psp stage init for symbols"
            stage_init, header_symbol, entity_table_symbol = parse_psp_stage_init(
                ovl_config.asm_path.joinpath(ovl_config.nonmatchings_path)
            )
            if stage_init:
                stage_init_name, stage_init_address = stage_init
                if stage_init_name and stage_init_address:
                    parsed_symbols += (
                        decomp_utils.Symbol(
                            f"{ovl_config.name.upper()}_Load", stage_init_address
                        ),
                    )
                if not ovl_config.symexport_path.exists():
                    spinner.message = "creating symexport file"
                    symexport_text = f"EXTERN(_binary_assets_{ovl_config.path_prefix}{"_" if ovl_config.ovl_prefix else ""}{ovl_config.name}_mwo_header_bin_start);\n"
                    symexport_text += f"EXTERN({stage_init_name});\n"
                    ovl_config.symexport_path.write_text(symexport_text)
                    decomp_utils.shell(f"git add {ovl_config.symexport_path}")

        if first_data_text:
            spinner.message = f"parsing the overlay header for symbols"
            header_address, header_symbols, pStObjLayoutHorizontal_address = (
                parse_ovl_header(
                    first_data_text,
                    ovl_config.name,
                    ovl_config.platform,
                    ovl_config.ovl_type,
                    header_symbol,
                )
            )

        if pStObjLayoutHorizontal_address and not entity_table_symbol:
            spinner.message = f"finding the entity table"
            # Todo: Find a less complicated way to handle this
            end_of_header = first_data_text.find(".size")
            pStObjLayoutHorizontal_index = first_data_text.find(
                f"{pStObjLayoutHorizontal_address:08X}", end_of_header
            )
            first_entity_index = first_data_text.find(
                " func", pStObjLayoutHorizontal_index
            )
            entity_table_index = first_data_text.rfind(
                "glabel", pStObjLayoutHorizontal_index, first_entity_index
            )
            entity_table_symbol = (
                first_data_text[entity_table_index:first_entity_index]
                .split("\n")[0]
                .split()[1]
            )

        if entity_table_symbol:
            spinner.message = f"parsing the entity table for symbols"
            entity_table_address, entity_table_symbols = parse_entity_table(
                first_data_text, ovl_config.name, entity_table_symbol
            )

        if header_symbols or entity_table_symbols:
            parsed_symbols += tuple(
                symbol
                for symbols in (
                    header_symbols,
                    entity_table_symbols,
                    header_symbols,
                )
                if symbols is not None
                for symbol in symbols
            )
            if entity_table_address:
                parsed_symbols += (
                    decomp_utils.Symbol(
                        f"{ovl_config.name.upper()}_EntityUpdates", entity_table_address
                    ),
                )
            if header_address:
                parsed_symbols += (
                    decomp_utils.Symbol(
                        f"{ovl_config.name.upper()}_Overlay", header_address
                    ),
                )

        if parsed_symbols:
            spinner.message = f"adding {len(parsed_symbols)} parsed symbols and splitting using updated symbols"
            decomp_utils.add_symbols(ovl_config, parsed_symbols)
            decomp_utils.shell(
                f"git add {ovl_config.config_path} {ovl_config.ovl_symbol_addrs_path}"
            )
            if ovl_config.symexport_path and ovl_config.symexport_path.exists():
                decomp_utils.shell(f"git add {ovl_config.symexport_path}")
            decomp_utils.shell(f"git clean -fdx {ovl_config.asm_path}")
            decomp_utils.splat_split(ovl_config.config_path)

    with decomp_utils.Spinner(
        message="creating .elf files for extracting reference symbols"
    ) as spinner:
        ref_basenames, ref_ovls = [], []
        for file in Path("config").iterdir():
            ref_version = (
                "pspeu" in file.name
                if ovl_config.platform == "psp"
                else "us" in file.name or "hd" in file.name
            )

            # Todo: Evaluate whether limiting to stage and non-stage overlays as references makes a practical difference in execution time
            if (
                ref_version
                and "main" not in file.name
                and "weapon" not in file.name
                and "mad" not in file.name
                and ovl_config.name not in file.name
                and file.match("splat.*.yaml")
                and (match := RE_PATTERNS.ref_pattern.match(file.name))
            ):
                ref_basenames.append(
                    f'{match.group("prefix") or ""}{match.group("ref_ovl")}'
                )
                ref_ovls.append(match.group("ref_ovl"))

        ref_lds = tuple(
            ovl_config.build_path.joinpath(basename).with_suffix(".ld")
            for basename in ref_basenames
        )
        found_elfs = tuple(ovl_config.build_path.glob("*.elf"))
        missing_elfs = tuple(
            ld.with_suffix(".elf")
            for ld in ref_lds
            if ld.with_suffix(".elf") not in found_elfs
        )
        if missing_elfs:
            decomp_utils.build(missing_elfs, plan=True, version=ovl_config.version)

    with decomp_utils.Spinner(
        message=f"extracting symbols from {len(ref_basenames)} reference .elf files"
    ) as spinner:
        decomp_utils.force_symbols(
            tuple(ld.with_suffix(".elf") for ld in ref_lds),
            version=ovl_config.version,
        )

        spinner.message = f"disassembling {len(ref_basenames)} reference overlays"
        decomp_utils.shell(
            f"git clean -fdx asm/{ovl_config.version}/ -e {ovl_config.asm_path}"
        )
        decomp_utils.build(ref_lds, plan=False, version=ovl_config.version)

        # Removes forced symbols files
        # Todo: checkout each file instead of the whole dir
        decomp_utils.shell("git checkout config/")

    with decomp_utils.Spinner(
        message=f"parsing instructions from reference overlay asm files"
    ) as spinner:
        ref_files, check_files = [], []
        for dirpath, _, filenames in Path("asm").joinpath(ovl_config.version).walk():
            if any(x in dirpath.parts or f"{x}_psp" in dirpath.parts for x in ref_ovls):
                ref_files.extend(
                    dirpath / f
                    for f in filenames
                    if not f.startswith(f"func_{ovl_config.version}_")
                    and not f.startswith("D_")
                )
            if (
                ovl_config.name in dirpath.parts
                or f"{ovl_config.name}_psp" in dirpath.parts
            ):
                check_files.extend(
                    dirpath / f
                    for f in filenames
                    if f.startswith(f"func_{ovl_config.version}_")
                )
        parsed = SimpleNamespace(
            ref_files=decomp_utils.parse_files(ref_files),
            check_files=decomp_utils.parse_files(check_files),
        )

    with decomp_utils.Spinner(
        message="finding symbol names using reference overlays"
    ) as spinner:
        matches = find_symbols(
            parsed, ovl_config.version, ovl_config.name, threshold=0.95
        )
        num_symbols = rename_symbols(ovl_config, matches)

    if num_symbols:
        with decomp_utils.Spinner(
            message=f"renamed {num_symbols} symbols, splitting again"
        ):
            decomp_utils.shell(f"git clean -fdx {ovl_config.asm_path}")
            decomp_utils.splat_split(ovl_config.config_path, ovl_config.disassemble_all)

    nonmatchings_path = (
        f"{ovl_config.nonmatchings_path}/{ovl_config.name}_psp"
        if ovl_config.platform == "psp"
        else ovl_config.nonmatchings_path
    )
    init_room_entities_path = (
        ovl_config.asm_path
        / nonmatchings_path
        / f"first_{ovl_config.name}"
        / f"InitRoomEntities.s"
    )
    # Sel has an InitRoomEntities function, but the symbols it references are different
    if init_room_entities_path.exists() and ovl_config.name != "sel":
        with decomp_utils.Spinner(
            message=f"parsing InitRoomEntities.s for symbols"
        ) as spinner:
            init_room_entities_symbols, create_entity_bss_address = (
                parse_init_room_entities(
                    ovl_config.name, ovl_config.platform, init_room_entities_path
                )
            )

            # Todo: Add bss segment comment
            if create_entity_bss_address:
                create_entity_bss_start = (
                    create_entity_bss_address - ovl_config.vram + ovl_config.start
                )
                create_entity_bss_end = create_entity_bss_start + (
                    0x18 if ovl_config.platform == "psp" else 0x10
                )

            if init_room_entities_symbols:
                spinner.message = f"adding {len(init_room_entities_symbols)} extracted from InitRoomEntities.s"
                decomp_utils.add_symbols(ovl_config, init_room_entities_symbols)

    with decomp_utils.Spinner(
        message="looking for functions for cross referencing"
    ) as spinner:
        parsed.check_files = decomp_utils.parse_files(
            dirpath / f
            for dirpath, _, filenames in ovl_config.asm_path.walk()
            for f in filenames
            if not f.startswith(f"func_{ovl_config.version}_")
            and not f.startswith("D_")
        )
        check_files_by_name = {x.path.name: x for x in parsed.check_files}
        ref_files_by_name = defaultdict(list)
        for ref_file in parsed.ref_files:
            # Todo: Explore the functions that have identical ops, but differing normalized instructions
            if (
                ref_file.path.name in check_files_by_name
                and ref_file.instructions.normalized
                == check_files_by_name[ref_file.path.name].instructions.normalized
            ):
                ref_files_by_name[ref_file.path.name].append(ref_file)

        if ref_files_by_name:
            spinner.message = f"cross referencing {len(ref_files_by_name)} functions"

            # Todo: Clean up this logic
            new_syms = set()
            for file_name, parsed_ref_files in ref_files_by_name.items():
                parsed_check_file = check_files_by_name[file_name]
                for parsed_ref_file in parsed_ref_files:
                    for ref_instruction, check_instruction in zip(
                        parsed_ref_file.instructions.parsed,
                        parsed_check_file.instructions.parsed,
                    ):
                        # Todo: Warn if a symbol is not default and does not match the cross referenced symbol
                        name = RE_PATTERNS.cross_ref_name_pattern.match(ref_instruction)
                        address = RE_PATTERNS.cross_ref_address_pattern.match(
                            check_instruction
                        )
                        if (
                            name
                            and not name.group(1).startswith("D_")
                            and not name.group(1).startswith(
                                f"func_{ovl_config.version}_"
                            )
                            and address
                        ):
                            new_syms.add(
                                decomp_utils.Symbol(
                                    RE_PATTERNS.symbol_ovl_name_prefix.sub(
                                        ovl_config.name.upper(), name.group(1)
                                    ),
                                    int(address.group(1), 16),
                                )
                            )

    if new_syms:
        with decomp_utils.Spinner(
            message=f"adding {len(new_syms)} cross referenced symbols and splitting again"
        ):
            decomp_utils.add_symbols(ovl_config, tuple(new_syms))
            decomp_utils.shell(f"git clean -fdx {ovl_config.asm_path}")
            decomp_utils.splat_split(ovl_config.config_path, ovl_config.disassemble_all)

    with decomp_utils.Spinner(message="staging files"):
        decomp_utils.shell(
            f"git add {ovl_config.config_path} {ovl_config.ovl_symbol_addrs_path}"
        )

    with decomp_utils.Spinner(
        message=f"creating {ovl_config.ld_script_path.with_suffix(".elf")}"
    ):
        decomp_utils.build(
            [f'{ovl_config.ld_script_path.with_suffix(".elf")}'],
            version=ovl_config.version,
        )

    with decomp_utils.Spinner(
        message=f"finding segments and splitting source files"
    ) as spinner:
        first_text_index = next(
            i for i, subseg in enumerate(ovl_config.subsegments) if "c" in subseg
        )
        segments, rodata_segments = find_segments(ovl_config)
        text_subsegs = [
            decomp_utils.yaml.FlowSegment([segment.offset.int, "c", segment.name])
            for segment in segments
        ]
        rodata_subsegs = [decomp_utils.yaml.FlowSegment(x) for x in rodata_segments]
        ovl_config.subsegments[first_text_index : first_text_index + 1] = text_subsegs
        first_rodata_index = next(
            i for i, subseg in enumerate(ovl_config.subsegments) if ".rodata" in subseg
        )
        ovl_config.subsegments[first_rodata_index : first_rodata_index + 1] = (
            rodata_subsegs
        )

    ovl_config.write_config()

    with decomp_utils.Spinner(
        message=f"extracting overlay to validate configuration"
    ) as spinner:
        # Todo: Compare generated offsets to .elf segment offsets
        decomp_utils.shell(f"git clean -fdx {ovl_config.asm_path}")
        ovl_config.ld_script_path.unlink(missing_ok=True)
        output = decomp_utils.build(
            [f"{ovl_config.ld_script_path}"], version=ovl_config.version
        ).decode()
        splat_suggestions = RE_PATTERNS.splat_suggestions_full.finditer(output)

        suggested_segments = []
        for match in splat_suggestions:
            suggestions = RE_PATTERNS.splat_suggestion.findall(
                match.group("suggestions")
            )
            suggested_segments.extend(
                [offset, segment_type, match.group("segment")]
                for offset, segment_type in suggestions
            )
        if suggested_segments:
            # Todo: Improve logging formatting
            logger.warning(
                f"Additional segments suggested by splat: {suggested_segments}"
            )

    with decomp_utils.Spinner(message="populating e_inits"):
        create_extra_files(first_data_path.read_text(), ovl_config)

    built_bin = ovl_config.build_path / f"{ovl_config.target_path.name}"
    with decomp_utils.Spinner(message=f"building and validating {built_bin}"):
        # These blocks may result in misordered symbols, but it isn't worth addressing for one time use blocks
        # psx rchi has a data value that gets interpreted as a global symbol, so that symbol needs to be defined for the linker
        if ovl_config.name == "rchi" and ovl_config.platform == "psx":
            undefined_syms = Path(f"config/undefined_syms.{ovl_config.version}.txt")
            undefined_syms.write_text(
                f'PadRead{" "*13}= 0x80015288;\n{undefined_syms.read_text()}'
            )
            decomp_utils.shell(f"git add {undefined_syms}")
        # psp bo4 has data values that get interpreted as a global symbol, so that symbol needs to be defined for the linker
        elif ovl_config.name == "bo4" and ovl_config.platform == "psp":
            undefined_syms = Path(f"config/undefined_syms.{ovl_config.version}.txt")
            undefined_syms.write_text(
                f'g_Clut{" "*13}= 0x091F5DF8;\n{undefined_syms.read_text()}'
            )
            decomp_utils.shell(f"git add {undefined_syms}")

        decomp_utils.build(
            [
                f'{ovl_config.ld_script_path.with_suffix(".elf")}',
                f"{ovl_config.build_path}/{ovl_config.target_path.name}",
            ],
            version=ovl_config.version,
        )
        if built_bin.exists():
            built_sha1 = hashlib.sha1(built_bin.read_bytes()).hexdigest()
        else:
            logger.error(f"{built_bin} did not build properly")
            raise SystemExit

        if ovl_config.sha1 != built_sha1:
            logger.error(f"{built_bin} did not match {ovl_config.target_path}")
            raise SystemExit
        else:
            if ovl_config.symexport_path and ovl_config.symexport_path.exists():
                decomp_utils.shell(f"git add {ovl_config.symexport_path}")
            decomp_utils.shell(
                f"git add {ovl_config.config_path} {ovl_config.ovl_symbol_addrs_path}"
            )

    # with decomp_utils.Spinner(message=f"adding header.c") as spinner:
    # Todo: Build header.c
    # spinner.message = f"adding e_init.c"
    # Todo: Parse final entity table and build e_init.c


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create initial configuration for new overlays"
    )
    parser.add_argument(
        "overlay",
        help="Name of the overlay to create a configuration for",
    )
    parser.add_argument(
        "-v",
        "--version",
        required=False,
        default=os.getenv("VERSION") or "us",
        help="The version of the game to target",
    )
    parser.add_argument(
        "-f",
        "--force",
        required=False,
        action="store_true",
        help="DESTRUCTIVE: Force recreation of overly configuration, symbol, and source files",
    )
    # Todo: Add option to use mipsmatch instead of native matching
    # Todo: Add option to generate us and pspeu versions at the same time
    # Todo: Add option for specifying log file
    # Todo: Move this to a distinct concurrency file
    multiprocessing.log_to_stderr()
    multiprocessing.set_start_method("spawn")
    global args
    args = parser.parse_args()
    global logger
    logger = decomp_utils.get_logger()

    main(args)
