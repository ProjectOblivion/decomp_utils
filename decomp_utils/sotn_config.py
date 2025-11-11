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
Code for handling the creation of Castlevania SOTN Splat configs
"""

__all__ = [
    "get_known_starts",
    "find_segments",
    "find_symbols",
    "rename_symbols",
    "parse_psp_stage_init",
    "parse_psp_weapon_load",
    "parse_ovl_header",
    "parse_init_room_entities",
    "parse_entity_table",
    "create_extra_files",
    "ovl_sort",
    "clean_artifacts",
]

logger = decomp_utils.get_logger()

# TODO: Use mipsmatch to supplement segments.yaml
# TODO: mipsmatch scan some.yaml another.yaml evenmore.yaml import.bin for the bin you're importing
def get_known_starts(
    ovl_name, version, segments_path=Path("tools/decomp_utils/segments.yaml")
):
    segments_config = decomp_utils.yaml.safe_load(segments_path.read_text())

    known_segments = []
    # Todo: Simplify this logic
    for label, values in segments_config.items():
        if not values:
            continue

        if "ovl" not in values or ovl_name in values["ovl"]:
            if isinstance(values["start"], str):
                starts = [values["start"]]
            elif isinstance(values["start"], list):
                starts = values["start"]
            else:
                continue

            if "end" in values and isinstance(values["end"], str):
                end = values["end"]
            elif "allow" not in values:
                end = starts[0]
            elif isinstance(values["allow"], list):
                end = ""
            else:
                continue

            known_segments.extend(
                SimpleNamespace(
                    name=values.get("name", label).replace("${prefix}", ovl_name.upper()),
                    start=start.replace("${prefix}", ovl_name.upper()),
                    end=end.replace("${prefix}", ovl_name.upper()),
                    allow=tuple(v.replace("${prefix}", ovl_name.upper()) for v in values.get("allow", [])) or None
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
                or not segment_meta.name
                or not segment_meta.name.endswith(known_starts[current_function].name)
            )
        ) or (
            (current_function == "GetLang" or current_function.startswith("GetLang_"))
            and matches[i + 1][1] in known_starts
        ):
            if segment_meta:
                if not segment_meta.name and len(functions) == 1:
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
                allow=None,
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
        elif segment_meta and segment_meta.allow and current_function not in segment_meta.allow:
            logger.debug(
                f"Found text segment for {segment_meta.name} at 0x{segment_meta.offset.str}"
            )
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
            if rodata_offset:
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
                sorted((RE_PATTERNS.symbol_ovl_name_prefix.sub(ovl_name.upper(), func.stem) for func in ref_paths),
                       key=lambda x: 0 if re.match(r'^[A-Z0-9]{3,4}', x) else 1 if not x.startswith('func_') else 2)
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

    if symbols:
        # Todo: Figure out a better way to handle multiple functions mapping to multiple functions with the same name
        decomp_utils.add_symbols(
            ovl_config,
            tuple(
                sorted(syms, key=lambda x: x.address, reverse=True)[0]
                for syms in symbols.values()
            ),
        )
        return len(symbols)
    else:
        logger.warning("\nNo new symbols found\n")
        return 0


def parse_psp_stage_init(asm_path):
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
            match = RE_PATTERNS.psp_ovl_header_entity_table_pattern.search(file_text)
            if match:
                stage_init_address = first_address_pattern.search(file_text)
                stage_init_address = (
                    int(stage_init_address.group(1), 16) if stage_init_address else None
                )
                return {"name": file.stem,
                        "address": stage_init_address,
                        "ovl_header": match.group("header")}, {"name": match.group("entity")}
    else:
        return {}

# This function is deprecated and scheduled for demolition
def parse_psp_stage_init_fallback(asm_path):
    stage_init_name, header_symbol, entity_table_symbol = None, None, None
    first_address_pattern = re.compile(r"\s+/\*\s+[A-F0-9]{1,5}\s+([A-F0-9]{8})\s")
    for file in (
        dirpath / f
        for dirpath, _, filenames in asm_path.walk()
        for f in filenames
        if ".data.s" not in f
    ):
        file_text = file.read_text()
        if (
            " 1D09043C " in file_text
            and " 38F78424 " in file_text
            and " E127240E " in file_text
        ):
            match = RE_PATTERNS.psp_ovl_header_pattern.search(file_text)
            if match:
                stage_init_name = file.stem
                stage_init_address = first_address_pattern.search(file_text)
                stage_init_address = (
                    int(stage_init_address.group(1), 16)
                    if stage_init_address
                    else None
                )
                header_symbol = match.group("header")
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
        case _:
            return {}
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
        return {}
    # Todo: Should this be findall or finditer?
    matches = RE_PATTERNS.symbol_line_pattern.findall(header)
    if matches:
        if ovl_type != "weapon" and len(matches) > 7:
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
        return {"address": header_address,
                "symbols": symbols}, pStObjLayoutHorizontal_address
    else:
        return {}


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
            ovl_type=ovl_config.ovl_type,
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
            ovl_type=ovl_config.ovl_type,
            header_syms=header_syms,
        )
        ovl_config.src_path_full.joinpath("header.c").write_text(output)


def ovl_sort(name):
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
    with decomp_utils.Spinner(message="cleaning artifacts") as spinner:
        spinner.message=f"Removing config/check.{ovl_config.version}.sha"
        sha_check_path = Path(f"config/check.{ovl_config.version}.sha")
        sha_check_lines = (line for line in sha_check_path.read_text().splitlines() if ovl_config.sha1 not in line)
        fbin_path = ovl_config.target_path.with_name(
                f"{"f" if ovl_config.platform == "psp" else "F"}_{ovl_config.target_path.name}"
            )
        if fbin_path.exists():
            fbin_sha1 = hashlib.sha1(fbin_path.read_bytes()).hexdigest()
            sha_check_lines = (line for line in sha_check_lines if fbin_sha1 not in line)
        sha_check_path.write_text("\n".join(sha_check_lines) + "\n")
        
        spinner.message=f"Removing {ovl_config.ovl_symbol_addrs_path}"
        ovl_config.ovl_symbol_addrs_path.unlink(missing_ok=True)
        
        spinner.message=f"Removing {ovl_config.symexport_path}"
        if ovl_config.symexport_path:
            ovl_config.symexport_path.unlink(missing_ok=True)

        spinner.message=f"Removing {ovl_config.asm_path}"
        if ovl_config.asm_path.exists():
            decomp_utils.shell(f"git clean -fdx {ovl_config.asm_path}")

        spinner.message=f"Removing {ovl_config.build_path / ovl_config.src_path_full}"
        if (path := ovl_config.build_path / ovl_config.src_path_full).exists():
            shutil.rmtree(path)

        spinner.message=f"Removing {ovl_config.ld_script_path}"
        ovl_config.ld_script_path.unlink(missing_ok=True)

        spinner.message=f"Removing {ovl_config.ld_script_path.with_suffix(".elf")}"
        ovl_config.ld_script_path.with_suffix(".elf").unlink(missing_ok=True)

        spinner.message=f"Removing {ovl_config.ld_script_path.with_suffix(".map")}"
        ovl_config.ld_script_path.with_suffix(".map").unlink(missing_ok=True)

        spinner.message=f"Removing {ovl_config.build_path.joinpath(f"{ovl_config.target_path.name}")}"
        ovl_config.build_path.joinpath(f"{ovl_config.target_path.name}").unlink(
            missing_ok=True
        )

        spinner.message=f"Removing {ovl_config.src_path_full}"
        if ovl_config.version != "hd" and ovl_config.src_path_full.exists():
            shutil.rmtree(ovl_config.src_path_full)
        
        spinner.message=f"Removing {ovl_config.config_path}"
        ovl_config.config_path.unlink(missing_ok=True)

        spinner.message=f"cleaned artifacts"
