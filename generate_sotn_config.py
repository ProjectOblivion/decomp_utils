#!/usr/bin/env python3

import argparse
import os
import re
import shutil
import decomp_utils
from decomp_utils import sotn_config
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
# TODO: Add error handling to functions

def create_ovl_include(ovl_config):
    ovl_include_path = (
        ovl_config.src_path_full.with_name(ovl_config.name) / f"{ovl_config.name}.h"
    )
    template = Template(Path("tools/decomp_utils/templates/ovl.h.mako").read_text())
    ovl_header_text = template.render(
        ovl_name=ovl_config.name,
        ovl_type=ovl_config.ovl_type,
        e_inits=None,
    )
    if not ovl_include_path.exists():
        ovl_include_path.parent.mkdir(parents=True, exist_ok=True)
        ovl_include_path.write_text(ovl_header_text)

def add_sha1_hashes(ovl_config):
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
        sorted_lines = sorted(new_lines, key=lambda x: sotn_config.ovl_sort(x.split()[-1]))
        check_file_path.write_text(f"{"\n".join(sorted_lines)}\n")

    decomp_utils.git("add", check_file_path)

def find_psx_entity_table(first_data_text, pStObjLayoutHorizontal_address = None):
    # TODO: Find a less complicated way to handle this
    # we know that the entity table is always after the ovl header
    end_of_header = first_data_text.find(".size")
    # use the address of pStObjLayoutHorizontal if it was parsed from the header to reduce the amount of data we're searching through
    if pStObjLayoutHorizontal_address:
        start_index = first_data_text.find(
            f"{pStObjLayoutHorizontal_address:08X}", end_of_header
        )
    else:
        logger.warning("No address for found for pStObjLayoutHorizontal, starting at end of header")
        start_index = end_of_header

    # the first entity referenced after the ovl header, which should be the first element of the entity table
    first_entity_index = first_data_text.find(
        " func_", start_index
    )
    # the last glabel before the first function pointer should be the entity table symbol
    entity_table_index = first_data_text.rfind(
        "glabel", start_index, first_entity_index
    )
    # this is just a convoluted way of extracting the entity table symbol name
    # get the second word of the first line, which should be the entity table symbol name
    return {"name": first_data_text[entity_table_index:first_entity_index].splitlines()[0].split()[1]}

def add_undefined_symbol(version, symbol, address):
    symbol_line = f"{symbol}{' '*13}= 0x{address:X};"
    undefined_syms = Path(f"config/undefined_syms.{version}.txt")
    undefined_syms_lines = undefined_syms.read_text().splitlines()
    if symbol_line not in undefined_syms_lines:
        new_lines = sorted(undefined_syms_lines, key=decomp_utils.symbols_sort)
        undefined_syms.write_text("\n".join(new_lines))
        decomp_utils.git("add", undefined_syms)

def main(args):
    logger.info("Starting...")
    with decomp_utils.Spinner(message="generating config"):
        ovl_config = decomp_utils.SotnOverlayConfig(args.overlay, args.version)

    if ovl_config.config_path.exists() and not args.clean:
        logger.error(
            f"A configuration for {ovl_config.name} already exists.  Use the --clean option to remove all existing overlay artifacts and re-extract the overlay."
        )
        raise SystemExit
    if args.clean:
        sotn_config.clean_artifacts(ovl_config)

    with decomp_utils.Spinner(message="creating initial files") as spinner:
        ovl_config.write_config()
        for symbol_path in ovl_config.symbol_addrs_path:
            symbol_path.touch(exist_ok=True)
        
        create_ovl_include(ovl_config)

        spinner.message = f"adding sha1 hashes to check file"
        add_sha1_hashes(ovl_config)

        # psx rchi and psp bo4 have data values that get interpreted as global symbols, so those symbols need to be defined for the linker
        if ovl_config.name == "rchi" and ovl_config.platform == "psx":
            add_undefined_symbol(ovl_config.version, "PadRead", 0x80015288)
        elif ovl_config.name == "bo4" and ovl_config.platform == "psp":
            add_undefined_symbol(ovl_config.version, "g_Clut", 0x091F5DF8)

        spinner.message = f"performing initial split with {ovl_config.config_path}"
        decomp_utils.splat_split(ovl_config.config_path, ovl_config.disassemble_all)

        spinner.message = f"adjusting initial include asm paths"
        src_text = ovl_config.first_src_file.read_text()
        adjusted_text = src_text.replace(f'("asm/{args.version}/', '("')
        ovl_config.first_src_file.write_text(adjusted_text)

        spinner.message = f"generating new build plan including {ovl_config.name.upper()}"
        decomp_utils.build(build=False, version=ovl_config.version)

    with decomp_utils.Spinner(message=f"gathering initial symbols") as spinner:
        parsed_symbols = []
        # TODO: rename this
        stage_init = {}
        if ovl_config.platform == "psp":
            spinner.message = f"parsing the psp stage init for symbols"
            asm_path = ovl_config.asm_path.joinpath(ovl_config.nonmatchings_path)
            stage_init, entity_table = sotn_config.parse_psp_stage_init(asm_path)

            # build symexport lines, but only write if needed
            symexport_lines = []
            if ovl_config.path_prefix:
                symexport_lines.append(f"EXTERN(_binary_assets_{ovl_config.path_prefix}_{ovl_config.name}_mwo_header_bin_start);")
            else:
                symexport_lines.append(f"EXTERN(_binary_assets_{ovl_config.name}_mwo_header_bin_start);")
            if stage_init.get("name"):
                symexport_lines.append(f"EXTERN({stage_init.get("name")});")
            if not ovl_config.symexport_path.exists():
                spinner.message = "creating symexport file"
                ovl_config.symexport_path.write_text("\n".join(symexport_lines))
                decomp_utils.git("add", ovl_config.symexport_path)

            if stage_init.get("address"):
                decomp_utils.Symbol(f"{ovl_config.name.upper()}_Load", stage_init.get("address"))
                parsed_symbols.append(decomp_utils.Symbol(f"{ovl_config.name.upper()}_Load", stage_init.get("address")))

        first_data_offset = next(subseg[0] for subseg in ovl_config.subsegments if "data" in subseg)
        first_data_path = ovl_config.asm_path / "data" / f"{first_data_offset:X}.data.s"
        if first_data_path.exists():
            first_data_text = first_data_path.read_text()
            spinner.message = f"parsing the overlay header for symbols"
            ovl_header, pStObjLayoutHorizontal_address = (
                sotn_config.parse_ovl_header(
                    first_data_text,
                    ovl_config.name,
                    ovl_config.platform,
                    ovl_config.ovl_type,
                    stage_init.get("ovl_header"),
                )
            )
            if ovl_config.platform == "psx":
                spinner.message = f"finding the entity table"
                entity_table = find_psx_entity_table(first_data_text, pStObjLayoutHorizontal_address)
        else:
            first_data_text = None
            ovl_header, pStObjLayoutHorizontal_address = {}, None
            entity_table = {}


    with decomp_utils.Spinner(message=f"gathering initial symbols") as spinner:
        if entity_table.get("name") and first_data_text:
            spinner.message = f"parsing the entity table for symbols"
            entity_table["address"], entity_table["symbols"] = sotn_config.parse_entity_table(
                first_data_text, ovl_config.name, entity_table.get("name")
            )

        if ovl_header.get("symbols") or entity_table.get("symbols"):
            parsed_symbols.extend((
                symbol
                for symbols in (
                    ovl_header.get("symbols"),
                    entity_table.get("symbols"),
                )
                if symbols is not None
                for symbol in symbols
            ))
        if entity_table.get("address"):
            parsed_symbols.append(decomp_utils.Symbol(f"{ovl_config.name.upper()}_EntityUpdates", entity_table.get("address")))
        if ovl_header.get("address"):
            parsed_symbols.append(decomp_utils.Symbol(f"{ovl_config.name.upper()}_Overlay", ovl_header.get("address")))

        if parsed_symbols:
            spinner.message = f"adding {len(parsed_symbols)} parsed symbols and splitting using updated symbols"
            decomp_utils.add_symbols(ovl_config, parsed_symbols)
            decomp_utils.git("add", [ovl_config.config_path, ovl_config.ovl_symbol_addrs_path])
            if ovl_config.symexport_path and ovl_config.symexport_path.exists():
                decomp_utils.git("add", ovl_config.symexport_path)
            decomp_utils.git("clean", ovl_config.asm_path)
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
            # TODO: bin/mipsmatch --output mipsmatch-${ovl}.yaml fingerprint build/us/st${ovl}.map build/us/st${ovl}.elf for each reference overlay
            # Todo: Evaluate whether limiting to stage and non-stage overlays as references makes a practical difference in execution time
            if (
                ref_version
                and ovl_config.ovl_type != "weapon"
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
            elif (
                ref_version
                and ovl_config.ovl_type == "weapon"
                and "weapon" in file.name
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
        decomp_utils.git("clean", [f"asm/{ovl_config.version}/", f"-e {ovl_config.asm_path}"])
        if ref_basenames:
            decomp_utils.build(ref_lds, plan=False, version=ovl_config.version)

        # Removes forced symbols files
        # Todo: checkout each file instead of the whole dir
        decomp_utils.git("checkout", "config/")

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
        matches = sotn_config.find_symbols(
            parsed, ovl_config.version, ovl_config.name, threshold=0.95
        )
        num_symbols = sotn_config.rename_symbols(ovl_config, matches)

    if num_symbols:
        with decomp_utils.Spinner(
            message=f"renamed {num_symbols} symbols, splitting again"
        ):
            decomp_utils.git("clean", ovl_config.asm_path)
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
                sotn_config.parse_init_room_entities(
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

    if ref_files_by_name and new_syms:
        with decomp_utils.Spinner(
            message=f"adding {len(new_syms)} cross referenced symbols and splitting again"
        ):
            decomp_utils.add_symbols(ovl_config, tuple(new_syms))
            decomp_utils.git("clean", ovl_config.asm_path)
            decomp_utils.splat_split(ovl_config.config_path, ovl_config.disassemble_all)

    with decomp_utils.Spinner(message="staging files"):
        decomp_utils.git("add", [ovl_config.config_path, ovl_config.ovl_symbol_addrs_path])

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
        segments, rodata_segments = sotn_config.find_segments(ovl_config)
        text_subsegs = [
            decomp_utils.yaml.FlowSegment([segment.offset.int, "c", segment.name])
            for segment in segments
        ]
        rodata_subsegs = [decomp_utils.yaml.FlowSegment(x) for x in rodata_segments]
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

    ovl_config.write_config()

    with decomp_utils.Spinner(
        message=f"extracting overlay to validate configuration"
    ) as spinner:
        # Todo: Compare generated offsets to .elf segment offsets
        decomp_utils.git("clean", ovl_config.asm_path)
        ovl_config.ld_script_path.unlink(missing_ok=True)
        decomp_utils.build(
            [f"{ovl_config.ld_script_path}"], version=ovl_config.version
        )
        output = decomp_utils.splat_split(ovl_config.config_path)
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
        sotn_config.create_extra_files(first_data_path.read_text(), ovl_config)

    built_bin = ovl_config.build_path / f"{ovl_config.target_path.name}"
    with decomp_utils.Spinner(message=f"building and validating {built_bin}"):
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
                decomp_utils.git("add", ovl_config.symexport_path)
            decomp_utils.git("add", [ovl_config.config_path, ovl_config.ovl_symbol_addrs_path])

    # with decomp_utils.Spinner(message=f"adding header.c") as spinner:
    # Todo: Build header.c
    # spinner.message = f"adding e_init.c"
    # Todo: Parse final entity table and build e_init.c


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create initial configuration for new overlays"
    )
    required_args = parser.add_mutually_exclusive_group(required=True)
    required_args.add_argument(
        "overlay",
        nargs="?",
        help="Name of the overlay to create a configuration for",
    )
    required_args.add_argument(
        "--remove",
        type=str,
        metavar="CONFIG",
        help="DESTRUCTIVE: Use the specified config file to remove an overlay and all associated artifacts",
    )
    parser.add_argument(
        "-v",
        "--version",
        required=False,
        default=os.getenv("VERSION") or "us",
        help="The version of the game to target",
    )
    parser.add_argument(
        "--clean",
        required=False,
        action="store_true",
        help="DESTRUCTIVE: Remove any existing overlay artifacts before re-extracting the overlay from the source binary",
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

    if args.remove:
        if (config_path := Path(args.remove)).is_file():
            # TODO: Add loading existing overlay function so that this can work
            ovl_config = decomp_utils.yaml.safe_load(config_path.open())
            sotn_config.clean_artifacts(ovl_config)
        else:
            logger.error(f"{config_path} not found")
            raise SystemExit
    else:
        main(args)
