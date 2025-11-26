#!/usr/bin/env python3

import argparse
import os
import re
import shutil
import time
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
# Todo: Add symbols closer to where the address is gathered
# Todo: Add einit common symbols
# Todo: Add EInits to e_init.c
# Todo: Extract and import BackgroundBlockInit data
# Todo: Extract and import RedDoorTiles data
# Todo: Add g_eRedDoorUV data to e_red_door
# TODO: Add error handling to functions
# TODO: Add SrcAsmPair tools/dups/src/main.rs
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

def main(args, start_time):
    logger.info(f"Starting config generation for {args.version} overlay {args.overlay.upper()}")
    with decomp_utils.Spinner(message=f"generating config for overlay {args.overlay.upper()}") as spinner:
        ovl_config = decomp_utils.SotnOverlayConfig(args.overlay, args.version)
        if ovl_config.config_path.exists() and not args.clean:
            logger.error(
                f"Configuration {ovl_config.name} already exists.  Use the --clean option to remove all existing overlay artifacts and re-extract the overlay."
            )
            raise SystemExit

        sotn_config.clean_artifacts(ovl_config, args.clean, spinner)

### group change ###
        spinner.message=f"creating initial files for overlay {args.overlay.upper()}"
        ovl_config.write_config()
        for symbol_path in ovl_config.symbol_addrs_path:
            symbol_path.touch(exist_ok=True)
        
        sotn_config.create_ovl_include(ovl_config)

### group change ###
        spinner.message = f"adding sha1 hashes to check file"
        sotn_config.add_sha1_hashes(ovl_config)

        # psx rchi and psp bo4 have data values that get interpreted as global symbols, so those symbols need to be defined for the linker
        if ovl_config.name == "rchi" and ovl_config.platform == "psx":
            sotn_config.add_undefined_symbol(ovl_config.version, "PadRead", 0x80015288)
        elif ovl_config.name == "bo4" and ovl_config.platform == "psp":
            sotn_config.add_undefined_symbol(ovl_config.version, "g_Clut", 0x091F5DF8)

### group change ###
        spinner.message = f"performing initial split using config {ovl_config.config_path}"
        decomp_utils.splat_split(ovl_config.config_path, ovl_config.disassemble_all)

### group change ###
        spinner.message = f"adjusting initial include asm paths"
        src_text = ovl_config.first_src_file.read_text()
        adjusted_text = src_text.replace(f'("asm/{args.version}/', '("')
        ovl_config.first_src_file.write_text(adjusted_text)

### group change ###
        spinner.message = f"generating new {ovl_config.version} build plan including {ovl_config.name.upper()}"
        decomp_utils.build(build=False, version=ovl_config.version)

    with decomp_utils.Spinner(message=f"gathering initial symbols") as spinner:
        parsed_symbols = []
        # TODO: rename this
        stage_init = {}
        if ovl_config.platform == "psp":
### group change ###
            spinner.message = f"parsing the psp stage init for symbols"
            asm_path = ovl_config.asm_path.joinpath(ovl_config.nonmatchings_path)
            stage_init, entity_updates = sotn_config.parse_psp_stage_init(asm_path)

            # build symexport lines, but only write if needed
            symexport_lines = []
            if ovl_config.path_prefix:
                symexport_lines.append(f"EXTERN(_binary_assets_{ovl_config.path_prefix}_{ovl_config.name}_mwo_header_bin_start);")
            else:
                symexport_lines.append(f"EXTERN(_binary_assets_{ovl_config.name}_mwo_header_bin_start);")
            if stage_init.get("name"):
                symexport_lines.append(f"EXTERN({stage_init.get("name")});")
            if not ovl_config.symexport_path.exists():
### group change ###
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
### group change ###
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
            if ovl_header.get("symbols"):
                spinner.message = f"creating {ovl_config.name}/header.c"
                sotn_config.create_header_c(ovl_config, ovl_header.get("symbols"))
                spinner.message = f"adding header subsegment"
                data_subseg_index, data_subseg = next(item for item in enumerate(ovl_config.subsegments) if "data" in item[1])
                header_offset = (ovl_header["address"] - ovl_config.vram) + 0x80 if ovl_config.platform == "psp" else 0
                header_subseg = [header_offset, ".data", f"{ovl_config.name}/header" if ovl_config.platform == "psp" else "header"]
                next_offset = header_offset + ovl_header.get("size_bytes", 0) + 4
                new_data_subsegs = []
                if header_offset == data_subseg[0]:
                    new_data_subsegs.append(header_subseg)
                    data_subseg[0] += next_offset
                    first_data_path = first_data_path.with_stem(f"{next_offset:X}.data")
                    new_data_subsegs.append(data_subseg)
                else:
                    new_data_subsegs.append(data_subseg)
                    new_data_subsegs.append(header_subseg)
                    new_data_subsegs.append([next_offset, "data"])
                ovl_config.subsegments[data_subseg_index:data_subseg_index + 1] = [decomp_utils.yaml.FlowSegment(x) for x in new_data_subsegs]
            # TODO: Add data segments for follow-on header symbols
            if ovl_config.platform == "psx":
### group change ###
                spinner.message = f"finding the entity table"
                entity_updates = sotn_config.find_psx_entity_updates(first_data_text, pStObjLayoutHorizontal_address)
        else:
            first_data_text = None
            ovl_header, pStObjLayoutHorizontal_address = {}, None
            entity_updates = {}


### group change ###
        if entity_updates.get("name") and first_data_text:
### group change ###
            spinner.message = f"parsing the entity table for symbols"
            entity_table["address"], entity_table["symbols"] = sotn_config.parse_entity_table(
                first_data_text, ovl_config.name, entity_table.get("name")
            )

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
            parsed_symbols.append(decomp_utils.Symbol(f"{ovl_config.name.upper()}_EntityUpdates", entity_updates.get("address")))
        if ovl_header.get("address"):
            parsed_symbols.append(decomp_utils.Symbol(f"{ovl_config.name.upper()}_Overlay", ovl_header.get("address")))

        if parsed_symbols:
### group change ###
            spinner.message = f"adding {len(parsed_symbols)} parsed symbols and splitting using updated symbols"
            decomp_utils.add_symbols(ovl_config, parsed_symbols)
            add_files = tuple(f"{x}" for x in [ovl_config.config_path, ovl_config.ovl_symbol_addrs_path, ovl_config.symexport_path] if x and x.exists())
            decomp_utils.git("add", add_files)
            decomp_utils.git("clean", ovl_config.asm_path)
            decomp_utils.splat_split(ovl_config.config_path)

    with decomp_utils.Spinner(
        message="gathering reference overlays"
    ) as spinner:
        ref_ovls = []
        for file in Path("config").glob(f"splat.{ovl_config.version}.*.yaml"):
            # TODO: bin/mipsmatch --output mipsmatch-${ovl}.yaml fingerprint build/us/st${ovl}.map build/us/st${ovl}.elf for each reference overlay
            # Todo: Evaluate whether limiting to stage and non-stage overlays as references makes a practical difference in execution time
            if (
                ovl_config.name not in file.name
                and (match := RE_PATTERNS.ref_pattern.match(file.name))
            ):
                prefix = match.group("prefix") or ""
                ref_name = match.group("ref_ovl")
                ld_path = ovl_config.build_path.joinpath(prefix + ref_name).with_suffix(".ld")
                ref_ovls.append(SimpleNamespace(prefix=prefix, name=ref_name, ld_path=ld_path))

        if ref_ovls:
            ref_lds = tuple(ovl.ld_path for ovl in ref_ovls)
            found_elfs = tuple(ovl_config.build_path.glob("*.elf"))
            missing_elfs = tuple(
                ld.with_suffix(".elf")
                for ld in ref_lds
                if ld.with_suffix(".elf") not in found_elfs
            )
            if missing_elfs:
### group change ###
                spinner.message = f"extracting {len(missing_elfs)} missing reference .elf files"
                decomp_utils.build(missing_elfs, plan=True, version=ovl_config.version)

### group change ###
            spinner.message = "extracting dynamic symbols"
            decomp_utils.extract_dynamic_symbols(
                tuple(ld.with_suffix(".elf") for ld in ref_lds), f"build/{args.version}/config/extract_syms.", version=ovl_config.version
            )
            [ld.unlink(missing_ok=True) for ld in ref_lds]
### group change ###
            spinner.message = f"disassembling {len(ref_ovls)} reference overlays"
            decomp_utils.build(ref_lds, dynamic_syms=True, version=ovl_config.version)

### group change ###
            spinner.message=f"finding files to compare"
            ref_files, check_files = [], []
            for dirpath, _, filenames in Path("asm").joinpath(ovl_config.version).walk():
                if any(ovl.name in dirpath.parts or f"{ovl.name}_psp" in dirpath.parts for ovl in ref_ovls):
                    ref_files.extend(
                        dirpath / f
                        for f in filenames
                    )
                if (
                    ovl_config.name in dirpath.parts
                    or f"{ovl_config.name}_psp" in dirpath.parts
                ):
                    check_files.extend(
                        dirpath / f
                        for f in filenames
                        if f.startswith(f"func_{ovl_config.version}_")
                        or f.startswith(f"D_{ovl_config.version}_")
                    )
### group change ###
            spinner.message=f"parsing instructions from {len(check_files)} new files and {len(ref_files)} reference files"
            parsed_files = SimpleNamespace(
                ref_files=decomp_utils.parse_files(ref_files),
                check_files=decomp_utils.parse_files(check_files),
            )
        else:
            parsed_files = None
### group change ###
            spinner.message = f"found no reference overlays"            

    if parsed_files:
        with decomp_utils.Spinner(
            message="searching for similar functions"
        ) as spinner:
            matches = sotn_config.find_symbols(
                parsed_files, ovl_config.version, ovl_config.name, threshold=0.95
            )
### group change ###
            spinner.message = f"Renaming symbols found from {len(matches)} similar functions"
            num_symbols, unhandled_renames = sotn_config.rename_symbols(ovl_config, matches)

        if num_symbols:
### group change ###
            # TODO: Why isn't this showing?
            spinner.message=f"renamed {num_symbols} symbols from {len(matches)} similar functions, splitting again"
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
### group change ###
                spinner.message = f"adding {len(init_room_entities_symbols)} symbols extracted from InitRoomEntities.s"
                decomp_utils.add_symbols(ovl_config, init_room_entities_symbols)

    with decomp_utils.Spinner(
        message="looking for functions for cross referencing"
    ) as spinner:
        parsed_files.check_files = decomp_utils.parse_files(
            dirpath / f
            for dirpath, _, filenames in ovl_config.asm_path.walk()
            for f in filenames
            if not f.startswith(f"func_{ovl_config.version}_")
            and not f.startswith("D_")
        )
        check_files_by_name = {x.path.name: x for x in parsed_files.check_files}
        ref_files_by_name = defaultdict(list)
        for ref_file in parsed_files.ref_files:
            # Todo: Explore the functions that have identical ops, but differing normalized instructions
            if (
                ref_file.path.name in check_files_by_name
                and ref_file.instructions.normalized
                == check_files_by_name[ref_file.path.name].instructions.normalized
            ):
                ref_files_by_name[ref_file.path.name].append(ref_file)

        if check_files_by_name and ref_files_by_name:
### group change ###
            spinner.message = f"cross referencing {len(ref_files_by_name)} functions"
            new_syms = decomp_utils.cross_reference_asm(check_files_by_name, ref_files_by_name, ovl_config.name, ovl_config.version)
            if new_syms:
### group change ###
                spinner.message=f"adding {len(new_syms)} cross referenced symbols and splitting again"
                decomp_utils.add_symbols(ovl_config, tuple(new_syms))
                decomp_utils.git("clean", ovl_config.asm_path)
                decomp_utils.splat_split(ovl_config.config_path, ovl_config.disassemble_all)

    decomp_utils.git("add", [ovl_config.config_path, ovl_config.ovl_symbol_addrs_path])

    with decomp_utils.Spinner(
        message=f"creating {ovl_config.ld_script_path.with_suffix(".elf")}"
    ) as spinner:
        decomp_utils.build(
            [f'{ovl_config.ld_script_path.with_suffix(".elf")}'],
            version=ovl_config.version,
        )

### group change ###
        spinner.message=f"finding segments and splitting source files"
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
    built_bin_path = ovl_config.build_path / ovl_config.target_path.name
    with decomp_utils.Spinner(message=f"building and validating {built_bin_path}"):
        # TODO: Compare generated offsets to .elf segment offsets
        decomp_utils.git("clean", ovl_config.asm_path)
        ovl_config.ld_script_path.unlink(missing_ok=True)
        decomp_utils.build(
            [
                f"{ovl_config.ld_script_path}",
                f"{ovl_config.ld_script_path.with_suffix('.elf')}",
                f"{ovl_config.build_path}/{ovl_config.target_path.name}",
            ],
            version=ovl_config.version,
        )
        if built_bin_path.exists():
            built_sha1 = hashlib.sha1(built_bin_path.read_bytes()).hexdigest()
        else:
            logger.error(f"{built_bin_path} did not build properly")
            raise SystemExit

        if built_sha1 != ovl_config.sha1:
            logger.error(f"{built_bin_path} did not match {ovl_config.target_path}")
            raise SystemExit
        else:
            if ovl_config.symexport_path and ovl_config.symexport_path.exists():
                decomp_utils.git("add", ovl_config.symexport_path)
            decomp_utils.git("add", [ovl_config.config_path, ovl_config.ovl_symbol_addrs_path])

    with decomp_utils.Spinner(message="getting suggested segments"):
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
        spinner.message = "creating extra files"
        sotn_config.create_extra_files(first_data_path.read_text(), ovl_config)
        # with decomp_utils.Spinner(message=f"adding header.c") as spinner:
        # Todo: Build header.c
        # spinner.message = f"adding e_init.c"
        # Todo: Parse final entity table and build e_init.c

    # wrap up
    run_time = time.perf_counter() - start_time
    if run_time < 60:
        time_text = f"{round(run_time % 60, 0)} seconds"
    else:
        minutes = int(run_time // 60)
        seconds = round(run_time % 60, 0)
        minutes_text = f"{minutes}m"
        seconds_text = f"{int(seconds)}s" if seconds else ""
        time_text = f"{minutes_text}{seconds_text}"
    print(f"✅ {args.overlay} ({time_text})")

    if unhandled_renames:
        print(f"\n{len(unhandled_renames)} unhandled match(es) found, see {Path(args.log).relative_to(Path.cwd())} for details")
    if suggested_segments:
        print(f"\n{len(suggested_segments)} additional segments were suggested by Splat:")
        for segment in suggested_segments:
            print(f"    - [{segment[0]}, {segment[1]}, {segment[2]}]")
        logger.info(
            f"Additional segments suggested by splat: {suggested_segments}"
        )


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
        "-l",
        "--log",
        required=False,
        default=f"{Path(__file__).parent / 'logs' / 'sotn_log.json'}",
        help="Use an alternate path for the log file"
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
    logger = decomp_utils.init_logger(filename=args.log)

    if args.remove:
        if (config_path := Path(args.remove)).is_file():
            # TODO: Add loading existing overlay function so that this can work
            ovl_config = decomp_utils.yaml.safe_load(config_path.open())
            sotn_config.clean_artifacts(ovl_config)
        else:
            logger.error(f"{config_path} not found")
            raise SystemExit
    else:
        start_time = time.perf_counter()
        main(args, start_time)
        