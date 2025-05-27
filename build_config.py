#!/usr/bin/env python3

import argparse
import os
import re
import shutil
import decomp_utils
import hashlib
from collections import defaultdict
from pathlib import Path
from box import Box
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

Example usage: python3 tools/decomp_utils/generate_config.py lib --version=us
"""


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

    with decomp_utils.Spinner(message="creating initial files") as spinner:
        ovl_config.write_config()
        for symbol_path in ovl_config.symbol_addrs_path:
            symbol_path.touch(exist_ok=True)
        header_path = (
            ovl_config.src_path_full.with_name(ovl_config.name) / f"{ovl_config.name}.h"
        )
        ovl_header_text = decomp_utils.get_default("ovl.h").format(
            ovl_name=ovl_config.name.upper()
        )
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
            check_file_path.write_text(f"{"\n".join(new_lines)}\n")
        decomp_utils.shell(f"git add {check_file_path}")

        spinner.message = f"performing initial split with {ovl_config.config_path}"
        decomp_utils.splat_split(ovl_config.config_path, ovl_config.disassemble_all)
        src_text = ovl_config.first_src_file.read_text()
        adjusted_text = src_text.replace(f'("asm/{args.version}/', '("')
        ovl_config.first_src_file.write_text(adjusted_text)
        decomp_utils.build(build=False, version=ovl_config.version)

    with decomp_utils.Spinner(message=f"gathering initial symbols") as spinner:
        header_symbols, entity_table_symbols, export_table_symbols = None, None, None
        entity_table_symbol, export_table_symbol = None, None
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
        if ovl_config.platform == "psx" and first_data_text:
            spinner.message = f"parsing the psx header for symbols"
            pStObjLayoutHorizontal_address, header_symbols = (
                decomp_utils.parse_psx_header(ovl_config.name, first_data_text)
            )

        if ovl_config.platform == "psp":
            spinner.message = f"parsing the psp stage init for symbols"
            stage_init, export_table_symbol, entity_table_symbol = (
                decomp_utils.parse_psp_stage_init(
                    ovl_config.asm_path.joinpath(ovl_config.nonmatchings_path)
                )
            )
            if stage_init and not ovl_config.symexport_path.exists():
                spinner.message = "creating symexport file"
                symexport_text = f"EXTERN(_binary_assets_{ovl_config.ovl_prefix}{"_" if ovl_config.ovl_prefix else ""}{ovl_config.name}_mwo_header_bin_start);\n"
                symexport_text += f"EXTERN({stage_init});\n"
                ovl_config.symexport_path.write_text(symexport_text)
                decomp_utils.shell(f"git add {ovl_config.symexport_path}")
        else:
            spinner.message = f"looking for the entity table"
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
            entity_table_address, entity_table_symbols = (
                decomp_utils.parse_entity_table(
                    ovl_config.name, entity_table_symbol, first_data_text
                )
            )

        if export_table_symbol:
            spinner.message = f"parsing the export table for symbols"
            export_table_symbols = decomp_utils.parse_export_table(
                ovl_config.ovl_type, export_table_symbol, first_data_text
            )

        if header_symbols or entity_table_symbols or export_table_symbols:
            parsed_symbols = tuple(
                symbol
                for symbols in (
                    header_symbols,
                    entity_table_symbols,
                    export_table_symbols,
                )
                if symbols is not None
                for symbol in symbols
            ) + (
                decomp_utils.Symbol(
                    f"{ovl_config.name.upper()}_EntityUpdates", entity_table_address
                ),
            )
            spinner.message = f"adding {len(parsed_symbols)} parsed symbols and splitting using updated symbols"
            decomp_utils.add_symbols(ovl_config, parsed_symbols)
            decomp_utils.shell(
                f"git add {ovl_config.config_path} {ovl_config.ovl_symbol_addrs_path}"
            )
            decomp_utils.shell(f"git clean -fdx {ovl_config.asm_path}")
            decomp_utils.splat_split(ovl_config.config_path)

    with decomp_utils.Spinner(
        message="extracting reference symbols from .elf files"
    ) as spinner:
        if elf_files := ovl_config.build_path.glob("*.elf"):
            decomp_utils.force_symbols(
                ovl_config.version,
                tuple(x for x in elf_files if ovl_config.name not in x.name),
            )
        else:
            logger.error(
                f"No elf files found in {ovl_config.build_path}.  Rerun this tool after a successful build."
            )
            raise SystemExit

    with decomp_utils.Spinner(message="disassembling reference functions"):
        decomp_utils.shell(
            f"git clean -fdx asm/{ovl_config.version}/ -e {ovl_config.asm_path}"
        )
        ref_lds, ref_ovls = [], []
        for file in Path("config").iterdir():
            ref_version = (
                "pspeu" in file.name
                if ovl_config.platform == "psp"
                else "us" in file.name or "hd" in file.name
            )
            ref_pattern = re.compile(
                rf"splat\.\w+\.(?P<prefix>st|bo)?(?P<ref_ovl>\w+)\.yaml"
            )
            # Todo: This should probably differentiate between stage and non-stage to reduce execution time
            if (
                ref_version
                and "main" not in file.name
                and "weapon" not in file.name
                and "mad" not in file.name
                and ovl_config.name not in file.name
                and file.match("splat.*.yaml")
                and (match := ref_pattern.match(file.name))
            ):
                ref_lds.append(
                    ovl_config.build_path.joinpath(
                        f'{match.group("prefix") or ""}{match.group("ref_ovl")}'
                    ).with_suffix(".ld")
                )
                ref_ovls.append(match.group("ref_ovl"))

        decomp_utils.build(ref_lds, plan=False, version=ovl_config.version)

        # Removes forced symbols files
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
        parsed = Box(
            ref_files=decomp_utils.parse_files(ref_files),
            check_files=decomp_utils.parse_files(check_files),
        )

    with decomp_utils.Spinner(
        message="finding symbol names using reference overlays"
    ) as spinner:
        matches = decomp_utils.find_symbols(
            parsed, ovl_config.version, ovl_config.name, threshold=0.95
        )
        num_symbols = decomp_utils.rename_symbols(ovl_config, matches)

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
    if init_room_entities_path.exists():
        with decomp_utils.Spinner(
            message=f"parsing InitRoomEntities.s for symbols"
        ) as spinner:
            init_room_entities_symbols, create_entity_bss_address = (
                decomp_utils.parse_init_room_entities(
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
            name_pattern = re.compile(r"lui\s+.+?%hi\(((?:[A-Z]|g_|func_)\w+)\)")
            address_pattern = re.compile(
                r"lui\s+.+?%hi\((?:D_|func_)(?:\w+_)?([A-F0-9]{8})\)"
            )

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
                        name, address = name_pattern.match(
                            ref_instruction
                        ), address_pattern.match(check_instruction)
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
                                    re.sub(
                                        r"^[A-Z0-9]{3,4}_(\w+)",
                                        rf"{ovl_config.name.upper()}_\1",
                                        name.group(1),
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
        segments, rodata_segments = decomp_utils.find_segments(ovl_config)
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
        splat_suggestions = re.finditer(
            r"""
            The\srodata\ssegment\s'(?P<segment>\w+)'\shas\sjumptables.+\n
            File\ssplit\ssuggestions.+\n
            (?P<suggestions>(?:\s+-\s+\[0x[0-9A-Fa-f]+,\s.+?\]\n)+)
            \n
            """,
            output,
            re.VERBOSE,
        )

        suggested_segments = []
        for match in splat_suggestions:
            suggestions = re.findall(
                r"\s+-\s+\[(0x[0-9A-Fa-f]+),\s(.+?)\]", match.group("suggestions")
            )
            suggested_segments.extend(
                [offset, segment_type, match.group("segment")]
                for offset, segment_type in suggestions
            )
        if suggested_segments:
            # Todo: Improve logging formatting
            logger.info(f"Additional segments suggested by splat: {suggested_segments}")

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
            decomp_utils.shell(
                f"git add {ovl_config.config_path} {ovl_config.ovl_symbol_addrs_path}"
            )

    #with decomp_utils.Spinner(message=f"adding header.c") as spinner:
        # Todo: Build header.c
        #spinner.message = f"adding e_init.c"
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
    # Todo: Put this closer to where the multiprocessing is happening.
    multiprocessing.log_to_stderr()
    multiprocessing.set_start_method("spawn")
    global args
    args = parser.parse_args()
    global logger
    logger = decomp_utils.get_logger()

    main(args)
