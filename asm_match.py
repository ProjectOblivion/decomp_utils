#!/usr/bin/env python3

import json
import decomp_utils
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


def generate_report(clusters, style="single"):
    """Generate a Markdown report of duplicate functions."""

    header = ["Score", "D", "Name", "File Path"]
    divider = "---"
    rows = []
    for i, cluster in enumerate(clusters):
        rows.extend(
            [
                [
                    decomp_utils.bar((item.score - 0.9) * 1000),
                    "√" if item.decomp_status else "X",
                    item.name,
                    f"{f'{"└╴" if item.indent else "":>{item.indent}}'}{item.path}",
                ]
                for item in cluster
            ]
        )
        if i < len(clusters) - 1:
            rows.append(divider)

    return decomp_utils.create_table(rows, header=header, style=style)


def main():
    """Main entry point for the script."""
    arg_parser = decomp_utils.get_argparser(
        description="Generate a report of duplicates"
    )
    arg_parser.add_argument(
        "-o",
        "--overlay",
        required=False,
        type=str,
        action="append",
        help="Specifies which overlays or overlay classes to process",
    )
    arg_parser.add_argument(
        "-f",
        "--output-file",
        required=False,
        type=str,
        help="Specify a file to save report as",
    )
    arg_parser.add_argument(
        "--dump-hashes",
        required=False,
        type=str,
        help="Dump the hashes and filenames to a file",
    )
    arg_parser.add_argument(
        "-d",
        "--dynamic-symbols",
        required=False,
        action="store_true",
        help="Force symbols and extract files",
    )

    args = arg_parser.parse_args()

    # Default to processing all overlays if none are specified
    overlays = args.overlay or ["all"]

    if args.dynamic_symbols:
# NEW #
        with decomp_utils.Spinner(message="extracting dynamic symbols") as spinner:
            lds = tuple(
                Path(decomp_utils.yaml.safe_load(path.open())["options"]["ld_script_path"])
                for path in Path("config").glob(f"splat.{args.version}.*.yaml")
                if "main" not in path.name and "weapon" not in path.name
            )
            
            found_elfs = Path(f"build/{args.version}").glob("*.elf")
            missing_elfs = tuple(
                ld.with_suffix(".elf")
                for ld in lds
                if ld.with_suffix(".elf") not in found_elfs
            )
            if missing_elfs:
                spinner.message = f"extracting {len(missing_elfs)} missing reference .elf files"
                decomp_utils.build(missing_elfs, plan=True, version=args.version)
            spinner.message="extracting dynamic symbols"
            decomp_utils.extract_dynamic_symbols(
                tuple(ld.with_suffix(".elf") for ld in lds), f"build/{args.version}/config/extract_syms.", version=args.version
            )
            [ld.unlink(missing_ok=True) for ld in lds]
            spinner.message="cleaning existing asm"
            decomp_utils.git("clean", f"asm/{args.version}/")

            spinner.message = f"disassembling {len(lds)} overlays"
            decomp_utils.build(lds, dynamic_syms=True, version=args.version)

    if args.dump_hashes:
        files = (
            dirpath / f
            for dirpath, _, filenames in Path("asm").joinpath(args.version).walk()
            if "data" not in dirpath.parts
            for f in filenames
        )
        funcs_by_op_hash = decomp_utils.group_by_hash(
            (func.ops.hash, func) for func in decomp_utils.parse_files(files)
        )
        paths_by_op_hash = {
            k: [f"{x.path}" for x in v] for k, v in funcs_by_op_hash.items()
        }
        Path(args.dump_hashes).write_text(json.dumps(paths_by_op_hash))
    else:
        clusters = decomp_utils.generate_clusters(
            args.version, overlays, threshold=0.95, debug=False
        )
        report = generate_report(clusters)

        if args.output_file:
            Path(args.output_file).write_text(report)
        else:
            print(report)


if __name__ == "__main__":
    main()
