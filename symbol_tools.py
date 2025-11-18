import decomp_utils
from pathlib import Path


def main(args):
    match args.command:
        case "sort":
            symbols_path = Path(args.symbols_path)
            if symbols_path.is_dir():
                symbols_files = tuple(
                    file
                    for file in symbols_path.iterdir()
                    if args.version in file.name
                    and (
                        file.name.startswith("symbols.")
                        or file.name.startswith("undefined_syms.")
                    )
                    and file.suffix == ".txt"
                )
            elif symbols_path.is_file():
                symbols_files = (symbols_path,)
            else:
                raise FileNotFoundError(f"{args.symbols_path} does not exist")
            decomp_utils.sort_symbols_files(symbols_files)
        case "clean":
            logger = decomp_utils.init_logger(console_level=20)
            build_files = {"us": [], "pspeu": [], "hd": []}
            for config_file in args.config_file:
                _, version, basename, _ = config_file.split(".")
                ld_path = Path(f"build/{version}/{basename}.ld")
                if not ld_path.exists():
                    build_files[version].append(ld_path)
                if "weapon" not in config_file and not ld_path.with_suffix(".elf").exists():
                    build_files[version].append(ld_path.with_suffix(".elf"))

            for version, files in build_files.items():
                if files:
                    logger.info(f"Building {[f'{f}' for f in files]}")
                    decomp_utils.build(files, version=version)

            configs = [Path(file) for file in args.config_file]
            remove = not args.warn_orphans
            decomp_utils.clean_orphans(configs, remove)
            remove = args.remove_conflicts
            decomp_utils.clean_conflicts(configs, remove)
        case "parse":
            excluded_starts = {"LM", "__pad"}
            excluded_ends = {"_START", "_END", "_VRAM"}

            if args.no_default:
                excluded_starts |= {"D_", "func_", "jpt_", "jtbl_"}

            symbols = decomp_utils.get_symbols(
                args.file_name, list(excluded_starts), list(excluded_ends)
            )

            if args.output:
                Path(args.output).write_text(
                    f"{"\n".join(tuple(f"{symbol.name} = 0x{symbol.address:08X}; // allow_duplicated:True" for symbol in symbols))}\n"
                )
            else:
                for symbol in symbols:
                    print(
                        f"{symbol.name} = 0x{symbol.address:08X}; // allow_duplicated:True"
                    )
        case "force":
            decomp_utils.extract_dynamic_symbols(
                tuple(Path(x) for x in args.elf_file), f"build/{args.version}/config/dyn_syms.", version=args.version
            )
        case _:
            print("Unknown command. Use --help for usage information.")


if __name__ == "__main__":
    parser = decomp_utils.get_argparser(
        description="Perform operations on game symbols"
    )
    subparsers = parser.add_subparsers(dest="command")
    # Todo: Clean up arguments
    sort_parser = subparsers.add_parser(
        "sort",
        description="Sort all the symbols of a given GNU LD script by their offset",
    )
    sort_parser.add_argument(
        "symbols_path",
        help="A directory of symbols files or a single symbols file to sort",
    )

    dynamic_symbols_parser = subparsers.add_parser(
        "dynamic",
        description="Extract dynamic symbols using specified configs",
    )
    dynamic_symbols_parser.add_argument(
        "config",
        nargs="+",
        help="The config(s) to use to extract dynamic symbols",
    )

    clean_syms_parser = subparsers.add_parser(
        "clean",
        description="Clean symbol files"
    )
    clean_syms_parser.add_argument(
        "config_file",
        nargs="+",
        help="The config file for the overlay to clean the symbols file(s) for",
    )
    orphans_group = clean_syms_parser.add_mutually_exclusive_group(required=False)
    orphans_group.add_argument(
        "--remove-orphans",
        action="store_true",
        help="Remove all symbols that are not referenced (default)"
    )
    orphans_group.add_argument(
        "--warn-orphans",
        action="store_true",
        help="Show warnings for all symbols that are not referenced"
    )
    conflicts_group = clean_syms_parser.add_mutually_exclusive_group(required=False)
    conflicts_group.add_argument(
        "--remove-conflicts",
        action="store_true",
        help="Remove all symbols that are built with one name, but defined as a different name"
    )
    conflicts_group.add_argument(
        "--warn-conflicts",
        action="store_true",
        help="Show warnings for all symbols that are built with one name, but defined as a different name (default)"
    )

    parse_parser = subparsers.add_parser(
        "parse",
        description="Parse the symbols from an elf or map file",
    )
    parse_parser.add_argument(
        "file_name",
        help="The file to parse symbols from",
    )
    parse_parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output parsed symbols to specified file, rather than stdout",
        required=False,
    )
    parse_parser.add_argument(
        "--no-default",
        required=False,
        action="store_true",
        help="Do not include symbols that start with D_ or func_",
    )

    args = parser.parse_args()
    main(args)
