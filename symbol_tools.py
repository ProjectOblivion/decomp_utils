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
        case "remove-orphans":
            decomp_utils.remove_orphans_from_config(Path(args.config_file))
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

    force_parser = subparsers.add_parser(
        "force",
        description="Sort all the symbols of a given GNU LD script by their offset",
    )
    force_parser.add_argument(
        "elf_file",
        nargs="+",
        help="An overlay to force symbols for",
    )

    remove_orphans_parser = subparsers.add_parser(
        "remove-orphans",
        description="Remove all symbols that are not referenced from a specific group of assembly code",
    )
    remove_orphans_parser.add_argument(
        "config_file",
        help="The config file for the overlay to remove the orphan symbols from",
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
