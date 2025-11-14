#!/usr/bin/env python3

import re
import decomp_utils.yaml_ext as yaml
from pathlib import Path
from decomp_utils.helpers import get_logger, shell, RE_TEMPLATES, RE_PATTERNS
from collections import namedtuple

__all__ = [
    "sort_symbols_files",
    "sort_symbols_file",
    "symbols_sort",
    "remove_orphans_from_config",
    "add_symbols",
    "get_symbol_offset",
    "get_symbols",
    "extract_dynamic_symbols",
    "Symbol",
]

Symbol = namedtuple("Symbol", ["name", "address"])

logger = get_logger()


def sort_symbols_files(symbols_files):
    # Only write if there are changes
    for symbols_file in symbols_files:
        sort_symbols_file(symbols_file)

def sort_symbols_file(symbols_file):
    symbol_lines = symbols_file.read_text().splitlines()
    if symbol_lines:
        new_lines = sorted(symbol_lines, key=symbols_sort)
        if new_lines != symbol_lines:
            symbols_file.write_text(f"{"\n".join(new_lines)}\n")

# Intended to be used with .sort, .max, and .min methods for interables of symbols in the format of symbol = address; //ignores comments
def symbols_sort(symbol_line):
    # First splits by '=' to get address section, then splits by ';' to strip the ';' and any comments
    return int(symbol_line.split("=")[1].split(";")[0].strip(), 16)


def remove_orphans_from_config(config_path):
    with config_path.open() as config_yaml:
        config = yaml.safe_load(config_yaml)

    symbol_addrs_path = config["options"]["symbol_addrs_path"]
    # Todo: Identify ovl specific symbols file instead of simply taking the last file in the list
    symbol_file = (
        Path(symbol_addrs_path)
        if isinstance(symbol_addrs_path, str)
        else Path(symbol_addrs_path[-1])
    )
    symbol_lines = RE_PATTERNS.symbol_file_line.finditer(symbol_file.read_text())

    symbols = {
        match.group("name"): match.group(0).rstrip("\n") for match in symbol_lines
    }
    asm_path = Path(config["options"]["asm_path"])
    asm_file_list = [
        dirpath / f
        for dirpath, _, filenames in asm_path.walk()
        for f in filenames
        if ".data.s" not in f
    ]
    if not asm_file_list:
        logger.error(
            f"No files found in '{asm_path}' to extract symbols from for '{symbol_file}', making no changes."
        )
        raise SystemExit

    asm_file_list.append(config_path)
    remaining_symbols = set(symbols.keys())
    new_lines = []
    symbols_found = set()
    for asm_file in asm_file_list:
        content = asm_file.read_text()
        for symbol in remaining_symbols:
            symbol_index = content.find(symbol)
            if symbol_index != -1:
                symbol_slice = content[
                    symbol_index - 1 : symbol_index + len(symbol) + 1
                ]
                if (
                    "ignore:true" in symbols[symbol]
                    or "allow-duplicates:true" in symbols[symbol]
                    or re.search(rf"\b{symbol}\b", symbol_slice)
                ):
                    new_lines.append(symbols[symbol])
                    symbols_found.add(symbol)
        remaining_symbols -= symbols_found
        if not remaining_symbols:
            break

    if len(new_lines) < len(symbols):
        symbol_file.write_text(f"{"\n".join(sorted(new_lines, key=symbols_sort))}\n")


# TODO: add function to remove conflicts and unneeded symbols from symbols files

def add_symbols(ovl_config, add_symbols):
    # Todo: Adjust this to be able to handle a config passed as a path
    symbols_path = ovl_config.ovl_symbol_addrs_path
    symbols_text = symbols_path.read_text()
    existing_symbols = {
        symbol.group("address"): symbol.group("name")
        for symbol in RE_PATTERNS.existing_symbols.finditer(symbols_text)
    }
    # Any addresses not in the ovl vram address space are global and should not be included in the ovl symbols file
    new_symbols = {
        f"{symbol.address:08X}": symbol.name
        for symbol in add_symbols
        if symbol.address >= ovl_config.vram
        and symbol.name not in symbols_text
        and (
            f"{symbol.address:08X}" not in existing_symbols
            or (
                existing_symbols[f"{symbol.address:08X}"].startswith("D_")
                and existing_symbols[f"{symbol.address:08X}"].startswith("func_")
            )
        )
    }

    if new_symbols:
        new_lines = [
            f"{name} = 0x{address};"
            for address, name in sorted((existing_symbols | new_symbols).items())
        ]
        symbols_path.write_text(f"{"\n".join(new_lines)}\n")

        sym_prefix = ovl_config.symbol_name_format.replace("$VRAM", "")
        pattern = re.compile(
            RE_TEMPLATES.sym_replace.substitute(
                sym_prefix=sym_prefix, symbols_list="|".join(new_symbols.keys())
            )
        )
        for src_file in (
            dirpath / f
            for dirpath, _, filenames in ovl_config.src_path_full.walk()
            for f in filenames
            if f.endswith(".c")
        ):
            src_text = src_file.read_text()
            adjusted_text = pattern.sub(
                lambda match: new_symbols[match.group(1)], src_text
            )
            if adjusted_text != src_text:
                src_file.write_text(adjusted_text)
        if ovl_config.symexport_path and ovl_config.symexport_path.exists():
            adjusted_text = pattern.sub(
                lambda match: new_symbols[match.group(1)],
                ovl_config.symexport_path.read_text(),
            )
            ovl_config.symexport_path.write_text(adjusted_text)


def get_symbol_offset(ovl_config, symbol_name):
    # Todo: Adjust this to be able to handle a config passed as a path
    match = re.search(
        RE_TEMPLATES.find_symbol_by_name.substitute(symbol_name=symbol_name),
        ovl_config.ld_script_path.with_suffix(".map").read_text(),
    )
    if match:
        return int(match.group(1), 16) - ovl_config.vram + ovl_config.start
    else:
        return None


def extract_dynamic_symbols(elf_files, path_prefix = "", version="us"):
    # Excluding pspeu dra because it doesn't play nice with forced symbols currently
    for elf_file in (x for x in elf_files if version != "pspeu" or "dra" not in x.name):
        config_path = Path(f"config/splat.{version}.{elf_file.stem}.yaml")
        config = yaml.safe_load(
            config_path.open()
        )
        dynamic_syms_path = Path(f"{path_prefix}{elf_file.stem}.txt")
        dynamic_syms_path.parent.mkdir(parents=True, exist_ok=True)
        dynamic_config_path = dynamic_syms_path.parent.joinpath(f"{config_path.name}.dyn_syms")
        dynamic_config_path.write_bytes(
        yaml.dump(
                {"options": {"symbol_addrs_path": [dynamic_syms_path]}},
                Dumper=yaml.IndentDumper,
                encoding="utf-8",
                sort_keys=False,
            ))
        excluded_starts = {"LM", "__pad"}
        excluded_ends = {"_START", "_END", "_VRAM", "_data__s"}
        defined_symbols = get_defined_symbols(config["options"]["symbol_addrs_path"])
        defined_sym_by_address = {symbol.address: symbol.name for symbol in defined_symbols}
        dynamic_symbols = []
        for symbol in get_symbols(elf_file, excluded_starts, excluded_ends):
            if symbol not in defined_symbols and symbol.address in defined_sym_by_address:
                logger.warning(f"Symbol conflict detected in {elf_file}! Address 0x{symbol.address:08X} built as {symbol.name}, but defined as {defined_sym_by_address[symbol.address]}") 
            elif symbol not in defined_symbols:
                dynamic_symbols.append(symbol)
        symbols_lines = tuple(f"{symbol.name} = 0x{symbol.address:08X}; // allow_duplicated:True" for symbol in dynamic_symbols)
        dynamic_syms_path.write_text(f"{'\n'.join(symbols_lines)}\n")

def get_defined_symbols(symbols_files):
    if not isinstance(symbols_files, (list, tuple)):
        symbols_files = (symbols_files, )
    split_lines = tuple(line.split("=")
        for file in symbols_files
        for line in Path(file).read_text().splitlines()
        if line
    )
    return tuple(Symbol(line[0].strip(), int(line[1].split(";")[0].strip(), 16)) for line in split_lines)

def get_symbols(
    file_path,
    excluded_starts=["LM", "__pad"],
    excluded_ends=["_END", "_START", "_VRAM"],
    no_default=False,
):
    symbols = set()
    match file_path.suffix:
        case ".map":
            text = file_path.read_text()
            matches = RE_PATTERNS.map_symbol.finditer(text)
        case ".elf":
            text = shell(f"nm {file_path}").decode()
            matches = RE_PATTERNS.elf_symbol.finditer(text)
        case _:
            raise SystemError(
                "File to extract symbols from must be either .elf or .map"
            )

    for match in matches:
        symbol_name = match.group("name")
        if not no_default or (
            not symbol_name.startswith("func_") and not symbol_name.startswith("D_")
        ):
            filters = "_compiled" not in symbol_name and not symbol_name.endswith("_c")
            if (
                filters
                and not any(symbol_name.startswith(x) for x in excluded_starts)
                and not any(symbol_name.endswith(x) for x in excluded_ends)
            ):
                symbols.add(Symbol(symbol_name, int(match.group("address"), 16)))

    return sorted(symbols, key=lambda x: x.address)
