#!/usr/bin/env python3

import re
import sotn_utils.yaml_ext as yaml
from pathlib import Path
from sotn_utils.helpers import get_logger, shell
from sotn_utils.regex import RE_TEMPLATES, RE_PATTERNS
from collections import namedtuple

__all__ = [
    "sort_symbols_files",
    "symbols_sort",
    "clean_orphans",
    "find_orphans",
    "clean_conflicts",
    "find_conflicts",
    "add_symbols",
    "get_symbol_address",
    "get_symbol_offset",
    "get_symbols",
    "extract_dynamic_symbols",
    "Symbol",
    "add_undefined_symbol",
    "cross_reference_asm",
]

Symbol = namedtuple("Symbol", ["name", "address"])

logger = get_logger()


def sort_symbols_files(symbols_files):
    if isinstance(symbols_files, (str,Path)):
        symbols_files = (Path(symbols_files), )
    for symbols_file in symbols_files:
        symbol_lines = symbols_file.read_text().splitlines()
        if symbol_lines:
            new_lines = sorted(symbol_lines, key=symbols_sort)
            new_lines = [line for i,line in enumerate(new_lines) if i == len(new_lines) - 1 or line != new_lines[i+1]]
            if new_lines != symbol_lines:
                symbols_file.write_text(f"{"\n".join(new_lines)}\n")

# Intended to be used with .sort, .max, and .min methods for interables of symbols in the format of symbol = address; // ignores comments
def symbols_sort(symbol_line):
    # First splits by '=' to get address section, then splits by ';' to strip the ';' and any comments
    return int(symbol_line.split("=")[1].split(";")[0].strip(), 16)

def cross_reference_asm(check_files_by_name, ref_files_by_name, ovl_name, version):
    # TODO: Clean up this logic
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
                        f"func_{version}_"
                    )
                    and address
                ):
                    new_syms.add(
                        Symbol(
                            RE_PATTERNS.symbol_ovl_name_prefix.sub(
                                ovl_name.upper(), name.group(1)
                            ),
                            int(address.group(1), 16),
                        )
                    )

    return new_syms

def clean_orphans(configs, remove=False):
    if not isinstance(configs, (list, tuple)):
        configs = [configs]
    start_col = max([len(x.name) for x in configs]) + 2
    for config_path in configs:
        with config_path.open() as config_yaml:
            config = yaml.safe_load(config_yaml)

        basename = config["options"]["basename"]
        asm_path = Path(config["options"]["asm_path"])
        symbol_addrs_path = config["options"]["symbol_addrs_path"]

        data_parts = []
        for segment in config["segments"]:
            if (not "type" in segment or segment["type"] == "code") and "subsegments" in segment:
                for subsegment in segment["subsegments"]:
                    if isinstance(subsegment, list) and len(subsegment) > 2:
                        data_parts.append([subsegment[2], subsegment[1]])
                    elif isinstance(subsegment, list) and len(subsegment) == 2:
                        data_parts.append([f"{subsegment[0]:X}", subsegment[1]])
                    elif isinstance(subsegment, dict) and "name" in subsegment and "type" in subsegment:
                        data_parts.append([subsegment["name"], subsegment["type"]])
                    elif isinstance(subsegment, dict) and "start" in subsegment and "type" in subsegment:
                        data_parts.append([f"{subsegment["start"]:X}", subsegment["type"]])
            elif "name" in segment and "type" in segment:
                data_parts.append([segment["name"], segment["type"]])
            
        data_files = [Path(f"{parts[0]}.{parts[1]}.s").name for parts in data_parts if not parts[1].startswith(".") and (parts[1].endswith("data") or parts[1].endswith("bss"))]

        if isinstance(symbol_addrs_path, list):
            symbol_files = tuple(
                Path(path)
                for path in symbol_addrs_path
                if basename in path
            )
        else:
            symbol_files = tuple(Path(symbol_addrs_path))

        asm_files = [
            dirpath / f
            for dirpath, _, filenames in asm_path.walk()
            for f in filenames
            if "matchings" not in dirpath.parts
            and ("data" not in dirpath.parts or f in data_files)
        ]

        orphans = find_orphans(asm_files, symbol_files)

        if remove:
            for symbol_file in symbol_files:
                file_text = symbol_file.read_text().rstrip("\n")
                if not file_text:
                    logger.debug(f"{symbol_file.name}: empty symbol file")
                elif asm_files or orphans:
                    lines = file_text.splitlines()
                    new_lines = [line for line in lines if line and line.split()[0] not in orphans]
                    if (removed := len(lines) - len(new_lines)):
                        symbol_file.write_text("\n".join(new_lines) + "\n")
                    message = f"{symbol_file.name + ':':<{start_col}} removed: {removed:<4} remaining: {len(new_lines):<4} (searched {len(asm_files):<3} files)"

                    logger.info(message) if asm_files else logger.warning(message)
        else:
            logger.info(f"{config_path.name}: found {len(orphans)} unneeded symbol file entries (searched {len(asm_files)} files):")
            for orphan in orphans:
                logger.info(f"    {orphan}")


def find_orphans(asm_files, symbol_files):
    symbol_lines = {}
    for symbol_file in symbol_files:
        file_text = symbol_file.read_text().rstrip("\n")
        symbol_lines[symbol_file] = tuple(line.split() for line in file_text.splitlines())

    symbols_by_name = {}
    for file, lines in symbol_lines.items():
        symbols_by_name[file] = {fields[0]: {"comment": " ".join(fields[3:]).lstrip("/ ") if len(fields) > 3 else "", "line": " ".join(fields)} for fields in lines}
    
    symbols = {file: set(symbols_by_name[file].keys()) for file in symbols_by_name}
    orphans = {symbol for file in symbols for symbol in symbols[file] if "used:true" not in symbols_by_name[file][symbol]["comment"] and "ignore:true" not in symbols_by_name[file][symbol]["comment"] and "allow-duplicates:true" not in symbols_by_name[file][symbol]["comment"] and "allow_duplicated:True" not in symbols_by_name[file][symbol]["comment"]}
    pattern = re.compile(r"(?:glabel|\.word|jal|j|%(?:hi|lo)\()\s*([A-Z_a-z]\w+)")
    for asm_file in asm_files:
        asm_text = asm_file.read_text()
        orphans -= {match.group(1) for match in pattern.finditer(asm_text)}
        if not orphans:
            break

    return orphans

def clean_conflicts(configs, remove=False):
    if not isinstance(configs, (list, tuple)):
        configs = [configs]
    for config_path in configs:
        with config_path.open() as config_yaml:
            config = yaml.safe_load(config_yaml)

        basename = config["options"]["basename"]
        symbol_addrs_path = config["options"]["symbol_addrs_path"]

        if isinstance(symbol_addrs_path, list):
            symbol_files = tuple(
                Path(path)
                for path in symbol_addrs_path
                if basename in path
            )
        else:
            symbol_files = tuple(Path(symbol_addrs_path))

        conflicts = find_conflicts(config_path, symbol_files)
        if conflicts and remove:
            for symbol_file in symbol_files:
                lines = (line for line in symbol_file.read_text().splitlines() if line.rstrip("\n") and line.split()[0] not in [defined for _, defined in conflicts])
                symbol_file.write_text("\n".join(lines) + "\n")
        elif conflicts:
            for symbol, defined in conflicts:
                logger.warning(f"Symbol conflict detected in {config_path}! Address 0x{symbol.address:08X} built as {symbol.name}, but defined as {defined}") 


def find_conflicts(config_path, symbol_files):
    if "weapon" in config_path.name:
        logger.warning("Finding symbol conflicts for weapon overlays is not yet supported")
        return []

    config = yaml.safe_load(
        config_path.open()
    )
    defined_symbols = get_defined_symbols(symbol_files)
    syms_by_address = {symbol.address: symbol.name for symbol in defined_symbols}
    symbol_conflicts = []
    elf_path = Path(config["options"]["ld_script_path"]).with_suffix(".elf")
    for symbol in get_symbols(elf_path, {"LM", "__pad"}, {"_START", "_END", "_VRAM", "_data__s"}):
        if symbol not in defined_symbols and symbol.address in syms_by_address:
            symbol_conflicts.append((symbol,  syms_by_address[symbol.address]))

    return symbol_conflicts


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
            if f.endswith(".c") or f == f"{ovl_config.name}.h"
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

def add_undefined_symbol(version, symbol, address):
    symbol_line = f"{symbol}{' '*13}= 0x{address:X};"
    undefined_syms = Path(f"config/undefined_syms.{version}.txt")
    undefined_syms_lines = undefined_syms.read_text().splitlines()
    if symbol_line not in undefined_syms_lines:
        new_lines = sorted(undefined_syms_lines + [symbol_line], key=symbols_sort)
        undefined_syms.write_text("\n".join(new_lines) + "\n")

def get_symbol_offset(map_path, symbol_name, vram, start):
    if address := get_symbol_address(map_path, symbol_name):
        return address - vram + start
    else:
        return None

def get_symbol_address(map_path, symbol_name):
    if map_path and map_path.is_file():
        if match := re.search(RE_TEMPLATES.find_symbol_by_name.substitute(symbol_name=symbol_name), map_path.read_text()):
            return int(match.group(1), 16)
        else:
            return None
    else:
        logger.error(f"{map_path} not found")
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
