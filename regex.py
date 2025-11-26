import re
from collections import namedtuple
from string import Template

__all__ = [
    "RE_TEMPLATES",
    "RE_PATTERNS",
]

RE_STRINGS = namedtuple("ReStrings", ["psp_entity_table", "psp_ovl_header"])(
    psp_entity_table=r"""
        \s+/\*\s[A-F0-9]{1,5}(?:\s[A-F0-9]{8}){2}\s\*/\s+lui\s+\$v1,\s+%hi\((?P<entity>[A-Za-z0-9_]+)\)\n
        .*\n
        \s+/\*\s[A-F0-9]{1,5}\s[A-F0-9]{8}\sC708023C\s\*/.*\n
        \s+/\*\s[A-F0-9]{1,5}\s[A-F0-9]{8}\s30BC43AC\s\*/.*\n
    """,
    psp_ovl_header=r"""
        \s+/\*\s[A-F0-9]{1,5}\s[A-F0-9]{8}\s1D09043C\s\*/.*\n
        \s+/\*\s[A-F0-9]{1,5}\s[A-F0-9]{8}\s38F78424\s\*/.*\n
        \s+/\*\s[A-F0-9]{1,5}(?:\s[A-F0-9]{8}){2}\s\*/\s+lui\s+\$a1,\s+%hi\((?P<header>[A-Za-z0-9_]+)\)\n
        (?:.*\n){2}
        \s+/\*\s[A-F0-9]{1,5}\s[A-F0-9]{8}\sE127240E\s\*/.*\n
    """,
)
RE_TEMPLATES = namedtuple(
    "ReTemplates",
    ["sym_replace", "find_symbol_by_name", "rodata_offset", "asm_symbol_offset"],
)(
    sym_replace=Template(r"(?:D_|func_)${sym_prefix}(${symbols_list})"),
    find_symbol_by_name=Template(
        r"\n\s+0x(?P<address>[A-Fa-f0-9]{8})\s+${symbol_name}\n"
    ),
    rodata_offset=Template(
        r"glabel (?:jtbl|D)_${version}_[0-9A-F]{8}\n\s+/\*\s(?P<offset>[0-9A-F]{1,5})\s"
    ),
    asm_symbol_offset=Template(
        r"glabel ${symbol_name}\s+/\*\s(?P<offset>[0-9A-F]{1,5})\s"
    ),
)
RE_PATTERNS = namedtuple(
    "RePatterns",
    [
        "symbol_file_line",
        "op",
        "asm_line",
        "jtbl",
        "jtbl_line",
        "masking",
        "map_symbol",
        "elf_symbol",
        "include_asm",
        "include_rodata",
        "camel_case",
        "symbol_ovl_name_prefix",
        "psp_entity_table_pattern",
        "psp_ovl_header_pattern",
        "psp_ovl_header_entity_table_pattern",
        "symbol_line_pattern",
        "init_room_entities_symbol_pattern",
        "ref_pattern",
        "cross_ref_name_pattern",
        "cross_ref_address_pattern",
        "splat_suggestions_full",
        "splat_suggestion",
        "existing_symbols",
    ],
)(
    symbol_file_line=re.compile(r"(?P<name>\w+)\s*=\s*0x(?P<address>[A-Fa-f0-9]{8});(?:\s*//\s*(?P<comment>.*))?\n"),
    op=re.compile(
        rb"""
        /\*\s(?:[0-9A-F]{1,5})
        \s(?:[0-9A-F]{8})
        \s(?:[0-9A-F]{8})
        \s\*/\s+([a-z]{1,5})
        [ \t]*(?:[^\n]*)\n
        """,
        re.VERBOSE,
    ),
    asm_line=re.compile(
        rb"""
        /\*\s(?P<offset>[0-9A-F]{1,5})
        \s(?P<address>[0-9A-F]{8})
        \s(?P<word>[0-9A-F]{8})
        \s\*/\s+(?P<op>[a-z]{1,5})
        [ \t]*(?P<fields>[^\n]*)\n
        """,
        re.VERBOSE,
    ),
    jtbl=re.compile(
        rb"""
        glabel\s(?P<name>jtbl\w+[0-9A-F]{8})\n
        (?P<table>.+?)\n
        \.size\s(?P=name),\s\.\s\-\s(?P=name)\n
        """,
        re.DOTALL | re.VERBOSE,
    ),
    jtbl_line=re.compile(
        rb"""
        /\*\s(?P<offset>[0-9A-F]{1,5})
        \s(?P<address>[0-9A-F]{8})
        \s(?P<data>[0-9A-F]{8})
        \s\*/\s+(?P<data_type>\.[a-z]{1,5})
        \s+(?P<location>\.[0-9A-Za-z_]{9,})
        """,
        re.VERBOSE,
    ),
    masking=re.compile(r"(?:\s\.?\w+$|\(\w+\))"),
    map_symbol=re.compile(
        r"\n\s+0x(?P<address>[A-Fa-f0-9]{8})\s+(?P<name>[A-Za-z]\w+)\n"
    ),
    elf_symbol=re.compile(
        r"(?P<address>[A-Fa-f0-9]{8})\s+[^A]\s+(?P<name>[A-Za-z]\w+)"
    ),
    include_asm=re.compile(
        r'INCLUDE_ASM\("(?P<dir>[A-Za-z0-9/_]+)",\s?(?P<name>\w+)\);'
    ),
    include_rodata=re.compile(r'INCLUDE_RODATA\("[A-Za-z0-9/_]+",\s?(?P<name>\w+)\);'),
    camel_case=re.compile(r"([A-Za-z])([A-Z][a-z])"),
    symbol_ovl_name_prefix=re.compile(r"^[^U][A-Z0-9]{2,3}_"),
    psp_entity_table_pattern=re.compile(RE_STRINGS.psp_entity_table, re.VERBOSE),
    psp_ovl_header_pattern=re.compile(RE_STRINGS.psp_ovl_header, re.VERBOSE),
    psp_ovl_header_entity_table_pattern=re.compile(
        rf"{RE_STRINGS.psp_entity_table}(?:.*\n)+{RE_STRINGS.psp_ovl_header}",
        re.VERBOSE,
    ),
    symbol_line_pattern=re.compile(
        r"/\*\s[0-9A-F]{1,5}\s[0-9A-F]{8}\s(?P<address>[0-9A-F]{8})\s\*/\s+\.word\s+(?P<name>\w+)"
    ),
    init_room_entities_symbol_pattern=re.compile(
        r"\s+/\*\s[0-9A-F]{1,5}\s[0-9A-F]{8}\s[0-9A-F]{8}\s\*/\s+[a-z]{1,5}[ \t]*\$\w+,\s%hi\(D_(?:\w+_)?(?P<address>[A-F0-9]{8})\)\s*"
    ),
    ref_pattern=re.compile(r"splat\.\w+\.(?P<prefix>st|bo)(?P<ref_ovl>\w+)\.yaml"),
    cross_ref_name_pattern=re.compile(r"lui\s+.+?%hi\(((?:[A-Z]|g_|func_)\w+)\)"),
    cross_ref_address_pattern=re.compile(
        r"lui\s+.+?%hi\((?:D_|func_)(?:\w+_)?([A-F0-9]{8})\)"
    ),
    splat_suggestions_full=re.compile(
        r"""
        The\srodata\ssegment\s'(?P<segment>\w+)'\shas\sjumptables.+\n
        File\ssplit\ssuggestions.+\n
        (?P<suggestions>(?:\s+-\s+\[0x[0-9A-Fa-f]+,\s.+?\]\n)+)\n
        """,
        re.VERBOSE,
    ),
    splat_suggestion=re.compile(r"\s+-\s+\[(0x[0-9A-Fa-f]+),\s(.+?)\]"),
    existing_symbols=re.compile(r"(?P<name>\w+)\s=\s0x(?P<address>[A-Fa-f0-9]{8})"),
)