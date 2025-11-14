from __future__ import annotations
import logging
import itertools
import sys
import time
import datetime
import json
import threading
import os
import argparse
import re
import contextlib
import splat.scripts.split as split
import decomp_utils.yaml_ext as yaml
from subprocess import run, CalledProcessError
from logging import LogRecord
from pathlib import Path
from enum import StrEnum
from typing import Any, Union, Generator
from io import StringIO
from collections import namedtuple
from string import Template

__all__ = [
    "TTY",
    "RE_TEMPLATES",
    "RE_PATTERNS",
    "SotnDecompConsoleFormatter",
    "Spinner",
    "get_repo_root",
    "get_argparser",
    "get_logger",
    "shell",
    "create_table",
    "bar",
    "splat_split",
    "build",
    "git"
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
    symbol_file_line=re.compile(r"(?P<name>\w+)\s*=\s*0x[A-Fa-f0-9]{8};.*\n"),
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


class TTY(StrEnum):
    DARK_GREY = "\x1b[90m"
    GREY = "\x1b[38m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    RED = "\x1b[31m"
    BOLD = "\x1b[1m"
    HIDE_CURSOR = "\x1b[?25l"
    SHOW_CURSOR = "\x1b[?25h"
    CLEAR = "\x1b[K"
    RESET = "\x1b[0m"


class SotnDecompConsoleFormatter(logging.Formatter):
    """Formats log entries with color-coded severities"""

    format_preamble: str = f"\r{TTY.CLEAR}"
    source_id: str = "[%(levelname)s:%(filename)s:%(lineno)d] "
    message_format: str = "%(message)s"
    formats: dict[int, str] = {
        logging.DEBUG: f"{TTY.DARK_GREY}{format_preamble}{source_id}{message_format}{TTY.RESET}",
        logging.INFO: f"{TTY.GREY}{format_preamble}{message_format}{TTY.RESET}",
        logging.WARNING: f"{TTY.YELLOW}{format_preamble}{message_format}{TTY.RESET}",
        logging.ERROR: f"{TTY.RED}{format_preamble}{source_id}{message_format}{TTY.RESET}",
        logging.CRITICAL: f"{TTY.RED}{TTY.BOLD}{format_preamble}{source_id}{message_format}{TTY.RESET}",
    }

    def format(self, message: LogRecord) -> str:
        format_str: str = self.formats.get(message.levelno, self.message_format)
        return logging.Formatter(format_str).format(message)


class SotnDecompLogFormatter(logging.Formatter):
    """Formats log entries as JSON for file output"""

    def format(self, message: LogRecord) -> str:
        entry = {
            "timestamp": datetime.datetime.fromtimestamp(message.created).isoformat(),
            "level": message.levelname,
            "file": message.filename,
            "line": message.lineno,
            "message": message.getMessage(),
        }
        return json.dumps(entry)


class Spinner:
    def __init__(
        self,
        output_to="stderr",
        interval: float = None,
        message: str = "Generating witty line",
    ) -> None:
        self.output_to = sys.stdout if output_to == "stdout" else sys.stderr
        self.running: bool = False
        self.killed: bool = False
        self.generator: Generator[str, None, None] = self.spinner()
        self._message: str = f"{message}"
        self.interval: float = interval if interval and float(interval) else 0.1
        self.start_time = time.perf_counter()

    @staticmethod
    def spinner() -> Generator[str, None, None]:
        spin_cycle = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        for char in itertools.cycle(spin_cycle):
            yield f"{TTY.GREEN}{TTY.BOLD}{char}{TTY.RESET}"

    @staticmethod
    def write(s: str, output_to=sys.stderr) -> None:
        output_to.write(s)
        output_to.flush()

    @property
    def message(self) -> str:
        return self._message

    @message.setter
    def message(self, s: str) -> None:
        self.write(f"\r{next(self.generator)} {s}{TTY.CLEAR}", self.output_to)
        self._message = s

    def task(self) -> None:
        self.write(TTY.HIDE_CURSOR, self.output_to)
        while self.running:
            self.write(f"\r{next(self.generator)} {self.message}", self.output_to)
            time.sleep(self.interval)
        self.write(TTY.SHOW_CURSOR, self.output_to)

    def kill(self, message=None) -> None:
        if message:
            get_logger().warning(message)
        self.__exit__(None, None, None, killed=True)

    def __enter__(self) -> Spinner:
        self.running = True
        threading.Thread(target=self.task).start()
        return self

    def __exit__(
        self, exception: Any, value: Any, tb: Any, killed: bool = False
    ) -> Union[bool, None]:
        self.running = False
        time.sleep(self.interval)  # Wait for the last cycle to finish
        if killed or exception is not None:
            self.write(
                f"\r{TTY.RED}{TTY.BOLD}✖{TTY.RESET} {self.message} ({round(time.perf_counter() - self.start_time, 2)}s)\n",
                self.output_to,
            )
        else:
            self.write(
                f"\r{TTY.GREEN}{TTY.BOLD}✔{TTY.RESET} {self.message} ({round(time.perf_counter() - self.start_time, 2)}s)\n",
                self.output_to,
            )

        if exception is not None:
            return False


def get_logger(
    file_level=logging.INFO,
    console_level=logging.WARNING,
    stdout=True,
    filename="tools/decomp_utils/logs/sotn_log.json",
):
    """Simple wrapper function to make it easier to set up and use custom formatting"""
    logger = logging.getLogger(__name__.strip("_"))
    logger.setLevel(logging.INFO)

    if filename and not any(
        isinstance(h, logging.FileHandler)
        and h.baseFilename == Path(filename).resolve()
        for h in logger.handlers
    ):
        file_handler = logging.FileHandler(filename, mode="a")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(SotnDecompLogFormatter())
        logger.addHandler(file_handler)

    if stdout and not any(
        isinstance(h, logging.StreamHandler)
        and getattr(h, "stream", None) in (sys.stdout, sys.stderr)
        for h in logger.handlers
    ):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(SotnDecompConsoleFormatter())
        logger.addHandler(console_handler)

    return logger

def get_repo_root(current_path: Path = Path(__file__).resolve()) -> Path:
    """Attempts to find the root of the repo by stepping backward from directory containing __file__"""
    while current_path != current_path.root:
        if current_path.joinpath(".git").exists():
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError("Repository root with .git folder not found.")


def get_argparser(description):
    """Sets arguments that are used frequently to enable a consistent user experience"""
    arg_parser = argparse.ArgumentParser(description=description)
    arg_parser.add_argument(
        "-v",
        "--version",
        required=False,
        type=str,
        default=os.getenv("VERSION") or "us",
        help="Sets game version and overrides VERSION environment variable",
    )

    return arg_parser


def shell(cmd, env_vars = {}, version="us"):
    """Executes a string as a shell command and returns its output"""
    # Todo: Add both list and string handling
    env = os.environ.copy()
    # Ensure the correct VERSION is passed
    env["VERSION"] = version
    env.update(env_vars)
    cmd_output = run(cmd.split(), env=env, capture_output=True)
    if cmd_output.returncode != 0:
        logger = get_logger()
        logger.warning(cmd_output.stdout)
        logger.error(cmd_output.stderr)
        # raise CalledProcessError(cmd_output.returncode, cmd, cmd_output.stderr)
    return cmd_output.stdout


def create_table(rows, header=None, style="single"):
    # Define box-drawing characters for different styles
    styles = {
        "single": {
            "horizontal": "─",
            "vertical": "│",
            "top_left": "┌",
            "top_right": "┐",
            "top_mid": "┬",
            "bottom_left": "└",
            "bottom_right": "┘",
            "bottom_mid": "┴",
            "left_mid": "├",
            "right_mid": "┤",
            "center": "┼",
        },
        "double": {
            "horizontal": "═",
            "vertical": "║",
            "top_left": "╔",
            "top_right": "╗",
            "top_mid": "╦",
            "bottom_left": "╚",
            "bottom_right": "╝",
            "bottom_mid": "╩",
            "left_mid": "╠",
            "right_mid": "╣",
            "center": "╬",
        },
        "bold": {
            "horizontal": "━",
            "vertical": "┃",
            "top_left": "┏",
            "top_right": "┓",
            "top_mid": "┳",
            "bottom_left": "┗",
            "bottom_right": "┛",
            "bottom_mid": "┻",
            "left_mid": "┣",
            "right_mid": "┫",
            "center": "╋",
        },
    }

    chars = styles.get(style, styles["single"])

    # Determine column widths
    if header:
        rows = [header] + rows
    col_widths = [
        max(len(str(cell)) for cell in col)
        for col in zip(*[row for row in rows if row != "---"])
    ]

    def create_row(cells, left, mid, right, fill):
        row = left
        for i, cell in enumerate(cells):
            first_char = "▐" if "█" in str(cell) else " "
            row += f'{first_char}{str(cell).ljust(col_widths[i])}{" " if i else ""}{chars['vertical'] if i < len(cells) - 1 else right}'
        return row

    def create_divider():
        return (
            chars["left_mid"]
            + chars["center"].join(
                chars["horizontal"] * (w + 1) + (chars["horizontal"] if i else "")
                for i, w in enumerate(col_widths)
            )
            + chars["right_mid"]
        )

    table = []

    # Top border
    table.append(
        chars["top_left"]
        + chars["top_mid"].join(
            chars["horizontal"] * (w + 1) + (chars["horizontal"] if i else "")
            for i, w in enumerate(col_widths)
        )
        + chars["top_right"]
    )

    if header:
        table.append(
            create_row(
                header,
                chars["vertical"],
                chars["vertical"],
                chars["vertical"],
                chars["horizontal"],
            )
        )
        table.append(create_divider())

    for row in rows[1:] if header else rows:
        if row == "---":
            table.append(create_divider())
        else:
            table.append(
                create_row(
                    row,
                    chars["vertical"],
                    chars["vertical"],
                    chars["vertical"],
                    chars["horizontal"],
                )
            )

    # Bottom border
    table.append(
        chars["bottom_left"]
        + chars["bottom_mid"].join(
            chars["horizontal"] * (w + 1) + (chars["horizontal"] if i else "")
            for i, w in enumerate(col_widths)
        )
        + chars["bottom_right"]
    )

    return "\n".join(table)


def bar(score, width=7):
    # Normalize the score to a range of 0 to (width - 1) for full blocks
    normalized = score / (100 / (width - 1))
    full_blocks = int(normalized)

    half_block_start = 1  # First block is always a half block
    half_block_end = 1 if normalized - full_blocks >= 0.5 else 0
    empty_blocks = width - full_blocks - half_block_start - half_block_end

    bar = "█" * full_blocks + "▌" * half_block_end + " " * empty_blocks
    return bar


def build(targets=[], plan=True, dynamic_syms=False, build=True, version="us"):
    env_vars = {"FORCE_SYMBOLS": ""} if dynamic_syms else {}
    if plan:
        Path(f"build/{version}/").mkdir(parents=True, exist_ok=True)
        shell(f"python3 tools/builds/gen.py build/{version}/build.ninja", env_vars=env_vars, version=version)
    if build:
        return shell(f'ninja -f build/{version}/build.ninja {" ".join(f"{x}" for x in targets)}', version=version)


def splat_split(config_path, disassemble_all=True):
    if disassemble_all and re.search(
        r"disassemble_all:\s*(?:true|yes)", Path(config_path).read_text(), re.IGNORECASE
    ):
        disassemble_all = False
    output = StringIO()
    with contextlib.redirect_stdout(output):
        # Splat has a bug in versions prior to 0.32.3 where it will throw an exception if disassemble_all is True in the config and also passed as True
        if disassemble_all and re.search(
            r"disassemble_all:\\s*(?:true|yes)",
            Path(config_path).read_text(),
            re.IGNORECASE,
        ):
            disassemble_all = False
        split.main(
            config_path=[config_path],
            modes=None,
            verbose=False,
            use_cache=False,
            skip_version_check=False,
            disassemble_all=disassemble_all,
        )
    return output.getvalue()

def git(cmd, path):
    if isinstance(path, (list,tuple)):
        path = " ".join(f"{x}" for x in path)

    match cmd:
        case "add":
            shell(f"git add {path}")
        case "clean":
            shell(f"git clean -fdx {path}")
        case "reset":
            shell(f"git checkout {path}")

def mipsmatch(version, ref_ovls, bin_path):
    segments = []
    for ref_ovl in ref_ovls:
        shell(
            f"cargo run --release --manifest-path tools/mipsmatch/Cargo.toml -- --output build/{version}/fingerprint.{ref_ovl}.yaml fingerprint build/{version}/{ref_ovl}.map build/{version}/{ref_ovl}.elf"
        )
        match_stream = shell(
            f"cargo run --release --manifest-path tools/mipsmatch/Cargo.toml -- scan build/{version}/fingerprint.{ref_ovl}.yaml {bin_path}"
        )
        matches = yaml.load_all(match_stream, Loader=yaml.SafeLoader)
        for match in matches:
            if match not in segments:
                segments.append(match)
    return segments
