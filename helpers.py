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
import sotn_utils.yaml_ext as yaml
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
    "SotnDecompConsoleFormatter",
    "Spinner",
    "get_repo_root",
    "get_logger",
    "init_logger",
    "shell",
    "create_table",
    "bar",
    "splat_split",
    "build",
]

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

def get_logger():
    """Simple wrapper function to make it easier to get the logger"""
    return logging.getLogger(__name__.strip("_"))

def init_logger(
    file_level=logging.INFO,
    console_level=logging.WARNING,
    stdout=True,
    filename=Path(__file__).parent / "logs" / "sotn_log.json",
):
    """Simple wrapper function to make it easier to set up and use custom formatting"""
    if isinstance(filename, (str, Path)):
        filename = Path(filename)
    
    if not filename.exists():
        filename.parent.mkdir(parents=True, exist_ok=True)

    logger = get_logger()
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
        return shell(f"ninja -f build/{version}/build.ninja {' '.join(f"{x}" for x in targets)}", version=version)


def splat_split(config_path, disassemble_all=True):
    output = StringIO()
    with contextlib.redirect_stdout(output):
        split.main(
            config_path=[config_path],
            modes=None,
            verbose=False,
            use_cache=False,
            skip_version_check=False,
            disassemble_all=disassemble_all,
        )
    return output.getvalue()

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
