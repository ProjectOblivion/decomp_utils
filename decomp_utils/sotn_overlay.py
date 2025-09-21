from __future__ import annotations
import struct
import hashlib
import decomp_utils.yaml_ext as yaml
import decomp_utils.mips as mips
from decomp_utils.helpers import get_logger
from pathlib import Path
from types import SimpleNamespace

# Todo: Convert non-mutating SimpleNamespace to namedtuple
from collections import namedtuple
from typing import Union, Any, Dict, List, Optional

__all__ = [
    "NULL",
    "MwOverlayHeader",
    "SotnOverlayConfig",
    "get_text_offset",
    "get_bss_offset",
    "get_rodata_address",
    "align",
]

# Pseudo enum for my own convenience
NULL = namedtuple("Null", ["BYTE", "INT", "STR"])(b"\x00", 0x0, "")


class MwOverlayHeader:
    """A Python implementation of the MetroWerks overlay header"""

    # https://gist.sharingeye.com/Linblow/541a3b24559f9c89374fdbd9e0693c40
    """
    typedef struct {
    /* 0x0 */ uint8_t identifier[3];   // Header identifier "MWo"
    /* 0x3 */ uint8_t version;         // MWo version
    /* 0x4 */ uint32_t overlayID;      // Overlay ID
    /* 0x8 */ uint32_t address;        // Load address (ie. address of this structure, followed by the data)  
    /* 0xC */ uint32_t textSize;       // Size of the .text section   
    /* 0x10 */ uint32_t dataSize;      // Size of the .data section
    /* 0x14 */ uint32_t bssSize;       // Size of the .bss section
    /* 0x18 */ uint32_t staticInit;    // Start address of the static array of initialization function pointers
    /* 0x1C */ uint32_t staticInitEnd; // End address of the static array of initialization function pointers
    /* 0x20 */ uint8_t name[32];       // Overlay name
    } mwOverlayHeader; // size: 0x40/64 bytes
    """

    def __init__(self, obj: Union[Path, str, bytes]) -> None:
        if isinstance(obj, Path) or isinstance(obj, str):
            file_path = Path(obj).resolve()
            if file_path and file_path.exists():
                self.extract_header(file_path.read_bytes())
            else:
                raise FileNotFoundError(f"{file_path} does not exist or is invalid")
        else:
            self.extract_header(obj)

    def __repr__(self) -> str:
        return (
            f"MwOverlayHeader(identifier='{self.identifier}', mwo_version={self.mwo_version}, "
            f"overlay_id={self.overlay_id}, address={self.address}, text_size={self.text_size}, "
            f"data_size={self.data_size}, bss_size={self.bss_size}, static_init_start=0x{self.static_init_start}, "
            f"static_init_end={self.static_init_end}, name='{self.name}')"
        )

    def extract_header(
        self, data: bytes, struct_format: str = "<3sBIIIIIII32s"
    ) -> MwOverlayHeader:
        if not isinstance(struct_format, str):
            raise TypeError(f"Format must be a string, but got {type(struct_format)}")
        if not isinstance(data, bytes):
            raise TypeError(
                f"Data must be provided as {type(bytes())}, {type(Path())}, or {type(str())} but got {type(data)}"
            )
        format_size = struct.calcsize(struct_format)
        if len(data) < format_size:
            raise ValueError(
                f"Data size must be >= {format_size} bytes, but got {len(data)}"
            )
        unpacked_data = struct.unpack(struct_format, data[:format_size])
        self.identifier = unpacked_data[0].decode("ascii")
        self.mwo_version = unpacked_data[1]
        self.overlay_id = unpacked_data[2]
        self.address = yaml.Hex(unpacked_data[3])
        self.text_size = yaml.Hex(unpacked_data[4])
        self.data_size = yaml.Hex(unpacked_data[5])
        self.bss_size = yaml.Hex(unpacked_data[6])
        self.static_init_start = yaml.Hex(unpacked_data[7])
        self.static_init_end = yaml.Hex(unpacked_data[8])
        self.name = unpacked_data[9].rstrip(NULL.BYTE).decode("ascii")

        return self


# Todo: This is currently written for a new config, it needs to be modified to load a config if it exists or create a new one.
class SotnOverlayConfig:
    """
    A class for representing a SOTN overlay configuration.

    This class provides properties and methods to manage overlay configurations, including
    paths, options, and metadata for different platforms (currently only PSX and PSP).
    """

    # Base definitions for properties that use a pseudo-caching structure
    _mwo_header: Optional[MwOverlayHeader] = None
    _bin_bytes: bytes = b""
    _options_dict: Dict[Any] = {}
    _segments: List[Any] = []
    _subsegments: List[Any] = []

    def __init__(self, name: str, version: str) -> None:
        # common definitions
        # string options
        self.name: str = name
        self.version: str = version
        self.platform: str = "psp" if "psp" in version else "psx"
        self.basename: str = f"{self.ovl_prefix}{self.name}"
        self.compiler: str = "GCC"
        self.asm_jtbl_label_macro: str = "jlabel"
        self.symbol_name_format: str = f"{version}_$VRAM"
        self.nonmatchings_path: str = "nonmatchings"

        # ovl paths
        self.base_path: str = ".."
        self.build_path: Path = Path(f"build/{version}")
        self.target_path: Path = self._find_target_path()
        self.asm_path: Path = Path(f"asm/{version}")
        self.src_path: Path = Path("src")
        self.segment_prefix = f"{self.name}_psp/" if self.platform == "psp" else ""
        self.asset_path: Path = Path(f"assets").joinpath(self.path_prefix, self.name)
        self.ld_script_path: Path = self.build_path.joinpath(f"{self.basename}.ld")

        # splat paths
        self.config_path: Path = Path(f"config/splat.{version}.{self.basename}.yaml")
        self.extensions_path: Path = Path("tools/splat_ext")

        # symbols paths
        _symbols_base_path: Path = Path("config")
        self.symexport_path: Path = (
            Path(f"config/symexport.{self.version}.{self.basename}.txt")
            if self.platform == "psp"
            else ""
        )
        self.global_symbol_addrs_path: Path = _symbols_base_path.joinpath(
            f"symbols.{version}.txt"
        )
        self.ovl_symbol_addrs_path: Path = _symbols_base_path.joinpath(
            f"symbols.{self.version}.{self.basename}.txt"
        )
        self.undefined_funcs_auto_path: Path = _symbols_base_path.joinpath(
            f"undefined_funcs_auto.{self.version}.{self.basename}.txt"
        )
        self.undefined_syms_auto_path: Path = _symbols_base_path.joinpath(
            f"undefined_syms_auto.{self.version}.{self.basename}.txt"
        )
        self.symbol_addrs_path: tuple[Path] = (
            self.global_symbol_addrs_path,
            self.ovl_symbol_addrs_path,
        )

        # Boolean options
        self.find_file_boundaries: bool = True
        self.use_legacy_include_asm: bool = False
        self.migrate_rodata_to_functions: bool = True
        self.disassemble_all: bool = True
        self.disasm_unknown: bool = True
        self.ld_generate_symbol_per_data_segment: bool = True

        # Sections
        self.text_section: SimpleNamespace = SimpleNamespace(
            address=None, offset=None, size=None
        )
        self.data_section: SimpleNamespace = SimpleNamespace(
            address=None, offset=None, size=None
        )
        self.rodata_section: SimpleNamespace = SimpleNamespace(
            address=None, offset=None, size=None
        )
        self.bss_section: SimpleNamespace = SimpleNamespace(
            address=None, offset=None, size=None
        )
        self.sbss_section: SimpleNamespace = SimpleNamespace(
            address=None, offset=None, size=None
        )

        # Metadata
        self.sha1 = hashlib.sha1(self.bin_bytes).hexdigest()

        # Platform specific definitions
        if self.platform == "psx":
            self.asm_path = self.asm_path.joinpath(self.path_prefix, self.name)
            self.src_path = self.src_path.joinpath(self.path_prefix, self.name)
            self.src_path_full = self.src_path
            self.first_src_file = self.src_path_full / f"first_{self.name}.c"
            self.start: int = 0x0

            self.global_vram_start: int = 0x80010000
            self.ld_bss_is_noload: bool = False
            match self.name:
                case "dra":
                    self.vram = 0x800A0000
                case "ric":
                    self.vram = 0x8013C000
                case _:
                    match self.ovl_type:
                        case "stage" | "boss":
                            self.vram = 0x80180000
                        case "servant":
                            self.vram = 0x80170000
                        case _:
                            self.vram = self.global_vram_start

            self.align: int = 4
            self.subalign: int = 4
            self.section_order: tuple[str] = (
                ".data",
                ".rodata",
                ".text",
                ".bss",
                ".sbss",
            )
            self.asm_inc_header: None
            self.text_section.offset = get_text_offset(self.bin_bytes)
            self.bss_section.offset = get_bss_offset(self.bin_bytes)
            self.bss_section.address = self.bss_section.offset + self.vram - self.start
            _jtbl_address = get_rodata_address(
                self.bin_bytes[self.text_section.offset : self.bss_section.offset]
            )
            self.rodata_section.offset = (
                _jtbl_address - self.vram if _jtbl_address else None
            )
        elif self.platform == "psp":
            self.asm_path = self.asm_path.joinpath(self.path_prefix, f"{self.name}_psp")
            self.src_path = self.src_path.joinpath(self.path_prefix)
            self.src_path_full = self.src_path.joinpath(f"{self.name}_psp")
            self.first_src_file = self.src_path_full / f"first_{self.name}.c"
            self.start: int = 0x80
            self.vram: int = self.mwo_header.address + 0x80
            self.align: int = 128
            self.subalign: int = 8
            self.ld_bss_is_noload: bool = True
            self.global_vram_start: int = 0x08000000
            self.section_order: tuple[str] = (".text", ".data", ".rodata", ".bss")
            self.asm_inc_header: Optional[str] = (
                ".set noat      /* allow manual use of $at */\n.set noreorder /* don't insert nops after branches */\n"
            )
            self.text_section = SimpleNamespace(
                address=self.mwo_header.address + 0x80,
                offset=0x80,
                size=self.mwo_header.text_size,
            )
            self.bss_section = SimpleNamespace(
                address=self.mwo_header.static_init_start,
                offset=self.mwo_header.static_init_start - self.mwo_header.address,
                size=self.mwo_header.bss_size,
            )
            self.data_section = SimpleNamespace(
                address=align(self.text_section.address + self.text_section.size, 0x80),
                offset=align(self.text_section.size, 0x80),
                size=self.mwo_header.data_size,
                bytes=self.bin_bytes[
                    self.data_section.offset : self.bss_section.offset
                ],
            )
            self.rodata_section: SimpleNamespace = SimpleNamespace(
                address=None, offset=None, size=None
            )
            # Unpack the bytes normally, but iterate through the unpacked data in reverse order
            words = tuple(
                word[0] for word in struct.iter_unpack("<I", self.data_section.bytes)
            )
            self.rodata_section.offset = yaml.Hex(
                next(
                    (
                        align(self.bss_section.offset - (i * 4), 0x80)
                        for i, word in enumerate(words[::-1])
                        if word
                        and not self.text_section.address
                        <= word
                        < self.data_section.address
                    ),
                    self.data_section.offset + 0x80,
                ),
            )

            self.rodata_section.address = (
                self.bss_section.address - self.rodata_section.offset
            )

    @property
    def ovl_type(self):
        return self._ovl_type.label

    @property
    def ovl_prefix(self):
        return self._ovl_type.ovl_prefix

    @property
    def path_prefix(self):
        return self._ovl_type.path_prefix

    @property
    def _ovl_type(self):
        game = "main dra ric weapon maria "
        stage = "are cat cen chi dai dre lib mad no0 no1 no2 no3 no4 np3 nz0 nz1 sel st0 top wrp "
        r_stage = "rare rcat rcen rchi rdai rlib rno0 rno1 rno2 rno3 rno4 rnz0 rnz1 rtop rwrp "
        boss = "bo0 bo1 bo2 bo3 bo4 bo5 bo6 bo7 mar rbo0 rbo1 rbo2 rbo3 rbo4 rbo5 rbo6 rbo7 rbo8 "
        servant = "tt_000 tt_001 tt_002 tt_003 tt_004 tt_005 tt_006 "

        if f"{self.name} " in game:
            return namedtuple("GameOvl", ["label", "ovl_prefix", "path_prefix"])(
                "game", "", ""
            )
        elif f"{self.name} " in stage + r_stage:
            return namedtuple("StageOvl", ["label", "ovl_prefix", "path_prefix"])(
                "stage", "st", "st"
            )
        elif f"{self.name} " in boss:
            return namedtuple("BossOvl", ["label", "ovl_prefix", "path_prefix"])(
                "boss", "bo", "boss"
            )
        elif f"{self.name} " in servant:
            return namedtuple("ServantOvl", ["label", "ovl_prefix", "path_prefix"])(
                "servant", "", "servant"
            )
        elif "w0_" in self.name or "w1_" in self.name:
            return namedtuple("WeaponOvl", ["label", "ovl_prefix", "path_prefix"])(
                "weapon", "", "weapon"
            )
        else:
            raise ValueError(f"Unknown overlay type for '{self.name}'")

    def _find_target_path(self) -> Path:
        bin_name = (
            f"{self.name.upper()}.BIN" if self.version == "us" else f"{self.name}.bin"
        )
        target_path = next(self.disk_path.rglob(bin_name), None)
        if not target_path:
            get_logger().error(f"Could not find {bin_name} in {self.disk_path}")
            raise SystemExit
        return target_path

    @property
    def disk_path(self) -> Path:
        if self.version == "pspeu" or self.version == "hd":
            return Path(
                f"disks/pspeu/PSP_GAME/USRDIR/res/ps/{"hdbin" if self.version == "hd" else "PSPBIN"}"
            )
        else:
            return Path(f"disks/{self.version}")

    @property
    def config(self) -> Dict[Any]:
        return {
            "options": self.options,
            "sha1": self.sha1,
            "segments": self.segments,
        }

    @property
    def options(self) -> Dict[Any]:
        if not self._options_dict:
            options_dict = {
                "platform": self.platform,
                "basename": self.basename,
                "base_path": self.base_path,
                "build_path": self.build_path,
                "target_path": self.target_path,
                "asm_path": self.asm_path,
                "asset_path": self.asset_path,
                "src_path": self.src_path,
                "ld_script_path": self.ld_script_path,
                "compiler": self.compiler,
                "symbol_addrs_path": self.symbol_addrs_path,
                "undefined_funcs_auto_path": self.undefined_funcs_auto_path,
                "undefined_syms_auto_path": self.undefined_syms_auto_path,
                "find_file_boundaries": self.find_file_boundaries,
                "use_legacy_include_asm": self.use_legacy_include_asm,
                "migrate_rodata_to_functions": self.migrate_rodata_to_functions,
                "asm_jtbl_label_macro": self.asm_jtbl_label_macro,
                "symbol_name_format": self.symbol_name_format,
                "disassemble_all": self.disassemble_all,
                "section_order": self.section_order,
                "ld_bss_is_noload": self.ld_bss_is_noload,
                "disasm_unknown": self.disasm_unknown,
                "global_vram_start": yaml.Hex(self.global_vram_start),
                "ld_generate_symbol_per_data_segment": self.ld_generate_symbol_per_data_segment,
            }
            if self.platform == "psp":
                options_dict["asm_inc_header"] = self.asm_inc_header
            self._options_dict = options_dict
        return self._options_dict

    @options.setter
    def options(self, value: Dict[Any]):
        self._options_dict = value

    @property
    def segments(self) -> List[Any]:
        if not self._segments:
            self._segments = [
                x
                for x in [
                    (
                        yaml.FlowSegment([0x0, "bin", "mwo_header"])
                        if self.platform == "psp"
                        else None
                    ),
                    {
                        k: v
                        for k, v in {
                            "name": self.basename,
                            "type": "code",
                            "start": yaml.Hex(self.start),
                            "vram": yaml.Hex(self.vram),
                            "bss_start_address": (
                                yaml.Hex(self.bss_section.address)
                                if self.platform == "psp"
                                else None
                            ),
                            "bss_size": (
                                self.mwo_header.bss_size
                                if self.platform == "psp"
                                else None
                            ),
                            "align": self.align,
                            "subalign": self.subalign,
                            "subsegments": self.subsegments,
                        }.items()
                        if v is not None
                    },
                    yaml.FlowSegment(
                        [
                            (
                                self.bss_section.offset + self.bss_section.size
                                if self.platform == "psp"
                                else len(self.bin_bytes)
                            )
                        ]
                    ),
                ]
                if x is not None
            ]
        return self._segments

    @property
    def subsegments(self) -> List[Any]:
        if not self._subsegments and self.platform == "psx":
            self._subsegments = [
                x
                for x in [
                    yaml.FlowSegment([0x0, "data"]),
                    (
                        yaml.FlowSegment(
                            [
                                self.rodata_section.offset,
                                ".rodata",
                                self.first_src_file.stem,
                            ]
                        )
                        if self.rodata_section.offset
                        and self.migrate_rodata_to_functions
                        else None
                    ),
                    (
                        yaml.FlowSegment([self.rodata_section.offset, "rodata"])
                        if self.rodata_section.offset
                        and not self.migrate_rodata_to_functions
                        else None
                    ),
                    (
                        yaml.FlowSegment(
                            [self.text_section.offset, "c", self.first_src_file.stem]
                        )
                        if self.text_section.offset
                        else None
                    ),
                    (
                        yaml.FlowSegment([self.bss_section.offset, "bss"])
                        if self.bss_section.offset
                        else None
                    ),
                ]
                if x is not None
            ]
        elif not self._subsegments and self.platform == "psp":
            self._subsegments = [
                yaml.FlowSegment(
                    [0x80, "c", f"{self.segment_prefix}{self.first_src_file.stem}"]
                ),
                yaml.FlowSegment([self.data_section.offset, "data"]),
                yaml.FlowSegment(
                    [
                        self.rodata_section.offset,
                        ".rodata",
                        f"{self.segment_prefix}{self.first_src_file.stem}",
                    ]
                ),
                yaml.FlowSegment([self.bss_section.offset, "bss"]),
            ]
        return self._subsegments

    @property
    def bin_bytes(self) -> bytes:
        if not self._bin_bytes:
            self._bin_bytes = self.target_path.read_bytes()
        return self._bin_bytes

    @property
    def mwo_header(self) -> MwOverlayHeader:
        """
        Parse and return the MetroWerks overlay header that contains all the
        necessary metadata to successfully parse a psp binary file.
        """
        if (
            not self._mwo_header
            and self.platform == "psp"
            and len(self.bin_bytes) >= 64
        ):
            self._mwo_header = MwOverlayHeader(self.bin_bytes[:64])
        return self._mwo_header

    def write_config(self) -> None:
        self.config_path.write_bytes(
            yaml.dump(
                self.config,
                Dumper=yaml.IndentDumper,
                encoding="utf-8",
                sort_keys=False,
            )
        )


def get_text_offset(data: bytes) -> Optional[int]:
    addiu_sp = mips.Instruction.from_fields(
        opcode=mips.Opcode.ADDIU.value,
        rs=mips.Register.SP.value,
        rt=mips.Register.SP.value,
    ).instruction.lstrip(NULL.BYTE)
    # Search for 'addiu $sp, $sp, imm address'
    text_offset = data.find(addiu_sp) - 2

    # Checks each addiu $sp, $sp match until it finds one that is both
    # in the proper byte alignment and the imm address is not 0
    while text_offset > 0 and (
        text_offset % 4 != 0
        or (data[text_offset + 1] != NULL.BYTE and data[text_offset + 1] != 0xFF)
        or data[text_offset : text_offset + 2] == NULL.BYTE * 2
    ):
        text_offset = data.find(addiu_sp, text_offset + 4) - 2

    return text_offset if text_offset > 0 else None


def get_bss_offset(data: bytes) -> Optional[int]:
    # Find the final 'jr $ra' to identify the likely end of the text section
    bss_offset = data.rfind(
        mips.Instruction.from_fields(
            funct=mips.Opcode.JR.value, rs=mips.Register.RA.value
        ).instruction
    )
    return None if bss_offset == -1 else bss_offset + 8


def get_rodata_address(data: bytes) -> Optional[int]:
    # Look for 'jr $v0'
    jr_offset = data.find(
        mips.Instruction.from_fields(
            funct=mips.Opcode.JR.value, rs=mips.Register.V0.value
        ).instruction
    )
    if jr_offset == -1:
        return None

    lw_v0 = mips.Instruction.from_fields(
        opcode=mips.Opcode.LW.value, rt=mips.Register.V0.value
    )
    # Look for last 'lw $v0, %lo(XXX)(YYY)' before jr_offset
    lw_offset = data.rfind(lw_v0.instruction[3], 0, jr_offset) - 3
    if lw_offset == -1:
        return None

    lw_v0 = mips.Instruction.from_bytes(data[lw_offset : lw_offset + 4])
    lui_rs = mips.Instruction.from_fields(
        opcode=mips.Opcode.LUI.value, rt=lw_v0.rs, rs=NULL.INT
    ).instruction
    # Look for last 'lui $at, %hi(XXX) before lw_offset
    lui_offset = data.rfind(lui_rs.lstrip(NULL.BYTE), 0, jr_offset) - 2

    if lui_offset == -1:
        return None

    return (
        mips.Instruction.from_bytes(data[lui_offset : lui_offset + 4]).immu << 16
    ) + lw_v0.imm


def align(n: Union[int, yaml.Hex], alignment: Union[int, yaml.Hex]) -> int:
    """Aligns n to the nearest multiple of alignment."""
    if alignment & (alignment - 1) == 0:  # Check if alignment is a power of 2
        return (n + alignment - 1) & ~(alignment - 1)
    else:
        return ((n + alignment - 1) // alignment) * alignment
