from __future__ import annotations
from enum import Enum
from typing import Union

__all__ = ["Register", "Opcode", "Instruction"]


# Todo: Switch to class, multivalue enum, or namedtuple
class Register(Enum):
    ZERO = 0x00  # Always zero
    AT = 0x01  # Assembler temporary
    V0 = 0x02  # Function return value
    V1 = 0x03  # Function return value
    A0 = 0x04  # Function argument
    A1 = 0x05  # Function argument
    A2 = 0x06  # Function argument
    A3 = 0x07  # Function argument
    T0 = 0x08  # Temporary
    T1 = 0x09  # Temporary
    T2 = 0x0A  # Temporary
    T3 = 0x0B  # Temporary
    T4 = 0x0C  # Temporary
    T5 = 0x0D  # Temporary
    T6 = 0x0E  # Temporary
    T7 = 0x0F  # Temporary
    S0 = 0x10  # Saved temporary
    S1 = 0x11  # Saved temporary
    S2 = 0x12  # Saved temporary
    S3 = 0x13  # Saved temporary
    S4 = 0x14  # Saved temporary
    S5 = 0x15  # Saved temporary
    S6 = 0x16  # Saved temporary
    S7 = 0x17  # Saved temporary
    T8 = 0x18  # Temporary
    T9 = 0x19  # Temporary
    K0 = 0x1A  # Reserved for OS kernel
    K1 = 0x1B  # Reserved for OS kernel
    GP = 0x1C  # Global pointer
    SP = 0x1D  # Stack pointer
    FP = 0x1E  # Frame pointer
    RA = 0x1F  # Return address


# Todo: Switch to class, multivalue enum, or namedtuple
class Opcode(Enum):
    # Load and Store Instructions
    LB = 0x20  # Load Byte
    LBU = 0x24  # Load Byte Unsigned
    LH = 0x21  # Load Halfword
    LHU = 0x25  # Load Halfword Unsigned
    LW = 0x23  # Load Word
    SB = 0x28  # Store Byte
    SH = 0x29  # Store Halfword
    SW = 0x2B  # Store Word

    # Arithmetic Instructions
    ADD = 0x20  # Add
    ADDU = 0x21  # Add Unsigned
    SUB = 0x22  # Subtract
    SUBU = 0x23  # Subtract Unsigned
    MULT = 0x18  # Multiply
    MULTU = 0x19  # Multiply Unsigned
    DIV = 0x1A  # Divide
    DIVU = 0x1B  # Divide Unsigned

    # Logical Instructions
    AND = 0x24  # Bitwise AND
    OR = 0x25  # Bitwise OR
    XOR = 0x26  # Bitwise XOR
    NOR = 0x27  # Bitwise NOR

    # Shift Instructions
    SLL = 0x00  # Shift Left Logical
    SRL = 0x02  # Shift Right Logical
    SRA = 0x03  # Shift Right Arithmetic

    # Immediate Instructions
    ADDI = 0x08  # Add Immediate
    ADDIU = 0x09  # Add Immediate Unsigned
    ANDI = 0x0C  # AND Immediate
    ORI = 0x0D  # OR Immediate
    XORI = 0x0E  # XOR Immediate
    LUI = 0x0F  # Load Upper Immediate

    # Branch Instructions
    BEQ = 0x04  # Branch if Equal
    BNE = 0x05  # Branch if Not Equal
    BLEZ = 0x06  # Branch if Less Than or Equal to Zero
    BGTZ = 0x07  # Branch if Greater Than Zero

    # Jump Instructions
    J = 0x02  # Jump
    JAL = 0x03  # Jump and Link
    JR = 0x08  # Jump Register
    JALR = 0x09  # Jump and Link Register

    # Comparison Instructions
    SLT = 0x2A  # Set on Less Than
    SLTU = 0x2B  # Set on Less Than Unsigned
    SLTI = 0x0A  # Set on Less Than Immediate
    SLTIU = 0x0B  # Set on Less Than Immediate Unsigned

    # Special Instructions
    SYSCALL = 0x0C  # System Call
    BREAK = 0x0D  # Breakpoint
    NOP = 0x00  # No Operation

    # Allegrex-Specific Instructions
    VFPU_MOVE = 0xD0  # Move VFPU Register
    VFPU_ADD = 0xD1  # Add VFPU Registers
    VFPU_MUL = 0xD2  # Multiply VFPU Registers
    VFPU_DOT = 0xD3  # Dot Product VFPU Registers
    CACHE = 0x2F  # Cache Operation
    SYNC = 0x0F  # Synchronize Memory


class Instruction:
    def __init__(
        self,
        value: int,
        opcode=None,
        rs=None,
        rt=None,
        rd=None,
        shamt=None,
        funct=None,
        immediate=None,
        address=None,
    ):
        """
        Initialize the Instruction instance.
        """
        if not isinstance(value, int):
            raise TypeError(
                f"Instruction must be provided as {type(int())} but got {type(value)}"
            )

        self.value = value
        self.instruction = value.to_bytes(4, byteorder="little")
        self.opcode = self.op = opcode
        self.source_register = self.rs = rs
        self.target_register = self.rt = rt
        self.dest_register = self.rd = rd
        self.shift_amount = self.shamt = shamt
        self.function_code = self.funct = funct
        self.immediate_unsigned = self.immu = immediate
        self.immediate_signed = self.imm = immediate
        self.address = address

        if immediate and immediate & 0x8000:
            self.immediate_signed = self.imm = immediate - 0x10000

    @classmethod
    def from_bytes(cls, data: Union[bytes, int]) -> Instruction:
        """
        Parses a 32-bit instruction word into its components and returns a
        SimpleNamespace with both the long and short forms of the field names
        """
        if not isinstance(data, (bytes, int)):
            raise TypeError(
                f"Data must be provided as {type(bytes())} or {type(int())} but got {type(data)}"
            )
        if isinstance(data, bytes) and len(data) != 4:
            raise ValueError(
                f"Data must be provided in a 32-bit word (4 bytes), but got {len(data)*8} bits ({len(data)} bytes)"
            )

        value = (
            int.from_bytes(data, byteorder="little")
            if isinstance(data, bytes)
            else data
        )
        kwargs = {"opcode": (value >> 26) & 0x3F}  # Opcode (6 bits)
        if kwargs["opcode"] == Opcode.J.value or kwargs["opcode"] == Opcode.JAL.value:
            kwargs["address"] = (
                value & 0x3FFFFFF
            )  # Address (26 bits for J-type instructions)
        else:
            kwargs["rs"] = (value >> 21) & 0x1F  # Source register (5 bits)
            kwargs["rt"] = (value >> 16) & 0x1F  # Target register (5 bits)
            kwargs["rd"] = (value >> 11) & 0x1F  # Destination register (5 bits)
            kwargs["shamt"] = (value >> 6) & 0x1F  # Shift amount (5 bits)
            kwargs["funct"] = value & 0x3F  # Function code (6 bits)
            kwargs["immediate"] = value & 0xFFFF  # Immediate value (16 bits)
        return cls(value, **kwargs)

    @classmethod
    def from_fields(cls, **kwargs) -> Instruction:
        # Format types from https://en.wikibooks.org/wiki/MIPS_Assembly/Instruction_Formats
        # This function assumes that the caller validates the fields passed and only checks for the resulting instruction having too many bits,
        # Known hard crashes: Passing a non int value
        if "opcode" in kwargs and (
            kwargs["opcode"] == Opcode.J.value or kwargs["opcode"] == Opcode.JAL.value
        ):  # j-format instruction
            value = (kwargs.get("op", 0x0) << 26) | (
                kwargs.get("address", 0x0) & 0x3FFFFFF
            )
        elif "funct" in kwargs and kwargs["funct"]:  # r-format instruction
            value = (
                (kwargs.get("opcode", 0x0) << 26)
                | (kwargs.get("rs", 0x0) << 21)
                | (kwargs.get("rt", 0x0) << 16)
                | (kwargs.get("rd", 0x0) << 11)
                | (kwargs.get("shamt", 0x0) << 6)
                | kwargs.get("funct", 0x0)
            )
        else:  # i-format instruction
            value = (
                (kwargs.get("opcode", 0x0) << 26)
                | (kwargs.get("rs", 0x0) << 21)
                | (kwargs.get("rt", 0x0) << 16)
                | (kwargs.get("immediate", 0x0) & 0xFFFF)
            )

        if value.bit_length() > 32:
            raise ValueError(
                f"Fields provided created an invalid instruction.  Expected 32 bits, but got {value.bit_length()}"
            )

        return cls(value, **kwargs)
    
    def normalize_instruction(instruction):
        if isinstance(instruction, bytes):
            instruction = int.from_bytes(instruction, byteorder='little')

        opcode = (instruction >> 26) & 0x3F

        if opcode == 0:  # R-type
            # Mask out the shamt (bits 10-6) and funct (bits 5-0)
            normalized = instruction & 0xFC00FFFF
        elif opcode in {0x2, 0x3}:  # J-type (J, JAL)
            # Mask out the address (bits 25-0)
            normalized = instruction & 0xFC000000
        else:  # I-type
            # Mask out the immediate value (bits 15-0)
            normalized = instruction & 0xFFFF0000
    
        return normalized.to_bytes(4, byteorder="little")

    def __repr__(self) -> str:
        return (
            f"<Instruction instruction={self.instruction} "
            f"opcode={hex(self.opcode) if self.opcode is not None else 'None'} "
            f"rs={hex(self.rs) if self.rs is not None else 'None'} "
            f"rt={hex(self.rt) if self.rt is not None else 'None'} "
            f"rd={hex(self.rd) if self.rd is not None else 'None'} "
            f"shamt={hex(self.shamt) if self.shamt is not None else 'None'} "
            f"funct={hex(self.funct) if self.funct is not None else 'None'} "
            f"imm={hex(self.imm) if self.imm is not None else 'None'} "
            f"immu={hex(self.immu) if self.immu is not None else 'None'} "
            f"address={hex(self.address) if self.address is not None else 'None'}>"
        )

    def __str__(self) -> str:
        return f"0x{self.value:08X}"

    def __int__(self) -> int:
        return self.value

    def __bytes__(self) -> bytes:
        return self.word

    def __contains__(self, data: Union[bytes, int]) -> bool:
        if not isinstance(data, (bytes, int)):
            raise TypeError(
                f"Data to be checked for a match must be provided as {type(bytes())} or {type(int())} but got {type(data)}"
            )
        if isinstance(data, bytes) and len(data) != 4:
            raise ValueError(
                f"Data to be checked for a match must be provided in a 32-bit word (4 bytes), but got {len(data)*8} bits ({len(data)} bytes)"
            )

        data = (
            int.from_bytes(data, byteorder="little")
            if isinstance(data, bytes)
            else data
        )

        pattern = 0
        mask = 0

        if self.opcode is not None:
            pattern |= self.opcode << 26
            mask |= 0xFC000000
        if self.rs is not None:
            pattern |= self.rs << 21
            mask |= 0x03E00000
        if self.rt is not None:
            pattern |= self.rt << 16
            mask |= 0x001F0000
        if self.rd is not None:
            pattern |= self.rd << 11
            mask |= 0x0000F800
        if self.shamt is not None:
            pattern |= self.shamt << 6
            mask |= 0x000007C0
        if self.funct is not None:
            pattern |= self.funct
            mask |= 0x0000003F
        if self.imm is not None:
            pattern |= self.imm & 0xFFFF
            mask |= 0x0000FFFF

        return (data & mask) == (pattern & mask)

    def find(self):
        """
        Look for a specific instruction within a binary file.  Constructs a generator and returns one value at a time.
        """
        # Todo: Actually write the function
        return None
