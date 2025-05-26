from __future__ import annotations
from yaml import *
from typing import Any, Union, List, Dict
from pathlib import PurePath


# Pyyaml doesn't determine indentless correctly, so we have to specify it explicitly
class IndentDumper(Dumper):
    def increase_indent(self, flow: bool = False, *args: Any, **kwargs: Any) -> None:
        return super().increase_indent(flow=flow, indentless=False)


class Hex(YAMLObject, int):
    """
    Represents values >= 0x08000000 as hex padded to 8 digits and values < 0x800000000 as
    unpadded hex, but otherwise behaves as int
    """

    yaml_tag = "!sotnhex"

    def __init__(self, value: Union[int, str]) -> None:
        if isinstance(value, int):
            self.value: int = value
        elif isinstance(value, str):
            # Todo: Figure out why it doesn't handle strings correctly
            self.value: int = int(value, 16)
        else:
            raise ValueError(f"Value must be {type(int)} or {type(str)}, but got {type(value)}")

    def __str__(self) -> str:
        if self.value >= 0x08000000:
            return f"0x{self.value:08X}"
        else:
            return f"0x{self.value:X}"

    def __int__(self) -> int:
        return self.value

    def __add__(self, other: Union[int, Hex]) -> Hex:
        if isinstance(other, int):
            return Hex(self.value + other)
        elif isinstance(other, Hex):
            return Hex(self.value + other.value)
        else:
            return NotImplemented

    def __radd__(self, other: Union[int, Hex]) -> Hex:
        if isinstance(other, int):
            return Hex(self.value + other)
        elif isinstance(other, Hex):
            return Hex(self.value + other.value)
        else:
            return NotImplemented

    @classmethod
    def to_yaml(cls, dumper: Dumper, data: Hex) -> ScalarNode:
        return dumper.represent_scalar("tag:yaml.org,2002:int", f"{data}", style="")


class FlowSegment:
    """Takes a list or dict and adds the correct representation for a sotn-decomp yaml config file"""

    def __new__(
        cls, obj: Union[List[Any], Dict[Any, Any]]
    ) -> Union[FlowList, FlowDict]:
        if isinstance(obj, list):
            return FlowList(obj)
        elif isinstance(obj, dict):
            return FlowDict(obj)
        else:
            raise ValueError("Input must be a list or a dictionary")


class FlowList(YAMLObject, list):
    yaml_tag = "!sotnlistseg"

    def __init__(self, obj: List[Any]) -> None:
        self.object: List[Any] = [Hex(x) if isinstance(x, int) else x for x in obj]
        super().__init__(obj)

    @classmethod
    def to_yaml(cls, dumper: Dumper, obj: FlowList) -> SequenceNode:
        return dumper.represent_sequence(
            "tag:yaml.org,2002:seq", obj.object, flow_style=True
        )


class FlowDict(YAMLObject, dict):
    yaml_tag = "!sotndictseg"

    def __init__(self, obj: Dict[Any, Any]) -> None:
        self.object: Dict[Any, Any] = {
            Hex(k) if isinstance(k, int) else k: (Hex(v) if isinstance(v, int) else v)
            for k, v in obj.items()
        }
        super().__init__(obj)

    @classmethod
    def to_yaml(cls, dumper: Dumper, obj: FlowDict) -> MappingNode:
        return dumper.represent_mapping(
            "tag:yaml.org,2002:map", obj.object, flow_style=True
        )


def int_representer(dumper: Dumper, obj: Any) -> ScalarNode:
    """Tells the yaml dumper to output the passed object as an int"""
    return dumper.represent_scalar("tag:yaml.org,2002:int", obj)


# Pyyaml doesn't properly handle multiline strings in all cases, so this explicitly specifies the right style
def str_representer(dumper: Dumper, obj: Any) -> ScalarNode:
    """Tells the yaml dumper to output the passed object as a string, correctly applying the | style for multiline strings"""
    if isinstance(obj, str) and "\n" in obj:
        return dumper.represent_scalar("tag:yaml.org,2002:str", obj, style="|")
    else:
        return dumper.represent_scalar("tag:yaml.org,2002:str", f"{obj}")


def seq_representer(dumper: Dumper, obj: Any) -> SequenceNode:
    """Tells the yaml dumper to output the passed object as a sequence"""
    return dumper.represent_sequence("tag:yaml.org,2002:seq", obj)


def map_representer(dumper: Dumper, obj: Any) -> MappingNode:
    """Tells the yaml dumper to output the passed object as a mapping"""
    return dumper.represent_mapping("tag:yaml.org,2002:map", obj)


add_representer(
    str, str_representer
)  # Pyyaml doesn't handle multiline strings properly
add_representer(tuple, seq_representer)  # Pyyaml doesn't handle tuples right
add_multi_representer(
    PurePath, str_representer
)  # Pyyaml doesn't know how to handle Path
