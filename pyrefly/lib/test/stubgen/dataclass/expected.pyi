# @generated
from dataclasses import dataclass, field, InitVar
from typing import ClassVar


@dataclass
class OptionalName:
    name: str | None

    def __init__(self, name: str | None = ...) -> None: ...


@dataclass
class Mixed:
    required: int
    literal_default: str = "x"
    field_with_default: int
    field_with_none: str | None
    items: list[str]
    complex_factory: list[int]
    field_no_default: int
    field_ellipsis: int

    def __init__(self, required: int, literal_default: str = ..., field_with_default: int = ..., field_with_none: str | None = ..., items: list[str] = ..., complex_factory: list[int] = ..., field_no_default: int, *, field_ellipsis: int) -> None: ...


@dataclass
class WithInitFalse:
    path: str
    cached: dict[str, str]

    def __init__(self, path: str) -> None: ...


@dataclass
class WithInitVarAndClassVar:
    sentinel: ClassVar[str] = "x"
    raw: InitVar[bytes | None]
    text: str

    def __init__(self, raw: bytes | None, text: str = ...) -> None: ...


@dataclass(kw_only=True)
class AllKwOnly:
    a: int
    b: str = "y"

    def __init__(self, *, a: int, b: str = ...) -> None: ...


@dataclass
class CustomInit:
    x: int

    def __init__(self, x: int, *, tag: str = "") -> None: ...
