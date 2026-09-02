# @generated
import functools
from pydantic import BaseModel, computed_field, Field


class Mixed(BaseModel):
    required: int
    literal_default: int = 7
    field_with_default: int
    field_with_none: str | None
    items: list[str]
    complex_factory: list[int]
    field_bare: int
    field_ellipsis: int
    field_ellipsis_only: int

    def __init__(self, required: int, literal_default: int = ..., field_with_default: int = ..., field_with_none: str | None = ..., items: list[str] = ..., complex_factory: list[int] = ..., field_bare: int, field_ellipsis: int, field_ellipsis_only: int) -> None: ...


class WithInitFalse(BaseModel):
    id: int
    digest: bytes

    def __init__(self, id: int) -> None: ...


class WithValidationAlias(BaseModel):
    internal_id: int

    def __init__(self, internal_id: int) -> None: ...


class WithKwOnly(BaseModel):
    req: int
    opt: str

    def __init__(self, req: int, *, opt: str = ...) -> None: ...


class WithComputed(BaseModel):
    width: int
    height: int

    @computed_field
    @property
    def area(self) -> int: ...

    def __init__(self, width: int, height: int) -> None: ...


class CustomInit(BaseModel):
    name: str

    def __init__(self, name: str, *, eager: bool = False) -> None: ...
