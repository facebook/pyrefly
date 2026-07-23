# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools

from pydantic import BaseModel, computed_field, Field


# Mixed Pydantic `Field` defaults: literal, `default=`, `default_factory=` (simple + complex),
# `Field(...)` for required fields, bare `Field()`, plus a kwargs-only field.
class Mixed(BaseModel):
    required: int
    literal_default: int = 7
    field_with_default: int = Field(default=0)
    field_with_none: str | None = Field(default=None)
    items: list[str] = Field(default_factory=list, min_length=0)
    complex_factory: list[int] = Field(default_factory=functools.partial(list, [0]))
    field_bare: int = Field()
    field_ellipsis: int = Field(..., description="req")
    field_ellipsis_only: int = Field(...)


# `init=False` keeps the attribute on the model but drops it from `__init__`, matching the
# runtime constructor surface.
class WithInitFalse(BaseModel):
    id: int
    digest: bytes = Field(default=b"", init=False)


# Pydantic `validation_alias` (without `validate_by_name`) means the field is only populated
# via the alias keyword; the alias appears as a keyword-only parameter on `__init__`.
class WithValidationAlias(BaseModel):
    internal_id: int = Field(validation_alias="id")


# `Field(kw_only=True)` forces a field (and any later fields) to be keyword-only.
class WithKwOnly(BaseModel):
    req: int = Field(kw_only=False)
    opt: str = Field(default="z", kw_only=True)


# `@computed_field` exposes a read-only attribute that is not a constructor parameter; it
# should appear in the stub as a plain `@property`.
class WithComputed(BaseModel):
    width: int
    height: int

    @computed_field
    @property
    def area(self) -> int:
        return self.width * self.height


# A user-defined `__init__` must be kept verbatim; stubgen must not synthesize a replacement.
class CustomInit(BaseModel):
    name: str

    def __init__(self, name: str, *, eager: bool = False) -> None:
        super().__init__(name=name)
