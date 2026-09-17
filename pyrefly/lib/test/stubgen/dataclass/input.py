# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field, InitVar
from typing import ClassVar


# The original #3221 reproducer: a non-literal `field(default=None)` should still produce a
# default in the synthesized `__init__`, so `A()` is accepted by checkers against the stub.
@dataclass
class OptionalName:
    name: str | None = field(default=None)


# Mixed field defaults: literal, `field(default=...)`, `field(default_factory=...)`, bare
# `field()`, and a trailing `field(..., kw_only=True)` to force keyword-only positioning.
@dataclass
class Mixed:
    required: int
    literal_default: str = "x"
    field_with_default: int = field(default=0, metadata={"a": 1}, repr=True)
    field_with_none: str | None = field(default=None)
    items: list[str] = field(default_factory=list, metadata={"k": 1})
    complex_factory: list[int] = field(default_factory=lambda: [1, 2, 3])
    field_no_default: int = field()
    field_ellipsis: int = field(..., kw_only=True)


# `init=False` keeps the attribute typed on the class but drops it from `__init__`.
@dataclass
class WithInitFalse:
    path: str
    cached: dict[str, str] = field(default_factory=dict, init=False)


# `InitVar` participates in `__init__` (unwrapped to its inner type) but does not become a
# stored attribute. `ClassVar` stays on the class body and is omitted from `__init__`.
@dataclass
class WithInitVarAndClassVar:
    sentinel: ClassVar[str] = "x"
    raw: InitVar[bytes | None]
    text: str = field(default="")


# `@dataclass(kw_only=True)` makes every constructor parameter keyword-only.
@dataclass(kw_only=True)
class AllKwOnly:
    a: int
    b: str = "y"


# A user-defined `__init__` must be kept verbatim; stubgen must not synthesize a replacement.
@dataclass
class CustomInit:
    x: int

    def __init__(self, x: int, *, tag: str = "") -> None:
        self.x = x
