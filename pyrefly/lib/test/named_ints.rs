/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

fn env() -> TestEnv {
    TestEnv::one_with_path(
        "shape_extensions",
        "shape_extensions/__init__.pyi",
        r#"
class NamedInts: ...
class CaptureNamedInts[T](int): ...
"#,
    )
}

testcase!(
    captures_named_integer_kwargs,
    env(),
    r#"
from shape_extensions import CaptureNamedInts, NamedInts
from typing import NotRequired, reveal_type, TypedDict

class Box[T]: ...

def capture[Axes: NamedInts](**axes: CaptureNamedInts[Axes]) -> Box[Axes]: ...

reveal_type(capture())  # E: revealed type: Box[NamedInts[]]
reveal_type(capture(b=2, a=3))  # E: revealed type: Box[NamedInts[a=Int[3], b=Int[2]]]
reveal_type(capture(**{"copies": 4}))  # E: revealed type: Box[NamedInts[copies=Int[int]]]

class Axes(TypedDict, closed=True):
    height: int
    width: NotRequired[int]

def typed_dict(axes: Axes) -> None:
    reveal_type(capture(**axes))  # E: revealed type: Box[NamedInts[height=Int[int], width?=Int[int]]]

class OpenAxes(TypedDict):
    height: int

def open_typed_dict(axes: OpenAxes) -> None:
    reveal_type(capture(**axes))  # E: revealed type: Box[NamedInts[height=Int[int], ...]]

class Mixed(TypedDict, closed=True):
    label: str
    width: int

def capture_extra_axes[Axes: NamedInts](label: str, **axes: CaptureNamedInts[Axes]) -> Box[Axes]: ...

def mixed(values: Mixed) -> None:
    reveal_type(capture_extra_axes(**values))  # E: revealed type: Box[NamedInts[width=Int[int]]]

def dynamic(axes: dict[str, int]) -> None:
    reveal_type(capture(**axes))  # E: revealed type: Box[NamedInts[...]]

capture(bad="x")  # E: Keyword argument `bad` with type `Literal['x']` is not assignable to parameter `**axes` with type `int`
"#,
);

testcase!(
    named_int_capture_is_explicitly_opt_in,
    env(),
    r#"
from typing import assert_type, Callable

def ordinary[T](callback: Callable[[int], T], **kwargs: int) -> T: ...

assert_type(ordinary(lambda x: x + 1, width=2), int)
"#,
);

testcase!(
    validates_named_int_capture_declarations,
    env(),
    r#"
from shape_extensions import CaptureNamedInts, NamedInts
from typing import reveal_type

def unrestricted[T](**axes: CaptureNamedInts[T]) -> T: ...  # E: `CaptureNamedInts` argument must be a `NamedInts` type parameter
def positional[Axes: NamedInts](axes: CaptureNamedInts[Axes]) -> None: ...  # E: `CaptureNamedInts` is supported only as a `**kwargs` annotation
def bare(**axes: CaptureNamedInts) -> None: ...  # E: `CaptureNamedInts` argument must be a `NamedInts` type parameter
def defaulted[Axes: NamedInts = int](**axes: CaptureNamedInts[Axes]) -> Axes: ...  # E: `NamedInts` type parameter `Axes` cannot have a default
def orphan[Axes: NamedInts]() -> Axes: ...  # E: `NamedInts` type parameter `Axes` must source exactly one `CaptureNamedInts` parameter, found 0
def duplicate[Axes: NamedInts](  # E: `NamedInts` type parameter `Axes` must source exactly one `CaptureNamedInts` parameter, found 2
    axes: CaptureNamedInts[Axes],  # E: `CaptureNamedInts` is supported only as a `**kwargs` annotation
    **more: CaptureNamedInts[Axes],
) -> Axes: ...
type BadAlias[Axes: NamedInts] = tuple[Axes]  # E: `NamedInts` type parameters are not supported on type aliases

class MissingConstructorSource[Axes: NamedInts]:  # E: `NamedInts` type parameter `Axes` must source exactly one constructor `CaptureNamedInts` parameter, found 0
    def __init__(self) -> None: ...

class ValidConstructorSource[Axes: NamedInts]:
    def __init__(self, **axes: CaptureNamedInts[Axes]) -> None: ...

reveal_type(ValidConstructorSource(width=3))  # E: revealed type: ValidConstructorSource[NamedInts[width=Int[3]]]
"#,
);
