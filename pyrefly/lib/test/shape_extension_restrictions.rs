/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

fn shape_extension_env() -> TestEnv {
    TestEnv::one_with_path(
        "shape_extensions",
        "shape_extensions/__init__.pyi",
        r#"
class Flag[T]: ...
class Index: ...
class Int[T]: ...
from typing import _SpecialForm
IntVar: _SpecialForm
class ProxyMethod[T]: ...
"#,
    )
}

fn shape_extension_reexport_env() -> TestEnv {
    let mut env = shape_extension_env();
    env.add(
        "flag_reexport",
        r#"
from shape_extensions import Flag as ReexportedFlag
from shape_extensions import Index as ReexportedIndex
"#,
    );
    env.add("flag_wildcard_reexport", "from flag_reexport import *\n");
    env.add(
        "shape_extensions.torchscript",
        "from shape_extensions import Flag\n",
    );
    env.add(
        "unpack_reexport",
        "from typing import Unpack as ReexportedUnpack\n",
    );
    env
}

testcase!(
    test_index_literal_preservation_and_imports,
    shape_extension_reexport_env(),
    r#"
from types import EllipsisType
from typing import Any, Literal, assert_type
from shape_extensions import Index
from flag_reexport import ReexportedIndex
from flag_wildcard_reexport import ReexportedIndex as WildcardIndex

def identity[I: Index](index: I) -> I: ...
def reexported[I: ReexportedIndex](index: I) -> I: ...
def wildcard[I: WildcardIndex](index: I) -> I: ...

assert_type(identity(0), Literal[0])
def typed(
    part: slice[Literal[1], Literal[5], Literal[2]],
    nested: tuple[slice[Literal[1], None, None], None, EllipsisType],
    advanced: tuple[slice[None, None, None], tuple[Literal[0], Literal[2]]],
) -> None:
    assert_type(identity(part), slice[Literal[1], Literal[5], Literal[2]])
    assert_type(
        identity(nested),
        tuple[slice[Literal[1], None, None], None, EllipsisType],
    )
    assert_type(
        identity(advanced),
        tuple[slice[None, None, None], tuple[Literal[0], Literal[2]]],
    )
assert_type(reexported(-1), Literal[-1])
assert_type(wildcard(...), EllipsisType)

IndexAlias = Index
def aliased[I: IndexAlias](index: I) -> I: ...
assert_type(aliased(3), Literal[3])

broad_int: int = 0
dynamic: Any = 0
unbounded: tuple[int, ...] = (0, 1)
items: list[int] = [0, 1]
valid_union: int | None = 0
assert_type(identity(broad_int), int)
assert_type(identity(dynamic), Any)
assert_type(identity(unbounded), tuple[int, ...])
assert_type(identity(items), list[int])
assert_type(identity([0, 1]), list[int])
assert_type(identity(valid_union), int | None)

identity(True)  # E: `Literal[True]` is not a valid `Index` value for type variable `I`
identity("bad")  # E: `Literal['bad']` is not a valid `Index` value for type variable `I`
identity((slice(None), "bad"))  # E: is not a valid `Index` value for type variable `I`
assert_type(identity((..., ...)), tuple[EllipsisType, EllipsisType])
bad_union: int | str = 0
identity(bad_union)  # E: is not a valid `Index` value for type variable `I`
bad_items: list[str] = []
identity(bad_items)  # E: is not a valid `Index` value for type variable `I`
class Array: ...
class Tensor: ...
identity(Array())  # E: is not a valid `Index` value for type variable `I`
identity(Tensor())  # E: is not a valid `Index` value for type variable `I`
"#,
);

testcase!(
    test_index_symbolic_slices_and_unpacked_tuples,
    shape_extension_env(),
    r#"
from typing import Literal, assert_type
from shape_extensions import Flag, Index, Int, IntVar

def identity[I: Index](index: I) -> I: ...

def symbolic[N: IntVar](
    n: Int[N],
    first: slice[Int[N], None, Literal[-1]],
    nested: tuple[slice[None, Int[N], None], Literal[0]],
) -> None:
    assert_type(identity(first), slice[Int[N], None, Literal[-1]])
    assert_type(
        identity(nested),
        tuple[slice[None, Int[N], None], Literal[0]],
    )

def unpacked[*Ts](index: tuple[int, *Ts]) -> None:
    assert_type(identity(index), tuple[int, *Ts])

def invalid_nested_unpack[*Ts](index: tuple[tuple[str, *Ts]]) -> None:
    identity(index)  # E: is not a valid `Index` value for type variable `I`

def bounded[T: int](value: T) -> None:
    assert_type(identity(value), T)

def constrained[T: (Literal[1], Literal[2])](value: T) -> None:
    assert_type(identity(value), T)

def slice_bound[T: slice](value: T) -> None:
    assert_type(identity(value), T)

def optional_int_bound[T: int | None](value: T) -> None:
    assert_type(identity(value), T)

def integer_flag[K: Flag[int]](value: K) -> None:
    assert_type(identity(value), K)

def optional_slice_bound[T: int | None](value: slice[T, None, None]) -> None:
    assert_type(identity(value), slice[T, None, None])

def integer_tuple_flag[K: Flag[tuple[int, ...]]](value: K) -> None:
    assert_type(identity(value), K)

def integer_union(value: tuple[Literal[1] | Literal[2]]) -> None:
    assert_type(identity(value), tuple[Literal[1] | Literal[2]])
"#,
);

testcase!(
    test_index_sources_and_defaults,
    shape_extension_env(),
    r#"
from typing import Literal, assert_type
from shape_extensions import Index

def default_parameter[I: Index](index: I = 1) -> I: ...
def bad_parameter_default[I: Index](index: I = "bad") -> I: ...  # E: Default for parameter binding `Index`
def default_type_parameter[I: Index = 2](index: I = 2) -> I: ...
def tuple_type_parameter[I: Index = tuple[Literal[0], Literal[1]]](index: I = (0, 1)) -> I: ...
def bad_type_parameter_default[I: Index = str](index: I = ...) -> I: ...  # E: Default for `Index` type parameter `I` is not a valid index value

assert_type(default_parameter(), Literal[1])
assert_type(default_type_parameter(), Literal[2])
assert_type(tuple_type_parameter(), tuple[Literal[0], Literal[1]])

def no_source[I: Index](value: int) -> I: ...  # E: `Index` type parameter `I` must directly annotate exactly one function parameter, found 0
def two_sources[I: Index](left: I, right: I) -> I: ...  # E: `Index` type parameter `I` must directly annotate exactly one function parameter, found 2
def unpacked_source[I: Index](*indices: *I) -> I: ...  # E: `Index` type parameter `I` cannot bind an unpacked parameter
class BadClass[I: Index]: ...  # E: `Index` type parameters are not supported on classes
type BadAlias[I: Index] = I  # E: `Index` type parameters are not supported on type aliases
"#,
);

fn shape_extension_stub_default_env() -> TestEnv {
    let mut env = shape_extension_env();
    env.add_with_path(
        "flag_defaults",
        "flag_defaults.pyi",
        r#"
from shape_extensions import Flag

def no_default[K: Flag[int]](k: K = ...) -> K: ...
def type_parameter_default[K: Flag[int] = 3](k: K = ...) -> K: ...
"#,
    );
    env
}

#[test]
fn test_flag_same_module_marker_provenance() {
    let (state, handle) = TestEnv::one(
        "shape_extensions",
        r#"
from typing import Literal, assert_type

class Flag[T]: ...
class Index: ...

def canonical[K: Flag[int]](k: K) -> K: ...
assert_type(canonical(1), Literal[1])
def canonical_index[I: Index](index: I) -> I: ...
assert_type(canonical_index(1), Literal[1])

def scope() -> None:
    class Flag[T]: ...
    class Index: ...
    def shadowed[K: Flag[int]](k: K) -> K: ...
    def shadowed_index[I: Index](index: I) -> I: ...
    local_index = Index()
    assert_type(shadowed_index(local_index), Index)

class Namespace:
    class Flag[T]: ...
    def shadowed[K: Flag[int]](self, k: K) -> K: ...

assert_type(
    Namespace().shadowed(Namespace.Flag[int]()),
    Namespace.Flag[int],
)
"#,
    )
    .to_state();
    let handle = handle("shape_extensions");
    let errors = state
        .transaction()
        .get_errors([&handle])
        .collect_display_errors();
    assert!(errors.is_empty(), "{errors:?}");
}

testcase!(
    test_flag_literal_preservation_and_imports,
    shape_extension_env(),
    r#"
from typing import Any, Literal, assert_type
from shape_extensions import Flag
from shape_extensions import Flag as RenamedFlag
import shape_extensions as se
import shape_extensions

IntAlias = int

def direct[K: Flag[int]](k: K) -> tuple[K, K]: ...
def renamed[K: RenamedFlag[bool]](k: K) -> K: ...
def qualified[K: se.Flag[str]](k: K) -> K: ...
def qualified_unaliased[K: shape_extensions.Flag[bool]](k: K) -> K: ...
def inner_alias[K: Flag[IntAlias]](k: K) -> K: ...

assert_type(direct(1), tuple[Literal[1], Literal[1]])
assert_type(renamed(True), Literal[True])
assert_type(qualified("x"), Literal["x"])
assert_type(qualified_unaliased(False), Literal[False])
assert_type(inner_alias(2), Literal[2])

broad_int: int = 1
dynamic: Any = 1
mode: Literal["nearest", "bilinear"] = "nearest"
assert_type(direct(broad_int), tuple[int, int])
assert_type(direct(dynamic), tuple[Any, Any])
assert_type(qualified(mode), Literal["nearest", "bilinear"])

direct(True)  # E: `Literal[True]` is not a valid `Flag[int]` value for type variable `K`
mixed: int | str = 1
direct(mixed)  # E: `int | str` is not a valid `Flag[int]` value for type variable `K`
"#,
);

// A `str` domain has to admit `str` subclasses, which are indistinguishable from `str` for the
// literal preservation a Flag parameter performs, while still rejecting unrelated scalars.
testcase!(
    test_flag_string_assignability,
    shape_extension_env(),
    r#"
from typing import Any, Literal, LiteralString, assert_type, overload
from shape_extensions import Flag

class Mode(str): ...

def carry[K: Flag[str]](value: K) -> K: ...
def optional[K: Flag[str | None]](value: K) -> K: ...
def string_or_int[K: Flag[str | int]](value: K) -> K: ...

@overload
def rollback[K: Flag[str | None]](value: K, branch: Literal[0]) -> tuple[K]: ...
@overload
def rollback[K: Flag[str | int]](value: K, branch: Literal[1]) -> list[K]: ...
def rollback(value: str | int | None, branch: int) -> object: ...

assert_type(carry("literal"), Literal["literal"])
assert_type(optional("literal"), Literal["literal"])
assert_type(optional(None), None)
assert_type(string_or_int("literal"), Literal["literal"])
assert_type(string_or_int(1), Literal[1])

broad: str = "broad"
literal_string: LiteralString = "literal string"
mode = Mode()
dynamic: Any = "dynamic"
string_union: str | Mode = "union"
optional_mode: Mode | None = mode
mode_or_int: Mode | int = mode

assert_type(carry(broad), str)
assert_type(carry(literal_string), LiteralString)
assert_type(carry(mode), Mode)
assert_type(carry(dynamic), Any)
assert_type(carry(string_union), str | Mode)
assert_type(optional(broad), str)
assert_type(optional(literal_string), LiteralString)
assert_type(optional(mode), Mode)
assert_type(optional(optional_mode), Mode | None)
assert_type(string_or_int(broad), str)
assert_type(string_or_int(literal_string), LiteralString)
assert_type(string_or_int(mode), Mode)
assert_type(string_or_int(mode_or_int), Mode | int)
assert_type(rollback(mode_or_int, 1), list[Mode | int])

carry(1)  # E: not a valid `Flag[str]` value
nonstr_union: str | int = 1
carry(nonstr_union)  # E: not a valid `Flag[str]` value
optional(1.5)  # E: not a valid `Flag[str | None]` value
optional(True)  # E: not a valid `Flag[str | None]` value
string_or_int(1.5)  # E: not a valid `Flag[int | str]` value
string_or_int(True)  # E: not a valid `Flag[int | str]` value
"#,
);

testcase!(
    test_flag_single_direct_binding_source,
    shape_extension_env(),
    r#"
from shape_extensions import Flag
from typing import Literal, assert_type

type Carrier[T] = T

def missing[K: Flag[int]](x: int) -> K: ...  # E: `Flag` type parameter `K` must directly annotate exactly one function parameter, found 0
def multiple[K: Flag[bool]](x: K, y: K) -> K: ...  # E: `Flag` type parameter `K` must directly annotate exactly one function parameter, found 2
def wrapped[K: Flag[int]](x: Carrier[K]) -> K: ...  # E: `Flag` type parameter `K` must directly annotate exactly one function parameter, found 0
def union_wrapped[K: Flag[int]](x: K | None) -> K: ...  # E: `Flag` type parameter `K` must directly annotate exactly one function parameter, found 0
def variadic[K: Flag[int]](*args: K) -> K: ...  # E: `Flag` type parameter `K` must directly annotate exactly one function parameter, found 0
def keywords[K: Flag[int]](**kwargs: K) -> K: ...  # E: `Flag` type parameter `K` must directly annotate exactly one function parameter, found 0
def invalid_unpacked[K: Flag[int]](*args: *K) -> K: ...  # E: requires a domain containing an integer tuple

def variadic_tuple[Ks: Flag[tuple[int, ...]]](*args: *Ks) -> Ks: ...
def fixed_tuple[Ks: Flag[tuple[int, int]]](*args: *Ks) -> Ks: ...
def mixed_domain[Ks: Flag[int | tuple[int, ...]]](*args: *Ks) -> Ks: ...

invalid_unpacked(1, 2)  # E: not a valid `Flag[int]` value
assert_type(variadic_tuple(), tuple[()])
assert_type(variadic_tuple(1, 2), tuple[Literal[1], Literal[2]])
dimensions: tuple[Literal[3], Literal[4]] = (3, 4)
assert_type(variadic_tuple(*dimensions), tuple[Literal[3], Literal[4]])
assert_type(fixed_tuple(5, 6), tuple[Literal[5], Literal[6]])
assert_type(fixed_tuple(), tuple[()])  # E: not a valid `Flag[tuple[int, int]]` value
assert_type(fixed_tuple(5), tuple[Literal[5]])  # E: not a valid `Flag[tuple[int, int]]` value
assert_type(fixed_tuple(5, 6, 7), tuple[Literal[5], Literal[6], Literal[7]])  # E: not a valid `Flag[tuple[int, int]]` value
assert_type(mixed_domain(7, 8), tuple[Literal[7], Literal[8]])

assert_type(multiple(True, True), Literal[True])
multiple(True, False)  # E: Argument `Literal[False]` is not assignable to parameter `y` with type `Literal[True]`

type InvalidAlias[K: Flag[int]] = K  # E: `Flag` type parameters are not supported on type aliases
"#,
);

testcase!(
    test_class_flag_constructor_inference,
    shape_extension_env(),
    r#"
from typing import Any, Literal, TypedDict, assert_type
from shape_extensions import Flag

class Control[K: Flag[int]]:
    def __init__(self, value: K) -> None: ...
    def get(self) -> K: ...

class DefaultControl[K: Flag[int]]:
    def __init__(self, value: K = -1) -> None: ...
    def get(self) -> K: ...

class Pair[Count: Flag[int], Label: Flag[str]]:
    def __init__(self, label: Label, count: Count) -> None: ...
    def get(self) -> tuple[Count, Label]: ...

class RequiredValue(TypedDict):
    value: Literal[4]

assert_type(Control(1), Control[Literal[1]])
assert_type(Control(value=2).get(), Literal[2])
assert_type(Control(*(3,)).get(), Literal[3])
required: RequiredValue = {"value": 4}
assert_type(Control(**required).get(), Literal[4])
assert_type(DefaultControl().get(), Literal[-1])
assert_type(Pair("item", 6).get(), tuple[Literal[6], Literal["item"]])

broad: int = 1
dynamic: Any = 1
values: tuple[int, ...] = ()
keywords: dict[str, int] = {}
assert_type(Control(broad), Control[int])
assert_type(Control(dynamic), Control[Any])
assert_type(DefaultControl(*values).get(), int)
assert_type(DefaultControl(**keywords).get(), int)

def bare_is_gradual(control: Control) -> None:
    assert_type(control.get(), Any)
"#,
);

testcase!(
    test_class_flag_literal_preservation_in_inferred_fields,
    shape_extension_env(),
    r#"
from shape_extensions import Flag, ProxyMethod
from typing import Any, Literal, assert_type

class Control[K: Flag[int]]:
    __call__: ProxyMethod["forward"]
    def __init__(self, value: K) -> None: ...
    def forward(self) -> K: ...

class Box[T]:
    def __init__(self, value: T) -> None: ...

class Mixed[K: Flag[int], T]:
    def __init__(self, control: K, value: T) -> None: ...

class Holder:
    def __init__(self) -> None:
        self.control = Control(1)
        self.box = Box(1)
        self.plain = 1
        self.mixed = Mixed(2, 3)
        self.nested = ((Control(4),),)
        self.sequence = (Control(5), Control(6))

class BareHolder:
    def __init__(self, control: Control) -> None:
        self.control = control

holder = Holder()
assert_type(holder.control, Control[Literal[1]])
assert_type(holder.control.forward(), Literal[1])
assert_type(holder.control(), Literal[1])
assert_type(holder.box, Box[int])
assert_type(holder.plain, int)
assert_type(holder.mixed, Mixed[Literal[2], int])
assert_type(holder.nested, tuple[tuple[Control[Literal[4]]]])
assert_type(holder.sequence, tuple[Control[Literal[5]], Control[Literal[6]]])

def bare_is_gradual(holder: BareHolder) -> None:
    assert_type(holder.control, Control[Any])
    assert_type(holder.control(), Any)
"#,
);

testcase!(
    test_flag_literal_style_survives_generic_boundaries,
    shape_extension_env(),
    r#"
from typing import Any, Literal, LiteralString, assert_type, overload
from shape_extensions import Flag, ProxyMethod

class Control[K: Flag[int]]:
    __call__: ProxyMethod["forward"]
    def __init__(self, value: K) -> None: ...
    def forward(self) -> K: ...

class DefaultControl[K: Flag[int]]:
    def __init__(self, value: K = -1) -> None: ...
    def get(self) -> K: ...

class TupleControl[K: Flag[tuple[int, int]]]:
    def __init__(self, value: K) -> None: ...
    def get(self) -> K: ...

class Box[T]:
    def __init__(self, value: T) -> None: ...

class Tagged[T](str):
    pass

class Sequence[*Ts]:
    def __init__(self, *values: *Ts) -> None: ...
    def values(self) -> tuple[*Ts]: ...

class Base[T]:
    def __init__(self, value: T) -> None: ...
    def get(self) -> T: ...

class Inherited[K: Flag[int]](Base[K]):
    pass

def identity[T](value: T) -> T: ...
def from_list[T](values: list[T]) -> T: ...
def capture[K: Flag[int]](value: K) -> K: ...
def capture_str[K: Flag[str]](value: K) -> K: ...
def capture_bool[K: Flag[bool]](value: K) -> K: ...
def capture_none[K: Flag[None]](value: K) -> K: ...

assert_type(identity(Control(1)), Control[Literal[1]])
assert_type(from_list([Control(2)]), Control[Literal[2]])
assert_type(
    Sequence(Control(3), Box(4), 5).values(),
    tuple[Control[Literal[3]], Box[int], int],
)
assert_type(
    Sequence((Control(6),), Control(7)).values(),
    tuple[tuple[Control[Literal[6]]], Control[Literal[7]]],
)
assert_type(DefaultControl().get(), Literal[-1])
assert_type(TupleControl((1, 2)).get(), tuple[Literal[1], Literal[2]])
assert_type(
    identity(TupleControl((3, 4))).get(),
    tuple[Literal[3], Literal[4]],
)
assert_type(
    Sequence(TupleControl((5, 6))).values()[0].get(),
    tuple[Literal[5], Literal[6]],
)
assert_type(Inherited(8).get(), Literal[8])

partial = []
partial.append(Control(18))
assert_type(partial[0], Control[Literal[18]])

choice: Literal[9] | Literal[10] = 9
explicit: Literal[11] = 11
broad: int = 12
dynamic: Any = 13
assert_type(capture(choice), Literal[9] | Literal[10])
assert_type(capture(explicit), Literal[11])
assert_type(capture(broad), int)
assert_type(capture(dynamic), Any)
assert_type(identity(capture("bad")), str)  # E: not a valid `Flag[int]` value

tagged: Tagged[Literal[19]] = Tagged()
tagged_or_literal: Tagged[Literal[19]] | Literal["fallback"] = tagged
assert_type(capture_str(tagged), Tagged[Literal[19]])
assert_type(
    capture_str(tagged_or_literal),
    Tagged[Literal[19]] | Literal["fallback"],
)
literal_string: LiteralString = "literal string"
assert_type(capture_str(literal_string), LiteralString)
assert_type(identity(capture_str(literal_string)), LiteralString)
assert_type(capture_bool(True), Literal[True])
assert_type(identity(capture_bool(False)), Literal[False])
assert_type(capture_none(None), None)
assert_type(identity(capture_none(None)), None)

@overload
def rollback[K: Flag[int]](value: K, branch: Literal[0]) -> tuple[K]: ...
@overload
def rollback[T](value: T, branch: Literal[1]) -> list[T]: ...
def rollback(value: object, branch: int) -> object: ...

assert_type(rollback(14, 0), tuple[Literal[14]])
assert_type(rollback(14, 1), list[int])

class Holder:
    def __init__(self) -> None:
        self.sequence = Sequence(Control(15), Box(16), 17)

holder = Holder()
assert_type(
    holder.sequence.values(),
    tuple[Control[Literal[15]], Box[int], int],
)
assert_type(holder.sequence.values()[0](), Literal[15])
"#,
);

// A literal upper bound blocks ordinary promotion, so `Tagged("a")` is a `str` subclass whose
// class argument is a genuinely inferred *implicit* literal. Accepting it as a `Flag[str]`
// control has to reach that nested argument: without `capture_str`, `identity(tagged)` widens
// it to `Tagged[str]`.
testcase!(
    test_flag_literal_style_marks_nested_class_argument_literals,
    shape_extension_env(),
    r#"
from typing import Literal, assert_type
from shape_extensions import Flag

class Tagged[T: Literal["a", "b"]](str):
    def __init__(self, value: T) -> None: ...

def identity[T](value: T) -> T: ...
def capture_str[K: Flag[str]](value: K) -> K: ...

tagged = Tagged("a")
assert_type(identity(capture_str(tagged)), Tagged[Literal["a"]])
"#,
);

testcase!(
    test_class_flag_constructor_source_validation,
    shape_extension_env(),
    r#"
from shape_extensions import Flag

type Carrier[T] = T

class Missing[K: Flag[int]]:  # E: must directly annotate exactly one constructor parameter, found 0
    pass

class Multiple[K: Flag[int]]:  # E: must directly annotate exactly one constructor parameter, found 2
    def __init__(self, x: K, y: K) -> None: ...

class Wrapped[K: Flag[int]]:  # E: must directly annotate exactly one constructor parameter, found 0
    def __init__(self, x: Carrier[K]) -> None: ...

class UnionWrapped[K: Flag[int]]:  # E: must directly annotate exactly one constructor parameter, found 0
    def __init__(self, x: K | None) -> None: ...

class Variadic[K: Flag[int]]:  # E: must directly annotate exactly one constructor parameter, found 0
    def __init__(self, *args: K) -> None: ...

class Keywords[K: Flag[int]]:  # E: must directly annotate exactly one constructor parameter, found 0
    def __init__(self, **kwargs: K) -> None: ...

broad: int = 1
class BroadDefault[K: Flag[int]]:  # E: Default for parameter binding
    def __init__(self, value: K = broad) -> None: ...

class Shadowed[K: Flag[int]]:  # E: must directly annotate exactly one constructor parameter, found 0
    def __init__[K](self, value: K) -> None: ...  # E: Type parameter `K` shadows a type parameter of the same name from an enclosing scope
"#,
);

testcase!(
    test_class_flag_new_constructor,
    shape_extension_env(),
    r#"
from typing import Literal, Self, assert_type
from shape_extensions import Flag

# An overridden `__new__` suppresses the inherited `object.__init__`, so it can be the only
# constructor phase that binds a `Flag` parameter.
class Created[K: Flag[int]]:
    def __new__(cls, value: K) -> Self: ...
    def get(self) -> K: ...

assert_type(Created(1), Created[Literal[1]])
assert_type(Created(value=2).get(), Literal[2])

# `__new__` and `__init__` may each bind a different `Flag` parameter.
class Split[Count: Flag[int], Label: Flag[str]]:
    def __new__(cls, count: Count, label: str) -> Self: ...
    def __init__(self, count: int, label: Label) -> None: ...
    def get(self) -> tuple[Count, Label]: ...

assert_type(Split(3, "item").get(), tuple[Literal[3], Literal["item"]])

# When both constructor phases bind the same `Flag`, they must agree on how callers supply it.
class SameSource[K: Flag[int]]:
    def __new__(cls, value: K = 1) -> Self: ...
    def __init__(self, value: K = 1) -> None: ...
    def get(self) -> K: ...

assert_type(SameSource().get(), Literal[1])
assert_type(SameSource(2).get(), Literal[2])

class MismatchedName[K: Flag[int]]:  # E: must bind from the same constructor argument
    def __new__(cls, value: K) -> Self: ...
    def __init__(self, renamed: K) -> None: ...

class MismatchedPosition[K: Flag[int]]:  # E: must bind from the same constructor argument
    def __new__(cls, value: K, other: int) -> Self: ...
    def __init__(self, other: int, value: K) -> None: ...

class MismatchedDefault[K: Flag[int]]:  # E: must have the same default
    def __new__(cls, value: K = 1) -> Self: ...
    def __init__(self, value: K = 2) -> None: ...

class MissingInNew[K: Flag[int]]:  # E: must directly annotate exactly one constructor parameter, found 0
    def __new__(cls, value: list[K]) -> Self: ...

class MultipleInNew[K: Flag[int]]:  # E: must directly annotate exactly one constructor parameter, found 2
    def __new__(cls, value: K, other: K) -> Self: ...

broad: int = 1
class BadDefaultInNew[K: Flag[int]]:  # E: Default for parameter binding
    def __new__(cls, value: K = broad) -> Self: ...

# Provenance is recorded for plain generic base classes too, so an inherited `__new__` can bind
# a `Flag` parameter that only the subclass introduces.
class NewBase[T]:
    def __new__(cls, value: T) -> Self: ...
    def get(self) -> T: ...

class InheritedNew[K: Flag[int]](NewBase[K]):
    pass

assert_type(InheritedNew(4).get(), Literal[4])

class WrappedNewBase[T]:
    def __new__(cls, value: list[T]) -> Self: ...

class BadInheritedNew[K: Flag[int]](WrappedNewBase[K]):  # E: must directly annotate exactly one constructor parameter, found 0
    pass
"#,
);

testcase!(
    test_class_flag_inherited_constructor,
    shape_extension_env(),
    r#"
from typing import Literal, assert_type, overload
from shape_extensions import Flag

class Base[T]:
    def __init__(self, value: T) -> None: ...
    def get(self) -> T: ...

class Propagated[K: Flag[int]](Base[K]):
    pass

class Fixed(Base[Literal[7]]):
    pass

assert_type(Propagated(5), Propagated[Literal[5]])
assert_type(Propagated(5).get(), Literal[5])
assert_type(Fixed(7).get(), Literal[7])

class WrappedBase[T]:
    def __init__(self, value: list[T]) -> None: ...

class BadInherited[K: Flag[int]](WrappedBase[K]):  # E: must directly annotate exactly one constructor parameter, found 0
    pass

type Carrier[T] = T
class AliasBase[T]:
    def __init__(self, value: Carrier[T]) -> None: ...

class BadInheritedAlias[K: Flag[int]](AliasBase[K]):  # E: must directly annotate exactly one constructor parameter, found 0
    pass

broad: int = 1
class DefaultBase[T]:
    def __init__(self, value: T = broad) -> None: ...

class BadInheritedDefault[K: Flag[int]](DefaultBase[K]):  # E: Default for parameter binding
    pass

class OverloadedBase[T]:
    @overload
    def __init__(self, value: T, mode: Literal[0]) -> None: ...
    @overload
    def __init__(self, value: T, mode: Literal[1]) -> None: ...
    def __init__(self, value: T, mode: Literal[0] | Literal[1]) -> None: ...

class Overloaded[K: Flag[int]](OverloadedBase[K]):
    pass

assert_type(Overloaded(8, 0), Overloaded[Literal[8]])
assert_type(Overloaded(value=9, mode=1), Overloaded[Literal[9]])
mode: Literal[0] | Literal[1] = 0
assert_type(Overloaded(10, mode), Overloaded[Literal[10]])

class MixedBase[T]:
    @overload
    def __init__(self, value: T, mode: Literal[0]) -> None: ...
    @overload
    def __init__(self, value: Carrier[T], mode: Literal[1]) -> None: ...
    def __init__(self, value: T, mode: Literal[0] | Literal[1]) -> None: ...

class BadOverloadBranch[K: Flag[int]](MixedBase[K]):  # E: must directly annotate exactly one constructor parameter, found 0
    pass
"#,
);

testcase!(
    test_class_flag_constructor_field_does_not_bypass_validation,
    shape_extension_env(),
    r#"
from collections.abc import Callable
from shape_extensions import Flag

def replacement(value: int) -> None: ...
def erase_signature[F](fn: F) -> Callable[..., None]: ...

class Assigned[K: Flag[int]]:  # E: must directly annotate exactly one constructor parameter, found 0
    __init__ = replacement

class Annotated[K: Flag[int]]:  # E: must directly annotate exactly one constructor parameter, found 0
    __init__: Callable[[K], None]

class Rebound[K: Flag[int]]:  # E: must directly annotate exactly one constructor parameter, found 0
    def __init__(self, value: K) -> None: ...
    __init__ = replacement

class Decorated[K: Flag[int]]:  # E: must directly annotate exactly one constructor parameter
    @erase_signature
    def __init__(self, value: K) -> None: ...
"#,
);

testcase!(
    test_non_flag_unpacked_varargs_uses_normal_call_check,
    shape_extension_env(),
    r#"
from shape_extensions import Flag
from typing import assert_type

def ordinary[*Ts](*args: *Ts) -> tuple[*Ts]: ...
def ordinary_with_flag[K: Flag[int], *Ts](control: K, *args: *Ts) -> tuple[*Ts]: ...

assert_type(ordinary(1, "x"), tuple[int, str])
values: tuple[int, str] = (1, "x")
assert_type(ordinary(*values), tuple[int, str])
assert_type(ordinary_with_flag(1, "x", 2.0), tuple[str, float])
"#,
);

testcase!(
    test_flag_unpacked_source_is_authoritative,
    shape_extension_env(),
    r#"
from shape_extensions import Flag
from typing import Literal, assert_type

def variadic_with_witness[Ks: Flag[tuple[int, ...]]](
    *args: *Ks, witness: list[Ks]
) -> Ks: ...

matching: list[tuple[Literal[1], Literal[2]]] = [(1, 2)]
assert_type(
    variadic_with_witness(1, 2, witness=matching),
    tuple[Literal[1], Literal[2]],
)

conflicting: list[tuple[Literal[1], Literal[2], Literal[3]]] = [(1, 2, 3)]
variadic_with_witness(1, 2, witness=conflicting)  # E: is not assignable to parameter `witness` with type `list[tuple[Literal[1], Literal[2]]]`
"#,
);

testcase!(
    test_flag_explicit_unpack_source_uses_resolved_type,
    shape_extension_reexport_env(),
    r#"
from flag_reexport import ReexportedFlag
from unpack_reexport import ReexportedUnpack
from typing import Literal, Unpack, assert_type

def builtin_unpack[Ks: ReexportedFlag[tuple[int, ...]]](
    *args: Unpack[Ks],
) -> Ks: ...
def reexported_unpack[Ks: ReexportedFlag[tuple[int, ...]]](
    *args: ReexportedUnpack[Ks],
) -> Ks: ...

assert_type(builtin_unpack(1, 2), tuple[Literal[1], Literal[2]])
assert_type(reexported_unpack(3, 4), tuple[Literal[3], Literal[4]])
"#,
);

testcase!(
    test_flag_direct_source_binds_before_wrapped_occurrences,
    shape_extension_env(),
    r#"
from collections.abc import Callable
from typing import Any, Literal, assert_type
from shape_extensions import Flag

def source_after_wrapped[K: Flag[int]](xs: list[K], k: K) -> K: ...
def string_after_wrapped[M: Flag[str]](xs: list[M], m: M) -> M: ...
def source_after_consumer[K: Flag[int]](consumer: Callable[[K], None], k: K) -> K: ...
def source_between_bounds[K: Flag[int]](
    lower: list[K], k: K, upper: Callable[[K], None]
) -> K: ...
def independent[K: Flag[int], M: Flag[str]](
    xs: list[K], ys: list[M], k: K, m: M
) -> tuple[K, M]: ...

dynamic_items: list[Any] = []
dynamic_strings: list[Any] = []
assert_type(source_after_wrapped(dynamic_items, 2), Literal[2])
assert_type(source_after_wrapped([], 2), Literal[2])
assert_type(string_after_wrapped([], "x"), Literal["x"])
assert_type(
    independent(dynamic_items, dynamic_strings, 3, "x"),
    tuple[Literal[3], Literal["x"]],
)

one_items: list[Literal[1]] = [1]
source_after_wrapped(one_items, 2)  # E: `Literal[1]` is incompatible with selected `Flag` value `Literal[2]` for type variable `K`

def accepts_two(x: Literal[2]) -> None: ...
def accepts_one(x: Literal[1]) -> None: ...
def accepts_three(x: Literal[3]) -> None: ...

assert_type(source_after_consumer(accepts_two, 2), Literal[2])
source_after_consumer(accepts_one, 2)  # E: `Literal[1]` is incompatible with selected `Flag` value `Literal[2]` for type variable `K`
source_between_bounds(one_items, 2, accepts_three)  # E: `Literal[1]` is incompatible with selected `Flag` value `Literal[2]` for type variable `K`  # E: Argument `(x: Literal[3]) -> None` is not assignable to parameter `upper` with type `(Literal[2]) -> None` in function `source_between_bounds`
"#,
);

testcase!(
    test_flag_marker_and_domain_validation,
    shape_extension_env(),
    r#"
from typing import Literal, TypeVar, assert_type
from shape_extensions import Flag as ShapeFlag
from shape_extensions import Index as ShapeIndex

class Flag[T]: ...
def unrelated[K: Flag[int]](k: K) -> K: ...
ordinary_flag = Flag[int]()
assert_type(unrelated(ordinary_flag), Flag[int])

Marker = ShapeFlag
def marker_alias[K: Marker[int]](k: K) -> K: ...
assert_type(marker_alias(1), Literal[1])
def bare_alias[K: Marker](k: K) -> K: ...  # E: `shape_extensions.Flag` requires one domain argument: `int`, `bool`, `str`, `tuple[int, ...]`, `None`, or a union of these
def bare_direct[K: ShapeFlag](k: K) -> K: ...  # E: `shape_extensions.Flag` requires one domain argument: `int`, `bool`, `str`, `tuple[int, ...]`, `None`, or a union of these

Legacy = TypeVar("Legacy", bound=ShapeFlag[int])  # E: `shape_extensions.Flag` is supported only as a direct PEP 695 type parameter bound
LegacyIndex = TypeVar("LegacyIndex", bound=ShapeIndex)  # E: `shape_extensions.Index` is supported only as a direct PEP 695 type parameter bound

def union_domain[K: ShapeFlag[int | str]](k: K) -> K: ...
def noncore_domain[K: ShapeFlag[bytes]](k: K) -> K: ...  # E: `Flag` domain must resolve to a nonempty union
def float_domain[K: ShapeFlag[float]](k: K) -> K: ...  # E: `Flag` domain must resolve to a nonempty union
def fixed_tuple_domain[K: ShapeFlag[tuple[int, int]]](k: K) -> K: ...
def heterogeneous_tuple_domain[K: ShapeFlag[tuple[int, str]]](k: K) -> K: ...  # E: `Flag` domain must resolve to a nonempty union
def narrow_then_wide[K: ShapeFlag[tuple[int] | tuple[int, int]]](k: K) -> K: ...  # E: `Flag` domain must resolve to a nonempty union
def wide_then_narrow[K: ShapeFlag[tuple[int, int] | tuple[int]]](k: K) -> K: ...  # E: `Flag` domain must resolve to a nonempty union
def fixed_then_unbounded[K: ShapeFlag[tuple[int] | tuple[int, int] | tuple[int, ...]]](k: K) -> K: ...
def unbounded_then_fixed[K: ShapeFlag[tuple[int, ...] | tuple[int] | tuple[int, int]]](k: K) -> K: ...
def invalid_then_unbounded[K: ShapeFlag[bytes | tuple[int, ...]]](k: K) -> K: ...  # E: `Flag` domain must resolve to a nonempty union
def unbounded_then_invalid[K: ShapeFlag[tuple[int, ...] | bytes]](k: K) -> K: ...  # E: `Flag` domain must resolve to a nonempty union
"#,
);

testcase!(
    test_flag_fixed_tuple_domains,
    shape_extension_env(),
    r#"
from typing import Any, Literal, assert_type
from shape_extensions import Flag

type Pair = tuple[int, int]

def empty[A: Flag[tuple[()]]](value: A) -> A: ...
def single[A: Flag[tuple[int]]](value: A) -> A: ...
def pair[A: Flag[Pair]](value: A) -> A: ...
def triple[A: Flag[tuple[int, int, int]]](value: A) -> A: ...
def scalar_pair[A: Flag[int | Pair | None]](value: A) -> A: ...
def widened[A: Flag[Pair | tuple[int, ...]]](value: A) -> A: ...

assert_type(empty(()), tuple[()])
assert_type(single((1,)), tuple[Literal[1]])
assert_type(pair((1, 2)), tuple[Literal[1], Literal[2]])
assert_type(triple((1, 2, 3)), tuple[Literal[1], Literal[2], Literal[3]])
assert_type(scalar_pair(1), Literal[1])
assert_type(scalar_pair((1, 2)), tuple[Literal[1], Literal[2]])
assert_type(scalar_pair(None), None)
assert_type(widened((1,)), tuple[Literal[1]])

broad_pair: tuple[int, int] = (1, 2)
broad_tuple: tuple[int, ...] = (1, 2)
compatible_unpacked: tuple[int, *tuple[int, ...]] = (1, 2)
too_long_unpacked: tuple[int, int, int, *tuple[int, ...]] = (1, 2, 3)
dynamic: Any = (1, 2)
assert_type(pair(broad_pair), tuple[int, int])
assert_type(pair(broad_tuple), tuple[int, ...])
assert_type(pair(compatible_unpacked), tuple[int, *tuple[int, ...]])
assert_type(pair(dynamic), Any)

pair(())  # E: not a valid `Flag[tuple[int, int]]` value
pair((1,))  # E: not a valid `Flag[tuple[int, int]]` value
pair((1, 2, 3))  # E: not a valid `Flag[tuple[int, int]]` value
pair((1, "x"))  # E: not a valid `Flag[tuple[int, int]]` value
pair(too_long_unpacked)  # E: not a valid `Flag[tuple[int, int]]` value
scalar_pair((1,))  # E: not a valid `Flag[int | tuple[int, int] | None]` value

def default_pair[A: Flag[Pair]](value: A = (1, 2)) -> A: ...
def wrong_default_pair[A: Flag[Pair]](value: A = (1,)) -> A: ...  # E: Default for parameter binding `Flag[tuple[int, int]]`
assert_type(default_pair(), tuple[Literal[1], Literal[2]])

def accepts_unbounded[A: Flag[tuple[int, ...]]](value: A) -> A: ...
def forwards_pair[A: Flag[Pair]](value: A) -> A:
    return accepts_unbounded(value)
def rejects_unbounded[A: Flag[tuple[int, ...]]](value: A) -> None:
    pair(value)  # E: not a valid `Flag[tuple[int, int]]` value

def construct_pair[A: Flag[Pair]](value: A, kind: type[A]) -> None:
    kind()
def construct_scalar_pair[A: Flag[int | Pair | None]](value: A, kind: type[A]) -> None:
    kind()
"#,
);

testcase!(
    test_flag_union_and_tuple_literal_preservation,
    shape_extension_env(),
    r#"
from typing import Any, Literal, assert_type
from shape_extensions import Flag

type Axis = int | tuple[int, ...] | None
type ReorderedAxis = None | int | tuple[int, ...] | int

def capture[A: Flag[Axis]](axis: A) -> A: ...
def capture_reordered[A: Flag[ReorderedAxis]](axis: A) -> A: ...

assert_type(capture(0), Literal[0])
assert_type(capture(-1), Literal[-1])
assert_type(capture((0, -1)), tuple[Literal[0], Literal[-1]])
assert_type(capture(()), tuple[()])
assert_type(capture(None), None)
assert_type(capture_reordered((1, 2)), tuple[Literal[1], Literal[2]])

broad_int: int = 0
broad_tuple: tuple[int, ...] = (0, 1)
broad_unpacked: tuple[Literal[0], *tuple[int, ...], Literal[-1]] = (0, 1, -1)
broad_axis: Axis = 0
dynamic: Any = 0
assert_type(capture(broad_int), int)
assert_type(capture(broad_tuple), tuple[int, ...])
assert_type(
    capture(broad_unpacked),
    tuple[Literal[0], *tuple[int, ...], Literal[-1]],
)
assert_type(capture(broad_axis), Axis)
assert_type(capture(dynamic), Any)

capture(True)  # E: `Literal[True]` is not a valid `Flag[int | tuple[int, ...] | None]` value
capture((0, "x"))  # E: is not a valid `Flag[int | tuple[int, ...] | None]` value
"#,
);

testcase!(
    test_flag_union_defaults_context_and_overloads,
    shape_extension_env(),
    r#"
from typing import Literal, assert_type, overload, reveal_type
from shape_extensions import Flag

type Axis = int | tuple[int, ...] | None

def runtime_default[A: Flag[Axis]](axis: A = None) -> A: ...
def type_default[A: Flag[Axis] = None](axis: A = None) -> A: ...
def tuple_type_default[
    A: Flag[tuple[int, ...]] = tuple[Literal[0], Literal[-1]]
](axis: A = (0, -1)) -> A: ...
def bool_default[A: Flag[bool]](enabled: A = False) -> A: ...

assert_type(runtime_default(), None)
assert_type(type_default(), None)
assert_type(tuple_type_default(), tuple[Literal[0], Literal[-1]])
assert_type(tuple_type_default((1,)), tuple[Literal[1]])
assert_type(bool_default(), Literal[False])
assert_type(bool_default(True), Literal[True])

precise: tuple[Literal[0], Literal[-1]] = runtime_default((0, -1))
wrong: tuple[Literal[1], Literal[-1]] = runtime_default((0, -1))  # E: not assignable

@overload
def pick[A: Flag[Axis]](axis: A, mode: Literal[0]) -> A: ...
@overload
def pick[A: Flag[Axis]](axis: A, mode: Literal[1]) -> tuple[A]: ...
def pick(axis: Axis, mode: int) -> Axis | tuple[Axis]: ...

assert_type(pick((0, -1), 0), tuple[Literal[0], Literal[-1]])
assert_type(pick((0, -1), 1), tuple[tuple[Literal[0], Literal[-1]]])

def accepts_union[A: Flag[int | str]](value: A) -> A: ...
def capture_int[A: Flag[int]](value: A) -> A: ...
def constrained[T: (int, str)](value: T) -> T: ...
def forwards_int[A: Flag[int]](value: A) -> A:
    return accepts_union(value)
def forwards_constraints[A: Flag[int | str]](value: A) -> A:
    return constrained(value)
def rejects_union[A: Flag[int | str]](value: A) -> None:
    capture_int(value)  # E: is not a valid `Flag[int]` value

def rejects_wrong_tuple_element[A: Flag[tuple[int, ...]]](value: A) -> tuple[str, ...]:
    return value  # E: not assignable

def merge_narrowed[A: Flag[int | str]](value: A) -> None:
    if isinstance(value, int):
        merged = value
    else:
        merged = value
    reveal_type(merged)  # E: revealed type: A
"#,
);

// A union domain makes `type[K]` resolve to several constructors at once. Targeting the call
// used to assume a single class, so this pins that a union domain stays a graceful diagnostic.
testcase!(
    test_flag_union_constructor_target_does_not_panic,
    shape_extension_env(),
    r#"
from shape_extensions import Flag

def construct[K: Flag[int | str]](value: K, cls: type[K]) -> K:
    return cls()
"#,
);

testcase!(
    test_flag_reexports_activate_marker_syntax,
    shape_extension_reexport_env(),
    r#"
from typing import Literal, assert_type
from flag_reexport import ReexportedFlag
from flag_wildcard_reexport import ReexportedFlag as WildcardFlag
from shape_extensions.torchscript import Flag as TorchscriptFlag

def reexported[K: ReexportedFlag[int]](k: K) -> K: ...
def wildcard[K: WildcardFlag[int]](k: K) -> K: ...
def torchscript[K: TorchscriptFlag[int]](k: K) -> K: ...

assert_type(reexported(1), Literal[1])
assert_type(wildcard(2), Literal[2])
assert_type(torchscript(3), Literal[3])
"#,
);

testcase!(
    test_flag_default_precedence_and_validation,
    shape_extension_env(),
    r#"
from typing import Literal, assert_type
from shape_extensions import Flag

def runtime_default[K: Flag[int]](k: K = 1) -> K: ...
def both_defaults[K: Flag[int] = 2](k: K = 1) -> K: ...

assert_type(runtime_default(), Literal[1])
assert_type(both_defaults(), Literal[1])
assert_type(both_defaults(3), Literal[3])

source_wins: Literal[1] = runtime_default()
wrong_hint: Literal[2] = runtime_default()  # E: `Literal[1]` is not assignable to `Literal[2]`

def bad_type_default[K: Flag[int] = "x"](k: K) -> K: ...  # E: Default for `Flag[int]` type parameter `K` must be a `int` literal, got `Literal['x']`
def bad_runtime_default[K: Flag[int]](k: K = "x") -> K: ...  # E: Default for parameter binding `Flag[int]` type parameter `K` must be a `int` literal, got `Literal['x']`
def bad_runtime_bool[K: Flag[int]](k: K = True) -> K: ...  # E: Default for parameter binding `Flag[int]` type parameter `K` must be a `int` literal, got `Literal[True]`

computed_default: int = 1
def broad_runtime_default[K: Flag[int]](k: K = computed_default) -> K: ...  # E: Default for parameter binding `Flag[int]` type parameter `K` must be a `int` literal, got `int`
assert_type(broad_runtime_default(), int)

def ordinary[T](x: T = 1) -> T: ...
assert_type(ordinary(), int)
"#,
);

testcase!(
    test_flag_overload_inference,
    shape_extension_env(),
    r#"
from typing import Literal, assert_type, overload
from shape_extensions import Flag

@overload
def pick[K: Flag[int]](k: K, mode: Literal[0]) -> K: ...
@overload
def pick[K: Flag[int]](k: K, mode: Literal[1]) -> tuple[K]: ...
def pick(k: int, mode: int) -> int | tuple[int]: ...

assert_type(pick(3, 0), Literal[3])
assert_type(pick(4, 1), tuple[Literal[4]])
"#,
);

testcase!(
    test_flag_does_not_block_other_contextual_type_variables,
    shape_extension_env().enable_implicit_any_lambda_error(),
    r#"
from collections.abc import Callable
from typing import Literal
from shape_extensions import Flag

def mixed[K: Flag[int], T](k: K, callback: Callable[[T], None]) -> tuple[K, T]: ...
def takes_int(value: int) -> None: ...

result: tuple[Literal[5], str] = mixed(5, lambda value: print(value.upper()))
bad: tuple[Literal[5], str] = mixed(5, takes_int)  # E: `tuple[Literal[5], int | str]` is not assignable to `tuple[Literal[5], str]`
"#,
);

testcase!(
    test_flag_materialized_bound_and_type_constructor,
    shape_extension_env(),
    r#"
from shape_extensions import Flag

def materialized_subset[K: Flag[int]](source: K) -> int | str:
    return source

def construct[K: Flag[int]](source: K, cls: type[K]) -> K:
    return cls(unknown=source)  # E: No matching overload found for function `int.__new__`
"#,
);

testcase!(
    test_flag_forwarding_methods_and_call_forms,
    shape_extension_env(),
    r#"
from typing import TYPE_CHECKING, Literal, TypedDict, assert_type

if TYPE_CHECKING:
    from shape_extensions import Flag

def identity[K: Flag[int]](value: K) -> K: ...
def forward[K: Flag[int]](value: K) -> K:
    return identity(value)
def quoted_source[K: Flag[int]](value: "K") -> K: ...

class Selector:
    def select[K: Flag[str]](self, value: K) -> K: ...

class IdentityKwargs(TypedDict):
    value: Literal[9]

assert_type(forward(4), Literal[4])
assert_type(quoted_source(5), Literal[5])
assert_type(Selector().select("x"), Literal["x"])
assert_type(identity(value=6), Literal[6])

args: tuple[Literal[8]] = (8,)
assert_type(identity(*args), Literal[8])
kwargs: IdentityKwargs = {"value": 9}
assert_type(identity(**kwargs), Literal[9])
"#,
);

testcase!(
    test_flag_stub_ellipsis_defaults,
    shape_extension_stub_default_env(),
    r#"
from typing import Any, Literal, assert_type
from flag_defaults import no_default, type_parameter_default

assert_type(no_default(), Any)
assert_type(type_parameter_default(), Literal[3])
"#,
);
