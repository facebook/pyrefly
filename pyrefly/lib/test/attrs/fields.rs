/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::attrs_testcase;
use crate::test::attrs::util::attrs_env;
use crate::testcase;

// `@a.default` supplies the default, so `a` is optional and the `a.default` access resolves.
attrs_testcase!(
    field_default_decorator,
    r#"
from attrs import define, field

@define
class C:
    a: dict = field()

    @a.default
    def _default_a(self):
        return {}

C()    # OK
C({})  # OK
"#,
);

attrs_testcase!(
    field_validator_decorator,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    @x.validator
    def _check_x(self, attribute, value):
        if value < 0:
            raise ValueError("x must be non-negative")

C()   # E: Missing argument `x`
C(1)  # OK
"#,
);

// `in_(EnumClass)` infers `_ValidatorType[object]`, but the annotation stays authoritative (#3429).
attrs_testcase!(
    test_attrs_field_validator_does_not_widen_annotation,
    r#"
from enum import Enum
from attrs import define, field, validators

class Color(Enum):
    RED = 1
    GREEN = 2

@define
class C:
    color: Color = field(validator=validators.in_(Color))

C(Color.RED)  # OK
"#,
);

// A field's declared type flows to its `__init__` param, so construction args are
// type-checked: passing a `str` for an `int` field is an error.
attrs_testcase!(
    test_attrs_field_no_converter_construct_typecheck,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

C("nope")  # E: not assignable to parameter `x`
"#,
);

// With a `converter=`, the `__init__` param takes the converter's input type (`str`),
// while the stored attribute keeps the declared/output type (`int`).
attrs_testcase!(
    test_attrs_field_with_converter_still_uses_converter_input,
    r#"
from typing import assert_type, reveal_type
from attrs import define, field

def to_int(s: str) -> int:
    return int(s)

@define
class C:
    x: int = field(converter=to_int)

reveal_type(C.__init__)  # E: revealed type: (self: C, x: str) -> None
c = C("5")
assert_type(c.x, int)
"#,
);

// An overloaded converter only contributes the input types of overloads callable with a single
// positional argument; an overload requiring a second positional arg is ignored.
attrs_testcase!(
    test_attrs_field_overloaded_converter_single_positional,
    r#"
from typing import overload, reveal_type
from attrs import define, field

@overload
def conv(x: int) -> str: ...
@overload
def conv(x: str, y: int) -> str: ...
def conv(x: object, y: int = 0) -> str:
    return str(x)

@define
class C:
    a: str = field(converter=conv)

reveal_type(C.__init__)  # E: revealed type: (self: C, a: int) -> None
"#,
);

// An overload requiring a second keyword-only argument is also ignored (it can't be called with
// a single positional arg).
attrs_testcase!(
    test_attrs_field_overloaded_converter_required_kwonly,
    r#"
from typing import overload, reveal_type
from attrs import define, field

@overload
def conv(x: int) -> str: ...
@overload
def conv(x: bytes, *, mode: int) -> str: ...
def conv(x: object, *, mode: int = 0) -> str:
    return str(x)

@define
class C:
    a: str = field(converter=conv)

reveal_type(C.__init__)  # E: revealed type: (self: C, a: int) -> None
"#,
);

// A generic-class converter (`list[int]`) applies its type arguments: the `__init__` param
// takes the parameterized constructor's input type, not `Any`.
attrs_testcase!(
    test_attrs_field_generic_class_converter,
    r#"
from typing import assert_type
from attrs import define, field

@define
class C:
    xs: list[int] = field(converter=list[int])

assert_type(C([1, 2, 3]).xs, list[int])
C(5)  # E: not assignable to parameter `xs`
"#,
);

// The element type of a builtin generic converter is enforced: `list[int]` accepts `Iterable[int]`,
// so a `list[str]` argument is rejected.
attrs_testcase!(
    test_attrs_field_generic_converter_wrong_element,
    r#"
from attrs import define, field

@define
class C:
    xs: list[int] = field(converter=list[int])

C(["a"])  # E: not assignable to parameter `xs` with type `Iterable[int]`
"#,
);

// A user-defined generic converter applies its type argument directly: `Box[int]`'s `__init__`
// parameter `T` becomes `int`, while the stored attribute keeps the declared `Box[int]`.
attrs_testcase!(
    test_attrs_field_user_generic_converter,
    r#"
from typing import assert_type
from attrs import define, field

class Box[T]:
    def __init__(self, x: T) -> None: ...

@define
class C:
    b: Box[int] = field(converter=Box[int])

assert_type(C(5).b, Box[int])
C("x")  # E: not assignable to parameter `b` with type `int`
"#,
);

// The type argument is substituted into nested positions of the converter's parameter: `Sink[int]`,
// whose `__init__` takes `list[T]`, yields an `__init__` parameter of `list[int]`.
attrs_testcase!(
    test_attrs_field_generic_converter_nested_typevar,
    r#"
from attrs import define, field

class Sink[T]:
    def __init__(self, xs: list[T]) -> None: ...

@define
class C:
    s: Sink[int] = field(converter=Sink[int])

C(5)  # E: not assignable to parameter `s` with type `list[int]`
"#,
);

// A bare (unsubscripted) generic converter takes its element type from the field annotation:
// `list` for a `list[int]` field gives an `Iterable[int]` param.
attrs_testcase!(
    test_attrs_field_bare_generic_converter,
    r#"
from attrs import define, field

@define
class C:
    xs: list[int] = field(converter=list)

C(5)  # E: not assignable to parameter `xs` with type `Iterable[int]`
"#,
);

// `tuple`'s constructor is `(Iterable[_T]) -> tuple[_T, ...]`. Solving that output against the
// `Sequence[int]` annotation gives `_T = int`, so the `__init__` param is the converter's input,
// `Iterable[int]`, while the attribute keeps the declared `Sequence[int]`.
attrs_testcase!(
    test_attrs_field_bare_tuple_converter_passthrough,
    r#"
from typing import Sequence, assert_type, reveal_type
from attrs import define, field

@define
class C:
    a: Sequence[int] = field(converter=tuple)

reveal_type(C.__init__)  # E: revealed type: (self: C, a: Iterable[int]) -> None
assert_type(C([1, 2, 3]).a, Sequence[int])
C(["1", 2, 3])  # E: not assignable to parameter `a` with type `Iterable[int]`
"#,
);

// When the converter's output cannot satisfy the annotation (`tuple[_T, ...]` is never an `int`),
// there is no element type to solve, so the `__init__` param falls back to `Iterable[Unknown]`.
attrs_testcase!(
    test_attrs_field_bare_converter_incompatible_annotation,
    r#"
from typing import reveal_type
from attrs import define, field

@define
class C:
    a: int = field(converter=tuple)

reveal_type(C.__init__)  # E: revealed type: (self: C, a: Iterable[Unknown]) -> None
"#,
);

// The solved converter input type carries over to a field inherited from a base class.
attrs_testcase!(
    test_attrs_field_bare_converter_inherited,
    r#"
from typing import Sequence
from attrs import define, field

@define
class Base:
    a: Sequence[int] = field(converter=tuple)

@define
class Sub(Base):
    pass

Sub(["1", 2, 3])  # E: not assignable to parameter `a` with type `Iterable[int]`
"#,
);

// A generic function converter (`def identity[T](x: T) -> T`) is type-preserving, so solving its
// return against the `int` annotation gives an `__init__` param of `int` rather than a leaked `T`.
attrs_testcase!(
    test_attrs_field_generic_function_converter,
    r#"
from typing import assert_type, reveal_type
from attrs import define, field

def identity[T](x: T) -> T:
    return x

@define
class C:
    x: int = field(converter=identity)

reveal_type(C.__init__)  # E: revealed type: (self: C, x: int) -> None
assert_type(C(42).x, int)
C("nope")  # E: not assignable to parameter `x`
"#,
);

// A stdlib generic function converter is solved the same way, so its input is the field type
// instead of a leaked type variable.
attrs_testcase!(
    test_attrs_field_deepcopy_converter,
    r#"
import copy
from attrs import define, field

@define
class C:
    x: int = field(converter=copy.deepcopy)

C(42)     # OK
C("no")   # E: not assignable to parameter `x`
"#,
);

// A type argument is substituted into nested positions of the converter input, so a
// `(list[T]) -> list[T]` converter against a `list[int]` field gives a `list[int]` param.
attrs_testcase!(
    test_attrs_field_generic_function_converter_nested_typevar,
    r#"
from attrs import define, field

def clone_all[T](xs: list[T]) -> list[T]:
    return list(xs)

@define
class C:
    xs: list[int] = field(converter=clone_all)

C([1, 2])   # OK
C(["a"])    # E: not assignable to parameter `xs`
"#,
);

// The solved generic-function converter input carries over to a field inherited from a base class.
attrs_testcase!(
    test_attrs_field_generic_function_converter_inherited,
    r#"
from attrs import define, field

def identity[T](x: T) -> T:
    return x

@define
class Base:
    x: int = field(converter=identity)

@define
class Sub(Base):
    pass

Sub(5)       # OK
Sub("nope")  # E: not assignable to parameter `x`
"#,
);

// `attr.converters.optional(c)` makes the `__init__` param the inner converter's input type
// unioned with `None`.
attrs_testcase!(
    test_attrs_field_converters_optional,
    r#"
from attrs import define, field
import attr

@define
class C:
    x: int = field(converter=attr.converters.optional(int))

C(None)     # OK: optional converter accepts None
C([1, 2])   # E: not assignable to parameter `x`
"#,
);

// `attr.converters.pipe(c1, ...)` runs `c1` first, so the `__init__` param takes `c1`'s input type
// (here `int`'s) while the attribute keeps the declared output type. An argument `c1` can't accept
// is rejected, even though a later converter (`str`) could.
attrs_testcase!(
    test_attrs_field_converters_pipe,
    r#"
from typing import assert_type
from attrs import define, field
import attr

@define
class C:
    x: str = field(converter=attr.converters.pipe(int, str))

assert_type(C(3.4).x, str)
C("09")  # OK
C({})    # E: not assignable to parameter `x`
"#,
);

// In a `pipe`, the first converter's output feeds the next converter, not the field, so a bare
// generic first converter is NOT solved against the field annotation: `tuple` here stays
// `Iterable[Unknown]` rather than being (wrongly) solved to `Iterable[bool]` from `Sequence[bool]`.
attrs_testcase!(
    test_attrs_field_pipe_bare_generic_first_not_solved_from_annotation,
    r#"
from typing import Sequence, reveal_type
from attrs import define, field
import attr

def widen(xs: tuple[int, ...]) -> Sequence[bool]: ...

@define
class C:
    a: Sequence[bool] = field(converter=attr.converters.pipe(tuple, widen))

reveal_type(C.__init__)  # E: revealed type: (self: C, a: Iterable[Unknown]) -> None
"#,
);

// `attr.converters.default_if_none(d)` replaces `None` with `d` and passes other values through, so
// the `__init__` param is the field's declared type unioned with `None`, not a useless `Any`.
attrs_testcase!(
    test_attrs_field_converters_default_if_none,
    r#"
from typing import assert_type, reveal_type
from attrs import define, field
import attr

@define
class C:
    x: int = field(converter=attr.converters.default_if_none(42))

reveal_type(C.__init__)  # E: revealed type: (self: C, x: int | None) -> None
assert_type(C(1).x, int)
C(None)  # OK: None becomes the default
C({})    # E: not assignable to parameter `x`
"#,
);

// The `factory=` form behaves the same: non-`None` values pass through, so the input stays
// `field type | None`.
attrs_testcase!(
    test_attrs_field_converters_default_if_none_factory,
    r#"
from attrs import define, field
import attr

@define
class C:
    xs: list[int] = field(converter=attr.converters.default_if_none(factory=list))

C(None)    # OK
C([1, 2])  # OK
C("nope")  # E: not assignable to parameter `xs`
"#,
);

// A `factory=` field is optional in `__init__`, but its param keeps the declared
// annotation type so construction args are still type-checked.
attrs_testcase!(
    test_attrs_field_factory_param_type,
    r#"
from attrs import define, field

@define
class C:
    items: list[int] = field(factory=list)

C()              # OK: factory supplies the default
C("not a list")  # E: not assignable to parameter `items`
"#,
);

// A `factory=` callable's return type must be assignable to the field type, just like an
// explicit `default=` value.
attrs_testcase!(
    test_attrs_field_factory_return_type_mismatch,
    r#"
from attrs import define, field

def make_str() -> str:
    return ""

@define
class C:
    x: int = field(factory=make_str)  # E: `str` is not assignable to `int`
"#,
);

// A matching `factory=` return type is accepted.
attrs_testcase!(
    test_attrs_field_factory_return_type_match,
    r#"
from attrs import define, field

def make_int() -> int:
    return 0

@define
class C:
    x: int = field(factory=make_int)
"#,
);

// A `factory=` whose output feeds a `converter=` is checked against the converter's input,
// not the field type, so a "mismatched" factory return is not flagged.
attrs_testcase!(
    test_attrs_field_factory_with_converter_not_checked,
    r#"
from attrs import define, field

def make_str() -> str:
    return ""

def to_int(s: str) -> int:
    return int(s)

@define
class C:
    x: int = field(factory=make_str, converter=to_int)
"#,
);

// Likewise an explicit `default=` value with a `converter=` is the converter's input, so it
// is not checked against the field type.
attrs_testcase!(
    test_attrs_field_default_with_converter_not_checked,
    r#"
from attrs import define, field

def to_int(s: str) -> int:
    return int(s)

@define
class C:
    x: int = field(default="5", converter=to_int)
"#,
);

// A `converter=` that is itself a type constructor (`converter=int`) is supported: the init
// parameter accepts the constructor's input types while the attribute keeps the converted output
// type, and an argument the constructor can't accept is rejected.
attrs_testcase!(
    test_attrs_field_converter_is_constructor,
    r#"
from typing import assert_type
from attrs import define, field

@define
class C:
    x: int = field(default="5", converter=int)

assert_type(C("5").x, int)
C(b"10")
C([1, 2])  # E: not assignable to parameter `x`
"#,
);

// A `@<field>.converter` decorator (attrs 26.2.0+) supplies the converter. attrs invokes it as
// `converter(self, field, value)`, so the `value` parameter's type becomes the `__init__` param type
// while the attribute keeps its declared output type.
attrs_testcase!(
    test_attrs_field_converter_decorator,
    r#"
from typing import assert_type, reveal_type
from attrs import define, field

@define
class C:
    x: int = field()

    @x.converter
    def _to_int(self, attribute, value: str) -> int:
        return int(value)

reveal_type(C.__init__)  # E: revealed type: (self: C, x: str) -> None
assert_type(C("5").x, int)
C(5)  # E: not assignable to parameter `x`
"#,
);

// A union `value` type flows to the `__init__` parameter, accepting either member and rejecting
// anything else. (The runtime `int(...)` may still raise, but that is not a type error.)
attrs_testcase!(
    test_attrs_field_converter_decorator_union_input,
    r#"
from attrs import define, field

@define
class DecoratorConverter:
    x: int = field()

    @x.converter
    def _to_int(self, attribute, value: str | float) -> int:
        return int(value)

DecoratorConverter("foo")  # OK
DecoratorConverter(1.5)    # OK
DecoratorConverter([])     # E: not assignable to parameter `x`
"#,
);

// An unannotated `value` parameter makes the `__init__` parameter `Any`, so any argument is accepted.
attrs_testcase!(
    test_attrs_field_converter_decorator_unannotated_value,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    @x.converter
    def _to_int(self, attribute, value) -> int:
        return int(value)

C("anything")  # OK
C(123)         # OK
"#,
);

// attrs always calls the decorated converter as `converter(self, field, value)`. A method missing the
// `field` parameter can never be called, so its signature is rejected (this is the naive 2-arg form).
attrs_testcase!(
    test_attrs_field_converter_decorator_too_few_params,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    @x.converter
    def _to_int(self, value: str) -> int:  # E: The `@x.converter` method must accept `(self, field, value)`, but it accepts too few positional parameters
        return int(value)
"#,
);

// A converter with a required parameter beyond `(self, field, value)` can never be called by attrs.
attrs_testcase!(
    test_attrs_field_converter_decorator_too_many_params,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    @x.converter
    def _to_int(self, attribute, value: str, extra: int) -> int:  # E: The `@x.converter` method must accept `(self, field, value)`, but it has required parameters that attrs does not pass
        return int(value)
"#,
);

// attrs passes converter arguments positionally, so a required keyword-only parameter can never be
// filled.
attrs_testcase!(
    test_attrs_field_converter_decorator_required_kwonly,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    @x.converter
    def _to_int(self, attribute, value: str, *, mode: int) -> int:  # E: The `@x.converter` method must accept `(self, field, value)`, but it has a required keyword-only parameter that attrs cannot pass
        return int(value)
"#,
);

// An explicit `converter=` composes before a `@<field>.converter` decorator (attrs `pipe`), so the
// keyword converter's input type is what `__init__` accepts.
attrs_testcase!(
    test_attrs_field_converter_decorator_kwarg_takes_precedence,
    r#"
from typing import reveal_type
from attrs import define, field

def to_int(x: bytes) -> int:
    return 0

@define
class C:
    x: int = field(converter=to_int)

    @x.converter
    def _extra(self, attribute, value: str) -> int:
        return 0

reveal_type(C.__init__)  # E: revealed type: (self: C, x: bytes) -> None
"#,
);

// Multiple `@<field>.converter` methods compose via `pipe` in definition order, so the first one's
// input type is what `__init__` accepts.
attrs_testcase!(
    test_attrs_field_converter_decorator_first_wins,
    r#"
from typing import reveal_type
from attrs import define, field

@define
class C:
    x: int = field()

    @x.converter
    def _first(self, attribute, value: str) -> int:
        return 0

    @x.converter
    def _second(self, attribute, value: bytes) -> int:
        return 0

reveal_type(C.__init__)  # E: revealed type: (self: C, x: str) -> None
"#,
);

// A `default=` feeds the converter, so it is checked against the converter's input type (`str`), not
// the field's declared output type, and the field stays optional.
attrs_testcase!(
    test_attrs_field_converter_decorator_with_default,
    r#"
from attrs import define, field

@define
class C:
    x: int = field(default="5")

    @x.converter
    def _to_int(self, attribute, value: str) -> int:
        return int(value)

C()     # OK
C("9")  # OK
"#,
);

// Legacy `attr.ib` accepts a positional `default`, so it is checked against the annotation.
attrs_testcase!(
    test_attrs_attr_ib_positional_default_checked,
    r#"
import attr

@attr.s(auto_attribs=True)
class C:
    x: int = attr.ib("bad")  # E: `Literal['bad']` is not assignable to `int`
"#,
);

// Next-gen `field` is keyword-only: a positional arg is only an arg-count error and must NOT also
// be treated as a `default` and checked against the annotation (no spurious assignability error).
attrs_testcase!(
    test_attrs_field_positional_not_treated_as_default,
    r#"
from attrs import define, field

@define
class C:
    x: int = field("bad")  # E: No matching overload found
"#,
);

// attrs raises `ValueError` at runtime when both `default=` and `factory=` are given.
attrs_testcase!(
    test_attrs_field_default_and_factory_conflict,
    r#"
from attrs import define, field

@define
class C:
    x: int = field(default=1, factory=int)  # E: cannot specify both `default` and `factory`
"#,
);

// The same conflict applies to the classic `attr.ib()` specifier.
attrs_testcase!(
    test_attrs_attr_ib_default_and_factory_conflict,
    r#"
import attr

@attr.s(auto_attribs=True)
class C:
    x: int = attr.ib(default=1, factory=int)  # E: cannot specify both `default` and `factory`
"#,
);

// `attr.ib` accepts `default` positionally, so a positional default plus `factory=`
// is also a conflict (`field()` is keyword-only, so this can only happen via `attr.ib`).
attrs_testcase!(
    test_attrs_attr_ib_positional_default_and_factory_conflict,
    r#"
import attr

@attr.s(auto_attribs=True)
class C:
    x: int = attr.ib(1, factory=int)  # E: cannot specify both `default` and `factory`
"#,
);

// `default=Factory(...)` is the canonical desugaring of `factory=`: only `default` is
// passed, so it must NOT be reported as a conflict.
attrs_testcase!(
    test_attrs_field_default_factory_value_ok,
    r#"
from attrs import define, field, Factory

@define
class C:
    items: list[int] = field(default=Factory(list))
"#,
);

// A non-specifier call that merely happens to use `default`/`factory` keyword names is
// not a field specifier, so no conflict is reported.
attrs_testcase!(
    test_attrs_non_field_specifier_default_factory_ok,
    r#"
from attrs import define

def helper(default: int, factory: int) -> int:
    return default

@define
class C:
    x: int = helper(default=1, factory=2)
"#,
);

// NOTHING default ⇒ field is required, and the sentinel isn't checked against the annotation.
attrs_testcase!(
    test_attrs_field_nothing_default_required,
    r#"
import attr
from attrs import define, field

@define
class C:
    x: int = field(default=attr.NOTHING)

C()   # E: Missing argument `x`
C(1)  # OK
"#,
);

// Recognized through a variable, since identity follows local bindings.
attrs_testcase!(
    test_attrs_field_nothing_default_via_variable,
    r#"
import attr
from attrs import define, field

SENTINEL = attr.NOTHING

@define
class C:
    x: int = field(default=SENTINEL)

C()  # E: Missing argument `x`
"#,
);

// Also works for `attr.ib()` and a bare `from attr import NOTHING`.
attrs_testcase!(
    test_attrs_attr_ib_nothing_default_required,
    r#"
import attr
from attr import NOTHING

@attr.s(auto_attribs=True)
class C:
    x: int = attr.ib(default=NOTHING)

C()   # E: Missing argument `x`
C(1)  # OK
"#,
);

// Counts as "no default" for ordering: can't follow a field that has one.
attrs_testcase!(
    test_attrs_field_nothing_default_ordering,
    r#"
import attr
from attrs import define, field

@define
class C:
    a: int = field(default=5)
    b: int = field(default=attr.NOTHING)  # E: without a default may not follow
"#,
);

// A non-default field inherited from one base, ordered after a defaulted field from another,
// is a merge-induced ordering error reported at the subclass definition.
attrs_testcase!(
    test_attrs_inherited_nondefault_after_default,
    r#"
from attrs import define, field

@define
class Base:
    a: int = field(default=5)

@define
class Mixin:
    b: int = field()

@define
class Sub(Mixin, Base):  # E: without a default may not follow
    pass
"#,
);

// A conflict contained within a single base is reported once (on that base); a subclass that
// merely inherits it does NOT re-report it.
attrs_testcase!(
    test_attrs_inherited_conflict_not_reported_on_subclass,
    r#"
from attrs import define, field

@define
class Base:
    a: int = field(default=5)
    b: int = field()  # E: without a default may not follow

@define
class Sub(Base):
    pass
"#,
);

// A required field declared in a class that *inherits* a defaulted field: the conflict
// originates at — and is reported once at — that class; subclasses inheriting it stay silent.
attrs_testcase!(
    test_attrs_inherited_default_local_required_not_repeated,
    r#"
from attrs import define, field

@define
class HasDefault:
    where: int = field(default=0)

@define
class Origin(HasDefault):
    arg: int = field()  # E: without a default may not follow

@define
class Inheritor(Origin):
    pass
"#,
);

// Mixed class: NOTHING field required, real-default field optional, declared param types.
attrs_testcase!(
    test_attrs_field_nothing_default_init_signature,
    r#"
import attr
from typing import reveal_type
from attrs import define, field

@define
class C:
    a: int = field(default=attr.NOTHING)
    b: int = field(default=5)

reveal_type(C.__init__)  # E: revealed type: (self: C, a: int, b: int = ...) -> None
"#,
);

// A real default is still type-checked: suppression must not leak to ordinary defaults.
attrs_testcase!(
    test_attrs_field_real_default_still_checked,
    r#"
from attrs import define, field

@define
class C:
    x: int = field(default="oops")  # E: `Literal['oops']` is not assignable to `int`
"#,
);

// `kw_only=True` is orthogonal: a NOTHING field stays required, just keyword-only.
attrs_testcase!(
    test_attrs_field_nothing_default_kw_only,
    r#"
import attr
from typing import reveal_type
from attrs import define, field

@define
class C:
    x: int = field(default=attr.NOTHING, kw_only=True)

reveal_type(C.__init__)  # E: revealed type: (self: C, *, x: int) -> None
"#,
);

// A NOTHING field on a base class stays required in subclasses.
attrs_testcase!(
    test_attrs_field_nothing_default_inherited,
    r#"
import attr
from attrs import define, field

@define
class Base:
    x: int = field(default=attr.NOTHING)

@define
class Sub(Base):
    y: int = field(default=0)

Sub()   # E: Missing argument `x`
Sub(1)  # OK
"#,
);

// Regression: the NOTHING suppression must not hijack ordinary `default=` calls. The annotation
// hint must still flow into the call so an invariant generic resolves to the declared type rather
// than from the argument alone (`ContextVar[None]` vs `ContextVar[str | None]`).
attrs_testcase!(
    test_nothing_suppression_does_not_break_contextvar_default,
    r#"
from contextvars import ContextVar

x: ContextVar[str | None] = ContextVar("x", default=None)  # OK
"#,
);

// Regression: without the hint, a `default=` literal widens (`str` instead of `Literal["tcp"]`)
// and fails the invariant check.
attrs_testcase!(
    test_nothing_suppression_does_not_break_literal_default,
    r#"
from typing import Generic, Literal, TypeVar

T = TypeVar("T")

class Box(Generic[T]):
    def __init__(self, *, default: T) -> None: ...

x: Box[Literal["tcp"]] = Box(default="tcp")  # OK
"#,
);

// A non-specifier call passing `default=attr.NOTHING` is still checked.
attrs_testcase!(
    test_attrs_non_field_call_nothing_default_still_checked,
    r#"
import attr

def f(default: object) -> str:
    return ""

x: int = f(default=attr.NOTHING)  # E: `str` is not assignable to `int`
"#,
);

// `attr.ib`'s first positional arg is `default`, so positional NOTHING ⇒ required. The NOTHING
// sentinel means "no default", so it must not be checked against the field's declared type.
attrs_testcase!(
    test_attrs_field_nothing_positional_required,
    r#"
import attr

@attr.s(auto_attribs=True)
class C:
    x: int = attr.ib(attr.NOTHING)

C()   # E: Missing argument `x`
C(1)  # OK
"#,
);

// Inherited `field()` fields keep their declared param type in the subclass
// `__init__`, collected base-first in MRO order.
attrs_testcase!(
    test_attrs_field_inherited_param_type,
    r#"
from typing import reveal_type
from attrs import define, field

@define
class Base:
    x: int = field()

@define
class Sub(Base):
    y: str = field(default="a")

reveal_type(Sub.__init__)  # E: revealed type: (self: Sub, x: int, y: str = ...) -> None
"#,
);

attrs_testcase!(
    field_validator_decorator_with_explicit_default,
    r#"
from attrs import define, field

@define
class C:
    x: int = field(default=2)
    items: list[int] = field(factory=list)

    @x.validator
    def _check_x(self, attribute, value):
        pass

    @items.validator
    def _check_items(self, attribute, value):
        pass

C()  # OK
"#,
);

// The `Any` retype applies only to attrs specifiers: a `@x.default` on a plain class still errors.
attrs_testcase!(
    non_attrs_default_decorator_still_errors,
    r#"
class C:
    x: int = 0

    @x.default  # E: Object of class `int` has no attribute `default`
    def _x(self):
        return 0
"#,
);

attrs_testcase!(
    field_validator_decorator_multiple,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    @x.validator
    def _a(self, attribute, value):
        pass

    @x.validator
    def _b(self, attribute, value):
        pass

C(1)  # OK: validators are additive
"#,
);

// The `Any` retype is attrs-specific: a stdlib `@dataclass` field keeps its declared type, so
// `@x.default` errors there too.
attrs_testcase!(
    dataclass_field_default_decorator_still_errors,
    r#"
from dataclasses import dataclass, field

@dataclass
class C:
    x: int = field()

    @x.default  # E: Object of class `int` has no attribute `default`
    def _x(self):
        return 0
"#,
);

attrs_testcase!(
    field_default_and_validator_decorator,
    r#"
from attrs import define, field

@define
class C:
    a: int = field()

    @a.default
    def _a(self):
        return 0

    @a.validator
    def _check_a(self, attribute, value):
        pass

C()   # OK
C(1)  # OK
"#,
);

attrs_testcase!(
    field_default_decorator_legacy_attr_ib,
    r#"
import attr

@attr.s
class C:
    x = attr.ib()

    @x.default
    def _x(self):
        return 0

C()   # OK
C(1)  # OK
"#,
);

attrs_testcase!(
    field_default_decorator_legacy_attr_ib_with_type,
    r#"
import attr
from typing import reveal_type

@attr.s
class C:
    x = attr.ib(type=int)

    @x.default
    def _x(self):
        return 0

reveal_type(C.__init__)  # E: revealed type: (self: C, x: int = ...) -> None
C(1)  # OK
"#,
);

// A string `type=` is a forward reference: it resolves to the real type (here `list[int]`) for the
// synthesized parameter, instead of being rejected as a `Literal[str]` value.
attrs_testcase!(
    test_attrs_legacy_attr_ib_type_string_forward_ref,
    r#"
import attr
from typing import reveal_type

@attr.s
class C:
    x = attr.ib(type='list[int]')

reveal_type(C.__init__)  # E: revealed type: (self: C, x: list[int]) -> None
C([1])  # OK
C(0)    # E: Argument `Literal[0]` is not assignable to parameter `x`
"#,
);

// A forward reference to a class defined later in the module resolves too, since the string is
// bound as a type during binding rather than evaluated as a value.
attrs_testcase!(
    test_attrs_legacy_attr_ib_type_string_later_class,
    r#"
import attr
from typing import reveal_type

@attr.s
class C:
    other = attr.ib(type='D')

class D:
    pass

reveal_type(C.__init__)  # E: revealed type: (self: C, other: D) -> None
"#,
);

// A bare builtin name string resolves like an annotation, so construction is type-checked.
attrs_testcase!(
    test_attrs_attrib_type_string_builtin,
    r#"
import attr

@attr.s
class C:
    x = attr.ib(type='int')

attr.fields(C)  # keep `attr` used
C(5)            # OK
C("bad")        # E: Argument `Literal['bad']` is not assignable to parameter `x`
"#,
);

// A dotted/qualified name string resolves through the module it names.
attrs_testcase!(
    test_attrs_attrib_type_string_qualified,
    r#"
from typing import assert_type
import typing
import attr

@attr.s
class C:
    x = attr.ib(type='typing.List[int]')

assert_type(C([1]).x, list[int])
"#,
);

// A union written as a string resolves to the union type.
attrs_testcase!(
    test_attrs_attrib_type_string_union,
    r#"
from typing import assert_type
import attr

@attr.s
class C:
    x = attr.ib(type='int | None')

assert_type(C(None).x, int | None)
assert_type(C(5).x, int | None)
"#,
);

// A string naming the enclosing class resolves (self-referential forward reference).
attrs_testcase!(
    test_attrs_attrib_type_string_self_reference,
    r#"
from typing import assert_type
import attr

@attr.s
class C:
    nxt = attr.ib(type='C | None')

assert_type(C(None).nxt, "C | None")
"#,
);

// Unlike legacy `attr.ib` (whose `type=` is typed `object`), next-gen `field` types `type=` as a
// real `type[...]`, so a forward-reference string is rejected by the stub rather than resolved.
attrs_testcase!(
    test_attrs_field_type_string_rejected_by_stub,
    r#"
from attrs import define, field

@define
class C:
    x = field(type='str')  # E: Argument `Literal['str']` is not assignable to parameter `type`
"#,
);

// An unresolvable name in the string is reported, like any bad forward reference.
attrs_testcase!(
    test_attrs_attrib_type_string_unknown_name,
    r#"
import attr

@attr.s
class C:
    x = attr.ib(type='Nonexistent')  # E: Could not find name `Nonexistent`
"#,
);

attrs_testcase!(
    field_default_decorator_ordering,
    r#"
from attrs import define, field

@define
class C:
    a: int = field()

    @a.default
    def _a(self):
        return 0

    b: int = field()  # E: without a default may not follow
"#,
);

attrs_testcase!(
    field_default_decorator_kw_only,
    r#"
from typing import reveal_type
from attrs import define, field

@define
class C:
    x: int = field(kw_only=True)

    @x.default
    def _x(self):
        return 0

reveal_type(C.__init__)  # E: revealed type: (self: C, *, x: int = ...) -> None
"#,
);

// The `@a.default` param keeps the field's declared type (`int`), not the in-body specifier `Any`.
attrs_testcase!(
    field_default_decorator_init_signature,
    r#"
from typing import reveal_type
from attrs import define, field

@define
class C:
    b: int = field()
    a: int = field()

    @a.default
    def _a(self):
        return 0

reveal_type(C.__init__)  # E: revealed type: (self: C, b: int, a: int = ...) -> None
"#,
);

attrs_testcase!(
    field_default_decorator_inherited,
    r#"
from attrs import define, field

@define
class Base:
    x: int = field()

    @x.default
    def _x(self):
        return 0

@define
class Sub(Base):
    y: int = field(default=1)

Sub()   # OK
Sub(0)  # OK
"#,
);

// A subclass override may add a `@x.default`, making the field optional in the subclass only.
attrs_testcase!(
    field_default_decorator_override_in_subclass,
    r#"
from attrs import define, field

@define
class Base:
    x: int = field()

@define
class Sub(Base):
    x: int = field()

    @x.default
    def _x(self):
        return 0

Base()  # E: Missing argument `x`
Sub()   # OK
"#,
);

// A `@x.default` method decorates the `field()` object named in the current class body (see the
// decorator form under attrs "Defaults": https://www.attrs.org/en/stable/init.html#defaults), so a
// subclass re-declares `x` rather than decorating the inherited field; that re-declaration replaces
// (not combines with) the parent's default, so it is not a conflict.
attrs_testcase!(
    field_default_decorator_override_parent_default,
    r#"
from attrs import define, field

@define
class Base:
    x: int = field(default=1)

@define
class Sub(Base):
    x: int = field()

    @x.default
    def _x(self):
        return 0

Sub()  # OK
"#,
);

// `init=False` excludes the field from `__init__` regardless of the `@x.default`.
attrs_testcase!(
    field_default_decorator_init_false,
    r#"
from typing import reveal_type
from attrs import define, field

@define
class C:
    x: int = field(init=False)

    @x.default
    def _x(self):
        return 0

reveal_type(C.__init__)  # E: revealed type: (self: C) -> None
"#,
);

// Cross-module: a `@x.default` field in another module is still optional in the subclass.
testcase!(
    field_default_decorator_inherited_cross_module,
    {
        let mut env = attrs_env();
        env.add(
            "base",
            r#"
from attrs import define, field

@define
class Base:
    x: int = field()

    @x.default
    def _x(self):
        return 0
"#,
        );
        env
    },
    r#"
from attrs import define, field
from base import Base

@define
class Sub(Base):
    y: int = field(default=1)

Sub()  # OK
"#,
);

attrs_testcase!(
    field_default_decorator_nested_in_control_flow,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    if True:
        @x.default
        def _x(self):
            return 0

C()   # OK
C(1)  # OK
"#,
);

// The decorator scan stops at nested scopes, so `Inner`'s `@x.default` does not make `C.x` optional.
attrs_testcase!(
    field_default_decorator_not_leaked_from_nested_class,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    @define
    class Inner:
        x: int = field()

        @x.default
        def _x(self):
            return 0

C()          # E: Missing argument `x`
C.Inner()    # OK
"#,
);

// An undefined name in `@<name>.default` is an unbound-name error and does not make any field optional.
attrs_testcase!(
    field_default_decorator_undefined_name,
    r#"
from attrs import define, field

@define
class C:
    a: int = field()

    @b.default  # E: Could not find name `b`
    def _b(self):
        return 0

C()  # E: Missing argument `a`
"#,
);

// `@x.default` in a subclass cannot target a field inherited from a base: `x` is not in the
// subclass body, so it errors rather than making the inherited field optional.
attrs_testcase!(
    field_default_decorator_targets_inherited_field,
    r#"
from attrs import define, field

@define
class Base:
    x: int = field()

@define
class Sub(Base):
    @x.default  # E: Object of class `int` has no attribute `default`
    def _x(self):
        return 0
"#,
);

// `default=` and a `@x.default` method are mutually exclusive (attrs raises `DefaultAlreadySetError`).
attrs_testcase!(
    field_default_decorator_conflicts_with_explicit_default,
    r#"
from attrs import define, field

@define
class C:
    x: int = field(default=1)  # E: cannot specify both an explicit default and a

    @x.default
    def _x(self):
        return 0
"#,
);

// `factory=` is also a default, so it likewise conflicts with a `@x.default` method.
attrs_testcase!(
    field_default_decorator_conflicts_with_factory,
    r#"
from attrs import define, field

@define
class C:
    x: list[int] = field(factory=list)  # E: cannot specify both an explicit default and a

    @x.default
    def _x(self):
        return []
"#,
);

// `default=attr.NOTHING` means "no default", so a `@x.default` is the sole default, not a conflict.
attrs_testcase!(
    field_default_decorator_with_nothing_default,
    r#"
import attr
from attrs import define, field

@define
class C:
    x: int = field(default=attr.NOTHING)

    @x.default
    def _x(self):
        return 0

C()   # OK
C(1)  # OK
"#,
);

// `attr.ib`'s positional default also conflicts with a `@x.default` method.
attrs_testcase!(
    field_default_decorator_conflicts_with_positional_attr_ib,
    r#"
import attr

@attr.s
class C:
    x = attr.ib(5)  # E: cannot specify both an explicit default and a

    @x.default
    def _x(self):
        return 0
"#,
);

// attrs raises `DefaultAlreadySetError` for a second `@x.default` on the same field.
attrs_testcase!(
    field_default_decorator_duplicate,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()  # E: `x` cannot have more than one `@x.default` method

    @x.default
    def _a(self):
        return 0

    @x.default
    def _b(self):
        return 1
"#,
);

// A duplicate `@x.default` with a mismatched return type reports only the duplicate error.
attrs_testcase!(
    field_default_decorator_duplicate_skips_return_type_check,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()  # E: `x` cannot have more than one `@x.default` method

    @x.default
    def _a(self) -> str:
        return "a"

    @x.default
    def _b(self) -> str:
        return "b"
"#,
);

// The `@x.default` method's return type must be assignable to the field's declared type.
attrs_testcase!(
    field_default_decorator_return_type_mismatch,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()  # E: Return type `str` of the `@x.default` method is not assignable to field `x` of type `int`

    @x.default
    def _x(self) -> str:
        return "oops"
"#,
);

attrs_testcase!(
    field_default_decorator_return_type_match,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    @x.default
    def _x(self):
        return 0

C()  # OK
"#,
);

// With a converter the default flows through the converter's input type, so the return type is
// not checked against the field type.
attrs_testcase!(
    field_default_decorator_with_converter_not_checked,
    r#"
from attrs import define, field

def to_int(s: str) -> int:
    return int(s)

@define
class C:
    x: int = field(converter=to_int)

    @x.default
    def _x(self):
        return "0"

C()  # OK
"#,
);

// The return-type check resolves through string/forward-ref annotations (PEP 563).
attrs_testcase!(
    field_default_decorator_return_type_mismatch_forward_ref,
    r#"
from __future__ import annotations
from attrs import define, field

@define
class C:
    x: int = field()  # E: Return type `str` of the `@x.default` method is not assignable to field `x` of type `int`

    @x.default
    def _x(self) -> str:
        return "oops"
"#,
);

// The `@x.default` method is called as `meth(self)`; requiring another argument is an error.
attrs_testcase!(
    field_default_decorator_wrong_signature,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    @x.default
    def _x(self, extra):  # E: The `@x.default` method must be callable with no argument other than `self`, but it has required parameters that attrs does not pass
        return 0
"#,
);

// The `@x.validator` method is called as `validator(self, attribute, value)`; fewer params is an error.
attrs_testcase!(
    field_validator_decorator_wrong_signature,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    @x.validator
    def _check(self):  # E: The `@x.validator` method must accept `(self, attribute, value)`, but it accepts too few positional parameters
        pass
"#,
);

// `*args` absorbs attrs' call shape for both decorators, so neither signature is flagged.
attrs_testcase!(
    field_decorator_signature_varargs_ok,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    @x.default
    def _x(self, *args):
        return 0

    @x.validator
    def _check(self, *args):
        pass

C(1)  # OK
"#,
);

// `*args` does not satisfy a required parameter: `extra` is still unfilled by attrs' `meth(self)`.
attrs_testcase!(
    field_default_decorator_required_arg_with_varargs,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    @x.default
    def _x(self, extra, *args):  # E: The `@x.default` method must be callable with no argument other than `self`, but it has required parameters that attrs does not pass
        return 0
"#,
);

// A required keyword-only parameter can never be filled by attrs' positional call.
attrs_testcase!(
    field_validator_decorator_required_kwonly,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    @x.validator
    def _check(self, attribute, value, *, k):  # E: The `@x.validator` method must accept `(self, attribute, value)`, but it has a required keyword-only parameter that attrs cannot pass
        pass
"#,
);

// The default method, too, cannot be passed a required keyword-only parameter.
attrs_testcase!(
    field_default_decorator_required_kwonly,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    @x.default
    def _x(self, *, k):  # E: The `@x.default` method must be callable with no argument other than `self`, but it has a required keyword-only parameter that attrs cannot pass
        return 0
"#,
);

// attrs passes exactly `(self, attribute, value)`; an extra required positional is unfillable.
attrs_testcase!(
    field_validator_decorator_too_many_required,
    r#"
from attrs import define, field

@define
class C:
    x: int = field()

    @x.validator
    def _check(self, attribute, value, extra):  # E: The `@x.validator` method must accept `(self, attribute, value)`, but it has required parameters that attrs does not pass
        pass
"#,
);

// attrs keeps only the LAST of a duplicated field name, at the last position with the last type
// (desired: `(self: C, y: int, x: str)`). pyrefly keeps the FIRST position and loses the type —
// the root is a binding-phase bug (scope.rs `Static::upsert` re-keys the rebound name to
// `Anywhere`), separate from the `get_dataclass_fields` reorder. Tracked for a follow-up.
attrs_testcase!(
    bug = "Same-class duplicate field keeps first position and loses the override type",
    test_attrs_same_class_duplicate_field,
    r#"
from typing import reveal_type
import attr

@attr.s
class C:
    x: int = attr.ib()
    y: int = attr.ib()
    x: str = attr.ib()  # E: `x` cannot be annotated with `str`, it is already defined with type `int`

reveal_type(C.__init__)  # E: revealed type: (self: C, x: int, y: int) -> None
"#,
);

// The eq/order/cmp combination rules apply to field specifiers too, not just the decorator.
attrs_testcase!(
    test_attrs_field_eq_false_order_true,
    r#"
from attrs import define, field

@define
class C:
    x: int = field(eq=False, order=True)  # E: `order` cannot be True when `eq` is False
"#,
);

// Classic `attr.ib` rejects `cmp` mixed with `eq`/`order`.
attrs_testcase!(
    test_attrs_attr_ib_cmp_with_eq,
    r#"
import attr

@attr.s
class C:
    x = attr.ib(cmp=True, eq=True)  # E: Cannot mix `cmp` with `eq` or `order`
"#,
);

// A callable `eq` (a key function) is truthy, so `order=True` alongside it is legal.
attrs_testcase!(
    test_attrs_field_callable_eq_with_order_ok,
    r#"
from attrs import define, field

@define
class C:
    x: int = field(eq=str, order=True)
"#,
);

// Per-field `on_setattr=setters.frozen` makes only that field read-only; siblings stay writable.
attrs_testcase!(
    test_attrs_field_on_setattr_frozen,
    r#"
from attr import define, field, setters

@define
class C:
    x: int = field(on_setattr=setters.frozen)
    y: int = field()

c = C(1, 2)
c.x = 5  # E: Cannot set field `x`
c.y = 5  # OK
"#,
);

// A hook list bound to a variable still freezes (detection is type-based, not syntactic). The
// separate overload error is `list` invariance on `on_setattr`, orthogonal to read-only detection.
attrs_testcase!(
    test_attrs_field_on_setattr_frozen_via_list_variable,
    r#"
from attr import define, field, setters

hooks = [setters.frozen]

@define
class C:
    x: int = field(on_setattr=hooks)  # E: No matching overload found for function `attrs.field`

C(1).x = 5  # E: Cannot set field `x`
"#,
);

// A list with several hooks infers a `list[<union of hook types>]`; `frozen` anywhere in the union
// makes the field read-only (the union branch of the type walk).
attrs_testcase!(
    test_attrs_field_on_setattr_frozen_via_list_variable_union,
    r#"
from attr import define, field, setters

hooks = [setters.validate, setters.frozen]

@define
class C:
    x: int = field(on_setattr=hooks)  # E: No matching overload found for function `attrs.field`

C(1).x = 5  # E: Cannot set field `x`
"#,
);

// Negative: a variable holding only non-frozen hooks leaves the field writable.
attrs_testcase!(
    test_attrs_field_on_setattr_non_frozen_variable_writable,
    r#"
from attr import define, field, setters

hooks = [setters.validate]

@define
class C:
    x: int = field(on_setattr=hooks)  # E: No matching overload found for function `attrs.field`

C(1).x = 5  # OK: no `frozen` hook
"#,
);

// A single `frozen` hook bound to a variable freezes the field with no overload noise (a lone hook
// is a plain callable the stub accepts, unlike an invariant `list`/`tuple` of hooks).
attrs_testcase!(
    test_attrs_field_on_setattr_frozen_single_hook_variable,
    r#"
from attr import define, field, setters

hook = setters.frozen

@define
class C:
    x: int = field(on_setattr=hook)

C(1).x = 5  # E: Cannot set field `x`
"#,
);

// Negative: a single non-frozen hook bound to a variable leaves the field writable.
attrs_testcase!(
    test_attrs_field_on_setattr_non_frozen_single_hook_variable,
    r#"
from attr import define, field, setters

hook = setters.validate

@define
class C:
    x: int = field(on_setattr=hook)

C(1).x = 5  # OK: not the `frozen` hook
"#,
);

// Read-only-ness from a variable hook is per-field: a sibling without `on_setattr` stays writable.
attrs_testcase!(
    test_attrs_field_on_setattr_frozen_variable_is_per_field,
    r#"
from attr import define, field, setters

frozen_hook = setters.frozen

@define
class C:
    x: int = field(on_setattr=frozen_hook)
    y: int = field()

c = C(1, 2)
c.x = 5  # E: Cannot set field `x`
c.y = 5  # OK
"#,
);

// Tuple-unpacked assignment of parallel specifiers declares one field per name, just as if each
// were written on its own line (so `type=`, `default=`, etc. all apply per element).
attrs_testcase!(
    test_attrs_tuple_unpacked_specifiers,
    r#"
from typing import reveal_type
import attr

@attr.s
class A:
    x, y, z = attr.ib(), attr.ib(type=int), attr.ib(default=17)

reveal_type(A.__init__)  # E: revealed type: (self: A, x: Any, y: int, z: int = ...) -> None
A(1, 2, 3)  # OK
"#,
);

// Only the specifier elements become fields; a plain value bound alongside is an ordinary class var.
attrs_testcase!(
    test_attrs_tuple_unpacked_mixed,
    r#"
from typing import reveal_type
import attr

@attr.s
class A:
    x, y = attr.ib(), 5

reveal_type(A.__init__)  # E: revealed type: (self: A, x: Any) -> None
"#,
);

// Tuple-unpacked specifiers in a base class are inherited like any other attrs fields.
attrs_testcase!(
    test_attrs_tuple_unpacked_specifiers_inherited,
    r#"
from typing import reveal_type
import attr

@attr.s
class Base:
    x, y = attr.ib(), attr.ib(type=int)

@attr.s
class Sub(Base):
    z = attr.ib()

reveal_type(Sub.__init__)  # E: revealed type: (self: Sub, x: Any, y: int, z: Any) -> None
"#,
);

// A string `type=` forward reference resolves for every chained name, just like a lone specifier.
attrs_testcase!(
    test_attrs_chained_specifier_type_forward_ref,
    r#"
import attr
from typing import reveal_type

@attr.s
class C:
    p = q = attr.ib(type='D')

class D:
    pass

reveal_type(C.__init__)  # E: revealed type: (self: C, p: D, q: D) -> None
"#,
);

// One specifier chained to several names declares a field per name, matching runtime attrs.
attrs_testcase!(
    test_attrs_chained_specifier_makes_one_field_per_name,
    r#"
from typing import reveal_type
import attr

@attr.s
class B:
    p = q = attr.ib()

reveal_type(B.__init__)  # E: revealed type: (self: B, p: Any, q: Any) -> None
"#,
);

// A plain (non-attrs) tuple assignment is unaffected: both names are ordinary class variables.
attrs_testcase!(
    test_attrs_non_specifier_unpacking_unaffected,
    r#"
from typing import assert_type
import attr

@attr.s
class A:
    a, b = 1, 2
    x = attr.ib()

assert_type(A.a, int)
assert_type(A.b, int)
"#,
);

// A list-literal target unpacks the same way a tuple target does.
attrs_testcase!(
    test_attrs_list_target_unpacked_specifiers,
    r#"
import attr

@attr.s
class A:
    [x, y] = attr.ib(), attr.ib()

A(1, 2)  # OK
A(1)     # E: Missing argument `y`
"#,
);

// The right-hand side may itself be a list literal of specifiers.
attrs_testcase!(
    test_attrs_unpacked_specifiers_list_rhs,
    r#"
import attr

@attr.s
class A:
    x, y = [attr.ib(), attr.ib()]

A(1, 2)     # OK
A(1, 2, 3)  # E: Expected 2 positional arguments
"#,
);

// Next-gen `field()` works in an unpack too, including a per-element `default=`.
attrs_testcase!(
    test_attrs_unpacked_field_specifiers_with_default,
    r#"
from attrs import define, field

@define
class A:
    x, y = field(), field(default=0)

A(1)     # OK: y has a default
A(1, 2)  # OK
"#,
);

// Python evaluates the whole RHS tuple before binding any target name, so a later RHS element
// that mentions an earlier target sees the *pre-assignment* binding (the module-level `x`), not
// the field created on this same line.
attrs_testcase!(
    test_attrs_unpacked_specifier_rhs_evaluation_order,
    r#"
from typing import reveal_type
import attr

x = "outer"

@attr.s
class A:
    x, y = attr.ib(), x

reveal_type(A.y)  # E: revealed type: str
"#,
);

// The same ordering applies inside a specifier's arguments: `default=x` refers to the outer `x`,
// not the field being created on this line.
attrs_testcase!(
    test_attrs_unpacked_specifier_arg_evaluation_order,
    r#"
from typing import reveal_type
import attr

x = "outer"

@attr.s
class A:
    x, y = attr.ib(), attr.ib(default=x)

reveal_type(A.__init__)  # E: revealed type: (self: A, x: Any, y: str = ...) -> None
"#,
);

// The next-gen `field()` specifier chains the same way.
attrs_testcase!(
    test_attrs_chained_field_specifier_makes_one_field_per_name,
    r#"
from attrs import define, field

@define
class B:
    p = q = field()

B(1, 2)  # OK
B(1)     # E: Missing argument `q`
"#,
);

// A chained specifier's arguments apply to every name, so `default=` makes each field optional.
attrs_testcase!(
    test_attrs_chained_specifier_with_default,
    r#"
import attr

@attr.s
class A:
    p = q = attr.ib(default=0)

A()      # OK: both have a default
A(1, 2)  # OK
"#,
);

// Chained-specifier fields in a base class are inherited like any other attrs fields.
attrs_testcase!(
    test_attrs_chained_specifier_inherited,
    r#"
from typing import reveal_type
import attr

@attr.s
class B:
    p = q = attr.ib()

@attr.s
class C(B):
    x = attr.ib()

reveal_type(C.__init__)  # E: revealed type: (self: C, p: Any, q: Any, x: Any) -> None
"#,
);

// In a class that is not attrs-decorated, a chained `attr.ib()` is ordinary Python: no fields.
attrs_testcase!(
    test_attrs_chained_specifier_in_non_attrs_class_ok,
    r#"
import attr

class C:
    p = q = attr.ib()  # OK: C is not an attrs class
"#,
);

// A stdlib `@dataclass` is not attrs, so an unannotated chained `attr.ib()` declares no fields.
attrs_testcase!(
    test_attrs_chained_specifier_in_stdlib_dataclass_ok,
    r#"
from dataclasses import dataclass
import attr

@dataclass
class C:
    p = q = attr.ib()  # OK: C is a stdlib dataclass, not an attrs class
"#,
);

// In a non-attrs dataclass the names are ordinary annotated fields; the chained `attr.ib()` is
// just their default value.
attrs_testcase!(
    test_attrs_chained_specifier_non_attrs_dataclass_fields,
    r#"
from typing import reveal_type
from dataclasses import dataclass
import attr

@dataclass
class C:
    p: int
    q: int
    p = q = attr.ib()

reveal_type(C.__init__)  # E: revealed type: (self: C, p: int = ..., q: int = ...) -> None
"#,
);

// An arity mismatch is not the specifier pattern, so it falls back to ordinary unpacking (which
// reports the size mismatch) rather than the per-name field treatment.
attrs_testcase!(
    test_attrs_unpacked_specifiers_arity_mismatch_falls_back,
    r#"
import attr

@attr.s
class A:
    x, y = attr.ib(), attr.ib(), attr.ib()  # E: Cannot unpack
"#,
);

// A starred RHS element is not the positional specifier pattern, so it falls back to ordinary
// unpacking rather than binding a name to the `*` node.
attrs_testcase!(
    test_attrs_unpacked_specifiers_starred_rhs_falls_back,
    r#"
from typing import reveal_type
import attr

rest = (attr.ib(),)

@attr.s
class A:
    x, y = attr.ib(), *rest

reveal_type(A.__init__)  # E: revealed type: (self: A) -> None
"#,
);
