/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_config::base::UntypedDefBehavior;

use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    test_no_empty_container_inference,
    TestEnv::new_with_infer_with_first_use(false),
    r#"
from typing import assert_type, Any
x = []
x.append(1)
x.append("foo")
assert_type(x, list[Any])
x = {}
x[1] = 2
x["1"] = "2"
assert_type(x, dict[Any, Any])
"#,
);

testcase!(
    test_empty_container_inference_via_generic_function,
    TestEnv::new_with_infer_with_first_use(false),
    r#"
from typing import assert_type

def pair[T](a: T, b: T) -> tuple[T, T]:
    return (a, b)

# The empty list [] comes first, but it gets unified with list[int] from
# the second argument through the generic function's type variable T.
result = pair([], [1, 2, 3])
assert_type(result, tuple[list[int], list[int]])

# Same pattern with dict
dict_result = pair({}, {"a": 1})
assert_type(dict_result, tuple[dict[str, int], dict[str, int]])
"#,
);

testcase!(
    test_implicit_any_no_inference,
    TestEnv::new_with_untyped_def_behavior(UntypedDefBehavior::SkipAndInferReturnAny)
        .enable_unannotated_return_error()
        .enable_implicit_any_parameter_error(),
    r#"
def foo(x, y):  # E: `foo` is missing an annotation for parameter `x` # E: `foo` is missing an annotation for parameter `y` # E: `foo` is missing a return annotation
    return 1
"#,
);

testcase!(
    test_implicit_any_with_inference,
    TestEnv::new_with_untyped_def_behavior(UntypedDefBehavior::CheckAndInferReturnType)
        .enable_unannotated_return_error()
        .enable_implicit_any_parameter_error(),
    r#"
def foo(x, y):  # E: `foo` is missing an annotation for parameter `x` # E: `foo` is missing an annotation for parameter `y` # E: `foo` is missing a return annotation
    return 1
"#,
);

testcase!(
    test_implicit_any_self_cls_ignored,
    TestEnv::new().enable_implicit_any_error(),
    r#"
class C:
    def method(self) -> int:
        return 1

    @classmethod
    def clsmethod(cls) -> int:
        return 1
"#,
);

// https://github.com/facebook/pyrefly/issues/2327
testcase!(
    test_unannotated_parameter_first_param_by_position,
    TestEnv::new().enable_implicit_any_parameter_error(),
    r#"
class A:
    def __new__(cls, a: int) -> "A": ...

class B:
    def __new__(_cls, a: int) -> "B": ...

class C:
    def method(_self, x: int) -> int:
        return x

    @classmethod
    def clsmethod(_cls) -> int:
        return 1

    @classmethod
    def clsmethod2(klass) -> int:
        return 1

    @staticmethod
    def static_method(x) -> int:  # E: `static_method` is missing an annotation for parameter `x`
        return x

    def __init_subclass__(klass, **kwargs: int) -> None: ...

    # vararg as first param is NOT an implicit self param
    def vararg_method(*args, **kwargs) -> None: ...  # E: `vararg_method` is missing an annotation for parameter `args` # E: `vararg_method` is missing an annotation for parameter `kwargs`

    # keyword-only params after variadic first param should also error
    def vararg_method2(*args, name) -> None: ...  # E: `vararg_method2` is missing an annotation for parameter `args` # E: `vararg_method2` is missing an annotation for parameter `name`

# self/cls in standalone functions should still error
def f(a: str, self, cls, b: int) -> None: ...  # E: `f` is missing an annotation for parameter `self` # E: `f` is missing an annotation for parameter `cls`

# self/cls as first param of standalone function should error
def g(self) -> None: ...  # E: `g` is missing an annotation for parameter `self`
def h(cls) -> None: ...  # E: `h` is missing an annotation for parameter `cls`
"#,
);

// https://github.com/facebook/pyrefly/issues/3542
testcase!(
    test_underscore_params_suppress_implicit_any,
    TestEnv::new().enable_implicit_any_parameter_error(),
    r#"
# Single `_` and `_`-prefixed params (single underscore, not ending with `_`)
# should not trigger implicit-any-parameter error
def foo(_) -> int:  # ok - intentionally unused
    return 1

def bar(_x) -> int:  # ok
    return 1

# Params starting with `__` or ending with `_` should still error
def baz(__x) -> int:  # E: `baz` is missing an annotation for parameter `__x`
    return 1

def qux(x_) -> int:  # E: `qux` is missing an annotation for parameter `x_`
    return 1

def quux(__x__) -> int:  # E: `quux` is missing an annotation for parameter `__x__`
    return 1

def corge(_x_) -> int:  # E: `corge` is missing an annotation for parameter `_x_`
    return 1

# Regular params without annotation should still error
def grault(x) -> int:  # E: `grault` is missing an annotation for parameter `x`
    return 1
"#,
);

testcase!(
    test_implicit_any_with_complete_annotations,
    TestEnv::new().enable_implicit_any_error(),
    r#"
def foo(x: int) -> int:
    return x
"#,
);

testcase!(
    test_implicit_any_empty_containers,
    TestEnv::new_with_infer_with_first_use(false).enable_implicit_any_error(),
    r#"
from typing import Iterable, Mapping
x1 = [] # E: Cannot infer type of empty container
x2 = {} # E: Cannot infer type of empty container
x3: Iterable[int] = {} # ok
x4: Mapping[str, str] = {} # ok
def f(_symbols=[]): # E: Cannot infer type of empty container
    pass
"#,
);

testcase!(
    test_implicit_any_empty_containers_with_partial_inference,
    TestEnv::new().enable_implicit_any_error(),
    r#"
from typing import Iterable, Mapping
x1 = [] # E: Cannot infer type of empty container
x2 = {} # E: Cannot infer type of empty container
x3: Iterable[int] = {} # ok
x4: Mapping[str, str] = {} # ok
def f(_symbols=[]): # E: Cannot infer type of empty container
    pass
"#,
);

testcase!(
    test_implicit_any_default_disabled,
    r#"
from typing import Iterable
def foo(x):
    return x

x1 = []
x2 = {}
"#,
);

testcase!(
    test_explicit_any_default_disabled,
    r#"
from typing import Any

def f(value: Any) -> Any:
    print(value.asdfsdf)
    return value
"#,
);

testcase!(
    test_explicit_any_error,
    TestEnv::new().enable_explicit_any_error(),
    r#"
from typing import Any, TypeAlias

def f(value: Any) -> Any:  # E: Explicit `Any` is not allowed # E: Explicit `Any` is not allowed
    print(value.asdfsdf)
    return value

xs: list[Any] = []  # E: Explicit `Any` is not allowed
Alias: TypeAlias = dict[str, Any]  # E: Explicit `Any` is not allowed
    "#,
);

testcase!(
    test_explicit_any_in_tparam_default_does_not_leak_to_specialization,
    TestEnv::new().enable_explicit_any_error(),
    r#"
from typing import Any

class C[T = Any]:  # E: Explicit `Any` is not allowed
    pass

x: C[int]
"#,
);

// Each annotation below denotes a type that an `Any` is reachable from, either by expanding an
// alias body or by reading a type parameter's bound or default. The `Any` is reported once, at
// the declaration that writes it, and not again at each use.
testcase!(
    test_explicit_any_reported_at_declaration_not_at_uses,
    TestEnv::new().enable_explicit_any_error(),
    r#"
from typing import Any, Callable, TypeVar

class DefaultedTParam[T = Any]:  # E: Explicit `Any` is not allowed
    pass
type AliasIsAny = Any  # E: Explicit `Any` is not allowed
type AliasHasAny = list[Any]  # E: Explicit `Any` is not allowed
type AliasCallable = Callable[[int], Any]  # E: Explicit `Any` is not allowed
BoundAny = TypeVar("BoundAny", bound=Any)  # E: Explicit `Any` is not allowed
DefaultAny = TypeVar("DefaultAny", default=Any)  # E: Explicit `Any` is not allowed

def by_tparam_default(x: DefaultedTParam) -> None: ...
def by_alias_to_any(x: AliasIsAny) -> None: ...
def by_alias_containing_any(x: AliasHasAny) -> None: ...
def by_alias_callable(x: AliasCallable) -> None: ...
def by_typevar_bound(x: BoundAny) -> None: ...
def by_typevar_default(x: DefaultAny) -> None: ...
"#,
);

// A member's `Any` is not reachable from the type of an annotation naming the class, so these
// use sites are silent for a different reason than the ones above: nothing reaches the `Any`.
testcase!(
    test_explicit_any_in_a_member_is_not_reported_at_uses_of_the_class,
    TestEnv::new().enable_explicit_any_error(),
    r#"
from typing import Any, NamedTuple, Protocol, TypedDict

class TD(TypedDict):
    f: Any  # E: Explicit `Any` is not allowed

class NT(NamedTuple):
    f: Any  # E: Explicit `Any` is not allowed

class Proto(Protocol):
    def m(self) -> Any: ...  # E: Explicit `Any` is not allowed

def by_typed_dict_field(x: TD) -> None: ...
def by_named_tuple_field(x: NT) -> None: ...
def by_protocol_member(x: Proto) -> None: ...
"#,
);

// Writing `Any` is what gets reported, wherever it is written: nested inside a subscript, as an
// explicit type argument to a parameter that also defaults to `Any`, and once per occurrence.
testcase!(
    test_explicit_any_reported_for_each_written_occurrence,
    TestEnv::new().enable_explicit_any_error(),
    r#"
from typing import Any, Callable

class C[T = Any]:  # E: Explicit `Any` is not allowed
    pass

def bare(x: Any) -> None: ...  # E: Explicit `Any` is not allowed
def nested(x: list[Any]) -> None: ...  # E: Explicit `Any` is not allowed
def deeply_nested(x: dict[str, list[Any]]) -> None: ...  # E: Explicit `Any` is not allowed
def specialized(x: C[Any]) -> None: ...  # E: Explicit `Any` is not allowed
def twice(x: Callable[[Any], Any]) -> None: ...  # E: Explicit `Any` is not allowed # E: Explicit `Any` is not allowed
"#,
);

testcase!(
    test_warn_on_implicit_any_in_attribute,
    TestEnv::new().enable_implicit_any_attribute_error(),
    r#"
from typing import Any
class A:
    def __init__(self):
        self.x = None  # E: implicitly inferred to be `Any | None`
    "#,
);

testcase!(
    test_implicit_any_attribute_slots_excluded,
    TestEnv::new().enable_implicit_any_attribute_error(),
    r#"
class A:
    __slots__ = ()
    __match_args__ = ()
    x = ()  # E: implicitly inferred to be `tuple[Any, ...]`
    "#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/1774
testcase!(
    test_nested_list_set_item,
    r#"
from typing import Any
rows: list[str] = []
x = []
for i, row in enumerate(rows):
    x.append([])
    for j, item in enumerate(row):
        x[-1].append(item)

entries: list[Any] = []
for i, j in entries:
    x[i][j] = "x"
"#,
);

testcase!(
    test_unknown_variable_type,
    TestEnv::new().enable_unknown_variable_type_error(),
    r#"
def untyped(x):
    return x

y = untyped(1)  # E: The type of `y` is unknown
"#,
);

testcase!(
    test_unknown_variable_type_annotated_no_error,
    TestEnv::new().enable_unknown_variable_type_error(),
    r#"
def untyped(x):
    return x

y: int = untyped(1)
"#,
);

testcase!(
    test_unknown_variable_type_known_no_error,
    TestEnv::new().enable_unknown_variable_type_error(),
    r#"
y = 1
s = "hello"
"#,
);

testcase!(
    test_unknown_variable_type_not_suppressed_by_implicit_any,
    TestEnv::new().enable_unknown_variable_type_error(),
    r#"
def untyped(x):
    return x

# pyrefly: ignore[implicit-any]
y = untyped(1)  # E: The type of `y` is unknown
"#,
);

testcase!(
    test_unknown_variable_type_explicit_any_no_error,
    TestEnv::new().enable_unknown_variable_type_error(),
    r#"
from typing import Any, cast
# An explicit `Any` is intentional, so only implicit `Any` should trigger the rule.
v = cast(Any, 1)
"#,
);

testcase!(
    test_unknown_variable_type_class_body_no_error,
    TestEnv::new().enable_unknown_variable_type_error(),
    r#"
def untyped(x):
    return x

# A class-body assignment defines a class attribute, which is reported by
# implicit-any-attribute, not unknown-variable-type.
class C:
    x = untyped(1)
"#,
);

testcase!(
    test_unknown_variable_type_other_targets,
    TestEnv::new().enable_unknown_variable_type_error(),
    r#"
def untyped(x):
    return x

def f() -> None:
    a = untyped(1)  # E: The type of `a` is unknown
    for b in untyped(1):  # E: The type of `b` is unknown
        pass
    c, d = untyped(1)  # E: The type of `c` is unknown  # E: The type of `d` is unknown
    with untyped(1) as e:  # E: The type of `e` is unknown
        pass
    [h for h in untyped(1)]  # E: The type of `h` is unknown
"#,
);

// Only variables are reported, not the intermediate bindings for tuple patterns, subscript and
// attribute targets, or `with` items without `as`.
testcase!(
    test_unknown_variable_type_only_names,
    TestEnv::new().enable_unknown_variable_type_error(),
    r#"
def untyped(x):
    return x

def f(xs: list[int], o: object) -> None:
    for k, v in untyped(1):  # E: The type of `k` is unknown  # E: The type of `v` is unknown
        pass
    (g, h), i = untyped(1)  # E: The type of `g` is unknown  # E: The type of `h` is unknown  # E: The type of `i` is unknown
    xs[0], j = untyped(1)  # E: The type of `j` is unknown
    with untyped(1):
        pass
    [m for m, n in untyped(1)]  # E: The type of `m` is unknown  # E: The type of `n` is unknown
    [p for p in xs]
"#,
);

testcase!(
    test_unknown_variable_type_other_targets_annotated_no_error,
    TestEnv::new().enable_unknown_variable_type_error(),
    r#"
def untyped(x):
    return x

def f() -> None:
    b: int
    for b in untyped(1):
        pass
    e: int
    with untyped(1) as e:
        pass
"#,
);

testcase!(
    test_unknown_variable_type_other_targets_known_no_error,
    TestEnv::new().enable_unknown_variable_type_error(),
    r#"
def f(xs: list[int], ys: list[tuple[int, str]]) -> None:
    for a in xs:
        pass
    for b, c in ys:
        pass
    d, e = 1, "s"
"#,
);

testcase!(
    test_unknown_return_type,
    TestEnv::new().enable_unknown_return_type_error(),
    r#"
def call(cb):
    return cb()  # E: The type of the returned value is unknown

def read(obj, n: int):
    if n:
        return obj.value  # E: The type of the returned value is unknown
    return n

def known(n: int):
    return n + 1

def bare(cb):
    cb()
    return

# Annotated functions are covered by `no-any-return-implicit` instead.
def annotated(cb) -> int:
    return cb()
"#,
);

testcase!(
    test_unknown_return_type_lambda,
    TestEnv::new().enable_unknown_return_type_error(),
    r#"
from typing import Callable

def mk(cb, obj) -> None:
    f = lambda: cb()  # E: The return type of this lambda is unknown
    g = lambda: obj.value  # E: The return type of this lambda is unknown
    h: Callable[[], int] = lambda: cb()  # E: The return type of this lambda is unknown
    k = lambda: 1
"#,
);

testcase!(
    test_unknown_from_error,
    TestEnv::new().enable_unknown_from_error_error(),
    r#"
import missing_module  # pyrefly: ignore[missing-import]

class A:
    foo: NotDefined  # E: Could not find name `NotDefined`

def make() -> NotDefined: ...  # E: Could not find name `NotDefined`
def make_from(v: int) -> NotDefined: ...  # E: Could not find name `NotDefined`

def takes(x: int) -> None: ...

a = A()
x = make()  # E: The type of `x` is unknown because it comes from an error
takes(make())  # E: The type of this argument is unknown because it comes from an error

def get():
    return make()  # E: The type of the returned value is unknown because it comes from an error

f = lambda: make()  # E: The return type of this lambda is unknown because it comes from an error

for item in make():  # E: The type of `item` is unknown because it comes from an error
    pass
c = [i for i in make()]  # E: The type of `i` is unknown because it comes from an error
with make() as cm:  # E: The type of `cm` is unknown because it comes from an error
    pass
p1, p2 = make()  # E: The type of `p1` is unknown because it comes from an error  # E: The type of `p2` is unknown because it comes from an error
(r1, r2), r3 = make()  # E: The type of `r1` is unknown because it comes from an error  # E: The type of `r2` is unknown because it comes from an error  # E: The type of `r3` is unknown because it comes from an error

class B:
    attr = make()  # E: The type of this attribute is unknown because it comes from an error
    for k in make():  # E: The type of this attribute is unknown because it comes from an error
        pass
    def __init__(self) -> None:
        self.inst = make()  # E: The type of this attribute is unknown because it comes from an error

# A suppressed root error still makes its uses visible.
takes(missing_module.value)  # E: The type of attribute `value` is unknown because it comes from an error

# Only the innermost site of a nested unknown value is reported.
takes(a.foo)  # E: The type of attribute `foo` is unknown because it comes from an error
y = a.foo  # E: The type of attribute `foo` is unknown because it comes from an error

# Accesses on a value that is already unknown only propagate it.
z = a.foo.bar.baz  # E: The type of attribute `foo` is unknown because it comes from an error

# The error that causes the unknown type is reported instead.
takes(undefined_name)  # E: Could not find name `undefined_name`
takes(a.missing)  # E: Object of class `A` has no attribute `missing`
w = a.missing.bar  # E: Object of class `A` has no attribute `missing`
for item2 in undefined_name:  # E: Could not find name `undefined_name`
    pass
for u1, u2 in undefined_name:  # E: Could not find name `undefined_name`
    pass
c2 = [i2 for i2 in undefined_name]  # E: Could not find name `undefined_name`
with undefined_name as cm2:  # E: Could not find name `undefined_name`
    pass
q1, q2 = undefined_name  # E: Could not find name `undefined_name`
(n1, n2), n3 = undefined_name  # E: Could not find name `undefined_name`
c3 = [x3 for x3, y3 in undefined_name]  # E: Could not find name `undefined_name`
m4 = (p4, q4) = undefined_name  # E: Could not find name `undefined_name`
match undefined_name:  # E: Could not find name `undefined_name`
    case [first, *rest]:
        pass
class C:
    for k2 in undefined_name:  # E: Could not find name `undefined_name`
        pass
def add():
    return 1 + "a"  # E: `+` is not supported between `Literal[1]` and `Literal['a']`

# An error whose kind is disabled, here `unknown-argument-type` for `p`, does not hide the report.
def g(p):
    v = make_from(p)  # E: The type of `v` is unknown because it comes from an error
"#,
);

testcase!(
    bug = "An unrelated error within the range hides the unknown-from-error report",
    test_unknown_from_error_hidden_by_unrelated_error,
    TestEnv::new().enable_unknown_from_error_error(),
    r#"
def make_from(v: int) -> NotDefined: ...  # E: Could not find name `NotDefined`

def make() -> NotDefined: ...  # E: Could not find name `NotDefined`

# `y` is unknown because of `make_from`'s return annotation, not because of the bad argument.
y = make_from("s")  # E: Argument `Literal['s']` is not assignable to parameter `v` with type `int`

# A nested report with a different cause, here the argument's, also hides the report on `z`.
z = make_from(make())  # E: The type of this argument is unknown because it comes from an error
"#,
);

testcase!(
    test_unknown_from_error_not_hidden_by_unknown_kinds,
    TestEnv::new()
        .enable_unknown_from_error_error()
        .enable_unknown_argument_type_error(),
    r#"
from typing_extensions import deprecated

def make_from(v: int) -> NotDefined: ...  # E: Could not find name `NotDefined`

@deprecated("old")
def old() -> int: ...

# Soft errors, like the unknown argument or the deprecated call, don't cause the error `Any`, so
# they don't hide the report.
def g(p):
    v = make_from(p)  # E: The type of `v` is unknown because it comes from an error  # E: The type of this argument is unknown
w = make_from(old())  # E: The type of `w` is unknown because it comes from an error  # E: `old` is deprecated
"#,
);

// `explicit-any` and `no-any-return-explicit` are enabled to show which kind reports each of these
// places, and that the three do not overlap on any of them.
testcase!(
    test_explicit_any_use,
    TestEnv::new()
        .enable_explicit_any_use_error()
        .enable_explicit_any_error()
        .enable_no_any_return_explicit_error(),
    r#"
import json
from typing import Any, cast

def takes(x: int) -> None: ...

def f(a: Any) -> None:  # E: Explicit `Any` is not allowed
    b = a.attr  # E: The type of `b` is an explicit `Any`  # E: The type of attribute `attr` is an explicit `Any`
    chain = a.one.two  # E: The type of `chain` is an explicit `Any`  # E: The type of attribute `one` is an explicit `Any`  # E: The type of attribute `two` is an explicit `Any`
    takes(a)  # E: The type of this argument is an explicit `Any`
    c = json.loads("1")  # E: The type of `c` is an explicit `Any`
    for item in a:  # E: The type of `item` is an explicit `Any`
        pass
    p, q = a  # E: The type of `p` is an explicit `Any`  # E: The type of `q` is an explicit `Any`
    with a as w:  # E: The type of `w` is an explicit `Any`
        pass
    l = [i for i in a]  # E: The type of `i` is an explicit `Any`
    g = lambda: a  # E: The return type of this lambda is an explicit `Any`
    y = cast(Any, 1)  # E: Explicit `Any` is not allowed  # E: The type of `y` is an explicit `Any`
    lt = a < 1  # E: The type of `lt` is an explicit `Any`
    gt = 1 < a  # E: The type of `gt` is an explicit `Any`
    takes(-a)  # E: The type of this argument is an explicit `Any`
    m = 1
    m += a  # the target of an augmented assignment is not itself a reported position
    takes(m)  # E: The type of this argument is an explicit `Any`
    n = a  # E: The type of `n` is an explicit `Any`
    n += 1
    takes(n)  # E: The type of this argument is an explicit `Any`

def ret(a: Any):  # E: Explicit `Any` is not allowed
    return a  # E: The type of the returned value is an explicit `Any`

def annotated(a: Any) -> int:  # E: Explicit `Any` is not allowed
    return a  # E: Returning Any from function declared to return "int"

def get() -> Any: ...  # E: Explicit `Any` is not allowed

class C:
    for k in get():  # E: The type of this attribute is an explicit `Any`
        pass

    def __init__(self) -> None:
        self.inst = json.loads("1")  # E: The type of this attribute is an explicit `Any`
        self.declared: Any = json.loads("1")  # E: Explicit `Any` is not allowed

e: Any = 1  # E: Explicit `Any` is not allowed
"#,
);

testcase!(
    test_explicit_any_use_does_not_hide_unknown_from_error,
    TestEnv::new()
        .enable_explicit_any_use_error()
        .enable_unknown_from_error_error(),
    r#"
from typing import Any

def make_from(v: object) -> NotDefined: ...  # E: Could not find name `NotDefined`

def f(d: Any) -> None:
    x = make_from(d.attr)  # E: The type of `x` is unknown because it comes from an error  # E: The type of this argument is an explicit `Any`  # E: The type of attribute `attr` is an explicit `Any`
"#,
);

// With two `Any` operands the styles are merged, so an error or implicit `Any` is reported as
// itself rather than being blamed on the user's explicit `Any`, and the reported kind does not
// depend on which side the explicit `Any` is on.
testcase!(
    test_any_operand_styles_merge,
    TestEnv::new()
        .enable_explicit_any_use_error()
        .enable_unknown_from_error_error()
        .enable_unknown_variable_type_error(),
    r#"
from typing import Any

def make_from(v: object) -> NotDefined: ...  # E: Could not find name `NotDefined`

def f(a: Any, b) -> None:
    e = make_from(0)  # E: The type of `e` is unknown because it comes from an error
    ae = a < e  # E: The type of `ae` is unknown because it comes from an error
    ea = e < a  # E: The type of `ea` is unknown because it comes from an error
    ab = a < b  # E: The type of `ab` is unknown; it is inferred as an implicit `Any`
    ba = b < a  # E: The type of `ba` is unknown; it is inferred as an implicit `Any`
    aa = a < a  # E: The type of `aa` is an explicit `Any`

def g(a: Any, b) -> None:
    m = a  # E: The type of `m` is an explicit `Any`
    m += b
    n = m  # E: The type of `n` is unknown; it is inferred as an implicit `Any`

def h(a: Any, b) -> None:
    e = make_from(0)  # E: The type of `e` is unknown because it comes from an error
    m = a  # E: The type of `m` is an explicit `Any`
    m += e
    n = m  # E: The type of `n` is unknown because it comes from an error
    p = b  # E: The type of `p` is unknown; it is inferred as an implicit `Any`
    p += e
    q = p  # E: The type of `q` is unknown because it comes from an error
"#,
);

// A union keeps only its strongest `Any` member, including a union built by substituting a type
// variable. Otherwise `Any | T` with `T = Unknown` keeps both, and `in` reports the argument once
// for each style.
testcase!(
    test_union_any_styles_merge,
    TestEnv::new()
        .enable_explicit_any_use_error()
        .enable_unknown_argument_type_error()
        .enable_unknown_variable_type_error(),
    r#"
from typing import Any, TypeVar, assert_type

T = TypeVar("T")

def or_any(default: T) -> Any | T: ...

def f(s: str, b="") -> None:
    x = or_any(b)
    x in ["a"]  # E: The type of this argument is unknown
    y = or_any(s)
    assert_type(y, Any | str)
"#,
);

// An unannotated class-body assignment is contextually typed by the annotation it inherits, both
// where it is bound in the class body and where it becomes a class field. Otherwise the lambda
// parameter `n` is `int` or `Any` in one inference and unknown in the other, and both inferences
// report.
testcase!(
    test_class_body_assignment_uses_inherited_annotation_as_hint,
    TestEnv::new()
        .enable_explicit_any_use_error()
        .enable_implicit_any_lambda_error()
        .enable_unknown_attribute_access_error(),
    r#"
from typing import Any, Callable

class Base:
    f: Callable[[int], str]
    g: Callable[[Any], object]

class C(Base):
    f = lambda n: n.foo  # E: Object of class `int` has no attribute `foo`
    g = lambda n: n.foo  # E: The return type of this lambda is an explicit `Any`  # E: The type of attribute `foo` is an explicit `Any`
"#,
);

// When the inherited annotation carries no type, such as a bare `ClassVar`, its type is an implicit
// `Any`, which hints the value just as the class field does. Otherwise the class-body binding infers
// `[]` with no hint and reports it as an unknown-element container, which the class field does not.
testcase!(
    test_class_body_assignment_hint_from_typeless_inherited_annotation,
    TestEnv::new().enable_implicit_any_empty_container_error(),
    r#"
from typing import ClassVar

class Base:
    h: ClassVar = {}  # E: Cannot infer type of empty container

class C(Base):
    h = {}
"#,
);

// A value whose type contains an implicit `Any` without being one, such as `list[Unknown]`, is
// partially unknown. The exact-`Any` check reports a fully unknown value instead, so the two are
// disjoint: `v` is partially unknown and `w` is fully unknown.
testcase!(
    test_partially_unknown_variable_type,
    TestEnv::new()
        .enable_partially_unknown_type_error()
        .enable_unknown_variable_type_error(),
    r#"
def untyped(p):
    return p

v = [untyped(1)]  # E: The type of `v` is partially unknown; it contains an implicit `Any`
w = untyped(1)  # E: The type of `w` is unknown; it is inferred as an implicit `Any`
"#,
);

// Module globals whose type pyrefly does not model are an implicit `Any`: there is no annotation
// in the user's code that they could change, so blaming an explicit `Any` would be wrong.
testcase!(
    test_unmodelled_module_globals_are_implicit_any,
    TestEnv::new()
        .enable_explicit_any_use_error()
        .enable_unknown_variable_type_error(),
    r#"
s = __spec__  # E: The type of `s` is unknown; it is inferred as an implicit `Any`
l = __loader__  # E: The type of `l` is unknown; it is inferred as an implicit `Any`
b = __builtins__  # E: The type of `b` is unknown; it is inferred as an implicit `Any`
"#,
);

// Each element of a fixed-length splat is matched to its own parameter but reported at the range
// of the whole argument, so the argument gets one report with the element styles merged. Without
// the merge, elements with different styles each report, putting two kinds at the same range.
testcase!(
    test_splat_argument_any_styles_merge,
    TestEnv::new()
        .enable_explicit_any_use_error()
        .enable_unknown_from_error_error()
        .enable_unknown_argument_type_error(),
    r#"
from typing import Any

def make_from(v: object) -> NotDefined: ...  # E: Could not find name `NotDefined`

def two(a: int, b: int) -> None: ...

def f(a: Any, b) -> None:
    e = make_from(0)  # E: The type of `e` is unknown because it comes from an error
    two(*[a, e])  # E: The type of this argument is unknown because it comes from an error
    two(*[a, b])  # E: The type of this argument is unknown
    two(*[a, a])  # E: The type of this argument is an explicit `Any`
    two(*[a, 1])  # E: The type of this argument is an explicit `Any`
    two(*[1, 2])

def g(t: tuple[Any, int], u: tuple[Any, Any]) -> None:
    two(*t)  # E: The type of this argument is an explicit `Any`
    two(*u)  # E: The type of this argument is an explicit `Any`
"#,
);

// A return is only reported when the declared return type is concrete. A declared `Any` already
// tells the reader the value is unchecked, and `object` forces the caller to narrow before use.
testcase!(
    test_explicit_any_use_return_annotations,
    TestEnv::new()
        .enable_explicit_any_use_error()
        .enable_explicit_any_error()
        .enable_no_any_return_explicit_error(),
    r#"
from typing import Any

def to_any(a: Any) -> Any:  # E: Explicit `Any` is not allowed  # E: Explicit `Any` is not allowed
    return a

def to_object(a: Any) -> object:  # E: Explicit `Any` is not allowed
    return a

def to_concrete(a: Any) -> int:  # E: Explicit `Any` is not allowed
    return a  # E: Returning Any from function declared to return "int"
"#,
);

// Returning an `Any` as `int | Any` leaves the `int` member unchecked. `no-any-return` skips this
// declared type, as mypy does, so the returned value is reported as in an unannotated function. A
// union whose `Any` member comes from an unresolved annotation is skipped, like the bare annotation.
testcase!(
    test_any_return_to_union_with_any,
    TestEnv::new()
        .enable_explicit_any_use_error()
        .enable_unknown_return_type_error()
        .enable_unknown_from_error_error()
        .enable_no_any_return_error(),
    r#"
from typing import Any

def make() -> NotDefined: ...  # E: Could not find name `NotDefined`

def explicit(a: Any) -> int | Any:
    return a  # E: The type of the returned value is an explicit `Any`

def implicit(a) -> int | Any:
    return a  # E: The type of the returned value is unknown; it is inferred as an implicit `Any`

def from_error() -> int | Any:
    return make()  # E: The type of the returned value is unknown because it comes from an error

def unresolved(a: Any) -> NotDefined | None:  # E: Could not find name `NotDefined`
    return a
"#,
);

// Every argument shape goes through a different call-site path, so each is reported separately.
testcase!(
    test_explicit_any_use_argument_shapes,
    TestEnv::new().enable_explicit_any_use_error(),
    r#"
from typing import Any

def positional(x: int) -> None: ...
def keyword(*, k: int) -> None: ...
def star(*args: int, **kwargs: int) -> None: ...

def f(a: Any) -> None:
    positional(a)  # E: The type of this argument is an explicit `Any`
    keyword(k=a)  # E: The type of this argument is an explicit `Any`
    star(*a)  # E: The type of this argument is an explicit `Any`
    star(**a)  # E: The type of this argument is an explicit `Any`
"#,
);

// An argument whose type contains an implicit `Any` without being one is partially unknown.
testcase!(
    test_partially_unknown_argument_type,
    TestEnv::new().enable_partially_unknown_type_error(),
    r#"
def untyped(p):
    return p

def takes(x: object) -> None: ...

def f() -> None:
    takes([untyped(1)])  # E: The type of this argument is partially unknown; it contains an implicit `Any`
"#,
);

// A partially unknown value is reported at every attribute position: a class-body attribute value,
// an instance attribute, a class-body loop target, and an attribute access.
testcase!(
    test_partially_unknown_attribute,
    TestEnv::new().enable_partially_unknown_type_error(),
    r#"
def untyped(p):
    return p

class C:
    x = [untyped(1)]  # E: This expression's type is partially unknown; it contains an implicit `Any`. Please provide an explicit type annotation.
    def set(self) -> None:
        self.y = [untyped(1)]  # E: This expression's type is partially unknown; it contains an implicit `Any`. Please provide an explicit type annotation.
    for item in [[untyped(1)]]:  # E: This expression's type is partially unknown; it contains an implicit `Any`. Please provide an explicit type annotation.
        pass

def access(c: C) -> None:
    c.x  # E: The type of attribute `x` is partially unknown; it contains an implicit `Any`
"#,
);

// A partially unknown value is reported when returned from an unannotated function and from a
// lambda body. The lambda's target is annotated so only the lambda return, not the variable, fires.
testcase!(
    test_partially_unknown_return_type,
    TestEnv::new().enable_partially_unknown_type_error(),
    r#"
from typing import Callable

def untyped(p):
    return p

def g():
    return [untyped(1)]  # E: The type of the returned value is partially unknown; it contains an implicit `Any`

f: Callable[[], object] = lambda: [untyped(1)]  # E: The return type of this lambda is partially unknown; it contains an implicit `Any`
"#,
);
