/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_one_level_simple,
    r#"
from typing import reveal_type
class Foo:
    x: Foo
class Bar(Foo):
    pass
def f(foo: Foo):
    if isinstance(foo.x, Bar):
        reveal_type(foo)  # E: revealed type: Foo (_.x: Bar)
"#,
);

testcase!(
    test_one_level_and,
    r#"
from typing import reveal_type
class Foo:
    x: Foo
    y: Foo
class Bar(Foo):
    pass
class Baz(Foo):
    pass
def f(foo: Foo):
    if isinstance(foo.x, Bar) and isinstance(foo.y, Baz):
        reveal_type(foo)  # E: revealed type: Foo (_.x: Bar, _.y: Baz)
"#,
);

testcase!(
    test_two_levels,
    r#"
from typing import reveal_type
class Foo:
    x: Foo
    y: Foo
class Bar(Foo):
    pass
def f(foo: Foo):
    if isinstance(foo.x.y, Bar):
        reveal_type(foo)  # E: revealed type: Foo (_.x.y: Bar)
"#,
);

testcase!(
    test_attribute_read_simple,
    r#"
from typing import assert_type
class Foo:
    x: Foo
class Bar(Foo):
    pass
def f(foo: Foo):
    if isinstance(foo.x, Bar):
        assert_type(foo.x, Bar)
    if isinstance(foo.x.x, Bar):
        # TODO(stroxler): At some point we should make `unions` simplify for linearly ordered types.
        assert_type(foo.x, Bar | Foo)
        assert_type(foo.x.x, Bar)
        assert_type(foo.x.x.x, Foo)
"#,
);

testcase!(
    test_attribute_narrow_type_algebra_for_correct_narrows,
    r#"
from typing import assert_type
class Foo:
    x: Foo | None
def f(foo: Foo):
    if foo.x is not None:
        assert_type(foo.x, Foo)
    if foo.x is not None and foo.x.x is not None:
        assert_type(foo.x, Foo)
        assert_type(foo.x.x, Foo)
    if foo.x is not None and foo.x.x is not None and foo.x.x.x is None:
        assert_type(foo.x, Foo)
        assert_type(foo.x.x, Foo)
        assert_type(foo.x.x.x, None)
"#,
);

// The expected behavior when narrowing an invalid attribute chain is to produce
// type errors at the narrow site, but apply the narrowing downstream
// (motivation: being noisy downstream could be quite frustrating for gradually
// typed code)
testcase!(
    test_invalid_narrows_on_bad_attribute_access,
    r#"
from typing import assert_type, Any
class Foo:
    x: Foo | None
def f(foo: Foo):
    if foo.x is not None:
        assert_type(foo.x, Foo)
    if foo.x.x is not None:  # E: Object of class `NoneType` has no attribute `x`
        assert_type(foo.x, Foo | None)
        # Why `Foo | object`? Because the lookup on `None` fails, and we fall back to `object`
        # in that branch of the union.
        assert_type(foo.x.x, Foo | object)
    if isinstance(foo.x.y, Foo) and foo.x.y.x is not None:  # E: Object of class `Foo` has no attribute `y`\nObject of class `NoneType` has no attribute `y`
        assert_type(foo.x, Foo | None)
        assert_type(foo.x.y, Foo)
        assert_type(foo.x.y.x, Foo)
"#,
);

testcase!(
    test_attr_assignment_introduction,
    r#"
from typing import Any, Literal, assert_type
class C:
    x: Any
    y: int | None
    z: C | None
def test_introduce_narrow_with_assignment(c0: C, c1: C):
    assert_type(c0.x, Any)
    assert_type(c0.y, int | None)
    assert_type(c0.z, C | None)
    c0.x = 42
    c0.y = 43
    c0.z = c1
    assert_type(c0.x, Literal[42])
    assert_type(c0.y, Literal[43])
    assert_type(c0.z, C)
"#,
);

testcase!(
    test_attr_assignment_invalidation,
    r#"
from typing import Any, Literal, assert_type
class C:
    x: Any
    z: C | None
def test_invalidate_narrow_with_assignment(c0: C, c1: C):
    assert isinstance(c0.z, C)
    assert isinstance(c0.z.x, int)
    assert isinstance(c0.z.z, C)
    assert isinstance(c0.z.z.x, int)
    assert_type(c0.z, C)
    assert_type(c0.z.x, int)
    assert_type(c0.z.z, C)
    assert_type(c0.z.z.x, int)
    c0.z.z = c1
    assert_type(c0.z, C)
    assert_type(c0.z.x, int)
    assert_type(c0.z.z, C)
    assert_type(c0.z.z.x, Any)
    c0.z = c1
    assert_type(c0.z, C)
    assert_type(c0.z.x, Any)
    assert_type(c0.z.z, C | None)
    print(c0.z.z.x)  # E: Object of class `NoneType` has no attribute `x`
"#,
);

testcase!(
    bug = "TODO(stroxler) We should fine-tune descriptor narrowing more; this is not high-priority",
    test_descriptor_narrowing,
    r#"
from typing import Any, Literal, assert_type
class C:
    @property
    def x(self) -> int | None: ...
    @x.setter
    def x(self, value: int) -> None: ...
def f(c: C):
    assert_type(c.x, int | None)
    # No narrowing occurs on assignment to a descriptor. This is debatable, although
    # in the case where the set type is mismatched vs get it would be absurd to narrow,
    # and it could be very difficult in the presence of overloads.
    c.x = 42
    assert_type(c.x, int | None)
    # Narrowing does occur on conditional checks. Ideally we would produce
    # a strict-mode error on reads, that is not yet implemented.
    assert isinstance(c.x, int)
    assert_type(c.x, int)
"#,
);

// This test is adapted from a Pytorch example.
testcase!(
    test_isinstance_getitem,
    r#"
from typing import Any, Optional, assert_type

Arg = Optional[tuple["Arg", ...]]

class DataType: pass

class N:
    _args: tuple["Arg", ...]
    type: Optional[DataType]

    @property
    def args(self) -> tuple[Arg, ...]:
        return self._args


def f(n: N):
    assert isinstance(n.args[0], N)
    t1 = n.args[0].type
    assert_type(t1, DataType | None)
"#,
);

testcase!(
    test_attribute_access_after_narrowing_in_subclass,
    r#"
from typing import assert_type
from unittest import TestCase

class A:
    def __init__(self) -> None:
        self.id = 5

class B:
    def __init__(self, a : A | None) -> None:
        self.a = a

b = B(A())
if b.a is not None:
    x = b.a
    assert_type(x, A)
"#,
);

testcase!(
    test_or_narrowing_one_level,
    r#"
from typing import reveal_type
class Foo:
    x: Foo
class Bar(Foo):
    pass
class Baz(Foo):
    pass
def f(foo: Foo):
    if isinstance(foo.x, Bar) or isinstance(foo.x, Baz):
        reveal_type(foo)  # E: revealed type: Foo (_.x: Bar | Baz)
"#,
);

testcase!(
    test_or_narrowing_with_top_level,
    r#"
from typing import reveal_type
class Foo:
    x: Foo
class Bar(Foo):
    pass
class Baz(Foo):
    pass
def f(foo: object):
    if (
        (isinstance(foo, Bar) and isinstance(foo.x.x, Bar))
        or (isinstance(foo, Bar) and isinstance(foo.x.x, Baz))
    ):
        reveal_type(foo)  # E: revealed type: Bar (_.x.x: Bar | Baz)
"#,
);

testcase!(
    test_join_empty_facets_vs_nonempty,
    r#"
from typing import reveal_type, Literal
class A:
    x: int | None
def test(y: A | None) -> None:
    if y is not None:
        if y.x:
            reveal_type(y)  # E: revealed type: A (_.x: int)
        else:
            reveal_type(y)  # E: revealed type: A (_.x: Literal[0] | None)
    else:
        reveal_type(y)  # E: revealed type: None
    reveal_type(y)  # E: revealed type: A | None
"#,
);

testcase!(
    test_join_empty_facets_in_or,
    r#"
from typing import assert_type
class A:
    kind: str
def test(y: A | None) -> None:
    if not y or y.kind:
        assert_type(y, A | None)
"#,
);

testcase!(
    test_or_joins_subtrees,
    r#"
from typing import reveal_type
class Foo:
    x: Foo
    y: Foo
class Bar(Foo):
    pass
class Baz(Foo):
    pass
def f(foo: Foo):
    if (
        (isinstance(foo.x.x, Bar) and isinstance(foo.x.y, Bar))
        or (isinstance(foo.x.x, Bar) and isinstance(foo.x.y, Baz))
    ):
        reveal_type(foo)  # E: revealed type: Foo (_.x.x: Bar, _.x.y: Bar | Baz)
"#,
);

testcase!(
    test_or_multiple_levels,
    r#"
from typing import reveal_type
class Foo:
    x: Foo
class Bar(Foo):
    pass
class Baz(Foo):
    pass
def f(foo: Foo):
    if (
        (isinstance(foo.x, Bar) and isinstance(foo.x.x, Bar))
        or (isinstance(foo.x, Bar) and isinstance(foo.x.x, Baz))
    ):
        reveal_type(foo)  # E: revealed type: Foo (_.x: Bar, _.x.x: Bar | Baz)
"#,
);

testcase!(
    test_or_join_drops_subtree,
    r#"
from typing import reveal_type
class Foo:
    x: Foo
    y: Foo
class Bar(Foo):
    pass
class Baz(Foo):
    pass
def f(foo: Foo):
    if (
        (isinstance(foo.x, Bar) and isinstance(foo.x.x, Bar))
        or (isinstance(foo.x, Baz) and isinstance(foo.x.y, Bar))
    ):
        reveal_type(foo)  # E: revealed type: Foo (_.x: Bar | Baz)
"#,
);

testcase!(
    test_or_join_drops_root,
    r#"
from typing import reveal_type
class Foo:
    x: Foo
    y: Foo
class Bar(Foo):
    pass
class Baz(Foo):
    pass
def f(foo: Foo):
    if (
        (isinstance(foo.x, Bar) and isinstance(foo.x.x, Bar))
        or isinstance(foo.x.x, Baz)
    ):
        reveal_type(foo)  # E: revealed type: Foo (_.x.x: Bar | Baz)
"#,
);

testcase!(
    test_propagate_through_no_op_control_flow,
    r#"
from typing import assert_type, Callable
class Foo:
    x: Foo
class Bar(Foo):
    pass
def f(foo: Foo, condition: Callable[[], bool]):
    if isinstance(foo.x, Bar):
        assert_type(foo.x, Bar)
        if condition():
            assert_type(foo.x, Bar)
        assert_type(foo.x, Bar)
        while condition():
            assert_type(foo.x, Bar)
        assert_type(foo.x, Bar)
"#,
);

testcase!(
    test_propagate_through_further_narrowing_control_flow,
    r#"
from typing import assert_type, Callable
class Foo:
    x: Foo
class Bar(Foo):
    pass
class Baz(Bar):
    pass
def f(foo: Foo, condition: Callable[[], bool]):
    if isinstance(foo.x, Bar):
        assert_type(foo.x, Bar)
        if isinstance(foo.x, Baz):
            assert_type(foo.x, Baz)
        assert_type(foo.x, Bar)
        while condition():
            assert isinstance(foo.x, Baz)
            assert_type(foo.x, Baz)
        assert_type(foo.x, Bar)
"#,
);

testcase!(
    test_join_on_branching_control_flow,
    r#"
from typing import assert_type, Callable
class Foo:
    x: Foo
class Bar(Foo):
    pass
class Baz(Foo):
    pass
def f(foo: Foo):
    if isinstance(foo.x, Bar):
        assert_type(foo.x, Bar)
    else:
        assert isinstance(foo.x, Baz)
        assert_type(foo.x, Baz)
    assert_type(foo.x, Bar | Baz)
"#,
);

testcase!(
    test_assignment_to_names,
    r#"
from typing import assert_type, Callable
class Foo:
    x: Foo
class Bar(Foo):
    pass
def f(foo: Foo):
    if isinstance(foo.x, Bar) and isinstance(foo.x.x, Bar):
        assert_type(foo.x, Bar)
        assert_type(foo.x.x, Bar)
        # Narrows on attributes do not propagate across assignment. This is
        # intended - without mutation restrictions it is unsound and could lead
        # to difficult bugs.
        baz = foo
        assert_type(baz.x, Foo)
        # Types of narrowed attributes themselves do propagate, but chained
        # attribute narrows do not.
        qux = foo.x
        assert_type(qux, Bar)
        assert_type(qux.x, Foo)
        # The behavior is the same for other styles of assignment (we use different
        # binding kinds for "target" assignment vs bare name assignment, this test
        # is verifying they work the same way)
        baz, qux = foo, foo.x
        assert_type(baz.x, Foo)
        assert_type(qux, Bar)
        assert_type(qux.x, Foo)
"#,
);

testcase!(
    test_none_to_iterable,
    r#"
from typing import assert_type, Callable
class Foo:
    xs: list[int] | None
def f(foo: Foo):
    if foo.xs is not None:
        assert_type(foo.xs, list[int])
        for x in foo.xs:
            assert_type(x, int)
"#,
);

testcase!(
    test_hasattr_narrowing,
    r#"
from typing import assert_type, Any
class C:
    x: int

def test(c: C):
    if hasattr(c, "x"):
        assert_type(c.x, int)
    if hasattr(c, "y"):
        assert_type(c.y, Any)
    "#,
);

testcase!(
    test_getattr_narrowing,
    r#"
from typing import assert_type, Any
class C:
    x: int | None
class C2:
    pass

def test(c: C, c2: C2):
    # if the class has the attribute declared, we do a truthy refinement if the default is statically falsy
    if getattr(c, "x"):
        assert_type(c.x, int)
    if getattr(c, "x", None):
        assert_type(c.x, int)
    if getattr(c, "x", 1):
        assert_type(c.x, int | None)

    # if not, we infer `Any` if the default is statically falsy
    if getattr(c2, "x"):
        assert_type(c2.x, Any)
    if getattr(c2, "x", None):
        assert_type(c2.x, Any)
    if getattr(c2, "x", 1):
        assert_type(c2.x, Any)  # E: Object of class `C2` has no attribute `x`
    "#,
);

testcase!(
    test_getattr_binop_narrowing,
    r#"
from typing import assert_type, Any
class C:
    pass

def test(c: C):
    if getattr(c, "x") is not None:
        assert_type(c.x, Any)
    if getattr(c, "x") != None:
        assert_type(c.x, Any)
    if getattr(c, "x", None) is not None:
        assert_type(c.x, Any)
    if getattr(c, "x", None) != None:
        assert_type(c.x, Any)
    # these do nothing
    if getattr(c, "x", 1) is not None:
        assert_type(c.x, Any)  # E: Object of class `C` has no attribute `x`
    if getattr(c, "x", 1) != 1:
        assert_type(c.x, Any)  # E: Object of class `C` has no attribute `x`
    "#,
);
