/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_use_self,
    r#"
from typing import assert_type
from typing import Self
import typing
from typing import Self as Myself

class C:
    def m(self, x: Self, y: Myself) -> list[typing.Self]:
        return [self, x, y]

assert_type(C().m(C(), C()), list[C])
"#,
);

testcase!(
    test_typing_self_return,
    r#"
from typing import Self, assert_type
class A:
    def f(self) -> Self:
        return self
    @classmethod
    def g(cls) -> type[Self]:
        return cls
class B(A):
    pass
assert_type(B().f(), B)
assert_type(B().g(), type[B])
    "#,
);

testcase!(
    test_typing_self_param,
    r#"
from typing import Self
class A:
    def f(self, x: Self):
        pass
class B(A):
    pass
def f(a: A, b: B):
    b.f(b)  # OK
    b.f(a)  # E:
    "#,
);

testcase!(
    test_typing_self_new_param,
    r#"
from typing import Self
class A:
    def __new__(cls, x: type[Self]):
        return super().__new__(cls)
class B(A):
    pass
B(B)  # OK
B(A)  # E:
    "#,
);

testcase!(
    test_assert_type,
    r#"
from typing import Self, assert_type
class A:
    def __new__(cls, *args, **kwargs):
        assert_type(cls, type[Self])
        super().__new__(cls, *args, **kwargs)

    def __init_subclass__(cls, **kwargs):
        assert_type(cls, type[Self])
        super().__init_subclass__(**kwargs)

    @classmethod
    def f1(cls):
        assert_type(cls, type[Self])

    def f2(self):
        assert_type(self, Self)
    
    def f3(self: Self) -> Self:
        assert_type(self, Self)
        return self
    "#,
);

testcase!(
    test_instance_attr,
    r#"
from typing import Self, assert_type
class A:
    x: Self
    y: int
    def f(self):
        assert_type(self.x, Self)
        assert_type(self.x.y, int)
class B(A):
    pass

assert_type(A().x, A)
assert_type(B().x, B)
    "#,
);

testcase!(
    test_class_attr,
    r#"
from typing import ClassVar, Self, assert_type
class A:
    x: ClassVar[Self]
    y: int
class B(A):
    pass

assert_type(A.x, A)
assert_type(B.x, B)
    "#,
);

testcase!(
    test_cast_self,
    r#"
from typing import cast, Self
class Foo:
    def hello(self): pass
    def check(self, other):
        other2 = cast(Self, other)
        other2.hello()
    "#,
);

testcase!(
    test_inherit_overloaded_dunder_new_with_self,
    r#"
from typing import Self, overload

class A:
    @overload
    def __new__(cls, x: int) -> Self: ...
    @overload
    def __new__(cls, x: str) -> Self: ...
    def __new__(cls, x: int | str) -> Self:
        return super().__new__(cls)

class B(A):
    pass

x: B = B(1)
"#,
);

testcase!(
    test_literal_self,
    r#"
from typing import Self, Literal, assert_type
import enum

class E(enum.Enum):
    A = 1
    B = 2

    def m(self, other: Self) -> Self:
        return other

a: Literal[E.A] = E.A

assert_type(a.m(E.B), E)
assert_type(E.A.m(E.B), E)
"#,
);

testcase!(
    test_callable_self,
    r#"
from typing import Self, assert_type

class C:
    def __call__(self) -> Self:
        return self

    def m(self):
        assert_type(self(), Self)
"#,
);
