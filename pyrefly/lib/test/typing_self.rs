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
    def f(self: Self) -> Self:
        assert_type(self, Self)
        return self
    "#,
);

testcase!(
    bug = "The display and solve semantics for `Self` are incorrect",
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
#
# Two issues here:
# - We don't display `Self` in a way that makes good sense for users or debugging,
#   there's no indication of when it is bound versus the raw special form.
# - We aren't treating `Self` the ways we need to theoretically: it should behave
#   like a type variable in calls, and in some slightly more complex way in
#   attribute lookups.
#
assert_type(A().x, A)  # E: assert_type(Self, A)
assert_type(B().x, B)  # E: assert_type(Self, B)
#
# That said, the existing implementation does actually have most of the basic
# behaviors we need, since it "behaves like" the class it is bound to for many
# purposes
assert_type(A().x.y, int)
assert_type(B().x.y, int)
    "#,
);

testcase!(
    bug = "The display and solve semantics for `Self` are incorrect",
    test_class_attr,
    r#"
from typing import ClassVar, Self, assert_type
class A:
    x: ClassVar[Self]
    y: int
class B(A):
    pass
# 
# Similar issues here to the test above; we need Self to be more type-var
# like and play better with inheritance.
assert_type(A.x, A)  # E: assert_type(Self, A)
assert_type(B.x, B)  # E: assert_type(Self, B)
#
# But again, quite a few of the desired behaviors are present.
assert_type(A.x.y, int)
assert_type(B.x.y, int)
    "#,
);
