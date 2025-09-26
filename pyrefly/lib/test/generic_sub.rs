/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_generic_sub_simple,
    r#"
from typing import Protocol

class P(Protocol):
    def m[T](self, x: T) -> T: ...

class C:
    def m[T](self, x: T) -> T: ...

x: P = C()
    "#,
);

testcase!(
    test_generic_sub_multiple_tparams,
    r#"
from typing import Protocol

class P(Protocol):
    def m[T](self, x: T, y: T) -> T: ...

class C:
    def m[A, B](self, x: A, y: B) -> B: ...

x: P = C() # OK, C.m is more general than P.m
"#,
);

testcase!(
    test_generic_sub_bounds,
    r#"
from typing import Protocol

class A: ...
class B(A): ...
class C(B): ...

class P(Protocol):
    def m[T: B](self, x: T) -> T: ...

class C1:
    def m[T: A](self, x: T) -> T: ...

x1: P = C1() # OK, [T: A] accepts all types that [T: B] does

class C2:
    def m[T: C](self, x: T) -> T: ...

x2: P = C2() # E: `C2` is not assignable to `P`
"#,
);

testcase!(
    test_non_generic_sub_generic,
    r#"
from typing import Protocol, Any, Never

class P(Protocol):
    def m[T](self, x: T) -> T: ...

class C1:
    def m(self, x: Any) -> Any: ...

x1: P = C1() # OK

class C2:
    def m(self, x: object) -> Never: ...

x2: P = C2() # OK

class C3:
    def m(self, x: int) -> int: ...

x3: P = C3() # E: `C3` is not assignable to `P`
"#,
);

testcase!(
    test_generic_sub_non_generic,
    r#"
from typing import Protocol

class P(Protocol):
    def m(self, x: int) -> int: ...

class C:
    def m[T](self, x: T) -> T: ...

x: P = C() # OK
"#,
);

testcase!(
    test_generic_sub_tricky,
    r#"
from typing import Protocol

class P(Protocol):
    def m[T](self, x: list[T]) -> list[T]: ...

class C:
    def m[T](self, x: T) -> T: ...

x: P = C()
"#,
);
