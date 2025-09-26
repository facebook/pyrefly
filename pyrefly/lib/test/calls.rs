/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_generic_call_happy_case,
    r#"
from typing import Never
def force_error(x: Never) -> None: ...
def f[S, T](x: S, y: T) -> tuple[S, T]: ...
force_error(f(1, "foo"))  # E: Argument `tuple[int, str]` is not assignable to parameter `x`
"#,
);

testcase!(
    test_generic_call_fails_to_solve_output_var_simple,
    r#"
from typing import Never
def force_error(x: Never) -> None: ...
def f[S, T](x: S) -> tuple[S, T]: ...
force_error(f(1))  # E: Argument `tuple[int, @_]` is not assignable to parameter `x`
"#,
);

testcase!(
    test_generic_call_fails_to_solve_output_var_union_case,
    r#"
from typing import Never
def force_error(x: Never) -> None: ...
def f[S, T](x: S, y: list[T] | None) -> tuple[S, T]: ...
force_error(f(1, None))  # E: Argument `tuple[int, @_]` is not assignable to parameter `x`
"#,
);

testcase!(
    test_self_type_subst,
    r#"
from typing import assert_type, Self
class A:
    def __new__(cls) -> Self: ...
class B[T](A): ...
class C[T]: ...
assert_type(A.__new__(A), A)
assert_type(A.__new__(B[int]), A)
assert_type(A.__new__(C[int]), A) # E: Argument `type[C[int]]` is not assignable to parameter `cls` with type `type[A]` in function `A.__new__`

o = A()
assert_type(o.__new__(A), A)
assert_type(o.__new__(B[int]), A)
assert_type(o.__new__(C[int]), A) # E: Argument `type[C[int]]` is not assignable to parameter `cls` with type `type[A]` in function `A.__new__`
    "#,
);

testcase!(
    test_self_type_subst_overloaded_dunder_new,
    r#"
from typing import Self, assert_type, overload
class C:
    @overload
    def __new__(cls, x: int) -> Self: ...
    @overload
    def __new__(cls, x: str) -> Self: ...
    def __new__(cls, x: int | str) -> Self:
        return super().__new__(cls)

assert_type(C.__new__(C, 0), C)
assert_type(C.__new__(C, ""), C)
    "#,
);

testcase!(
    test_self_type_subst_use_receiver,
    r#"
from typing import assert_type, Self
class A[T]:
    def __new__(cls: type[Self], x: T) -> Self: ...
o = A[int].__new__(A[str], "foo") # E: `type[A[str]]` is not assignable to parameter `cls` with type `type[A[int]]` # E: Argument `Literal['foo']` is not assignable to parameter `x` with type `int` in function `A.__new__`
assert_type(o, A[int])
    "#,
);

testcase!(
    test_deprecated_call,
    r#"
from warnings import deprecated
@deprecated("function is deprecated")
def old_function() -> None: ...
old_function()  # E: `old_function` is deprecated
    "#,
);

testcase!(
    test_deprecated_function_reference,
    r#"
from typing import Callable
from warnings import deprecated
@deprecated("function is deprecated")
def old_function() -> None: ...

def take_callable(f: Callable) -> None: ...
take_callable(old_function)  # E: `old_function` is deprecated
    "#,
);

testcase!(
    test_deprecated_method_call,
    r#"
from warnings import deprecated
class C:
    @deprecated("function is deprecated")
    def old_function(self) -> None: ...

c = C()
c.old_function()  # E: `C.old_function` is deprecated
    "#,
);

testcase!(
    test_deprecated_overloaded_call,
    r#"
from typing import overload
from warnings import deprecated

@overload
def f(x: int) -> int: ...
@overload
def f(x: str) -> str: ...
@deprecated("DEPRECATED")
def f(x: int | str) -> int | str:
    return x

f(0)  # E: `f` is deprecated
    "#,
);

testcase!(
    test_deprecated_overloaded_signature,
    r#"
from typing import overload
from warnings import deprecated

@deprecated("DEPRECATED")
@overload
def f(x: int) -> int: ...
@overload
def f(x: str) -> str: ...
def f(x: int | str) -> int | str:
    return x

f(0)  # E: Call to deprecated overload `f`
f("foo") # No error
    "#,
);

testcase!(
    test_deprecated_overloaded_signature_no_impl,
    r#"
from typing import overload
from warnings import deprecated

@deprecated("DEPRECATED")
@overload
def f(x: int) -> int: ...  # E: Overloaded function must have an implementation
@overload
def f(x: str) -> str: ...

f(0)  # E: Call to deprecated overload `f`
f("foo") # No error
    "#,
);

testcase!(
    test_deprecated_overload_shutil,
    r#"
import shutil
shutil.rmtree("/tmp")
    "#,
);

testcase!(
    test_reduce_call,
    r#"
from functools import reduce
reduce(max, [1,2])
    "#,
);

testcase!(
    test_union_with_type,
    r#"
from typing import assert_type
class A:
    pass
def identity[T](x: T) -> T:
    return x
def f(condition: bool):
    if condition:
        g = type
    else:
        g = identity
    assert_type(g(A()), type[A] | A)
    "#,
);

testcase!(
    test_any_constructor,
    r#"
from typing import Any
Any()  # E: `Any` can not be instantiated
    "#,
);
