/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::sys_info::PythonVersion;

use crate::test::util::TestEnv;
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
assert_type(A.__new__(B[int]), B[int])
assert_type(A.__new__(C[int]), C[int]) # E: `C[int]` is not assignable to upper bound `A` of type variable `Self@A`

o = A()
assert_type(o.__new__(A), A)
assert_type(o.__new__(B[int]), B[int])
assert_type(o.__new__(C[int]), C[int]) # E: `C[int]` is not assignable to upper bound `A` of type variable `Self@A`
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

fn test_env_3_12() -> TestEnv {
    TestEnv::new_with_version(PythonVersion {
        major: 3,
        minor: 12,
        micro: 0,
    })
}

testcase!(
    test_deprecated_call_3_12,
    test_env_3_12(),
    r#"
from typing_extensions import deprecated
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
    test_nondeprecated_overload_shutil,
    r#"
import shutil
shutil.rmtree("/tmp")
    "#,
);

testcase!(
    test_deprecated_message,
    r#"
from warnings import deprecated
@deprecated("I am a special super-important message about the extended warranty on your car")
def f(): ...

f()  # E: I am a special super-important message about the extended warranty on your car
    "#,
);

testcase!(
    test_deprecated_fqn,
    r#"
import warnings
@warnings.deprecated("Deprecated")
def f(): ...
f()  # E: Deprecated
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

testcase!(
    test_object_new_explicit_call,
    r#"
from typing import assert_type

class A: pass
class B(A): pass

# Direct object.__new__ calls should return the argument class type
x1 = object.__new__(A)
assert_type(x1, A)

x2 = object.__new__(B)
assert_type(x2, B)

# Works with builtin classes too
x3 = object.__new__(int)
assert_type(x3, int)

# Works with `type` annotations too
def f(cls: type[A]):
    x4 = object.__new__(cls)
    assert_type(x4, A)
    "#,
);

testcase!(
    test_object_new_with_generics,
    r#"
from typing import assert_type

class Container[T]: pass

# object.__new__ with generic class should preserve type params
x = object.__new__(Container[int])
assert_type(x, Container[int])
    "#,
);

testcase!(
    test_custom_new_unaffected,
    r#"
from typing import Self, assert_type

class A[T]:
    def __new__(cls: type[Self], x: T) -> Self: ...

# Custom __new__ should continue to work correctly
o = A[int].__new__(A[int], 42)
assert_type(o, A[int])

# Receiver type binding is preserved
class B:
    def __new__(cls) -> Self: ...

b = B.__new__(B)
assert_type(b, B)
    "#,
);

testcase!(
    test_inherit_custom_new,
    r#"
from typing import assert_type, Self
class A:
    def __new__(cls) -> Self:
        return super().__new__(cls)
class B(A):
    pass
assert_type(A().__new__(B), B)
assert_type(A.__new__(B), B)
    "#,
);

testcase!(
    test_inherit_generic_custom_new,
    r#"
from typing import assert_type, Self
class A:
    def __new__[T](cls, x: T, y: T) -> Self:
        return super().__new__(cls)
class B(A):
    pass
assert_type(A.__new__(B, 0, 0), B)
    "#,
);

testcase!(
    test_inherit_overloaded_custom_new,
    r#"
from typing import assert_type, overload, Self
class A:
    @overload
    def __new__(cls) -> Self: ...
    @overload
    def __new__(cls, x) -> Self: ...
    def __new__(cls, x=None) -> Self:
        return super().__new__(cls)
class B(A):
    pass
assert_type(A.__new__(B), B)
assert_type(A.__new__(B, 0), B)
    "#,
);
