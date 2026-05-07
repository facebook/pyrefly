/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_class_init,
    r#"
from typing import assert_type
class Foo:
    def __init__(self, x: int): pass
v = Foo(1)
assert_type(v, Foo)
"#,
);

testcase!(
    test_constructor_union,
    r#"
from typing import assert_type
class A: ...
class B: ...
def test(f: type[A | B]) -> A | B:
    return f()
"#,
);

testcase!(
    test_generic_class,
    r#"
from typing import assert_type
class Box[T]:
    def __init__(self, x: T): pass

    def wrap(self) -> Box[Box[T]]:
        return Box(self)

def f() -> int:
    return 1
b3 = Box(f()).wrap().wrap()
assert_type(b3, Box[Box[Box[int]]])

assert_type(Box[int](1), Box[int])
Box[int]("oops")  # E: Argument `Literal['oops']` is not assignable to parameter `x` with type `int`
"#,
);

testcase!(
    test_self_in_generic_class,
    r#"
from typing import reveal_type
class A[T]:
    x: T
    def __init__(self, x: T):
        reveal_type(self)  # E: revealed type: Self@A
        reveal_type(self.x)  # E: revealed type: T
        self.x = 1  # E: `Literal[1]` is not assignable to attribute `x` with type `T`
        self.x = x  # OK
    "#,
);

testcase!(
    bug = "int should not be assignable to type var with upper bound = int; bounded type var should be assignable to itself, but it is not because the T in x: T is different from the default T used to parameterize self",
    test_bounded_self_in_generic_class,
    r#"
from typing import reveal_type
class A[T: int]:
    x: T
    def __init__(self, x: T):
        reveal_type(self)  # E: revealed type: Self@A
        reveal_type(self.x)  # E: revealed type: T
        self.x = 1  # E: `Literal[1]` is not assignable to attribute `x` with type `T`
        self.x = x  # OK
    "#,
);

testcase!(
    test_typing_self_param_in_generic_class,
    r#"
from typing import Self, reveal_type
class A[T]:
    x: T
    def __init__(self, other: Self):
        reveal_type(other.x)  # E: revealed type: T
        self.x = other.x  # OK
    "#,
);

testcase!(
    test_generic_init_in_generic_class,
    r#"
from typing import assert_type
class Box[T]:
    def __init__[S](self, x: S, y: S):
        pass
    def wrap(self, x: bool) -> Box[Box[T]]:
        if x:
            return Box(self, self)
        else:
            return Box(self, 42)
b = Box[int]("hello", "world")
assert_type(b, Box[int])
assert_type(b.wrap(True), Box[Box[int]])
    "#,
);

testcase!(
    test_init_self_annotation,
    r#"
class C:
    def __init__[T](self: T, x: T):
        pass
def test(c: C):
    C(c)  # OK
    C(0)  # E: Argument `Literal[0]` is not assignable to parameter `x` with type `C`
    "#,
);

testcase!(
    test_init_self_annotation_in_generic_class,
    r#"
class C[T1]:
    def __init__[T2](self: T2, x: T2):
        pass
def test(c: C[int]):
    C[int](c)
    # Even though T1 is bivariant in C, we follow mypy and pyright's lead in treating it as invariant.
    C[str](c)  # E: `C[int]` is not assignable to parameter `x` with type `C[str]`
    "#,
);

testcase!(
    test_metaclass_call,
    r#"
class Meta(type):
    def __call__[T](cls: type[T], x: int) -> T: ...
class C(metaclass=Meta):
    def __init__(self, *args, **kwargs):
        pass
C(5)
C()     # E: Missing argument `x`
C("5")  # E: Argument `Literal['5']` is not assignable to parameter `x` with type `int`
    "#,
);

testcase!(
    test_metaclass_call_bad_classdef,
    r#"
class Meta(type):
    def __call__[T](cls: type[T], x: int) -> T: ...
# C needs to define __new__ and/or __init__ taking `x: int` to be compatible with Meta.
class C(metaclass=Meta):
    pass
# Both of these calls error at runtime.
C()   # E: Missing argument `x`
C(0)  # E: Expected 0 positional arguments
    "#,
);

testcase!(
    test_metaclass_call_returns_something_else,
    r#"
from typing import assert_type
class Meta(type):
    def __call__(cls) -> int:
        return 0
class C(metaclass=Meta):
    pass
x = C()
assert_type(x, int)
    "#,
);

testcase!(
    test_metaclass_invalid_generic,
    r#"
from typing import Any, assert_type
class Meta[T](type):
    def __call__(cls, x: T): ...
class C[T](metaclass=Meta[T]): # E: Metaclass may not be an unbound generic
    pass
assert_type(C(), C[Any]) # Correct, because invalid metaclass.
    "#,
);

testcase!(
    test_metaclass_invalid_generic_legacy_typevar,
    r#"
from typing import Any, Generic, TypeVar, assert_type
T = TypeVar("T")
class Meta(type, Generic[T]):
    foo: T
class C(metaclass=Meta[T]): # E: Metaclass may not be an unbound generic
    pass
    "#,
);

testcase!(
    test_metaclass_invalid_generic_legacy_typevar_with_default,
    r#"
from typing import Any, Generic, TypeVar, assert_type
T = TypeVar("T", default=int)
class Meta(type, Generic[T]):
    foo: T
class C(metaclass=Meta[T]): # E: Metaclass may not be an unbound generic
    pass
# After gradualization, T (default=int) becomes int.
assert_type(C.foo, int)
    "#,
);

testcase!(
    test_metaclass_invalid_generic_nested_targ,
    r#"
from typing import Any, assert_type
class Meta[T](type):
    foo: list[T]
class C[T](metaclass=Meta[list[T]]): # E: Metaclass may not be an unbound generic
    pass
# After gradualization, Meta[list[T]] becomes Meta[list[Any]], so foo: list[list[Any]].
assert_type(C.foo, list[list[Any]])
    "#,
);

testcase!(
    test_metaclass_invalid_generic_inherited,
    r#"
from typing import Any, assert_type
class Meta[T](type):
    foo: T
class Base[T](metaclass=Meta[T]): # E: Metaclass may not be an unbound generic
    pass
class Child(Base[int]):
    pass
    "#,
);

// Test reflects behavior of existing type checkers, but is probably worth revisiting with
// the typing council. Two reasons: (1) `class Meta(type)` is almost certainly a metaclass,
// and metaclass __call__ cls param is not actually `type[Self]`; (2) instantiating `T=str`
// may be reasonable here.
testcase!(
    test_metaclass_call_cls_param_does_not_instantiate,
    r#"
from typing import assert_type
class Meta(type):
    def __call__(cls: 'type[C[str]]', *args, **kwargs): ...  # E: `__call__` method self type `type[C[str]]` is not a superclass of class `Meta`
class C[T](metaclass=Meta):
    def __init__(self, x: T):
        pass
assert_type(C(0), C[int]) # Correct, because metaclass call does not instantiate T=str
    "#,
);

// Test reflects behavior of existing type checkers, but is probably worth revisiting with
// the typing council. I think instantiating `T=str` may be reasonable here.
testcase!(
    test_metaclass_call_does_not_instantiate,
    r#"
from typing import assert_type
class Meta(type):
    def __call__(cls, *args, **kwargs) -> 'C[str]':
        ...
class C[T](metaclass=Meta):
    def __init__(self, x: T):
        pass
assert_type(C(0), C[int])
    "#,
);

testcase!(
    test_new,
    r#"
class C:
    def __new__[T](cls: type[T], x: int) -> T: ...
C(5)
C()     # E: Missing argument `x`
C("5")  # E: Argument `Literal['5']` is not assignable to parameter `x` with type `int`
    "#,
);

testcase!(
    test_new_and_init,
    r#"
class C:
    def __new__[T](cls: type[T], x: int) -> T: ...
    def __init__(self, x: int):
        pass
C(5)
C()     # E: Missing argument `x` in function `C.__new__`
C("5")  # E: Argument `Literal['5']` is not assignable to parameter `x` with type `int` in function `C.__new__
    "#,
);

testcase!(
    test_new_and_inherited_init,
    r#"
class Parent1:
    def __init__(self):
        pass
class Parent2:
    def __init__(self, x: int):
        pass
class GoodChild(Parent2):
    def __new__[T](cls: type[T], x: int) -> T: ...
class BadChild(Parent1):
    # Incompatible with inherited __init__
    def __new__[T](cls: type[T], x: int) -> T: ...
GoodChild(0)
GoodChild()  # E: Missing argument `x` in function `GoodChild.__new__`
# Both of these calls error at runtime.
BadChild()   # E: Missing argument `x`
BadChild(0)  # E: Expected 0 positional arguments
    "#,
);

testcase!(
    test_new_returns_something_else,
    r#"
from typing import assert_type
class C:
    def __new__(cls) -> int:
        return 0
x = C()
assert_type(x, int)
    "#,
);

testcase!(
    test_new_returns_any,
    r#"
from typing import assert_type, Any
class C:
    def __new__(cls) -> Any:
        return 0
x = C()
assert_type(x, Any)
    "#,
);

testcase!(
    test_new_returns_error,
    r#"
from typing import assert_type, overload, Self
class C:
    @overload
    def __new__(cls, x: str) -> Self: ...
    @overload
    def __new__(cls, x: bytes) -> Self: ...
    def __new__(cls, x: str | bytes) -> Self: ...

# Intentionally make it such that `__new__` returns an error-induced `Any`
x = C(42)  # E: No matching overload found for function `C.__new__`
assert_type(x, C)
    "#,
);

testcase!(
    test_new_returns_something_else_generic,
    r#"
from typing import assert_type
class C[T]:
    def __new__(cls, x: T) -> list[T]:
        return []
x = C(0)
assert_type(x, list[int])
    "#,
);

testcase!(
    test_generic_new,
    r#"
from typing import Self, assert_type
class C[T]:
    def __new__(cls, x: T) -> Self: ...
assert_type(C(0), C[int])
assert_type(C[bool](True), C[bool])
assert_type(C[bool](0), C[bool])  # E: Argument `Literal[0]` is not assignable to parameter `x` with type `bool`
    "#,
);

testcase!(
    test_new_return_self,
    r#"
from typing import Self, assert_type
class C:
    x: int
    def __new__(cls) -> Self: ...
    def __init__(self):
        self.x = 42
assert_type(C().x, int)
    "#,
);

testcase!(
    test_inherit_dunder_init,
    r#"
class A:
    def __init__(self, x: int): pass
class B(A): pass
B(1)
B("")  # E: Argument `Literal['']` is not assignable to parameter `x` with type `int`
    "#,
);

testcase!(
    test_decorated_init,
    r#"
from typing import Any, assert_type
def decorator(func) -> Any: ...
class C:
    @decorator
    def __init__(self): ...
assert_type(C(), C)
    "#,
);

testcase!(
    test_metaclass_call_noreturn,
    r#"
from typing import Self, NoReturn, Never, assert_type
class Meta(type):
    def __call__(cls, *args, **kwargs) -> NoReturn:
        raise TypeError("Cannot instantiate class")
class MyClass(metaclass=Meta):
    def __new__(cls, *args, **kwargs) -> Self:
        return super().__new__(cls, *args, **kwargs)
assert_type(MyClass(), Never)
    "#,
);

testcase!(
    test_metaclass_call_unannotated,
    r#"
from typing import Self
class Meta(type):
    # This is unannotated, so we should treat it as compatible and use the signature of __new__
    def __call__(cls, *args, **kwargs):
        raise TypeError("Cannot instantiate class")
class MyClass(metaclass=Meta):
    def __new__(cls, x: int) -> Self:
        return super().__new__(cls)
MyClass()  # E: Missing argument `x` in function `MyClass.__new__`
    "#,
);

testcase!(
    test_new_explicit_any_return,
    r#"
from typing import Any, assert_type
class MyClass:
    def __new__(cls) -> Any:
        return 0
    # The __init__ method will not be called in this case, so
    # it should not be evaluated.
    def __init__(self, x: int):
        pass
assert_type(MyClass(), Any)
    "#,
);

testcase!(
    test_cls_type_in_new_annotated,
    r#"
from typing import Self
class A:
    def __new__(cls: type[Self]): ...
A.__new__(A)  # OK
A.__new__(int) # E: `int` is not assignable to upper bound `A` of type variable `Self@A`
    "#,
);

testcase!(
    test_cls_type_in_new_unannotated,
    r#"
class A:
    def __new__(cls): ...
A.__new__(A)  # OK
A.__new__(int)  # E: `int` is not assignable to upper bound `A` of type variable `Self@A`
    "#,
);

testcase!(
    test_subst_self_type_in_static_new,
    r#"
class A: pass
a: A = A.__new__(A)
    "#,
);

testcase!(
    test_call_self,
    r#"
class Foo:
    @classmethod
    def foo(cls) -> None:
      cls()

class Bar:
    def __call__(self) -> None:
      pass

    def bar(self) -> None:
      self()
    "#,
);

// See https://typing.python.org/en/latest/spec/generics.html#instantiating-generic-classes-and-type-erasure.
// When no type argument is provided, we fall back to the default or Any. Specifically, the bound is not used.
testcase!(
    test_construction_no_targ,
    r#"
from typing import assert_type, Any
class C1[T: float]:
    pass
class C2[T: float = int]:
    pass
assert_type(C1(), C1[Any])
assert_type(C2(), C2[int])
    "#,
);

testcase!(
    test_specialize_in_new,
    r#"
from typing import assert_type
class C[T]:
    def __new__[T2](cls, x: T2) -> C[T2]: ...
assert_type(C(0), C[int])
    "#,
);

testcase!(
    test_specialize_in_init,
    r#"
from typing import assert_type
class A[T]:
    def __init__(self, x: type[T]):
        self._x = x
        self.f()
    def f(self):
        pass
assert_type(A(int), A[int])
    "#,
);

testcase!(
    test_overload_init,
    r#"
from typing import overload, assert_type
class C[T]:
    @overload
    def __init__(self, x: T, y: int): ... # E: Overloaded function must have an implementation
    @overload
    def __init__(self, x: int, y: T): ...
assert_type(C(0, "foo"), C[str])
"#,
);

testcase!(
    test_new_and_init_generic,
    r#"
from typing import Self,assert_type

class Class2[T]:
    def __new__(cls, *args, **kwargs) -> Self: ...
    def __init__(self, x: T) -> None: ...

assert_type(Class2(1), Class2[int])
    "#,
);

// Per the typing spec:
// > If any class-scoped type variables are not solved when evaluating the __new__ method call
// > using the supplied arguments, these type variables should be left unsolved, allowing the
// >__init__ method (if applicable) to be used to solve them.
//
// However, neither mypy nor pyright pass this test, so either the spec wording is imprecise,
// or the other type checkers are incorrect.
testcase!(
    test_new_and_init_partial_instantiation,
    r#"
from typing import Any, Self, assert_type

class C[T, U]:
    def __new__(cls, x: T, y: Any) -> Self:
        return super().__new__(cls)

    def __init__(self, x: Any, y: U):
        pass

assert_type(C(0, ""), C[int, str])
    "#,
);

testcase!(
    test_deprecated_new,
    r#"
from warnings import deprecated
class A:
    @deprecated("old old old")
    def __new__(cls, x: int):
        return super().__new__(cls)
    def __init__(self, x: str):
        pass
A(0) # E: `A.__new__` is deprecated # E: `Literal[0]` is not assignable to parameter `x` with type `str`
    "#,
);

testcase!(
    test_annotate_self,
    r#"
from typing import assert_type
class A[T]:
    def __init__(self: A[str]): pass
assert_type(A(), A[str])
    "#,
);

testcase!(
    test_targ_mismatch,
    r#"
class A[T]:
    def __init__(self, x: T):
        pass
a = A(0)
b: A[str] = a  # E: `A[int]` is not assignable to `A[str]`
    "#,
);

testcase!(
    test_overloaded_init,
    r#"
from typing import Literal, assert_type, overload

class A: ...
class B: ...

class C[T]:
    @overload
    def __init__(self: C[A], x: Literal[True]) -> None: ...
    @overload
    def __init__(self: C[B], x: Literal[False]) -> None: ...
    def __init__(self, x):
        pass

assert_type(C(True), C[A])
assert_type(C(False), C[B])
    "#,
);

testcase!(
    test_init_bad_receiver_annotation,
    r#"
from typing import Literal, assert_type, overload

class A: ...
class B: ...
class D:
    def __init__(self: A): pass  # E: `__init__` method self type `A` is not a superclass of class `D`
class E(A):
    def __init__(self: A): pass

class C[T]:
    @overload
    def __init__(self: A, x: Literal[True]) -> None: ...  # E: `__init__` method self type `A` is not a superclass of class `C`
    @overload
    def __init__(self: B, x: Literal[False]) -> None: ...  # E: `__init__` method self type `B` is not a superclass of class `C`
    def __init__(self, x):
        pass

    "#,
);

testcase!(
    test_init_transitive_superclass_annotation,
    r#"
class A: ...
class B(A): ...
class C(B):
    def __init__(self: A): pass

class D(B):
    def __init__(self: B): pass
    "#,
);

testcase!(
    test_init_non_classtype_self_annotation,
    r#"
from typing import Self, Any

class E:
    def __init__(self: Self): pass

class F:
    def __init__(self: Any): pass
    "#,
);

testcase!(
    test_new_bad_receiver_annotation,
    r#"
from typing import Literal, assert_type, overload, Self, Any

class A: ...
class B: ...
class D:
    def __new__(cls: type[A]): pass  # E: `__new__` method cls type `type[A]` is not a superclass of class `D`
class E(A):
    def __new__(cls: type[A]): pass
class F:
    def __new__(cls: A): pass  # E: `__new__` method cls type `A` is not a valid `type[...]` annotation
class G:
    def __new__(cls: Self): pass  # E: `__new__` method cls type `Self@G` is not a valid `type[...]` annotation
class H:
    def __new__(cls: type[Self]): pass
class I:
    def __new__(cls: type): pass
class J:
    def __new__(cls: Any): pass
class K:
    def __new__(cls: type[Any]): pass

class C[T]:
    @overload
    def __new__(cls: type[A], x: Literal[True]): ...  # E: `__new__` method cls type `type[A]` is not a superclass of class `C`  # E: Implementation signature `(cls: type[Self@C], x: Unknown) -> Self@C` does not accept all arguments that overload signature `(cls: type[A], x: Literal[True]) -> Self@C`
    @overload
    def __new__(cls: type[B], x: Literal[False]): ...  # E: `__new__` method cls type `type[B]` is not a superclass of class `C`  # E: Implementation signature `(cls: type[Self@C], x: Unknown) -> Self@C` does not accept all arguments that overload signature `(cls: type[B], x: Literal[False]) -> Self@C` accepts
    def __new__(cls, x):
        pass

    "#,
);

testcase!(
    test_generic_in_generic,
    r#"
from typing import Literal, assert_type, overload

class A: ...
class B: ...

class TypeEngine[T]: ...

class UUID[T: (A, B)](TypeEngine[T]):
    @overload
    def __init__(self: UUID[A], as_uuid: Literal[True]) -> None: ...
    @overload
    def __init__(self: UUID[B], as_uuid: Literal[False]) -> None: ...
    def __init__(self, as_uuid): ...

class Column[T]:
    def __init__(self, ty: TypeEngine[T]) -> None: ...

assert_type(Column(UUID(as_uuid=False)), Column[B])
    "#,
);

testcase!(
    test_typevar_with_explicit_any_default,
    r#"
from typing import Any, Generic, TypeVar, assert_type

T = TypeVar("T", bound=int, default=Any)

class A(Generic[T]):
    def __new__(cls, x: T) -> A[T]: ...

assert_type(A(0), A[int])
A("oops")  # E: `str` is not assignable to upper bound `int` of type variable `T`
    "#,
);

testcase!(
    test_typevar_with_explicit_any_default_in_nested_constructor_call,
    r#"
from typing import Any, assert_type

class A[T = Any]:
    def __new__(cls, x: T) -> A[T]: ...

class B[T: int | A[Any] = Any]:
    def __new__(cls, x: list[A[T]]) -> B[A[T]]: ...

assert_type(B([A(0)]), B[A[int]])
B([A("oops")])  # E: `str` is not assignable to upper bound `A | int` of type variable `T`
    "#,
);

testcase!(
    test_init_overload_with_self,
    r#"
from typing import Generic, TypeVar, overload, Callable
T = TypeVar("T")
class C(Generic[T]):
    @overload
    def __init__(self: "C[int]", x: int) -> None:
        ...
    @overload
    def __init__(self: "C[str]", x: str) -> None:
        ...
    def __init__(self, x: int | str) -> None:
        ...

def takes_Cint(x: Callable[[int], C[int]]) -> None:
    pass
def takes_Cstr(x: Callable[[str], C[str]]) -> None:
    pass
def takes_Cstr_wrong(x: Callable[[str], C[int]]) -> None:
    pass
takes_Cint(C)
takes_Cstr(C)
takes_Cstr_wrong(C) # E: Argument `type[C]` is not assignable to parameter `x` with type `(str) -> C[int]` in function `takes_Cstr_wrong`
    "#,
);

testcase!(
    test_init_to_callable_generics,
    r#"
from typing import Generic, TypeVar, assert_type, Callable
T = TypeVar("T")
class C(Generic[T]):
    def __init__[V](self: "C[V]", x: V) -> None: pass
def takes_callable[V](x: Callable[[V], C[V]], y: V) -> C[V]: ...
out1 = takes_callable(C, 42)
assert_type(out1, C[int])
out2 = takes_callable(C, "hello")
assert_type(out2, C[str])
    "#,
);

testcase!(
    test_init_class_scoped_typevars_in_self,
    r#"
from typing import Generic, TypeVar

T1 = TypeVar("T1")
T2 = TypeVar("T2")

class Class8(Generic[T1, T2]):
    def __init__(self: "Class8[T2, T1]") -> None:  # E: `__init__` method self type cannot reference class type parameters `T2`, `T1`
        pass
"#,
);

testcase!(
    test_constructor_typevar_scope,
    r#"
from typing import Generic, TypeVar
T = TypeVar("T")
class Ok1(Generic[T]):
    def __init__(self: "Ok1[int]") -> None:
        pass
class Ok2[T]:
    def __init__(self: "Ok2[int]") -> None:
        pass
class Ok3(Generic[T]):
    def __init__(self) -> None:
        pass
class Ok4[T]:
    def __init__(self) -> None:
        pass
class Ok5(Generic[T]):
    def __init__[V](self: "Ok5[V]", arg: V) -> None:
        pass
class Ok6[T]:
    def __init__[V](self: "Ok6[V]", arg: V) -> None:
        pass
class Bad1(Generic[T]):
    def __init__(self: "Bad1[T]") -> None: # E: `__init__` method self type cannot reference class type parameter `T`
        pass
class Bad2[T]:
    def __init__(self: "Bad2[T]") -> None: # E: `__init__` method self type cannot reference class type parameter `T`
        pass
"#,
);

testcase!(
    test_constructor_typevar_scope_nested,
    r#"
from typing import Generic, TypeVar
T = TypeVar("T")
# Nested type variables should also be detected (e.g., Foo[list[T]])
class Bad1(Generic[T]):
    def __init__(self: "Bad1[list[T]]") -> None: # E: `__init__` method self type cannot reference class type parameter `T`
        pass
class Bad2[T]:
    def __init__(self: "Bad2[tuple[T, int]]") -> None: # E: `__init__` method self type cannot reference class type parameter `T`
        pass
"#,
);

testcase!(
    test_constructor_typevar_scope_overload,
    r#"
from typing import Generic, TypeVar, overload
T = TypeVar("T")
# Overloaded __init__ methods should also be checked
class Bad1(Generic[T]):
    @overload
    def __init__(self: "Bad1[T]", x: int) -> None: # E: `__init__` method self type cannot reference class type parameter `T`
        ...
    @overload
    def __init__(self: "Bad1[str]", x: str) -> None:
        ...
    def __init__(self, x: int | str) -> None:
        pass
class Ok1(Generic[T]):
    @overload
    def __init__(self: "Ok1[int]", x: int) -> None:
        ...
    @overload
    def __init__(self: "Ok1[str]", x: str) -> None:
        ...
    def __init__(self, x: int | str) -> None:
        pass
"#,
);

testcase!(
    test_class_scoped_typevar_in_decorated_init,
    r#"
from typing import Any
def decorate(f) -> Any: ...
class A[T]:
    @decorate
    def __init__(self: A[T]): ...  # E: self type cannot reference class type parameter `T`
    "#,
);

testcase!(
    test_new_returns_concrete_inside_method,
    r#"
from typing import Self, reveal_type

class C:
    def __new__(cls) -> "C": ...

    def method(self) -> None:
        # __new__ explicitly returns C, not Self, so type(self)() returns C.
        reveal_type(type(self)())  # E: revealed type: C

class D(C): ...

def check_subclass(d: D) -> None:
    reveal_type(type(d)())  # E: revealed type: C
    "#,
);

testcase!(
    test_new_returns_list_self_inside_method,
    r#"
from typing import Self, reveal_type

class C:
    def __new__(cls) -> list[Self]: ...

    def method(self) -> None:
        # __new__ returns list[Self], so type(self)() preserves Self.
        reveal_type(type(self)())  # E: revealed type: list[Self@C]

class D(C): ...

def check_subclass(d: D) -> None:
    reveal_type(type(d)())  # E: revealed type: list[D]
    "#,
);

testcase!(
    test_metaclass_call_with_overridden_new,
    r#"
from typing import Self, assert_type

class Meta(type):
    def __call__[T](cls: type[T], x: int) -> T: ...

class C(metaclass=Meta):
    def __new__(cls, x: int) -> Self:
        return super().__new__(cls)

c = C(5)
assert_type(c, C)
C()     # E: Missing argument `x`  # E: Missing argument `x` in function `C.__new__`
C("5")  # E: Argument `Literal['5']` is not assignable to parameter `x` with type `int`  # E: Argument `Literal['5']` is not assignable to parameter `x` with type `int` in function `C.__new__`
    "#,
);

// Regression test for a problem in networkx: https://github.com/facebook/pyrefly/issues/3121
testcase!(
    test_return_type_inference_for_constructors,
    r#"
from typing import assert_type

class A:
    def __new__(cls, x: int | None = None):
        if x is None:
            return cls.__new__(cls, 5)
        else:
            return object.__new__(cls)

    def __init__(self):
        return "x"

class B(A): ...

a = A()
assert_type(a, A)
b = B()
assert_type(b, B)
"#,
);

testcase!(
    test_redundant_dict_constructor_call_ok,
    r#"
from collections.abc import Mapping
from typing import Literal

type Kind = Literal["a", "b"]

def g(x: Mapping[Kind, int]) -> None: ...

def f(x: dict[Kind, int]) -> None:
    g(dict(x))
    "#,
);

testcase!(
    test_overloaded_constructor_with_hint,
    r#"
from collections.abc import Mapping
from typing import assert_type, Generic, Never, overload, SupportsInt, TypeVar

_T_co = TypeVar("_T_co", covariant=True)
_T = TypeVar("_T")

class Box(Generic[_T_co]):
    @overload
    def __init__(self: "Box[Never]", val: Mapping[Never, SupportsInt], /) -> None: ...
    @overload
    def __init__(self: "Box[_T]", val: Mapping[_T, SupportsInt], /) -> None: ...
    def __init__(self, val: object, /) -> None:
        pass

def process(items: Box[_T]) -> "Box[_T]": ...

assert_type(process(Box({1: 1})), Box[int])
    "#,
);

testcase!(
    test_construct_list_with_union_input,
    r#"
Y = list[int] | list[str] | str
def f(x: str):
    y: Y = list(x)
    "#,
);

testcase!(
    test_construct_list_from_iterator_with_parent_hint,
    r#"
from collections.abc import Iterator

class ParentItem: ...
class ChildItem(ParentItem): ...

def f() -> Iterator[ChildItem]: ...
def g() -> list[ParentItem] | None:
    return list(f())
    "#,
);

// Overloaded __new__ where one overload has an explicit return annotation
// and one doesn't. The unannotated overload should assume Self; the
// annotated overload should keep its declared return type.
testcase!(
    test_overloaded_new_mixed_annotation,
    r#"
from typing import assert_type, overload

class C:
    @overload
    def __new__(cls, x: int) -> "C": ...
    @overload
    def __new__(cls, x: str): ...
    def __new__(cls, x: int | str):
        return object.__new__(cls)

class D(C): ...

# int overload: explicit return annotation -> C (not D)
c1 = C(1)
assert_type(c1, C)
d1 = D(1)
assert_type(d1, C)

# str overload: unannotated -> Self (D for D(), C for C())
c2 = C("a")
assert_type(c2, C)
d2 = D("a")
assert_type(d2, D)
    "#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/3236
testcase!(
    test_construct_with_hint_and_overloads,
    r#"
from typing import Generic, Never, overload, Protocol, TypeVar

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
_AddWithT_contra = TypeVar("_AddWithT_contra", contravariant=True)
_ResultT_co = TypeVar("_ResultT_co", covariant=True)
_AddWithT = TypeVar("_AddWithT")
_ResultT = TypeVar("_ResultT")

class CanAdd(Protocol[_AddWithT_contra, _ResultT_co]):
    def __add__(self, other: _AddWithT_contra, /) -> _ResultT_co: ...

class H(Generic[_T_co]):
    @overload
    def __init__(self: "H[Never]", init_val: dict[Never, int], /) -> None: ...
    @overload
    def __init__(self: "H[_T]", init_val: dict[_T, int], /) -> None: ...
    @overload
    def __init__(self: "H[int]", init_val: int, /) -> None: ...
    def __init__(self, init_val: object, /) -> None: ...

def explode_n(source: "H[CanAdd[_AddWithT, _ResultT]]") -> "H[_ResultT]":
    raise NotImplementedError

result: "H[int]" = explode_n(H(10))
    "#,
);

testcase!(
    test_hint_and_bound_interaction,
    r#"
from typing import assert_type, Self, Sequence
class C[T: int | list[str]]:
    def __new__(cls, data: T | Sequence[T]) -> Self: ...
x = C([1, 2])
assert_type(x, C[int])
    "#,
);
