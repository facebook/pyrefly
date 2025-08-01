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
from typing import assert_type, reveal_type
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
            return Box(self, self)  # ok
        else:
            return Box(self, 42)  # E: Argument `Literal[42]` is not assignable to parameter `y` with type `Self@Box`
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

// This is the same pyre1 behavior. We infer bivariance here in T1 as well as T2"
testcase!(
    test_init_self_annotation_in_generic_class,
    r#"
class C[T1]:
    def __init__[T2](self: T2, x: T2):
        pass
def test(c: C[int]):
    C[int](c)  
    C[str](c)  
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
    bug = "Generic metaclasses are not allowed, should error on `C` classdef.",
    test_metaclass_invalid_generic,
    r#"
from typing import Any, assert_type
class Meta[T](type):
    def __call__(cls, x: T): ...
class C[T](metaclass=Meta[T]): # TODO: error here (or possibly on Meta classdef)
    pass
assert_type(C(), C[Any]) # Correct, because invalid metaclass.
    "#,
);

// Test reflects behavior of existing type checkers, but is probably worth revisiting with
// the typing council. Two reasons: (1) `class Meta(type)` is almost certainly a metaclass,
// and metaclass __call__ cls param is not actually `type[Self]`; (2) instantiating `T=str`
// may be reasonable here.
testcase!(
    bug = "Missing check that self/cls param is a supertype of the defining class",
    test_metaclass_call_cls_param_does_not_instantiate,
    r#"
from typing import assert_type
class Meta(type):
    def __call__(cls: 'type[C[str]]', *args, **kwargs): ... # TODO: error because annot is not supertype of Meta
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
    bug = "We should specialize `type[Self@A]` to `type[A]` in the call to `A.__new__`",
    test_cls_type_in_new_annotated,
    r#"
from typing import Self
class A:
    def __new__(cls: type[Self]): ...
A.__new__(A)  # OK
A.__new__(int) # E: Argument `type[int]` is not assignable to parameter `cls` with type `type[Self@A]` in function `A.__new__`
    "#,
);

testcase!(
    bug = "We should specialize `type[Self@A]` to `type[A]` in the call to `A.__new__`",
    test_cls_type_in_new_unannotated,
    r#"
class A:
    def __new__(cls): ...
A.__new__(A)  # OK
A.__new__(int)  # E: Argument `type[int]` is not assignable to parameter `cls` with type `type[Self@A]` in function `A.__new__`
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
