/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_class_super_no_args,
    r#"
from typing import assert_type

class A:
    def m(self) -> int:
        return 0

class B(A):
    def m(self) -> int:
        return super().m()

class C(A):
    def m(self) -> bool:
        return True
    def f(self):
        assert_type(super().m(), int)
"#,
);

testcase!(
    test_class_super_with_args,
    r#"
from typing import assert_type

class A:
    def f(self) -> int:
        return 0

class B:
    def f(self) -> bool:
        return True

class C(B, A):
    def g(self):
        assert_type(super(C, self).f(), bool)
        assert_type(super(B, self).f(), int)
    "#,
);

testcase!(
    bug = "Demonstration of a limitation of our super() implementation",
    test_inherit_method_with_super,
    r#"
from typing import assert_type

class A:
    def f(self) -> int:
        return 0
class B(A):
    def f(self):
        return super().f()
class C:
    def f(self) -> bool:
        return True
class D(B, C, A):
    pass

# At runtime, the super() call in B.f is evaluated with D as the starting class, so that C.f is called.
# We can't do this statically without re-analyzing the body of B.f (too expensive).
assert_type(D().f(), bool)  # E: assert_type(int, bool)
    "#,
);

testcase!(
    test_bad_args,
    r#"
super(1, 2, 3)  # E: `super` takes at most 2 arguments, got 3

class Unrelated:
    pass

class C:
    def f1(self):
        super(C, self, oops=42)  # E: `super` got an unexpected keyword argument `oops`
    def f2(self):
        super(42, self)  # E: Expected first argument to `super` to be a class object, got `Literal[42]`
    def f3(self):
        super(C, int | str)  # E: Expected second argument to `super` to be a class object or instance, got `type[int | str]`
    def f4(self):
        super(Unrelated, self)  # E: Illegal `super(type[Unrelated], C)` call: `C` is not an instance or subclass of `type[Unrelated]`
    "#,
);

testcase!(
    test_super_object,
    r#"
# Trying to call super() on `object` is a weird thing to do (although the Python runtime allows it).
# Either accepting this or producing a good error message would be ok.
class C:
    def f(self):
        super(object, self)
    "#,
);

testcase!(
    test_super_alias,
    r#"
_super = super
class C:
    def f(self):
        _super(C, self)
    "#,
);

testcase!(
    test_super_protocol_unimplemented_method,
    r#"
from typing import Protocol

class PColor(Protocol):
    def draw(self) -> str: ...

class BadColor(PColor):
    def draw(self) -> str:
        return super().draw()  # E: Method `draw` inherited from class `PColor` has no implementation and cannot be accessed via `super()`
    "#,
);

testcase!(
    test_super_abstract_method,
    r#"
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        ...

class Triangle(Shape):
    def area(self) -> float:
        return super().area()  # E: Method `area` inherited from class `Shape` has no implementation and cannot be accessed via `super()`
    "#,
);

testcase!(
    test_super_method_assigned_to_self_attribute,
    r#"
class Parent:
    def meth1(self) -> None: ...
    @classmethod
    def meth2(cls) -> None: ...
    @staticmethod
    def meth3() -> None: ...

class Child(Parent):
    def __init__(self) -> None:
        self.meth1 = super().meth1
        self.meth2 = super().meth2
        self.meth3 = super().meth3

Parent().meth1()
Child().meth1()
Parent().meth2()
Child().meth2()
Parent().meth3()
Child().meth3()
"#,
);

testcase!(
    test_instance_method_assigned_to_incompatible_inherited_classmethod,
    r#"
from typing import assert_type

class Parent:
    @classmethod
    def meth1(self) -> int:
        return 2

    @classmethod
    def meth2(self) -> str:
        return "'4'"


class Child(Parent):
    def __init__(self) -> None:
        self.meth2 = super().meth1  # E: `(self: type[Self@Child]) -> int` is not assignable to attribute `meth2` with type `(self: type[Self@Child]) -> str`

# At runtime, this is a call to the inherited `Parent.meth2` classmethod.
# We don't have a good way of modeling this, so we treat this as an (illegal) class access of the
# `Child.meth2` instance attribute.
Child.meth2()  # E: Instance-only attribute `meth2` of class `Child` is not visible on the class

# At runtime, this is a call to the `Child.meth2` instance attribute, which returns ` int` because
# it was assigned to `Parent.meth1`. However, we rejected the assignment above due to incompatible
# signatures, so we get the inherited return type of `Parent.meth2`.
assert_type(Child().meth2(), str)
    "#,
);

testcase!(
    test_illegal_location,
    r#"
class A:
    pass

# This is unusual but legal
super(A, A())

# The following are runtime errors
super()  # E: `super` call with no arguments is valid only inside a method
class B:
    super()  # E: `super` call with no arguments is valid only inside a method
    "#,
);

testcase!(
    test_dunder_new_implicit,
    r#"
class A:
    def __new__(cls, x):
        return super().__new__(cls)
    def __init__(self, x):
        self.x = x

class B(A):
    def __new__(cls, x):
        return super().__new__(cls, x)
    def __init__(self, x):
        super().__init__(x)
    "#,
);

testcase!(
    test_dunder_new_explicit_with_annotated_cls,
    r#"
from typing import Self

class A:
    def __new__(cls: type[Self], x):
        return super(A, cls).__new__(cls)
    def __init__(self, x):
        self.x = x

class B(A):
    def __new__(cls: type[Self], x):
        return super(B, cls).__new__(cls, x)
    def __init__(self, x):
        super().__init__(x)
    "#,
);

testcase!(
    test_dunder_new_explicit_with_unannotated_cls,
    r#"
from typing import Self

class A:
    def __new__(cls, x):
        return super(A, cls).__new__(cls)
    def __init__(self, x):
        self.x = x

class B(A):
    def __new__(cls, x):
        return super(B, cls).__new__(cls, x)
    def __init__(self, x):
        super().__init__(x)
    "#,
);

testcase!(
    test_super_new_return,
    r#"
from typing import Self
class A:
    def __new__(cls) -> Self:
        return super().__new__(cls)
    "#,
);

testcase!(
    test_staticmethod,
    r#"
from typing import assert_type

class A:
    @staticmethod
    def f() -> int:
        return 0

class B(A):
    @staticmethod
    def g():
        # Two-argument super() works fine
        assert_type(super(B, B).f(), int)
    @staticmethod
    def h():
        # No-argument super() is a runtime error
        super().f()  # E: `super` call with no arguments is not valid inside a staticmethod
    "#,
);

testcase!(
    test_classmethod,
    r#"
from typing import assert_type

class A:
    @classmethod
    def f(cls) -> int:
        return 0

class B(A):
    @classmethod
    def g(cls):
        assert_type(super().f(), int)
        assert_type(super(B, cls).f(), int)
    "#,
);

testcase!(
    test_call_instance_method_from_classmethod,
    r#"
class A:
    def f(self):
        pass

class B(A):
    @classmethod
    def g(cls):
        super().f(cls())
    "#,
);

// This is beyond what we support. We don't care what errors are generated as long as we don't crash.
testcase!(
    test_super_in_base_classes,
    r#"
import types
from typing import Iterable
class Alias(types.GenericAlias):
    def __mro_entries__(self, bases: Iterable[object], /) -> tuple[type, ...]:
        class C(*super().__mro_entries__(bases)): # E:
            pass
        return (C,)
    "#,
);

testcase!(
    test_super_multiple_inheritance,
    r#"
class A:
    def method_a(self):
        print("Method from A")
class B(A):
    def method_b(self):
        super().method_a()
        print("Method from B")
class MixinC:
    def method_c(self):
        print("Method from MixinC")
class D(MixinC, B):
    def method_a(self):
        super().method_a()
    def method_b(self):
        super().method_b()
    def method_c(self):
        super().method_c()
    "#,
);

testcase!(
    test_super_with_error_base,
    r#"
from nowhere import bob  # E: Cannot find module
class Foo(bob):
    def __init__(self):
        super().__init__(1)
class Bar(bob):
    def __new__(cls):
        return super().__new__(cls, 1)
    "#,
);

// Verify super().__init__() correctly handles kwargs forwarding with default params.
testcase!(
    test_super_init_kwargs_forwarding,
    r#"
class Parent:
    def __init__(self, x: int, y: int = 0) -> None: ...

class Child(Parent):
    def __init__(self, x: int, **kwargs) -> None:
        super().__init__(x, **kwargs)

class OnlyKwargs(Parent):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

class NoArgsOptionalParent:
    def __init__(self, x: int = 0, y: int = 0) -> None: ...

class NoArgsChild(NoArgsOptionalParent):
    def __init__(self) -> None:
        super().__init__()
    "#,
);

// Verify super().__init__() works with explicit kwargs alongside **kwargs.
testcase!(
    test_super_init_explicit_and_kwargs,
    r#"
class Parent:
    def __init__(self, size: dict, **kwargs) -> None: ...

class Child(Parent):
    def __init__(self, size: dict = {}, **kwargs) -> None:
        super().__init__(size=size, **kwargs)
    "#,
);

// Verify super().__init__() works with all positional args, including unannotated.
testcase!(
    test_super_init_all_positional,
    r#"
class AnnotatedParent:
    def __init__(self, config: int, layer_idx: int) -> None: ...

class AnnotatedChild(AnnotatedParent):
    def __init__(self, config: int, layer_idx: int) -> None:
        super().__init__(config, layer_idx)

class UnannotatedParent:
    def __init__(self, config, layer_idx): ...

class UnannotatedChild(UnannotatedParent):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
    "#,
);
