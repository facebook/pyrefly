/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    test_slots_manual_tuple_rejects_undeclared,
    r#"
class C:
    __slots__ = ("x",)

    def __init__(self):
        self.x = 1
        self.y = 2  # E: not declared in `__slots__`
"#,
);

testcase!(
    test_slots_manual_list_rejects_undeclared,
    r#"
class C:
    __slots__ = ["x"]

    def __init__(self):
        self.x = 1
        self.y = 2  # E: not declared in `__slots__`
"#,
);

testcase!(
    test_slots_inherited_union,
    r#"
class Base:
    __slots__ = ("x",)

class Child(Base):
    __slots__ = ("y",)

    def method(self):
        self.x = 1
        self.y = 2
        self.z = 3  # E: not declared in `__slots__`
"#,
);

testcase!(
    test_slots_unslotted_child_allows_dynamic,
    r#"
class Base:
    __slots__ = ("x",)

class Child(Base):
    pass  # no __slots__, so dynamic attributes allowed

    def method(self):
        self.x = 1
        self.y = 2  # OK: Child is unslotted
"#,
);

testcase!(
    test_slots_dict_allows_dynamic,
    r#"
class C:
    __slots__ = ("x", "__dict__")

    def __init__(self):
        self.x = 1
        self.y = 2  # OK: __dict__ in slots allows arbitrary attrs
"#,
);

testcase!(
    test_slots_dynamic_expression_no_enforcement,
    r#"
from typing import Sequence

def get_slots() -> Sequence[str]:
    return ("x",)

class C:
    __slots__ = get_slots()  # dynamic, can't determine slot names statically

    def __init__(self):
        self.x = 1
        self.y = 2  # OK: no enforcement for dynamic slots
"#,
);

testcase!(
    test_slots_property_setter_allowed,
    r#"
class C:
    __slots__ = ("_x",)

    @property
    def x(self) -> int:
        return self._x

    @x.setter
    def x(self, value: int) -> None:
        self._x = value

c = C()
c.x = 42  # OK: property setter, not direct instance storage
"#,
);

testcase!(
    test_slots_custom_setattr_suppresses_enforcement,
    r#"
class C:
    __slots__ = ("x",)

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)

    def __init__(self):
        self.x = 1
        self.y = 2  # OK: custom __setattr__ suppresses slots enforcement
"#,
);

testcase!(
    test_slots_single_string_literal,
    r#"
class C:
    __slots__ = "x"

    def __init__(self):
        self.x = 1
"#,
);

testcase!(
    test_slots_external_write_rejected,
    r#"
class C:
    __slots__ = ("x",)

    def __init__(self):
        self.x = 1

c = C()
c.x = 2
c.y = 3  # E: not declared in `__slots__`
"#,
);

testcase!(
    test_slots_dataclass_slots_true,
    r#"
from dataclasses import dataclass

@dataclass(slots=True)
class C:
    x: int
    y: str

c = C(x=1, y="a")
c.x = 2
c.z = 3  # E: not declared in `__slots__`
"#,
);

testcase!(
    test_slots_dataclass_slots_true_with_dict,
    r#"
from dataclasses import dataclass
from typing import Any

@dataclass(slots=True)
class C:
    x: int
    __dict__: dict[str, Any]

    def method(self):
        self.x = 1
        self.y = 2  # OK: __dict__ in dataclass fields disables slots enforcement
"#,
);

testcase!(
    test_slots_dataclass_inherits_unslotted,
    r#"
from dataclasses import dataclass

@dataclass
class Base:
    x: int

@dataclass(slots=True)
class Child(Base):
    y: str

c = Child(x=1, y="a")
c.x = 2
c.y = "b"
"#,
);

testcase!(
    test_slots_annotation_not_in_slots,
    r#"
class C:
    __slots__ = ("x",)
    x: int
    y: int  # annotation for attribute not in __slots__

    def __init__(self):
        self.x = 1

c = C()
c.x = 2
c.y = 3  # E: not declared in `__slots__`
"#,
);

testcase!(
    test_slots_classvar_not_in_slots,
    r#"
class C:
    __slots__ = ("x",)
    x: int
    y: int = 5  # class variable, not in __slots__

    def __init__(self):
        self.x = 1

c = C()
c.x = 2
c.y = 3  # E: not declared in `__slots__`
"#,
);

testcase!(
    test_slots_generic_class,
    r#"
from typing import Generic, TypeVar

T = TypeVar("T")

class C(Generic[T]):
    __slots__ = ("value",)

    def __init__(self, value: T) -> None:
        self.value = value

c: C[int] = C(1)
c.value = 2
c.other = 3  # E: not declared in `__slots__`
"#,
);

testcase!(
    test_slots_descriptor_allowed,
    r#"
class MyDescriptor:
    def __get__(self, obj: object, type: type | None = None) -> int:
        return 42

    def __set__(self, obj: object, value: int) -> None:
        pass

class C:
    __slots__ = ("_x",)
    x = MyDescriptor()  # descriptor not in __slots__, but should be allowed

    def __init__(self) -> None:
        self._x = 1

c = C()
c.x = 2  # OK: descriptor __set__ handles this, not instance storage
"#,
);

// https://github.com/facebook/pyrefly/issues/2917
testcase!(
    test_slots_multiple_inheritance_layout_conflict,
    r#"
class Left:
    __slots__ = ("a", "b")

class Right:
    __slots__ = ("c", "d")

# Inheriting from two classes that both define non-empty __slots__
# causes a TypeError at runtime.
class Combined(Left, Right): ...  # E: multiple base classes with non-empty `__slots__`
"#,
);

// https://github.com/facebook/pyrefly/issues/2916
testcase!(
    test_slots_layout_conflict_same_names,
    r#"
class First:
    __slots__ = ("x",)

class Second:
    __slots__ = ("x",)

# Even though the slot names match, these are different C-level layouts.
class Both(First, Second): ...  # E: multiple base classes with non-empty `__slots__`
"#,
);

testcase!(
    test_slots_layout_conflict_empty_slots_ok,
    r#"
class A:
    __slots__ = ("x",)

class B:
    __slots__ = ()

# One base has non-empty slots, the other has empty slots - this is fine.
class C(A, B): ...
"#,
);

testcase!(
    test_slots_layout_conflict_both_empty_ok,
    r#"
class A:
    __slots__ = ()

class B:
    __slots__ = ()

# Both bases have empty slots - no conflict.
class C(A, B): ...
"#,
);

testcase!(
    test_slots_layout_conflict_three_bases,
    r#"
class A:
    __slots__ = ("x",)

class B:
    __slots__ = ("y",)

class C:
    __slots__ = ("z",)

class D(A, B, C): ...  # E: multiple base classes with non-empty `__slots__`
"#,
);

testcase!(
    test_slots_dunder_name_no_false_positive,
    r#"
from typing import assert_type

class Foo:
    __slots__ = ["__name__"]

    def __init__(self):
        self.__name__ = "foo_instance"

assert_type(Foo.__name__, str)
"#,
);

testcase!(
    bug = "Foo.b returns a member_descriptor at runtime, not an error",
    test_slots_instance_only_attr_not_visible_on_class,
    r#"
class Foo:
    __slots__ = ["b"]

    def __init__(self):
        self.b = "b"

print(Foo.b)  # E: Instance-only attribute `b` of class `Foo` is not visible on the class
"#,
);

testcase!(
    test_slots_metaclass_plain_attr_no_override,
    r#"
class Meta(type):
    x: int = 42

class Baz(metaclass=Meta):
    __slots__ = ["x"]

    def __init__(self):
        self.x = 10

# Meta.x is a plain attribute, not a data descriptor, so it does not
# override the slot descriptor in the MRO.
print(Baz.x)  # E: Instance-only attribute `x` of class `Baz` is not visible on the class
"#,
);

fn env_slots_cross_module() -> TestEnv {
    TestEnv::one(
        "m1",
        r#"
class A:
    __slots__ = ('ok',)

    def __init__(self, ok: int) -> None:
        self.ok = ok
"#,
    )
}

// Regression test: when class A (with __slots__) is imported from another
// module, and the current module also defines a class B with __slots__, the
// per-file ClassDefIndex could collide, causing pyrefly to check A's
// attribute writes against B's slot names instead of A's.
testcase!(
    test_slots_cross_module_no_false_positive,
    env_slots_cross_module(),
    r#"
from m1 import A

class B:
    __slots__ = ('x',)

    def __init__(self, x: int) -> None:
        self.x = x

def f(a: A) -> None:
    a.ok = 1   # ok
    a.bad = 2  # E: not declared in `__slots__`
"#,
);

testcase!(
    test_instance_attr_dunder_name_metaclass_fallback,
    r#"
from typing import assert_type

class Foo:
    def __init__(self):
        self.__name__ = "foo_instance"

assert_type(Foo.__name__, str)
"#,
);
