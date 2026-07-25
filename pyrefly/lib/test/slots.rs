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
    test_slots_manual_dict_rejects_undeclared,
    r#"
class C:
    __slots__ = {"x": "docstring for x"}

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
    test_slots_dict_literal_allows_dynamic,
    r#"
class C:
    __slots__ = {"x": "docstring for x", "__dict__": "instance dict"}

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

// Both list-literal and single-string-literal extractor shapes promote.
// https://github.com/facebook/pyrefly/issues/2917
testcase!(
    test_slots_multiple_inheritance_layout_conflict,
    r#"
class Left:
    __slots__ = ["a", "b"]

class Right:
    __slots__ = "c"

class Combined(Left, Right): ...  # E: incompatible disjoint bases
"#,
);

testcase!(
    test_slots_dict_layout_conflict,
    r#"
class Left:
    __slots__ = {"x": "docstring for x"}

class Right:
    __slots__ = ("y",)

class Combined(Left, Right): ...  # E: inherits from incompatible disjoint bases `Left`, `Right`
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

class Both(First, Second): ...  # E: incompatible disjoint bases
"#,
);

// Empty list `__slots__ = []` does not promote.
testcase!(
    test_slots_layout_conflict_empty_slots_ok,
    r#"
class A:
    __slots__ = ("x",)

class B:
    __slots__ = []

class C(A, B): ...
"#,
);

// Empty tuple `__slots__ = ()` does not promote.
testcase!(
    test_slots_layout_conflict_both_empty_ok,
    r#"
class A:
    __slots__ = ()

class B:
    __slots__ = ()

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

class D(A, B, C): ...  # E: incompatible disjoint bases
"#,
);

// `Combined(Child, Base)` is OK when `Child(Base)` already covers `Base`'s layout.
testcase!(
    test_slots_disjoint_base_subclass_and_ancestor_ok,
    r#"
class Base:
    __slots__ = ("x",)

class Child(Base):
    __slots__ = ("y",)

class Combined(Child, Base): ...
"#,
);

testcase!(
    test_slots_disjoint_base_propagation_through_unslotted,
    r#"
class Left:
    __slots__ = ("x",)

class LeftChild(Left):
    pass

class Right:
    __slots__ = ("y",)

class Bad(LeftChild, Right): ...  # E: incompatible disjoint bases
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

testcase!(
    test_slots_dataclass_slots_disjoint_base_conflicts,
    r#"
from dataclasses import dataclass

@dataclass(slots=True)
class Left:
    x: int

@dataclass(slots=True)
class Right:
    y: int

class Bad(Left, Right): ...  # E: inherits from incompatible disjoint bases `Left`, `Right`

class Explicit:
    __slots__ = ("y",)

class MixedBad(Left, Explicit): ...  # E: inherits from incompatible disjoint bases `Left`, `Explicit`
"#,
);

// Empty / pseudo-field-only dataclasses synthesize empty `__slots__`.
testcase!(
    test_slots_dataclass_empty_and_pseudo_slots_do_not_promote,
    r#"
from dataclasses import KW_ONLY, InitVar, dataclass
from typing import ClassVar

@dataclass(slots=True)
class Empty:
    pass

@dataclass(slots=True)
class OnlyClassVar:
    x: ClassVar[int]

@dataclass(slots=True)
class OnlyInitVar:
    x: InitVar[int]

@dataclass(slots=True)
class OnlyKwOnly:
    _: KW_ONLY

class Slotted:
    __slots__ = ("y",)

class EmptyOK(Empty, Slotted): ...
class ClassVarOK(OnlyClassVar, Slotted): ...
class InitVarOK(OnlyInitVar, Slotted): ...
class KwOnlyOK(OnlyKwOnly, Slotted): ...
"#,
);

// Subclasses without a new non-empty slot tuple propagate the inherited
// representative; `ChildWithLocalField` becomes its own.
testcase!(
    test_slots_dataclass_slots_propagate_through_subclasses,
    r#"
from dataclasses import InitVar, dataclass
from typing import ClassVar

@dataclass(slots=True)
class Left:
    x: int

class LeftChild(Left):
    pass

@dataclass(slots=True)
class DecoratedLeftChild(Left):
    pass

@dataclass(slots=True)
class ChildWithLocalField(Left):
    child: int

@dataclass(slots=True)
class ChildWithOnlyPseudoFields(Left):
    class_var: ClassVar[int]
    init_var: InitVar[int]

@dataclass(slots=True)
class Right:
    y: int

class Bad(LeftChild, Right): ...  # E: inherits from incompatible disjoint bases `Left`, `Right`
class DecoratedBad(DecoratedLeftChild, Right): ...  # E: inherits from incompatible disjoint bases `Left`, `Right`
class ChildBad(ChildWithLocalField, Right): ...  # E: inherits from incompatible disjoint bases `ChildWithLocalField`, `Right`
class PseudoChildBad(ChildWithOnlyPseudoFields, Right): ...  # E: inherits from incompatible disjoint bases `Left`, `Right`
"#,
);

testcase!(
    test_slots_empty_dataclass_slot_base_with_nonempty_dataclass_slot_base_ok,
    r#"
from dataclasses import dataclass

@dataclass(slots=True)
class Empty:
    pass

@dataclass(slots=True)
class NonEmpty:
    x: int

class OK(Empty, NonEmpty): ...
"#,
);

// All three `dataclass_transform` entry points (base, metaclass, decorator).
testcase!(
    test_slots_dataclass_transform_slots_disjoint_base_conflict,
    r#"
from typing import dataclass_transform

@dataclass_transform()
class ModelBase: ...

class BaseLeft(ModelBase, slots=True):
    x: int

@dataclass_transform()
class ModelMeta(type): ...

class MetaLeft(metaclass=ModelMeta, slots=True):
    x: int

@dataclass_transform()
def transform(**kwargs): ...

@transform(slots=True)
class DecoratedLeft:
    x: int

class Right:
    __slots__ = ("y",)

class BaseBad(BaseLeft, Right): ...  # E: inherits from incompatible disjoint bases `BaseLeft`, `Right`
class MetaBad(MetaLeft, Right): ...  # E: inherits from incompatible disjoint bases `MetaLeft`, `Right`
class DecoratedBad(DecoratedLeft, Right): ...  # E: inherits from incompatible disjoint bases `DecoratedLeft`, `Right`
"#,
);

// `fields(cls)` includes inherited dataclass fields, so an unslotted
// dataclass ancestor still materializes a non-empty `__slots__` in a
// slotted subclass.
testcase!(
    test_slots_dataclass_slots_promotes_via_inherited_fields,
    r#"
from dataclasses import dataclass

@dataclass
class UnslottedBase:
    x: int

@dataclass(slots=True)
class Promoted(UnslottedBase):
    pass

class Right:
    __slots__ = ("y",)

class Bad(Promoted, Right): ...  # E: inherits from incompatible disjoint bases `Promoted`, `Right`
"#,
);

// CPython dedups child slots against transitive ancestor slots, so the
// representative reported here must be `A`, not `C`.
testcase!(
    test_slots_dataclass_slots_dedup_through_unslotted_middle,
    r#"
from dataclasses import dataclass

@dataclass(slots=True)
class A:
    x: int

@dataclass
class B(A):
    pass

@dataclass(slots=True)
class C(B):
    pass

@dataclass(slots=True)
class Right:
    y: int

# The diagnostic must name `A`, not `C`: `C`'s generated `__slots__` is
# empty because `x` is already covered by `A`, so `C` inherits `A`'s
# representative through the MRO rather than introducing its own.
class Bad(C, Right): ...  # E: inherits from incompatible disjoint bases `A`, `Right`
"#,
);

// Inherited pseudo-fields must not count as storage for slot synthesis.
testcase!(
    test_slots_dataclass_slots_ignores_inherited_pseudo_fields,
    r#"
from dataclasses import dataclass
from typing import ClassVar

@dataclass
class PseudoOnlyBase:
    x: ClassVar[int] = 0

@dataclass(slots=True)
class Child(PseudoOnlyBase):
    pass

class Right:
    __slots__ = ("y",)

class OK(Child, Right): ...  # no conflict: Child's slots are empty
"#,
);

// A local pseudo-field annotation overrides the inherited dataclass entry,
// dropping it from `fields(cls)` and the synthesized slots.
testcase!(
    test_slots_dataclass_slots_local_pseudo_override_drops_inherited_storage,
    r#"
from dataclasses import InitVar, dataclass
from typing import ClassVar

@dataclass
class Base:
    x: int
    y: int

@dataclass(slots=True)
class Child(Base):
    x: ClassVar[int] = 0  # E: ClassVar `Child.x` overrides instance variable of the same name in parent class `Base`
    y: InitVar[int] = 0

class Right:
    __slots__ = ("z",)

class OK(Child, Right): ...  # no conflict: x, y were pseudo-overridden
"#,
);

// `@dataclass(slots=True)` + explicit `__slots__` errors at synthesis;
// the disjoint-base gate must also suppress promotion.
testcase!(
    test_slots_dataclass_slots_explicit_slots_conflict_does_not_promote,
    r#"
from dataclasses import dataclass

@dataclass(slots=True)
class C:  # E: Cannot specify both `slots=True` and `__slots__`
    __slots__ = ()
    x: int

class Right:
    __slots__ = ("y",)

class OK(C, Right): ...  # no conflict
"#,
);

// Like the prior test but with dynamic explicit `__slots__`: slot names are
// unknown, but the class-body `__slots__` presence still blocks synthesis.
testcase!(
    test_slots_dataclass_slots_dynamic_explicit_slots_does_not_promote,
    r#"
from dataclasses import dataclass
from typing import Sequence

def get_slots() -> Sequence[str]: ...

@dataclass(slots=True)
class C:  # E: Cannot specify both `slots=True` and `__slots__`
    __slots__ = get_slots()
    x: int

class Right:
    __slots__ = ("y",)

class OK(C, Right): ...  # no conflict
"#,
);
