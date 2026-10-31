/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::sys_info::PythonVersion;

use crate::pydantic_testcase;
use crate::test::pydantic::util::pydantic_env;
use crate::test::util::TestEnv;
use crate::testcase;

pydantic_testcase!(
    test_model_validate_mutable_input_provenance,
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel):
    x: int = Field(ge=0)

payload = {"x": -1}
payload.update({"x": 1})
Model.model_validate(payload)

aliased = {"x": -1}
other = aliased
other["x"] = 1
Model.model_validate(aliased)

class Outer(BaseModel):
    items: list[Model]

items = [{"x": -1}]
items.clear()
Outer.model_validate({"items": items})

# Inline structure and scalar literal types remain checkable.
negative = -1
Model.model_validate({"x": negative})  # E: violates Pydantic `ge` constraint
Outer.model_validate({"items": [{"x": -1}]})  # E: violates Pydantic `ge` constraint
"#,
);

pydantic_testcase!(
    test_model_validate_transforming_field_validators,
    r#"
from pydantic import BaseModel, Field, field_validator

class Before(BaseModel):
    x: int = Field(ge=0)

    @field_validator("x", mode="before")
    @classmethod
    def normalize(cls, value: int) -> int:
        return abs(value)

class Child(Before):
    pass

Before.model_validate({"x": -1})
Child.model_validate({"x": -1})

class Plain(BaseModel):
    x: int = Field(ge=0)

    @field_validator("x", mode="plain")
    @classmethod
    def normalize(cls, value: object) -> int:
        return 1

Plain.model_validate({"x": -1})

class Item(BaseModel):
    x: int = Field(ge=0)

class Wrapper(BaseModel):
    child: Item
    unchecked: int = Field(ge=0)

    @field_validator("child", mode="before")
    @classmethod
    def normalize_child(cls, value: object) -> Item:
        return Item(x=1)

Wrapper.model_validate({"child": {"x": -1}, "unchecked": 1})
Wrapper.model_validate({"child": {"x": 1}, "unchecked": -1})  # E: violates Pydantic `ge` constraint
"#,
);

pydantic_testcase!(
    test_model_validate_additional_input_validators,
    r#"
from typing import Annotated
from pydantic import BaseModel, BeforeValidator, Field, PlainValidator, field_validator

def positive(value: int) -> int:
    return abs(value)

class Wildcard(BaseModel):
    x: int = Field(ge=0)

    @field_validator("*", mode="before")
    @classmethod
    def normalize(cls, value: int) -> int:
        return abs(value)

class WildcardChild(Wildcard):
    pass

Wildcard.model_validate({"x": -1})
WildcardChild.model_validate({"x": -1})

class Assigned(BaseModel):
    x: Annotated[int, BeforeValidator(positive)] = Field(ge=0)

class AnnotationOnly(BaseModel):
    x: Annotated[int, Field(ge=0), BeforeValidator(positive)]

class Plain(BaseModel):
    x: Annotated[int, Field(ge=0), PlainValidator(positive)]

Assigned.model_validate({"x": -1})
AnnotationOnly.model_validate({"x": -1})
Plain.model_validate({"x": -1})
"#,
);

pydantic_testcase!(
    bug = "Unsupported validation aliases are confused with absent aliases",
    test_model_validate_unsupported_validation_aliases,
    r#"
from typing import Annotated
from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field

class Choice(BaseModel):
    x: int = Field(ge=0, validation_alias=AliasChoices("y", "x"))

Choice.model_validate({"y": 1, "x": -1})  # E: violates Pydantic `ge` constraint
Choice.model_validate({"y": 1, "x": -1}, by_alias=False, by_name=True)  # E: violates Pydantic `ge` constraint

class Fallback(BaseModel):
    x: int = Field(ge=0, alias="z", validation_alias=AliasChoices("y", "z"))

Fallback.model_validate({"y": 1, "z": -1})  # E: violates Pydantic `ge` constraint

class Path(BaseModel):
    model_config = ConfigDict(validate_by_name=True)
    x: int = Field(ge=0, validation_alias=AliasPath("values", 0))

Path.model_validate({"values": [1], "x": -1})  # E: violates Pydantic `ge` constraint

class AnnotatedChoice(BaseModel):
    x: Annotated[int, Field(ge=0, validation_alias=AliasChoices("y", "x"))]

AnnotatedChoice.model_validate({"y": 1, "x": -1})  # E: violates Pydantic `ge` constraint

class StringAlias(BaseModel):
    x: int = Field(ge=0, validation_alias="y")

StringAlias.model_validate({"y": -1})  # E: violates Pydantic `ge` constraint
"#,
);

pydantic_testcase!(
    test_field_conflicting_defaults_uses_overload_error,
    r#"
from pydantic import BaseModel, Field, PrivateAttr

class Model(BaseModel):
    value: int = Field(default=1, default_factory=int)  # E: No matching overload found for function `pydantic.fields.Field`
    _private: int = PrivateAttr(default=1, default_factory=int)  # E: No matching overload found for function `pydantic.fields.PrivateAttr`
"#,
);

// See https://github.com/facebook/pyrefly/issues/4569. Not pydantic-specific — `Field`
// just happens to be a TypeVar-returning generic function, the shape that triggers the
// underlying bug (see test_call_hint_with_wide_literal_union in test::contextual for the
// general-purpose reproduction and fix).
pydantic_testcase!(
    test_field_default_with_wide_literal_type,
    r#"
from typing import Literal

from pydantic import BaseModel, Field

T4 = Literal["a", "b", "c", "d"]
T5 = Literal["a", "b", "c", "d", "e"]

class Works(BaseModel):
    value: T4 = Field("a")

class AlsoWorks(BaseModel):
    value: T5 = Field("a")
"#,
);

pydantic_testcase!(
    test_field_right_type,
    r#"
from pydantic import BaseModel, Field
class Model(BaseModel):
   x: int = Field(gt=0, lt=10)

Model(x=5)
Model(x=0)  # E: Argument value `Literal[0]` violates Pydantic `gt` constraint `Literal[0]` for field `x`
Model(x=15)  # E: Argument value `Literal[15]` violates Pydantic `lt` constraint `Literal[10]` for field `x`
"#,
);

pydantic_testcase!(
    test_field_range_ge_le,
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel):
    x: int = Field(ge=0, le=10)

Model(x=0)
Model(x=10)
Model(x=-1)  # E: Argument value `Literal[-1]` violates Pydantic `ge` constraint `Literal[0]` for field `x`
Model(x=11)  # E: Argument value `Literal[11]` violates Pydantic `le` constraint `Literal[10]` for field `x`
"#,
);

pydantic_testcase!(
    test_field_range_positional,
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel):
    x: int = Field(gt=0, kw_only=False)
    y: int = Field(lt=3, kw_only=False)

Model(1, 2)
Model(0, 2)  # E: Argument value `Literal[0]` violates Pydantic `gt` constraint `Literal[0]` for field `x`
Model(1, 3)  # E: Argument value `Literal[3]` violates Pydantic `lt` constraint `Literal[3]` for field `y`
"#,
);

pydantic_testcase!(
    test_field_range_kw_only,
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel):
    x: int = Field(ge=1, kw_only=True)

Model(x=1)
Model(x=0)  # E: Argument value `Literal[0]` violates Pydantic `ge` constraint `Literal[1]` for field `x`
"#,
);

pydantic_testcase!(
    test_field_range_alias,
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel, validate_by_name=True, validate_by_alias=True):
    x: int = Field(gt=0, validation_alias="y")

Model(x=0)  # E: Argument value `Literal[0]` violates Pydantic `gt` constraint `Literal[0]` for field `x`
Model(y=0)  # E: Argument value `Literal[0]` violates Pydantic `gt` constraint `Literal[0]` for field `x`
"#,
);

pydantic_testcase!(
    test_field_range_alias_only,
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel, validate_by_name=False, validate_by_alias=True):
    x: int = Field(gt=0, validation_alias="y")

Model(y=0)  # E: Argument value `Literal[0]` violates Pydantic `gt` constraint `Literal[0]` for field `x`
Model(x=0)  # E: Missing argument `y`
"#,
);

pydantic_testcase!(
    test_field_range_multiple,
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel):
    x: int = Field(gt=1, kw_only=False)
    y: int = Field(lt=0, kw_only=False)

Model(2, -1)
Model(y=-1, x=2)

Model(2, 0)  # E: violates Pydantic `lt` constraint `Literal[0]` for field `y`
Model(y=-1, x=1)  # E: violates Pydantic `gt` constraint `Literal[1]` for field `x`
    "#,
);

pydantic_testcase!(
    test_field_wrong_type,
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel):
    x: int = Field(gt="A", lt="B") # E:  Pydantic `gt` value has type `Literal['A']`, which is not assignable to field type `int` # E: Pydantic `lt` value has type `Literal['B']`, which is not assignable to field type `int`

Model(x=5)
"#,
);

pydantic_testcase!(
    test_field_ge,
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel):
    x: int = Field(ge="B") # E: Pydantic `ge` value has type `Literal['B']`, which is not assignable to field type `int`

Model(x=5)
"#,
);

pydantic_testcase!(
    test_field_range_decimal,
    r#"
from decimal import Decimal
from typing import Annotated
from pydantic import BaseModel, Field

class DecimalField(BaseModel):
    a: Decimal = Field(ge=0)
    b: Decimal = Field(gt=0)
    c: Decimal = Field(le=100)
    d: Decimal = Field(lt=100)
    e: Decimal | None = Field(default=None, ge=0)
    f: Decimal = Field(ge="bad")  # E: Pydantic `ge` value has type `Literal['bad']`, which is not assignable to field type `Decimal`
    g: Annotated[Decimal, Field(ge=0)]
    h: Annotated[Decimal, Field(ge="bad")]  # E: Pydantic `ge` value has type `Literal['bad']`, which is not assignable to field type `Decimal`
    i: Decimal = Field(ge=True)  # E: Pydantic `ge` value has type `Literal[True]`, which is not assignable to field type `Decimal`
    j: Annotated[Decimal, Field(ge=False)]  # E: Pydantic `ge` value has type `Literal[False]`, which is not assignable to field type `Decimal`
"#,
);

pydantic_testcase!(
    test_field_range_decimal_numeric_types,
    r#"
from decimal import Decimal
from enum import IntEnum
from typing import Annotated, Literal, NewType
from pydantic import BaseModel, Field

class IntBound(int):
    pass

class FloatBound(float):
    pass

class Bound(IntEnum):
    ZERO = 0

NominalBound = NewType("NominalBound", int)

def numeric_bounds(
    literal: Literal[0],
    numeric: int | float,
    mixed: int | Decimal,
    integer: IntBound,
    floating: FloatBound,
    enum: Bound,
    nominal: NominalBound,
) -> None:
    class Model(BaseModel):
        a: Decimal = Field(ge=literal)
        b: Decimal = Field(gt=numeric)
        c: Decimal = Field(le=mixed)
        d: Decimal = Field(lt=integer)
        e: Decimal = Field(ge=floating)
        f: Decimal = Field(ge=enum)
        g: Decimal = Field(ge=nominal)
        h: Decimal | None = Field(default=None, ge=literal)
        i: Annotated[Decimal, Field(ge=literal)]
        j: Annotated[Decimal, "metadata"] | None = Field(default=None, ge=literal)

def constrained[T: (int, Decimal)](bound: T) -> None:
    class Model(BaseModel):
        a: Decimal = Field(ge=bound)

def bounded[T: float](bound: T) -> None:
    class Model(BaseModel):
        a: Decimal = Field(ge=bound)

type DecimalAnnotated = Annotated[Decimal, "metadata"]

class ExistingForms(BaseModel):
    a: Decimal = Field(ge=0.5)
    b: Decimal = Field(gt=Decimal("0"))
    c: DecimalAnnotated | None = Field(default=None, ge=0)
"#,
);

pydantic_testcase!(
    test_field_range_decimal_invalid_types,
    r#"
from decimal import Decimal
from pydantic import BaseModel, Field

def invalid_bounds(
    boolean: bool,
    maybe_bool: bool | Decimal,
    maybe_bytes: int | bytes,
    maybe_complex: int | complex,
) -> None:
    class Model(BaseModel):
        a: Decimal = Field(ge=boolean)  # E: Pydantic `ge` value has type `bool`, which is not assignable to field type `Decimal`
        b: Decimal = Field(ge=maybe_bool)  # E: Pydantic `ge` value has type `Decimal | bool`, which is not assignable to field type `Decimal`
        c: Decimal = Field(ge=maybe_bytes)  # E: Pydantic `ge` value has type `bytes | int`, which is not assignable to field type `Decimal`
        d: Decimal = Field(ge=maybe_complex)  # E: Pydantic `ge` value has type `complex | int`, which is not assignable to field type `Decimal`

def boolean_bound[T: bool](bound: T) -> None:
    class Model(BaseModel):
        a: Decimal = Field(ge=bound)  # E: Pydantic `ge` value has type `T`, which is not assignable to field type `Decimal`

def boolean_constraint[T: (bool, Decimal)](bound: T) -> None:
    class Model(BaseModel):
        a: Decimal = Field(ge=bound)  # E: Pydantic `ge` value has type `T`, which is not assignable to field type `Decimal`
"#,
);

pydantic_testcase!(
    bug = "Decimal range bounds still reject runtime-coercible numeric strings",
    test_field_range_decimal_numeric_strings,
    r#"
from decimal import Decimal
from typing import Annotated
from pydantic import BaseModel, Field

class Model(BaseModel):
    a: Decimal = Field(ge="0")  # E: Pydantic `ge` value has type `Literal['0']`, which is not assignable to field type `Decimal`
    b: Annotated[Decimal, Field(ge="0")]  # E: Pydantic `ge` value has type `Literal['0']`, which is not assignable to field type `Decimal`
"#,
);

pydantic_testcase!(
    test_field_optional,
    r#"
from pydantic import BaseModel, Field

class Example(BaseModel):
    id: str
    attribute_1: str = Field(..., description="A required attribute")
    optional_attribute1: str | None = Field(None, description="An optional attribute")
    optional_attribute2: int = Field(0, description="Another optional attribute")

Example(id="123", attribute_1="value1")
Example(id="123")  # E: Missing argument `attribute_1`
"#,
);

pydantic_testcase!(
    test_model_construct,
    r#"
from typing import assert_type

from pydantic import BaseModel, Field

class Model(BaseModel):
    x: int
    y: str = "default"
    z: bool = Field(alias="aliased")

model = Model.model_construct(x=1, z=True)
assert_type(model, Model)
Model.model_construct(x="1", z=True)  # E: Argument `Literal['1']` is not assignable to parameter `x`
Model.model_construct(x=1, z=0)  # E: Argument `Literal[0]` is not assignable to parameter `z`
Model.model_construct(x=1, aliased=True)  # E: Missing argument `z`
Model.model_construct(z=True)  # E: Missing argument `x`
Model.model_construct(x=1, z=True, _fields_set={"x"})
Model.model_construct(x=1, z=True, _fields_set={1})  # E: `set[int]` is not assignable to parameter `_fields_set`

# Normal construction performs validation and can accept coercible input.
Model(x="1", aliased=True)
"#,
);

pydantic_testcase!(
    test_model_construct_generic,
    r#"
from typing import assert_type

from pydantic import BaseModel

class Model[T](BaseModel):
    value: T

model = Model[int].model_construct(value=1)
assert_type(model, Model[int])
Model[int].model_construct(value="1")  # E: `Literal['1']` is not assignable to parameter `value`
"#,
);

pydantic_testcase!(
    test_model_construct_instance_binding,
    r#"
from typing import assert_type

from pydantic import BaseModel

class Model(BaseModel):
    value: int

model = Model(value=1)
assert_type(Model.model_construct(value=1), Model)
assert_type(model.model_construct(value=1), Model)
assert_type(model.model_construct({"value"}, value=1), Model)
construct = model.model_construct
assert_type(construct(value=1), Model)

class GenericModel[T](BaseModel):
    value: T

instance = GenericModel[int](value=1)
assert_type(instance.model_construct(value=1), GenericModel[int])

class ReceiverNames(BaseModel):
    cls: int
    self: str

assert_type(ReceiverNames.model_construct(cls=1, self="value"), ReceiverNames)
"#,
);

pydantic_testcase!(
    test_model_construct_inference,
    r#"
from typing import assert_type

from pydantic import BaseModel

class Model[T](BaseModel):
    value: T

assert_type(Model.model_construct(value=1), Model[int])
construct = Model.model_construct
assert_type(construct(value="s"), Model[str])

class Child(Model[int]):
    pass

assert_type(Child.model_construct(value=1), Child)
Child.model_construct(value="s")  # E: `Literal['s']` is not assignable to parameter `value`
assert_type(Child(value=1).model_construct(value=1), Child)
"#,
);

pydantic_testcase!(
    test_model_construct_subclass_override,
    r#"
from typing import Any, Self, assert_type

from pydantic import BaseModel

class Standard(BaseModel):
    value: int

class OverridingChild(Standard):
    @classmethod
    def model_construct(cls, _fields_set: set[str] | None = None, **values: Any) -> Self:
        return super().model_construct(_fields_set, **values)

class Grandchild(OverridingChild):
    pass

assert_type(OverridingChild.model_construct(), OverridingChild)
assert_type(OverridingChild(value=1).model_construct(), OverridingChild)
assert_type(Grandchild.model_construct(), Grandchild)
"#,
);

pydantic_testcase!(
    test_model_construct_instance_self_return,
    r#"
from typing import Self, assert_type

from pydantic import BaseModel

class Model(BaseModel):
    value: int

    def clone(self) -> Self:
        return self.model_construct(value=self.value)

class Child(Model):
    pass

assert_type(Child(value=1).clone(), Child)
"#,
);

pydantic_testcase!(
    test_model_construct_self_return,
    r#"
from typing import Self, assert_type

from pydantic import BaseModel

class Model(BaseModel):
    value: int

    @classmethod
    def make(cls, value: int) -> Self:
        return cls.model_construct(value=value)

class Child(Model):
    label: str = "default"

assert_type(Child.make(1), Child)

def make[T: Model](cls: type[T], value: int) -> T:
    return cls.model_construct(value=value)

assert_type(make(Child, 1), Child)

class GenericModel[T](BaseModel):
    value: T

    @classmethod
    def make(cls, value: T) -> Self:
        return cls.model_construct(value=value)

class GenericChild(GenericModel[int]):
    pass

assert_type(GenericModel[int].make(1), GenericModel[int])
assert_type(GenericChild.make(1), GenericChild)
"#,
);

pydantic_testcase!(
    test_model_construct_inherited_override,
    r#"
from typing import Any, Self, assert_type

from pydantic import BaseModel

class Model(BaseModel):
    value: int

    @classmethod
    def model_construct(cls, _fields_set: set[str] | None = None, **values: Any) -> Self:
        values.setdefault("value", 1)
        values.setdefault("other", "default")
        return super().model_construct(_fields_set=_fields_set, **values)

class Child(Model):
    other: str

class Grandchild(Child):
    pass

assert_type(Model.model_construct(), Model)
assert_type(Child.model_construct(), Child)
assert_type(Grandchild.model_construct(), Grandchild)
Child.model_construct(value="accepted by override")
Grandchild.model_construct(other=1)

class Standard(BaseModel):
    value: int

class StandardChild(Standard):
    other: str

StandardChild.model_construct(value=1, other="valid")
StandardChild.model_construct(value=1)  # E: Missing argument `other`
"#,
);

pydantic_testcase!(
    test_model_construct_annotated_defaults,
    r#"
from typing import Annotated

from pydantic import BaseModel, Field

class Model(BaseModel):
    value: Annotated[int, Field(default=1)] = Field()
    factory: Annotated[int, Field(default_factory=int)] = Field()
    assigned: Annotated[int, Field()] = Field(default=2)
    required: Annotated[int, Field()] = Field()

Model(required=3)
Model.model_construct(required=3)
Model.model_construct()  # E: Missing argument `required`
Model.model_construct(required=3, value="1")  # E: `Literal['1']` is not assignable to parameter `value`
"#,
);

pydantic_testcase!(
    test_model_construct_instance_fields_set_keyword,
    r#"
from typing import assert_type
from pydantic import BaseModel

class Model(BaseModel):
    value: int

model = Model(value=1)
assert_type(model.model_construct(value=2, _fields_set={"value"}), Model)
"#,
);

pydantic_testcase!(
    test_model_construct_multiple_inheritance_override,
    r#"
from typing import Any, Self, assert_type
from pydantic import BaseModel

class Standard(BaseModel):
    value: int

class Override(BaseModel):
    @classmethod
    def model_construct(cls, _fields_set: set[str] | None = None, **values: Any) -> Self:
        values.setdefault("value", 1)
        return super().model_construct(_fields_set=_fields_set, **values)

class Child(Standard, Override):
    pass

assert_type(Child.model_construct(), Child)
instance = Child(value=1)
assert_type(instance.model_construct(), Child)
construct = instance.model_construct
assert_type(construct(), Child)

class LocalOverride(Child):
    @classmethod
    def model_construct(cls, _fields_set: set[str] | None = None, **values: Any) -> Self:
        return super().model_construct(_fields_set, **values)

assert_type(LocalOverride.model_construct(), LocalOverride)
"#,
);

pydantic_testcase!(
    test_model_construct_multiple_inheritance_generic_override,
    r#"
from typing import Any, Self, assert_type
from pydantic import BaseModel

class Standard(BaseModel):
    value: int

class Override[T](BaseModel):
    @classmethod
    def model_construct(cls, _fields_set: set[str] | None = None, *, marker: T | None = None, **values: Any) -> Self:
        return super().model_construct(_fields_set, **values)

class Child(Standard, Override[int]):
    pass

class Grandchild(Child):
    pass

assert_type(Child.model_construct(marker=1), Child)
assert_type(Grandchild.model_construct(marker=1), Grandchild)
Child.model_construct(value=1, marker="invalid")  # E: `Literal['invalid']` is not assignable to parameter `marker`
Grandchild.model_construct(value=1, marker="invalid")  # E: `Literal['invalid']` is not assignable to parameter `marker`
"#,
);

// Pydantic's intentionally permissive extras are explicit Any, not unknown input.
testcase!(
    test_model_construct_explicit_kwargs,
    pydantic_env().enable_partially_unknown_type_error(),
    r#"
from pydantic import BaseModel

class Model(BaseModel):
    value: int

Model.model_construct(value=1)
Model.model_construct(value=1, extra="accepted")
"#,
);

pydantic_testcase!(
    test_frozen_field_override_covariant,
    r#"
from pydantic import BaseModel, Field

class Foo(BaseModel):
    id: int | None = Field(frozen=True)

class Bar(Foo):
    id: int = Field(frozen=True)
"#,
);

pydantic_testcase!(
    test_model_validate_nested_field_range,
    r#"
from pydantic import BaseModel, Field

class Item(BaseModel):
    sku: str
    quantity: int = Field(ge=1)

class Inventory(BaseModel):
    items: list[Item]

Inventory.model_validate({"items": [{"sku": "A-100", "quantity": 5}]})
Inventory.model_validate({"items": [{"sku": "B-200", "quantity": -3}]})  # E: Argument value `Literal[-3]` violates Pydantic `ge` constraint `Literal[1]` for field `quantity`

# Named mutable input is intentionally not inspected structurally.
bad = {"items": [{"sku": "B-200", "quantity": -3}]}
Inventory.model_validate(bad)
"#,
);

pydantic_testcase!(
    test_model_validate_union_alternatives,
    r#"
from pydantic import BaseModel, Field

class Positive(BaseModel):
    x: int = Field(ge=0)

class Negative(BaseModel):
    x: int = Field(le=-1)

class Either(BaseModel):
    item: Positive | Negative
    items: list[Positive | Negative]

Either.model_validate({"item": {"x": -1}, "items": [{"x": -1}, {"x": 1}]})
Either.model_validate({"item": {"x": 1}, "items": []})
"#,
);

pydantic_testcase!(
    test_model_validate_custom_input,
    r#"
from typing import Self, assert_type
from pydantic import BaseModel, Field

class Parent(BaseModel):
    x: int = Field(ge=0)

    @classmethod
    def model_validate(
        cls, obj: object = None, *, strict=None, from_attributes=None,
        context=None, by_alias=None, by_name=None,
    ) -> Self:
        return cls(x=1)

class Child(Parent):
    pass

class Builtin(BaseModel):
    pass

class Multiple(Builtin, Parent):
    pass

assert_type(Child.model_validate(), Child)
assert_type(Multiple.model_validate(), Multiple)
Parent.model_validate({"x": -1})
Child.model_validate({"x": -1})
"#,
);

pydantic_testcase!(
    test_model_validate_alias_selection,
    r#"
from pydantic import BaseModel, ConfigDict, Field

class Both(BaseModel):
    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)
    x: int = Field(ge=0, alias="y")

Both.model_validate({"x": -1, "y": 1})
Both.model_validate({"x": 1, "y": -1})  # E: violates Pydantic `ge` constraint
Both.model_validate({"x": 1, "y": -1}, by_name=True, by_alias=False)
Both.model_validate({"x": -1, "y": 1}, by_name=True, by_alias=False)  # E: violates Pydantic `ge` constraint

class AliasOnly(BaseModel):
    x: int = Field(ge=0, validation_alias="y")

AliasOnly.model_validate({"x": 1, "y": -1}, by_name=True, by_alias=False)
AliasOnly.model_validate({"x": -1, "y": 1}, by_name=True, by_alias=False)  # E: violates Pydantic `ge` constraint

class NameOnly(BaseModel):
    model_config = ConfigDict(validate_by_name=True, validate_by_alias=False)
    x: int = Field(ge=0, alias="y")

NameOnly.model_validate({"x": -1, "y": 1}, by_alias=True, by_name=False)
NameOnly.model_validate({"x": 1, "y": -1})

def dynamic(flag: bool) -> None:
    Both.model_validate({"x": -1, "y": 1}, by_alias=flag, by_name=True)
    Both.model_validate({"x": -1, "y": 1}, **{"by_alias": flag, "by_name": True})

class Outer(BaseModel):
    child: Both

Outer.model_validate({"child": {"x": 1, "y": -1}}, by_name=True, by_alias=False)
"#,
);

pydantic_testcase!(
    test_model_validate_dictionary_spreads,
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel):
    x: int = Field(ge=0)

patch: dict[str, int] = {"x": 1}
Model.model_validate({"x": -1, **patch})
Model.model_validate({"x": -1, **{"x": 1}})
Model.model_validate({**patch, "x": -1})  # E: violates Pydantic `ge` constraint
Model.model_validate({"x": 1, **{"x": -1}})  # E: violates Pydantic `ge` constraint

def dynamic_key(key: str) -> None:
    Model.model_validate({"x": -1, key: 1})

class Aliased(BaseModel):
    model_config = {"validate_by_name": True}
    x: int = Field(ge=0, alias="y")

Aliased.model_validate({**patch, "x": -1})
Aliased.model_validate({**patch, "x": -1, "y": 1})
Aliased.model_validate({**patch, "y": -1})  # E: violates Pydantic `ge` constraint
"#,
);

/// Supply a library signature with a version-specific keyword.
fn model_validate_library_env() -> TestEnv {
    TestEnv::one(
        "pydantic.main",
        r#"
from typing import Any, Literal, Self, dataclass_transform

@dataclass_transform()
class BaseModel:
    @classmethod
    def model_validate(
        cls, obj: object, *, extra: Literal["allow", "ignore", "forbid"] | None = None,
        context: Any = None,
    ) -> Self: ...
"#,
    )
}

testcase!(
    test_model_validate_library_signature,
    model_validate_library_env(),
    r#"
from typing import assert_type
from pydantic.main import BaseModel

class Model(BaseModel):
    x: int

assert_type(Model.model_validate({"x": 1}, extra="forbid"), Model)
Model.model_validate({"x": 1}, extra="invalid")  # E: is not assignable to parameter `extra`
"#,
);

pydantic_testcase!(
    test_model_validate_shared_descendants,
    r#"
from pydantic import BaseModel

class M0(BaseModel):
    x: int

class M1(BaseModel):
    left: M0
    right: M0

class M2(BaseModel):
    left: M1
    right: M1

class M3(BaseModel):
    left: M2
    right: M2

class M4(BaseModel):
    left: M3
    right: M3

class M5(BaseModel):
    left: M4
    right: M4

class M6(BaseModel):
    left: M5
    right: M5

class M7(BaseModel):
    left: M6
    right: M6

class M8(BaseModel):
    left: M7
    right: M7

class M9(BaseModel):
    left: M8
    right: M8

class M10(BaseModel):
    left: M9
    right: M9

class M11(BaseModel):
    left: M10
    right: M10

class M12(BaseModel):
    left: M11
    right: M11

class M13(BaseModel):
    left: M12
    right: M12

class M14(BaseModel):
    left: M13
    right: M13
"#,
);

pydantic_testcase!(
    test_model_validate_from_attributes_accepts_object,
    r#"
from pydantic import BaseModel

class Model(BaseModel):
    x: int

def validate(obj: object) -> Model:
    return Model.model_validate(obj, from_attributes=True)
"#,
);

pydantic_testcase!(
    test_custom_model_validate_is_not_overwritten,
    r#"
from typing import Self
from pydantic import BaseModel

class Model(BaseModel):
    x: int

    @classmethod
    def model_validate(
        cls,
        obj: object,
        *,
        strict: bool | None = None,
        from_attributes: bool | None = None,
        context: object | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        return cls(x=0)

def validate(obj: object) -> int:
    return Model.model_validate(obj).x
"#,
);

pydantic_testcase!(
    test_super_model_validate_returns_self,
    r#"
from typing import Self
from pydantic import BaseModel

class Parent(BaseModel):
    x: int

class Child(Parent):
    y: int

    @classmethod
    def model_validate(
        cls,
        obj: object,
        *,
        strict: bool | None = None,
        from_attributes: bool | None = None,
        context: object | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        return super().model_validate(
            obj,
            strict=strict,
            from_attributes=from_attributes,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

def validate(obj: object) -> int:
    return Child.model_validate(obj, from_attributes=True).y
"#,
);

pydantic_testcase!(
    bug = "consider erroring on invalid5 and invalid6",
    test_discriminated_unions,
    r#"
from typing import Literal, Union
from pydantic import BaseModel, Field

class A(BaseModel):
    kind: Literal["a"]
    val: int

class B(BaseModel):
    kind: Literal["b"]
    msg: str

class Wrapper(BaseModel):
    item: Union[A, B] = Field(discriminator="kind")

valid1 = Wrapper(item=A(kind="a", val=123))
valid2 = Wrapper(item=B(kind="b", msg="Bob"))

invalid1 = Wrapper(item=A(kind="a")) # E: Missing argument `val` in function `A.__init__`
invalid2 = Wrapper(item=B(kind="b", val=123)) # E: Missing argument `msg` in function `B.__init__`

valid3 = Wrapper.model_validate({"item": A(kind="a", val=123)})
valid4 = Wrapper.model_validate({"item": B(kind="b", msg="Bob")})

invalid3 = Wrapper.model_validate({"item": A(kind="a")}) # E: Missing argument `val` in function `A.__init__`
invalid4 =  Wrapper.model_validate({"item": B(kind="b", val=123)}) # E: Missing argument `msg` in function `B.__init__`

valid5 = Wrapper.model_validate({"item": {"kind": "a", "val": 123}})
valid6 = Wrapper.model_validate({"item": {"kind": "b", "msg": "Bob"}})

invalid5 = Wrapper.model_validate({"item": {"kind": "a"}})
invalid6 = Wrapper.model_validate({"item": {"kind": "b", "name": 123}})

    "#,
);

pydantic_testcase!(
    test_discriminated_unions_annotated,
    r#"
from typing import Annotated, Literal
from pydantic import BaseModel, Field

class A(BaseModel):
    input_type: Literal["A"] = "A"
class B(BaseModel):
    input_type: Literal["B"] = "B"

T = Annotated[A | B, Field(discriminator="input_type")]

def foo(ts: list[T]) -> list[A]:
    return [
        t for t in ts
        if t.input_type == "A"
    ]

print(foo([A(), B()]))
    "#,
);

pydantic_testcase!(
    test_required_field,
    r#"
from pydantic import BaseModel
class A(BaseModel, validate_by_name=True, validate_by_alias=True):
    x: int
A()  # E: Missing argument `x`
    "#,
);

pydantic_testcase!(
    test_private_attributes_not_init_fields,
    r#"
from pydantic import BaseModel

class Foo(BaseModel):
    a: int
    _a: int

    def initialize(self):
        self._a = 1

Foo(a=1)
Foo()  # E: Missing argument `a`
    "#,
);

pydantic_testcase!(
    test_inherited_fields_are_init_fields,
    r#"
from pydantic import BaseModel

class SuperBase(BaseModel, extra="forbid"):
    x: int

class Derived(SuperBase, extra="forbid"):
    z: int

Derived(x=1, z=1)
Derived(z=1)  # E: Missing argument `x`
    "#,
);

pydantic_testcase!(
    test_pydantic_dataclass_underscore_field_is_init_param,
    r#"
from pydantic.dataclasses import dataclass

@dataclass
class Bar:
    _name: str

Bar(_name="hello")
    "#,
);

pydantic_testcase!(
    test_field_default_gt_violation,
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel):
    value: int = Field(0, gt=0)  # E: Default value `Literal[0]` violates Pydantic `gt` constraint `Literal[0]` for field `value`

class Model2(BaseModel):
    value: int = Field(default=0, gt=0)  # E: violates Pydantic `gt` constraint

class Model3(BaseModel):
    value: int = Field(default_factory=lambda: 0, gt=0)  # E: violates Pydantic `gt` constraint
    "#,
);

pydantic_testcase!(
    test_field_default_gt_ok,
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel):
    value: int = Field(1, gt=0)

class Model2(BaseModel):
    value: int = Field(default=1, gt=0)

class Model3(BaseModel):
    value: int = Field(default_factory=lambda: 1, gt=0)

def f() -> int: ...
class Model4(BaseModel):
    value: int = Field(f(), gt=0)
    "#,
);

fn pydantic_env_3_10() -> TestEnv {
    let env = pydantic_env();
    env.with_version(PythonVersion::new(3, 10, 0))
}

testcase!(
    test_default_keywords_3_9,
    pydantic_env().with_version(PythonVersion::new(3, 9, 0)),
    r#"
from pydantic import BaseModel, Field
class A(BaseModel):
    x: int = Field(default='oops')  # E: `str` is not assignable to `int`
    y: int = Field(...)
    "#,
);

testcase!(
    test_model_3_10,
    pydantic_env_3_10(),
    r#"
from pydantic import BaseModel
class A(BaseModel, strict=True):
    x: int
A(x='')  # E: `Literal['']` is not assignable to parameter `x` with type `int`
    "#,
);

pydantic_testcase!(
    test_default_keywords,
    r#"
from pydantic import BaseModel, Field
class A(BaseModel):
    x: int = Field(default='oops')  # E: `str` is not assignable to `int`
class B(BaseModel):
    x: int = Field(default_factory=lambda: 'oops')  # E: `str` is not assignable to `int`
    "#,
);

pydantic_testcase!(
    test_annotated_field_with_defaults,
    r#"
from typing import Annotated
from pydantic import BaseModel, Field

class House(BaseModel):
    street: str
    city: str
    zipcode: str
    notes: Annotated[str, Field(default="")]
    extra_fields: Annotated[dict, Field(default_factory=dict)]
    something: dict = Field(default_factory=dict)

house = House(
    street="House Street",
    city="House City",
    zipcode="House Zipcode",
)
    "#,
);

pydantic_testcase!(
    test_model_fields_with_bounded_typevar,
    r#"
from pydantic import BaseModel

class MyModel(BaseModel):
    field: int

class A[T: BaseModel]:
    def __init__(self, model_type: type[T]) -> None:
        self._model_type = model_type

    def print_model_fields(self) -> None:
        for field in self._model_type.model_fields:
            print(field)

a = A(MyModel)
a.print_model_fields()
    "#,
);

pydantic_testcase!(
    test_field_validator,
    r#"
from pydantic import BaseModel
from pydantic import field_validator

class MyBaseModel(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v:
            raise ValueError("Name cannot be empty")
        return v
    "#,
);

pydantic_testcase!(
    test_field_validator_mode_before,
    r#"
from pydantic import BaseModel, field_validator

class C(BaseModel):
    x: str
    @field_validator('x', mode='before')
    @classmethod
    def validate_x(cls, x):
        return str(x)

C(x=0)
C(x="hello")
    "#,
);

pydantic_testcase!(
    test_field_validator_mode_before_inherited,
    r#"
from pydantic import BaseModel, field_validator

class Parent(BaseModel):
    x: str
    @field_validator('x', mode='before')
    @classmethod
    def validate_x(cls, x):
        return str(x)

class Child(Parent):
    pass

Child(x=0)
Child(x="hello")
    "#,
);

pydantic_testcase!(
    test_field_validator_mode_plain,
    r#"
from pydantic import BaseModel, field_validator

class P(BaseModel):
    x: int
    @field_validator('x', mode='plain')
    @classmethod
    def validate_x(cls, x) -> int:
        return int(x)

P(x="99")
P(x=42)
    "#,
);

pydantic_testcase!(
    test_field_without_annotation,
    r#"
from pydantic import BaseModel, Field
class Model(BaseModel):
    x = Field(default=1)  # E: `x` is a dataclass field but has no type annotation
"#,
);

// Pydantic `_`-prefixed private fields are not storage; the inherited
// `BaseModel` representative wins — a regression to `Left` would name it
// in the diagnostic.
pydantic_testcase!(
    test_pydantic_dataclass_slots_private_fields_do_not_promote,
    r#"
from pydantic import BaseModel

class Left(BaseModel, slots=True):
    _x: int

class Right:
    __slots__ = ("y",)

class Conflict(Left, Right): ...  # E: inherits from incompatible disjoint bases `BaseModel`, `Right`
"#,
);

// pydantic field named 'self' must not collide with the synthesized '__init__''s implicit 'self' param.
// same as the stdlib dataclass fix.
pydantic_testcase!(
    test_pydantic_field_named_self,
    r#"
from pydantic import BaseModel
from typing import assert_type

class Model(BaseModel):
    self: str

m = Model(self="test")
assert_type(m.self, str)
"#,
);

// pydantic `BaseModel` does not honor `Field(init=False)`,
// unlike stdlib/attrs dataclasses and unlike pydantic.dataclass
// The field stays a real keyword parameter of the synthesized `__init__`,
// and it is type-checked.
pydantic_testcase!(
    test_pydantic_model_field_init_false_ignored,
    r#"
from typing import reveal_type
from pydantic import BaseModel, Field

class Model(BaseModel):
    x: int = Field(init=False, default=0)

reveal_type(Model.__init__)  # E: revealed type: (self: Model, *, x: LaxInt = ..., **Unknown) -> None
"#,
);

pydantic_testcase!(
    test_private_attr_is_not_frozen,
    r#"
from pydantic import BaseModel, ConfigDict

class Model(BaseModel):
    model_config = ConfigDict(frozen=True)
    _private: int = 0

model = Model()
# A private attribute is not frozen even when the model is.
model._private = 1
    "#,
);
