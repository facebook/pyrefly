/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::pydantic_testcase;

pydantic_testcase!(
    bug = "Nested annotations and composed Field metadata are not interpreted",
    test_model_validate_nested_schema_uncertainty,
    r#"
from typing import Annotated
from pydantic import BaseModel, BeforeValidator, Field

class Item(BaseModel):
    x: int = Field(ge=0)

def normalize(value: object) -> Item:
    return Item(x=1)

class Outer(BaseModel):
    items: list[Annotated[Item, BeforeValidator(normalize)]]

Outer.model_validate({"items": [{"x": -1}]})  # E: violates Pydantic `ge` constraint

class Multiple(BaseModel):
    x: Annotated[int, Field(ge=0), Field(ge=-10)]

Multiple.model_validate({"x": -1})  # E: violates Pydantic `ge` constraint

class AssignedAlias(BaseModel):
    x: Annotated[int, Field(validation_alias="y")] = Field(ge=0)

AssignedAlias.model_validate({"x": -1, "y": 1})  # E: violates Pydantic `ge` constraint
"#,
);

pydantic_testcase!(
    bug = "Wrapping, model, aliased validators and custom init can transform input",
    test_model_validate_custom_validation_schema,
    r#"
from typing import Any
from pydantic import BaseModel, Field, field_validator as fv, model_validator

class Aliased(BaseModel):
    x: int = Field(ge=0)

    @fv("x", mode="before")
    @classmethod
    def normalize(cls, value: int) -> int:
        return abs(value)

Aliased.model_validate({"x": -1})  # E: violates Pydantic `ge` constraint

class Wrapped(BaseModel):
    x: int = Field(ge=0)

    @fv("x", mode="wrap")
    @classmethod
    def normalize(cls, value: int, handler: Any) -> int:
        return handler(abs(value))

Wrapped.model_validate({"x": -1})  # E: violates Pydantic `ge` constraint

class ModelBefore(BaseModel):
    x: int = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def normalize(cls, value: object) -> dict[str, int]:
        return {"x": 1}

class Child(ModelBefore):
    pass

ModelBefore.model_validate({"x": -1})  # E: violates Pydantic `ge` constraint
Child.model_validate({"x": -1})  # E: violates Pydantic `ge` constraint

class CustomInit(BaseModel):
    x: int = Field(ge=0)

    def __init__(self, **data: Any) -> None:
        super().__init__(x=1)

CustomInit.model_validate({"x": -1})  # E: violates Pydantic `ge` constraint
"#,
);

pydantic_testcase!(
    bug = "Unmodeled model alias configuration is confused with defaults",
    test_model_validate_unmodeled_alias_config,
    r#"
from pydantic import BaseModel, ConfigDict, Field

def alias(name: str) -> str:
    return "y"

class Generated(BaseModel):
    model_config = ConfigDict(alias_generator=alias)
    x: int = Field(ge=0)

Generated.model_validate({"x": -1, "y": 1})  # E: violates Pydantic `ge` constraint

flag = False

class Dynamic(BaseModel):
    model_config = ConfigDict(validate_by_alias=flag, validate_by_name=True)
    x: int = Field(ge=0, alias="y")

Dynamic.model_validate({"x": 1, "y": -1})  # E: violates Pydantic `ge` constraint
"#,
);
