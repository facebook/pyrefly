/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::pydantic_testcase;

pydantic_testcase!(
    bug = "Annotation aliases and nonliteral validator modes lose uncertainty",
    test_model_validate_alias_and_mode_uncertainty,
    r#"
from typing import Annotated, Literal
from pydantic import BaseModel, BeforeValidator, Field, field_validator

Positive = Annotated[int, BeforeValidator(abs)]

class Aliased(BaseModel):
    x: Positive = Field(ge=0)

Aliased.model_validate({"x": -1})  # E: violates Pydantic `ge` constraint

MODE: Literal["before"] = "before"

class Configured(BaseModel):
    x: int = Field(ge=0)

    @field_validator("x", mode=MODE)
    @classmethod
    def normalize(cls, value: int) -> int:
        return abs(value)

Configured.model_validate({"x": -1})  # E: violates Pydantic `ge` constraint
"#,
);
