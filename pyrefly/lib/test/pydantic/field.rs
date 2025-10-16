/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::pydantic_testcase;

pydantic_testcase!(
    bug = "we could support ranges, but this is not for v1",
    test_field_right_type,
    r#"
from pydantic import BaseModel, Field
class Model(BaseModel):
   x: int = Field(gt=0, lt=10)

Model(x=5) 
Model(x=0)  
Model(x=15)
"#,
);

pydantic_testcase!(
    test_field_wrong_type,
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel):
    x: int = Field(gt="A", lt="B") # E:  Pydantic `gt` value is of type `Literal['A']` but the field is annotated with `int` # E: Pydantic `lt` value is of type `Literal['B']` but the field is annotated with `int`

Model(x=5)
"#,
);

pydantic_testcase!(
    test_field_ge,
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel):
    x: int = Field(ge="B") # E: Pydantic `ge` value is of type `Literal['B']` but the field is annotated with `int`

Model(x=5)
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
    test_required_field,
    r#"
from pydantic import BaseModel
class A(BaseModel, validate_by_name=True, validate_by_alias=True):
    x: int
A()  # E: Missing argument `x`
    "#,
);
