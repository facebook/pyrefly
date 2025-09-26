/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

fn pydantic_env() -> TestEnv {
    let path = std::env::var("PYDANTIC_TEST_PATH").expect("PYDANTIC_TEST_PATH must be set");
    TestEnv::new_with_site_package_path(&path)
}

testcase!(
    bug = "we could support ranges, but this is not for v1",
    test_field_right_type,
    pydantic_env(),
    r#"
from pydantic import BaseModel, Field
class Model(BaseModel):
   x: int = Field(gt=0, lt=10)

Model(x=5) 
Model(x=0)  
Model(x=15)
"#,
);

testcase!(
    test_field_wrong_type,
    pydantic_env(),
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel):
    x: int = Field(gt="A", lt="B") # E:  Pydantic `gt` value is of type `Literal['A']` but the field is annotated with `int` # E: Pydantic `lt` value is of type `Literal['B']` but the field is annotated with `int`

Model(x=5)
"#,
);

testcase!(
    test_field_ge,
    pydantic_env(),
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel):
    x: int = Field(ge="B") # E: Pydantic `ge` value is of type `Literal['B']` but the field is annotated with `int`

Model(x=5)
"#,
);

testcase!(
    test_field_optional,
    pydantic_env(),
    r#"
from pydantic import BaseModel, Field

class Example(BaseModel):
    id: str
    attribute_1: str
    optional_attribute: str | None = Field(None, description="An optional attribute")

Example(id="123", attribute_1="value1")
"#,
);
