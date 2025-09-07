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
    test_alias,
    pydantic_env(),
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel):
    x: int = Field(alias="y")

m = Model(y=0)
m.x
m.y # E: Object of class `Model` has no attribute `y`
"#,
);

testcase!(
    test_validation_alias,
    pydantic_env(),
    r#"
from pydantic import BaseModel, Field

class Model(BaseModel):
    x: int = Field(validation_alias="y", alias="z")

m = Model(y=0)
m = Model(z=0) # E: Missing argument `y` in function `Model.__init__`

m.x
m.y  # E: Object of class `Model` has no attribute `y`
m.z  # E: Object of class `Model` has no attribute `z`
"#,
);

testcase!(
    test_validation_by_alias_and_name,
    pydantic_env(),
    r#"
from pydantic import BaseModel, Field
class Model(BaseModel, validate_by_name=True, validate_by_alias=True):
    x: int = Field(alias='y')
Model(x=0) 
Model(y=0)
"#,
);

testcase!(
    test_validation_by_alias,
    pydantic_env(),
    r#"
from pydantic import BaseModel, Field
class Model(BaseModel, validate_by_alias=True):
    x: int = Field(alias='y')
Model(x=0) # E: Missing argument `y` in function `Model.__init__` 
Model(y=0)
"#,
);

testcase!(
    test_validation_by_name,
    pydantic_env(),
    r#"
from pydantic import BaseModel, Field
class Model(BaseModel, validate_by_name=True):
    x: int = Field(alias='y')
Model(x=0)
Model(y=0)
"#,
);

testcase!(
    test_validation_by_name_only,
    pydantic_env(),
    r#"
from pydantic import BaseModel, Field
class Model(BaseModel, validate_by_name=True, validate_by_alias=False):
    x: int = Field(alias='y')
Model(x=0)
Model(y=0) # E: Missing argument `x` in function `Model.__init__` 
"#,
);

testcase!(
    bug = "when both validation keys are true, aliases should be required. mypy doesn't support the right behavior here either. This requires generating overloads. We might shelve this for v2",
    test_validation_defaults,
    pydantic_env(),
    r#"
from pydantic import BaseModel, Field
class Model(BaseModel, validate_by_name=True, validate_by_alias=True):
    x: int = Field(alias='y')
Model()
"#,
);

testcase!(
    bug = "We should respect validate_by_name and accept both Model(...) calls",
    test_configdict_validate_by_name,
    pydantic_env(),
    r#"
from pydantic import BaseModel, Field, ConfigDict
class Model(BaseModel):
    x: str = Field(..., alias="y")
    model_config = ConfigDict(validate_by_name=True)
Model(y="123")
Model(x="123")  
    "#,
);
