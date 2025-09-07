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
    test_config_conditional_extra,
    pydantic_env(),
    r#"
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

u = User(name="Alice", age=30)
print(u)
"#,
);

testcase!(
    test_model_config,
    pydantic_env(),
    r#"
from pydantic import BaseModel, ConfigDict

class Model(BaseModel):
    model_config = ConfigDict(frozen=True)
    x: int = 42


m = Model()
m.x = 10 # E: Cannot set field `x`
"#,
);

testcase!(
    test_not_a_pydantic_model,
    pydantic_env(),
    r#"
from pydantic import ConfigDict

class Model:
    model_config = ConfigDict(frozen=True)
    x: int = 42


m = Model()
m.x = 10 
"#,
);

// This is a corner case, but since y is annotated, we consider it a field
testcase!(
    test_model_config_alias,
    pydantic_env(),
    r#"
from pydantic import BaseModel, ConfigDict

class Model(BaseModel):
    y: ConfigDict = ConfigDict(frozen=True)
    model_config = y
    x: int = 42

m = Model()
m.x = 10

"#,
);

testcase!(
    bug = "We should raise an error on y because all model fields require an annotation.",
    test_model_config_y,
    pydantic_env(),
    r#"
from pydantic import BaseModel, ConfigDict

class Model(BaseModel):
    y = ConfigDict(frozen=True)
    x: int = 42

m = Model()
m.x = 10 
"#,
);

// Only the last config is considered
testcase!(
    test_model_two_config,
    pydantic_env(),
    r#"
from pydantic import BaseModel, ConfigDict

class Model(BaseModel):
    model_config = ConfigDict(frozen=True)
    model_config = ConfigDict(frozen=False)
    x: int = 42

m = Model()
m.x = 10
"#,
);

testcase!(
    test_frozen_model_subclass,
    pydantic_env(),
    r#"
from pydantic import BaseModel, ConfigDict

class Model(BaseModel):
    model_config = ConfigDict(frozen=True)
    x: int = 42

class Model2(Model):
    model_config = ConfigDict(frozen=False)

m = Model2()
m.x = 10
"#,
);

testcase!(
    test_frozen_model_default,
    pydantic_env(),
    r#"
from pydantic import BaseModel
class A(BaseModel, frozen=True):
    x: int

class B(A):
    pass

b = B(x=0)
b.x = 1 # E: Cannot set field `x`
"#,
);

testcase!(
    bug = "Nested config not supported yet. Fields should be frozen.",
    test_nested_model_config,
    pydantic_env(),
    r#"
from pydantic import BaseModel

class Model(BaseModel):
    class Config:
        frozen = True

    x: int = 42

m = Model()
m.x = 10

"#,
);
