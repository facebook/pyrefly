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
    bug = "we should raise an error that x is immutable",
    test_model_config,
    pydantic_env(),
    r#"
from pydantic import BaseModel, ConfigDict

class Model(BaseModel):
    model_config = ConfigDict(frozen=True)
    x: int = 42


m = Model()
m.x = 10 
"#,
);
