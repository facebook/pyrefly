/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::pydantic_testcase;

pydantic_testcase!(
    bug = "BaseSettings fields can be populated from environment variables but pyrefly doesn't know this",
    test_base_settings_no_defaults,
    r#"
import os

from pydantic_settings import BaseSettings 

class AppConfig(BaseSettings):
    database_url: str
    api_key: str
    port: int

# Set environment variables
os.environ["DATABASE_URL"] = "postgres://localhost:5432/mydb" 
os.environ["API_KEY"] = "my-secret-api-key" 
os.environ["PORT"] = "8080"

# This should work - fields come from environment variables
config = AppConfig()  # E: Missing argument `database_url` in function `AppConfig.__init__` # E: Missing argument `api_key` in function `AppConfig.__init__` # E: Missing argument `port` in function `AppConfig.__init__`
"#,
);

pydantic_testcase!(
    bug = "This is not a correct program to write since a model cannot be both a BaseSettings and a RootModel",
    test_base_settings_wrong_def,
    r#"
from pydantic_settings import BaseSettings
from pydantic import RootModel

class AppConfig(BaseSettings, RootModel):
    database_url: str
    api_key: str
    port: int

"#,
);
