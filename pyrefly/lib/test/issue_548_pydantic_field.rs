/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

// Create a test environment with mocked pydantic module
fn env_with_pydantic() -> TestEnv {
    let mut env = TestEnv::new();
    env.add(
        "pydantic",
        r#"
from typing import TypeVar, overload, Any

_T = TypeVar("_T")

class BaseModel:
    pass

# Field overloads based on typical pydantic signatures
@overload
def Field(default: _T, *, ge: int | None = None, le: int | None = None, description: str | None = None, min_length: int | None = None) -> _T: ...

@overload
def Field(*, default: _T, ge: int | None = None, le: int | None = None, description: str | None = None, min_length: int | None = None) -> _T: ...

@overload
def Field(*, ge: int | None = None, le: int | None = None, description: str | None = None, min_length: int | None = None) -> Any: ...

def Field(default: Any | None = None, *, ge: int | None = None, le: int | None = None, description: str | None = None, min_length: int | None = None) -> Any:
    return default
"#,
    );
    env
}

// Test case for Issue #548: Pydantic Field Overload Resolution Failure
testcase!(
    test_pydantic_field_positional,
    env_with_pydantic(),
    r#"
from typing import assert_type
from pydantic import BaseModel, Field

class DatabaseConfig(BaseModel):
    # Positional default argument should work
    port: int = Field(5432, ge=1, le=65535, description="Database port")
    assert_type(port, int)
    
    # Multiple fields with positional defaults
    min_connections: int = Field(5, ge=1, le=100)
    max_connections: int = Field(20, ge=1, le=1000)
    
    # Keyword default should also work
    timeout: int = Field(default=30, ge=1)
"#,
);

testcase!(
    test_pydantic_field_type_inference,
    env_with_pydantic(),
    r#"
from typing import assert_type
from pydantic import BaseModel, Field

class Config(BaseModel):
    # TypeVar inference should work with positional args
    name: str = Field("default_name", min_length=1)
    assert_type(name, str)
    
    count: int = Field(42, ge=0)
    assert_type(count, int)
    
    flag: bool = Field(True, description="Feature flag")
    assert_type(flag, bool)
    
    # None as default
    optional_value: int | None = Field(None, description="Optional")
    assert_type(optional_value, int | None)
"#,
);
