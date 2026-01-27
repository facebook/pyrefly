/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

fn env_pytest_fixture() -> TestEnv {
    let mut t = TestEnv::new();
    t.add_with_path(
        "pytest",
        "pytest.pyi",
        r#"
from typing import Callable, TypeVar

T = TypeVar("T")

def fixture(func: Callable[..., T]) -> Callable[..., T]: ...
"#,
    );
    t
}

testcase!(
    bug = "Fixture injection should propagate fixture return types to parameters.",
    test_pytest_fixture_injected_parameter_type,
    env_pytest_fixture(),
    r#"
import pytest
from typing import reveal_type

@pytest.fixture
def my_fixture() -> int:
    return 42

def test_foo(my_fixture):
    reveal_type(my_fixture)  # E: revealed type: Unknown
"#,
);
