/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

// Creates a test environment with sklearn stubs matching the bundled stubs.
fn env_with_sklearn_stubs() -> TestEnv {
    let mut env = TestEnv::new();
    env.add(
        "sklearn._typing",
        r#"
import typing_extensions

Int: typing_extensions.TypeAlias
        "#,
    );
    env.add(
        "sklearn._config",
        r#"
from collections.abc import Generator
from contextlib import contextmanager
from typing import Literal

from ._typing import Int

@contextmanager
def config_context(
    *,
    assume_finite: None | bool = None,
    working_memory: None | Int = None,
    print_changed_only: None | bool = None,
    display: None | Literal["text", "diagram"] = None,
    pairwise_dist_chunk_size: None | Int = None,
    enable_cython_pairwise_dist: None | bool = None,
    array_api_dispatch: None | bool = None,
    transform_output: None | str = None,
    enable_metadata_routing: None | bool = None,
    skip_parameter_validation: None | bool = None,
) -> Generator[None, None, None]: ...
"#,
    );
    env.add(
        "sklearn",
        "from sklearn._config import config_context as config_context",
    );
    env
}

// The sklearn bundled stub types `config_context` as returning a valid context manager
testcase!(
    test_sklearn_config_context,
    env_with_sklearn_stubs(),
    r#"
import sklearn

with sklearn.config_context(transform_output="pandas"):
    pass
"#,
);
