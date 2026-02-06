/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_unittest_mock_patch_target_string_checked,
    r#"
from unittest.mock import patch

def bar() -> None: ...

with patch("main.foo"):  # E: No attribute `foo` in module `main`
    pass
"#,
);
