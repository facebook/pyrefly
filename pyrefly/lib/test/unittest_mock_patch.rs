/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
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

testcase!(
    test_unittest_mock_patch_target_builtin_checked,
    r#"
from unittest.mock import patch

with patch("main.open"):
    pass
"#,
);

testcase!(
    test_unittest_mock_patch_target_create_true_skipped,
    r#"
from unittest.mock import patch

with patch("main.foo", create=True):
    pass
"#,
);

testcase!(
    test_unittest_mock_patch_target_private_skipped,
    r#"
from unittest.mock import patch

with patch("main._MAXLINE"):
    pass
"#,
);

testcase!(
    test_unittest_mock_patch_target_other_module_skipped,
    TestEnv::one("other", ""),
    r#"
from unittest.mock import patch

with patch("other.foo"):
    pass
"#,
);
