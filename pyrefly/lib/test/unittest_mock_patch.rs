/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    test_target_string_checked,
    TestEnv::one("other", ""),
    r#"
from unittest.mock import patch

with patch("other.foo"):  # E: No attribute `foo` in module `other`
    pass
"#,
);

testcase!(
    test_current_module_checked,
    TestEnv::one("other", ""),
    r#"
from unittest.mock import patch

with patch("main.foo"):  # E: No attribute `foo` in module `main`
    pass
"#,
);

testcase!(
    test_builtin_name_checked,
    TestEnv::one("other", ""),
    r#"
from unittest.mock import patch

with patch("other.open"):  # E: No attribute `open` in module `other`
    pass
"#,
);

testcase!(
    test_create_true_skipped,
    TestEnv::one("other", ""),
    r#"
from unittest.mock import patch

with patch("other.foo", create=True):
    pass
"#,
);

testcase!(
    test_unknown_create_skipped,
    TestEnv::one("other", ""),
    r#"
from unittest.mock import patch

def test(create: bool) -> None:
    with patch("other.foo", create=create):
        pass
"#,
);

testcase!(
    test_positional_create_skipped,
    TestEnv::one("other", ""),
    r#"
from unittest.mock import DEFAULT, patch

with patch("other.foo", DEFAULT, None, True):
    pass
"#,
);

testcase!(
    test_kwargs_create_skipped,
    TestEnv::one("other", ""),
    r#"
from unittest.mock import patch

with patch("other.foo", **{"create": True}):
    pass
"#,
);

testcase!(
    test_create_false_checked,
    TestEnv::one("other", ""),
    r#"
from unittest.mock import patch

with patch("other.foo", create=False):  # E: No attribute `foo` in module `other`
    pass
"#,
);

testcase!(
    test_private_checked,
    TestEnv::one("other", ""),
    r#"
from unittest.mock import patch

with patch("other._MAXLINE"):  # E: No attribute `_MAXLINE` in module `other`
    pass
"#,
);

testcase!(
    test_private_valid,
    TestEnv::one("other", "_private = 0"),
    r#"
from unittest.mock import patch

with patch("other._private"):
    pass
"#,
);

testcase!(
    test_nested_attributes_checked,
    TestEnv::one("other", "class C:\n    existing = 0\ninstance = C()"),
    r#"
from unittest.mock import patch

with patch("other.C.missing"):  # E: Class `C` has no class attribute `missing`
    pass

with patch("other.instance.missing"):  # E: Object of class `C` has no attribute `missing`
    pass
"#,
);

testcase!(
    test_nested_attributes_valid,
    TestEnv::one("other", "class C:\n    existing = 0\ninstance = C()"),
    r#"
from unittest.mock import patch

with patch("other.C.existing"):
    pass

with patch("other.instance.existing"):
    pass
"#,
);

testcase!(
    test_module_getattr,
    TestEnv::one("other", "def __getattr__(name: str) -> int: ..."),
    r#"
from unittest.mock import patch

with patch("other.dynamic"):
    pass
"#,
);

testcase!(
    test_keyword_checked,
    TestEnv::one("other", ""),
    r#"
from unittest.mock import patch

with patch(target="other.foo"):  # E: No attribute `foo` in module `other`
    pass
"#,
);

testcase!(
    test_other_module_valid,
    TestEnv::one("other", "def foo() -> None: ..."),
    r#"
from unittest.mock import patch

with patch("other.foo"):
    pass
"#,
);

testcase!(
    test_patch_object_not_treated_as_patch,
    TestEnv::one("other", ""),
    r#"
from unittest.mock import patch

with patch.object("other.missing", "attribute"):
    pass
"#,
);

testcase!(
    test_patch_multiple_not_treated_as_patch,
    TestEnv::one("other", ""),
    r#"
from unittest.mock import DEFAULT, patch

with patch.multiple("other.missing", attribute=DEFAULT):
    pass
"#,
);
