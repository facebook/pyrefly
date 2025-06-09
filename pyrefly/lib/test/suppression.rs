/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_pyrefly_suppression,
    r#"
def foo() -> str:
  # pyrefly: ignore
  return 1
"#,
);

testcase!(
    test_pyrefly_suppression_pass_comment,
    r#"
def foo() -> str:
  # pyrefly: ignore
  # some explanation about the ignore
  # some other explanation about the ignore
  return 1
"#,
);

testcase!(
    test_pyrefly_suppression_stop_at_empty_line,
    r#"
def foo() -> str:
  # pyrefly: ignore

  return 1  # E: not assignable
"#,
);

testcase!(
    test_pyrefly_top_level_ignore,
    r#"
# pyrefly: ignore-all-errors
3 + "3"
3 + "3"
"#,
);

testcase!(
    test_pyrefly_top_level_ignore_wrong_same_line,
    r#"
3 + "3" # pyrefly: ignore-all-errors # E:
3 + "3" # E:
"#,
);

testcase!(
    test_pyrefly_top_level_ignore_wrong_own_line,
    r#"
3 + "3" # E:
# pyrefly: ignore-all-errors
3 + "3" # E:
"#,
);
