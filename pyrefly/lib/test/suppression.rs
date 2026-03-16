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
# pyrefly: ignore-errors
3 + "3"
3 + "3"
"#,
);

testcase!(
    test_pyrefly_top_level_ignore_wrong_same_line,
    r#"
3 + "3" # pyrefly: ignore-errors # E:
3 + "3" # E:
"#,
);

testcase!(
    test_pyrefly_top_level_ignore_wrong_own_line,
    r#"
3 + "3" # E:
# pyrefly: ignore-errors
3 + "3" # E:
"#,
);

testcase!(
    test_pyrefly_suppression_typed,
    r#"
def foo() -> str:
  # pyrefly: ignore[bad-return]
  return 1
"#,
);

testcase!(
    test_pyrefly_suppression_typed_inline,
    r#"
def foo() -> str:
  return 1  # pyrefly: ignore[bad-return]
"#,
);

testcase!(
    test_pyrefly_suppression_typed_wrong_type,
    r#"
def foo() -> str:
  # pyrefly: ignore[bad-assignment]
  return 1 # E: Returned type `Literal[1]` is not assignable to declared return type `str`
"#,
);

testcase!(
    test_pyrefly_suppression_typed_inline_wrong_type,
    r#"
def foo() -> str:
  return 1  # pyrefly: ignore[bad-assignment]  # E: Returned type `Literal[1]` is not assignable to declared return type `str`
"#,
);

testcase!(
    test_pyrefly_suppression_typed_bad_type,
    r#"
def foo() -> str:
  # pyrefly: ignore[bad-]
  return 1 # E: Returned type `Literal[1]` is not assignable to declared return type `str`
"#,
);

testcase!(
    test_pyrefly_suppression_typed_empty,
    r#"
def foo() -> str:
  # pyrefly: ignore[]
  return 1 # E: Returned type `Literal[1]` is not assignable to declared return type `str`
"#,
);

testcase!(
    test_pyrefly_suppression_typed_whitespace_variation,
    r#"
def foo() -> str:
  #   pyrefly:    ignore   [  bad-return  ]
  return 1
"#,
);

testcase!(
    test_pyrefly_suppression_typed_multiple_codes,
    r#"
def foo() -> str:
  # pyrefly: ignore[bad-return, bad-assignment]
  return 1
"#,
);

testcase!(
    test_pyrefly_suppression_typed_multiple_valid_codes,
    r#"
def foo() -> str:
  # pyrefly: ignore[bad-return]
  # pyrefly: ignore[bad-argument-type]
  return len(1)
"#,
);

testcase!(
    test_pyrefly_suppression_typed_multiple_codes_line,
    r#"
def foo() -> str:
  # pyrefly: ignore[bad-return] # pyrefly: ignore[bad-argument-type]
  return len(1)
"#,
);

testcase!(
    test_ignore_whitespace,
    r#"
x0: int = "hello" #type:ignore
x1: int = "hello" # type:ignore
x2: int = "hello" #type: ignore
x3: int = "hello" # type: ignore
x4: int = "hello" # type:  ignore
x5: int = "hello" #  type:  ignore    # more
"#,
);

testcase!(
    test_ignore,
    r#"
x: int = "1"  # type: ignore

# type: ignore
y: int = "2"

z: int = "3"  # E: `Literal['3']` is not assignable to `int`
"#,
);

testcase!(
    test_ignore_attachment,
    r#"
# type: ignore
w: int = "0"

x: int = "1"  # type: ignore
y: int = "2"  # E: is not assignable
"#,
);

testcase!(
    test_pyrefly_suppression_fstring_above,
    r#"
def foo() -> str:
  # pyrefly: ignore
  return f"""
result: {1 + "a"}
"""
"#,
);

testcase!(
    test_pyrefly_suppression_fstring_inline_end,
    r#"
def foo() -> str:
  return f"""
result: {1 + "a"}
"""  # pyrefly: ignore
"#,
);

testcase!(
    test_pyrefly_suppression_fstring_typed,
    r#"
def foo() -> str:
  # pyrefly: ignore[unsupported-operation]
  return f"""
result: {1 + "a"}
"""
"#,
);

testcase!(
    test_pyrefly_suppression_nested_fstring_single_line_inner,
    r#"
def foo() -> str:
  # pyrefly: ignore
  return f"""
result: {f"{1 + 'a'}"}
"""
"#,
);

testcase!(
    test_pyrefly_suppression_nested_fstring_multi_line_inner,
    r#"
def foo() -> str:
  # pyrefly: ignore
  return f"""
result: {f'''
{1 + "a"}
'''}
"""
"#,
);

testcase!(
    test_pyrefly_suppression_consecutive_fstrings_above_first,
    r#"
def foo():
  # pyrefly: ignore
  f"""hello"""
  f"result: {1 + "a"}"  # E: is not supported
"#,
);

testcase!(
    test_pyrefly_suppression_consecutive_fstrings_above_second,
    r#"
def foo():
  f"""hello"""
  # pyrefly: ignore
  f"result: {1 + "a"}"
"#,
);

testcase!(
    test_pyrefly_suppression_fstring_embedded_comment,
    r#"
def foo() -> str:
  return f"""
# pyrefly: ignore
result: {1 + "a"}  # E: is not supported
"""
"#,
);
