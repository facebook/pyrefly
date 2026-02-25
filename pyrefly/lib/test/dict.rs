/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ruff_text_size::Ranged;
use ruff_text_size::TextSize;

use crate::test::util;
use crate::testcase;

testcase!(
    test_literal_dict,
    r#"
dict(x = 1, y = "test")
    "#,
);

testcase!(
    test_anonymous_typed_dict_union_promotion,
    r#"
from typing import assert_type

def test(cond: bool):
    x = {"a": 1, "b": "2"}
    y = {"a": 1, "b": "2", "c": 3}
    # we promote anonymous typed dicts when unioning
    z = x if cond else y
    assert_type(z["a"], int | str)
    assert_type(z, dict[str, int | str])
"#,
);

testcase!(
    test_unpack_empty,
    r#"
from typing import assert_type
x = {**{}}
x['x'] = 0
assert_type(x, dict[str, int])
    "#,
);

testcase!(
    test_typeddict_interaction,
    r#"
from typing import TypedDict
class C(TypedDict):
    x: int
x: C | dict[str, int] = {"y": 0}
    "#,
);

testcase!(
    test_kwargs_unpack_dict_union,
    r#"
from typing import Any

def foo(**kwargs: Any) -> None:
    pass

def bar(yes: bool) -> None:
    if yes:
        kwargs = {"hello": "world"}
    else:
        kwargs = {"goodbye": 1}

    foo(**kwargs)
"#,
);

#[test]
fn test_dict_literal_error_range_points_to_value() {
    util::init_test();
    let code = r#"x: dict[str, str] = {
    "1": 2,
}
"#;
    let (handle, state) = util::mk_state(code);
    let errors = state
        .transaction()
        .get_errors([&handle])
        .collect_errors()
        .shown;
    assert_eq!(errors.len(), 1);
    let err = &errors[0];
    let value_offset = code.find("2").expect("missing dict value literal");
    let expected_start = TextSize::try_from(value_offset).unwrap();
    assert_eq!(err.range().start(), expected_start);
}
