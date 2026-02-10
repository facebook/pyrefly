/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
