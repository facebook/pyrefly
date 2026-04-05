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

testcase!(
    test_get_dict_value_even_with_error,
    r#"
from typing import assert_type
d: dict[str, int] = {}
def f(k: str | None):
    # We should report the mismatch between `str` and `str | None` rather than "No matching overload".
    v = d.get(k)  # E: Argument `str | None` is not assignable to parameter `key` with type `str`
    # Because only one overload of `dict.get` can match based on argument count, we should use its
    # return type of `int | None`.
    assert_type(v, int | None)
    "#,
);

testcase!(
    test_dict_get_return,
    r#"
from typing import Any
def f(outcomes: list[Any]) -> dict[str, int]:
    ret = {noun: int(count) for (count, noun) in outcomes}
    to_plural = {
        "warning": "warnings",
        "error": "errors",
    }
    return {to_plural.get(k, k): v for k, v in ret.items()}
"#,
);
