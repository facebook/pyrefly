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

testcase!(
    test_setdefault_append,
    r#"
d = {}
items = [("news", "token1"), ("sports", "token2"), ("news", "token3")]
for topic, token in items:
    d.setdefault(topic, []).append(token)
"#,
);

testcase!(
    test_large_dict_literal_mixed_none,
    r#"
# Regression test: dict literals with many entries of mixed str | None values
# previously caused exponential memory blowup during overload resolution
# because partial type variables were not restored after failed overload attempts.
# This test completes in bounded time only with the fix in place.
d = {
    "a": None,
    "b": "v1",
    "c": None,
    "d": "v2",
    "e": None,
    "f": "v3",
    "g": None,
    "h": "v4",
    "i": None,
    "j": "v5",
    "k": None,
    "l": "v6",
    "m": None,
    "n": "v7",
    "o": None,
}
x: dict[str, str | None] = d
"#,
);

testcase!(
    bug = "False positives (see in-line comments)",
    test_dict_literal_should_be_typeddict,
    r#"
from typing import assert_type, TypedDict
class TD(TypedDict):
    x: int
def f[T: TD](x: T) -> T:
    return x

d = {"x": 0}
# We represent d as an anonymous TypedDict, so the `f` call succeeds, but the returned type is wrong.
assert_type(f(d), TD)  # E: assert_type(dict[str, int], TD)

# This `f` call incorrectly fails, and the returned type is wrong.
assert_type(f({"x": 0}), TD)  # E: `dict[str, int]` is not assignable to upper bound `TD`  # E: assert_type(dict[str, int], TD)
    "#,
);

// Regression test: deeply nested dict literals previously caused exponential memory growth
// because AnonymousTypedDictInner stored the value type both in `fields` and a redundant
// `value_type` field, doubling the cloned type tree at each nesting level. The fix removed
// the redundant field and computes the value type on demand from `fields`.
//
// Depth 15 is used for CI speed. The fix was verified at depth 25 (239 MB, down from 7.7 GB)
// and depth 50 (236 MB), confirming linear rather than exponential growth.
testcase!(
    test_deeply_nested_dict_literal,
    r#"
from typing import assert_type

x = {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": {"i": {"j": {"k": {"l": {"m": {"n": {"o": "deep"}}}}}}}}}}}}}}}
assert_type(x, dict[str, dict[str, dict[str, dict[str, dict[str, dict[str, dict[str, dict[str, dict[str, dict[str, dict[str, dict[str, dict[str, dict[str, dict[str, str]]]]]]]]]]]]]]])
"#,
);
