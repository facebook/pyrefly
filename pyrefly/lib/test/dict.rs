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
    test_anonymous_typed_dict_spread_unpack,
    r#"
from typing import assert_type

# Basic dict unpacking should preserve anonymous typed dict
defaults = {"host": "localhost", "port": 8080}
overrides = {"port": 9090}
config = {**defaults, **overrides}
port: int = config["port"]
host: str = config["host"]

# Multiple dict unpacking with different value types
base = {"name": "test", "count": 0}
extra = {"count": 5, "active": True}
combined = {**base, **extra}
count: int = combined["count"]
name: str = combined["name"]
active: bool = combined["active"]

# Passing unpacked dict value to typed function
def process_port(port: int) -> str:
    return f":{port}"

merged = {**defaults, **overrides}
process_port(merged["port"])

# assert_type works on the dict
assert_type(config["port"], int)
assert_type(config["host"], str)
"#,
);

testcase!(
    test_anonymous_typed_dict_spread_with_explicit_keys,
    r#"
from typing import assert_type
base = {"x": 1, "y": "hello"}
extended = {**base, "z": True}
assert_type(extended["x"], int)
assert_type(extended["y"], str)
assert_type(extended["z"], bool)
assert_type(extended, dict[str, int | str | bool])
"#,
);

testcase!(
    test_anonymous_typed_dict_spread_override,
    r#"
from typing import assert_type
original = {"val": "string"}
updated = {**original, "val": 42}
assert_type(updated["val"], int)
assert_type(updated, dict[str, int])
"#,
);
