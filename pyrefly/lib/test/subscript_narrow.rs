/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    test_index_narrow,
    r#"
from typing import assert_type
class C1:
    x: list[object]
class C2:
    x: object
def test(x: list[object], c1: C1, c2s: list[C2]):
    assert_type(x[0], object)
    assert isinstance(x[0], int)
    assert_type(x[0], int)

    assert_type(c1.x[0], object)
    assert isinstance(c1.x[0], int)
    assert_type(c1.x[0], int)

    assert_type(c2s[0].x, object)
    assert isinstance(c2s[0].x, int)
    assert_type(c2s[0].x, int)
 "#,
);

testcase!(
    test_index_narrow_invalidation,
    r#"
from typing import assert_type
class C1:
    x: list[object]
class C2:
    x: object
def test(x: list[object], c1: C1, c2s: list[C2], s: str):
    assert isinstance(x[0], int)
    x[0] = s
    assert_type(x[0], str)
    x = []
    assert isinstance(x[0], object)

    assert isinstance(c1.x[0], int)
    c1.x[0] = s
    assert_type(c1.x[0], str)

    assert isinstance(c2s[0].x, int)
    c2s[0].x = s
    assert_type(c2s[0].x, str)
 "#,
);

testcase!(
    test_index_narrow_prefix_invalidation,
    r#"
from typing import assert_type
class C1:
    x: list[object]
class C2:
    x: object
def test(x: list[object], c1: C1, c2s: list[C2], s: str, idx: int):
    assert isinstance(x[0], int)
    assert_type(x[0], int)
    x[idx] = s
    assert_type(x[0], object)

    assert isinstance(c1.x[0], int)
    assert_type(c1.x[0], int)
    c1.x[idx] = s
    assert_type(c1.x[0], object)

    assert isinstance(c2s[0].x, int)
    assert_type(c2s[0].x, int)
    c2s[idx].x = s
    assert_type(c2s[0].x, object)
 "#,
);

testcase!(
    test_key_narrow,
    r#"
from typing import assert_type
class C1:
    x: dict[str, object]
class C2:
    x: object
def test(x: dict[str, object], c1: C1, c2s: dict[str, C2]):
    assert_type(x["key1"], object)
    assert isinstance(x["key1"], int)
    assert_type(x["key1"], int)

    assert_type(c1.x["key1"], object)
    assert isinstance(c1.x["key1"], int)
    assert_type(c1.x["key1"], int)

    assert_type(c2s["key1"].x, object)
    assert isinstance(c2s["key1"].x, int)
    assert_type(c2s["key1"].x, int)
 "#,
);

testcase!(
    test_key_narrow_invalidation,
    r#"
from typing import assert_type
class C1:
    x: dict[str, object]
class C2:
    x: object
def test(x: dict[str, object], c1: C1, c2s: dict[str, C2], s: str):
    assert isinstance(x["key1"], int)
    x["key1"] = s
    assert_type(x["key1"], str)
    x = {}
    assert isinstance(x["key1"], object)

    assert isinstance(c1.x["key1"], int)
    c1.x["key1"] = s
    assert_type(c1.x["key1"], str)

    assert isinstance(c2s["key1"].x, int)
    c2s["key1"].x = s
    assert_type(c2s["key1"].x, str)
 "#,
);

testcase!(
    test_key_narrow_prefix_invalidation,
    r#"
from typing import assert_type
class C1:
    x: dict[str, object]
class C2:
    x: object
def test(x: dict[str, object], c1: C1, c2s: dict[str, C2], key: str, s: str):
    assert isinstance(x["key1"], int)
    assert_type(x["key1"], int)
    x[key] = s
    assert_type(x["key1"], object)

    assert isinstance(c1.x["key1"], int)
    assert_type(c1.x["key1"], int)
    c1.x[key] = s
    assert_type(c1.x["key1"], object)

    assert isinstance(c2s["key1"].x, int)
    assert_type(c2s["key1"].x, int)
    c2s[key].x = s
    assert_type(c2s["key1"].x, object)
 "#,
);

testcase!(
    test_subscript_narrow_does_not_invalidate_attribute,
    r#"
from typing import Optional, Dict, Any, assert_type, Literal

class ErrorContext:
    def __init__(self):
        self.system_context: dict[str, Any] | None = None

    def update_context(self, data: dict[str, Any]) -> None:
        # Explicit None check
        if self.system_context is not None:
            assert_type(self.system_context, dict[str, Any])

            self.system_context["updated"] = True

            assert_type(self.system_context, dict[str, Any])
            assert_type(self.system_context["updated"], Literal[True])

            value = self.system_context.get("key", "default")

    def update_context_2(self, data: dict[str, Any]) -> None:
        if self.system_context:
            self.system_context["status"] = "active"
        else:
            self.system_context = {"status": "active"}

        assert_type(self.system_context, dict[str, Any])

        self.system_context["timestamp"] = "2024-01-01"

        assert_type(self.system_context, dict[str, Any])
        assert_type(self.system_context["timestamp"], Literal["2024-01-01"])
"#,
);

testcase!(
    test_dict_get_literal_key_narrow,
    r#"
from typing import assert_type, Literal

def use(mapping: dict[str, int | None]) -> None:
    if mapping.get("foo") is not None:
        assert_type(mapping.get("foo"), int)
        assert_type(mapping["foo"], int)
    else:
        assert_type(mapping.get("foo"), None)
        assert_type(mapping["foo"], None)

def use2(mapping: dict[str, int | None]) -> None:
    if mapping.get("foo"):
        assert_type(mapping.get("foo"), int)
        assert_type(mapping["foo"], int)
    else:
        assert_type(mapping.get("foo"), int | None)
        assert_type(mapping["foo"], int | None)
"#,
);

testcase!(
    test_typeddict_get_literal_key_narrow,
    TestEnv::new().enable_not_required_key_access_error(),
    r#"
from typing import TypedDict, assert_type, Literal

class TD(TypedDict, total=False):
    foo: int | None
    bar: str

def use(mapping: TD) -> None:
    if mapping.get("foo") is not None:
        assert_type(mapping.get("foo"), int)
        assert_type(mapping["foo"], int)
    else:
        assert_type(mapping.get("foo"), None)
        assert_type(mapping["foo"], None)

def use2(mapping: TD) -> None:
    if mapping.get("foo"):
        assert_type(mapping.get("foo"), int)
        assert_type(mapping["foo"], int)
    else:
        assert_type(mapping.get("foo"), int | None)
        assert_type(mapping["foo"], int | None)  # E: TypedDict key `foo` may be absent
"#,
);

testcase!(
    test_typeddict_contains_not_required_key_basic,
    TestEnv::new().enable_not_required_key_access_error(),
    r#"
from typing import TypedDict, NotRequired, assert_type

class TD(TypedDict):
    foo: NotRequired[int]

def use(mapping: TD) -> None:
    if "foo" in mapping:
        assert_type(mapping["foo"], int)
        if "foo" not in mapping:
            mapping["foo"]  # E: TypedDict key `foo` may be absent
    else:
        mapping["foo"]  # E: TypedDict key `foo` may be absent
    if "foo" not in mapping:
        mapping["foo"]  # E: TypedDict key `foo` may be absent
    else:
        assert_type(mapping["foo"], int)
"#,
);

testcase!(
    test_typeddict_contains_not_required_key_get,
    TestEnv::new().enable_not_required_key_access_error(),
    r#"
from typing import TypedDict, NotRequired, assert_type

class TD(TypedDict):
    foo: NotRequired[int]

def use(mapping: TD) -> None:
    if mapping.get("foo"):
        assert_type(mapping["foo"], int)
    else:
        mapping["foo"]  # E: TypedDict key `foo` may be absent
    mapping["foo"]  # E: TypedDict key `foo` may be absent
"#,
);

testcase!(
    test_non_total_typed_dict_not_required_key_warning,
    TestEnv::new().enable_not_required_key_access_error(),
    r#"
from typing import TypedDict

class TD(TypedDict, total=False):
    foo: int

def bad(mapping: TD) -> int:
    return mapping["foo"]  # E: TypedDict key `foo` may be absent
"#,
);

testcase!(
    test_typeddict_contains_not_required_key_compound_condition,
    TestEnv::new().enable_not_required_key_access_error(),
    r#"
from typing import TypedDict, NotRequired, assert_type

class TD(TypedDict):
    foo: NotRequired[int]
    bar: NotRequired[int]

def use(mapping: TD, cond: bool) -> None:
    if "foo" in mapping and "bar" in mapping:
        assert_type(mapping["foo"], int)
        assert_type(mapping["bar"], int)
    else:
        mapping["foo"]  # E: TypedDict key `foo` may be absent
        mapping["bar"]  # E: TypedDict key `bar` may be absent

    if "foo" in mapping or "bar" in mapping:
        mapping["foo"]  # E: TypedDict key `foo` may be absent
        mapping["bar"]  # E: TypedDict key `bar` may be absent
    else:
        mapping["foo"]  # E: TypedDict key `foo` may be absent
        mapping["bar"]  # E: TypedDict key `bar` may be absent

    if "foo" not in mapping and "bar" not in mapping:
        mapping["foo"]  # E: TypedDict key `foo` may be absent
        mapping["bar"]  # E: TypedDict key `bar` may be absent
    else:
        mapping["foo"]  # E: TypedDict key `foo` may be absent
        mapping["bar"]  # E: TypedDict key `bar` may be absent

    if "foo" not in mapping or "bar" not in mapping:
        mapping["foo"]  # E: TypedDict key `foo` may be absent
        mapping["bar"]  # E: TypedDict key `bar` may be absent
    else:
        assert_type(mapping["foo"], int)
        assert_type(mapping["bar"], int)

    if "foo" in mapping and cond:
        assert_type(mapping["foo"], int)
    else:
        mapping["foo"]  # E: TypedDict key `foo` may be absent

    if "foo" in mapping or cond:
        mapping["foo"]  # E: TypedDict key `foo` may be absent
    else:
        mapping["foo"]  # E: TypedDict key `foo` may be absent
"#,
);

testcase!(
    test_typeddict_literal_variable_key_narrow,
    r#"
from typing import TypedDict, Literal, assert_type

class Payload(TypedDict):
    my_key: list[str] | None

def local_case() -> None:
    data: Payload = {"my_key": None}
    key: Literal["my_key"] = "my_key"
    if data[key] is None:
        data[key] = []
    else:
        assert_type(data[key], list[str])
        assert_type(data["my_key"], list[str])
        data[key].append("a")
        data["my_key"].append("b")

def param_case(data: Payload, key: Literal["my_key"]) -> None:
    if data[key] is not None:
        assert_type(data[key], list[str])
        data[key].append("c")
"#,
);

testcase!(
    test_non_dict_get_does_not_narrow,
    r#"
from typing import assert_type

class NotDict:
    def get(self, key: str) -> int | None: ...
    def __getitem__(self, key: str) -> int | None: ...

def use(mapping: NotDict) -> None:
    if mapping.get("foo") is not None:
        assert_type(mapping.get("foo"), int | None)
        assert_type(mapping["foo"], int | None)
    else:
        assert_type(mapping.get("foo"), int | None)
        assert_type(mapping["foo"], int | None)

def use2(mapping: NotDict) -> None:
    if mapping.get("foo"):
        assert_type(mapping.get("foo"), int | None)
        assert_type(mapping["foo"], int | None)
    else:
        assert_type(mapping.get("foo"), int | None)
        assert_type(mapping["foo"], int | None)
"#,
);
