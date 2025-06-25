/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
