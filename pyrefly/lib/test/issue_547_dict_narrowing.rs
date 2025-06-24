/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

// Test case for Issue #547: Cannot Handle None Checks Before Dictionary Access
testcase!(
    test_dict_none_narrow_attribute_chain,
    r#"
from typing import Optional, Dict, Any, assert_type

class ConfigManager:
    def __init__(self):
        self.system_context: Optional[Dict[str, Any]] = None
        
    def test_explicit_none_check(self) -> None:
        if self.system_context is not None:
            # After narrowing, should be Dict[str, Any], not Optional
            assert_type(self.system_context, Dict[str, Any])
            # Should be able to set items without error
            self.system_context["updated"] = True
            self.system_context["data"] = {"key": "value"}
            
    def test_dict_methods(self) -> Any:
        if self.system_context is not None:
            # Dictionary methods should work without error
            value = self.system_context.get("key", "default")
            keys = self.system_context.keys()
            items = self.system_context.items()
            return value
        return None
        
    def test_truthy_check(self) -> list[str]:
        if self.system_context:
            # Truthy check should also narrow
            assert_type(self.system_context, Dict[str, Any])
            return list(self.system_context.keys())
        return []
"#,
);

testcase!(
    test_dict_none_narrow_early_return,
    r#"
from typing import Optional, Dict, Any, assert_type

def process_config(config: Optional[Dict[str, Any]]) -> str:
    if config is None:
        return "no config"
    # After early return, config cannot be None
    assert_type(config, Dict[str, Any])
    # Should not error on dictionary methods
    return config.get("setting", "default")

def process_with_isinstance(data: Optional[Dict[str, Any]]) -> None:
    if isinstance(data, dict):
        # After isinstance check, should narrow to dict
        assert_type(data, Dict[str, Any])
        data["checked"] = True
        items = data.items()
        for k, v in items:
            print(f"{k}: {v}")
"#,
);

testcase!(
    test_dict_none_narrow_nested,
    r#"
from typing import Optional, Dict, Any, assert_type

class NestedConfig:
    def __init__(self):
        self.outer: Optional[Dict[str, Optional[Dict[str, Any]]]] = None
        
    def test_nested_narrowing(self) -> None:
        if self.outer is not None:
            assert_type(self.outer, Dict[str, Optional[Dict[str, Any]]])
            self.outer["key"] = {"nested": "value"}
            
            inner = self.outer.get("key")
            if inner is not None:
                assert_type(inner, Dict[str, Any])
                inner["updated"] = True
"#,
);
