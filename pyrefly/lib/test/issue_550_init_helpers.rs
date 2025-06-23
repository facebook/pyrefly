/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

// Test case for Issue #550: Support initialization helper methods called from __init__
testcase!(
    test_init_helper_methods,
    r#"
class MetricsCollector:
    def __init__(self):
        # Call helper methods during initialization
        self._init_performance_metrics()
        self._init_error_metrics()
    
    def _init_performance_metrics(self) -> None:
        # These should not error when called from __init__
        self.request_duration = 0.0
        self.db_query_duration = 0.0
    
    def _init_error_metrics(self) -> None:
        # These should not error when called from __init__
        self.error_count = 0
        self.error_rate = 0.0
        
    def get_metrics(self) -> dict[str, float]:
        # These attributes should be recognized as defined
        return {
            "request_duration": self.request_duration,
            "error_count": self.error_count,
        }
"#,
);

testcase!(
    test_conditional_init_helpers,
    r#"
class ConfigurableService:
    def __init__(self, enable_cache: bool = True):
        self.enabled = True
        if enable_cache:
            self._setup_cache()
    
    def _setup_cache(self) -> None:
        # Attributes defined in conditionally-called helpers
        self.cache_size = 1000
        self.cache = {}
        
    def use_cache(self) -> None:
        # Should understand cache might not be defined
        if hasattr(self, "cache"):
            self.cache.clear()
"#,
);

testcase!(
    test_nested_init_helpers,
    r#"
class ComplexService:
    def __init__(self):
        self._init_base()
    
    def _init_base(self) -> None:
        self.base_attr = "base"
        self._init_extended()
    
    def _init_extended(self) -> None:
        # Nested helper called from another helper
        self.extended_attr = "extended"
        
    def get_attrs(self) -> tuple[str, str]:
        return (self.base_attr, self.extended_attr)
"#,
);

testcase!(
    bug = "Should still error when helper is not called from init",
    test_uncalled_helper_method,
    r#"
class Service:
    def __init__(self):
        self.initialized = True
        # Note: _setup is NOT called from __init__
    
    def _setup(self) -> None:
        # This SHOULD error - not called from __init__
        self.uncalled_attr = "error"  # E: Attribute `uncalled_attr` is implicitly defined
        
    def use_attr(self) -> None:
        # This should error - uncalled_attr might not exist
        print(self.uncalled_attr)  # E: Object of class `Service` has no attribute `uncalled_attr`
"#,
);