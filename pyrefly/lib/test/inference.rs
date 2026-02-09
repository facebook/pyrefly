/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_config::base::UntypedDefBehavior;

use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    test_no_empty_container_inference,
    TestEnv::new_with_infer_with_first_use(false),
    r#"
from typing import assert_type, Any
x = []
x.append(1)
x.append("foo")
assert_type(x, list[Any])
x = {}
x[1] = 2
x["1"] = "2"
assert_type(x, dict[Any, Any])
"#,
);

testcase!(
    test_empty_container_inference_via_generic_function,
    TestEnv::new_with_infer_with_first_use(false),
    r#"
from typing import assert_type

def pair[T](a: T, b: T) -> tuple[T, T]:
    return (a, b)

# The empty list [] comes first, but it gets unified with list[int] from
# the second argument through the generic function's type variable T.
result = pair([], [1, 2, 3])
assert_type(result, tuple[list[int], list[int]])

# Same pattern with dict
dict_result = pair({}, {"a": 1})
assert_type(dict_result, tuple[dict[str, int], dict[str, int]])
"#,
);

testcase!(
    test_implicit_any_no_inference,
    TestEnv::new_with_untyped_def_behavior(UntypedDefBehavior::SkipAndInferReturnAny)
        .enable_unannotated_return_error()
        .enable_unannotated_parameter_error(),
    r#"
def foo(x, y):  # E: `foo` is missing an annotation for parameter `x` # E: `foo` is missing an annotation for parameter `y` # E: `foo` is missing a return annotation
    return 1
"#,
);

testcase!(
    test_implicit_any_with_inference,
    TestEnv::new_with_untyped_def_behavior(UntypedDefBehavior::CheckAndInferReturnType)
        .enable_unannotated_return_error()
        .enable_unannotated_parameter_error(),
    r#"
def foo(x, y):  # E: `foo` is missing an annotation for parameter `x` # E: `foo` is missing an annotation for parameter `y` # E: `foo` is missing a return annotation
    return 1
"#,
);

testcase!(
    test_implicit_any_self_cls_ignored,
    TestEnv::new().enable_implicit_any_error(),
    r#"
class C:
    def method(self) -> int:
        return 1

    @classmethod
    def clsmethod(cls) -> int:
        return 1
"#,
);

testcase!(
    test_implicit_any_with_complete_annotations,
    TestEnv::new().enable_implicit_any_error(),
    r#"
def foo(x: int) -> int:
    return x
"#,
);

testcase!(
    test_implicit_any_empty_containers,
    TestEnv::new_with_infer_with_first_use(false).enable_implicit_any_error(),
    r#"
from typing import Iterable, Mapping
x1 = [] # E: Cannot infer type of empty container
x2 = {} # E: Cannot infer type of empty container
x3: Iterable[int] = {} # ok
x4: Mapping[str, str] = {} # ok
"#,
);

testcase!(
    test_implicit_any_empty_containers_with_partial_inference,
    TestEnv::new().enable_implicit_any_error(),
    r#"
from typing import Iterable, Mapping
x1 = [] # E: Cannot infer type of empty container
x2 = {} # E: Cannot infer type of empty container
x3: Iterable[int] = {} # ok
x4: Mapping[str, str] = {} # ok
"#,
);

testcase!(
    test_implicit_any_default_disabled,
    r#"
from typing import Iterable
def foo(x):
    return x

x1 = []
x2 = {}
"#,
);

testcase!(
    test_warn_on_implicit_any_in_attribute,
    TestEnv::new().enable_unannotated_attribute_error(),
    r#"
from typing import Any
class A:
    def __init__(self):
        self.x = None  # E: implicitly inferred to be `Any | None`
    "#,
);
