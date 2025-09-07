/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::config::base::UntypedDefBehavior;
use crate::test::util::TestEnv;
use crate::testcase;

testcase!(
    test_function_check_and_inference_with_mode_infer_return_type,
    TestEnv::new_with_untyped_def_behavior(UntypedDefBehavior::CheckAndInferReturnType),
    r#"
from typing import assert_type, Any, Callable, Coroutine, Generator, AsyncGenerator

x: int = ...  # E:

def f():
    oops: int = "oops"  # E:
    return x
assert_type(f, Callable[[], int])

async def async_f():
    oops: int = "oops"  # E:
    return x
assert_type(async_f, Callable[[], Coroutine[Any, Any, int]])

def gen():
    oops: int = "oops"  # E:
    yield x
assert_type(gen, Callable[[], Generator[int, Any, None]])

def gen_w_return():
    oops: int = "oops"  # E:
    yield x
    return x
assert_type(gen_w_return, Callable[[], Generator[int, Any, int]])

async def async_gen():
    oops: int = "oops"  # E:
    yield x
assert_type(async_gen, Callable[[], AsyncGenerator[int, Any]])
"#,
);

testcase!(
    test_function_check_and_inference_with_mode_infer_return_any,
    TestEnv::new_with_untyped_def_behavior(UntypedDefBehavior::CheckAndInferReturnAny),
    r#"
from typing import assert_type, Any, Callable, Coroutine, Generator, AsyncGenerator

x: int = ...  # E:

def f():
    oops: int = "oops"  # E:
    return x
assert_type(f, Callable[[], Any])

async def async_f():
    oops: int = "oops"  # E:
    return x
assert_type(async_f, Callable[[], Coroutine[Any, Any, Any]])

def gen():
    oops: int = "oops"  # E:
    yield x
assert_type(gen, Callable[[], Any])

def gen_w_return():
    oops: int = "oops"  # E:
    yield x
    return x
assert_type(gen_w_return, Callable[[], Any])

async def async_gen():
    oops: int = "oops"  # E:
    yield x
assert_type(async_gen, Callable[[], Any])
"#,
);

testcase!(
    test_function_check_and_inference_with_mode_skip_and_infer_return_any,
    TestEnv::new_with_untyped_def_behavior(UntypedDefBehavior::SkipAndInferReturnAny),
    r#"
from typing import assert_type, Any, Callable, Coroutine, Generator, AsyncGenerator

x: int = ...  # E:

def f():
    oops: int = "oops"
    return x
assert_type(f, Callable[[], Any])

async def async_f():
    oops: int = "oops"
    return x
assert_type(async_f, Callable[[], Any])

def gen():
    oops: int = "oops"
    yield x
assert_type(gen, Callable[[], Any])

def gen_w_return():
    oops: int = "oops"
    yield x
    return x
assert_type(gen_w_return, Callable[[], Any])

async def async_gen():
    oops: int = "oops"
    yield x
assert_type(async_gen, Callable[[], Any])
"#,
);

// Because the yield and return type plumbing works a bit differently when inferring
// any, we want to be sure to make sure that in this mode
// - an annotated function (or async function or generators) has its returns and yields checked
// - we correctly flag a function annotated as a generator that has no yields
// - we correctly flag an async generator with a return as invalid (even with no annotation)
testcase!(
    test_verify_return_and_yield_with_mode_infer_return_any,
    TestEnv::new_with_untyped_def_behavior(UntypedDefBehavior::CheckAndInferReturnAny),
    r#"
from typing import assert_type, Any, Callable, Coroutine, Generator, AsyncGenerator

def simple_return() -> int:
    return "oops"  # E: Returned type `Literal['oops']` is not assignable to declared return type `int`

def simple_implicit_return() -> int:  # E: missing an explicit `return`
    pass

def generator_with_return() -> Generator[int, Any, str]:
    # TODO(stroxler): this yield error message needs some wordsmithing!
    yield "oops"  # E: Type of yielded value `Literal['oops']` is not assignable to declared return type `int`
    return 55  # E: Returned type `Literal[55]` is not assignable to declared return type `str`

async def simple_async() -> int:
    return "oops"  # E: Returned type `Literal['oops']` is not assignable to declared return type `int`

async def async_generator() -> AsyncGenerator[int, None]:
    # TODO(stroxler): this yield error message needs some wordsmithing!
    yield "oops"  # E: Type of yielded value `Literal['oops']` is not assignable to declared return type `int`

def marked_as_generator_but_does_not_yield() -> Generator[int, Any, str]:
    return "str"  # E: Returned type `Literal['str']` is not assignable to declared return type `Generator[int, Any, str]`

async def async_generator_with_return():
    yield "s"
    return 42  # E: Return statement with value is not allowed in async generator
"#,
);

testcase!(
    test_self_attrs_with_mode_check_and_infer_return_any,
    TestEnv::new_with_untyped_def_behavior(UntypedDefBehavior::CheckAndInferReturnAny)
        .enable_implicitly_defined_attribute_error(),
    r#"
from typing import assert_type, Any
class C:
    def __init__(self):
        self.x: int = 5
    def f(self):
        self.y: str = "y"  # E: Attribute `y` is implicitly defined by assignment in method `f`
c = C()
assert_type(c.x, int)
assert_type(c.y, str)
assert_type(c.f(), Any)
"#,
);

testcase!(
    test_self_attrs_with_mode_skip_and_infer_return_any,
    TestEnv::new_with_untyped_def_behavior(UntypedDefBehavior::SkipAndInferReturnAny)
        .enable_implicitly_defined_attribute_error(),
    r#"
from typing import assert_type, Any
class C:
    def __init__(self):
        self.x: int = 5
    def f(self):
        self.y: str = "y"  # E: Attribute `y` is implicitly defined by assignment in method `f`
c = C()
assert_type(c.x, Any)
assert_type(c.y, Any)
assert_type(c.f(), Any)
"#,
);

testcase!(
    test_annotated_defs_with_mode_skip_and_infer_return_any,
    TestEnv::new_with_untyped_def_behavior(UntypedDefBehavior::SkipAndInferReturnAny),
    r#"
from typing import assert_type
def unannotated():
    x: int = "x"
def annotated_return() -> None:
    x: int = "x"  # E:
def annotated_param(_: str):
    x: int = "x"  # E:
"#,
);

testcase!(
    test_annotated_defs_check_and_transform_with_mode_infer_return_any,
    TestEnv::new_with_untyped_def_behavior(UntypedDefBehavior::CheckAndInferReturnAny),
    r#"
from typing import assert_type, Any, Callable, Coroutine, Generator, AsyncGenerator

x: int = ...  # E:

def f() -> str:
    oops: int = "oops"  # E:
    return x  # E:
assert_type(f, Callable[[], str])

async def async_f() -> str:
    oops: int = "oops"  # E:
    return x  # E:
assert_type(async_f, Callable[[], Coroutine[Any, Any, str]])

def gen() -> Generator[str, Any, None]:
    oops: int = "oops"  # E:
    yield x  # E:
assert_type(gen, Callable[[], Generator[str, Any, None]])

async def async_gen() -> AsyncGenerator[str, Any]:
    oops: int = "oops"  # E:
    yield x  # E:
assert_type(async_gen, Callable[[], AsyncGenerator[str, Any]])
"#,
);

testcase!(
    stress_tests_for_mode_skip_and_infer_return_any,
    TestEnv::new_with_untyped_def_behavior(UntypedDefBehavior::SkipAndInferReturnAny),
    r#"
from typing import assert_type
def u0():
    x: int = "x"
def u1(y, *args, **kwargs):
    x: int = "x"
class C:
    def __init__(self):
        x: int = "x"
        pass
    def __init__(self, y, *args, **kwargs):
        x: int = "x"
        pass
"#,
);

testcase!(
    bug = "We do not yet implement @no_type_check",
    test_no_type_check_decorator,
    r#"
from typing import no_type_check, assert_type, Any

@no_type_check
def f(x: int) -> int:
    y: int = "y"
    return "f"

class C:
    @no_type_check
    def __init__(self, x: int) -> None:
        self.x = x

assert_type(C(42).x, Any)
"#,
);
