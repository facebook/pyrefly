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
    yield "oops"  # E: Yielded type `Literal['oops']` is not assignable to declared yield type `int`
    return 55  # E: Returned type `Literal[55]` is not assignable to declared return type `str`

async def simple_async() -> int:
    return "oops"  # E: Returned type `Literal['oops']` is not assignable to declared return type `int`

async def async_generator() -> AsyncGenerator[int, None]:
    yield "oops"  # E: Yielded type `Literal['oops']` is not assignable to declared yield type `int`

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

// State 1: check-unannotated-defs=false, infer-return-types=never.
// Unannotated functions are skipped entirely; annotated functions are checked
// but return types are never inferred.
testcase!(
    test_skip_check_no_infer,
    TestEnv::new_skip_check_no_infer(),
    r#"
from typing import assert_type, Any, Callable

# Unannotated: body is not checked, return type is Any
def unchecked(x, y):
    z: str = 0  # no error
    return x + y
assert_type(unchecked(0, 0), Any)

# Annotated: body is checked, but return type is NOT inferred (still Any)
def annotated_params(x: int, y: int):
    return x + y
assert_type(annotated_params(0, 0), Any)

# Explicitly annotated return: body is checked, annotation is used
def annotated_return(x: int) -> int:
    return x + 1
assert_type(annotated_return(0), int)

# Annotated function with missing return path: error is reported
def missing_return(x: int) -> int:  # E: missing an explicit `return`
    if x > 0:
        return x
"#,
);

// State 1b: async/generators with check-unannotated-defs=false, infer-return-types=never.
// Return types must always be Any (wrapped in Coroutine for async).
testcase!(
    test_skip_check_no_infer_async_and_generators,
    TestEnv::new_skip_check_no_infer(),
    r#"
from typing import assert_type, Any, Callable, Coroutine

# Unannotated async: return type is Any (no Coroutine wrapping without inference)
async def async_f():
    return 42
assert_type(async_f, Callable[[], Any])

# Unannotated generator: return type is Any (no inference)
def gen():
    yield 42
assert_type(gen, Callable[[], Any])

# Unannotated async generator: return type is Any (no inference)
async def async_gen():
    yield 42
assert_type(async_gen, Callable[[], Any])

# Annotated async: return type is NOT inferred (still Any), wrapped in Coroutine
async def annotated_async(x: int):
    return x + 1
assert_type(annotated_async(0), Coroutine[Any, Any, Any])

# Annotated generator: return type is NOT inferred (still Any)
def annotated_gen(x: int):
    yield x
assert_type(annotated_gen(0), Any)
"#,
);

// State 2: check-unannotated-defs=false, infer-return-types=annotated.
// Unannotated functions are skipped; annotated functions get return inference.
testcase!(
    test_skip_check_and_infer_return_type,
    TestEnv::new_skip_check_infer_return_types(),
    r#"
from typing import assert_type, Any

# check-unannotated-defs=false -> totally unchecked
def unchecked(x, y):
    z: str = 0 # no error
    return x + y
# infer-return-types=annotated -> inferred for annotated functions
def inferred_return(x: int, y: int):
    return x + y
assert_type(unchecked(0, 0), Any)
assert_type(inferred_return(0, 0), int)
"#,
);

// State 2b: async/generators with check-unannotated-defs=false, infer-return-types=annotated.
// Unannotated are Any; annotated get inference (including async/generator wrapping).
testcase!(
    test_skip_check_infer_annotated_async_and_generators,
    TestEnv::new_skip_check_infer_return_types(),
    r#"
from typing import assert_type, Any, Callable, Coroutine, Generator, AsyncGenerator

# Unannotated async: return type is Any (no inference, no Coroutine wrapping)
async def unannotated_async():
    return 42
assert_type(unannotated_async, Callable[[], Any])

# Unannotated generator: return type is Any (no inference)
def unannotated_gen():
    yield 42
assert_type(unannotated_gen, Callable[[], Any])

# Annotated async: return type IS inferred
async def annotated_async(x: int):
    return x + 1
assert_type(annotated_async(0), Coroutine[Any, Any, int])

# Annotated generator: return type IS inferred
def annotated_gen(x: int):
    yield x
assert_type(annotated_gen(0), Generator[int, Any, None])

# Annotated async generator: return type IS inferred
async def annotated_async_gen(x: int):
    yield x
assert_type(annotated_async_gen(0), AsyncGenerator[int, Any])
"#,
);

// State 5: check-unannotated-defs=true, infer-return-types=never.
// All bodies are checked, but return types are never inferred.
testcase!(
    test_check_all_no_infer,
    TestEnv::new_check_all_no_infer(),
    r#"
from typing import assert_type, Any, Callable

# Unannotated: body IS checked, but return type is Any
def unannotated():
    oops: int = "oops"  # E:
    return 42
assert_type(unannotated, Callable[[], Any])

# Annotated params: body is checked, return type is NOT inferred
def annotated_params(x: int, y: int):
    return x + y
assert_type(annotated_params(0, 0), Any)

# Explicit return annotation: respected as always
def annotated_return(x: int) -> int:
    return x + 1
assert_type(annotated_return(0), int)
"#,
);

// State 5b: async/generators with check-unannotated-defs=true, infer-return-types=never.
// All bodies are checked, return types are always Any.
testcase!(
    test_check_all_no_infer_async_and_generators,
    TestEnv::new_check_all_no_infer(),
    r#"
from typing import assert_type, Any, Callable, Coroutine

# Unannotated async: body IS checked, return type is Coroutine[Any, Any, Any]
async def unannotated_async():
    oops: int = "oops"  # E:
    return 42
assert_type(unannotated_async, Callable[[], Coroutine[Any, Any, Any]])

# Unannotated generator: body IS checked, return type is Any
def unannotated_gen():
    oops: int = "oops"  # E:
    yield 42
assert_type(unannotated_gen, Callable[[], Any])

# Annotated async: body IS checked, return type is NOT inferred
async def annotated_async(x: int):
    return x + 1
assert_type(annotated_async(0), Coroutine[Any, Any, Any])

# Annotated generator: body IS checked, return type is NOT inferred
def annotated_gen(x: int):
    yield x
assert_type(annotated_gen(0), Any)
"#,
);

// State 6: check-unannotated-defs=true, infer-return-types=annotated.
// All bodies are checked, but return types are only inferred for functions
// with at least one annotation.
testcase!(
    test_check_all_infer_annotated_only,
    TestEnv::new_check_infer_annotated_only(),
    r#"
from typing import assert_type, Any, Callable

# Unannotated: body is checked but return type is Any
def unannotated():
    oops: int = "oops"  # E:
    return 42
assert_type(unannotated, Callable[[], Any])

# Annotated parameters: body is checked and return type is inferred
def annotated_params(x: int, y: int):
    return x + y
assert_type(annotated_params(0, 0), int)
"#,
);

testcase!(
    bug = "@no_type_check on classes (applying to all methods) is not yet supported",
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

assert_type(f(0), Any)
assert_type(C(42).x, Any)
"#,
);

// @no_type_check must return Any even when infer-return-types=never.
testcase!(
    test_no_type_check_with_skip_check_no_infer,
    TestEnv::new_skip_check_no_infer(),
    r#"
from typing import no_type_check, assert_type, Any

@no_type_check
def f(x: int) -> int:
    y: int = "y"
    return "f"

assert_type(f(0), Any)
"#,
);
