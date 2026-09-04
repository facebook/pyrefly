/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use itertools::Itertools as _;
use pretty_assertions::assert_eq;
use pyrefly_build::handle::Handle;
use pyrefly_python::module::TextRangeWithModule;
use ruff_text_size::TextSize;

use crate::state::state::State;
use crate::test::util::code_frame_of_source_at_range;
use crate::test::util::get_batched_lsp_operations_report;

fn get_test_report(state: &State, handle: &Handle, position: TextSize) -> String {
    // `goto_type_definition` never returns an empty `Vec`: every success path yields at
    // least one range, and having nothing to report is spelled as an error instead.
    match state.transaction().goto_type_definition(handle, position) {
        Err(reason) => format!("Type Definition Result: None ({})", reason.as_str()),
        Ok(defs) => defs
            .into_iter()
            .map(
                |TextRangeWithModule {
                     module: module_info,
                     range,
                 }| {
                    format!(
                        "Type Definition Result:\n{}",
                        code_frame_of_source_at_range(module_info.contents(), range)
                    )
                },
            )
            .join("\n"),
    }
}

#[test]
fn function_goes_to_its_own_def() {
    let code = r#"
def f(x: list[int], y: str) -> bytes: ...

g = f
#   ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
4 | g = f
        ^
Type Definition Result:
2 | def f(x: list[int], y: str) -> bytes: ...
        ^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn imported_function_goes_to_its_own_def() {
    let lib = r#"
def g(x: int) -> str: ...
"#;
    let code = r#"
from lib import g

x = g
#   ^
"#;
    let report =
        get_batched_lsp_operations_report(&[("main", code), ("lib", lib)], get_test_report);
    assert_eq!(
        r#"
# main.py
4 | x = g
        ^
Type Definition Result:
2 | def g(x: int) -> str: ...
        ^


# lib.py
"#
        .trim(),
        report.trim(),
    );
}

/// A decorator whose parameters have class-typed defaults, which is the shape of
/// `numba.jit`. Before functions resolved to their own `def`, this reported
/// `MappingProxyType` and `bool` — the classes of the parameter defaults.
#[test]
fn decorator_does_not_report_parameter_types() {
    let code = r#"
from types import MappingProxyType

def jit(signature_or_function=None, locals=MappingProxyType({}), cache=False):
    def wrapper(func):
        return func
    return wrapper

@jit
# ^
def foo(x, y):
    return x + y
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
9 | @jit
      ^
Type Definition Result:
4 | def jit(signature_or_function=None, locals=MappingProxyType({}), cache=False):
        ^^^
"#
        .trim(),
        report.trim(),
    );
}

/// `FunctionKind` special-cases a set of well-known functions and discards their
/// `FuncId`, so they cannot be resolved by `def_index`. `numba.jit` is one of them, and
/// resolves instead through the module `FunctionKind` records it under.
#[test]
fn special_cased_function_resolves_through_declaring_module() {
    let numba = r#"
from numba.core.decorators import jit
"#;
    let decorators = r#"
def jit(signature_or_function=None, cache=False):
    def wrapper(func):
        return func
    return wrapper
"#;
    let code = r#"
from numba import jit

@jit
# ^
def foo(x, y):
    return x + y
"#;
    let report = get_batched_lsp_operations_report(
        &[
            ("main", code),
            ("numba", numba),
            ("numba.core.decorators", decorators),
        ],
        get_test_report,
    );
    assert_eq!(
        r#"
# main.py
4 | @jit
      ^
Type Definition Result:
2 | def jit(signature_or_function=None, cache=False):
        ^^^


# numba.py

# numba.core.decorators.py
"#
        .trim(),
        report.trim(),
    );
}

/// The decorated symbol carries the decorator's return type, which is how you reach
/// the type a decorator produces.
#[test]
fn decorated_symbol_goes_to_decorator_return_type() {
    let code = r#"
class Dispatcher: ...

def deco(func) -> Dispatcher: ...

@deco
def bar(): ...

x = bar
#   ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
9 | x = bar
        ^
Type Definition Result:
2 | class Dispatcher: ...
          ^^^^^^^^^^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn method_goes_to_its_own_def() {
    let code = r#"
class A:
    def m(self, x: int) -> str: ...

a = A()
x = a.m
#     ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
6 | x = a.m
          ^
Type Definition Result:
3 |     def m(self, x: int) -> str: ...
            ^
"#
        .trim(),
        report.trim(),
    );
}

/// An overload set carries a single `FuncMetadata`, so it produces one result: the
/// implementation, not the individual overload signatures.
#[test]
fn overloaded_function() {
    let code = r#"
from typing import overload

@overload
def h(x: int) -> int: ...
@overload
def h(x: str) -> str: ...
def h(x: int | str) -> int | str:
    return x

y = h
#   ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
11 | y = h
         ^
Type Definition Result:
8 | def h(x: int | str) -> int | str:
        ^
"#
        .trim(),
        report.trim(),
    );
}

/// For a name in "the thing being called" position, an overload gets collapsed down
/// to the single signature that matched, leaving a bare callable shape with no link back
/// to the original `def`. This check finds no function identity, falls through, and reports
/// the types inside that signature.
#[test]
fn overloaded_function_call() {
    let code = r#"
from typing import overload

@overload
def h(x: int) -> int: ...
@overload
def h(x: str) -> str: ...
def h(x: int | str) -> int | str:
    return x

y = h(1)
#   ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
11 | y = h(1)
         ^
Type Definition Result:
418 | class int:
            ^^^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn class_instance_is_unchanged() {
    let code = r#"
class A: ...

a = A()
b = a
#   ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
5 | b = a
        ^
Type Definition Result:
2 | class A: ...
          ^
"#
        .trim(),
        report.trim(),
    );
}

/// A `Callable` annotation carries no function identity, so it still falls through to
/// the generic walk over the type and reports the parameter and return types.
#[test]
fn callable_annotation_falls_through_to_signature_types() {
    let code = r#"
from typing import Callable

def takes(c: Callable[[int], str]) -> None:
    d = c
#       ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
5 |     d = c
            ^
Type Definition Result:
418 | class int:
            ^^^
Type Definition Result:
1033 | class str(Sequence[str]):
             ^^^
"#
        .trim(),
        report.trim(),
    );
}

/// A synthesized `__init__` has no `def_index`, so it also falls through.
#[test]
fn synthesized_init_falls_through() {
    let code = r#"
from dataclasses import dataclass

@dataclass
class D:
    x: int

D.__init__
#     ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
8 | D.__init__
          ^
Type Definition Result:
5 | class D:
          ^
Type Definition Result:
418 | class int:
            ^^^
"#
        .trim(),
        report.trim(),
    );
}

/// `FunctionKind` records the *public* module a special-cased function is declared
/// under, so importing one straight out of its private module leaves nothing to look
/// up. Reporting nothing is deliberate: the alternative is the parameter types.
#[test]
fn special_cased_function_not_reexported_reports_nothing() {
    let numba = r#"
"#;
    let decorators = r#"
def jit(signature_or_function=None, cache=False):
    def wrapper(func):
        return func
    return wrapper
"#;
    let code = r#"
from numba.core.decorators import jit

@jit
# ^
def foo(x, y):
    return x + y
"#;
    let report = get_batched_lsp_operations_report(
        &[
            ("main", code),
            ("numba", numba),
            ("numba.core.decorators", decorators),
        ],
        get_test_report,
    );
    assert_eq!(
        r#"
# main.py
4 | @jit
      ^
Type Definition Result: None (definition_not_found)


# numba.py

# numba.core.decorators.py
"#
        .trim(),
        report.trim(),
    );
}

/// A `@singledispatch` dispatcher is typed as a callback protocol over its fallback's
/// signature, so its definition is the protocol class. Walking the type instead would
/// report `object`, the fallback's parameter type.
#[test]
fn singledispatch_dispatcher_goes_to_its_protocol_class() {
    let code = r#"
from functools import singledispatch

@singledispatch
def fun(arg: object) -> None: ...

r = fun
#     ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
7 | r = fun
          ^
Type Definition Result:
266 | class _SingleDispatchCallable(Generic[_T]):
            ^^^^^^^^^^^^^^^^^^^^^^^
"#
        .trim(),
        report.trim(),
    );
}

/// `register` is reached through the dispatcher rather than through `functools`, so it
/// has no name to resolve and reports nothing.
#[test]
fn singledispatch_register_reports_nothing() {
    let code = r#"
from functools import singledispatch

@singledispatch
def fun(arg: object) -> None: ...

@fun.register
#      ^
def _(arg: int) -> None: ...
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
7 | @fun.register
           ^
Type Definition Result: None (definition_not_found)
"#
        .trim(),
        report.trim(),
    );
}

/// `def_index` identifies a `def` within its module, so a function nested in another
/// function resolves to its own `def` rather than the enclosing one.
#[test]
fn nested_function_goes_to_its_own_def() {
    let code = r#"
def outer() -> None:
    def inner(x: int) -> str: ...
    y = inner
#       ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
4 |     y = inner
            ^
Type Definition Result:
3 |     def inner(x: int) -> str: ...
            ^^^^^
"#
        .trim(),
        report.trim(),
    );
}
