/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_build::handle::Handle;

use crate::state::require::Require;
use crate::state::state::State;
use crate::test::util::mk_multi_file_state;

fn get_unused_import_diagnostics(state: &State, handle: &Handle) -> String {
    let transaction = state.transaction();
    if let Some(bindings) = transaction.get_bindings(handle) {
        let unused_imports = bindings.unused_imports();
        if unused_imports.is_empty() {
            return "No unused imports".to_owned();
        }
        let mut report = String::new();
        for (i, unused) in unused_imports.iter().enumerate() {
            if i > 0 {
                report.push_str(", ");
            }
            report.push_str(&format!("Import `{}` is unused", unused.name.as_str()));
        }
        report
    } else {
        "No bindings".to_owned()
    }
}

fn get_unused_variable_diagnostics(state: &State, handle: &Handle) -> String {
    let transaction = state.transaction();
    if let Some(bindings) = transaction.get_bindings(handle) {
        let unused_variables = bindings.unused_variables();
        if unused_variables.is_empty() {
            return "No unused variables".to_owned();
        }
        let mut report = String::new();
        for (i, unused) in unused_variables.iter().enumerate() {
            if i > 0 {
                report.push_str(", ");
            }
            report.push_str(&format!("Variable `{}` is unused", unused.name.as_str()));
        }
        report
    } else {
        "No bindings".to_owned()
    }
}

#[test]
fn test_dotted_import_used() {
    let code = r#"
import os.path

def check_exists(path: str) -> bool:
    return os.path.exists(path)
"#;
    let (handles, state) = mk_multi_file_state(&[("main", code)], Require::indexing(), true);
    let handle = handles.get("main").unwrap();
    let report = get_unused_import_diagnostics(&state, handle);
    assert_eq!(report, "No unused imports");
}

#[test]
fn test_dotted_import_unused() {
    let code = r#"
import os.path

def foo() -> str:
    return "hello"
"#;
    let (handles, state) = mk_multi_file_state(&[("main", code)], Require::indexing(), true);
    let handle = handles.get("main").unwrap();
    let report = get_unused_import_diagnostics(&state, handle);
    assert_eq!(report, "Import `os` is unused");
}

#[test]
fn test_simple_import_used() {
    let code = r#"
import os

def get_cwd() -> str:
    return os.getcwd()
"#;
    let (handles, state) = mk_multi_file_state(&[("main", code)], Require::indexing(), true);
    let handle = handles.get("main").unwrap();
    let report = get_unused_import_diagnostics(&state, handle);
    assert_eq!(report, "No unused imports");
}

#[test]
fn test_simple_import_unused() {
    let code = r#"
import os

def foo() -> str:
    return "hello"
"#;
    let (handles, state) = mk_multi_file_state(&[("main", code)], Require::indexing(), true);
    let handle = handles.get("main").unwrap();
    let report = get_unused_import_diagnostics(&state, handle);
    assert_eq!(report, "Import `os` is unused");
}

#[test]
fn test_from_import_used() {
    let code = r#"
from typing import List

def process(items: List[str]):
    return [item.upper() for item in items]
"#;
    let (handles, state) = mk_multi_file_state(&[("main", code)], Require::indexing(), true);
    let handle = handles.get("main").unwrap();
    let report = get_unused_import_diagnostics(&state, handle);
    assert_eq!(report, "No unused imports");
}

#[test]
fn test_from_import_unused() {
    let code = r#"
from typing import Dict, List

def process(items: List[str]):
    return [item.upper() for item in items]
"#;
    let (handles, state) = mk_multi_file_state(&[("main", code)], Require::indexing(), true);
    let handle = handles.get("main").unwrap();
    let report = get_unused_import_diagnostics(&state, handle);
    assert_eq!(report, "Import `Dict` is unused");
}

#[test]
fn test_star_import_not_reported_as_unused() {
    let code = r#"
from typing import *

def foo() -> str:
    return "hello"
"#;
    let (handles, state) = mk_multi_file_state(&[("main", code)], Require::indexing(), true);
    let handle = handles.get("main").unwrap();
    let report = get_unused_import_diagnostics(&state, handle);
    assert_eq!(report, "No unused imports");
}

#[test]
fn test_future_import_not_reported_as_unused() {
    let code = r#"
from __future__ import annotations

def foo() -> str:
    return "hello"
"#;
    let (handles, state) = mk_multi_file_state(&[("main", code)], Require::indexing(), true);
    let handle = handles.get("main").unwrap();
    let report = get_unused_import_diagnostics(&state, handle);
    assert_eq!(report, "No unused imports");
}

#[test]
fn test_future_import_with_alias_not_reported_as_unused() {
    let code = r#"
from __future__ import annotations as _annotations

def foo() -> str:
    return "hello"
"#;
    let (handles, state) = mk_multi_file_state(&[("main", code)], Require::indexing(), true);
    let handle = handles.get("main").unwrap();
    let report = get_unused_import_diagnostics(&state, handle);
    assert_eq!(report, "No unused imports");
}

#[test]
fn test_generator_with_send() {
    let code = r#"
from typing import Generator

def test() -> Generator[float, float, None]:
    new = yield 0.0
    while True:
        new = yield new - 1
"#;
    let (handles, state) = mk_multi_file_state(&[("main", code)], Require::indexing(), true);
    let handle = handles.get("main").unwrap();
    let report = get_unused_variable_diagnostics(&state, handle);
    assert_eq!(report, "No unused variables");
}

// TODO: x = 7 should be highlighted as unused
#[test]
fn test_reassignment_false_negative() {
    let code = r#"
def f():
    x = 5
    print(x)
    x = 7
"#;
    let (handles, state) = mk_multi_file_state(&[("main", code)], Require::indexing(), true);
    let handle = handles.get("main").unwrap();
    let report = get_unused_variable_diagnostics(&state, handle);
    assert_eq!(report, "No unused variables");
}

#[test]
fn test_loop_with_reassignment() {
    let code = r#"
def test_loop() -> str:
    foo = 0
    while True:
        if foo > 100:
            return f"bar {foo}"
        foo = foo + 1
        continue
"#;
    let (handles, state) = mk_multi_file_state(&[("main", code)], Require::indexing(), true);
    let handle = handles.get("main").unwrap();
    let report = get_unused_variable_diagnostics(&state, handle);
    assert_eq!(report, "No unused variables");
}

#[test]
fn test_loop_with_augmented_assignment() {
    let code = r#"
def test_loop_aug() -> str:
    foo = 0
    while True:
        if foo > 100:
            return f"bar {foo}"
        foo += 1
        continue
"#;
    let (handles, state) = mk_multi_file_state(&[("main", code)], Require::indexing(), true);
    let handle = handles.get("main").unwrap();
    let report = get_unused_variable_diagnostics(&state, handle);
    assert_eq!(report, "No unused variables");
}

#[test]
fn test_unused_variable_basic() {
    let code = r#"
def main():
    unused_var = "this is unused"
    used_var = "this is used"
    print(used_var)
"#;
    let (handles, state) = mk_multi_file_state(&[("main", code)], Require::indexing(), true);
    let handle = handles.get("main").unwrap();
    let report = get_unused_variable_diagnostics(&state, handle);
    assert_eq!(report, "Variable `unused_var` is unused");
}
