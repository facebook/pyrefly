/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::report::pysa::context::ModuleContext;
use crate::report::pysa::is_test_module::is_test_module;
use crate::report::pysa::module::ModuleIds;
use crate::test::pysa::utils::create_state;
use crate::test::pysa::utils::get_handle_for_module_name;

fn test_is_test_module(python_code: &str, expected: bool) {
    let state = create_state("test", python_code);
    let transaction = state.transaction();
    let handles = transaction.handles();
    let module_ids = ModuleIds::new(&handles);

    let test_module_handle = get_handle_for_module_name("test", &transaction);
    let context = ModuleContext::create(test_module_handle, &transaction, &module_ids).unwrap();

    let result = is_test_module(&context);
    assert_eq!(result, expected);
}

#[test]
fn test_unittest_module() {
    test_is_test_module(
        r#"
import unittest

class TestExample(unittest.TestCase):
    def test_something(self):
        self.assertEqual(1, 1)
"#,
        true,
    );
}

#[test]
fn test_pytest_module() {
    test_is_test_module(
        r#"
import pytest  # type: ignore

def test_something():
    assert 1 == 1
"#,
        true,
    );
}

#[test]
fn test_pytest_module_without_test_function() {
    test_is_test_module(
        r#"
def helper_function():
    return 42
"#,
        false,
    );
}

#[test]
fn test_unittest_module_from_import() {
    test_is_test_module(
        r#"
from unittest import TestCase

class MyTest(TestCase):
    def test_method(self):
        pass
"#,
        true,
    );
}

#[test]
fn test_regular_module() {
    test_is_test_module(
        r#"
def regular_function():
    return "hello"

class RegularClass:
    def method(self):
        pass
"#,
        false,
    );
}

#[test]
fn test_module_with_test_function_but_no_pytest() {
    test_is_test_module(
        r#"
def test_something():
    assert 1 == 1
"#,
        false,
    );
}

#[test]
fn test_complex_unittest_hierarchy() {
    test_is_test_module(
        r#"
import unittest

class BaseTest(unittest.TestCase):
    pass

class SpecificTest(BaseTest):
    def test_specific_case(self):
        self.assertTrue(True)
"#,
        true,
    );
}

#[test]
fn test_mixed_pytest_and_regular_functions() {
    test_is_test_module(
        r#"
import pytest  # type: ignore

def helper():
    return 1

def test_with_helper():
    assert helper() == 1

def another_helper():
    return 2
"#,
        true,
    );
}
