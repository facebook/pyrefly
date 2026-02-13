/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use serde_json::json;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_provide_type_unopened_file() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("basic"));
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    // NOTE: We do NOT call did_open on bar.py, testing that provide_type
    // can work with files that haven't been explicitly opened
    interaction
        .client
        .provide_type("bar.py", 7, 5)
        .expect_response(json!({
            "contents": [{
                "kind": "plaintext",
                "value": "typing.Literal[3]",
            }]
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_provide_type_unopened_file_with_dependencies() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("basic"));
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    // NOTE: We do NOT call did_open on foo.py, testing that provide_type
    // can work with files that have dependencies (like importing from bar.py)
    interaction
        .client
        .provide_type("foo.py", 6, 16)
        .expect_response(json!({
            "contents": [{
                "kind": "plaintext",
                "value": "type[bar.Bar]".to_owned()
            }]
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_provide_type_basic() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("basic"));
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("bar.py");

    interaction
        .client
        .provide_type("bar.py", 7, 5)
        .expect_response(json!({
            "contents": [{
                "kind": "plaintext",
                "value": "typing.Literal[3]",
            }]
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_provide_type() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("basic"));
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("foo.py");

    interaction
        .client
        .provide_type("foo.py", 6, 16)
        .expect_response(json!({
            "contents": [{
                "kind": "plaintext",
                "value": "type[bar.Bar]".to_owned()
            }]
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_provide_type_from_pyi_file() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("provide_type_pyi"));
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    // Test type from .pyi file - MyClass constructor
    interaction
        .client
        .provide_type("usage.py", 8, 11) // position of MyClass in "MyClass(42)"
        .expect_response(json!({
            "contents": [{
                "kind": "plaintext",
                "value": "(\n    self: types_stub.MyClass,\n    value: builtins.int\n) -> None"
            }]
        }))
        .unwrap();

    // Test variable type from .pyi file - result should be int (return type of get_value)
    interaction
        .client
        .provide_type("usage.py", 11, 0) // position of result variable
        .expect_response(json!({
            "contents": [{
                "kind": "plaintext",
                "value": "builtins.int"
            }]
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_provide_type_directly_from_pyi_file() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("provide_type_pyi"));
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    // Test accessing types directly from .pyi file without opening it
    interaction
        .client
        .provide_type("types_stub.pyi", 7, 6) // position of MyClass definition
        .expect_response(json!({
            "contents": [{
                "kind": "plaintext",
                "value": "type[types_stub.MyClass]"
            }]
        }))
        .unwrap();

    // Test constant definition in .pyi file
    interaction
        .client
        .provide_type("types_stub.pyi", 14, 0) // position of MY_CONSTANT
        .expect_response(json!({
            "contents": [{
                "kind": "plaintext",
                "value": "builtins.int"
            }]
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}
