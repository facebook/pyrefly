/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use serde_json::json;

use crate::object_model::InitializeSettings;
use crate::object_model::LspInteraction;
use crate::util::get_test_files_root;

#[test]
fn test_provide_type_unopened_file() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("provide_type"));
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
        .provide_type("bar.py", 9, 5)
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
    interaction.set_root(root.path().join("provide_type"));
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
                "value": "builtins.type[bar.Bar]".to_owned()
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
                "value": "(self: types_stub.MyClass, value: builtins.int) -> None"
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
                "value": "builtins.type[types_stub.MyClass]"
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

fn do_test(line: u32, col: u32, expected: &'static str) {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("provide_type"));
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("bar.py");

    interaction
        .client
        .provide_type("bar.py", line, col)
        .expect_response(json!({
            "contents": [{
                "kind": "plaintext",
                "value": expected,
            }]
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_provide_type() {
    do_test(8, 9, "builtins.type[bar.Bar]");
}

#[test]
fn test_provide_type_basic() {
    do_test(9, 5, "typing.Literal[3]");
}

#[test]
fn test_provide_type_nested_class() {
    do_test(11, 13, "builtins.type[bar.Bar.Baz]");
}

#[test]
fn test_provide_type_function_fully_qualified() {
    do_test(18, 9, "def bar.Bar.Baz.f(self: bar.Bar.Baz) -> None");
}

#[test]
fn test_provide_type_variable() {
    do_test(23, 4, "T@bar.f_generic1");
}

#[test]
fn test_provide_generic_signature() {
    do_test(27, 4, "(T@bar.f_generic2) -> T@bar.f_generic2");
}

#[test]
fn test_provide_generic_signature_with_declaration() {
    do_test(30, 0, "[T: builtins.int = builtins.bool](t: T) -> T");
}

#[test]
fn test_self() {
    do_test(13, 14, "typing.Self@bar.Bar.Baz");
}

#[test]
fn test_builtin_method_type() {
    do_test(
        38,
        8,
        "def builtins.int.bit_length(self: builtins.int) -> builtins.int",
    );
}

#[test]
fn test_builtin_function_type() {
    do_test(
        39,
        0,
        "def builtins.any(iterable: typing.Iterable[builtins.object], /) -> builtins.bool",
    );
}

#[test]
fn test_function_type_with_constraints() {
    do_test(
        42,
        8,
        "def bar.f_constraints[T: (builtins.bool, builtins.str) = builtins.bool](t: T) -> T",
    );
}

#[test]
fn test_bound_method() {
    do_test(19, 10, "def bar.Bar.Baz.f() -> None");
}

#[test]
fn test_super() {
    do_test(47, 10, "builtins.super[builtins.object, bar.S]");
}

#[test]
fn test_new_type() {
    // NOTE: This is a bit of a hack, because it's not actually a `builtins.type`,
    //  but the client should realize that when they look up the qname
    do_test(50, 1, "builtins.type[bar.N]");
}

#[test]
fn test_overload() {
    do_test(
        61,
        0,
        "Overload[def bar.overloaded_func(x: None) -> None, def bar.overloaded_func(x: builtins.int) -> None]",
    );
}

#[test]
fn test_generic_mixed_scopes() {
    do_test(
        69,
        14,
        "def bar.A.f1.B.f2[F2](self: typing.Self@bar.A.f1.B, x: F1@bar.A.f1, y: F2, a: T@bar.A, b: T2@bar.A.f1.B) -> None",
    );
}

#[test]
fn test_tuple() {
    do_test(
        72,
        0,
        "builtins.tuple[typing.Literal[1], typing.Literal[2]]",
    );
}

#[test]
fn test_tuple_item() {
    do_test(72, 11, "typing.Literal[2]");
}

#[test]
fn test_type_alias() {
    do_test(77, 6, "def bar.alias() -> bar.TA");
}

#[test]
fn test_pos() {
    do_test(86, 0, "typing.Literal[False]");
}
