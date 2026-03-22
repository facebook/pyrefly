/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Integration tests for the `typeServer/getDeclaredType`,
//! `typeServer/getComputedType`, and `typeServer/getExpectedType` TSP requests.

use lsp_types::Url;
use tempfile::TempDir;
use tsp_types::TypeKind;

use crate::test::tsp::tsp_interaction::object_model::TspInteraction;
use crate::test::tsp::tsp_interaction::object_model::get_current_snapshot;
use crate::test::tsp::tsp_interaction::object_model::write_pyproject;

/// Set up a project with a single Python file and return (tsp, file_uri, snapshot).
fn setup_project(file_content: &str) -> (TspInteraction, String, i32) {
    let temp_dir = TempDir::new().unwrap();
    write_pyproject(temp_dir.path());

    let test_file = temp_dir.path().join("main.py");
    std::fs::write(&test_file, file_content).unwrap();

    let mut tsp = TspInteraction::new();
    tsp.set_root(temp_dir.path().to_path_buf());
    tsp.initialize(Default::default());

    tsp.server.did_open("main.py");
    tsp.client.expect_any_message();

    let snapshot = get_current_snapshot(&mut tsp, 2);
    let file_uri = Url::from_file_path(&test_file).unwrap().to_string();

    (tsp, file_uri, snapshot)
}

/// Helper to extract the "kind" field from a type query result.
fn assert_kind(result: &serde_json::Value, expected_kind: TypeKind) {
    let kind = result
        .get("kind")
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| panic!("Expected 'kind' field (integer) in type result: {result}"));
    let expected = expected_kind as u64;
    assert_eq!(
        kind, expected,
        "Expected kind={expected_kind:?} ({expected}), got kind={kind} in: {result}"
    );
}

/// Helper to send a getComputedType request and return a successful result.
fn get_computed_type_ok(
    tsp: &mut TspInteraction,
    file_uri: &str,
    line: u32,
    character: u32,
    snapshot: i32,
) -> serde_json::Value {
    tsp.server
        .get_computed_type(file_uri, line, character, snapshot);
    let resp = tsp.client.receive_response_skip_notifications();
    assert!(
        resp.error.is_none(),
        "Expected success, got error: {:?}",
        resp.error
    );
    let result = resp.result.expect("Expected result");
    assert!(!result.is_null(), "Expected non-null type result");
    result
}

// =======================================================================
// getDeclaredType
// =======================================================================

#[test]
fn test_get_declared_type_int_annotation() {
    let (mut tsp, file_uri, snapshot) = setup_project("x: int = 42\n");

    tsp.server.get_declared_type(&file_uri, 0, 0, snapshot);

    let resp = tsp.client.receive_response_skip_notifications();
    assert!(
        resp.error.is_none(),
        "Expected success, got error: {:?}",
        resp.error
    );
    let result = resp.result.expect("Expected result");
    assert!(!result.is_null(), "Expected non-null type");
    assert_kind(&result, TypeKind::Class);

    tsp.shutdown();
}

#[test]
fn test_get_declared_type_stale_snapshot() {
    let (mut tsp, file_uri, _snapshot) = setup_project("x = 1\n");

    tsp.server.get_declared_type(&file_uri, 0, 0, 9999);

    let resp = tsp.client.receive_response_skip_notifications();
    assert!(
        resp.error.is_some(),
        "Expected error for stale snapshot, got success: {:?}",
        resp.result
    );

    tsp.shutdown();
}

#[test]
fn test_get_declared_type_invalid_params() {
    let (tsp, _file_uri, _snapshot) = setup_project("x = 1\n");

    tsp.server
        .send_message(crate::lsp::non_wasm::protocol::Message::Request(
            crate::lsp::non_wasm::protocol::Request {
                id: lsp_server::RequestId::from(99),
                method: "typeServer/getDeclaredType".to_owned(),
                params: serde_json::json!({ "bad": "params" }),
                activity_key: None,
            },
        ));

    let resp = tsp.client.receive_response_skip_notifications();
    assert!(
        resp.error.is_some(),
        "Expected error for invalid params, got success"
    );

    tsp.shutdown();
}

// =======================================================================
// getComputedType — assertions on kind and structure
// =======================================================================

#[test]
fn test_get_computed_type_int_is_class() {
    // `x = 42` infers Literal[42], which is a Class with literalValue
    let (mut tsp, file_uri, snapshot) = setup_project("x = 42\n");

    let result = get_computed_type_ok(&mut tsp, &file_uri, 0, 0, snapshot);
    assert_kind(&result, TypeKind::Class);

    // Literal types carry a literalValue field
    let literal_value = result.get("literalValue");
    assert!(
        literal_value.is_some(),
        "Expected literalValue for int literal, got: {result}"
    );
    assert_eq!(
        literal_value.and_then(|v| v.as_i64()),
        Some(42),
        "Expected literalValue=42"
    );

    tsp.shutdown();
}

#[test]
fn test_get_computed_type_string_is_class() {
    // `s = "hello"` infers Literal["hello"], which is a Class with literalValue
    let (mut tsp, file_uri, snapshot) = setup_project("s = \"hello\"\n");

    let result = get_computed_type_ok(&mut tsp, &file_uri, 0, 0, snapshot);
    assert_kind(&result, TypeKind::Class);

    let literal_value = result.get("literalValue");
    assert!(
        literal_value.is_some(),
        "Expected literalValue for string literal, got: {result}"
    );
    assert_eq!(
        literal_value.and_then(|v| v.as_str()),
        Some("hello"),
        "Expected literalValue='hello'"
    );

    tsp.shutdown();
}

#[test]
fn test_get_computed_type_none_is_builtin() {
    let (mut tsp, file_uri, snapshot) = setup_project("x = None\n");

    let result = get_computed_type_ok(&mut tsp, &file_uri, 0, 0, snapshot);
    assert_kind(&result, TypeKind::Builtin);

    let name = result.get("name").and_then(|v| v.as_str());
    assert_eq!(name, Some("none"), "Expected builtin name 'none'");

    tsp.shutdown();
}

#[test]
fn test_get_computed_type_bool_is_class() {
    let (mut tsp, file_uri, snapshot) = setup_project("b = True\n");

    let result = get_computed_type_ok(&mut tsp, &file_uri, 0, 0, snapshot);
    assert_kind(&result, TypeKind::Class);

    tsp.shutdown();
}

#[test]
fn test_get_computed_type_list_is_class() {
    let (mut tsp, file_uri, snapshot) = setup_project("xs = [1, 2, 3]\n");

    let result = get_computed_type_ok(&mut tsp, &file_uri, 0, 0, snapshot);
    assert_kind(&result, TypeKind::Class);

    let decl = result.get("declaration").expect("Expected declaration");
    let name = decl.get("name").and_then(|v| v.as_str());
    assert_eq!(name, Some("list"), "Expected class name 'list'");

    tsp.shutdown();
}

#[test]
fn test_get_computed_type_function_is_synthesized() {
    // A function definition should produce a Synthesized or Callable-flagged type
    let (mut tsp, file_uri, snapshot) =
        setup_project("def foo(x: int) -> str:\n    return str(x)\n");

    let result = get_computed_type_ok(&mut tsp, &file_uri, 0, 4, snapshot);
    // Function types are emitted as SynthesizedType with CALLABLE flag
    assert_kind(&result, TypeKind::Synthesized);

    let flags = result.get("flags").and_then(|v| v.as_i64());
    // CALLABLE = 4
    assert!(
        flags.is_some_and(|f| f & 4 != 0),
        "Expected CALLABLE flag (4), got flags={flags:?}"
    );

    tsp.shutdown();
}

#[test]
fn test_get_computed_type_union_annotation() {
    // `x: int | str` → querying `x` should give Union
    let (mut tsp, file_uri, snapshot) = setup_project("x: int | str = 42\n");

    let result = get_computed_type_ok(&mut tsp, &file_uri, 0, 0, snapshot);
    // The computed type of x could be Class(int) since 42 narrows the union,
    // but this depends on pyrefly's inference. Let's just verify the response
    // is a valid type with a known kind.
    let kind = result
        .get("kind")
        .and_then(|v| v.as_u64())
        .expect("Expected kind");
    assert!(
        kind == TypeKind::Class as u64 || kind == TypeKind::Union as u64,
        "Expected Class or Union for union-annotated var, got {kind}"
    );

    tsp.shutdown();
}

#[test]
fn test_get_computed_type_class_definition() {
    // The class name itself should be Instantiable (type[MyClass])
    let (mut tsp, file_uri, snapshot) = setup_project("class MyClass:\n    pass\n");

    let result = get_computed_type_ok(&mut tsp, &file_uri, 0, 6, snapshot);
    assert_kind(&result, TypeKind::Class);

    // INSTANTIABLE = 1
    let flags = result.get("flags").and_then(|v| v.as_i64());
    assert!(
        flags.is_some_and(|f| f & 1 != 0),
        "Expected INSTANTIABLE flag (1) for class definition, got flags={flags:?}"
    );

    tsp.shutdown();
}

#[test]
fn test_get_computed_type_has_id_field() {
    // Every type result must have an "id" field (positive integer)
    let (mut tsp, file_uri, snapshot) = setup_project("x = 42\n");

    let result = get_computed_type_ok(&mut tsp, &file_uri, 0, 0, snapshot);
    let id = result.get("id").and_then(|v| v.as_i64());
    assert!(
        id.is_some_and(|i| i > 0),
        "Expected positive 'id' field, got {id:?}"
    );

    tsp.shutdown();
}

// =======================================================================
// getExpectedType
// =======================================================================

#[test]
fn test_get_expected_type_str_annotation() {
    let (mut tsp, file_uri, snapshot) = setup_project("y: str = \"world\"\n");

    tsp.server.get_expected_type(&file_uri, 0, 0, snapshot);

    let resp = tsp.client.receive_response_skip_notifications();
    assert!(
        resp.error.is_none(),
        "Expected success, got error: {:?}",
        resp.error
    );
    let result = resp.result.expect("Expected result");
    assert!(!result.is_null(), "Expected non-null type");
    // The variable `y` should have a Class kind
    assert_kind(&result, TypeKind::Class);

    tsp.shutdown();
}

#[test]
fn test_get_expected_type_stale_snapshot() {
    let (mut tsp, file_uri, _snapshot) = setup_project("x = 1\n");

    tsp.server.get_expected_type(&file_uri, 0, 0, 9999);

    let resp = tsp.client.receive_response_skip_notifications();
    assert!(resp.error.is_some(), "Expected error for stale snapshot");

    tsp.shutdown();
}

// =======================================================================
// Cross-cutting: all three methods agree
// =======================================================================

#[test]
fn test_all_three_methods_return_same_kind_for_simple_var() {
    // For `x = 42`, all three type queries should return a Class type
    let (mut tsp, file_uri, snapshot) = setup_project("x = 42\n");

    tsp.server.get_declared_type(&file_uri, 0, 0, snapshot);
    let resp1 = tsp.client.receive_response_skip_notifications();
    let r1 = resp1.result.expect("declared result");

    tsp.server.get_computed_type(&file_uri, 0, 0, snapshot);
    let resp2 = tsp.client.receive_response_skip_notifications();
    let r2 = resp2.result.expect("computed result");

    tsp.server.get_expected_type(&file_uri, 0, 0, snapshot);
    let resp3 = tsp.client.receive_response_skip_notifications();
    let r3 = resp3.result.expect("expected result");

    // All should have the same kind
    let kind1 = r1.get("kind").and_then(|v| v.as_u64()).unwrap();
    let kind2 = r2.get("kind").and_then(|v| v.as_u64()).unwrap();
    let kind3 = r3.get("kind").and_then(|v| v.as_u64()).unwrap();
    assert_eq!(kind1, kind2, "declared vs computed kind mismatch");
    assert_eq!(kind2, kind3, "computed vs expected kind mismatch");

    tsp.shutdown();
}
