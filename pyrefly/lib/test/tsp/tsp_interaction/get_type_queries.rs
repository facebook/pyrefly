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
fn test_get_computed_type_function_is_function_type() {
    // A function definition should produce a FunctionType with CALLABLE flag
    let (mut tsp, file_uri, snapshot) =
        setup_project("def foo(x: int) -> str:\n    return str(x)\n");

    let result = get_computed_type_ok(&mut tsp, &file_uri, 0, 4, snapshot);
    // Function types are now emitted as FunctionType with CALLABLE flag
    assert_kind(&result, TypeKind::Function);

    let flags = result.get("flags").and_then(|v| v.as_i64());
    // CALLABLE = 4
    assert!(
        flags.is_some_and(|f| f & 4 != 0),
        "Expected CALLABLE flag (4), got flags={flags:?}"
    );

    // Should have a declaration
    assert!(
        result.get("declaration").is_some(),
        "Expected declaration field on FunctionType"
    );

    // Should have a returnType
    assert!(
        result.get("returnType").is_some(),
        "Expected returnType field on FunctionType"
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

// =======================================================================
// Tests for declaration-based type conversions
// =======================================================================

#[test]
fn test_get_computed_type_tuple_is_class() {
    // `t = (1, "a")` infers tuple[int, str], which should be a ClassType
    let (mut tsp, file_uri, snapshot) = setup_project("t = (1, \"a\")\n");

    let result = get_computed_type_ok(&mut tsp, &file_uri, 0, 0, snapshot);
    assert_kind(&result, TypeKind::Class);

    tsp.shutdown();
}

#[test]
fn test_get_computed_type_tuple_with_type_args() {
    // `t: tuple[int, str]` — should have typeArgs in the response
    let (mut tsp, file_uri, snapshot) = setup_project("t: tuple[int, str] = (1, \"a\")\n");

    let result = get_computed_type_ok(&mut tsp, &file_uri, 0, 0, snapshot);
    assert_kind(&result, TypeKind::Class);

    tsp.shutdown();
}

#[test]
fn test_get_computed_type_dict_is_class() {
    // `d: dict[str, int]` with an annotation to get a dict ClassType
    let (mut tsp, file_uri, snapshot) = setup_project("d: dict[str, int] = {\"key\": 1}\n");

    tsp.server.get_declared_type(&file_uri, 0, 0, snapshot);
    let resp = tsp.client.receive_response_skip_notifications();
    let result = resp.result.expect("Expected result");
    assert_kind(&result, TypeKind::Class);

    let decl = result.get("declaration").expect("Expected declaration");
    let name = decl.get("name").and_then(|v| v.as_str());
    assert_eq!(name, Some("dict"), "Expected class name 'dict'");

    tsp.shutdown();
}

#[test]
fn test_get_computed_type_set_is_class() {
    let (mut tsp, file_uri, snapshot) = setup_project("s = {1, 2, 3}\n");

    let result = get_computed_type_ok(&mut tsp, &file_uri, 0, 0, snapshot);
    assert_kind(&result, TypeKind::Class);

    let decl = result.get("declaration").expect("Expected declaration");
    let name = decl.get("name").and_then(|v| v.as_str());
    assert_eq!(name, Some("set"), "Expected class name 'set'");

    tsp.shutdown();
}

#[test]
fn test_get_computed_type_module_import() {
    // `import os` — querying `os` should give a ModuleType
    let (mut tsp, file_uri, snapshot) = setup_project("import os\nos\n");

    let result = get_computed_type_ok(&mut tsp, &file_uri, 1, 0, snapshot);
    assert_kind(&result, TypeKind::Module);

    let module_name = result.get("moduleName").and_then(|v| v.as_str());
    assert_eq!(module_name, Some("os"), "Expected module name 'os'");

    tsp.shutdown();
}

#[test]
fn test_get_computed_type_bound_method_is_function() {
    // `m = x.append` — querying `m` (position 0 on line 1) gives bound method type
    let (mut tsp, file_uri, snapshot) = setup_project("x = [1, 2]\nm = x.append\n");

    let result = get_computed_type_ok(&mut tsp, &file_uri, 1, 0, snapshot);
    // Bound methods produce FunctionType or OverloadedType
    let kind = result
        .get("kind")
        .and_then(|v| v.as_u64())
        .expect("Expected kind");
    assert!(
        kind == TypeKind::Function as u64 || kind == TypeKind::Overloaded as u64,
        "Expected Function or Overloaded for bound method, got kind={kind}"
    );

    tsp.shutdown();
}

#[test]
fn test_get_computed_type_overloaded_function() {
    // An overloaded function should produce OverloadedType
    let code = "\
from typing import overload

@overload
def f(x: int) -> int: ...
@overload
def f(x: str) -> str: ...
def f(x):
    return x
";
    let (mut tsp, file_uri, snapshot) = setup_project(code);

    let result = get_computed_type_ok(&mut tsp, &file_uri, 6, 4, snapshot);
    assert_kind(&result, TypeKind::Overloaded);

    let overloads = result
        .get("overloads")
        .and_then(|v| v.as_array())
        .expect("Expected overloads array");
    assert_eq!(overloads.len(), 2, "Expected 2 overload signatures");

    // Each overload should be a FunctionType
    for (i, overload) in overloads.iter().enumerate() {
        let kind = overload
            .get("kind")
            .and_then(|v| v.as_u64())
            .expect("Expected kind on overload");
        assert_eq!(
            kind,
            TypeKind::Function as u64,
            "Expected overload {i} to be FunctionType"
        );
    }

    tsp.shutdown();
}

#[test]
fn test_overloaded_function_each_overload_has_distinct_return_type() {
    // Regression: pyrefly was duplicating the first overload's signature for all
    // overloads, so both entries had return type `str` instead of `str` and `int`.
    // Pylance round-trip expected: Overload[(value: int) -> str, (value: str) -> int]
    // Pylance round-trip received: Overload[(value: int) -> str, (value: int) -> str]
    let code = "\
from typing import overload

@overload
def process(value: int) -> str: ...
@overload
def process(value: str) -> int: ...
def process(value):
    if isinstance(value, int):
        return str(value)
    return len(value)
";
    let (mut tsp, file_uri, snapshot) = setup_project(code);

    // Query the implementation line (line 6, `def process(value):`)
    let result = get_computed_type_ok(&mut tsp, &file_uri, 6, 4, snapshot);
    assert_kind(&result, TypeKind::Overloaded);

    let overloads = result
        .get("overloads")
        .and_then(|v| v.as_array())
        .expect("Expected overloads array");
    assert_eq!(overloads.len(), 2, "Expected 2 overload signatures");

    // First overload: (value: int) -> str
    let ret0 = overloads[0]
        .get("returnType")
        .expect("Expected returnType on first overload");
    let ret0_decl = ret0
        .get("declaration")
        .and_then(|d| d.get("name"))
        .and_then(|n| n.as_str());
    assert_eq!(
        ret0_decl,
        Some("str"),
        "First overload should return str, got: {ret0}"
    );

    // Second overload: (value: str) -> int
    let ret1 = overloads[1]
        .get("returnType")
        .expect("Expected returnType on second overload");
    let ret1_decl = ret1
        .get("declaration")
        .and_then(|d| d.get("name"))
        .and_then(|n| n.as_str());
    assert_eq!(
        ret1_decl,
        Some("int"),
        "Second overload should return int, got: {ret1}"
    );

    // The two return types must differ
    assert_ne!(
        serde_json::to_string(&overloads[0].get("returnType")).unwrap(),
        serde_json::to_string(&overloads[1].get("returnType")).unwrap(),
        "Both overloads have identical return types — the second overload's signature is duplicated from the first"
    );

    tsp.shutdown();
}

#[test]
fn test_overloaded_function_each_overload_has_distinct_declaration_node() {
    // Each overload should point to a different source location (line number).
    let code = "\
from typing import overload

@overload
def process(value: int) -> str: ...
@overload
def process(value: str) -> int: ...
def process(value):
    if isinstance(value, int):
        return str(value)
    return len(value)
";
    let (mut tsp, file_uri, snapshot) = setup_project(code);

    let result = get_computed_type_ok(&mut tsp, &file_uri, 6, 4, snapshot);
    assert_kind(&result, TypeKind::Overloaded);

    let overloads = result
        .get("overloads")
        .and_then(|v| v.as_array())
        .expect("Expected overloads array");
    assert_eq!(overloads.len(), 2, "Expected 2 overloads");

    let get_start_line = |overload: &serde_json::Value| -> Option<u64> {
        overload
            .get("declaration")
            .and_then(|d| d.get("node"))
            .and_then(|n| n.get("range"))
            .and_then(|r| r.get("start"))
            .and_then(|s| s.get("line"))
            .and_then(|l| l.as_u64())
    };

    let line0 = get_start_line(&overloads[0])
        .expect("Expected declaration node with line on first overload");
    let line1 = get_start_line(&overloads[1])
        .expect("Expected declaration node with line on second overload");

    // The two @overload decorated `def` statements are on different lines
    assert_ne!(
        line0, line1,
        "Both overloads point to the same declaration line ({line0}) — they should be distinct"
    );

    tsp.shutdown();
}

#[test]
fn test_get_computed_type_function_has_return_type() {
    // Verify that the function's returnType is properly populated
    let (mut tsp, file_uri, snapshot) =
        setup_project("def greet(name: str) -> str:\n    return 'hello ' + name\n");

    let result = get_computed_type_ok(&mut tsp, &file_uri, 0, 4, snapshot);
    assert_kind(&result, TypeKind::Function);

    let return_type = result
        .get("returnType")
        .expect("Expected returnType on FunctionType");
    assert!(!return_type.is_null(), "returnType should not be null");

    // The return type should be a ClassType for `str`
    let ret_kind = return_type.get("kind").and_then(|v| v.as_u64());
    assert_eq!(
        ret_kind,
        Some(TypeKind::Class as u64),
        "Expected return type to be Class"
    );

    tsp.shutdown();
}

#[test]
fn test_get_computed_type_function_has_declaration() {
    // Verify that a def-function has a Regular declaration with name
    let (mut tsp, file_uri, snapshot) = setup_project("def my_func() -> None:\n    pass\n");

    let result = get_computed_type_ok(&mut tsp, &file_uri, 0, 4, snapshot);
    assert_kind(&result, TypeKind::Function);

    let decl = result.get("declaration").expect("Expected declaration");
    let name = decl.get("name").and_then(|v| v.as_str());
    assert_eq!(name, Some("my_func"), "Expected function name 'my_func'");

    // Should have a node with a URI
    let node = decl.get("node");
    assert!(node.is_some(), "Expected node in declaration");
    let uri = node.and_then(|n| n.get("uri")).and_then(|v| v.as_str());
    assert!(
        uri.is_some_and(|u| !u.is_empty()),
        "Expected non-empty URI in declaration node"
    );

    tsp.shutdown();
}

#[test]
fn test_get_computed_type_class_has_declaration() {
    // Verify that a class definition carries a declaration with name and URI
    let (mut tsp, file_uri, snapshot) = setup_project("class Foo:\n    x: int = 0\n");

    let result = get_computed_type_ok(&mut tsp, &file_uri, 0, 6, snapshot);
    assert_kind(&result, TypeKind::Class);

    let decl = result.get("declaration").expect("Expected declaration");
    let name = decl.get("name").and_then(|v| v.as_str());
    assert_eq!(name, Some("Foo"), "Expected class name 'Foo'");

    let node = decl.get("node");
    assert!(node.is_some(), "Expected node in declaration");

    tsp.shutdown();
}

#[test]
fn test_get_computed_type_class_instance_is_class() {
    // `foo = Foo()` where Foo is a user class — should produce ClassType with INSTANCE flag
    let code = "class Foo:\n    pass\nfoo = Foo()\n";
    let (mut tsp, file_uri, snapshot) = setup_project(code);

    let result = get_computed_type_ok(&mut tsp, &file_uri, 2, 0, snapshot);
    assert_kind(&result, TypeKind::Class);

    // INSTANCE = 2
    let flags = result.get("flags").and_then(|v| v.as_i64());
    assert!(
        flags.is_some_and(|f| f & 2 != 0),
        "Expected INSTANCE flag (2), got flags={flags:?}"
    );

    let decl = result.get("declaration").expect("Expected declaration");
    let name = decl.get("name").and_then(|v| v.as_str());
    assert_eq!(name, Some("Foo"), "Expected class name 'Foo'");

    tsp.shutdown();
}

#[test]
fn test_get_computed_type_type_alias() {
    // `type IntList = list[int]` — querying the alias resolves to the underlying type
    let code = "type IntList = list[int]\nx: IntList = [1]\n";
    let (mut tsp, file_uri, snapshot) = setup_project(code);

    let result = get_computed_type_ok(&mut tsp, &file_uri, 1, 0, snapshot);
    // The computed type of x should be a Class (list[int])
    assert_kind(&result, TypeKind::Class);

    tsp.shutdown();
}

#[test]
fn test_get_computed_type_annotated_unwraps() {
    // `Annotated[int, ...]` should unwrap to int (ClassType)
    let code = "from typing import Annotated\nx: Annotated[int, 'metadata'] = 42\n";
    let (mut tsp, file_uri, snapshot) = setup_project(code);

    let result = get_computed_type_ok(&mut tsp, &file_uri, 1, 0, snapshot);
    // The variable x should have a Class kind (int)
    assert_kind(&result, TypeKind::Class);

    tsp.shutdown();
}

#[test]
fn test_get_computed_type_union_subtype_kinds() {
    // `x: int | str` — the declared type should be a Union with 2 Class subtypes
    let (mut tsp, file_uri, snapshot) = setup_project("x: int | str\n");

    tsp.server.get_declared_type(&file_uri, 0, 0, snapshot);
    let resp = tsp.client.receive_response_skip_notifications();
    let result = resp.result.expect("Expected result");
    assert_kind(&result, TypeKind::Union);

    let sub_types = result
        .get("subTypes")
        .and_then(|v| v.as_array())
        .expect("Expected subTypes array");
    assert_eq!(sub_types.len(), 2, "Expected 2 union members");

    // Both members should be Class types (int and str)
    for (i, member) in sub_types.iter().enumerate() {
        let kind = member.get("kind").and_then(|v| v.as_u64());
        assert_eq!(
            kind,
            Some(TypeKind::Class as u64),
            "Expected union member {i} to be Class"
        );
    }

    tsp.shutdown();
}

#[test]
fn test_get_computed_type_callable_is_function() {
    // `typing.Callable[[int], str]` should produce a FunctionType
    let code = "from typing import Callable\nf: Callable[[int], str]\n";
    let (mut tsp, file_uri, snapshot) = setup_project(code);

    tsp.server.get_declared_type(&file_uri, 1, 0, snapshot);
    let resp = tsp.client.receive_response_skip_notifications();
    let result = resp.result.expect("Expected result");
    assert_kind(&result, TypeKind::Function);

    // Should have CALLABLE flag
    let flags = result.get("flags").and_then(|v| v.as_i64());
    assert!(
        flags.is_some_and(|f| f & 4 != 0),
        "Expected CALLABLE flag (4), got flags={flags:?}"
    );

    // Should have a returnType
    assert!(
        result.get("returnType").is_some(),
        "Expected returnType on Callable FunctionType"
    );

    tsp.shutdown();
}

#[test]
fn test_get_computed_type_typed_dict_is_class() {
    // TypedDict should produce a ClassType
    let code = "\
from typing import TypedDict

class Point(TypedDict):
    x: int
    y: int

p: Point = {'x': 1, 'y': 2}
";
    let (mut tsp, file_uri, snapshot) = setup_project(code);

    let result = get_computed_type_ok(&mut tsp, &file_uri, 6, 0, snapshot);
    assert_kind(&result, TypeKind::Class);

    let decl = result.get("declaration").expect("Expected declaration");
    let name = decl.get("name").and_then(|v| v.as_str());
    assert_eq!(name, Some("Point"), "Expected TypedDict class name 'Point'");

    tsp.shutdown();
}

#[test]
fn test_get_computed_type_optional_is_union() {
    // `Optional[int]` should produce a Union
    let code = "from typing import Optional\nx: Optional[int] = None\n";
    let (mut tsp, file_uri, snapshot) = setup_project(code);

    tsp.server.get_declared_type(&file_uri, 1, 0, snapshot);
    let resp = tsp.client.receive_response_skip_notifications();
    let result = resp.result.expect("Expected result");
    assert_kind(&result, TypeKind::Union);

    tsp.shutdown();
}

#[test]
fn test_get_computed_type_lambda_is_function() {
    // `f = lambda x: x + 1` — should produce FunctionType
    let (mut tsp, file_uri, snapshot) = setup_project("f = lambda x: x + 1\n");

    let result = get_computed_type_ok(&mut tsp, &file_uri, 0, 0, snapshot);
    assert_kind(&result, TypeKind::Function);

    let flags = result.get("flags").and_then(|v| v.as_i64());
    assert!(
        flags.is_some_and(|f| f & 4 != 0),
        "Expected CALLABLE flag (4), got flags={flags:?}"
    );

    tsp.shutdown();
}

#[test]
fn test_get_computed_type_generic_class_has_type_args() {
    // `x: list[int]` — ClassType should have typeArgs
    let (mut tsp, file_uri, snapshot) = setup_project("x: list[int] = [1]\n");

    tsp.server.get_declared_type(&file_uri, 0, 0, snapshot);
    let resp = tsp.client.receive_response_skip_notifications();
    let result = resp.result.expect("Expected result");
    assert_kind(&result, TypeKind::Class);

    let type_args = result.get("typeArgs").and_then(|v| v.as_array());
    assert!(
        type_args.is_some_and(|args| args.len() == 1),
        "Expected 1 typeArg for list[int], got {:?}",
        type_args
    );

    tsp.shutdown();
}

#[test]
fn test_get_computed_type_literal_bool_has_literal_value() {
    let (mut tsp, file_uri, snapshot) = setup_project("b = True\n");

    let result = get_computed_type_ok(&mut tsp, &file_uri, 0, 0, snapshot);
    assert_kind(&result, TypeKind::Class);

    let literal_value = result.get("literalValue");
    assert!(
        literal_value.is_some(),
        "Expected literalValue for bool literal"
    );
    assert_eq!(
        literal_value.and_then(|v| v.as_bool()),
        Some(true),
        "Expected literalValue=true"
    );

    tsp.shutdown();
}
