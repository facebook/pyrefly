/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests for the CinderX type report.

use pretty_assertions::assert_eq;

use crate::report::cinderx::collect::collect_module_types;
use crate::report::cinderx::display::format_module_types;
use crate::report::cinderx::types::StructuredType;
use crate::report::cinderx::write_results;
use crate::state::require::Require;
use crate::test::util::TestEnv;

/// Minimal `__static__` stub for CinderX primitive type tests.
/// In production, the real `__static__` package is provided by CinderX.
const STATIC_MODULE_STUB: &str = r#"
class int8(int): pass
class int16(int): pass
class int32(int): pass
class int64(int): pass
class uint8(int): pass
class uint16(int): pass
class uint32(int): pass
class uint64(int): pass
class cbool(int): pass
class char(int): pass
class double(float): pass
class single(float): pass
"#;

// ---------------------------------------------------------------------------
// Fixture-based expect tests
// ---------------------------------------------------------------------------

/// Resolve the `fixtures/` directory from the current working directory.
///
/// Buck and cargo run tests from different roots, so we try several candidate
/// paths. The function panics if none of the candidates exist.
fn fixtures_dir() -> std::path::PathBuf {
    let cwd = std::env::current_dir().expect("cwd should be available");
    let mut candidates = vec![
        cwd.join("fbcode/pyrefly/pyrefly/lib/test/cinderx/fixtures"),
        cwd.join("pyrefly/lib/test/cinderx/fixtures"),
        cwd.join("lib/test/cinderx/fixtures"),
    ];
    if let Some(manifest_dir) = option_env!("CARGO_MANIFEST_DIR") {
        candidates.push(std::path::Path::new(manifest_dir).join("lib/test/cinderx/fixtures"));
    }
    candidates
        .into_iter()
        .find(|p| p.exists())
        .unwrap_or_else(|| panic!("cinderx fixtures directory not found; tried multiple paths"))
}

/// Run a fixture-based expect test.
///
/// Reads `fixtures/<name>.py`, type-checks it (with the `__static__` stub
/// always loaded), renders the output with `format_module_types`, then
/// compares it against `fixtures/<name>.expected`.
///
/// Set the environment variable `UPDATE_EXPECT=1` to (re)generate the
/// `.expected` file instead of asserting equality. Use this when adding a
/// new fixture or when intentionally changing Pyrefly's output.
fn check_fixture(name: &str) {
    let fixtures = fixtures_dir();

    let py_path = fixtures.join(format!("{name}.py"));
    let source = std::fs::read_to_string(&py_path)
        .unwrap_or_else(|e| panic!("could not read fixture {py_path:?}: {e}"));

    let state = create_state_with_static(name, &source);
    let transaction = state.transaction();
    let handle = get_handle(name, &transaction);
    let data = collect_module_types(&transaction, &handle).expect("should collect types");
    let actual = format_module_types(&data.entries, &data.locations);

    let expected_path = fixtures.join(format!("{name}.expected"));

    if std::env::var("UPDATE_EXPECT").is_ok() {
        std::fs::write(&expected_path, &actual)
            .unwrap_or_else(|e| panic!("could not write {expected_path:?}: {e}"));
        return;
    }

    let expected = std::fs::read_to_string(&expected_path)
        .unwrap_or_else(|_| {
            panic!(
                "expected file {expected_path:?} not found.\n\
                 To create it, run: UPDATE_EXPECT=1 cargo test {name}\n\
                 Actual output:\n{actual}"
            )
        })
        // Normalize Windows line endings so tests pass on all platforms.
        .replace("\r\n", "\n");

    assert_eq!(
        expected, actual,
        "fixture output mismatch for `{name}`.\n\
         To update, run: UPDATE_EXPECT=1 cargo test {name}"
    );
}

// ---------------------------------------------------------------------------
// Fixture tests
// ---------------------------------------------------------------------------

#[test]
fn test_fixture_function_types() {
    check_fixture("function_types");
}

#[test]
fn test_fixture_simple_variable() {
    check_fixture("simple_variable");
}

#[test]
fn test_fixture_optional_type() {
    check_fixture("optional_type");
}

#[test]
fn test_fixture_literal_type() {
    check_fixture("literal_type");
}

#[test]
fn test_fixture_class_with_type_args() {
    check_fixture("class_with_type_args");
}

#[test]
fn test_fixture_bound_method_defining_class() {
    check_fixture("bound_method_defining_class");
}

#[test]
fn test_fixture_callable_defining_func() {
    check_fixture("callable_defining_func");
}

#[test]
fn test_fixture_callable_defining_func_method() {
    check_fixture("callable_defining_func_method");
}

#[test]
fn test_fixture_facet_narrow_mismatch() {
    check_fixture("facet_narrow_mismatch");
}

#[test]
fn test_fixture_facet_narrow_no_mismatch() {
    check_fixture("facet_narrow_no_mismatch");
}

#[test]
fn test_fixture_facet_narrow_chained_attr() {
    check_fixture("facet_narrow_chained_attr");
}

#[test]
fn test_fixture_no_facets_no_reresolution() {
    check_fixture("no_facets_no_reresolution");
}

#[test]
fn test_fixture_facet_narrow_index() {
    check_fixture("facet_narrow_index");
}

#[test]
fn test_fixture_facet_narrow_key() {
    check_fixture("facet_narrow_key");
}

#[test]
fn test_fixture_facet_narrow_mixed_chain() {
    check_fixture("facet_narrow_mixed_chain");
}

#[test]
fn test_fixture_static_int64_literal_contextual_type() {
    check_fixture("static_int64_literal_contextual_type");
}

#[test]
fn test_fixture_static_assign_after_annotation() {
    check_fixture("static_assign_after_annotation");
}

#[test]
fn test_fixture_static_assign_without_prior_value() {
    check_fixture("static_assign_without_prior_value");
}

#[test]
fn test_fixture_static_double_literal_contextual_type() {
    check_fixture("static_double_literal_contextual_type");
}

#[test]
fn test_fixture_static_call_positional_arg() {
    check_fixture("static_call_positional_arg");
}

#[test]
fn test_fixture_static_call_bound_method_arg() {
    check_fixture("static_call_bound_method_arg");
}

#[test]
fn test_fixture_static_call_multiple_args() {
    check_fixture("static_call_multiple_args");
}

#[test]
fn test_fixture_static_attr_assign() {
    check_fixture("static_attr_assign");
}

#[test]
fn test_fixture_static_attr_ann_assign() {
    check_fixture("static_attr_ann_assign");
}

#[test]
fn test_fixture_static_int64_constructor_call() {
    check_fixture("static_int64_constructor_call");
}

#[test]
fn test_fixture_static_call_keyword_arg() {
    check_fixture("static_call_keyword_arg");
}

#[test]
fn test_fixture_static_call_mixed_args() {
    check_fixture("static_call_mixed_args");
}

// ---------------------------------------------------------------------------
// Property-based unit tests
// ---------------------------------------------------------------------------

/// Create a type-checked state with the `__static__` stub and multiple modules.
fn create_state_multi_module(modules: &[(&str, &str)]) -> crate::state::state::State {
    let mut test_env = TestEnv::new();
    test_env.add("__static__", STATIC_MODULE_STUB);
    for (name, code) in modules {
        test_env.add(name, code);
    }
    let (state, _) = test_env
        .with_default_require_level(Require::Everything)
        .to_state();
    state
}

/// Create a type-checked state from a single module's Python source.
fn create_state(module_name: &str, python_code: &str) -> crate::state::state::State {
    let mut test_env = TestEnv::new();
    test_env.add(module_name, python_code);
    let (state, _) = test_env
        .with_default_require_level(Require::Everything)
        .to_state();
    state
}

/// Create a type-checked state with the `__static__` stub and a test module.
fn create_state_with_static(module_name: &str, python_code: &str) -> crate::state::state::State {
    let mut test_env = TestEnv::new();
    test_env.add("__static__", STATIC_MODULE_STUB);
    test_env.add(module_name, python_code);
    let (state, _) = test_env
        .with_default_require_level(Require::Everything)
        .to_state();
    state
}

/// Find the handle for a module by name in a transaction.
fn get_handle(
    module_name: &str,
    transaction: &crate::state::state::Transaction,
) -> pyrefly_build::handle::Handle {
    transaction
        .handles()
        .into_iter()
        .find(|h| h.module().as_str() == module_name)
        .unwrap_or_else(|| panic!("module `{module_name}` not found"))
}

#[test]
fn test_deduplication() {
    let state = create_state(
        "test",
        r#"
x: int = 1
y: int = 2
"#,
    );
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // `builtins.int` should appear only once in the type table despite being used twice
    let int_count = data.entries.iter().filter(|entry| {
        matches!(&entry.ty, StructuredType::Class { qname, args, .. } if qname == "builtins.int" && args.is_empty())
    }).count();
    assert_eq!(
        int_count, 1,
        "expected exactly one `builtins.int` entry (deduplication)"
    );
}

#[test]
fn test_json_serialization() {
    let state = create_state("test", "x: int = 42");
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // Verify the data can be serialized to JSON without errors
    let entries_json =
        serde_json::to_string_pretty(&data.entries).expect("type table should be serializable");
    let locations_json =
        serde_json::to_string_pretty(&data.locations).expect("locations should be serializable");

    // Basic sanity: JSON should contain expected strings
    assert!(
        entries_json.contains("\"kind\""),
        "entries JSON should have 'kind' field"
    );
    assert!(
        locations_json.contains("\"loc\""),
        "locations JSON should have 'loc' field"
    );
    assert!(
        locations_json.contains("\"type\""),
        "locations JSON should have 'type' field"
    );
}

#[test]
fn test_mro_collection() {
    let state = create_state(
        "test",
        r#"
class Base:
    pass

class Child(Base):
    pass

x: Child = Child()
"#,
    );
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // The classes list should contain the classes we referenced
    assert!(
        !data.classes.is_empty(),
        "expected at least one class in the classes list"
    );
}

#[test]
fn test_mro_in_report() {
    let state = create_state(
        "test",
        r#"
class Base:
    pass

class Child(Base):
    pass

x: Child = Child()
"#,
    );
    let transaction = state.transaction();
    let output_dir = tempfile::tempdir().expect("should create temp dir");
    let handles = transaction.handles();
    write_results(output_dir.path(), &transaction, &handles, false).expect("should write results");

    // Read and parse class_metadata.json
    let metadata_json = std::fs::read_to_string(output_dir.path().join("class_metadata.json"))
        .expect("class_metadata.json should exist");
    let metadata: serde_json::Value =
        serde_json::from_str(&metadata_json).expect("class_metadata.json should be valid JSON");

    let entries = metadata["entries"]
        .as_array()
        .expect("entries should be an array");

    // Find the Child class entry
    let child_entry = entries
        .iter()
        .find(|e| e["qname"].as_str() == Some("test.Child"));
    assert!(
        child_entry.is_some(),
        "expected entry for test.Child, got entries: {metadata_json}"
    );

    let ancestors = child_entry.unwrap()["ancestors"]
        .as_array()
        .expect("ancestors should be an array");
    let ancestor_names: Vec<&str> = ancestors
        .iter()
        .map(|a| a.as_str().expect("ancestor should be a string"))
        .collect();
    assert!(
        ancestor_names.contains(&"test.Base"),
        "expected test.Base in Child's MRO, got: {ancestor_names:?}"
    );

    // Non-protocol classes should have no tags
    let child_tags = child_entry.unwrap().get("tags");
    assert!(
        child_tags.is_none() || child_tags.unwrap().as_array().unwrap().is_empty(),
        "expected no tags on non-protocol class"
    );
}

#[test]
fn test_protocol_tags() {
    let state = create_state(
        "test",
        r#"
from typing import Protocol

class MyProto(Protocol):
    def method(self) -> int: ...

class Impl(MyProto):
    def method(self) -> int:
        return 42

x: MyProto = Impl()
y: Impl = Impl()
"#,
    );
    let transaction = state.transaction();
    let output_dir = tempfile::tempdir().expect("should create temp dir");
    let handles = transaction.handles();
    write_results(output_dir.path(), &transaction, &handles, false).expect("should write results");

    let metadata_json = std::fs::read_to_string(output_dir.path().join("class_metadata.json"))
        .expect("class_metadata.json should exist");
    let metadata: serde_json::Value =
        serde_json::from_str(&metadata_json).expect("class_metadata.json should be valid JSON");

    let entries = metadata["entries"]
        .as_array()
        .expect("entries should be an array");

    // MyProto should have the "protocol" tag
    let proto_entry = entries
        .iter()
        .find(|e| e["qname"].as_str() == Some("test.MyProto"));
    assert!(
        proto_entry.is_some(),
        "expected entry for test.MyProto, got entries: {metadata_json}"
    );
    let proto_tags: Vec<&str> = proto_entry.unwrap()["tags"]
        .as_array()
        .expect("tags should be an array")
        .iter()
        .map(|t| t.as_str().expect("tag should be a string"))
        .collect();
    assert!(
        proto_tags.contains(&"protocol"),
        "expected 'protocol' tag on MyProto, got: {proto_tags:?}"
    );

    // Impl should have the "inherits_protocol" tag
    let impl_entry = entries
        .iter()
        .find(|e| e["qname"].as_str() == Some("test.Impl"));
    assert!(
        impl_entry.is_some(),
        "expected entry for test.Impl, got entries: {metadata_json}"
    );
    let impl_tags: Vec<&str> = impl_entry.unwrap()["tags"]
        .as_array()
        .expect("tags should be an array")
        .iter()
        .map(|t| t.as_str().expect("tag should be a string"))
        .collect();
    assert!(
        impl_tags.contains(&"inherits_protocol"),
        "expected 'inherits_protocol' tag on Impl, got: {impl_tags:?}"
    );
}

#[test]
fn test_cross_module_attr_contextual_type() {
    let state = create_state_multi_module(&[
        // base_mod defines a class with x: int64 (ClassDefIndex 0 in base_mod)
        (
            "base_mod",
            r#"
from __static__ import int64

class Base:
    x: int64
"#,
        ),
        // child_mod defines a local class also at ClassDefIndex 0 with x: double,
        // then imports Base and assigns to b.x on a Base instance directly
        // (no inheritance, so x is purely a cross-module attribute).
        (
            "child_mod",
            r#"
from __static__ import double
from base_mod import Base

class Local:
    x: double

def f(b: Base) -> None:
    b.x = 42
"#,
        ),
    ]);
    let transaction = state.transaction();
    let handle = get_handle("child_mod", &transaction);
    let data = collect_module_types(&transaction, &handle).expect("should collect types");
    let output = format_module_types(&data.entries, &data.locations);

    // The contextual type for `42` in `b.x = 42` should be int64 (from
    // Base.x), NOT double (from Local.x which happens to share the same
    // ClassDefIndex in the local module).
    assert!(
        output.contains("contextual: __static__.int64"),
        "Expected contextual type __static__.int64 from Base.x.\nOutput:\n{output}"
    );
}
