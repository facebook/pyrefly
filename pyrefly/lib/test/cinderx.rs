/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests for the CinderX type report.

use pretty_assertions::assert_eq;

use crate::report::cinderx::collect::collect_module_types;
use crate::report::cinderx::types::StructuredType;
use crate::report::cinderx::write_results;
use crate::state::require::Require;
use crate::test::util::TestEnv;

/// Create a type-checked state from a single module's Python source.
fn create_state(module_name: &str, python_code: &str) -> crate::state::state::State {
    let mut test_env = TestEnv::new();
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
fn test_simple_variable() {
    let state = create_state("test", "x: int = 42");
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // We should have at least one located type (the variable `x`)
    assert!(
        !data.locations.is_empty(),
        "expected at least one located type"
    );
    assert!(
        !data.entries.is_empty(),
        "expected at least one type table entry"
    );

    // The type table should contain a `builtins.int` class entry
    let has_int = data.entries.iter().any(|entry| {
        matches!(&entry.ty, StructuredType::Class { qname, args, .. } if qname == "builtins.int" && args.is_empty())
    });
    assert!(
        has_int,
        "expected `builtins.int` in the type table, got: {:#?}",
        data.entries
    );
}

#[test]
fn test_function_types() {
    let state = create_state(
        "test",
        r#"
def foo(x: int) -> str:
    return str(x)

y = foo(42)
"#,
    );
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // Should have `int`, `str` class entries and a callable entry for `foo`
    let has_str = data.entries.iter().any(
        |entry| matches!(&entry.ty, StructuredType::Class { qname, .. } if qname == "builtins.str"),
    );
    let has_callable = data
        .entries
        .iter()
        .any(|entry| matches!(&entry.ty, StructuredType::Callable { .. }));
    assert!(has_str, "expected `builtins.str` in the type table");
    assert!(has_callable, "expected a Callable entry for `foo`");

    // Every location should reference a valid type table index
    for loc in &data.locations {
        assert!(
            loc.type_index < data.entries.len(),
            "location references out-of-bounds type index {} (table has {} entries)",
            loc.type_index,
            data.entries.len(),
        );
    }
}

#[test]
fn test_optional_type() {
    let state = create_state(
        "test",
        r#"
x: int | None = None
"#,
    );
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // Should have a typing.Optional wrapping int
    let has_optional = data.entries.iter().any(|entry| {
        matches!(&entry.ty, StructuredType::OtherForm { qname, args } if qname == "typing.Optional" && args.len() == 1)
    });
    assert!(
        has_optional,
        "expected `typing.Optional` in the type table, got: {:#?}",
        data.entries,
    );
}

#[test]
fn test_literal_type() {
    let state = create_state("test", "x = 42");
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // The literal value `42` should produce a Literal entry
    let has_literal = data
        .entries
        .iter()
        .any(|entry| matches!(&entry.ty, StructuredType::Literal { value, .. } if value == "42"));
    assert!(
        has_literal,
        "expected Literal(42) in the type table, got: {:#?}",
        data.entries,
    );
}

#[test]
fn test_class_with_type_args() {
    let state = create_state(
        "test",
        r#"
x: list[int] = [1, 2, 3]
"#,
    );
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // Should have `builtins.list` with an `builtins.int` arg
    let has_list = data.entries.iter().any(|entry| {
        matches!(&entry.ty, StructuredType::Class { qname, args, .. } if qname == "builtins.list" && args.len() == 1)
    });
    assert!(
        has_list,
        "expected `builtins.list` with type arg in the type table, got: {:#?}",
        data.entries,
    );
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
    write_results(output_dir.path(), &transaction).expect("should write results");

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
    write_results(output_dir.path(), &transaction).expect("should write results");

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
fn test_bound_method_defining_class() {
    let state = create_state(
        "test",
        r#"
class MyClass:
    def greet(self) -> str:
        return "hello"

obj = MyClass()
result = obj.greet()
"#,
    );
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // Should have a BoundMethod entry with defining_class = "test.MyClass"
    let has_bound_method = data.entries.iter().any(|entry| {
        matches!(
            &entry.ty,
            StructuredType::BoundMethod { defining_class: Some(dc), .. } if dc == "test.MyClass"
        )
    });
    assert!(
        has_bound_method,
        "expected a BoundMethod with defining_class 'test.MyClass', got: {:#?}",
        data.entries
    );
}

#[test]
fn test_facet_narrow_mismatch() {
    let state = create_state(
        "test",
        r#"
class Foo:
    x: int | None

def f(foo: Foo) -> None:
    if foo.x is not None:
        y = foo.x
"#,
    );
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // Find the located type for `foo.x` inside the if-branch (the narrowed access).
    // It should have unnarrowed_type set and is_narrowed_mismatch == true because
    // the narrowed type (int) differs from the unnarrowed type (int | None).
    let narrowed_locs: Vec<_> = data
        .locations
        .iter()
        .filter(|loc| loc.unnarrowed_type.is_some())
        .collect();
    assert!(
        !narrowed_locs.is_empty(),
        "expected at least one location with unnarrowed_type set, got locations: {:#?}",
        data.locations,
    );

    let has_mismatch = narrowed_locs.iter().any(|loc| loc.is_narrowed_mismatch);
    assert!(
        has_mismatch,
        "expected is_narrowed_mismatch == true for the narrowed foo.x access",
    );

    // The unnarrowed type index should be valid and different from the narrowed type index.
    for loc in &narrowed_locs {
        let unnarrowed_idx = loc.unnarrowed_type.unwrap();
        assert!(
            unnarrowed_idx < data.entries.len(),
            "unnarrowed_type index {unnarrowed_idx} is out of bounds (table has {} entries)",
            data.entries.len(),
        );
    }
}

#[test]
fn test_facet_narrow_no_mismatch() {
    let state = create_state(
        "test",
        r#"
class Foo:
    x: int

def f(foo: Foo) -> None:
    y = foo.x
"#,
    );
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // No facet narrow exists on foo.x since it's always `int`.
    // All locations should have unnarrowed_type == None and is_narrowed_mismatch == false.
    for loc in &data.locations {
        assert!(
            loc.unnarrowed_type.is_none(),
            "expected no unnarrowed_type for non-narrowed attribute access, got: {:#?}",
            loc,
        );
        assert!(
            !loc.is_narrowed_mismatch,
            "expected is_narrowed_mismatch == false for non-narrowed attribute access",
        );
    }
}

#[test]
fn test_facet_narrow_chained_attr() {
    let state = create_state(
        "test",
        r#"
class Inner:
    value: int | None

class Outer:
    inner: Inner

def f(outer: Outer) -> None:
    if outer.inner.value is not None:
        y = outer.inner.value
"#,
    );
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // Find the located type for `outer.inner.value` inside the if-branch.
    // It should have unnarrowed_type set and is_narrowed_mismatch == true because
    // the narrowed type (int) differs from the unnarrowed type (int | None).
    let narrowed_locs: Vec<_> = data
        .locations
        .iter()
        .filter(|loc| loc.unnarrowed_type.is_some())
        .collect();
    assert!(
        !narrowed_locs.is_empty(),
        "expected at least one location with unnarrowed_type set for chained attr, got locations: {:#?}",
        data.locations,
    );

    let has_mismatch = narrowed_locs.iter().any(|loc| loc.is_narrowed_mismatch);
    assert!(
        has_mismatch,
        "expected is_narrowed_mismatch == true for the narrowed outer.inner.value access",
    );

    // The unnarrowed type index should be valid.
    for loc in &narrowed_locs {
        let unnarrowed_idx = loc.unnarrowed_type.unwrap();
        assert!(
            unnarrowed_idx < data.entries.len(),
            "unnarrowed_type index {unnarrowed_idx} is out of bounds (table has {} entries)",
            data.entries.len(),
        );
    }
}

#[test]
fn test_no_facets_no_reresolution() {
    let state = create_state(
        "test",
        r#"
x: int = 42
y = x
"#,
    );
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // No attribute access at all, so no facet narrow detection should trigger.
    for loc in &data.locations {
        assert!(
            loc.unnarrowed_type.is_none(),
            "expected no unnarrowed_type for simple variable access, got: {:#?}",
            loc,
        );
        assert!(
            !loc.is_narrowed_mismatch,
            "expected is_narrowed_mismatch == false for simple variable access",
        );
    }
}

#[test]
fn test_facet_narrow_index() {
    let state = create_state(
        "test",
        r#"
from typing import Tuple

def f(t: tuple[int | None, str]) -> None:
    if t[0] is not None:
        y = t[0]
"#,
    );
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // Find the located type for `t[0]` inside the if-branch (the narrowed access).
    // It should have unnarrowed_type set and is_narrowed_mismatch == true because
    // the narrowed type (int) differs from the unnarrowed type (int | None).
    let narrowed_locs: Vec<_> = data
        .locations
        .iter()
        .filter(|loc| loc.unnarrowed_type.is_some())
        .collect();
    assert!(
        !narrowed_locs.is_empty(),
        "expected at least one location with unnarrowed_type set for index facet, got locations: {:#?}",
        data.locations,
    );

    let has_mismatch = narrowed_locs.iter().any(|loc| loc.is_narrowed_mismatch);
    assert!(
        has_mismatch,
        "expected is_narrowed_mismatch == true for the narrowed t[0] access",
    );
}

#[test]
fn test_facet_narrow_key() {
    let state = create_state(
        "test",
        r#"
from typing import TypedDict

class MyDict(TypedDict):
    x: int | None

def f(d: MyDict) -> None:
    if d["x"] is not None:
        y = d["x"]
"#,
    );
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // Find the located type for `d["x"]` inside the if-branch (the narrowed access).
    // It should have unnarrowed_type set and is_narrowed_mismatch == true because
    // the narrowed type (int) differs from the unnarrowed type (int | None).
    let narrowed_locs: Vec<_> = data
        .locations
        .iter()
        .filter(|loc| loc.unnarrowed_type.is_some())
        .collect();
    assert!(
        !narrowed_locs.is_empty(),
        "expected at least one location with unnarrowed_type set for key facet, got locations: {:#?}",
        data.locations,
    );

    let has_mismatch = narrowed_locs.iter().any(|loc| loc.is_narrowed_mismatch);
    assert!(
        has_mismatch,
        "expected is_narrowed_mismatch == true for the narrowed d[\"x\"] access",
    );
}

#[test]
fn test_callable_defining_func() {
    let state = create_state(
        "test",
        r#"
def greet(name: str) -> str:
    return "hello " + name

x = greet
"#,
    );
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // A Type::Function should produce a Callable entry with defining_func set
    let has_defining_func = data.entries.iter().any(|entry| {
        matches!(
            &entry.ty,
            StructuredType::Callable { defining_func: Some(df), .. } if df == "test.greet"
        )
    });
    assert!(
        has_defining_func,
        "expected a Callable with defining_func 'test.greet', got: {:#?}",
        data.entries
    );
}

#[test]
fn test_callable_defining_func_method() {
    let state = create_state(
        "test",
        r#"
class MyClass:
    def greet(self) -> str:
        return "hello"

f = MyClass.greet
"#,
    );
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // Accessing MyClass.greet (unbound) should produce a Callable with class prefix
    let has_method_defining_func = data.entries.iter().any(|entry| {
        matches!(
            &entry.ty,
            StructuredType::Callable { defining_func: Some(df), .. } if df == "test.MyClass.greet"
        )
    });
    assert!(
        has_method_defining_func,
        "expected a Callable with defining_func 'test.MyClass.greet', got: {:#?}",
        data.entries
    );
}

#[test]
fn test_facet_narrow_mixed_chain() {
    let state = create_state(
        "test",
        r#"
class Inner:
    value: int | None

def f(t: tuple[Inner, str]) -> None:
    if t[0].value is not None:
        y = t[0].value
"#,
    );
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // Find the located type for `t[0].value` inside the if-branch.
    // This is a mixed chain (Index then Attribute) with a facet narrow:
    // the narrowed type (int) differs from the unnarrowed type (int | None).
    let narrowed_locs: Vec<_> = data
        .locations
        .iter()
        .filter(|loc| loc.unnarrowed_type.is_some())
        .collect();
    assert!(
        !narrowed_locs.is_empty(),
        "expected at least one location with unnarrowed_type set for mixed chain, got locations: {:#?}",
        data.locations,
    );

    let has_mismatch = narrowed_locs.iter().any(|loc| loc.is_narrowed_mismatch);
    assert!(
        has_mismatch,
        "expected is_narrowed_mismatch == true for the narrowed t[0].value access",
    );
}

#[test]
fn test_literal_promoted_type() {
    let state = create_state("test", "x = 42");
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // Find the Literal entry for "42" and verify its promoted_type points to builtins.int
    let literal_entry = data
        .entries
        .iter()
        .find(|entry| matches!(&entry.ty, StructuredType::Literal { value, .. } if value == "42"));
    assert!(
        literal_entry.is_some(),
        "expected Literal(42) in the type table, got: {:#?}",
        data.entries,
    );

    let promoted_idx = match &literal_entry.unwrap().ty {
        StructuredType::Literal { promoted_type, .. } => *promoted_type,
        _ => unreachable!("already matched as Literal"),
    };

    // The promoted_type index should point to a builtins.int Class entry
    let promoted_entry = &data.entries[promoted_idx];
    assert!(
        matches!(&promoted_entry.ty, StructuredType::Class { qname, args, .. } if qname == "builtins.int" && args.is_empty()),
        "expected promoted_type to point to builtins.int, got: {:#?}",
        promoted_entry.ty,
    );
}
