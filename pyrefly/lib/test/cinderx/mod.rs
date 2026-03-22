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

/// When a literal int is assigned to a variable annotated with `__static__.int64`,
/// the CinderX report should record the contextual type `__static__.int64` for the
/// literal expression via the `contextual_type` field on `LocatedType`.
#[test]
fn test_static_int64_literal_contextual_type() {
    let state = create_state_with_static(
        "test",
        r#"
from __static__ import int64

x: int64 = 42
"#,
    );
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // The type table should contain `__static__.int64` as a class entry,
    // proving that the annotation resolved correctly.
    let has_int64_class = data.entries.iter().any(|entry| {
        matches!(
            &entry.ty,
            StructuredType::Class { qname, .. } if qname == "__static__.int64"
        )
    });
    assert!(
        has_int64_class,
        "expected `__static__.int64` in the type table, got: {:#?}",
        data.entries,
    );

    // The literal `42` should still be recorded with its inferred type `Literal[42]`.
    let has_literal_42 = data.entries.iter().any(|entry| {
        matches!(
            &entry.ty,
            StructuredType::Literal { value, .. } if value == "42"
        )
    });
    assert!(
        has_literal_42,
        "expected Literal(42) in the type table, got: {:#?}",
        data.entries,
    );

    // Find the located type for the literal `42` and verify that its
    // `contextual_type` points to `__static__.int64`.
    let int64_idx = data
        .entries
        .iter()
        .position(|entry| {
            matches!(
                &entry.ty,
                StructuredType::Class { qname, .. } if qname == "__static__.int64"
            )
        })
        .expect("__static__.int64 should exist in the type table");

    let literal_42_idx = data
        .entries
        .iter()
        .position(|entry| {
            matches!(
                &entry.ty,
                StructuredType::Literal { value, .. } if value == "42"
            )
        })
        .expect("Literal(42) should exist in the type table");

    let loc_with_contextual = data
        .locations
        .iter()
        .find(|loc| loc.type_index == literal_42_idx && loc.contextual_type.is_some());
    assert!(
        loc_with_contextual.is_some(),
        "expected a located type for literal 42 with contextual_type set, got locations: {:#?}",
        data.locations,
    );

    let ctx_idx = loc_with_contextual.unwrap().contextual_type.unwrap();
    assert_eq!(
        ctx_idx, int64_idx,
        "expected contextual_type to point to __static__.int64 (index {int64_idx}), got index {ctx_idx}",
    );
}

/// When a literal int is re-assigned to a variable previously annotated with
/// `__static__.int64`, the CinderX report should record the contextual type
/// `__static__.int64` for the RHS literal expression of the plain `Assign`.
#[test]
fn test_static_assign_after_annotation() {
    let state = create_state_with_static(
        "test",
        r#"
from __static__ import int64

x: int64 = 0
x = 42
"#,
    );
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // The type table should contain `__static__.int64` as a class entry.
    let int64_idx = data
        .entries
        .iter()
        .position(|entry| {
            matches!(
                &entry.ty,
                StructuredType::Class { qname, .. } if qname == "__static__.int64"
            )
        })
        .expect("__static__.int64 should exist in the type table");

    // Find a located type with contextual_type pointing to __static__.int64.
    // The RHS `42` of the plain `x = 42` assign should have it.
    let loc_with_contextual = data
        .locations
        .iter()
        .find(|loc| loc.contextual_type == Some(int64_idx));
    assert!(
        loc_with_contextual.is_some(),
        "expected a located type for literal 42 (plain assign) with contextual_type pointing to __static__.int64, got locations: {:#?}",
        data.locations,
    );
}

/// When a variable is declared with a `__static__` annotation but no initial
/// value, and then assigned via a plain `Assign`, the RHS should still get
/// the contextual type from the annotation.
#[test]
fn test_static_assign_without_prior_value() {
    let state = create_state_with_static(
        "test",
        r#"
from __static__ import int64

x: int64
x = 42
"#,
    );
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // The type table should contain `__static__.int64` as a class entry.
    let int64_idx = data
        .entries
        .iter()
        .position(|entry| {
            matches!(
                &entry.ty,
                StructuredType::Class { qname, .. } if qname == "__static__.int64"
            )
        })
        .expect("__static__.int64 should exist in the type table");

    // Find a located type with contextual_type pointing to __static__.int64.
    // The RHS `42` of `x = 42` should have it.
    let loc_with_contextual = data
        .locations
        .iter()
        .find(|loc| loc.contextual_type == Some(int64_idx));
    assert!(
        loc_with_contextual.is_some(),
        "expected a located type for literal 42 (assign after bare annotation) with contextual_type pointing to __static__.int64, got locations: {:#?}",
        data.locations,
    );
}

/// When a literal float is assigned to a variable annotated with `__static__.double`,
/// the CinderX report should record the contextual type `__static__.double` for the
/// literal expression via the `contextual_type` field on `LocatedType`.
#[test]
fn test_static_double_literal_contextual_type() {
    let state = create_state_with_static(
        "test",
        r#"
from __static__ import double

y: double = 3.14
"#,
    );
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // The type table should contain `__static__.double` as a class entry,
    // proving that the annotation resolved correctly.
    let has_double_class = data.entries.iter().any(|entry| {
        matches!(
            &entry.ty,
            StructuredType::Class { qname, .. } if qname == "__static__.double"
        )
    });
    assert!(
        has_double_class,
        "expected `__static__.double` in the type table, got: {:#?}",
        data.entries,
    );

    // The literal `3.14` should have its inferred type (builtins.float) recorded,
    // and the contextual type should point to `__static__.double`.
    let double_idx = data
        .entries
        .iter()
        .position(|entry| {
            matches!(
                &entry.ty,
                StructuredType::Class { qname, .. } if qname == "__static__.double"
            )
        })
        .expect("__static__.double should exist in the type table");

    let float_class_idx = data
        .entries
        .iter()
        .position(|entry| {
            matches!(
                &entry.ty,
                StructuredType::Class { qname, args, .. } if qname == "builtins.float" && args.is_empty()
            )
        })
        .expect("builtins.float should exist");

    // Find a located type for the literal 3.14 (type = builtins.float) with
    // contextual_type pointing to __static__.double.
    let loc_with_contextual = data
        .locations
        .iter()
        .find(|loc| loc.type_index == float_class_idx && loc.contextual_type.is_some());
    assert!(
        loc_with_contextual.is_some(),
        "expected a located type for literal 3.14 with contextual_type set, got locations: {:#?}",
        data.locations,
    );

    let ctx_idx = loc_with_contextual.unwrap().contextual_type.unwrap();
    assert_eq!(
        ctx_idx, double_idx,
        "expected contextual_type to point to __static__.double (index {double_idx}), got index {ctx_idx}",
    );
}

/// When a positional argument to a function call has a corresponding parameter
/// annotated with `__static__.int64`, the CinderX report should record the
/// contextual type `__static__.int64` for the argument expression.
#[test]
fn test_static_call_positional_arg() {
    let state = create_state_with_static(
        "test",
        r#"
from __static__ import int64

def foo(x: int64) -> None:
    pass

foo(42)
"#,
    );
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // The type table should contain `__static__.int64` as a class entry.
    let int64_idx = data
        .entries
        .iter()
        .position(|entry| {
            matches!(
                &entry.ty,
                StructuredType::Class { qname, .. } if qname == "__static__.int64"
            )
        })
        .expect("__static__.int64 should exist in the type table");

    // Find a located type with contextual_type pointing to __static__.int64.
    // The positional argument `42` in `foo(42)` should have it.
    let loc_with_contextual = data
        .locations
        .iter()
        .find(|loc| loc.contextual_type == Some(int64_idx));
    assert!(
        loc_with_contextual.is_some(),
        "expected a located type for positional arg 42 with contextual_type pointing to __static__.int64, got locations: {:#?}",
        data.locations,
    );
}

/// When a positional argument to a bound method call has a corresponding
/// parameter annotated with `__static__.int64`, the CinderX report should
/// record the contextual type, correctly skipping the `self` parameter.
#[test]
fn test_static_call_bound_method_arg() {
    let state = create_state_with_static(
        "test",
        r#"
from __static__ import int64

class MyClass:
    def bar(self, x: int64) -> None:
        pass

obj = MyClass()
obj.bar(42)
"#,
    );
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // The type table should contain `__static__.int64` as a class entry.
    let int64_idx = data
        .entries
        .iter()
        .position(|entry| {
            matches!(
                &entry.ty,
                StructuredType::Class { qname, .. } if qname == "__static__.int64"
            )
        })
        .expect("__static__.int64 should exist in the type table");

    // Find a located type with contextual_type pointing to __static__.int64.
    // The positional argument `42` in `obj.bar(42)` should have it.
    let loc_with_contextual = data
        .locations
        .iter()
        .find(|loc| loc.contextual_type == Some(int64_idx));
    assert!(
        loc_with_contextual.is_some(),
        "expected a located type for bound method arg 42 with contextual_type pointing to __static__.int64, got locations: {:#?}",
        data.locations,
    );
}

/// When a function has multiple parameters, only those annotated with
/// `__static__` primitive types should get contextual types on their
/// corresponding positional arguments.
#[test]
fn test_static_call_multiple_args() {
    let state = create_state_with_static(
        "test",
        r#"
from __static__ import int64, double

def baz(x: int64, y: str, z: double) -> None:
    pass

baz(42, "hello", 3.14)
"#,
    );
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // The type table should contain both `__static__.int64` and `__static__.double`.
    let int64_idx = data
        .entries
        .iter()
        .position(|entry| {
            matches!(
                &entry.ty,
                StructuredType::Class { qname, .. } if qname == "__static__.int64"
            )
        })
        .expect("__static__.int64 should exist in the type table");

    let double_idx = data
        .entries
        .iter()
        .position(|entry| {
            matches!(
                &entry.ty,
                StructuredType::Class { qname, .. } if qname == "__static__.double"
            )
        })
        .expect("__static__.double should exist in the type table");

    // The literal `42` should have contextual_type pointing to __static__.int64.
    let literal_42_idx = data
        .entries
        .iter()
        .position(|entry| {
            matches!(
                &entry.ty,
                StructuredType::Literal { value, .. } if value == "42"
            )
        })
        .expect("Literal(42) should exist in the type table");

    let loc_42 = data
        .locations
        .iter()
        .find(|loc| loc.type_index == literal_42_idx && loc.contextual_type == Some(int64_idx));
    assert!(
        loc_42.is_some(),
        "expected literal 42 to have contextual_type __static__.int64, got locations: {:#?}",
        data.locations,
    );

    // The literal `3.14` should have contextual_type pointing to __static__.double.
    let float_class_idx = data
        .entries
        .iter()
        .position(|entry| {
            matches!(
                &entry.ty,
                StructuredType::Class { qname, args, .. } if qname == "builtins.float" && args.is_empty()
            )
        })
        .expect("builtins.float should exist");

    let loc_314 = data
        .locations
        .iter()
        .find(|loc| loc.type_index == float_class_idx && loc.contextual_type == Some(double_idx));
    assert!(
        loc_314.is_some(),
        "expected literal 3.14 to have contextual_type __static__.double, got locations: {:#?}",
        data.locations,
    );

    // The string literal `"hello"` should NOT have a contextual type.
    let str_literal_idx = data.entries.iter().position(|entry| {
        matches!(
            &entry.ty,
            StructuredType::Literal { value, .. } if value == "\"hello\""
        )
    });
    if let Some(str_idx) = str_literal_idx {
        let loc_hello = data
            .locations
            .iter()
            .find(|loc| loc.type_index == str_idx && loc.contextual_type.is_some());
        assert!(
            loc_hello.is_none(),
            "expected string literal \"hello\" to NOT have a contextual type, got locations: {:#?}",
            data.locations,
        );
    }
}

/// When a literal int is assigned to an attribute annotated with `__static__.int64`
/// (e.g. `self.x = 42` where `x: int64`), the CinderX report should record
/// the contextual type `__static__.int64` for the literal expression.
#[test]
fn test_static_attr_assign() {
    let state = create_state_with_static(
        "test",
        r#"
from __static__ import int64

class Foo:
    x: int64
    def __init__(self) -> None:
        self.x = 42
"#,
    );
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // The type table should contain `__static__.int64` as a class entry.
    let int64_idx = data
        .entries
        .iter()
        .position(|entry| {
            matches!(
                &entry.ty,
                StructuredType::Class { qname, .. } if qname == "__static__.int64"
            )
        })
        .expect("__static__.int64 should exist in the type table");

    // Find a located type with contextual_type pointing to __static__.int64.
    // The RHS `42` of `self.x = 42` should have it.
    let loc_with_contextual = data
        .locations
        .iter()
        .find(|loc| loc.contextual_type == Some(int64_idx));
    assert!(
        loc_with_contextual.is_some(),
        "expected a located type for literal 42 (attr assign) with contextual_type pointing to __static__.int64, got locations: {:#?}",
        data.locations,
    );
}

/// When a class body has an annotated assignment like `x: int64 = 42`,
/// the CinderX report should record the contextual type `__static__.int64`
/// for the literal expression.
#[test]
fn test_static_attr_ann_assign() {
    let state = create_state_with_static(
        "test",
        r#"
from __static__ import int64

class Bar:
    x: int64 = 42
"#,
    );
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // The type table should contain `__static__.int64` as a class entry.
    let int64_idx = data
        .entries
        .iter()
        .position(|entry| {
            matches!(
                &entry.ty,
                StructuredType::Class { qname, .. } if qname == "__static__.int64"
            )
        })
        .expect("__static__.int64 should exist in the type table");

    // Find a located type with contextual_type pointing to __static__.int64.
    // The RHS `42` of `x: int64 = 42` in the class body should have it.
    let loc_with_contextual = data
        .locations
        .iter()
        .find(|loc| loc.contextual_type == Some(int64_idx));
    assert!(
        loc_with_contextual.is_some(),
        "expected a located type for literal 42 (class body ann assign) with contextual_type pointing to __static__.int64, got locations: {:#?}",
        data.locations,
    );
}

/// Test that `int64(0)` (a constructor call) gets contextual type treatment.
/// This mirrors a pattern the CinderX team hits in practice: using the
/// `int64` constructor to initialize a primitive-typed local.
///
/// Both annotated (`x: int64 = int64(0)`) and unannotated (`y = int64(0)`)
/// assignments should work: `x` gets contextual_type from the AnnAssign,
/// and `y` gets it from the Assign because pyrefly infers `y`'s type as
/// `__static__.int64` from the constructor call.
#[test]
fn test_static_int64_constructor_call() {
    let state = create_state_with_static(
        "test",
        r#"
import __static__
from __static__ import int64

def main() -> None:
    x: int64 = int64(0)
    y = int64(0)
"#,
    );
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // The type table should contain `__static__.int64` as a class entry.
    let int64_idx = data
        .entries
        .iter()
        .position(|entry| {
            matches!(
                &entry.ty,
                StructuredType::Class { qname, .. } if qname == "__static__.int64"
            )
        })
        .expect("__static__.int64 should exist in the type table");

    // Both `int64(0)` call expressions should get contextual_type pointing
    // to __static__.int64. Count the locations with this contextual type.
    let locs_with_contextual: Vec<_> = data
        .locations
        .iter()
        .filter(|loc| loc.contextual_type == Some(int64_idx))
        .collect();

    // We expect at least 2: the RHS of `x: int64 = int64(0)` and `y = int64(0)`.
    assert!(
        locs_with_contextual.len() >= 2,
        "expected at least 2 located types with contextual_type __static__.int64, got {}: {:#?}",
        locs_with_contextual.len(),
        data.locations,
    );

    // Each of these should have inferred type __static__.int64 as well
    // (since `int64(0)` returns `int64`).
    for loc in &locs_with_contextual {
        assert_eq!(
            loc.type_index, int64_idx,
            "expected inferred type to also be __static__.int64 for int64(0) call",
        );
    }
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

/// When a keyword argument to a function call has a corresponding parameter
/// annotated with a `__static__` primitive type, the CinderX report should
/// record the contextual type on the keyword argument's value expression.
#[test]
fn test_static_call_keyword_arg() {
    let state = create_state_with_static(
        "test",
        r#"
from __static__ import int64

def foo(x: int64) -> None:
    pass

foo(x=42)
"#,
    );
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // The type table should contain `__static__.int64` as a class entry.
    let int64_idx = data
        .entries
        .iter()
        .position(|entry| {
            matches!(
                &entry.ty,
                StructuredType::Class { qname, .. } if qname == "__static__.int64"
            )
        })
        .expect("__static__.int64 should exist in the type table");

    // Find a located type with contextual_type pointing to __static__.int64.
    // The keyword argument value `42` in `foo(x=42)` should have it.
    let loc_with_contextual = data
        .locations
        .iter()
        .find(|loc| loc.contextual_type == Some(int64_idx));
    assert!(
        loc_with_contextual.is_some(),
        "expected a located type for keyword arg 42 with contextual_type pointing to __static__.int64, got locations: {:#?}",
        data.locations,
    );
}

/// When a function call mixes positional and keyword arguments, contextual
/// types should be recorded for both positional and keyword arguments whose
/// corresponding parameters are `__static__` primitive types. Non-static
/// parameters should not get contextual types.
#[test]
fn test_static_call_mixed_args() {
    let state = create_state_with_static(
        "test",
        r#"
from __static__ import int64, double

def bar(x: int64, y: str, z: double) -> None:
    pass

bar(42, z=3.14, y="hello")
"#,
    );
    let transaction = state.transaction();
    let handle = get_handle("test", &transaction);

    let data = collect_module_types(&transaction, &handle).expect("should collect types");

    // The type table should contain both `__static__.int64` and `__static__.double`.
    let int64_idx = data
        .entries
        .iter()
        .position(|entry| {
            matches!(
                &entry.ty,
                StructuredType::Class { qname, .. } if qname == "__static__.int64"
            )
        })
        .expect("__static__.int64 should exist in the type table");

    let double_idx = data
        .entries
        .iter()
        .position(|entry| {
            matches!(
                &entry.ty,
                StructuredType::Class { qname, .. } if qname == "__static__.double"
            )
        })
        .expect("__static__.double should exist in the type table");

    // The literal `42` (positional) should have contextual_type pointing to __static__.int64.
    let literal_42_idx = data
        .entries
        .iter()
        .position(|entry| {
            matches!(
                &entry.ty,
                StructuredType::Literal { value, .. } if value == "42"
            )
        })
        .expect("Literal(42) should exist in the type table");

    let loc_42 = data
        .locations
        .iter()
        .find(|loc| loc.type_index == literal_42_idx && loc.contextual_type == Some(int64_idx));
    assert!(
        loc_42.is_some(),
        "expected positional arg 42 to have contextual_type __static__.int64, got locations: {:#?}",
        data.locations,
    );

    // The literal `3.14` (keyword `z`) should have contextual_type pointing to __static__.double.
    let float_class_idx = data
        .entries
        .iter()
        .position(|entry| {
            matches!(
                &entry.ty,
                StructuredType::Class { qname, args, .. } if qname == "builtins.float" && args.is_empty()
            )
        })
        .expect("builtins.float should exist");

    let loc_314 = data
        .locations
        .iter()
        .find(|loc| loc.type_index == float_class_idx && loc.contextual_type == Some(double_idx));
    assert!(
        loc_314.is_some(),
        "expected keyword arg 3.14 (z) to have contextual_type __static__.double, got locations: {:#?}",
        data.locations,
    );

    // The string literal `"hello"` (keyword `y`) should NOT have a contextual type.
    let str_class_idx = data
        .entries
        .iter()
        .position(|entry| {
            matches!(
                &entry.ty,
                StructuredType::Class { qname, args, .. } if qname == "builtins.str" && args.is_empty()
            )
        })
        .expect("builtins.str should exist");

    let loc_hello = data
        .locations
        .iter()
        .find(|loc| loc.type_index == str_class_idx && loc.contextual_type.is_some());
    assert!(
        loc_hello.is_none(),
        "expected string arg 'hello' to have no contextual_type, but it has one: {:#?}",
        data.locations,
    );
}
