/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests for the query interface.

use pretty_assertions::assert_eq;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::fs_anyhow;
use pyrefly_util::thread_pool::TEST_THREAD_COUNT;
use serde_json::Value;
use tempfile::TempDir;

use crate::config::config::ConfigFile;
use crate::config::config::ConfigSource;
use crate::config::finder::ConfigFinder;
use crate::query::Query;
use crate::query::SerializedTypeTableEntry;
use crate::test::util::init_test;

/// Helper to create a Query with a ConfigFinder that doesn't use sourcedb.
fn create_query() -> Query {
    init_test();
    let mut config = ConfigFile::default();
    config.python_environment.set_empty_to_default();
    config.configure();
    let config = ArcId::new(config);
    Query::new(ConfigFinder::new_constant(config), TEST_THREAD_COUNT)
}

fn indexed_shape_values(type_table: &[SerializedTypeTableEntry]) -> Vec<Value> {
    type_table
        .iter()
        .map(|type_shape| serde_json::to_value(type_shape).unwrap())
        .collect()
}

fn is_indexed_named_shape(shape: &Value, name: &str, args: &[usize]) -> bool {
    is_named_shape(shape, name)
        && shape
            .get("args")
            .and_then(Value::as_array)
            .is_some_and(|actual_args| {
                actual_args.len() == args.len()
                    && actual_args
                        .iter()
                        .zip(args.iter())
                        .all(|(actual, expected)| actual.as_u64() == Some(*expected as u64))
            })
}

fn is_named_shape(shape: &Value, name: &str) -> bool {
    shape.get("kind").and_then(Value::as_str) == Some("named")
        && shape.get("name").and_then(Value::as_str) == Some(name)
}

#[test]
fn test_type_table_direct_conversion_includes_callable_union_optional_and_dedup() {
    let tdir = TempDir::new().unwrap();
    let file_path = tdir.path().join("main.py");
    let code = r#"from typing import Callable

f: Callable[[int, str], bool]
x: int | str
y: int | None
first: list[int]
second: list[int]
"#;
    fs_anyhow::write(&file_path, code).unwrap();

    let query = create_query();
    let module_name = ModuleName::from_str("main");
    let path = ModulePath::filesystem(file_path.clone());

    let errors = query.add_files(vec![(module_name, path.clone())]);
    assert!(errors.is_empty(), "Unexpected errors: {:?}", errors);

    let response = query
        .get_type_table_in_file_with_timing(module_name, path)
        .unwrap()
        .0;
    let table = indexed_shape_values(&response.type_table);

    assert!(
        table.iter().any(|shape| {
            shape.get("kind").and_then(Value::as_str) == Some("callable")
                && shape
                    .get("params")
                    .and_then(Value::as_array)
                    .is_some_and(|params| params.len() == 2)
                && shape.get("return_type").and_then(Value::as_u64).is_some()
        }),
        "Expected direct callable entry in type table:\n{table:#?}",
    );

    let int_index = table
        .iter()
        .position(|shape| is_indexed_named_shape(shape, "builtins.int", &[]))
        .unwrap();
    let str_index = table
        .iter()
        .position(|shape| is_indexed_named_shape(shape, "builtins.str", &[]))
        .unwrap();
    assert!(
        table.iter().any(|shape| is_indexed_named_shape(
            shape,
            "typing.Union",
            &[int_index, str_index]
        )),
        "Expected int | str to map to typing.Union with int and str args:\n{table:#?}",
    );
    assert!(
        table
            .iter()
            .any(|shape| is_indexed_named_shape(shape, "typing.Optional", &[int_index])),
        "Expected int | None to map to typing.Optional[int]:\n{table:#?}",
    );

    let list_int_entries = table
        .iter()
        .filter(|shape| is_indexed_named_shape(shape, "builtins.list", &[int_index]))
        .count();
    assert_eq!(
        1, list_int_entries,
        "Expected repeated list[int] annotations to share one structural table entry:\n{table:#?}",
    );

    // Every wire entry carries its structural hash so clients can key a global
    // (cross-file) hash -> parsed shape cache on it.
    assert!(
        table
            .iter()
            .all(|shape| shape.get("hash").and_then(Value::as_u64).is_some()),
        "Expected every type_table entry to carry a u64 hash:\n{table:#?}",
    );
    // Structurally distinct shapes must hash differently.
    let int_hash = table[int_index]
        .get("hash")
        .and_then(Value::as_u64)
        .unwrap();
    let str_hash = table[str_index]
        .get("hash")
        .and_then(Value::as_u64)
        .unwrap();
    assert_ne!(
        int_hash, str_hash,
        "Expected builtins.int and builtins.str to have distinct structural hashes",
    );
}

#[test]
fn test_annotated_metadata_has_expression_type_traces() {
    let tdir = TempDir::new().unwrap();
    let main_path = tdir.path().join("main.py");
    let pkg_path = tdir.path().join("pkg.py");
    let utils_path = tdir.path().join("utils.py");
    let code = r#"from typing import Annotated
import pkg

def takes(x: Annotated[int, pkg.utils.ValueRange, pkg.utils.Unit]) -> None: ...
"#;
    fs_anyhow::write(&main_path, code).unwrap();
    fs_anyhow::write(&pkg_path, "import utils\n").unwrap();
    fs_anyhow::write(&utils_path, "class ValueRange: ...\nclass Unit: ...\n").unwrap();

    let mut config = ConfigFile::default();
    config.python_environment.set_empty_to_default();
    config.search_path_from_args.push(tdir.path().to_path_buf());
    config.configure();
    let query = Query::new(
        ConfigFinder::new_constant(ArcId::new(config)),
        TEST_THREAD_COUNT,
    );
    let main = ModuleName::from_str("main");
    let pkg = ModuleName::from_str("pkg");
    let utils = ModuleName::from_str("utils");
    let errors = query.add_files(vec![
        (main, ModulePath::filesystem(main_path.clone())),
        (pkg, ModulePath::filesystem(pkg_path)),
        (utils, ModulePath::filesystem(utils_path)),
    ]);
    assert!(errors.is_empty(), "Unexpected errors: {errors:?}");

    let response = query
        .get_type_table_in_file_with_timing(main, ModulePath::filesystem(main_path))
        .unwrap()
        .0;
    let table = indexed_shape_values(&response.type_table);
    let type_at_range = |start_col, end_col| {
        response
            .types
            .iter()
            .find_map(|entry| {
                let range = &entry.location;
                (range.start_line.get() == 4
                    && range.start_col == start_col
                    && range.end_line.get() == 4
                    && range.end_col == end_col)
                    .then_some(entry.type_index)
            })
            .unwrap_or_else(|| {
                panic!(
                    "Expected a type trace at 4:{start_col}-4:{end_col}; got {:?}",
                    response.types
                )
            })
    };

    let value_range = table
        .iter()
        .position(|shape| is_indexed_named_shape(shape, "utils.ValueRange", &[]))
        .unwrap();
    let unit = table
        .iter()
        .position(|shape| is_indexed_named_shape(shape, "utils.Unit", &[]))
        .unwrap();
    assert!(
        is_indexed_named_shape(&table[type_at_range(28, 48)], "typing.Type", &[value_range],),
        "Unexpected ValueRange trace: {table:#?}"
    );
    assert!(
        is_indexed_named_shape(&table[type_at_range(50, 64)], "typing.Type", &[unit],),
        "Unexpected Unit trace: {table:#?}"
    );
    assert_eq!(type_at_range(28, 37), type_at_range(50, 59));
}

/// Enum-member literals must carry the enum class's fully module-qualified name
/// (`main.Color.RED`), not the bare class name (`Color.RED`), in the inline
/// shape path — matching the module-qualified display string and the
/// `ClassType` shape arm.
#[test]
fn test_type_table_qualifies_enum_literal_members() {
    let tdir = TempDir::new().unwrap();
    let file_path = tdir.path().join("main.py");
    let code = r#"import enum

class Color(enum.Enum):
    RED = 1
    GREEN = 2

c = Color.RED
"#;
    fs_anyhow::write(&file_path, code).unwrap();

    let query = create_query();
    let module_name = ModuleName::from_str("main");
    let path = ModulePath::filesystem(file_path.clone());

    let errors = query.add_files(vec![(module_name, path.clone())]);
    assert!(errors.is_empty(), "Unexpected errors: {:?}", errors);

    let response = query
        .get_type_table_in_file_with_timing(module_name, path)
        .unwrap()
        .0;
    let table = indexed_shape_values(&response.type_table);

    let member_index = table
        .iter()
        .position(|shape| is_indexed_named_shape(shape, "main.Color.RED", &[]))
        .expect("expected fully-qualified enum member leaf `main.Color.RED` in the type table");
    assert!(
        table
            .iter()
            .any(|shape| is_indexed_named_shape(shape, "typing.Literal", &[member_index])),
        "Expected typing.Literal entry referencing the qualified enum member:\n{table:#?}",
    );
}

#[test]
fn test_type_table_anonymous_typed_dict_is_dict() {
    let tdir = TempDir::new().unwrap();
    let file_path = tdir.path().join("main.py");
    // An unannotated dict literal with string-literal keys synthesizes an
    // anonymous TypedDict. Its structured shape must be `dict[str, int | str]`
    // (keeping the field value types), not an opaque `TypedDictionary` marker
    // that drops them.
    let code = r#"d = {"count": 1, "name": "x"}
"#;
    fs_anyhow::write(&file_path, code).unwrap();

    let query = create_query();
    let module_name = ModuleName::from_str("main");
    let path = ModulePath::filesystem(file_path.clone());

    let errors = query.add_files(vec![(module_name, path.clone())]);
    assert!(errors.is_empty(), "Unexpected errors: {:?}", errors);

    let response = query
        .get_type_table_in_file_with_timing(module_name, path)
        .unwrap()
        .0;
    let table = indexed_shape_values(&response.type_table);

    assert!(
        !table
            .iter()
            .any(|shape| is_named_shape(shape, "TypedDictionary")
                || is_named_shape(shape, "NonTotalTypedDictionary")),
        "anonymous TypedDict should not emit an opaque TypedDictionary marker:\n{table:#?}",
    );

    let str_index = table
        .iter()
        .position(|shape| is_indexed_named_shape(shape, "builtins.str", &[]))
        .expect("expected builtins.str leaf");
    let int_index = table
        .iter()
        .position(|shape| is_indexed_named_shape(shape, "builtins.int", &[]))
        .expect("expected builtins.int leaf");
    let value_index = table
        .iter()
        .position(|shape| {
            is_indexed_named_shape(shape, "typing.Union", &[int_index, str_index])
                || is_indexed_named_shape(shape, "typing.Union", &[str_index, int_index])
        })
        .expect("expected `int | str` field value union");
    assert!(
        table.iter().any(|shape| is_indexed_named_shape(
            shape,
            "builtins.dict",
            &[str_index, value_index]
        )),
        "expected `dict[str, int | str]` shape for the anonymous TypedDict:\n{table:#?}",
    );
}

#[test]
fn test_callees_annotated_type() {
    let tdir = TempDir::new().unwrap();
    let file_path = tdir.path().join("main.py");
    // A type alias whose body is Annotated[Foo, ...] stores Type::Annotated
    // internally. Calling the alias as a value makes callee_from_type recurse
    // into the TypeAlias body, reaching Type::Annotated.
    let code = r#"
from typing import Annotated, TypeAlias

class Foo:
    def bar(self) -> int:
        return 42

MyType: TypeAlias = Annotated[Foo, "metadata"]

def f() -> None:
    MyType()
"#;
    fs_anyhow::write(&file_path, code).unwrap();

    let query = create_query();
    let module_name = ModuleName::from_str("main");
    let path = ModulePath::filesystem(file_path.clone());

    let errors = query.add_files(vec![(module_name, path.clone())]);
    assert!(
        !errors.is_empty(),
        "Annotated[Foo, ...] is not callable, expected errors"
    );
    assert!(
        errors.iter().any(|e| e.contains("not-callable")),
        "Expected a not-callable error, got: {errors:?}",
    );

    // get_callees_with_location triggers callee_from_type which must handle
    // Type::Annotated rather than panicking. Annotated is not callable, so
    // MyType() should produce no callees.
    let callees = query
        .get_callees_with_location(module_name, path, None)
        .unwrap();
    assert!(
        callees.is_empty(),
        "Annotated is not callable, expected no callees"
    );
}

#[test]
fn test_callees_attribute_narrow_does_not_overwrite_rhs_trace() {
    // Regression test: narrowing on an attribute facet (e.g. `c.p == k.v`) used to
    // record the LHS property getter's trace against the narrow expression's range,
    // which clobbered the legitimate trace for the RHS. As a result, querying callees
    // on the RHS attribute returned the LHS property getter.
    let tdir = TempDir::new().unwrap();
    let file_path = tdir.path().join("main.py");
    let code = r#"
class C:
    @property
    def p(self) -> int:
        return 0

class K:
    v: int = 0

def foo(c: C, k: K) -> None:
    if c.p == k.v:
        pass
"#;
    fs_anyhow::write(&file_path, code).unwrap();

    let query = create_query();
    let module_name = ModuleName::from_str("main");
    let path = ModulePath::filesystem(file_path.clone());

    let errors = query.add_files(vec![(module_name, path.clone())]);
    assert!(errors.is_empty(), "Unexpected errors: {:?}", errors);

    let callees = query
        .get_callees_with_location(module_name, path, None)
        .unwrap();

    // The property getter `C.p` should be reported exactly once, at the `c.p`
    // access (line 11), not at the `k.v` access on the RHS.
    let p_getters: Vec<_> = callees
        .iter()
        .filter(|(_, c)| c.target == "main.C.p")
        .collect();
    assert_eq!(
        p_getters.len(),
        1,
        "Expected exactly one callee for property C.p, got: {p_getters:?}"
    );
    let (range, _) = p_getters[0];
    assert_eq!(
        range.start_line.get(),
        11,
        "C.p getter callee should be on line 11 (the `c.p` access), got: {range:?}"
    );

    // The RHS `k.v` is a plain attribute, not a property — it should produce no
    // callees at all. (Pre-fix, it had a spurious C.p property getter trace.)
    let k_v_callees: Vec<_> = callees
        .iter()
        .filter(|(r, _)| r.start_line.get() == 11 && r.start_col >= 13)
        .collect();
    assert!(
        k_v_callees.is_empty(),
        "Expected no callees on the RHS `k.v`, got: {k_v_callees:?}"
    );
}

#[test]
fn test_callees_final_stub_class_constructor_in_gather() {
    let tdir = TempDir::new().unwrap();
    let main_path = tdir.path().join("main.py");
    let dep_path = tdir.path().join("dep.pyi");
    let main_code = r#"
import asyncio
from dep import C

async def use_c(x: C) -> None: ...

async def f(x: bool | None):
    first, second = await asyncio.gather(use_c(C()), use_c(x or C()))
"#;
    let dep_code = r#"
from typing import final

@final
class C:
    def __init__(self) -> None: ...
    def __call__(self) -> C: ...
"#;
    fs_anyhow::write(&main_path, main_code).unwrap();
    fs_anyhow::write(&dep_path, dep_code).unwrap();

    let mut config = ConfigFile {
        source: ConfigSource::File(tdir.path().join(ConfigFile::PYREFLY_FILE_NAME)),
        enable_fallback_search_path: true,
        ..Default::default()
    };
    config.python_environment.set_empty_to_default();
    config.interpreters.skip_interpreter_query = true;
    config.configure();
    let query = Query::new(
        ConfigFinder::new_constant(ArcId::new(config)),
        TEST_THREAD_COUNT,
    );
    let main_name = ModuleName::from_str("main");
    let main_module_path = ModulePath::filesystem(main_path);
    query.add_files(vec![
        (
            ModuleName::from_str("dep"),
            ModulePath::filesystem(dep_path),
        ),
        (main_name, main_module_path.clone()),
    ]);

    let callees = query
        .get_callees_with_location(main_name, main_module_path, None)
        .unwrap();
    assert!(
        callees
            .iter()
            .filter(|(_, callee)| callee.target == "dep.C.__init__")
            .count()
            == 2,
        "Expected two constructor callees for C(), got: {callees:?}"
    );
}
