/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests for the query interface, specifically get_types_in_file.

use pretty_assertions::assert_eq;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::fs_anyhow;
use pyrefly_util::lined_buffer::PythonASTRange;
use pyrefly_util::thread_pool::TEST_THREAD_COUNT;
use serde_json::Value;
use serde_json::json;
use tempfile::TempDir;

use crate::config::config::ConfigFile;
use crate::config::finder::ConfigFinder;
use crate::query::Query;
use crate::query::TypeShape;
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

/// Convert the result of get_types_in_file to a pretty-printed JSON string.
/// This format makes test failures easy to patch by copy-pasting the actual output.
fn types_to_json_string(types: Vec<(PythonASTRange, String)>) -> String {
    let entries: Vec<serde_json::Value> = types
        .into_iter()
        .map(|(range, type_str)| {
            json!({
                "location": {
                    "start_line": range.start_line.get(),
                    "start_col": range.start_col,
                    "end_line": range.end_line.get(),
                    "end_col": range.end_col
                },
                "type": type_str
            })
        })
        .collect();
    serde_json::to_string_pretty(&entries).unwrap()
}

fn type_shape_values(types: Vec<(PythonASTRange, TypeShape)>) -> Vec<Value> {
    types
        .into_iter()
        .map(|(_, type_shape)| serde_json::to_value(type_shape).unwrap())
        .collect()
}

fn is_named_shape(shape: &Value, name: &str) -> bool {
    shape.get("kind").and_then(Value::as_str) == Some("named")
        && shape.get("name").and_then(Value::as_str) == Some(name)
}

fn is_named_shape_with_args(shape: &Value, name: &str, arg_names: &[&str]) -> bool {
    is_named_shape(shape, name)
        && shape
            .get("args")
            .and_then(Value::as_array)
            .is_some_and(|args| {
                args.len() == arg_names.len()
                    && args
                        .iter()
                        .zip(arg_names.iter())
                        .all(|(arg, arg_name)| is_named_shape(arg, arg_name))
            })
}

fn unspecified_type_arg_count(shape: &Value) -> Option<u64> {
    shape
        .get("unspecified_type_arg_count")
        .and_then(Value::as_u64)
}

fn contains_named_shape_with_unspecified_type_arg_count(
    shape: &Value,
    name: &str,
    unspecified_count: u64,
) -> bool {
    if is_named_shape_with_args(shape, name, &[])
        && unspecified_type_arg_count(shape) == Some(unspecified_count)
    {
        return true;
    }

    ["args", "bounds", "params"].iter().any(|field| {
        shape
            .get(field)
            .and_then(Value::as_array)
            .is_some_and(|children| {
                children.iter().any(|child| {
                    contains_named_shape_with_unspecified_type_arg_count(
                        child,
                        name,
                        unspecified_count,
                    )
                })
            })
    }) || shape.get("return_type").is_some_and(|child| {
        contains_named_shape_with_unspecified_type_arg_count(child, name, unspecified_count)
    })
}

#[test]
fn test_simple_int_annotation() {
    let tdir = TempDir::new().unwrap();
    let file_path = tdir.path().join("main.py");
    let code = "x: int = 42";
    fs_anyhow::write(&file_path, code).unwrap();

    let query = create_query();
    let module_name = ModuleName::from_str("main");
    let path = ModulePath::filesystem(file_path.clone());

    // Load the file
    let errors = query.add_files(vec![(module_name, path.clone())]);
    assert!(errors.is_empty(), "Unexpected errors: {:?}", errors);

    // Get types as pretty-printed JSON
    let types = query.get_types_in_file(module_name, path).unwrap();
    let actual = types_to_json_string(types);

    // Expected: the variable 'x', the annotation 'int', and the literal '42'
    let expected = r#"[
  {
    "location": {
      "end_col": 1,
      "end_line": 1,
      "start_col": 0,
      "start_line": 1
    },
    "type": "builtins.int"
  },
  {
    "location": {
      "end_col": 6,
      "end_line": 1,
      "start_col": 3,
      "start_line": 1
    },
    "type": "builtins.type[builtins.int]"
  },
  {
    "location": {
      "end_col": 11,
      "end_line": 1,
      "start_col": 9,
      "start_line": 1
    },
    "type": "typing.Literal[42]"
  }
]"#;

    assert_eq!(expected, actual);
}

#[test]
fn test_type_shapes_include_structured_named_callable_and_type_variable_data() {
    let tdir = TempDir::new().unwrap();
    let file_path = tdir.path().join("main.py");
    let code = r#"from typing import Callable, TypeVar

T = TypeVar("T", bound=int)

def apply(f: Callable[[int, str], bool], x: T) -> bool:
    values: list[int] = [1]
    return f(x, "ok")
"#;
    fs_anyhow::write(&file_path, code).unwrap();

    let query = create_query();
    let module_name = ModuleName::from_str("main");
    let path = ModulePath::filesystem(file_path.clone());

    let errors = query.add_files(vec![(module_name, path.clone())]);
    assert!(errors.is_empty(), "Unexpected errors: {:?}", errors);

    let shapes = type_shape_values(query.get_type_shapes_in_file(module_name, path).unwrap());
    assert!(
        shapes.iter().all(|shape| {
            shape.get("display").is_some_and(Value::is_string)
                && shape.get("kind").is_some_and(Value::is_string)
        }),
        "Expected every type shape to include display and kind:\n{shapes:#?}",
    );
    assert!(
        shapes.iter().any(|shape| is_named_shape_with_args(
            shape,
            "builtins.list",
            &["builtins.int"]
        )),
        "Expected a structured list[int] named shape:\n{shapes:#?}",
    );
    assert!(
        shapes.iter().any(|shape| {
            shape.get("kind").and_then(Value::as_str) == Some("callable")
                && shape
                    .get("params")
                    .and_then(Value::as_array)
                    .is_some_and(|params| {
                        params.len() == 2
                            && is_named_shape(&params[0], "builtins.int")
                            && is_named_shape(&params[1], "builtins.str")
                    })
                && shape
                    .get("return_type")
                    .is_some_and(|return_type| is_named_shape(return_type, "builtins.bool"))
        }),
        "Expected a structured callable shape:\n{shapes:#?}",
    );
    assert!(
        shapes.iter().any(|shape| {
            shape.get("kind").and_then(Value::as_str) == Some("type_variable")
                && shape.get("display").and_then(Value::as_str)
                    == Some("Variable[T (bound to builtins.int)]")
                && shape.get("name").and_then(Value::as_str) == Some("T")
                && shape
                    .get("bounds")
                    .and_then(Value::as_array)
                    .is_some_and(|bounds| {
                        bounds.len() == 1 && is_named_shape(&bounds[0], "builtins.int")
                    })
        }),
        "Expected a structured TypeVar shape with a bound:\n{shapes:#?}",
    );
}

#[test]
fn test_type_shapes_include_unspecified_type_arg_count_for_generic_classes() {
    let tdir = TempDir::new().unwrap();
    let file_path = tdir.path().join("main.py");
    let code = r#"from typing import Generic, TypeVar

T = TypeVar("T")

class Box(Generic[T]):
    pass

bare = Box
value: Box[int] = Box()
"#;
    fs_anyhow::write(&file_path, code).unwrap();

    let query = create_query();
    let module_name = ModuleName::from_str("main");
    let path = ModulePath::filesystem(file_path.clone());

    let errors = query.add_files(vec![(module_name, path.clone())]);
    assert!(errors.is_empty(), "Unexpected errors: {:?}", errors);

    let shapes = type_shape_values(query.get_type_shapes_in_file(module_name, path).unwrap());
    assert!(
        shapes.iter().any(|shape| {
            contains_named_shape_with_unspecified_type_arg_count(shape, "main.Box", 1)
        }),
        "Expected bare generic class `Box` to report one unspecified type arg:\n{shapes:#?}",
    );
    assert!(
        shapes.iter().any(|shape| {
            is_named_shape_with_args(shape, "main.Box", &["builtins.int"])
                && unspecified_type_arg_count(shape).is_none()
        }),
        "Expected instantiated `Box[int]` to omit unspecified type args:\n{shapes:#?}",
    );
}

#[test]
fn test_if_else_in_loop() {
    let tdir = TempDir::new().unwrap();
    let file_path = tdir.path().join("main.py");
    let code = r#"
class Foo:
    x: int | None
def f(foos: list[Foo]) -> int:
    n = 0
    xs = set()
    for foo in foos:
        if foo.x:
            xs.add(foo.x)
        else:
            n += 1
    return n + len(xs)
"#;
    fs_anyhow::write(&file_path, code).unwrap();

    let query = create_query();
    let module_name = ModuleName::from_str("main");
    let path = ModulePath::filesystem(file_path.clone());

    // Load the file
    let errors = query.add_files(vec![(module_name, path.clone())]);
    assert!(errors.is_empty(), "Unexpected errors: {:?}", errors);

    // Get types as pretty-printed JSON
    let types = query.get_types_in_file(module_name, path).unwrap();
    let actual = types_to_json_string(types);

    let expected = r#"[
  {
    "location": {
      "end_col": 5,
      "end_line": 3,
      "start_col": 4,
      "start_line": 3
    },
    "type": "builtins.int | None"
  },
  {
    "location": {
      "end_col": 17,
      "end_line": 3,
      "start_col": 7,
      "start_line": 3
    },
    "type": "builtins.type[builtins.int | None]"
  },
  {
    "location": {
      "end_col": 10,
      "end_line": 3,
      "start_col": 7,
      "start_line": 3
    },
    "type": "builtins.type[builtins.int]"
  },
  {
    "location": {
      "end_col": 17,
      "end_line": 3,
      "start_col": 13,
      "start_line": 3
    },
    "type": "None"
  },
  {
    "location": {
      "end_col": 21,
      "end_line": 4,
      "start_col": 12,
      "start_line": 4
    },
    "type": "builtins.type[builtins.list[main.Foo]]"
  },
  {
    "location": {
      "end_col": 16,
      "end_line": 4,
      "start_col": 12,
      "start_line": 4
    },
    "type": "builtins.type[builtins.list]"
  },
  {
    "location": {
      "end_col": 20,
      "end_line": 4,
      "start_col": 17,
      "start_line": 4
    },
    "type": "builtins.type[main.Foo]"
  },
  {
    "location": {
      "end_col": 29,
      "end_line": 4,
      "start_col": 26,
      "start_line": 4
    },
    "type": "builtins.type[builtins.int]"
  },
  {
    "location": {
      "end_col": 5,
      "end_line": 5,
      "start_col": 4,
      "start_line": 5
    },
    "type": "typing.Literal[0]"
  },
  {
    "location": {
      "end_col": 9,
      "end_line": 5,
      "start_col": 8,
      "start_line": 5
    },
    "type": "typing.Literal[0]"
  },
  {
    "location": {
      "end_col": 6,
      "end_line": 6,
      "start_col": 4,
      "start_line": 6
    },
    "type": "builtins.set[builtins.int]"
  },
  {
    "location": {
      "end_col": 14,
      "end_line": 6,
      "start_col": 9,
      "start_line": 6
    },
    "type": "builtins.set[builtins.int]"
  },
  {
    "location": {
      "end_col": 12,
      "end_line": 6,
      "start_col": 9,
      "start_line": 6
    },
    "type": "builtins.type[builtins.set]"
  },
  {
    "location": {
      "end_col": 11,
      "end_line": 7,
      "start_col": 8,
      "start_line": 7
    },
    "type": "main.Foo"
  },
  {
    "location": {
      "end_col": 19,
      "end_line": 7,
      "start_col": 15,
      "start_line": 7
    },
    "type": "builtins.list[main.Foo]"
  },
  {
    "location": {
      "end_col": 16,
      "end_line": 8,
      "start_col": 11,
      "start_line": 8
    },
    "type": "builtins.int | None"
  },
  {
    "location": {
      "end_col": 14,
      "end_line": 8,
      "start_col": 11,
      "start_line": 8
    },
    "type": "main.Foo"
  },
  {
    "location": {
      "end_col": 25,
      "end_line": 9,
      "start_col": 12,
      "start_line": 9
    },
    "type": "None"
  },
  {
    "location": {
      "end_col": 18,
      "end_line": 9,
      "start_col": 12,
      "start_line": 9
    },
    "type": "BoundMethod[builtins.set[builtins.int], (self: builtins.set[builtins.int], element: builtins.int, /) -> None]"
  },
  {
    "location": {
      "end_col": 14,
      "end_line": 9,
      "start_col": 12,
      "start_line": 9
    },
    "type": "builtins.set[builtins.int]"
  },
  {
    "location": {
      "end_col": 24,
      "end_line": 9,
      "start_col": 19,
      "start_line": 9
    },
    "type": "builtins.int"
  },
  {
    "location": {
      "end_col": 22,
      "end_line": 9,
      "start_col": 19,
      "start_line": 9
    },
    "type": "main.Foo"
  },
  {
    "location": {
      "end_col": 13,
      "end_line": 11,
      "start_col": 12,
      "start_line": 11
    },
    "type": "builtins.int"
  },
  {
    "location": {
      "end_col": 18,
      "end_line": 11,
      "start_col": 17,
      "start_line": 11
    },
    "type": "typing.Literal[1]"
  },
  {
    "location": {
      "end_col": 22,
      "end_line": 12,
      "start_col": 11,
      "start_line": 12
    },
    "type": "builtins.int"
  },
  {
    "location": {
      "end_col": 12,
      "end_line": 12,
      "start_col": 11,
      "start_line": 12
    },
    "type": "builtins.int"
  },
  {
    "location": {
      "end_col": 22,
      "end_line": 12,
      "start_col": 15,
      "start_line": 12
    },
    "type": "builtins.int"
  },
  {
    "location": {
      "end_col": 18,
      "end_line": 12,
      "start_col": 15,
      "start_line": 12
    },
    "type": "(obj: typing.Sized, /) -> builtins.int"
  },
  {
    "location": {
      "end_col": 21,
      "end_line": 12,
      "start_col": 19,
      "start_line": 12
    },
    "type": "builtins.set[builtins.int]"
  }
]"#;

    assert_eq!(expected, actual);
}

#[test]
fn test_lambda_param_var_leak_regression() {
    let tdir = TempDir::new().unwrap();
    let file_path = tdir.path().join("main.py");
    // Minimal reproducer from mypy-primer: a lambda used as an argument can
    // leave a Binding::LambdaParameter Var that is not present in the solving
    // thread's Variables map.
    let code = r#"
from typing import Callable

def find_self_type(t: object, f: Callable[[str], object]) -> bool:
    return True

class A:
    def m(self, t: object) -> None:
        if find_self_type(t, lambda name: name):
            pass
"#;
    fs_anyhow::write(&file_path, code).unwrap();

    let query = create_query();
    let module_name = ModuleName::from_str("main");
    let path = ModulePath::filesystem(file_path.clone());

    let errors = query.add_files(vec![(module_name, path.clone())]);
    assert!(errors.is_empty(), "Unexpected errors: {:?}", errors);

    // This currently panics with:
    // "Internal error: a variable has leaked across thread boundaries."
    // Keep as a regression to ensure lambda-parameter Vars cannot leak.
    let _ = query.get_types_in_file(module_name, path).unwrap();
}

/// Regression test: legacy implicit type alias to a builtin container must not
/// produce double-qualified names like `typing.builtins.type[...]` in query mode.
#[test]
fn test_legacy_implicit_type_alias_no_double_qualification() {
    let tdir = TempDir::new().unwrap();
    let file_path = tdir.path().join("main.py");
    let code = r#"
from typing import Any
RawData = dict[str, Any]
def f(x: RawData) -> None:
    pass
"#;
    fs_anyhow::write(&file_path, code).unwrap();

    let query = create_query();
    let module_name = ModuleName::from_str("main");
    let path = ModulePath::filesystem(file_path.clone());

    let errors = query.add_files(vec![(module_name, path.clone())]);
    assert!(errors.is_empty(), "Unexpected errors: {:?}", errors);

    let types = query.get_types_in_file(module_name, path).unwrap();
    let actual = types_to_json_string(types);

    // The type of `x` should NOT contain "typing.builtins." — that's double-qualification.
    assert!(
        !actual.contains("typing.builtins."),
        "Double-qualified 'typing.builtins.' found in output:\n{actual}",
    );
}

/// Regression test: `Annotated` type alias must not produce double-qualified
/// names like `typing.typing.Annotated[...]` in query mode.
#[test]
fn test_annotated_type_alias_no_double_qualification() {
    let tdir = TempDir::new().unwrap();
    let file_path = tdir.path().join("main.py");
    let code = r#"
from typing import Annotated, TypeAlias
PrimitiveIntID = Annotated[int, "metadata"]
def f(x: PrimitiveIntID) -> None:
    pass
"#;
    fs_anyhow::write(&file_path, code).unwrap();

    let query = create_query();
    let module_name = ModuleName::from_str("main");
    let path = ModulePath::filesystem(file_path.clone());

    let errors = query.add_files(vec![(module_name, path.clone())]);
    assert!(errors.is_empty(), "Unexpected errors: {:?}", errors);

    let types = query.get_types_in_file(module_name, path).unwrap();
    let actual = types_to_json_string(types);

    // The output should NOT contain "typing.typing." — that's double-qualification.
    assert!(
        !actual.contains("typing.typing."),
        "Double-qualified 'typing.typing.' found in output:\n{actual}",
    );
}

/// Explicit TypeAlias should show `typing.TypeAlias[...]`, not
/// `typing.typing.TypeAlias[...]`.
#[test]
fn test_explicit_type_alias_no_double_qualification() {
    let tdir = TempDir::new().unwrap();
    let file_path = tdir.path().join("main.py");
    let code = r#"
from typing import Any, TypeAlias
MyDict: TypeAlias = dict[str, Any]
def f(x: MyDict) -> None:
    pass
"#;
    fs_anyhow::write(&file_path, code).unwrap();

    let query = create_query();
    let module_name = ModuleName::from_str("main");
    let path = ModulePath::filesystem(file_path.clone());

    let errors = query.add_files(vec![(module_name, path.clone())]);
    assert!(errors.is_empty(), "Unexpected errors: {:?}", errors);

    let types = query.get_types_in_file(module_name, path).unwrap();
    let actual = types_to_json_string(types);

    // Should not have double-qualified typing prefix.
    assert!(
        !actual.contains("typing.typing."),
        "Double-qualified 'typing.typing.' found in output:\n{actual}",
    );
    // Should not have typing.builtins. either.
    assert!(
        !actual.contains("typing.builtins."),
        "Double-qualified 'typing.builtins.' found in output:\n{actual}",
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
