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
use serde_json::json;
use tempfile::TempDir;

use crate::config::config::ConfigFile;
use crate::config::finder::ConfigFinder;
use crate::query::Query;
use crate::test::util::init_test;

/// Helper to create a Query with a ConfigFinder that doesn't use sourcedb.
fn create_query() -> Query {
    init_test();
    let mut config = ConfigFile::default();
    config.python_environment.set_empty_to_default();
    config.configure();
    let config = ArcId::new(config);
    Query::new(ConfigFinder::new_constant(config))
}

/// Convert the result of get_types_in_file to a pretty-printed JSON string.
/// This format makes test failures easy to patch by copy-pasting the actual output.
fn types_to_json_string(types: Vec<(crate::query::PythonASTRange, String)>) -> String {
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
    "type": "type[builtins.int]"
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
    "type": "type[builtins.int | None]"
  },
  {
    "location": {
      "end_col": 10,
      "end_line": 3,
      "start_col": 7,
      "start_line": 3
    },
    "type": "type[builtins.int]"
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
    "type": "type[builtins.list[main.Foo]]"
  },
  {
    "location": {
      "end_col": 16,
      "end_line": 4,
      "start_col": 12,
      "start_line": 4
    },
    "type": "type[builtins.list]"
  },
  {
    "location": {
      "end_col": 20,
      "end_line": 4,
      "start_col": 17,
      "start_line": 4
    },
    "type": "type[main.Foo]"
  },
  {
    "location": {
      "end_col": 29,
      "end_line": 4,
      "start_col": 26,
      "start_line": 4
    },
    "type": "type[builtins.int]"
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
    "type": "type[builtins.set]"
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
