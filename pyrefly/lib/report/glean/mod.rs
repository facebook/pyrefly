/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_build::handle::Handle;

use crate::report::glean::facts::Glean;
use crate::state::state::Transaction;

pub mod convert;
pub mod facts;
pub mod schema;

pub fn glean(transaction: &Transaction, handle: &Handle) -> String {
    fn f(transaction: &Transaction, handle: &Handle) -> Option<Glean> {
        Some(Glean::new(transaction, handle))
    }

    let data = f(transaction, handle).expect("Glean data be ready").entries;
    serde_json::to_string_pretty(&data).unwrap()
}

#[test]
fn snapshot_test() {
    use std::env;
    use std::fs;

    use crate::state::require::Require;
    use crate::test::util::mk_multi_file_state_assert_no_errors;

    // Helper function to sort JSON for deterministic comparison
    fn normalize_json(json_str: &str) -> String {
        let mut value: serde_json::Value =
            serde_json::from_str(json_str).expect("Failed to parse JSON");

        fn sort_arrays(value: &mut serde_json::Value) {
            match value {
                serde_json::Value::Array(arr) => {
                    for item in arr.iter_mut() {
                        sort_arrays(item);
                    }
                    arr.sort_by(|a, b| {
                        serde_json::to_string(a)
                            .unwrap_or_default()
                            .cmp(&serde_json::to_string(b).unwrap_or_default())
                    });
                }
                serde_json::Value::Object(obj) => {
                    for (_, v) in obj.iter_mut() {
                        sort_arrays(v);
                    }
                }
                _ => {}
            }
        }

        sort_arrays(&mut value);
        serde_json::to_string_pretty(&value).unwrap()
    }

    // Simple line-by-line diff like git/source control
    fn create_simple_diff(expected: &str, actual: &str) -> String {
        let expected_lines: Vec<&str> = expected.lines().collect();
        let actual_lines: Vec<&str> = actual.lines().collect();

        let mut diff_lines = Vec::new();
        let max_lines = expected_lines.len().max(actual_lines.len());

        for i in 0..max_lines {
            match (expected_lines.get(i), actual_lines.get(i)) {
                (Some(exp_line), Some(act_line)) if exp_line != act_line => {
                    diff_lines.push(format!("- {exp_line}"));
                    diff_lines.push(format!("+ {act_line}"));
                }
                (Some(exp_line), None) => {
                    diff_lines.push(format!("- {exp_line}"));
                }
                (None, Some(act_line)) => {
                    diff_lines.push(format!("+ {act_line}"));
                }
                _ => {} // Lines are the same or both None
            }
        }

        if diff_lines.is_empty() {
            "No differences found".to_owned()
        } else {
            format!("Diff (- expected, + actual):\n{}", diff_lines.join("\n"))
        }
    }

    let simple_code = r#"def hello_world():
    return "Hello, World!"

x = 42
"""Answer to everything"""
y = hello_world()"#;

    let classes_code = r#"
import typing

T = typing.TypeVar('T')

class Animal(typing.Generic[T]):
    def __init__(self, name: str):
        self.name = name

    def speak(self) -> str:
        return f"{self.name} makes a sound"

class Dog(Animal):
    def speak(self) -> str:
        return f"{self.name} barks"

dog = Dog("Buddy")
sound = dog.speak()"#;

    let imports_code = r#"import os
from typing import List
from .simple import hello_world
from os import *

def process_files(paths: List[str]) -> int:
    """Count files in given paths."""
    count = 0
    for path in paths:
        if os.path.exists(path):
            count += 1
    return count

# Use relative import
greeting = hello_world()
result = process_files(["/tmp", "/home"])"#;

    let try_except_code = r#"try:
    pass
except Exception as ex:
    pass"#;

    let return_types = r#"from typing import Callable, Dict, List, Optional, Set, Union

class FooClass:
    pass

def complex_return() -> (
    Union[
        Union[FooClass, Callable[[int], bool], int | str | None],
        List[int | str],
        Dict[str, int | str],
        Optional[int | str],
        Set[List[str]],
    ]
):
    return None
    "#;

    let calls = r#"from dataclasses import dataclass
@dataclass(frozen=True)
class A:
    pass

print("hello")
print(b"world")

"#;

    let type_lit_str = r#"from typing import Union
class B:
    class A:
        pass

    def other(self, x: "B") -> "Union[list[B], A, bool]":
        return False
"#;

    let exports_all = r#"from typing import Union as U
x = 1
y = 2

__all__ = ["x", "U"]

__all__ += ["y"]
"#;

    let generated_file = format!(
        r#" # {}generated SignedSource<<abcde123456>>
 # @codegen-command : scripts/foo/bar
 # @codegen-source FooConfig: this/is/the/Way.php
 # @codegen-class : BarCodeGen

def helloworld():
    pass
"#,
        "@",
    );

    let files = [
        ("simple", simple_code),
        ("classes", classes_code),
        ("imports", imports_code),
        ("try_except", try_except_code),
        ("return_types", return_types),
        ("calls", calls),
        ("type_lit_str", type_lit_str),
        ("exports_all", exports_all),
        ("generated_file", &generated_file),
    ];
    let (handles, state) = mk_multi_file_state_assert_no_errors(&files, Require::Everything);
    let transaction = state.transaction();

    let update_snapshots = env::var("GLEAN_SNAPSHOTS_WRITE_PATH").is_ok();

    let snapshots_path = if !update_snapshots {
        env::var("GLEAN_SNAPSHOTS_PATH")
            .expect("GLEAN_SNAPSHOTS_PATH env var not set: buck should set this automatically")
    } else {
        env::var("GLEAN_SNAPSHOTS_WRITE_PATH").unwrap()
    };

    let mut snapshot_dir = std::env::current_dir().expect("Failed to get current directory");
    snapshot_dir.push(snapshots_path);

    for (filename, _) in &files {
        let handle = handles
            .get(filename)
            .expect("Handle should exist for test file");
        let glean_output = glean(&transaction, handle);
        let output_path = snapshot_dir.join(format!("{filename}.json"));

        if update_snapshots {
            let normalized_output = normalize_json(&glean_output);
            let output_with_header = &normalized_output;
            fs::write(&output_path, output_with_header).expect("Failed to write JSON output");
            println!(
                "Updated snapshot for {} -> {}",
                filename,
                output_path.display()
            );
            continue;
        }

        assert!(
            output_path.exists(),
            "Snapshot file does not exist: {}\nRun with GLEAN_SNAPSHOTS_WRITE_PATH set to update snapshots.",
            output_path.display()
        );

        let expected_content =
            fs::read_to_string(&output_path).expect("Failed to read existing snapshot");

        let normalized_actual = normalize_json(&glean_output);
        let normalized_expected = normalize_json(&expected_content);

        assert!(
            normalized_actual.trim() == normalized_expected.trim(),
            "Snapshot mismatch for {filename}!\n\n{}\n\nTo update snapshots, run with GLEAN_SNAPSHOTS_WRITE_PATH set. If using buck, add -- --env GLEAN_SNAPSHOTS_WRITE_PATH=\"path\" to the command.",
            create_simple_diff(&normalized_expected, &normalized_actual)
        );

        println!("✓ Snapshot matches for {filename}");
    }

    if update_snapshots {
        println!("All snapshots updated successfully!");
    } else {
        println!("All snapshots match!");
    }
}
