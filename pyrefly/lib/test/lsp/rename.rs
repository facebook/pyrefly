/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fs;
use std::ops::Range;

use dupe::Dupe;
use itertools::Itertools;
use pretty_assertions::assert_eq;
use pyrefly_build::handle::Handle;
use pyrefly_python::module::TextRangeWithModule;
use pyrefly_python::module_path::ModulePath;
use ruff_text_size::TextSize;
use tempfile::TempDir;

use crate::state::lsp::FindPreference;
use crate::state::lsp::ReferenceOptions;
use crate::state::require::Require;
use crate::state::state::State;
use crate::test::util::TestEnv;
use crate::test::util::code_frame_of_source_at_range;
use crate::test::util::get_batched_lsp_operations_report;

/// Applies the reference edits used by rename to a source file, its stub, and a consumer.
///
/// The rename starts in `lib.py` unless `shadowing_source` is given. In that case, the rename
/// starts in `shadow/lib.py`, a module that is also named `lib` but that imports never resolve to.
fn rename_paired_symbol(
    source: &str,
    stub: &str,
    consumer: &str,
    shadowing_source: Option<&str>,
    position: usize,
) -> Vec<(String, String)> {
    let root = TempDir::new().unwrap();
    let mut env = TestEnv::new();
    for (name, file, code) in [
        ("lib", "lib.py", source),
        ("lib", "lib.pyi", stub),
        ("consumer", "consumer.py", consumer),
    ] {
        let path = root.path().join(file);
        fs::write(&path, code).unwrap();
        env.add_real_path(name, path);
    }
    let renamed_path = match shadowing_source {
        Some(code) => {
            let path = root.path().join("shadow").join("lib.py");
            fs::create_dir(path.parent().unwrap()).unwrap();
            fs::write(&path, code).unwrap();
            path
        }
        None => root.path().join("lib.py"),
    };
    let (state, handle) = env.to_state();
    let stub_handle = handle("lib");
    let source_handle = Handle::new(
        stub_handle.module(),
        ModulePath::filesystem(renamed_path),
        *stub_handle.sys_info(),
    );
    let mut transaction = state.transaction();
    transaction.run(&[source_handle.dupe()], Require::Everything, None);
    let definitions = transaction
        .find_definition(
            &source_handle,
            TextSize::try_from(position).unwrap(),
            FindPreference {
                resolve_call_dunders: false,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(definitions.len(), 1);
    let definition = definitions.first();
    let references = transaction
        .find_global_references_from_definition(
            *source_handle.sys_info(),
            definition.metadata.clone(),
            TextRangeWithModule::new(definition.module.clone(), definition.definition_range),
            ReferenceOptions::textual_only(true),
        )
        .unwrap();
    let mut result = Vec::new();
    for (module, ranges) in references {
        let mut code = module.contents().as_str().to_owned();
        for range in ranges.into_iter().rev() {
            code.replace_range(Range::<usize>::from(range), "renamed");
        }
        let relative_path = module
            .path()
            .as_path()
            .strip_prefix(root.path())
            .unwrap()
            .components()
            .map(|component| component.as_os_str().to_str().unwrap())
            .join("/");
        result.push((relative_path, code));
    }
    result.sort();
    result
}

#[test]
fn test_rename_source_updates_stub_and_consumers() {
    for (source, stub, consumer, name) in [
        (
            "def item() -> int:\n    return 1\n",
            "def item() -> int: ...\n",
            "from lib import item\nitem()\n",
            "item",
        ),
        (
            "class Item:\n    pass\n",
            "class Item: ...\n",
            "from lib import Item\nx: Item = Item()\n",
            "Item",
        ),
        (
            "item = 1\n",
            "item: int\n",
            "from lib import item\nx = item\n",
            "item",
        ),
        (
            "item: int = 1\n",
            "item: int\n",
            "from lib import item\nx = item\n",
            "item",
        ),
        (
            "class Item:\n    def item(self) -> int:\n        return 1\n",
            "class Item:\n    def item(self) -> int: ...\n",
            "from lib import Item\nItem().item()\n",
            "item",
        ),
        (
            "class Item:\n    item: int = 1\n",
            "class Item:\n    item: int\n",
            "from lib import Item\nx = Item.item\n",
            "item",
        ),
        (
            "class Outer:\n    class Item:\n        pass\n",
            "class Outer:\n    class Item: ...\n",
            "from lib import Outer\nx: Outer.Item = Outer.Item()\n",
            "Item",
        ),
    ] {
        assert_eq!(
            rename_paired_symbol(source, stub, consumer, None, source.find(name).unwrap()),
            vec![
                ("consumer.py".to_owned(), consumer.replace(name, "renamed")),
                ("lib.py".to_owned(), source.replace(name, "renamed")),
                ("lib.pyi".to_owned(), stub.replace(name, "renamed")),
            ],
        );
    }
}

#[test]
fn test_rename_local_symbols_does_not_change_stub_or_consumers() {
    for (body, position) in [
        (
            "def wrapper(item: int) -> int:\n    return item\n",
            "item: int",
        ),
        (
            "def wrapper() -> int:\n    item = 1\n    return item\n",
            "item =",
        ),
        (
            "def wrapper() -> int:\n    def item() -> int:\n        return 1\n    return item()\n",
            "item()",
        ),
    ] {
        for class_scope in [false, true] {
            let (source, stub, consumer) = if class_scope {
                (
                    format!(
                        "class Item:\n    def item(self) -> int:\n        return 1\n{}",
                        body.replacen("wrapper(", "wrapper(self, ", 1)
                            .lines()
                            .map(|line| format!("    {line}\n"))
                            .collect::<String>()
                    ),
                    "class Item:\n    def item(self) -> int: ...\n",
                    "from lib import Item\nItem().item()\n",
                )
            } else {
                (
                    format!("def item() -> int:\n    return 1\n{body}"),
                    "def item() -> int: ...\n",
                    "from lib import item\nitem()\n",
                )
            };
            let wrapper_start = source.find("def wrapper").unwrap();
            let start = wrapper_start + source[wrapper_start..].find(position).unwrap();
            let mut expected = source[..start].to_owned();
            expected.push_str(&source[start..].replace("item", "renamed"));
            assert_eq!(
                rename_paired_symbol(&source, stub, consumer, None, start),
                vec![("lib.py".to_owned(), expected)],
            );
        }
    }
}

#[test]
fn test_rename_shadowed_source_does_not_change_stub_or_consumers() {
    let source = "def item() -> int:\n    return 1\n";
    let stub = "def item() -> int: ...\n";
    let consumer = "from lib import item\nitem()\n";
    // Imports of `lib` resolve to `lib.pyi`, which describes `lib.py` rather than
    // `shadow/lib.py`, so the stub and its consumers are not renamed.
    assert_eq!(
        rename_paired_symbol(
            source,
            stub,
            consumer,
            Some(source),
            source.find("item").unwrap()
        ),
        vec![(
            "shadow/lib.py".to_owned(),
            source.replace("item", "renamed")
        )],
    );
}

fn get_test_report(state: &State, handle: &Handle, position: TextSize) -> String {
    let transaction = state.transaction();
    // Mirror the reference-collection half of `textDocument/rename`: only ranges whose text
    // is the symbol being renamed, since every returned range is rewritten in place.
    let ranges =
        transaction.find_local_references(handle, position, ReferenceOptions::textual_only(true));
    let module_info = transaction.get_module_info(handle).unwrap();
    format!(
        "Rename locations:\n{}",
        ranges
            .into_iter()
            .map(|range| code_frame_of_source_at_range(module_info.contents(), range))
            .join("\n")
    )
}

// BUG: prepare_rename returns None for __all__ and __slots__ entries.
// It should return the range of foo or bar inside the quotes (issue #4344).
#[test]
fn prepare_rename_symbol_literals() {
    let code = r#"
foo = 1
__all__ = ["foo"]
#            ^

class C:
    __slots__ = ("bar",)
#                  ^

    def __init__(self) -> None:
        self.bar = 1
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], |state, handle, position| {
        format!(
            "Rename range: {:?}",
            state.transaction().prepare_rename(handle, position)
        )
    });
    assert_eq!(
        r#"
# main.py
3 | __all__ = ["foo"]
                 ^
Rename range: None

7 |     __slots__ = ("bar",)
                       ^
Rename range: None
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn test_rename_parameter_updates_keyword_arguments() {
    let code = r#"
def greet(name, message):
    """Greet someone with a message."""
    print(f"{message}, {name}!")
    return name

def another_func():
    result = greet(name="Alice", message="Hello")
#                  ^
    return result
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
8 |     result = greet(name="Alice", message="Hello")
                       ^
Rename locations:
2 | def greet(name, message):
              ^^^^
4 |     print(f"{message}, {name}!")
                            ^^^^
5 |     return name
               ^^^^
8 |     result = greet(name="Alice", message="Hello")
                       ^^^^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn test_rename_parameter_only_updates_correct_function() {
    let code = r#"
def func1(name, message):
    """First function with name parameter."""
    print(f"{message}, {name}!")
    return name

def func2(name, value):
    """Second function with same name parameter."""
    print(f"Value: {value}, Name: {name}")
    return name

def caller():
    result1 = func1(name="Alice", message="Hello")
#                   ^
    result2 = func2(name="Bob", value=42)
    return result1, result2
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
13 |     result1 = func1(name="Alice", message="Hello")
                         ^
Rename locations:
2 | def func1(name, message):
              ^^^^
4 |     print(f"{message}, {name}!")
                            ^^^^
5 |     return name
               ^^^^
13 |     result1 = func1(name="Alice", message="Hello")
                         ^^^^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn test_rename_function_parameter_updates_call_sites() {
    let code = r#"
def greet(name, message):
#         ^
    """Greet someone with a message."""
    print(f"{message}, {name}!")
    return name

def caller():
    result1 = greet(name="Alice", message="Hello")
    result2 = greet(message="Hi", name="Bob")
    result3 = greet(name="Charlie", message="Hey")
    return result1, result2, result3
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | def greet(name, message):
              ^
Rename locations:
2 | def greet(name, message):
              ^^^^
5 |     print(f"{message}, {name}!")
                            ^^^^
6 |     return name
               ^^^^
9 |     result1 = greet(name="Alice", message="Hello")
                        ^^^^
10 |     result2 = greet(message="Hi", name="Bob")
                                       ^^^^
11 |     result3 = greet(name="Charlie", message="Hey")
                         ^^^^
"#
        .trim(),
        report.trim(),
    );
}

/// Find-references reports `Foo()` as a reference to `Foo.__init__`, but that range spells the
/// class name. Rename must skip it, or renaming `__init__` would rewrite the constructor call.
#[test]
fn test_rename_dunder_init_skips_constructor_call_sites() {
    let code = r#"
class Foo:
    def __init__(self): ...
    #   ^

Foo()
Foo().__init__()
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
3 |     def __init__(self): ...
            ^
Rename locations:
3 |     def __init__(self): ...
            ^^^^^^^^
7 | Foo().__init__()
          ^^^^^^^^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn test_rename_legacy_type_parameters_updates_constructor_names() {
    let code = r#"
from typing import Callable, ParamSpec, TypeVar, TypeVarTuple

T = TypeVar("T")
P = ParamSpec(name="P")
Ts = TypeVarTuple("Ts")
unrelated = "T"

def f(value: T) -> T:
#            ^
    return value

def g(func: Callable[P, None]) -> None:
#                    ^
    pass

def h(value: tuple[*Ts]) -> None:
#                   ^
    pass
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
9 | def f(value: T) -> T:
                 ^
Rename locations:
4 | T = TypeVar("T")
    ^
4 | T = TypeVar("T")
                 ^
9 | def f(value: T) -> T:
                 ^
9 | def f(value: T) -> T:
                       ^

13 | def g(func: Callable[P, None]) -> None:
                          ^
Rename locations:
5 | P = ParamSpec(name="P")
    ^
5 | P = ParamSpec(name="P")
                        ^
13 | def g(func: Callable[P, None]) -> None:
                          ^

17 | def h(value: tuple[*Ts]) -> None:
                         ^
Rename locations:
6 | Ts = TypeVarTuple("Ts")
    ^^
6 | Ts = TypeVarTuple("Ts")
                       ^^
17 | def h(value: tuple[*Ts]) -> None:
                         ^^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn test_rename_from_inside_quoted_annotation() {
    let code = r#"
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lib import Helper

class Model: ...

def make(helper: "Helper") -> "list[Model]":
#                                   ^
    return [Model()]
x: "Helper | None" = None
#   ^
"#;
    let report = get_batched_lsp_operations_report(
        &[("main", code), ("lib", "class Helper: ...\n")],
        get_test_report,
    );
    assert_eq!(
        r#"
# main.py
9 | def make(helper: "Helper") -> "list[Model]":
                                        ^
Rename locations:
7 | class Model: ...
          ^^^^^
9 | def make(helper: "Helper") -> "list[Model]":
                                        ^^^^^
11 |     return [Model()]
                 ^^^^^

12 | x: "Helper | None" = None
         ^
Rename locations:
5 |     from lib import Helper
                        ^^^^^^
9 | def make(helper: "Helper") -> "list[Model]":
                      ^^^^^^
12 | x: "Helper | None" = None
         ^^^^^^


# lib.py
"#
        .trim(),
        report.trim(),
    );
}
