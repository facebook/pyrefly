/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::Url;
use serde_json::json;

use crate::object_model::InitializeSettings;
use crate::object_model::LspInteraction;
use crate::util::get_test_files_root;

#[test]
fn test_hover_basic() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("basic"));
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("bar.py");

    interaction
        .client
        .hover("bar.py", 7, 5)
        .expect_response(json!({
            "contents": {
                "kind": "markdown",
                "value": "```python\n(variable) foo: Literal[3]\n```",
            }
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn hover_on_attr_of_pyi_assignment_shows_pyi_type() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            ..Default::default()
        })
        .unwrap();
    let file = "attributes_of_py/src_with_assignments.py";
    interaction.client.did_open(file);

    interaction
        .client
        .hover(file, 8, 8)
        .expect_hover_response_with_markup(|x| x.is_some_and(|x| x.contains("y: int")))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn hover_attribute_prefers_py_docstring_over_pyi() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    let file = "attributes_of_py_docstrings/src.py";
    interaction.client.did_open(file);
    interaction
        .client
        .hover(file, 9, 10)
        .expect_hover_response_with_markup(|x| {
            x.is_some_and(|x| {
                // a link to the .pyi file proves that the type is coming from the .pyi
                x.contains("Docstring coming from the .py implementation.") && x.contains("lib.pyi")
            })
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn hover_shows_third_party_function_name() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("rename_third_party"));
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("user_code.py");
    // Column/line values follow LSP's zero-based positions
    interaction
        .client
        .hover("user_code.py", 14, 25)
        .expect_hover_response_with_markup(|value| {
            value.is_some_and(|text| text.contains("external_function"))
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_hover_import() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("basic"));
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("foo.py");

    interaction
        .client
        .hover("foo.py", 6, 16)
        .expect_hover_response_with_markup(|value| {
            value.is_some_and(|text| {
                text.contains("(class) Bar: def Bar() -> Bar: ...")
                    && text.contains(
                        Url::from_file_path(root.path().join("basic/bar.py"))
                            .unwrap()
                            .as_str(),
                    )
            })
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_hover_suppressed_error() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("suppression.py");

    // Standalone suppression, next line has a suppressed error
    interaction
        .client
        .hover("suppression.py", 5, 10)
        .expect_response(json!({
            "contents": {
                "kind": "markdown",
                "value": "**Suppressed Error**\n\n`unsupported-operation`: `+` is not supported between `Literal[1]` and `Literal['']`\n  Argument `Literal['']` is not assignable to parameter `value` with type `int` in function `int.__add__`",
            }
        })).unwrap();

    // Trailing suppression, same line has a suppressed error
    interaction
        .client
        .hover("suppression.py", 8, 15)
        .expect_response(json!({
            "contents": {
                "kind": "markdown",
                "value": "**Suppressed Error**\n\n`unsupported-operation`: `+` is not supported between `Literal[2]` and `Literal['']`\n  Argument `Literal['']` is not assignable to parameter `value` with type `int` in function `int.__add__`",
            }
        })).unwrap();

    // Trailing suppression, suppressed error does not match
    interaction
        .client
        .hover("suppression.py", 10, 15)
        .expect_response(json!({
            "contents": {
                "kind": "markdown",
                "value": "**No errors suppressed by this ignore**\n\n_The ignore comment may have an incorrect error code or there may be no errors on this line._",
            }
        })).unwrap();

    // Trailing suppression, next line has an unsuppressed error
    interaction
        .client
        .hover("suppression.py", 12, 15)
        .expect_response(json!({
            "contents": {
                "kind": "markdown",
                "value": "**No errors suppressed by this ignore**\n\n_The ignore comment may have an incorrect error code or there may be no errors on this line._",
            }
        })).unwrap();

    // Standalone suppression, no errors
    interaction
        .client
        .hover("suppression.py", 15, 10)
        .expect_response(json!({
            "contents": {
                "kind": "markdown",
                "value": "**No errors suppressed by this ignore**\n\n_The ignore comment may have an incorrect error code or there may be no errors on this line._",
            }
        })).unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_hover_suppressed_error_subkind() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("suppression_subkind.py");

    // bad-override-mutable-attribute is a sub-kind of bad-override, so
    // `# pyrefly: ignore[bad-override]` should suppress it and hovering
    // over the comment should show the suppressed error.
    interaction
        .client
        .hover("suppression_subkind.py", 11, 15)
        .expect_response(json!({
            "contents": {
                "kind": "markdown",
                "value": "**Suppressed Error**\n\n`bad-override-mutable-attribute`: Class member `B.x` overrides parent class `A` in an inconsistent manner\n  `B.x` has type `int`, which is not consistent with `int | str` in `A.x` (the type of read-write attributes cannot be changed)",
            }
        })).unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_hover_suppressed_error_deprecated_alias() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction
        .client
        .did_open("suppression_deprecated_alias.py");

    // bad-override-param-name error suppressed via the deprecated alias
    // bad-param-name-override. Hovering should show the suppressed error
    // with the new error code name.
    interaction
        .client
        .hover("suppression_deprecated_alias.py", 12, 40)
        .expect_response(json!({
            "contents": {
                "kind": "markdown",
                "value": "**Suppressed Error**\n\n`bad-override-param-name`: Class member `B.f` overrides parent class `A` in an inconsistent manner\n  Got parameter name `x1`, expected `x`",
            }
        })).unwrap();

    interaction.shutdown().unwrap();
}
