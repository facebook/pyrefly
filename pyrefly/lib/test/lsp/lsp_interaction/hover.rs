/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_server::Response;
use lsp_types::Url;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_hover_basic() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("basic"));
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });

    interaction.server.did_open("bar.py");
    interaction.server.hover("bar.py", 7, 5);

    interaction.client.expect_response(Response {
        id: interaction.server.current_request_id(),
        result: Some(serde_json::json!({
            "contents": {
                "kind": "markdown",
                "value": "```python\n(variable) foo: Literal[3]\n```",
            }
        })),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_hover_import() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("basic"));
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });

    interaction.server.did_open("foo.py");
    interaction.server.hover("foo.py", 6, 16);

    interaction.client.expect_response(Response {
        id: interaction.server.current_request_id(),
        result: Some(serde_json::json!({
            "contents": {
                "kind": "markdown",
                "value": "```python\n(class) Bar: type[Bar]\n```\n\nGo to [Bar](".to_owned()
                    + Url::from_file_path(root.path().join("basic/bar.py")).unwrap().as_str()
                    + "#L7,7)",
            }
        })),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_hover_suppressed_error() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(None),
        ..Default::default()
    });

    interaction.server.did_open("suppression.py");

    // Standalone suppression, next line has a suppressed error
    interaction.server.hover("suppression.py", 5, 10);
    interaction.client.expect_response(Response {
        id: interaction.server.current_request_id(),
        result: Some(serde_json::json!({
            "contents": {
                "kind": "markdown",
                "value": "**Suppressed Error**\n\n`unsupported-operation`: `+` is not supported between `Literal[1]` and `Literal['']`\n  Argument `Literal['']` is not assignable to parameter `value` with type `int` in function `int.__add__`",
            }
        })),
        error: None,
    });

    // Trailing suppression, same line has a suppressed error
    interaction.server.hover("suppression.py", 8, 15);
    interaction.client.expect_response(Response {
        id: interaction.server.current_request_id(),
        result: Some(serde_json::json!({
            "contents": {
                "kind": "markdown",
                "value": "**Suppressed Error**\n\n`unsupported-operation`: `+` is not supported between `Literal[2]` and `Literal['']`\n  Argument `Literal['']` is not assignable to parameter `value` with type `int` in function `int.__add__`",
            }
        })),
        error: None,
    });

    // Trailing suppression, suppressed error does not match
    interaction.server.hover("suppression.py", 10, 15);
    interaction.client.expect_response(Response {
        id: interaction.server.current_request_id(),
        result: Some(serde_json::json!({
            "contents": {
                "kind": "markdown",
                "value": "**No errors suppressed by this ignore**\n\n_The ignore comment may have an incorrect error code or there may be no errors on this line._",
            }
        })),
        error: None,
    });

    // Trailing suppression, next line has an unsuppressed error
    interaction.server.hover("suppression.py", 12, 15);
    interaction.client.expect_response(Response {
        id: interaction.server.current_request_id(),
        result: Some(serde_json::json!({
            "contents": {
                "kind": "markdown",
                "value": "**No errors suppressed by this ignore**\n\n_The ignore comment may have an incorrect error code or there may be no errors on this line._",
            }
        })),
        error: None,
    });

    // Standalone suppression, no errors
    interaction.server.hover("suppression.py", 15, 10);
    interaction.client.expect_response(Response {
        id: interaction.server.current_request_id(),
        result: Some(serde_json::json!({
            "contents": {
                "kind": "markdown",
                "value": "**No errors suppressed by this ignore**\n\n_The ignore comment may have an incorrect error code or there may be no errors on this line._",
            }
        })),
        error: None,
    });

    interaction.shutdown();
}
