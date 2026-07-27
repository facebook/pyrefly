/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::CodeActionKind;
use lsp_types::CodeActionOrCommand;
use lsp_types::Url;
use lsp_types::request::CodeActionRequest;
use pyrefly_lsp_test::object_model::InitializeSettings;
use pyrefly_lsp_test::object_model::LspInteraction;
use serde_json::json;

use crate::test::lsp::lsp_interaction::util::get_test_files_root;

const TEST_FILE: &str = "insert_type_annotation_code_action.py";
const ACTION_TITLE: &str = "Insert inferred type annotation";

fn request_has_type_annotation_edit(
    interaction: &LspInteraction,
    uri: &Url,
    request_line: u32,
    request_start: u32,
    request_end: u32,
    edit_line: u32,
    edit_character: u32,
    new_text: &'static str,
) {
    let uri = uri.clone();
    interaction
        .client
        .send_request::<CodeActionRequest>(json!({
            "textDocument": { "uri": uri },
            "range": {
                "start": { "line": request_line, "character": request_start },
                "end": { "line": request_line, "character": request_end }
            },
            "context": {
                "diagnostics": [],
                "only": ["quickfix"]
            }
        }))
        .expect_response_with(move |response: Option<Vec<CodeActionOrCommand>>| {
            response.is_some_and(|actions| {
                actions.iter().any(|action| {
                    let CodeActionOrCommand::CodeAction(action) = action else {
                        return false;
                    };
                    action.title == ACTION_TITLE
                        && action.kind == Some(CodeActionKind::QUICKFIX)
                        && action
                            .edit
                            .as_ref()
                            .and_then(|edit| edit.changes.as_ref())
                            .and_then(|changes| changes.get(&uri))
                            .is_some_and(|edits| {
                                edits.iter().any(|edit| {
                                    edit.range.start.line == edit_line
                                        && edit.range.start.character == edit_character
                                        && edit.range.end == edit.range.start
                                        && edit.new_text == new_text
                                })
                            })
                })
            })
        })
        .unwrap();
}

fn request_has_no_type_annotation_action(
    interaction: &LspInteraction,
    uri: &Url,
    line: u32,
    character: u32,
    only: &str,
) {
    interaction
        .client
        .send_request::<CodeActionRequest>(json!({
            "textDocument": { "uri": uri },
            "range": {
                "start": { "line": line, "character": character },
                "end": { "line": line, "character": character }
            },
            "context": {
                "diagnostics": [],
                "only": [only]
            }
        }))
        .expect_response_with(|response: Option<Vec<CodeActionOrCommand>>| {
            response.is_none_or(|actions| {
                actions.iter().all(|action| {
                    let CodeActionOrCommand::CodeAction(action) = action else {
                        return true;
                    };
                    action.title != ACTION_TITLE
                })
            })
        })
        .unwrap();
}

#[test]
fn test_insert_inferred_type_annotation_code_actions() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(json!([{
                "pyrefly": {"displayTypeErrors": "force-on"},
                "analysis": {
                    "inlayHints": {
                        "callArgumentNames": "all"
                    }
                }
            }]))),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open(TEST_FILE);
    let uri = Url::from_file_path(root.path().join(TEST_FILE)).unwrap();

    // Code actions are matched at the inlay hint's insertion position. A selection ending at
    // that position also matches because the check is inclusive.
    request_has_type_annotation_edit(&interaction, &uri, 6, 16, 16, 6, 16, " -> list[int]");
    request_has_type_annotation_edit(&interaction, &uri, 10, 0, 5, 10, 5, ": list[int]");
    request_has_type_annotation_edit(&interaction, &uri, 15, 18, 18, 15, 18, ": list[int]");

    // Unpacked annotations are invalid Python, call-argument hints are not types, and an
    // explicitly annotated variable has no inferred annotation to insert.
    request_has_no_type_annotation_action(&interaction, &uri, 18, 4, "quickfix");
    request_has_no_type_annotation_action(&interaction, &uri, 25, 8, "quickfix");
    request_has_no_type_annotation_action(&interaction, &uri, 28, 2, "quickfix");

    // The action must honor the client's requested code-action kind.
    request_has_no_type_annotation_action(&interaction, &uri, 10, 2, "refactor");

    interaction.shutdown().unwrap();
}

#[test]
fn test_insert_inferred_type_annotation_in_notebook_cell() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
            )),
            ..Default::default()
        })
        .unwrap();

    interaction.open_notebook(
        "notebook.ipynb",
        vec!["def make_items():\n    return [1, 2, 3]\n\nitems = make_items()"],
    );
    let cell_uri = interaction.cell_uri("notebook.ipynb", "cell1");

    interaction
        .inlay_hint_cell("notebook.ipynb", "cell1", 0, 0, 4, 0)
        .expect_response_with(|hints| {
            hints.is_some_and(|hints| {
                hints.iter().any(|hint| {
                    hint.position.line == 3
                        && hint.position.character == 5
                        && hint.text_edits.as_ref().is_some_and(|edits| {
                            edits.iter().any(|edit| edit.new_text == ": list[int]")
                        })
                })
            })
        })
        .unwrap();
    request_has_type_annotation_edit(&interaction, &cell_uri, 3, 5, 5, 3, 5, ": list[int]");

    interaction.shutdown().unwrap();
}

#[test]
fn test_insert_inferred_type_annotation_respects_inlay_hint_config() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(json!([{
                "pyrefly": {"displayTypeErrors": "force-on"},
                "analysis": {
                    "inlayHints": {
                        "functionReturnTypes": false,
                        "variableTypes": false
                    }
                }
            }]))),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open(TEST_FILE);
    let uri = Url::from_file_path(root.path().join(TEST_FILE)).unwrap();

    request_has_no_type_annotation_action(&interaction, &uri, 6, 16, "quickfix");
    request_has_no_type_annotation_action(&interaction, &uri, 10, 5, "quickfix");
    request_has_no_type_annotation_action(&interaction, &uri, 15, 18, "quickfix");

    interaction.shutdown().unwrap();
}
