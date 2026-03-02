/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::CodeActionOrCommand;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_notebook_code_action_import() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.open_notebook("notebook.ipynb", vec!["TypedDict", "NamedTuple"]);

    let cell1_uri = interaction.cell_uri("notebook.ipynb", "cell1");
    interaction
        .code_action_cell("notebook.ipynb", "cell1", 0, 0, 0, 9)
        .expect_response_with(|response| {
            let Some(actions) = response else {
                return false;
            };
            actions.iter().any(|action| {
                let CodeActionOrCommand::CodeAction(code_action) = action else {
                    return false;
                };
                let Some(text_edits) = code_action
                    .edit
                    .as_ref()
                    .and_then(|edit| edit.changes.as_ref())
                    .and_then(|changes| changes.get(&cell1_uri))
                else {
                    return false;
                };
                text_edits.iter().any(|text_edit| {
                    text_edit.range.start.line == 0
                        && text_edit.range.start.character == 0
                        && text_edit.new_text.as_str() == "from typing import TypedDict\n"
                })
            })
        })
        .unwrap();

    // Code actions for later cells will still insert imports to the first cell
    interaction
        .code_action_cell("notebook.ipynb", "cell2", 0, 0, 0, 10)
        .expect_response_with(|response| {
            let Some(actions) = response else {
                return false;
            };
            actions.iter().any(|action| {
                let CodeActionOrCommand::CodeAction(code_action) = action else {
                    return false;
                };
                let Some(text_edits) = code_action
                    .edit
                    .as_ref()
                    .and_then(|edit| edit.changes.as_ref())
                    .and_then(|changes| changes.get(&cell1_uri))
                else {
                    return false;
                };
                text_edits.iter().any(|text_edit| {
                    text_edit.range.start.line == 0
                        && text_edit.range.start.character == 0
                        && text_edit.new_text.as_str() == "from typing import NamedTuple\n"
                })
            })
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
