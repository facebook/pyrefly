/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fs;

use lsp_types::CodeActionResponse;
use lsp_types::Position;
use lsp_types::Range;
use lsp_types::Uri;
use pyrefly_lsp_test::object_model::InitializeSettings;
use pyrefly_lsp_test::object_model::LspInteraction;
use tempfile::TempDir;

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
                let CodeActionResponse::CodeAction(code_action) = action else {
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

    // Code actions for later cells insert imports into the current cell, not the first cell
    let cell2_uri = interaction.cell_uri("notebook.ipynb", "cell2");
    interaction
        .code_action_cell("notebook.ipynb", "cell2", 0, 0, 0, 10)
        .expect_response_with(|response| {
            let Some(actions) = response else {
                return false;
            };
            actions.iter().any(|action| {
                let CodeActionResponse::CodeAction(code_action) = action else {
                    return false;
                };
                let Some(text_edits) = code_action
                    .edit
                    .as_ref()
                    .and_then(|edit| edit.changes.as_ref())
                    .and_then(|changes| changes.get(&cell2_uri))
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

#[test]
fn test_notebook_code_action_implicit_import_cross_cell() {
    let root = TempDir::new().unwrap();
    fs::create_dir(root.path().join("foo")).unwrap();
    fs::write(root.path().join("foo/__init__.py"), "").unwrap();
    fs::write(root.path().join("foo/bar.py"), "x = 1\n").unwrap();
    fs::write(
        root.path().join("pyrefly.toml"),
        "[errors]\nimplicit-import = \"error\"\n",
    )
    .unwrap();
    let root_uri = Uri::from_file_path(root.path()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), root_uri)]),
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();
    interaction.open_notebook("notebook.ipynb", vec!["import foo", "x = foo.bar.x"]);

    // The replacement targets the import's cell, not the cell requesting the fix.
    let target_uri = interaction.cell_uri("notebook.ipynb", "cell1");
    interaction
        .code_action_cell("notebook.ipynb", "cell2", 0, 4, 0, 4)
        .expect_response_with(|response| {
            let Some(actions) = response else {
                return false;
            };
            actions.iter().any(|action| {
                let CodeActionResponse::CodeAction(action) = action else {
                    return false;
                };
                if action.title != "Insert import: `import foo.bar`" {
                    return false;
                }
                let Some(changes) = action.edit.as_ref().and_then(|edit| edit.changes.as_ref())
                else {
                    return false;
                };
                let Some(edits) = changes.get(&target_uri) else {
                    return false;
                };
                changes.len() == 1
                    && edits.len() == 1
                    && edits[0].range == Range::new(Position::new(0, 7), Position::new(0, 10))
                    && edits[0].new_text == "foo.bar"
            })
        })
        .unwrap();
    interaction.shutdown().unwrap();
}
