/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cell::RefCell;

use lsp_types::CodeAction;
use lsp_types::CodeActionOrCommand;
use lsp_types::Url;
use lsp_types::request::CodeActionRequest;
use lsp_types::request::CodeActionResolveRequest;
use serde_json::json;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_code_action_resolve_introduce_parameter() {
    let root = get_test_files_root();
    let root_path = root.path().join("code_action_resolve");
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    let file = "introduce_parameter.py";
    let file_path = root_path.join(file);
    let uri = Url::from_file_path(&file_path).unwrap();
    interaction.client.did_open(file);

    let action_to_resolve: RefCell<Option<CodeAction>> = RefCell::new(None);
    interaction
        .client
        .send_request::<CodeActionRequest>(json!({
            "textDocument": { "uri": uri },
            "range": {
                "start": { "line": 1, "character": 11 },
                "end": { "line": 1, "character": 16 }
            },
            "context": {
                "diagnostics": [],
                "triggerKind": 1
            }
        }))
        .expect_response_with(|response| {
            let Some(actions) = response else {
                return false;
            };
            for action in actions {
                let CodeActionOrCommand::CodeAction(code_action) = action else {
                    continue;
                };
                if code_action.title == "Introduce parameter `param`" {
                    *action_to_resolve.borrow_mut() = Some(code_action);
                    return true;
                }
            }
            false
        })
        .unwrap();

    let action = action_to_resolve
        .into_inner()
        .expect("expected introduce_parameter code action");

    interaction
        .client
        .send_request::<CodeActionResolveRequest>(serde_json::to_value(action).unwrap())
        .expect_response_with(|resolved| {
            let Some(edit) = resolved.edit else {
                return false;
            };
            let Some(changes) = edit.changes.as_ref() else {
                return false;
            };
            let Some(edits) = changes.get(&uri) else {
                return false;
            };
            let has_signature_edit = edits.iter().any(|edit| edit.new_text == ", param");
            let has_replacement_edit = edits.iter().any(|edit| edit.new_text == "param");
            has_signature_edit && has_replacement_edit
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
