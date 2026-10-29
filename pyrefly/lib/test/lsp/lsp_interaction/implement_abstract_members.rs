/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fs;

use lsp_types::CodeActionKind;
use lsp_types::CodeActionRequest;
use lsp_types::CodeActionResponse;
use lsp_types::Uri;
use pyrefly_lsp_test::object_model::InitializeSettings;
use pyrefly_lsp_test::object_model::LspInteraction;
use serde_json::json;

use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_implement_abstract_members_action_filter() {
    let root = get_test_files_root();
    let path = root.path().join("abstract_members.py");
    fs::write(&path, "from abc import ABC, abstractmethod\nclass Base(ABC):\n    @abstractmethod\n    def foo(self) -> int: ...\nclass Child(Base): pass\n").unwrap();
    let uri = Uri::from_file_path(&path).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();
    interaction.client.did_open("abstract_members.py");
    for (only, trigger_kind, expected) in [
        (json!(["quickfix"]), 1, true),
        (json!(["quickfix"]), 2, true),
        (json!(["refactor"]), 1, false),
        (json!(null), 1, true),
    ] {
        interaction
            .client
            .send_request::<CodeActionRequest>(json!({
                "textDocument": { "uri": uri },
                "range": {
                    "start": { "line": 4, "character": 6 },
                    "end": { "line": 4, "character": 11 }
                },
                "context": { "diagnostics": [], "only": only, "triggerKind": trigger_kind }
            }))
            .expect_response_with(|response| {
                let action = response.as_ref().and_then(|actions| {
                    actions.iter().find_map(|action| match action {
                        CodeActionResponse::CodeAction(action)
                            if action.title == "Implement abstract members" =>
                        {
                            Some(action)
                        }
                        _ => None,
                    })
                });
                assert_eq!(action.is_some(), expected);
                if let Some(action) = action {
                    assert_eq!(action.kind, Some(CodeActionKind::QuickFix));
                }
                true
            })
            .unwrap();
    }
    interaction.shutdown().unwrap();
}
