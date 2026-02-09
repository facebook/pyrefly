/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_server::RequestId;
use lsp_types::SymbolKind;
use lsp_types::Url;
use lsp_types::request::Request as _;
use lsp_types::request::TypeHierarchyPrepare;
use lsp_types::request::TypeHierarchySubtypes;
use lsp_types::request::TypeHierarchySupertypes;
use serde_json::json;

use crate::lsp::non_wasm::protocol::Message;
use crate::lsp::non_wasm::protocol::Request;
use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_type_hierarchy_basic() {
    let root = get_test_files_root();
    let root_path = root.path().join("type_hierarchy_test");
    let scope_uri = Url::from_file_path(&root_path).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("classes.py");
    let uri = Url::from_file_path(root_path.join("classes.py")).unwrap();

    interaction.client.send_message(Message::Request(Request {
        id: RequestId::from(1),
        method: TypeHierarchyPrepare::METHOD.to_owned(),
        params: json!({
            "textDocument": {
                "uri": uri.to_string()
            },
            "position": {
                "line": 10,
                "character": 6
            }
        }),
        activity_key: None,
    }));

    interaction
        .client
        .expect_response_with::<TypeHierarchyPrepare>(RequestId::from(1), |result| {
            let Some(items) = result else {
                return false;
            };
            items.len() == 1 && items[0].name == "B"
        })
        .unwrap();

    let class_b_item = json!({
        "name": "B",
        "kind": SymbolKind::CLASS,
        "uri": uri.to_string(),
        "range": {
            "start": {"line": 10, "character": 0},
            "end": {"line": 11, "character": 8}
        },
        "selectionRange": {
            "start": {"line": 10, "character": 6},
            "end": {"line": 10, "character": 7}
        }
    });

    interaction
        .client
        .send_request::<TypeHierarchySupertypes>(json!({
            "item": class_b_item.clone()
        }))
        .expect_response_with(|result| {
            let Some(items) = result else {
                return false;
            };
            items.iter().any(|item| item.name == "A")
        })
        .unwrap();

    interaction
        .client
        .send_request::<TypeHierarchySubtypes>(json!({
            "item": class_b_item
        }))
        .expect_response_with(|result| {
            let Some(items) = result else {
                return false;
            };
            items.iter().any(|item| item.name == "C")
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
