/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::Url;
use lsp_types::WorkspaceSymbolResponse;
use lsp_types::request::WorkspaceSymbolRequest;
use serde_json::json;

use crate::object_model::InitializeSettings;
use crate::object_model::LspInteraction;
use crate::util::get_test_files_root;

#[test]
fn test_workspace_symbol() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            configuration: Some(Some(json!([{ "indexing_mode": "lazy_blocking"}]))),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("autoimport_provider.py");

    interaction
        .client
        .send_request::<WorkspaceSymbolRequest>(
            json!({
                "query": "this_is_a_very_long_function_name_so_we_can"
            }),
        )
        .expect_response(json!([
            {
                "kind": 12,
                "location": {
                    "range": {
                        "start": {"line": 6, "character": 4},
                        "end": {"line": 6, "character": 99}
                    },
                    "uri": Url::from_file_path(root_path.join("autoimport_provider.py")).unwrap().to_string()
                },
                "name": "this_is_a_very_long_function_name_so_we_can_deterministically_test_autoimport_with_fuzzy_search"
            }
        ]))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_workspace_symbol_prefers_non_init_result() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            configuration: Some(Some(json!([{ "indexing_mode": "lazy_blocking"}]))),
            ..Default::default()
        })
        .unwrap();

    interaction
        .client
        .did_open("workspace_symbol_prefer_non_init/implementation.py");
    interaction
        .client
        .did_open("workspace_symbol_prefer_non_init/__init__.py");

    let implementation_uri =
        Url::from_file_path(root_path.join("workspace_symbol_prefer_non_init/implementation.py"))
            .unwrap();
    let init_uri =
        Url::from_file_path(root_path.join("workspace_symbol_prefer_non_init/__init__.py"))
            .unwrap();
    let symbol_name = "workspace_symbol_prefers_non_init_over_init_reexport";

    interaction
        .client
        .send_request::<WorkspaceSymbolRequest>(json!({ "query": symbol_name }))
        .expect_response_with(|result| {
            let Some(WorkspaceSymbolResponse::Flat(symbols)) = result else {
                panic!("Unexpected workspace symbol response: {result:?}");
            };
            assert!(symbols.iter().all(|symbol| symbol.name == symbol_name));
            assert!(symbols.iter().all(|symbol| {
                symbol.location.uri == implementation_uri || symbol.location.uri == init_uri
            }));

            let first_init_index = symbols
                .iter()
                .position(|symbol| symbol.location.uri == init_uri)
                .expect("expected at least one __init__.py result");
            let last_non_init_index = symbols
                .iter()
                .rposition(|symbol| symbol.location.uri == implementation_uri)
                .expect("expected at least one non-__init__.py result");

            assert!(last_non_init_index < first_init_index);
            true
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
