/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::Url;
use lsp_types::request::PrepareRenameRequest;
use lsp_types::request::Rename;
use serde_json::json;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_prepare_rename() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("basic"));
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    interaction.client.did_open("foo.py");

    let path = root.path().join("basic/foo.py");

    interaction
        .client
        .send_request::<PrepareRenameRequest>(json!({
            "textDocument": {
                "uri": Url::from_file_path(&path).unwrap().to_string()
            },
            "position": {
                "line": 6,
                "character": 16
            }
        }))
        .expect_response(json!({
            "start": {"line": 6, "character": 16},
            "end": {"line": 6, "character": 19},
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_rename_third_party_symbols_in_venv_is_not_allowed() {
    let root = get_test_files_root();
    let root_path = root.path().join("rename_third_party");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();

    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
            configuration: Some(Some(json!([{ "indexing_mode": "lazy_blocking" }]))),
            ..Default::default()
        })
        .unwrap();

    let user_code = root_path.join("user_code.py");

    interaction.client.did_open("user_code.py");

    // Verify that prepareRename returns null, indicating that renaming third party symbols is not allowed
    interaction
        .client
        .send_request::<PrepareRenameRequest>(json!({
            "textDocument": {
                "uri": Url::from_file_path(&user_code).unwrap().to_string()
            },
            "position": {
                "line": 14,  // Line with "external_result = external_function()"
                "character": 25  // Position on "external_function"
            }
        }))
        .expect_response(serde_json::Value::Null)
        .unwrap();

    // Verify that attempting to rename a third party symbol returns an error
    interaction
        .client
        .send_request::<Rename>(json!({
            "textDocument": {
                "uri": Url::from_file_path(&user_code).unwrap().to_string()
            },
            "position": {
                "line": 14,  // Line with "external_result = external_function()"
                "character": 25  // Position on "external_function"
            },
            "newName": "new_external_function"
        }))
        .expect_response_error(json!({
            "code": -32600,
            "message": "Third-party symbols cannot be renamed",
            "data": null,
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_rename_editable_package_symbols_is_allowed() {
    // This test verifies that symbols from editable packages CAN be renamed.
    // Editable packages (installed via `pip install -e .`) appear in both site-packages
    // AND the search_path, and should be treated like first-party code for renaming purposes.
    let root = get_test_files_root();
    let root_path = root.path().join("rename_editable_package");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();

    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
            configuration: Some(Some(json!([{ "indexing_mode": "lazy_blocking" }]))),
            ..Default::default()
        })
        .unwrap();

    let user_code = root_path.join("user_code.py");
    let editable_module = root_path.join("editable_module.py");

    interaction.client.did_open("user_code.py");
    interaction.client.did_open("editable_module.py");

    // Verify that prepareRename returns a range for editable package symbols
    interaction
        .client
        .send_request::<PrepareRenameRequest>(json!({
            "textDocument": {
                "uri": Url::from_file_path(&user_code).unwrap().to_string()
            },
            "position": {
                "line": 14,  // Line with "editable_result = editable_function()"
                "character": 25  // Position on "editable_function"
            }
        }))
        .expect_response(json!({
            "start": {"line": 14, "character": 22},
            "end": {"line": 14, "character": 39},
        }))
        .unwrap();

    // Verify that renaming an editable package symbol succeeds
    interaction
        .client
        .send_request::<Rename>(json!({
            "textDocument": {
                "uri": Url::from_file_path(&user_code).unwrap().to_string()
            },
            "position": {
                "line": 14,  // Line with "editable_result = editable_function()"
                "character": 25  // Position on "editable_function"
            },
            "newName": "new_editable_function"
        }))
        .expect_response(json!({
            "changes": {
                Url::from_file_path(&user_code).unwrap().to_string(): [
                    {
                        "newText": "new_editable_function",
                        "range": {"start": {"line": 5, "character": 28}, "end": {"line": 5, "character": 45}}
                    },
                    {
                        "newText": "new_editable_function",
                        "range": {"start": {"line": 14, "character": 22}, "end": {"line": 14, "character": 39}}
                    },
                ],
                Url::from_file_path(&editable_module).unwrap().to_string(): [
                    {
                        "newText": "new_editable_function",
                        "range": {"start": {"line": 6, "character": 4}, "end": {"line": 6, "character": 21}}
                    },
                ]
            }
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_rename() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();

    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
            configuration: Some(Some(json!([{ "indexing_mode": "lazy_blocking" }]))),
            ..Default::default()
        })
        .unwrap();

    let foo = root_path.join("foo.py");
    let bar = root_path.join("bar.py");
    let various_imports = root_path.join("various_imports.py");
    let with_synthetic_bindings = root_path.join("with_synthetic_bindings.py");

    interaction.client.did_open("bar.py");
    interaction.client.did_open("foo.py");
    interaction.client.did_open("various_imports.py");
    interaction.client.did_open("with_synthetic_bindings.py");

    interaction
        .client
        .send_request::<Rename>(json!({
            "textDocument": {
                "uri": Url::from_file_path(&bar).unwrap().to_string()
            },
            "position": {
                "line": 10,
                "character": 1
            },
            "newName": "Baz"
        }))
        .expect_response(json!({
            "changes": {
                Url::from_file_path(&foo).unwrap().to_string(): [
                    {
                        "newText":"Baz",
                        "range":{"start":{"line":6,"character":16},"end":{"line":6,"character":19}}
                    },
                    {
                        "newText":"Baz",
                        "range":{"start":{"line":8,"character":0},"end":{"line":8,"character":3}}
                    },
                    {
                        "newText":"Baz",
                        "range":{"start":{"line":9,"character":4},"end":{"line":9,"character":7}}
                    },
                ],
                Url::from_file_path(&various_imports).unwrap().to_string(): [
                    {
                        "newText":"Baz",
                        "range":{"start":{"line":5,"character":16},"end":{"line":5,"character":19}}
                    },
                    {
                        "newText":"Baz",
                        "range":{"start":{"line":5,"character":26},"end":{"line":5,"character":29}}
                    },
                ],
                Url::from_file_path(&with_synthetic_bindings).unwrap().to_string(): [
                    {
                        "newText":"Baz",
                        "range":{"start":{"line":5,"character":16},"end":{"character":19,"line":5}}
                    },
                    {
                        "newText":"Baz",
                        "range":{"start":{"line":10,"character":4},"end":{"character":7,"line":10}}
                    },
                ],
                Url::from_file_path(&bar).unwrap().to_string(): [
                    {
                        "newText":"Baz",
                        "range":{"start":{"line":6,"character":6},"end":{"character":9,"line":6}}
                    },
                    {
                        "newText":"Baz",
                        "range":{"start":{"line":10,"character":0},"end":{"character":3,"line":10}}
                    },
                ]
            }
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}
