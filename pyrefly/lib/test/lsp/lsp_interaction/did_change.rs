/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::Url;
use lsp_types::notification::DidChangeTextDocument;
use pyrefly::commands::lsp::IndexingMode;
use serde_json::json;

use crate::object_model::InitializeSettings;
use crate::object_model::LspInteraction;
use crate::util::get_test_files_root;

#[test]
fn test_text_document_did_change() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    interaction.client.did_open("text_document.py");

    let filepath = root.path().join("text_document.py");
    interaction
        .client
        .send_notification::<DidChangeTextDocument>(json!({
            "textDocument": {
                "uri": Url::from_file_path(&filepath).unwrap().to_string(),
                "languageId": "python",
                "version": 2
            },
            "contentChanges": [{
                "range": {
                    "start": {"line": 6, "character": 0},
                    "end": {"line": 7, "character": 0}
                },
                "text": format!("{}\n", "rint(\"another change\")")
            }],
        }));

    interaction
        .client
        .send_notification::<DidChangeTextDocument>(json!({
            "textDocument": {
                "uri": Url::from_file_path(&filepath).unwrap().to_string(),
                "languageId": "python",
                "version": 3
            },
            "contentChanges": [{
                "range": {
                    "start": {"line": 6, "character": 0},
                    "end": {"line": 6, "character": 0}
                },
                "text": "p"
            }],
        }));

    interaction
        .client
        .diagnostic("text_document.py")
        .expect_response(json!({"items": [], "kind": "full"}))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_text_document_did_change_backwards_version() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    interaction.client.did_open("text_document.py");

    let filepath = root.path().join("text_document.py");

    // Send version 3 first (skipping version 2).
    interaction
        .client
        .send_notification::<DidChangeTextDocument>(json!({
            "textDocument": {
                "uri": Url::from_file_path(&filepath).unwrap().to_string(),
                "version": 3
            },
            "contentChanges": [{
                "range": {
                    "start": {"line": 6, "character": 0},
                    "end": {"line": 6, "character": 0}
                },
                "text": "x: int = 'not_an_int'\n"
            }],
        }));

    // Send version 2 (backwards!). The server should accept it.
    interaction
        .client
        .send_notification::<DidChangeTextDocument>(json!({
            "textDocument": {
                "uri": Url::from_file_path(&filepath).unwrap().to_string(),
                "version": 2
            },
            "contentChanges": [{
                "range": {
                    "start": {"line": 6, "character": 0},
                    "end": {"line": 7, "character": 0}
                },
                "text": "x: int = 42\n"
            }],
        }));

    // If the backwards version was rejected, we'd still see the type error
    // from version 3. Since it was accepted, the content should be valid.
    interaction
        .client
        .diagnostic("text_document.py")
        .expect_response(json!({"items": [], "kind": "full"}))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_text_document_did_change_unicode() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    interaction.client.did_open("utf.py");

    let utf_filepath = root.path().join("utf.py");
    interaction
        .client
        .send_notification::<DidChangeTextDocument>(json!({
            "textDocument": {
                "uri": Url::from_file_path(&utf_filepath).unwrap().to_string(),
                "languageId": "python",
                "version": 2
            },
            "contentChanges": [{
                "range": {
                    "start": { "line": 7, "character": 8 },
                    "end": { "line": 8, "character": 2 }
                },
                "rangeLength": 3,
                "text": ""
            }],
        }));

    interaction
        .client
        .send_notification::<DidChangeTextDocument>(json!({
            "textDocument": {
                "uri": Url::from_file_path(&utf_filepath).unwrap().to_string(),
                "languageId": "python",
                "version": 3
            },
            "contentChanges": [{
                "range": {
                    "start": { "line": 7, "character": 8 },
                    "end": { "line": 7, "character": 8 }
                },
                "rangeLength": 0,
                "text": format!("\n{}", "print(\"")
            }],
        }));

    interaction
        .client
        .diagnostic("utf.py")
        .expect_response(json!({"items": [], "kind": "full"}))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[cfg(unix)]
#[test]
fn test_text_document_did_change_updates_symlinked_imports() {
    use std::os::unix::fs::symlink;

    let root = tempfile::tempdir().unwrap();
    let module_path = root.path().join("test_module.py");
    let symlink_path = root.path().join("sym.py");
    let importer_path = root.path().join("test_import.py");
    std::fs::write(&module_path, "def hello(name: str) -> None:\n    pass\n").unwrap();
    symlink(&module_path, &symlink_path).unwrap();
    std::fs::write(&importer_path, "from sym import hello\nhello(\"John\")\n").unwrap();

    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
            )),
            workspace_folders: Some(vec![(
                "test".to_owned(),
                Url::from_file_path(root.path()).unwrap(),
            )]),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("test_module.py");
    interaction.client.did_open("test_import.py");
    interaction.client.did_change(
        "test_module.py",
        "def hello(name: str, times: int) -> None:\n    pass\n",
    );
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(importer_path, 1)
        .unwrap();

    interaction.shutdown().unwrap();
}
