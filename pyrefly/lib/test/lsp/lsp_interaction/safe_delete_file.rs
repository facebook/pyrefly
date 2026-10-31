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
use lsp_types::DocumentChange;
use lsp_types::Location;
use lsp_types::Uri;
use pyrefly_lsp_test::IndexingMode;
use pyrefly_lsp_test::object_model::InitializeSettings;
use pyrefly_lsp_test::object_model::LspInteraction;
use pyrefly_lsp_test::object_model::LspInteractionArgs;
use serde_json::json;

use crate::test::lsp::lsp_interaction::util::get_test_files_root;

fn init_with_delete_support(
    root_path: &std::path::Path,
    supports_find_usages: bool,
    indexing_mode: IndexingMode,
) -> (LspInteraction, Uri) {
    let scope_uri = Uri::from_file_path(root_path).unwrap();
    let mut args = LspInteractionArgs::default();
    args.args.indexing_mode = indexing_mode;
    let mut interaction = LspInteraction::new_with_args(args);
    interaction.set_root(root_path.to_path_buf());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
            capabilities: Some(json!({
                "workspace": {
                    "workspaceEdit": {
                        "documentChanges": true,
                        "resourceOperations": ["delete"]
                    }
                }
            })),
            initialization_options: supports_find_usages
                .then(|| json!({"pyrefly": {"findFileUsages": true}})),
            ..Default::default()
        })
        .unwrap();
    (interaction, scope_uri)
}

fn find_usage_locations(
    actions: &[CodeActionResponse],
    file_name: &str,
    target_uri: &Uri,
) -> Option<Vec<Location>> {
    let title = format!("Find usages of file `{file_name}`");
    let code_action = actions.iter().find_map(|action| {
        let CodeActionResponse::CodeAction(code_action) = action else {
            return None;
        };
        (code_action.title == title).then_some(code_action)
    })?;
    if code_action.kind.as_ref()? != &CodeActionKind::new("refactor.delete") {
        return None;
    }
    let command = code_action.command.as_ref()?;
    if command.command != "pyrefly.findFileUsages" {
        return None;
    }
    let payload = command.arguments.as_ref()?.first()?.as_object()?;
    if payload.get("uri")? != &json!(target_uri) {
        return None;
    }
    serde_json::from_value(payload.get("locations")?.clone()).ok()
}

#[test]
fn test_safe_delete_file_unused() {
    let root = get_test_files_root();
    let root_path = root.path().join("safe_delete_file");
    let (interaction, _scope_uri) = init_with_delete_support(&root_path, true, IndexingMode::None);

    let file = "unused.py";
    let file_path = root_path.join(file);
    let uri = Uri::from_file_path(&file_path).unwrap();
    interaction.client.did_open(file);

    interaction
        .client
        .send_request::<CodeActionRequest>(json!({
            "textDocument": { "uri": uri },
            "range": {
                "start": { "line": 0, "character": 0 },
                "end": { "line": 0, "character": 0 }
            },
            "context": { "diagnostics": [] }
        }))
        .expect_response_with(|response: Option<Vec<CodeActionResponse>>| {
            let Some(actions) = response else {
                return false;
            };
            actions.iter().any(|action| {
                let CodeActionResponse::CodeAction(code_action) = action else {
                    return false;
                };
                if code_action.title != "Safe delete file `unused.py`" {
                    return false;
                }
                let Some(edit) = &code_action.edit else {
                    return false;
                };
                let Some(ops) = &edit.document_changes else {
                    return false;
                };
                if ops.len() != 1 {
                    return false;
                }
                match &ops[0] {
                    DocumentChange::DeleteFile(delete) => delete.uri == uri,
                    _ => false,
                }
            })
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_safe_delete_file_finds_usage_in_indexed_closed_file() {
    let root = get_test_files_root();
    let root_path = root.path().join("safe_delete_file");
    let (interaction, _scope_uri) =
        init_with_delete_support(&root_path, true, IndexingMode::LazyBlocking);

    let file = "target.py";
    let file_path = root_path.join(file);
    let uri = Uri::from_file_path(&file_path).unwrap();
    let consumer_uri = Uri::from_file_path(root_path.join("consumer.py")).unwrap();

    interaction.client.did_open(file);

    interaction
        .client
        .send_request::<CodeActionRequest>(json!({
            "textDocument": { "uri": uri },
            "range": {
                "start": { "line": 0, "character": 0 },
                "end": { "line": 0, "character": 0 }
            },
            "context": { "diagnostics": [] }
        }))
        .expect_response_with(|response: Option<Vec<CodeActionResponse>>| {
            let Some(actions) = response else {
                return false;
            };
            let locations = find_usage_locations(&actions, file, &uri).unwrap_or_default();
            let has_consumer = locations
                .iter()
                .any(|location| location.uri == consumer_uri);
            let has_delete = actions.iter().any(|action| {
                matches!(
                    action,
                    CodeActionResponse::CodeAction(action)
                        if action.title == "Delete file `target.py` anyway"
                )
            });
            let has_safe = actions.iter().any(|action| {
                matches!(
                    action,
                    CodeActionResponse::CodeAction(action)
                        if action.title == "Safe delete file `target.py`"
                )
            });
            has_consumer && has_delete && !has_safe
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_safe_delete_file_prefers_open_importer() {
    let root = get_test_files_root();
    let root_path = root.path().join("safe_delete_file");
    let (interaction, _scope_uri) =
        init_with_delete_support(&root_path, true, IndexingMode::LazyBlocking);

    let file = "target.py";
    let uri = Uri::from_file_path(root_path.join(file)).unwrap();
    let consumer_uri = Uri::from_file_path(root_path.join("consumer.py")).unwrap();
    interaction.client.did_open(file);
    // Complete indexing before opening a buffer for the indexed importer.
    interaction
        .client
        .send_request::<CodeActionRequest>(json!({
            "textDocument": { "uri": uri },
            "range": {
                "start": { "line": 0, "character": 0 },
                "end": { "line": 0, "character": 0 }
            },
            "context": { "diagnostics": [] }
        }))
        .expect_response_with(|response: Option<Vec<CodeActionResponse>>| {
            let actions = response.expect("expected file actions after indexing");
            let locations = find_usage_locations(&actions, file, &uri)
                .expect("expected usages from the indexed importer");
            assert_eq!(locations.len(), 1);
            assert_eq!(locations[0].uri, consumer_uri);
            true
        })
        .unwrap();
    interaction.client.did_open("consumer.py");
    interaction
        .client
        .did_change("consumer.py", "\n\nimport target\n");

    interaction
        .client
        .send_request::<CodeActionRequest>(json!({
            "textDocument": { "uri": uri },
            "range": {
                "start": { "line": 0, "character": 0 },
                "end": { "line": 0, "character": 0 }
            },
            "context": { "diagnostics": [] }
        }))
        .expect_response_with(|response: Option<Vec<CodeActionResponse>>| {
            let Some(actions) = response else {
                return false;
            };
            let locations = find_usage_locations(&actions, file, &uri)
                .expect("expected usages from the open importer");
            assert_eq!(locations.len(), 1);
            assert_eq!(locations[0].uri, consumer_uri);
            assert_eq!(locations[0].range.start.line, 2);
            true
        })
        .unwrap();

    interaction
        .client
        .did_change("consumer.py", "answer = 123\n");
    interaction
        .client
        .send_request::<CodeActionRequest>(json!({
            "textDocument": { "uri": uri },
            "range": {
                "start": { "line": 0, "character": 0 },
                "end": { "line": 0, "character": 0 }
            },
            "context": { "diagnostics": [] }
        }))
        .expect_response_with(|response: Option<Vec<CodeActionResponse>>| {
            let actions =
                response.expect("expected a delete-anyway action for indexed dependencies");
            assert!(find_usage_locations(&actions, file, &uri).is_none());
            assert!(actions.iter().any(|action| {
                matches!(action, CodeActionResponse::CodeAction(action)
                    if action.title == "Delete file `target.py` anyway")
            }));
            assert!(actions.iter().all(|action| {
                !matches!(action, CodeActionResponse::CodeAction(action)
                    if action.title == "Safe delete file `target.py`")
            }));
            true
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_safe_delete_file_finds_package_reexports() {
    for chained in [false, true] {
        let root = get_test_files_root();
        let root_path = root.path().join("safe_delete_file_relative");
        fs::write(root_path.join("pkg/__init__.py"), "from . import target\n").unwrap();
        let consumer_source = if chained {
            fs::write(
                root_path.join("bridge.py"),
                "from pkg import target as exported\n",
            )
            .unwrap();
            "from bridge import exported as target\ntarget.func()\n"
        } else {
            "from pkg import target\ntarget.func()\n"
        };
        fs::write(root_path.join("consumer.py"), consumer_source).unwrap();
        let (interaction, _scope_uri) =
            init_with_delete_support(&root_path, true, IndexingMode::LazyBlocking);
        let uri = Uri::from_file_path(root_path.join("pkg/target.py")).unwrap();
        let package_uri = Uri::from_file_path(root_path.join("pkg/__init__.py")).unwrap();
        let consumer_uri = Uri::from_file_path(root_path.join("consumer.py")).unwrap();
        interaction.client.did_open("pkg/target.py");
        interaction
            .client
            .send_request::<CodeActionRequest>(json!({
                "textDocument": { "uri": uri },
                "range": {
                    "start": { "line": 0, "character": 0 },
                    "end": { "line": 0, "character": 0 }
                },
                "context": { "diagnostics": [] }
            }))
            .expect_response_with(|response: Option<Vec<CodeActionResponse>>| {
                let actions = response.expect("expected actions for a re-exported submodule");
                let locations = find_usage_locations(&actions, "target.py", &uri)
                    .expect("expected references through package re-exports");
                assert!(locations.iter().any(|location| location.uri == package_uri));
                assert!(
                    locations
                        .iter()
                        .any(|location| location.uri == consumer_uri)
                );
                assert!(actions.iter().any(|action| {
                    matches!(action, CodeActionResponse::CodeAction(action)
                        if action.title == "Delete file `target.py` anyway")
                }));
                true
            })
            .unwrap();
        interaction.shutdown().unwrap();
    }
}

#[test]
fn test_safe_delete_file_negotiates_find_usages_command() {
    let root = get_test_files_root();
    let root_path = root.path().join("safe_delete_file");
    let (interaction, _scope_uri) = init_with_delete_support(&root_path, false, IndexingMode::None);

    let file = "target.py";
    let uri = Uri::from_file_path(root_path.join(file)).unwrap();
    interaction.client.did_open(file);
    interaction.client.did_open("consumer.py");

    interaction
        .client
        .send_request::<CodeActionRequest>(json!({
            "textDocument": { "uri": uri },
            "range": {
                "start": { "line": 0, "character": 0 },
                "end": { "line": 0, "character": 0 }
            },
            "context": { "diagnostics": [] }
        }))
        .expect_response_with(|response: Option<Vec<CodeActionResponse>>| {
            let Some(actions) = response else {
                return false;
            };
            let titles = actions.iter().filter_map(|action| match action {
                CodeActionResponse::CodeAction(action) => Some(action.title.as_str()),
                CodeActionResponse::Command(_) => None,
            });
            let titles = titles.collect::<Vec<_>>();
            titles.contains(&"Delete file `target.py` anyway")
                && !titles.contains(&"Find usages of file `target.py`")
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_safe_delete_file_maps_notebook_usage_to_cell() {
    let root = get_test_files_root();
    let root_path = root.path().join("safe_delete_file");
    let (interaction, _scope_uri) = init_with_delete_support(&root_path, true, IndexingMode::None);

    let file = "target.py";
    let uri = Uri::from_file_path(root_path.join(file)).unwrap();
    interaction.client.did_open(file);
    interaction.open_notebook("consumer.ipynb", vec!["import target"]);
    let cell_uri = interaction.cell_uri("consumer.ipynb", "cell1");

    interaction
        .client
        .send_request::<CodeActionRequest>(json!({
            "textDocument": { "uri": uri },
            "range": {
                "start": { "line": 0, "character": 0 },
                "end": { "line": 0, "character": 0 }
            },
            "context": { "diagnostics": [] }
        }))
        .expect_response_with(|response: Option<Vec<CodeActionResponse>>| {
            let Some(actions) = response else {
                return false;
            };
            find_usage_locations(&actions, file, &uri)
                .is_some_and(|locations| locations.iter().any(|location| location.uri == cell_uri))
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_safe_delete_file_does_not_report_attribute_import_as_submodule() {
    let root = get_test_files_root();
    let root_path = root.path().join("safe_delete_file_attribute_import");
    let (interaction, _scope_uri) = init_with_delete_support(&root_path, true, IndexingMode::None);

    let file = "package/target.py";
    let uri = Uri::from_file_path(root_path.join(file)).unwrap();
    let package_uri = Uri::from_file_path(root_path.join("package/__init__.py")).unwrap();
    let consumer_uri = Uri::from_file_path(root_path.join("consumer.py")).unwrap();
    interaction.client.did_open(file);
    interaction.client.did_open("package/__init__.py");
    interaction.client.did_open("consumer.py");

    interaction
        .client
        .send_request::<CodeActionRequest>(json!({
            "textDocument": { "uri": uri },
            "range": {
                "start": { "line": 0, "character": 0 },
                "end": { "line": 0, "character": 0 }
            },
            "context": { "diagnostics": [] }
        }))
        .expect_response_with(|response: Option<Vec<CodeActionResponse>>| {
            let Some(actions) = response else {
                return false;
            };
            find_usage_locations(&actions, "target.py", &uri).is_some_and(|locations| {
                locations.iter().any(|location| location.uri == package_uri)
                    && locations
                        .iter()
                        .all(|location| location.uri != consumer_uri)
            })
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_safe_delete_file_rejects_from_import() {
    let root = get_test_files_root();
    let root_path = root.path().join("safe_delete_file_from_import");
    let (interaction, _scope_uri) = init_with_delete_support(&root_path, true, IndexingMode::None);

    let file = "target.py";
    let file_path = root_path.join(file);
    let uri = Uri::from_file_path(&file_path).unwrap();

    interaction.client.did_open(file);
    interaction.client.did_open("consumer.py");

    interaction
        .client
        .send_request::<CodeActionRequest>(json!({
            "textDocument": { "uri": uri },
            "range": {
                "start": { "line": 0, "character": 0 },
                "end": { "line": 0, "character": 0 }
            },
            "context": { "diagnostics": [] }
        }))
        .expect_response_with(|response| {
            let Some(actions) = response else {
                return false;
            };
            let mut saw_find = false;
            let mut saw_delete = false;
            let mut saw_safe = false;
            for action in actions {
                let CodeActionResponse::CodeAction(code_action) = action else {
                    continue;
                };
                match code_action.title.as_str() {
                    "Find usages of file `target.py`" => saw_find = true,
                    "Delete file `target.py` anyway" => saw_delete = true,
                    "Safe delete file `target.py`" => saw_safe = true,
                    _ => {}
                }
            }
            saw_find && saw_delete && !saw_safe
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_safe_delete_file_from_import_unused() {
    let root = get_test_files_root();
    let root_path = root.path().join("safe_delete_file_from_import");
    let (interaction, _scope_uri) = init_with_delete_support(&root_path, true, IndexingMode::None);

    let file = "unused.py";
    let file_path = root_path.join(file);
    let uri = Uri::from_file_path(&file_path).unwrap();

    interaction.client.did_open(file);

    interaction
        .client
        .send_request::<CodeActionRequest>(json!({
            "textDocument": { "uri": uri },
            "range": {
                "start": { "line": 0, "character": 0 },
                "end": { "line": 0, "character": 0 }
            },
            "context": { "diagnostics": [] }
        }))
        .expect_response_with(|response| {
            let Some(actions) = response else {
                return false;
            };
            actions.iter().any(|action| {
                let CodeActionResponse::CodeAction(code_action) = action else {
                    return false;
                };
                if code_action.title != "Safe delete file `unused.py`" {
                    return false;
                }
                let Some(edit) = &code_action.edit else {
                    return false;
                };
                let Some(ops) = &edit.document_changes else {
                    return false;
                };
                if ops.len() != 1 {
                    return false;
                }
                match &ops[0] {
                    DocumentChange::DeleteFile(delete) => delete.uri == uri,
                    _ => false,
                }
            })
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_safe_delete_file_rejects_relative_import() {
    let root = get_test_files_root();
    let root_path = root.path().join("safe_delete_file_relative");
    let (interaction, _scope_uri) = init_with_delete_support(&root_path, true, IndexingMode::None);

    let file = "pkg/target.py";
    let file_path = root_path.join(file);
    let uri = Uri::from_file_path(&file_path).unwrap();

    interaction.client.did_open(file);
    interaction.client.did_open("pkg/consumer.py");

    interaction
        .client
        .send_request::<CodeActionRequest>(json!({
            "textDocument": { "uri": uri },
            "range": {
                "start": { "line": 0, "character": 0 },
                "end": { "line": 0, "character": 0 }
            },
            "context": { "diagnostics": [] }
        }))
        .expect_response_with(|response| {
            let Some(actions) = response else {
                return false;
            };
            let mut saw_find = false;
            let mut saw_delete = false;
            let mut saw_safe = false;
            for action in actions {
                let CodeActionResponse::CodeAction(code_action) = action else {
                    continue;
                };
                match code_action.title.as_str() {
                    "Find usages of file `target.py`" => saw_find = true,
                    "Delete file `target.py` anyway" => saw_delete = true,
                    "Safe delete file `target.py`" => saw_safe = true,
                    _ => {}
                }
            }
            saw_find && saw_delete && !saw_safe
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

/// A Configerator `.cinc` file reaches its Thrift stub through an
/// `import_thrift(...)` call rather than an `import` statement. The dependency
/// is real -- deleting the stub breaks the config -- and the dependency graph
/// records it, so deletion must carry the "anyway" warning.
#[test]
fn test_safe_delete_file_rejects_special_import() {
    let root = get_test_files_root();
    let root_path = root.path().join("safe_delete_special_import");
    let (interaction, _scope_uri) = init_with_delete_support(&root_path, true, IndexingMode::None);

    interaction.client.did_open("config.cinc");

    let uri = Uri::from_file_path(root_path.join("service/types.thrift.pyi")).unwrap();
    interaction
        .client
        .send_request::<CodeActionRequest>(json!({
            "textDocument": { "uri": uri },
            "range": {
                "start": { "line": 0, "character": 0 },
                "end": { "line": 0, "character": 0 }
            },
            "context": { "diagnostics": [] }
        }))
        .expect_response_with(|response: Option<Vec<CodeActionResponse>>| {
            let actions = response.expect("expected a delete-anyway action for a special import");
            assert!(actions.iter().any(|action| {
                matches!(action, CodeActionResponse::CodeAction(action)
                    if action.title == "Delete file `types.thrift.pyi` anyway")
            }));
            actions.iter().all(|action| {
                let CodeActionResponse::CodeAction(code_action) = action else {
                    return true;
                };
                !code_action.title.starts_with("Safe delete file")
            })
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
