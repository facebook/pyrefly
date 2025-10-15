/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_server::Message;
use lsp_server::Notification;
use lsp_server::Request;
use lsp_server::RequestId;
use lsp_server::Response;
use lsp_types::CompletionItem;
use lsp_types::CompletionItemKind;
use lsp_types::ConfigurationItem;
use lsp_types::ConfigurationParams;
use lsp_types::Url;
use lsp_types::notification::DidChangeConfiguration;
use lsp_types::notification::Notification as _;
use lsp_types::request::Request as _;
use lsp_types::request::WorkspaceConfiguration;
use pyrefly_python::keywords::get_keywords;

use crate::commands::lsp::IndexingMode;
use crate::config::environment::environment::PythonEnvironment;
use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::TestCase;
use crate::test::lsp::lsp_interaction::util::build_did_open_notification;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;
#[allow(deprecated)]
use crate::test::lsp::lsp_interaction::util::run_test_lsp;

fn get_all_builtin_completions() -> Vec<CompletionItem> {
    get_keywords(
        PythonEnvironment::get_default_interpreter_env()
            .python_version
            .unwrap(),
    )
    .into_iter()
    .map(|kw| CompletionItem {
        label: (*kw).to_owned(),
        kind: Some(CompletionItemKind::KEYWORD),
        sort_text: Some("0".to_owned()),
        ..Default::default()
    })
    .collect()
}

/// Creates a completion response message sorting the completion_items.
/// completion_items is a Vec of CompletionItem to include in the response
pub fn make_sorted_completion_result_with_all_keywords(
    request_id: i32,
    completion_items: Vec<CompletionItem>,
) -> Message {
    let mut all_items = get_all_builtin_completions();
    all_items.extend(completion_items);
    all_items.sort_by(|item1, item2| {
        item1
            .sort_text
            .cmp(&item2.sort_text)
            .then_with(|| item1.label.cmp(&item2.label))
            .then_with(|| item1.detail.cmp(&item2.detail))
    });

    let items_json: Vec<serde_json::Value> = all_items
        .into_iter()
        .map(|item| serde_json::to_value(item).unwrap())
        .collect();

    Message::Response(Response {
        id: RequestId::from(request_id),
        result: Some(serde_json::json!({
            "isIncomplete": false,
            "items": items_json,
        })),
        error: None,
    })
}

#[test]
fn test_completion_basic() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings::default());

    interaction.server.did_open("foo.py");

    let foo_path = root.path().join("foo.py");
    interaction
        .server
        .send_message(Message::Notification(Notification {
            method: "textDocument/didChange".to_owned(),
            params: serde_json::json!({
                "textDocument": {
                    "uri": Url::from_file_path(&foo_path).unwrap().to_string(),
                    "languageId": "python",
                    "version": 2
                },
                "contentChanges": [{
                    "range": {
                        "start": {"line": 10, "character": 0},
                        "end": {"line": 12, "character": 0}
                    },
                    "text": format!("\n{}\n", "Ba")
                }],
            }),
        }));

    interaction.server.completion("foo.py", 11, 1);

    interaction.client.expect_response_with(
        |response| {
            if response.id != RequestId::from(2) {
                return false;
            }
            if let Some(result) = &response.result
                && let Some(items) = result.get("items")
                && let Some(items_array) = items.as_array()
            {
                return items_array.iter().any(|item| {
                    if let Some(label) = item.get("label")
                        && let Some(label_str) = label.as_str()
                    {
                        label_str == "Bar"
                    } else {
                        false
                    }
                });
            }
            false
        },
        "Expected completion response with 'Bar' in items",
    );

    interaction.shutdown();
}

#[test]
fn test_completion_sorting() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings::default());

    interaction.server.did_open("foo.py");

    let foo_path = root.path().join("foo.py");
    interaction
        .server
        .send_message(Message::Notification(Notification {
            method: "textDocument/didChange".to_owned(),
            params: serde_json::json!({
                "textDocument": {
                    "uri": Url::from_file_path(&foo_path).unwrap().to_string(),
                    "languageId": "python",
                    "version": 2
                },
                "contentChanges": [{
                    "range": {
                        "start": {"line": 10, "character": 0},
                        "end": {"line": 12, "character": 0}
                    },
                    "text": format!("\n{}\n", "Ba")
                }],
            }),
        }));

    interaction.server.completion("foo.py", 11, 1);

    interaction.client.expect_response_with(
        |response| {
            if response.id != RequestId::from(2) {
                return false;
            }
            if let Some(result) = &response.result
                && let Some(items) = result.get("items")
                && let Some(items_array) = items.as_array()
            {
                let mut prev_sort_text: Option<String> = None;
                let mut prev_label: Option<String> = None;

                for item in items_array {
                    let sort_text = item.get("sortText").and_then(|v| v.as_str()).unwrap_or("");
                    let label = item.get("label").and_then(|v| v.as_str()).unwrap_or("");

                    if let Some(prev_st) = &prev_sort_text {
                        if sort_text < prev_st.as_str() {
                            return false;
                        }
                        if sort_text == prev_st.as_str()
                            && let Some(prev_l) = &prev_label
                            && label < prev_l.as_str()
                        {
                            return false;
                        }
                    }

                    prev_sort_text = Some(sort_text.to_owned());
                    prev_label = Some(label.to_owned());
                }

                return true;
            }
            false
        },
        "Expected completion items to be sorted by sortText then label",
    );

    interaction.shutdown();
}

#[test]
fn test_completion_keywords() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings::default());

    interaction.server.did_open("foo.py");

    let foo_path = root.path().join("foo.py");
    interaction
        .server
        .send_message(Message::Notification(Notification {
            method: "textDocument/didChange".to_owned(),
            params: serde_json::json!({
                "textDocument": {
                    "uri": Url::from_file_path(&foo_path).unwrap().to_string(),
                    "languageId": "python",
                    "version": 2
                },
                "contentChanges": [{
                    "range": {
                        "start": {"line": 10, "character": 0},
                        "end": {"line": 12, "character": 0}
                    },
                    "text": format!("\n{}\n", "i")
                }],
            }),
        }));

    interaction.server.completion("foo.py", 11, 1);

    interaction.client.expect_response_with(
        |response| {
            if response.id != RequestId::from(2) {
                return false;
            }
            if let Some(result) = &response.result
                && let Some(items) = result.get("items")
                && let Some(items_array) = items.as_array()
            {
                // Verify that common Python keywords are present
                let has_if = items_array.iter().any(|item| {
                    item.get("label").and_then(|v| v.as_str()) == Some("if")
                        && item.get("kind").and_then(|v| v.as_u64()) == Some(14) // KEYWORD kind
                });
                let has_import = items_array.iter().any(|item| {
                    item.get("label").and_then(|v| v.as_str()) == Some("import")
                        && item.get("kind").and_then(|v| v.as_u64()) == Some(14)
                });
                let has_def = items_array.iter().any(|item| {
                    item.get("label").and_then(|v| v.as_str()) == Some("def")
                        && item.get("kind").and_then(|v| v.as_u64()) == Some(14)
                });

                return has_if && has_import && has_def;
            }
            false
        },
        "Expected completion response to include Python keywords like 'if', 'import', 'def'",
    );

    interaction.shutdown();
}

#[test]
#[allow(deprecated)]
fn test_completion_with_autoimport() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();

    run_test_lsp(TestCase {
        messages_from_language_client: vec![
            Message::Response(Response {
                id: RequestId::from(1),
                result: Some(serde_json::json!([])),
                error: None,
            }),
            Message::from(build_did_open_notification(root_path.join("foo.py"))),
            Message::from(Notification {
                method: "textDocument/didChange".to_owned(),
                params: serde_json::json!({
                    "textDocument": {
                        "uri": Url::from_file_path(root_path.join("foo.py")).unwrap().to_string(),
                        "languageId": "python",
                        "version": 2
                    },
                    "contentChanges": [{
                        "text": "this_is_a_very_long_function_name_so_we_can".to_owned()
                    }],
                }),
            }),
            Message::from(Request {
                id: RequestId::from(2),
                method: "textDocument/completion".to_owned(),
                params: serde_json::json!({
                    "textDocument": {
                        "uri": Url::from_file_path(root_path.join("foo.py")).unwrap().to_string()
                    },
                    "position": {
                        "line": 0,
                        "character": 43
                    }
                }),
            }),
            Message::Notification(Notification {
                method: DidChangeConfiguration::METHOD.to_owned(),
                params: serde_json::json!([{"settings": {}}
                ]),
            }),
            Message::Response(Response {
                id: RequestId::from(2),
                result: Some(serde_json::json!([
                    {
                        "analysis": {
                            "importFormat": "relative",
                        }
                    },
                ])),
                error: None,
            }),
            Message::from(Request {
                id: RequestId::from(4),
                method: "textDocument/completion".to_owned(),
                params: serde_json::json!({
                    "textDocument": {
                        "uri": Url::from_file_path(root_path.join("foo.py")).unwrap().to_string()
                    },
                    "position": {
                        "line": 0,
                        "character": 43
                    }
                }),
            }),
        ],
        expected_messages_from_language_server: vec![
            Message::Request(Request {
                id: RequestId::from(1),
                method: WorkspaceConfiguration::METHOD.to_owned(),
                params: serde_json::json!(ConfigurationParams {
                    items: Vec::from([
                        ConfigurationItem {
                            scope_uri: Some(Url::from_file_path(root_path.as_path()).unwrap()),
                            section: Some("python".to_owned()),
                        },
                        ConfigurationItem {
                            scope_uri: None,
                            section: Some("python".to_owned()),
                        }
                    ]),
                }),
            }),
            make_sorted_completion_result_with_all_keywords(
                2,
                vec![
                    CompletionItem {
                        label: "this_is_a_very_long_function_name_so_we_can_deterministically_test_autoimport_with_fuzzy_search".to_owned(),
                        detail: Some("from autoimport_provider import this_is_a_very_long_function_name_so_we_can_deterministically_test_autoimport_with_fuzzy_search\n".to_owned()),
                        kind: Some(CompletionItemKind::FUNCTION),
                        sort_text: Some("4".to_owned()),
                        additional_text_edits: Some(vec![lsp_types::TextEdit {
                            range: lsp_types::Range {
                                start: lsp_types::Position {
                                    line: 0,
                                    character: 0,
                                },
                                end: lsp_types::Position {
                                    line: 0,
                                    character: 0,
                                },
                            },
                            new_text: "from autoimport_provider import this_is_a_very_long_function_name_so_we_can_deterministically_test_autoimport_with_fuzzy_search\n".to_owned(),
                        }]),
                        ..Default::default()
                    },
                ],
            ),
            Message::Request(Request {
                id: RequestId::from(2),
                method: WorkspaceConfiguration::METHOD.to_owned(),
                params: serde_json::json!(ConfigurationParams {
                    items: Vec::from([
                        ConfigurationItem {
                            scope_uri: Some(Url::from_file_path(root_path.as_path()).unwrap()),
                            section: Some("python".to_owned()),
                        },
                        ConfigurationItem {
                            scope_uri: None,
                            section: Some("python".to_owned()),
                        }
                    ]),
                }),
            }),
            make_sorted_completion_result_with_all_keywords(
                4,
                vec![
                    CompletionItem {
                        label: "this_is_a_very_long_function_name_so_we_can_deterministically_test_autoimport_with_fuzzy_search".to_owned(),
                        detail: Some("from autoimport_provider import this_is_a_very_long_function_name_so_we_can_deterministically_test_autoimport_with_fuzzy_search\n".to_owned()),
                        kind: Some(CompletionItemKind::FUNCTION),
                        sort_text: Some("4".to_owned()),
                        additional_text_edits: Some(vec![lsp_types::TextEdit {
                            range: lsp_types::Range {
                                start: lsp_types::Position {
                                    line: 0,
                                    character: 0,
                                },
                                end: lsp_types::Position {
                                    line: 0,
                                    character: 0,
                                },
                            },
                            new_text: "from autoimport_provider import this_is_a_very_long_function_name_so_we_can_deterministically_test_autoimport_with_fuzzy_search\n".to_owned(),
                        }]),
                        ..Default::default()
                    },
                ],
            ),
        ],
        indexing_mode: IndexingMode::LazyBlocking,
        workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
        configuration: true,
        ..Default::default()
    });
}

#[test]
fn test_completion_with_autoimport_without_config() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    let root_path = root.path();
    let scope_uri = Url::from_file_path(root_path).unwrap();

    interaction.set_root(root_path.to_path_buf());
    interaction.initialize(InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
        ..Default::default()
    });

    let foo_path = root_path.join("foo.py");
    interaction.server.did_open("foo.py");

    interaction
        .server
        .send_message(Message::Notification(Notification {
            method: "textDocument/didChange".to_owned(),
            params: serde_json::json!({
                "textDocument": {
                    "uri": Url::from_file_path(&foo_path).unwrap().to_string(),
                    "languageId": "python",
                    "version": 2
                },
                "contentChanges": [{
                    "text": "Bar".to_owned()
                }],
            }),
        }));

    interaction.server.completion("foo.py", 0, 3);

    interaction.client.expect_response_with(
        |response| {
            if response.id != RequestId::from(2) {
                return false;
            }
            if let Some(result) = &response.result
                && let Some(items) = result.get("items")
                && let Some(items_array) = items.as_array()
            {
                return !items_array.is_empty();
            }
            false
        },
        "Expected completion response with items",
    );

    interaction.shutdown();
}

#[test]
fn test_completion_with_autoimport_in_defined_module() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(&root_path).unwrap();

    interaction.set_root(root_path.clone());
    interaction.initialize(InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
        ..Default::default()
    });

    let file = root_path.join("autoimport_provider.py");
    interaction.server.did_open("autoimport_provider.py");

    let file_content = std::fs::read_to_string(&file).unwrap();
    interaction
        .server
        .send_message(Message::Notification(Notification {
            method: "textDocument/didChange".to_owned(),
            params: serde_json::json!({
                "textDocument": {
                    "uri": Url::from_file_path(&file).unwrap().to_string(),
                    "languageId": "python",
                    "version": 2
                },
                "contentChanges": [{
                    "text": format!("{}\n{}", file_content, "this_is_a_very_long_function_name_so_we_can")
                }],
            }),
        }));

    interaction.server.send_message(Message::Request(Request {
        id: RequestId::from(2),
        method: "textDocument/completion".to_owned(),
        params: serde_json::json!({
            "textDocument": {
                "uri": Url::from_file_path(&file).unwrap().to_string()
            },
            "position": {
                "line": 12,
                "character": 95
            }
        }),
    }));

    interaction.client.expect_response_with(
        |response| {
            if response.id != RequestId::from(2) {
                return false;
            }
            if let Some(result) = &response.result
                && let Some(items) = result.get("items")
                && let Some(items_array) = items.as_array()
            {
                return items_array.iter().any(|item| {
                    if let Some(label) = item.get("label")
                        && let Some(label_str) = label.as_str()
                        && let Some(detail) = item.get("detail")
                        && let Some(detail_str) = detail.as_str()
                    {
                        label_str == "this_is_a_very_long_function_name_so_we_can_deterministically_test_autoimport_with_fuzzy_search"
                            && detail_str == "() -> None"
                    } else {
                        false
                    }
                });
            }
            false
        },
        "Expected completion response with local function",
    );

    interaction.shutdown();
}

#[test]
fn test_completion_with_autoimport_duplicates() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    let root_path = root.path().join("duplicate_export_test");
    let scope_uri = Url::from_file_path(&root_path).unwrap();

    interaction.set_root(root_path.clone());
    interaction.initialize(InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
        ..Default::default()
    });

    interaction.server.did_open("foo.py");

    interaction.server.completion("foo.py", 5, 14);

    interaction.client.expect_response_with(
        |response| {
            if response.id != RequestId::from(2) {
                return false;
            }
            if let Some(result) = &response.result
                && let Some(items) = result.get("items")
                && let Some(items_array) = items.as_array()
            {
                return !items_array.is_empty();
            }
            false
        },
        "Expected completion response with items",
    );

    interaction.shutdown();
}

#[test]
fn test_module_completion() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("tests_requiring_config"));
    interaction.initialize(InitializeSettings::default());

    interaction.server.did_open("foo.py");

    interaction.server.completion("foo.py", 5, 10);

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({
            "isIncomplete": false,
            "items": [{
                "label": "bar",
                "detail": "bar",
                "kind": 9,
                "sortText": "0"
            }],
        })),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_relative_module_completion() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings::default());

    interaction
        .server
        .did_open("relative_test/relative_import.py");

    interaction
        .server
        .completion("relative_test/relative_import.py", 5, 10);

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({
            "isIncomplete": false,
            "items": [],
        })),
        error: None,
    });

    interaction.shutdown();
}
