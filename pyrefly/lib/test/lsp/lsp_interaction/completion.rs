/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use itertools::Itertools;
use lsp_types::CompletionItemKind;
use lsp_types::CompletionResponse;
use lsp_types::Url;
use lsp_types::notification::DidChangeTextDocument;
use lsp_types::request::Completion;
use serde_json::json;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_completion_basic() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("basic"));
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    interaction.client.did_open("foo.py");

    let root_path = root.path().join("basic");
    let foo_path = root_path.join("foo.py");
    interaction
        .client
        .send_notification::<DidChangeTextDocument>(json!({
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
        }));

    interaction
        .client
        .completion("foo.py", 11, 1)
        .expect_completion_response_with(|list| list.items.iter().any(|item| item.label == "Bar"))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_completion_sorted_in_sorttext_order() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("basic"));
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    interaction.client.did_open("foo.py");

    let root_path = root.path().join("basic");
    let foo_path = root_path.join("foo.py");
    interaction
        .client
        .send_notification::<DidChangeTextDocument>(json!({
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
        }));

    interaction
        .client
        .completion("foo.py", 11, 1)
        .expect_completion_response_with(|list| {
            list.items
                .iter()
                .is_sorted_by_key(|x| (&x.sort_text, &x.label))
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_completion_keywords() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("basic"));
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    interaction.client.did_open("foo.py");

    let root_path = root.path().join("basic");
    let foo_path = root_path.join("foo.py");

    interaction
        .client
        .send_notification::<DidChangeTextDocument>(json!({
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
        }));

    interaction
        .client
        .completion("foo.py", 11, 1)
        .expect_completion_response_with(|list| {
            let mut has_if = false;
            let mut has_import = false;
            let mut has_def = false;
            for item in &list.items {
                if item.kind == Some(CompletionItemKind::KEYWORD) {
                    has_if = has_if || item.label == "if";
                    has_import = has_import || item.label == "import";
                    has_def = has_def || item.label == "def";
                }
            }
            has_if && has_import && has_def
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_import_completion_skips_hidden_directories() {
    let root = get_test_files_root();
    let workspace = root.path().join("basic");
    let hidden_dir = workspace.join(".hiddenpkg");
    std::fs::create_dir_all(&hidden_dir).unwrap();
    std::fs::write(hidden_dir.join("__init__.py"), "").unwrap();

    let foo_path = workspace.join("foo.py");

    let mut interaction = LspInteraction::new();
    interaction.set_root(workspace);
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    interaction.client.did_open("foo.py");

    interaction
        .client
        .send_notification::<DidChangeTextDocument>(json!({
            "textDocument": {
                "uri": Url::from_file_path(&foo_path).unwrap().to_string(),
                "languageId": "python",
                "version": 2
            },
            "contentChanges": [{
                "text": "import ".to_owned()
            }],
        }));

    interaction
        .client
        .completion("foo.py", 0, 7)
        .expect_completion_response_with(|list| {
            assert!(list.items.iter().all(|item| item.label != ".hiddenpkg"));
            true
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_completion_with_autoimport() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");

    let mut interaction =
        LspInteraction::new_with_indexing_mode(crate::commands::lsp::IndexingMode::LazyBlocking);

    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    let file = root_path.join("foo.py");
    interaction.client.did_open("foo.py");

    interaction
        .client
        .send_notification::<DidChangeTextDocument>(json!({
            "textDocument": {
                "uri": Url::from_file_path(&file).unwrap().to_string(),
                "languageId": "python",
                "version": 2
            },
            "contentChanges": [{
                "text": "this_is_a_very_long_function_name_so_we_can".to_owned()
            }],
        }));

    interaction.client.completion("foo.py", 0, 43).expect_completion_response_with(|list| {
        list.items.iter().any(|item| {
            item.label == "this_is_a_very_long_function_name_so_we_can_deterministically_test_autoimport_with_fuzzy_search"
            && item.detail.as_ref().is_some_and(|detail| detail.contains("from autoimport_provider import"))
            && item.additional_text_edits.as_ref().is_some_and(|edits| !edits.is_empty())
        })
    }).unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_completion_with_autoimport_without_config() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    let root_path = root.path().join("basic");
    let scope_uri = Url::from_file_path(&root_path).unwrap();

    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            ..Default::default()
        })
        .unwrap();

    let foo_path = root_path.join("foo.py");
    interaction.client.did_open("foo.py");

    interaction
        .client
        .send_notification::<DidChangeTextDocument>(json!({
            "textDocument": {
                "uri": Url::from_file_path(&foo_path).unwrap().to_string(),
                "languageId": "python",
                "version": 2
            },
            "contentChanges": [{
                "text": "Bar".to_owned()
            }],
        }));

    interaction
        .client
        .completion("foo.py", 0, 3)
        .expect_completion_response_with(|list| !list.items.is_empty())
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_completion_with_autoimport_in_defined_module() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(&root_path).unwrap();

    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            ..Default::default()
        })
        .unwrap();

    let file = root_path.join("autoimport_provider.py");
    interaction.client.did_open("autoimport_provider.py");

    let file_content = std::fs::read_to_string(&file).unwrap();
    interaction
        .client
        .send_notification::<DidChangeTextDocument>(json!({
                "textDocument": {
                    "uri": Url::from_file_path(&file).unwrap().to_string(),
                    "languageId": "python",
                    "version": 2
                },
                "contentChanges": [{
                    "text": format!("{}\n{}", file_content, "this_is_a_very_long_function_name_so_we_can")
                }],
            }));

    interaction.client.send_request::<Completion>(
        json!({
            "textDocument": {
                "uri": Url::from_file_path(&file).unwrap().to_string()
            },
            "position": {
                "line": 12,
                "character": 95
            }
        }),
    ).expect_completion_response_with(|list| {
        list.items.iter().any(|item| {
            item.label == "this_is_a_very_long_function_name_so_we_can_deterministically_test_autoimport_with_fuzzy_search"
                && item.detail.as_ref().is_some_and(|detail| detail == "() -> None")
        })
    }).unwrap();

    interaction.shutdown().unwrap();
}

// TODO: figure out why this test fails on Windows.
#[cfg(unix)]
#[test]
fn test_completion_with_autoimport_duplicates() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    let root_path = root.path().join("duplicate_export_test");
    let scope_uri = Url::from_file_path(&root_path).unwrap();

    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("foo.py");

    interaction
        .client
        .completion("foo.py", 5, 14)
        .expect_completion_response_with(|list| !list.items.is_empty())
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_module_completion() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("tests_requiring_config"));
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    interaction.client.did_open("foo.py");

    interaction
        .client
        .completion("foo.py", 5, 10)
        .expect_response(json!({
            "isIncomplete": false,
            "items": [{
                "label": "bar",
                "detail": "bar",
                "kind": 9,
                "sortText": "0"
            }],
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_module_completion_reexports_sorted_lower() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("reexport_test"));
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    interaction.client.did_open("test.py");

    let test_path = root.path().join("reexport_test/test.py");
    interaction
        .client
        .send_notification::<DidChangeTextDocument>(json!({
            "textDocument": {
                "uri": Url::from_file_path(&test_path).unwrap().to_string(),
                "languageId": "python",
                "version": 2
            },
            "contentChanges": [{
                "text": "import module_with_reexports\n\nmodule_with_reexports.".to_owned()
            }],
        }));

    interaction
        .client
        .completion("test.py", 2, 23)
        .expect_completion_response_with(|list| {
            let mut direct_definitions = vec![];
            let mut reexports = vec![];
            for item in &list.items {
                if item.label == "another_direct_function" || item.label == "AnotherDirectClass" {
                    direct_definitions.push(&item.sort_text);
                } else if item.label == "reexported_function" || item.label == "ReexportedClass" {
                    reexports.push(&item.sort_text);
                }
            }
            !direct_definitions.is_empty()
                && !reexports.is_empty()
                && direct_definitions
                    .iter()
                    .cartesian_product(reexports.iter())
                    .all(|(direct_sort, reexport_sort)| reexport_sort > direct_sort)
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_relative_module_completion() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    interaction
        .client
        .did_open("relative_test/relative_import.py");

    interaction
        .client
        .completion("relative_test/relative_import.py", 5, 10)
        .expect_response(json!({
            "isIncomplete": false,
            "items": [],
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_stdlib_submodule_completion() {
    let root = get_test_files_root();
    let root_path = root.path().join("basic");

    let mut interaction =
        LspInteraction::new_with_indexing_mode(crate::commands::lsp::IndexingMode::LazyBlocking);

    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    interaction.client.did_open("foo.py");
    interaction.client.did_change("foo.py", "import email.");
    interaction
        .client
        .completion("foo.py", 0, 13)
        .expect_completion_response_with(|list| {
            list.items.iter().any(|item| {
                item.label == "errors"
                    && item.detail.as_deref() == Some("email.errors")
                    && item.kind == Some(CompletionItemKind::MODULE)
            })
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_stdlib_class_completion() {
    let root = get_test_files_root();
    let root_path = root.path().join("basic");

    let mut interaction =
        LspInteraction::new_with_indexing_mode(crate::commands::lsp::IndexingMode::LazyBlocking);

    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    interaction.client.did_open("foo.py");
    interaction.client.did_change("foo.py", "FirstHeader");
    interaction
        .client
        .completion("foo.py", 0, 11)
        .expect_completion_response_with(|list| {
            list.items
                .iter()
                .any(|item| item.label == "FirstHeaderLineIsContinuationDefect")
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_completion_incomplete_below_autoimport_threshold() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("basic"));
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    interaction.client.did_open("foo.py");

    // Type only 2 characters (below MIN_CHARACTERS_TYPED_AUTOIMPORT = 3)
    interaction.client.did_change("foo.py", "xy");

    interaction
        .client
        .completion("foo.py", 0, 2)
        .expect_response_with(|response| {
            // Since we typed only 2 characters and there are no local completions,
            // autoimport suggestions are skipped due to MIN_CHARACTERS_TYPED_AUTOIMPORT,
            // so is_incomplete should be true
            match response {
                Some(CompletionResponse::List(list)) => list.is_incomplete,
                _ => false,
            }
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_completion_complete_above_autoimport_threshold() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("basic"));
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    interaction.client.did_open("foo.py");

    // Type 3 characters (meets MIN_CHARACTERS_TYPED_AUTOIMPORT = 3)
    interaction.client.did_change("foo.py", "xyz");

    interaction
        .client
        .completion("foo.py", 0, 3)
        .expect_response_with(|response| {
            // Since we typed 3 characters (meets threshold), autoimport suggestions
            // are included, so is_incomplete should be false
            match response {
                Some(CompletionResponse::List(list)) => !list.is_incomplete,
                _ => false,
            }
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_completion_complete_with_local_completions() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("basic"));
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    interaction.client.did_open("foo.py");

    // Type 2 characters (below threshold) but match local completion "Ba" -> "Bar"
    interaction.client.did_change("foo.py", "Ba");

    // Even though we have local completions (like "Bar"), since we typed only 2 characters
    // (below MIN_CHARACTERS_TYPED_AUTOIMPORT), is_incomplete should be true to ensure
    // the client keeps asking for completions as the user types more characters.
    // This prevents the Zed bug where local completions prevent autoimport checks.
    interaction
        .client
        .completion("foo.py", 0, 2)
        .expect_completion_response_with(|list| list.is_incomplete)
        .unwrap();

    interaction.shutdown().unwrap();
}
