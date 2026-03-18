/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::CodeLens;
use lsp_types::CodeLensOptions;
use lsp_types::Url;
use lsp_types::request::CodeLensRequest;
use pyrefly::commands::lsp::IndexingMode;
use serde_json::Value;
use serde_json::json;

use crate::object_model::InitializeSettings;
use crate::object_model::LspInteraction;
use crate::util::get_test_files_root;

fn runnable_code_lens_config() -> serde_json::Value {
    json!([{
        "pyrefly": {
            "runnableCodeLens": true
        }
    }])
}

#[test]
fn test_code_lens_for_tests_and_main() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    let test_root = root.path().join("code_lens");
    interaction.set_root(test_root.clone());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(runnable_code_lens_config())),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("main_and_tests.py");

    let path = test_root.join("main_and_tests.py");
    let uri = Url::from_file_path(&path).unwrap();

    interaction
        .client
        .send_request::<CodeLensRequest>(json!({
            "textDocument": {
                "uri": uri.to_string()
            },
        }))
        .expect_response_with(|response: Option<Vec<CodeLens>>| {
            let Some(lenses) = response else {
                return false;
            };

            let mut has_run = false;
            let mut pytest_test = false;
            let mut unittest_test = false;
            let mut top_level_test = false;
            for lens in lenses {
                let Some(command) = lens.command else {
                    continue;
                };
                if command.command == "pyrefly.runMain" && command.title == "Run" {
                    has_run |= lens.range.start.line == 26;
                }
                if command.command == "pyrefly.runTest" && command.title == "Test" {
                    let args = command.arguments.clone().unwrap_or_default();
                    let Some(Value::Object(obj)) = args.first() else {
                        continue;
                    };
                    let is_unittest = obj
                        .get("isUnittest")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                    let test_name = obj.get("testName").and_then(Value::as_str);
                    let class_name = obj.get("className").and_then(Value::as_str);
                    match (is_unittest, class_name, test_name) {
                        (false, Some("TestPytest"), Some("test_method")) => pytest_test = true,
                        (true, Some("MyTestCase"), Some("test_unittest")) => unittest_test = true,
                        (false, None, Some("test_top_level")) => top_level_test = true,
                        _ => {}
                    }
                }
            }

            has_run && pytest_test && unittest_test && top_level_test
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_code_lens_uses_config_root_for_cwd() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    let test_root = root.path().join("code_lens");
    interaction.set_root(test_root.clone());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(runnable_code_lens_config())),
            ..Default::default()
        })
        .unwrap();

    interaction
        .client
        .did_open("nested_project/main_and_tests.py");

    let path = test_root.join("nested_project/main_and_tests.py");
    let uri = Url::from_file_path(&path).unwrap();
    let expected_cwd = test_root
        .join("nested_project")
        .to_string_lossy()
        .into_owned();

    interaction
        .client
        .send_request::<CodeLensRequest>(json!({
            "textDocument": {
                "uri": uri.to_string()
            },
        }))
        .expect_response_with(|response: Option<Vec<CodeLens>>| {
            let Some(lenses) = response else {
                return false;
            };
            let mut saw_lens = false;
            lenses.into_iter().all(|lens| {
                saw_lens = true;
                lens.command
                    .and_then(|command| command.arguments)
                    .and_then(|args| args.into_iter().next())
                    .and_then(|arg| arg.get("cwd").and_then(Value::as_str).map(str::to_owned))
                    .is_some_and(|cwd| cwd == expected_cwd)
            }) && saw_lens
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_code_lens_ignores_stub_files() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    let test_root = root.path().join("code_lens");
    interaction.set_root(test_root.clone());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(runnable_code_lens_config())),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("main_and_tests.pyi");

    let path = test_root.join("main_and_tests.pyi");
    let uri = Url::from_file_path(&path).unwrap();

    interaction
        .client
        .send_request::<CodeLensRequest>(json!({
            "textDocument": {
                "uri": uri.to_string()
            },
        }))
        .expect_response_with(|response: Option<Vec<CodeLens>>| {
            response.is_some_and(|lenses| lenses.is_empty())
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_code_lens_disabled_by_default() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    let test_root = root.path().join("code_lens");
    interaction.set_root(test_root.clone());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(json!([{}]))),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("main_and_tests.py");

    let path = test_root.join("main_and_tests.py");
    let uri = Url::from_file_path(&path).unwrap();

    interaction
        .client
        .send_request::<CodeLensRequest>(json!({
            "textDocument": {
                "uri": uri.to_string()
            },
        }))
        .expect_response_with(|response: Option<Vec<CodeLens>>| {
            response.is_some_and(|lenses| lenses.is_empty())
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_initialize_advertises_code_lens_with_indexing() {
    let interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);

    interaction
        .client
        .send_initialize(
            interaction
                .client
                .get_initialize_params(&InitializeSettings::default()),
        )
        .expect_response_with(|response| {
            response.capabilities.code_lens_provider
                == Some(CodeLensOptions {
                    resolve_provider: Some(false),
                })
        })
        .unwrap();
    interaction.client.send_initialized();
    interaction.shutdown().unwrap();
}

#[test]
fn test_code_lens_shows_reference_counts() {
    let root = get_test_files_root();
    let root_path = root.path().join("code_lens_references");
    let scope_uri = Url::from_file_path(&root_path).unwrap();
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(root_path);
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("symbols.py");
    interaction.client.did_open("usage.py");

    interaction
        .client
        .code_lens("symbols.py")
        .expect_response_with(|response| {
            let Some(lenses) = response else {
                return false;
            };
            has_reference_lens(&lenses, 0, "3 references", 3)
                && has_reference_lens(&lenses, 1, "2 references", 2)
                && has_reference_lens(&lenses, 4, "2 references", 2)
                && lenses.len() == 3
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

fn has_reference_lens(
    lenses: &[CodeLens],
    line: u32,
    expected_title: &str,
    expected_locations: usize,
) -> bool {
    lenses.iter().any(|lens| {
        let Some(command) = &lens.command else {
            return false;
        };
        if lens.range.start.line != line
            || command.title != expected_title
            || command.command != "editor.action.showReferences"
        {
            return false;
        }
        let Some(arguments) = &command.arguments else {
            return false;
        };
        arguments
            .get(2)
            .and_then(|value| value.as_array())
            .is_some_and(|locations| locations.len() == expected_locations)
    })
}
