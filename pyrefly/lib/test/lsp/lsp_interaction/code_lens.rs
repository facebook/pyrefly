/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cell::RefCell;
use std::path::Path;

use lsp_types::CodeLens;
use lsp_types::CodeLensOptions;
use lsp_types::Location;
use lsp_types::Url;
use lsp_types::request::CodeLensRequest;
use lsp_types::request::CodeLensResolve;
use pyrefly_lsp_test::IndexingMode;
use pyrefly_lsp_test::LspArgs;
use pyrefly_lsp_test::object_model::InitializeSettings;
use pyrefly_lsp_test::object_model::LspInteraction;
use pyrefly_lsp_test::object_model::LspInteractionArgs;
use serde_json::Value;
use serde_json::json;

use crate::test::lsp::lsp_interaction::util::get_test_files_root;

fn runnable_code_lens_config() -> serde_json::Value {
    json!([{
        "pyrefly": {
            "runnableCodeLens": true
        }
    }])
}

fn indexed_interaction(root_path: &Path) -> LspInteraction {
    let scope_uri = Url::from_file_path(root_path).unwrap();
    let mut interaction = LspInteraction::new_with_args(LspInteractionArgs {
        args: LspArgs {
            indexing_mode: IndexingMode::LazyBlocking,
            ..LspInteractionArgs::default().args
        },
        ..Default::default()
    });
    interaction.set_root(root_path.to_path_buf());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            ..Default::default()
        })
        .unwrap();
    interaction
}

fn request_code_lenses(interaction: &LspInteraction, uri: Url) -> Vec<CodeLens> {
    let lenses = RefCell::new(Vec::new());
    interaction
        .client
        .send_request::<CodeLensRequest>(json!({
            "textDocument": {
                "uri": uri
            }
        }))
        .expect_response_with(|response| {
            lenses.replace(response.unwrap_or_default());
            true
        })
        .unwrap();
    lenses.into_inner()
}

fn resolve_code_lens(interaction: &LspInteraction, lens: CodeLens) -> CodeLens {
    let resolved = RefCell::new(None);
    interaction
        .client
        .send_request::<CodeLensResolve>(serde_json::to_value(lens).unwrap())
        .expect_response_with(|lens| {
            resolved.replace(Some(lens));
            true
        })
        .unwrap();
    resolved.into_inner().unwrap()
}

#[test]
fn test_code_lens_advertises_resolve_provider() {
    let interaction = LspInteraction::new();
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
                    resolve_provider: Some(true),
                })
        })
        .unwrap();
    interaction.client.send_initialized();
    interaction.shutdown().unwrap();
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
fn test_code_lens_shows_reference_counts() {
    let root = get_test_files_root();
    let root_path = root.path().join("code_lens_references");
    let symbols_uri = Url::from_file_path(root_path.join("symbols.py")).unwrap();
    let interaction = indexed_interaction(&root_path);

    interaction.client.did_open("symbols.py");
    interaction.client.did_open("usage.py");

    let lenses = request_code_lenses(&interaction, symbols_uri);
    assert_eq!(lenses.len(), 3);
    assert!(
        lenses
            .iter()
            .all(|lens| lens.command.is_none() && lens.data.is_some())
    );
    for (line, title, count) in [
        (0, "3 references", 3),
        (1, "2 references", 2),
        (4, "2 references", 2),
    ] {
        let lens = lenses
            .iter()
            .find(|lens| lens.range.start.line == line)
            .unwrap()
            .clone();
        let lens = resolve_code_lens(&interaction, lens);
        assert!(has_reference_lens(&lens, line, title, count));
    }

    interaction.shutdown().unwrap();
}

#[test]
fn test_reference_code_lens_request_from_notebook_is_empty() {
    let root = get_test_files_root();
    let root_path = root.path().join("code_lens_references");
    let interaction = indexed_interaction(&root_path);
    interaction.open_notebook("notebook.ipynb", vec!["class NotebookSymbol:\n    pass"]);

    let lenses = request_code_lenses(
        &interaction,
        interaction.cell_uri("notebook.ipynb", "cell1"),
    );
    assert!(lenses.is_empty());

    interaction.shutdown().unwrap();
}

#[test]
fn test_reference_code_lens_excludes_notebook_references() {
    let root = get_test_files_root();
    let root_path = root.path().join("code_lens_references");
    let symbols_uri = Url::from_file_path(root_path.join("symbols.py")).unwrap();
    let interaction = indexed_interaction(&root_path);
    interaction.client.did_open("symbols.py");
    interaction.client.did_open("usage.py");
    interaction.open_notebook(
        "notebook.ipynb",
        vec!["from symbols import Greeter", "Greeter()"],
    );
    let notebook_uris = [
        interaction.cell_uri("notebook.ipynb", "cell1"),
        interaction.cell_uri("notebook.ipynb", "cell2"),
    ];

    let lenses = request_code_lenses(&interaction, symbols_uri);
    let greeter_lens = lenses
        .into_iter()
        .find(|lens| lens.range.start.line == 0)
        .unwrap();
    let resolved = resolve_code_lens(&interaction, greeter_lens);
    let locations = reference_locations(&resolved);
    assert_eq!(locations.len(), 3);
    assert!(
        locations
            .iter()
            .all(|location| !notebook_uris.contains(&location.uri))
    );

    interaction.shutdown().unwrap();
}

fn has_reference_lens(
    lens: &CodeLens,
    line: u32,
    expected_title: &str,
    expected_locations: usize,
) -> bool {
    let Some(command) = &lens.command else {
        return false;
    };
    lens.range.start.line == line
        && command.title == expected_title
        && command.command == "editor.action.showReferences"
        && reference_locations(lens).len() == expected_locations
}

fn reference_locations(lens: &CodeLens) -> Vec<Location> {
    let arguments = lens
        .command
        .as_ref()
        .and_then(|command| command.arguments.as_ref())
        .expect("Resolved reference lens should have command arguments");
    serde_json::from_value(
        arguments
            .get(2)
            .expect("Reference locations should be the third argument")
            .clone(),
    )
    .expect("Reference locations should deserialize")
}
