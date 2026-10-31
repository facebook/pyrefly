/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fs;

use lsp_types::CodeLens;
use lsp_types::CodeLensRequest;
use lsp_types::Uri;
use pyrefly_lsp_test::IndexingMode;
use pyrefly_lsp_test::LspArgs;
use pyrefly_lsp_test::Message;
use pyrefly_lsp_test::object_model::CellKind;
use pyrefly_lsp_test::object_model::InitializeSettings;
use pyrefly_lsp_test::object_model::LspInteraction;
use pyrefly_lsp_test::object_model::LspInteractionArgs;
use serde_json::Value;
use serde_json::json;
use tempfile::TempDir;

use crate::lsp::non_wasm::code_lens::OverrideCodeLensRequest;
use crate::lsp::non_wasm::code_lens::OverrideCodeLensTargets;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

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
    let uri = Uri::from_file_path(&path).unwrap();

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
    let uri = Uri::from_file_path(&path).unwrap();
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
    let uri = Uri::from_file_path(&path).unwrap();

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
    let uri = Uri::from_file_path(&path).unwrap();

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
fn test_override_code_lens_batch_and_document_version() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    let test_root = root.path().join("code_lens");
    interaction.set_root(test_root.clone());
    interaction.initialize(Default::default()).unwrap();
    let uri = Uri::from_file_path(test_root.join("overrides.py")).unwrap();
    interaction.client.did_open_uri(
        &uri,
        "python",
        "class Base:\n    def f(self): pass\nclass Child(Base):\n    def f(self): pass\n",
    );
    let params = json!({
        "textDocument": {"uri": uri.to_string(), "version": 1},
        "ranges": [
            {"start": {"line": 1, "character": 8}, "end": {"line": 1, "character": 9}},
            {"start": {"line": 3, "character": 8}, "end": {"line": 3, "character": 9}}
        ]
    });
    interaction
        .client
        .send_request::<OverrideCodeLensRequest>(params.clone())
        .expect_response_with(|targets: Vec<OverrideCodeLensTargets>| {
            targets.len() == 2
                && targets[0].parents.is_empty()
                && targets[0].implementations.len() == 1
                && targets[0].implementations[0].location.range.start.line == 3
                && targets[0].implementations[0].name == "Child.f"
                && targets[1].parents.len() == 1
                && targets[1].parents[0].location.range.start.line == 1
                && targets[1].parents[0].name == "Base.f"
                && targets[1].implementations.is_empty()
        })
        .unwrap();
    let mut stale_params = params;
    stale_params["textDocument"]["version"] = json!(0);
    interaction
        .client
        .send_request::<OverrideCodeLensRequest>(stale_params)
        .expect_response_error_code(-32800)
        .unwrap();
    interaction.shutdown().unwrap();
}

#[test]
fn test_override_code_lens_notebook_cell_versions() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(Default::default()).unwrap();
    interaction.open_notebook(
        "overrides.ipynb",
        vec!["class Base:\n    def f(self): pass\nclass Child(Base):\n    def f(self): pass\n"],
    );
    let uri = interaction.cell_uri("overrides.ipynb", "cell1");
    let mut params = json!({
        "textDocument": {"uri": uri.to_string(), "version": 1},
        "ranges": [{"start": {"line": 3, "character": 8}, "end": {"line": 3, "character": 9}}]
    });
    for notebook_version in [1, 2] {
        if notebook_version == 2 {
            interaction.change_notebook(
                "overrides.ipynb",
                2,
                json!({"metadata": {"language_info": {"name": "python"}}}),
            );
        }
        interaction
            .client
            .send_request::<OverrideCodeLensRequest>(params.clone())
            .expect_response_with(|targets: Vec<OverrideCodeLensTargets>| {
                targets.len() == 1
                    && targets[0].parents.len() == 1
                    && targets[0].parents[0].name == "Base.f"
                    && targets[0].parents[0].location.uri == uri
                    && targets[0].parents[0].location.range.start.line == 1
            })
            .unwrap();
    }
    // The cell text version changes independently of the notebook version.
    interaction.change_notebook("overrides.ipynb", 3, json!({"cells": {"textContent": [{
        "document": {"uri": uri.to_string(), "version": 7},
        "changes": [{"text": "class Base:\n    def f(self): pass\nclass Child(Base):\n    def f(self): pass\n"}]
    }]}}));
    interaction
        .client
        .send_request::<OverrideCodeLensRequest>(params.clone())
        .expect_response_error_code(-32800)
        .unwrap();
    params["textDocument"]["version"] = json!(7);
    interaction
        .client
        .send_request::<OverrideCodeLensRequest>(params)
        .expect_response_with(|targets: Vec<OverrideCodeLensTargets>| {
            targets.len() == 1 && targets[0].parents.len() == 1
        })
        .unwrap();
    interaction.shutdown().unwrap();
}

#[test]
fn test_override_code_lens_unsaved_notebook() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(Default::default()).unwrap();
    interaction.open_notebook_with_uri(
        "untitled:Untitled-Overrides.ipynb",
        "jupyter-notebook",
        "unsaved_overrides.ipynb",
        vec![(
            CellKind::Code,
            "class Base:\n    def f(self): pass\nclass Child(Base):\n    def f(self): pass\n",
        )],
    );
    let uri = interaction.cell_uri("unsaved_overrides.ipynb", "cell1");
    let mut params = json!({
        "textDocument": {"uri": uri.to_string(), "version": 1},
        "ranges": [{"start": {"line": 3, "character": 8}, "end": {"line": 3, "character": 9}}]
    });
    interaction
        .client
        .send_request::<OverrideCodeLensRequest>(params.clone())
        .expect_response_with(|targets: Vec<OverrideCodeLensTargets>| {
            targets.len() == 1
                && targets[0].parents.len() == 1
                && targets[0].parents[0].name == "Base.f"
                && targets[0].parents[0].location.uri == uri
                && targets[0].parents[0].location.range.start.line == 1
        })
        .unwrap();
    params["textDocument"]["version"] = json!(0);
    interaction
        .client
        .send_request::<OverrideCodeLensRequest>(params)
        .expect_response_error_code(-32800)
        .unwrap();
    interaction.shutdown().unwrap();
}

// Known bug: target conversion uses code-cell indices to select from all notebook cells.
#[test]
fn test_override_code_lens_mixed_notebook_cells() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(Default::default()).unwrap();
    interaction.open_notebook_with_kinds(
        "mixed_overrides.ipynb",
        vec![
            (CellKind::Markdown, "# Models"),
            (CellKind::Code, "class Base:\n    def f(self): pass\n"),
            (
                CellKind::Code,
                "class Child(Base):\n    def f(self): pass\n",
            ),
        ],
    );
    let markdown_uri = interaction.cell_uri("mixed_overrides.ipynb", "cell1");
    let base_uri = interaction.cell_uri("mixed_overrides.ipynb", "cell2");
    let child_uri = interaction.cell_uri("mixed_overrides.ipynb", "cell3");
    let ranges = json!([{
        "start": {"line": 1, "character": 8},
        "end": {"line": 1, "character": 9}
    }]);
    interaction
        .client
        .send_request::<OverrideCodeLensRequest>(json!({
            "textDocument": {"uri": base_uri.to_string(), "version": 1},
            "ranges": ranges
        }))
        .expect_response_with(|targets: Vec<OverrideCodeLensTargets>| {
            targets.len() == 1
                && targets[0].implementations.len() == 1
                && targets[0].implementations[0].name == "Child.f"
                && targets[0].implementations[0].location.uri == base_uri
                && targets[0].implementations[0].location.range.start.line == 1
        })
        .unwrap();
    interaction
        .client
        .send_request::<OverrideCodeLensRequest>(json!({
            "textDocument": {"uri": child_uri.to_string(), "version": 1},
            "ranges": ranges
        }))
        .expect_response_with(|targets: Vec<OverrideCodeLensTargets>| {
            targets.len() == 1
                && targets[0].parents.len() == 1
                && targets[0].parents[0].name == "Base.f"
                && targets[0].parents[0].location.uri == markdown_uri
                && targets[0].parents[0].location.range.start.line == 1
        })
        .unwrap();
    interaction.shutdown().unwrap();
}

#[test]
fn test_override_code_lens_refresh_after_subclass_change() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    let test_root = root.path().join("code_lens");
    interaction.set_root(test_root.clone());
    interaction.initialize(Default::default()).unwrap();
    let uri = Uri::from_file_path(test_root.join("base.py")).unwrap();
    interaction
        .client
        .did_open_uri(&uri, "python", "class Base:\n    def f(self): pass\n");
    interaction
        .client
        .send_request::<OverrideCodeLensRequest>(json!({
            "textDocument": {"uri": uri.to_string(), "version": 1},
            "refreshSupport": true,
            "ranges": [{"start": {"line": 1, "character": 8}, "end": {"line": 1, "character": 9}}]
        }))
        .expect_response_with(|_: Vec<OverrideCodeLensTargets>| true)
        .unwrap();
    let child_uri = Uri::from_file_path(test_root.join("child.py")).unwrap();
    interaction.client.did_open_uri(
        &child_uri,
        "python",
        "from base import Base\nclass Child(Base):\n    def f(self): pass\n",
    );
    interaction.client.did_change(
        "child.py",
        "from base import Base\nclass Child(Base):\n    def f(self): return 1\n",
    );
    interaction
        .client
        .expect_message("override CodeLens invalidation", |message| match message {
            Message::Notification(notification)
                if notification.method == "pyrefly/overrideCodeLens/refresh" =>
            {
                Some(Ok(()))
            }
            _ => None,
        })
        .unwrap();
    interaction.shutdown().unwrap();
}

#[test]
fn test_override_code_lens_refresh_after_indexed_file_change() {
    let root = TempDir::new().unwrap();
    fs::write(
        root.path().join("pyrefly.toml"),
        "project-includes = ['*.py']\n",
    )
    .unwrap();
    fs::write(
        root.path().join("base.py"),
        "class Base:\n    def f(self): pass\n",
    )
    .unwrap();
    let child = root.path().join("child.py");
    fs::write(
        &child,
        "from base import Base\nclass Child(Base):\n    def f(self): pass\n",
    )
    .unwrap();
    let uri = Uri::from_file_path(root.path().join("base.py")).unwrap();
    let scope = Uri::from_file_path(root.path()).unwrap();
    let mut interaction = LspInteraction::new_with_args(LspInteractionArgs {
        args: LspArgs {
            indexing_mode: IndexingMode::LazyBlocking,
            ..LspInteractionArgs::default().args
        },
        ..Default::default()
    });
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope)]),
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();
    interaction.client.did_open("base.py");
    let params = json!({
        "textDocument": {"uri": uri.to_string(), "version": 1},
        "refreshSupport": true,
        "ranges": [{"start": {"line": 1, "character": 8}, "end": {"line": 1, "character": 9}}]
    });
    interaction
        .client
        .send_request::<OverrideCodeLensRequest>(params.clone())
        .expect_response_with(|targets: Vec<OverrideCodeLensTargets>| {
            targets.len() == 1 && targets[0].implementations.len() == 1
        })
        .unwrap();
    fs::write(
        &child,
        "from base import Base\nclass Child(Base):\n    def g(self): pass\n",
    )
    .unwrap();
    interaction.client.file_modified("child.py");
    interaction
        .client
        .expect_message(
            "CodeLens invalidation after indexing",
            |message| match message {
                Message::Notification(notification)
                    if notification.method == "pyrefly/overrideCodeLens/refresh" =>
                {
                    Some(Ok(()))
                }
                _ => None,
            },
        )
        .unwrap();
    interaction
        .client
        .send_request::<OverrideCodeLensRequest>(params)
        .expect_response_with(|targets: Vec<OverrideCodeLensTargets>| {
            targets.len() == 1 && targets[0].implementations.is_empty()
        })
        .unwrap();
    interaction.shutdown().unwrap();
}
