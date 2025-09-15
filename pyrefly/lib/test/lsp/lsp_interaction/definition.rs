/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::env::temp_dir;

use lsp_server::Message;
use lsp_server::Request;
use lsp_server::RequestId;
use lsp_server::Response;
use lsp_server::ResponseError;
use lsp_types::Url;
use tempfile::TempDir;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

fn test_go_to_def(
    root: &TempDir,
    workspace_folders: Option<Vec<(String, Url)>>,
    // request file name, relative to root
    request_file_name: &'static str,
    // (line, character, response_file_name (relative to root), response_line_start, response_character_start, response_line_end, response_character_end)
    requests: Vec<(u32, u32, &'static str, u32, u32, u32, u32)>,
) {
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        workspace_folders,
        ..Default::default()
    });
    interaction.server.did_open(request_file_name);

    for (
        request_line,
        request_character,
        response_file_name,
        response_line_start,
        response_character_start,
        response_line_end,
        response_character_end,
    ) in requests
    {
        interaction
            .server
            .definition(request_file_name, request_line, request_character);
        interaction.client.expect_definition_response_from_root(
            response_file_name,
            response_line_start,
            response_character_start,
            response_line_end,
            response_character_end,
        );
    }
}

fn test_go_to_def_basic(root: &TempDir, workspace_folders: Option<Vec<(String, Url)>>) {
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    let file = "foo.py";
    interaction.initialize(InitializeSettings {
        workspace_folders: workspace_folders.clone(),
        ..Default::default()
    });
    interaction.server.did_open(file);
    interaction.server.definition(file, 5, 7);
    interaction
        .client
        .expect_definition_response_from_root("bar.py", 0, 0, 0, 0);
    interaction.server.definition(file, 6, 16);
    interaction
        .client
        .expect_definition_response_from_root("bar.py", 6, 6, 6, 9);
    interaction.server.definition(file, 8, 9);
    interaction
        .client
        .expect_definition_response_from_root("bar.py", 7, 4, 7, 7);
    interaction.server.definition(file, 9, 7);
    interaction
        .client
        .expect_definition_response_from_root("bar.py", 6, 6, 6, 9);
}

#[test]
fn test_go_to_def_single_root() {
    let root = get_test_files_root();
    test_go_to_def_basic(
        &root,
        Some(vec![(
            "test".to_owned(),
            Url::from_file_path(root.path()).unwrap(),
        )]),
    );
}

#[test]
fn test_go_to_def_no_workspace_folders() {
    let root = get_test_files_root();
    test_go_to_def_basic(&root, Some(vec![]));
}

#[test]
fn test_go_to_def_no_folder_capability() {
    let root = get_test_files_root();
    test_go_to_def_basic(&root, None);
}

#[test]
fn test_go_to_def_relative_path() {
    test_go_to_def(
        &get_test_files_root(),
        None,
        "foo_relative.py",
        vec![
            (5, 14, "bar.py", 0, 0, 0, 0),
            (6, 17, "bar.py", 6, 6, 6, 9),
            (8, 9, "bar.py", 7, 4, 7, 7),
            (9, 7, "bar.py", 6, 6, 6, 9),
        ],
    );
}

#[test]
fn definition_in_builtins() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        ..Default::default()
    });
    interaction
        .server
        .did_open("imports_builtins/imports_builtins.py");
    interaction
        .server
        .definition("imports_builtins/imports_builtins.py", 7, 7);
    interaction.client.expect_response_with(
        |response| {
            // expect typing.py, NOT typing.pyi
            response.result.as_ref().is_some_and(|r| {
                r.get("uri")
                    .is_some_and(|uri| uri.as_str().is_some_and(|x| x.ends_with("typing.py")))
            })
        },
        "response must return the file `typing.py` from a site package",
    );
}

// todo(kylei): definition of an attribute of a pyi should still point to py
#[test]
fn definition_on_attr_of_pyi() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        ..Default::default()
    });
    let file = "attributes_of_py/src.py";
    interaction.server.did_open(file);
    interaction.server.definition(file, 7, 4);
    interaction
        .client
        .expect_definition_response_from_root("attributes_of_py/lib.pyi", 6, 8, 6, 9);
    interaction.shutdown();
}

#[test]
fn definition_in_builtins_without_interpreter_goes_to_stub() {
    let root = get_test_files_root();
    let pyrefly_typeshed_materialized = temp_dir().join("pyrefly_bundled_typeshed");
    let result_file = pyrefly_typeshed_materialized.join("typing.pyi");
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(Some(
            serde_json::json!([{"pythonPath": "/fake/python/path"}]),
        )),
        ..Default::default()
    });
    interaction.server.did_open("imports_builtins_no_config.py");
    interaction
        .server
        .definition("imports_builtins_no_config.py", 7, 7);
    interaction.client.expect_definition_response_absolute(
        result_file.to_string_lossy().to_string(),
        425,
        0,
        425,
        4,
    );
}

#[test]
fn malformed_missing_position() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        ..Default::default()
    });
    interaction.server.did_open("foo.py");
    interaction.server.send_message(Message::Request(Request {
        id: RequestId::from(2),
        method: "textDocument/definition".to_owned(),
        // Missing position
        params: serde_json::json!({
            "textDocument": {
                "uri": Url::from_file_path(root.path().join("foo.py")).unwrap().to_string()
            },
        }),
    }));
    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: None,
        error: Some(ResponseError {
            code: -32602,
            message: "missing field `position`".to_owned(),
            data: None,
        }),
    });
}

// we generally want to prefer py. but if it's missing in the py, we should prefer the pyi
#[test]
fn prefer_pyi_when_missing_in_py() {
    let root = get_test_files_root();
    let test_root = root.path().join("prefer_pyi_when_missing_in_py");
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_root);
    interaction.initialize(InitializeSettings {
        ..Default::default()
    });
    interaction.server.did_open("main.py");
    interaction.server.definition("main.py", 5, 18);
    interaction
        .client
        .expect_definition_response_from_root("foo.pyi", 5, 4, 5, 7);
}
