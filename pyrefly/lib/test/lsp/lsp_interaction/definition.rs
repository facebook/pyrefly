/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;

use lsp_server::Message;
use lsp_server::Request;
use lsp_server::RequestId;
use lsp_types::Url;
use serde_json::json;
use tempfile::TempDir;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::bundled_typeshed_path;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

fn test_go_to_def(
    root: PathBuf,
    workspace_folders: Option<Vec<(String, Url)>>,
    // request file name, relative to root
    request_file_name: &'static str,
    // (line, character, response_file_name (relative to root), response_line_start, response_character_start, response_line_end, response_character_end)
    requests: Vec<(u32, u32, &'static str, u32, u32, u32, u32)>,
) {
    let mut interaction = LspInteraction::new();
    interaction.set_root(root);
    interaction.initialize(InitializeSettings {
        workspace_folders,
        ..Default::default()
    });
    interaction.client.did_open(request_file_name);

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
            .client
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

#[test]
fn definition_on_attr_of_pyi_assignment_goes_to_py() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        ..Default::default()
    });
    let file = "attributes_of_py/src_with_assignments.py";
    interaction.client.did_open(file);
    // Test annotated assignment (x: int = 100)
    interaction.client.definition(file, 7, 8);
    interaction.client.expect_definition_response_from_root(
        "attributes_of_py/lib_with_assignments.py",
        7,
        4,
        7,
        5,
    );
    // Test regular assignment (y = "world")
    interaction.client.definition(file, 8, 8);
    interaction.client.expect_definition_response_from_root(
        "attributes_of_py/lib_with_assignments.py",
        8,
        4,
        8,
        5,
    );
    interaction.shutdown();
}

fn test_go_to_def_basic(root: &TempDir, workspace_folders: Option<Vec<(String, Url)>>) {
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("basic"));
    let file = "foo.py";
    interaction.initialize(InitializeSettings {
        workspace_folders: workspace_folders.clone(),
        ..Default::default()
    });
    interaction.client.did_open(file);
    interaction.client.definition(file, 5, 7);
    interaction
        .client
        .expect_definition_response_from_root("bar.py", 0, 0, 0, 0);
    interaction.client.definition(file, 6, 16);
    interaction
        .client
        .expect_definition_response_from_root("bar.py", 6, 6, 6, 9);
    interaction.client.definition(file, 8, 9);
    interaction
        .client
        .expect_definition_response_from_root("bar.py", 7, 4, 7, 7);
    interaction.client.definition(file, 9, 7);
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
            Url::from_file_path(root.path().join("basic")).unwrap(),
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
    let root = get_test_files_root();
    let basic_root = root.path().join("basic");
    test_go_to_def(
        basic_root,
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
fn test_go_to_def_relative_path_helper() {
    let root = get_test_files_root();
    let basic_root = root.path().join("basic");
    test_go_to_def(
        basic_root,
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
        .client
        .did_open("imports_builtins/imports_builtins.py");
    interaction
        .client
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

#[test]
fn definition_on_attr_of_pyi_goes_to_py() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        ..Default::default()
    });
    let file = "attributes_of_py/src.py";
    interaction.client.did_open(file);
    interaction.client.definition(file, 7, 4);
    interaction
        .client
        .expect_definition_response_from_root("attributes_of_py/lib.py", 7, 8, 7, 9);
    interaction.shutdown();
}

#[test]
fn definition_in_builtins_without_interpreter_goes_to_stub() {
    let root = get_test_files_root();
    let pyrefly_typeshed_materialized = bundled_typeshed_path();
    let result_file = pyrefly_typeshed_materialized.join("typing.pyi");
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(Some(json!([{"pythonPath": "/fake/python/path"}]))),
        ..Default::default()
    });
    interaction.client.did_open("imports_builtins_no_config.py");
    interaction
        .client
        .definition("imports_builtins_no_config.py", 7, 7);
    interaction.client.expect_definition_response_absolute(
        result_file.to_string_lossy().to_string(),
        444,
        0,
        444,
        4,
    );
}

#[test]
fn malformed_missing_position() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("basic"));
    interaction.initialize(InitializeSettings {
        ..Default::default()
    });
    interaction.client.did_open("foo.py");
    interaction.client.send_message(Message::Request(Request {
        id: RequestId::from(2),
        method: "textDocument/definition".to_owned(),
        // Missing position - intentionally malformed to test error handling
        params: json!({
            "textDocument": {
                "uri": Url::from_file_path(root.path().join("basic/foo.py")).unwrap().to_string()
            },
        }),
    }));
    interaction.client.expect_response_error(
        RequestId::from(2),
        json!({
            "code": -32602,
            "message": "missing field `position`",
            "data": null,
        }),
    );
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
    interaction.client.did_open("main.py");
    interaction.client.definition("main.py", 5, 18);
    interaction
        .client
        .expect_definition_response_from_root("foo.pyi", 5, 4, 5, 7);
}

#[test]
fn goto_type_def_on_str_primitive_goes_to_builtins_stub() {
    let root = get_test_files_root();
    let pyrefly_typeshed_materialized = bundled_typeshed_path();
    let result_file = pyrefly_typeshed_materialized.join("builtins.pyi");
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        ..Default::default()
    });
    interaction.client.did_open("primitive_type_test.py");
    interaction
        .client
        .type_definition("primitive_type_test.py", 5, 0);

    interaction.client.expect_definition_response_absolute(
        result_file.to_string_lossy().to_string(),
        1023,
        6,
        1023,
        9,
    );

    assert!(
        result_file.exists(),
        "Expected builtins.pyi to exist at {result_file:?}",
    );
}

#[test]
fn goto_type_def_on_int_primitive_goes_to_builtins_stub() {
    let root = get_test_files_root();
    let pyrefly_typeshed_materialized = bundled_typeshed_path();
    let result_file = pyrefly_typeshed_materialized.join("builtins.pyi");
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        ..Default::default()
    });
    interaction.client.did_open("primitive_type_test.py");

    interaction
        .client
        .type_definition("primitive_type_test.py", 6, 0);

    // Expect to go to the int class definition in builtins.pyi
    // Line 252 is 0-indexed (253 - 1), where "class int:" is defined
    interaction.client.expect_definition_response_absolute(
        result_file.to_string_lossy().to_string(),
        417,
        6,
        417,
        9,
    );

    assert!(
        result_file.exists(),
        "Expected builtins.pyi to exist at {result_file:?}",
    );
}

#[test]
fn goto_type_def_on_bool_primitive_goes_to_builtins_stub() {
    let root = get_test_files_root();
    let pyrefly_typeshed_materialized = bundled_typeshed_path();
    let result_file = pyrefly_typeshed_materialized.join("builtins.pyi");
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        ..Default::default()
    });
    interaction.client.did_open("primitive_type_test.py");
    interaction
        .client
        .type_definition("primitive_type_test.py", 7, 0);

    // Expect to go to the bool class definition in builtins.pyi
    // Line 953 is 0-indexed (954 - 1), where "class bool:" is defined
    interaction.client.expect_definition_response_absolute(
        result_file.to_string_lossy().to_string(),
        3098,
        6,
        3098,
        10,
    );

    assert!(
        result_file.exists(),
        "Expected builtins.pyi to exist at {result_file:?}",
    );
}

#[test]
fn goto_type_def_on_bytes_primitive_goes_to_builtins_stub() {
    let root = get_test_files_root();
    let pyrefly_typeshed_materialized = bundled_typeshed_path();
    let result_file = pyrefly_typeshed_materialized.join("builtins.pyi");
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        ..Default::default()
    });
    interaction.client.did_open("primitive_type_test.py");

    interaction
        .client
        .type_definition("primitive_type_test.py", 8, 0);

    // Expect to go to the bytes class definition in builtins.pyi
    // Line 662 is 0-indexed (663 - 1), where "class bytes:" is defined
    interaction.client.expect_definition_response_absolute(
        result_file.to_string_lossy().to_string(),
        1835,
        6,
        1835,
        11,
    );

    assert!(
        result_file.exists(),
        "Expected builtins.pyi to exist at {result_file:?}",
    );
}

#[test]
fn goto_type_def_on_custom_class_goes_to_class_definition() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        ..Default::default()
    });
    interaction.client.did_open("custom_class_type_test.py");

    interaction
        .client
        .type_definition("custom_class_type_test.py", 8, 6);

    // Expect to go to the Foo class definition (line 6, columns 6-9)
    interaction.client.expect_definition_response_from_root(
        "custom_class_type_test.py",
        6,
        6,
        6,
        9,
    );
}

#[test]
fn goto_type_def_on_list_of_primitives_shows_selector() {
    let root = get_test_files_root();
    let pyrefly_typeshed_materialized = bundled_typeshed_path();
    let builtins_file = pyrefly_typeshed_materialized.join("builtins.pyi");
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        ..Default::default()
    });
    interaction.client.did_open("primitive_type_test.py");

    interaction
        .client
        .type_definition("primitive_type_test.py", 9, 0);

    interaction.client.expect_response_with(
        |response| {
            if let Some(result) = &response.result
                && let Some(locations_array) = result.as_array()
            {
                if locations_array.len() != 2 {
                    return false;
                }

                let has_int = locations_array.iter().any(|loc| {
                    loc.get("uri")
                        .and_then(|uri| uri.as_str())
                        .is_some_and(|u| u.ends_with("builtins.pyi"))
                        && loc
                            .get("range")
                            .and_then(|r| r.get("start"))
                            .and_then(|s| s.get("line"))
                            .and_then(|l| l.as_u64())
                            == Some(417)
                });

                let has_list = locations_array.iter().any(|loc| {
                    loc.get("uri")
                        .and_then(|uri| uri.as_str())
                        .is_some_and(|u| u.ends_with("builtins.pyi"))
                        && loc
                            .get("range")
                            .and_then(|r| r.get("start"))
                            .and_then(|s| s.get("line"))
                            .and_then(|l| l.as_u64())
                            == Some(3348)
                });

                return has_int && has_list;
            }
            false
        },
        "response should contain two locations: one for int (line 417) and one for list (line 3348) in builtins.pyi",
    );

    assert!(
        builtins_file.exists(),
        "Expected builtins.pyi to exist at {builtins_file:?}",
    );
}

#[test]
fn test_go_to_def_constructor_calls() {
    // Note: go-to-definition currently goes to the class definition, not __init__.
    let root = get_test_files_root();
    let constructor_root = root.path().join("constructor_references");
    test_go_to_def(
        constructor_root,
        None,
        "usage.py",
        vec![
            // Person("Alice", 30) - goes to class Person definition
            (7, 7, "person.py", 6, 6, 6, 12),
            // Person("Bob", 25) - goes to class Person definition
            (8, 7, "person.py", 6, 6, 6, 12),
        ],
    );
}
