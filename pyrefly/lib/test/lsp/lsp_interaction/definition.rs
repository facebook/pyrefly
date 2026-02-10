/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;

use lsp_server::RequestId;
use lsp_types::GotoDefinitionResponse;
use lsp_types::Location;
use lsp_types::Url;
use serde_json::json;
use tempfile::TempDir;

use crate::lsp::non_wasm::protocol::Message;
use crate::lsp::non_wasm::protocol::Request;
use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::bundled_typeshed_path;
use crate::test::lsp::lsp_interaction::util::expect_definition_points_to_symbol;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;
use crate::test::lsp::lsp_interaction::util::line_at_location;

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
    interaction
        .initialize(InitializeSettings {
            workspace_folders,
            ..Default::default()
        })
        .unwrap();
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
            .definition(request_file_name, request_line, request_character)
            .expect_definition_response_from_root(
                response_file_name,
                response_line_start,
                response_character_start,
                response_line_end,
                response_character_end,
            )
            .unwrap();
    }
}

#[test]
fn definition_on_attr_of_pyi_assignment_goes_to_py() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            ..Default::default()
        })
        .unwrap();
    let file = "attributes_of_py/src_with_assignments.py";
    interaction.client.did_open(file);
    // Test annotated assignment (x: int = 100)
    interaction
        .client
        .definition(file, 7, 8)
        .expect_definition_response_from_root(
            "attributes_of_py/lib_with_assignments.py",
            7,
            4,
            7,
            5,
        )
        .unwrap();
    // Test regular assignment (y = "world")
    interaction
        .client
        .definition(file, 8, 8)
        .expect_definition_response_from_root(
            "attributes_of_py/lib_with_assignments.py",
            8,
            4,
            8,
            5,
        )
        .unwrap();
    interaction.shutdown().unwrap();
}

fn test_go_to_def_basic(root: &TempDir, workspace_folders: Option<Vec<(String, Url)>>) {
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("basic"));
    let file = "foo.py";
    interaction
        .initialize(InitializeSettings {
            workspace_folders: workspace_folders.clone(),
            ..Default::default()
        })
        .unwrap();
    interaction.client.did_open(file);
    interaction
        .client
        .definition(file, 5, 7)
        .expect_definition_response_from_root("bar.py", 0, 0, 0, 0)
        .unwrap();
    interaction
        .client
        .definition(file, 6, 16)
        .expect_definition_response_from_root("bar.py", 6, 6, 6, 9)
        .unwrap();
    interaction
        .client
        .definition(file, 8, 9)
        .expect_definition_response_from_root("bar.py", 7, 4, 7, 7)
        .unwrap();
    interaction
        .client
        .definition(file, 9, 7)
        .expect_definition_response_from_root("bar.py", 6, 6, 6, 9)
        .unwrap();
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
    interaction
        .initialize(InitializeSettings {
            ..Default::default()
        })
        .unwrap();
    interaction
        .client
        .did_open("imports_builtins/imports_builtins.py");
    interaction
        .client
        .definition("imports_builtins/imports_builtins.py", 7, 7)
        .expect_response_with(|response| {
            expect_definition_points_to_symbol(response.as_ref(), "typing", "List")
        })
        .unwrap();
}

#[test]
fn definition_on_attr_of_pyi_goes_to_py() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            ..Default::default()
        })
        .unwrap();
    let file = "attributes_of_py/src.py";
    interaction.client.did_open(file);
    interaction
        .client
        .definition(file, 7, 4)
        .expect_definition_response_from_root("attributes_of_py/lib.py", 7, 8, 7, 9)
        .unwrap();
    interaction.shutdown().unwrap();
}

#[test]
fn definition_in_builtins_without_interpreter_goes_to_stub() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(json!([{"pythonPath": "/fake/python/path"}]))),
            ..Default::default()
        })
        .unwrap();
    interaction.client.did_open("imports_builtins_no_config.py");
    interaction
        .client
        .definition("imports_builtins_no_config.py", 7, 7)
        .expect_response_with(|response| {
            expect_definition_points_to_symbol(response.as_ref(), "typing.pyi", "List =")
        })
        .unwrap();
}

#[test]
fn malformed_missing_position() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("basic"));
    interaction
        .initialize(InitializeSettings {
            ..Default::default()
        })
        .unwrap();
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
        activity_key: None,
    }));
    interaction
        .client
        .expect_response_error(
            RequestId::from(2),
            json!({
                "code": -32602,
                "message": "missing field `position`",
                "data": null,
            }),
        )
        .unwrap();
}

// we generally want to prefer py. but if it's missing in the py, we should prefer the pyi
#[test]
fn prefer_pyi_when_missing_in_py() {
    let root = get_test_files_root();
    let test_root = root.path().join("prefer_pyi_when_missing_in_py");
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_root);
    interaction
        .initialize(InitializeSettings {
            ..Default::default()
        })
        .unwrap();
    interaction.client.did_open("main.py");
    interaction
        .client
        .definition("main.py", 5, 18)
        .expect_definition_response_from_root("foo.pyi", 5, 4, 5, 7)
        .unwrap();
}

#[test]
fn goto_type_def_on_str_primitive_goes_to_builtins_stub() {
    let root = get_test_files_root();
    let pyrefly_typeshed_materialized = bundled_typeshed_path();
    let result_file = pyrefly_typeshed_materialized.join("builtins.pyi");
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            ..Default::default()
        })
        .unwrap();
    interaction.client.did_open("primitive_type_test.py");
    interaction
        .client
        .type_definition("primitive_type_test.py", 5, 0)
        .expect_response_with(|response| {
            expect_definition_points_to_symbol(response.as_ref(), "builtins.pyi", "class str")
        })
        .unwrap();

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
    interaction
        .initialize(InitializeSettings {
            ..Default::default()
        })
        .unwrap();
    interaction.client.did_open("primitive_type_test.py");

    interaction
        .client
        .type_definition("primitive_type_test.py", 6, 0)
        .expect_response_with(|response| {
            expect_definition_points_to_symbol(response.as_ref(), "builtins.pyi", "class int")
        })
        .unwrap();

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
    interaction
        .initialize(InitializeSettings {
            ..Default::default()
        })
        .unwrap();
    interaction.client.did_open("primitive_type_test.py");

    interaction
        .client
        .type_definition("primitive_type_test.py", 7, 0)
        .expect_response_with(|response| {
            expect_definition_points_to_symbol(response.as_ref(), "builtins.pyi", "class bool")
        })
        .unwrap();

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
    interaction
        .initialize(InitializeSettings {
            ..Default::default()
        })
        .unwrap();
    interaction.client.did_open("primitive_type_test.py");

    interaction
        .client
        .type_definition("primitive_type_test.py", 8, 0)
        .expect_response_with(|response| {
            expect_definition_points_to_symbol(response.as_ref(), "builtins.pyi", "class bytes")
        })
        .unwrap();

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
    interaction
        .initialize(InitializeSettings {
            ..Default::default()
        })
        .unwrap();
    interaction.client.did_open("custom_class_type_test.py");

    // Expect to go to the Foo class definition (line 6, columns 6-9)
    interaction
        .client
        .type_definition("custom_class_type_test.py", 8, 6)
        .expect_definition_response_from_root("custom_class_type_test.py", 6, 6, 6, 9)
        .unwrap();
}

#[test]
fn goto_type_def_on_list_of_primitives_shows_selector() {
    let root = get_test_files_root();
    let pyrefly_typeshed_materialized = bundled_typeshed_path();
    let builtins_file = pyrefly_typeshed_materialized.join("builtins.pyi");
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            ..Default::default()
        })
        .unwrap();
    interaction.client.did_open("primitive_type_test.py");

    interaction
        .client
        .type_definition("primitive_type_test.py", 9, 0)
        .expect_response_with(|response| match response {
            Some(GotoDefinitionResponse::Array(xs)) => {
                if xs.len() != 2 {
                    return false;
                }

                let mut has_int = false;
                let mut has_list = false;

                for x in xs {
                    if x.uri.to_file_path().unwrap() == builtins_file
                        && let Some(line) = line_at_location(&x)
                    {
                        has_int = has_int || line.contains("class int");
                        has_list = has_list || line.contains("class list");
                    }
                }

                has_int && has_list
            }
            _ => false,
        })
        .unwrap();
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

#[test]
fn goto_def_on_none_goes_to_builtins_stub() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            ..Default::default()
        })
        .unwrap();
    interaction.client.did_open("primitive_type_test.py");

    let check_none_type_location = |x: &Location| {
        let path = x.uri.to_file_path().unwrap();
        let file_name = path.file_name().and_then(|n| n.to_str());
        // NoneType can be in types.pyi (Python 3.10+) or __init__.pyi (older versions)
        file_name == Some("types.pyi") || file_name == Some("__init__.pyi")
    };

    // Test goto definition on None - should go to NoneType in types.pyi or builtins.pyi
    interaction
        .client
        .definition("primitive_type_test.py", 10, 4)
        .expect_response_with(|response| match response {
            Some(GotoDefinitionResponse::Scalar(x)) => check_none_type_location(&x),
            Some(GotoDefinitionResponse::Array(xs)) if !xs.is_empty() => {
                check_none_type_location(&xs[0])
            }
            _ => false,
        })
        .unwrap();
}

#[test]
fn test_goto_def_imported_submodule_with_alias() {
    let root = get_test_files_root();
    let root_path = root.path().join("nested_package_imports");
    test_go_to_def(
        root_path,
        None,
        "main.py",
        vec![
            // `from pkg import sub as sub` -> first `sub`
            (5, 16, "pkg/sub.py", 0, 0, 0, 0),
            // `from pkg import sub as sub` -> second `sub`
            (5, 23, "pkg/sub.py", 0, 0, 0, 0),
        ],
    );
}

#[test]
fn test_goto_def_submodule_access() {
    // Test go-to-definition on `autograd` in `torch.autograd.Function`
    // This tests submodule access with proper package structure (torch/__init__.py, torch/autograd/__init__.py)
    let root = get_test_files_root();
    let root_path = root.path().join("submodule_access");
    test_go_to_def(
        root_path,
        None,
        "main.py",
        vec![
            // `torch.autograd.Function` -> `autograd` (line 7, char 6)
            // Should navigate to torch/autograd/__init__.py
            (7, 6, "torch/autograd/__init__.py", 0, 0, 0, 0),
        ],
    );
}

#[test]
fn test_goto_def_deep_submodule_chain() {
    // Test go-to-definition on submodule components in `a.b.c.D`
    // This tests accessing nested packages with proper package structure
    let root = get_test_files_root();
    let root_path = root.path().join("deep_submodule_chain");
    test_go_to_def(
        root_path,
        None,
        "main.py",
        vec![
            // `a.b.c.D` -> `a` (line 7, char 0)
            // Should navigate to a/__init__.py
            (7, 0, "a/__init__.py", 0, 0, 0, 0),
            // `a.b.c.D` -> `b` (line 7, char 2)
            // Should navigate to a/b/__init__.py
            (7, 2, "a/b/__init__.py", 0, 0, 0, 0),
            // `a.b.c.D` -> `c` (line 7, char 4)
            // Should navigate to a/b/c.py
            (7, 4, "a/b/c.py", 0, 0, 0, 0),
            // `a.b.c.D` -> `D` (line 7, char 6)
            // Should navigate to class D definition in a/b/c.py
            (7, 6, "a/b/c.py", 6, 6, 6, 7),
        ],
    );
}

#[test]
fn test_goto_def_deep_submodule_chain_reexport() {
    // Test go-to-definition on submodule components in `a.b.c.D`
    // This tests the same pattern as deep_submodule_chain but with explicit re-exports
    // using `from . import x as x` pattern (similar to D91081404's implicit_submodule test).
    // Unlike deep_submodule_chain (which has empty __init__.py files), this should work
    // because the submodules are explicitly re-exported.
    let root = get_test_files_root();
    let root_path = root.path().join("deep_submodule_chain_reexport");
    test_go_to_def(
        root_path,
        None,
        "main.py",
        vec![
            // `a.b.c.D` -> `a` (line 7, char 0)
            // Should navigate to a/__init__.py
            (7, 0, "a/__init__.py", 0, 0, 0, 0),
            // `a.b.c.D` -> `b` (line 7, char 2)
            // Should navigate to a/b/__init__.py
            (7, 2, "a/b/__init__.py", 0, 0, 0, 0),
            // `a.b.c.D` -> `c` (line 7, char 4)
            // Should navigate to a/b/c.py
            (7, 4, "a/b/c.py", 0, 0, 0, 0),
            // `a.b.c.D` -> `D` (line 7, char 6)
            // Should navigate to class D definition in a/b/c.py
            (7, 6, "a/b/c.py", 6, 6, 6, 7),
        ],
    );
}

#[test]
fn test_goto_def_dunder_all_submodule() {
    // Test go-to-definition on a submodule name in __all__.
    // When __all__ = ["sub"] in pkg/__init__.py, clicking on "sub" should
    // navigate to pkg/sub.py.
    let root = get_test_files_root();
    let root_path = root.path().join("dunder_all_submodule");
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path);
    interaction
        .initialize(InitializeSettings {
            ..Default::default()
        })
        .unwrap();
    interaction.client.did_open("pkg/__init__.py");
    // Click on "sub" in __all__ = ["sub"] (line 5, char 12 is inside "sub")
    interaction
        .client
        .definition("pkg/__init__.py", 5, 12)
        .expect_definition_response_from_root("pkg/sub.py", 0, 0, 0, 0)
        .unwrap();
    interaction.shutdown().unwrap();
}
