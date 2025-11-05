/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_server::RequestId;
use lsp_server::Response;
use lsp_types::Url;

use crate::commands::lsp::IndexingMode;
use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_references_for_usage_with_config() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction.initialize(InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
        configuration: Some(None),
        ..Default::default()
    });

    let foo = root_path.join("foo.py");
    let bar = root_path.join("bar.py");
    let various_imports = root_path.join("various_imports.py");
    let with_synthetic_bindings = root_path.join("with_synthetic_bindings.py");

    interaction.server.did_open("foo.py");
    interaction.server.did_open("various_imports.py");
    interaction.server.did_open("with_synthetic_bindings.py");
    interaction.server.did_open("bar.py");

    interaction.server.references("bar.py", 10, 1, true);

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!([
            {
                "range": {"start":{"line":6,"character":6},"end":{"character":9,"line":6}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":10,"character":0},"end":{"character":3,"line":10}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":6,"character":16},"end":{"line":6,"character":19}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":8,"character":0},"end":{"line":8,"character":3}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":9,"character":4},"end":{"line":9,"character":7}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":5,"character":16},"end":{"line":5,"character":19}},
                "uri": Url::from_file_path(various_imports.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":5,"character":26},"end":{"line":5,"character":29}},
                "uri": Url::from_file_path(various_imports.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":5,"character":16},"end":{"character":19,"line":5}},
                "uri": Url::from_file_path(with_synthetic_bindings.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":10,"character":4},"end":{"character":7,"line":10}},
                "uri": Url::from_file_path(with_synthetic_bindings.clone()).unwrap().to_string()
            },
        ])),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_finds_references_outside_config_when_workspace_larger_than_config() {
    let root = get_test_files_root();
    let root_path = root.path().join("config_with_workspace_larger");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction.initialize(InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
        configuration: Some(None),
        ..Default::default()
    });

    let core = root_path.join("module_dir/core.py");
    let usage = root_path.join("module_dir/usage.py");

    interaction.server.did_open("module_dir/core.py");
    interaction.server.did_open("module_dir/usage.py");
    interaction.server.did_open("outside_usage.py");

    interaction
        .server
        .references("module_dir/core.py", 6, 7, true);

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!([
            {
                "range": {"start":{"line":6,"character":6},"end":{"character":12,"line":6}},
                "uri": Url::from_file_path(core.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":5,"character":17},"end":{"character":23,"line":5}},
                "uri": Url::from_file_path(usage.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":7,"character":0},"end":{"character":6,"line":7}},
                "uri": Url::from_file_path(usage.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":8,"character":0},"end":{"character":6,"line":8}},
                "uri": Url::from_file_path(usage.clone()).unwrap().to_string()
            },
        ])),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_references_workspace_smaller_than_config() {
    let root = get_test_files_root();
    let root_path = root.path().join("config_with_workspace_smaller");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(root_path.clone());
    interaction.initialize(InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
        configuration: Some(None),
        ..Default::default()
    });

    let core = root_path.join("core.py");
    let usage_in_config = root_path.join("usage_in_config.py");
    let usage_outside_workspace = root_path.join("subdir/usage_outside_workspace.py");

    interaction.server.did_open("core.py");
    interaction.server.did_open("usage_in_config.py");

    interaction.server.references("core.py", 6, 7, true);

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!([
            {
                "range": {"start":{"line":5,"character":17},"end":{"character":22,"line":5}},
                "uri": Url::from_file_path(usage_outside_workspace.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":7,"character":0},"end":{"character":5,"line":7}},
                "uri": Url::from_file_path(usage_outside_workspace.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":8,"character":0},"end":{"character":5,"line":8}},
                "uri": Url::from_file_path(usage_outside_workspace.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":6,"character":6},"end":{"character":11,"line":6}},
                "uri": Url::from_file_path(core.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":5,"character":17},"end":{"character":22,"line":5}},
                "uri": Url::from_file_path(usage_in_config.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":7,"character":0},"end":{"character":5,"line":7}},
                "uri": Url::from_file_path(usage_in_config.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":8,"character":0},"end":{"character":5,"line":8}},
                "uri": Url::from_file_path(usage_in_config.clone()).unwrap().to_string()
            },
        ])),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_references_cross_file_no_config() {
    let root = get_test_files_root();
    let root_path = root.path().join("basic");
    let scope_uri = Url::from_file_path(&root_path).unwrap();
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(root_path.clone());
    interaction.initialize(InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
        ..Default::default()
    });

    let bar = root_path.join("bar.py");
    let foo = root_path.join("foo.py");
    let foo_relative = root_path.join("foo_relative.py");

    interaction.server.did_open("bar.py");

    interaction.server.references("bar.py", 10, 1, true);

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!([
            {
                "range": {"start":{"line":6,"character":16},"end":{"character":19,"line":6}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range":{"end":{"character":3,"line":8},"start":{"character":0,"line":8}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range":{"end":{"character":7,"line":9},"start":{"character":4,"line":9}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":6,"character":17},"end":{"character":20,"line":6}},
                "uri": Url::from_file_path(foo_relative.clone()).unwrap().to_string()
            },
            {
                "range":{"end":{"character":3,"line":8},"start":{"character":0,"line":8}},
                "uri": Url::from_file_path(foo_relative.clone()).unwrap().to_string()
            },
            {
                "range":{"end":{"character":7,"line":9},"start":{"character":4,"line":9}},
                "uri": Url::from_file_path(foo_relative.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":6,"character":6},"end":{"character":9,"line":6}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":10,"character":0},"end":{"character":3,"line":10}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
        ])),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_references_cross_file_no_config_nested() {
    let root = get_test_files_root();
    let root_path = root.path().join("nested_test").to_path_buf();
    let scope_uri = Url::from_file_path(&root_path).unwrap();
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(root_path.clone());
    interaction.initialize(InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
        ..Default::default()
    });

    let bar = root_path.join("models/bar.py");
    let foo = root_path.join("services/foo.py");
    let foo_relative = root_path.join("utils/foo_relative.py");

    interaction.server.did_open("models/bar.py");

    interaction.server.references("models/bar.py", 10, 1, true);

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!([
            {
                "range": {"start":{"line":6,"character":23},"end":{"character":26,"line":6}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range":{"end":{"character":3,"line":8},"start":{"character":0,"line":8}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range":{"end":{"character":14,"line":9},"start":{"character":11,"line":9}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":6,"character":25},"end":{"character":28,"line":6}},
                "uri": Url::from_file_path(foo_relative.clone()).unwrap().to_string()
            },
            {
                "range":{"end":{"character":3,"line":8},"start":{"character":0,"line":8}},
                "uri": Url::from_file_path(foo_relative.clone()).unwrap().to_string()
            },
            {
                "range":{"end":{"character":7,"line":9},"start":{"character":4,"line":9}},
                "uri": Url::from_file_path(foo_relative.clone()).unwrap().to_string()
            },            {
                "range": {"start":{"line":6,"character":6},"end":{"character":9,"line":6}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":10,"character":0},"end":{"character":3,"line":10}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
        ])),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_references_cross_file_with_marker_file() {
    let root = get_test_files_root();
    let root_path = root.path().join("marker_file_no_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(root_path.clone());
    interaction.initialize(InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
        ..Default::default()
    });

    let bar = root_path.join("bar.py");
    let foo = root_path.join("foo.py");

    interaction.server.did_open("bar.py");

    interaction.server.references("bar.py", 10, 1, true);

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!([
            {
                "range": {"start":{"line":6,"character":16},"end":{"character":19,"line":6}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range":{"end":{"character":3,"line":8},"start":{"character":0,"line":8}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range":{"end":{"character":7,"line":9},"start":{"character":4,"line":9}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range":{"end":{"character":9,"line":6},"start":{"character":6,"line":6}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":10,"character":0},"end":{"character":3,"line":10}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
        ])),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_references_for_definition_with_config() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction.initialize(InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
        configuration: Some(None),
        ..Default::default()
    });

    let foo = root_path.join("foo.py");
    let bar = root_path.join("bar.py");
    let various_imports = root_path.join("various_imports.py");
    let with_synthetic_bindings = root_path.join("with_synthetic_bindings.py");

    interaction.server.did_open("foo.py");
    interaction.server.did_open("various_imports.py");
    interaction.server.did_open("with_synthetic_bindings.py");
    interaction.server.did_open("bar.py");

    interaction.server.references("bar.py", 6, 7, true);

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!([
            {
                "range": {"start":{"line":6,"character":6},"end":{"character":9,"line":6}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":10,"character":0},"end":{"character":3,"line":10}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":6,"character":16},"end":{"line":6, "character":19}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":8,"character":0},"end":{"line":8,"character":3}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":9,"character":4},"end":{"line":9,"character":7}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":5,"character":16},"end":{"line":5,"character":19}},
                "uri": Url::from_file_path(various_imports.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":5,"character":26},"end":{"line":5,"character":29}},
                "uri": Url::from_file_path(various_imports.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":5,"character":16},"end":{"character":19,"line":5}},
                "uri": Url::from_file_path(with_synthetic_bindings.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":10,"character":4},"end":{"character":7,"line":10}},
                "uri": Url::from_file_path(with_synthetic_bindings.clone()).unwrap().to_string()
            },
        ])),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_references_for_import_with_config() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction.initialize(InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
        configuration: Some(None),
        ..Default::default()
    });

    let foo = root_path.join("foo.py");
    let bar = root_path.join("bar.py");
    let various_imports = root_path.join("various_imports.py");
    let with_synthetic_bindings = root_path.join("with_synthetic_bindings.py");

    interaction.server.did_open("foo.py");
    interaction.server.did_open("various_imports.py");
    interaction.server.did_open("with_synthetic_bindings.py");
    interaction.server.did_open("bar.py");

    interaction.server.references("foo.py", 6, 17, true);

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!([
            {
                "range": {"start":{"line":6,"character":6},"end":{"character":9,"line":6}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":10,"character":0},"end":{"character":3,"line":10}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":6,"character":16},"end":{"line":6, "character":19}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":8,"character":0},"end":{"line":8,"character":3}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":9,"character":4},"end":{"line":9,"character":7}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":5,"character":16},"end":{"line":5,"character":19}},
                "uri": Url::from_file_path(various_imports.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":5,"character":26},"end":{"line":5,"character":29}},
                "uri": Url::from_file_path(various_imports.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":5,"character":16},"end":{"character":19,"line":5}},
                "uri": Url::from_file_path(with_synthetic_bindings.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":10,"character":4},"end":{"character":7,"line":10}},
                "uri": Url::from_file_path(with_synthetic_bindings.clone()).unwrap().to_string()
            },
        ])),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_references_for_aliased_import_with_config() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction.initialize(InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
        configuration: Some(None),
        ..Default::default()
    });

    let various_imports = root_path.join("various_imports.py");

    interaction.server.did_open("foo.py");
    interaction.server.did_open("various_imports.py");
    interaction.server.did_open("with_synthetic_bindings.py");
    interaction.server.did_open("bar.py");

    interaction
        .server
        .references("various_imports.py", 7, 0, true);

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!([
            {
                "range": {"start":{"line":5,"character":23},"end":{"line":5,"character":24}},
                "uri": Url::from_file_path(various_imports.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":7,"character":0},"end":{"line":7,"character":1}},
                "uri": Url::from_file_path(various_imports.clone()).unwrap().to_string()
            },
        ])),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_references_after_file_modification_with_config() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction.initialize(InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
        configuration: Some(None),
        ..Default::default()
    });

    let foo = root_path.join("foo.py");
    let bar = root_path.join("bar.py");
    let various_imports = root_path.join("various_imports.py");
    let with_synthetic_bindings = root_path.join("with_synthetic_bindings.py");

    interaction.server.did_open("foo.py");
    interaction.server.did_open("various_imports.py");
    interaction.server.did_open("with_synthetic_bindings.py");
    interaction.server.did_open("bar.py");

    let modified_contents = format!("\n\n{}", std::fs::read_to_string(bar.clone()).unwrap());
    interaction.server.did_change("bar.py", &modified_contents);

    interaction.server.references("foo.py", 6, 17, true);

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!([
            {
                "range": {"start":{"line":6,"character":6},"end":{"character":9,"line":6}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":10,"character":0},"end":{"character":3,"line":10}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":6,"character":16},"end":{"line":6, "character":19}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":8,"character":0},"end":{"line":8,"character":3}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":9,"character":4},"end":{"line":9,"character":7}},
                "uri": Url::from_file_path(foo.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":5,"character":16},"end":{"line":5,"character":19}},
                "uri": Url::from_file_path(various_imports.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":5,"character":26},"end":{"line":5,"character":29}},
                "uri": Url::from_file_path(various_imports.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":5,"character":16},"end":{"character":19,"line":5}},
                "uri": Url::from_file_path(with_synthetic_bindings.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":10,"character":4},"end":{"character":7,"line":10}},
                "uri": Url::from_file_path(with_synthetic_bindings.clone()).unwrap().to_string()
            },
        ])),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_references_after_file_modification_with_line_offset_with_config() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction.initialize(InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
        configuration: Some(None),
        ..Default::default()
    });

    let bar = root_path.join("bar.py");

    interaction.server.did_open("foo.py");
    interaction.server.did_open("various_imports.py");
    interaction.server.did_open("with_synthetic_bindings.py");
    interaction.server.did_open("bar.py");

    let modified_contents = format!("\n\n{}", std::fs::read_to_string(bar.clone()).unwrap());
    interaction.server.did_change("bar.py", &modified_contents);

    interaction.server.references("bar.py", 8, 7, true);

    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!([
            {
                "range": {"start":{"line":8,"character":6},"end":{"character":9,"line":8}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":12,"character":0},"end":{"character":3,"line":12}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
        ])),
        error: None,
    });

    interaction.shutdown();
}
