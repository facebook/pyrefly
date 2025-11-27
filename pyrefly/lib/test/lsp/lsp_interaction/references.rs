/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::Url;
use serde_json::json;

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
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    let foo = root_path.join("foo.py");
    let bar = root_path.join("bar.py");
    let various_imports = root_path.join("various_imports.py");
    let with_synthetic_bindings = root_path.join("with_synthetic_bindings.py");

    interaction.client.did_open("foo.py");
    interaction.client.did_open("various_imports.py");
    interaction.client.did_open("with_synthetic_bindings.py");
    interaction.client.did_open("bar.py");

    interaction
        .client
        .references("bar.py", 10, 1, true)
        .expect_response(json!([
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
        ]))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_finds_references_outside_config_when_workspace_larger_than_config() {
    let root = get_test_files_root();
    let root_path = root.path().join("config_with_workspace_larger");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    let core = root_path.join("module_dir/core.py");
    let usage = root_path.join("module_dir/usage.py");

    interaction.client.did_open("module_dir/core.py");
    interaction.client.did_open("module_dir/usage.py");
    interaction.client.did_open("outside_usage.py");

    interaction
        .client
        .references("module_dir/core.py", 6, 7, true)
        .expect_response(json!([
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
        ]))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_references_workspace_smaller_than_config() {
    let root = get_test_files_root();
    let root_path = root.path().join("config_with_workspace_smaller");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    let core = root_path.join("core.py");
    let usage_in_config = root_path.join("usage_in_config.py");
    let usage_outside_workspace = root_path.join("subdir/usage_outside_workspace.py");

    interaction.client.did_open("core.py");
    interaction.client.did_open("usage_in_config.py");

    interaction
        .client
        .references("core.py", 6, 7, true)
        .expect_response(json!([
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
        ]))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_references_cross_file_no_config() {
    let root = get_test_files_root();
    let root_path = root.path().join("basic");
    let scope_uri = Url::from_file_path(&root_path).unwrap();
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            ..Default::default()
        })
        .unwrap();

    let bar = root_path.join("bar.py");
    let foo = root_path.join("foo.py");
    let foo_relative = root_path.join("foo_relative.py");

    interaction.client.did_open("bar.py");

    interaction
        .client
        .references("bar.py", 10, 1, true)
        .expect_response(json!([
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
        ]))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_references_cross_file_no_config_nested() {
    let root = get_test_files_root();
    let root_path = root.path().join("nested_test").to_path_buf();
    let scope_uri = Url::from_file_path(&root_path).unwrap();
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            ..Default::default()
        })
        .unwrap();

    let bar = root_path.join("models/bar.py");
    let foo = root_path.join("services/foo.py");
    let foo_relative = root_path.join("utils/foo_relative.py");

    interaction.client.did_open("models/bar.py");

    interaction
        .client
        .references("models/bar.py", 10, 1, true)
        .expect_response(json!([
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
        ]))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_references_cross_file_with_marker_file() {
    let root = get_test_files_root();
    let root_path = root.path().join("marker_file_no_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            ..Default::default()
        })
        .unwrap();

    let bar = root_path.join("bar.py");
    let foo = root_path.join("foo.py");

    interaction.client.did_open("bar.py");

    interaction
        .client
        .references("bar.py", 10, 1, true)
        .expect_response(json!([
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
        ]))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_references_for_definition_with_config() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    let foo = root_path.join("foo.py");
    let bar = root_path.join("bar.py");
    let various_imports = root_path.join("various_imports.py");
    let with_synthetic_bindings = root_path.join("with_synthetic_bindings.py");

    interaction.client.did_open("foo.py");
    interaction.client.did_open("various_imports.py");
    interaction.client.did_open("with_synthetic_bindings.py");
    interaction.client.did_open("bar.py");

    interaction
        .client
        .references("bar.py", 6, 7, true)
        .expect_response(json!([
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
        ]))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_references_for_import_with_config() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    let foo = root_path.join("foo.py");
    let bar = root_path.join("bar.py");
    let various_imports = root_path.join("various_imports.py");
    let with_synthetic_bindings = root_path.join("with_synthetic_bindings.py");

    interaction.client.did_open("foo.py");
    interaction.client.did_open("various_imports.py");
    interaction.client.did_open("with_synthetic_bindings.py");
    interaction.client.did_open("bar.py");

    interaction
        .client
        .references("foo.py", 6, 17, true)
        .expect_response(json!([
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
        ]))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_references_for_aliased_import_with_config() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    let various_imports = root_path.join("various_imports.py");

    interaction.client.did_open("foo.py");
    interaction.client.did_open("various_imports.py");
    interaction.client.did_open("with_synthetic_bindings.py");
    interaction.client.did_open("bar.py");

    interaction
        .client
        .references("various_imports.py", 7, 0, true)
        .expect_response(json!([
            {
                "range": {"start":{"line":5,"character":23},"end":{"line":5,"character":24}},
                "uri": Url::from_file_path(various_imports.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":7,"character":0},"end":{"line":7,"character":1}},
                "uri": Url::from_file_path(various_imports.clone()).unwrap().to_string()
            },
        ]))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_references_after_file_modification_with_config() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    let foo = root_path.join("foo.py");
    let bar = root_path.join("bar.py");
    let various_imports = root_path.join("various_imports.py");
    let with_synthetic_bindings = root_path.join("with_synthetic_bindings.py");

    interaction.client.did_open("foo.py");
    interaction.client.did_open("various_imports.py");
    interaction.client.did_open("with_synthetic_bindings.py");
    interaction.client.did_open("bar.py");

    let modified_contents = format!("\n\n{}", std::fs::read_to_string(bar.clone()).unwrap());
    interaction.client.did_change("bar.py", &modified_contents);

    interaction
        .client
        .references("foo.py", 6, 17, true)
        .expect_response(json!([
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
        ]))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_references_after_file_modification_with_line_offset_with_config() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    let bar = root_path.join("bar.py");

    interaction.client.did_open("foo.py");
    interaction.client.did_open("various_imports.py");
    interaction.client.did_open("with_synthetic_bindings.py");
    interaction.client.did_open("bar.py");

    let modified_contents = format!("\n\n{}", std::fs::read_to_string(bar.clone()).unwrap());
    interaction.client.did_change("bar.py", &modified_contents);

    interaction
        .client
        .references("bar.py", 8, 7, true)
        .expect_response(json!([
            {
                "range": {"start":{"line":8,"character":6},"end":{"character":9,"line":8}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":12,"character":0},"end":{"character":3,"line":12}},
                "uri": Url::from_file_path(bar.clone()).unwrap().to_string()
            },
        ]))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_references_cross_file_method_inheritance() {
    let root = get_test_files_root();
    let root_path = root.path().join("references_cross_file_method_inheritance");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            ..Default::default()
        })
        .unwrap();

    let base_py = root_path.join("base.py");
    let child_py = root_path.join("child.py");
    let child_of_child_py = root_path.join("child_of_child.py");
    let usage_py = root_path.join("usage.py");

    interaction.client.did_open("base.py");

    // Find references for Base.method (line 7, character 8 in base.py)
    interaction
        .client
        .references("base.py", 7, 8, true)
        .expect_response(json!([
            {
                "range": {"start":{"line":9,"character":8},"end":{"line":9,"character":14}},
                "uri": Url::from_file_path(child_py.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":14,"character":6},"end":{"line":14,"character":12}},
                "uri": Url::from_file_path(child_py.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":9,"character":8},"end":{"line":9,"character":14}},
                "uri": Url::from_file_path(child_of_child_py.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":14,"character":6},"end":{"line":14,"character":12}},
                "uri": Url::from_file_path(child_of_child_py.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":8,"character":2},"end":{"line":8,"character":8}},
                "uri": Url::from_file_path(usage_py.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":7,"character":8},"end":{"line":7,"character":14}},
                "uri": Url::from_file_path(base_py.clone()).unwrap().to_string()
            },
        ]))
        .unwrap();

    interaction.shutdown().unwrap();
}

// Test for __init__ priority
// When only __init__ is overridden (not __new__), __init__ handles the constructor call
#[test]
fn test_references_for_init_priority() {
    let root = get_test_files_root();
    let root_path = root
        .path()
        .join("constructor_priority_references/init_priority");
    let scope_uri = Url::from_file_path(&root_path).unwrap();
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            ..Default::default()
        })
        .unwrap();

    let person_py = root_path.join("person.py");
    let usage_py = root_path.join("usage.py");

    interaction.client.did_open("person.py");
    interaction.client.did_open("usage.py");

    // Find references for Person.__init__
    interaction
        .client
        .references("person.py", 9, 12, true)
        .expect_response(json!([
            {
                "range": {"start":{"line":9,"character":8},"end":{"line":9,"character":16}},
                "uri": Url::from_file_path(person_py.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":7,"character":5},"end":{"line":7,"character":11}},
                "uri": Url::from_file_path(usage_py.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":8,"character":5},"end":{"line":8,"character":11}},
                "uri": Url::from_file_path(usage_py.clone()).unwrap().to_string()
            },
        ]))
        .unwrap();

    interaction.shutdown().unwrap();
}

// Test for __new__ priority
// When __new__ is overridden, it takes priority over __init__ for constructor calls
#[test]
fn test_references_for_new_priority() {
    let root = get_test_files_root();
    let root_path = root
        .path()
        .join("constructor_priority_references/new_priority");
    let scope_uri = Url::from_file_path(&root_path).unwrap();
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            ..Default::default()
        })
        .unwrap();

    let singleton_py = root_path.join("singleton.py");
    let usage_py = root_path.join("usage.py");

    interaction.client.did_open("singleton.py");

    // Find references for Singleton.__new__ (line 8, character 12 in singleton.py)
    interaction
        .client
        .references("singleton.py", 7, 12, true)
        .expect_response(json!([
            {
                "range": {"start":{"line":7,"character":5},"end":{"line":7,"character":14}},
                "uri": Url::from_file_path(usage_py.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":8,"character":5},"end":{"line":8,"character":14}},
                "uri": Url::from_file_path(usage_py.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":7,"character":8},"end":{"line":7,"character":15}},
                "uri": Url::from_file_path(singleton_py.clone()).unwrap().to_string()
            },
        ]))
        .unwrap();

    interaction.shutdown().unwrap();
}

// Test for metaclass __call__ priority
// When a metaclass defines __call__, it takes priority over __new__ and __init__
#[test]
fn test_references_for_metaclass_call_priority() {
    let root = get_test_files_root();
    let root_path = root
        .path()
        .join("constructor_priority_references/metaclass_priority");
    let scope_uri = Url::from_file_path(&root_path).unwrap();
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            ..Default::default()
        })
        .unwrap();

    let singleton_meta_py = root_path.join("singleton_meta.py");
    let usage_py = root_path.join("usage.py");

    interaction.client.did_open("singleton_meta.py");
    interaction.client.did_open("usage.py");

    // Find references for SingletonMeta.__call__
    interaction
        .client
        .references("singleton_meta.py", 7, 12, true)
        .expect_response(json!([
            {
                "range": {"start":{"line":7,"character":8},"end":{"line":7,"character":16}},
                "uri": Url::from_file_path(singleton_meta_py.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":7,"character":5},"end":{"line":7,"character":14}},
                "uri": Url::from_file_path(usage_py.clone()).unwrap().to_string()
            },
            {
                "range": {"start":{"line":8,"character":5},"end":{"line":8,"character":14}},
                "uri": Url::from_file_path(usage_py.clone()).unwrap().to_string()
            },
        ]))
        .unwrap();

    interaction.shutdown().unwrap();
}
