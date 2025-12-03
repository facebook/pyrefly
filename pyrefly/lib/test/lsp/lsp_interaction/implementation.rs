/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::Url;

use crate::commands::lsp::IndexingMode;
use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn implementation_on_definition_test() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    let test_root = root.path().join("references_cross_file_method_inheritance");
    interaction.set_root(test_root.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![(
                "test".to_owned(),
                Url::from_file_path(test_root.clone()).unwrap(),
            )]),
            ..Default::default()
        })
        .unwrap();

    // Open the base file
    interaction.client.did_open("base.py");

    // Request implementations for Base.method
    // Expect two implementations: Child.method and ChildOfChild.method
    // Note: Order may vary between runs due to HashMap ordering
    interaction
        .client
        .implementation("base.py", 7, 8)
        .expect_implementation_response_from_root(vec![
            ("child.py", 9, 8, 9, 14),          // Child.method
            ("child_of_child.py", 9, 8, 9, 14), // ChildOfChild.method
        ])
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn implementation_on_call_test() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new_with_indexing_mode(IndexingMode::LazyBlocking);
    let test_root = root.path().join("references_cross_file_method_inheritance");
    interaction.set_root(test_root.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![(
                "test".to_owned(),
                Url::from_file_path(test_root.clone()).unwrap(),
            )]),
            ..Default::default()
        })
        .unwrap();

    // Open the usage file
    interaction.client.did_open("usage.py");

    // Request implementations for b.method() call on line 9
    // Expect two implementations: Child.method and ChildOfChild.method
    // Note: Order may vary between runs due to HashMap ordering
    interaction
        .client
        .implementation("usage.py", 8, 2)
        .expect_implementation_response_from_root(vec![
            ("child.py", 9, 8, 9, 14),          // Child.method
            ("child_of_child.py", 9, 8, 9, 14), // ChildOfChild.method
        ])
        .unwrap();

    interaction.shutdown().unwrap();
}
