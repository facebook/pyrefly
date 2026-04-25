/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::Url;
use pyrefly::commands::lsp::IndexingMode;

use crate::object_model::InitializeSettings;
use crate::object_model::LspInteraction;
use crate::util::get_test_files_root;

#[test]
fn test_notebook_references() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();
    // Two cells: cell1 defines x, cell2 uses x
    interaction.open_notebook("notebook.ipynb", vec!["x = 1", "y = x"]);

    let cell1_uri = interaction.cell_uri("notebook.ipynb", "cell1");
    let cell2_uri = interaction.cell_uri("notebook.ipynb", "cell2");

    // Find all references to "x" (including declaration) from cell1
    interaction
        .references_cell("notebook.ipynb", "cell1", 0, 0, true)
        .expect_response_with(|response| {
            let Some(locations) = response else {
                return false;
            };
            // Should find the declaration in cell1 and the usage in cell2
            let has_declaration = locations.iter().any(|loc| {
                loc.uri == cell1_uri
                    && loc.range.start.line == 0
                    && loc.range.start.character == 0
                    && loc.range.end.character == 1
            });
            let has_usage = locations.iter().any(|loc| {
                loc.uri == cell2_uri
                    && loc.range.start.line == 0
                    && loc.range.start.character == 4
                    && loc.range.end.character == 5
            });
            has_declaration && has_usage
        })
        .unwrap();
    interaction.shutdown().unwrap();
}

/// Notebooks (.ipynb) are indexed from disk just like .py files, so references
/// inside a notebook are found even when the notebook is not open.
/// This test verifies that an on-disk notebook contributes references.
#[test]
fn test_references_from_file_includes_indexed_notebook() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
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

    // Open only the .py files — do NOT open the notebook
    interaction.client.did_open("bar.py");
    interaction.client.did_open("foo.py");
    interaction.client.did_open("various_imports.py");
    interaction.client.did_open("with_synthetic_bindings.py");

    // Find references to "Bar" from bar.py line 10.
    // Should include results from .py files (9) plus the indexed notebook (2),
    // for 11 total. The notebook references use file:// URIs since the notebook
    // is not open (no vscode-notebook-cell remapping).
    interaction
        .client
        .references("bar.py", 10, 1, true)
        .expect_response_with(|response| {
            let Some(locations) = response else {
                return false;
            };
            assert_eq!(
                locations.len(),
                11,
                "Expected 11 references (9 .py + 2 from indexed notebook)"
            );
            let notebook_refs: Vec<_> = locations
                .iter()
                .filter(|loc| loc.uri.path().ends_with("notebook_refs.ipynb"))
                .collect();
            assert_eq!(
                notebook_refs.len(),
                2,
                "Expected 2 references from the indexed notebook"
            );
            true
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

/// Tests that find-references from a .py file includes results from an open notebook
/// that references the same symbol.
#[test]
fn test_references_from_file_includes_open_notebook() {
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

    // Open the .py files
    interaction.client.did_open("bar.py");
    interaction.client.did_open("foo.py");
    interaction.client.did_open("various_imports.py");
    interaction.client.did_open("with_synthetic_bindings.py");

    // Open a notebook that references Bar from bar.py
    interaction.open_notebook("notebook.ipynb", vec!["from bar import Bar", "Bar()"]);

    let cell1_uri = interaction.cell_uri("notebook.ipynb", "cell1");
    let cell2_uri = interaction.cell_uri("notebook.ipynb", "cell2");

    // Find references to "Bar" from bar.py line 10.
    // Should include results from both .py files and the open notebook cells.
    interaction
        .client
        .references("bar.py", 10, 1, true)
        .expect_response_with(|response| {
            let Some(locations) = response else {
                return false;
            };
            // Should have 11 total references: 9 from .py files + 2 from notebook cells
            assert_eq!(
                locations.len(),
                11,
                "Expected 11 references (9 .py + 2 notebook)"
            );
            // Verify the notebook cell references are present
            let has_cell1_import = locations.iter().any(|loc| {
                loc.uri == cell1_uri
                    && loc.range.start.line == 0
                    && loc.range.start.character == 16
                    && loc.range.end.character == 19
            });
            let has_cell2_usage = locations.iter().any(|loc| {
                loc.uri == cell2_uri
                    && loc.range.start.line == 0
                    && loc.range.start.character == 0
                    && loc.range.end.character == 3
            });
            assert!(
                has_cell1_import,
                "Missing Bar import reference in notebook cell1"
            );
            assert!(
                has_cell2_usage,
                "Missing Bar usage reference in notebook cell2"
            );
            true
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
