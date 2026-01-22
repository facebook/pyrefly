/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::FoldingRange;
use lsp_types::FoldingRangeKind;
use lsp_types::Url;
use serde_json::json;
use tempfile::TempDir;

use super::object_model::InitializeSettings;
use super::object_model::LspInteraction;

fn get_test_files_root() -> TempDir {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let test_dir = temp_dir.path();

    // Create a simple test file with comment sections
    std::fs::write(
        test_dir.join("test.py"),
        r#"# Section 1 ----

def foo():
    pass

# Section 2 ----

class MyClass:
    pass
"#,
    )
    .expect("Failed to write test file");

    temp_dir
}

#[test]
fn test_folding_ranges_comment_sections_disabled_by_default() {
    let test_files_root = get_test_files_root();
    let root_path = test_files_root.path().to_path_buf();
    let scope_uri = Url::from_file_path(&root_path).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());

    // Initialize without setting commentFoldingRanges (defaults to false)
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
            initialization_options: None,
            configuration: Some(None),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_open("test.py");

    // Request folding ranges
    interaction
        .client
        .folding_range("test.py")
        .expect_response_with(|ranges: Option<Vec<FoldingRange>>| {
            let ranges = ranges.unwrap_or_default();
            // Should NOT contain any Region kind ranges (comment sections)
            // because commentFoldingRanges defaults to false
            !ranges
                .iter()
                .any(|r| r.kind == Some(FoldingRangeKind::Region))
        })
        .expect("Expected no Region folding ranges when commentFoldingRanges is disabled");

    interaction.shutdown().expect("Failed to shutdown");
}

#[test]
fn test_folding_ranges_comment_sections_enabled() {
    let test_files_root = get_test_files_root();
    let root_path = test_files_root.path().to_path_buf();
    let scope_uri = Url::from_file_path(&root_path).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());

    // Initialize with commentFoldingRanges set to true
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
            initialization_options: Some(json!({
                "commentFoldingRanges": true
            })),
            configuration: Some(None),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_open("test.py");

    // Request folding ranges
    interaction
        .client
        .folding_range("test.py")
        .expect_response_with(|ranges: Option<Vec<FoldingRange>>| {
            let ranges = ranges.unwrap_or_default();
            // Should contain Region kind ranges (comment sections)
            // because commentFoldingRanges is set to true
            ranges
                .iter()
                .any(|r| r.kind == Some(FoldingRangeKind::Region))
        })
        .expect("Expected Region folding ranges when commentFoldingRanges is enabled");

    interaction.shutdown().expect("Failed to shutdown");
}

#[test]
fn test_folding_ranges_comment_sections_explicitly_disabled() {
    let test_files_root = get_test_files_root();
    let root_path = test_files_root.path().to_path_buf();
    let scope_uri = Url::from_file_path(&root_path).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());

    // Initialize with commentFoldingRanges explicitly set to false
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
            initialization_options: Some(json!({
                "commentFoldingRanges": false
            })),
            configuration: Some(None),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_open("test.py");

    // Request folding ranges
    interaction
        .client
        .folding_range("test.py")
        .expect_response_with(|ranges: Option<Vec<FoldingRange>>| {
            let ranges = ranges.unwrap_or_default();
            // Should NOT contain any Region kind ranges (comment sections)
            !ranges
                .iter()
                .any(|r| r.kind == Some(FoldingRangeKind::Region))
        })
        .expect("Expected no Region folding ranges when commentFoldingRanges is explicitly false");

    interaction.shutdown().expect("Failed to shutdown");
}
