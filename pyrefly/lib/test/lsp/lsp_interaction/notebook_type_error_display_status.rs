/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::Url;

use crate::object_model::InitializeSettings;
use crate::object_model::LspInteraction;
use crate::util::get_test_files_root;

/// Verifies that typeErrorDisplayStatus resolves the notebook file's config
/// for notebook cells, rather than defaulting to NoConfigFile.
#[test]
fn test_notebook_type_error_display_status() {
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
    interaction.open_notebook("notebook.ipynb", vec!["x = 1", "y = x"]);

    // The notebook is inside tests_requiring_config/ which has a pyrefly.toml,
    // so the status should reflect that a config file was found.
    let status = interaction.type_error_display_status_cell("notebook.ipynb", "cell1");
    assert_eq!(status, "enabled-in-config-file");

    let status = interaction.type_error_display_status_cell("notebook.ipynb", "cell2");
    assert_eq!(status, "enabled-in-config-file");

    interaction.shutdown().unwrap();
}
