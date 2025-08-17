/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::env;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn builtins_import_goes_to_typeshed() {
    let root = get_test_files_root();
    let pyrefly_typeshed_materialized = env::temp_dir().join("pyrefly_bundled_typeshed");
    let result_file = pyrefly_typeshed_materialized.join("typing.pyi");
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        ..Default::default()
    });
    interaction.server.did_open("imports_builtins.py");
    interaction
        .server
        .type_definition("imports_builtins.py", 7, 7);
    interaction.client.expect_definition_response_absolute(
        result_file.to_string_lossy().to_string(),
        425,
        0,
        425,
        4,
    );
    assert!(
        pyrefly_typeshed_materialized.join("pyrefly.toml").exists(),
        "Expected pyrefly.toml to exist at {:?}",
        pyrefly_typeshed_materialized.to_str()
    );
    assert!(
        result_file.exists(),
        "Expected pyrefly.toml to exist at {result_file:?}",
    );
}
