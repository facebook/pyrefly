/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::bundled_typeshed_path;
use crate::test::lsp::lsp_interaction::util::expect_definition_points_to_symbol;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn builtins_import_goes_to_typeshed() {
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
    interaction.client.did_open("imports_builtins_no_config.py");
    interaction
        .client
        .type_definition("imports_builtins_no_config.py", 7, 7)
        .expect_response_with(|response| {
            expect_definition_points_to_symbol(response.as_ref(), "builtins.pyi", "class list")
        })
        .unwrap();
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
