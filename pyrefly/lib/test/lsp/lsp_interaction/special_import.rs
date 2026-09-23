/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Go-to-definition through the Configerator special import functions
//! (`import_thrift`/`import_python`) as they appear in `.cinc` files. The fixture's
//! `pyrefly.toml` declares `extra-file-extensions`, so `.cinc` and `.thrift` resolve
//! as modules; without that, none of these names would bind at all.

use pyrefly_lsp_test::object_model::InitializeSettings;
use pyrefly_lsp_test::object_model::LspInteraction;
use tempfile::TempDir;

use crate::test::lsp::lsp_interaction::util::get_test_files_root;

/// The `TempDir` is returned alongside the interaction because it owns the fixture
/// files, which must outlive the language server.
fn open_cinc(file: &'static str) -> (TempDir, LspInteraction) {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().join("special_import_cinc"));
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();
    interaction.client.did_open(file);
    (root, interaction)
}

/// `import_thrift(<module>, "*")` behaves like `from <module> import *`. Two of them in
/// one file each contribute their own names: wildcard names are keyed on
/// `(name, <range of the import_thrift call>)`, so the second call must not displace the
/// first.
#[test]
fn go_to_def_through_wildcard_special_import() {
    let (_root, interaction) = open_cinc("wildcard.cinc");
    // Two names from the first wildcard import.
    interaction
        .client
        .definition("wildcard.cinc", 8, 6)
        .expect_definition_response_from_root("service/types.thrift.pyi", 5, 6, 5, 14)
        .unwrap();
    interaction
        .client
        .definition("wildcard.cinc", 9, 6)
        .expect_definition_response_from_root("service/types.thrift.pyi", 8, 6, 8, 17)
        .unwrap();
    // A name from the second wildcard import, resolving to the other module.
    interaction
        .client
        .definition("wildcard.cinc", 10, 6)
        .expect_definition_response_from_root("service/other.thrift.pyi", 5, 6, 5, 17)
        .unwrap();
    interaction.shutdown().unwrap();
}

/// A bare string second argument is a module alias, so the name is reached through
/// an attribute access rather than being bound directly.
#[test]
fn go_to_def_through_special_import_alias() {
    let (_root, interaction) = open_cinc("alias.cinc");
    interaction
        .client
        .definition("alias.cinc", 7, 16)
        .expect_definition_response_from_root("service/types.thrift.pyi", 5, 6, 5, 14)
        .unwrap();
    interaction.shutdown().unwrap();
}
