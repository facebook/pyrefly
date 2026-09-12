/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::Url;
use lsp_types::WorkspaceSymbolResponse;
use pyrefly_lsp_test::IndexingMode;
use pyrefly_lsp_test::LspArgs;
use pyrefly_lsp_test::object_model::InitializeSettings;
use pyrefly_lsp_test::object_model::LspInteraction;
use pyrefly_lsp_test::object_model::LspInteractionArgs;
use serde_json::json;

use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_workspace_symbol() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            configuration: Some(Some(json!([{ "indexing_mode": "lazy_blocking"}]))),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("autoimport_provider.py");

    interaction
        .client
        .send_workspace_symbol("this_is_a_very_long_function_name_so_we_can")
        .expect_response(json!([
            {
                "kind": 12,
                "location": {
                    "range": {
                        "start": {"line": 6, "character": 4},
                        "end": {"line": 6, "character": 99}
                    },
                    "uri": Url::from_file_path(root_path.join("autoimport_provider.py")).unwrap().to_string()
                },
                "name": "this_is_a_very_long_function_name_so_we_can_deterministically_test_autoimport_with_fuzzy_search"
            }
        ]))
        .unwrap();

    interaction.shutdown().unwrap();
}

// Score outranks the `__init__.py` preference. This test's two candidates tie
// on score, so the preference is what decides between them.
#[test]
fn test_workspace_symbol_prefers_non_init_result_on_equal_score() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            configuration: Some(Some(json!([{ "indexing_mode": "lazy_blocking"}]))),
            ..Default::default()
        })
        .unwrap();

    interaction
        .client
        .did_open("workspace_symbol_prefer_non_init/implementation.py");
    interaction
        .client
        .did_open("workspace_symbol_prefer_non_init/__init__.py");

    let implementation_uri =
        Url::from_file_path(root_path.join("workspace_symbol_prefer_non_init/implementation.py"))
            .unwrap();
    let init_uri =
        Url::from_file_path(root_path.join("workspace_symbol_prefer_non_init/__init__.py"))
            .unwrap();
    let symbol_name = "workspace_symbol_prefers_non_init_over_init_reexport";

    interaction
        .client
        .send_workspace_symbol(symbol_name)
        .expect_response_with(|result| {
            let Some(WorkspaceSymbolResponse::Flat(symbols)) = result else {
                panic!("Unexpected workspace symbol response: {result:?}");
            };
            assert!(symbols.iter().all(|symbol| symbol.name == symbol_name));
            assert!(symbols.iter().all(|symbol| {
                symbol.location.uri == implementation_uri || symbol.location.uri == init_uri
            }));

            let first_init_index = symbols
                .iter()
                .position(|symbol| symbol.location.uri == init_uri)
                .expect("expected at least one __init__.py result");
            let last_non_init_index = symbols
                .iter()
                .rposition(|symbol| symbol.location.uri == implementation_uri)
                .expect("expected at least one non-__init__.py result");

            assert!(last_non_init_index < first_init_index);
            true
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

// Re-export paths may resolve the same canonical declaration several times.
// Collapse those byte-identical rows without removing the distinct local
// re-export in `__init__.py`.
#[test]
fn test_workspace_symbol_deduplicates_reexported_definitions() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            configuration: Some(Some(json!([{ "indexing_mode": "lazy_blocking"}]))),
            ..Default::default()
        })
        .unwrap();

    interaction
        .client
        .did_open("workspace_symbol_prefer_non_init/implementation.py");
    interaction
        .client
        .did_open("workspace_symbol_prefer_non_init/__init__.py");

    let implementation_uri =
        Url::from_file_path(root_path.join("workspace_symbol_prefer_non_init/implementation.py"))
            .unwrap();
    let init_uri =
        Url::from_file_path(root_path.join("workspace_symbol_prefer_non_init/__init__.py"))
            .unwrap();
    let symbol_name = "workspace_symbol_prefers_non_init_over_init_reexport";

    interaction
        .client
        .send_workspace_symbol(symbol_name)
        .expect_response_with(|result| {
            let Some(WorkspaceSymbolResponse::Flat(symbols)) = result else {
                panic!("Unexpected workspace symbol response: {result:?}");
            };

            let canonical: Vec<_> = symbols
                .iter()
                .filter(|s| s.location.uri == implementation_uri)
                .collect();
            let reexport: Vec<_> = symbols
                .iter()
                .filter(|s| s.location.uri == init_uri)
                .collect();

            assert_eq!(
                symbols.len(),
                canonical.len() + reexport.len(),
                "every result should come from one of the package's two files"
            );

            assert_eq!(canonical.len(), 1);
            assert_eq!(canonical[0].name, symbol_name);
            assert_ne!(
                canonical[0].location.range,
                lsp_types::Range::default(),
                "the canonical row should point at the definition, not the file start"
            );

            // The re-export row is distinct: a zero range, because it stands for
            // the re-exporting module rather than a definition within it.
            assert_eq!(reexport.len(), 1);
            assert_eq!(reexport[0].name, symbol_name);
            assert_eq!(reexport[0].location.range, lsp_types::Range::default());
            true
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

// Methods live inside a `ClassDef`, so they are not module exports and do not
// appear via the export-table path that backs `workspace/symbol`. They are
// surfaced from the cached per-module symbol tables (`Exports::symbols`),
// scanned by `search_workspace_symbols_fuzzy`.
#[test]
fn test_workspace_symbol_includes_methods_of_open_files() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            configuration: Some(Some(json!([{ "indexing_mode": "lazy_blocking"}]))),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("workspace_symbol_methods.py");

    let uri = Url::from_file_path(root_path.join("workspace_symbol_methods.py")).unwrap();
    interaction
        .client
        .send_workspace_symbol("workspace_symbol_method_deterministic_name")
        .expect_response_with(|result| {
            let Some(WorkspaceSymbolResponse::Flat(symbols)) = result else {
                panic!("Unexpected workspace symbol response: {result:?}");
            };
            let method = symbols
                .iter()
                .find(|s| s.name == "workspace_symbol_method_deterministic_name")
                .expect("expected the method to appear in workspace symbols");
            assert_eq!(method.kind, lsp_types::SymbolKind::METHOD);
            assert_eq!(method.location.uri, uri);
            assert_eq!(
                method.container_name.as_deref(),
                Some("WorkspaceSymbolMethodHost")
            );
            true
        })
        .unwrap();

    interaction
        .client
        .send_workspace_symbol("workspace_symbol_class_attribute_deterministic_name")
        .expect_response_with(|result| {
            let Some(WorkspaceSymbolResponse::Flat(symbols)) = result else {
                panic!("Unexpected workspace symbol response: {result:?}");
            };
            let attribute = symbols
                .iter()
                .find(|s| s.name == "workspace_symbol_class_attribute_deterministic_name")
                .expect("expected the class attribute to appear in workspace symbols");
            assert_eq!(attribute.kind, lsp_types::SymbolKind::FIELD);
            assert_eq!(attribute.location.uri, uri);
            assert_eq!(
                attribute.container_name.as_deref(),
                Some("WorkspaceSymbolMethodHost")
            );
            true
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

// The same coverage as the test above, for a file the user never opened, which
// project indexing has loaded on their behalf. The symbol table is built during
// indexing, so it is populated for these files too.
#[test]
fn test_workspace_symbol_includes_methods_of_indexed_files() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new_with_args(LspInteractionArgs {
        args: LspArgs {
            indexing_mode: IndexingMode::LazyBlocking,
            ..LspInteractionArgs::default().args
        },
        ..Default::default()
    });
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            ..Default::default()
        })
        .unwrap();

    // Opening any file in the project triggers indexing of the whole config; the
    // queried method lives in an unopened sibling.
    interaction.client.did_open("autoimport_provider.py");

    let uri = Url::from_file_path(root_path.join("workspace_symbol_methods_indexed.py")).unwrap();
    interaction
        .client
        .send_workspace_symbol("workspace_symbol_indexed_only_method_name")
        .expect_response_with(|result| {
            let Some(WorkspaceSymbolResponse::Flat(symbols)) = result else {
                panic!("Unexpected workspace symbol response: {result:?}");
            };
            let method = symbols
                .iter()
                .find(|s| s.name == "workspace_symbol_indexed_only_method_name")
                .expect("expected the method from the indexed (unopened) file");
            assert_eq!(method.kind, lsp_types::SymbolKind::METHOD);
            assert_eq!(method.location.uri, uri);
            assert_eq!(
                method.container_name.as_deref(),
                Some("WorkspaceSymbolIndexedHost")
            );
            true
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

// Regression test for https://github.com/facebook/pyrefly/issues/3041
#[test]
fn test_workspace_symbol_multibyte_no_panic() {
    let root = get_test_files_root();
    let root_path = root.path().join("tests_requiring_config");
    let scope_uri = Url::from_file_path(root_path.clone()).unwrap();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            configuration: Some(Some(json!([{ "indexing_mode": "lazy_blocking"}]))),
            ..Default::default()
        })
        .unwrap();

    interaction
        .client
        .did_open("workspace_symbol_multibyte/__init__.py");
    interaction
        .client
        .did_open("workspace_symbol_multibyte/impl_mod.py");

    interaction
        .client
        .send_workspace_symbol("workspace_symbol_multibyte_repro")
        .expect_response_with(|result| {
            let Some(WorkspaceSymbolResponse::Flat(symbols)) = result else {
                panic!("Unexpected workspace symbol response: {result:?}");
            };
            assert!(
                !symbols.is_empty(),
                "Expected at least one result for workspace_symbol_multibyte_repro"
            );
            true
        })
        .unwrap();

    let path = root_path.join("workspace_symbol_multibyte/impl_mod.py");
    let uri = Url::from_file_path(path).unwrap();
    interaction
        .client
        .send_workspace_symbol("workspace_symbol_multibyte_nested_method")
        .expect_response_with(|result| {
            let Some(WorkspaceSymbolResponse::Flat(symbols)) = result else {
                panic!("Unexpected workspace symbol response: {result:?}");
            };
            let method = symbols
                .iter()
                .find(|s| s.name == "workspace_symbol_multibyte_nested_method")
                .expect("expected the nested symbol after multibyte text");
            assert_eq!(method.kind, lsp_types::SymbolKind::METHOD);
            assert_eq!(method.location.uri, uri);
            assert_eq!(
                method.container_name.as_deref(),
                Some("WorkspaceSymbolMultibyteHost")
            );
            true
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
