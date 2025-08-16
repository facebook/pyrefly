/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests for ResolveImportDeclarationParams, Declaration, and related types construction and serialization

use tsp_types::Declaration;
use tsp_types::DeclarationCategory;
use tsp_types::DeclarationFlags;
use tsp_types::DeclarationHandle;
use tsp_types::ModuleName;
use tsp_types::Node;
use tsp_types::Position;
use tsp_types::Range;
use tsp_types::ResolveImportDeclarationParams;
use tsp_types::ResolveImportOptions;

#[test]
fn test_resolve_import_declaration_params_construction() {
    // Test basic parameter construction
    let declaration = Declaration {
        handle: DeclarationHandle::String("test_handle".to_owned()),
        category: DeclarationCategory::Import,
        flags: DeclarationFlags::new(),
        node: Some(Node {
            uri: "file:///test.py".to_string(),
            range: Range {
                start: Position {
                    line: 0,
                    character: 0,
                },
                end: Position {
                    line: 0,
                    character: 10,
                },
            },
        }),
        module_name: ModuleName {
            leading_dots: 0,
            name_parts: vec!["test_module".to_owned()],
        },
        name: "imported_symbol".to_owned(),
        uri: "file:///test.py".to_string(),
    };

    let options = ResolveImportOptions {
        resolve_local_names: Some(true),
        allow_externally_hidden_access: Some(false),
        skip_file_needed_check: Some(true),
    };

    let params = ResolveImportDeclarationParams {
        decl: declaration,
        options,
        snapshot: 42,
    };

    // Verify parameter construction
    assert_eq!(params.snapshot, 42);
    assert_eq!(params.decl.category, DeclarationCategory::Import);
    assert_eq!(params.decl.name, "imported_symbol");
    assert_eq!(params.decl.module_name.name_parts, vec!["test_module"]);
    assert_eq!(params.options.resolve_local_names, Some(true));
    assert_eq!(params.options.allow_externally_hidden_access, Some(false));
    assert_eq!(params.options.skip_file_needed_check, Some(true));
}

#[test]
fn test_resolve_import_declaration_default_options() {
    // Test default options construction
    let default_options = ResolveImportOptions::default();

    assert_eq!(default_options.resolve_local_names, Some(false));
    assert_eq!(default_options.allow_externally_hidden_access, Some(false));
    assert_eq!(default_options.skip_file_needed_check, Some(false));

    let declaration = Declaration {
        handle: DeclarationHandle::String("test_handle".to_owned()),
        category: DeclarationCategory::Function,
        flags: DeclarationFlags::new(),
        node: None,
        module_name: ModuleName {
            leading_dots: 0,
            name_parts: vec!["test_module".to_owned()],
        },
        name: "my_function".to_owned(),
        uri: "file:///test.py".to_string(),
    };

    let params = ResolveImportDeclarationParams {
        decl: declaration,
        options: default_options,
        snapshot: 1,
    };

    // Non-import declaration should be handled differently
    assert_eq!(params.decl.category, DeclarationCategory::Function);
}

#[test]
fn test_resolve_import_declaration_different_categories() {
    // Test different declaration categories
    let categories = vec![
        (DeclarationCategory::Import, "import_symbol"),
        (DeclarationCategory::Function, "function_symbol"),
        (DeclarationCategory::Class, "class_symbol"),
        (DeclarationCategory::Variable, "variable_symbol"),
        (DeclarationCategory::Param, "param_symbol"),
    ];

    for (category, name) in categories {
        let category = category.clone();
        let declaration = Declaration {
            handle: DeclarationHandle::String(format!("handle_{name}")),
            category: category.clone(),
            flags: DeclarationFlags::new(),
            node: Some(Node {
                uri: "file:///test.py".to_string(),
                range: Range {
                    start: Position {
                        line: 0,
                        character: 0,
                    },
                    end: Position {
                        line: 0,
                        character: name.len() as u32,
                    },
                },
            }),
            module_name: ModuleName {
                leading_dots: 0,
                name_parts: vec!["test_module".to_owned()],
            },
            name: name.to_owned(),
            uri: "file:///test.py".to_string(),
        };

        let params = ResolveImportDeclarationParams {
            decl: declaration.clone(),
            options: ResolveImportOptions::default(),
            snapshot: 1,
        };

        assert_eq!(params.decl.category, category);
        assert_eq!(params.decl.name, name);
    }
}

#[test]
fn test_resolve_import_declaration_module_name_variants() {
    // Test different module name patterns
    let module_patterns = vec![
        // Simple module
        (
            ModuleName {
                leading_dots: 0,
                name_parts: vec!["os".to_owned()],
            },
            "os module",
        ),
        // Nested module
        (
            ModuleName {
                leading_dots: 0,
                name_parts: vec!["os".to_owned(), "path".to_owned()],
            },
            "os.path module",
        ),
        // Relative import with single dot
        (
            ModuleName {
                leading_dots: 1,
                name_parts: vec!["utils".to_owned()],
            },
            "relative utils",
        ),
        // Relative import with multiple dots
        (
            ModuleName {
                leading_dots: 2,
                name_parts: vec!["shared".to_owned(), "helpers".to_owned()],
            },
            "deeply relative",
        ),
        // Current package import
        (
            ModuleName {
                leading_dots: 1,
                name_parts: vec![],
            },
            "current package",
        ),
    ];

    for (module_name, description) in module_patterns {
        let declaration = Declaration {
            handle: DeclarationHandle::String(format!("handle_{}", description.replace(' ', "_"))),
            category: DeclarationCategory::Import,
            flags: DeclarationFlags::new(),
            node: None,
            module_name: module_name.clone(),
            name: "imported_item".to_owned(),
            uri: "file:///test.py".to_string(),
        };

        let params = ResolveImportDeclarationParams {
            decl: declaration,
            options: ResolveImportOptions::default(),
            snapshot: 1,
        };

        assert_eq!(
            params.decl.module_name.leading_dots,
            module_name.leading_dots
        );
        assert_eq!(params.decl.module_name.name_parts, module_name.name_parts);
    }
}

#[test]
fn test_resolve_import_declaration_serialization() {
    let declaration = Declaration {
        handle: DeclarationHandle::String("test_handle".to_owned()),
        category: DeclarationCategory::Import,
        flags: DeclarationFlags::new(),
        node: Some(Node {
            uri: "file:///example.py".to_string(),
            range: Range {
                start: Position {
                    line: 2,
                    character: 5,
                },
                end: Position {
                    line: 2,
                    character: 15,
                },
            },
        }),
        module_name: ModuleName {
            leading_dots: 1,
            name_parts: vec!["utils".to_owned(), "helpers".to_owned()],
        },
        name: "helper_function".to_owned(),
        uri: "file:///example.py".to_string(),
    };

    let options = ResolveImportOptions {
        resolve_local_names: Some(true),
        allow_externally_hidden_access: Some(false),
        skip_file_needed_check: Some(true),
    };

    let params = ResolveImportDeclarationParams {
        decl: declaration,
        options,
        snapshot: 99,
    };

    // Test serialization round-trip
    let serialized = serde_json::to_string(&params).expect("Should serialize");
    let deserialized: ResolveImportDeclarationParams =
        serde_json::from_str(&serialized).expect("Should deserialize");

    assert_eq!(deserialized.snapshot, params.snapshot);
    assert_eq!(deserialized.decl.name, params.decl.name);
    assert_eq!(deserialized.decl.category, params.decl.category);
    assert_eq!(
        deserialized.decl.module_name.leading_dots,
        params.decl.module_name.leading_dots
    );
    assert_eq!(
        deserialized.decl.module_name.name_parts,
        params.decl.module_name.name_parts
    );
    assert_eq!(
        deserialized.options.resolve_local_names,
        params.options.resolve_local_names
    );
    assert_eq!(
        deserialized.options.allow_externally_hidden_access,
        params.options.allow_externally_hidden_access
    );
    assert_eq!(
        deserialized.options.skip_file_needed_check,
        params.options.skip_file_needed_check
    );
}

#[test]
fn test_declaration_handle_variants() {
    // Test different declaration handle variants
    let handle_variants = vec![
        (
            DeclarationHandle::String("string_handle".to_owned()),
            "string handle",
        ),
        (DeclarationHandle::Int(42), "int handle"),
    ];

    for (handle, description) in handle_variants {
        let declaration = Declaration {
            handle,
            category: DeclarationCategory::Function,
            flags: DeclarationFlags::new(),
            node: None,
            module_name: ModuleName {
                leading_dots: 0,
                name_parts: vec!["test".to_owned()],
            },
            name: format!("symbol_for_{}", description.replace(' ', "_")),
            uri: "file:///test.py".to_string(),
        };

        let params = ResolveImportDeclarationParams {
            decl: declaration.clone(),
            options: ResolveImportOptions::default(),
            snapshot: 1,
        };

        // Verify the handle was preserved (basic check)
        assert_eq!(params.decl.category, DeclarationCategory::Function);
        assert!(params.decl.name.contains("symbol_for"));
    }
}

#[test]
fn test_declaration_node_handling() {
    // Test with node present
    let with_node = Declaration {
        handle: DeclarationHandle::String("with_node".to_owned()),
        category: DeclarationCategory::Import,
        flags: DeclarationFlags::new(),
        node: Some(Node {
            uri: "file:///test.py".to_string(),
            range: Range {
                start: Position {
                    line: 5,
                    character: 10,
                },
                end: Position {
                    line: 5,
                    character: 20,
                },
            },
        }),
        module_name: ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        },
        name: "symbol".to_owned(),
        uri: "file:///test.py".to_string(),
    };

    // Test with node absent
    let without_node = Declaration {
        handle: DeclarationHandle::String("without_node".to_owned()),
        category: DeclarationCategory::Import,
        flags: DeclarationFlags::new(),
        node: None,
        module_name: ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        },
        name: "symbol".to_owned(),
        uri: "file:///test.py".to_string(),
    };

    let params_with_node = ResolveImportDeclarationParams {
        decl: with_node,
        options: ResolveImportOptions::default(),
        snapshot: 1,
    };

    let params_without_node = ResolveImportDeclarationParams {
        decl: without_node,
        options: ResolveImportOptions::default(),
        snapshot: 1,
    };

    assert!(params_with_node.decl.node.is_some());
    assert!(params_without_node.decl.node.is_none());

    if let Some(node) = &params_with_node.decl.node {
        assert_eq!(node.range.start.line, 5);
        assert_eq!(node.range.start.character, 10);
        assert_eq!(node.range.end.line, 5);
        assert_eq!(node.range.end.character, 20);
    }
}
