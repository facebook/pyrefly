/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Unit tests for TSP GetSymbolsForTypeParams and GetSymbolsForTypeRequest type construction and serialization
//!
//! These tests verify the GetSymbolsForTypeParams parameter construction and validation by:
//! 1. Testing TSP GetSymbolsForTypeParams construction with various parameter combinations
//! 2. Testing Type parameter variations
//! 3. Testing snapshot validation logic
//! 4. Testing flag combinations
//! 5. Testing optional name and node parameters
//! 6. Testing serialization/deserialization round-trips

use tsp_types::Declaration;
use tsp_types::DeclarationCategory;
use tsp_types::DeclarationFlags;
use tsp_types::DeclarationHandle;
use tsp_types::GetSymbolsForTypeParams;
use tsp_types::GetSymbolsForTypeRequest;
use tsp_types::LSPId;
use tsp_types::ModuleName;
use tsp_types::Node;
use tsp_types::Position;
use tsp_types::Range;
use tsp_types::SymbolSearchFlags;
use tsp_types::TSPRequestMethods;
use tsp_types::Type;
use tsp_types::TypeCategory;
use tsp_types::TypeFlags;
use tsp_types::TypeHandle;

#[test]
fn test_get_symbols_for_type_params_construction() {
    // Test basic parameter construction
    let type_param = Type {
        alias_name: None,
        handle: TypeHandle::String("test_class".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["mymodule".to_owned()],
        }),
        name: "TestClass".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params = GetSymbolsForTypeParams {
        flags: SymbolSearchFlags::new(),
        name: Some("method_name".to_owned()),
        node: None,
        snapshot: 42,
        type_: type_param.clone(),
    };

    // Verify parameter construction
    assert_eq!(params.snapshot, 42);
    assert_eq!(params.name, Some("method_name".to_owned()));
    assert!(params.node.is_none());
    assert_eq!(params.type_.name, "TestClass");
    assert_eq!(params.type_.category, TypeCategory::Class);
}

#[test]
fn test_get_symbols_for_type_request_construction() {
    // Test basic request construction
    let type_param = Type {
        alias_name: None,
        handle: TypeHandle::String("test_function".to_owned()),
        category: TypeCategory::Function,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["module".to_owned()],
        }),
        name: "test_func".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params = GetSymbolsForTypeParams {
        flags: SymbolSearchFlags::new(),
        name: None,
        node: None,
        snapshot: 10,
        type_: type_param,
    };

    let request = GetSymbolsForTypeRequest {
        method: TSPRequestMethods::TypeServerGetSymbolsForType,
        id: LSPId::Int(1),
        params,
    };

    // Verify request construction
    assert_eq!(
        request.method,
        TSPRequestMethods::TypeServerGetSymbolsForType
    );
    match request.id {
        LSPId::Int(n) => assert_eq!(n, 1),
        _ => panic!("Expected Number id"),
    }
    assert_eq!(request.params.snapshot, 10);
    assert_eq!(request.params.type_.name, "test_func");
    assert_eq!(request.params.type_.category, TypeCategory::Function);
}

#[test]
fn test_get_symbols_for_type_params_different_type_categories() {
    // Test with CLASS category
    let class_type = Type {
        alias_name: None,
        handle: TypeHandle::String("class_handle".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        }),
        name: "MyClass".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_class = GetSymbolsForTypeParams {
        flags: SymbolSearchFlags::new(),
        name: None,
        node: None,
        snapshot: 1,
        type_: class_type,
    };

    // Test with FUNCTION category
    let function_type = Type {
        alias_name: None,
        handle: TypeHandle::String("function_handle".to_owned()),
        category: TypeCategory::Function,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        }),
        name: "my_function".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_function = GetSymbolsForTypeParams {
        flags: SymbolSearchFlags::new(),
        name: None,
        node: None,
        snapshot: 1,
        type_: function_type,
    };

    // Test with MODULE category
    let module_type = Type {
        alias_name: None,
        handle: TypeHandle::String("module_handle".to_owned()),
        category: TypeCategory::Module,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        }),
        name: "test_module".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_module = GetSymbolsForTypeParams {
        flags: SymbolSearchFlags::new(),
        name: None,
        node: None,
        snapshot: 1,
        type_: module_type,
    };

    // Verify different categories
    assert_eq!(params_class.type_.category, TypeCategory::Class);
    assert_eq!(params_function.type_.category, TypeCategory::Function);
    assert_eq!(params_module.type_.category, TypeCategory::Module);
}

#[test]
fn test_get_symbols_for_type_params_with_node() {
    // Test with node parameter provided
    let type_param = Type {
        alias_name: None,
        handle: TypeHandle::String("type_with_node".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["module".to_owned()],
        }),
        name: "ClassWithNode".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let node = Node {
        uri: "file:///test.py".to_owned(),
        range: Range {
            start: Position {
                line: 10,
                character: 5,
            },
            end: Position {
                line: 10,
                character: 20,
            },
        },
    };

    let params_with_node = GetSymbolsForTypeParams {
        flags: SymbolSearchFlags::new(),
        name: Some("specific_symbol".to_owned()),
        node: Some(node.clone()),
        snapshot: 15,
        type_: type_param,
    };

    // Test without node parameter
    let type_param_no_node = Type {
        alias_name: None,
        handle: TypeHandle::String("type_no_node".to_owned()),
        category: TypeCategory::Function,
        flags: TypeFlags::new(),
        module_name: None,
        name: "function_no_node".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_no_node = GetSymbolsForTypeParams {
        flags: SymbolSearchFlags::new(),
        name: None,
        node: None,
        snapshot: 15,
        type_: type_param_no_node,
    };

    // Verify node handling
    assert!(params_with_node.node.is_some());
    let node_ref = params_with_node.node.as_ref().unwrap();
    assert_eq!(node_ref.uri, "file:///test.py");
    assert_eq!(node_ref.range.start.line, 10);
    assert_eq!(node_ref.range.start.character, 5);

    assert!(params_no_node.node.is_none());
}

#[test]
fn test_get_symbols_for_type_params_with_name_variations() {
    // Test with specific name
    let type_param = Type {
        alias_name: None,
        handle: TypeHandle::String("named_search".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        }),
        name: "SearchClass".to_owned(),
        category_flags: 0,
        decl: None,
    };

    // Test with specific name
    let params_with_name = GetSymbolsForTypeParams {
        flags: SymbolSearchFlags::new(),
        name: Some("specific_method".to_owned()),
        node: None,
        snapshot: 5,
        type_: type_param.clone(),
    };

    // Test with empty name (but Some)
    let params_empty_name = GetSymbolsForTypeParams {
        flags: SymbolSearchFlags::new(),
        name: Some("".to_owned()),
        node: None,
        snapshot: 5,
        type_: type_param.clone(),
    };

    // Test with None name (returns all symbols)
    let params_no_name = GetSymbolsForTypeParams {
        flags: SymbolSearchFlags::new(),
        name: None,
        node: None,
        snapshot: 5,
        type_: type_param,
    };

    // Verify name handling
    assert_eq!(params_with_name.name, Some("specific_method".to_owned()));
    assert_eq!(params_empty_name.name, Some("".to_owned()));
    assert!(params_no_name.name.is_none());
}

#[test]
fn test_get_symbols_for_type_params_serialization() {
    // Test that parameters can be properly serialized and deserialized
    let original_params = GetSymbolsForTypeParams {
        flags: SymbolSearchFlags::new(),
        name: Some("serialization_test".to_owned()),
        node: Some(Node {
            uri: "file:///serialization/test.py".to_owned(),
            range: Range {
                start: Position {
                    line: 1,
                    character: 0,
                },
                end: Position {
                    line: 1,
                    character: 10,
                },
            },
        }),
        snapshot: 999,
        type_: Type {
            alias_name: None,
            handle: TypeHandle::String("serialization_type".to_owned()),
            category: TypeCategory::Class,
            flags: TypeFlags::new(),
            module_name: Some(ModuleName {
                leading_dots: 1,
                name_parts: vec!["serialization".to_owned(), "test".to_owned()],
            }),
            name: "SerializableClass".to_owned(),
            category_flags: 2,
            decl: Some(Declaration {
                category: DeclarationCategory::Class,
                flags: DeclarationFlags::new(),
                handle: DeclarationHandle::String("decl_handle_serializable".to_owned()),
                module_name: ModuleName {
                    leading_dots: 1,
                    name_parts: vec!["serialization".to_owned(), "test".to_owned()],
                },
                name: "SerializableClass".to_owned(),
                node: None,
                uri: "file:///serialization/test.py".to_owned(),
            }),
        },
    };

    // Serialize to JSON
    let json_str = serde_json::to_string(&original_params).expect("Failed to serialize");

    // Deserialize back from JSON
    let deserialized_params: GetSymbolsForTypeParams =
        serde_json::from_str(&json_str).expect("Failed to deserialize");

    // Verify round-trip serialization
    assert_eq!(deserialized_params.snapshot, original_params.snapshot);
    assert_eq!(deserialized_params.name, original_params.name);
    assert_eq!(deserialized_params.type_.name, original_params.type_.name);
    assert_eq!(
        deserialized_params.type_.category,
        original_params.type_.category
    );

    // Verify node serialization
    let orig_node = original_params.node.as_ref().unwrap();
    let deser_node = deserialized_params.node.as_ref().unwrap();
    assert_eq!(deser_node.uri, orig_node.uri);
    assert_eq!(deser_node.range.start.line, orig_node.range.start.line);
}
