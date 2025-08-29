/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Unit tests for TSP GetDocstringParams and GetDocstringRequest type construction and serialization
//!
//! These tests verify the GetDocstringParams parameter construction and validation by:
//! 1. Testing TSP GetDocstringParams construction with various parameter combinations
//! 2. Testing Declaration parameter handling
//! 3. Testing optional Type parameters (bound_object_or_class and type_)
//! 4. Testing snapshot validation logic
//! 5. Testing serialization/deserialization round-trips

use tsp_types::Declaration;
use tsp_types::DeclarationCategory;
use tsp_types::DeclarationFlags;
use tsp_types::DeclarationHandle;
use tsp_types::GetDocstringParams;
use tsp_types::GetDocstringRequest;
use tsp_types::LSPId;
use tsp_types::ModuleName;
use tsp_types::TSPRequestMethods;
use tsp_types::Type;
use tsp_types::TypeCategory;
use tsp_types::TypeFlags;
use tsp_types::TypeHandle;

#[test]
fn test_get_docstring_params_construction() {
    // Test basic parameter construction
    let decl = Declaration {
        category: DeclarationCategory::Function,
        flags: DeclarationFlags::new(),
        handle: DeclarationHandle::String("test_function_handle".to_owned()),
        module_name: ModuleName {
            leading_dots: 0,
            name_parts: vec!["mymodule".to_owned()],
        },
        name: "test_function".to_owned(),
        node: None,
        uri: "file:///test.py".to_owned(),
    };

    let params = GetDocstringParams {
        bound_object_or_class: None,
        decl: decl.clone(),
        snapshot: 42,
        type_: None,
    };

    // Verify parameter construction
    assert_eq!(params.snapshot, 42);
    assert!(params.bound_object_or_class.is_none());
    assert!(params.type_.is_none());
    assert_eq!(params.decl.name, "test_function");
    assert_eq!(params.decl.category, DeclarationCategory::Function);
}

#[test]
fn test_get_docstring_request_construction() {
    // Test basic request construction
    let decl = Declaration {
        category: DeclarationCategory::Class,
        flags: DeclarationFlags::new(),
        handle: DeclarationHandle::String("test_class_handle".to_owned()),
        module_name: ModuleName {
            leading_dots: 0,
            name_parts: vec!["module".to_owned()],
        },
        name: "TestClass".to_owned(),
        node: None,
        uri: "file:///module.py".to_owned(),
    };

    let params = GetDocstringParams {
        bound_object_or_class: None,
        decl,
        snapshot: 10,
        type_: None,
    };

    let request = GetDocstringRequest {
        method: TSPRequestMethods::TypeServerGetDocstring,
        id: LSPId::Int(1),
        params,
    };

    // Verify request construction
    assert_eq!(request.method, TSPRequestMethods::TypeServerGetDocstring);
    match request.id {
        LSPId::Int(n) => assert_eq!(n, 1),
        _ => panic!("Expected Number id"),
    }
    assert_eq!(request.params.snapshot, 10);
    assert_eq!(request.params.decl.name, "TestClass");
    assert_eq!(request.params.decl.category, DeclarationCategory::Class);
}

#[test]
fn test_get_docstring_params_with_bound_object() {
    // Test with bound object or class type
    let bound_type = Type {
        alias_name: None,
        handle: TypeHandle::String("bound_object_handle".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        }),
        name: "BoundClass".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let decl = Declaration {
        category: DeclarationCategory::Function,
        flags: DeclarationFlags::new(),
        handle: DeclarationHandle::String("method_handle".to_owned()),
        module_name: ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        },
        name: "method_with_bound_object".to_owned(),
        node: None,
        uri: "file:///test.py".to_owned(),
    };

    let params_with_bound = GetDocstringParams {
        bound_object_or_class: Some(bound_type.clone()),
        decl,
        snapshot: 15,
        type_: None,
    };

    // Test without bound object
    let decl_no_bound = Declaration {
        category: DeclarationCategory::Function,
        flags: DeclarationFlags::new(),
        handle: DeclarationHandle::String("standalone_function_handle".to_owned()),
        module_name: ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        },
        name: "standalone_function".to_owned(),
        node: None,
        uri: "file:///test.py".to_owned(),
    };

    let params_no_bound = GetDocstringParams {
        bound_object_or_class: None,
        decl: decl_no_bound,
        snapshot: 15,
        type_: None,
    };

    // Verify bound object handling
    assert!(params_with_bound.bound_object_or_class.is_some());
    let bound_ref = params_with_bound.bound_object_or_class.as_ref().unwrap();
    assert_eq!(bound_ref.name, "BoundClass");
    assert_eq!(bound_ref.category, TypeCategory::Class);

    assert!(params_no_bound.bound_object_or_class.is_none());
}

#[test]
fn test_get_docstring_params_with_type_context() {
    // Test with type context
    let type_context = Type {
        alias_name: None,
        handle: TypeHandle::String("context_type_handle".to_owned()),
        category: TypeCategory::Module,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["context".to_owned()],
        }),
        name: "ContextModule".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let decl = Declaration {
        category: DeclarationCategory::Variable,
        flags: DeclarationFlags::new(),
        handle: DeclarationHandle::String("variable_handle".to_owned()),
        module_name: ModuleName {
            leading_dots: 0,
            name_parts: vec!["context".to_owned()],
        },
        name: "context_variable".to_owned(),
        node: None,
        uri: "file:///context.py".to_owned(),
    };

    let params_with_type = GetDocstringParams {
        bound_object_or_class: None,
        decl,
        snapshot: 20,
        type_: Some(type_context.clone()),
    };

    // Test without type context
    let decl_no_type = Declaration {
        category: DeclarationCategory::Variable,
        flags: DeclarationFlags::new(),
        handle: DeclarationHandle::String("simple_variable_handle".to_owned()),
        module_name: ModuleName {
            leading_dots: 0,
            name_parts: vec!["simple".to_owned()],
        },
        name: "simple_variable".to_owned(),
        node: None,
        uri: "file:///simple.py".to_owned(),
    };

    let params_no_type = GetDocstringParams {
        bound_object_or_class: None,
        decl: decl_no_type,
        snapshot: 20,
        type_: None,
    };

    // Verify type context handling
    assert!(params_with_type.type_.is_some());
    let type_ref = params_with_type.type_.as_ref().unwrap();
    assert_eq!(type_ref.name, "ContextModule");
    assert_eq!(type_ref.category, TypeCategory::Module);

    assert!(params_no_type.type_.is_none());
}

#[test]
fn test_get_docstring_params_different_declaration_categories() {
    // Test with function declaration
    let function_decl = Declaration {
        category: DeclarationCategory::Function,
        flags: DeclarationFlags::new(),
        handle: DeclarationHandle::String("function_handle".to_owned()),
        module_name: ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        },
        name: "test_function".to_owned(),
        node: None,
        uri: "file:///test.py".to_owned(),
    };

    let params_function = GetDocstringParams {
        bound_object_or_class: None,
        decl: function_decl,
        snapshot: 1,
        type_: None,
    };

    // Test with class declaration
    let class_decl = Declaration {
        category: DeclarationCategory::Class,
        flags: DeclarationFlags::new(),
        handle: DeclarationHandle::String("class_handle".to_owned()),
        module_name: ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        },
        name: "TestClass".to_owned(),
        node: None,
        uri: "file:///test.py".to_owned(),
    };

    let params_class = GetDocstringParams {
        bound_object_or_class: None,
        decl: class_decl,
        snapshot: 1,
        type_: None,
    };

    // Test with variable declaration
    let variable_decl = Declaration {
        category: DeclarationCategory::Variable,
        flags: DeclarationFlags::new(),
        handle: DeclarationHandle::String("variable_handle".to_owned()),
        module_name: ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        },
        name: "test_variable".to_owned(),
        node: None,
        uri: "file:///test.py".to_owned(),
    };

    let params_variable = GetDocstringParams {
        bound_object_or_class: None,
        decl: variable_decl,
        snapshot: 1,
        type_: None,
    };

    // Verify different declaration categories
    assert_eq!(params_function.decl.category, DeclarationCategory::Function);
    assert_eq!(params_class.decl.category, DeclarationCategory::Class);
    assert_eq!(params_variable.decl.category, DeclarationCategory::Variable);
}

#[test]
fn test_get_docstring_params_serialization() {
    // Test that parameters can be properly serialized and deserialized
    let original_params = GetDocstringParams {
        bound_object_or_class: Some(Type {
            alias_name: None,
            handle: TypeHandle::String("serialization_bound_type".to_owned()),
            category: TypeCategory::Class,
            flags: TypeFlags::new(),
            module_name: Some(ModuleName {
                leading_dots: 0,
                name_parts: vec!["serialization".to_owned()],
            }),
            name: "SerializationBoundClass".to_owned(),
            category_flags: 1,
            decl: None,
        }),
        decl: Declaration {
            category: DeclarationCategory::Function,
            flags: DeclarationFlags::new(),
            handle: DeclarationHandle::String("serialization_decl_handle".to_owned()),
            module_name: ModuleName {
                leading_dots: 0,
                name_parts: vec!["serialization".to_owned()],
            },
            name: "serialization_function".to_owned(),
            node: None,
            uri: "file:///serialization.py".to_owned(),
        },
        snapshot: 999,
        type_: Some(Type {
            alias_name: None,
            handle: TypeHandle::String("serialization_type_context".to_owned()),
            category: TypeCategory::Module,
            flags: TypeFlags::new(),
            module_name: Some(ModuleName {
                leading_dots: 0,
                name_parts: vec!["serialization".to_owned()],
            }),
            name: "SerializationModule".to_owned(),
            category_flags: 2,
            decl: None,
        }),
    };

    // Serialize to JSON
    let json_str = serde_json::to_string(&original_params).expect("Failed to serialize");

    // Deserialize back from JSON
    let deserialized_params: GetDocstringParams =
        serde_json::from_str(&json_str).expect("Failed to deserialize");

    // Verify round-trip serialization
    assert_eq!(deserialized_params.snapshot, original_params.snapshot);
    assert_eq!(deserialized_params.decl.name, original_params.decl.name);
    assert_eq!(
        deserialized_params.decl.category,
        original_params.decl.category
    );

    // Verify bound object serialization
    let orig_bound = original_params.bound_object_or_class.as_ref().unwrap();
    let deser_bound = deserialized_params.bound_object_or_class.as_ref().unwrap();
    assert_eq!(deser_bound.name, orig_bound.name);
    assert_eq!(deser_bound.category, orig_bound.category);

    // Verify type context serialization
    let orig_type = original_params.type_.as_ref().unwrap();
    let deser_type = deserialized_params.type_.as_ref().unwrap();
    assert_eq!(deser_type.name, orig_type.name);
    assert_eq!(deser_type.category, orig_type.category);
}
