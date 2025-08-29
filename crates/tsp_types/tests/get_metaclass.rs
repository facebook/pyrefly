/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Unit tests for TSP GetMetaclassParams and GetMetaclassRequest type construction and serialization
//!
//! These tests verify the GetMetaclassParams parameter construction and validation by:
//! 1. Testing TSP GetMetaclassParams construction with various parameter combinations
//! 2. Testing Type parameter variations
//! 3. Testing snapshot validation logic
//! 4. Testing different type categories for metaclass queries
//! 5. Testing serialization/deserialization round-trips

use tsp_types::Declaration;
use tsp_types::DeclarationCategory;
use tsp_types::DeclarationFlags;
use tsp_types::DeclarationHandle;
use tsp_types::GetMetaclassParams;
use tsp_types::GetMetaclassRequest;
use tsp_types::LSPId;
use tsp_types::ModuleName;
use tsp_types::TSPRequestMethods;
use tsp_types::Type;
use tsp_types::TypeCategory;
use tsp_types::TypeFlags;
use tsp_types::TypeHandle;

#[test]
fn test_get_metaclass_params_construction() {
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

    let params = GetMetaclassParams {
        snapshot: 42,
        type_: type_param.clone(),
    };

    // Verify parameter construction
    assert_eq!(params.snapshot, 42);
    assert_eq!(params.type_.name, "TestClass");
    assert_eq!(params.type_.category, TypeCategory::Class);
    match &params.type_.handle {
        TypeHandle::String(s) => assert_eq!(s, "test_class"),
        _ => panic!("Expected String handle"),
    }
}

#[test]
fn test_get_metaclass_request_construction() {
    // Test basic request construction
    let type_param = Type {
        alias_name: None,
        handle: TypeHandle::String("class_for_metaclass".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["module".to_owned()],
        }),
        name: "ClassForMetaclass".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params = GetMetaclassParams {
        snapshot: 10,
        type_: type_param,
    };

    let request = GetMetaclassRequest {
        method: TSPRequestMethods::TypeServerGetMetaclass,
        id: LSPId::Int(1),
        params,
    };

    // Verify request construction
    assert_eq!(request.method, TSPRequestMethods::TypeServerGetMetaclass);
    match request.id {
        LSPId::Int(n) => assert_eq!(n, 1),
        _ => panic!("Expected Number id"),
    }
    assert_eq!(request.params.snapshot, 10);
    assert_eq!(request.params.type_.name, "ClassForMetaclass");
    assert_eq!(request.params.type_.category, TypeCategory::Class);
}

#[test]
fn test_get_metaclass_params_different_type_categories() {
    // Test with CLASS category (typical case for metaclass queries)
    let class_type = Type {
        alias_name: None,
        handle: TypeHandle::String("class_handle".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        }),
        name: "RegularClass".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_class = GetMetaclassParams {
        snapshot: 1,
        type_: class_type,
    };

    // Test with UNKNOWN category (edge case)
    let unknown_type = Type {
        alias_name: None,
        handle: TypeHandle::String("unknown_handle".to_owned()),
        category: TypeCategory::Any,
        flags: TypeFlags::new(),
        module_name: None,
        name: "UnknownType".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_unknown = GetMetaclassParams {
        snapshot: 1,
        type_: unknown_type,
    };

    // Test with ANY category (edge case)
    let any_type = Type {
        alias_name: None,
        handle: TypeHandle::String("any_handle".to_owned()),
        category: TypeCategory::Any,
        flags: TypeFlags::new(),
        module_name: None,
        name: "AnyType".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_any = GetMetaclassParams {
        snapshot: 1,
        type_: any_type,
    };

    // Verify different categories
    assert_eq!(params_class.type_.category, TypeCategory::Class);
    assert_eq!(params_unknown.type_.category, TypeCategory::Any);
    assert_eq!(params_any.type_.category, TypeCategory::Any);
}

#[test]
fn test_get_metaclass_params_type_handle_variants() {
    // Test with String handle
    let string_handle_type = Type {
        alias_name: None,
        handle: TypeHandle::String("string_handle_123".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["module".to_owned()],
        }),
        name: "StringHandleClass".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_string = GetMetaclassParams {
        snapshot: 5,
        type_: string_handle_type,
    };

    // Test with Integer handle
    let integer_handle_type = Type {
        alias_name: None,
        handle: TypeHandle::Int(42),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["module".to_owned()],
        }),
        name: "IntegerHandleClass".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_integer = GetMetaclassParams {
        snapshot: 5,
        type_: integer_handle_type,
    };

    // Verify handle types
    match &params_string.type_.handle {
        TypeHandle::String(s) => assert_eq!(s, "string_handle_123"),
        _ => panic!("Expected String handle"),
    }

    match &params_integer.type_.handle {
        TypeHandle::Int(i) => assert_eq!(*i, 42),
        _ => panic!("Expected Int handle"),
    }
}

#[test]
fn test_get_metaclass_params_with_declaration() {
    // Test with type that has declaration
    let type_with_decl = Type {
        alias_name: None,
        handle: TypeHandle::String("type_with_decl".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        }),
        name: "ClassWithDecl".to_owned(),
        category_flags: 0,
        decl: Some(Declaration {
            category: DeclarationCategory::Class,
            flags: DeclarationFlags::new(),
            handle: DeclarationHandle::String("decl_handle_class".to_owned()),
            module_name: ModuleName {
                leading_dots: 0,
                name_parts: vec!["test".to_owned()],
            },
            name: "ClassWithDecl".to_owned(),
            node: None,
            uri: "file:///test.py".to_owned(),
        }),
    };

    let params_with_decl = GetMetaclassParams {
        snapshot: 15,
        type_: type_with_decl,
    };

    // Test with type that has no declaration
    let type_no_decl = Type {
        alias_name: None,
        handle: TypeHandle::String("type_no_decl".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        }),
        name: "ClassNoDecl".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_no_decl = GetMetaclassParams {
        snapshot: 15,
        type_: type_no_decl,
    };

    // Verify declaration handling
    assert!(params_with_decl.type_.decl.is_some());
    let decl_ref = params_with_decl.type_.decl.as_ref().unwrap();
    assert_eq!(decl_ref.category, DeclarationCategory::Class);
    assert_eq!(decl_ref.name, "ClassWithDecl");

    assert!(params_no_decl.type_.decl.is_none());
}

#[test]
fn test_get_metaclass_params_snapshot_validation() {
    // Test with different snapshot values
    let type_param = Type {
        alias_name: None,
        handle: TypeHandle::String("snapshot_test".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        }),
        name: "SnapshotTestClass".to_owned(),
        category_flags: 0,
        decl: None,
    };

    // Test with zero snapshot
    let params_zero = GetMetaclassParams {
        snapshot: 0,
        type_: type_param.clone(),
    };

    // Test with positive snapshot
    let params_positive = GetMetaclassParams {
        snapshot: 12345,
        type_: type_param.clone(),
    };

    // Test with negative snapshot (should be valid in parameter construction)
    let params_negative = GetMetaclassParams {
        snapshot: -1,
        type_: type_param,
    };

    // Verify snapshot values
    assert_eq!(params_zero.snapshot, 0);
    assert_eq!(params_positive.snapshot, 12345);
    assert_eq!(params_negative.snapshot, -1);
}

#[test]
fn test_get_metaclass_request_different_id_types() {
    let type_param = Type {
        alias_name: None,
        handle: TypeHandle::String("id_test_type".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        }),
        name: "IdTestClass".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params = GetMetaclassParams {
        snapshot: 7,
        type_: type_param,
    };

    // Test with Number ID
    let request_number = GetMetaclassRequest {
        method: TSPRequestMethods::TypeServerGetMetaclass,
        id: LSPId::Int(42),
        params: params.clone(),
    };

    // Test with String ID
    let request_string = GetMetaclassRequest {
        method: TSPRequestMethods::TypeServerGetMetaclass,
        id: LSPId::String("metaclass-req-123".to_owned()),
        params,
    };

    // Verify different ID types
    match request_number.id {
        LSPId::Int(n) => assert_eq!(n, 42),
        _ => panic!("Expected Number id"),
    }

    match request_string.id {
        LSPId::String(s) => assert_eq!(s, "metaclass-req-123"),
        _ => panic!("Expected String id"),
    }
}

#[test]
fn test_get_metaclass_params_serialization() {
    // Test that parameters can be properly serialized and deserialized
    let original_params = GetMetaclassParams {
        snapshot: 999,
        type_: Type {
            alias_name: None,
            handle: TypeHandle::String("serialization_test".to_owned()),
            category: TypeCategory::Class,
            flags: TypeFlags::new().with_callable(),
            module_name: Some(ModuleName {
                leading_dots: 1,
                name_parts: vec!["serialization".to_owned(), "test".to_owned()],
            }),
            name: "SerializableClass".to_owned(),
            category_flags: 2,
            decl: Some(Declaration {
                category: DeclarationCategory::Class,
                flags: DeclarationFlags::new(),
                handle: DeclarationHandle::String("decl_handle_serialization_test".to_owned()),
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
    let deserialized_params: GetMetaclassParams =
        serde_json::from_str(&json_str).expect("Failed to deserialize");

    // Verify round-trip serialization
    assert_eq!(deserialized_params.snapshot, original_params.snapshot);
    assert_eq!(deserialized_params.type_.name, original_params.type_.name);
    assert_eq!(
        deserialized_params.type_.category,
        original_params.type_.category
    );
    assert_eq!(
        deserialized_params.type_.category_flags,
        original_params.type_.category_flags
    );

    match (
        &deserialized_params.type_.handle,
        &original_params.type_.handle,
    ) {
        (TypeHandle::String(d), TypeHandle::String(o)) => assert_eq!(d, o),
        _ => panic!("Handle type mismatch"),
    }

    let orig_module = original_params.type_.module_name.as_ref().unwrap();
    let deser_module = deserialized_params.type_.module_name.as_ref().unwrap();
    assert_eq!(deser_module.leading_dots, orig_module.leading_dots);
    assert_eq!(deser_module.name_parts, orig_module.name_parts);
}
