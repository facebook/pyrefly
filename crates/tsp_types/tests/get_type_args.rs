/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Unit tests for TSP GetTypeArgsParams and GetTypeArgsRequest type construction and serialization
//!
//! These tests verify the GetTypeArgsParams parameter construction and validation by:
//! 1. Testing TSP GetTypeArgsParams construction with various parameter combinations
//! 2. Testing Type parameter variations for generic types
//! 3. Testing snapshot validation logic
//! 4. Testing different type categories that may have type arguments
//! 5. Testing serialization/deserialization round-trips

use tsp_types::Declaration;
use tsp_types::DeclarationCategory;
use tsp_types::DeclarationFlags;
use tsp_types::DeclarationHandle;
use tsp_types::GetTypeArgsParams;
use tsp_types::GetTypeArgsRequest;
use tsp_types::LSPId;
use tsp_types::ModuleName;
use tsp_types::TSPRequestMethods;
use tsp_types::Type;
use tsp_types::TypeCategory;
use tsp_types::TypeFlags;
use tsp_types::TypeHandle;

#[test]
fn test_get_type_args_params_construction() {
    // Test basic parameter construction
    let type_param = Type {
        alias_name: None,
        handle: TypeHandle::String("generic_type".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["typing".to_owned()],
        }),
        name: "List".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params = GetTypeArgsParams {
        snapshot: 42,
        type_: type_param,
    };

    // Verify parameter construction
    assert_eq!(params.snapshot, 42);
    assert_eq!(params.type_.name, "List");
    assert_eq!(params.type_.category, TypeCategory::Class);
    match &params.type_.handle {
        TypeHandle::String(s) => assert_eq!(s, "generic_type"),
        _ => panic!("Expected String handle"),
    }
}

#[test]
fn test_get_type_args_request_construction() {
    // Test basic request construction
    let type_param = Type {
        alias_name: None,
        handle: TypeHandle::String("dict_type".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["builtins".to_owned()],
        }),
        name: "dict".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params = GetTypeArgsParams {
        snapshot: 10,
        type_: type_param,
    };

    let request = GetTypeArgsRequest {
        method: TSPRequestMethods::TypeServerGetTypeArgs,
        id: LSPId::Int(1),
        params,
    };

    // Verify request construction
    assert_eq!(request.method, TSPRequestMethods::TypeServerGetTypeArgs);
    match request.id {
        LSPId::Int(n) => assert_eq!(n, 1),
        _ => panic!("Expected Number id"),
    }
    assert_eq!(request.params.snapshot, 10);
    assert_eq!(request.params.type_.name, "dict");
    assert_eq!(request.params.type_.category, TypeCategory::Class);
}

#[test]
fn test_get_type_args_params_different_generic_types() {
    // Test with List type (common generic)
    let list_type = Type {
        alias_name: None,
        handle: TypeHandle::String("list_handle".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["builtins".to_owned()],
        }),
        name: "list".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_list = GetTypeArgsParams {
        snapshot: 1,
        type_: list_type,
    };

    // Test with Dict type (two type arguments)
    let dict_type = Type {
        alias_name: None,
        handle: TypeHandle::String("dict_handle".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["builtins".to_owned()],
        }),
        name: "dict".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_dict = GetTypeArgsParams {
        snapshot: 1,
        type_: dict_type,
    };

    // Test with Optional type (Union generic)
    let optional_type = Type {
        alias_name: None,
        handle: TypeHandle::String("optional_handle".to_owned()),
        category: TypeCategory::Union,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["typing".to_owned()],
        }),
        name: "Optional".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_optional = GetTypeArgsParams {
        snapshot: 1,
        type_: optional_type,
    };

    // Verify different generic types
    assert_eq!(params_list.type_.name, "list");
    assert_eq!(params_list.type_.category, TypeCategory::Class);
    assert_eq!(params_dict.type_.name, "dict");
    assert_eq!(params_dict.type_.category, TypeCategory::Class);
    assert_eq!(params_optional.type_.name, "Optional");
    assert_eq!(params_optional.type_.category, TypeCategory::Union);
}

#[test]
fn test_get_type_args_params_callable_types() {
    // Test with Callable type (function signature generic)
    let callable_type = Type {
        alias_name: None,
        handle: TypeHandle::String("callable_handle".to_owned()),
        category: TypeCategory::Function,
        flags: TypeFlags::new().with_callable(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["typing".to_owned()],
        }),
        name: "Callable".to_owned(),
        category_flags: 1,
        decl: None,
    };

    let params_callable = GetTypeArgsParams {
        snapshot: 15,
        type_: callable_type,
    };

    // Test with generic function type
    let generic_func_type = Type {
        alias_name: None,
        handle: TypeHandle::String("generic_func_handle".to_owned()),
        category: TypeCategory::Function,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["module".to_owned()],
        }),
        name: "generic_function".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_generic_func = GetTypeArgsParams {
        snapshot: 15,
        type_: generic_func_type,
    };

    // Verify callable type handling
    assert_eq!(params_callable.type_.name, "Callable");
    assert_eq!(params_callable.type_.category, TypeCategory::Function);
    assert_eq!(params_callable.type_.category_flags, 1);

    assert_eq!(params_generic_func.type_.name, "generic_function");
    assert_eq!(params_generic_func.type_.category, TypeCategory::Function);
}

#[test]
fn test_get_type_args_params_type_handle_variants() {
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
        name: "StringHandleGeneric".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_string = GetTypeArgsParams {
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
        name: "IntegerHandleGeneric".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_integer = GetTypeArgsParams {
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
fn test_get_type_args_params_with_declaration() {
    // Test with type that has declaration
    let type_with_decl = Type {
        alias_name: None,
        handle: TypeHandle::String("type_with_decl".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["typing".to_owned()],
        }),
        name: "Generic".to_owned(),
        category_flags: 0,
        decl: Some(Declaration {
            category: DeclarationCategory::Class,
            flags: DeclarationFlags::new(),
            handle: DeclarationHandle::String("decl_handle_generic".to_owned()),
            module_name: ModuleName {
                leading_dots: 0,
                name_parts: vec!["typing".to_owned()],
            },
            name: "Generic".to_owned(),
            node: None,
            uri: "file:///typing.py".to_owned(),
        }),
    };

    let params_with_decl = GetTypeArgsParams {
        snapshot: 20,
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
            name_parts: vec!["builtins".to_owned()],
        }),
        name: "tuple".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_no_decl = GetTypeArgsParams {
        snapshot: 20,
        type_: type_no_decl,
    };

    // Verify declaration handling
    assert!(params_with_decl.type_.decl.is_some());
    let decl_ref = params_with_decl.type_.decl.as_ref().unwrap();
    assert_eq!(decl_ref.category, DeclarationCategory::Class);
    assert_eq!(decl_ref.name, "Generic");

    assert!(params_no_decl.type_.decl.is_none());
}

#[test]
fn test_get_type_args_params_snapshot_validation() {
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
        name: "SnapshotTestGeneric".to_owned(),
        category_flags: 0,
        decl: None,
    };

    // Test with zero snapshot
    let params_zero = GetTypeArgsParams {
        snapshot: 0,
        type_: type_param.clone(),
    };

    // Test with positive snapshot
    let params_positive = GetTypeArgsParams {
        snapshot: 12345,
        type_: type_param.clone(),
    };

    // Test with negative snapshot (should be valid in parameter construction)
    let params_negative = GetTypeArgsParams {
        snapshot: -1,
        type_: type_param,
    };

    // Verify snapshot values
    assert_eq!(params_zero.snapshot, 0);
    assert_eq!(params_positive.snapshot, 12345);
    assert_eq!(params_negative.snapshot, -1);
}

#[test]
fn test_get_type_args_request_different_id_types() {
    let type_param = Type {
        alias_name: None,
        handle: TypeHandle::String("id_test_type".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        }),
        name: "IdTestGeneric".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params = GetTypeArgsParams {
        snapshot: 7,
        type_: type_param,
    };

    // Test with Number ID
    let request_number = GetTypeArgsRequest {
        method: TSPRequestMethods::TypeServerGetTypeArgs,
        id: LSPId::Int(42),
        params: params.clone(),
    };

    // Test with String ID
    let request_string = GetTypeArgsRequest {
        method: TSPRequestMethods::TypeServerGetTypeArgs,
        id: LSPId::String("type-args-req-123".to_owned()),
        params,
    };

    // Verify different ID types
    match request_number.id {
        LSPId::Int(n) => assert_eq!(n, 42),
        _ => panic!("Expected Number id"),
    }

    match request_string.id {
        LSPId::String(s) => assert_eq!(s, "type-args-req-123"),
        _ => panic!("Expected String id"),
    }
}

#[test]
fn test_get_type_args_params_serialization() {
    // Test that parameters can be properly serialized and deserialized
    let original_params = GetTypeArgsParams {
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
            name: "SerializableGeneric".to_owned(),
            category_flags: 2,
            decl: Some(Declaration {
                category: DeclarationCategory::Class,
                flags: DeclarationFlags::new(),
                handle: DeclarationHandle::String("decl_handle_serialization_test".to_owned()),
                module_name: ModuleName {
                    leading_dots: 1,
                    name_parts: vec!["serialization".to_owned(), "test".to_owned()],
                },
                name: "SerializableGeneric".to_owned(),
                node: None,
                uri: "file:///serialization/test.py".to_owned(),
            }),
        },
    };

    // Serialize to JSON
    let json_str = serde_json::to_string(&original_params).expect("Failed to serialize");

    // Deserialize back from JSON
    let deserialized_params: GetTypeArgsParams =
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
