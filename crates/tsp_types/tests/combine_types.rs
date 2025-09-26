/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Unit tests for TSP CombineTypesParams and CombineTypesRequest type construction and serialization
//!
//! These tests verify the CombineTypesParams parameter construction and validation by:
//! 1. Testing TSP CombineTypesParams construction with various parameter combinations
//! 2. Testing Vec<Type> parameter handling with different type combinations
//! 3. Testing snapshot validation logic
//! 4. Testing edge cases like empty type lists
//! 5. Testing serialization/deserialization round-trips

use tsp_types::CombineTypesParams;
use tsp_types::CombineTypesRequest;
use tsp_types::LSPId;
use tsp_types::ModuleName;
use tsp_types::TSPRequestMethods;
use tsp_types::Type;
use tsp_types::TypeCategory;
use tsp_types::TypeFlags;
use tsp_types::TypeHandle;

#[test]
fn test_combine_types_params_construction() {
    // Test basic parameter construction with multiple types
    let type1 = Type {
        alias_name: None,
        handle: TypeHandle::String("type1_handle".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["module1".to_owned()],
        }),
        name: "Type1".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let type2 = Type {
        alias_name: None,
        handle: TypeHandle::String("type2_handle".to_owned()),
        category: TypeCategory::Function,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["module2".to_owned()],
        }),
        name: "Type2".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params = CombineTypesParams {
        snapshot: 42,
        types: vec![type1.clone(), type2.clone()],
    };

    // Verify parameter construction
    assert_eq!(params.snapshot, 42);
    assert_eq!(params.types.len(), 2);
    assert_eq!(params.types[0].name, "Type1");
    assert_eq!(params.types[0].category, TypeCategory::Class);
    assert_eq!(params.types[1].name, "Type2");
    assert_eq!(params.types[1].category, TypeCategory::Function);
}

#[test]
fn test_combine_types_request_construction() {
    // Test basic request construction
    let type1 = Type {
        alias_name: None,
        handle: TypeHandle::String("request_type1".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        }),
        name: "RequestType1".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let type2 = Type {
        alias_name: None,
        handle: TypeHandle::String("request_type2".to_owned()),
        category: TypeCategory::Module,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        }),
        name: "RequestType2".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params = CombineTypesParams {
        snapshot: 10,
        types: vec![type1, type2],
    };

    let request = CombineTypesRequest {
        method: TSPRequestMethods::TypeServerCombineTypes,
        id: LSPId::Int(1),
        params,
    };

    // Verify request construction
    assert_eq!(request.method, TSPRequestMethods::TypeServerCombineTypes);
    match request.id {
        LSPId::Int(n) => assert_eq!(n, 1),
        _ => panic!("Expected Number id"),
    }
    assert_eq!(request.params.snapshot, 10);
    assert_eq!(request.params.types.len(), 2);
    assert_eq!(request.params.types[0].name, "RequestType1");
    assert_eq!(request.params.types[1].name, "RequestType2");
}

#[test]
fn test_combine_types_params_single_type() {
    // Test with single type (edge case)
    let single_type = Type {
        alias_name: None,
        handle: TypeHandle::String("single_type".to_owned()),
        category: TypeCategory::Any,
        flags: TypeFlags::new(),
        module_name: None,
        name: "SingleType".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_single = CombineTypesParams {
        snapshot: 5,
        types: vec![single_type],
    };

    // Verify single type handling
    assert_eq!(params_single.types.len(), 1);
    assert_eq!(params_single.types[0].name, "SingleType");
    assert_eq!(params_single.types[0].category, TypeCategory::Any);
}

#[test]
fn test_combine_types_params_empty_types() {
    // Test with empty types list (edge case)
    let params_empty = CombineTypesParams {
        snapshot: 1,
        types: vec![],
    };

    // Verify empty types handling
    assert_eq!(params_empty.types.len(), 0);
    assert!(params_empty.types.is_empty());
}

#[test]
fn test_combine_types_params_many_types() {
    // Test with many types
    let mut types = Vec::new();
    for i in 0..10 {
        types.push(Type {
            alias_name: None,
            handle: TypeHandle::String(format!("type_{i}_handle")),
            category: if i % 2 == 0 {
                TypeCategory::Class
            } else {
                TypeCategory::Function
            },
            flags: TypeFlags::new(),
            module_name: Some(ModuleName {
                leading_dots: 0,
                name_parts: vec![format!("module_{}", i)],
            }),
            name: format!("Type{i}"),
            category_flags: 0,
            decl: None,
        });
    }

    let params_many = CombineTypesParams {
        snapshot: 100,
        types: types.clone(),
    };

    // Verify many types handling
    assert_eq!(params_many.types.len(), 10);
    for (i, type_) in params_many.types.iter().enumerate() {
        assert_eq!(type_.name, format!("Type{i}"));
        assert_eq!(
            type_.category,
            if i % 2 == 0 {
                TypeCategory::Class
            } else {
                TypeCategory::Function
            }
        );
    }
}

#[test]
fn test_combine_types_params_different_handle_types() {
    // Test with different handle types
    let string_handle_type = Type {
        alias_name: None,
        handle: TypeHandle::String("string_handle".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        }),
        name: "StringHandleType".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let int_handle_type = Type {
        alias_name: None,
        handle: TypeHandle::Int(42),
        category: TypeCategory::Function,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        }),
        name: "IntHandleType".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_mixed_handles = CombineTypesParams {
        snapshot: 25,
        types: vec![string_handle_type, int_handle_type],
    };

    // Verify different handle types
    assert_eq!(params_mixed_handles.types.len(), 2);

    match &params_mixed_handles.types[0].handle {
        TypeHandle::String(s) => assert_eq!(s, "string_handle"),
        _ => panic!("Expected String handle"),
    }

    match &params_mixed_handles.types[1].handle {
        TypeHandle::Int(i) => assert_eq!(*i, 42),
        _ => panic!("Expected Int handle"),
    }
}

#[test]
fn test_combine_types_params_different_categories() {
    // Test combining different type categories
    let categories_and_names = [
        (TypeCategory::Class, "ClassType"),
        (TypeCategory::Function, "FunctionType"),
        (TypeCategory::Module, "ModuleType"),
        (TypeCategory::Any, "AnyType"),
        (TypeCategory::Any, "AnyType2"),
    ];

    let types: Vec<Type> = categories_and_names
        .iter()
        .enumerate()
        .map(|(i, (category, name))| Type {
            alias_name: None,
            handle: TypeHandle::String(format!("handle_{i}")),
            category: category.clone(),
            flags: TypeFlags::new(),
            module_name: Some(ModuleName {
                leading_dots: 0,
                name_parts: vec!["test".to_owned()],
            }),
            name: name.to_string(),
            category_flags: 0,
            decl: None,
        })
        .collect();

    let params_mixed_categories = CombineTypesParams {
        snapshot: 50,
        types,
    };

    // Verify different categories
    assert_eq!(params_mixed_categories.types.len(), 5);
    for (i, (expected_category, expected_name)) in categories_and_names.iter().enumerate() {
        assert_eq!(
            params_mixed_categories.types[i].category,
            *expected_category
        );
        assert_eq!(params_mixed_categories.types[i].name, *expected_name);
    }
}

#[test]
fn test_combine_types_request_different_id_types() {
    let type1 = Type {
        alias_name: None,
        handle: TypeHandle::String("id_test_type".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        }),
        name: "IdTestType".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params = CombineTypesParams {
        snapshot: 7,
        types: vec![type1],
    };

    // Test with Number ID
    let request_number = CombineTypesRequest {
        method: TSPRequestMethods::TypeServerCombineTypes,
        id: LSPId::Int(42),
        params: params.clone(),
    };

    // Test with String ID
    let request_string = CombineTypesRequest {
        method: TSPRequestMethods::TypeServerCombineTypes,
        id: LSPId::String("combine-types-req-123".to_owned()),
        params,
    };

    // Verify different ID types
    match request_number.id {
        LSPId::Int(n) => assert_eq!(n, 42),
        _ => panic!("Expected Number id"),
    }

    match request_string.id {
        LSPId::String(s) => assert_eq!(s, "combine-types-req-123"),
        _ => panic!("Expected String id"),
    }
}

#[test]
fn test_combine_types_params_serialization() {
    // Test that parameters can be properly serialized and deserialized
    let original_params = CombineTypesParams {
        snapshot: 999,
        types: vec![
            Type {
                alias_name: None,
                handle: TypeHandle::String("serialization_type1".to_owned()),
                category: TypeCategory::Class,
                flags: TypeFlags::new(),
                module_name: Some(ModuleName {
                    leading_dots: 1,
                    name_parts: vec!["serialization".to_owned(), "test".to_owned()],
                }),
                name: "SerializationType1".to_owned(),
                category_flags: 1,
                decl: None,
            },
            Type {
                alias_name: None,
                handle: TypeHandle::Int(123),
                category: TypeCategory::Function,
                flags: TypeFlags::new(),
                module_name: Some(ModuleName {
                    leading_dots: 0,
                    name_parts: vec!["other".to_owned()],
                }),
                name: "SerializationType2".to_owned(),
                category_flags: 2,
                decl: None,
            },
        ],
    };

    // Serialize to JSON
    let json_str = serde_json::to_string(&original_params).expect("Failed to serialize");

    // Deserialize back from JSON
    let deserialized_params: CombineTypesParams =
        serde_json::from_str(&json_str).expect("Failed to deserialize");

    // Verify round-trip serialization
    assert_eq!(deserialized_params.snapshot, original_params.snapshot);
    assert_eq!(deserialized_params.types.len(), original_params.types.len());

    for (i, (orig, deser)) in original_params
        .types
        .iter()
        .zip(deserialized_params.types.iter())
        .enumerate()
    {
        assert_eq!(deser.name, orig.name, "Type {i} name mismatch");
        assert_eq!(deser.category, orig.category, "Type {i} category mismatch");
        assert_eq!(
            deser.category_flags, orig.category_flags,
            "Type {i} category_flags mismatch"
        );

        match (&deser.handle, &orig.handle) {
            (TypeHandle::String(d), TypeHandle::String(o)) => {
                assert_eq!(d, o, "Type {i} string handle mismatch")
            }
            (TypeHandle::Int(d), TypeHandle::Int(o)) => {
                assert_eq!(d, o, "Type {i} int handle mismatch")
            }
            _ => panic!("Type {i} handle type mismatch"),
        }
    }
}
