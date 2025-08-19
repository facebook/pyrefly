/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests for TSP GetFunctionPartsParams type construction and serialization

use tsp_types::GetFunctionPartsParams;
use tsp_types::Type;
use tsp_types::TypeCategory;
use tsp_types::TypeFlags;
use tsp_types::TypeHandle;
use tsp_types::TypeReprFlags;

#[test]
fn test_get_function_parts_params_construction() {
    // Test basic parameter construction
    let type_handle = TypeHandle::Int(42);
    let tsp_type = Type {
        alias_name: None,
        handle: type_handle.clone(),
        category: TypeCategory::Function,
        flags: TypeFlags::new().with_callable(),
        module_name: None,
        name: "test_function".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params = GetFunctionPartsParams {
        type_: tsp_type.clone(),
        flags: TypeReprFlags::NONE,
        snapshot: 123,
    };

    // Verify parameter construction
    if let TypeHandle::Int(handle_value) = &params.type_.handle {
        assert_eq!(*handle_value, 42);
    } else {
        panic!("Expected integer type handle");
    }
    assert_eq!(params.snapshot, 123);
    assert_eq!(params.type_.name, "test_function");

    // Test with different flags
    let params_with_flags = GetFunctionPartsParams {
        type_: tsp_type.clone(),
        flags: TypeReprFlags::EXPAND_TYPE_ALIASES,
        snapshot: 456,
    };

    if let TypeHandle::Int(handle_value) = &params_with_flags.type_.handle {
        assert_eq!(*handle_value, 42);
    } else {
        panic!("Expected integer type handle");
    }
    assert!(params_with_flags.flags.has_expand_type_aliases());
    assert_eq!(params_with_flags.snapshot, 456);

    // Test parameter serialization/deserialization
    let json_str = serde_json::to_string(&params).unwrap();
    let deserialized: GetFunctionPartsParams = serde_json::from_str(&json_str).unwrap();
    assert_eq!(deserialized.type_.name, params.type_.name);
    assert_eq!(deserialized.snapshot, params.snapshot);
}

#[test]
fn test_get_function_parts_params_with_different_types() {
    // Test parameter construction for different function types
    let type1 = Type {
        alias_name: None,
        handle: TypeHandle::Int(100),
        category: TypeCategory::Function,
        flags: TypeFlags::new().with_callable(),
        module_name: None,
        name: "simple_func".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let type2 = Type {
        alias_name: None,
        handle: TypeHandle::String("async_func_handle".to_owned()),
        category: TypeCategory::Function,
        flags: TypeFlags::new().with_callable().with_instance(),
        module_name: None,
        name: "async_func".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let type3 = Type {
        alias_name: None,
        handle: TypeHandle::Int(300),
        category: TypeCategory::Overloaded,
        flags: TypeFlags::new().with_callable(),
        module_name: None,
        name: "overloaded_func".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params1 = GetFunctionPartsParams {
        type_: type1,
        flags: TypeReprFlags::NONE,
        snapshot: 1,
    };

    let params2 = GetFunctionPartsParams {
        type_: type2,
        flags: TypeReprFlags::CONVERT_TO_INSTANCE_TYPE,
        snapshot: 2,
    };

    let params3 = GetFunctionPartsParams {
        type_: type3,
        flags: TypeReprFlags::PRINT_TYPE_VAR_VARIANCE,
        snapshot: 3,
    };

    // Verify each parameter set is distinct
    assert_eq!(params1.type_.name, "simple_func");
    assert_eq!(params2.type_.name, "async_func");
    assert_eq!(params3.type_.name, "overloaded_func");

    assert_eq!(params1.snapshot, 1);
    assert_eq!(params2.snapshot, 2);
    assert_eq!(params3.snapshot, 3);

    // Test different flag combinations
    assert!(!params1.flags.has_expand_type_aliases());
    assert!(params2.flags.has_convert_to_instance_type());
    assert!(params3.flags.has_print_type_var_variance());
}
