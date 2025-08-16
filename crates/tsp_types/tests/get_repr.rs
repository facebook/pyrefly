/*
 * Unit tests for TSP GetReprParams type construction and serialization
 */

use tsp_types::GetReprParams;
use tsp_types::Type;
use tsp_types::TypeCategory;
use tsp_types::TypeFlags;
use tsp_types::TypeHandle;
use tsp_types::TypeReprFlags;

#[test]
fn test_basic_get_repr_params_construction() {
    let params = GetReprParams {
        type_: Type {
            handle: TypeHandle::String("test".to_owned()),
            category: TypeCategory::Any,
            flags: TypeFlags::new(),
            module_name: None,
            name: "str".to_owned(),
            category_flags: 0,
            decl: None,
            alias_name: None,
        },
        flags: TypeReprFlags::NONE,
        snapshot: 1,
    };

    // Verify parameter construction
    assert_eq!(params.snapshot, 1);
    assert_eq!(params.type_.name, "str");
    assert_eq!(params.type_.category, TypeCategory::Any);
    match &params.type_.handle {
        TypeHandle::String(s) => assert_eq!(s, "test"),
        _ => panic!("Expected String handle"),
    }
}

#[test]
fn test_get_repr_params_with_flags() {
    let type_obj = Type {
        handle: TypeHandle::Int(42),
        category: TypeCategory::Class,
        flags: TypeFlags::new().with_instantiable(),
        module_name: None,
        name: "MyClass".to_owned(),
        category_flags: 0,
        decl: None,
        alias_name: None,
    };

    let params = GetReprParams {
        type_: type_obj,
        flags: TypeReprFlags::EXPAND_TYPE_ALIASES | TypeReprFlags::CONVERT_TO_INSTANCE_TYPE,
        snapshot: 100,
    };

    // Verify parameter construction with flags
    assert_eq!(params.snapshot, 100);
    assert_eq!(params.type_.name, "MyClass");
    assert!(params.flags.has_expand_type_aliases());
    assert!(params.flags.has_convert_to_instance_type());
    match &params.type_.handle {
        TypeHandle::Int(i) => assert_eq!(*i, 42),
        _ => panic!("Expected Int handle"),
    }
}

#[test]
fn test_get_repr_params_serialization() {
    let params = GetReprParams {
        type_: Type {
            handle: TypeHandle::String("serialization_test".to_owned()),
            category: TypeCategory::Function,
            flags: TypeFlags::new().with_callable(),
            module_name: None,
            name: "test_func".to_owned(),
            category_flags: 0,
            decl: None,
            alias_name: None,
        },
        flags: TypeReprFlags::PRINT_TYPE_VAR_VARIANCE,
        snapshot: 42,
    };

    // Test serialization round-trip
    let json_str = serde_json::to_string(&params).expect("Failed to serialize");
    let deserialized: GetReprParams =
        serde_json::from_str(&json_str).expect("Failed to deserialize");

    // Verify round-trip
    assert_eq!(deserialized.snapshot, params.snapshot);
    assert_eq!(deserialized.type_.name, params.type_.name);
    assert_eq!(deserialized.type_.category, params.type_.category);

    match (&deserialized.type_.handle, &params.type_.handle) {
        (TypeHandle::String(d), TypeHandle::String(o)) => assert_eq!(d, o),
        _ => panic!("Handle type mismatch"),
    }
}
