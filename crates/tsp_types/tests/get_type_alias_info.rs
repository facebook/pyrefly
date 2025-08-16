/*
 * Unit tests for GetTypeAliasInfoParams and TypeAliasInfo type construction and serialization
 */

use tsp_types::GetTypeAliasInfoParams;
use tsp_types::ModuleName;
use tsp_types::Type;
use tsp_types::TypeAliasInfo;
use tsp_types::TypeCategory;
use tsp_types::TypeFlags;
use tsp_types::TypeHandle;

#[test]
fn test_get_type_alias_info_params_construction() {
    // Test basic parameter construction
    let type_handle = TypeHandle::Int(42);
    let tsp_type = Type {
        alias_name: None,
        handle: type_handle.clone(),
        category: TypeCategory::Class,
        flags: TypeFlags::new().with_from_alias(), // Use FROM_ALIAS flag to indicate type alias
        module_name: None,
        name: "MyTypeAlias".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params = GetTypeAliasInfoParams {
        type_: tsp_type.clone(),
        snapshot: 123,
    };

    // Verify parameter construction
    if let TypeHandle::Int(handle_value) = &params.type_.handle {
        assert_eq!(*handle_value, 42);
    } else {
        panic!("Expected integer type handle");
    }
    assert_eq!(params.snapshot, 123);
    assert_eq!(params.type_.name, "MyTypeAlias");
    assert_eq!(params.type_.category, TypeCategory::Class);
}

#[test]
fn test_get_type_alias_info_params_different_handles() {
    // Test with different handle types

    // Test with string handle
    let string_type = Type {
        alias_name: None,
        handle: TypeHandle::String("alias_handle".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new().with_from_alias(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["typing".to_owned()],
        }),
        name: "List[str]".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params = GetTypeAliasInfoParams {
        type_: string_type,
        snapshot: 456,
    };

    assert_eq!(params.snapshot, 456);
    assert_eq!(params.type_.name, "List[str]");
    if let Some(module_name) = &params.type_.module_name {
        assert_eq!(module_name.name_parts, vec!["typing"]);
        assert_eq!(module_name.leading_dots, 0);
    }
}

#[test]
fn test_type_alias_info_creation() {
    // Test TypeAliasInfo struct creation and validation
    let str_type = Type {
        alias_name: None,
        handle: TypeHandle::Int(1),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: None,
        name: "str".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let type_alias_info = TypeAliasInfo {
        name: "MyList".to_owned(),
        type_args: Some(vec![str_type]),
    };

    assert_eq!(type_alias_info.name, "MyList");
    assert!(type_alias_info.type_args.is_some());
    assert_eq!(type_alias_info.type_args.unwrap().len(), 1);
}

#[test]
fn test_type_alias_info_no_type_arguments() {
    // Test TypeAliasInfo for non-generic type alias
    let type_alias_info = TypeAliasInfo {
        name: "SimpleAlias".to_owned(),
        type_args: None,
    };

    assert_eq!(type_alias_info.name, "SimpleAlias");
    assert!(type_alias_info.type_args.is_none());
}

#[test]
fn test_type_alias_info_multiple_type_arguments() {
    // Test TypeAliasInfo with multiple type arguments
    let str_type = Type {
        alias_name: None,
        handle: TypeHandle::Int(1),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: None,
        name: "str".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let int_type = Type {
        alias_name: None,
        handle: TypeHandle::Int(2),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: None,
        name: "int".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let type_alias_info = TypeAliasInfo {
        name: "MyDict".to_owned(),
        type_args: Some(vec![str_type, int_type]),
    };

    assert_eq!(type_alias_info.name, "MyDict");
    assert!(type_alias_info.type_args.is_some());
    let type_args = type_alias_info.type_args.unwrap();
    assert_eq!(type_args.len(), 2);
    assert_eq!(type_args[0].name, "str");
    assert_eq!(type_args[1].name, "int");
}

#[test]
fn test_params_serialization_structure() {
    // Test that the params structure matches expected JSON structure
    let type_handle = TypeHandle::String("test_handle".to_owned());
    let tsp_type = Type {
        alias_name: None,
        handle: type_handle,
        category: TypeCategory::Class,
        flags: TypeFlags::new().with_from_alias(),
        module_name: None,
        name: "TestAlias".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params = GetTypeAliasInfoParams {
        type_: tsp_type,
        snapshot: 789,
    };

    // Basic validation that the structure is correct
    assert_eq!(params.snapshot, 789);
    assert_eq!(params.type_.name, "TestAlias");

    // Test serialization/deserialization round-trip
    let serialized = serde_json::to_string(&params).expect("Should serialize");
    let deserialized: GetTypeAliasInfoParams =
        serde_json::from_str(&serialized).expect("Should deserialize");

    assert_eq!(deserialized.snapshot, params.snapshot);
    assert_eq!(deserialized.type_.name, params.type_.name);
    assert_eq!(deserialized.type_.category, params.type_.category);
}
