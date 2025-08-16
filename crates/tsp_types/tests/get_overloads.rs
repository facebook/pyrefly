/*
 * Unit tests for TSP GetOverloadsParams type construction and serialization
 *
 * These tests verify the GetOverloadsParams parameter construction and validation by:
 * 1. Testing TSP GetOverloadsParams construction with various parameter combinations
 * 2. Validating proper handling of different Type parameter variations
 * 3. Testing snapshot validation logic
 * 4. Testing TypeHandle variations (String and Integer handles)
 * 5. Testing different type categories and flags combinations
 * 6. Testing module name handling and edge cases
 * 7. Testing serialization/deserialization round-trips
 */

use tsp_types::Declaration;
use tsp_types::DeclarationCategory;
use tsp_types::DeclarationFlags;
use tsp_types::DeclarationHandle;
use tsp_types::GetOverloadsParams;
use tsp_types::ModuleName;
use tsp_types::Node;
use tsp_types::Position;
use tsp_types::Range;
use tsp_types::Type;
use tsp_types::TypeCategory;
use tsp_types::TypeFlags;
use tsp_types::TypeHandle;

#[test]
fn test_get_overloads_params_construction() {
    // Test basic parameter construction
    let type_param = Type {
        alias_name: None,
        handle: TypeHandle::String("test_overloaded_function".to_owned()),
        category: TypeCategory::Overloaded,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["mymodule".to_owned()],
        }),
        name: "overloaded_func".to_owned(),
        category_flags: 0,
        decl: Some(Declaration {
            category: DeclarationCategory::Function,
            flags: DeclarationFlags::new(),
            handle: DeclarationHandle::String("decl_handle_overloaded_func".to_owned()),
            module_name: ModuleName {
                leading_dots: 0,
                name_parts: vec!["mymodule".to_owned()],
            },
            name: "overloaded_func".to_owned(),
            node: None,
            uri: "file:///mymodule.py".to_owned(),
        }),
    };

    let params = GetOverloadsParams {
        type_: type_param.clone(),
        snapshot: 42,
    };

    // Verify parameter construction
    assert_eq!(params.snapshot, 42);
    assert_eq!(params.type_.name, "overloaded_func");
    assert_eq!(params.type_.category, TypeCategory::Overloaded);
    match &params.type_.handle {
        TypeHandle::String(s) => assert_eq!(s, "test_overloaded_function"),
        _ => panic!("Expected String handle"),
    }
}

#[test]
fn test_get_overloads_different_type_categories() {
    // Test with OVERLOADED category (expected case)
    let overloaded_type = Type {
        alias_name: None,
        handle: TypeHandle::String("overloaded_handle".to_owned()),
        category: TypeCategory::Overloaded,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        }),
        name: "overloaded_function".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_overloaded = GetOverloadsParams {
        type_: overloaded_type,
        snapshot: 1,
    };

    // Test with FUNCTION category (should not have overloads)
    let function_type = Type {
        alias_name: None,
        handle: TypeHandle::String("function_handle".to_owned()),
        category: TypeCategory::Function,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        }),
        name: "simple_function".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_function = GetOverloadsParams {
        type_: function_type,
        snapshot: 1,
    };

    // Test with ANY category (should not have overloads)
    let any_type = Type {
        alias_name: None,
        handle: TypeHandle::String("any_handle".to_owned()),
        category: TypeCategory::Any,
        flags: TypeFlags::new(),
        module_name: None,
        name: "any_type".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_any = GetOverloadsParams {
        type_: any_type,
        snapshot: 1,
    };

    // Verify different categories
    assert_eq!(params_overloaded.type_.category, TypeCategory::Overloaded);
    assert_eq!(params_function.type_.category, TypeCategory::Function);
    assert_eq!(params_any.type_.category, TypeCategory::Any);
}

#[test]
fn test_get_overloads_type_handle_variants() {
    // Test with String handle
    let string_handle_type = Type {
        alias_name: None,
        handle: TypeHandle::String("string_handle_123".to_owned()),
        category: TypeCategory::Overloaded,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["module".to_owned()],
        }),
        name: "string_handle_func".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_string = GetOverloadsParams {
        type_: string_handle_type,
        snapshot: 5,
    };

    // Test with Integer handle
    let integer_handle_type = Type {
        alias_name: None,
        handle: TypeHandle::Int(42),
        category: TypeCategory::Overloaded,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["module".to_owned()],
        }),
        name: "integer_handle_func".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_integer = GetOverloadsParams {
        type_: integer_handle_type,
        snapshot: 5,
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
fn test_get_overloads_module_name_variants() {
    // Test with simple module name
    let simple_module = ModuleName {
        leading_dots: 0,
        name_parts: vec!["simple".to_owned()],
    };

    // Test with nested module name
    let nested_module = ModuleName {
        leading_dots: 0,
        name_parts: vec!["package".to_owned(), "submodule".to_owned()],
    };

    // Test with relative import (leading dots)
    let relative_module = ModuleName {
        leading_dots: 2,
        name_parts: vec!["relative".to_owned()],
    };

    // Test with no module name (None)
    let type_no_module = Type {
        alias_name: None,
        handle: TypeHandle::String("no_module".to_owned()),
        category: TypeCategory::Overloaded,
        flags: TypeFlags::new(),
        module_name: None,
        name: "builtin_function".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let type_simple = Type {
        alias_name: None,
        handle: TypeHandle::String("simple".to_owned()),
        category: TypeCategory::Overloaded,
        flags: TypeFlags::new(),
        module_name: Some(simple_module.clone()),
        name: "simple_func".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let type_nested = Type {
        alias_name: None,
        handle: TypeHandle::String("nested".to_owned()),
        category: TypeCategory::Overloaded,
        flags: TypeFlags::new(),
        module_name: Some(nested_module.clone()),
        name: "nested_func".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let type_relative = Type {
        alias_name: None,
        handle: TypeHandle::String("relative".to_owned()),
        category: TypeCategory::Overloaded,
        flags: TypeFlags::new(),
        module_name: Some(relative_module.clone()),
        name: "relative_func".to_owned(),
        category_flags: 0,
        decl: None,
    };

    // Create params
    let params_no_module = GetOverloadsParams {
        type_: type_no_module,
        snapshot: 1,
    };
    let params_simple = GetOverloadsParams {
        type_: type_simple,
        snapshot: 1,
    };
    let params_nested = GetOverloadsParams {
        type_: type_nested,
        snapshot: 1,
    };
    let params_relative = GetOverloadsParams {
        type_: type_relative,
        snapshot: 1,
    };

    // Verify module name handling
    assert!(params_no_module.type_.module_name.is_none());

    let simple_mod = params_simple.type_.module_name.as_ref().unwrap();
    assert_eq!(simple_mod.leading_dots, 0);
    assert_eq!(simple_mod.name_parts, vec!["simple"]);

    let nested_mod = params_nested.type_.module_name.as_ref().unwrap();
    assert_eq!(nested_mod.leading_dots, 0);
    assert_eq!(nested_mod.name_parts, vec!["package", "submodule"]);

    let relative_mod = params_relative.type_.module_name.as_ref().unwrap();
    assert_eq!(relative_mod.leading_dots, 2);
    assert_eq!(relative_mod.name_parts, vec!["relative"]);
}

#[test]
fn test_get_overloads_flags_handling() {
    // Test with no flags
    let no_flags_type = Type {
        alias_name: None,
        handle: TypeHandle::String("no_flags".to_owned()),
        category: TypeCategory::Overloaded,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        }),
        name: "no_flags_func".to_owned(),
        category_flags: 0,
        decl: None,
    };

    // Test with some flags set
    let with_flags_type = Type {
        alias_name: None,
        handle: TypeHandle::String("with_flags".to_owned()),
        category: TypeCategory::Overloaded,
        flags: TypeFlags::new().with_callable(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        }),
        name: "callable_func".to_owned(),
        category_flags: 1,
        decl: None,
    };

    let params_no_flags = GetOverloadsParams {
        type_: no_flags_type,
        snapshot: 10,
    };
    let params_with_flags = GetOverloadsParams {
        type_: with_flags_type,
        snapshot: 10,
    };

    // Verify flag handling - just verify construction works properly
    // We can't compare TypeFlags directly since it doesn't implement PartialEq
    let _no_flags = params_no_flags.type_.flags;
    let _with_flags = params_with_flags.type_.flags;
    assert_eq!(params_no_flags.type_.category_flags, 0);
    assert_eq!(params_with_flags.type_.category_flags, 1);
}

#[test]
fn test_get_overloads_declaration_handling() {
    // Test with null declaration
    let null_decl_type = Type {
        alias_name: None,
        handle: TypeHandle::String("null_decl".to_owned()),
        category: TypeCategory::Overloaded,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        }),
        name: "null_decl_func".to_owned(),
        category_flags: 0,
        decl: None,
    };

    // Test with simple declaration
    let simple_decl_type = Type {
        alias_name: None,
        handle: TypeHandle::String("simple_decl".to_owned()),
        category: TypeCategory::Overloaded,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        }),
        name: "simple_decl_func".to_owned(),
        category_flags: 0,
        decl: Some(Declaration {
            category: DeclarationCategory::Function,
            flags: DeclarationFlags::new(),
            handle: DeclarationHandle::String("decl_handle_simple".to_owned()),
            module_name: ModuleName {
                leading_dots: 0,
                name_parts: vec!["test".to_owned()],
            },
            name: "simple_decl_func".to_owned(),
            node: Some(Node {
                uri: "file:///test.py".to_owned(),
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
            uri: "file:///test.py".to_owned(),
        }),
    };

    // Test with complex declaration
    let complex_decl_type = Type {
        alias_name: None,
        handle: TypeHandle::String("complex_decl".to_owned()),
        category: TypeCategory::Overloaded,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["complex".to_owned(), "module".to_owned()],
        }),
        name: "complex_func".to_owned(),
        category_flags: 0,
        decl: Some(Declaration {
            category: DeclarationCategory::Function,
            flags: DeclarationFlags::new(),
            handle: DeclarationHandle::String("decl_handle_complex".to_owned()),
            module_name: ModuleName {
                leading_dots: 0,
                name_parts: vec!["complex".to_owned(), "module".to_owned()],
            },
            name: "complex_func".to_owned(),
            node: None,
            uri: "file:///complex/module.py".to_owned(),
        }),
    };

    let params_null = GetOverloadsParams {
        type_: null_decl_type,
        snapshot: 20,
    };
    let params_simple = GetOverloadsParams {
        type_: simple_decl_type,
        snapshot: 20,
    };
    let params_complex = GetOverloadsParams {
        type_: complex_decl_type,
        snapshot: 20,
    };

    // Verify declaration handling
    assert!(params_null.type_.decl.is_none());

    let simple_decl = params_simple.type_.decl.as_ref().unwrap();
    assert_eq!(simple_decl.category, DeclarationCategory::Function);
    assert_eq!(simple_decl.name, "simple_decl_func");

    let complex_decl = params_complex.type_.decl.as_ref().unwrap();
    assert_eq!(complex_decl.category, DeclarationCategory::Function);
    assert_eq!(complex_decl.name, "complex_func");
}

#[test]
fn test_get_overloads_snapshot_validation() {
    // Test with different snapshot values
    let type_param = Type {
        alias_name: None,
        handle: TypeHandle::String("snapshot_test".to_owned()),
        category: TypeCategory::Overloaded,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        }),
        name: "snapshot_func".to_owned(),
        category_flags: 0,
        decl: None,
    };

    // Test with zero snapshot
    let params_zero = GetOverloadsParams {
        type_: type_param.clone(),
        snapshot: 0,
    };

    // Test with positive snapshot
    let params_positive = GetOverloadsParams {
        type_: type_param.clone(),
        snapshot: 12345,
    };

    // Test with negative snapshot (should be valid in parameter construction)
    let params_negative = GetOverloadsParams {
        type_: type_param.clone(),
        snapshot: -1,
    };

    // Verify snapshot values
    assert_eq!(params_zero.snapshot, 0);
    assert_eq!(params_positive.snapshot, 12345);
    assert_eq!(params_negative.snapshot, -1);
}

#[test]
fn test_get_overloads_serialization_deserialization() {
    // Test that parameters can be properly serialized and deserialized
    let original_params = GetOverloadsParams {
        type_: Type {
            alias_name: None,
            handle: TypeHandle::String("serialization_test".to_owned()),
            category: TypeCategory::Overloaded,
            flags: TypeFlags::new().with_callable(),
            module_name: Some(ModuleName {
                leading_dots: 1,
                name_parts: vec!["serialization".to_owned(), "test".to_owned()],
            }),
            name: "serializable_func".to_owned(),
            category_flags: 2,
            decl: Some(Declaration {
                category: DeclarationCategory::Function,
                flags: DeclarationFlags::new(),
                handle: DeclarationHandle::String("decl_handle_serialization_test".to_owned()),
                module_name: ModuleName {
                    leading_dots: 1,
                    name_parts: vec!["serialization".to_owned(), "test".to_owned()],
                },
                name: "serializable_func".to_owned(),
                node: None,
                uri: "file:///serialization/test.py".to_owned(),
            }),
        },
        snapshot: 999,
    };

    // Serialize to JSON
    let json_str = serde_json::to_string(&original_params).expect("Failed to serialize");

    // Deserialize back from JSON
    let deserialized_params: GetOverloadsParams =
        serde_json::from_str(&json_str).expect("Failed to deserialize");

    // Verify round-trip serialization
    assert_eq!(deserialized_params.snapshot, original_params.snapshot);
    assert_eq!(deserialized_params.type_.name, original_params.type_.name);
    assert_eq!(
        deserialized_params.type_.category,
        original_params.type_.category
    );
    // Note: TypeFlags doesn't implement PartialEq so we can't directly compare
    // but serialization/deserialization should preserve the flag structure
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
