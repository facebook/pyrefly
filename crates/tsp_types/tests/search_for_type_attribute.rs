/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests for SearchForTypeAttributeParams and related types construction and serialization

use tsp_types::AttributeAccessFlags;
use tsp_types::ModuleName;
use tsp_types::Node;
use tsp_types::Position;
use tsp_types::Range;
use tsp_types::SearchForTypeAttributeParams;
use tsp_types::Type;
use tsp_types::TypeCategory;
use tsp_types::TypeFlags;
use tsp_types::TypeHandle;

#[test]
fn test_search_for_type_attribute_params_construction() {
    // Test basic parameter construction
    let start_type = Type {
        alias_name: None,
        handle: TypeHandle::String("test_class_type".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["mymodule".to_owned()],
        }),
        name: "MyClass".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params = SearchForTypeAttributeParams {
        start_type: start_type.clone(),
        attribute_name: "my_method".to_owned(),
        access_flags: AttributeAccessFlags::NONE,
        expression_node: None,
        instance_type: None,
        snapshot: 42,
    };

    // Verify parameter construction
    assert_eq!(params.snapshot, 42);
    assert_eq!(params.attribute_name, "my_method");
    assert_eq!(params.start_type.name, "MyClass");
    assert_eq!(params.start_type.category, TypeCategory::Class);
    match &params.start_type.handle {
        TypeHandle::String(s) => assert_eq!(s, "test_class_type"),
        _ => panic!("Expected String handle"),
    }
    assert!(params.expression_node.is_none());
    assert!(params.instance_type.is_none());
}

#[test]
fn test_search_for_type_attribute_different_start_types() {
    // Test with CLASS category (expected case)
    let class_type = Type {
        alias_name: None,
        handle: TypeHandle::String("class_handle".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["models".to_owned()],
        }),
        name: "User".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_class = SearchForTypeAttributeParams {
        start_type: class_type,
        attribute_name: "name".to_owned(),
        access_flags: AttributeAccessFlags::NONE,
        expression_node: None,
        instance_type: None,
        snapshot: 1,
    };

    // Test with FUNCTION category (should not have attributes typically)
    let function_type = Type {
        alias_name: None,
        handle: TypeHandle::String("function_handle".to_owned()),
        category: TypeCategory::Function,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["utils".to_owned()],
        }),
        name: "helper_function".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_function = SearchForTypeAttributeParams {
        start_type: function_type,
        attribute_name: "__name__".to_owned(),
        access_flags: AttributeAccessFlags::NONE,
        expression_node: None,
        instance_type: None,
        snapshot: 1,
    };

    // Test with OVERLOADED category
    let overloaded_type = Type {
        alias_name: None,
        handle: TypeHandle::String("overloaded_handle".to_owned()),
        category: TypeCategory::Overloaded,
        flags: TypeFlags::new(),
        module_name: None,
        name: "overloaded_func".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_overloaded = SearchForTypeAttributeParams {
        start_type: overloaded_type,
        attribute_name: "__call__".to_owned(),
        access_flags: AttributeAccessFlags::NONE,
        expression_node: None,
        instance_type: None,
        snapshot: 1,
    };

    // Verify different categories
    assert_eq!(params_class.start_type.category, TypeCategory::Class);
    assert_eq!(params_function.start_type.category, TypeCategory::Function);
    assert_eq!(
        params_overloaded.start_type.category,
        TypeCategory::Overloaded
    );
}

#[test]
fn test_search_for_type_attribute_attribute_name_variants() {
    let base_type = Type {
        alias_name: None,
        handle: TypeHandle::String("test_type".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        }),
        name: "TestClass".to_owned(),
        category_flags: 0,
        decl: None,
    };

    // Test with regular method name
    let params_method = SearchForTypeAttributeParams {
        start_type: base_type.clone(),
        attribute_name: "regular_method".to_owned(),
        access_flags: AttributeAccessFlags::NONE,
        expression_node: None,
        instance_type: None,
        snapshot: 1,
    };

    // Test with dunder method
    let params_dunder = SearchForTypeAttributeParams {
        start_type: base_type.clone(),
        attribute_name: "__init__".to_owned(),
        access_flags: AttributeAccessFlags::NONE,
        expression_node: None,
        instance_type: None,
        snapshot: 1,
    };

    // Test with property
    let params_property = SearchForTypeAttributeParams {
        start_type: base_type.clone(),
        attribute_name: "my_property".to_owned(),
        access_flags: AttributeAccessFlags::NONE,
        expression_node: None,
        instance_type: None,
        snapshot: 1,
    };

    // Test with private attribute
    let params_private = SearchForTypeAttributeParams {
        start_type: base_type.clone(),
        attribute_name: "_private_attr".to_owned(),
        access_flags: AttributeAccessFlags::NONE,
        expression_node: None,
        instance_type: None,
        snapshot: 1,
    };

    // Test with mangled attribute
    let params_mangled = SearchForTypeAttributeParams {
        start_type: base_type.clone(),
        attribute_name: "__very_private".to_owned(),
        access_flags: AttributeAccessFlags::NONE,
        expression_node: None,
        instance_type: None,
        snapshot: 1,
    };

    // Verify attribute names
    assert_eq!(params_method.attribute_name, "regular_method");
    assert_eq!(params_dunder.attribute_name, "__init__");
    assert_eq!(params_property.attribute_name, "my_property");
    assert_eq!(params_private.attribute_name, "_private_attr");
    assert_eq!(params_mangled.attribute_name, "__very_private");
}

#[test]
fn test_search_for_type_attribute_access_flags() {
    let base_type = Type {
        alias_name: None,
        handle: TypeHandle::String("test_type".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        }),
        name: "TestClass".to_owned(),
        category_flags: 0,
        decl: None,
    };

    // Test with NONE flags
    let params_none = SearchForTypeAttributeParams {
        start_type: base_type.clone(),
        attribute_name: "test_attr".to_owned(),
        access_flags: AttributeAccessFlags::NONE,
        expression_node: None,
        instance_type: None,
        snapshot: 1,
    };

    // Test with SKIP_INSTANCE_ATTRIBUTES flag
    let params_skip_instance = SearchForTypeAttributeParams {
        start_type: base_type.clone(),
        attribute_name: "test_attr".to_owned(),
        access_flags: AttributeAccessFlags::SKIP_INSTANCE_ATTRIBUTES,
        expression_node: None,
        instance_type: None,
        snapshot: 1,
    };

    // Test with SKIP_TYPE_BASE_CLASS flag
    let params_skip_base = SearchForTypeAttributeParams {
        start_type: base_type.clone(),
        attribute_name: "test_attr".to_owned(),
        access_flags: AttributeAccessFlags::SKIP_TYPE_BASE_CLASS,
        expression_node: None,
        instance_type: None,
        snapshot: 1,
    };

    // Test with multiple flags combined
    let combined_flags = AttributeAccessFlags(
        AttributeAccessFlags::SKIP_INSTANCE_ATTRIBUTES.0
            | AttributeAccessFlags::SKIP_TYPE_BASE_CLASS.0,
    );
    let params_combined = SearchForTypeAttributeParams {
        start_type: base_type.clone(),
        attribute_name: "test_attr".to_owned(),
        access_flags: combined_flags,
        expression_node: None,
        instance_type: None,
        snapshot: 1,
    };

    // Verify flags - we can check the inner values
    assert_eq!(params_none.access_flags.0, 0);
    assert_eq!(params_skip_instance.access_flags.0, 1);
    assert_eq!(params_skip_base.access_flags.0, 2);
    assert_eq!(params_combined.access_flags.0, 3); // 1 | 2 = 3
}

#[test]
fn test_search_for_type_attribute_optional_parameters() {
    let base_type = Type {
        alias_name: None,
        handle: TypeHandle::String("test_type".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        }),
        name: "TestClass".to_owned(),
        category_flags: 0,
        decl: None,
    };

    // Test with expression_node provided
    let test_uri = "file:///test.py".to_string();
    let expression_node = Node {
        uri: test_uri.clone(),
        range: Range {
            start: Position {
                line: 10,
                character: 5,
            },
            end: Position {
                line: 10,
                character: 15,
            },
        },
    };

    let params_with_node = SearchForTypeAttributeParams {
        start_type: base_type.clone(),
        attribute_name: "test_attr".to_owned(),
        access_flags: AttributeAccessFlags::NONE,
        expression_node: Some(expression_node.clone()),
        instance_type: None,
        snapshot: 1,
    };

    // Test with instance_type provided
    let instance_type = Type {
        alias_name: None,
        handle: TypeHandle::Int(123),
        category: TypeCategory::Class,
        flags: TypeFlags::new().with_instance(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        }),
        name: "TestClassInstance".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_with_instance = SearchForTypeAttributeParams {
        start_type: base_type.clone(),
        attribute_name: "test_attr".to_owned(),
        access_flags: AttributeAccessFlags::NONE,
        expression_node: None,
        instance_type: Some(instance_type.clone()),
        snapshot: 1,
    };

    // Test with both optional parameters
    let params_with_both = SearchForTypeAttributeParams {
        start_type: base_type.clone(),
        attribute_name: "test_attr".to_owned(),
        access_flags: AttributeAccessFlags::NONE,
        expression_node: Some(expression_node.clone()),
        instance_type: Some(instance_type.clone()),
        snapshot: 1,
    };

    // Verify optional parameters
    assert!(params_with_node.expression_node.is_some());
    assert!(params_with_node.instance_type.is_none());

    assert!(params_with_instance.expression_node.is_none());
    assert!(params_with_instance.instance_type.is_some());

    assert!(params_with_both.expression_node.is_some());
    assert!(params_with_both.instance_type.is_some());

    // Verify node details
    let node = params_with_node.expression_node.as_ref().unwrap();
    assert_eq!(node.uri, test_uri);
    assert_eq!(node.range.start.line, 10);
    assert_eq!(node.range.start.character, 5);

    // Verify instance type details
    let instance = params_with_instance.instance_type.as_ref().unwrap();
    assert_eq!(instance.name, "TestClassInstance");
    match &instance.handle {
        TypeHandle::Int(i) => assert_eq!(*i, 123),
        _ => panic!("Expected Integer handle"),
    }
}

#[test]
fn test_search_for_type_attribute_type_handle_variants() {
    // Test with String handle
    let string_handle_type = Type {
        alias_name: None,
        handle: TypeHandle::String("string_handle_class".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["models".to_owned()],
        }),
        name: "StringHandleClass".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_string = SearchForTypeAttributeParams {
        start_type: string_handle_type,
        attribute_name: "string_attr".to_owned(),
        access_flags: AttributeAccessFlags::NONE,
        expression_node: None,
        instance_type: None,
        snapshot: 5,
    };

    // Test with Integer handle
    let integer_handle_type = Type {
        alias_name: None,
        handle: TypeHandle::Int(789),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["models".to_owned()],
        }),
        name: "IntegerHandleClass".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let params_integer = SearchForTypeAttributeParams {
        start_type: integer_handle_type,
        attribute_name: "integer_attr".to_owned(),
        access_flags: AttributeAccessFlags::NONE,
        expression_node: None,
        instance_type: None,
        snapshot: 5,
    };

    // Verify handle types
    match &params_string.start_type.handle {
        TypeHandle::String(s) => assert_eq!(s, "string_handle_class"),
        _ => panic!("Expected String handle"),
    }

    match &params_integer.start_type.handle {
        TypeHandle::Int(i) => assert_eq!(*i, 789),
        _ => panic!("Expected Integer handle"),
    }
    // Int handle already validated above
}

#[test]
fn test_search_for_type_attribute_module_name_variants() {
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
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: None,
        name: "BuiltinClass".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let type_simple = Type {
        alias_name: None,
        handle: TypeHandle::String("simple".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(simple_module.clone()),
        name: "SimpleClass".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let type_nested = Type {
        alias_name: None,
        handle: TypeHandle::String("nested".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(nested_module.clone()),
        name: "NestedClass".to_owned(),
        category_flags: 0,
        decl: None,
    };

    let type_relative = Type {
        alias_name: None,
        handle: TypeHandle::String("relative".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(relative_module.clone()),
        name: "RelativeClass".to_owned(),
        category_flags: 0,
        decl: None,
    };

    // Create params
    let params_no_module = SearchForTypeAttributeParams {
        start_type: type_no_module,
        attribute_name: "builtin_attr".to_owned(),
        access_flags: AttributeAccessFlags::NONE,
        expression_node: None,
        instance_type: None,
        snapshot: 1,
    };

    let params_simple = SearchForTypeAttributeParams {
        start_type: type_simple,
        attribute_name: "simple_attr".to_owned(),
        access_flags: AttributeAccessFlags::NONE,
        expression_node: None,
        instance_type: None,
        snapshot: 1,
    };

    let params_nested = SearchForTypeAttributeParams {
        start_type: type_nested,
        attribute_name: "nested_attr".to_owned(),
        access_flags: AttributeAccessFlags::NONE,
        expression_node: None,
        instance_type: None,
        snapshot: 1,
    };

    let params_relative = SearchForTypeAttributeParams {
        start_type: type_relative,
        attribute_name: "relative_attr".to_owned(),
        access_flags: AttributeAccessFlags::NONE,
        expression_node: None,
        instance_type: None,
        snapshot: 1,
    };

    // Verify module name handling
    assert!(params_no_module.start_type.module_name.is_none());

    let simple_mod = params_simple.start_type.module_name.as_ref().unwrap();
    assert_eq!(simple_mod.leading_dots, 0);
    assert_eq!(simple_mod.name_parts, vec!["simple"]);

    let nested_mod = params_nested.start_type.module_name.as_ref().unwrap();
    assert_eq!(nested_mod.leading_dots, 0);
    assert_eq!(nested_mod.name_parts, vec!["package", "submodule"]);

    let relative_mod = params_relative.start_type.module_name.as_ref().unwrap();
    assert_eq!(relative_mod.leading_dots, 2);
    assert_eq!(relative_mod.name_parts, vec!["relative"]);
}

#[test]
fn test_search_for_type_attribute_snapshot_validation() {
    let base_type = Type {
        alias_name: None,
        handle: TypeHandle::String("snapshot_test".to_owned()),
        category: TypeCategory::Class,
        flags: TypeFlags::new(),
        module_name: Some(ModuleName {
            leading_dots: 0,
            name_parts: vec!["test".to_owned()],
        }),
        name: "SnapshotClass".to_owned(),
        category_flags: 0,
        decl: None,
    };

    // Test with zero snapshot
    let params_zero = SearchForTypeAttributeParams {
        start_type: base_type.clone(),
        attribute_name: "test_attr".to_owned(),
        access_flags: AttributeAccessFlags::NONE,
        expression_node: None,
        instance_type: None,
        snapshot: 0,
    };

    // Test with positive snapshot
    let params_positive = SearchForTypeAttributeParams {
        start_type: base_type.clone(),
        attribute_name: "test_attr".to_owned(),
        access_flags: AttributeAccessFlags::NONE,
        expression_node: None,
        instance_type: None,
        snapshot: 12345,
    };

    // Test with negative snapshot (should be valid in parameter construction)
    let params_negative = SearchForTypeAttributeParams {
        start_type: base_type.clone(),
        attribute_name: "test_attr".to_owned(),
        access_flags: AttributeAccessFlags::NONE,
        expression_node: None,
        instance_type: None,
        snapshot: -1,
    };

    // Verify snapshot values
    assert_eq!(params_zero.snapshot, 0);
    assert_eq!(params_positive.snapshot, 12345);
    assert_eq!(params_negative.snapshot, -1);
}

#[test]
fn test_search_for_type_attribute_serialization_deserialization() {
    // Test that parameters can be properly serialized and deserialized
    let test_uri = "file:///test_serialization.py".to_string();
    let original_params = SearchForTypeAttributeParams {
        start_type: Type {
            alias_name: None,
            handle: TypeHandle::String("serialization_test".to_owned()),
            category: TypeCategory::Class,
            flags: TypeFlags::new().with_instantiable(),
            module_name: Some(ModuleName {
                leading_dots: 1,
                name_parts: vec!["serialization".to_owned(), "test".to_owned()],
            }),
            name: "SerializableClass".to_owned(),
            category_flags: 2,
            decl: None,
        },
        attribute_name: "serialize_method".to_owned(),
        access_flags: AttributeAccessFlags::SKIP_INSTANCE_ATTRIBUTES,
        expression_node: Some(Node {
            uri: test_uri.clone(),
            range: Range {
                start: Position {
                    line: 5,
                    character: 10,
                },
                end: Position {
                    line: 5,
                    character: 25,
                },
            },
        }),
        instance_type: Some(Type {
            alias_name: None,
            handle: TypeHandle::Int(456),
            category: TypeCategory::Class,
            flags: TypeFlags::new().with_instance(),
            module_name: Some(ModuleName {
                leading_dots: 0,
                name_parts: vec!["instance".to_owned()],
            }),
            name: "InstanceType".to_owned(),
            category_flags: 0,
            decl: None,
        }),
        snapshot: 999,
    };

    // Serialize to JSON
    let json_str = serde_json::to_string(&original_params).expect("Failed to serialize");

    // Deserialize back from JSON
    let deserialized_params: SearchForTypeAttributeParams =
        serde_json::from_str(&json_str).expect("Failed to deserialize");

    // Verify round-trip serialization
    assert_eq!(deserialized_params.snapshot, original_params.snapshot);
    assert_eq!(
        deserialized_params.attribute_name,
        original_params.attribute_name
    );
    assert_eq!(
        deserialized_params.start_type.name,
        original_params.start_type.name
    );
    assert_eq!(
        deserialized_params.start_type.category,
        original_params.start_type.category
    );
    assert_eq!(
        deserialized_params.start_type.category_flags,
        original_params.start_type.category_flags
    );
    assert_eq!(
        deserialized_params.access_flags.0,
        original_params.access_flags.0
    );

    match (
        &deserialized_params.start_type.handle,
        &original_params.start_type.handle,
    ) {
        (TypeHandle::String(d), TypeHandle::String(o)) => assert_eq!(d, o),
        _ => panic!("Handle type mismatch"),
    }

    let orig_module = original_params.start_type.module_name.as_ref().unwrap();
    let deser_module = deserialized_params.start_type.module_name.as_ref().unwrap();
    assert_eq!(deser_module.leading_dots, orig_module.leading_dots);
    assert_eq!(deser_module.name_parts, orig_module.name_parts);

    // Verify optional parameters
    assert!(deserialized_params.expression_node.is_some());
    assert!(deserialized_params.instance_type.is_some());

    let deser_node = deserialized_params.expression_node.as_ref().unwrap();
    let orig_node = original_params.expression_node.as_ref().unwrap();
    assert_eq!(deser_node.uri, orig_node.uri);
    assert_eq!(deser_node.range.start.line, orig_node.range.start.line);

    let deser_instance = deserialized_params.instance_type.as_ref().unwrap();
    let orig_instance = original_params.instance_type.as_ref().unwrap();
    assert_eq!(deser_instance.name, orig_instance.name);
}
