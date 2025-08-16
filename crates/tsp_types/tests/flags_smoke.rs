//! Smoke test to exercise all flag constants and builder helpers so they aren't optimized away.
use tsp_types::AttributeAccessFlags;
use tsp_types::AttributeFlags;
use tsp_types::ClassFlags;
use tsp_types::DeclarationFlags;
use tsp_types::FunctionFlags;
use tsp_types::INVALID_HANDLE;
use tsp_types::TypeFlags;
use tsp_types::TypeReprFlags;
use tsp_types::TypeVarFlags;

#[test]
fn test_tsp_all_flag_builders_and_constants() {
    // TypeFlags
    let tf = TypeFlags::new()
        .with_instantiable()
        .with_instance()
        .with_callable()
        .with_literal()
        .with_interface()
        .with_generic()
        .with_from_alias();
    assert!(tf.contains(TypeFlags::INSTANTIABLE));
    assert!(tf.contains(TypeFlags::INSTANCE));
    assert!(tf.contains(TypeFlags::CALLABLE));
    assert!(tf.contains(TypeFlags::LITERAL));
    assert!(tf.contains(TypeFlags::INTERFACE));
    assert!(tf.contains(TypeFlags::GENERIC));
    assert!(tf.contains(TypeFlags::FROM_ALIAS));

    // FunctionFlags
    let ff = FunctionFlags::new()
        .with_async()
        .with_generator()
        .with_abstract()
        .with_static();
    assert!(ff.contains(FunctionFlags::ASYNC));
    assert!(ff.contains(FunctionFlags::GENERATOR));
    assert!(ff.contains(FunctionFlags::ABSTRACT));
    assert!(ff.contains(FunctionFlags::STATIC));

    // ClassFlags
    let cf = ClassFlags::new().with_enum().with_typed_dict();
    assert!(cf.contains(ClassFlags::ENUM));
    assert!(cf.contains(ClassFlags::TYPED_DICT));

    // TypeVarFlags
    let tvf = TypeVarFlags::new().with_is_param_spec();
    assert!(tvf.contains(TypeVarFlags::IS_PARAM_SPEC));

    // AttributeFlags
    let af = AttributeFlags::new()
        .with_is_args_list()
        .with_is_kwargs_dict();
    assert!(af.contains(AttributeFlags::IS_ARGS_LIST));
    assert!(af.contains(AttributeFlags::IS_KWARGS_DICT));

    // AttributeAccessFlags
    let aaf = AttributeAccessFlags::new()
        .with_skip_instance_attributes()
        .with_skip_type_base_class()
        .with_skip_attribute_access_overrides()
        .with_get_bound_attributes();
    assert!(aaf.contains(AttributeAccessFlags::SKIP_INSTANCE_ATTRIBUTES));
    assert!(aaf.contains(AttributeAccessFlags::SKIP_TYPE_BASE_CLASS));
    assert!(aaf.contains(AttributeAccessFlags::SKIP_ATTRIBUTE_ACCESS_OVERRIDES));
    assert!(aaf.contains(AttributeAccessFlags::GET_BOUND_ATTRIBUTES));

    // DeclarationFlags
    let df = DeclarationFlags::new()
        .with_class_member()
        .with_constant()
        .with_final()
        .with_is_defined_by_slots()
        .with_uses_local_name()
        .with_unresolved_import();
    assert!(df.contains(DeclarationFlags::CLASS_MEMBER));
    assert!(df.contains(DeclarationFlags::CONSTANT));
    assert!(df.contains(DeclarationFlags::FINAL));
    assert!(df.contains(DeclarationFlags::IS_DEFINED_BY_SLOTS));
    assert!(df.contains(DeclarationFlags::USES_LOCAL_NAME));
    assert!(df.contains(DeclarationFlags::UNRESOLVED_IMPORT));

    // TypeReprFlags
    let trf = TypeReprFlags::new()
        .with_expand_type_aliases()
        .with_print_type_var_variance()
        .with_convert_to_instance_type();
    assert!(trf.contains(TypeReprFlags::EXPAND_TYPE_ALIASES));
    assert!(trf.contains(TypeReprFlags::PRINT_TYPE_VAR_VARIANCE));
    assert!(trf.contains(TypeReprFlags::CONVERT_TO_INSTANCE_TYPE));

    // Touch INVALID_HANDLE constant so it is considered used.
    let invalid = INVALID_HANDLE;
    assert_eq!(invalid, -1);

    // Exercise error_response to mark it used.
    let _resp = tsp_types::error_response(lsp_server::RequestId::from(0), 42, "msg".to_owned());
}
