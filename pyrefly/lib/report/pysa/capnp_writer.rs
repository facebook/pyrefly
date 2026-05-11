/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Serializes pysa report structs into Cap'n Proto binary format.

use std::io::Write;

use pyrefly_python::module_path::ModulePathDetails;

use super::PysaModuleCallGraphs;
use super::PysaModuleDefinitions;
use super::PysaModuleTypeOfExpressions;
use super::PysaProjectFile;
use super::PysaTypeErrorsFile;
use super::pysa_report_capnp;
use crate::report::pysa::call_graph::AttributeAccessCallees;
use crate::report::pysa::call_graph::CallCallees;
use crate::report::pysa::call_graph::CallGraph;
use crate::report::pysa::call_graph::DefineCallees;
use crate::report::pysa::call_graph::ExpressionCallees;
use crate::report::pysa::call_graph::ExpressionIdentifier;
use crate::report::pysa::call_graph::FormatStringArtificialCallees;
use crate::report::pysa::call_graph::FormatStringStringifyCallees;
use crate::report::pysa::call_graph::HigherOrderParameter;
use crate::report::pysa::call_graph::IdentifierCallees;
use crate::report::pysa::call_graph::ImplicitReceiver;
use crate::report::pysa::call_graph::PysaCallTarget;
use crate::report::pysa::call_graph::ReturnShimArgumentMapping;
use crate::report::pysa::call_graph::ReturnShimCallees;
use crate::report::pysa::call_graph::Target;
use crate::report::pysa::call_graph::Unresolved;
use crate::report::pysa::call_graph::UnresolvedReason;
use crate::report::pysa::captured_variable::CapturedVariableRef;
use crate::report::pysa::class::ClassDefinition;
use crate::report::pysa::class::ClassRef;
use crate::report::pysa::class::PysaClassField;
use crate::report::pysa::class::PysaClassFieldDeclaration;
use crate::report::pysa::class::PysaClassMro;
use crate::report::pysa::function::FunctionDefinition;
use crate::report::pysa::function::FunctionId;
use crate::report::pysa::function::FunctionParameter;
use crate::report::pysa::function::FunctionParameters;
use crate::report::pysa::function::FunctionRef;
use crate::report::pysa::function::FunctionSignature;
use crate::report::pysa::location::PysaLocation;
use crate::report::pysa::scope::ScopeParent;
use crate::report::pysa::types::ClassNamesFromType;
use crate::report::pysa::types::ClassWithModifiers;
use crate::report::pysa::types::PysaType;
use crate::report::pysa::types::ScalarTypeProperties;
use crate::report::pysa::types::TypeModifier;

fn set_source_path(mut builder: pysa_report_capnp::source_path::Builder, path: &ModulePathDetails) {
    let path_str = |p: &std::path::PathBuf| p.to_string_lossy().into_owned();
    match path {
        ModulePathDetails::FileSystem(p) => builder.set_file_system(path_str(p)),
        ModulePathDetails::Namespace(p) => builder.set_namespace(path_str(p)),
        ModulePathDetails::Memory(p) => builder.set_memory(path_str(p)),
        ModulePathDetails::BundledTypeshed(p) => builder.set_bundled_typeshed(path_str(p)),
        ModulePathDetails::BundledTypeshedThirdParty(p) => {
            builder.set_bundled_typeshed_third_party(path_str(p))
        }
        ModulePathDetails::BundledThirdParty(p) => builder.set_bundled_third_party(path_str(p)),
    }
}

fn set_location(mut builder: pysa_report_capnp::pysa_location::Builder, loc: &PysaLocation) {
    builder.set_line(loc.line());
    builder.set_col(loc.col());
    builder.set_end_line(loc.end_line());
    builder.set_end_col(loc.end_col());
}

fn set_class_ref(mut builder: pysa_report_capnp::class_ref::Builder, class_ref: &ClassRef) {
    builder.set_module_id(class_ref.module_id.to_int());
    builder.set_class_id(class_ref.class_id.to_int());
}

fn set_function_ref(mut builder: pysa_report_capnp::function_ref::Builder, func_ref: &FunctionRef) {
    builder.set_module_id(func_ref.module_id.to_int());
    builder.set_function_id(func_ref.function_id.serialize_to_string());
}

fn set_global_variable_ref(
    mut builder: pysa_report_capnp::global_variable_ref::Builder,
    global_variable_ref: &crate::report::pysa::global_variable::GlobalVariableRef,
) {
    builder.set_module_id(global_variable_ref.module_id.to_int());
    builder.set_name(global_variable_ref.name.as_str());
}

fn set_scalar_type_properties(
    mut builder: pysa_report_capnp::scalar_type_properties::Builder,
    properties: ScalarTypeProperties,
) {
    builder.set_is_bool(properties.is_bool);
    builder.set_is_int(properties.is_int);
    builder.set_is_float(properties.is_float);
    builder.set_is_enum(properties.is_enum);
}

fn convert_type_modifier(modifier: TypeModifier) -> pysa_report_capnp::TypeModifier {
    match modifier {
        TypeModifier::Optional => pysa_report_capnp::TypeModifier::Optional,
        TypeModifier::Coroutine => pysa_report_capnp::TypeModifier::Coroutine,
        TypeModifier::Awaitable => pysa_report_capnp::TypeModifier::Awaitable,
        TypeModifier::TypeVariableBound => pysa_report_capnp::TypeModifier::TypeVariableBound,
        TypeModifier::TypeVariableConstraint => {
            pysa_report_capnp::TypeModifier::TypeVariableConstraint
        }
        TypeModifier::Type => pysa_report_capnp::TypeModifier::Type,
    }
}

fn set_class_with_modifiers(
    mut builder: pysa_report_capnp::class_with_modifiers::Builder,
    class_with_modifiers: &ClassWithModifiers,
) {
    set_class_ref(builder.reborrow().init_class(), &class_with_modifiers.class);
    let mut modifiers = builder.init_modifiers(class_with_modifiers.modifiers.len() as u32);
    for (i, modifier) in class_with_modifiers.modifiers.iter().copied().enumerate() {
        modifiers.set(i as u32, convert_type_modifier(modifier));
    }
}

fn set_class_names(
    mut builder: pysa_report_capnp::class_names_from_type::Builder,
    class_names: &ClassNamesFromType,
) {
    let mut classes = builder
        .reborrow()
        .init_classes(class_names.classes.len() as u32);
    for (i, class_with_modifiers) in class_names.classes.iter().enumerate() {
        set_class_with_modifiers(classes.reborrow().get(i as u32), class_with_modifiers);
    }
    builder.set_is_exhaustive(class_names.is_exhaustive);
}

fn set_pysa_type(mut builder: pysa_report_capnp::pysa_type::Builder, ty: &PysaType) {
    builder.reborrow().set_string(&ty.string);
    set_scalar_type_properties(
        builder.reborrow().init_scalar_type_properties(),
        ty.scalar_type_properties,
    );
    if !ty.class_names.classes.is_empty() {
        set_class_names(builder.init_class_names(), &ty.class_names);
    }
}

fn set_scope_parent(mut builder: pysa_report_capnp::scope_parent::Builder, parent: &ScopeParent) {
    match parent {
        ScopeParent::Function { func_def_index } => {
            builder.set_function(func_def_index.0);
        }
        ScopeParent::Class { class_id } => {
            builder.set_class(class_id.to_int());
        }
        ScopeParent::TopLevel => {
            builder.set_top_level(());
        }
    }
}

fn set_function_parameter(
    builder: pysa_report_capnp::function_parameter::Builder,
    param: &FunctionParameter,
) {
    match param {
        FunctionParameter::PosOnly {
            name,
            annotation,
            required,
        } => {
            let mut param = builder.init_pos_only();
            if let Some(name) = name {
                param.reborrow().set_name(name.as_str());
            }
            set_pysa_type(param.reborrow().init_annotation(), annotation);
            param.set_required(*required);
        }
        FunctionParameter::Pos {
            name,
            annotation,
            required,
        } => {
            let mut param = builder.init_pos();
            param.reborrow().set_name(name.as_str());
            set_pysa_type(param.reborrow().init_annotation(), annotation);
            param.set_required(*required);
        }
        FunctionParameter::VarArg { name, annotation } => {
            let mut param = builder.init_var_arg();
            if let Some(name) = name {
                param.reborrow().set_name(name.as_str());
            }
            set_pysa_type(param.init_annotation(), annotation);
        }
        FunctionParameter::KwOnly {
            name,
            annotation,
            required,
        } => {
            let mut param = builder.init_kw_only();
            param.reborrow().set_name(name.as_str());
            set_pysa_type(param.reborrow().init_annotation(), annotation);
            param.set_required(*required);
        }
        FunctionParameter::Kwargs { name, annotation } => {
            let mut param = builder.init_kwargs();
            if let Some(name) = name {
                param.reborrow().set_name(name.as_str());
            }
            set_pysa_type(param.init_annotation(), annotation);
        }
    }
}

fn set_function_parameters(
    mut builder: pysa_report_capnp::function_parameters::Builder,
    params: &FunctionParameters,
) {
    match params {
        FunctionParameters::List(list) => {
            let mut params = builder.init_list(list.len() as u32);
            for (i, param) in list.iter().enumerate() {
                set_function_parameter(params.reborrow().get(i as u32), param);
            }
        }
        FunctionParameters::Ellipsis => {
            builder.set_ellipsis(());
        }
        FunctionParameters::ParamSpec => {
            builder.set_param_spec(());
        }
    }
}

fn set_function_signature(
    mut builder: pysa_report_capnp::function_signature::Builder,
    signature: &FunctionSignature,
) {
    set_function_parameters(builder.reborrow().init_parameters(), &signature.parameters);
    set_pysa_type(
        builder.init_return_annotation(),
        &signature.return_annotation,
    );
}

fn set_target(mut builder: pysa_report_capnp::target::Builder, target: &Target<FunctionRef>) {
    match target {
        Target::Function(func_ref) => {
            set_function_ref(builder.init_function(), func_ref);
        }
        Target::Overrides(func_ref) => {
            set_function_ref(builder.init_overrides(), func_ref);
        }
        Target::FormatString => {
            builder.set_format_string(());
        }
    }
}

fn set_captured_variable_ref(
    mut builder: pysa_report_capnp::captured_variable_ref::Builder,
    captured_variable_ref: &CapturedVariableRef<FunctionRef>,
) {
    set_function_ref(
        builder.reborrow().init_outer_function(),
        &captured_variable_ref.outer_function,
    );
    builder.set_name(captured_variable_ref.name.as_str());
}

fn convert_implicit_receiver(
    implicit_receiver: ImplicitReceiver,
) -> pysa_report_capnp::ImplicitReceiver {
    match implicit_receiver {
        ImplicitReceiver::TrueWithClassReceiver => {
            pysa_report_capnp::ImplicitReceiver::TrueWithClassReceiver
        }
        ImplicitReceiver::TrueWithObjectReceiver => {
            pysa_report_capnp::ImplicitReceiver::TrueWithObjectReceiver
        }
        ImplicitReceiver::False => pysa_report_capnp::ImplicitReceiver::False,
    }
}

fn set_pysa_call_target(
    mut builder: pysa_report_capnp::pysa_call_target::Builder,
    call_target: &PysaCallTarget<FunctionRef>,
) {
    set_target(builder.reborrow().init_target(), &call_target.target);
    builder
        .reborrow()
        .set_implicit_receiver(convert_implicit_receiver(call_target.implicit_receiver));
    builder
        .reborrow()
        .set_implicit_dunder_call(call_target.implicit_dunder_call);
    if let Some(receiver_class) = &call_target.receiver_class {
        set_class_ref(builder.reborrow().init_receiver_class(), receiver_class);
    }
    builder
        .reborrow()
        .set_is_class_method(call_target.is_class_method);
    builder
        .reborrow()
        .set_is_static_method(call_target.is_static_method);
    set_scalar_type_properties(builder.init_return_type(), call_target.return_type);
}

fn set_unresolved(mut builder: pysa_report_capnp::unresolved::Builder, unresolved: &Unresolved) {
    match unresolved {
        Unresolved::False => {
            builder.set_false(());
        }
        Unresolved::True(reason) => {
            builder.set_true(convert_unresolved_reason(*reason));
        }
    }
}

fn convert_unresolved_reason(reason: UnresolvedReason) -> pysa_report_capnp::UnresolvedReason {
    match reason {
        UnresolvedReason::LambdaArgument => pysa_report_capnp::UnresolvedReason::LambdaArgument,
        UnresolvedReason::UnexpectedPyreflyTarget => {
            pysa_report_capnp::UnresolvedReason::UnexpectedPyreflyTarget
        }
        UnresolvedReason::EmptyPyreflyCallTarget => {
            pysa_report_capnp::UnresolvedReason::EmptyPyreflyCallTarget
        }
        UnresolvedReason::UnknownClassField => {
            pysa_report_capnp::UnresolvedReason::UnknownClassField
        }
        UnresolvedReason::ClassFieldOnlyExistInObject => {
            pysa_report_capnp::UnresolvedReason::ClassFieldOnlyExistInObject
        }
        UnresolvedReason::UnsupportedFunctionTarget => {
            pysa_report_capnp::UnresolvedReason::UnsupportedFunctionTarget
        }
        UnresolvedReason::UnexpectedDefiningClass => {
            pysa_report_capnp::UnresolvedReason::UnexpectedDefiningClass
        }
        UnresolvedReason::UnexpectedInitMethod => {
            pysa_report_capnp::UnresolvedReason::UnexpectedInitMethod
        }
        UnresolvedReason::UnexpectedNewMethod => {
            pysa_report_capnp::UnresolvedReason::UnexpectedNewMethod
        }
        UnresolvedReason::UnexpectedCalleeExpression => {
            pysa_report_capnp::UnresolvedReason::UnexpectedCalleeExpression
        }
        UnresolvedReason::UnresolvedMagicDunderAttr => {
            pysa_report_capnp::UnresolvedReason::UnresolvedMagicDunderAttr
        }
        UnresolvedReason::UnresolvedMagicDunderAttrDueToNoBase => {
            pysa_report_capnp::UnresolvedReason::UnresolvedMagicDunderAttrDueToNoBase
        }
        UnresolvedReason::UnresolvedMagicDunderAttrDueToNoAttribute => {
            pysa_report_capnp::UnresolvedReason::UnresolvedMagicDunderAttrDueToNoAttribute
        }
        UnresolvedReason::Mixed => pysa_report_capnp::UnresolvedReason::Mixed,
    }
}

fn set_higher_order_parameter(
    mut builder: pysa_report_capnp::higher_order_parameter::Builder,
    higher_order_param: &HigherOrderParameter<FunctionRef>,
) {
    builder.reborrow().set_index(higher_order_param.index);
    let mut targets = builder
        .reborrow()
        .init_call_targets(higher_order_param.call_targets.len() as u32);
    for (i, call_target) in higher_order_param.call_targets.iter().enumerate() {
        set_pysa_call_target(targets.reborrow().get(i as u32), call_target);
    }
    set_unresolved(builder.init_unresolved(), &higher_order_param.unresolved);
}

fn set_call_callees(
    mut builder: pysa_report_capnp::call_callees::Builder,
    call_callees: &CallCallees<FunctionRef>,
) {
    {
        let mut targets = builder
            .reborrow()
            .init_call_targets(call_callees.call_targets.len() as u32);
        for (i, call_target) in call_callees.call_targets.iter().enumerate() {
            set_pysa_call_target(targets.reborrow().get(i as u32), call_target);
        }
    }
    {
        let mut targets = builder
            .reborrow()
            .init_init_targets(call_callees.init_targets.len() as u32);
        for (i, call_target) in call_callees.init_targets.iter().enumerate() {
            set_pysa_call_target(targets.reborrow().get(i as u32), call_target);
        }
    }
    {
        let mut targets = builder
            .reborrow()
            .init_new_targets(call_callees.new_targets.len() as u32);
        for (i, call_target) in call_callees.new_targets.iter().enumerate() {
            set_pysa_call_target(targets.reborrow().get(i as u32), call_target);
        }
    }
    {
        let mut list = builder
            .reborrow()
            .init_higher_order_parameters(call_callees.higher_order_parameters.len() as u32);
        for (i, higher_order_param) in call_callees.higher_order_parameters.values().enumerate() {
            set_higher_order_parameter(list.reborrow().get(i as u32), higher_order_param);
        }
    }
    set_unresolved(builder.init_unresolved(), &call_callees.unresolved);
}

fn set_expression_callees(
    builder: pysa_report_capnp::expression_callees::Builder,
    expression_callees: &ExpressionCallees<FunctionRef>,
) {
    match expression_callees {
        ExpressionCallees::Call(call_callees) => {
            set_call_callees(builder.init_call(), call_callees);
        }
        ExpressionCallees::Identifier(identifier_callees) => {
            set_identifier_callees(builder.init_identifier(), identifier_callees);
        }
        ExpressionCallees::AttributeAccess(attribute_access_callees) => {
            set_attribute_access_callees(builder.init_attribute_access(), attribute_access_callees);
        }
        ExpressionCallees::Define(define_callees) => {
            set_define_callees(builder.init_define(), define_callees);
        }
        ExpressionCallees::FormatStringArtificial(format_string_artificial) => {
            set_format_string_artificial_callees(
                builder.init_format_string_artificial(),
                format_string_artificial,
            );
        }
        ExpressionCallees::FormatStringStringify(format_string_stringify) => {
            set_format_string_stringify_callees(
                builder.init_format_string_stringify(),
                format_string_stringify,
            );
        }
        ExpressionCallees::Return(return_shim) => {
            set_return_shim_callees(builder.init_return(), return_shim);
        }
    }
}

fn set_identifier_callees(
    mut builder: pysa_report_capnp::identifier_callees::Builder,
    identifier_callees: &IdentifierCallees<FunctionRef>,
) {
    set_call_callees(
        builder.reborrow().init_if_called(),
        &identifier_callees.if_called,
    );
    {
        let mut targets = builder
            .reborrow()
            .init_global_targets(identifier_callees.global_targets.len() as u32);
        for (i, global_variable_ref) in identifier_callees.global_targets.iter().enumerate() {
            set_global_variable_ref(targets.reborrow().get(i as u32), global_variable_ref);
        }
    }
    {
        let mut captured_variables =
            builder.init_captured_variables(identifier_callees.captured_variables.len() as u32);
        for (i, captured_variable_ref) in identifier_callees.captured_variables.iter().enumerate() {
            set_captured_variable_ref(
                captured_variables.reborrow().get(i as u32),
                captured_variable_ref,
            );
        }
    }
}

fn set_attribute_access_callees(
    mut builder: pysa_report_capnp::attribute_access_callees::Builder,
    attribute_access: &AttributeAccessCallees<FunctionRef>,
) {
    set_call_callees(
        builder.reborrow().init_if_called(),
        &attribute_access.if_called,
    );
    {
        let mut targets = builder
            .reborrow()
            .init_property_setters(attribute_access.property_setters.len() as u32);
        for (i, call_target) in attribute_access.property_setters.iter().enumerate() {
            set_pysa_call_target(targets.reborrow().get(i as u32), call_target);
        }
    }
    {
        let mut targets = builder
            .reborrow()
            .init_property_getters(attribute_access.property_getters.len() as u32);
        for (i, call_target) in attribute_access.property_getters.iter().enumerate() {
            set_pysa_call_target(targets.reborrow().get(i as u32), call_target);
        }
    }
    {
        let mut targets = builder
            .reborrow()
            .init_global_targets(attribute_access.global_targets.len() as u32);
        for (i, global_variable_ref) in attribute_access.global_targets.iter().enumerate() {
            set_global_variable_ref(targets.reborrow().get(i as u32), global_variable_ref);
        }
    }
    builder.set_is_attribute(attribute_access.is_attribute);
}

fn set_define_callees(
    builder: pysa_report_capnp::define_callees::Builder,
    define_callees: &DefineCallees<FunctionRef>,
) {
    let mut targets = builder.init_define_targets(define_callees.define_targets.len() as u32);
    for (i, call_target) in define_callees.define_targets.iter().enumerate() {
        set_pysa_call_target(targets.reborrow().get(i as u32), call_target);
    }
}

fn set_format_string_artificial_callees(
    builder: pysa_report_capnp::format_string_artificial_callees::Builder,
    format_string_artificial: &FormatStringArtificialCallees<FunctionRef>,
) {
    let mut targets = builder.init_targets(format_string_artificial.targets.len() as u32);
    for (i, call_target) in format_string_artificial.targets.iter().enumerate() {
        set_pysa_call_target(targets.reborrow().get(i as u32), call_target);
    }
}

fn set_format_string_stringify_callees(
    mut builder: pysa_report_capnp::format_string_stringify_callees::Builder,
    format_string_stringify: &FormatStringStringifyCallees<FunctionRef>,
) {
    {
        let mut targets = builder
            .reborrow()
            .init_targets(format_string_stringify.targets.len() as u32);
        for (i, call_target) in format_string_stringify.targets.iter().enumerate() {
            set_pysa_call_target(targets.reborrow().get(i as u32), call_target);
        }
    }
    set_unresolved(
        builder.init_unresolved(),
        &format_string_stringify.unresolved,
    );
}

fn set_return_shim_callees(
    mut builder: pysa_report_capnp::return_shim_callees::Builder,
    return_shim: &ReturnShimCallees<FunctionRef>,
) {
    {
        let mut targets = builder
            .reborrow()
            .init_targets(return_shim.targets.len() as u32);
        for (i, call_target) in return_shim.targets.iter().enumerate() {
            set_pysa_call_target(targets.reborrow().get(i as u32), call_target);
        }
    }
    {
        let mut arguments = builder.init_arguments(return_shim.arguments.len() as u32);
        for (i, arg) in return_shim.arguments.iter().enumerate() {
            arguments.set(
                i as u32,
                match arg {
                    ReturnShimArgumentMapping::ReturnExpression => {
                        pysa_report_capnp::ReturnShimArgumentMapping::ReturnExpression
                    }
                    ReturnShimArgumentMapping::ReturnExpressionElement => {
                        pysa_report_capnp::ReturnShimArgumentMapping::ReturnExpressionElement
                    }
                },
            );
        }
    }
}

fn set_function_definition(
    mut builder: pysa_report_capnp::function_definition::Builder,
    func_id: &FunctionId,
    func_def: &FunctionDefinition,
) {
    // Flattened FunctionBaseDefinition fields
    builder.reborrow().set_name(func_def.base.name.as_str());
    if let Some(name_location) = &func_def.base.name_location {
        set_location(
            builder.reborrow().init_define_name_location(),
            name_location,
        );
    }
    set_scope_parent(builder.reborrow().init_parent(), &func_def.base.parent);
    builder
        .reborrow()
        .set_is_overload(func_def.base.is_overload);
    builder
        .reborrow()
        .set_is_staticmethod(func_def.base.is_staticmethod);
    builder
        .reborrow()
        .set_is_classmethod(func_def.base.is_classmethod);
    builder
        .reborrow()
        .set_is_property_getter(func_def.base.is_property_getter);
    builder
        .reborrow()
        .set_is_property_setter(func_def.base.is_property_setter);
    builder.reborrow().set_is_stub(func_def.base.is_stub);
    builder
        .reborrow()
        .set_is_def_statement(func_def.base.is_def_statement);
    if let Some(defining_class) = &func_def.base.defining_class {
        set_class_ref(builder.reborrow().init_defining_class(), defining_class);
    }

    // FunctionDefinition-specific fields
    builder
        .reborrow()
        .set_function_id(func_id.serialize_to_string());
    {
        let mut signatures = builder
            .reborrow()
            .init_undecorated_signatures(func_def.undecorated_signatures.len() as u32);
        for (i, signature) in func_def.undecorated_signatures.iter().enumerate() {
            set_function_signature(signatures.reborrow().get(i as u32), signature);
        }
    }
    {
        let mut captured_variables = builder
            .reborrow()
            .init_captured_variables(func_def.captured_variables.len() as u32);
        for (i, captured_variable_ref) in func_def.captured_variables.iter().enumerate() {
            set_captured_variable_ref(
                captured_variables.reborrow().get(i as u32),
                captured_variable_ref,
            );
        }
    }
    {
        let mut list = builder
            .reborrow()
            .init_decorator_callees(func_def.decorator_callees.len() as u32);
        for (i, (loc, targets)) in func_def.decorator_callees.iter().enumerate() {
            let mut entry = list.reborrow().get(i as u32);
            set_location(entry.reborrow().init_location(), loc);
            let mut target_list = entry.init_targets(targets.len() as u32);
            for (j, target) in targets.iter().enumerate() {
                set_target(target_list.reborrow().get(j as u32), target);
            }
        }
    }
    if let Some(overridden_base_method) = &func_def.overridden_base_method {
        set_function_ref(
            builder.reborrow().init_overridden_base_method(),
            overridden_base_method,
        );
    }
}

fn set_class_definition(
    mut builder: pysa_report_capnp::class_definition::Builder,
    class_def: &ClassDefinition,
) {
    builder.reborrow().set_class_id(class_def.class_id.to_int());
    builder.reborrow().set_name(&class_def.name);
    set_location(
        builder.reborrow().init_name_location(),
        &class_def.name_location,
    );
    {
        let mut bases = builder.reborrow().init_bases(class_def.bases.len() as u32);
        for (i, base) in class_def.bases.iter().enumerate() {
            set_class_ref(bases.reborrow().get(i as u32), base);
        }
    }
    // MRO
    match &class_def.mro {
        PysaClassMro::Resolved(classes) => {
            let mro_builder = builder.reborrow().init_mro();
            let mut resolved = mro_builder.init_resolved(classes.len() as u32);
            for (i, class_ref) in classes.iter().enumerate() {
                set_class_ref(resolved.reborrow().get(i as u32), class_ref);
            }
        }
        PysaClassMro::Cyclic => {
            builder.reborrow().init_mro().set_cyclic(());
        }
    }
    set_scope_parent(builder.reborrow().init_parent(), &class_def.parent);
    builder
        .reborrow()
        .set_is_synthesized(class_def.is_synthesized);
    builder.reborrow().set_is_dataclass(class_def.is_dataclass);
    builder
        .reborrow()
        .set_is_named_tuple(class_def.is_named_tuple);
    builder
        .reborrow()
        .set_is_typed_dict(class_def.is_typed_dict);
    {
        let mut list = builder
            .reborrow()
            .init_fields(class_def.fields.len() as u32);
        for (i, (name, field)) in class_def.fields.iter().enumerate() {
            set_class_field(list.reborrow().get(i as u32), name.as_str(), field);
        }
    }
    {
        let mut list = builder.init_decorator_callees(class_def.decorator_callees.len() as u32);
        for (i, (loc, targets)) in class_def.decorator_callees.iter().enumerate() {
            let mut entry = list.reborrow().get(i as u32);
            set_location(entry.reborrow().init_location(), loc);
            let mut target_list = entry.init_targets(targets.len() as u32);
            for (j, target) in targets.iter().enumerate() {
                set_target(target_list.reborrow().get(j as u32), target);
            }
        }
    }
}

fn set_class_field(
    mut builder: pysa_report_capnp::pysa_class_field::Builder,
    name: &str,
    field: &PysaClassField,
) {
    builder.reborrow().set_name(name);
    set_pysa_type(builder.reborrow().init_type(), &field.type_);
    if let Some(annotation) = &field.explicit_annotation {
        builder.reborrow().set_explicit_annotation(annotation);
    }
    if let Some(loc) = &field.location {
        set_location(builder.reborrow().init_location(), loc);
    }
    builder.set_declaration_kind(convert_class_field_declaration_kind(
        &field.declaration_kind,
    ));
}

fn convert_class_field_declaration_kind(
    kind: &Option<PysaClassFieldDeclaration>,
) -> pysa_report_capnp::PysaClassFieldDeclaration {
    match kind {
        None => pysa_report_capnp::PysaClassFieldDeclaration::None,
        Some(PysaClassFieldDeclaration::DeclaredByAnnotation) => {
            pysa_report_capnp::PysaClassFieldDeclaration::DeclaredByAnnotation
        }
        Some(PysaClassFieldDeclaration::DeclaredWithoutAnnotation) => {
            pysa_report_capnp::PysaClassFieldDeclaration::DeclaredWithoutAnnotation
        }
        Some(PysaClassFieldDeclaration::AssignedInBody) => {
            pysa_report_capnp::PysaClassFieldDeclaration::AssignedInBody
        }
        Some(PysaClassFieldDeclaration::DefinedWithoutAssign) => {
            pysa_report_capnp::PysaClassFieldDeclaration::DefinedWithoutAssign
        }
        Some(PysaClassFieldDeclaration::DefinedInMethod) => {
            pysa_report_capnp::PysaClassFieldDeclaration::DefinedInMethod
        }
    }
}

/// Write module definitions in Cap'n Proto format.
pub fn write_definitions<W: Write>(writer: W, defs: &PysaModuleDefinitions) -> anyhow::Result<()> {
    let mut message = capnp::message::Builder::new_default();
    {
        let mut root = message.init_root::<pysa_report_capnp::module_definitions::Builder>();
        root.reborrow().set_module_id(defs.module_id.to_int());
        root.reborrow().set_module_name(defs.module_name.as_str());
        set_source_path(root.reborrow().init_source_path(), &defs.source_path);

        // Function definitions
        let func_defs_map = defs.function_definitions.as_map();
        let mut funcs = root
            .reborrow()
            .init_function_definitions(func_defs_map.len() as u32);
        for (i, (func_id, func_def)) in func_defs_map.iter().enumerate() {
            set_function_definition(funcs.reborrow().get(i as u32), func_id, func_def);
        }

        // Class definitions
        let mut classes = root
            .reborrow()
            .init_class_definitions(defs.class_definitions.len() as u32);
        for (i, (_class_id, class_def)) in defs.class_definitions.iter().enumerate() {
            set_class_definition(classes.reborrow().get(i as u32), class_def);
        }

        // Global variables
        let mut global_variables = root.init_global_variables(defs.global_variables.len() as u32);
        for (i, (name, global_variable)) in defs.global_variables.iter().enumerate() {
            let mut entry = global_variables.reborrow().get(i as u32);
            entry.reborrow().set_name(name.as_str());
            if let Some(ty) = &global_variable.type_ {
                set_pysa_type(entry.reborrow().init_type(), ty);
            }
            set_location(entry.init_location(), &global_variable.location);
        }
    }
    capnp::serialize::write_message(writer, &message)?;
    Ok(())
}

/// Write module type-of-expressions in Cap'n Proto format.
pub fn write_type_of_expressions<W: Write>(
    writer: W,
    exprs: &PysaModuleTypeOfExpressions,
) -> anyhow::Result<()> {
    let mut message = capnp::message::Builder::new_default();
    {
        let mut root =
            message.init_root::<pysa_report_capnp::module_type_of_expressions::Builder>();
        root.reborrow().set_module_id(exprs.module_id.to_int());
        root.reborrow().set_module_name(exprs.module_name.as_str());
        set_source_path(root.reborrow().init_source_path(), &exprs.source_path);

        let mut functions_list = root.init_functions(exprs.functions.len() as u32);
        for (i, (func_id, func_data)) in exprs.functions.iter().enumerate() {
            let mut func_builder = functions_list.reborrow().get(i as u32);
            func_builder
                .reborrow()
                .set_function_id(func_id.serialize_to_string());

            // Write deduplicated type table
            let mut types_list = func_builder
                .reborrow()
                .init_types(func_data.type_table.len() as u32);
            for (j, ty) in func_data.type_table.iter().enumerate() {
                set_pysa_type(types_list.reborrow().get(j as u32), ty);
            }

            // Write location -> type_id entries
            let mut locations_list = func_builder.init_locations(func_data.locations.len() as u32);
            for (j, (loc, type_id)) in func_data.locations.iter().enumerate() {
                let mut entry = locations_list.reborrow().get(j as u32);
                set_location(entry.reborrow().init_location(), loc);
                entry.set_type_id(type_id.0);
            }
        }
    }
    capnp::serialize::write_message(writer, &message)?;
    Ok(())
}

/// Write module call graphs in Cap'n Proto format.
pub fn write_call_graphs<W: Write>(writer: W, graphs: &PysaModuleCallGraphs) -> anyhow::Result<()> {
    let mut message = capnp::message::Builder::new_default();
    {
        let mut root = message.init_root::<pysa_report_capnp::module_call_graphs::Builder>();
        root.reborrow().set_module_id(graphs.module_id.to_int());
        root.reborrow().set_module_name(graphs.module_name.as_str());
        set_source_path(root.reborrow().init_source_path(), &graphs.source_path);

        let mut call_graphs_list = root.init_call_graphs(graphs.call_graphs.len() as u32);
        for (i, (func_id, call_graph)) in graphs.call_graphs.iter().enumerate() {
            set_function_call_graph(
                call_graphs_list.reborrow().get(i as u32),
                func_id,
                call_graph,
            );
        }
    }
    capnp::serialize::write_message(writer, &message)?;
    Ok(())
}

fn set_function_call_graph(
    mut builder: pysa_report_capnp::function_call_graph::Builder,
    func_id: &FunctionId,
    call_graph: &CallGraph<ExpressionIdentifier, FunctionRef>,
) {
    builder
        .reborrow()
        .set_function_id(func_id.serialize_to_string());

    let call_graph_map = call_graph.as_map();
    let mut list = builder.init_entries(call_graph_map.len() as u32);
    for (i, (expr_id, callees)) in call_graph_map.iter().enumerate() {
        let mut entry = list.reborrow().get(i as u32);
        entry.reborrow().set_expression_id(expr_id.as_key());
        set_expression_callees(entry.init_callees(), callees);
    }
}

fn set_project_module(
    mut builder: pysa_report_capnp::pysa_project_module::Builder,
    module: &super::PysaProjectModule,
) {
    builder.reborrow().set_module_id(module.module_id.to_int());
    builder
        .reborrow()
        .set_module_name(module.module_name.as_str());
    set_source_path(builder.reborrow().init_source_path(), &module.source_path);
    if let Some(rel_path) = &module.relative_source_path {
        builder
            .reborrow()
            .set_relative_source_path(rel_path.to_string_lossy().as_ref());
    }
    if let Some(info) = &module.info_filename {
        builder
            .reborrow()
            .set_info_filename(info.to_string_lossy().as_ref());
    }
    builder
        .reborrow()
        .set_python_version(module.python_version.to_string());
    builder.reborrow().set_platform(module.platform.to_string());
    builder.reborrow().set_is_test(module.is_test);
    builder.reborrow().set_is_interface(module.is_interface);
    builder.reborrow().set_is_init(module.is_init);
    builder.set_is_internal(module.is_internal);
}

/// Write the project file in Cap'n Proto format.
pub fn write_project_file<W: Write>(writer: W, project: &PysaProjectFile) -> anyhow::Result<()> {
    let mut message = capnp::message::Builder::new_default();
    {
        let mut root = message.init_root::<pysa_report_capnp::project_file::Builder>();
        // builtin_module_ids: List(UInt32)
        {
            let mut list = root
                .reborrow()
                .init_builtin_module_ids(project.builtin_module_ids.len() as u32);
            for (i, id) in project.builtin_module_ids.iter().enumerate() {
                list.set(i as u32, id.to_int());
            }
        }
        // object_class_refs: List(ClassRef)
        {
            let mut list = root
                .reborrow()
                .init_object_class_refs(project.object_class_refs.len() as u32);
            for (i, class_ref) in project.object_class_refs.iter().enumerate() {
                set_class_ref(list.reborrow().get(i as u32), class_ref);
            }
        }
        // dict_class_refs: List(ClassRef)
        {
            let mut list = root
                .reborrow()
                .init_dict_class_refs(project.dict_class_refs.len() as u32);
            for (i, class_ref) in project.dict_class_refs.iter().enumerate() {
                set_class_ref(list.reborrow().get(i as u32), class_ref);
            }
        }
        // typing_module_ids: List(UInt32)
        {
            let mut list = root
                .reborrow()
                .init_typing_module_ids(project.typing_module_ids.len() as u32);
            for (i, id) in project.typing_module_ids.iter().enumerate() {
                list.set(i as u32, id.to_int());
            }
        }
        // typing_mapping_class_refs: List(ClassRef)
        {
            let mut list = root
                .reborrow()
                .init_typing_mapping_class_refs(project.typing_mapping_class_refs.len() as u32);
            for (i, class_ref) in project.typing_mapping_class_refs.iter().enumerate() {
                set_class_ref(list.reborrow().get(i as u32), class_ref);
            }
        }

        let mut list = root.init_modules(project.modules.len() as u32);
        for (i, module) in project.modules.values().enumerate() {
            set_project_module(list.reborrow().get(i as u32), module);
        }
    }
    capnp::serialize::write_message(writer, &message)?;
    Ok(())
}

/// Write errors file in Cap'n Proto format.
pub fn write_errors<W: Write>(writer: W, errors: &PysaTypeErrorsFile) -> anyhow::Result<()> {
    let mut message = capnp::message::Builder::new_default();
    {
        let root = message.init_root::<pysa_report_capnp::type_errors::Builder>();
        let mut list = root.init_errors(errors.errors.len() as u32);
        for (i, error) in errors.errors.iter().enumerate() {
            let mut entry = list.reborrow().get(i as u32);
            entry.reborrow().set_module_name(error.module_name.as_str());
            set_source_path(entry.reborrow().init_module_path(), &error.module_path);
            set_location(entry.reborrow().init_location(), &error.location);
            entry.reborrow().set_kind(error.kind.to_string());
            entry.set_message(&error.message);
        }
    }
    capnp::serialize::write_message(writer, &message)?;
    Ok(())
}
