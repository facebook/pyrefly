/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Shared type formatting utilities for TSP requests

use tsp_types as tsp;

use crate::state::state::Transaction;
use crate::types::types::Type;

/// Apply TypeReprFlags formatting to a type
pub fn format_type_for_display(
    typ: Type,
    flags: tsp::TypeReprFlags,
    transaction: &Transaction<'_>,
) -> String {
    let mut processed_type = typ;

    // Apply type alias expansion first
    if flags.has_expand_type_aliases() {
        processed_type = expand_type_aliases(processed_type, transaction);
    }

    // Apply instance type conversion
    if flags.has_convert_to_instance_type() {
        processed_type = convert_to_instance_type(processed_type, transaction);
    }

    // Apply variance formatting for type variables
    if flags.has_print_type_var_variance() {
        format_type_with_variance(&processed_type)
    } else {
        processed_type.to_string()
    }
}

/// Expand type aliases recursively
pub fn expand_type_aliases(type_obj: Type, transaction: &Transaction<'_>) -> Type {
    use pyrefly_types::simplify::unions;

    match type_obj {
        Type::TypeAlias(alias) => {
            // Get the underlying type from the alias
            let underlying = alias.as_type();
            // Recursively expand any nested aliases
            expand_type_aliases(underlying, transaction)
        }
        Type::Union(union_types) => {
            // Expand aliases in union members
            let expanded_types: Vec<_> = union_types
                .iter()
                .map(|t| expand_type_aliases(t.clone(), transaction))
                .collect();
            unions(expanded_types)
        }
        Type::ClassType(class_type) => {
            // Expand aliases in type arguments
            let expanded_targs = expand_type_args_aliases(class_type.targs(), transaction);
            if &expanded_targs != class_type.targs() {
                Type::ClassType(crate::types::class::ClassType::new(
                    class_type.class_object().clone(),
                    expanded_targs,
                ))
            } else {
                Type::ClassType(class_type)
            }
        }
        Type::Function(func) => {
            // Expand aliases in function signature
            let expanded_callable = expand_callable_aliases(&func.signature, transaction);
            if expanded_callable != func.signature {
                // Create a new function with expanded signature
                // Note: We can't modify the existing function, so we return the original for now
                // In a real implementation, you'd need to create a new Function type
                Type::Function(func)
            } else {
                Type::Function(func)
            }
        }
        Type::Callable(callable) => {
            // Expand aliases in callable signature
            let expanded = expand_callable_aliases(&callable, transaction);
            Type::Callable(Box::new(expanded))
        }
        // For other types, return as-is
        _ => type_obj,
    }
}

/// Convert class types to instance types
pub fn convert_to_instance_type(type_obj: Type, _transaction: &Transaction<'_>) -> Type {
    use pyrefly_types::simplify::unions;

    match type_obj {
        Type::ClassDef(class_def) => {
            // Convert ClassDef to ClassType (instance)
            let empty_tparams = std::sync::Arc::new(crate::types::types::TParams::new(Vec::new()));
            let empty_targs = crate::types::types::TArgs::new(empty_tparams, Vec::new());
            Type::ClassType(crate::types::class::ClassType::new(class_def, empty_targs))
        }
        Type::Union(union_types) => {
            // Convert class types in union members
            let converted_types: Vec<_> = union_types
                .iter()
                .map(|t| convert_to_instance_type(t.clone(), _transaction))
                .collect();
            unions(converted_types)
        }
        // For other types, return as-is
        _ => type_obj,
    }
}

/// Format a type with variance information for type variables
pub fn format_type_with_variance(type_obj: &Type) -> String {
    use pyrefly_types::type_var::PreInferenceVariance;

    match type_obj {
        Type::TypeVar(tvar) => {
            let base_name = tvar.qname().id().to_string();

            // Add variance information if available
            match tvar.variance() {
                PreInferenceVariance::PCovariant => format!("{base_name} (covariant)"),
                PreInferenceVariance::PContravariant => format!("{base_name} (contravariant)"),
                PreInferenceVariance::PInvariant => format!("{base_name} (invariant)"),
                PreInferenceVariance::PUndefined => base_name,
            }
        }
        _ => type_obj.to_string(),
    }
}

/// Helper function to expand type aliases in type arguments
fn expand_type_args_aliases(
    targs: &crate::types::types::TArgs,
    _transaction: &Transaction<'_>,
) -> crate::types::types::TArgs {
    // For now, return the original targs since we need access to the internal structure
    // In a full implementation, you'd iterate through targs and expand each type
    targs.clone()
}

/// Helper function to expand type aliases in callable signatures
pub fn expand_callable_aliases(
    callable: &crate::types::callable::Callable,
    transaction: &Transaction<'_>,
) -> crate::types::callable::Callable {
    use crate::types::callable::Callable;
    use crate::types::callable::Params;

    let expanded_ret = expand_type_aliases(callable.ret.clone(), transaction);

    let expanded_params = match &callable.params {
        Params::List(param_list) => {
            let expanded_items: Vec<_> = param_list
                .items()
                .iter()
                .map(|param| expand_param_aliases(param, transaction))
                .collect();
            Params::List(crate::types::callable::ParamList::new(expanded_items))
        }
        Params::Ellipsis => Params::Ellipsis,
        Params::ParamSpec(types, param_spec) => {
            let expanded_types: Vec<_> = types
                .iter()
                .map(|t| expand_type_aliases(t.clone(), transaction))
                .collect();
            let expanded_param_spec = expand_type_aliases(param_spec.clone(), transaction);
            Params::ParamSpec(expanded_types.into_boxed_slice(), expanded_param_spec)
        }
    };

    // Create a new callable based on the original method
    match (&callable.params, &expanded_params) {
        (Params::List(_), Params::List(new_list)) => Callable::list(new_list.clone(), expanded_ret),
        (Params::Ellipsis, Params::Ellipsis) => Callable::ellipsis(expanded_ret),
        (Params::ParamSpec(_, _), Params::ParamSpec(new_types, new_param_spec)) => {
            Callable::concatenate(new_types.clone(), new_param_spec.clone(), expanded_ret)
        }
        // Fallback to original callable if structure doesn't match
        _ => callable.clone(),
    }
}

/// Helper function to expand type aliases in function parameters
pub fn expand_param_aliases(
    param: &crate::types::callable::Param,
    transaction: &Transaction<'_>,
) -> crate::types::callable::Param {
    use crate::types::callable::Param;

    match param {
        Param::PosOnly(name, param_type, required) => {
            let expanded_type = expand_type_aliases(param_type.clone(), transaction);
            Param::PosOnly(name.clone(), expanded_type, required.clone())
        }
        Param::Pos(name, param_type, required) => {
            let expanded_type = expand_type_aliases(param_type.clone(), transaction);
            Param::Pos(name.clone(), expanded_type, required.clone())
        }
        Param::VarArg(name, param_type) => {
            let expanded_type = expand_type_aliases(param_type.clone(), transaction);
            Param::VarArg(name.clone(), expanded_type)
        }
        Param::KwOnly(name, param_type, required) => {
            let expanded_type = expand_type_aliases(param_type.clone(), transaction);
            Param::KwOnly(name.clone(), expanded_type, required.clone())
        }
        Param::Kwargs(name, param_type) => {
            let expanded_type = expand_type_aliases(param_type.clone(), transaction);
            Param::Kwargs(name.clone(), expanded_type)
        }
    }
}
