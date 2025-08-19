/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Implementation of the getTypeAttributes TSP request

use lsp_server::ResponseError;
use tsp_types::snapshot_outdated_error;
use tsp_types::tsp_debug;
use tsp_types::{self as tsp};

use crate::module::module_info::ModuleInfo;
use crate::state::handle::Handle;
use crate::state::state::Transaction;
use crate::tsp::common::convert_to_tsp_type;
use crate::tsp::server::TspServer;
use crate::types::types::Type;

/// Helper function to extract parameter name from a Param
fn get_param_name(param: &crate::types::callable::Param, index: usize) -> String {
    use crate::types::callable::Param;

    match param {
        Param::PosOnly(name, _, _) => name
            .as_ref()
            .map(|n| n.as_str().to_owned())
            .unwrap_or_else(|| format!("param{index}")),
        Param::Pos(name, _, _) => name.as_str().to_owned(),
        Param::VarArg(name, _) => name
            .as_ref()
            .map(|n| format!("*{}", n.as_str()))
            .unwrap_or_else(|| "*args".to_owned()),
        Param::KwOnly(name, _, _) => name.as_str().to_owned(),
        Param::Kwargs(name, _) => name
            .as_ref()
            .map(|n| format!("**{}", n.as_str()))
            .unwrap_or_else(|| "**kwargs".to_owned()),
    }
}

/// Helper function to extract parameter type from a Param
fn get_param_type(param: &crate::types::callable::Param) -> &Type {
    use crate::types::callable::Param;

    match param {
        Param::PosOnly(_, param_type, _) => param_type,
        Param::Pos(_, param_type, _) => param_type,
        Param::VarArg(_, param_type) => param_type,
        Param::KwOnly(_, param_type, _) => param_type,
        Param::Kwargs(_, param_type) => param_type,
    }
}

/// Extract all attributes from a pyrefly type
/// This function examines a type and returns all its attributes (class members, function parameters, etc.)
pub fn extract_type_attributes(
    transaction: &Transaction<'_>,
    pyrefly_type: &Type,
    handle_factory: impl Fn(&ModuleInfo) -> Handle,
) -> Result<Vec<tsp::Attribute>, ResponseError> {
    let mut attributes = Vec::new();

    match pyrefly_type {
        Type::ClassType(class_type) => {
            // Get the module info from the class type
            let class_qname = class_type.class_object().qname();
            let module_info = class_qname.module();

            // Create a handle for the module containing the class
            let handle = handle_factory(module_info);

            // Use ad_hoc_solve to access the solver and get all class attributes
            let result = transaction.ad_hoc_solve(&handle, |solver| {
                let class_instance_type = Type::ClassType(class_type.clone());

                // Get all field names from the class
                let field_names: Vec<_> = class_type.class_object().fields().collect();

                let mut class_attributes = Vec::new();

                // For each field, get its type using the solver
                for field_name in field_names {
                    let attribute_type = solver.type_of_attr_get(
                        &class_instance_type,
                        field_name, // This is already &Name from the iterator
                        ruff_text_size::TextRange::default(), // Use a default range
                        &crate::error::collector::ErrorCollector::new(
                            module_info.clone(),
                            crate::error::style::ErrorStyle::Never,
                        ), // Create a temporary error collector
                        None,       // No context
                        "get_type_attributes", // Context description
                    );

                    // Convert the pyrefly type to TSP type
                    let tsp_type = convert_to_tsp_type(attribute_type);

                    let attribute = tsp::Attribute {
                        name: field_name.as_str().to_owned(),
                        type_: tsp_type,
                        owner: None, // Could be enhanced to include owner type
                        bound_type: None,
                        flags: tsp::AttributeFlags::NONE.0,
                        decls: Vec::new(), // Could be enhanced with declaration info
                    };
                    class_attributes.push(attribute);
                }

                class_attributes
            });

            if let Some(class_attrs) = result {
                attributes.extend(class_attrs);
            }
        }

        Type::Function(function_type) => {
            // For functions, extract parameters and return type as attributes
            // Reuse logic from get_function_parts
            let signature = &function_type.signature;

            // Extract parameters as attributes
            match &signature.params {
                crate::types::callable::Params::List(param_list) => {
                    for (i, param) in param_list.items().iter().enumerate() {
                        let param_name = get_param_name(param, i);
                        let param_type = get_param_type(param);
                        let tsp_type = convert_to_tsp_type(param_type.clone());

                        let attribute = tsp::Attribute {
                            name: param_name,
                            type_: tsp_type,
                            owner: None,
                            bound_type: None,
                            flags: tsp::AttributeFlags::NONE.0,
                            decls: Vec::new(),
                        };
                        attributes.push(attribute);
                    }
                }
                crate::types::callable::Params::Ellipsis => {
                    // Handle ellipsis parameters
                    let attribute = tsp::Attribute {
                        name: "...".to_owned(),
                        type_: convert_to_tsp_type(Type::Any(
                            crate::types::types::AnyStyle::Implicit,
                        )),
                        owner: None,
                        bound_type: None,
                        flags: tsp::AttributeFlags::NONE.0,
                        decls: Vec::new(),
                    };
                    attributes.push(attribute);
                }
                crate::types::callable::Params::ParamSpec(types, param_spec) => {
                    // Handle concatenated parameters with ParamSpec
                    for (i, param_type) in types.iter().enumerate() {
                        let param_name = format!("param{i}");
                        let tsp_type = convert_to_tsp_type(param_type.clone());

                        let attribute = tsp::Attribute {
                            name: param_name,
                            type_: tsp_type,
                            owner: None,
                            bound_type: None,
                            flags: tsp::AttributeFlags::NONE.0,
                            decls: Vec::new(),
                        };
                        attributes.push(attribute);
                    }

                    // Add the ParamSpec itself
                    let param_spec_tsp = convert_to_tsp_type(param_spec.clone());
                    let attribute = tsp::Attribute {
                        name: "*param_spec".to_owned(),
                        type_: param_spec_tsp,
                        owner: None,
                        bound_type: None,
                        flags: tsp::AttributeFlags::NONE.0,
                        decls: Vec::new(),
                    };
                    attributes.push(attribute);
                }
            }

            // Add return type as an attribute
            let return_tsp_type = convert_to_tsp_type(signature.ret.clone());
            let return_attribute = tsp::Attribute {
                name: "return".to_owned(),
                type_: return_tsp_type,
                owner: None,
                bound_type: None,
                flags: tsp::AttributeFlags::NONE.0,
                decls: Vec::new(),
            };
            attributes.push(return_attribute);
        }

        Type::Callable(callable_type) => {
            // For callable types, extract parameters and return type as attributes
            // Similar logic to Function but working directly with Callable
            match &callable_type.params {
                crate::types::callable::Params::List(param_list) => {
                    for (i, param) in param_list.items().iter().enumerate() {
                        let param_name = get_param_name(param, i);
                        let param_type = get_param_type(param);
                        let tsp_type = convert_to_tsp_type(param_type.clone());

                        let attribute = tsp::Attribute {
                            name: param_name,
                            type_: tsp_type,
                            owner: None,
                            bound_type: None,
                            flags: tsp::AttributeFlags::NONE.0,
                            decls: Vec::new(),
                        };
                        attributes.push(attribute);
                    }
                }
                crate::types::callable::Params::Ellipsis => {
                    let attribute = tsp::Attribute {
                        name: "...".to_owned(),
                        type_: convert_to_tsp_type(Type::Any(
                            crate::types::types::AnyStyle::Implicit,
                        )),
                        owner: None,
                        bound_type: None,
                        flags: tsp::AttributeFlags::NONE.0,
                        decls: Vec::new(),
                    };
                    attributes.push(attribute);
                }
                crate::types::callable::Params::ParamSpec(types, param_spec) => {
                    for (i, param_type) in types.iter().enumerate() {
                        let param_name = format!("param{i}");
                        let tsp_type = convert_to_tsp_type(param_type.clone());

                        let attribute = tsp::Attribute {
                            name: param_name,
                            type_: tsp_type,
                            owner: None,
                            bound_type: None,
                            flags: tsp::AttributeFlags::NONE.0,
                            decls: Vec::new(),
                        };
                        attributes.push(attribute);
                    }

                    let param_spec_tsp = convert_to_tsp_type(param_spec.clone());
                    let attribute = tsp::Attribute {
                        name: "*param_spec".to_owned(),
                        type_: param_spec_tsp,
                        owner: None,
                        bound_type: None,
                        flags: tsp::AttributeFlags::NONE.0,
                        decls: Vec::new(),
                    };
                    attributes.push(attribute);
                }
            }

            // Add return type as an attribute
            let return_tsp_type = convert_to_tsp_type(callable_type.ret.clone());
            let return_attribute = tsp::Attribute {
                name: "return".to_owned(),
                type_: return_tsp_type,
                owner: None,
                bound_type: None,
                flags: tsp::AttributeFlags::NONE.0,
                decls: Vec::new(),
            };
            attributes.push(return_attribute);
        }

        Type::Module(module_type) => {
            // For modules, we could potentially list exported symbols
            // For now, we'll return empty as module introspection is complex
            tsp_debug!(
                "Module attribute extraction not yet implemented for: {:?}",
                module_type
            );
        }

        Type::Overload(overload_type) => {
            // For overloaded functions, we could show all overload signatures
            // For now, we'll return a summary
            for (i, signature) in overload_type.signatures.iter().enumerate() {
                let signature_name = format!("overload_{i}");
                match signature {
                    crate::types::types::OverloadType::Function(function) => {
                        let function_type = Type::Function(Box::new(function.clone()));
                        let tsp_type = convert_to_tsp_type(function_type);

                        let attribute = tsp::Attribute {
                            name: signature_name,
                            type_: tsp_type,
                            owner: None,
                            bound_type: None,
                            flags: tsp::AttributeFlags::NONE.0,
                            decls: Vec::new(),
                        };
                        attributes.push(attribute);
                    }
                    crate::types::types::OverloadType::Forall(forall) => {
                        let function_type = Type::Function(Box::new(forall.body.clone()));
                        let tsp_type = convert_to_tsp_type(function_type);

                        let attribute = tsp::Attribute {
                            name: signature_name,
                            type_: tsp_type,
                            owner: None,
                            bound_type: None,
                            flags: tsp::AttributeFlags::NONE.0,
                            decls: Vec::new(),
                        };
                        attributes.push(attribute);
                    }
                }
            }
        }

        _ => {
            // For other types (primitives, unions, etc.), there are no attributes
            tsp_debug!("No attributes available for type: {:?}", pyrefly_type);
        }
    }

    Ok(attributes)
}

impl TspServer {
    pub fn get_type_attributes(
        &self,
        transaction: &Transaction<'_>,
        params: tsp::GetTypeAttributesParams,
    ) -> Result<Option<Vec<tsp::Attribute>>, ResponseError> {
        // Check if the snapshot is still valid
        if params.snapshot != self.current_snapshot() {
            return Err(snapshot_outdated_error());
        }

        tsp_debug!("Getting attributes for type: {:?}", params.type_);

        // Convert TSP type to pyrefly type
        let pyrefly_type = match self.lookup_type_from_tsp_type(&params.type_) {
            Some(pyrefly_type) => pyrefly_type,
            None => {
                tsp_debug!("Could not convert TSP type to pyrefly type");
                return Ok(Some(Vec::new()));
            }
        };

        // Extract attributes from the pyrefly type
        let attributes = extract_type_attributes(transaction, &pyrefly_type, |module_info| {
            let module_path = module_info.path().clone();
            let module_name = module_info.name();
            let config = self
                .inner
                .state
                .config_finder()
                .python_file(module_name, &module_path);
            Handle::new(module_name, module_path, config.get_sys_info())
        })?;

        tsp_debug!("Found {} attributes", attributes.len());
        Ok(Some(attributes))
    }
}
