/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
*
* This source code is licensed under the M                                let attr_type = solver.type_of_attr_get(
                       &base_type,
                       &attr_name,
                       ruff_text_size::TextRange::default(),
                       &crate::error::collector::ErrorCollector::new(
                           module_info.clone(),
                           crate::error::style::ErrorStyle::Never,
                       ),
                       None,
                       "search_for_type_attribute",
                   );

                   attr_type attr_type = solver.type_of_attr_get(
                       &base_type,
                       &attr_name,
                       ruff_text_size::TextRange::default(),
                       &crate::error::collector::ErrorCollector::new(
                           module_info.clone(),
                           crate::error::style::ErrorStyle::Never,
                       ),
                       None,
                       "search_for_type_attribute",
                   );

                   attr_typefound in the
* LICENSE file in the root directory of this source tree.
*/

//! TSP search for type attribute request implementation

use dupe::Dupe;
use lsp_server::ResponseError;
use tsp_types::snapshot_outdated_error;
use tsp_types::tsp_debug;
use tsp_types::{self as tsp};

use crate::module::module_info::ModuleInfo;
use crate::state::handle::Handle;
use crate::state::state::Transaction;
use crate::tsp::server::TspServer;

/// Convert a pyrefly Type to TSP Attribute format
///
/// This helper function creates a TSP Attribute from a pyrefly Type and attribute name.
pub fn create_tsp_attribute_from_type(
    attribute_type: crate::types::types::Type,
    attribute_name: &str,
    type_converter: impl Fn(crate::types::types::Type) -> tsp::Type,
) -> tsp::Attribute {
    let tsp_type = type_converter(attribute_type);
    let flags = tsp::AttributeFlags::NONE.0;
    tsp::Attribute {
        name: attribute_name.to_owned(),
        type_: tsp_type,
        owner: None,
        bound_type: None,
        flags,
        decls: Vec::new(),
    }
}

impl TspServer {
    pub fn search_for_type_attribute(
        &self,
        transaction: &Transaction<'_>,
        params: tsp::SearchForTypeAttributeParams,
    ) -> Result<Option<tsp::Attribute>, ResponseError> {
        // Check if the snapshot is still valid
        if params.snapshot != self.current_snapshot() {
            return Err(snapshot_outdated_error());
        }

        tsp_debug!(
            "Searching for attribute '{}' with access flags: {:?}",
            params.attribute_name,
            params.access_flags
        );

        // Get the internal type from the start_type handle
        let internal_type = match self.lookup_type_from_tsp_type(&params.start_type) {
            Some(t) => t,
            None => {
                tsp_debug!(
                    "Could not resolve type handle: {:?}",
                    params.start_type.handle
                );
                return Ok(None);
            }
        };

        // Only work on class types - this method is specifically for class attribute lookup
        match &internal_type {
            crate::types::types::Type::ClassType(class_type) => {
                // Convert string attribute name to ruff_python_ast::name::Name
                use ruff_python_ast::name::Name;
                let attr_name = Name::new(&params.attribute_name);

                // Get the module info from the class type
                let class_qname = class_type.class_object().qname();
                let module_info = class_qname.module();

                // Create a handle for the module containing the class
                let handle = self.create_handle_for_module(module_info);

                // Use ad_hoc_solve to access the solver and get the attribute
                let maybe_attribute_type = transaction.ad_hoc_solve(&handle, |solver| {
                    use tsp_types::AttributeAccessFlags;

                    // Check if SKIP_INSTANCE_ATTRIBUTES flag is set
                    let skip_instance_attrs = params
                        .access_flags
                        .contains(AttributeAccessFlags::SKIP_INSTANCE_ATTRIBUTES);

                    // Choose the appropriate base type for attribute lookup:
                    // - ClassType (instance) includes both instance and class attributes
                    // - ClassDef (class definition) includes only class attributes
                    let base_type = if skip_instance_attrs {
                        // When skipping instance attributes, use ClassDef to only find class attributes
                        crate::types::types::Type::ClassDef(class_type.class_object().dupe())
                    } else {
                        // When including instance attributes, use ClassType to find both
                        crate::types::types::Type::ClassType(class_type.clone())
                    };

                    solver.type_of_attr_get(
                        &base_type,
                        &attr_name,
                        ruff_text_size::TextRange::default(),
                        &crate::error::collector::ErrorCollector::new(
                            module_info.clone(),
                            crate::error::style::ErrorStyle::Never,
                        ),
                        None,
                        "search_for_type_attribute",
                    )
                });

                match maybe_attribute_type {
                    Some(attr_type) => {
                        match &attr_type {
                            crate::types::types::Type::Any(
                                crate::types::types::AnyStyle::Error,
                            ) => {
                                tsp_debug!(
                                    "Attribute '{}' not found in class type (got error type)",
                                    params.attribute_name
                                );
                                Ok(None)
                            }
                            _ => {
                                tsp_debug!(
                                    "Found attribute '{}' in class type with type: {:?}",
                                    params.attribute_name,
                                    attr_type
                                );

                                // Convert to TSP attribute using standalone function
                                let tsp_attribute = create_tsp_attribute_from_type(
                                    attr_type,
                                    &params.attribute_name,
                                    |attr_type| self.convert_and_register_type(attr_type),
                                );
                                Ok(Some(tsp_attribute))
                            }
                        }
                    }
                    None => {
                        tsp_debug!(
                            "Attribute '{}' not found in class type",
                            params.attribute_name
                        );
                        Ok(None)
                    }
                }
            }
            _ => {
                tsp_debug!(
                    "search_for_type_attribute only works on class types, got: {:?}",
                    internal_type
                );
                Ok(None)
            }
        }
    }

    /// Create a handle for a module (Server-specific functionality)
    fn create_handle_for_module(&self, module_info: &ModuleInfo) -> Handle {
        let module_path = module_info.path().clone();
        let module_name = module_info.name();
        let config = self
            .inner
            .state
            .config_finder()
            .python_file(module_name, &module_path);
        Handle::new(module_name, module_path, config.get_sys_info())
    }
}
