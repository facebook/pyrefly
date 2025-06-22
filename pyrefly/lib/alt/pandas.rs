/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Type inference support for pandas DataFrame objects

use std::sync::Arc;

use crate::types::types::Type;
use crate::types::class::Class;
use crate::module::module_info::ModuleInfo;
use crate::module::module_name::ModuleName;

/// Handles inference for pandas DataFrame types through re-exports
pub fn infer_pandas_type(
    name: &str,
    module_path: &str,
    class_info: &ModuleInfo,
) -> Option<Type> {
    // Map of pandas types to their implementation modules
    let type_map = {
        let mut map = std::collections::HashMap::new();
        map.insert("DataFrame", "pandas.core.frame");
        map.insert("Series", "pandas.core.series"); 
        map.insert("Index", "pandas.core.index");
        map
    };

    // Look up implementation module for this type
    if let Some(impl_module) = type_map.get(name) {
        // Create class type pointing to implementation
        let class = Class::new(
            class_info.clone(),
            name.to_string(),
            ModuleName::new(impl_module),
            vec![],
        );
        Some(Type::ClassDef(Arc::new(class))) 
    } else {
        None
    }
}

/// Checks if this is a pandas re-export module
pub fn is_pandas_reexport(module_path: &str) -> bool {
    (module_path == "pandas" || module_path.starts_with("pandas.")) && 
    !module_path.starts_with("pandas.core.")
}
