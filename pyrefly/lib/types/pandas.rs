/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;
use std::collections::HashMap;

use crate::types::types::Type;
use crate::types::class::Class;
use crate::types::class::ClassType;
use crate::module::module_info::ModuleInfo;
use crate::module::module_name::ModuleName;

/// Maps of pandas types to their implementation details
lazy_static! {
    static ref PANDAS_TYPE_MAP: HashMap<&'static str, &'static str> = {
        let mut map = HashMap::new();
        map.insert("DataFrame", "pandas.core.frame");
        map.insert("Series", "pandas.core.series");
        map.insert("Index", "pandas.core.index");
        map
    };

    static ref PANDAS_ATTRIBUTE_MAP: HashMap<&'static str, &'static str> = {
        let mut map = HashMap::new();
        // DataFrame attributes
        map.insert("columns", "pandas.core.index.Index");
        map.insert("index", "pandas.core.index.Index");
        map.insert("dtypes", "pandas.core.series.Series");
        // Series attributes  
        map.insert("values", "numpy.ndarray");
        map
    };
}

/// Resolves pandas type hints through re-exports and attribute lookups
pub fn resolve_pandas_type(
    name: &str,
    module_path: &str,
    class_info: &ModuleInfo,
    attr_name: Option<&str>,
) -> Option<Type> {
    // For attribute lookups, check the attribute map
    if let Some(attr) = attr_name {
        if let Some(impl_type) = PANDAS_ATTRIBUTE_MAP.get(attr) {
            let (module, name) = impl_type.rsplit_once('.').unwrap();
            return Some(Type::ClassType(ClassType::new(
                Class::new(
                    class_info.clone(), 
                    name.to_string(),
                    ModuleName::new(module),
                    vec![],
                ),
                Default::default(),
            )));
        }
    }

    // For direct type lookups, check the type map
    if let Some(impl_module) = PANDAS_TYPE_MAP.get(name) {
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
