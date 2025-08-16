/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Common utilities for TSP request handling that depend on pyrefly types

use tsp_types::*;

/// Convert a pyrefly Type to a TSP Type
pub fn convert_to_tsp_type(py_type: crate::types::types::Type) -> Type {
    use crate::types::types::Type as PyType;

    Type {
        handle: TypeHandle::String(format!("{:p}", &py_type as *const _)),
        category: match &py_type {
            PyType::Any(_) => TypeCategory::Any,
            PyType::Function(_) | PyType::Callable(_) => TypeCategory::Function,
            PyType::Overload(_) => TypeCategory::Overloaded,
            PyType::ClassType(_) | PyType::ClassDef(_) => TypeCategory::Class,
            PyType::Module(_) => TypeCategory::Module,
            PyType::Union(_) => TypeCategory::Union,
            PyType::TypeVar(_) => TypeCategory::TypeVar,
            _ => TypeCategory::Any,
        },
        flags: calculate_type_flags(&py_type),
        module_name: extract_module_name(&py_type),
        name: py_type.to_string(),
        category_flags: 0,
        decl: None,
        alias_name: None,
    }
}

/// Convert a pyrefly Type to a TSP Type with a specific handle
pub fn convert_to_tsp_type_with_handle(py_type: crate::types::types::Type, handle: String) -> Type {
    use crate::types::types::Type as PyType;

    Type {
        handle: TypeHandle::String(handle),
        category: match &py_type {
            PyType::Any(_) => TypeCategory::Any,
            PyType::Function(_) | PyType::Callable(_) => TypeCategory::Function,
            PyType::Overload(_) => TypeCategory::Overloaded,
            PyType::ClassType(_) | PyType::ClassDef(_) => TypeCategory::Class,
            PyType::Module(_) => TypeCategory::Module,
            PyType::Union(_) => TypeCategory::Union,
            PyType::TypeVar(_) => TypeCategory::TypeVar,
            _ => TypeCategory::Any,
        },
        flags: calculate_type_flags(&py_type),
        module_name: extract_module_name(&py_type),
        name: py_type.to_string(),
        category_flags: 0,
        decl: None,
        alias_name: None,
    }
}

/// Calculate type flags for a pyrefly Type
pub fn calculate_type_flags(py_type: &crate::types::types::Type) -> TypeFlags {
    use crate::types::types::Type as PyType;
    match py_type {
        PyType::ClassDef(_) => TypeFlags::INSTANTIABLE,
        PyType::ClassType(_) => TypeFlags::INSTANCE,
        PyType::Function(_) | PyType::Callable(_) => TypeFlags::CALLABLE,
        PyType::Literal(_) => TypeFlags::LITERAL,
        PyType::TypeAlias(_) => TypeFlags::FROM_ALIAS,
        _ => TypeFlags::NONE,
    }
}

/// Extract module name from a pyrefly Type
pub fn extract_module_name(py_type: &crate::types::types::Type) -> Option<ModuleName> {
    use crate::types::types::Type as PyType;

    match py_type {
        PyType::ClassType(ct) => Some(convert_module_name(ct.qname().module_name())),
        PyType::ClassDef(cd) => Some(convert_module_name(cd.qname().module_name())),
        PyType::Module(m) => Some(convert_module_name_from_string(&m.to_string())),
        _ => None,
    }
}

/// Convert a pyrefly ModuleName to a TSP ModuleName
pub fn convert_module_name(pyrefly_module: pyrefly_python::module_name::ModuleName) -> ModuleName {
    ModuleName {
        leading_dots: 0, // pyrefly modules don't have leading dots in this context
        name_parts: pyrefly_module
            .as_str()
            .split('.')
            .map(|s| s.to_owned())
            .collect(),
    }
}

/// Convert TSP ModuleName back to pyrefly ModuleName
pub fn convert_tsp_module_name_to_pyrefly(
    tsp_module: &ModuleName,
) -> pyrefly_python::module_name::ModuleName {
    let module_str = tsp_module.name_parts.join(".");

    // Normalize __builtins__ to builtins so that the builtins module can be found
    let normalized_module_str = if module_str == "__builtins__" {
        "builtins".to_owned()
    } else {
        module_str
    };

    pyrefly_python::module_name::ModuleName::from_str(&normalized_module_str)
}

/// Convert a module string to a TSP ModuleName
pub fn convert_module_name_from_string(module_str: &str) -> ModuleName {
    ModuleName {
        leading_dots: 0,
        name_parts: module_str.split('.').map(|s| s.to_owned()).collect(),
    }
}
