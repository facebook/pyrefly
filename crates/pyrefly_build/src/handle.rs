/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_name::ModuleNameWithKind;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::sys_info::SysInfo;

/// A handle to a Python module, containing its name, path, and system information.
#[derive(Debug, Clone, Dupe, PartialEq, Eq, Hash)]
pub struct Handle {
    module: ModuleNameWithKind,
    path: ModulePath,
    sys_info: SysInfo,
}

impl Handle {
    /// Create a new handle with a guaranteed module name.
    pub fn new(module: ModuleName, path: ModulePath, sys_info: SysInfo) -> Self {
        Self {
            module: ModuleNameWithKind::guaranteed(module),
            path,
            sys_info,
        }
    }

    /// Create a new handle from a ModuleNameWithKind.
    pub fn from_with_module_name_kind(
        module: ModuleNameWithKind,
        path: ModulePath,
        sys_info: SysInfo,
    ) -> Self {
        Self {
            module,
            path,
            sys_info,
        }
    }

    /// Get the underlying module name.
    pub fn module(&self) -> ModuleName {
        self.module.name()
    }

    /// Get the module name kind (guaranteed or fallback).
    pub fn module_kind(&self) -> ModuleNameWithKind {
        self.module
    }

    pub fn path(&self) -> &ModulePath {
        &self.path
    }

    pub fn sys_info(&self) -> &SysInfo {
        &self.sys_info
    }

    /// Returns true if this handle's module name was created using fallback heuristics.
    /// When true, the module name is not reliable for determining project structure.
    pub fn is_fallback(&self) -> bool {
        self.module.is_fallback()
    }
}
