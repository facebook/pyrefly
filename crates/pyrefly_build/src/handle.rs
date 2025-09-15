/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::sys_info::SysInfo;

#[derive(Debug, Clone, Dupe, PartialEq, Eq, Hash)]
pub struct Handle {
    module: ModuleName,
    path: ModulePath,
    sys_info: SysInfo,
}

impl Handle {
    pub fn new(module: ModuleName, path: ModulePath, sys_info: SysInfo) -> Self {
        Self {
            module,
            path,
            sys_info,
        }
    }

    pub fn module(&self) -> ModuleName {
        self.module
    }

    pub fn path(&self) -> &ModulePath {
        &self.path
    }

    pub fn sys_info(&self) -> &SysInfo {
        &self.sys_info
    }
}
