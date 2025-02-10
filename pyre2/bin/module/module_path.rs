/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ffi::OsStr;
use std::fmt;
use std::fmt::Display;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use dupe::Dupe;

use crate::dunder;

#[derive(Debug, Clone, Dupe, Copy, PartialEq, Eq, Hash, Default)]
pub enum ModuleStyle {
    /// .py - executable code.
    #[default]
    Executable,
    /// .pyi - just types that form an interface.
    Interface,
    /// directory - a namespace package.
    Namespace,
}

/// Store information about where a module is sourced from.
#[derive(Debug, Clone, Dupe, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct ModulePath(Arc<ModulePathDetails>);

#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub enum ModulePathDetails {
    /// The module source comes from a file on disk.
    /// The path may point to a directory, in which case the module will be backed by namespace package.
    FileSystem(PathBuf),
    /// The module source comes from memory.
    Memory(PathBuf),
    /// The module source comes from typeshed bundled with Pyre (which gets stored in-memory).
    /// The path is relative to the root of the typeshed directory.
    BundledTypeshed(PathBuf),
}

fn is_path_init(path: &Path) -> bool {
    path.file_stem() == Some(OsStr::new(dunder::INIT.as_str()))
}

impl ModuleStyle {
    fn of_path(path: &Path) -> Self {
        let extension = path.extension();
        if extension == Some(OsStr::new("pyi")) {
            ModuleStyle::Interface
        } else if extension == Some(OsStr::new("py")) {
            ModuleStyle::Executable
        } else {
            ModuleStyle::Namespace
        }
    }
}

impl Display for ModulePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &*self.0 {
            ModulePathDetails::FileSystem(path) | ModulePathDetails::Memory(path) => {
                write!(f, "{}", path.display())
            }
            ModulePathDetails::BundledTypeshed(relative_path) => {
                write!(
                    f,
                    "bundled /pyre2/third_party/typeshed/{}",
                    relative_path.display()
                )
            }
        }
    }
}

impl ModulePath {
    pub fn filesystem(path: PathBuf) -> Self {
        Self(Arc::new(ModulePathDetails::FileSystem(path)))
    }

    pub fn memory(path: PathBuf) -> Self {
        Self(Arc::new(ModulePathDetails::Memory(path)))
    }

    pub fn bundled_typeshed(relative_path: PathBuf) -> Self {
        Self(Arc::new(ModulePathDetails::BundledTypeshed(relative_path)))
    }

    pub fn is_init(&self) -> bool {
        self.as_path().is_some_and(is_path_init)
    }

    /// Whether things imported by this module are reexported.
    pub fn style(&self) -> ModuleStyle {
        self.as_path().map(ModuleStyle::of_path).unwrap_or_default()
    }

    pub fn is_interface(&self) -> bool {
        self.style() == ModuleStyle::Interface
    }

    /// Convert to a path, that may not exist on disk.
    fn as_path(&self) -> Option<&Path> {
        match &*self.0 {
            ModulePathDetails::FileSystem(path)
            | ModulePathDetails::BundledTypeshed(path)
            | ModulePathDetails::Memory(path) => Some(path),
        }
    }

    pub fn details(&self) -> &ModulePathDetails {
        &self.0
    }
}
