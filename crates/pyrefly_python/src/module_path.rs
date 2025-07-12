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
use pyrefly_util::with_hash::WithHash;
use serde::Serialize;
use serde::Serializer;

use crate::dunder;
use crate::module_name::ModuleName;

#[derive(Debug, Clone, Dupe, Copy, PartialEq, Eq, Hash, Default)]
pub enum ModuleStyle {
    /// .py - executable code.
    #[default]
    Executable,
    /// .pyi - just types that form an interface.
    Interface,
}

/// Store information about where a module is sourced from.
#[derive(Debug, Clone, Dupe, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct ModulePath(Arc<WithHash<ModulePathDetails>>);

#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub enum ModulePathDetails {
    /// The module source comes from a file on disk. Probably a `.py` or `.pyi` file.
    FileSystem(PathBuf),
    /// A directory where the module is backed by a namespace package.
    Namespace(PathBuf),
    /// The module source comes from memory, only for files (not namespace).
    Memory(PathBuf),
    /// The module source comes from typeshed bundled with Pyrefly (which gets stored in-memory).
    /// The path is relative to the root of the typeshed directory.
    BundledTypeshed(PathBuf),
}

fn is_path_init(path: &Path) -> bool {
    path.file_stem() == Some(dunder::INIT.as_str().as_ref())
}

impl ModuleStyle {
    fn of_path(path: &Path) -> Self {
        if path.extension() == Some("pyi".as_ref()) {
            ModuleStyle::Interface
        } else {
            ModuleStyle::Executable
        }
    }
}

impl Display for ModulePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &**self.0 {
            ModulePathDetails::FileSystem(path) | ModulePathDetails::Namespace(path) => {
                write!(f, "{}", path.display())
            }
            ModulePathDetails::Memory(path) => {
                write!(f, "in-memory {}", path.display())
            }
            ModulePathDetails::BundledTypeshed(relative_path) => {
                write!(
                    f,
                    "bundled /pyrefly/third_party/typeshed/stdlib/{}",
                    relative_path.display()
                )
            }
        }
    }
}

impl Serialize for ModulePath {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match &**self.0 {
            ModulePathDetails::FileSystem(path)
            | ModulePathDetails::Memory(path)
            | ModulePathDetails::Namespace(path) => path.serialize(serializer),
            ModulePathDetails::BundledTypeshed(_) => self.to_string().serialize(serializer),
        }
    }
}

impl ModulePath {
    fn new(details: ModulePathDetails) -> Self {
        Self(Arc::new(WithHash::new(details)))
    }

    pub fn filesystem(path: PathBuf) -> Self {
        Self::new(ModulePathDetails::FileSystem(path))
    }

    pub fn namespace(path: PathBuf) -> Self {
        Self::new(ModulePathDetails::Namespace(path))
    }

    pub fn memory(path: PathBuf) -> Self {
        Self::new(ModulePathDetails::Memory(path))
    }

    pub fn bundled_typeshed(relative_path: PathBuf) -> Self {
        Self::new(ModulePathDetails::BundledTypeshed(relative_path))
    }

    pub fn is_init(&self) -> bool {
        is_path_init(self.as_path())
    }

    /// Whether things imported by this module are reexported.
    pub fn style(&self) -> ModuleStyle {
        ModuleStyle::of_path(self.as_path())
    }

    pub fn is_interface(&self) -> bool {
        self.style() == ModuleStyle::Interface
    }

    /// Attempt to match the given [`ModuleName`]'s components to this `ModulePath`,
    /// returning the directory that is the import root for the `ModuleName`, if
    /// *all* module components could be matched to directories. `ModulePath`s
    /// with `-stubs` components are matched iff the *first* `ModuleName` component
    /// prefix matches the directory it's being compared to.
    ///
    /// Example:
    /// - `/some/path/to/root/a/b/c/d.py`, `a.b.c.d` -> `Some(/some/path/to/root)`
    /// - `/some/path/to/root/a-stubs/b/c/d.py`, `a.b.c.d` -> `Some(/some/path/to/root)`
    /// - `/some/path/to/root/a/b/c/d.py`, `z.b.c.d` -> `None`
    /// - `/some/path/to/root/a-stubs/b/c/d.py`, `a.b.z.d` -> `None`
    /// - `/some/path/to/root/a-stubs/b/c/d.py`, `root.a.b.c.d` -> `None`
    ///   - because `a` can't match `a-stubs` if we're not looking at the first
    ///     component of the `ModuleName`
    pub fn root_of(&self, name: ModuleName) -> Option<PathBuf> {
        if matches!(self.details(), ModulePathDetails::BundledTypeshed(_)) {
            return None;
        }
        let mut path = self.as_path().to_path_buf();
        path.set_extension("");

        if path.file_name() == Some(dunder::INIT.as_str().as_ref()) {
            path.pop();
        }

        let components = name.components();
        let mut components = components.iter().rev().peekable();
        while let Some(part) = components.next() {
            let file_name = path.file_name();

            // does this `part` match the next part of the `path`?
            let direct_match = file_name == Some(part.as_str().as_ref());
            // if we're looking at the first component (import root) of the
            // `ModuleName`, does it match the `part` if we postfix `-stubs`?
            let stubs_match = components.peek().is_none()
                && file_name == Some(OsStr::new(&(part.to_string() + "-stubs")));

            if !(direct_match || stubs_match) {
                return None;
            }
            path.pop();
        }
        Some(path)
    }

    /// Convert to a path, that may not exist on disk.
    pub fn as_path(&self) -> &Path {
        match &**self.0 {
            ModulePathDetails::FileSystem(path)
            | ModulePathDetails::BundledTypeshed(path)
            | ModulePathDetails::Memory(path)
            | ModulePathDetails::Namespace(path) => path,
        }
    }

    pub fn details(&self) -> &ModulePathDetails {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_root_of() {
        let path = ModulePath::filesystem(PathBuf::from("hello/foo/bar/baz.py"));
        assert_eq!(
            path.root_of(ModuleName::from_str("foo.bar.baz")),
            Some(PathBuf::from("hello")),
        );
        assert_eq!(
            path.root_of(ModuleName::from_str("baz")),
            Some(PathBuf::from("hello/foo/bar")),
        );
        assert_eq!(path.root_of(ModuleName::from_str("baaz")), None);

        let path = ModulePath::filesystem(PathBuf::from("hello/foo/bar/__init__.pyi"));
        assert_eq!(
            path.root_of(ModuleName::from_str("foo.bar")),
            Some(PathBuf::from("hello")),
        );
    }

    #[test]
    fn test_root_of_stubs() {
        let path = ModulePath::filesystem(PathBuf::from("hello/foo-stubs/bar/baz.py"));
        assert_eq!(
            path.root_of(ModuleName::from_str("foo.bar.baz")),
            Some(PathBuf::from("hello")),
        );
        assert_eq!(
            path.root_of(ModuleName::from_str("baz")),
            Some(PathBuf::from("hello/foo-stubs/bar")),
        );
        assert_eq!(
            path.root_of(ModuleName::from_str("hello.foo.bar.baz")),
            None,
        );
    }
}
