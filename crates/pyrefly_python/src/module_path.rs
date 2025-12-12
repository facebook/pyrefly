/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::Ordering;
use std::ffi::OsStr;
use std::fmt;
use std::fmt::Display;
use std::ops::Deref;
use std::path::Path;
use std::path::PathBuf;

use dupe::Dupe;
use equivalent::Equivalent;
use serde::Deserialize;
use serde::Serialize;
use serde::Serializer;
use static_interner::Intern;
use static_interner::Interner;

use crate::dunder;
use crate::module_name::ModuleName;

static MODULE_PATH_INTERNER: Interner<PathBuf> = Interner::new();

#[derive(Clone, Dupe, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct ModulePathBuf(Intern<PathBuf>);

impl Deref for ModulePathBuf {
    type Target = PathBuf;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Serialize for ModulePathBuf {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ModulePathBuf {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let path: &Path = Deserialize::deserialize(d)?;
        Ok(ModulePathBuf::from_path(path))
    }
}

#[derive(Debug, Hash, PartialEq, Eq)]
struct PathRef<'a>(&'a Path);

impl<'a> Equivalent<PathBuf> for PathRef<'a> {
    fn equivalent(&self, key: &PathBuf) -> bool {
        *self.0 == *key
    }
}

impl<'a> From<PathRef<'a>> for PathBuf {
    fn from(value: PathRef<'a>) -> Self {
        value.0.to_path_buf()
    }
}

impl fmt::Debug for ModulePathBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Self as fmt::Display>::fmt(self, f)
    }
}

impl fmt::Display for ModulePathBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", (**self.0).display())
    }
}

impl ModulePathBuf {
    pub fn new(path: PathBuf) -> Self {
        Self(MODULE_PATH_INTERNER.intern(path))
    }

    pub fn from_path(path: &Path) -> Self {
        Self(MODULE_PATH_INTERNER.intern(PathRef(path)))
    }
}

#[derive(Debug, Clone, Dupe, Copy, PartialEq, Eq, Hash, Default)]
pub enum ModuleStyle {
    /// .py - executable code.
    #[default]
    Executable,
    /// .pyi - just types that form an interface.
    Interface,
}

impl ModuleStyle {
    pub fn of_path(path: &Path) -> Self {
        if path.extension() == Some("pyi".as_ref()) {
            ModuleStyle::Interface
        } else {
            // Both .py and .ipynb are executable
            ModuleStyle::Executable
        }
    }
}

/// Store information about where a module is sourced from.
#[derive(Debug, Clone, Dupe, PartialEq, Eq, Hash)]
pub struct ModulePath(ModulePathDetails);

#[derive(Debug, Clone, Dupe, PartialOrd, Ord, PartialEq, Eq, Hash, Serialize)]
pub enum ModulePathDetails {
    /// The module source comes from a file on disk. Probably a `.py` or `.pyi` file.
    FileSystem(ModulePathBuf),
    /// A directory where the module is backed by a namespace package.
    Namespace(ModulePathBuf),
    /// The module source comes from memory, only for files (not namespace).
    Memory(ModulePathBuf),
    /// The module source comes from typeshed bundled with Pyrefly (which gets stored in-memory).
    /// The path is relative to the root of the typeshed/stdlib directory.
    BundledTypeshed(ModulePathBuf),
    /// The module source comes from typeshed bundled with Pyrefly (which gets stored in-memory).
    /// Although the module root is the same the third party stubs are stored in a subdirectory called stubs.
    BundledTypeshedThirdParty(ModulePathBuf),
    /// The module source comes from custom third-party stubs bundled with Pyrefly.
    /// These are stubs not included in typeshed
    BundledThirdParty(ModulePathBuf),
}

impl PartialOrd for ModulePath {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ModulePath {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.0 == other.0 {
            // In the common case of equality (as we usually just matched ModuleName),
            // we can short circuit the comparison entirely.
            Ordering::Equal
        } else {
            self.0.cmp(&other.0)
        }
    }
}

fn is_path_init(path: &Path) -> bool {
    path.file_stem() == Some(dunder::INIT.as_str().as_ref())
}

impl Display for ModulePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            ModulePathDetails::FileSystem(path) | ModulePathDetails::Namespace(path) => {
                write!(f, "{}", path.display())
            }
            ModulePathDetails::Memory(path) => {
                write!(f, "in-memory {}", path.display())
            }
            ModulePathDetails::BundledTypeshed(relative_path) => {
                write!(
                    f,
                    "bundled /crates/pyrefly_bundled/third_party/typeshed/stdlib/{}",
                    relative_path.display()
                )
            }
            ModulePathDetails::BundledTypeshedThirdParty(relative_path) => {
                write!(
                    f,
                    "bundled /crates/pyrefly_bundled/third_party/typeshed/stubs/{}",
                    relative_path.display()
                )
            }
            ModulePathDetails::BundledThirdParty(relative_path) => {
                write!(
                    f,
                    "bundled /crates/pyrefly_bundled/third_party/stubs/{}",
                    relative_path.display()
                )
            }
        }
    }
}

impl Serialize for ModulePath {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            ModulePathDetails::FileSystem(path)
            | ModulePathDetails::Memory(path)
            | ModulePathDetails::Namespace(path) => path.serialize(serializer),
            ModulePathDetails::BundledTypeshed(_)
            | ModulePathDetails::BundledTypeshedThirdParty(_)
            | ModulePathDetails::BundledThirdParty(_) => self.to_string().serialize(serializer),
        }
    }
}

impl ModulePath {
    fn new(details: ModulePathDetails) -> Self {
        Self(details)
    }

    pub fn filesystem(path: PathBuf) -> Self {
        Self::new(ModulePathDetails::FileSystem(ModulePathBuf::new(path)))
    }

    pub fn namespace(path: PathBuf) -> Self {
        Self::new(ModulePathDetails::Namespace(ModulePathBuf::new(path)))
    }

    pub fn memory(path: PathBuf) -> Self {
        Self::new(ModulePathDetails::Memory(ModulePathBuf::new(path)))
    }

    pub fn bundled_typeshed(relative_path: PathBuf) -> Self {
        Self::new(ModulePathDetails::BundledTypeshed(ModulePathBuf::new(
            relative_path,
        )))
    }

    pub fn bundled_typeshed_third_party(relative_path: PathBuf) -> Self {
        Self::new(ModulePathDetails::BundledTypeshedThirdParty(
            ModulePathBuf::new(relative_path),
        ))
    }

    pub fn bundled_third_party(relative_path: PathBuf) -> Self {
        Self::new(ModulePathDetails::BundledThirdParty(ModulePathBuf::new(
            relative_path,
        )))
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

    pub fn is_notebook(&self) -> bool {
        self.as_path().extension() == Some("ipynb".as_ref())
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
        match &self.0 {
            ModulePathDetails::FileSystem(path)
            | ModulePathDetails::BundledTypeshed(path)
            | ModulePathDetails::BundledTypeshedThirdParty(path)
            | ModulePathDetails::BundledThirdParty(path)
            | ModulePathDetails::Memory(path)
            | ModulePathDetails::Namespace(path) => path,
        }
    }

    /// Convert to a path, that may not exist on disk.
    pub fn module_path_buf(&self) -> ModulePathBuf {
        ModulePathBuf::from_path(self.as_path())
    }

    /// For nominal types, we consider FileSystem and Memory to be equal. This is important in the
    /// IDE when an in-memory module reaches its own nominal type through a cycle, where we end up
    /// with two classes, one from the Memory path and one from the FileSystem path.
    pub fn to_key_eq(&self) -> ModulePath {
        match &self.0 {
            ModulePathDetails::FileSystem(path) | ModulePathDetails::Memory(path) => {
                ModulePath::new(ModulePathDetails::FileSystem(*path))
            }
            ModulePathDetails::Namespace(path) => {
                ModulePath::new(ModulePathDetails::Namespace(*path))
            }
            ModulePathDetails::BundledTypeshed(path) => {
                ModulePath::new(ModulePathDetails::BundledTypeshed(*path))
            }
            ModulePathDetails::BundledTypeshedThirdParty(path) => {
                ModulePath::new(ModulePathDetails::BundledTypeshedThirdParty(*path))
            }
            ModulePathDetails::BundledThirdParty(path) => {
                ModulePath::new(ModulePathDetails::BundledThirdParty(*path))
            }
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
