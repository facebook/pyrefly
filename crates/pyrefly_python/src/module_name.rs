/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ffi::OsString;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;
use std::path::Component;
use std::path::Path;
use std::path::PathBuf;

use dupe::Dupe;
use equivalent::Equivalent;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::name::Name;
use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;
use serde::Serializer;
use static_interner::Intern;
use static_interner::Interner;
use thiserror::Error;

use crate::dunder;

static MODULE_NAME_INTERNER: Interner<String> = Interner::new();

/// The name of a python module. Examples: `foo.bar.baz`, `.foo.bar`.
#[derive(Clone, Dupe, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct ModuleName(Intern<String>);

impl<To: 'static> Visit<To> for ModuleName {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a To)) {}
}

impl<To: 'static> VisitMut<To> for ModuleName {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut To)) {}
}

impl Serialize for ModuleName {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ModuleName {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s: &str = Deserialize::deserialize(deserializer)?;
        Ok(ModuleName::from_str(s))
    }
}

impl Display for ModuleName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            write!(f, ".")
        } else {
            write!(f, "{}", self.0)
        }
    }
}

#[derive(Debug, Error)]
enum PathConversionError {
    #[error("invalid source file extension (file name: `{file_name}`")]
    InvalidExtension { file_name: String },
    #[error("path component is not UTF-8 encoded: `{component:?}`")]
    ComponentNotUTF8 { component: OsString },
}

impl Debug for ModuleName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut f = f.debug_tuple("ModuleName");
        f.field(&self.as_str());
        f.finish()
    }
}

#[derive(Hash, Eq, PartialEq)]
struct StrRef<'a>(&'a str);

impl Equivalent<String> for StrRef<'_> {
    fn equivalent(&self, key: &String) -> bool {
        self.0 == key
    }
}

impl From<StrRef<'_>> for String {
    fn from(value: StrRef<'_>) -> Self {
        value.0.to_owned()
    }
}

impl ModuleName {
    pub fn builtins() -> Self {
        Self::from_str("builtins")
    }

    pub fn extra_builtins() -> Self {
        Self::from_str("__builtins__")
    }

    pub fn typing() -> Self {
        Self::from_str("typing")
    }

    pub fn typing_extensions() -> Self {
        Self::from_str("typing_extensions")
    }

    pub fn future() -> Self {
        Self::from_str("__future__")
    }

    pub fn types() -> Self {
        Self::from_str("types")
    }

    pub fn warnings() -> Self {
        Self::from_str("warnings")
    }

    pub fn collections() -> Self {
        Self::from_str("collections")
    }

    pub fn enum_() -> Self {
        Self::from_str("enum")
    }

    pub fn abc() -> Self {
        Self::from_str("abc")
    }

    pub fn dataclasses() -> Self {
        Self::from_str("dataclasses")
    }

    pub fn functools() -> Self {
        Self::from_str("functools")
    }

    pub fn type_checker_internals() -> Self {
        Self::from_str("_typeshed._type_checker_internals")
    }

    pub fn collections_abc() -> Self {
        Self::from_str("_collections_abc")
    }

    pub fn pydantic() -> Self {
        Self::from_str("pydantic.main")
    }

    #[allow(dead_code)]
    pub fn pydantic_settings() -> Self {
        Self::from_str("pydantic_settings.main")
    }

    pub fn pydantic_root_model() -> Self {
        Self::from_str("pydantic.root_model")
    }

    pub fn django_models_enums() -> Self {
        Self::from_str("django.db.models.enums")
    }

    pub fn django_models() -> Self {
        Self::from_str("django.db.models.base")
    }

    pub fn django_models_fields() -> Self {
        Self::from_str("django.db.models.fields")
    }

    pub fn django_models_fields_related() -> Self {
        Self::from_str("django.db.models.fields.related")
    }

    pub fn django_models_fields_related_descriptors() -> Self {
        Self::from_str("django.db.models.fields.related_descriptors")
    }

    pub fn django_utils_functional() -> Self {
        Self::from_str("django.utils.functional")
    }

    pub fn pydantic_types() -> Self {
        Self::from_str("pydantic.types")
    }

    /// The "unknown" module name, which corresponds to `__unknown__`.
    /// Used for files directly opened or passed on the command line which aren't on the search path.
    pub fn unknown() -> Self {
        Self::from_str("__unknown__")
    }

    pub fn from_str(x: &str) -> Self {
        ModuleName(MODULE_NAME_INTERNER.intern(StrRef(x)))
    }

    pub fn from_string(x: String) -> Self {
        ModuleName(MODULE_NAME_INTERNER.intern(x))
    }

    pub fn from_name(x: &Name) -> Self {
        Self::from_str(x)
    }

    pub fn from_parts(parts: impl IntoIterator<Item = impl Display + AsRef<str>>) -> Self {
        Self::from_string(itertools::join(parts, "."))
    }

    fn from_relative_path_components(mut components: Vec<&str>) -> anyhow::Result<Self> {
        let last_element = components.pop();
        match last_element {
            None => {}
            Some(file_name) => {
                let splits: Vec<&str> = file_name.rsplitn(2, '.').collect();
                if splits.len() != 2
                    || !(splits[0] == "py" || splits[0] == "pyi" || splits[0] == "ipynb")
                {
                    return Err(anyhow::anyhow!(PathConversionError::InvalidExtension {
                        file_name: file_name.to_owned(),
                    }));
                }
                if splits[1] != dunder::INIT {
                    components.push(splits[1])
                }
            }
        }
        Ok(ModuleName::from_parts(components))
    }

    pub fn from_relative_path(path: &Path) -> anyhow::Result<Self> {
        let mut components = Vec::new();
        for raw_component in path.components() {
            if let Some(component) = raw_component.as_os_str().to_str() {
                components.push(component)
            } else {
                return Err(anyhow::anyhow!(PathConversionError::ComponentNotUTF8 {
                    component: raw_component.as_os_str().to_owned(),
                }));
            }
        }
        Self::from_relative_path_components(components)
    }

    pub fn relative_module_name_between(from: &Path, to: &Path) -> Option<ModuleName> {
        let relative_path = pathdiff::diff_paths(to, from.parent()?)?;
        // In the following loop, we aim to generate a list of components that can be joined by `.`
        // to form a correct relative import module name,
        let mut components = vec![""];
        for raw_component in relative_path.as_path().components() {
            match &raw_component {
                // For each parent, we should create a `.`.
                // The `.` is already provided during the join, so we only need an empty component.
                Component::ParentDir => components.push(""),
                Component::CurDir => {}
                Component::Prefix(_) | Component::RootDir | Component::Normal(_) => {
                    components.push(raw_component.as_os_str().to_str()?);
                }
            };
        }
        Self::from_relative_path_components(components).ok()
    }

    pub fn append(self, name: &Name) -> Self {
        Self::from_string(format!("{}.{}", self.as_str(), name))
    }

    /// Create a new ModuleName instance based off the current instance, with:
    /// - specified number of dots removed
    /// - specified suffix appended
    ///
    /// * `is_init` - Whether the current module is an __init__.py file
    /// * `dots` - The number of dots to remove
    /// * `suffix` - The suffix to append to the current module
    pub fn new_maybe_relative(
        self,
        is_init: bool,
        mut dots: u32,
        suffix: Option<&Name>,
    ) -> Option<Self> {
        if dots == 0
            && let Some(s) = suffix
        {
            return Some(ModuleName::from_name(s));
        }
        let mut components = self.components();
        if is_init {
            dots = dots.saturating_sub(1);
        }
        for _ in 0..dots {
            components.pop()?;
        }
        if let Some(suffix) = suffix {
            components.push(suffix.clone());
        }
        Some(ModuleName::from_parts(components))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn first_component(self) -> Name {
        match self.0.split_once('.') {
            None => Name::new(self.as_str()),
            Some(x) => Name::new(x.0),
        }
    }

    pub fn components(self) -> Vec<Name> {
        self.0.split('.').map(Name::new).collect()
    }

    /// If the module is on the search path, return its name from that path. Otherwise, return None.
    pub fn from_path<'a>(
        path: &Path,
        includes: impl Iterator<Item = &'a PathBuf>,
    ) -> Option<ModuleName> {
        // Return a module name, and a boolean as to whether it is any good.
        fn path_to_module(mut path: &Path) -> Option<ModuleName> {
            if path.file_stem() == Some(dunder::INIT.as_str().as_ref()) {
                path = path.parent()?;
            }
            let mut out = Vec::new();
            let path = path.with_extension("");
            for x in path.components() {
                if let Component::Normal(x) = x
                    && !x.is_empty()
                {
                    out.push(x.to_string_lossy());
                }
            }
            if out.is_empty() {
                None
            } else {
                Some(ModuleName::from_parts(out))
            }
        }

        for include in includes {
            if let Ok(x) = path.strip_prefix(include)
                && let Some(res) = path_to_module(x)
            {
                return Some(res);
            }
        }
        None
    }

    /// Pop off the last name component from this [`ModuleName`]. If the `ModuleName`
    /// would be empty, return `None` instead.
    pub fn parent(&self) -> Option<Self> {
        Some(Self::from_str(self.as_str().rsplit_once('.')?.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_component() {
        assert_eq!(
            ModuleName::from_str("a.b.c").first_component(),
            Name::new_static("a")
        );
        assert_eq!(
            ModuleName::from_str("a").first_component(),
            Name::new_static("a")
        );
    }

    #[test]
    fn test_relative() {
        let base = ModuleName::from_str("a.b.c");
        assert_eq!(
            base.new_maybe_relative(false, 0, Some(&Name::new_static("d")))
                .unwrap(),
            ModuleName::from_str("d")
        );
        assert_eq!(
            base.new_maybe_relative(false, 1, Some(&Name::new_static("d")))
                .unwrap(),
            ModuleName::from_str("a.b.d")
        );
        assert_eq!(
            base.new_maybe_relative(false, 2, Some(&Name::new_static("d")))
                .unwrap(),
            ModuleName::from_str("a.d")
        );
        assert_eq!(
            base.new_maybe_relative(false, 3, Some(&Name::new_static("d")))
                .unwrap(),
            ModuleName::from_str("d")
        );
        // TODO: This is wrong. The relative level 4 should be invalid
        assert_eq!(
            base.new_maybe_relative(false, 4, Some(&Name::new_static("d"))),
            None
        );
        assert_eq!(
            base.new_maybe_relative(false, 1, None).unwrap(),
            ModuleName::from_str("a.b")
        );
        assert_eq!(
            base.new_maybe_relative(false, 2, None).unwrap(),
            ModuleName::from_str("a")
        );
        assert_eq!(
            ModuleName::from_str("sys")
                .new_maybe_relative(true, 1, None)
                .unwrap(),
            ModuleName::from_str("sys")
        );
    }

    #[test]
    fn test_from_relative_path() {
        fn assert_module_name(path: &str, expected: &str) {
            assert_eq!(
                ModuleName::from_relative_path(Path::new(path)).unwrap(),
                ModuleName::from_str(expected)
            );
        }
        assert_module_name("foo.py", "foo");
        assert_module_name("foo.pyi", "foo");
        assert_module_name("foo.ipynb", "foo");
        assert_module_name("foo/bar.py", "foo.bar");
        assert_module_name("foo/bar.pyi", "foo.bar");
        assert_module_name("foo/bar.ipynb", "foo.bar");
        assert_module_name("foo/bar/__init__.py", "foo.bar");
        assert_module_name("foo/bar/__init__.pyi", "foo.bar");

        fn assert_conversion_error(path: &str) {
            assert!(ModuleName::from_relative_path(Path::new(path)).is_err());
        }
        assert_conversion_error("foo/bar.derp");
        assert_conversion_error("foo/bar/baz");
        assert_conversion_error("foo/bar/__init__.derp");
    }

    #[test]
    fn test_relative_module_name_between() {
        fn assert_module_name(from: &str, to: &str, expected: &str) {
            let from = Path::new(from);
            let to = Path::new(to);
            let actual = ModuleName::relative_module_name_between(from, to);
            assert_eq!(Some(ModuleName::from_str(expected)), actual);
        }
        assert_module_name("foo/bar.py", "foo/baz.py", ".baz");
        assert_module_name("bar.py", "foo/baz.py", ".foo.baz");
        assert_module_name("foo/bar.py", "baz.py", "..baz");
        assert_module_name("foo/bar/boz.py", "baz.py", "...baz");
    }

    #[test]
    fn test_module_from_path() {
        let includes = [PathBuf::from("/foo/bar")];
        assert_eq!(
            ModuleName::from_path(Path::new("/foo/bar/baz.py"), includes.iter()),
            Some(ModuleName::from_str("baz"))
        );
        assert_eq!(
            ModuleName::from_path(Path::new("/foo/bar/baz/qux.pyi"), includes.iter()),
            Some(ModuleName::from_str("baz.qux"))
        );
        assert_eq!(
            ModuleName::from_path(Path::new("/foo/bar/baz/test/magic.py"), includes.iter()),
            Some(ModuleName::from_str("baz.test.magic"))
        );
        assert_eq!(
            ModuleName::from_path(Path::new("/foo/bar/baz/__init__.pyi"), includes.iter()),
            Some(ModuleName::from_str("baz"))
        );
        assert_eq!(
            ModuleName::from_path(Path::new("/test.py"), includes.iter()),
            None
        );
        assert_eq!(
            ModuleName::from_path(Path::new("/not_foo/test.py"), includes.iter()),
            None
        );
    }
}
