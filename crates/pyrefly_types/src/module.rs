/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Display;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_derive::TypeEq;
use pyrefly_python::module_name::ModuleName;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::name::Name;
use starlark_map::ordered_set::OrderedSet;

use crate::types::Type;

/// In Python if you do `import foo.bar` and `import foo.baz` then what you are really
/// doing is importing a single symbol `foo` that contains the two modules accessible from it.
///
/// To represent that, we have a set of modules and a `path` of how far down we are.
/// Any module that does not start with a prefix of the `path` is no longer accessible,
/// but we keep them around (under an `Arc`) since it's more efficient not to recreate
/// the `SmallMap` on each access.
#[derive(Debug, Clone, TypeEq, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModuleType {
    parts: Box<[Name]>,
    /// Use an OrderedMap so we have a table Hash/Ord instance.
    modules: Arc<OrderedSet<ModuleName>>,
}

impl Visit<Type> for ModuleType {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

impl VisitMut<Type> for ModuleType {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}

impl Display for ModuleType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.parts.join("."))
    }
}

impl ModuleType {
    /// Created from an import, e.g. `import foo.bar.baz`
    pub fn new(name: Name, modules: OrderedSet<ModuleName>) -> ModuleType {
        assert!(
            modules.iter().all(|x| x.first_component() == name),
            "{name} {modules:?}"
        );
        Self {
            parts: Box::new([name]),
            modules: Arc::new(modules),
        }
    }

    /// Created from an alias, e.g. `import foo.bar.baz as bar`
    pub fn new_as(name: ModuleName) -> ModuleType {
        Self {
            parts: name.as_str().split('.').map(Name::new).collect(),
            modules: Arc::new(OrderedSet::from_iter([name])),
        }
    }

    pub fn to_type(self) -> Type {
        Type::Module(self)
    }

    pub fn push_part(&self, component: Name) -> Self {
        let mut path = Vec::with_capacity(self.parts.len() + 1);
        path.extend(self.parts.iter().cloned());
        path.push(component);
        ModuleType {
            parts: path.into_boxed_slice(),
            modules: self.modules.dupe(),
        }
    }

    pub fn parts(&self) -> &[Name] {
        &self.parts
    }

    pub fn add_module(&self, m: ModuleName) -> Self {
        let mut modules = (*self.modules).clone();
        modules.insert(m);
        Self {
            parts: self.parts.clone(),
            modules: Arc::new(modules),
        }
    }

    pub fn merge(&mut self, m: &ModuleType) {
        assert_eq!(self.parts, m.parts);
        let mut modules = (*self.modules).clone();
        modules.extend(m.modules.iter().copied());
        self.modules = Arc::new(modules);
    }

    pub fn is_submodules_imported_directly(&self) -> bool {
        let prefix = self.parts();
        self.modules
            .iter()
            .any(|name| name.components().starts_with(prefix))
    }
}
