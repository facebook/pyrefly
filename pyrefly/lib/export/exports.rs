/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::sync::Arc;

use pyrefly_graph::calculation::Calculation;
use pyrefly_python::docstring::Docstring;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_types::callable::Deprecation;
use ruff_python_ast::Stmt;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::export::definitions::DefinitionStyle;
use crate::export::definitions::Definitions;
use crate::export::definitions::DunderAllEntry;
use crate::export::definitions::DunderAllKind;
use crate::export::special::SpecialExport;
use crate::module::module_info::ModuleInfo;
use crate::state::loader::FindingOrError;

/// Find the exports of a given module. Beware: these APIs record dependencies between modules during lookups. Using the
/// wrong API can lead to invalidation bugs.
pub trait LookupExport {
    /// Check if a specific export exists in a module. Records a dependency on `name` from `module` regardless of if it exists.
    fn export_exists(&self, module: ModuleName, name: &Name) -> bool;

    /// Check if a module exists and do nothing with it. Note: if we rely on the exports of `module`, we need to call
    /// `module_exists_and_record_export_dependency` instead.
    fn module_exists(&self, module: ModuleName) -> FindingOrError<()>;

    /// Get the wildcard exports for a module. Records a dependency on `module` regardless of if it exists.
    fn get_wildcard(&self, module: ModuleName) -> Option<Arc<SmallSet<Name>>>;

    /// Get all export names for a module. Records no dependencies.
    fn get_every_export_untracked(&self, module: ModuleName) -> Option<SmallSet<Name>>;

    /// Check if a submodule is imported implicitly. Records a dependency on `name` from `module` regardless of if it exists.
    fn is_submodule_imported_implicitly(&self, module: ModuleName, name: &Name) -> bool;

    /// Get deprecation info for an export. Records a dependency on `name` from `module` regardless of if it exists.
    fn get_deprecated(&self, module: ModuleName, name: &Name) -> Option<Deprecation>;

    /// Check if an export is a re-export from another module. Records a dependency on `name` from `module` regardless of if it exists.
    fn is_reexport(&self, module: ModuleName, name: &Name) -> bool;

    /// Check if an export is a special export. Records a dependency on `name` from `module` regardless of if it exists.
    fn is_special_export(&self, module: ModuleName, name: &Name) -> Option<SpecialExport>;

    /// Get the docstring range for an export. Records a dependency on `name` from `module` regardless of if it exists.
    fn docstring_range(&self, module: ModuleName, name: &Name) -> Option<TextRange>;

    /// Check if an export is marked as `Final`. Records a dependency on `name` from `module` regardless of if it exists.
    fn is_final(&self, module: ModuleName, name: &Name) -> bool;
}

#[derive(Debug, Clone)]
pub struct Export {
    pub location: TextRange,
    pub symbol_kind: Option<SymbolKind>,
    pub docstring_range: Option<TextRange>,
    pub deprecation: Option<Deprecation>,
    /// Whether the exported symbol is marked with the `Final` special form.
    pub is_final: bool,
    pub special_export: Option<SpecialExport>,
}

/// Where is this export defined?
#[derive(Debug, Clone)]
pub enum ExportLocation {
    /// This export is defined in this module.
    ThisModule(Export),
    /// Export from another module ModuleName. If it's aliased, the old name (before the alias) is provided.
    OtherModule(ModuleName, Option<Name>),
}

#[derive(Debug, Default)]
pub struct Exports {
    /// The underlying definitions.
    /// Note that these aren't actually required, once we have calculated the other fields,
    /// but they take up very little space, so not worth the hassle to detect when
    /// calculation completes.
    definitions: Definitions,
    /// Names that are available via `from <this_module> import *`
    wildcard: Calculation<Arc<SmallSet<Name>>>,
    /// Names that are available via `from <this_module> import <name>` along with their locations
    exports: Calculation<Arc<SmallMap<Name, ExportLocation>>>,
    /// If this module has a docstring, the range is stored here. Docstrings for exports themselves are stored in exports.
    /// While putting the module docstring range on exports is a bit weird (it doesn't actually have much to do with exports),
    /// we can't put it on the Module as that doesn't have the AST, and we can't get it from the AST as we often throw that away,
    /// so here makes sense.
    docstring_range: Option<TextRange>,
}

impl Display for Exports {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for x in self.definitions.dunder_all.entries.iter() {
            match x {
                DunderAllEntry::Name(_, x) => writeln!(f, "export {x}")?,
                DunderAllEntry::Module(_, x) => writeln!(f, "from {x} import *")?,
                DunderAllEntry::Remove(_, x) => writeln!(f, "unexport {x}")?,
            }
        }
        Ok(())
    }
}

impl Exports {
    pub fn new(x: &[Stmt], module_info: &ModuleInfo, sys_info: &SysInfo) -> Self {
        let mut definitions = Definitions::new(
            x,
            module_info.name(),
            module_info.path().is_init(),
            sys_info,
        );
        definitions.inject_implicit_globals();
        definitions.ensure_dunder_all(module_info.path().style());
        if module_info.name() == ModuleName::builtins() {
            // The `builtins` module is a bit weird. It has no `__all__` in TypeShed,
            // if you do `from builtins import *` it behaves weirdly, and things that
            // would otherwise be hidden show up.
            //
            // Eventually it would be good to make TypeShed the source of truth, but
            // until then, manually extend the synthetic `__all__` to match runtime.
            definitions.extend_dunder_all(&[
                Name::new_static("__build_class__"),
                Name::new_static("__import__"),
            ]);
        }

        Self {
            definitions,
            wildcard: Calculation::new(),
            exports: Calculation::new(),
            docstring_range: Docstring::range_from_stmts(x),
        }
    }

    /// What symbols will I get if I do `from <this_module> import *`?
    pub fn wildcard(&self, lookup: &dyn LookupExport) -> Arc<SmallSet<Name>> {
        let f = || {
            let mut result = SmallSet::new();
            for x in &self.definitions.dunder_all.entries {
                match x {
                    DunderAllEntry::Name(_, x) => {
                        result.insert(x.clone());
                    }
                    DunderAllEntry::Module(_, x) => {
                        // They did `__all__.extend(foo.__all__)`, but didn't import `foo`.
                        if let Some(wildcard) = lookup.get_wildcard(*x) {
                            for y in wildcard.iter_hashed() {
                                result.insert_hashed(y.cloned());
                            }
                        }
                    }
                    DunderAllEntry::Remove(_, x) => {
                        // This is O(n), but we'd appreciate the determism, and remove is rare in `__all__`.
                        result.shift_remove(x);
                    }
                }
            }
            Arc::new(result)
        };
        self.wildcard.calculate(f).unwrap_or_default()
    }

    /// Get the names that were added or removed between self and other.
    /// Returns the symmetric difference: names that exist in one but not the other.
    pub fn changed_names(&self, other: &Self) -> SmallSet<Name> {
        let self_set = self.wildcard.get();
        let other_set = other.wildcard.get();

        let (self_set, other_set) = match (self_set, other_set) {
            (Some(s), Some(o)) => (s, o),
            (None, None) => return SmallSet::new(),
            (Some(s), None) => return s.iter().cloned().collect(),
            (None, Some(o)) => return o.iter().cloned().collect(),
        };

        // Compute symmetric difference: names in self but not other, plus names in other but not self
        let mut result = SmallSet::new();
        for name in self_set.iter() {
            if !other_set.contains(name) {
                result.insert(name.clone());
            }
        }
        for name in other_set.iter() {
            if !self_set.contains(name) {
                result.insert(name.clone());
            }
        }
        result
    }

    /// Get the names where metadata changed between self and other.
    /// Checks: is_import status, implicitly_imported_submodules, deprecated, special_exports.
    /// Only checks names that exist in both versions (existence changes tracked separately).
    /// Ignores TextRange fields (range, docstring_range) per design doc.
    pub fn changed_metadata_names(&self, other: &Self) -> SmallSet<Name> {
        let self_defs = &self.definitions;
        let other_defs = &other.definitions;

        let mut changed = SmallSet::new();

        // Check names that exist in both
        for (name, self_def) in self_defs.definitions.iter() {
            if let Some(other_def) = other_defs.definitions.get(name) {
                // Check is_import status (is_reexport)
                if self_def.style.is_import() != other_def.style.is_import() {
                    changed.insert(name.clone());
                    continue;
                }
                // Check implicitly_imported_submodules
                let self_implicit = self_defs.implicitly_imported_submodules.contains(name);
                let other_implicit = other_defs.implicitly_imported_submodules.contains(name);
                if self_implicit != other_implicit {
                    changed.insert(name.clone());
                    continue;
                }
                // Check deprecated
                if self_defs.deprecated.get(name) != other_defs.deprecated.get(name) {
                    changed.insert(name.clone());
                    continue;
                }
                // Check special_exports
                if self_defs.special_exports.get(name) != other_defs.special_exports.get(name) {
                    changed.insert(name.clone());
                }
            }
        }
        changed
    }

    /// Get the docstring for this module.
    pub fn docstring_range(&self) -> Option<TextRange> {
        self.docstring_range
    }

    /// If `position` is inside a user-specified `__all__` string entry, return its range and name.
    pub fn dunder_all_name_at(&self, position: TextSize) -> Option<(TextRange, Name)> {
        if self.definitions.dunder_all.kind != DunderAllKind::Specified {
            return None;
        }
        self.definitions
            .dunder_all
            .entries
            .iter()
            .find_map(|entry| match entry {
                DunderAllEntry::Name(range, name) if range.contains_inclusive(position) => {
                    Some((*range, name.clone()))
                }
                _ => None,
            })
    }

    pub fn is_submodule_imported_implicitly(&self, name: &Name) -> bool {
        self.definitions
            .implicitly_imported_submodules
            .contains(name)
    }

    /// Return an iterator with entries in `__all__` that are user-defined or None if `__all__` was not present.
    pub fn get_explicit_dunder_all_names_iter(&self) -> Option<impl Iterator<Item = &Name>> {
        match self.definitions.dunder_all.kind {
            DunderAllKind::Specified => Some(
                self.definitions
                    .dunder_all
                    .entries
                    .iter()
                    .filter_map(|entry| match entry {
                        DunderAllEntry::Name(_, name) => Some(name),
                        _ => None,
                    }),
            ),
            _ => None,
        }
    }

    /// Returns entries in `__all__` that don't exist in the module's definitions.
    /// Only validates explicitly user-defined `__all__` entries, not synthesized ones.
    /// Returns a vector of (range, name) tuples for invalid entries.
    pub fn invalid_dunder_all_entries(
        &self,
        lookup: &dyn LookupExport,
        module_info: &ModuleInfo,
    ) -> Vec<(TextRange, Name)> {
        // Only validate if __all__ was explicitly defined by the user
        if self.definitions.dunder_all.kind == DunderAllKind::Inferred {
            return Vec::new();
        }
        let mut invalid = Vec::new();
        for entry in &self.definitions.dunder_all.entries {
            if let DunderAllEntry::Name(range, name) = entry {
                // Check if name exists in definitions
                if self.definitions.definitions.contains_key(name) {
                    continue;
                }
                // Check if name is available through a wildcard import
                let mut found_in_import = false;
                for (module, _) in self.definitions.import_all.iter() {
                    if let Some(wildcard) = lookup.get_wildcard(*module)
                        && wildcard.contains(name)
                    {
                        found_in_import = true;
                        break;
                    }
                }
                if found_in_import {
                    continue;
                }
                // In __init__.py, __all__ can list submodule names
                if module_info.path().is_init() {
                    let submodule = module_info.name().append(name);
                    if lookup.module_exists(submodule).finding().is_some() {
                        continue;
                    }
                }
                invalid.push((*range, name.clone()));
            }
        }
        invalid
    }

    pub fn exports(&self, lookup: &dyn LookupExport) -> Arc<SmallMap<Name, ExportLocation>> {
        let f = || {
            let mut result: SmallMap<Name, ExportLocation> = SmallMap::new();
            for (name, definition) in self.definitions.definitions.iter_hashed() {
                let deprecation = self.definitions.deprecated.get_hashed(name).cloned();
                let special_export = self.definitions.special_exports.get_hashed(name).copied();
                let is_final = self.definitions.final_names.contains_hashed(name);
                let export = match &definition.style {
                    DefinitionStyle::Annotated(symbol_kind, ..)
                    | DefinitionStyle::Unannotated(symbol_kind) => {
                        ExportLocation::ThisModule(Export {
                            location: definition.range,
                            symbol_kind: Some(*symbol_kind),
                            docstring_range: definition.docstring_range,
                            deprecation,
                            is_final,
                            special_export,
                        })
                    }
                    // The final location is this module in several edge cases that can occur analyzing invalid code:
                    // - An invalid import
                    // - A variable defined only by a `del` statement but never initialized
                    // - A mutable capture at the top-level
                    DefinitionStyle::ImportInvalidRelative
                    | DefinitionStyle::Delete
                    | DefinitionStyle::MutableCapture(..) => ExportLocation::ThisModule(Export {
                        location: definition.range,
                        symbol_kind: None,
                        docstring_range: definition.docstring_range,
                        deprecation,
                        is_final,
                        special_export,
                    }),
                    DefinitionStyle::ImplicitGlobal => ExportLocation::ThisModule(Export {
                        location: definition.range,
                        symbol_kind: Some(SymbolKind::Constant),
                        docstring_range: None,
                        deprecation,
                        is_final,
                        special_export,
                    }),
                    DefinitionStyle::ImportAs(from, name) => {
                        ExportLocation::OtherModule(*from, Some(name.clone()))
                    }
                    DefinitionStyle::ImportAsEq(from)
                    | DefinitionStyle::Import(from)
                    | DefinitionStyle::ImportModule(from) => {
                        ExportLocation::OtherModule(*from, None)
                    }
                };
                result.insert_hashed(name.cloned(), export);
            }
            for m in self.definitions.import_all.keys() {
                if let Some(wildcard) = lookup.get_wildcard(*m) {
                    for name in wildcard.iter_hashed() {
                        result.insert_hashed(name.cloned(), ExportLocation::OtherModule(*m, None));
                    }
                }
            }
            Arc::new(result)
        };
        self.exports.calculate(f).unwrap_or_default()
    }

    pub fn is_explicit_reexport(&self, name: &Name) -> bool {
        self.definitions
            .definitions
            .get(name)
            .is_some_and(|definition| matches!(definition.style, DefinitionStyle::ImportAsEq(_)))
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use anyhow::anyhow;
    use dupe::Dupe;
    use pyrefly_python::ast::Ast;
    use pyrefly_python::module_path::ModulePath;
    use pyrefly_python::module_path::ModuleStyle;
    use ruff_python_ast::PySourceType;
    use starlark_map::small_map::SmallMap;
    use starlark_map::smallmap;

    use super::*;
    use crate::state::loader::FindError;

    impl LookupExport for SmallMap<ModuleName, Arc<Exports>> {
        fn export_exists(&self, module: ModuleName, k: &Name) -> bool {
            self.get(&module)
                .map(|x| x.exports(self).contains_key(k))
                .unwrap_or(false)
        }

        fn get_wildcard(&self, module: ModuleName) -> Option<Arc<SmallSet<Name>>> {
            self.get(&module).map(|x| x.wildcard(self))
        }

        fn get_every_export_untracked(&self, module: ModuleName) -> Option<SmallSet<Name>> {
            self.get(&module)
                .map(|x| x.exports(self).keys().cloned().collect::<SmallSet<Name>>())
        }

        fn module_exists(&self, module: ModuleName) -> FindingOrError<()> {
            match self.get(&module) {
                Some(_) => FindingOrError::new_finding(()),
                None => FindingOrError::Error(FindError::not_found(anyhow!("Error"), module)),
            }
        }

        fn get_deprecated(&self, _module: ModuleName, _name: &Name) -> Option<Deprecation> {
            None
        }

        fn is_reexport(&self, _module: ModuleName, _name: &Name) -> bool {
            false
        }

        fn docstring_range(&self, _module: ModuleName, _name: &Name) -> Option<TextRange> {
            None
        }

        fn is_special_export(&self, _module: ModuleName, _name: &Name) -> Option<SpecialExport> {
            None
        }

        fn is_submodule_imported_implicitly(&self, _module: ModuleName, _name: &Name) -> bool {
            false
        }

        fn is_final(&self, _module: ModuleName, _name: &Name) -> bool {
            false
        }
    }

    fn mk_exports(contents: &str, style: ModuleStyle) -> Arc<Exports> {
        let ast = Ast::parse(contents, PySourceType::Python).0;
        let path = ModulePath::filesystem(PathBuf::from(if style == ModuleStyle::Interface {
            "foo.pyi"
        } else {
            "foo.py"
        }));
        let module_info = ModuleInfo::new(
            ModuleName::from_str("foo"),
            path,
            Arc::new(contents.to_owned()),
        );
        Arc::new(Exports::new(&ast.body, &module_info, &SysInfo::default()))
    }

    fn eq_wildcards(exports: &Exports, lookup: &dyn LookupExport, all: &[&str]) {
        assert_eq!(
            exports
                .wildcard(lookup)
                .iter()
                .map(|x| x.as_str())
                .collect::<Vec<_>>(),
            all
        );
    }

    #[must_use]
    fn contains(exports: &Exports, lookup: &dyn LookupExport, name: &str) -> bool {
        exports.exports(lookup).contains_key(&Name::new(name))
    }

    #[test]
    fn test_exports() {
        let simple = mk_exports("simple_val = 1\n_simple_val = 2", ModuleStyle::Executable);
        eq_wildcards(&simple, &SmallMap::new(), &["simple_val"]);

        let imports = smallmap! {ModuleName::from_str("simple") => simple};
        let contents = r#"
from simple import *
from bar import X, Y as Z, Q as Q
import baz
import test as test

x = 1
_x = 2
"#;

        let executable = mk_exports(contents, ModuleStyle::Executable);
        let interface = mk_exports(contents, ModuleStyle::Interface);

        eq_wildcards(
            &executable,
            &imports,
            &["simple_val", "X", "Z", "Q", "baz", "test", "x"],
        );
        eq_wildcards(&interface, &imports, &["Q", "test", "x"]);

        for x in [&executable, &interface] {
            assert!(contains(x, &imports, "Z"));
            assert!(contains(x, &imports, "baz"));
            assert!(!contains(x, &imports, "magic"));
        }
        assert!(contains(&executable, &imports, "simple_val"));
    }

    #[test]
    fn test_reexport() {
        // `a` is not in the `import *` of `b`, but it can be used as `b.a`
        let a = mk_exports("a = 1", ModuleStyle::Interface);
        let b = mk_exports("from a import *", ModuleStyle::Interface);
        let imports = smallmap! {ModuleName::from_str("a") => a};
        assert!(contains(&b, &imports, "a"));
        eq_wildcards(&b, &imports, &[]);
    }

    #[test]
    fn test_cyclic() {
        let a = mk_exports("from b import *", ModuleStyle::Interface);
        let b = mk_exports("from a import *\nx = 1", ModuleStyle::Interface);
        let imports = smallmap! {
                ModuleName::from_str("a") => a.dupe(),
                ModuleName::from_str("b") => b.dupe(),
        };
        eq_wildcards(&a, &imports, &[]);
        eq_wildcards(&b, &imports, &["x"]);
        assert!(contains(&b, &imports, "x"));
        assert!(!contains(&b, &imports, "y"));
    }

    #[test]
    fn over_export() {
        let a = mk_exports("from b import *", ModuleStyle::Executable);
        let b = mk_exports("from a import magic\n__all__ = []", ModuleStyle::Executable);
        let imports = smallmap! {
                ModuleName::from_str("a") => a.dupe(),
                ModuleName::from_str("b") => b.dupe(),
        };
        eq_wildcards(&a, &imports, &[]);
        eq_wildcards(&b, &imports, &[]);
        assert!(!contains(&a, &imports, "magic"));
        assert!(contains(&b, &imports, "magic"));
    }
}
