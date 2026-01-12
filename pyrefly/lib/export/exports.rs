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

use dupe::Dupe;
use pyrefly_graph::calculation::Calculation;
use pyrefly_python::docstring::Docstring;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_types::callable::Deprecation;
use ruff_python_ast::Stmt;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::export::definitions::DefinitionStyle;
use crate::export::definitions::Definitions;
use crate::export::definitions::DunderAllEntry;
use crate::export::definitions::DunderAllKind;
use crate::export::special::SpecialExport;
use crate::module::module_info::ModuleInfo;
use crate::state::loader::FindingOrError;

/// Find the exports of a given module.
pub trait LookupExport {
    /// Get the exports of a given module, or an error if the module is not available.
    fn get(&self, module: ModuleName) -> FindingOrError<Exports>;
}

#[derive(Debug, Clone)]
pub struct Export {
    pub location: TextRange,
    pub symbol_kind: Option<SymbolKind>,
    pub docstring_range: Option<TextRange>,
    pub deprecation: Option<Deprecation>,
    pub special_export: Option<SpecialExport>,
}

/// Where is this export defined?
#[derive(Debug, Clone)]
pub enum ExportLocation {
    // This export is defined in this module.
    ThisModule(Export),
    // Export from another module ModuleName. If it's aliased, the old name (before the alias) is provided.
    OtherModule(ModuleName, Option<Name>),
}

#[derive(Debug, Default, Clone, Dupe)]
pub struct Exports(Arc<ExportsInner>);

#[derive(Debug, Default)]
struct ExportsInner {
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
        for x in self.0.definitions.dunder_all.entries.iter() {
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

        Self(Arc::new(ExportsInner {
            definitions,
            wildcard: Calculation::new(),
            exports: Calculation::new(),
            docstring_range: Docstring::range_from_stmts(x),
        }))
    }

    /// What symbols will I get if I do `from <this_module> import *`?
    pub fn wildcard(&self, lookup: &dyn LookupExport) -> Arc<SmallSet<Name>> {
        let f = || {
            let mut result = SmallSet::new();
            for x in &self.0.definitions.dunder_all.entries {
                match x {
                    DunderAllEntry::Name(_, x) => {
                        result.insert(x.clone());
                    }
                    DunderAllEntry::Module(_, x) => {
                        // They did `__all__.extend(foo.__all__)`, but didn't import `foo`.
                        if let Some(import) = lookup.get(*x).finding() {
                            let wildcard = import.wildcard(lookup);
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
        self.0.wildcard.calculate(f).unwrap_or_default()
    }

    /// Get the docstring for this module.
    pub fn docstring_range(&self) -> Option<TextRange> {
        self.0.docstring_range
    }

    pub fn is_submodule_imported_implicitly(&self, name: &Name) -> bool {
        self.0
            .definitions
            .implicitly_imported_submodules
            .contains(name)
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
        if self.0.definitions.dunder_all.kind == DunderAllKind::Inferred {
            return Vec::new();
        }
        let mut invalid = Vec::new();
        for entry in &self.0.definitions.dunder_all.entries {
            if let DunderAllEntry::Name(range, name) = entry {
                // Check if name exists in definitions
                if self.0.definitions.definitions.contains_key(name) {
                    continue;
                }
                // Check if name is available through a wildcard import
                let mut found_in_import = false;
                for (module, _) in self.0.definitions.import_all.iter() {
                    if let Some(exports) = lookup.get(*module).finding()
                        && exports.wildcard(lookup).contains(name)
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
                    if lookup.get(submodule).finding().is_some() {
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
            for (name, definition) in self.0.definitions.definitions.iter_hashed() {
                let deprecation = self.0.definitions.deprecated.get_hashed(name).cloned();
                let special_export = self.0.definitions.special_exports.get_hashed(name).copied();
                let export = match &definition.style {
                    DefinitionStyle::Annotated(symbol_kind, ..)
                    | DefinitionStyle::Unannotated(symbol_kind) => {
                        ExportLocation::ThisModule(Export {
                            location: definition.range,
                            symbol_kind: Some(*symbol_kind),
                            docstring_range: definition.docstring_range,
                            deprecation,
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
                        special_export,
                    }),
                    DefinitionStyle::ImplicitGlobal => ExportLocation::ThisModule(Export {
                        location: definition.range,
                        symbol_kind: Some(SymbolKind::Constant),
                        docstring_range: None,
                        deprecation,
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
            for m in self.0.definitions.import_all.keys() {
                if let Some(exports) = lookup.get(*m).finding() {
                    for name in exports.wildcard(lookup).iter_hashed() {
                        result.insert_hashed(name.cloned(), ExportLocation::OtherModule(*m, None));
                    }
                }
            }
            Arc::new(result)
        };
        self.0.exports.calculate(f).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use anyhow::anyhow;
    use pyrefly_python::ast::Ast;
    use pyrefly_python::module_path::ModulePath;
    use pyrefly_python::module_path::ModuleStyle;
    use ruff_python_ast::PySourceType;
    use starlark_map::small_map::SmallMap;
    use starlark_map::smallmap;

    use super::*;
    use crate::state::loader::FindError;

    impl LookupExport for SmallMap<ModuleName, Exports> {
        fn get(&self, module: ModuleName) -> FindingOrError<Exports> {
            match self.get(&module) {
                Some(x) => FindingOrError::new_finding(x.dupe()),
                None => FindingOrError::Error(FindError::not_found(anyhow!("Error"), module)),
            }
        }
    }

    fn mk_exports(contents: &str, style: ModuleStyle) -> Exports {
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
        Exports::new(&ast.body, &module_info, &SysInfo::default())
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
