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
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_python::sys_info::SysInfo;
use ruff_python_ast::Stmt;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::export::definitions::DefinitionStyle;
use crate::export::definitions::Definitions;
use crate::export::definitions::DocString;
use crate::export::definitions::DunderAllEntry;
use crate::graph::calculation::Calculation;
use crate::module::module_info::ModuleInfo;
use crate::state::loader::FindError;

/// Find the exports of a given module.
pub trait LookupExport {
    /// Get the exports of a given module, or an error if the module is not available.
    fn get(&self, module: ModuleName) -> Result<Exports, FindError>;
}

#[derive(Debug, Clone)]
pub struct Export {
    pub location: TextRange,
    pub symbol_kind: Option<SymbolKind>,
    pub docstring: Option<DocString>,
}

/// Where is this export defined?
#[derive(Debug, Clone)]
pub enum ExportLocation {
    // This export is defined in this module.
    ThisModule(Export),
    // Exported from another module. Module optional in case
    // we could not find it.
    OtherModule(ModuleName),
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
    /// If this module has a docstring, it's stored here. Docstrings for exports themselves are stored in exports.
    docstring: Option<DocString>,
}

impl Display for Exports {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for x in self.0.definitions.dunder_all.iter() {
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
        definitions.inject_globals();
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
            docstring: DocString::from_stmts(x),
        }))
    }

    /// What symbols will I get if I do `from <this_module> import *`?
    pub fn wildcard(&self, lookup: &dyn LookupExport) -> Arc<SmallSet<Name>> {
        let f = || {
            let mut result = SmallSet::new();
            for x in &self.0.definitions.dunder_all {
                match x {
                    DunderAllEntry::Name(_, x) => {
                        result.insert(x.clone());
                    }
                    DunderAllEntry::Module(_, x) => {
                        // They did `__all__.extend(foo.__all__)`, but didn't import `foo`.
                        if let Ok(import) = lookup.get(*x) {
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
    pub fn docstring(&self) -> Option<&DocString> {
        self.0.docstring.as_ref()
    }

    pub fn is_submodule_imported_implicitly(&self, name: &Name) -> bool {
        self.0
            .definitions
            .implicitly_imported_submodules
            .contains(name)
    }

    pub fn exports(&self, lookup: &dyn LookupExport) -> Arc<SmallMap<Name, ExportLocation>> {
        let f = || {
            let mut result: SmallMap<Name, ExportLocation> = SmallMap::new();
            for (name, definition) in self.0.definitions.definitions.iter_hashed() {
                let export = match definition.style {
                    DefinitionStyle::Local(symbol_kind) => ExportLocation::ThisModule(Export {
                        location: definition.range,
                        symbol_kind: Some(symbol_kind),
                        docstring: definition.docstring.clone(),
                    }),
                    // If the import is invalid, the final location is this module.
                    DefinitionStyle::ImportInvalidRelative => ExportLocation::ThisModule(Export {
                        location: definition.range,
                        symbol_kind: None,
                        docstring: definition.docstring.clone(),
                    }),
                    DefinitionStyle::Global => ExportLocation::ThisModule(Export {
                        location: definition.range,
                        symbol_kind: Some(SymbolKind::Constant),
                        docstring: None,
                    }),
                    DefinitionStyle::ImportAs(from)
                    | DefinitionStyle::ImportAsEq(from)
                    | DefinitionStyle::Import(from)
                    | DefinitionStyle::ImportModule(from) => ExportLocation::OtherModule(from),
                };
                result.insert_hashed(name.cloned(), export);
            }
            for m in self.0.definitions.import_all.keys() {
                if let Ok(exports) = lookup.get(*m) {
                    for name in exports.wildcard(lookup).iter_hashed() {
                        result.insert_hashed(name.cloned(), ExportLocation::OtherModule(*m));
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
    use starlark_map::small_map::SmallMap;
    use starlark_map::smallmap;

    use super::*;

    impl LookupExport for SmallMap<ModuleName, Exports> {
        fn get(&self, module: ModuleName) -> Result<Exports, FindError> {
            match self.get(&module) {
                Some(x) => Ok(x.dupe()),
                None => Err(FindError::not_found(anyhow!("Error"), module)),
            }
        }
    }

    fn mk_exports(contents: &str, style: ModuleStyle) -> Exports {
        let ast = Ast::parse(contents).0;
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
