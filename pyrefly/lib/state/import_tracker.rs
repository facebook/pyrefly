/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Resolve structured annotation references against imports in a module.

use dupe::Dupe;
use pyrefly_python::module_name::ModuleName;
use pyrefly_types::type_output::AnnotationPart;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::name::Name;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

/// Tracks names made available by top-level imports.
#[derive(Default)]
pub struct ImportTracker {
    canonical_modules: SmallSet<ModuleName>,
    alias_modules: Vec<(ModuleName, String)>,
    imported_names: SmallMap<ModuleName, SmallMap<String, String>>,
}

impl ImportTracker {
    /// Build an import tracker from the top-level imports in a module.
    pub fn from_ast(ast: &ModModule, current_module: ModuleName, is_init: bool) -> Self {
        let mut tracker = Self::default();
        for stmt in &ast.body {
            if let Stmt::Import(stmt_import) = stmt {
                for alias in &stmt_import.names {
                    let module = ModuleName::from_str(alias.name.as_str());
                    if let Some(asname) = &alias.asname {
                        tracker.alias_modules.push((module, asname.id.to_string()));
                    } else {
                        tracker.canonical_modules.insert(module);
                    }
                }
            } else if let Stmt::ImportFrom(stmt_import_from) = stmt {
                let module = if stmt_import_from.level == 0 {
                    ModuleName::from_str(
                        stmt_import_from
                            .module
                            .as_ref()
                            .expect("absolute import-from must name a module")
                            .as_str(),
                    )
                } else {
                    let suffix = stmt_import_from
                        .module
                        .as_ref()
                        .map(|module| Name::new(module.as_str()));
                    let Some(module) = current_module.new_maybe_relative(
                        is_init,
                        stmt_import_from.level,
                        suffix.as_ref(),
                    ) else {
                        continue;
                    };
                    module
                };
                let names = tracker.imported_names.entry(module).or_default();
                for alias in &stmt_import_from.names {
                    let name = alias.name.as_str();
                    if name != "*" {
                        names.insert(
                            name.to_owned(),
                            alias
                                .asname
                                .as_ref()
                                .map(|id| id.id.to_string())
                                .unwrap_or_else(|| name.to_owned()),
                        );
                    }
                }
            }
        }
        tracker
    }

    /// Resolve annotation references to names available in the current module.
    pub fn resolve_annotation(
        &self,
        parts: &[AnnotationPart],
        current_module: ModuleName,
    ) -> (String, SmallSet<ModuleName>) {
        let mut text = String::new();
        let mut missing = SmallSet::new();
        for part in parts {
            let (module, name) = match part {
                AnnotationPart::Text(part) => {
                    text.push_str(part);
                    continue;
                }
                AnnotationPart::Reference { module, name } => (module, name),
            };
            if module.as_str().is_empty()
                || *module == current_module
                || *module == ModuleName::builtins()
                || *module == ModuleName::extra_builtins()
            {
                text.push_str(name);
            } else {
                let mut name_parts = name.splitn(2, '.');
                let head = name_parts
                    .next()
                    .expect("splitn always returns at least one part");
                let suffix = name_parts.next();
                if let Some(imported) = self
                    .imported_names
                    .get(module)
                    .and_then(|names| names.get(head))
                {
                    text.push_str(imported);
                    if let Some(suffix) = suffix {
                        text.push('.');
                        text.push_str(suffix);
                    }
                } else if let Some(alias) = self.alias_for(module.dupe()) {
                    text.push_str(alias);
                    text.push('.');
                    text.push_str(name);
                } else {
                    text.push_str(module.as_str());
                    text.push('.');
                    text.push_str(name);
                    if !self.has_canonical(module.dupe()) {
                        missing.insert(module.dupe());
                    }
                }
            }
        }
        (text, missing)
    }

    fn alias_for(&self, module: ModuleName) -> Option<&str> {
        self.alias_modules
            .iter()
            .find_map(|(imported, alias)| (*imported == module).then_some(alias.as_str()))
    }

    fn has_canonical(&self, module: ModuleName) -> bool {
        self.canonical_modules.contains(&module)
    }
}

#[cfg(test)]
mod tests {
    use pyrefly_python::ast::Ast;
    use ruff_python_ast::PySourceType;

    use super::*;

    fn reference(module: &str, name: &str) -> AnnotationPart {
        AnnotationPart::Reference {
            module: ModuleName::from_str(module),
            name: name.to_owned(),
        }
    }

    #[test]
    fn resolves_module_and_imported_name_aliases_without_touching_text() {
        let ast = Ast::parse(
            "import foo as f\nfrom bar import Bar as B\nfrom typing import Literal\n",
            PySourceType::Python,
        )
        .0;
        let tracker = ImportTracker::from_ast(&ast, ModuleName::from_str("current"), false);
        let parts = vec![
            reference("typing", "Literal"),
            AnnotationPart::Text("['é'] | ".to_owned()),
            reference("foo", "C"),
            AnnotationPart::Text(" | ".to_owned()),
            reference("bar", "Bar.Inner"),
        ];

        let (text, missing) = tracker.resolve_annotation(&parts, ModuleName::from_str("current"));
        assert_eq!(text, "Literal['é'] | f.C | B.Inner");
        assert!(missing.is_empty());
    }

    #[test]
    fn tracks_missing_modules_per_reference() {
        let ast = Ast::parse("from foo import Other\n", PySourceType::Python).0;
        let tracker = ImportTracker::from_ast(&ast, ModuleName::from_str("current"), false);
        let parts = vec![
            reference("foo", "Foo"),
            AnnotationPart::Text(" | ".to_owned()),
            reference("bar", "Bar"),
            AnnotationPart::Text(" | ".to_owned()),
            reference("foo", "Foo"),
        ];

        let (text, missing) = tracker.resolve_annotation(&parts, ModuleName::from_str("current"));
        assert_eq!(text, "foo.Foo | bar.Bar | foo.Foo");
        assert_eq!(missing.len(), 2);
        assert!(missing.contains(&ModuleName::from_str("foo")));
        assert!(missing.contains(&ModuleName::from_str("bar")));
    }

    #[test]
    fn parent_module_imports_do_not_resolve_descendants() {
        let ast = Ast::parse("import foo as f\nimport bar\n", PySourceType::Python).0;
        let tracker = ImportTracker::from_ast(&ast, ModuleName::from_str("current"), false);
        let parts = vec![
            reference("foo.child", "C"),
            AnnotationPart::Text(" | ".to_owned()),
            reference("bar.child", "D"),
        ];

        let (text, missing) = tracker.resolve_annotation(&parts, ModuleName::from_str("current"));
        assert_eq!(text, "foo.child.C | bar.child.D");
        assert_eq!(missing.len(), 2);
        assert!(missing.contains(&ModuleName::from_str("foo.child")));
        assert!(missing.contains(&ModuleName::from_str("bar.child")));
    }

    #[test]
    fn resolves_relative_imports_against_current_module() {
        let ast = Ast::parse(
            "from .types import Local as L\n\
             from ..shared import Shared\n\
             from absolute import Absolute\n",
            PySourceType::Python,
        )
        .0;
        let tracker =
            ImportTracker::from_ast(&ast, ModuleName::from_str("package.sub.current"), false);
        let parts = vec![
            reference("package.sub.types", "Local"),
            AnnotationPart::Text(" | ".to_owned()),
            reference("package.shared", "Shared"),
            AnnotationPart::Text(" | ".to_owned()),
            reference("absolute", "Absolute"),
        ];

        let (text, missing) =
            tracker.resolve_annotation(&parts, ModuleName::from_str("package.sub.current"));
        assert_eq!(text, "L | Shared | Absolute");
        assert!(missing.is_empty());
    }

    #[test]
    fn relative_import_does_not_resolve_same_named_absolute_module() {
        let ast = Ast::parse("from .foo import Bar\n", PySourceType::Python).0;
        let tracker = ImportTracker::from_ast(&ast, ModuleName::from_str("package.current"), false);

        let (text, missing) = tracker.resolve_annotation(
            &[reference("foo", "Bar")],
            ModuleName::from_str("package.current"),
        );
        assert_eq!(text, "foo.Bar");
        assert!(missing.contains(&ModuleName::from_str("foo")));
    }
}
