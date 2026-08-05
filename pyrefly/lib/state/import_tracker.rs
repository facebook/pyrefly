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
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::name::Name;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

/// Split a dotted name into the leading identifier and the rest, e.g.
/// `Outer.Inner` into `("Outer", Some("Inner"))`. The leading identifier is the
/// only part an import can bind.
fn split_head(name: &str) -> (&str, Option<&str>) {
    let mut parts = name.splitn(2, '.');
    let head = parts
        .next()
        .expect("splitn always returns at least one part");
    (head, parts.next())
}

/// Tracks imports and selected module-scope bindings used to resolve annotations.
#[derive(Default)]
pub struct ImportTracker {
    canonical_modules: SmallSet<ModuleName>,
    alias_modules: SmallMap<ModuleName, String>,
    imported_names: SmallMap<ModuleName, SmallMap<String, String>>,
    /// Names bound by top-level imports, definitions, and simple assignments.
    /// A generated `from <module> import <head>` binds `head` for the whole file,
    /// so it must not be generated when `head` is already bound here. Binding
    /// forms other than the ones walked below (conditional imports, `for`, `with`,
    /// walrus) are not tracked, so this check is necessary but not sufficient.
    module_scope_names: SmallSet<String>,
}

impl ImportTracker {
    /// Build an import tracker from the top-level statements in a module.
    pub fn from_ast(ast: &ModModule, current_module: ModuleName, is_init: bool) -> Self {
        let mut tracker = Self::default();
        for stmt in &ast.body {
            match stmt {
                Stmt::Import(stmt_import) => {
                    for alias in &stmt_import.names {
                        let module = ModuleName::from_str(alias.name.as_str());
                        if let Some(asname) = &alias.asname {
                            tracker.alias_modules.insert(module, asname.id.to_string());
                            tracker.module_scope_names.insert(asname.id.to_string());
                        } else {
                            tracker.canonical_modules.insert(module);
                            // `import foo.bar` binds `foo`, not `foo.bar`.
                            tracker
                                .module_scope_names
                                .insert(split_head(module.as_str()).0.to_owned());
                        }
                    }
                }
                Stmt::ImportFrom(stmt_import_from) => {
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
                    for alias in &stmt_import_from.names {
                        let name = alias.name.as_str();
                        if name != "*" {
                            let local = alias
                                .asname
                                .as_ref()
                                .map(|id| id.id.to_string())
                                .unwrap_or_else(|| name.to_owned());
                            tracker.module_scope_names.insert(local.clone());
                            tracker
                                .imported_names
                                .entry(module)
                                .or_default()
                                .insert(name.to_owned(), local);
                        }
                    }
                }
                Stmt::ClassDef(class) => {
                    tracker.module_scope_names.insert(class.name.id.to_string());
                }
                Stmt::FunctionDef(function) => {
                    tracker
                        .module_scope_names
                        .insert(function.name.id.to_string());
                }
                Stmt::Assign(assign) => {
                    for target in &assign.targets {
                        if let Expr::Name(name) = target {
                            tracker.module_scope_names.insert(name.id.to_string());
                        }
                    }
                }
                Stmt::AnnAssign(assign) => {
                    if let Expr::Name(name) = &*assign.target {
                        tracker.module_scope_names.insert(name.id.to_string());
                    }
                }
                _ => {}
            }
        }
        tracker
    }

    /// The name a `from <module> import ...` would have to bind to render this
    /// reference unqualified, or `None` when the reference needs no new import or
    /// when binding that name would collide with a name the file already binds.
    fn direct_import_head<'a>(
        &self,
        module: ModuleName,
        name: &'a str,
        current_module: ModuleName,
    ) -> Option<&'a str> {
        if module.as_str().is_empty()
            || module == current_module
            || module == ModuleName::builtins()
            || module == ModuleName::extra_builtins()
        {
            return None;
        }
        let (head, _) = split_head(name);
        if self.canonical_modules.contains(&module)
            || self.alias_modules.contains_key(&module)
            || self
                .imported_names
                .get(&module)
                .is_some_and(|names| names.contains_key(head))
            || module
                .as_str()
                .rsplit_once('.')
                .is_some_and(|(parent, child)| {
                    self.imported_names
                        .get(&ModuleName::from_str(parent))
                        .is_some_and(|names| names.contains_key(child))
                })
            || self.module_scope_names.contains(head)
        {
            return None;
        }
        Some(head)
    }

    /// Distinct `(module, head)` pairs that could become `from <module> import <head>`.
    /// A head referenced from more than one module is excluded: importing it from
    /// one module would leave the other reference indistinguishable from it. This
    /// is why the whole annotation must be inspected before any reference is
    /// resolved, rather than deciding reference by reference.
    pub fn direct_import_candidates(
        &self,
        parts: &[AnnotationPart],
        current_module: ModuleName,
    ) -> Vec<(ModuleName, String)> {
        let heads = parts
            .iter()
            .filter_map(|part| match part {
                AnnotationPart::Reference { module, name } => self
                    .direct_import_head(*module, name, current_module)
                    .map(|head| (*module, head)),
                AnnotationPart::Text(_) => None,
            })
            .collect::<Vec<_>>();
        let mut candidates: Vec<(ModuleName, String)> = Vec::new();
        for (module, head) in &heads {
            let candidate = (*module, (*head).to_owned());
            if heads
                .iter()
                .all(|(other, other_head)| other_head != head || other == module)
                && !candidates.contains(&candidate)
            {
                candidates.push(candidate);
            }
        }
        candidates
    }

    /// Resolve annotation references to names available in the current module.
    /// A reference whose `(module, head)` is in `direct_imports` renders
    /// unqualified, on the caller's promise to emit `from <module> import <head>`.
    pub fn resolve_annotation(
        &self,
        parts: &[AnnotationPart],
        current_module: ModuleName,
        direct_imports: &[(ModuleName, &str)],
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
                let (head, suffix) = split_head(name);
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
                } else if let Some(imported) =
                    module
                        .as_str()
                        .rsplit_once('.')
                        .and_then(|(parent, child)| {
                            self.imported_names
                                .get(&ModuleName::from_str(parent))
                                .and_then(|names| names.get(child))
                        })
                {
                    text.push_str(imported);
                    text.push('.');
                    text.push_str(name);
                } else if let Some(alias) = self.alias_modules.get(module) {
                    text.push_str(alias);
                    text.push('.');
                    text.push_str(name);
                } else if self.canonical_modules.contains(module) {
                    text.push_str(module.as_str());
                    text.push('.');
                    text.push_str(name);
                } else if direct_imports.iter().any(|(direct_module, direct_head)| {
                    direct_module == module && *direct_head == head
                }) {
                    // The import binds the head, so the whole dotted name reads as written.
                    text.push_str(name);
                } else {
                    text.push_str(module.as_str());
                    text.push('.');
                    text.push_str(name);
                    missing.insert(module.dupe());
                }
            }
        }
        (text, missing)
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

    fn candidates(source: &str, parts: &[AnnotationPart]) -> Vec<(String, String)> {
        let ast = Ast::parse(source, PySourceType::Python).0;
        let current_module = ModuleName::from_str("current");
        ImportTracker::from_ast(&ast, current_module, false)
            .direct_import_candidates(parts, current_module)
            .into_iter()
            .map(|(module, head)| (module.as_str().to_owned(), head))
            .collect()
    }

    #[test]
    fn proposes_one_candidate_per_distinct_reference() {
        let parts = vec![
            reference("foo", "Outer.Inner"),
            AnnotationPart::Text(" | ".to_owned()),
            reference("bar", "Bar"),
            AnnotationPart::Text(" | ".to_owned()),
            reference("foo", "Outer"),
        ];

        assert_eq!(
            candidates("", &parts),
            vec![
                ("foo".to_owned(), "Outer".to_owned()),
                ("bar".to_owned(), "Bar".to_owned())
            ]
        );
    }

    #[test]
    fn skips_heads_shared_by_two_modules() {
        let parts = vec![
            reference("foo", "Value"),
            AnnotationPart::Text(" | ".to_owned()),
            reference("bar", "Value"),
            AnnotationPart::Text(" | ".to_owned()),
            reference("bar", "Other"),
        ];

        assert_eq!(
            candidates("", &parts),
            vec![("bar".to_owned(), "Other".to_owned())]
        );
    }

    #[test]
    fn skips_references_an_existing_import_resolves() {
        let source =
            "import foo\nimport baz as b\nfrom qux import Qux\nfrom package import module\n";
        let parts = vec![
            reference("foo", "Foo"),
            reference("baz", "Baz"),
            reference("qux", "Qux"),
            reference("package.module", "Type"),
            reference("current", "Local"),
            reference("builtins", "int"),
        ];

        assert_eq!(candidates(source, &parts), Vec::new());
    }

    #[test]
    fn skips_heads_bound_at_module_scope() {
        let source = "from elsewhere import Imported\nclass Klass: pass\ndef fun(): pass\nAssigned = 1\nAnnotated: int = 2\n";
        let parts = vec![
            reference("foo", "Imported"),
            reference("foo", "Klass"),
            reference("foo", "fun"),
            reference("foo", "Assigned"),
            reference("foo", "Annotated"),
            reference("foo", "Free"),
        ];

        assert_eq!(
            candidates(source, &parts),
            vec![("foo".to_owned(), "Free".to_owned())]
        );
    }

    #[test]
    fn renders_direct_imports_and_leaves_the_rest_qualified() {
        let ast = Ast::parse("", PySourceType::Python).0;
        let tracker = ImportTracker::from_ast(&ast, ModuleName::from_str("current"), false);
        let parts = vec![
            reference("foo", "Outer.Inner"),
            AnnotationPart::Text(" | ".to_owned()),
            reference("bar", "Bar"),
        ];

        let (text, missing) = tracker.resolve_annotation(
            &parts,
            ModuleName::from_str("current"),
            &[(ModuleName::from_str("foo"), "Outer")],
        );
        assert_eq!(text, "Outer.Inner | bar.Bar");
        assert_eq!(missing.len(), 1);
        assert!(missing.contains(&ModuleName::from_str("bar")));
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

        let (text, missing) =
            tracker.resolve_annotation(&parts, ModuleName::from_str("current"), &[]);
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

        let (text, missing) =
            tracker.resolve_annotation(&parts, ModuleName::from_str("current"), &[]);
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

        let (text, missing) =
            tracker.resolve_annotation(&parts, ModuleName::from_str("current"), &[]);
        assert_eq!(text, "foo.child.C | bar.child.D");
        assert_eq!(missing.len(), 2);
        assert!(missing.contains(&ModuleName::from_str("foo.child")));
        assert!(missing.contains(&ModuleName::from_str("bar.child")));
    }

    #[test]
    fn resolves_submodules_imported_from_their_parent() {
        let ast = Ast::parse(
            "from package import module\nfrom other import child as alias\n",
            PySourceType::Python,
        )
        .0;
        let tracker = ImportTracker::from_ast(&ast, ModuleName::from_str("current"), false);
        let parts = vec![
            reference("package.module", "Type"),
            AnnotationPart::Text(" | ".to_owned()),
            reference("other.child", "Outer.Inner"),
        ];

        let (text, missing) =
            tracker.resolve_annotation(&parts, ModuleName::from_str("current"), &[]);
        assert_eq!(text, "module.Type | alias.Outer.Inner");
        assert!(missing.is_empty());
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
            tracker.resolve_annotation(&parts, ModuleName::from_str("package.sub.current"), &[]);
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
            &[],
        );
        assert_eq!(text, "foo.Bar");
        assert!(missing.contains(&ModuleName::from_str("foo")));
    }
}
