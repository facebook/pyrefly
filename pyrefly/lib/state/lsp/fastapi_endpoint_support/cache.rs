/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Source-versioned route and binding summaries without retaining whole-module ASTs.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Weak;

use pyrefly_python::ast::Ast;
use pyrefly_python::docstring::Docstring;
use pyrefly_python::module::Module;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::lock::Mutex;
use ruff_python_ast::ModModule;
use ruff_python_ast::PySourceType;
use ruff_python_ast::Stmt;
use ruff_python_ast::visitor::Visitor;
use ruff_python_ast::visitor::walk_stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use super::EndpointDecoratorSpec;
use super::association_summary::SourceBindings;
use super::endpoint_decorator_spec;

#[derive(Default)]
pub(crate) struct EndpointModuleCache {
    entries: Mutex<HashMap<ModulePath, CacheEntry>>,
}

struct CacheEntry {
    source: Weak<String>,
    summary: Arc<ModuleSummary>,
}

pub(super) struct ModuleSummary {
    pub bindings: SourceBindings,
    pub routes: Vec<RouteSummary>,
}

pub(super) struct RouteSummary {
    pub application: TextRange,
    pub spec: EndpointDecoratorSpec,
    pub definition_range: TextRange,
    pub docstring_range: Option<TextRange>,
}

impl EndpointModuleCache {
    pub(super) fn get(&self, module: &Module, ast: Option<&ModModule>) -> Arc<ModuleSummary> {
        if let Some(entry) = self.entries.lock().get(module.path())
            && entry
                .source
                .upgrade()
                .is_some_and(|source| Arc::ptr_eq(&source, module.contents()))
        {
            return entry.summary.clone();
        }
        let summary = Arc::new(match ast {
            Some(ast) => ModuleSummary::from_ast(ast),
            None => {
                let (ast, _, _) = Ast::parse(module.contents(), PySourceType::Python);
                ModuleSummary::from_ast(&ast)
            }
        });
        let mut entries = self.entries.lock();
        entries.retain(|_, entry| entry.source.strong_count() > 0);
        entries.insert(
            module.path().clone(),
            CacheEntry {
                source: Arc::downgrade(module.contents()),
                summary: summary.clone(),
            },
        );
        summary
    }
}

impl ModuleSummary {
    fn from_ast(ast: &ModModule) -> Self {
        let mut summary = Self {
            bindings: SourceBindings::from_ast(ast),
            routes: Vec::new(),
        };
        summary.visit_body(&ast.body);
        summary
    }
}

impl<'a> Visitor<'a> for ModuleSummary {
    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        if let Stmt::FunctionDef(function) = stmt {
            for decorator in &function.decorator_list {
                let Some(application) = self
                    .bindings
                    .route_owners
                    .get(&decorator.expression.range())
                else {
                    continue;
                };
                if let Some(spec) = endpoint_decorator_spec(&decorator.expression)
                    && spec.path.starts_with('/')
                {
                    self.routes.push(RouteSummary {
                        application: *application,
                        spec,
                        definition_range: function.name.range(),
                        docstring_range: Docstring::range_from_stmts(&function.body),
                    });
                }
            }
        }
        walk_stmt(self, stmt);
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use pyrefly_python::module::Module;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;

    use super::EndpointModuleCache;

    fn module(path: &str, source: &str) -> Module {
        Module::new(
            ModuleName::from_str("app"),
            ModulePath::memory(PathBuf::from(path)),
            Arc::new(source.to_owned()),
        )
    }

    #[test]
    fn source_version_reuses_and_replaces_summary() {
        let cache = EndpointModuleCache::default();
        let old = module(
            "app.py",
            "app = Factory()\n@app.get('/old')\ndef old(): pass\n",
        );
        let first = cache.get(&old, None);
        assert!(Arc::ptr_eq(&first, &cache.get(&old, None)));
        let new = module(
            "app.py",
            "app = Factory()\n@app.get('/new')\ndef new(): pass\n",
        );
        let second = cache.get(&new, None);
        assert!(!Arc::ptr_eq(&first, &second));
        assert_eq!(first.routes[0].spec.path, "/old");
        assert_eq!(second.routes[0].spec.path, "/new");
        assert_eq!(cache.entries.lock().len(), 1);
    }

    #[test]
    fn cache_does_not_keep_obsolete_source_alive() {
        let cache = EndpointModuleCache::default();
        let old = module(
            "old.py",
            "app = Factory()\n@app.get('/old')\ndef old(): pass\n",
        );
        let source = Arc::downgrade(old.contents());
        let summary = cache.get(&old, None);
        drop(old);
        assert!(source.upgrade().is_none());
        assert_eq!(summary.routes[0].spec.path, "/old");
        let new = module(
            "new.py",
            "app = Factory()\n@app.get('/new')\ndef new(): pass\n",
        );
        cache.get(&new, None);
        assert_eq!(cache.entries.lock().len(), 1);
    }
}
