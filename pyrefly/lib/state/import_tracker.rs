/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Helpers for harvesting imports and formatting type strings for inlay hints.

use std::cmp::Reverse;

use pyrefly_python::module_name::ModuleName;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtImport;
use ruff_python_ast::StmtImportFrom;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

/// Tracks imports already present in a module and can determine which modules are still missing
/// for a given set of referenced modules. Also supports alias-aware replacement when displaying
/// type strings.
#[derive(Default)]
pub struct ImportTracker {
    canonical_modules: SmallSet<ModuleName>,
    alias_modules: Vec<(ModuleName, String)>,
    imported_names: SmallMap<ModuleName, SmallMap<String, String>>,
}

impl ImportTracker {
    /// Build an import tracker from the top-level `import ...` statements in a module.
    pub fn from_ast(ast: &ModModule) -> Self {
        let mut tracker = Self::default();
        for stmt in &ast.body {
            if let Stmt::Import(stmt_import) = stmt {
                tracker.record_import(stmt_import);
            } else if let Stmt::ImportFrom(stmt_import_from) = stmt {
                tracker.record_import_from(stmt_import_from);
            }
        }
        tracker
            .alias_modules
            .sort_by_key(|(module, _)| Reverse(module.as_str().len()));
        tracker
    }

    /// Record an `import ...` statement into the tracker.
    pub fn record_import(&mut self, stmt_import: &StmtImport) {
        for alias in &stmt_import.names {
            let module_name = ModuleName::from_str(alias.name.as_str());
            if let Some(asname) = &alias.asname {
                self.alias_modules
                    .push((module_name, asname.id.to_string()));
            } else {
                self.canonical_modules.insert(module_name);
            }
        }
    }

    /// Record a `from ... import ...` statement into the tracker.
    pub fn record_import_from(&mut self, stmt_import_from: &StmtImportFrom) {
        let Some(module) = &stmt_import_from.module else {
            return;
        };
        let module_name = ModuleName::from_str(module.as_str());
        let entry = self.imported_names.entry(module_name).or_default();
        for alias in &stmt_import_from.names {
            let name = alias.name.as_str();
            if name == "*" {
                continue;
            }
            let alias_name = alias
                .asname
                .as_ref()
                .map(|id| id.id.to_string())
                .unwrap_or_else(|| name.to_owned());
            entry.insert(name.to_owned(), alias_name);
        }
    }

    fn module_is_imported(&self, module: ModuleName) -> bool {
        self.alias_for(module).is_some() || self.has_canonical(module)
    }

    /// Whether the module is imported via `import module` (with or without alias).
    pub fn has_module_import(&self, module: ModuleName) -> bool {
        self.module_is_imported(module)
    }

    /// Returns the alias for a module if it was imported as `import module as alias`.
    /// If a parent module was aliased, returns the alias with the remaining suffix.
    pub fn alias_for_module(&self, module: ModuleName) -> Option<String> {
        self.alias_for(module)
    }

    /// Returns the locally imported name for a `from module import name` statement.
    pub fn imported_name_alias(&self, module: ModuleName, name: &str) -> Option<&str> {
        self.imported_names
            .get(&module)
            .and_then(|names| names.get(name))
            .map(|s| s.as_str())
    }

    fn alias_for(&self, module: ModuleName) -> Option<String> {
        let target = module.as_str();
        for (alias_module, alias_name) in &self.alias_modules {
            let alias_module_str = alias_module.as_str();
            if alias_module_str.is_empty() {
                continue;
            }
            if target == alias_module_str {
                return Some(alias_name.clone());
            }
            if target.len() > alias_module_str.len()
                && target.starts_with(alias_module_str)
                && target.as_bytes()[alias_module_str.len()] == b'.'
            {
                let remainder = &target[alias_module_str.len()..];
                return Some(format!("{alias_name}{remainder}"));
            }
        }
        None
    }

    fn has_canonical(&self, module: ModuleName) -> bool {
        let target = module.as_str();
        self.canonical_modules.iter().any(|imported| {
            let imported_str = imported.as_str();
            imported_str == target
                || (target.len() > imported_str.len()
                    && target.starts_with(imported_str)
                    && target.as_bytes()[imported_str.len()] == b'.')
        })
    }
}
