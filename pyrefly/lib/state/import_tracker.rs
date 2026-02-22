/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Helpers for harvesting imports and formatting type strings for inlay hints.

use std::cmp::Reverse;

use dupe::Dupe;
use pyrefly_python::module_name::ModuleName;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtImport;
use ruff_python_ast::StmtImportFrom;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::types::display::TypeDisplayContext;
use crate::types::types::Type;

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
        let entry = self
            .imported_names
            .entry(module_name)
            .or_insert(SmallMap::new());
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

    /// Replace any module prefixes that have been imported under an alias (e.g. `import typing as t`).
    pub fn apply_aliases(&self, text: &str) -> String {
        if self.alias_modules.is_empty() {
            return text.to_owned();
        }
        let bytes = text.as_bytes();
        let mut result = String::with_capacity(text.len());
        let mut i = 0;
        while i < bytes.len() {
            let mut replaced = false;
            for (module, alias) in &self.alias_modules {
                let module_str = module.as_str();
                if module_str.is_empty() {
                    continue;
                }
                let module_bytes = module_str.as_bytes();
                if i + module_bytes.len() <= bytes.len()
                    && &bytes[i..i + module_bytes.len()] == module_bytes
                    && Self::is_boundary(bytes, i, i + module_bytes.len())
                {
                    result.push_str(alias);
                    i += module_bytes.len();
                    replaced = true;
                    break;
                }
            }
            if !replaced {
                result.push(bytes[i] as char);
                i += 1;
            }
        }
        result
    }

    /// Modules that are referenced in the type string but not yet imported (excluding builtins/current).
    pub fn missing_modules(
        &self,
        modules: &SmallSet<ModuleName>,
        current_module: ModuleName,
    ) -> SmallSet<ModuleName> {
        let mut missing = SmallSet::new();
        for module in modules.iter() {
            let module = module.dupe();
            if module.as_str().is_empty()
                || module == current_module
                || module == ModuleName::builtins()
                || module == ModuleName::extra_builtins()
            {
                continue;
            }
            if self.module_is_imported(module) {
                continue;
            }
            missing.insert(module);
        }
        missing
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

    fn is_boundary(bytes: &[u8], start: usize, end: usize) -> bool {
        (start == 0 || !Self::is_ident(bytes[start - 1]))
            && (end == bytes.len() || !Self::is_ident(bytes[end]))
    }

    fn is_ident(byte: u8) -> bool {
        matches!(byte, b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_')
    }
}

/// Produce a user-facing type string (without module qualifiers) together with all referenced modules
/// (captured with module qualification) so callers can insert the necessary imports.
pub fn format_type_for_annotation(ty: &Type) -> (String, SmallSet<ModuleName>) {
    // First pass: force module names so referenced_modules collects everything, but ignore the text.
    let mut module_ctx = TypeDisplayContext::new(&[ty]);
    module_ctx.always_display_module_name_except_builtins();
    let _ = module_ctx.display(ty).to_string();
    let modules = module_ctx.referenced_modules();

    // Second pass: produce a concise label without module qualifiers.
    let display_ctx = TypeDisplayContext::new(&[ty]);
    let text = display_ctx.display(ty).to_string();
    (text, modules)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::literal::LitStyle;

    #[test]
    fn aliases_are_applied_at_boundaries_only() {
        let module = ModuleName::from_str("typing");
        let mut tracker = ImportTracker::default();
        tracker.alias_modules.push((module, "t".to_owned()));
        assert_eq!(tracker.apply_aliases("typing.Literal"), "t.Literal");
        // Do not replace inside longer identifiers
        assert_eq!(tracker.apply_aliases("mytyping"), "mytyping");
    }

    #[test]
    fn missing_modules_skips_builtin_and_current() {
        let tracker = ImportTracker::default();
        let mut modules = SmallSet::new();
        let current = ModuleName::from_str("pkg.mod");
        modules.insert(current.dupe());
        modules.insert(ModuleName::builtins());
        modules.insert(ModuleName::from_str("typing"));
        let missing = tracker.missing_modules(&modules, current);
        assert!(missing.contains(&ModuleName::from_str("typing")));
        assert_eq!(missing.len(), 1);
    }

    #[test]
    fn format_type_collects_modules_but_returns_short_label() {
        let ty = Type::LiteralString(LitStyle::Implicit);
        let (text, modules) = format_type_for_annotation(&ty);
        assert_eq!(text, "LiteralString");
        assert!(modules.contains(&ModuleName::from_str("typing")));
    }
}
