/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::LazyLock;

use dupe::Dupe;
use pyrefly_build::handle::Handle;
use pyrefly_config::finder::ConfigFinder;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_util::gas::Gas;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtImportFrom;
use ruff_python_ast::helpers::is_docstring_stmt;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use starlark_map::Hashed;
use starlark_map::small_map::SmallMap;
use starlark_map::smallmap;

use crate::binding::binding::Binding;
use crate::binding::binding::BindingClass;
use crate::binding::binding::ClassBinding;
use crate::binding::binding::Key;
use crate::binding::bindings::Bindings;
use crate::binding::narrow::identifier_and_chain_for_expr;
use crate::binding::narrow::identifier_and_chain_prefix_for_expr;
use crate::export::exports::Export;
use crate::state::lsp::ImportFormat;

const KEY_TO_DEFINITION_INITIAL_GAS: Gas = Gas::new(100);

pub enum IntermediateDefinition {
    Local(Export),
    NamedImport(TextRange, ModuleName, Name, Option<TextRange>),
    Module(TextRange, ModuleName),
}

pub(crate) struct ImportEdit {
    pub range: TextRange,
    pub new_text: String,
    pub display_text: String,
    pub module_name: String,
}

pub fn key_to_intermediate_definition(
    bindings: &Bindings,
    key: &Key,
) -> Option<IntermediateDefinition> {
    let def_key = find_definition_key_from(bindings, key)?;
    create_intermediate_definition_from(bindings, def_key)
}

/// If `key` is already a definition, return it.
/// Otherwise, follow the use-def chain in bindings, and return non-None if we could reach a definition.
fn find_definition_key_from<'a>(bindings: &'a Bindings, key: &'a Key) -> Option<&'a Key> {
    let mut gas = KEY_TO_DEFINITION_INITIAL_GAS;
    let mut current_idx = bindings.key_to_idx_hashed_opt(Hashed::new(key))?;
    let base_key_of_assign_target = |expr: &Expr| {
        if let Some((id, _)) = identifier_and_chain_for_expr(expr) {
            Some(Key::BoundName(ShortIdentifier::new(&id)))
        } else if let Some((id, _)) = identifier_and_chain_prefix_for_expr(expr) {
            Some(Key::BoundName(ShortIdentifier::new(&id)))
        } else {
            None
        }
    };
    while !gas.stop() {
        let current_key = bindings.idx_to_key(current_idx);
        match current_key {
            Key::Definition(..) | Key::Import(..) => {
                // These keys signal that we've reached a definition within the current module
                return Some(current_key);
            }
            _ => {}
        }
        match bindings.get(current_idx) {
            Binding::Forward(k)
            | Binding::Narrow(k, _, _)
            | Binding::CompletedPartialType(k, ..)
            | Binding::PartialTypeWithUpstreamsCompleted(k, ..)
            | Binding::LoopPhi(k, ..) => {
                current_idx = *k;
            }
            Binding::Phi(_, branches) if !branches.is_empty() => {
                current_idx = branches[0].value_key
            }
            Binding::PossibleLegacyTParam(k, _) => {
                let binding = bindings.get(*k);
                current_idx = binding.idx();
            }
            Binding::AssignToSubscript(subscript, _)
                if let Some(key) =
                    base_key_of_assign_target(&Expr::Subscript(subscript.clone())) =>
            {
                current_idx = bindings.key_to_idx(&key);
            }
            Binding::AssignToAttribute {
                attr: attribute, ..
            } if let Some(key) = base_key_of_assign_target(&Expr::Attribute(attribute.clone())) => {
                current_idx = bindings.key_to_idx(&key);
            }
            _ => {
                // We have reached the end of the forwarding chain, and did not find any definitions
                break;
            }
        }
    }
    None
}

/// Given a `def_key` which is guaranteed to point to a definition, do our best to construct a
/// `IntermediateDefinition` that holds the most exact information for it.
fn create_intermediate_definition_from(
    bindings: &Bindings,
    def_key: &Key,
) -> Option<IntermediateDefinition> {
    let mut gas = KEY_TO_DEFINITION_INITIAL_GAS;
    let mut current_binding = bindings.get(bindings.key_to_idx(def_key));

    while !gas.stop() {
        match current_binding {
            Binding::Forward(k) => current_binding = bindings.get(*k),
            Binding::PossibleLegacyTParam(k, _) => {
                let binding = bindings.get(*k);
                current_binding = bindings.get(binding.idx());
            }
            Binding::Import(m, name, original_name_range) => {
                return Some(IntermediateDefinition::NamedImport(
                    def_key.range(),
                    *m,
                    name.clone(),
                    *original_name_range,
                ));
            }
            Binding::ImportViaGetattr(m, _name) => {
                // For __getattr__ imports, the name doesn't exist directly in the module,
                // so we point to __getattr__ instead.
                return Some(IntermediateDefinition::NamedImport(
                    def_key.range(),
                    *m,
                    pyrefly_python::dunder::GETATTR.clone(),
                    None,
                ));
            }
            Binding::Module(name, path, ..) => {
                let imported_module_name = if path.len() == 1 {
                    // This corresponds to the case for `import x.y` -- the corresponding key would
                    // always be `Key::Import(x)`, so the actual module that corresponds to the key
                    // should be `x` instead of `x.y`.
                    ModuleName::from_name(&path[0])
                } else {
                    // This corresponds to all other cases (e.g. `import x.y as z` or `from x.y
                    // import z`) -- the corresponding key would be `Key::Definition(z)` so the
                    // actual module that corresponds to the key must be `x.y`.
                    name.dupe()
                };
                return Some(IntermediateDefinition::Module(
                    def_key.range(),
                    imported_module_name,
                ));
            }
            Binding::Function(idx, ..) => {
                let func = bindings.get(*idx);
                let undecorated = bindings.get(func.undecorated_idx);
                let symbol_kind = if undecorated.class_key.is_some() {
                    SymbolKind::Method
                } else {
                    SymbolKind::Function
                };
                return Some(IntermediateDefinition::Local(Export {
                    location: undecorated.def.name.range,
                    symbol_kind: Some(symbol_kind),
                    docstring_range: func.docstring_range,
                    deprecation: None,
                    is_final: false,
                    special_export: None,
                }));
            }
            Binding::ClassDef(idx, ..) => {
                return match bindings.get(*idx) {
                    BindingClass::FunctionalClassDef(..) => {
                        Some(IntermediateDefinition::Local(Export {
                            location: def_key.range(),
                            symbol_kind: Some(SymbolKind::Class),
                            docstring_range: None,
                            deprecation: None,
                            is_final: false,
                            special_export: None,
                        }))
                    }
                    BindingClass::ClassDef(ClassBinding {
                        def,
                        docstring_range,
                        ..
                    }) => Some(IntermediateDefinition::Local(Export {
                        location: def.name.range,
                        symbol_kind: Some(SymbolKind::Class),
                        docstring_range: *docstring_range,
                        deprecation: None,
                        is_final: false,
                        special_export: None,
                    })),
                };
            }
            _ => {
                return Some(IntermediateDefinition::Local(Export {
                    location: def_key.range(),
                    symbol_kind: current_binding.symbol_kind(),
                    docstring_range: None,
                    deprecation: None,
                    is_final: false,
                    special_export: None,
                }));
            }
        }
    }
    None
}

pub fn insert_import_edit(
    ast: &ModModule,
    config_finder: &ConfigFinder,
    handle_to_insert_import: Handle,
    handle_to_import_from: Handle,
    export_name: &str,
    import_format: ImportFormat,
) -> ImportEdit {
    let use_absolute_import = match import_format {
        ImportFormat::Absolute => true,
        ImportFormat::Relative => {
            handle_require_absolute_import(config_finder, &handle_to_import_from)
        }
    };
    insert_import_edit_with_forced_import_format(
        ast,
        handle_to_insert_import,
        handle_to_import_from,
        export_name,
        use_absolute_import,
    )
}

/// Common alias -> module mappings used for auto-imports.
/// These are modeled after Pylance's built-in aliases.
static COMMON_MODULE_ALIASES: LazyLock<SmallMap<&'static str, &'static str>> =
    LazyLock::new(|| {
        smallmap! {
            "np" => "numpy",
            "pd" => "pandas",
            "tf" => "tensorflow",
            "mpl" => "matplotlib",
            "plt" => "matplotlib.pyplot",
            "m" => "math",
            "sp" => "scipy",
            "spio" => "scipy.io",
            "pn" => "panel",
            "hv" => "holoviews",
        }
    });

pub fn common_alias_target_module(alias: &str) -> Option<&'static str> {
    COMMON_MODULE_ALIASES.get(alias).copied()
}

/// Insert `import <module>` (optionally with an alias) at the top of the file.
/// Returns (position, import text, completion label).
pub fn import_regular_import_edit(
    ast: &ModModule,
    handle_to_import_from: Handle,
    alias: Option<&str>,
) -> (TextSize, String, String) {
    let position = if let Some(first_stmt) = ast.body.iter().find(|stmt| !is_docstring_stmt(stmt)) {
        first_stmt.range().start()
    } else {
        ast.range.end()
    };
    let module_name_to_import = handle_to_import_from.module();
    let (import_text, completion_label) = match alias {
        Some(alias) => (
            format!("import {} as {}\n", module_name_to_import.as_str(), alias),
            alias.to_owned(),
        ),
        None => (
            format!("import {}\n", module_name_to_import.as_str()),
            module_name_to_import.as_str().to_owned(),
        ),
    };
    (position, import_text, completion_label)
}

pub fn insert_import_edit_with_forced_import_format(
    ast: &ModModule,
    handle_to_insert_import: Handle,
    handle_to_import_from: Handle,
    export_name: &str,
    use_absolute_import: bool,
) -> ImportEdit {
    let module_name_to_import = if use_absolute_import {
        handle_to_import_from.module()
    } else if let Some(relative_module) = ModuleName::relative_module_name_between(
        handle_to_insert_import.path().as_path(),
        handle_to_import_from.path().as_path(),
    ) {
        relative_module
    } else {
        handle_to_import_from.module()
    };
    let display_text = format!(
        "from {} import {}",
        module_name_to_import.as_str(),
        export_name
    );
    if let Some((range, new_text)) =
        find_import_merge_edit(ast, module_name_to_import.as_str(), export_name)
    {
        return ImportEdit {
            range,
            new_text,
            display_text,
            module_name: module_name_to_import.to_string(),
        };
    }

    let insert_position = find_import_block_insertion_point(ast, module_name_to_import.as_str())
        .unwrap_or_else(|| {
            if let Some(first_stmt) = ast.body.iter().find(|stmt| !is_docstring_stmt(stmt)) {
                first_stmt.range().start()
            } else {
                ast.range.end()
            }
        });
    let insert_text = format!(
        "from {} import {}\n",
        module_name_to_import.as_str(),
        export_name
    );
    ImportEdit {
        range: TextRange::at(insert_position, TextSize::new(0)),
        new_text: insert_text,
        display_text,
        module_name: module_name_to_import.to_string(),
    }
}

fn find_import_merge_edit(
    ast: &ModModule,
    module_name: &str,
    export_name: &str,
) -> Option<(TextRange, String)> {
    for stmt in import_block_stmts(ast) {
        if let Stmt::ImportFrom(import_from) = stmt
            && import_from_module_matches(import_from, module_name)
        {
            if let Some(range) = merge_range_for_import(import_from, export_name) {
                return Some((range, format!(", {}", export_name)));
            }
        }
    }
    None
}

fn find_import_block_insertion_point(ast: &ModModule, module_name: &str) -> Option<TextSize> {
    for stmt in import_block_stmts(ast) {
        if let Stmt::ImportFrom(import_from) = stmt
            && import_from_module_matches(import_from, module_name)
        {
            return Some(stmt.range().start());
        }
    }
    None
}

fn import_block_stmts<'a>(ast: &'a ModModule) -> impl Iterator<Item = &'a Stmt> {
    ast.body
        .iter()
        .skip_while(|stmt| is_docstring_stmt(stmt))
        .take_while(|stmt| matches!(stmt, Stmt::Import(_) | Stmt::ImportFrom(_)))
}

fn import_from_module_matches(import_from: &StmtImportFrom, module_name: &str) -> bool {
    let dots = ".".repeat(import_from.level as usize);
    let module = import_from
        .module
        .as_ref()
        .map(|module| module.as_str())
        .unwrap_or("");
    format!("{dots}{module}") == module_name
}

fn merge_range_for_import(import_from: &StmtImportFrom, export_name: &str) -> Option<TextRange> {
    let mut last_alias_end: Option<TextSize> = None;
    for alias in &import_from.names {
        if alias.name.as_str() == "*" {
            return None;
        }
        if alias.name.as_str() == export_name && alias.asname.is_none() {
            return None;
        }
        let alias_end = alias
            .asname
            .as_ref()
            .map_or(alias.name.range.end(), |asname| asname.range.end());
        last_alias_end = Some(alias_end);
    }
    last_alias_end.map(|end| TextRange::at(end, TextSize::new(0)))
}

/// Some handles must be imported in absolute style,
/// even if the user has `importFormat: "relative"` in their settings.
///
/// For now, we use the following criteria:
/// 1. Bundled typeshed
/// 2. In search path or site packages
fn handle_require_absolute_import(config_finder: &ConfigFinder, handle: &Handle) -> bool {
    if matches!(
        handle.path().details(),
        ModulePathDetails::BundledTypeshed(_)
    ) {
        return true;
    }
    let config = config_finder.python_file(handle.module_kind(), handle.path());
    config
        .search_path()
        .any(|search_path| handle.path().as_path().starts_with(search_path))
        || config
            .site_package_path()
            .any(|search_path| handle.path().as_path().starts_with(search_path))
}
