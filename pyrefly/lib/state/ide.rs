/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
use ruff_python_ast::helpers::is_docstring_stmt;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

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
    Module(ModuleName),
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
    let mut current_idx = bindings.key_to_idx(key);
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
            Binding::Phi(_, ks) if !ks.is_empty() => current_idx = *ks.iter().next().unwrap(),
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
                return Some(IntermediateDefinition::Module(imported_module_name));
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
) -> (TextSize, String, String) {
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

/// Insert `import <>` import at the top of the file
pub fn import_regular_import_edit(
    ast: &ModModule,
    handle_to_import_from: Handle,
) -> (TextSize, String) {
    let position = if let Some(first_stmt) = ast.body.iter().find(|stmt| !is_docstring_stmt(stmt)) {
        first_stmt.range().start()
    } else {
        ast.range.end()
    };
    let module_name_to_import = handle_to_import_from.module();
    let insert_text = format!("import {}\n", module_name_to_import.as_str());
    (position, insert_text)
}

pub fn insert_import_edit_with_forced_import_format(
    ast: &ModModule,
    handle_to_insert_import: Handle,
    handle_to_import_from: Handle,
    export_name: &str,
    use_absolute_import: bool,
) -> (TextSize, String, String) {
    let position = if let Some(first_stmt) = ast.body.iter().find(|stmt| !is_docstring_stmt(stmt)) {
        first_stmt.range().start()
    } else {
        ast.range.end()
    };
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
    let insert_text = format!(
        "from {} import {}\n",
        module_name_to_import.as_str(),
        export_name
    );
    (position, insert_text, module_name_to_import.to_string())
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
    let config = config_finder.python_file(handle.module(), handle.path());
    config
        .search_path()
        .any(|search_path| handle.path().as_path().starts_with(search_path))
        || config
            .site_package_path()
            .any(|search_path| handle.path().as_path().starts_with(search_path))
}
