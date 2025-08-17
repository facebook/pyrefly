/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use pyrefly_python::module::Module;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_util::gas::Gas;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
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
use crate::error::error::Error;
use crate::export::exports::Export;
use crate::state::handle::Handle;

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
            | Binding::Pin(k, ..)
            | Binding::PinUpstream(k, ..)
            | Binding::Default(k, ..) => {
                current_idx = *k;
            }
            Binding::Phi(ks) if !ks.is_empty() => current_idx = *ks.iter().next().unwrap(),
            Binding::CheckLegacyTypeParam(k, _) => {
                let binding = bindings.get(*k);
                current_idx = binding.0;
            }
            Binding::AssignToSubscript(subscript, _)
                if let Some(key) =
                    base_key_of_assign_target(&Expr::Subscript(subscript.clone())) =>
            {
                current_idx = bindings.key_to_idx(&key);
            }
            Binding::AssignToAttribute(attribute, _)
                if let Some(key) =
                    base_key_of_assign_target(&Expr::Attribute(attribute.clone())) =>
            {
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
            Binding::CheckLegacyTypeParam(k, _) => {
                let binding = bindings.get(*k);
                current_binding = bindings.get(binding.0);
            }
            Binding::Import(m, name, original_name_range) => {
                return Some(IntermediateDefinition::NamedImport(
                    def_key.range(),
                    *m,
                    name.clone(),
                    *original_name_range,
                ));
            }
            Binding::Module(name, ..) => return Some(IntermediateDefinition::Module(*name)),
            Binding::Function(idx, ..) => {
                let func = bindings.get(*idx);
                let undecorated = bindings.get(func.undecorated_idx);
                return Some(IntermediateDefinition::Local(Export {
                    location: undecorated.def.name.range,
                    symbol_kind: Some(SymbolKind::Function),
                    docstring_range: func.docstring_range,
                    is_deprecated: false,
                }));
            }
            Binding::ClassDef(idx, ..) => {
                return match bindings.get(*idx) {
                    BindingClass::FunctionalClassDef(..) => {
                        Some(IntermediateDefinition::Local(Export {
                            location: def_key.range(),
                            symbol_kind: Some(SymbolKind::Class),
                            docstring_range: None,
                            is_deprecated: false,
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
                        is_deprecated: false,
                    })),
                };
            }
            _ => {
                return Some(IntermediateDefinition::Local(Export {
                    location: def_key.range(),
                    symbol_kind: current_binding.symbol_kind(),
                    docstring_range: None,
                    is_deprecated: false,
                }));
            }
        }
    }
    None
}

pub fn insert_import_edit(
    ast: &ModModule,
    handle_to_import_from: Handle,
    export_name: &str,
) -> (TextSize, String) {
    let position = if let Some(first_stmt) = ast.body.first() {
        first_stmt.range().start()
    } else {
        ast.range.end()
    };
    let insert_text = format!(
        "from {} import {}\n",
        handle_to_import_from.module().as_str(),
        export_name
    );
    (position, insert_text)
}

pub fn create_ignore_code_action(
    error: &Error,
    module_info: &Module,
) -> Option<(String, Module, TextRange, String)> {
    let ignore_lines_in_module = module_info.ignore().get_pyrefly_ignores();
    let start_line = error.display_range().start.line;

    if ignore_lines_in_module.contains(&start_line) {
        create_add_to_existing_ignore_action(error, module_info)
    } else {
        create_new_ignore_action(error, module_info)
    }
}

fn create_add_to_existing_ignore_action(
    error: &Error,
    module_info: &Module,
) -> Option<(String, Module, TextRange, String)> {
    let dec = error.display_range().start.line.decrement()?;
    let suppression_comment = module_info
        .lined_buffer()
        .content_in_line_range(dec, dec)
        .trim_end();
    let bracket_pos = suppression_comment.rfind(']')?;
    let row_offset = module_info.lined_buffer().line_start(dec);
    let text_range = TextRange::new(
        row_offset + TextSize::from(bracket_pos as u32),
        row_offset + TextSize::from((bracket_pos + 1) as u32),
    );

    Some((
        format!("Add {} to above ignore comment", error.error_kind()),
        module_info.dupe(),
        text_range,
        format!(", {}]", error.error_kind().to_name()),
    ))
}

fn create_new_ignore_action(
    error: &Error,
    module_info: &Module,
) -> Option<(String, Module, TextRange, String)> {
    let start = error.range().start();
    let display_pos = module_info.display_pos(start);
    let start = module_info.lined_buffer().line_start(display_pos.line);
    let line = module_info
        .lined_buffer()
        .content_in_line_range(display_pos.line, display_pos.line);
    let offset = line.find(|c: char| !c.is_whitespace()).unwrap_or(0);
    let leading_indentation = " ".repeat(offset as usize);

    Some((
        format!("Suppress {} with ignore comment", error.error_kind()),
        module_info.dupe(),
        TextRange::new(start, start),
        format!(
            "{}# pyrefly: ignore[{}]\n",
            leading_indentation,
            error.error_kind().to_name()
        ),
    ))
}
