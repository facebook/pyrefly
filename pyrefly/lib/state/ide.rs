/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::module_name::ModuleName;
use pyrefly_util::gas::Gas;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::binding::binding::Binding;
use crate::binding::binding::Key;
use crate::binding::bindings::Bindings;
use crate::binding::narrow::identifier_and_chain_for_expr;
use crate::binding::narrow::identifier_and_chain_prefix_for_expr;
use crate::export::exports::Export;
use crate::module::short_identifier::ShortIdentifier;
use crate::state::handle::Handle;

pub enum IntermediateDefinition {
    Local(Export),
    NamedImport(TextRange, ModuleName, Name, Option<TextRange>),
    Module(ModuleName),
}

pub fn key_to_intermediate_definition(
    bindings: &Bindings,
    key: &Key,
    gas: &mut Gas,
) -> Option<IntermediateDefinition> {
    let idx = bindings.key_to_idx(key);
    let binding = bindings.get(idx);
    let res = binding_to_intermediate_definition(bindings, binding, key, gas);
    match &res {
        Some(IntermediateDefinition::Local(_))
        | Some(IntermediateDefinition::Module(_))
        | Some(IntermediateDefinition::NamedImport(_, _, _, _)) => res,
        None => {
            if let Key::Definition(x) = key {
                Some(IntermediateDefinition::Local(Export {
                    location: x.range(),
                    symbol_kind: binding.symbol_kind(),
                    docstring: None,
                }))
            } else {
                None
            }
        }
    }
}

fn binding_to_intermediate_definition(
    bindings: &Bindings,
    binding: &Binding,
    key: &Key,
    gas: &mut Gas,
) -> Option<IntermediateDefinition> {
    if gas.stop() {
        return None;
    }

    let mut resolve_assign_to_expr = |expr: &Expr| {
        if let Some((id, _)) = identifier_and_chain_for_expr(expr) {
            key_to_intermediate_definition(
                bindings,
                &Key::BoundName(ShortIdentifier::new(&id)),
                gas,
            )
        } else if let Some((id, _)) = identifier_and_chain_prefix_for_expr(expr) {
            key_to_intermediate_definition(
                bindings,
                &Key::BoundName(ShortIdentifier::new(&id)),
                gas,
            )
        } else {
            None
        }
    };
    match binding {
        Binding::Forward(k) | Binding::Narrow(k, _, _) | Binding::Pin(k, ..) => {
            key_to_intermediate_definition(bindings, bindings.idx_to_key(*k), gas)
        }
        Binding::Default(k, m) => {
            binding_to_intermediate_definition(bindings, m, bindings.idx_to_key(*k), gas)
        }
        Binding::Phi(ks) if !ks.is_empty() => key_to_intermediate_definition(
            bindings,
            bindings.idx_to_key(*ks.iter().next().unwrap()),
            gas,
        ),
        Binding::Import(m, name, original_name_range) => Some(IntermediateDefinition::NamedImport(
            key.range(),
            *m,
            name.clone(),
            *original_name_range,
        )),
        Binding::Module(name, _, _) => Some(IntermediateDefinition::Module(*name)),
        Binding::CheckLegacyTypeParam(k, _) => {
            let binding = bindings.get(*k);
            key_to_intermediate_definition(bindings, bindings.idx_to_key(binding.0), gas)
        }
        Binding::AssignToSubscript(subscript, _) => {
            let expr = Expr::Subscript(subscript.clone());
            resolve_assign_to_expr(&expr)
        }
        Binding::AssignToAttribute(attribute, _) => {
            let expr = Expr::Attribute(attribute.clone());
            resolve_assign_to_expr(&expr)
        }
        _ => None,
    }
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
