/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_util::gas::Gas;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;

use crate::binding::binding::Binding;
use crate::binding::binding::Key;
use crate::binding::bindings::Bindings;
use crate::export::exports::Export;
use crate::module::module_name::ModuleName;

pub enum IntermediateDefinition {
    Local(Export),
    NamedImport(ModuleName, Name),
    Module(ModuleName),
}

pub fn key_to_intermediate_definition(
    bindings: &Bindings,
    key: &Key,
    gas: &mut Gas,
) -> Option<IntermediateDefinition> {
    let idx = bindings.key_to_idx(key);
    let res = binding_to_intermediate_definition(bindings, bindings.get(idx), gas);
    if res.is_none()
        && let Key::Definition(x) = key
    {
        return Some(IntermediateDefinition::Local(Export {
            location: x.range(),
            docstring: None,
        }));
    }
    if res.is_none()
        && let Key::Anywhere(_, range) = key
    {
        return Some(IntermediateDefinition::Local(Export {
            location: *range,
            docstring: None,
        }));
    }
    res
}

pub fn binding_to_intermediate_definition(
    bindings: &Bindings,
    binding: &Binding,
    gas: &mut Gas,
) -> Option<IntermediateDefinition> {
    if gas.stop() {
        return None;
    }
    match binding {
        Binding::Forward(k) | Binding::Narrow(k, _, _) => {
            key_to_intermediate_definition(bindings, bindings.idx_to_key(*k), gas)
        }
        Binding::Default(_, m) => binding_to_intermediate_definition(bindings, m, gas),
        Binding::Phi(ks) if !ks.is_empty() => key_to_intermediate_definition(
            bindings,
            bindings.idx_to_key(*ks.iter().next().unwrap()),
            gas,
        ),
        Binding::Import(m, name) => Some(IntermediateDefinition::NamedImport(*m, name.clone())),
        Binding::Module(name, _, _) => Some(IntermediateDefinition::Module(*name)),
        Binding::CheckLegacyTypeParam(k, _) => {
            let binding = bindings.get(*k);
            key_to_intermediate_definition(bindings, bindings.idx_to_key(binding.0), gas)
        }
        _ => None,
    }
}
