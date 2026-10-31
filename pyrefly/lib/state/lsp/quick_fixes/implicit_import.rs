/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::module::Module;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_util::gas::Gas;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::ModModule;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::Hashed;

use crate::ModuleInfo;
use crate::binding::binding::Binding;
use crate::binding::binding::Key;
use crate::state::lsp::Transaction;

/// Extend an unaliased import without changing its scope or binding a new name.
pub(crate) fn import_module_code_action(
    transaction: &Transaction<'_>,
    handle: &Handle,
    module_info: &ModuleInfo,
    ast: &ModModule,
    error_range: TextRange,
    module: ModuleName,
) -> Option<(String, Module, TextRange, String)> {
    let receiver = Ast::locate_node(ast, error_range.start())
        .into_iter()
        .find_map(|node| match node {
            AnyNodeRef::ExprName(name) => Some(name),
            _ => None,
        })?;
    if receiver.id != module.first_component() {
        return None;
    }
    let answers = transaction.get_answers(handle)?;
    let bindings = answers.bindings();
    let use_key = Key::BoundName(ShortIdentifier::expr_name(receiver));
    let mut idx = bindings.key_to_idx_hashed_opt(Hashed::new(&use_key))?;
    let mut gas = Gas::new(100);
    let import_range = loop {
        if gas.stop() {
            return None;
        }
        let key = bindings.idx_to_key(idx);
        match bindings.get(idx) {
            Binding::Module(_) if matches!(key, Key::Import(..) | Key::Definition(..)) => {
                break key.range();
            }
            Binding::Forward(next)
            | Binding::PromoteForward(next)
            | Binding::ForwardToFirstUse(next)
            | Binding::Narrow(next, _, _) => idx = *next,
            // Navigation selects one source at joins. An edit must not select one
            // branch or ignore imports carried over from another loop iteration.
            Binding::Phi(..) | Binding::LoopPhi(..) => return None,
            _ => return None,
        }
    };
    let nodes = Ast::locate_node(ast, import_range.start());
    if !nodes
        .iter()
        .any(|node| matches!(node, AnyNodeRef::StmtImport(_)))
    {
        return None;
    }
    let alias = nodes.into_iter().find_map(|node| match node {
        AnyNodeRef::Alias(alias) => Some(alias),
        _ => None,
    })?;
    // Aliases need a reference rewrite or an extra binding, which can shadow names
    // or produce an unused import that the editor would offer to remove.
    if alias.asname.is_some()
        || !module
            .components()
            .starts_with(&ModuleName::from_name(&alias.name.id).components())
    {
        return None;
    }
    Some((
        format!("Insert import: `import {module}`"),
        module_info.dupe(),
        alias.name.range(),
        module.as_str().to_owned(),
    ))
}
