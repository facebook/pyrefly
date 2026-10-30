/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Associate a statically constructed TestClient with its application binding.

use std::sync::Arc;

use dupe::Dupe;
use pyrefly_build::handle::Handle;
use pyrefly_python::module::Module;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::types::Type;
use ruff_python_ast::Expr;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::Hashed;

use super::association_summary::Declaration;
use super::provenance;
use crate::binding::binding::Binding;
use crate::binding::binding::Key;
use crate::state::state::Transaction;

struct Definition {
    handle: Handle,
    module: Module,
    range: TextRange,
}

/// Navigation chooses one branch at joins; application identity must not do so.
fn unambiguous_reference(transaction: &Transaction<'_>, handle: &Handle, reference: &Expr) -> bool {
    let name = match reference {
        Expr::Name(name) => name,
        Expr::Attribute(attribute) => {
            return matches!(
                transaction.get_type_trace(handle, attribute.value.range()),
                Some(Type::Module(_))
            ) && unambiguous_reference(transaction, handle, &attribute.value);
        }
        _ => return false,
    };
    let Some(answers) = transaction.get_answers(handle) else {
        return false;
    };
    let bindings = answers.bindings();
    let key = Key::BoundName(ShortIdentifier::expr_name(name));
    let Some(mut index) = bindings.key_to_idx_hashed_opt(Hashed::new(&key)) else {
        return false;
    };
    for _ in 0..100 {
        let current_key = bindings.idx_to_key(index);
        let binding = bindings.get(index);
        if matches!(binding, Binding::Phi(..) | Binding::LoopPhi(..)) {
            return false;
        }
        if matches!(current_key, Key::Definition(..) | Key::Import(..)) {
            return true;
        }
        match binding {
            Binding::Forward(next)
            | Binding::PromoteForward(next)
            | Binding::ForwardToFirstUse(next)
            | Binding::Narrow(next, _, _) => index = *next,
            Binding::Module(module)
                if let (Key::ModuleAlias(..), Some(previous)) = (current_key, module.2) =>
            {
                index = previous
            }
            _ => return false,
        }
    }
    false
}

fn reference_definition(
    transaction: &Transaction<'_>,
    handle: &Handle,
    reference: &Expr,
) -> Option<Definition> {
    if !unambiguous_reference(transaction, handle, reference) {
        return None;
    }
    let position = match reference {
        Expr::Name(name) => name.range().start(),
        Expr::Attribute(attribute) => attribute.attr.range().start(),
        _ => return None,
    };
    let definitions = transaction.goto_definition(handle, position).ok()?;
    if definitions.len() != 1 {
        return None;
    }
    let definition = definitions.into_iter().next()?;
    let target = if definition.module.path() == handle.path()
        && definition.module.name() == handle.module()
    {
        handle.dupe()
    } else {
        let target = transaction
            .import_handle(handle, definition.module.name(), None)
            .finding()?;
        if target.path() != definition.module.path() {
            return None;
        }
        target
    };
    let target = if let ModulePathDetails::FileSystem(path) = target.path().details() {
        let memory = transaction
            .get_config(&target)?
            .handle_from_module_path(ModulePath::memory(path.to_path_buf()));
        if transaction.get_module_info(&memory).is_some() {
            memory
        } else {
            target
        }
    } else {
        target
    };
    let module = transaction.get_module_info(&target)?;
    let summary = transaction.endpoint_module_cache().get(&module, None);
    let range = if module.path() != definition.module.path()
        || !Arc::ptr_eq(module.contents(), definition.module.contents())
    {
        // Declaration ranges belong to one source version; exported names survive buffer edits.
        let name = definition.module.code_at(definition.range);
        *summary.bindings.exported_origins.get(name)?
    } else {
        if let Some(origin) = summary.bindings.application_origins.get(&definition.range) {
            let name = module.code_at(definition.range);
            if summary.bindings.exported_origins.get(name) != Some(origin) {
                return None;
            }
        } else if definition.module.path() != handle.path()
            || matches!(reference, Expr::Attribute(_))
        {
            return None;
        }
        definition.range
    };
    Some(Definition {
        handle: target,
        module,
        range,
    })
}

fn direct_constructor(transaction: &Transaction<'_>, handle: &Handle, callee: &Expr) -> bool {
    let Some(Type::ClassDef(class)) = transaction.get_type_trace(handle, callee.range()) else {
        return false;
    };
    class.has_toplevel_qname("starlette.testclient", "TestClient")
        || class.has_toplevel_qname("fastapi.testclient", "TestClient")
}

/// Only a direct constructor exposes the meaning of its `app` argument; factories and
/// custom subclass constructors can give those arguments a different meaning.
pub(super) fn application_for_client(
    transaction: &Transaction<'_>,
    handle: &Handle,
    receiver: &Expr,
) -> Option<(Module, TextRange)> {
    let mut handle = handle.dupe();
    let mut value = receiver.clone();
    for _ in 0..16 {
        let declaration = if matches!(value, Expr::Call(_)) {
            Declaration::from_value(&value)?
        } else {
            let definition = reference_definition(transaction, &handle, &value)?;
            let ast = transaction.get_ast(&definition.handle);
            let summary = transaction
                .endpoint_module_cache()
                .get(&definition.module, ast.as_deref());
            let declaration = summary
                .bindings
                .declarations
                .get(&definition.range)?
                .clone();
            handle = definition.handle;
            declaration
        };
        match declaration {
            Declaration::Alias(alias) => value = alias,
            Declaration::Call {
                callee,
                application,
            } => {
                let application = application?;
                if !direct_constructor(transaction, &handle, &callee)
                    || !provenance::is_application(transaction, &handle, application.range())
                {
                    return None;
                }
                let definition = reference_definition(transaction, &handle, &application)?;
                let ast = transaction.get_ast(&definition.handle);
                let summary = transaction
                    .endpoint_module_cache()
                    .get(&definition.module, ast.as_deref());
                let origin = *summary
                    .bindings
                    .application_origins
                    .get(&definition.range)?;
                return Some((definition.module, origin));
            }
        }
    }
    None
}
