/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use pyrefly_build::handle::Handle;
use pyrefly_python::module::Module;
use ruff_python_ast::ModModule;

use crate::alt::answers::Answers;
use crate::binding::bindings::Bindings;
use crate::report::pysa::module::ModuleId;
use crate::report::pysa::module::ModuleIds;
use crate::report::pysa::module::ModuleKey;
use crate::state::state::Transaction;
use crate::types::stdlib::Stdlib;

/// Pyrefly information about a module.
pub struct ModuleContext<'a> {
    pub handle: &'a Handle,
    pub transaction: &'a Transaction<'a>,
    pub bindings: Bindings,
    pub answers: Arc<Answers>,
    pub stdlib: Arc<Stdlib>,
    pub ast: Arc<ModModule>,
    pub module_info: Module,
    pub module_id: ModuleId,
    pub module_ids: &'a ModuleIds,
}

impl ModuleContext<'_> {
    pub fn create<'a>(
        handle: &'a Handle,
        transaction: &'a Transaction<'a>,
        module_ids: &'a ModuleIds,
    ) -> Option<ModuleContext<'a>> {
        Some(ModuleContext {
            handle,
            transaction,
            bindings: transaction.get_bindings(handle)?,
            answers: transaction.get_answers(handle)?,
            stdlib: transaction.get_stdlib(handle),
            ast: transaction.get_ast(handle)?,
            module_info: transaction.get_module_info(handle)?,
            module_id: module_ids.get(ModuleKey::from_handle(handle))?,
            module_ids,
        })
    }
}
