/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_build::handle::Handle;
use pyrefly_python::module::Module;
use pyrefly_util::display::DisplayWith;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::Arguments;
use ruff_python_ast::Decorator;
use ruff_python_ast::Expr;
use ruff_python_ast::Identifier;
use ruff_python_ast::ModModule;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use vec1::Vec1;

use crate::alt::answers::Answers;
use crate::alt::answers_solver::AnswersSolver;
use crate::binding::bindings::Bindings;
use crate::report::pysa::PysaSolutions;
use crate::report::pysa::module::ModuleId;
use crate::report::pysa::module::ModuleIds;
use crate::state::lsp::FindDefinitionItemWithDocstring;
use crate::state::lsp::FindPreference;
use crate::state::state::Transaction;
use crate::state::state::TransactionHandle;
use crate::types::stdlib::Stdlib;

/// Wraps `&Transaction` to provide only the cross-module operations that pysa needs.
/// Owns a local cache of resolved PysaSolutions to avoid redundant lookups.
pub struct PysaResolver<'a> {
    /* private */ transaction: &'a Transaction<'a>,
    module_ids: &'a ModuleIds,
    /// Handle of the current module
    current_handle: Handle,
    /// The current module's PysaSolutions, for fast direct access.
    current_module_solutions: Arc<PysaSolutions>,
    /// Cache of resolved PysaSolutions, keyed by ModuleId.
    cache: RefCell<HashMap<ModuleId, Arc<PysaSolutions>>>,
}

impl<'a> PysaResolver<'a> {
    pub fn new(
        transaction: &'a Transaction<'a>,
        module_ids: &'a ModuleIds,
        current_handle: Handle,
    ) -> Self {
        let current_module_solutions = transaction.resolve_pysa_solutions(&current_handle);
        let mut cache = HashMap::new();
        let module_id = current_module_solutions.module_id;
        cache.insert(module_id, current_module_solutions.clone());
        Self {
            transaction,
            module_ids,
            current_handle,
            current_module_solutions,
            cache: RefCell::new(cache),
        }
    }

    /// Construct a resolver for tests, pre-building and caching PysaSolutions
    /// for all handles.
    #[cfg(test)]
    pub fn new_for_test(
        transaction: &'a Transaction<'a>,
        module_ids: &'a ModuleIds,
        current_handle: Handle,
        handles: &[Handle],
    ) -> Self {
        let mut cache = HashMap::new();
        for handle in handles {
            let module_id = module_ids.get_from_handle(handle);
            let answers_context =
                ModuleAnswersContext::create(handle.clone(), transaction, module_ids);
            cache.insert(module_id, PysaSolutions::build(&answers_context));
        }
        let current_module_id = module_ids.get_from_handle(&current_handle);
        let current_module_solutions = cache
            .get(&current_module_id)
            .expect("current_handle must be in handles")
            .dupe();
        Self {
            transaction,
            module_ids,
            current_handle,
            current_module_solutions,
            cache: RefCell::new(cache),
        }
    }

    /// Resolve pysa solutions for a given module, demanding it to Solutions
    /// if needed. Caches the result for subsequent lookups by ModuleId.
    pub fn resolve_pysa_solutions(&self, module: &Module) -> Arc<PysaSolutions> {
        let handle = Handle::new(
            module.name(),
            module.path().dupe(),
            self.current_handle.sys_info().dupe(),
        );
        let module_id = self.module_ids.get_from_handle(&handle);
        if let Some(cached) = self.cache.borrow().get(&module_id) {
            return cached.clone();
        }
        let solutions = self.transaction.resolve_pysa_solutions(&handle);
        self.cache.borrow_mut().insert(module_id, solutions.dupe());
        solutions
    }

    /// Look up cached pysa solutions by ModuleId. The module must have been
    /// previously resolved via `resolve_pysa_solutions`.
    pub fn get_cached_solutions(&self, module_id: ModuleId) -> Arc<PysaSolutions> {
        self.cache
            .borrow()
            .get(&module_id)
            .expect("PysaSolutions must be resolved before lookup by ModuleId")
            .clone()
    }

    pub(crate) fn with_solver<R: Sized>(
        &self,
        label: &'static str,
        f: impl FnOnce(AnswersSolver<TransactionHandle>) -> R,
    ) -> Option<R> {
        self.transaction
            .ad_hoc_solve(&self.current_handle, label, f)
    }

    pub fn find_definition(
        &self,
        position: TextSize,
        preference: FindPreference,
    ) -> Vec<FindDefinitionItemWithDocstring> {
        self.transaction
            .find_definition(&self.current_handle, position, preference)
            .map(Vec1::into_vec)
            .unwrap_or_default()
    }

    pub fn find_definition_for_name_use(
        &self,
        name: &Identifier,
        preference: FindPreference,
    ) -> Option<FindDefinitionItemWithDocstring> {
        self.transaction
            .find_definition_for_name_use(&self.current_handle, name, preference)
            .unwrap_or(None)
    }

    pub fn find_definition_for_attribute(
        &self,
        base_range: TextRange,
        name: &Name,
        preference: FindPreference,
    ) -> Vec<FindDefinitionItemWithDocstring> {
        self.transaction
            .find_definition_for_attribute(&self.current_handle, base_range, name, preference)
            .map(Vec1::into_vec)
            .unwrap_or_default()
    }

    pub fn current_module_solutions(&self) -> &PysaSolutions {
        &self.current_module_solutions
    }

    pub fn module_ids(&self) -> &ModuleIds {
        self.module_ids
    }

    #[cfg(test)]
    pub fn transaction_for_tests(&self) -> &Transaction<'_> {
        self.transaction
    }
}

/// Pyrefly information about a single module.
///
/// This is available when building `PysaSolutions`, which includes all the
/// information we will need about a module.
#[derive(Clone, Dupe)]
pub struct ModuleAnswersContext {
    pub handle: Handle,
    pub module_id: ModuleId,
    pub module_info: Module,
    pub stdlib: Arc<Stdlib>,
    pub ast: Arc<ModModule>,
    pub bindings: Bindings,
    pub answers: Arc<Answers>,
}

/// Pyrefly information about a module.
///
/// This includes the `resolver` which allows access to cross-module information.
pub struct ModuleContext<'a> {
    pub answers_context: ModuleAnswersContext,
    pub resolver: &'a PysaResolver<'a>,
}

impl ModuleAnswersContext {
    pub fn create(
        handle: Handle,
        transaction: &Transaction,
        module_ids: &ModuleIds,
    ) -> ModuleAnswersContext {
        let bindings = transaction
            .get_bindings(&handle)
            .expect("bindings should be available for handle");
        let answers = transaction
            .get_answers(&handle)
            .expect("answers should be available for handle");
        let stdlib = transaction.get_stdlib(&handle);
        let ast = transaction
            .get_ast(&handle)
            .expect("AST should be available for handle");
        let module_info = transaction
            .get_module_info(&handle)
            .expect("module info should be available for handle");
        let module_id = module_ids.get_from_handle(&handle);
        ModuleAnswersContext {
            handle,
            module_id,
            module_info,
            stdlib,
            ast,
            bindings,
            answers,
        }
    }
}

impl ModuleContext<'_> {
    /// Convenience accessor for module_ids.
    pub fn module_ids(&self) -> &ModuleIds {
        self.resolver.module_ids()
    }
}

impl<'a, T: DisplayWith<ModuleContext<'a>>> DisplayWith<ModuleContext<'a>> for Option<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, context: &ModuleContext<'a>) -> fmt::Result {
        match self {
            Some(value) => pyrefly_util::display::DisplayWith::fmt(&value, f, context),
            None => write!(f, "Option::None"),
        }
    }
}

impl DisplayWith<ModuleContext<'_>> for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, context: &ModuleContext) -> fmt::Result {
        pyrefly_util::display::DisplayWith::fmt(&self, f, &context.answers_context.module_info)
    }
}

impl DisplayWith<ModuleContext<'_>> for AnyNodeRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, context: &ModuleContext) -> fmt::Result {
        pyrefly_util::display::DisplayWith::fmt(&self, f, &context.answers_context.module_info)
    }
}

impl DisplayWith<ModuleContext<'_>> for Arguments {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, context: &ModuleContext) -> fmt::Result {
        pyrefly_util::display::DisplayWith::fmt(&self, f, &context.answers_context.module_info)
    }
}

impl DisplayWith<ModuleContext<'_>> for Decorator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, context: &ModuleContext) -> fmt::Result {
        pyrefly_util::display::DisplayWith::fmt(&self, f, &context.answers_context.module_info)
    }
}
