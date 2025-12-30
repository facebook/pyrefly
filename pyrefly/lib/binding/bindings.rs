/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cell::RefCell;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_graph::index::Idx;
use pyrefly_graph::index::Index;
use pyrefly_graph::index_map::IndexMap;
use pyrefly_python::ast::Ast;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::nesting_context::NestingContext;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_types::type_info::JoinStyle;
use pyrefly_types::types::Type;
use pyrefly_util::display::DisplayWithCtx;
use pyrefly_util::gas::Gas;
use pyrefly_util::uniques::UniqueFactory;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::Identifier;
use ruff_python_ast::ModModule;
use ruff_python_ast::Parameter;
use ruff_python_ast::Stmt;
use ruff_python_ast::TypeParam;
use ruff_python_ast::TypeParams;
use ruff_python_ast::name::Name;
use ruff_python_parser::semantic_errors::SemanticSyntaxChecker;
use ruff_python_parser::semantic_errors::SemanticSyntaxContext;
use ruff_python_parser::semantic_errors::SemanticSyntaxError;
use ruff_python_parser::semantic_errors::SemanticSyntaxErrorKind;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use starlark_map::Hashed;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use vec1::Vec1;
use vec1::vec1;

use crate::binding::binding::AnnotationTarget;
use crate::binding::binding::Binding;
use crate::binding::binding::BindingAnnotation;
use crate::binding::binding::BindingExport;
use crate::binding::binding::BindingLegacyTypeParam;
use crate::binding::binding::FirstUse;
use crate::binding::binding::FunctionParameter;
use crate::binding::binding::Key;
use crate::binding::binding::KeyAnnotation;
use crate::binding::binding::KeyClass;
use crate::binding::binding::KeyDecoratedFunction;
use crate::binding::binding::KeyExport;
use crate::binding::binding::KeyLegacyTypeParam;
use crate::binding::binding::KeyUndecoratedFunction;
use crate::binding::binding::Keyed;
use crate::binding::binding::LastStmt;
use crate::binding::binding::NarrowUseLocation;
use crate::binding::binding::TypeParameter;
use crate::binding::expr::Usage;
use crate::binding::narrow::NarrowOps;
use crate::binding::scope::Exportable;
use crate::binding::scope::FlowStyle;
use crate::binding::scope::NameReadInfo;
use crate::binding::scope::ScopeTrace;
use crate::binding::scope::Scopes;
use crate::binding::scope::UnusedImport;
use crate::binding::scope::UnusedParameter;
use crate::binding::scope::UnusedVariable;
use crate::binding::table::TableKeyed;
use crate::config::base::UntypedDefBehavior;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorInfo;
use crate::export::definitions::MutableCaptureKind;
use crate::export::exports::Exports;
use crate::export::exports::LookupExport;
use crate::export::special::SpecialExport;
use crate::module::module_info::ModuleInfo;
use crate::solver::solver::Solver;
use crate::state::loader::FindError;
use crate::state::loader::FindingOrError;
use crate::table;
use crate::table_for_each;
use crate::table_try_for_each;
use crate::types::globals::ImplicitGlobal;
use crate::types::quantified::QuantifiedKind;
use crate::types::types::Var;

/// The result of looking up a name. Similar to `NameReadInfo`, but
/// differs because the `BindingsBuilder` layer is responsible for both
/// intercepting first-usage reads and for wrapping forward-reference `Key`s
/// in `Idx<Key>` by inserting them into the bindings table.
#[derive(Debug)]
pub enum NameLookupResult {
    /// I am the bound key for this name in the current scope stack.
    /// I might be:
    /// - initialized (either part of the current flow, or an anywhere-style
    ///   lookup across a barrier)
    /// - possibly-initialized (I come from the current flow, but somewhere upstream
    ///   there is branching flow where I was only defined by some branches)
    /// - uninitialized (I am definitely not initialized in a way static analysis
    ///   understands) and this key is either the most recent stale flow key (e.g.
    ///   if I am used after a `del` or is an anywhere-style lookup)
    Found {
        idx: Idx<Key>,
        initialized: InitializedInFlow,
    },
    /// This name is not defined in the current scope stack.
    NotFound,
}

impl NameLookupResult {
    fn found(self) -> Option<Idx<Key>> {
        match self {
            NameLookupResult::Found { idx, .. } => Some(idx),
            NameLookupResult::NotFound => None,
        }
    }
}

#[derive(Debug)]
pub enum InitializedInFlow {
    Yes,
    Conditionally,
    No,
}

impl InitializedInFlow {
    pub fn as_error_message(&self, name: &Name) -> Option<String> {
        match self {
            InitializedInFlow::Yes => None,
            InitializedInFlow::Conditionally => Some(format!("`{name}` may be uninitialized")),
            InitializedInFlow::No => Some(format!("`{name}` is uninitialized")),
        }
    }
}

#[derive(Clone, Dupe, Debug)]
pub struct Bindings(Arc<BindingsInner>);

pub type BindingEntry<K> = (Index<K>, IndexMap<K, <K as Keyed>::Value>);

table! {
    #[derive(Debug, Clone, Default)]
    pub struct BindingTable(pub BindingEntry)
}

#[derive(Clone, Debug)]
struct BindingsInner {
    module_info: ModuleInfo,
    table: BindingTable,
    scope_trace: Option<ScopeTrace>,
    unused_parameters: Vec<UnusedParameter>,
    unused_imports: Vec<UnusedImport>,
    unused_variables: Vec<UnusedVariable>,
}

impl Display for Bindings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn go<K: Keyed>(
            items: &BindingEntry<K>,
            me: &Bindings,
            f: &mut fmt::Formatter<'_>,
        ) -> fmt::Result {
            for (idx, k) in items.0.items() {
                writeln!(
                    f,
                    "{} = {}",
                    me.module().display(k),
                    items.1.get_exists(idx).display_with(me)
                )?;
            }
            Ok(())
        }
        table_try_for_each!(self.0.table, |items| go(items, self, f));
        Ok(())
    }
}

pub struct BindingsBuilder<'a> {
    pub module_info: ModuleInfo,
    pub lookup: &'a dyn LookupExport,
    pub sys_info: &'a SysInfo,
    pub class_count: u32,
    await_context: AwaitContext,
    errors: &'a ErrorCollector,
    solver: &'a Solver,
    uniques: &'a UniqueFactory,
    pub has_docstring: bool,
    pub scopes: Scopes,
    table: BindingTable,
    pub untyped_def_behavior: UntypedDefBehavior,
    unused_parameters: Vec<UnusedParameter>,
    unused_imports: Vec<UnusedImport>,
    unused_variables: Vec<UnusedVariable>,
    semantic_checker: SemanticSyntaxChecker,
    semantic_syntax_errors: RefCell<Vec<SemanticSyntaxError>>,
}

/// An enum tracking whether we are in a generator expression
/// like `(x for x in xs)` - used to allow `await` inside of generators
/// even when a function is not async, for example (await x for x in xs).
///
/// This is legal because the resulting AsyncGenerator does not actually
/// await until iterated (which can only be done in an `async def`).
///
/// In any other comprehension, `await` requires us to be in an `async def`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AwaitContext {
    #[default]
    General,
    GeneratorElement,
}

impl Bindings {
    #[expect(dead_code)] // Useful API
    fn len(&self) -> usize {
        let mut res = 0;
        table_for_each!(&self.0.table, |x: &BindingEntry<_>| res += x.1.len());
        res
    }

    pub fn display<K: Keyed>(&self, idx: Idx<K>) -> impl Display + '_
    where
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        self.module().display(self.idx_to_key(idx))
    }

    pub fn module(&self) -> &ModuleInfo {
        &self.0.module_info
    }

    pub fn unused_parameters(&self) -> &[UnusedParameter] {
        &self.0.unused_parameters
    }

    pub fn unused_imports(&self) -> &[UnusedImport] {
        &self.0.unused_imports
    }

    pub fn unused_variables(&self) -> &[UnusedVariable] {
        &self.0.unused_variables
    }

    pub fn available_definitions(&self, position: TextSize) -> SmallSet<Idx<Key>> {
        if let Some(trace) = &self.0.scope_trace {
            trace.available_definitions(&self.0.table, position)
        } else {
            SmallSet::new()
        }
    }

    pub fn definition_at_position(&self, position: TextSize) -> Option<&Key> {
        if let Some(trace) = &self.0.scope_trace {
            trace.definition_at_position(&self.0.table, position)
        } else {
            None
        }
    }

    /// Within the LSP, check if a key exists.
    /// It may not exist within `if False:` or `if sys.version == 0:` style code.
    pub fn is_valid_key(&self, k: &Key) -> bool {
        self.0.table.get::<Key>().0.key_to_idx(k).is_some()
    }

    pub fn key_to_idx<K: Keyed>(&self, k: &K) -> Idx<K>
    where
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        self.key_to_idx_hashed(Hashed::new(k))
    }

    pub fn key_to_idx_hashed_opt<K: Keyed>(&self, k: Hashed<&K>) -> Option<Idx<K>>
    where
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        self.0.table.get::<K>().0.key_to_idx_hashed(k)
    }

    pub fn key_to_idx_hashed<K: Keyed>(&self, k: Hashed<&K>) -> Idx<K>
    where
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        self.key_to_idx_hashed_opt(k).unwrap_or_else(|| {
            panic!(
                "Internal error: key not found, module `{}`, path `{}`, key {k:?}",
                self.0.module_info.name(),
                self.0.module_info.path(),
            )
        })
    }

    pub fn get<K: Keyed>(&self, idx: Idx<K>) -> &K::Value
    where
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        self.0.table.get::<K>().1.get(idx).unwrap_or_else(|| {
            let key = self.idx_to_key(idx);
            panic!(
                "Internal error: key lacking binding, module={}, path={}, key={}, key-debug={key:?}",
                self.module().name(),
                self.module().path(),
                self.module().display(key),
            )
        })
    }

    pub fn idx_to_key<K: Keyed>(&self, idx: Idx<K>) -> &K
    where
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        self.0.table.get::<K>().0.idx_to_key(idx)
    }

    pub fn keys<K: Keyed>(&self) -> impl ExactSizeIterator<Item = Idx<K>> + '_
    where
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        self.0.table.get::<K>().0.items().map(|(k, _)| k)
    }

    pub fn get_lambda_param(&self, name: &Identifier) -> Var {
        let b = self.get(self.key_to_idx(&Key::Definition(ShortIdentifier::new(name))));
        if let Binding::LambdaParameter(var) = b {
            *var
        } else {
            panic!(
                "Internal error: unexpected binding for lambda parameter `{}` @  {:?}: {}, module={}, path={}",
                &name.id,
                name.range,
                b.display_with(self),
                self.module().name(),
                self.module().path(),
            )
        }
    }

    pub fn get_function_param(&self, name: &Identifier) -> &FunctionParameter {
        let b = self.get(self.key_to_idx(&Key::Definition(ShortIdentifier::new(name))));
        if let Binding::FunctionParameter(p) = b {
            p
        } else {
            panic!(
                "Internal error: unexpected binding for parameter `{}` @  {:?}: {}, module={}, path={}",
                &name.id,
                name.range,
                b.display_with(self),
                self.module().name(),
                self.module().path(),
            )
        }
    }

    pub fn function_has_return_annotation(&self, name: &Identifier) -> bool {
        let b = self.get(self.key_to_idx(&Key::ReturnType(ShortIdentifier::new(name))));
        if let Binding::ReturnType(box r) = b {
            r.kind.has_return_annotation()
        } else if let Binding::Type(_) = b {
            // This happens when we have an un-annotated return & the inference behavior is "skip and infer Any"
            false
        } else {
            panic!(
                "Internal error: unexpected binding for return type `{}` @  {:?}: {}, module={}, path={}",
                &name.id,
                name.range,
                b.display_with(self),
                self.module().name(),
                self.module().path(),
            )
        }
    }

    pub fn new(
        x: ModModule,
        module_info: ModuleInfo,
        exports: Exports,
        solver: &Solver,
        lookup: &dyn LookupExport,
        sys_info: &SysInfo,
        errors: &ErrorCollector,
        uniques: &UniqueFactory,
        enable_trace: bool,
        untyped_def_behavior: UntypedDefBehavior,
    ) -> Self {
        let mut builder = BindingsBuilder {
            module_info: module_info.dupe(),
            lookup,
            sys_info,
            errors,
            solver,
            uniques,
            class_count: 0,
            await_context: AwaitContext::General,
            has_docstring: Ast::has_docstring(&x),
            scopes: Scopes::module(x.range, enable_trace),
            table: Default::default(),
            untyped_def_behavior,
            unused_parameters: Vec::new(),
            unused_imports: Vec::new(),
            unused_variables: Vec::new(),
            semantic_checker: SemanticSyntaxChecker::new(),
            semantic_syntax_errors: RefCell::new(Vec::new()),
        };
        builder.init_static_scope(&x.body, true);
        if module_info.name() != ModuleName::builtins() {
            builder.inject_builtins(ModuleName::builtins(), false);
            if module_info.name() != ModuleName::extra_builtins() {
                builder.inject_builtins(ModuleName::extra_builtins(), true);
            }
        }
        builder.inject_globals();
        builder.stmts(x.body, &NestingContext::toplevel());
        assert_eq!(builder.scopes.loop_depth(), 0);

        // Validate that all entries in __all__ are defined in the module
        for (range, name) in exports.invalid_dunder_all_entries(lookup, &module_info) {
            builder.error(
                range,
                ErrorInfo::Kind(ErrorKind::BadDunderAll),
                format!("Name `{name}` is listed in `__all__` but is not defined in the module"),
            );
        }

        let unused_imports = builder.scopes.collect_module_unused_imports();
        builder.record_unused_imports(unused_imports);
        let scope_trace = builder.scopes.finish();

        let semantic_errors = builder.semantic_syntax_errors.into_inner();
        for error in semantic_errors {
            if Self::should_emit_semantic_syntax_error(&error) {
                builder.errors.add(
                    error.range,
                    ErrorInfo::Kind(ErrorKind::InvalidSyntax),
                    vec1![error.to_string()],
                );
            }
        }

        let exported = exports.exports(lookup);
        for (name, exportable) in scope_trace.exportables().into_iter_hashed() {
            let binding = match exportable {
                Exportable::Initialized(key, Some(ann)) => {
                    Binding::AnnotatedType(ann, Box::new(Binding::Forward(key)))
                }
                Exportable::Initialized(key, None) => Binding::Forward(key),
                Exportable::Uninitialized(key) => {
                    Binding::Forward(builder.table.types.0.insert(key))
                }
            };
            if exported.contains_key_hashed(name.as_ref()) {
                builder
                    .table
                    .insert(KeyExport(name.into_key()), BindingExport(binding));
            }
        }
        Self(Arc::new(BindingsInner {
            module_info,
            table: builder.table,
            scope_trace: if enable_trace {
                Some(scope_trace)
            } else {
                None
            },
            unused_parameters: builder.unused_parameters,
            unused_imports: builder.unused_imports,
            unused_variables: builder.unused_variables,
        }))
    }

    fn should_emit_semantic_syntax_error(error: &SemanticSyntaxError) -> bool {
        match error.kind {
            SemanticSyntaxErrorKind::BreakOutsideLoop
            | SemanticSyntaxErrorKind::ContinueOutsideLoop
            | SemanticSyntaxErrorKind::SingleStarredAssignment
            | SemanticSyntaxErrorKind::DifferentMatchPatternBindings
            | SemanticSyntaxErrorKind::IrrefutableCasePattern(_)
            | SemanticSyntaxErrorKind::LateFutureImport
            | SemanticSyntaxErrorKind::ReboundComprehensionVariable
            | SemanticSyntaxErrorKind::DuplicateParameter(_)
            | SemanticSyntaxErrorKind::NonlocalDeclarationAtModuleLevel
            | SemanticSyntaxErrorKind::MultipleCaseAssignment(_)
            | SemanticSyntaxErrorKind::DuplicateMatchKey(_)
            | SemanticSyntaxErrorKind::DuplicateMatchClassAttribute(_)
            | SemanticSyntaxErrorKind::DuplicateTypeParameter
            | SemanticSyntaxErrorKind::NonModuleImportStar(_) => true,
            // TODO: the following errors aren't being emitted even when enabled
            // we should investigate that
            SemanticSyntaxErrorKind::WriteToDebug(_)
            | SemanticSyntaxErrorKind::MultipleStarredExpressions
            // pyrefly already handles these errors - we should weigh the pros and cons of enabling them
            | SemanticSyntaxErrorKind::InvalidExpression(_, _)
            | SemanticSyntaxErrorKind::FutureFeatureNotDefined(_)
            | SemanticSyntaxErrorKind::AsyncComprehensionInSyncComprehension(_)
            | SemanticSyntaxErrorKind::InvalidStarExpression
            | SemanticSyntaxErrorKind::AwaitOutsideAsyncFunction(_)
            | SemanticSyntaxErrorKind::ReturnOutsideFunction
            | SemanticSyntaxErrorKind::YieldFromInAsyncFunction
            | SemanticSyntaxErrorKind::YieldOutsideFunction(_)
            // The following errors involve modifying our scope implementation
            | SemanticSyntaxErrorKind::LoadBeforeGlobalDeclaration { .. }
            | SemanticSyntaxErrorKind::GlobalParameter(_)
            | SemanticSyntaxErrorKind::LoadBeforeNonlocalDeclaration { .. }
            | SemanticSyntaxErrorKind::NonlocalAndGlobal(_)
            | SemanticSyntaxErrorKind::AnnotatedGlobal(_)
            | SemanticSyntaxErrorKind::AnnotatedNonlocal(_)
            | SemanticSyntaxErrorKind::NonlocalWithoutBinding(_) => false,
        }
    }
}

impl BindingTable {
    pub fn insert<K: Keyed>(&mut self, key: K, value: K::Value) -> Idx<K>
    where
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        let entry = self.get_mut::<K>();
        let idx = entry.0.insert(key);
        self.insert_idx(idx, value)
    }

    pub fn insert_idx<K: Keyed>(&mut self, idx: Idx<K>, value: K::Value) -> Idx<K>
    where
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        let entry = self.get_mut::<K>();
        let existing = entry.1.insert(idx, value);
        if let Some(existing) = existing {
            panic!(
                "Key {:?} already exists with value {:?}, cannot insert new value {:?}",
                entry.0.idx_to_key(idx),
                existing,
                entry.1.get_exists(idx)
            );
        }
        idx
    }

    fn insert_overwrite(&mut self, key: Key, value: Binding) -> Idx<Key> {
        let idx = self.types.0.insert(key);
        self.types.1.insert(idx, value);
        idx
    }

    /// Record the binding of a value to a variable in an Anywhere binding (which
    /// will take the phi of all values bound at different points). If necessary, we
    /// insert the Anywhere.
    fn record_bind_in_anywhere(&mut self, name: Name, range: TextRange, idx: Idx<Key>) {
        let phi_idx = self.types.0.insert(Key::Anywhere(name, range));
        match self.types.1.insert_if_missing(phi_idx, || {
            Binding::Phi(JoinStyle::SimpleMerge, SmallSet::new())
        }) {
            Binding::Phi(_, phi) => {
                phi.insert(idx);
            }
            _ => unreachable!(),
        }
    }

    fn link_predecessor_function(
        &mut self,
        pred_function_idx: Idx<KeyDecoratedFunction>,
        function_idx: Idx<KeyDecoratedFunction>,
    ) {
        let pred_binding = self
            .decorated_functions
            .1
            .get_mut(pred_function_idx)
            .unwrap();
        pred_binding.successor = Some(function_idx);
    }
}

/// An abstraction representing the `Idx<Key>` for a binding that we
/// are currently constructing, which can be used as a factory to create
/// usage values for `ensure_expr`.
///
/// Note that while it wraps a `Usage`, that usage is always `Usage::CurrentIdx`,
/// never some other variant.
#[derive(Debug)]
pub struct CurrentIdx(Usage);

impl CurrentIdx {
    pub fn new(idx: Idx<Key>) -> Self {
        Self(Usage::CurrentIdx(idx, SmallSet::new()))
    }

    pub fn usage(&mut self) -> &mut Usage {
        &mut self.0
    }

    fn idx(&self) -> Idx<Key> {
        match self.0 {
            Usage::CurrentIdx(idx, ..) => idx,
            _ => unreachable!(),
        }
    }

    pub fn into_idx(self) -> Idx<Key> {
        self.idx()
    }

    pub fn decompose(self) -> (SmallSet<Idx<Key>>, Idx<Key>) {
        match self.0 {
            Usage::CurrentIdx(idx, first_used_by) => (first_used_by, idx),
            _ => unreachable!(),
        }
    }
}

impl<'a> BindingsBuilder<'a> {
    /// Given a `key: K = impl Keyed`, get an `Idx<K>` for it. The intended use case
    /// is when creating a complex binding where the process of creating the binding
    /// requires being able to identify what we are binding.
    pub fn idx_for_promise<K>(&mut self, key: K) -> Idx<K>
    where
        K: Keyed,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        self.table.get_mut::<K>().0.insert(key)
    }

    pub fn idx_to_key<K>(&self, idx: Idx<K>) -> &K
    where
        K: Keyed,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        self.table.get::<K>().0.idx_to_key(idx)
    }

    /// Declare a `Key` as a usage, which can be used for name lookups. Like `idx_for_promise`,
    /// this is a promise to later provide a `Binding` corresponding this key.
    pub fn declare_current_idx(&mut self, key: Key) -> CurrentIdx {
        CurrentIdx::new(self.idx_for_promise(key))
    }

    /// Insert a binding into the bindings table immediately, given a `key`
    pub fn insert_binding<K: Keyed>(&mut self, key: K, value: K::Value) -> Idx<K>
    where
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        self.table.insert(key, value)
    }

    /// Like `insert_binding` but will overwrite any existing binding.
    /// Should only be used in exceptional cases.
    pub fn insert_binding_overwrite(&mut self, key: Key, value: Binding) -> Idx<Key> {
        self.table.insert_overwrite(key, value)
    }

    /// Insert a binding into the bindings table, given the `idx` of a key that we obtained previously.
    pub fn insert_binding_idx<K: Keyed>(&mut self, idx: Idx<K>, value: K::Value) -> Idx<K>
    where
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        self.table.insert_idx(idx, value)
    }

    pub fn record_unused_parameters(&mut self, unused: Vec<UnusedParameter>) {
        self.unused_parameters.extend(unused);
    }

    pub fn record_unused_imports(&mut self, unused: Vec<UnusedImport>) {
        self.unused_imports.extend(unused);
    }

    pub fn record_unused_variables(&mut self, unused: Vec<UnusedVariable>) {
        self.unused_variables.extend(unused);
    }

    pub(crate) fn with_await_context<R>(
        &mut self,
        ctx: AwaitContext,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let prev = self.await_context;
        self.await_context = ctx;
        let result = f(self);
        self.await_context = prev;
        result
    }

    pub(crate) fn in_generator_await_context(&self) -> bool {
        matches!(self.await_context, AwaitContext::GeneratorElement)
    }

    /// Insert a binding into the bindings table, given a `Usage`. This will panic if the usage
    /// is `Usage::NoUsageTracking`.
    pub fn insert_binding_current(&mut self, current: CurrentIdx, value: Binding) -> Idx<Key> {
        self.insert_binding_idx(current.into_idx(), value)
    }

    /// Allow access to an `Idx<Key>` given a `LastStmt` coming from a scan of a function body.
    /// This index will not be dangling under two assumptions:
    /// - we bind the function body (note that this isn't true for, e.g. a `@no_type_check` function!)
    /// - our scan of the function body is consistent with our traversal when binding
    pub fn last_statement_idx_for_implicit_return(&mut self, last: LastStmt, x: &Expr) -> Idx<Key> {
        self.table.types.0.insert(match last {
            LastStmt::Expr => Key::StmtExpr(x.range()),
            LastStmt::With(_) => Key::ContextExpr(x.range()),
        })
    }

    /// Given the name of a function def, return a new `Idx<KeyDecoratedFunction>` at which
    /// we will store the result of binding it along with an optional `Idx<Key>` at which
    /// we have the binding for the TypeInfo of any preceding function def of the same name.
    ///
    /// An invariant is that the caller must store a binding for the returned
    /// `Idx<KeyDecoratedFunction>`; failure to do so will lead to a dangling Idx and
    /// a panic at solve time.
    ///
    /// Function bindings are unusual because the `@overload` decorator causes bindings
    /// that would normally be unrelated in control flow to become tied together.
    ///
    /// As a result, when we create a Idx<KeyDecoratedFunction> for binding a function def, we
    /// will want to track any pre-existing binding associated with the same name and
    /// link the bindings together.
    pub fn create_function_index(
        &mut self,
        function_identifier: &Identifier,
    ) -> (Idx<KeyDecoratedFunction>, Option<Idx<Key>>) {
        // Get the index of both the `Key` and `KeyDecoratedFunction` for the preceding function definition, if any
        let (pred_idx, pred_function_idx) = match self
            .scopes
            .function_predecessor_indices(&function_identifier.id)
        {
            Some((pred_idx, pred_function_idx)) => (Some(pred_idx), Some(pred_function_idx)),
            None => (None, None),
        };
        // Create the Idx<KeyDecoratedFunction> at which we'll store the def we are ready to bind now.
        // The caller *must* eventually store a binding for it.
        let function_idx = self.idx_for_promise(KeyDecoratedFunction(ShortIdentifier::new(
            function_identifier,
        )));
        // If we found a previous def, we store a forward reference inside its `BindingDecoratedFunction`.
        if let Some(pred_function_idx) = pred_function_idx {
            self.table
                .link_predecessor_function(pred_function_idx, function_idx);
        }
        (function_idx, pred_idx)
    }

    pub fn init_static_scope(&mut self, x: &[Stmt], top_level: bool) {
        self.scopes.init_current_static(
            x,
            &self.module_info,
            top_level,
            self.lookup,
            self.sys_info,
            &mut |x| {
                self.table
                    .annotations
                    .0
                    .insert(KeyAnnotation::Annotation(x))
            },
        );
    }

    pub fn stmts(&mut self, xs: Vec<Stmt>, parent: &NestingContext) {
        for x in xs {
            self.stmt(x, parent);
        }
    }

    fn inject_globals(&mut self) {
        for global in ImplicitGlobal::implicit_globals(self.has_docstring) {
            let key = Key::ImplicitGlobal(global.name().clone());
            let idx = self.table.insert(key, Binding::Global(global.clone()));
            self.bind_name(global.name(), idx, FlowStyle::Other);
        }
    }

    fn inject_builtins(&mut self, builtins_module: ModuleName, ignore_if_missing: bool) {
        match self.lookup.get(builtins_module) {
            FindingOrError::Finding(builtins_export) => {
                for name in builtins_export.finding.wildcard(self.lookup).iter() {
                    let key = Key::Import(name.clone(), TextRange::default());
                    let idx = self
                        .table
                        .insert(key, Binding::Import(builtins_module, name.clone(), None));
                    self.bind_name(name, idx, FlowStyle::Import(builtins_module, name.clone()));
                }
            }
            FindingOrError::Error(err @ FindError::NotFound(..)) if !ignore_if_missing => {
                let (ctx, msg) = err.display();
                self.error_multiline(
                    TextRange::default(),
                    ErrorInfo::new(ErrorKind::InternalError, ctx.as_deref()),
                    msg,
                );
            }
            FindingOrError::Error(_) => (),
        }
    }

    // Only works for things with `Foo`, or `source.Foo`, or `F` where `from module import Foo as F`.
    // Does not work for things with nested modules - but no SpecialExport's have that.
    pub fn as_special_export(&self, e: &Expr) -> Option<SpecialExport> {
        let mut visited_names: SmallSet<Name> = SmallSet::new();
        let mut visited_keys: SmallSet<Idx<Key>> = SmallSet::new();
        self.as_special_export_inner(e, &mut visited_names, &mut visited_keys)
    }

    fn as_special_export_inner(
        &self,
        e: &Expr,
        visited_names: &mut SmallSet<Name>,
        visited_keys: &mut SmallSet<Idx<Key>>,
    ) -> Option<SpecialExport> {
        match e {
            Expr::Name(name) => {
                if !visited_names.insert(name.id.clone()) {
                    return None;
                }
                self.scopes
                    .as_special_export(&name.id, None, self.module_info.name(), self.lookup)
                    .or_else(|| {
                        self.special_export_via_alias(&name.id, visited_names, visited_keys)
                    })
            }
            Expr::Attribute(ExprAttribute {
                value, attr: name, ..
            }) if let Expr::Name(base_name) = &**value => self.scopes.as_special_export(
                &name.id,
                Some(&base_name.id),
                self.module_info.name(),
                self.lookup,
            ),
            _ => None,
        }
    }

    fn special_export_via_alias(
        &self,
        name: &Name,
        visited_names: &mut SmallSet<Name>,
        visited_keys: &mut SmallSet<Idx<Key>>,
    ) -> Option<SpecialExport> {
        let (idx, style) = self.scopes.binding_idx_for_name(name)?;
        match style {
            FlowStyle::Other
            | FlowStyle::ClassField { .. }
            | FlowStyle::PossiblyUninitialized
            | FlowStyle::Uninitialized => {
                self.special_export_from_binding_idx(idx, visited_names, visited_keys)
            }
            FlowStyle::MergeableImport(_)
            | FlowStyle::Import(..)
            | FlowStyle::ImportAs(_)
            | FlowStyle::FunctionDef(..)
            | FlowStyle::ClassDef
            | FlowStyle::LoopRecursion => None,
        }
    }

    fn special_export_from_binding_idx(
        &self,
        mut idx: Idx<Key>,
        visited_names: &mut SmallSet<Name>,
        visited_keys: &mut SmallSet<Idx<Key>>,
    ) -> Option<SpecialExport> {
        for _ in 0..16 {
            if !visited_keys.insert(idx) {
                return None;
            }
            let binding = self.table.types.1.get(idx)?;
            match binding {
                Binding::CompletedPartialType(inner_idx, _) => {
                    idx = *inner_idx;
                }
                Binding::PartialTypeWithUpstreamsCompleted(inner_idx, _) => {
                    idx = *inner_idx;
                }
                Binding::Forward(inner_idx) => {
                    idx = *inner_idx;
                }
                Binding::NameAssign { expr: value, .. } => {
                    return self.as_special_export_inner(value, visited_names, visited_keys);
                }
                _ => return None,
            }
        }
        None
    }

    pub fn error(&self, range: TextRange, info: ErrorInfo, msg: String) {
        self.errors.add(range, info, vec1![msg]);
    }

    pub fn error_multiline(&self, range: TextRange, info: ErrorInfo, msg: Vec1<String>) {
        self.errors.add(range, info, msg);
    }

    pub fn declare_mutable_capture(&mut self, name: &Identifier, kind: MutableCaptureKind) {
        // Record any errors finding the identity of the mutable capture, and get a binding
        // that provides the type coming from the parent scope.
        let binding = match self
            .scopes
            .validate_mutable_capture_and_get_key(Hashed::new(&name.id), kind)
        {
            Ok(key) => Binding::Forward(self.table.types.0.insert(key)),
            Err(error) => {
                let should_suppress = matches!(kind, MutableCaptureKind::Nonlocal)
                    && self.scopes.in_module_or_class_top_level()
                    && !self.scopes.in_class_body();
                if !should_suppress {
                    self.error(
                        name.range,
                        ErrorInfo::Kind(ErrorKind::UnknownName),
                        error.message(name),
                    );
                }
                Binding::Type(Type::any_error())
            }
        };
        // Insert that type into the current flow.
        let idx = self.insert_binding(Key::MutableCapture(ShortIdentifier::new(name)), binding);
        self.bind_name(&name.id, idx, FlowStyle::Other);
    }

    pub fn lookup_name(&mut self, name: Hashed<&Name>, usage: &mut Usage) -> NameLookupResult {
        match self.scopes.look_up_name_for_read(name) {
            NameReadInfo::Flow {
                idx,
                initialized: is_initialized,
            } => {
                let (idx, first_use) = self.detect_first_use(idx, usage);
                if let Some(used_idx) = first_use {
                    self.record_first_use(used_idx, usage);
                }
                self.scopes.mark_parameter_used(name.key());
                self.scopes.mark_import_used(name.key());
                self.scopes.mark_variable_used(name.key());
                NameLookupResult::Found {
                    idx,
                    initialized: is_initialized,
                }
            }
            NameReadInfo::Anywhere {
                key,
                initialized: is_initialized,
            } => {
                self.scopes.mark_parameter_used(name.key());
                self.scopes.mark_import_used(name.key());
                self.scopes.mark_variable_used(name.key());
                NameLookupResult::Found {
                    idx: self.table.types.0.insert(key),
                    initialized: is_initialized,
                }
            }
            NameReadInfo::NotFound => NameLookupResult::NotFound,
        }
    }

    /// Look up the idx for a name. The first output is the idx to use for the
    /// lookup itself, and the second is possibly used to record the first-usage
    /// for pinning:
    /// - If this is not the first use of a `Binding::Pin`, then the result is just
    ///   `(flow_idx, None)`.
    /// - If this is the first use of a `Binding::Pin` then we look at the usage:
    ///   - If it is `Usage(idx)`, then we return `(unpinned_idx, Some(pinned_idx))`
    ///     which will allow us to expose unpinned types to the first use, then pin.
    ///   - Otherwise, we return `(pinned_idx, Some(pinned_idx))` which will tell
    ///     us to record that the first usage does not pin (and therefore the
    ///     `Binding::Pin` should force placeholder types to default values).
    /// - If this is a secondary read of a `Binding::Pin` and the usage is the same
    ///   usage as the first read, return `(pinned_idx, None)`: we don't need to
    ///   record first use because that is done already, but we want to continue
    ///   forwarding the raw binding throughout this first use.
    fn detect_first_use(
        &self,
        flow_idx: Idx<Key>,
        usage: &mut Usage,
    ) -> (Idx<Key>, Option<Idx<Key>>) {
        match self.table.types.1.get(flow_idx) {
            Some(Binding::CompletedPartialType(unpinned_idx, FirstUse::Undetermined)) => {
                match usage {
                    Usage::StaticTypeInformation | Usage::Narrowing(_) => {
                        (flow_idx, Some(flow_idx))
                    }
                    Usage::CurrentIdx(..) => (*unpinned_idx, Some(flow_idx)),
                }
            }
            Some(Binding::CompletedPartialType(unpinned_idx, first_use)) => match first_use {
                FirstUse::DoesNotPin => (flow_idx, None),
                FirstUse::Undetermined => match usage {
                    Usage::StaticTypeInformation | Usage::Narrowing(_) => {
                        (flow_idx, Some(flow_idx))
                    }
                    Usage::CurrentIdx(..) => (*unpinned_idx, Some(flow_idx)),
                },
                FirstUse::UsedBy(usage_idx) => {
                    // Detect secondary reads of the same name from a first use, and make
                    // sure they all use the raw binding rather than the `Pin`.
                    // TODO(grievejia): This would eliminate cycles formed on the secondary reads,
                    // but may cause nondeterminism in presence of multiple walrus operators
                    // nested in a single expression. We really need to re-think how first-usage
                    // pinning can interact with things like narrowing and walrus.
                    let currently_in_first_use =
                        usage.current_idx().is_some_and(|idx| &idx == usage_idx);
                    if currently_in_first_use {
                        (*unpinned_idx, None)
                    } else {
                        (flow_idx, None)
                    }
                }
            },
            _ => (flow_idx, None),
        }
    }

    /// Record a first use detected in `detect_possible_first_use`.
    fn record_first_use(&mut self, used: Idx<Key>, usage: &mut Usage) {
        match self.table.types.1.get_mut(used) {
            Some(Binding::CompletedPartialType(.., first_use @ FirstUse::Undetermined)) => {
                *first_use = match usage {
                    Usage::CurrentIdx(use_idx, first_uses_of) => {
                        first_uses_of.insert(used);
                        FirstUse::UsedBy(*use_idx)
                    }
                    Usage::StaticTypeInformation | Usage::Narrowing(_) => FirstUse::DoesNotPin,
                };
            }
            b => {
                unreachable!("Expected a Binding::Pin needing first use, got {:?}", b)
            }
        }
    }

    pub fn bind_definition(
        &mut self,
        name: &Identifier,
        binding: Binding,
        style: FlowStyle,
    ) -> Option<Idx<KeyAnnotation>> {
        // Ignore imports and other items from unused variable detection
        if matches!(style, FlowStyle::Other) {
            self.scopes.register_variable(name);
        }
        let idx = self.insert_binding(Key::Definition(ShortIdentifier::new(name)), binding);
        self.bind_name(&name.id, idx, style)
    }

    /// Bind a name in scope to the idx of `current`, inserting `binding` as the binding.
    pub fn bind_current_as(
        &mut self,
        name: &Identifier,
        current: CurrentIdx,
        binding: Binding,
        style: FlowStyle,
    ) -> Option<Idx<KeyAnnotation>> {
        let idx = self.insert_binding_current(current, binding);
        self.bind_name(&name.id, idx, style)
    }

    /// Bind a name in scope to the idx of `current`, without inserting a binding.
    ///
    /// Returns the same data as `bind_name`, which a caller might use to produce the binding
    /// for `current` (which they are responsible for inserting later).
    pub fn bind_current(
        &mut self,
        name: &Name,
        current: &CurrentIdx,
        style: FlowStyle,
    ) -> Option<Idx<KeyAnnotation>> {
        self.bind_name(name, current.idx(), style)
    }

    /// Bind a name in the current flow. Panics if the name is not in the current static scope.
    ///
    /// Return the first annotation for this variable, if one exists, which the binding we
    /// eventually produce for `idx` will often use to verify we don't assign an incompatible type.
    pub fn bind_name(
        &mut self,
        name: &Name,
        idx: Idx<Key>,
        style: FlowStyle,
    ) -> Option<Idx<KeyAnnotation>> {
        let name = Hashed::new(name);
        let write_info = self
            .scopes
            .define_in_current_flow(name, idx, style)
            .unwrap_or_else(|| {
                panic!(
                    "Name `{name}` not found in static scope of module `{}`.",
                    self.module_info.name(),
                )
            });
        if let Some(range) = write_info.anywhere_range {
            self.table
                .record_bind_in_anywhere(name.into_key().clone(), range, idx);
        }
        write_info.annotation
    }

    pub fn type_params(&mut self, x: &mut TypeParams) -> SmallSet<Name> {
        let mut names = SmallSet::new();
        for x in x.type_params.iter_mut() {
            let name = x.name().clone();
            names.insert(name.id.clone());

            // Check for shadowing of type parameters in enclosing Annotation scopes
            if self
                .scopes
                .name_shadows_enclosing_annotation_scope(&name.id)
            {
                self.error(
                    name.range,
                    ErrorInfo::Kind(ErrorKind::InvalidTypeVar),
                    format!(
                        "Type parameter `{}` shadows a type parameter of the same name from an enclosing scope",
                        name.id
                    ),
                );
            }

            let mut default = None;
            let mut bound = None;
            let mut constraints = None;
            let kind = match x {
                TypeParam::TypeVar(tv) => {
                    if let Some(bound_expr) = &mut tv.bound {
                        if let Expr::Tuple(tuple) = &mut **bound_expr {
                            let mut constraint_exprs = Vec::new();
                            for constraint in &mut tuple.elts {
                                self.ensure_type(constraint, &mut None);
                                constraint_exprs.push(constraint.clone());
                            }
                            constraints = Some((constraint_exprs, bound_expr.range()))
                        } else {
                            self.ensure_type(bound_expr, &mut None);
                            bound = Some((**bound_expr).clone());
                        }
                    }
                    if let Some(default_expr) = &mut tv.default {
                        self.ensure_type(default_expr, &mut None);
                        default = Some((**default_expr).clone());
                    }
                    QuantifiedKind::TypeVar
                }
                TypeParam::ParamSpec(x) => {
                    if let Some(default_expr) = &mut x.default {
                        self.ensure_type(default_expr, &mut None);
                        default = Some((**default_expr).clone());
                    }
                    QuantifiedKind::ParamSpec
                }
                TypeParam::TypeVarTuple(x) => {
                    if let Some(default_expr) = &mut x.default {
                        self.ensure_type(default_expr, &mut None);
                        default = Some((**default_expr).clone());
                    }
                    QuantifiedKind::TypeVarTuple
                }
            };
            self.scopes.add_parameter_to_current_static(&name, None);
            self.bind_definition(
                &name,
                Binding::TypeParameter(Box::new(TypeParameter {
                    name: name.id.clone(),
                    unique: self.uniques.fresh(),
                    kind,
                    default,
                    bound,
                    constraints,
                })),
                FlowStyle::Other,
            );
        }
        names
    }

    pub fn bind_narrow_ops(
        &mut self,
        narrow_ops: &NarrowOps,
        use_location: NarrowUseLocation,
        usage: &Usage,
    ) {
        for (name, (op, op_range)) in narrow_ops.0.iter_hashed() {
            if let Some(initial_idx) = self
                .lookup_name(name, &mut Usage::narrowing_from(usage))
                .found()
            {
                let narrowed_idx = self.insert_binding(
                    Key::Narrow(name.into_key().clone(), *op_range, use_location),
                    Binding::Narrow(initial_idx, Box::new(op.clone()), use_location),
                );
                self.scopes.narrow_in_current_flow(name, narrowed_idx);
            }
        }
    }

    pub fn bind_lambda_param(&mut self, name: &Identifier) {
        // Create a parameter var; the binding for the lambda expr itself will use this to pass
        // any contextual typing information as a side-effect to the parameter binding used in
        // the lambda body.
        let var = self.solver.fresh_unwrap(self.uniques);
        let idx = self.insert_binding(
            Key::Definition(ShortIdentifier::new(name)),
            Binding::LambdaParameter(var),
        );
        self.scopes.add_parameter_to_current_static(name, None);
        self.bind_name(&name.id, idx, FlowStyle::Other);
    }

    pub fn bind_function_param(
        &mut self,
        target: AnnotationTarget,
        x: &Parameter,
        undecorated_idx: Idx<KeyUndecoratedFunction>,
        class_key: Option<Idx<KeyClass>>,
        is_variadic: bool,
    ) {
        let name = x.name();
        let allow_unused = name.id.as_str().starts_with('_')
            || matches!(name.id.as_str(), "self" | "cls")
            || is_variadic;
        let annot = x.annotation().map(|x| {
            self.insert_binding(
                KeyAnnotation::Annotation(ShortIdentifier::new(name)),
                BindingAnnotation::AnnotateExpr(target.clone(), x.clone(), class_key),
            )
        });
        let key = self.insert_binding(
            Key::Definition(ShortIdentifier::new(name)),
            Binding::FunctionParameter(match annot {
                Some(annot) => FunctionParameter::Annotated(annot),
                None => FunctionParameter::Unannotated(
                    self.solver.fresh_parameter(self.uniques),
                    undecorated_idx,
                    target,
                ),
            }),
        );
        self.scopes.add_parameter_to_current_static(name, annot);
        self.scopes.register_parameter(name, allow_unused);
        self.bind_name(&name.id, key, FlowStyle::Other);
    }
}

#[derive(Debug)]
pub enum LegacyTParamId {
    /// A simple name referring to a legacy type parameter.
    Name(Identifier),
    /// A <name>.<name> reference to a legacy type parameter.
    Attr(Identifier, Identifier),
}

impl LegacyTParamId {
    /// Get the identifier of the name that will actually be bound (for a normal name, this is
    /// just itself; for a `<base>.<attr>` attribute it is the base portion, which gets narrowed).
    fn as_identifier(&self) -> &Identifier {
        match self {
            Self::Name(name) => name,
            Self::Attr(base, _) => base,
        }
    }

    /// Create the `Key` actually used to model the legacy type parameter
    /// name (or an attribute narrow of the base name, if this is an attribute
    /// of an imported module like `foo.T`) as a type.
    ///
    /// Note that the range here is not the range of the full `LegacyTParamId`, but
    /// just of the name being bound (which in the `Attr` case is just the base
    /// rather than the entire identifier).
    fn as_possible_legacy_tparam_key(&self) -> Key {
        Key::PossibleLegacyTParam(self.as_identifier().range)
    }

    /// Get the key used to track this potential legacy tparam in the `legacy_tparams` map.
    fn tvar_name(&self) -> String {
        match self {
            Self::Name(name) => name.id.as_str().to_owned(),
            Self::Attr(base, attr) => format!("{base}.{attr}"),
        }
    }
}

impl Ranged for LegacyTParamId {
    fn range(&self) -> TextRange {
        match self {
            Self::Name(name) => name.range,
            Self::Attr(_, attr) => attr.range,
        }
    }
}

/// A name we found that might either be a legacy type variable or be a module
/// that has a legacy type variable as an attribute.
struct PossibleTParam {
    id: LegacyTParamId,
    idx: Idx<Key>,
    tparam_idx: Idx<KeyLegacyTypeParam>,
}

enum TParamLookupResult {
    MaybeTParam(PossibleTParam),
    NotTParam(Idx<Key>),
    NotFound,
}

impl TParamLookupResult {
    fn idx(&self) -> Option<Idx<Key>> {
        match self {
            Self::MaybeTParam(possible_tparam) => Some(possible_tparam.idx),
            Self::NotTParam(idx) => Some(*idx),
            Self::NotFound => None,
        }
    }

    fn as_name_lookup_result(&self) -> NameLookupResult {
        self.idx()
            .map_or(NameLookupResult::NotFound, |idx| NameLookupResult::Found {
                idx,
                initialized: InitializedInFlow::Yes,
            })
    }
}

/// Handle intercepting names inside either function parameter/return
/// annotations or base class lists of classes, in order to check whether they
/// point at type variable declarations and need to be converted to type
/// parameters.
pub struct LegacyTParamCollector {
    /// All of the names used. Each one may or may not point at a type variable
    /// and therefore bind a legacy type parameter.
    legacy_tparams: SmallMap<String, TParamLookupResult>,
    /// Are there scoped type parameters? Used to control downstream errors.
    has_scoped_tparams: bool,
}

impl LegacyTParamCollector {
    pub fn new(has_scoped_tparams: bool) -> Self {
        Self {
            legacy_tparams: SmallMap::new(),
            has_scoped_tparams,
        }
    }

    /// Get the keys that correspond to the result of checking whether a name
    /// corresponds to a legacy type param. This is used when actually computing
    /// the final type parameters for classes and functions, which have to take
    /// all the names that *do* map to type variable declarations and combine
    /// them (potentially) with scoped type parameters.
    pub fn lookup_keys(&self) -> Vec<Idx<KeyLegacyTypeParam>> {
        self.legacy_tparams
            .values()
            .filter_map(|x| match x {
                TParamLookupResult::MaybeTParam(possible_tparam) => {
                    Some(possible_tparam.tparam_idx)
                }
                _ => None,
            })
            .collect()
    }
}

/// The legacy-tparams-specifc logic is in a second impl because that lets us define it
/// just under where the key data structures live.
impl<'a> BindingsBuilder<'a> {
    /// Perform a lookup of a name used in either base classes of a class or
    /// parameter/return annotations of a function.
    ///
    /// We have a special "intercepted" lookup to create bindings that allow us
    /// to later determine whether this name points at a type variable
    /// declaration, in which case we intercept it to treat it as a type
    /// parameter in the current scope.
    pub fn intercept_lookup(
        &mut self,
        legacy_tparams: &mut LegacyTParamCollector,
        id: LegacyTParamId,
    ) -> NameLookupResult {
        let result = legacy_tparams
            .legacy_tparams
            .entry(id.tvar_name())
            .or_insert_with(|| self.lookup_legacy_tparam(id, legacy_tparams.has_scoped_tparams));
        result.as_name_lookup_result()
    }

    /// Look up a name that might refer to a legacy tparam. This is used by `intercept_lookup`
    /// when in a setting where we have to check values currently in scope to see if they are
    /// legacy type parameters and need to be re-bound into quantified type variables.
    ///
    /// The returned value will be:
    /// - Either::Right(None) if the name is not in scope; we'll just skip it (the same
    ///   code will be traversed elsewhere, so no need for a duplicate type error)
    /// - Either::Right(Idx<Key>) if the name is in scope and does not point at a
    ///   legacy type parameter. In this case, the intercepted lookup should just forward
    ///   the existing binding.
    /// - Either::Left(Idx<KeyLegacyTypeParameter>) if the name might be a legacy type
    ///   parameter. We actually cannot currently be sure; imported names have to be treated
    ///   as though they *might* be legacy type parameters. Making a final decision is deferred
    ///   until the solve stage.
    fn lookup_legacy_tparam(
        &mut self,
        id: LegacyTParamId,
        has_scoped_type_params: bool,
    ) -> TParamLookupResult {
        let name = id.as_identifier();
        self.lookup_name(Hashed::new(&name.id), &mut Usage::StaticTypeInformation)
            .found()
            .map_or(TParamLookupResult::NotFound, |original_idx| {
                match self.lookup_legacy_tparam_from_idx(id, original_idx, has_scoped_type_params) {
                    Some(possible_tparam) => TParamLookupResult::MaybeTParam(possible_tparam),
                    None => TParamLookupResult::NotTParam(original_idx),
                }
            })
    }

    pub fn get_original_binding(
        &'a self,
        mut original_idx: Idx<Key>,
    ) -> Option<(Idx<Key>, Option<&'a Binding>)> {
        // Follow Forwards to get to the actual original binding.
        // Short circuit if there are too many forwards - it may mean there's a cycle.
        let mut original_binding = self.table.types.1.get(original_idx);
        let mut gas = Gas::new(100);
        while let Some(
            Binding::Forward(fwd_idx)
            | Binding::CompletedPartialType(fwd_idx, _)
            | Binding::Phi(JoinStyle::NarrowOf(fwd_idx), _),
        ) = original_binding
        {
            if gas.stop() {
                return None;
            } else {
                original_idx = *fwd_idx;
                original_binding = self.table.types.1.get(original_idx);
            }
        }
        Some((original_idx, original_binding))
    }

    /// Perform the inner loop of looking up a possible legacy type parameter, given a starting
    /// binding. The loop follows `Forward` nodes backward, and returns:
    /// - Some(...) if we find either a legacy type variable or an import (in which case it *might*
    ///   be a legacy type variable, so we'll let the solve stage decide)
    /// - None if we find something that is definitely not a legacy type variable.
    fn lookup_legacy_tparam_from_idx(
        &mut self,
        id: LegacyTParamId,
        original_idx: Idx<Key>,
        has_scoped_type_params: bool,
    ) -> Option<PossibleTParam> {
        let (original_idx, original_binding) = self.get_original_binding(original_idx)?;
        // If we found a potential legacy type variable, first insert the key / binding pair
        // for the raw lookup, then insert another key / binding pair for the
        // `CheckLegacyTypeParam`, and return the `Idx<Key>`.
        let tparam_idx = Self::make_legacy_tparam(&id, original_binding, original_idx)
            .map(|(k, v)| self.insert_binding(k, v))?;
        let idx = self.insert_binding(
            id.as_possible_legacy_tparam_key(),
            Binding::PossibleLegacyTParam(
                tparam_idx,
                if has_scoped_type_params {
                    Some(id.range())
                } else {
                    None
                },
            ),
        );
        Some(PossibleTParam {
            id,
            idx,
            tparam_idx,
        })
    }

    /// Given a name (either a bare name or a `<base>.<attribute>`) name, produce
    /// `Some((key, binding))` if we cannot rule out that the name is a legacy type
    /// variable; the solver will make the final decision.
    ///
    /// To break down "when we cannot rule out":
    /// - We know for certain that a bare name whose binding is a legacy type
    ///   variable *is* a legacy type varaible
    /// - We cannot be sure in a few cases:
    ///   - a bare name that is an imported name
    ///   - a `module.attr` name, where the base is an imported module
    ///   - either kind of name and a forward reference where we don't yet know
    ///     what it will be
    /// - In all other cases, we know for sure the name is *not* a legacy
    ///   type variable, and we will return `None`
    fn make_legacy_tparam(
        id: &LegacyTParamId,
        binding: Option<&Binding>,
        original_idx: Idx<Key>,
    ) -> Option<(KeyLegacyTypeParam, BindingLegacyTypeParam)> {
        match id {
            LegacyTParamId::Name(name) => match binding {
                Some(
                    Binding::TypeVar(..)
                    | Binding::ParamSpec(..)
                    | Binding::TypeVarTuple(..)
                    | Binding::Import(..),
                )
                | None => Some((
                    KeyLegacyTypeParam(ShortIdentifier::new(name)),
                    BindingLegacyTypeParam::ParamKeyed(original_idx),
                )),
                Some(_) => None,
            },
            LegacyTParamId::Attr(_, attr) => match binding {
                Some(Binding::Module(..)) | None => Some((
                    KeyLegacyTypeParam(ShortIdentifier::new(attr)),
                    BindingLegacyTypeParam::ModuleKeyed(original_idx, Box::new(attr.id.clone())),
                )),
                Some(_) => None,
            },
        }
    }

    /// Add `Definition` bindings to a class or function body scope for all the names
    /// referenced in the function parameter/return annotations or the class bases.
    ///
    /// We do this so that AnswersSolver has the opportunity to determine whether any
    /// of those names point at legacy (pre-PEP-695) type variable declarations, in which
    /// case the name should be treated as a Quantified type parameter inside this scope.
    pub fn add_name_definitions(&mut self, legacy_tparams: &LegacyTParamCollector) {
        for entry in legacy_tparams.legacy_tparams.values() {
            match entry {
                TParamLookupResult::MaybeTParam(possible_tparam) => {
                    self.scopes
                        .add_possible_legacy_tparam(possible_tparam.id.as_identifier());
                }
                _ => {}
            }
        }
    }

    pub fn with_semantic_checker(&mut self, f: impl FnOnce(&mut SemanticSyntaxChecker, &Self)) {
        let mut checker = std::mem::take(&mut self.semantic_checker);
        f(&mut checker, self);
        self.semantic_checker = checker;
    }
}

impl<'a> SemanticSyntaxContext for BindingsBuilder<'a> {
    fn python_version(&self) -> ruff_python_ast::PythonVersion {
        ruff_python_ast::PythonVersion {
            major: self.sys_info.version().major as u8,
            minor: self.sys_info.version().minor as u8,
        }
    }

    fn source(&self) -> &str {
        self.module_info.contents()
    }

    fn future_annotations_or_stub(&self) -> bool {
        self.module_info.source_type() == ruff_python_ast::PySourceType::Stub
    }

    fn report_semantic_error(&self, error: SemanticSyntaxError) {
        self.semantic_syntax_errors.borrow_mut().push(error);
    }

    fn global(&self, name: &str) -> Option<TextRange> {
        self.scopes.get_global_declaration(name)
    }

    fn has_nonlocal_binding(&self, name: &str) -> bool {
        self.scopes.has_nonlocal_binding(name)
    }

    fn in_async_context(&self) -> bool {
        self.scopes.is_in_async_def()
    }

    fn in_await_allowed_context(&self) -> bool {
        // await is allowed in functions, lambdas, and notebooks
        self.scopes.in_function_scope() || self.in_notebook()
    }

    fn in_yield_allowed_context(&self) -> bool {
        // yield is allowed in functions and lambdas, but not in comprehensions or classes
        self.scopes.in_function_scope()
    }

    fn in_sync_comprehension(&self) -> bool {
        self.scopes.in_sync_comprehension()
    }

    fn in_module_scope(&self) -> bool {
        self.scopes.in_module_or_class_top_level() && !self.scopes.in_class_body()
    }

    fn in_function_scope(&self) -> bool {
        self.scopes.in_function_scope()
    }

    fn in_generator_scope(&self) -> bool {
        self.scopes.in_generator_expression()
    }

    fn in_notebook(&self) -> bool {
        self.module_info.source_type() == ruff_python_ast::PySourceType::Ipynb
    }

    fn in_loop_context(&self) -> bool {
        self.scopes.loop_depth() > 0
    }

    fn is_bound_parameter(&self, name: &str) -> bool {
        self.scopes.is_bound_parameter(name)
    }
}
