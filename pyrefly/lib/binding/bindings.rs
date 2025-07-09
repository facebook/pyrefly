/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::mem;
use std::sync::Arc;

use dupe::Dupe;
use itertools::Either;
use itertools::Itertools;
use pyrefly_python::ast::Ast;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::display::DisplayWithCtx;
use pyrefly_util::uniques::UniqueFactory;
use ruff_python_ast::AnyParameterRef;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::Identifier;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::TypeParam;
use ruff_python_ast::TypeParams;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use starlark_map::Hashed;
use starlark_map::small_map::Entry;
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
use crate::binding::binding::Key;
use crate::binding::binding::KeyAnnotation;
use crate::binding::binding::KeyClass;
use crate::binding::binding::KeyExport;
use crate::binding::binding::KeyFunction;
use crate::binding::binding::KeyLegacyTypeParam;
use crate::binding::binding::Keyed;
use crate::binding::binding::LastStmt;
use crate::binding::binding::TypeParameter;
use crate::binding::expr::Usage;
use crate::binding::narrow::NarrowOps;
use crate::binding::scope::Flow;
use crate::binding::scope::FlowInfo;
use crate::binding::scope::FlowStyle;
use crate::binding::scope::Loop;
use crate::binding::scope::LoopExit;
use crate::binding::scope::ScopeKind;
use crate::binding::scope::ScopeTrace;
use crate::binding::scope::Scopes;
use crate::binding::table::TableKeyed;
use crate::config::base::UntypedDefBehavior;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorContext;
use crate::error::kind::ErrorKind;
use crate::export::exports::Exports;
use crate::export::exports::LookupExport;
use crate::export::special::SpecialExport;
use crate::graph::index::Idx;
use crate::graph::index::Index;
use crate::graph::index_map::IndexMap;
use crate::module::module_info::ModuleInfo;
use crate::module::short_identifier::ShortIdentifier;
use crate::solver::solver::Solver;
use crate::state::loader::FindError;
use crate::table;
use crate::table_for_each;
use crate::table_try_for_each;
use crate::types::globals::Global;
use crate::types::quantified::QuantifiedKind;
use crate::types::types::Var;

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
                    me.module_info().display(k),
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
    errors: &'a ErrorCollector,
    solver: &'a Solver,
    uniques: &'a UniqueFactory,
    pub has_docstring: bool,
    pub scopes: Scopes,
    table: BindingTable,
    pub untyped_def_behavior: UntypedDefBehavior,
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
        self.module_info().display(self.idx_to_key(idx))
    }

    pub fn module_info(&self) -> &ModuleInfo {
        &self.0.module_info
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

    pub fn key_to_idx_hashed<K: Keyed>(&self, k: Hashed<&K>) -> Idx<K>
    where
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
    {
        self.0
            .table
            .get::<K>()
            .0
            .key_to_idx_hashed(k)
            .unwrap_or_else(|| {
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
                self.module_info().name(),
                self.module_info().path(),
                self.module_info().display(key),
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
                self.module_info().name(),
                self.module_info().path(),
            )
        }
    }

    pub fn get_function_param(&self, name: &Identifier) -> Either<Idx<KeyAnnotation>, Var> {
        let b = self.get(self.key_to_idx(&Key::Definition(ShortIdentifier::new(name))));
        if let Binding::FunctionParameter(p) = b {
            match p {
                Either::Left(idx) => Either::Left(*idx),
                Either::Right(var) => Either::Right(*var),
            }
        } else {
            panic!(
                "Internal error: unexpected binding for parameter `{}` @  {:?}: {}, module={}, path={}",
                &name.id,
                name.range,
                b.display_with(self),
                self.module_info().name(),
                self.module_info().path(),
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
            has_docstring: Ast::has_docstring(&x),
            scopes: Scopes::module(x.range, enable_trace),
            table: Default::default(),
            untyped_def_behavior,
        };
        builder.init_static_scope(&x.body, true);
        if module_info.name() != ModuleName::builtins() {
            builder.inject_builtins();
        }
        builder.inject_globals();
        builder.stmts(x.body);
        assert_eq!(builder.scopes.loop_depth(), 0);
        let scope_trace = builder.scopes.finish();
        let last_scope = scope_trace.toplevel_scope();
        let exported = exports.exports(lookup);
        for (k, static_info) in last_scope.stat.0.iter_hashed() {
            let info = last_scope.flow.info.get_hashed(k);
            let binding = match info {
                Some(FlowInfo { key, .. }) => {
                    if let Some(ann) = static_info.annot {
                        Binding::AnnotatedType(ann, Box::new(Binding::Forward(*key)))
                    } else {
                        Binding::Forward(*key)
                    }
                }
                None => {
                    // The variable is not in the flow scope, so probably it has not been defined
                    // in any flow that reaches the end. So just use the anywhere version.
                    Binding::Forward(
                        builder
                            .table
                            .types
                            .0
                            .insert(static_info.as_key(k.into_key())),
                    )
                }
            };
            if exported.contains_key_hashed(k) {
                builder
                    .table
                    .insert(KeyExport(k.into_key().clone()), BindingExport(binding));
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
        }))
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

    fn insert_anywhere(&mut self, name: Name, range: TextRange, idx: Idx<Key>) {
        let phi_idx = self.types.0.insert(Key::Anywhere(name, range));
        match self
            .types
            .1
            .insert_if_missing(phi_idx, || Binding::Phi(SmallSet::new()))
        {
            Binding::Phi(phi) => {
                phi.insert(idx);
            }
            _ => unreachable!(),
        }
    }

    fn link_predecessor_function(
        &mut self,
        pred_function_idx: Idx<KeyFunction>,
        function_idx: Idx<KeyFunction>,
    ) {
        let pred_binding = self.functions.1.get_mut(pred_function_idx).unwrap();
        pred_binding.successor = Some(function_idx);
    }
}

/// Errors that can occur when we try to look up a name
pub enum LookupError {
    /// We can't find the name at all
    NotFound,
    /// We expected the name to be mutable from the current scope, but it's not
    NotMutable,
}

impl LookupError {
    pub fn message(&self, name: &Identifier) -> String {
        match self {
            Self::NotFound => format!("Could not find name `{name}`"),
            Self::NotMutable => format!("`{name}` is not mutable from the current scope"),
        }
    }
}

#[derive(PartialEq, Eq)]
pub enum LookupKind {
    Regular,
    /// Look up a name that must be mutable from the current scope, like a `del` or augmented assignment statement
    Mutable,
}

pub enum MutableCaptureLookupError {
    /// We can't find the name at all
    NotFound,
    /// We expected the name to be in an enclosing, non-global scope, but it's not
    NonlocalScope,
    /// This variable was assigned before the nonlocal declaration
    AssignedBeforeNonlocal,
    /// We expected the name to be in the global scope, but it's not
    GlobalScope,
    /// This variable was assigned before the global declaration
    AssignedBeforeGlobal,
}

impl MutableCaptureLookupError {
    pub fn message(&self, name: &Identifier) -> String {
        match self {
            Self::NotFound => format!("Could not find name `{name}`"),
            Self::NonlocalScope => {
                format!("Found `{name}`, but it was not in a valid enclosing scope")
            }
            Self::AssignedBeforeNonlocal => {
                format!(
                    "`{name}` was assigned in the current scope before the nonlocal declaration"
                )
            }
            Self::GlobalScope => {
                format!("Found `{name}`, but it was not the global scope")
            }
            Self::AssignedBeforeGlobal => {
                format!("`{name}` was assigned in the current scope before the global declaration")
            }
        }
    }
}

#[derive(PartialEq, Eq)]
pub enum MutableCaptureLookupKind {
    /// Look up a name in a `global` statement
    Global,
    /// Look up a name in a `nonlocal` statement
    Nonlocal,
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

    /// Given the name of a function def, return a new `Idx<KeyFunction>` at which
    /// we will store the result of binding it along with an optional `Idx<Key>` at which
    /// we have the binding for the TypeInfo of any preceding function def of the same name.
    ///
    /// An invariant is that the caller must store a binding for the returned
    /// `Idx<KeyFunction>`; failure to do so will lead to a dangling Idx and
    /// a panic at solve time.
    ///
    /// Function bindings are unusual because the `@overload` decorator causes bindings
    /// that would normally be unrelated in control flow to become tied together.
    ///
    /// As a result, when we create a Idx<KeyFunction> for binding a function def, we
    /// will want to track any pre-existing binding associated with the same name and
    /// link the bindings together.
    pub fn create_function_index(
        &mut self,
        function_identifier: &Identifier,
    ) -> (Idx<KeyFunction>, Option<Idx<Key>>) {
        // Get the index of both the `Key` and `KeyFunction` for the preceding function definition, if any
        let (pred_idx, pred_function_idx) = match self
            .scopes
            .function_predecessor_indices(&function_identifier.id)
        {
            Some((pred_idx, pred_function_idx)) => (Some(pred_idx), Some(pred_function_idx)),
            None => (None, None),
        };
        // Create the Idx<KeyFunction> at which we'll store the def we are ready to bind now.
        // The caller *must* eventually store a binding for it.
        let function_idx =
            self.idx_for_promise(KeyFunction(ShortIdentifier::new(function_identifier)));
        // If we found a previous def, we store a forward reference inside its `BindingFunction`.
        if let Some(pred_function_idx) = pred_function_idx {
            self.table
                .link_predecessor_function(pred_function_idx, function_idx);
        }
        (function_idx, pred_idx)
    }

    pub fn init_static_scope(&mut self, x: &[Stmt], top_level: bool) {
        let current = self.scopes.current_mut();
        current.stat.stmts(
            x,
            &self.module_info,
            top_level,
            self.lookup,
            self.sys_info,
            |x| {
                self.table
                    .annotations
                    .0
                    .insert(KeyAnnotation::Annotation(x))
            },
        );
        // Presize the flow, as its likely to need as much space as static
        current.flow.info.reserve(current.stat.0.capacity());
    }

    pub fn stmts(&mut self, xs: Vec<Stmt>) {
        for x in xs {
            self.stmt(x);
        }
    }

    fn inject_globals(&mut self) {
        for global in Global::globals(self.has_docstring) {
            let key = Key::Global(global.name().clone());
            let idx = self.table.insert(key, Binding::Global(global.clone()));
            self.bind_key(global.name(), idx, FlowStyle::Other);
        }
    }

    fn inject_builtins(&mut self) {
        let builtins_module = ModuleName::builtins();
        match self.lookup.get(builtins_module) {
            Ok(builtins_export) => {
                for name in builtins_export.wildcard(self.lookup).iter() {
                    let key = Key::Import(name.clone(), TextRange::default());
                    let idx = self
                        .table
                        .insert(key, Binding::Import(builtins_module, name.clone(), None));
                    self.bind_key(name, idx, FlowStyle::Import(builtins_module, name.clone()));
                }
            }
            Err(err @ FindError::NotFound(..)) => {
                let (ctx, msg) = err.display();
                self.error_multiline(
                    TextRange::default(),
                    ErrorKind::InternalError,
                    ctx.as_deref(),
                    msg,
                );
            }
            Err(FindError::Ignored | FindError::NoPyTyped | FindError::NoSource(_)) => (),
        }
    }

    // Only works for things with `Foo`, or `source.Foo`, or `F` where `from module import Foo as F`.
    // Does not work for things with nested modules - but no SpecialExport's have that.
    pub fn as_special_export(&self, e: &Expr) -> Option<SpecialExport> {
        match e {
            Expr::Name(name) => {
                self.scopes
                    .as_special_export(&name.id, None, self.module_info.name())
            }
            Expr::Attribute(ExprAttribute {
                value, attr: name, ..
            }) if let Expr::Name(base_name) = &**value => self.scopes.as_special_export(
                &name.id,
                Some(&base_name.id),
                self.module_info.name(),
            ),
            _ => None,
        }
    }

    pub fn error(
        &self,
        range: TextRange,
        error_kind: ErrorKind,
        context: Option<&dyn Fn() -> ErrorContext>,
        msg: String,
    ) {
        self.errors.add(range, error_kind, context, vec1![msg]);
    }

    pub fn error_multiline(
        &self,
        range: TextRange,
        error_kind: ErrorKind,
        context: Option<&dyn Fn() -> ErrorContext>,
        msg: Vec1<String>,
    ) {
        self.errors.add(range, error_kind, context, msg);
    }

    pub fn lookup_mutable_captured_name(
        &mut self,
        name: &Name,
        kind: MutableCaptureLookupKind,
    ) -> Result<Idx<Key>, MutableCaptureLookupError> {
        let name = Hashed::new(name);
        let mut barrier = false;
        let allow_nonlocal_reference = kind == MutableCaptureLookupKind::Nonlocal;
        let allow_global_reference = kind == MutableCaptureLookupKind::Global;
        let mut result = Err(MutableCaptureLookupError::NotFound);
        // If there is static info for the name in the current scope and this value is not None
        // set the `annot` field to this value
        let mut static_annot_override = None;
        for (idx, scope) in self.scopes.iter_rev().enumerate() {
            let in_current_scope = idx == 0;
            let valid_nonlocal_reference = allow_nonlocal_reference
                && !in_current_scope
                && !matches!(scope.kind, ScopeKind::Module | ScopeKind::Class(_));
            let valid_global_reference = allow_global_reference
                && !in_current_scope
                && matches!(scope.kind, ScopeKind::Module);
            if scope.flow.info.get_hashed(name).is_some() {
                match kind {
                    MutableCaptureLookupKind::Nonlocal => {
                        if in_current_scope {
                            // If there's a flow type for the name in the current scope
                            // it must have been assigned before
                            return Err(MutableCaptureLookupError::AssignedBeforeNonlocal);
                        }
                    }
                    MutableCaptureLookupKind::Global => {
                        if in_current_scope {
                            // If there's a flow type for the name in the current scope
                            // it must have been assigned before
                            return Err(MutableCaptureLookupError::AssignedBeforeGlobal);
                        }
                    }
                }
            }
            if !matches!(scope.kind, ScopeKind::Class(_))
                && let Some(info) = scope.stat.0.get_hashed(name)
            {
                match kind {
                    MutableCaptureLookupKind::Nonlocal => {
                        if valid_nonlocal_reference {
                            let key = info.as_key(name.into_key());
                            result = Ok(self.table.types.0.insert(key));
                            // We can't return immediately, because we need to override
                            // the static annotation in the current scope with the one we found
                            static_annot_override = info.annot;
                            break;
                        } else if !in_current_scope {
                            return Err(MutableCaptureLookupError::NonlocalScope);
                        }
                    }
                    MutableCaptureLookupKind::Global => {
                        if valid_global_reference {
                            let key = info.as_key(name.into_key());
                            result = Ok(self.table.types.0.insert(key));
                            // We can't return immediately, because we need to override
                            // the static annotation in the current scope with the one we found
                            static_annot_override = info.annot;
                            break;
                        } else if !in_current_scope {
                            return Err(MutableCaptureLookupError::GlobalScope);
                        }
                    }
                }
            }
            barrier = barrier || scope.barrier;
        }
        self.scopes
            .set_annotation_for_mutable_capture(name, static_annot_override);
        result
    }

    pub fn lookup_name(
        &mut self,
        name: Hashed<&Name>,
        kind: LookupKind,
    ) -> Result<Idx<Key>, LookupError> {
        self.lookup_name_impl(name, kind, &mut Usage::StaticTypeInformation)
    }

    pub fn lookup_name_usage(
        &mut self,
        name: Hashed<&Name>,
        usage: &mut Usage,
    ) -> Result<Idx<Key>, LookupError> {
        self.lookup_name_impl(name, LookupKind::Regular, usage)
    }

    fn lookup_name_impl(
        &mut self,
        name: Hashed<&Name>,
        kind: LookupKind,
        usage: &mut Usage,
    ) -> Result<Idx<Key>, LookupError> {
        self.lookup_name_inner(name, kind, usage)
            .map(|(result, first_use)| {
                if let Some(used_idx) = first_use {
                    self.record_possible_first_use(used_idx, usage);
                }
                result
            })
    }

    /// Helper function, needed to work around the borrow checker given heavy use of mutable refs.
    ///
    /// When lookup succeeds, returns a pair `idx, maybe_first_use`, where `maybe_first_use`
    /// is an option of a possible first-use `(used_idx)` to track for deterministic
    /// type inference.
    fn lookup_name_inner(
        &mut self,
        name: Hashed<&Name>,
        kind: LookupKind,
        usage: &mut Usage,
    ) -> Result<(Idx<Key>, Option<Idx<Key>>), LookupError> {
        let mut barrier = false;
        let ok_no_usage = |idx| Ok((idx, None));
        for scope in self.scopes.iter_rev() {
            if let Some(flow) = scope.flow.info.get_hashed(name) {
                if !barrier {
                    let (idx, maybe_pinned_idx) = self.detect_possible_first_use(flow.key, usage);
                    if let Some(pinned_idx) = maybe_pinned_idx {
                        return Ok((idx, Some(pinned_idx)));
                    } else {
                        return ok_no_usage(idx);
                    }
                }
            }
            if !matches!(scope.kind, ScopeKind::Class(_))
                && let Some(info) = scope.stat.0.get_hashed(name)
            {
                match kind {
                    LookupKind::Regular => {
                        let key = info.as_key(name.into_key());
                        return ok_no_usage(self.table.types.0.insert(key));
                    }
                    LookupKind::Mutable => {
                        if barrier {
                            return Err(LookupError::NotMutable);
                        }
                    }
                }
            }
            barrier = barrier || scope.barrier;
        }
        Err(LookupError::NotFound)
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
    fn detect_possible_first_use(
        &self,
        flow_idx: Idx<Key>,
        usage: &mut Usage,
    ) -> (Idx<Key>, Option<Idx<Key>>) {
        match self.table.types.1.get(flow_idx) {
            Some(Binding::Pin(unpinned_idx, FirstUse::Undetermined)) => match usage {
                Usage::StaticTypeInformation | Usage::Narrowing => (flow_idx, Some(flow_idx)),
                Usage::CurrentIdx(..) => (*unpinned_idx, Some(flow_idx)),
            },
            Some(Binding::Pin(unpinned_idx, first_use)) => match first_use {
                FirstUse::DoesNotPin => (flow_idx, None),
                FirstUse::Undetermined => match usage {
                    Usage::StaticTypeInformation | Usage::Narrowing => (flow_idx, Some(flow_idx)),
                    Usage::CurrentIdx(..) => (*unpinned_idx, Some(flow_idx)),
                },
                FirstUse::UsedBy(usage_idx) => {
                    // Detect secondary reads of the same name from a first use, and make
                    // sure they all use the raw binding rather than the `Pin`.
                    let currently_in_first_use = match usage {
                        Usage::CurrentIdx(idx, ..) => idx == usage_idx,
                        Usage::Narrowing | Usage::StaticTypeInformation => false,
                    };
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
    fn record_possible_first_use(&mut self, used: Idx<Key>, usage: &mut Usage) {
        match self.table.types.1.get_mut(used) {
            Some(Binding::Pin(.., first_use @ FirstUse::Undetermined)) => {
                *first_use = match usage {
                    Usage::CurrentIdx(use_idx, first_uses_of) => {
                        first_uses_of.insert(used);
                        FirstUse::UsedBy(*use_idx)
                    }
                    Usage::StaticTypeInformation | Usage::Narrowing => FirstUse::DoesNotPin,
                };
            }
            b => {
                unreachable!("Expected a Binding::Pin needing first use, got {:?}", b)
            }
        }
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
        name: &Identifier,
    ) -> Either<Idx<KeyLegacyTypeParam>, Option<Idx<Key>>> {
        let found = self
            .lookup_name(Hashed::new(&name.id), LookupKind::Regular)
            .ok();
        if let Some(idx) = found {
            match self.lookup_legacy_tparam_from_idx(name, idx) {
                Some(left) => Either::Left(left),
                None => Either::Right(Some(idx)),
            }
        } else {
            Either::Right(None)
        }
    }

    /// Perform the inner loop of looking up a possible legacy type parameter, given a starting
    /// binding. The loop follows `Forward` nodes backward, and returns:
    /// - Some(...) if we find either a legacy type variable or an import (in which case it *might*
    ///   be a legacy type variable, so we'll let the solve stage decide)
    /// - None if we find somethign that is definitely not a legacy type variable.
    fn lookup_legacy_tparam_from_idx(
        &mut self,
        name: &Identifier,
        mut idx: Idx<Key>,
    ) -> Option<Idx<KeyLegacyTypeParam>> {
        // We are happy to follow some forward bindings, but it's possible to have a cycle of such bindings.
        // Therefore we arbitrarily cut off at 100 forward hops.
        for _ in 1..100 {
            if let Some(b) = self.table.types.1.get(idx) {
                match b {
                    Binding::Forward(fwd_idx) => {
                        idx = *fwd_idx;
                    }
                    Binding::TypeVar(..) | Binding::ParamSpec(..) | Binding::TypeVarTuple(..) => {
                        return Some(self.insert_binding(
                            KeyLegacyTypeParam(ShortIdentifier::new(name)),
                            BindingLegacyTypeParam(idx),
                        ));
                    }
                    Binding::Import(..) => {
                        // TODO: We need to recursively look through imports to determine
                        // whether it is a legacy type parameter. We can't simply walk through
                        // bindings, because we could recursively reach ourselves, resulting in
                        // a deadlock.
                        return Some(self.insert_binding(
                            KeyLegacyTypeParam(ShortIdentifier::new(name)),
                            BindingLegacyTypeParam(idx),
                        ));
                    }
                    _ => {
                        // If we hit anything other than a type variable, an import, or a Forward,
                        // then we know this name does not point at a type variable
                        return None;
                    }
                }
            } else {
                // This case happens if the name is associated with a promised binding
                // that is not yet in the table. I'm fuzzy when exactly this occurs, but
                // such names cannot point at legacy type variables.
                //
                // TODO(stroxler): it would be nice to have an actual example here, but I am
                // still not sure when exactly it happens.
                return None;
            }
        }
        None
    }

    pub fn bind_definition(
        &mut self,
        name: &Identifier,
        binding: Binding,
        style: FlowStyle,
    ) -> Option<Idx<KeyAnnotation>> {
        let idx = self.insert_binding(Key::Definition(ShortIdentifier::new(name)), binding);
        self.bind_key(&name.id, idx, style).0
    }

    pub fn bind_definition_current(
        &mut self,
        name: &Identifier,
        current: CurrentIdx,
        binding: Binding,
        style: FlowStyle,
    ) -> Option<Idx<KeyAnnotation>> {
        let idx = self.insert_binding_current(current, binding);
        self.bind_key(&name.id, idx, style).0
    }

    pub fn bind_current(
        &mut self,
        name: &Name,
        current: &CurrentIdx,
        style: FlowStyle,
    ) -> (Option<Idx<KeyAnnotation>>, Option<Idx<Key>>) {
        self.bind_key(name, current.idx(), style)
    }

    /// Return a pair of:
    /// 1. The annotation that should be used at the moment, if one was provided.
    /// 2. The default that should be used if you are in a loop.
    pub fn bind_key(
        &mut self,
        name: &Name,
        idx: Idx<Key>,
        style: FlowStyle,
    ) -> (Option<Idx<KeyAnnotation>>, Option<Idx<Key>>) {
        let name = Hashed::new(name);
        let default = self.scopes.update_flow_info(name, idx, Some(style));
        let info = self
            .scopes
            .current()
            .stat
            .0
            .get_hashed(name)
            .unwrap_or_else(|| {
                let module = self.module_info.name();
                panic!("Name `{name}` not found in static scope of module `{module}`")
            });
        if info.count > 1 {
            self.table
                .insert_anywhere(name.into_key().clone(), info.loc, idx);
        }
        (info.annot, default)
    }

    pub fn type_params(&mut self, x: &mut TypeParams) {
        for x in x.type_params.iter_mut() {
            let name = x.name().clone();
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
            self.scopes.add_to_current_static(
                name.id.clone(),
                name.range,
                SymbolKind::TypeParameter,
                None,
            );
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
    }

    pub fn add_loop_exitpoint(&mut self, exit: LoopExit, range: TextRange) {
        let in_loop = self.scopes.add_loop_exitpoint(exit);
        if !in_loop {
            // Python treats break and continue outside of a loop as a syntax error.
            self.error(
                range,
                ErrorKind::ParseError,
                None,
                format!("Cannot `{exit}` outside loop"),
            );
        }
    }

    pub fn bind_narrow_ops(&mut self, narrow_ops: &NarrowOps, use_range: TextRange) {
        for (name, (op, op_range)) in narrow_ops.0.iter_hashed() {
            if let Ok(name_key) = self.lookup_name(name, LookupKind::Regular) {
                let binding_key = self.insert_binding(
                    Key::Narrow(name.into_key().clone(), *op_range, use_range),
                    Binding::Narrow(name_key, Box::new(op.clone()), use_range),
                );
                self.scopes.update_flow_info(name, binding_key, None);
            }
        }
    }

    pub fn bind_lambda_param(&mut self, name: &Identifier) {
        let var = self.solver.fresh_contained(self.uniques);
        let idx = self.insert_binding(
            Key::Definition(ShortIdentifier::new(name)),
            Binding::LambdaParameter(var),
        );
        self.scopes
            .add_to_current_static(name.id.clone(), name.range, SymbolKind::Parameter, None);
        self.bind_key(&name.id, idx, FlowStyle::Other);
    }

    pub fn bind_function_param(
        &mut self,
        target: AnnotationTarget,
        x: AnyParameterRef,
        class_key: Option<Idx<KeyClass>>,
    ) {
        let name = x.name();
        let annot = x.annotation().map(|x| {
            self.insert_binding(
                KeyAnnotation::Annotation(ShortIdentifier::new(name)),
                BindingAnnotation::AnnotateExpr(target.clone(), x.clone(), class_key),
            )
        });
        let (annot, def) = match annot {
            Some(annot) => (annot, Either::Left(annot)),
            None => {
                let var = self.solver.fresh_contained(self.uniques);
                let annot = self.insert_binding(
                    KeyAnnotation::Annotation(ShortIdentifier::new(name)),
                    BindingAnnotation::Type(target.clone(), var.to_type()),
                );
                (annot, Either::Right(var))
            }
        };
        let key = self.insert_binding(
            Key::Definition(ShortIdentifier::new(name)),
            Binding::FunctionParameter(def),
        );
        self.scopes.add_to_current_static(
            name.id.clone(),
            name.range,
            SymbolKind::Parameter,
            Some(annot),
        );
        self.bind_key(&name.id, key, FlowStyle::Other);
    }

    /// Helper for loops, inserts a phi key for every name in the given flow.
    fn insert_phi_keys(&mut self, mut flow: Flow, range: TextRange) -> Flow {
        for (name, info) in flow.info.iter_mut() {
            // The promise is that we will insert a Phi binding when the control flow merges.
            info.key = self.idx_for_promise(Key::Phi(name.clone(), range));
        }
        flow
    }

    pub fn setup_loop(&mut self, range: TextRange, narrow_ops: &NarrowOps) {
        let base = mem::take(&mut self.scopes.current_mut().flow);
        // To account for possible assignments to existing names in a loop, we
        // speculatively insert phi keys upfront.
        self.scopes.current_mut().flow = self.insert_phi_keys(base.clone(), range);
        self.scopes
            .current_mut()
            .loops
            .push(Loop(vec![(LoopExit::NeverRan, base)]));
        self.bind_narrow_ops(narrow_ops, range);
    }

    pub fn teardown_loop(&mut self, range: TextRange, narrow_ops: &NarrowOps, orelse: Vec<Stmt>) {
        let done = self.scopes.finish_current_loop();
        let (breaks, other_exits): (Vec<Flow>, Vec<Flow>) =
            done.0.into_iter().partition_map(|(exit, flow)| match exit {
                LoopExit::Break => Either::Left(flow),
                LoopExit::NeverRan | LoopExit::Continue => Either::Right(flow),
            });
        // We associate a range to the non-`break` exits from the loop; it doesn't matter much what
        // it is as long as it's different from the loop's range.
        let other_range = TextRange::new(range.start(), range.start());
        if breaks.is_empty() {
            // When there are no `break`s, the loop condition is always false once the body has exited,
            // and any `orelse` always runs.
            self.merge_loop_into_current(other_exits, range);
            self.bind_narrow_ops(&narrow_ops.negate(), other_range);
            self.stmts(orelse);
        } else {
            // Otherwise, we negate the loop condition and run the `orelse` only when we don't `break`.
            self.merge_loop_into_current(other_exits, range);
            self.bind_narrow_ops(&narrow_ops.negate(), other_range);
            self.stmts(orelse);
            self.merge_loop_into_current(breaks, other_range);
        }
    }

    fn merge_flow(&mut self, mut xs: Vec<Flow>, range: TextRange, is_loop: bool) -> Flow {
        if xs.len() == 1 && xs[0].has_terminated {
            return xs.pop().unwrap();
        }
        let (hidden_branches, mut visible_branches): (Vec<_>, Vec<_>) =
            xs.into_iter().partition(|x| x.has_terminated);

        // We normally go through the visible branches, but if nothing is visible no one is going to
        // fill in the Phi keys we promised. So just give up and use the hidden branches instead.
        let no_next = visible_branches.is_empty();
        if visible_branches.is_empty() {
            visible_branches = hidden_branches;
        }

        // Collect all the information that we care about from all branches
        let mut names: SmallMap<Name, (Idx<Key>, Idx<Key>, SmallSet<Idx<Key>>, Vec<FlowStyle>)> =
            SmallMap::with_capacity(visible_branches.first().map_or(0, |x| x.info.len()));
        let visible_branches_len = visible_branches.len();
        for flow in visible_branches {
            for (name, info) in flow.info.into_iter_hashed() {
                let f = |v: &mut (Idx<Key>, Idx<Key>, SmallSet<Idx<Key>>, Vec<FlowStyle>)| {
                    if info.key != v.0 {
                        // Optimization: instead of x = phi(x, ...), we can skip the x.
                        // Avoids a recursive solving step later.
                        v.2.insert(info.key);
                    }
                    v.3.push(info.style);
                };

                match names.entry_hashed(name) {
                    Entry::Occupied(mut e) => f(e.get_mut()),
                    Entry::Vacant(e) => {
                        // The promise is that the next block will create a binding for all names in `names`.
                        //
                        // Note that in some cases (e.g. variables defined above a loop) we already promised
                        // a binding and this lookup will just give us back the same `Idx<Key::Phi(...)>` we
                        // created initially.
                        let key = self.idx_for_promise(Key::Phi(e.key().clone(), range));
                        f(e.insert((
                            key,
                            info.default,
                            SmallSet::new(),
                            Vec::with_capacity(visible_branches_len),
                        )));
                    }
                };
            }
        }

        let mut res = SmallMap::with_capacity(names.len());
        for (name, (key, default, values, styles)) in names.into_iter_hashed() {
            let style = FlowStyle::merged(styles);
            self.insert_binding_idx(
                key,
                match () {
                    _ if values.len() == 1 => Binding::Forward(values.into_iter().next().unwrap()),
                    _ if is_loop => Binding::Default(default, Box::new(Binding::Phi(values))),
                    _ => Binding::Phi(values),
                },
            );
            res.insert_hashed(
                name,
                FlowInfo {
                    key,
                    default: if self.scopes.loop_depth() > 0 {
                        default
                    } else {
                        key
                    },
                    style,
                },
            );
        }
        Flow {
            info: res,
            has_terminated: no_next,
        }
    }

    fn merge_into_current(&mut self, mut branches: Vec<Flow>, range: TextRange, is_loop: bool) {
        branches.push(mem::take(&mut self.scopes.current_mut().flow));
        self.scopes.current_mut().flow = self.merge_flow(branches, range, is_loop);
    }

    fn merge_loop_into_current(&mut self, branches: Vec<Flow>, range: TextRange) {
        self.merge_into_current(branches, range, true);
    }

    pub fn merge_branches_into_current(&mut self, branches: Vec<Flow>, range: TextRange) {
        self.merge_into_current(branches, range, false);
    }

    pub fn set_current_flow_to_merged_branches(&mut self, branches: Vec<Flow>, range: TextRange) {
        let flow = self.merge_flow(branches, range, false);
        self.scopes.replace_current_flow(flow);
    }
}

/// Handle intercepting names inside either function parameter/return
/// annotations or base class lists of classes, in order to check whether they
/// point at type variable declarations and need to be converted to type
/// parameters.
pub struct LegacyTParamBuilder {
    /// All of the names used. Each one may or may not point at a type variable
    /// and therefore bind a legacy type parameter.
    legacy_tparams: SmallMap<Name, Either<(Identifier, Idx<KeyLegacyTypeParam>), Option<Idx<Key>>>>,
    /// Are there scoped type parameters? Used to control downstream errors.
    has_scoped_tparams: bool,
}

impl LegacyTParamBuilder {
    pub fn new(has_scoped_tparams: bool) -> Self {
        Self {
            legacy_tparams: SmallMap::new(),
            has_scoped_tparams,
        }
    }

    /// Perform a lookup of a name used in either base classes of a class or
    /// parameter/return annotations of a function.
    ///
    /// We have a special "intercepted" lookup to create bindings that allow us
    /// to later determine whether this name points at a type variable
    /// declaration, in which case we intercept it to treat it as a type
    /// parameter in the current scope.
    pub fn intercept_lookup(
        &mut self,
        builder: &mut BindingsBuilder,
        name: &Identifier,
    ) -> Option<Binding> {
        let result = self
            .legacy_tparams
            .entry(name.id.clone())
            .or_insert_with(|| {
                builder
                    .lookup_legacy_tparam(name)
                    .map_left(|idx| (name.clone(), idx))
            });
        match result {
            Either::Left((_, idx)) => {
                let range_if_scoped_params_exist = if self.has_scoped_tparams {
                    Some(name.range())
                } else {
                    None
                };
                Some(Binding::CheckLegacyTypeParam(
                    *idx,
                    range_if_scoped_params_exist,
                ))
            }
            Either::Right(idx) => idx.map(Binding::Forward),
        }
    }

    /// Add `Definition` bindings to a class or function body scope for all the names
    /// referenced in the function parameter/return annotations or the class bases.
    ///
    /// We do this so that AnswersSolver has the opportunity to determine whether any
    /// of those names point at legacy (pre-PEP-695) type variable declarations, in which
    /// case the name should be treated as a Quantified type parameter inside this scope.
    pub fn add_name_definitions(&self, builder: &mut BindingsBuilder) {
        for entry in self.legacy_tparams.values() {
            if let Either::Left((name, idx)) = entry {
                builder.scopes.add_to_current_static(
                    name.id.clone(),
                    name.range,
                    SymbolKind::TypeParameter,
                    None,
                );
                builder.bind_definition(
                    name,
                    // Note: we use None as the range here because the range is
                    // used to error if legacy tparams are mixed with scope
                    // tparams, and we only want to do that once (which we do in
                    // the binding created by `forward_lookup`).
                    Binding::CheckLegacyTypeParam(*idx, None),
                    FlowStyle::Other,
                );
            }
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
            .filter_map(|x| x.as_ref().left().as_ref().map(|(_, idx)| *idx))
            .collect()
    }
}
