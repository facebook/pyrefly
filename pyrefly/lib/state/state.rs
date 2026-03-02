/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::any::Any;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::hash_map::Entry;
use std::fmt::Display;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::mem;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::MutexGuard;
use std::sync::RwLockReadGuard;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;

use dupe::Dupe;
use dupe::OptionDupedExt;
use enum_iterator::Sequence;
use itertools::Itertools;
use pyrefly_build::handle::Handle;
use pyrefly_python::module::Module;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_types::type_alias::TypeAliasIndex;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::events::CategorizedEvents;
use pyrefly_util::fs_anyhow;
use pyrefly_util::lock::Mutex;
use pyrefly_util::lock::RwLock;
use pyrefly_util::locked_map::LockedMap;
use pyrefly_util::no_hash::BuildNoHash;
use pyrefly_util::prelude::VecExt;
use pyrefly_util::task_heap::CancellationHandle;
use pyrefly_util::task_heap::Cancelled;
use pyrefly_util::task_heap::TaskHeap;
use pyrefly_util::telemetry::TelemetryEvent;
use pyrefly_util::telemetry::TelemetryTransactionStats;
use pyrefly_util::thread_pool::ThreadPool;
use pyrefly_util::uniques::UniqueFactory;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::Hashed;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use tracing::debug;
use tracing::info;
use vec1::vec1;
use web_time::Instant;

use crate::alt::answers::AnswerEntry;
use crate::alt::answers::AnswerTable;
use crate::alt::answers::Answers;
use crate::alt::answers::LookupAnswer;
use crate::alt::answers::Solutions;
use crate::alt::answers::SolutionsEntry;
use crate::alt::answers::SolutionsTable;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::answers_solver::CalcId;
use crate::alt::answers_solver::ThreadState;
use crate::alt::traits::Solve;
use crate::binding::binding::AnyExportedKey;
use crate::binding::binding::Exported;
use crate::binding::binding::KeyAbstractClassCheck;
use crate::binding::binding::KeyClassBaseType;
use crate::binding::binding::KeyClassField;
use crate::binding::binding::KeyClassMetadata;
use crate::binding::binding::KeyClassMro;
use crate::binding::binding::KeyClassSynthesizedFields;
use crate::binding::binding::KeyExport;
use crate::binding::binding::KeyTParams;
use crate::binding::binding::KeyVariance;
use crate::binding::binding::Keyed;
use crate::binding::bindings::BindingEntry;
use crate::binding::bindings::BindingTable;
use crate::binding::bindings::Bindings;
use crate::binding::table::TableKeyed;
use crate::config::config::ConfigFile;
use crate::config::error_kind::ErrorKind;
use crate::config::finder::ConfigError;
use crate::config::finder::ConfigFinder;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorInfo;
use crate::export::exports::Export;
use crate::export::exports::ExportLocation;
use crate::export::exports::Exports;
use crate::export::exports::LookupExport;
use crate::export::special::SpecialExport;
use crate::module::bundled::BundledStub;
use crate::module::finder::find_import_prefixes;
use crate::module::typeshed::BundledTypeshedStdlib;
use crate::solver::solver::VarRecurser;
use crate::state::dirty::AtomicDirty;
use crate::state::epoch::Epoch;
use crate::state::errors::Errors;
use crate::state::load::FileContents;
use crate::state::load::Load;
use crate::state::loader::FindingOrError;
use crate::state::loader::LoaderFindCache;
use crate::state::memory::MemoryFiles;
use crate::state::memory::MemoryFilesLookup;
use crate::state::memory::MemoryFilesOverlay;
use crate::state::module::CleanGuard;
use crate::state::module::ModuleState;
use crate::state::module::ModuleStateMut;
use crate::state::module::ModuleStateReader;
use crate::state::require::Require;
use crate::state::require::RequireLevels;
use crate::state::steps::Context;
use crate::state::steps::Step;
use crate::state::steps::StepsMut;
use crate::state::subscriber::Subscriber;
use crate::types::callable::Deprecation;
use crate::types::class::Class;
use crate::types::class::ClassDefIndex;
use crate::types::stdlib::Stdlib;
use crate::types::types::TParams;
use crate::types::types::Type;

/// Tracks fine-grained dependency on a single exported name.
///
/// Presence in a `ModuleDeps::names` map implies we depend on the name's existence;
/// if the name is added or removed, we should be invalidated regardless of the
/// `metadata` and `type_` flags. The flags below control additional dependencies.
///
/// In a `ModuleChanges::names` map, default NameDep (both flags false) means
/// the name's existence changed (added/removed). Flags indicate type/metadata
/// changed without existence changing.
#[derive(Debug, Clone, Default)]
pub struct NameDep {
    /// Depend on metadata (deprecation, docstring)?
    pub metadata: bool,
    /// Depend on the type of the export?
    pub type_: bool,
}

/// Per-module dependency tracking for fine-grained incremental invalidation.
/// Represents what an rdep depends on.
#[derive(Debug, Clone, Default)]
pub struct ModuleDeps {
    /// Per-name dependencies. Presence implies existence dependency.
    pub names: SmallMap<Name, NameDep>,
    /// Do we depend on the wildcard export set?
    pub wildcard: bool,
    /// Which classes do we depend on?
    pub classes: SmallSet<ClassDefIndex>,
    /// Which type aliases do we depend on?
    pub type_aliases: SmallSet<TypeAliasIndex>,
}

/// Per-module change tracking. Represents what changed in a module's exports.
///
/// Uses the same underlying fields as `ModuleDeps`, but with different semantics
/// for `NameDep`: default (both flags false) means existence changed (name
/// added/removed); flags indicate type/metadata changed without existence changing.
#[derive(Debug, Clone, Default)]
pub struct ModuleChanges(pub ModuleDeps);

// A single dependency, passed during lookup. Can be merged into ModuleDeps.
pub enum ModuleDep {
    // Depend on the existence of a module
    Exists,
    // Depend on the TypeEq result of an exported key
    Key(AnyExportedKey),
    // Depend on the existence of an exported name, not necessarily it's type
    // Currently unused, but we should use this in LookupExport
    #[allow(unused)]
    NameExists(Name),
    // Depend on metadata (deprecation, docstring) of an exported name
    NameMetadata(Name),
    // Depend on the set of wildcard exported names
    Wildcard,
}

impl ModuleChanges {
    /// Record that a key's value changed (key still exists).
    /// For name keys, sets `type_ = true` to indicate a type-level change.
    pub fn add_key(&mut self, key: AnyExportedKey) {
        self.0.add_key(key);
    }

    /// Record that a key was added or removed (existence change).
    /// For name keys, uses default NameDep (both flags false) to denote
    /// existence-level change. For classes/type aliases, equivalent to add_key.
    pub fn add_key_existence(&mut self, key: AnyExportedKey) {
        match key {
            AnyExportedKey::KeyExport(k) => {
                self.0.names.entry(k.0).or_default();
            }
            // Classes and type aliases don't distinguish between existence and change.
            _ => self.add_key(key),
        }
    }

    /// Merge another `ModuleChanges` into this one, mutating in place.
    pub fn merge(&mut self, other: ModuleChanges) {
        self.0.merge(other.0);
    }

    /// Returns true if no changes were recorded.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Check if two change sets overlap, for cycle detection.
    ///
    /// This is symmetric: both sides are change sets. Two changes overlap if
    /// they affect the same name/class/type_alias. An existence change (default
    /// NameDep) overlaps with any change on the same name, since it's strictly
    /// more impactful than a type/metadata-only change.
    pub fn overlaps(&self, other: &ModuleChanges) -> bool {
        if self.0.wildcard || other.0.wildcard {
            return true;
        }
        for (name, self_dep) in &self.0.names {
            if let Some(other_dep) = other.0.names.get(name) {
                // Either side is an existence change — overlaps with any
                // change on the same name.
                if (!self_dep.type_ && !self_dep.metadata)
                    || (!other_dep.type_ && !other_dep.metadata)
                {
                    return true;
                }
                // Both sides have specific flags. Check flag overlap.
                if (self_dep.type_ && other_dep.type_) || (self_dep.metadata && other_dep.metadata)
                {
                    return true;
                }
            }
        }
        if self.0.classes.iter().any(|c| other.0.classes.contains(c)) {
            return true;
        }
        self.0
            .type_aliases
            .iter()
            .any(|t| other.0.type_aliases.contains(t))
    }
}

impl ModuleDeps {
    /// Record a dependency on an exported key.
    /// For name keys, sets `type_ = true` (depends on the type of the export).
    fn add_key(&mut self, key: AnyExportedKey) {
        match key {
            AnyExportedKey::KeyExport(k) => {
                self.names.entry(k.0).or_default().type_ = true;
            }
            AnyExportedKey::KeyTypeAlias(k) => {
                self.type_aliases.insert(k.0);
            }
            AnyExportedKey::KeyTParams(KeyTParams(c))
            | AnyExportedKey::KeyClassBaseType(KeyClassBaseType(c))
            | AnyExportedKey::KeyClassField(KeyClassField(c, _))
            | AnyExportedKey::KeyClassSynthesizedFields(KeyClassSynthesizedFields(c))
            | AnyExportedKey::KeyVariance(KeyVariance(c))
            | AnyExportedKey::KeyClassMetadata(KeyClassMetadata(c))
            | AnyExportedKey::KeyClassMro(KeyClassMro(c))
            | AnyExportedKey::KeyAbstractClassCheck(KeyAbstractClassCheck(c)) => {
                self.classes.insert(c);
            }
        }
    }

    pub fn add_dep(&mut self, dep: ModuleDep) {
        match dep {
            ModuleDep::Exists => {}
            ModuleDep::Key(key) => self.add_key(key),
            ModuleDep::NameExists(name) => {
                self.names.entry(name).or_default();
            }
            ModuleDep::NameMetadata(name) => {
                self.names.entry(name).or_default().metadata = true;
            }
            ModuleDep::Wildcard => {
                self.wildcard = true;
            }
        }
    }

    pub fn with_dep(mut self, dep: ModuleDep) -> Self {
        self.add_dep(dep);
        self
    }

    /// Merge another `ModuleDeps` into this one, mutating in place.
    pub fn merge(&mut self, other: ModuleDeps) {
        for (name, dep) in other.names.into_iter_hashed() {
            match self.names.entry_hashed(name) {
                starlark_map::small_map::Entry::Occupied(mut e) => {
                    e.get_mut().metadata |= dep.metadata;
                    e.get_mut().type_ |= dep.type_;
                }
                starlark_map::small_map::Entry::Vacant(e) => {
                    e.insert(dep);
                }
            }
        }
        self.classes.extend(other.classes);
        self.type_aliases.extend(other.type_aliases);
        self.wildcard |= other.wildcard;
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
            && !self.wildcard
            && self.classes.is_empty()
            && self.type_aliases.is_empty()
    }

    /// Check if these dependencies are affected by the given change.
    ///
    /// `self` is the dependency set (what an rdep depends on).
    /// `change` is the change set (what changed in a module).
    ///
    /// For names, the flags have asymmetric semantics:
    /// - In a dep set, default NameDep (both flags false) = depends on existence only.
    ///   Depending on type/metadata implies depending on existence.
    /// - In a change set, default NameDep = existence changed (name added/removed).
    ///   Flags indicate type/metadata changed without existence changing.
    ///
    /// An existence change (default in change) invalidates any dep on that name.
    /// A type/metadata-only change only invalidates deps with the matching flag.
    pub fn invalidated_by(&self, changed: &ModuleChanges) -> bool {
        if self.wildcard || changed.0.wildcard {
            return true;
        }
        for (name, dep) in &self.names {
            if let Some(ch) = changed.0.names.get(name) {
                // Existence changed — invalidates any dep on this name.
                if !ch.type_ && !ch.metadata {
                    return true;
                }
                // Type/metadata-only change — check matching flags.
                if (dep.type_ && ch.type_) || (dep.metadata && ch.metadata) {
                    return true;
                }
            }
        }
        if self.classes.iter().any(|c| changed.0.classes.contains(c)) {
            return true;
        }
        self.type_aliases
            .iter()
            .any(|t| changed.0.type_aliases.contains(t))
    }
}

/// `ModuleData` is a snapshot of `ArcId<ModuleDataMut>` in the main state.
/// The snapshot is readonly most of the times. It will only be overwritten with updated information
/// from `Transaction` when we decide to commit a `Transaction` into the main state.
#[derive(Debug)]
struct ModuleData {
    handle: Handle,
    config: ArcId<ConfigFile>,
    state: ModuleState,
    imports: HashMap<ModuleName, FindingOrError<ModulePath>, BuildNoHash>,
    deps: HashMap<Handle, ModuleDeps>,
    rdeps: HashSet<Handle>,
}

#[derive(Debug)]
struct ModuleDataMut {
    handle: Handle,
    config: RwLock<ArcId<ConfigFile>>,
    state: ModuleStateMut,
    /// Import resolution cache: module names from import statements → resolved paths.
    /// Only contains deps that were resolved via `find_import`.
    imports: RwLock<HashMap<ModuleName, FindingOrError<ModulePath>, BuildNoHash>>,
    /// All forward dependencies keyed by Handle.
    /// Invariant: If deps contains h2, then h2.rdeps.contains(self.handle).
    /// To ensure atomicity, rdeps is modified while holding the deps write lock.
    deps: RwLock<HashMap<Handle, ModuleDeps>>,
    /// The reverse dependencies of this module. This is used to invalidate on change.
    /// Note that if we are only running once, e.g. on the command line, this isn't valuable.
    /// But we create it anyway for simplicity, since it doesn't seem to add much overhead.
    rdeps: Mutex<HashSet<Handle>>,
}

impl ModuleData {
    /// Make a copy of the data that can be mutated.
    fn clone_for_mutation(&self) -> ModuleDataMut {
        ModuleDataMut {
            handle: self.handle.dupe(),
            config: RwLock::new(self.config.dupe()),
            state: self.state.clone_for_mutation(),
            imports: RwLock::new(self.imports.clone()),
            deps: RwLock::new(self.deps.clone()),
            rdeps: Mutex::new(self.rdeps.clone()),
        }
    }
}

impl ModuleDataMut {
    fn new(handle: Handle, require: Require, config: ArcId<ConfigFile>, now: Epoch) -> Self {
        Self {
            handle,
            config: RwLock::new(config),
            state: ModuleStateMut::new(require, now),
            imports: Default::default(),
            deps: Default::default(),
            rdeps: Default::default(),
        }
    }

    /// Take the data out of the `ModuleDataMut`, leaving a `ModuleData`.
    /// Reusing the `ModuleDataMut` is not possible.
    fn take_and_freeze(&self) -> ModuleData {
        let ModuleDataMut {
            handle,
            config,
            state,
            imports,
            deps,
            rdeps,
        } = self;
        let imports = mem::take(&mut *imports.write());
        let deps = mem::take(&mut *deps.write());
        let rdeps = mem::take(&mut *rdeps.lock());
        let state = state.take_and_freeze();
        ModuleData {
            handle: handle.dupe(),
            config: config.read().dupe(),
            state,
            imports,
            deps,
            rdeps,
        }
    }

    /// Look up how this module depends on a specific source handle.
    /// Returns the `ModuleDep` if this module depends on `source_handle`, or `None` if not found.
    fn get_depends_on(&self, source_handle: &Handle) -> Option<ModuleDeps> {
        self.deps.read().get(source_handle).cloned()
    }
}

/// A subset of State that contains readable information for various systems (e.g. IDE, error reporting, etc).
struct StateData {
    stdlib: SmallMap<SysInfo, Arc<Stdlib>>,
    modules: HashMap<Handle, ModuleData>,
    loaders: SmallMap<ArcId<ConfigFile>, Arc<LoaderFindCache>>,
    /// The contents for ModulePath::memory values
    memory: MemoryFiles,
    /// The current epoch, gets incremented every time we recompute
    now: Epoch,
}

impl StateData {
    fn new() -> Self {
        Self {
            stdlib: Default::default(),
            modules: Default::default(),
            loaders: Default::default(),
            memory: Default::default(),
            now: Epoch::zero(),
        }
    }
}

/// `TransactionData` contains most of the information in `Transaction`, but it doesn't lock
/// the read of `State`.
/// It is used to store uncommitted transaction state in between transaction runs.
pub(crate) struct TransactionData<'a> {
    state: &'a State,
    stdlib: SmallMap<SysInfo, Arc<Stdlib>>,
    updated_modules: LockedMap<Handle, ArcId<ModuleDataMut>>,
    updated_loaders: LockedMap<ArcId<ConfigFile>, Arc<LoaderFindCache>>,
    memory_overlay: MemoryFilesOverlay,
    default_require: Require,
    /// The epoch when this transaction was created
    base: Epoch,
    /// The current epoch, gets incremented every time we recompute
    now: Epoch,
    /// Items we still need to process. Stored in a max heap, so that
    /// the highest step (the module that is closest to being finished)
    /// gets picked first, ensuring we release its memory quickly.
    todo: TaskHeap<Step, ArcId<ModuleDataMut>>,
    /// Values whose solutions changed value since the last time we recomputed
    changed: Mutex<Vec<(ArcId<ModuleDataMut>, ModuleChanges)>>,
    /// Handles which are dirty
    dirty: Mutex<SmallSet<ArcId<ModuleDataMut>>>,
    /// Thing to tell about each action.
    subscriber: Option<Box<dyn Subscriber + 'a>>,
}

impl<'a> TransactionData<'a> {
    /// Convert saved transaction data back into a full transaction. We can only restore if the
    /// underlying state is unchanged, otherwise the transaction data might make inconsistent
    /// assumptions, in particular about deps/rdeps.
    pub(crate) fn restore(self) -> Option<Transaction<'a>> {
        let start = Instant::now();
        let readable = self.state.state.read();
        let state_lock_blocked = start.elapsed();
        if self.base == readable.now {
            Some(Transaction {
                data: self,
                stats: Mutex::new(TelemetryTransactionStats {
                    state_lock_blocked,
                    ..Default::default()
                }),
                ad_hoc_solve_recorder: None,
                readable,
            })
        } else {
            None
        }
    }
}

/// `Transaction` is a collection of state that's only relevant during a type checking job.
/// Most importantly, it holds `updated_modules`, which contains module information that are copied
/// over from main state, potentially with updates as a result of recheck.
/// At the end of a check, the updated modules information can be committed back to the main `State`
/// in a transaction.
pub struct Transaction<'a> {
    data: TransactionData<'a>,
    stats: Mutex<TelemetryTransactionStats>,
    /// Optional callback that logs each ad-hoc solve event the instant it completes.
    /// When set, each call to `ad_hoc_solve` immediately invokes this recorder with the
    /// operation label, start time, and duration, rather than batching stats for later.
    ad_hoc_solve_recorder: Option<Box<dyn Fn(&'static str, Instant, Duration) + Send + Sync + 'a>>,
    readable: RwLockReadGuard<'a, StateData>,
}

impl<'a> Transaction<'a> {
    /// Drops the lock and retains just the underlying data.
    pub(crate) fn save(self, telemetry: &mut TelemetryEvent) -> TransactionData<'a> {
        let Transaction {
            data,
            stats,
            ad_hoc_solve_recorder: _,
            readable,
        } = self;
        drop(readable);
        telemetry.set_transaction_stats(stats.into_inner());
        data
    }

    pub fn set_subscriber(&mut self, subscriber: Option<Box<dyn Subscriber>>) {
        self.data.subscriber = subscriber;
    }

    /// Mark this transaction as freshly created (not restored from saved state).
    pub fn set_fresh(&mut self) {
        self.stats.lock().fresh = true;
    }

    /// Sets a callback that will be invoked immediately each time an ad-hoc solve completes,
    /// recording the operation label, start time, and duration as a telemetry event.
    pub fn set_ad_hoc_solve_recorder(
        &mut self,
        recorder: Box<dyn Fn(&'static str, Instant, Duration) + Send + Sync + 'a>,
    ) {
        self.ad_hoc_solve_recorder = Some(recorder);
    }

    pub fn get_solutions(&self, handle: &Handle) -> Option<Arc<Solutions>> {
        self.with_module_inner(handle, |x| x.get_solutions())
    }

    pub fn get_bindings(&self, handle: &Handle) -> Option<Bindings> {
        self.with_module_inner(handle, |x| x.get_answers().map(|a| a.0.dupe()))
    }

    pub fn get_answers(&self, handle: &Handle) -> Option<Arc<Answers>> {
        self.with_module_inner(handle, |x| x.get_answers().map(|a| a.1.dupe()))
    }

    pub fn get_ast(&self, handle: &Handle) -> Option<Arc<ruff_python_ast::ModModule>> {
        self.with_module_inner(handle, |x| x.get_ast())
    }

    pub fn get_config(&self, handle: &Handle) -> Option<ArcId<ConfigFile>> {
        // We ignore the ModuleState, but no worries, this is not on a critical path
        self.with_module_config_inner(handle, |c, _| Some(c.dupe()))
    }

    pub fn get_load(&self, handle: &Handle) -> Option<Arc<Load>> {
        self.with_module_inner(handle, |x| x.get_load())
    }

    pub fn get_errors<'b>(&self, handles: impl IntoIterator<Item = &'b Handle>) -> Errors {
        Errors::new(
            handles
                .into_iter()
                .filter_map(|handle| {
                    self.with_module_config_inner(handle, |config, x| {
                        Some((x.get_load()?, config.dupe()))
                    })
                })
                .collect(),
        )
    }

    pub fn get_all_errors(&self) -> Errors {
        if self.data.updated_modules.is_empty() {
            // Optimized path
            return Errors::new(
                self.readable
                    .modules
                    .values()
                    .filter_map(|x| Some((x.state.get_load()?, x.config.dupe())))
                    .collect(),
            );
        }
        let mut res = self
            .data
            .updated_modules
            .iter_unordered()
            .filter_map(|x| Some((x.1.state.get_load()?, x.1.config.read().dupe())))
            .collect::<Vec<_>>();
        for (k, v) in self.readable.modules.iter() {
            if self.data.updated_modules.get(k).is_none()
                && let Some(load) = v.state.get_load()
            {
                res.push((load, v.config.dupe()));
            }
        }
        Errors::new(res)
    }

    pub fn config_finder(&self) -> &ConfigFinder {
        &self.data.state.config_finder
    }

    /// Search through the export table of every module we know about.
    /// Searches will be performed in parallel on chunks of modules, to speed things up.
    /// The order of the resulting `Vec` is unspecified.
    pub fn search_exports<V: Send + Sync>(
        &self,
        searcher: impl Fn(&Handle, &Exports, &SmallMap<Name, ExportLocation>) -> Vec<V> + Sync,
    ) -> Vec<V> {
        // Make sure all the modules are in updated_modules.
        // We have to get a mutable module data to do the lookup we need anyway.
        for x in self.readable.modules.keys() {
            self.get_module(x);
        }

        let all_results = Mutex::new(Vec::new());

        let tasks = TaskHeap::new();
        // It's very fast to find whether a module contains an export, but the cost will
        // add up for a large codebase. Therefore, we will parallelize the work. The work is
        // distributed in the task heap above.
        // To avoid too much lock contention, we chunk the work into size of 1000 modules.
        for chunk in &self.data.updated_modules.iter_unordered().chunks(1000) {
            tasks.push((), chunk.collect_vec(), false);
        }
        self.data.state.threads.spawn_many(|| {
            tasks.work_without_cancellation(|_, modules| {
                let mut thread_local_results = Vec::new();
                for (handle, module_data) in modules {
                    let exports_data = self.lookup_export(module_data);
                    let exports = exports_data.exports(&self.lookup(module_data.dupe()));
                    thread_local_results.extend(searcher(handle, &exports_data, &exports));
                }
                if !thread_local_results.is_empty() {
                    all_results.lock().push(thread_local_results);
                }
            });
        });

        all_results.into_inner().into_iter().flatten().collect()
    }

    pub fn get_config_errors(&self) -> Vec<ConfigError> {
        self.data.state.config_finder.errors()
    }

    pub fn get_module_info(&self, handle: &Handle) -> Option<Module> {
        self.get_load(handle).map(|x| x.module_info.dupe())
    }

    /// Compute transitive dependency closure for the given handle.
    /// Note that for IDE services, if the given handle is an in-memory one, then you are probably
    /// not getting what you want, because the set of rdeps of in-memory file for IDE service will
    /// only contain itself.
    pub fn get_transitive_rdeps(&self, handle: Handle) -> HashSet<Handle> {
        let mut transitive_rdeps = HashSet::new();
        let mut work_list = vec![handle];
        loop {
            let Some(handle) = work_list.pop() else {
                break;
            };
            if !transitive_rdeps.insert(handle.dupe()) {
                continue;
            }
            for rdep in self.get_module(&handle).rdeps.lock().iter() {
                work_list.push(rdep.dupe());
            }
        }
        transitive_rdeps
    }

    /// Return all handles for which there is data, in a non-deterministic order.
    pub fn handles(&self) -> Vec<Handle> {
        if self.data.updated_modules.is_empty() {
            // Optimized path
            self.readable.modules.keys().cloned().collect()
        } else {
            let mut res = self
                .data
                .updated_modules
                .iter_unordered()
                .map(|x| x.0.clone())
                .collect::<Vec<_>>();
            for x in self.readable.modules.keys() {
                if self.data.updated_modules.get(x).is_none() {
                    res.push(x.clone());
                }
            }
            res
        }
    }

    /// Return all modules for which there is data, in a non-deterministic order.
    pub fn modules(&self) -> SmallSet<ModuleName> {
        self.readable
            .modules
            .keys()
            .map(|x| x.module())
            .chain(
                self.data
                    .updated_modules
                    .iter_unordered()
                    .map(|x| x.0.module()),
            )
            .collect()
    }

    pub fn module_count(&self) -> usize {
        let transaction = self.data.updated_modules.len();
        let base = self.readable.modules.len();
        if transaction == 0 || base == 0 {
            transaction + base
        } else {
            let mut res = transaction;
            for x in self.readable.modules.keys() {
                if self.data.updated_modules.get(x).is_none() {
                    res += 1;
                }
            }
            res
        }
    }

    /// Computes line count split between user-owned and dependency modules.
    /// Returns (user_lines, dependency_lines).
    pub fn split_line_count(&self, user_handles: &HashSet<&Handle>) -> (usize, usize) {
        let mut user_lines = 0;
        let mut dep_lines = 0;

        if self.data.updated_modules.is_empty() {
            for (handle, module) in self.readable.modules.iter() {
                let lines = module.state.line_count();
                if user_handles.contains(handle) {
                    user_lines += lines;
                } else {
                    dep_lines += lines;
                }
            }
        } else {
            for (handle, module) in self.data.updated_modules.iter_unordered() {
                let lines = module.state.line_count();
                if user_handles.contains(handle) {
                    user_lines += lines;
                } else {
                    dep_lines += lines;
                }
            }

            for (handle, module) in self.readable.modules.iter() {
                if self.data.updated_modules.get(handle).is_none() {
                    let lines = module.state.line_count();
                    if user_handles.contains(handle) {
                        user_lines += lines;
                    } else {
                        dep_lines += lines;
                    }
                }
            }
        }

        (user_lines, dep_lines)
    }

    /// Create a handle for import `module` within the handle `handle`
    pub fn import_handle(
        &self,
        handle: &Handle,
        module: ModuleName,
        path: Option<&ModulePath>,
    ) -> FindingOrError<Handle> {
        let path = match path {
            Some(path) => FindingOrError::new_finding(path.dupe()),
            None => self
                .get_cached_loader(&self.get_module(handle).config.read())
                .find_import(module, Some(handle.path())),
        };
        path.map(|path| Handle::new(module, path, handle.sys_info().dupe()))
    }

    /// Create a handle for import `module` within the handle `handle`, preferring `.py` over `.pyi`
    pub fn import_handle_prefer_executable(
        &self,
        handle: &Handle,
        module: ModuleName,
        path: Option<&ModulePath>,
    ) -> FindingOrError<Handle> {
        let path = match path {
            Some(path) => FindingOrError::new_finding(path.dupe()),
            None => self
                .get_cached_loader(&self.get_module(handle).config.read())
                .find_import_prefer_executable(module, Some(handle.path())),
        };
        path.map(|path| Handle::new(module, path, handle.sys_info().dupe()))
    }

    /// Create a handle for import `module` within the handle `handle`
    pub fn import_prefixes(&self, handle: &Handle, module: ModuleName) -> Vec<ModuleName> {
        find_import_prefixes(&self.get_module(handle).config.read(), module)
    }

    fn clean(&self, module_data: &ArcId<ModuleDataMut>, guard: CleanGuard) {
        // We need to clean up the state.
        // If things have changed, we need to update the last_step.
        // We clear memory as an optimisation only.

        let dirty = guard.take_dirty();

        // Helper: rebuild and clear imports/deps/rdeps.
        let rebuild = |clear_ast: bool| {
            if let Some(subscriber) = &self.data.subscriber {
                subscriber.start_work(&module_data.handle);
            }
            let mut imports_lock = module_data.imports.write();
            let mut deps_lock = module_data.deps.write();
            let _imports = mem::take(&mut *imports_lock);
            let deps = mem::take(&mut *deps_lock);
            guard.rebuild(clear_ast, self.data.now);
            for dep_handle in deps.keys() {
                let removed = self
                    .get_module(dep_handle)
                    .rdeps
                    .lock()
                    .remove(&module_data.handle);
                assert!(removed);
            }
            // Hold both locks until after rdeps are updated
            drop(deps_lock);
            drop(imports_lock);
        };

        if dirty.require() {
            // We have increased the `Require` level, so redo everything to make sure
            // we capture everything.
            // Could be optimized to do less work (e.g. if you had Retain::Error before don't need to reload)
            guard.store_load(None);
            rebuild(true);
            return;
        }

        // Validate the load flag.
        if dirty.load()
            && let Some(old_load) = guard.get_load()
        {
            let (file_contents, self_error) =
                Load::load_from_path(module_data.handle.path(), &self.memory_lookup());
            if self_error.is_some()
                || match &file_contents {
                    FileContents::Source(code) => {
                        old_load.module_info.is_notebook()
                            || code.as_str() != old_load.module_info.contents().as_str()
                    }
                    FileContents::Notebook(notebook) => {
                        if let Some(old_notebook) = old_load.module_info.notebook() {
                            **notebook != *old_notebook
                        } else {
                            true
                        }
                    }
                }
            {
                guard.store_load(Some(Arc::new(Load::load_from_data(
                    module_data.handle.module(),
                    module_data.handle.path().dupe(),
                    old_load.errors.style(),
                    file_contents,
                    self_error,
                ))));
                rebuild(true);
                return;
            }
        }

        // The contents are the same, so we can just reuse the old load contents. But errors could have changed from deps.
        if dirty.deps()
            && let Some(old_load) = guard.get_load()
        {
            guard.store_load(Some(Arc::new(Load {
                errors: ErrorCollector::new(old_load.module_info.dupe(), old_load.errors.style()),
                module_info: old_load.module_info.clone(),
            })));
            rebuild(false);
            return;
        }

        // Validate the find flag.
        if dirty.find() {
            let loader = self.get_cached_loader(&module_data.config.read());
            let mut is_dirty = false;

            // Only check imports (not all deps), since only import-statement deps
            // need `find_import` re-validation. The cached value is the same type
            // returned by `find_import`, so we just compare for equality.
            for (module_name, cached) in module_data.imports.read().iter() {
                let fresh = loader.find_import(*module_name, Some(module_data.handle.path()));
                if *cached != fresh {
                    is_dirty = true;
                    break;
                }
            }

            if is_dirty {
                // Create new ErrorCollector to clear old errors from the previous config
                if let Some(old_load) = guard.get_load() {
                    guard.store_load(Some(Arc::new(Load {
                        errors: ErrorCollector::new(
                            old_load.module_info.dupe(),
                            old_load.errors.style(),
                        ),
                        module_info: old_load.module_info.clone(),
                    })));
                }
                rebuild(false);
                return;
            }
        }

        // The module was not dirty.
        guard.finish_clean(self.data.now);
    }

    /// Try to mark a module as dirty due to dependency changes.
    /// Returns true if the module was newly marked dirty.
    fn try_mark_module_dirty(
        &self,
        module_data: &ArcId<ModuleDataMut>,
        dirtied: &mut Vec<ArcId<ModuleDataMut>>,
    ) -> bool {
        let marked = module_data.state.try_mark_deps_dirty(self.data.now);
        if marked {
            dirtied.push(module_data.dupe());
        }
        marked
    }

    /// Compute a module up to the given step, performing single-level fine-grained
    /// invalidation of direct dependents when exports change.
    ///
    /// When a module's exports change during the Solutions step, this function
    /// invalidates only those direct rdeps that import the specific names that changed.
    /// This is the normal incremental path. For transitive invalidation (used when
    /// mutable dependency cycles are detected), see `invalidate_rdeps`.
    fn demand(&self, module_data: &ArcId<ModuleDataMut>, step: Step) {
        let mut computed = false;
        loop {
            // Check if the module has been cleaned in this epoch.
            if !module_data.state.is_checked(self.data.now) {
                if let Some(guard) = module_data.state.try_start_clean() {
                    self.clean(module_data, guard);
                    // We might have done some cleaning
                    computed = true;
                }
                continue;
            }

            // Check if next step needs computing.
            let todo = match module_data.state.next_step() {
                Some(todo) if todo <= step => todo,
                _ => break,
            };

            // Try to acquire exclusive compute access for this step.
            let guard = match module_data.state.try_start_compute(todo) {
                Some(guard) => guard,
                None => {
                    // Another thread is computing this step; check again.
                    continue;
                }
            };

            // Re-check next_step under exclusive access, since another thread
            // may have completed the step between our check and lock acquisition.
            let todo = match guard.next_step() {
                Some(todo) if todo <= step => todo,
                _ => break,
            };

            computed = true;
            let require = guard.require();
            let stdlib = self.get_stdlib(&module_data.handle);
            let config = module_data.config.read();
            let ctx = Context {
                require,
                module: module_data.handle.module(),
                path: module_data.handle.path(),
                sys_info: module_data.handle.sys_info(),
                memory: &self.memory_lookup(),
                uniques: &self.data.state.uniques,
                stdlib: &stdlib,
                lookup: &self.lookup(module_data.dupe()),
                untyped_def_behavior: config
                    .untyped_def_behavior(module_data.handle.path().as_path()),
                infer_with_first_use: config
                    .infer_with_first_use(module_data.handle.path().as_path()),
                tensor_shapes: config.tensor_shapes(module_data.handle.path().as_path()),
                strict_callable_subtyping: config
                    .strict_callable_subtyping(module_data.handle.path().as_path()),
                recursion_limit_config: config.recursion_limit_config(),
            };

            guard.compute(todo, &ctx);
            {
                let mut load_result = None;
                // Compute which exports changed for fine-grained invalidation.
                // All diffing is done at the Solutions step, using old data
                // saved during reset_for_rebuild().
                let mut changed = ModuleChanges::default();
                if todo == Step::Solutions {
                    // Take old data saved during reset_for_rebuild (swap clears slot).
                    let old_exports = guard.take_old_exports();
                    let old_answers = guard.take_old_answers();
                    let old_solutions = guard.take_old_solutions();

                    // Exports diffing: compare old vs new exports.
                    if let Some(old_exp) = old_exports {
                        let new_exports = module_data
                            .state
                            .get_exports()
                            .expect("exports must exist after computing Solutions");
                        old_exp.changed_exports(&new_exports, ctx.lookup, &mut changed);
                    }

                    // Solutions diffing: compare old vs new solutions.
                    let new_solutions = module_data
                        .state
                        .get_solutions()
                        .expect("solutions must exist after computing Solutions");
                    if let Some(old_sol) = old_solutions {
                        old_sol.changed_exports(&new_solutions, &mut changed);
                    } else if let Some(old_ans) = old_answers {
                        // Old solutions were None but old exports existed — module
                        // was previously computed to Answers but not Solutions.
                        // Diff new solutions against old answers.
                        new_solutions.changed_exports_vs_answers(
                            &old_ans.0,
                            &old_ans.1,
                            &mut changed,
                        );
                    }
                }
                if !changed.is_empty() {
                    debug!(
                        "Exports changed for `{}`: {:?}",
                        module_data.handle.module(),
                        changed
                    );
                }
                if todo == Step::Answers && !require.keep_ast() {
                    // We have captured the Ast, and must have already built Exports (we do it serially),
                    // so won't need the Ast again.
                    guard.evict_ast();
                } else if todo == Step::Solutions {
                    if !require.keep_bindings() && !require.keep_answers() {
                        // From now on we can use the answers directly, so evict the bindings/answers.
                        guard.evict_answers();
                    }
                    load_result = module_data.state.get_load();
                }
                if !changed.is_empty() {
                    self.data
                        .changed
                        .lock()
                        .push((module_data.dupe(), changed.clone()));
                    let mut dirtied = Vec::new();
                    // We clone so we drop the lock immediately
                    let rdeps: Vec<Handle> = module_data.rdeps.lock().iter().cloned().collect();
                    for rdep_handle in rdeps.iter() {
                        let rdep_module = self.get_module(rdep_handle);
                        let should_invalidate = rdep_module
                            .get_depends_on(&module_data.handle)
                            .is_none_or(|d| d.invalidated_by(&changed));
                        if !should_invalidate {
                            continue;
                        }
                        self.try_mark_module_dirty(&rdep_module, &mut dirtied);
                    }

                    self.stats.lock().dirty_rdeps += dirtied.len();
                    self.data.dirty.lock().extend(dirtied);
                }
                if let Some(load) = load_result
                    && let Some(subscriber) = &self.data.subscriber
                {
                    subscriber.finish_work(self, &module_data.handle, &load, !changed.is_empty());
                }
            }
            if todo == step {
                break; // Fast path - avoid asking again since we just did it.
            }
        }

        // Eagerly compute the next, if we computed this one. This makes sure that all modules
        // eventually reach the "Solutions" step, where we can evict previous results to free
        // memory.
        //
        // This can also help with performance by eliminating bottlenecks. By being eager, we can
        // increase overall thread utilization. In many cases, this eager behavior means that a
        // result has already been computed when we need it. This is especially useful when imports
        // form large strongly-connected components.
        //
        // !! NOTE !!
        //
        // This eager behavior has the effect of checking all modules transitively reachable by
        // imports. To understand why, consider that computing an all solutions will demand the
        // types of all imports.
        //
        // Usually, a project only uses a small fraction of its 3rd party dependencies. In cases
        // like this, the additional cost (time + memory) of checking all transitive modules is
        // much higher than the cost of just keeping Answers around. So, we want some modules to
        // behave "eagerly" -- for the benefits described at the beginning of this comment -- and
        // some to behave "lazily" -- to avoid the pitfalls described above.
        //
        // For now, we use the "Require" level of a module to determine whether it should be eager
        // or lazy. This works because in practice we always ask for Require >= Errors for modules
        // being checked, and only use Require::Exports as the "default" require level, for files
        // reached _only_ through imports.
        //
        // However, this only works for "check" and the IDE. The latter uses the default level
        // Require::Indexing but falls back to Require::Exports as a performance optimization.
        //
        // This does not affect laziness for glean, pysa, or other "tracing" check modes. This is
        // by design, since those modes currently require all modules to have completed Solutions
        // to operate correctly.
        //
        // TODO: It would be much nicer to identify when a module is a 3rd party dependency directly
        // instead of approximating it using require levels.
        if computed
            && let Some(next) = step.next()
            && /* See "NOTE" */ module_data.state.require().compute_errors()
        {
            // For a large benchmark, LIFO is 10Gb retained, FIFO is 13Gb.
            // Perhaps we are getting to the heart of the graph with LIFO?
            self.data.todo.push_lifo(next, module_data.dupe());
        }
    }

    /// Like `get_module` but if the data isn't yet in this transaction will not copy it over.
    /// Saves copying if it is just a query.
    fn with_module_inner<R>(
        &self,
        handle: &Handle,
        f: impl FnOnce(&dyn ModuleStateReader) -> Option<R>,
    ) -> Option<R> {
        if let Some(v) = self.data.updated_modules.get(handle) {
            f(&v.state)
        } else if let Some(v) = self.readable.modules.get(handle) {
            f(&v.state)
        } else {
            None
        }
    }

    /// Like `with_module_inner`, but also gives access to the config.
    fn with_module_config_inner<R>(
        &self,
        handle: &Handle,
        f: impl FnOnce(&ArcId<ConfigFile>, &dyn ModuleStateReader) -> Option<R>,
    ) -> Option<R> {
        if let Some(v) = self.data.updated_modules.get(handle) {
            f(&v.config.read(), &v.state)
        } else if let Some(v) = self.readable.modules.get(handle) {
            f(&v.config, &v.state)
        } else {
            None
        }
    }

    fn get_module(&self, handle: &Handle) -> ArcId<ModuleDataMut> {
        self.get_module_ex(handle, self.data.default_require).0
    }

    /// Get a module discovered via an import.
    fn get_imported_module(&self, handle: &Handle) -> ArcId<ModuleDataMut> {
        self.get_module_ex(handle, self.data.default_require).0
    }

    /// Return the module, plus true if the module was newly created.
    fn get_module_ex(&self, handle: &Handle, require: Require) -> (ArcId<ModuleDataMut>, bool) {
        let mut created = false;
        let (res, inserted) = self.data.updated_modules.ensure(handle, || {
            if let Some(m) = self.readable.modules.get(handle) {
                ArcId::new(m.clone_for_mutation())
            } else {
                created = true;
                let config = self.data.state.get_config(handle);
                ArcId::new(ModuleDataMut::new(
                    handle.dupe(),
                    require,
                    config,
                    self.data.now,
                ))
            }
        });
        // Due to race conditions, we might create two ModuleDataMut, but only the first is returned.
        // Figure out if we won the race, and thus are the person who actually did the creation.
        if inserted {
            self.stats.lock().modules += 1;
            if created && let Some(subscriber) = &self.data.subscriber {
                subscriber.start_work(handle);
            }
        }
        (res.dupe(), created)
    }

    fn add_error(
        &self,
        module_data: &ArcId<ModuleDataMut>,
        range: TextRange,
        msg: String,
        kind: ErrorKind,
    ) {
        let load = module_data.state.get_load().unwrap();
        load.errors.add(range, ErrorInfo::Kind(kind), vec1![msg]);
    }

    fn lookup<'b>(&'b self, module_data: ArcId<ModuleDataMut>) -> TransactionHandle<'b> {
        TransactionHandle {
            transaction: self,
            module_data,
        }
    }

    fn lookup_stdlib(
        &self,
        handle: &Handle,
        name: &Name,
        thread_state: &ThreadState,
    ) -> Option<(Class, Arc<TParams>)> {
        let module_data = self.get_module(handle);
        if !self
            .lookup_export(&module_data)
            .exports(&self.lookup(module_data.dupe()))
            .contains_key(name)
        {
            self.add_error(
                &module_data,
                TextRange::default(),
                format!(
                    "Stdlib import failure, was expecting `{}` to contain `{name}`",
                    module_data.handle.module()
                ),
                ErrorKind::MissingModuleAttribute,
            );
            return None;
        }

        let t = self.lookup_answer(module_data.dupe(), &KeyExport(name.clone()), thread_state);
        let class = match t.as_deref() {
            Some(Type::ClassDef(cls)) => Some(cls.dupe()),
            ty => {
                self.add_error(
                    &module_data,
                    TextRange::default(),
                    format!(
                        "Did not expect non-class type `{}` for stdlib import `{}.{name}`",
                        ty.map_or_else(|| "<KeyError>".to_owned(), |t| t.to_string()),
                        module_data.handle.module()
                    ),
                    ErrorKind::MissingModuleAttribute,
                );
                None
            }
        };
        class.map(|class| {
            let tparams = match class.precomputed_tparams() {
                Some(tparams) => tparams.dupe(),
                None => self
                    .lookup_answer(module_data.dupe(), &KeyTParams(class.index()), thread_state)
                    .unwrap_or_default(),
            };
            (class, tparams)
        })
    }

    fn lookup_export(&self, module_data: &ArcId<ModuleDataMut>) -> Arc<Exports> {
        self.demand(module_data, Step::Exports);
        module_data.state.get_exports().unwrap()
    }

    /// Look up the location of an exported name in a module.
    /// Follows re-exports (ExportLocation::OtherModule) to find the original definition.
    /// Returns the module and text range where the name is defined.
    fn lookup_export_location(&self, handle: &Handle, name: &Name) -> Option<(Module, TextRange)> {
        let module_data = self.get_module(handle);
        let exports = self.lookup_export(&module_data);
        let export_map = exports.exports(&self.lookup(module_data.dupe()));

        match export_map.get(name)? {
            ExportLocation::ThisModule(export) => {
                let load = module_data.state.get_load()?;
                Some((load.module_info.dupe(), export.location))
            }
            ExportLocation::OtherModule(other_module, alias) => {
                let actual_name = alias.as_ref().unwrap_or(name);
                let loader = self.get_cached_loader(&module_data.config.read());
                let other_path = loader
                    .find_import(*other_module, Some(handle.path()))
                    .finding()?;
                let other_handle = Handle::new(*other_module, other_path, handle.sys_info().dupe());
                self.lookup_export_location(&other_handle, actual_name)
            }
        }
    }

    fn lookup_answer<'b, K: Solve<TransactionHandle<'b>> + Exported>(
        &'b self,
        module_data: ArcId<ModuleDataMut>,
        key: &K,
        thread_state: &ThreadState,
    ) -> Option<Arc<<K as Keyed>::Answer>>
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        let key = Hashed::new(key);

        // Either: We have solutions (use that), or we have answers (calculate that), or we have none (demand and try again)
        // Check; demand; check - the second check is guaranteed to work.
        for _ in 0..2 {
            if module_data.state.is_checked(self.data.now) {
                // Only use existing solutions or answers if the module data is current.
                // Otherwise, the module might be dirty and require computation.
                if let Some(solutions) = module_data.state.get_solutions()
                    && module_data.state.last_step() == Some(Step::Solutions)
                {
                    return solutions.get_hashed_opt(key).duped();
                } else if let Some(answers) = module_data.state.get_answers() {
                    // Fast path: check if the answer is already computed in the
                    // Calculation cell. This avoids duping Arcs and constructing
                    // a TransactionHandle when the value is cached.
                    if let Some(idx) = answers.0.key_to_idx_hashed_opt(key)
                        && let Some(v) = answers.1.get_idx(idx)
                    {
                        return Some(v);
                    }
                    // Slow path: need full solve_exported_key for computation.
                    let load = module_data.state.get_load().unwrap();
                    let stdlib = self.get_stdlib(&module_data.handle);
                    let lookup = self.lookup(module_data);
                    let result = answers.1.solve_exported_key(
                        &lookup,
                        &lookup,
                        &answers.0,
                        &load.errors,
                        &stdlib,
                        &self.data.state.uniques,
                        key,
                        thread_state,
                    );
                    return result;
                }
            }
            self.demand(&module_data, Step::Answers);
        }
        unreachable!("We demanded the answers, either answers or solutions should be present");
    }

    fn memory_lookup<'b>(&'b self) -> MemoryFilesLookup<'b> {
        MemoryFilesLookup::new(&self.readable.memory, &self.data.memory_overlay)
    }

    fn get_cached_loader(&self, loader: &ArcId<ConfigFile>) -> Arc<LoaderFindCache> {
        self.data
            .updated_loaders
            .ensure(loader, || match self.readable.loaders.get(loader) {
                Some(v) => v.dupe(),
                None => Arc::new(LoaderFindCache::new(loader.dupe())),
            })
            .0
            .dupe()
    }

    pub fn get_stdlib(&self, handle: &Handle) -> Arc<Stdlib> {
        if self.data.stdlib.len() == 1 {
            // Since we know our one must exist, we can shortcut
            return self.data.stdlib.first().unwrap().1.dupe();
        }

        self.data.stdlib.get(handle.sys_info()).unwrap().dupe()
    }

    fn compute_stdlib(&mut self, sys_infos: SmallSet<SysInfo>) {
        let loader = self.get_cached_loader(&BundledTypeshedStdlib::config());
        // Use defaults (disabled) for stdlib - depth limiting is for user code
        let thread_state = ThreadState::new(None);
        for k in sys_infos.into_iter_hashed() {
            self.data
                .stdlib
                .insert_hashed(k.to_owned(), Arc::new(Stdlib::for_bootstrapping()));
            let v = Arc::new(Stdlib::new(
                k.version(),
                // Existing lookup_class callback
                &|module, name| {
                    let path = loader.find_import(module, None).finding()?;
                    self.lookup_stdlib(&Handle::new(module, path, (*k).dupe()), name, &thread_state)
                },
                // New lookup_export_location callback
                &|module, name| {
                    let path = loader.find_import(module, None).finding()?;
                    let handle = Handle::new(module, path, (*k).dupe());
                    self.lookup_export_location(&handle, name)
                },
            ));
            self.data.stdlib.insert_hashed(k, v);
        }
    }

    fn work(&self) -> Result<(), Cancelled> {
        // ensure we have answers for everything, keep going until we don't discover any new modules
        self.data.todo.work(|_, x| {
            self.demand(&x, Step::last());
        })
    }

    fn run_step(&mut self, handles: &[Handle], require: Require) -> Result<(), Cancelled> {
        let run_start = Instant::now();

        self.data.now.next();
        let sys_infos = handles
            .iter()
            .map(|x| x.sys_info().dupe())
            .collect::<SmallSet<_>>();
        let stdlib_start = Instant::now();
        self.compute_stdlib(sys_infos);
        let compute_stdlib_time = stdlib_start.elapsed();
        self.stats.lock().compute_stdlib_time += compute_stdlib_time;

        let mut todo_count = 0;
        let dirty_count;
        {
            let dirty = mem::take(&mut *self.data.dirty.lock());
            dirty_count = dirty.len();
            for h in handles {
                let (m, created) = self.get_module_ex(h, require);
                let dirty_require = m.state.increase_require(require);
                if (created || dirty_require) && !dirty.contains(&m) {
                    self.data.todo.push_fifo(Step::first(), m);
                    todo_count += 1;
                }
            }
            for m in dirty {
                self.data.todo.push_fifo(Step::first(), m);
                todo_count += 1;
            }
        }

        let work_start = Instant::now();
        let cancelled = AtomicBool::new(false);
        // When the todo queue is empty, run `work()` on the calling thread instead of
        // dispatching to the shared thread pool. `spawn_many` uses rayon `scope` which
        // blocks until all spawned closures complete. If pool threads are parked inside
        // another transaction's work loop (e.g. a concurrent recheck), acquiring them
        // can take as long as the recheck itself — even when our todo queue is empty.
        if todo_count == 0 {
            cancelled.fetch_or(self.work().is_err(), Ordering::Relaxed);
        } else {
            self.data.state.threads.spawn_many(|| {
                cancelled.fetch_or(self.work().is_err(), Ordering::Relaxed);
            });
        }
        let run_work_time = work_start.elapsed();

        let mut stats = self.stats.lock();
        stats.run_steps += 1;
        stats.run_time += run_start.elapsed();
        stats.run_dirty_count += dirty_count;
        stats.run_todo_count += todo_count;
        stats.run_work_time += run_work_time;

        if cancelled.into_inner() {
            Err(Cancelled)
        } else {
            Ok(())
        }
    }

    /// Transitively invalidate all modules in the dependency chain of the changed modules.
    ///
    /// Unlike the single-level invalidation in `demand`, this follows the entire rdeps
    /// chain using a BFS worklist algorithm. Every module that transitively depends on
    /// any of the changed modules is marked dirty.
    ///
    /// This is called from `run_internal` when a mutable dependency cycle is detected
    /// (i.e., the same module changes twice in one run), as a fallback to ensure all
    /// cyclic modules reach a stable state.
    fn invalidate_rdeps(&mut self, mut follow: Vec<ArcId<ModuleDataMut>>) {
        // All modules discovered so far (to avoid revisiting).
        let mut dirty: SmallMap<Handle, ArcId<ModuleDataMut>> =
            follow.iter().map(|m| (m.handle.dupe(), m.dupe())).collect();

        while let Some(module) = follow.pop() {
            let rdeps: Vec<Handle> = module.rdeps.lock().iter().cloned().collect();

            for rdep_handle in rdeps {
                let hashed_rdep = Hashed::new(&rdep_handle);

                if dirty.contains_key_hashed(hashed_rdep) {
                    continue;
                }

                let m = self.get_module(&rdep_handle);
                dirty.insert_hashed(hashed_rdep.cloned(), m.dupe());
                follow.push(m.dupe());
            }
        }
        self.stats.lock().cycle_rdeps += dirty.len();

        let mut dirty_set: std::sync::MutexGuard<'_, SmallSet<ArcId<ModuleDataMut>>> =
            self.data.dirty.lock();
        for x in dirty.into_values() {
            x.state.dirty.set_deps();
            dirty_set.insert(x);
        }
    }

    fn run_internal(&mut self, handles: &[Handle], require: Require) -> Result<(), Cancelled> {
        let run_number = self.data.state.run_count.fetch_add(1, Ordering::SeqCst);
        // We first compute all the modules that are either new or have changed.
        // Then we repeatedly compute all the modules who depend on modules that changed.
        //
        // ## Termination Guarantee
        //
        // To ensure termination, we detect when the same export changes twice for a module.
        // A true non-terminating cycle requires the same export to keep changing:
        //   - Export X in A depends on export Y in B
        //   - Export Y in B depends on export X in A
        //   - X changes -> Y changes -> X changes -> infinite loop
        //
        // However, a module appearing twice with DIFFERENT exports is not a cycle:
        //   - Module A exports {x} due to change in dependency B
        //   - Later, A exports {y} due to change in dependency C
        //   - These are independent dependency chains that will each stabilize
        //
        // We track per-module ModuleDeps and check overlap to distinguish these
        // cases. This avoids false positives where independent exports happen to
        // be processed in the same module across different epochs.
        //
        // As a defense-in-depth measure, we also cap the total number of epochs to prevent
        // runaway computation in case of unforeseen edge cases.
        const MAX_EPOCHS: usize = 100;
        let mut seen_deps: SmallMap<ArcId<ModuleDataMut>, ModuleChanges> = SmallMap::new();

        for i in 1..=MAX_EPOCHS {
            debug!("Running epoch {i} of run {run_number}");
            self.run_step(handles, require)?;
            let changed = mem::take(&mut *self.data.changed.lock());
            if changed.is_empty() {
                return Ok(());
            }
            // Check for cycle: any module with overlapping export changes indicates
            // a mutable dependency cycle (e.g., A depends on B depends on A, and exports
            // keep oscillating).
            let has_cycle = changed.iter().any(|(module, changed_dep)| {
                seen_deps
                    .get(module)
                    .is_some_and(|seen| seen.overlaps(changed_dep))
            });

            if has_cycle {
                debug!(
                    "Mutable dependency cycle detected: overlapping export changes. \
                     Invalidating cycle."
                );
                // We are in a cycle of mutual dependencies, so give up.
                // Just invalidate everything in the cycle and recompute it all.
                // Use coarse-grained invalidation to ensure all cyclic modules reach stable state
                self.invalidate_rdeps(changed.into_map(|(m, _)| m));
                return self.run_step(handles, require);
            }

            // No cycle detected. Merge the new deps into our tracking set.
            for (module, changed_dep) in changed {
                match seen_deps.entry(module.dupe()) {
                    starlark_map::small_map::Entry::Vacant(e) => {
                        e.insert(changed_dep);
                    }
                    starlark_map::small_map::Entry::Occupied(mut e) => {
                        e.get_mut().merge(changed_dep);
                    }
                }
            }
        }
        // If we reach here, we've exceeded MAX_EPOCHS without stabilizing.
        // This should be extremely rare and indicates an unexpected edge case.
        // Force invalidation and one final run as a fallback.
        tracing::warn!(
            "Exceeded maximum epochs ({MAX_EPOCHS}) without stabilizing. \
             This may indicate an unexpected dependency pattern. Forcing invalidation."
        );
        let changed = mem::take(&mut *self.data.changed.lock());
        self.invalidate_rdeps(changed.into_map(|(m, _)| m));
        self.run_step(handles, require)
    }

    pub fn run(&mut self, handles: &[Handle], require: Require) {
        let _ = self.run_internal(handles, require);
    }

    pub(crate) fn ad_hoc_solve<R: Sized, F: FnOnce(AnswersSolver<TransactionHandle>) -> R>(
        &self,
        handle: &Handle,
        label: &'static str,
        solve: F,
    ) -> Option<R> {
        let module_data = self.get_module(handle);
        let lookup = self.lookup(module_data.dupe());
        let load = module_data.state.get_load()?;
        let answers = module_data.state.get_answers()?;
        let errors = &load.errors;
        let stdlib = self.get_stdlib(handle);
        let recurser = VarRecurser::new();
        let config = module_data.config.read();
        let thread_state = ThreadState::new(config.recursion_limit_config());
        let solver = AnswersSolver::new(
            &lookup,
            &answers.1,
            errors,
            &answers.0,
            &lookup,
            &self.data.state.uniques,
            &recurser,
            &stdlib,
            &thread_state,
            answers.1.heap(),
        );
        let start = Instant::now();
        let result = solve(solver);
        let duration = start.elapsed();
        if let Some(recorder) = &self.ad_hoc_solve_recorder {
            recorder(label, start, duration);
        }
        Some(result)
    }

    fn invalidate(&mut self, pred: impl Fn(&Handle) -> bool, dirty: impl Fn(&AtomicDirty)) {
        let mut dirty_set = self.data.dirty.lock();
        // We need to mark as dirty all those in updated_modules, and lift those in readable.modules up if they are dirty.
        // Most things in updated are also in readable, so we are likely to set them twice - but it's not too expensive.
        // Make sure we do updated first, as doing readable will cause them all to move to dirty.
        for (handle, module_data) in self.data.updated_modules.iter_unordered() {
            if pred(handle) {
                dirty(&module_data.state.dirty);
                dirty_set.insert(module_data.dupe());
            }
        }
        for handle in self.readable.modules.keys() {
            if pred(handle) {
                let module_data = self.get_module(handle);
                dirty(&module_data.state.dirty);
                dirty_set.insert(module_data.dupe());
            }
        }
    }

    /// Invalidate based on what a watcher told you.
    pub fn invalidate_events(&mut self, events: &CategorizedEvents) {
        // If any files were added or removed, we need to invalidate the find step.
        if !events.created.is_empty() || !events.removed.is_empty() || !events.unknown.is_empty() {
            self.invalidate_find();
        }

        // Any files that change need to be invalidated
        let files = events.iter().cloned().collect::<Vec<_>>();
        self.invalidate_disk(&files);

        // If any config files changed, we need to invalidate the config step.
        if events.iter().any(|x| {
            x.file_name()
                .and_then(|x| x.to_str())
                .is_some_and(|x| ConfigFile::CONFIG_FILE_NAMES.contains(&x))
        }) {
            self.invalidate_config();
        }
    }

    /// Called if the `find` portion of loading might have changed.
    /// E.g. you have include paths, and a new file appeared earlier on the path.
    fn invalidate_find(&mut self) {
        let new_loaders = LockedMap::new();
        for loader in self.data.updated_loaders.keys() {
            new_loaders.insert(loader.dupe(), Arc::new(LoaderFindCache::new(loader.dupe())));
        }
        for loader in self.readable.loaders.keys() {
            new_loaders.insert(loader.dupe(), Arc::new(LoaderFindCache::new(loader.dupe())));
        }
        self.data.updated_loaders = new_loaders;

        self.invalidate(|_| true, |dirty| dirty.set_find());
    }

    /// The data returned by the ConfigFinder might have changed. Note: invalidate find is not also required to run. When
    /// a config changes, this function guarantees the next transaction run will invalidate find accordingly.
    pub fn invalidate_config(&mut self) {
        // We clear the global config cache, rather than making a dedicated copy.
        // This is reasonable, because we will cache the result on ModuleData.
        self.data.state.config_finder.clear();

        // Wipe the copy of ConfigFile on each module that has changed.
        // If they change, set find to dirty.
        let mut dirty_set = self.data.dirty.lock();
        for (handle, module_data) in self.data.updated_modules.iter_unordered() {
            let config2 = self.data.state.get_config(handle);
            if config2 != *module_data.config.read() {
                *module_data.config.write() = config2;
                module_data.state.dirty.set_find();
                dirty_set.insert(module_data.dupe());
            }
        }
        for (handle, module_data) in self.readable.modules.iter() {
            if self.data.updated_modules.get(handle).is_none() {
                let config2 = self.data.state.get_config(handle);
                if module_data.config != config2 {
                    let module_data = self.get_module(handle);
                    *module_data.config.write() = config2;
                    module_data.state.dirty.set_find();
                    dirty_set.insert(module_data.dupe());
                }
            }
        }
    }

    /// Called if the `find` portion of loading might have changed for specific configs,
    /// without wanting to fully reload all configs (and pay the performance penalty of
    /// requerying a build system). If `configs` is empty, we short circuit.
    /// E.g. a file was opened or closed, changing the set of 'open' build system targets,
    /// and affecting how a go-to-definition or hover result would be produced.
    pub fn invalidate_find_for_configs(&mut self, configs: SmallSet<ArcId<ConfigFile>>) {
        if configs.is_empty() {
            return;
        }

        // First do the work of clearing out the loaders for our config, but preserve all the other
        // loaders.
        let new_loaders = LockedMap::new();
        self.data
            .updated_loaders
            .iter_unordered()
            .chain(self.readable.loaders.iter())
            .filter(|(c, _)| !configs.contains(*c))
            .for_each(|(c, l)| {
                new_loaders.insert(c.dupe(), l.dupe());
            });
        configs.iter().for_each(|config| {
            new_loaders.insert(config.dupe(), Arc::new(LoaderFindCache::new(config.dupe())));
        });
        self.data.updated_loaders = new_loaders;

        // Then mark all handles under that config as dirty.
        let mut dirty_set = self.data.dirty.lock();
        for module_data in self.data.updated_modules.values() {
            if configs.contains(&*module_data.config.read()) {
                module_data.state.dirty.set_find();
                dirty_set.insert(module_data.dupe());
            }
        }
        for (handle, module_data) in self.readable.modules.iter() {
            if self.data.updated_modules.get(handle).is_none()
                && configs.contains(&module_data.config)
            {
                let module_data = self.get_module(handle);
                module_data.state.dirty.set_find();
                dirty_set.insert(module_data.dupe());
            }
        }
    }

    /// Called if the `load_from_memory` portion of loading might have changed.
    /// Specify which in-memory files might have changed, use None to say they don't exist anymore.
    pub fn set_memory(&mut self, files: Vec<(PathBuf, Option<Arc<FileContents>>)>) {
        let mut changed = SmallSet::new();
        for (path, contents) in files {
            if self.memory_lookup().get(&path) != contents.as_ref() {
                self.data.memory_overlay.set(path.clone(), contents);
                changed.insert(ModulePath::memory(path));
            }
        }
        self.stats.lock().set_memory_dirty = changed.len();
        if changed.is_empty() {
            return;
        }
        self.invalidate(
            |handle| changed.contains(handle.path()),
            |dirty| dirty.set_load(),
        );
    }

    /// Called if the files read from the disk might have changed.
    /// Specify which files might have changed.
    /// You must use the same absolute/relative paths as were given by `find`.
    pub fn invalidate_disk(&mut self, files: &[PathBuf]) {
        if files.is_empty() {
            return;
        }
        // We create the set out of ModulePath as it allows us to reuse the fact `ModulePath` has cheap hash
        // when checking the modules.
        let files = files
            .iter()
            .map(|x| ModulePath::filesystem(x.clone()))
            .collect::<SmallSet<_>>();
        self.invalidate(
            |handle| files.contains(handle.path()),
            |dirty| dirty.set_load(),
        );
    }

    pub fn report_timings(&mut self, path: &Path) -> anyhow::Result<()> {
        let mut file = BufWriter::new(File::create(path)?);
        writeln!(file, "Module,Step,Seconds")?;
        file.flush()?;

        // Ensure all committed modules are in self.data, so we can iterate one list
        for h in self.readable.modules.keys() {
            self.get_module(h);
        }

        if let Some(subscriber) = &self.data.subscriber {
            // Start everything so we have the right size progress bar.
            for h in self.data.updated_modules.keys() {
                subscriber.start_work(h);
            }
        }
        let mut timings: SmallMap<String, f32> = SmallMap::new();
        for m in self.data.updated_modules.values() {
            let mut write = |step: &dyn Display, start: Instant| -> anyhow::Result<()> {
                let duration = start.elapsed().as_secs_f32();
                let step = step.to_string();
                writeln!(file, "{},{step},{duration}", m.handle.module())?;
                // Always flush, so if a user aborts we get the timings thus-far
                file.flush()?;
                *timings.entry(step).or_default() += duration;
                Ok(())
            };

            let m = self.get_module(&m.handle);
            let alt = StepsMut::new_loaded(m.state.get_load().unwrap());
            let require = m.state.require();
            let stdlib = self.get_stdlib(&m.handle);
            let config = m.config.read();
            let ctx = Context {
                require,
                module: m.handle.module(),
                path: m.handle.path(),
                sys_info: m.handle.sys_info(),
                memory: &self.memory_lookup(),
                uniques: &self.data.state.uniques,
                stdlib: &stdlib,
                lookup: &self.lookup(m.dupe()),
                untyped_def_behavior: config.untyped_def_behavior(m.handle.path().as_path()),
                infer_with_first_use: config.infer_with_first_use(m.handle.path().as_path()),
                tensor_shapes: config.tensor_shapes(m.handle.path().as_path()),
                strict_callable_subtyping: config
                    .strict_callable_subtyping(m.handle.path().as_path()),
                recursion_limit_config: config.recursion_limit_config(),
            };
            while let Some(step) = alt.next_step() {
                let start = Instant::now();
                alt.compute(step, &ctx);
                write(&step, start)?;
                if step == Step::Exports {
                    let start = Instant::now();
                    let exports = alt.exports.load_full().unwrap();
                    exports.wildcard(ctx.lookup);
                    exports.exports(ctx.lookup);
                    write(&"Exports-force", start)?;
                }
            }
            if let Some(subscriber) = &self.data.subscriber {
                subscriber.finish_work(self, &m.handle, &alt.load.load_full().unwrap(), false);
            }
        }
        self.data.subscriber = None; // Finalize the progress bar before printing to stderr

        fn line_key(x: &str) -> Option<(u64, &str)> {
            let (_, x) = x.rsplit_once(',')?;
            let (whole, frac) = x.split_once('.').unwrap_or((x, ""));
            Some((whole.parse::<u64>().unwrap_or(u64::MAX), frac))
        }

        // Often what the person wants is what is taking most time, so sort that way.
        // But sometimes they abort, so we can't just buffer the results in memory.
        file.flush()?;
        drop(file);
        let contents = fs_anyhow::read_to_string(path)?;
        let mut lines = contents.lines().collect::<Vec<_>>();
        lines.sort_by_cached_key(|x| line_key(x));
        lines.reverse();
        fs_anyhow::write(path, lines.join("\n") + "\n")?;

        for (step, duration) in timings {
            info!("Step {step} took {duration:.3} seconds");
        }
        Ok(())
    }

    /// Return the forward dependencies for each module in `handles`, filtered to
    /// only include dependencies that also appear in `handles`.
    /// Each entry maps a module's absolute filesystem path to the list of absolute
    /// paths of its direct dependencies. Bundled typeshed and in-memory modules are
    /// excluded.
    pub fn get_dependency_graph(&self, handles: &[Handle]) -> Vec<(PathBuf, Vec<PathBuf>)> {
        // Build a set of module names → filesystem paths for the handles the
        // caller cares about (i.e. the project files from the config globs).
        let mut included: HashMap<ModuleName, PathBuf> = HashMap::with_capacity(handles.len());
        for handle in handles {
            if let ModulePathDetails::FileSystem(path) = handle.path().details() {
                included.insert(handle.module(), path.to_path_buf());
            }
        }
        let mut graph: Vec<(PathBuf, Vec<PathBuf>)> = Vec::with_capacity(included.len());
        for handle in handles {
            if let Some(entry_path) = included.get(&handle.module()) {
                let module_data = self.get_module(handle);
                let deps: Vec<PathBuf> = module_data
                    .deps
                    .read()
                    .iter()
                    .flat_map(|(h, _)| included.get(&h.module()).cloned())
                    .collect();
                graph.push((entry_path.clone(), deps));
            }
        }
        graph
    }

    pub fn get_exports(&self, handle: &Handle) -> Arc<SmallMap<Name, ExportLocation>> {
        let module_data = self.get_module(handle);
        self.lookup_export(&module_data)
            .exports(&self.lookup(module_data))
    }

    pub(crate) fn get_exports_data(&self, handle: &Handle) -> Arc<Exports> {
        let module_data = self.get_module(handle);
        self.lookup_export(&module_data)
    }

    pub fn get_module_docstring_range(&self, handle: &Handle) -> Option<TextRange> {
        let module_data = self.get_module(handle);
        self.lookup_export(&module_data).docstring_range()
    }
}

pub(crate) struct TransactionHandle<'a> {
    transaction: &'a Transaction<'a>,
    module_data: ArcId<ModuleDataMut>,
}

impl<'a> TransactionHandle<'a> {
    fn get_module(
        &self,
        module: ModuleName,
        path: Option<&ModulePath>,
        dep: ModuleDep,
    ) -> FindingOrError<ArcId<ModuleDataMut>> {
        let handle = match path {
            Some(path) => {
                // Explicit path — already resolved. Bypass imports entirely.
                FindingOrError::new_finding(Handle::new(
                    module,
                    path.dupe(),
                    self.module_data.handle.sys_info().dupe(),
                ))
            }
            None => {
                // No path — needs find_import. Check imports cache first.
                let imports_read = self.module_data.imports.read();
                let path = match imports_read.get(&module) {
                    Some(path) => path.dupe(),
                    None => {
                        drop(imports_read);
                        let finding = self
                            .transaction
                            .get_cached_loader(&self.module_data.config.read())
                            .find_import(module, Some(self.module_data.handle.path()));
                        self.module_data
                            .imports
                            .write()
                            .insert(module, finding.dupe());
                        finding
                    }
                };
                path.map(|path| {
                    Handle::new(
                        module,
                        path.dupe(),
                        self.module_data.handle.sys_info().dupe(),
                    )
                })
            }
        };

        handle.map(|handle| {
            let res = self.transaction.get_imported_module(&handle);
            let mut deps = self.module_data.deps.write();
            match deps.entry(handle) {
                Entry::Occupied(mut e) => {
                    e.get_mut().add_dep(dep);
                }
                Entry::Vacant(e) => {
                    e.insert(ModuleDeps::default().with_dep(dep));
                    let inserted = res.rdeps.lock().insert(self.module_data.handle.dupe());
                    assert!(inserted);
                }
            }
            res
        })
    }

    /// Helper to get exports for a module with the correct lookup context.
    fn with_exports<T>(
        &self,
        module: ModuleName,
        f: impl FnOnce(&Exports, &Self) -> T,
        dep: ModuleDep,
    ) -> Option<T> {
        let module_data = self.get_module(module, None, dep).finding()?;
        let exports = self.transaction.lookup_export(&module_data);
        let lookup = TransactionHandle {
            transaction: self.transaction,
            module_data,
        };
        Some(f(&exports, &lookup))
    }
}

impl<'a> LookupExport for TransactionHandle<'a> {
    fn export_exists(&self, module: ModuleName, name: &Name) -> bool {
        // TODO: This should be ModuleDep::NameExists instead
        // but tests fail.
        let dep = ModuleDep::Key(AnyExportedKey::KeyExport(KeyExport(name.clone())));
        self.with_exports(
            module,
            |exports, lookup| exports.exports(lookup).contains_key(name),
            dep,
        )
        .unwrap_or(false)
    }

    fn get_wildcard(&self, module: ModuleName) -> Option<Arc<SmallSet<Name>>> {
        self.with_exports(
            module,
            |exports, lookup| exports.wildcard(lookup),
            ModuleDep::Wildcard,
        )
    }

    fn module_exists(&self, module: ModuleName) -> FindingOrError<()> {
        self.get_module(module, None, ModuleDep::Exists)
            .map(|module_data| {
                self.transaction.lookup_export(&module_data);
            })
    }

    fn is_submodule_imported_implicitly(&self, module: ModuleName, name: &Name) -> bool {
        self.with_exports(
            module,
            |exports, _lookup| exports.is_submodule_imported_implicitly(name),
            ModuleDep::NameMetadata(name.clone()),
        )
        .unwrap_or(false)
    }

    fn get_every_export_untracked(&self, module: ModuleName) -> Option<SmallSet<Name>> {
        self.with_exports(
            module,
            |exports, lookup| {
                exports
                    .exports(lookup)
                    .keys()
                    .cloned()
                    .collect::<SmallSet<Name>>()
            },
            ModuleDep::Exists,
        )
    }

    fn get_deprecated(&self, module: ModuleName, name: &Name) -> Option<Deprecation> {
        self.with_exports(
            module,
            |exports, lookup| match exports.exports(lookup).get(name)? {
                ExportLocation::ThisModule(Export {
                    deprecation: Some(d),
                    ..
                }) => Some(d.clone()),
                _ => None,
            },
            ModuleDep::NameMetadata(name.clone()),
        )?
    }

    fn is_reexport(&self, module: ModuleName, name: &Name) -> bool {
        self.with_exports(
            module,
            |exports, lookup| {
                matches!(
                    exports.exports(lookup).get(name),
                    Some(ExportLocation::OtherModule(..))
                )
            },
            ModuleDep::NameMetadata(name.clone()),
        )
        .unwrap_or(false)
    }

    fn is_special_export(&self, mut module: ModuleName, name: &Name) -> Option<SpecialExport> {
        let mut seen = HashSet::new();
        let mut name = name.clone();

        loop {
            if let Some(special) = SpecialExport::new(&name)
                && special.defined_in(module)
            {
                return Some(special);
            }

            if !seen.insert(module) {
                return None; // Cycle detected
            }

            let next = self.with_exports(
                module,
                |exports, lookup| match exports.exports(lookup).get(&name)? {
                    ExportLocation::ThisModule(export) => Some(Err(export.special_export)),
                    ExportLocation::OtherModule(other_module, original_name) => {
                        Some(Ok((*other_module, original_name.clone())))
                    }
                },
                ModuleDep::NameMetadata(name.clone()),
            )??;

            match next {
                Err(special) => return special,
                Ok((other_module, original_name)) => {
                    if let Some(original_name) = original_name {
                        name = original_name.clone();
                    }
                    module = other_module;
                }
            }
        }
    }

    fn docstring_range(&self, module: ModuleName, name: &Name) -> Option<TextRange> {
        self.with_exports(
            module,
            |exports, lookup| match exports.exports(lookup).get(name)? {
                ExportLocation::ThisModule(Export {
                    docstring_range, ..
                }) => *docstring_range,
                _ => None,
            },
            ModuleDep::NameMetadata(name.clone()),
        )?
    }

    fn is_final(&self, mut module: ModuleName, name: &Name) -> bool {
        let mut seen = HashSet::new();
        let mut name = name.clone();

        loop {
            if !seen.insert(module) {
                return false; // Cycle detected
            }

            let next = self.with_exports(
                module,
                |exports, lookup| match exports.exports(lookup).get(&name) {
                    Some(ExportLocation::ThisModule(Export { is_final, .. })) => Err(*is_final),
                    Some(ExportLocation::OtherModule(other_module, original_name)) => {
                        Ok((*other_module, original_name.clone()))
                    }
                    None => Err(false),
                },
                ModuleDep::NameMetadata(name.clone()),
            );

            match next {
                Some(Err(is_final)) => return is_final,
                Some(Ok((other_module, original_name))) => {
                    if let Some(original_name) = original_name {
                        name = original_name;
                    }
                    module = other_module;
                }
                None => return false,
            }
        }
    }
}

impl<'a> LookupAnswer for TransactionHandle<'a> {
    fn get<K: Solve<Self> + Exported>(
        &self,
        module: ModuleName,
        path: Option<&ModulePath>,
        k: &K,
        thread_state: &ThreadState,
    ) -> Option<Arc<K::Answer>>
    where
        AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
        SolutionsTable: TableKeyed<K, Value = SolutionsEntry<K>>,
    {
        // The unwrap is safe because we must have said there were no exports,
        // so no one can be trying to get at them
        let module_data = self
            .get_module(module, path, ModuleDep::Key(k.to_anykey()))
            .finding()
            .unwrap();
        let res = self.transaction.lookup_answer(module_data, k, thread_state);
        if res.is_none() {
            let msg = format!(
                "LookupAnswer::get failed to find key, {module} {k:?} (concurrent changes?)"
            );
            if self.transaction.data.state.run_count.load(Ordering::SeqCst) <= 1 {
                // We failed to find the key, but we are the only one running, and have never had any invalidation.
                // We should panic.
                panic!("{msg}");
            } else {
                debug!("{msg}");
            }
        }
        res
    }

    fn commit_to_module(
        &self,
        calc_id: CalcId,
        answer: Arc<dyn Any + Send + Sync>,
        errors: Option<Arc<ErrorCollector>>,
    ) -> bool {
        let CalcId(ref bindings, ref any_idx) = calc_id;
        let module = bindings.module().name();
        let path = bindings.module().path();

        // Look up the target module. Use default ModuleDep since cross-module
        // commits don't establish new dependencies.
        let module_data = match self
            .get_module(module, Some(path), ModuleDep::Exists)
            .finding()
        {
            Some(data) => data,
            None => return false,
        };

        // Access the target module's Answers and commit the answer.
        if let Some(answers_pair) = module_data.state.get_answers() {
            let answers = answers_pair.1.dupe();
            // Get the target module's error collector for error propagation.
            let target_load = module_data.state.get_load();
            let did_write = answers.commit_preliminary(any_idx, answer);
            // Only extend errors if this write won the first-write-wins race.
            if did_write && let (Some(errors), Some(target_load)) = (errors, target_load) {
                // The errors Arc should have refcount 1 here: batch_commit_scc
                // consumes the Scc (moved into the method), and each NodeState::Done
                // is destructured by the for loop, so no other references remain.
                // If this invariant is violated, something is holding an unexpected
                // reference to the error collector, which could cause silent error
                // loss and nondeterministic output.
                let errors = Arc::try_unwrap(errors).expect(
                    "cross-module batch commit: errors Arc has unexpected extra references; \
                         the SCC should have been consumed, giving us sole ownership",
                );
                target_load.errors.extend(errors);
            }
            true
        } else if module_data.state.get_solutions().is_some()
            && module_data.state.last_step() == Some(Step::Solutions)
        {
            // The target module's Answers have been evicted, but Solutions
            // exist. This is a benign race: another thread ran
            // `demand(Step::Solutions)` on the target module concurrently
            // with our SCC resolution. The sequence is:
            //
            //   1. Our thread duped the target's Arc<Answers> (in
            //      `lookup_answer`) and dropped the module state lock.
            //   2. While our thread was solving the SCC using that duped
            //      Arc, another thread acquired the exclusive lock on the
            //      target module and ran `step_solutions`, which solves
            //      all keys independently (Calculation cells allow
            //      multi-thread parallel compute via `propose_calculation`).
            //   3. After computing Solutions, `demand` evicted the Answers
            //      as a memory optimization (`steps.answers.take()`),
            //      then released the exclusive lock.
            //   4. Our batch commit now reads `steps.answers` and finds
            //      None — but the Calculation cells were already filled
            //      by the other thread's `step_solutions`, so no data is
            //      lost. Our commit is redundant and can be safely skipped.
            true
        } else {
            false
        }
    }
}

/// A checking state that will eventually commit.
/// `State` will ensure that at most one of them can exist.
pub struct CommittingTransaction<'a> {
    transaction: Transaction<'a>,
    committing_transaction_guard: MutexGuard<'a, ()>,
}

impl<'a> AsMut<Transaction<'a>> for CommittingTransaction<'a> {
    fn as_mut(&mut self) -> &mut Transaction<'a> {
        &mut self.transaction
    }
}

/// A thin wrapper around `Transaction`, so that the ability to cancel the transaction is only
/// exposed for this struct.
pub struct CancellableTransaction<'a>(Transaction<'a>);

impl CancellableTransaction<'_> {
    pub fn run(&mut self, handles: &[Handle], require: Require) -> Result<(), Cancelled> {
        self.0.run_internal(handles, require)
    }

    pub fn get_cancellation_handle(&self) -> CancellationHandle {
        self.0.data.todo.get_cancellation_handle()
    }
}

impl<'a> AsRef<Transaction<'a>> for CancellableTransaction<'a> {
    fn as_ref(&self) -> &Transaction<'a> {
        &self.0
    }
}

impl<'a> AsMut<Transaction<'a>> for CancellableTransaction<'a> {
    fn as_mut(&mut self) -> &mut Transaction<'a> {
        &mut self.0
    }
}

/// `State` coordinates between potential parallel operations over itself.
/// It enforces that
/// 1. There can be at most one ongoing recheck that can eventually commit.
/// 2. All the reads over the state are reads over a consistent view
///    (i.e. it won't observe a mix of state between different epochs),
///    which is enforced by
///
///     1. There can be as many concurrent reads over state as possible,
///        but they will block committing.
///     2. During the committing of `Transaction`, all reads will be blocked.
pub struct State {
    threads: ThreadPool,
    uniques: UniqueFactory,
    config_finder: ConfigFinder,
    state: RwLock<StateData>,
    run_count: AtomicUsize,
    committing_transaction_lock: Mutex<()>,
}

impl State {
    pub fn new(config_finder: ConfigFinder) -> Self {
        Self {
            threads: ThreadPool::new(),
            uniques: UniqueFactory::new(),
            config_finder,
            state: RwLock::new(StateData::new()),
            run_count: AtomicUsize::new(0),
            committing_transaction_lock: Mutex::new(()),
        }
    }

    pub fn config_finder(&self) -> &ConfigFinder {
        &self.config_finder
    }

    fn get_config(&self, handle: &Handle) -> ArcId<ConfigFile> {
        if matches!(
            handle.path().details(),
            ModulePathDetails::BundledTypeshed(_)
        ) {
            BundledTypeshedStdlib::config()
        } else {
            self.config_finder
                .python_file(handle.module_kind(), handle.path())
        }
    }

    pub fn new_transaction<'a>(
        &'a self,
        default_require: Require,
        subscriber: Option<Box<dyn Subscriber + 'a>>,
    ) -> Transaction<'a> {
        let start = Instant::now();
        let readable = self.state.read();
        let state_lock_blocked = start.elapsed();
        let now = readable.now;
        let stdlib = readable.stdlib.clone();
        Transaction {
            readable,
            stats: Mutex::new(TelemetryTransactionStats {
                state_lock_blocked,
                ..Default::default()
            }),
            ad_hoc_solve_recorder: None,
            data: TransactionData {
                state: self,
                stdlib,
                updated_modules: Default::default(),
                updated_loaders: Default::default(),
                memory_overlay: Default::default(),
                base: now,
                now,
                default_require,
                todo: Default::default(),
                changed: Default::default(),
                dirty: Default::default(),
                subscriber,
            },
        }
    }

    pub fn transaction<'a>(&'a self) -> Transaction<'a> {
        // IMPORTANT: the LSP depends on default_require here being Require::Exports for good
        // startup time performance.
        self.new_transaction(Require::Exports, None)
    }

    pub fn cancellable_transaction<'a>(&'a self) -> CancellableTransaction<'a> {
        CancellableTransaction(self.transaction())
    }

    pub fn new_committable_transaction<'a>(
        &'a self,
        default_require: Require,
        subscriber: Option<Box<dyn Subscriber + 'a>>,
    ) -> CommittingTransaction<'a> {
        let committing_transaction_guard = self.committing_transaction_lock.lock();
        let transaction = self.new_transaction(default_require, subscriber);
        CommittingTransaction {
            transaction,
            committing_transaction_guard,
        }
    }

    pub fn try_new_committable_transaction<'a>(
        &'a self,
        default_require: Require,
        subscriber: Option<Box<dyn Subscriber + 'a>>,
    ) -> Option<CommittingTransaction<'a>> {
        if let Some(committing_transaction_guard) = self.committing_transaction_lock.try_lock() {
            let transaction = self.new_transaction(default_require, subscriber);
            Some(CommittingTransaction {
                transaction,
                committing_transaction_guard,
            })
        } else {
            None
        }
    }

    pub fn commit_transaction(
        &self,
        transaction: CommittingTransaction,
        telemetry: Option<&mut TelemetryEvent>,
    ) {
        debug!("Committing transaction");
        let CommittingTransaction {
            transaction:
                Transaction {
                    readable,
                    stats,
                    ad_hoc_solve_recorder: _,
                    data:
                        TransactionData {
                            stdlib,
                            updated_modules,
                            updated_loaders,
                            memory_overlay,
                            base,
                            now,
                            default_require: _,
                            state: _,
                            todo: _,
                            changed: _,
                            dirty,
                            subscriber: _,
                        },
                },
            committing_transaction_guard,
        } = transaction;
        // Drop the read lock the transaction holds.
        drop(readable);

        let mut stats = stats.into_inner();
        stats.committed = true;

        // If you make a transaction dirty, e.g. by calling an invalidate method,
        // you must subsequently call `run` to drain the dirty queue.
        // We could relax this restriction by storing `dirty` in the `State`,
        // but no one wants to do this, so don't bother.
        assert!(dirty.into_inner().is_empty(), "Transaction is dirty");

        let state_lock_start = Instant::now();
        let mut state = self.state.write();
        stats.state_lock_blocked += state_lock_start.elapsed();

        if let Some(telemetry) = telemetry {
            telemetry.set_transaction_stats(stats);
        }
        assert_eq!(
            state.now, base,
            "Attempted to commit a stale transaction from epoch {:?} into state at epoch {:?}",
            base, state.now
        );
        state.stdlib = stdlib;
        state.now = now;
        for (handle, new_module_data) in updated_modules.iter_unordered() {
            state
                .modules
                .insert(handle.dupe(), new_module_data.take_and_freeze());
        }
        state.memory.apply_overlay(memory_overlay);
        for (loader_id, additional_loader) in updated_loaders.iter_unordered() {
            state
                .loaders
                .insert(loader_id.dupe(), additional_loader.dupe());
        }
        drop(committing_transaction_guard)
    }

    pub fn run(
        &self,
        handles: &[Handle],
        require: RequireLevels,
        subscriber: Option<Box<dyn Subscriber>>,
        telemetry: Option<&mut TelemetryEvent>,
    ) {
        let mut transaction = self.new_committable_transaction(require.default, subscriber);
        transaction.transaction.run(handles, require.specified);
        self.commit_transaction(transaction, telemetry);
    }

    pub fn run_with_committing_transaction(
        &self,
        mut transaction: CommittingTransaction<'_>,
        handles: &[Handle],
        require: Require,
        telemetry: Option<&mut TelemetryEvent>,
    ) {
        transaction.transaction.run(handles, require);
        self.commit_transaction(transaction, telemetry);
    }
}
