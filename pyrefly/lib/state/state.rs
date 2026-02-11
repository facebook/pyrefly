/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
use pyrefly_util::small_map1::SmallMap1;
use pyrefly_util::task_heap::CancellationHandle;
use pyrefly_util::task_heap::Cancelled;
use pyrefly_util::task_heap::TaskHeap;
use pyrefly_util::telemetry::TelemetryEvent;
use pyrefly_util::telemetry::TelemetryTransactionStats;
use pyrefly_util::thread_pool::ThreadPool;
use pyrefly_util::uniques::UniqueFactory;
use pyrefly_util::upgrade_lock::UpgradeLock;
use pyrefly_util::upgrade_lock::UpgradeLockExclusiveGuard;
use pyrefly_util::upgrade_lock::UpgradeLockWriteGuard;
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
use crate::alt::answers_solver::ThreadState;
use crate::alt::traits::Solve;
use crate::binding::binding::AnyExportedKey;
use crate::binding::binding::ChangedExport;
use crate::binding::binding::Exported;
use crate::binding::binding::KeyExport;
use crate::binding::binding::KeyTParams;
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
use crate::state::dirty::Dirty;
use crate::state::epoch::Epoch;
use crate::state::epoch::Epochs;
use crate::state::errors::Errors;
use crate::state::load::FileContents;
use crate::state::load::Load;
use crate::state::loader::FindError;
use crate::state::loader::Finding;
use crate::state::loader::FindingOrError;
use crate::state::loader::LoaderFindCache;
use crate::state::memory::MemoryFiles;
use crate::state::memory::MemoryFilesLookup;
use crate::state::memory::MemoryFilesOverlay;
use crate::state::require::Require;
use crate::state::require::RequireLevels;
use crate::state::steps::Context;
use crate::state::steps::Step;
use crate::state::steps::Steps;
use crate::state::subscriber::Subscriber;
use crate::types::callable::Deprecation;
use crate::types::class::Class;
use crate::types::class::ClassDefIndex;
use crate::types::stdlib::Stdlib;
use crate::types::types::TParams;
use crate::types::types::Type;

/// Represents which exports changed in a module for fine-grained invalidation.
#[derive(Debug, Clone)]
enum ChangedExports {
    /// No exports changed
    NoChange,
    /// Specific exports changed (either names or class indices)
    Changed(SmallSet<ChangedExport>),
    /// Invalidate all dependents (too complex to track)
    InvalidateAll,
}

/// Tracks fine-grained dependency on a single exported name.
///
/// Presence in a `ModuleDep::names` map implies we depend on the name's existence;
/// if the name is added or removed, we should be invalidated regardless of the
/// `metadata` and `type_` flags. The flags below control additional dependencies.
#[derive(Debug, Clone, Default)]
pub struct NameDep {
    /// Depend on metadata (deprecation, docstring)?
    pub metadata: bool,
    /// Depend on the type of the export?
    pub type_: bool,
}

/// Per-module dependency tracking for fine-grained incremental invalidation.
#[derive(Debug, Clone, Default)]
pub struct ModuleDep {
    /// Per-name dependencies. Presence implies existence dependency.
    pub names: SmallMap<Name, NameDep>,
    /// Do we depend on the wildcard export set?
    pub wildcard: bool,
    /// Which classes do we depend on?
    pub classes: SmallSet<ClassDefIndex>,
    /// Which type aliases do we depend on?
    pub type_aliases: SmallSet<TypeAliasIndex>,
}

impl ModuleDep {
    /// Create a dependency on a name's type.
    pub fn name_type(name: Name) -> Self {
        let mut names = SmallMap::new();
        names.insert(
            name,
            NameDep {
                metadata: false,
                type_: true,
            },
        );
        Self {
            names,
            wildcard: false,
            classes: SmallSet::new(),
            type_aliases: SmallSet::new(),
        }
    }

    /// Create a dependency on a name's metadata.
    pub fn name_metadata(name: Name) -> Self {
        let mut names = SmallMap::new();
        names.insert(
            name,
            NameDep {
                metadata: true,
                type_: false,
            },
        );
        Self {
            names,
            wildcard: false,
            classes: SmallSet::new(),
            type_aliases: SmallSet::new(),
        }
    }

    /// Create a dependency on the wildcard export set.
    pub fn wildcard_dep() -> Self {
        Self {
            names: SmallMap::new(),
            wildcard: true,
            classes: SmallSet::new(),
            type_aliases: SmallSet::new(),
        }
    }

    /// Create a dependency on a specific class.
    pub fn class_dep(class: ClassDefIndex) -> Self {
        let mut classes = SmallSet::new();
        classes.insert(class);
        Self {
            names: SmallMap::new(),
            wildcard: false,
            classes,
            type_aliases: SmallSet::new(),
        }
    }

    /// Create a dependency on a specific type alias
    pub fn type_alias_dep(type_alias: TypeAliasIndex) -> Self {
        let mut type_aliases = SmallSet::new();
        type_aliases.insert(type_alias);
        Self {
            names: SmallMap::new(),
            wildcard: false,
            classes: SmallSet::new(),
            type_aliases,
        }
    }

    /// Create a dependency with no dependencies (just module existence).
    pub fn none() -> Self {
        Self::default()
    }

    /// Merge another `ModuleDep` into this one, mutating in place.
    pub fn merge_in_place(&mut self, mut other: ModuleDep) {
        // SmallMap doesn't support drain; take ownership by replacing with an empty map
        let mut other_names = SmallMap::new();
        std::mem::swap(&mut other_names, &mut other.names);

        for (name, dep) in other_names {
            if let Some(existing) = self.names.get_mut(&name) {
                existing.metadata |= dep.metadata;
                existing.type_ |= dep.type_;
            } else {
                self.names.insert(name, dep);
            }
        }

        self.classes.extend(other.classes);
        self.type_aliases.extend(other.type_aliases);

        self.wildcard |= other.wildcard;
    }

    /// Check if this dependency should be invalidated given a set of changed exports.
    /// Returns true if any of the changed exports overlap with what this dependency imports.
    ///
    /// In this version, wildcard = true invalidates on ANY change for compatibility.
    fn should_invalidate(&self, changed_exports: &ChangedExports) -> bool {
        match changed_exports {
            ChangedExports::NoChange => false,
            ChangedExports::InvalidateAll => true,
            ChangedExports::Changed(changed) => {
                // If wildcard is set, invalidate on any change (same as old DependsOn::All)
                if self.wildcard {
                    return true;
                }
                changed.iter().any(|change| self.matches_change(change))
            }
        }
    }

    /// Compute which exports to propagate to this dependent based on what changed.
    ///
    /// This uses the same matching logic as `should_invalidate` but filters the set
    /// rather than short-circuiting. We keep them separate because `should_invalidate`
    /// can return early (better for the common case), while this must compute the
    /// full filtered set for transitive propagation.
    fn propagate_exports(&self, changed_exports: &ChangedExports) -> ChangedExports {
        match changed_exports {
            ChangedExports::NoChange => ChangedExports::NoChange,
            ChangedExports::InvalidateAll => ChangedExports::InvalidateAll,
            ChangedExports::Changed(changed) => {
                // If wildcard is set, propagate all changes (same as old DependsOn::All)
                if self.wildcard {
                    return ChangedExports::Changed(changed.clone());
                }
                let propagated: SmallSet<ChangedExport> = changed
                    .iter()
                    .filter(|change| self.matches_change(change))
                    .cloned()
                    .collect();
                if propagated.is_empty() {
                    ChangedExports::NoChange
                } else {
                    ChangedExports::Changed(propagated)
                }
            }
        }
    }

    /// Check if a single changed export matches this dependency.
    /// Used by both `should_invalidate` and `propagate_exports`.
    fn matches_change(&self, change: &ChangedExport) -> bool {
        match change {
            ChangedExport::Name(name) => self.names.get(name).is_some_and(|d| d.type_),
            ChangedExport::NameExistence(name) => self.names.contains_key(name),
            ChangedExport::ClassDefIndex(idx) => self.classes.contains(idx),
            ChangedExport::TypeAliasIndex(idx) => self.type_aliases.contains(idx),
            ChangedExport::Metadata(name) => self.names.get(name).is_some_and(|d| d.metadata),
        }
    }
}

impl AnyExportedKey {
    /// Convert this exported key to a `ModuleDep`.
    /// `KeyExport` maps to a name type dependency, all class-related keys map to class dependencies.
    pub fn to_module_dep(&self) -> ModuleDep {
        match self {
            AnyExportedKey::KeyExport(k) => ModuleDep::name_type(k.0.clone()),
            AnyExportedKey::KeyTParams(k) => ModuleDep::class_dep(k.0),
            AnyExportedKey::KeyClassBaseType(k) => ModuleDep::class_dep(k.0),
            AnyExportedKey::KeyClassField(k) => ModuleDep::class_dep(k.0),
            AnyExportedKey::KeyClassSynthesizedFields(k) => ModuleDep::class_dep(k.0),
            AnyExportedKey::KeyVariance(k) => ModuleDep::class_dep(k.0),
            AnyExportedKey::KeyClassMetadata(k) => ModuleDep::class_dep(k.0),
            AnyExportedKey::KeyClassMro(k) => ModuleDep::class_dep(k.0),
            AnyExportedKey::KeyAbstractClassCheck(k) => ModuleDep::class_dep(k.0),
            AnyExportedKey::KeyTypeAlias(k) => ModuleDep::type_alias_dep(k.0),
        }
    }
}

/// Represents a resolved or failed import.
#[derive(Debug, Clone)]
enum ImportResolution {
    /// Successfully resolved import - maps module name to handle(s) with optional dependency tracking.
    /// `None` means the import was resolved for caching only (used during Exports phase).
    /// `Some(ModuleDep)` means the import is tracked for fine-grained invalidation (used during Solutions phase).
    Resolved(SmallMap1<Handle, ModuleDep>),
    /// Failed import - stores the error for incremental invalidation.
    Failed(FindError),
}

/// `ModuleData` is a snapshot of `ArcId<ModuleDataMut>` in the main state.
/// The snapshot is readonly most of the times. It will only be overwritten with updated information
/// from `Transaction` when we decide to commit a `Transaction` into the main state.
#[derive(Debug)]
struct ModuleData {
    handle: Handle,
    config: ArcId<ConfigFile>,
    state: ModuleDataInner,
    /// The dependencies of this module, including both resolved and failed imports.
    /// Most modules exist in exactly one place, but it can be possible to load the same module multiple times with different paths.
    deps: HashMap<ModuleName, ImportResolution, BuildNoHash>,
    rdeps: HashSet<Handle>,
}

#[derive(Debug)]
struct ModuleDataMut {
    handle: Handle,
    config: RwLock<ArcId<ConfigFile>>,
    state: UpgradeLock<Step, ModuleDataInner>,
    /// The dependencies of this module, including both resolved and failed imports.
    /// Invariant: If `h1` depends on `h2` then we must have both of:
    /// data[h1].deps[h2.module] == ImportResolution::Resolved(set) where set.contains(h2)
    /// data[h2].rdeps.contains(h1)
    ///
    /// To ensure that is atomic, we always modify the rdeps while holding the deps write lock.
    deps: RwLock<HashMap<ModuleName, ImportResolution, BuildNoHash>>,
    /// The reverse dependencies of this module. This is used to invalidate on change.
    /// Note that if we are only running once, e.g. on the command line, this isn't valuable.
    /// But we create it anyway for simplicity, since it doesn't seem to add much overhead.
    rdeps: Mutex<HashSet<Handle>>,
}

/// The fields of `ModuleDataMut` that are stored together as they might be mutated.
#[derive(Debug, Clone)]
struct ModuleDataInner {
    require: Require,
    epochs: Epochs,
    dirty: Dirty,
    steps: Steps,
}

impl ModuleDataInner {
    fn new(require: Require, now: Epoch) -> Self {
        Self {
            require,
            epochs: Epochs::new(now),
            dirty: Dirty::default(),
            steps: Steps::default(),
        }
    }

    fn update_require(&mut self, require: Require) -> bool {
        let dirty = require > self.require;
        if dirty {
            self.require = require;
        }
        dirty
    }
}

impl ModuleData {
    /// Make a copy of the data that can be mutated.
    fn clone_for_mutation(&self) -> ModuleDataMut {
        ModuleDataMut {
            handle: self.handle.dupe(),
            config: RwLock::new(self.config.dupe()),
            state: UpgradeLock::new(self.state.clone()),
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
            state: UpgradeLock::new(ModuleDataInner::new(require, now)),
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
            deps,
            rdeps,
        } = self;
        let deps = mem::take(&mut *deps.write());
        let rdeps = mem::take(&mut *rdeps.lock());
        let state = state.read().clone();
        ModuleData {
            handle: handle.dupe(),
            config: config.read().dupe(),
            state,
            deps,
            rdeps,
        }
    }

    /// Look up how this module depends on a specific source handle.
    /// Returns the `ModuleDep` if this module imports from `source_handle`, or `None` if not found.
    fn get_depends_on(
        &self,
        source_module: ModuleName,
        source_handle: &Handle,
    ) -> Option<ModuleDep> {
        let deps_guard = self.deps.read();
        deps_guard
            .get(&source_module)
            .and_then(|resolution| match resolution {
                ImportResolution::Resolved(handles_map) => handles_map
                    .into_iter()
                    .find(|(h, _)| *h == source_handle)
                    .map(|(_, d)| d.clone()),
                ImportResolution::Failed(_) => None,
            })
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
    changed: Mutex<Vec<(ArcId<ModuleDataMut>, ChangedExports)>>,
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
    readable: RwLockReadGuard<'a, StateData>,
}

impl<'a> Transaction<'a> {
    /// Drops the lock and retains just the underlying data.
    pub(crate) fn save(self, telemetry: &mut TelemetryEvent) -> TransactionData<'a> {
        let Transaction {
            data,
            stats,
            readable,
        } = self;
        drop(readable);
        telemetry.set_transaction_stats(stats.into_inner());
        data
    }

    pub fn set_subscriber(&mut self, subscriber: Option<Box<dyn Subscriber>>) {
        self.data.subscriber = subscriber;
    }

    pub fn get_solutions(&self, handle: &Handle) -> Option<Arc<Solutions>> {
        self.with_module_inner(handle, |x| x.steps.solutions.dupe())
    }

    pub fn get_bindings(&self, handle: &Handle) -> Option<Bindings> {
        self.with_module_inner(handle, |x| x.steps.answers.as_ref().map(|x| x.0.dupe()))
    }

    pub fn get_answers(&self, handle: &Handle) -> Option<Arc<Answers>> {
        self.with_module_inner(handle, |x| x.steps.answers.as_ref().map(|x| x.1.dupe()))
    }

    pub fn get_ast(&self, handle: &Handle) -> Option<Arc<ruff_python_ast::ModModule>> {
        self.with_module_inner(handle, |x| x.steps.ast.dupe())
    }

    pub fn get_config(&self, handle: &Handle) -> Option<ArcId<ConfigFile>> {
        // We ignore the ModuleDataInner, but no worries, this is not on a critical path
        self.with_module_config_inner(handle, |c, _| Some(c.dupe()))
    }

    pub fn get_load(&self, handle: &Handle) -> Option<Arc<Load>> {
        self.with_module_inner(handle, |x| x.steps.load.dupe())
    }

    pub fn get_errors<'b>(&self, handles: impl IntoIterator<Item = &'b Handle>) -> Errors {
        Errors::new(
            handles
                .into_iter()
                .filter_map(|handle| {
                    self.with_module_config_inner(handle, |config, x| {
                        Some((x.steps.load.dupe()?, config.dupe()))
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
                    .filter_map(|x| Some((x.state.steps.load.dupe()?, x.config.dupe())))
                    .collect(),
            );
        }
        let mut res = self
            .data
            .updated_modules
            .iter_unordered()
            .filter_map(|x| {
                Some((
                    x.1.state.read().steps.load.dupe()?,
                    x.1.config.read().dupe(),
                ))
            })
            .collect::<Vec<_>>();
        for (k, v) in self.readable.modules.iter() {
            if self.data.updated_modules.get(k).is_none()
                && let Some(load) = v.state.steps.load.dupe()
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
        searcher: impl Fn(&Handle, &SmallMap<Name, ExportLocation>) -> Vec<V> + Sync,
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
                    let exports = self
                        .lookup_export(module_data)
                        .exports(&self.lookup(module_data.dupe()));
                    thread_local_results.extend(searcher(handle, &exports));
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
                let lines = module.state.steps.line_count();
                if user_handles.contains(handle) {
                    user_lines += lines;
                } else {
                    dep_lines += lines;
                }
            }
        } else {
            for (handle, module) in self.data.updated_modules.iter_unordered() {
                let lines = module.state.read().steps.line_count();
                if user_handles.contains(handle) {
                    user_lines += lines;
                } else {
                    dep_lines += lines;
                }
            }

            for (handle, module) in self.readable.modules.iter() {
                if self.data.updated_modules.get(handle).is_none() {
                    let lines = module.state.steps.line_count();
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

    fn clean(
        &self,
        module_data: &ArcId<ModuleDataMut>,
        exclusive: UpgradeLockExclusiveGuard<Step, ModuleDataInner>,
    ) {
        // We need to clean up the state.
        // If things have changed, we need to update the last_step.
        // We clear memory as an optimisation only.

        // Mark ourselves as having completed everything.
        let finish = |w: &mut ModuleDataInner| {
            w.epochs.checked = self.data.now;
            w.dirty.clean();
        };
        // Rebuild stuff. Pass clear_ast to indicate we need to rebuild the AST, otherwise can reuse it (if present).
        let rebuild = |mut w: UpgradeLockWriteGuard<Step, ModuleDataInner>, clear_ast: bool| {
            w.steps.last_step = if clear_ast || w.steps.ast.is_none() {
                if w.steps.load.is_none() {
                    None
                } else {
                    Some(Step::Load)
                }
            } else {
                Some(Step::Ast)
            };
            if clear_ast {
                w.steps.ast = None;
            }
            // Do not clear solutions or exports, since we use them for equality.
            w.steps.answers = None;
            w.epochs.computed = self.data.now;
            if let Some(subscriber) = &self.data.subscriber {
                subscriber.start_work(&module_data.handle);
            }
            let mut deps_lock = module_data.deps.write();
            let deps = mem::take(&mut *deps_lock);
            finish(&mut w);
            if !deps.is_empty() {
                // Downgrade to exclusive, so other people can read from us, or we lock up.
                // But don't give up the lock entirely, so we don't recompute anything
                let _exclusive = w.exclusive();
                for resolution in deps.values() {
                    if let ImportResolution::Resolved(handles_map) = resolution {
                        for (dep_handle, _depends_on) in handles_map {
                            let removed = self
                                .get_module(dep_handle)
                                .rdeps
                                .lock()
                                .remove(&module_data.handle);
                            assert!(removed);
                        }
                    }
                }
            }
            // Make sure we hold deps write lock while mutating rdeps
            drop(deps_lock);
        };

        if exclusive.dirty.require {
            // We have increased the `Require` level, so redo everything to make sure
            // we capture everything.
            // Could be optimized to do less work (e.g. if you had Retain::Error before don't need to reload)
            let mut write = exclusive.write();
            write.steps.load = None;
            rebuild(write, true);
            return;
        }

        // Validate the load flag.
        if exclusive.dirty.load
            && let Some(old_load) = exclusive.steps.load.dupe()
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
                let mut write = exclusive.write();
                write.steps.load = Some(Arc::new(Load::load_from_data(
                    module_data.handle.module(),
                    module_data.handle.path().dupe(),
                    old_load.errors.style(),
                    file_contents,
                    self_error,
                )));
                rebuild(write, true);
                return;
            }
        }

        // The contents are the same, so we can just reuse the old load contents. But errors could have changed from deps.
        if exclusive.dirty.deps
            && let Some(old_load) = exclusive.steps.load.dupe()
        {
            let mut write = exclusive.write();
            write.steps.load = Some(Arc::new(Load {
                errors: ErrorCollector::new(old_load.module_info.dupe(), old_load.errors.style()),
                module_info: old_load.module_info.clone(),
            }));
            rebuild(write, false);
            return;
        }

        // Validate the find flag.
        if exclusive.dirty.find {
            let loader = self.get_cached_loader(&module_data.config.read());
            let mut is_dirty = false;

            // Check dependencies for changes
            for (module_name, resolution) in module_data.deps.read().iter() {
                match resolution {
                    ImportResolution::Resolved(handles) => {
                        // Check if any of the resolved imports have changed
                        for (dependency_handle, _) in handles {
                            match loader.find_import(
                                dependency_handle.module(),
                                Some(module_data.handle.path()),
                            ) {
                                FindingOrError::Finding(path)
                                    if &path.finding == dependency_handle.path() => {}
                                _ => {
                                    is_dirty = true;
                                    break;
                                }
                            }
                        }
                    }
                    ImportResolution::Failed(import_failure) => {
                        // Check if failed imports now succeed or have different errors
                        match loader.find_import(*module_name, Some(module_data.handle.path())) {
                            // If we can now resolve an import, we need to rebuild
                            FindingOrError::Finding(_) => {
                                is_dirty = true;
                            }
                            // If the error changes, we need to rebuild
                            FindingOrError::Error(error) if error != *import_failure => {
                                is_dirty = true;
                            }
                            FindingOrError::Error(_) => {}
                        }
                    }
                }
                if is_dirty {
                    break;
                }
            }

            if is_dirty {
                let mut write = exclusive.write();
                // Create new ErrorCollector to clear old errors from the previous config
                if let Some(old_load) = write.steps.load.dupe() {
                    write.steps.load = Some(Arc::new(Load {
                        errors: ErrorCollector::new(
                            old_load.module_info.dupe(),
                            old_load.errors.style(),
                        ),
                        module_info: old_load.module_info.clone(),
                    }));
                }
                rebuild(write, false);
                return;
            }
        }

        // The module was not dirty. Make sure our dependencies aren't dirty either.
        let mut write = exclusive.write();
        finish(&mut write);
    }

    /// Try to mark a module as dirty due to dependency changes.
    /// Returns true if the module was newly marked dirty.
    fn try_mark_module_dirty(
        &self,
        module_data: &ArcId<ModuleDataMut>,
        dirtied: &mut Vec<ArcId<ModuleDataMut>>,
    ) -> bool {
        loop {
            let reader = module_data.state.read();
            if reader.epochs.computed == self.data.now || reader.dirty.deps {
                // Either doesn't need setting, or already set
                return false;
            }
            // This can potentially race with `clean`, so make sure we use the `last` as our exclusive key,
            // which importantly is a different key to the `first` that `clean` uses.
            // Slight risk of a busy-loop, but better than a deadlock.
            if let Some(exclusive) = reader.exclusive(Step::last()) {
                if exclusive.epochs.computed == self.data.now || exclusive.dirty.deps {
                    return false;
                }
                dirtied.push(module_data.dupe());
                exclusive.write().dirty.deps = true;
                return true;
            }
            // continue around the loop - failed to get the lock, but we really want it
        }
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
            let reader = module_data.state.read();
            if reader.epochs.checked != self.data.now {
                if let Some(ex) = reader.exclusive(Step::first()) {
                    self.clean(module_data, ex);
                    // We might have done some cleaning
                    computed = true;
                }
                continue;
            }

            let todo = match reader.steps.next_step() {
                Some(todo) if todo <= step => todo,
                _ => break,
            };
            let exclusive = match reader.exclusive(todo) {
                Some(exclusive) => exclusive,
                None => {
                    // The world changed, we should check again
                    continue;
                }
            };
            let todo = match exclusive.steps.next_step() {
                Some(todo) if todo <= step => todo,
                _ => break,
            };

            computed = true;
            let require = exclusive.require;
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
                recursion_limit_config: config.recursion_limit_config(),
            };
            let set = todo.compute(&exclusive.steps, &ctx);
            {
                let mut to_drop_ast = None;
                let mut to_drop_answers = None;
                let mut writer = exclusive.write();
                let mut load_result = None;
                let old_solutions = if todo == Step::Solutions {
                    writer.steps.solutions.take()
                } else {
                    None
                };
                let old_exports = if todo == Step::Exports {
                    writer.steps.exports.take()
                } else {
                    None
                };
                set.0(&mut writer.steps);
                // Compute which exports changed for fine-grained invalidation.
                // Check at both the Exports step (for wildcard set changes) and
                // the Solutions step (for type changes).
                let changed_exports: ChangedExports = if todo == Step::Solutions {
                    match (old_solutions.as_ref(), writer.steps.solutions.as_ref()) {
                        (Some(old), Some(new)) => {
                            let changed = old.changed_exports(new);
                            if changed.is_empty() {
                                ChangedExports::NoChange
                            } else {
                                debug!(
                                    "Exports changed for `{}`: {:?}",
                                    module_data.handle.module(),
                                    changed
                                );
                                ChangedExports::Changed(changed)
                            }
                        }
                        (Some(_old), None) => ChangedExports::InvalidateAll, // Had solutions, now don't
                        (None, _) => ChangedExports::NoChange, // No old solutions = no change to propagate
                    }
                } else if todo == Step::Exports {
                    // Check if exports changed at the Exports step.
                    // This detects both wildcard set changes and definition name changes.
                    // Wildcard changes affect `from M import *`.
                    // Name existence changes affect `from M import name`.
                    match (old_exports.as_ref(), writer.steps.exports.as_ref()) {
                        (Some(old), Some(new)) => {
                            let mut changed_set: SmallSet<ChangedExport> = SmallSet::new();
                            // Check for definition name changes (added/removed names)
                            for name in old.changed_names(new) {
                                changed_set.insert(ChangedExport::NameExistence(name));
                            }

                            // Check for metadata changes (is_reexport, implicitly_imported_submodule, deprecation, special_export)
                            for name in old.changed_metadata_names(new) {
                                changed_set.insert(ChangedExport::Metadata(name));
                            }

                            if changed_set.is_empty() {
                                ChangedExports::NoChange
                            } else {
                                debug!(
                                    "Exports changed for `{}`: {:?}",
                                    module_data.handle.module(),
                                    changed_set
                                );
                                ChangedExports::Changed(changed_set)
                            }
                        }
                        (Some(_), None) => ChangedExports::InvalidateAll,
                        (None, _) => ChangedExports::NoChange,
                    }
                } else {
                    ChangedExports::NoChange
                };
                if todo == Step::Answers && !require.keep_ast() {
                    // We have captured the Ast, and must have already built Exports (we do it serially),
                    // so won't need the Ast again.
                    to_drop_ast = writer.steps.ast.take();
                } else if todo == Step::Solutions {
                    if !require.keep_bindings() && !require.keep_answers() {
                        // From now on we can use the answers directly, so evict the bindings/answers.
                        to_drop_answers = writer.steps.answers.take();
                    }
                    load_result = writer.steps.load.dupe();
                }
                drop(writer);
                // Release the lock before dropping
                drop(to_drop_ast);
                drop(to_drop_answers);
                if !matches!(changed_exports, ChangedExports::NoChange) {
                    self.data
                        .changed
                        .lock()
                        .push((module_data.dupe(), changed_exports.clone()));
                    let mut dirtied = Vec::new();
                    // We clone so we drop the lock immediately
                    let rdeps: Vec<Handle> = module_data.rdeps.lock().iter().cloned().collect();
                    let our_module = module_data.handle.module();
                    for rdep_handle in rdeps.iter() {
                        let rdep_module = self.get_module(rdep_handle);
                        let should_invalidate = rdep_module
                            .get_depends_on(our_module, &module_data.handle)
                            .is_none_or(|d| d.should_invalidate(&changed_exports));
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
                    subscriber.finish_work(
                        self,
                        &module_data.handle,
                        &load,
                        !matches!(changed_exports, ChangedExports::NoChange),
                    );
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
            && /* See "NOTE" */ module_data.state.read().require.compute_errors()
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
        f: impl FnOnce(&ModuleDataInner) -> Option<R>,
    ) -> Option<R> {
        if let Some(v) = self.data.updated_modules.get(handle) {
            f(&v.state.read())
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
        f: impl FnOnce(&ArcId<ConfigFile>, &ModuleDataInner) -> Option<R>,
    ) -> Option<R> {
        if let Some(v) = self.data.updated_modules.get(handle) {
            f(&v.config.read(), &v.state.read())
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
        let load = module_data.state.read().steps.load.dupe().unwrap();
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

    fn lookup_export(&self, module_data: &ArcId<ModuleDataMut>) -> Exports {
        self.demand(module_data, Step::Exports);
        let lock = module_data.state.read();
        lock.steps.exports.dupe().unwrap()
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
                let load = module_data.state.read().steps.load.dupe()?;
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
            let lock = module_data.state.read();
            if lock.epochs.checked == self.data.now {
                // Only use existing solutions or answers if the module data is current.
                // Otherwise, the module might be dirty and require computation.
                if let Some(solutions) = &lock.steps.solutions
                    && lock.steps.last_step == Some(Step::Solutions)
                {
                    return solutions.get_hashed_opt(key).duped();
                } else if let Some(answers) = &lock.steps.answers {
                    let load = lock.steps.load.dupe().unwrap();
                    let answers = answers.dupe();
                    drop(lock);
                    let stdlib = self.get_stdlib(&module_data.handle);
                    let lookup = self.lookup(module_data);
                    return answers.1.solve_exported_key(
                        &lookup,
                        &lookup,
                        &answers.0,
                        &load.errors,
                        &stdlib,
                        &self.data.state.uniques,
                        key,
                        thread_state,
                    );
                }
            }
            drop(lock);
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
        self.compute_stdlib(sys_infos);

        {
            let dirty = mem::take(&mut *self.data.dirty.lock());
            for h in handles {
                let (m, created) = self.get_module_ex(h, require);
                let mut state = m.state.write(Step::first()).unwrap();
                let dirty_require = state.update_require(require);
                state.dirty.require = dirty_require || state.dirty.require;
                drop(state);
                if (created || dirty_require) && !dirty.contains(&m) {
                    self.data.todo.push_fifo(Step::first(), m);
                }
            }
            for m in dirty {
                self.data.todo.push_fifo(Step::first(), m);
            }
        }

        let cancelled = AtomicBool::new(false);
        self.data.state.threads.spawn_many(|| {
            cancelled.fetch_or(self.work().is_err(), Ordering::Relaxed);
        });

        let mut stats = self.stats.lock();
        stats.run_steps += 1;
        stats.run_time += run_start.elapsed();

        if cancelled.into_inner() {
            Err(Cancelled)
        } else {
            Ok(())
        }
    }

    /// Transitively invalidate all modules in the dependency chain of the changed modules.
    ///
    /// Unlike the single-level invalidation in `demand`, this follows the entire rdeps
    /// chain using a worklist algorithm. It propagates changed export names through the
    /// dependency graph, only invalidating modules that import (directly or transitively)
    /// the names that changed.
    ///
    /// This is called from `run_internal` when a mutable dependency cycle is detected
    /// (i.e., the same module changes twice in one run), as a fallback to ensure all
    /// cyclic modules reach a stable state.
    fn invalidate_rdeps(&mut self, changed: &[(ArcId<ModuleDataMut>, ChangedExports)]) {
        let mut changed_exports: SmallMap<Handle, ChangedExports> = changed
            .iter()
            .map(|(m, exports)| (m.handle.dupe(), exports.clone()))
            .collect();

        // Those that I have yet to follow
        let mut follow: Vec<(Handle, ChangedExports)> = changed
            .iter()
            .map(|(m, exports)| (m.handle.dupe(), exports.clone()))
            .collect();

        // Those that I know are dirty
        let mut dirty: SmallMap<Handle, ArcId<ModuleDataMut>> = changed
            .iter()
            .map(|(m, _)| (m.handle.dupe(), m.dupe()))
            .collect();

        while let Some((handle, item_changed_exports)) = follow.pop() {
            let module = self.get_module(&handle);
            let module_name = handle.module();
            let rdeps: Vec<Handle> = module.rdeps.lock().iter().cloned().collect();

            for rdep_handle in rdeps {
                let hashed_rdep = Hashed::new(&rdep_handle);

                let rdep_module = self.get_module(&rdep_handle);
                let propagated = rdep_module
                    .get_depends_on(module_name, &handle)
                    .map_or(ChangedExports::InvalidateAll, |d| {
                        d.propagate_exports(&item_changed_exports)
                    });
                if matches!(&propagated, ChangedExports::NoChange) {
                    continue; // Nothing to propagate
                }

                if dirty.contains_key_hashed(hashed_rdep) {
                    // Already marked dirty, merge the propagated names into existing
                    if let Some(existing) = changed_exports.get_mut(&rdep_handle) {
                        match (&propagated, &*existing) {
                            (ChangedExports::InvalidateAll, _) => {
                                *existing = ChangedExports::InvalidateAll
                            }
                            (_, ChangedExports::InvalidateAll) => {} // Already invalidating all
                            (ChangedExports::Changed(new), ChangedExports::Changed(old)) => {
                                let mut merged = old.clone();
                                merged.extend(new.iter().cloned());
                                *existing = ChangedExports::Changed(merged);
                            }
                            (ChangedExports::Changed(_), ChangedExports::NoChange) => {
                                *existing = propagated.clone();
                            }
                            (ChangedExports::NoChange, _) => {} // Nothing to merge
                        }
                    }
                    continue;
                }

                let m = self.get_module(&rdep_handle);
                dirty.insert_hashed(hashed_rdep.cloned(), m.dupe());
                changed_exports.insert(rdep_handle.dupe(), propagated.clone());
                follow.push((rdep_handle, propagated));
            }
        }
        self.stats.lock().cycle_rdeps += dirty.len();

        let mut dirty_set: std::sync::MutexGuard<'_, SmallSet<ArcId<ModuleDataMut>>> =
            self.data.dirty.lock();
        for x in dirty.into_values() {
            x.state.write(Step::Load).unwrap().dirty.deps = true;
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
        // We track (module, export_name) pairs rather than just modules to distinguish
        // these cases. This avoids false positives where independent exports happen to
        // be processed in the same module across different epochs.
        //
        // As a defense-in-depth measure, we also cap the total number of epochs to prevent
        // runaway computation in case of unforeseen edge cases.
        const MAX_EPOCHS: usize = 100;
        let mut seen_exports: SmallMap<ArcId<ModuleDataMut>, ChangedExports> = SmallMap::new();

        for i in 1..=MAX_EPOCHS {
            debug!("Running epoch {i} of run {run_number}");
            self.run_step(handles, require)?;
            let changed = mem::take(&mut *self.data.changed.lock());
            if changed.is_empty() {
                return Ok(());
            }
            for (module, changed_exports) in &changed {
                let dominated = match seen_exports.get(module) {
                    None => false,
                    Some(ChangedExports::InvalidateAll) => {
                        // We previously saw InvalidateAll, so any new change is dominated
                        true
                    }
                    Some(ChangedExports::NoChange) => false,
                    Some(ChangedExports::Changed(seen_names)) => {
                        // Check if the new changes overlap with previously seen exports
                        match changed_exports {
                            ChangedExports::InvalidateAll => {
                                // InvalidateAll dominates any previous specific names
                                !seen_names.is_empty()
                            }
                            ChangedExports::Changed(new_names) => {
                                new_names.iter().any(|n| seen_names.contains(n))
                            }
                            ChangedExports::NoChange => false,
                        }
                    }
                };

                if dominated {
                    debug!(
                        "Mutable dependency cycle detected: module `{}` has overlapping export changes. \
                         Previously seen: {:?}, now: {:?}. Invalidating cycle.",
                        module.handle.module(),
                        seen_exports.get(module),
                        changed_exports
                    );
                    // We are in a cycle of mutual dependencies, so give up.
                    // Just invalidate everything in the cycle and recompute it all.
                    // Use coarse-grained invalidation to ensure all cyclic modules reach stable state
                    let coarse_grained_changed: Vec<_> = changed
                        .iter()
                        .map(|(m, _)| (m.dupe(), ChangedExports::InvalidateAll))
                        .collect();
                    self.invalidate_rdeps(&coarse_grained_changed);
                    return self.run_step(handles, require);
                }

                // Merge the new exports into our tracking set
                match seen_exports.entry(module.dupe()) {
                    starlark_map::small_map::Entry::Vacant(e) => {
                        e.insert(changed_exports.clone());
                    }
                    starlark_map::small_map::Entry::Occupied(mut e) => {
                        let existing = e.get_mut();
                        match (existing, changed_exports) {
                            (ChangedExports::InvalidateAll, _) => {
                                // Already tracking all, nothing to do
                            }
                            (existing, ChangedExports::InvalidateAll) => {
                                *existing = ChangedExports::InvalidateAll;
                            }
                            (ChangedExports::NoChange, new) => {
                                *e.get_mut() = new.clone();
                            }
                            (_, ChangedExports::NoChange) => {
                                // Nothing new to add
                            }
                            (ChangedExports::Changed(seen), ChangedExports::Changed(new)) => {
                                for name in new {
                                    seen.insert(name.clone());
                                }
                            }
                        }
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
        let coarse_grained_changed: Vec<_> = changed
            .iter()
            .map(|(m, _)| (m.dupe(), ChangedExports::InvalidateAll))
            .collect();
        self.invalidate_rdeps(&coarse_grained_changed);
        self.run_step(handles, require)
    }

    pub fn run(&mut self, handles: &[Handle], require: Require) {
        let _ = self.run_internal(handles, require);
    }

    pub(crate) fn ad_hoc_solve<R: Sized, F: FnOnce(AnswersSolver<TransactionHandle>) -> R>(
        &self,
        handle: &Handle,
        solve: F,
    ) -> Option<R> {
        let module_data = self.get_module(handle);
        let lookup = self.lookup(module_data.dupe());
        let steps = &module_data.state.read().steps;
        let errors = &steps.load.as_ref()?.errors;
        let (bindings, answers) = steps.answers.as_deref().as_ref()?;
        let stdlib = self.get_stdlib(handle);
        let recurser = VarRecurser::new();
        let config = module_data.config.read();
        let thread_state = ThreadState::new(config.recursion_limit_config());
        let solver = AnswersSolver::new(
            &lookup,
            answers,
            errors,
            bindings,
            &lookup,
            &self.data.state.uniques,
            &recurser,
            &stdlib,
            &thread_state,
            answers.heap(),
        );
        let result = solve(solver);
        Some(result)
    }

    fn invalidate(&mut self, pred: impl Fn(&Handle) -> bool, dirty: impl Fn(&mut Dirty)) {
        let mut dirty_set = self.data.dirty.lock();
        // We need to mark as dirty all those in updated_modules, and lift those in readable.modules up if they are dirty.
        // Most things in updated are also in readable, so we are likely to set them twice - but it's not too expensive.
        // Make sure we do updated first, as doing readable will cause them all to move to dirty.
        for (handle, module_data) in self.data.updated_modules.iter_unordered() {
            if pred(handle) {
                dirty(&mut module_data.state.write(Step::Load).unwrap().dirty);
                dirty_set.insert(module_data.dupe());
            }
        }
        for handle in self.readable.modules.keys() {
            if pred(handle) {
                let module_data = self.get_module(handle);
                dirty(&mut module_data.state.write(Step::Load).unwrap().dirty);
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

        self.invalidate(|_| true, |dirty| dirty.find = true);
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
                module_data.state.write(Step::Load).unwrap().dirty.find = true;
                dirty_set.insert(module_data.dupe());
            }
        }
        for (handle, module_data) in self.readable.modules.iter() {
            if self.data.updated_modules.get(handle).is_none() {
                let config2 = self.data.state.get_config(handle);
                if module_data.config != config2 {
                    let module_data = self.get_module(handle);
                    *module_data.config.write() = config2;
                    module_data.state.write(Step::Load).unwrap().dirty.find = true;
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
                module_data.state.write(Step::Load).unwrap().dirty.find = true;
                dirty_set.insert(module_data.dupe());
            }
        }
        for (handle, module_data) in self.readable.modules.iter() {
            if self.data.updated_modules.get(handle).is_none()
                && configs.contains(&module_data.config)
            {
                let module_data = self.get_module(handle);
                module_data.state.write(Step::Load).unwrap().dirty.find = true;
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
        if changed.is_empty() {
            return;
        }
        self.invalidate(
            |handle| changed.contains(handle.path()),
            |dirty| dirty.load = true,
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
            |dirty| dirty.load = true,
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
            let mut alt = Steps::default();
            let lock = m.state.read();
            let stdlib = self.get_stdlib(&m.handle);
            let config = m.config.read();
            let ctx = Context {
                require: lock.require,
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
                recursion_limit_config: config.recursion_limit_config(),
            };
            let mut step = Step::Load; // Start at AST (Load.next)
            alt.load = lock.steps.load.dupe();
            while let Some(s) = step.next() {
                step = s;
                let start = Instant::now();
                step.compute(&alt, &ctx).0(&mut alt);
                write(&step, start)?;
                if step == Step::Exports {
                    let start = Instant::now();
                    let exports = alt.exports.as_ref().unwrap();
                    exports.wildcard(ctx.lookup);
                    exports.exports(ctx.lookup);
                    write(&"Exports-force", start)?;
                }
            }
            if let Some(subscriber) = &self.data.subscriber {
                subscriber.finish_work(self, &m.handle, &alt.load.unwrap(), false);
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

    pub fn get_exports(&self, handle: &Handle) -> Arc<SmallMap<Name, ExportLocation>> {
        let module_data = self.get_module(handle);
        self.lookup_export(&module_data)
            .exports(&self.lookup(module_data))
    }

    pub(crate) fn get_exports_data(&self, handle: &Handle) -> Exports {
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
        if module == self.module_data.handle.module()
            && path.is_none_or(|path| path == self.module_data.handle.path())
        {
            return FindingOrError::new_finding(self.module_data.dupe());
        }

        let cached = {
            let deps_read = self.module_data.deps.read();
            if let Some(ImportResolution::Resolved(handles)) = deps_read.get(&module)
                && path.is_none_or(|path| path == handles.first().0.path())
            {
                Some(handles.first().0.dupe())
            } else {
                None
            }
        };

        if let Some(handle) = cached {
            // Only acquire write lock if we actually have new dependencies to add.
            // This avoids lock contention on the hot path when the same import is
            // looked up repeatedly with no new dependency information.
            // First check with read lock if merge is needed
            let needs_merge = {
                let deps_read = self.module_data.deps.read();
                if let Some(ImportResolution::Resolved(handles)) = deps_read.get(&module)
                    && let Some(_existing) = handles.get(&handle)
                {
                    // Check if dep has any dependencies that aren't already tracked
                    !dep.names.is_empty()
                        || dep.wildcard
                        || !dep.classes.is_empty()
                        || !dep.type_aliases.is_empty()
                } else {
                    true
                }
            };
            if needs_merge {
                let mut write = self.module_data.deps.write();
                if let Some(ImportResolution::Resolved(handles)) = write.get_mut(&module)
                    && let Some(existing) = handles.get_mut(&handle)
                {
                    existing.merge_in_place(dep);
                }
            }
            return FindingOrError::new_finding(self.transaction.get_imported_module(&handle));
        }

        let handle = self
            .transaction
            .import_handle(&self.module_data.handle, module, path);

        match handle {
            FindingOrError::Finding(finding) => {
                let handle = finding.finding;
                let error = finding.error;
                let res = self.transaction.get_imported_module(&handle);

                let mut write = self.module_data.deps.write();
                let did_insert = match write.entry(module) {
                    Entry::Vacant(e) => {
                        e.insert(ImportResolution::Resolved(SmallMap1::new(
                            handle,
                            dep.clone(),
                        )));
                        true
                    }
                    Entry::Occupied(mut e) => {
                        match e.get_mut() {
                            ImportResolution::Resolved(handles) => {
                                if let Some(existing) = handles.get_mut(&handle) {
                                    existing.merge_in_place(dep);
                                    false
                                } else {
                                    handles.insert(handle, dep.clone());
                                    true
                                }
                            }
                            ImportResolution::Failed(_) => {
                                // A prior lookup (without explicit path) failed, but this lookup
                                // (with explicit path) succeeded. This can happen when an import
                                // is first resolved via search paths (fails) and later via an
                                // explicit path (e.g., from bundled typeshed). Upgrade to Resolved.
                                e.insert(ImportResolution::Resolved(SmallMap1::new(
                                    handle,
                                    dep.clone(),
                                )));
                                true
                            }
                        }
                    }
                };
                if did_insert {
                    let inserted = res.rdeps.lock().insert(self.module_data.handle.dupe());
                    assert!(inserted);
                }
                // Make sure we hold the deps write lock until after we insert into rdeps.
                drop(write);
                FindingOrError::Finding(Finding {
                    finding: res,
                    error,
                })
            }
            FindingOrError::Error(err) => {
                // Store the failed import so we can retry it when the config changes
                self.module_data
                    .deps
                    .write()
                    .insert(module, ImportResolution::Failed(err.dupe()));
                FindingOrError::Error(err)
            }
        }
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
        self.with_exports(
            module,
            |exports, lookup| exports.exports(lookup).contains_key(name),
            ModuleDep::name_type(name.clone()),
        )
        .unwrap_or(false)
    }

    fn get_wildcard(&self, module: ModuleName) -> Option<Arc<SmallSet<Name>>> {
        self.with_exports(
            module,
            |exports, lookup| exports.wildcard(lookup),
            ModuleDep::wildcard_dep(),
        )
    }

    fn module_exists(&self, module: ModuleName) -> FindingOrError<()> {
        self.get_module(module, None, ModuleDep::none())
            .map(|module_data| {
                self.transaction.lookup_export(&module_data);
            })
    }

    fn is_submodule_imported_implicitly(&self, module: ModuleName, name: &Name) -> bool {
        self.with_exports(
            module,
            |exports, _lookup| exports.is_submodule_imported_implicitly(name),
            ModuleDep::name_metadata(name.clone()),
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
            ModuleDep::none(),
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
            ModuleDep::name_metadata(name.clone()),
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
            ModuleDep::name_metadata(name.clone()),
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
                ModuleDep::name_type(name.clone()),
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
            ModuleDep::name_metadata(name.clone()),
        )?
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
            .get_module(module, path, k.to_anykey().to_module_dep())
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
