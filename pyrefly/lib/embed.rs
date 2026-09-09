/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A small, programmatic type-checking API for embedders (e.g. sandboxed
//! interpreters, REPLs) that want "source in, diagnostics out" against a reused,
//! warm checker — without driving the editor-oriented [`crate::playground`].
//!
//! This interface is experimental and NOT stable. It will change without notice
//! during minor version increments, and should not be relied upon.
//!
//! [`Checker`] holds one warm [`State`]. The first [`Checker::check`] pays the
//! one-time typeshed load; later checks reuse it, overlaying the supplied module
//! contents in a single transaction and solving only the target module
//! ([`Require::Errors`]) — so context modules (stubs) and typeshed are resolved at
//! export level, not re-checked, and only the target's diagnostics are collected.

use std::path::Path;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_build::handle::Handle;
use pyrefly_build::source_db::LiveSourceDatabase;
use pyrefly_build::source_db::SourceDatabase;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModuleStyle;
use pyrefly_python::sys_info::PythonPlatform;
use pyrefly_python::sys_info::PythonVersion;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::lock::Mutex;
use pyrefly_util::thread_pool::ThreadCount;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::config::config::ConfigFile;
pub use crate::config::error_kind::Severity;
use crate::config::finder::ConfigFinder;
use crate::error::error::Error;
use crate::state::load::FileContents;
use crate::state::require::Require;
use crate::state::state::State;

/// A reusable type checker holding one warm [`State`].
///
/// Construct once, amortizing the typeshed load, then call [`check`](Checker::check)
/// per snippet. Cheap to keep alive and share (`&self` checks).
pub struct Checker {
    state: State,
    sys_info: SysInfo,
    /// The in-memory modules visible to the current check, shared with the source
    /// database so that import resolution sees whatever [`Checker::check`] was given.
    modules: Arc<Mutex<SmallMap<ModuleName, ModulePath>>>,
    /// Held so that a changed module set can invalidate the cached import resolutions
    /// made under it.
    config: ArcId<ConfigFile>,
}

impl Checker {
    /// Build a checker for the given Python version (e.g. `"3.14"`, or the default
    /// when `None`). Everything not supplied to [`Checker::check`] resolves to the
    /// bundled typeshed. No interpreter is queried.
    pub fn new(python_version: Option<&str>) -> Result<Self, String> {
        let mut config = ConfigFile::default();
        config.python_environment.set_empty_to_default();
        config.interpreters.skip_interpreter_query = true;

        let sys_info = match python_version {
            Some(version) => {
                let parsed = PythonVersion::from_str(version)
                    .map_err(|e| format!("invalid Python version '{version}': {e}"))?;
                config.python_environment.python_version = Some(parsed);
                SysInfo::new(parsed, PythonPlatform::linux())
            }
            None => SysInfo::default(),
        };

        let modules = Arc::new(Mutex::new(SmallMap::new()));
        config.source_db = Some(ArcId::new(Box::new(MemorySourceDb {
            modules: modules.dupe(),
            sys_info: sys_info.dupe(),
        })));

        config.configure();
        let config = ArcId::new(config);
        let config_finder = ConfigFinder::new_constant(config.dupe());
        Ok(Self {
            state: State::new(config_finder, ThreadCount::default()),
            sys_info,
            modules,
            config,
        })
    }

    /// Type check the `target` module, returning diagnostics for it only.
    ///
    /// `files` supplies the source for each in-memory module (each
    /// `(module_name, source)`), which are importable from one another. Modules other
    /// than `target` are importable but their own diagnostics are not reported.
    pub fn check(&self, target: &str, files: &[(&str, &str)]) -> Vec<Diagnostic> {
        let modules: SmallMap<_, _> = files
            .iter()
            .map(|(name, _)| (ModuleName::from_str(name), memory_path(name)))
            .collect();
        // Import resolutions are cached per config, so a changed module set has to
        // discard them; otherwise a module dropped since the last check still resolves.
        let modules_changed = {
            let mut current = self.modules.lock();
            let changed = *current != modules;
            *current = modules;
            changed
        };

        let target_handle = self.handle(target);
        let memory = files
            .iter()
            .map(|(name, source)| {
                (
                    memory_path(name).as_path().to_path_buf(),
                    Some(Arc::new(FileContents::from_source((*source).to_owned()))),
                )
            })
            .collect();

        // One transaction, one solve of just the target handle; committing keeps the
        // typeshed/State warm for the next call.
        let mut transaction = self
            .state
            .new_committable_transaction(Require::Exports, None);
        transaction.as_mut().set_memory(memory);
        if modules_changed {
            transaction
                .as_mut()
                .invalidate_find_for_configs(SmallSet::from_iter([self.config.dupe()]));
        }
        self.state.run_with_committing_transaction(
            transaction,
            &[target_handle.dupe()],
            Require::Errors,
            None,
            None,
        );

        self.state
            .transaction()
            .get_errors([&target_handle])
            .collect_errors()
            .ordinary
            .iter()
            .map(Diagnostic::from_error)
            .collect()
    }

    fn handle(&self, name: &str) -> Handle {
        Handle::new(
            ModuleName::from_str(name),
            memory_path(name),
            self.sys_info.dupe(),
        )
    }
}

/// In-memory module path for `name`, e.g. `name.py`. Shared by the source database
/// and `set_memory` so import resolution and file contents agree.
fn memory_path(name: &str) -> ModulePath {
    ModulePath::memory(PathBuf::from(format!("{name}.py")))
}

/// Resolves the embedder's declared in-memory modules by name; everything else
/// (typeshed, stdlib) falls through to normal resolution.
#[derive(Debug)]
struct MemorySourceDb {
    modules: Arc<Mutex<SmallMap<ModuleName, ModulePath>>>,
    sys_info: SysInfo,
}

impl SourceDatabase for MemorySourceDb {
    fn lookup(
        &self,
        module: ModuleName,
        _origin: Option<&Path>,
        _style_filter: Option<ModuleStyle>,
    ) -> Option<ModulePath> {
        self.modules.lock().get(&module).cloned()
    }

    fn handle_from_module_path(&self, module_path: &ModulePath) -> Option<Handle> {
        let modules = self.modules.lock();
        let (name, _) = modules.iter().find(|(_, p)| *p == module_path)?;
        Some(Handle::new(
            name.dupe(),
            module_path.dupe(),
            self.sys_info.dupe(),
        ))
    }

    /// Never live: the module set is fixed at construction, so there is nothing to requery.
    fn as_live_source_database(&self) -> Option<&dyn LiveSourceDatabase> {
        None
    }
}

/// A single type-checking diagnostic, with owned data so it outlives the checker
/// transaction. Positions are 1-based (line and column), matching editor display.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub start_line: u32,
    pub start_col: u32,
    pub end_line: u32,
    pub end_col: u32,
    pub severity: Severity,
    /// Kebab-case rule id, e.g. `bad-assignment`.
    pub kind: String,
    /// One-line summary of the problem.
    pub message: String,
    /// Extra context, empty when the diagnostic has none.
    pub details: String,
}

impl Diagnostic {
    fn from_error(error: &Error) -> Self {
        let range = error.display_range();
        Self {
            start_line: range.start.line_within_file().get(),
            start_col: range.start.column().get(),
            end_line: range.end.line_within_file().get(),
            end_col: range.end.column().get(),
            severity: error.severity(),
            kind: error.error_kind().to_name().to_owned(),
            message: error.msg_header().to_owned(),
            details: error.msg_details().unwrap_or("").to_owned(),
        }
    }
}
