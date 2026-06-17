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
//! [`Checker`] holds one warm [`State`]: the first [`Checker::check`] pays the
//! one-time typeshed load, later checks reuse it. Each check overlays its files in
//! a single transaction and solves only the target module ([`Require::Errors`]),
//! leaving dependencies (stubs, typeshed) at [`Require::Exports`] — so a stub
//! context is resolved, not fully re-checked, and only the target's diagnostics
//! are collected.

use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_build::handle::Handle;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::sys_info::PythonPlatform;
use pyrefly_python::sys_info::PythonVersion;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::thread_pool::ThreadCount;

use crate::config::config::ConfigFile;
use crate::config::finder::ConfigFinder;
use crate::error::error::Error;
use crate::state::load::FileContents;
use crate::state::require::Require;
use crate::state::state::State;

pub use crate::config::error_kind::Severity;

/// A reusable type checker holding one warm [`State`].
///
/// Cheap to keep alive and share (`&self` checks); construct once so the typeshed
/// load is amortized across calls. Not tied to any on-disk project — all input is
/// in-memory source supplied per [`check`](Checker::check).
pub struct Checker {
    state: State,
    sys_info: SysInfo,
}

impl Checker {
    /// Build a checker for the given Python version (e.g. `"3.14"`), or the default
    /// version when `None`. No interpreter is queried and the bundled typeshed is used.
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

        config.configure();
        let config_finder = ConfigFinder::new_constant(ArcId::new(config));
        Ok(Self {
            state: State::new(config_finder, ThreadCount::default()),
            sys_info,
        })
    }

    /// Type check `main_source` (as module `main_name`) against optional in-memory
    /// `context` modules, returning diagnostics for the main module only.
    ///
    /// Context modules (each `(module_name, source)`) are importable by the main
    /// module — e.g. monty's accumulated stubs — but their own diagnostics are not
    /// reported. Only the main module is fully solved; context and typeshed are
    /// resolved at export level.
    pub fn check(
        &self,
        main_name: &str,
        main_source: &str,
        context: &[(&str, &str)],
    ) -> Vec<Diagnostic> {
        let main_handle = self.handle(main_name);

        let mut files = Vec::with_capacity(context.len() + 1);
        for (name, source) in context {
            files.push(memory_file(name, source));
        }
        files.push(memory_file(main_name, main_source));

        // One transaction, one solve of just the main handle; committing keeps the
        // typeshed/State warm for the next call.
        let mut transaction = self
            .state
            .new_committable_transaction(Require::Exports, None);
        transaction.as_mut().set_memory(files);
        self.state.run_with_committing_transaction(
            transaction,
            &[main_handle.dupe()],
            Require::Errors,
            None,
            None,
        );

        self.state
            .transaction()
            .get_errors([&main_handle])
            .collect_errors()
            .ordinary
            .iter()
            .map(Diagnostic::from_error)
            .collect()
    }

    fn handle(&self, name: &str) -> Handle {
        Handle::new(
            ModuleName::from_str(name),
            ModulePath::memory(PathBuf::from(format!("{name}.py"))),
            self.sys_info.dupe(),
        )
    }
}

/// Path-keyed in-memory file for `set_memory`, matching the module name used by [`Checker::handle`].
fn memory_file(name: &str, source: &str) -> (PathBuf, Option<Arc<FileContents>>) {
    (
        PathBuf::from(format!("{name}.py")),
        Some(Arc::new(FileContents::from_source(source.to_owned()))),
    )
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
