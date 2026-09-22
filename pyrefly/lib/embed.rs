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

use std::path::PathBuf;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_build::handle::Handle;
use pyrefly_build::source_db::map_db::MapDatabase;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::thread_pool::ThreadCount;
use starlark_map::small_set::SmallSet;

use crate::config::config::ConfigFile;
pub use crate::config::error_kind::Severity;
use crate::config::finder::ConfigFinder;
use crate::error::error::Error;
pub use crate::memory_project::InvalidPythonVersionError;
use crate::memory_project::SharedMapDatabase;
use crate::memory_project::memory_config;
use crate::state::load::FileContents;
use crate::state::require::Require;
use crate::state::state::State;

/// A reusable type checker holding one warm [`State`].
///
/// Construct once, amortizing the typeshed load, then call [`check`](Checker::check)
/// per snippet.
pub struct Checker {
    state: State,
    sys_info: SysInfo,
    /// Shared with the config's source database so that import resolution sees
    /// whatever module set [`Checker::check`] was last given.
    source_db: SharedMapDatabase,
    /// Held so that a changed module set can invalidate the cached import
    /// resolutions made under it.
    config: ArcId<ConfigFile>,
}

impl Checker {
    /// Build a checker for the given Python version (e.g. `"3.14"`, or the default
    /// when `None`). Everything not supplied to [`Checker::check`] resolves to the
    /// bundled typeshed. No interpreter is queried.
    pub fn try_new(python_version: Option<&str>) -> Result<Self, InvalidPythonVersionError> {
        let (mut config, sys_info) = memory_config(python_version)?;

        let source_db = SharedMapDatabase::new(MapDatabase::new(sys_info.dupe()));
        config.source_db = Some(ArcId::new(Box::new(source_db.clone())));

        config.configure();
        let config = ArcId::new(config);
        let config_finder = ConfigFinder::new_constant(config.dupe());
        Ok(Self {
            state: State::new(config_finder, ThreadCount::default()),
            sys_info,
            source_db,
            config,
        })
    }

    /// Type check the `target` module, returning diagnostics for it only.
    ///
    /// `files` supplies the source for each in-memory module (each
    /// `(module_name, source)`), which are importable from one another. Modules other
    /// than `target` are importable but their own diagnostics are not reported.
    pub fn check(&mut self, target: &str, files: &[(&str, &str)]) -> Vec<Diagnostic> {
        let mut new_db = MapDatabase::new(self.sys_info.dupe());
        for (name, _) in files {
            new_db.insert(ModuleName::from_str(name), memory_path(name));
        }
        let modules_changed = self.source_db.replace(new_db);

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

        let mut transaction = self
            .state
            .new_committable_transaction(Require::Exports, None);
        transaction.as_mut().set_memory(memory);
        if modules_changed {
            // Without this, import resolution for a module dropped from the new
            // module set could still be served from the state's cached lookups
            // made under the old `source_db` contents.
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
            // Only ordinary diagnostics are intentionally returned here; directives
            // (e.g. `reveal_type`) are excluded from this API.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_python_version_is_typed_error() {
        let err = match Checker::try_new(Some("not-a-version")) {
            Ok(_) => panic!("expected an invalid Python version to be rejected"),
            Err(err) => err,
        };
        assert_eq!(
            err.to_string(),
            "invalid Python version \"not-a-version\": Invalid version string: not-a-version."
        );
    }

    #[test]
    fn test_basic_diagnostic_round_trip() {
        let mut checker = Checker::try_new(None).unwrap();
        let diagnostics = checker.check("main", &[("main", "x: int = 'hello'")]);
        assert_eq!(diagnostics.len(), 1);
        let d = &diagnostics[0];
        assert_eq!(d.kind, "bad-assignment");
        assert_eq!(d.severity, Severity::Error);
        assert_eq!(d.start_line, 1);
        assert!(d.start_col > 0);
    }

    #[test]
    fn test_changed_module_set_invalidates_imports() {
        let mut checker = Checker::try_new(None).unwrap();

        // First check: target imports from helper_a, which exists.
        let diags1 = checker.check(
            "main",
            &[
                ("main", "from helper_a import value\nx: int = value"),
                ("helper_a", "value: int = 1"),
            ],
        );
        assert!(
            diags1.iter().all(|d| d.kind != "missing-import"),
            "helper_a should resolve: {diags1:?}",
        );

        // Second check: replace the module set — helper_a is gone, helper_b is
        // present. The target now imports helper_b. This exercises
        // SharedMapDatabase replacement and import-cache invalidation: without
        // invalidation the stale cache would still map to the old module set.
        let diags2 = checker.check(
            "main",
            &[
                ("main", "from helper_b import value\nx: int = value"),
                ("helper_b", "value: int = 2"),
            ],
        );
        assert!(
            diags2.iter().all(|d| d.kind != "missing-import"),
            "helper_b should resolve after module set change: {diags2:?}",
        );

        // Also verify the old module is no longer importable.
        let diags3 = checker.check(
            "main",
            &[
                ("main", "from helper_a import value\nx: int = value"),
                ("helper_b", "value: int = 2"),
            ],
        );
        assert!(
            diags3.iter().any(|d| d.kind == "missing-import"),
            "helper_a should be missing after module set changed: {diags3:?}",
        );
    }
}
