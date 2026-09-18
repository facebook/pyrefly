/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Shared infrastructure for in-memory type-checking sessions that supply
//! source strings rather than reading from disk (playground, embedder API).

use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_build::handle::Handle;
use pyrefly_build::source_db::LiveSourceDatabase;
use pyrefly_build::source_db::SourceDatabase;
use pyrefly_build::source_db::map_db::MapDatabase;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModuleStyle;
use pyrefly_python::sys_info::PythonPlatform;
use pyrefly_python::sys_info::PythonVersion;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::lock::Mutex;

use crate::config::config::ConfigFile;

/// The Python version string given to [`memory_config`] failed to parse.
#[derive(thiserror::Error, Debug)]
#[error("invalid Python version {version:?}: {cause}")]
pub struct InvalidPythonVersionError {
    version: String,
    #[source]
    cause: anyhow::Error,
}

/// Build a default [`ConfigFile`] and [`SysInfo`] for in-memory type checking.
/// Disables interpreter discovery and sets default Python environment values so
/// only the bundled typeshed is used.
pub(crate) fn memory_config(
    python_version: Option<&str>,
) -> Result<(ConfigFile, SysInfo), InvalidPythonVersionError> {
    let mut config = ConfigFile::default();
    config.python_environment.set_empty_to_default();
    config.interpreters.skip_interpreter_query = true;

    let sys_info = match python_version {
        Some(version) => {
            let parsed =
                PythonVersion::from_str(version).map_err(|cause| InvalidPythonVersionError {
                    version: version.to_owned(),
                    cause,
                })?;
            config.python_environment.python_version = Some(parsed);
            SysInfo::new(parsed, PythonPlatform::linux())
        }
        None => SysInfo::default(),
    };

    Ok((config, sys_info))
}

/// A thread-safe [`SourceDatabase`] wrapper around [`MapDatabase`].
///
/// The embedder API keeps one warm [`crate::state::state::State`] across calls,
/// but the in-memory module set can change between checks. This wrapper lets the
/// source database stored inside the config (and thus inside the `State`) see
/// updated modules without rebuilding the `State`.
#[derive(Debug)]
pub(crate) struct SharedMapDatabase(Arc<Mutex<MapDatabase>>);

impl SharedMapDatabase {
    pub(crate) fn new(db: MapDatabase) -> Self {
        Self(Arc::new(Mutex::new(db)))
    }

    /// Replace the inner database, returning whether the contents changed.
    pub(crate) fn replace(&self, new_db: MapDatabase) -> bool {
        let mut guard = self.0.lock();
        if *guard == new_db {
            false
        } else {
            *guard = new_db;
            true
        }
    }
}

impl Clone for SharedMapDatabase {
    fn clone(&self) -> Self {
        Self(self.0.dupe())
    }
}

impl SourceDatabase for SharedMapDatabase {
    fn may_contain_module(&self, module: ModuleName) -> bool {
        self.0.lock().may_contain_module(module)
    }

    fn lookup(
        &self,
        module: ModuleName,
        origin: Option<&Path>,
        style_filter: Option<ModuleStyle>,
    ) -> Option<ModulePath> {
        self.0.lock().lookup(module, origin, style_filter)
    }

    fn handle_from_module_path(&self, module_path: &ModulePath) -> Option<Handle> {
        self.0.lock().handle_from_module_path(module_path)
    }

    fn as_live_source_database(&self) -> Option<&dyn LiveSourceDatabase> {
        None
    }
}
