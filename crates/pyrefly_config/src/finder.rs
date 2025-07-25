/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ffi::OsString;
use std::mem;
use std::path::Path;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_util::absolutize::Absolutize as _;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::lock::Mutex;
use pyrefly_util::upward_search::UpwardSearch;
use tracing::Level;
use tracing::debug;
use tracing::enabled;
use tracing::error;
use tracing::info;
use tracing::warn;

use crate::config::ConfigFile;
use crate::error_kind::Severity;

pub struct ConfigError {
    severity: Severity,
    msg: anyhow::Error,
}

impl ConfigError {
    pub fn error(msg: anyhow::Error) -> Self {
        Self {
            severity: Severity::Error,
            msg,
        }
    }

    pub fn warn(msg: anyhow::Error) -> Self {
        Self {
            severity: Severity::Warn,
            msg,
        }
    }

    pub fn print(&self) {
        match self.severity {
            Severity::Error => {
                error!("{:#}", self.msg);
            }
            Severity::Warn => {
                warn!("{:#}", self.msg);
            }
            Severity::Info => {
                info!("{:#}", self.msg);
            }
            Severity::Ignore => {}
        }
    }

    pub fn context(self, context: String) -> Self {
        ConfigError {
            severity: self.severity,
            msg: self.msg.context(context),
        }
    }

    pub fn severity(&self) -> Severity {
        self.severity
    }

    pub fn get_message(&self) -> String {
        self.msg.to_string()
    }
}

/// When debugging is enabled, log errors.
pub fn debug_log(errors: Vec<ConfigError>) {
    if enabled!(Level::DEBUG) {
        for e in errors {
            debug!("{:#}", e.msg);
        }
    }
}

/// A way to find a config file given a directory or Python file.
/// Uses a lot of caching.
pub struct ConfigFinder {
    /// The cached state, with previously found entries.
    search: UpwardSearch<ArcId<ConfigFile>>,
    /// The errors that have occurred when loading.
    errors: Arc<Mutex<Vec<ConfigError>>>,

    /// Function to run before checking the state. If this returns a value, it is _not_ cached.
    /// If this returns anything other than `Ok`, the rest of the functions are used.
    before: Box<
        dyn Fn(ModuleName, &ModulePath) -> anyhow::Result<Option<ArcId<ConfigFile>>> + Send + Sync,
    >,
    /// If there is no config file, or loading it fails, use this fallback.
    fallback: Box<dyn Fn(ModuleName, &ModulePath) -> ArcId<ConfigFile> + Send + Sync>,

    clear_extra_caches: Box<dyn Fn() + Send + Sync>,
}

impl ConfigFinder {
    /// Create a new ConfigFinder a way to load a config file, and a default if that errors or there is no file.
    pub fn new(
        load: Box<dyn Fn(&Path) -> (ArcId<ConfigFile>, Vec<ConfigError>) + Send + Sync>,
        fallback: Box<dyn Fn(ModuleName, &ModulePath) -> ArcId<ConfigFile> + Send + Sync>,
        clear_extra_caches: Box<dyn Fn() + Send + Sync>,
    ) -> Self {
        Self::new_custom(
            Box::new(|_, _| Ok(None)),
            load,
            fallback,
            clear_extra_caches,
        )
    }

    /// Create a new ConfigFinder that always returns the same constant.
    pub fn new_constant(constant: ArcId<ConfigFile>) -> Self {
        let c1 = constant.dupe();
        let c2 = constant.dupe();
        let c3 = constant;

        Self::new_custom(
            Box::new(move |_, _| Ok(Some(c1.dupe()))),
            Box::new(move |_| (c2.dupe(), Vec::new())),
            Box::new(move |_, _| c3.dupe()),
            Box::new(|| {}),
        )
    }

    /// Create a new ConfigFinder, but with a custom way to produce a result from a Python file.
    /// If the `before` function fails to produce a config, then the other methods will be used.
    /// The `before` function is not cached in any way.
    fn new_custom(
        before: Box<
            dyn Fn(ModuleName, &ModulePath) -> anyhow::Result<Option<ArcId<ConfigFile>>>
                + Send
                + Sync,
        >,
        load: Box<dyn Fn(&Path) -> (ArcId<ConfigFile>, Vec<ConfigError>) + Send + Sync>,
        fallback: Box<dyn Fn(ModuleName, &ModulePath) -> ArcId<ConfigFile> + Send + Sync>,
        clear_extra_caches: Box<dyn Fn() + Send + Sync>,
    ) -> Self {
        let errors = Arc::new(Mutex::new(Vec::new()));
        let errors2 = errors.dupe();

        Self {
            search: UpwardSearch::new(
                ConfigFile::CONFIG_FILE_NAMES
                    .iter()
                    .chain(ConfigFile::ADDITIONAL_ROOT_FILE_NAMES)
                    .map(OsString::from)
                    .collect(),
                move |x| {
                    let (v, errors) = load(x);
                    errors2.lock().extend(errors);
                    v
                },
            ),
            errors,
            before,
            fallback,
            clear_extra_caches,
        }
    }

    /// Invalidate all data stored in the config.
    pub fn clear(&self) {
        self.search.clear();
        (self.clear_extra_caches)();
        *self.errors.lock() = Vec::new();
    }

    /// Collect all the current errors that have been produced, and clear them.
    pub fn errors(&self) -> Vec<ConfigError> {
        mem::take(&mut self.errors.lock())
    }

    pub fn add_errors(&self, errors: Vec<ConfigError>) {
        self.errors.lock().extend(errors);
    }

    /// Get the config file associated with a directory.
    pub fn directory(&self, dir: &Path) -> Option<ArcId<ConfigFile>> {
        self.search.directory(dir)
    }

    /// Get the config file given a Python file.
    pub fn python_file(&self, name: ModuleName, path: &ModulePath) -> ArcId<ConfigFile> {
        match (self.before)(name, path) {
            Ok(Some(x)) => return x,
            Ok(None) => {}
            Err(e) => {
                self.errors.lock().push(ConfigError::error(e));
            }
        }

        let f = |dir: Option<&Path>| match dir {
            Some(parent) => self
                .search
                .directory_absolute(parent)
                .unwrap_or_else(|| (self.fallback)(name, path)),
            None => (self.fallback)(name, path),
        };

        match path.details() {
            ModulePathDetails::FileSystem(x) | ModulePathDetails::Memory(x) => {
                let absolute = x.absolutize();
                f(absolute.parent())
            }
            ModulePathDetails::Namespace(x) => f(Some(&x.absolutize())),
            ModulePathDetails::BundledTypeshed(_) => f(None),
        }
    }

    /// If we have an error, print all the errors that the config finder has accumulated. This is used
    /// to ensure that config errors are still surfaced if we exit early.
    pub fn checkpoint<R, E>(&self, result: Result<R, E>) -> Result<R, E> {
        if result.is_err() {
            for error in self.errors() {
                error.print();
            }
        }
        result
    }
}
