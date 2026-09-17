/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Display;
use std::path::Path;
use std::path::PathBuf;
use std::sync::LazyLock;

use itertools::Itertools;
use pyrefly_python::sys_info::PythonPlatform;
use pyrefly_python::sys_info::PythonVersion;
use pyrefly_util::lock::Mutex;
use pyrefly_util::stdlib::register_stdlib_paths;
use serde::Deserialize;
use serde::Serialize;
use serde_with::skip_serializing_none;
use starlark_map::small_map::SmallMap;

#[cfg(not(target_arch = "wasm32"))]
use crate::environment::interpreter_query;
use crate::environment::interpreters::Interpreters;

static INTERPRETER_ENV_REGISTRY: LazyLock<
    Mutex<SmallMap<PathBuf, Result<PythonEnvironment, String>>>,
> = LazyLock::new(|| Mutex::new(SmallMap::new()));

/// Values representing the environment of the Python interpreter.
/// These values are `None` by default, so we can tell if a config
/// overrode them, or if we should query a Python interpreter for
/// any missing values. We can't query a Python interpreter
/// on config parsing, since we also won't know if an executable
/// other than the first available on the path should be used (i.e.
/// should we always look at a venv/conda environment instead?)
#[skip_serializing_none]
#[derive(Debug, PartialEq, Eq, Deserialize, Serialize, Clone, Default)]
#[serde(rename_all = "kebab-case")]
pub struct PythonEnvironment {
    /// The platform any `sys.platform` check should evaluate against.
    #[serde(
        // TODO(connernilsen): DON'T COPY THIS TO NEW FIELDS. This is a temporary
        // alias while we migrate existing fields from snake case to kebab case.
        alias = "python_platform"
    )]
    pub python_platform: Option<PythonPlatform>,

    /// The platform any `sys.version` check should evaluate against.
    #[serde(
        // TODO(connernilsen): DON'T COPY THIS TO NEW FIELDS. This is a temporary
        // alias while we migrate existing fields from snake case to kebab case.
        alias = "python_version"
    )]
    pub python_version: Option<PythonVersion>,

    /// Directories containing third-party package imports, searched
    /// after first checking `search_path` and `typeshed`.
    #[serde(
        // TODO(connernilsen): DON'T COPY THIS TO NEW FIELDS. This is a temporary
        // alias while we migrate existing fields from snake case to kebab case.
        alias = "site_package_path"
    )]
    pub site_package_path: Option<Vec<PathBuf>>,

    #[serde(skip)]
    pub interpreter_site_package_path: Vec<PathBuf>,

    /// The subset of `interpreter_site_package_path` that the interpreter
    /// reports as editable (PEP 610) installs, so callers can exempt those
    /// paths from Pyrefly's default project-source exclusion.
    #[serde(skip)]
    pub interpreter_editable_path: Vec<PathBuf>,

    #[serde(alias = "stdlib_paths", default, skip_serializing)]
    pub interpreter_stdlib_path: Vec<PathBuf>,
}

impl PythonEnvironment {
    fn pyrefly_default() -> Self {
        let mut env = Self::default();
        env.set_empty_to_default();
        env
    }

    /// If any Python environment values are `None`, set them to
    /// Pyrefly's default value.
    pub fn set_empty_to_default(&mut self) {
        if self.python_platform.is_none() {
            self.python_platform = Some(PythonPlatform::default());
        }
        if self.python_version.is_none() {
            self.python_version = Some(PythonVersion::default());
        }
        if self.site_package_path.is_none() {
            // The `typings/` default is applied in `ConfigFile::configure()` so it
            // can be resolved relative to the config root and applied regardless
            // of whether an interpreter was queried.
            self.site_package_path = Some(Vec::new());
        }
    }

    /// Given another `PythonEnvironment`, override any `None` values
    /// in this `PythonEnvironment` with the other environment's values.
    pub fn override_empty(&mut self, other: Self) {
        if self.python_platform.is_none() {
            self.python_platform = other.python_platform;
        }
        if self.python_version.is_none() {
            self.python_version = other.python_version;
        }
        if self.site_package_path.is_none() {
            self.site_package_path = other.site_package_path;
        }
        self.interpreter_site_package_path = other.interpreter_site_package_path.clone();
        self.interpreter_editable_path = other.interpreter_editable_path.clone();
        self.interpreter_stdlib_path = other.interpreter_stdlib_path.clone();
    }

    /// Given a path to a Python interpreter executable, query that interpreter for its
    /// version, platform, and site package path. Return an error in the case of failure during
    /// execution, parsing, or deserializing.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn get_env_from_interpreter(interpreter: &Path) -> anyhow::Result<PythonEnvironment> {
        let env = interpreter_query::query(interpreter)?;
        Self::cache_interpreter_stdlib_path(env.interpreter_stdlib_path.clone());
        Ok(env)
    }

    #[cfg(target_arch = "wasm32")]
    pub fn get_env_from_interpreter(_interpreter: &Path) -> anyhow::Result<PythonEnvironment> {
        Err(anyhow::anyhow!(
            "Python interpreter queries are not supported on WebAssembly"
        ))
    }

    /// Given a path to an interpreter, query the interpreter with
    /// [`Self::get_env_from_interpreter()`] and cache the result. If a cached
    /// result already exists, return that.
    ///
    /// In the case of failure, log an error message and return Pyrefly's
    /// [`PythonEnvironment::default()`].
    pub fn get_interpreter_env(interpreter: &Path) -> (PythonEnvironment, Option<anyhow::Error>) {
        let env = INTERPRETER_ENV_REGISTRY.lock()
        .entry(interpreter.to_path_buf()).or_insert_with(move || {
            Self::get_env_from_interpreter(interpreter).map_err(|e| {
                format!("Failed to query interpreter at {}, falling back to default Python environment settings\n{}", interpreter.display(), e)
            })
        }).clone();
        match env {
            Ok(env) => (env, None),
            Err(message) => (Self::pyrefly_default(), Some(anyhow::anyhow!(message))),
        }
    }

    fn cache_interpreter_stdlib_path(path: Vec<PathBuf>) {
        register_stdlib_paths(path);
    }

    /// [`Self::get_default_interpreter()`] and [`Self::get_interpreter_env()`] with the resulting value,
    /// or return [`PythonEnvironment::default()`] if `None`.
    pub fn get_default_interpreter_env() -> PythonEnvironment {
        Interpreters::get_default_interpreter().map_or_else(Self::pyrefly_default, |path| {
            Self::get_interpreter_env(path).0
        })
    }
}

impl Display for PythonEnvironment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{{python_platform: {}, python_version: {}, site_package_path: [{}], interpreter_site_package_path: [{}], interpreter_stdlib_path: [{}]}}",
            self.python_platform
                .as_ref()
                .map_or_else(|| "None".to_owned(), |platform| platform.to_string()),
            self.python_version
                .map_or_else(|| "None".to_owned(), |version| version.to_string()),
            self.site_package_path.as_ref().map_or_else(
                || "".to_owned(),
                |packages| packages.iter().map(|p| p.display()).join(", ")
            ),
            self.interpreter_site_package_path
                .iter()
                .map(|p| p.display())
                .join(", "),
            self.interpreter_stdlib_path
                .iter()
                .map(|p| p.display())
                .join(", "),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use pyrefly_python::sys_info::PythonPlatform;
    use pyrefly_python::sys_info::PythonVersion;

    use super::*;

    #[test]
    fn test_display_includes_stdlib_path() {
        let env = PythonEnvironment {
            python_platform: Some(PythonPlatform::mac()),
            python_version: Some(PythonVersion::new(3, 10, 5)),
            site_package_path: Some(vec![PathBuf::from("/path/to/site-packages")]),
            interpreter_site_package_path: vec![PathBuf::from("/path/to/site-packages")],
            interpreter_editable_path: Vec::new(),
            interpreter_stdlib_path: vec![
                PathBuf::from("/usr/lib/python3.10"),
                PathBuf::from("/usr/lib/python3.10/lib-dynload"),
            ],
        };

        let display = format!("{}", env);
        assert!(display.contains("interpreter_stdlib_path"));
        assert!(display.contains("/usr/lib/python3.10"));
    }

    #[test]
    fn test_override_empty_propagates_interpreter_paths() {
        let mut env1 = PythonEnvironment {
            python_platform: None,
            python_version: None,
            site_package_path: None,
            interpreter_site_package_path: Vec::new(),
            interpreter_editable_path: Vec::new(),
            interpreter_stdlib_path: Vec::new(),
        };

        let env2 = PythonEnvironment {
            python_platform: Some(PythonPlatform::mac()),
            python_version: Some(PythonVersion::new(3, 10, 0)),
            site_package_path: Some(vec![PathBuf::from("/path/to/site-packages")]),
            interpreter_site_package_path: vec![PathBuf::from("/path/to/site-packages")],
            interpreter_editable_path: vec![PathBuf::from("/path/to/editable")],
            interpreter_stdlib_path: vec![
                PathBuf::from("/usr/lib/python3.10"),
                PathBuf::from("/usr/lib/python3.10/lib-dynload"),
            ],
        };

        env1.override_empty(env2.clone());

        assert_eq!(env1.interpreter_stdlib_path, env2.interpreter_stdlib_path);
        assert_eq!(
            env1.interpreter_site_package_path,
            env2.interpreter_site_package_path
        );
        assert_eq!(
            env1.interpreter_editable_path,
            env2.interpreter_editable_path
        );
    }
}
