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
use std::process::Command;
use std::sync::LazyLock;

use anyhow::Context;
use anyhow::anyhow;
use itertools::Itertools;
use pyrefly_python::sys_info::PythonPlatform;
use pyrefly_python::sys_info::PythonVersion;
use pyrefly_util::lock::Mutex;
use pyrefly_util::lock::RwLock;
use serde::Deserialize;
use serde::Serialize;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use tracing::warn;

use crate::environment::interpreters::Interpreters;

static INTERPRETER_ENV_REGISTRY: LazyLock<
    Mutex<SmallMap<PathBuf, Result<PythonEnvironment, String>>>,
> = LazyLock::new(|| Mutex::new(SmallMap::new()));

static INTERPRETER_STDLIB_PATH_REGISTRY: LazyLock<RwLock<SmallSet<PathBuf>>> =
    LazyLock::new(|| RwLock::new(SmallSet::new()));

/// Values representing the environment of the Python interpreter.
/// These values are `None` by default, so we can tell if a config
/// overrode them, or if we should query a Python interpreter for
/// any missing values. We can't query a Python interpreter
/// on config parsing, since we also won't know if an executable
/// other than the first available on the path should be used (i.e.
/// should we always look at a venv/conda environment instead?)
#[derive(Debug, PartialEq, Eq, Deserialize, Serialize, Clone, Default)]
#[serde(rename_all = "kebab-case")]
pub struct PythonEnvironment {
    /// The platform any `sys.platform` check should evaluate against.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        // TODO(connernilsen): DON'T COPY THIS TO NEW FIELDS. This is a temporary
        // alias while we migrate existing fields from snake case to kebab case.
        alias = "python_platform"
    )]
    pub python_platform: Option<PythonPlatform>,

    /// The platform any `sys.version` check should evaluate against.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        // TODO(connernilsen): DON'T COPY THIS TO NEW FIELDS. This is a temporary
        // alias while we migrate existing fields from snake case to kebab case.
        alias = "python_version"
    )]
    pub python_version: Option<PythonVersion>,

    /// Directories containing third-party package imports, searched
    /// after first checking `search_path` and `typeshed`.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        // TODO(connernilsen): DON'T COPY THIS TO NEW FIELDS. This is a temporary
        // alias while we migrate existing fields from snake case to kebab case.
        alias = "site_package_path"
    )]
    pub site_package_path: Option<Vec<PathBuf>>,

    #[serde(skip, default)]
    pub interpreter_site_package_path: Vec<PathBuf>,

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
            let typings = PathBuf::from("./typings");
            if typings.exists() {
                self.site_package_path = Some(vec![PathBuf::from("./typings")]);
            } else {
                self.site_package_path = Some(Vec::new());
            }
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
        self.interpreter_stdlib_path = other.interpreter_stdlib_path.clone();
    }

    /// Given a path to a Python interpreter executable, query that interpreter for its
    /// version, platform, and site package path. Return an error in the case of failure during
    /// execution, parsing, or deserializing.
    pub fn get_env_from_interpreter(interpreter: &Path) -> anyhow::Result<PythonEnvironment> {
        if let Ok(pythonpath) = std::env::var("PYTHONPATH") {
            warn!(
                "PYTHONPATH environment variable is set to `{}`. Checks in other environments may not include these paths.",
                pythonpath
            );
        }

        let script = "\
import json, sys, sysconfig
platform = sys.platform
v = sys.version_info
version = '{}.{}.{}'.format(v.major, v.minor, v.micro)
site_package_path = list(filter(lambda x: x != '' and '.zip' not in x, sys.path))
stdlib_paths = [sysconfig.get_path('stdlib'), sysconfig.get_path('platstdlib')]
print(json.dumps({'python_platform': platform, 'python_version': version, 'site_package_path': site_package_path, 'stdlib_paths': stdlib_paths}))
";

        let mut command = Command::new(interpreter);
        command.arg("-c");
        command.arg(script);

        let python_info = command.output()?;

        let stdout = String::from_utf8(python_info.stdout).with_context(|| {
            format!(
                "while parsing Python interpreter (`{}`) stdout for environment configuration",
                interpreter.display()
            )
        })?;
        if !python_info.status.success() {
            let stderr = String::from_utf8(python_info.stderr)
                .unwrap_or("<Failed to parse STDOUT from UTF-8 string>".to_owned());
            return Err(anyhow::anyhow!(
                "Unable to query interpreter {} for environment info:\nSTDOUT: {}\nSTDERR: {}",
                interpreter.display(),
                stdout,
                stderr
            ));
        }

        let mut deserialized: PythonEnvironment = serde_json::from_str(&stdout)?;

        deserialized.python_platform.as_ref().ok_or_else(|| {
            anyhow!("Expected `python_platform` from Python interpreter query to be non-empty")
        })?;
        deserialized.python_version.as_ref().ok_or_else(|| {
            anyhow!("Expected `python_version` from Python interpreter query to be non-empty")
        })?;
        let site_package_path = deserialized
            .site_package_path
            .replace(Vec::new())
            .ok_or_else(|| {
                anyhow!(
                    "Expected `site_package_path` from Python interpreter query to be non-empty"
                )
            })?;
        deserialized.interpreter_site_package_path = site_package_path;

        Self::cache_interpreter_stdlib_path(deserialized.interpreter_stdlib_path.clone());

        Ok(deserialized)
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
        INTERPRETER_STDLIB_PATH_REGISTRY.write().extend(path);
    }

    /// todo(jvansch): Remove this once function is used to filter standard library files
    #[allow(dead_code)]
    pub fn get_interpreter_stdlib_path() -> &'static LazyLock<RwLock<SmallSet<PathBuf>>> {
        &INTERPRETER_STDLIB_PATH_REGISTRY
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
    fn test_override_empty_propagates_stdlib_path() {
        let mut env1 = PythonEnvironment {
            python_platform: None,
            python_version: None,
            site_package_path: None,
            interpreter_site_package_path: Vec::new(),
            interpreter_stdlib_path: Vec::new(),
        };

        let env2 = PythonEnvironment {
            python_platform: Some(PythonPlatform::mac()),
            python_version: Some(PythonVersion::new(3, 10, 0)),
            site_package_path: Some(vec![PathBuf::from("/path/to/site-packages")]),
            interpreter_site_package_path: vec![PathBuf::from("/path/to/site-packages")],
            interpreter_stdlib_path: vec![
                PathBuf::from("/usr/lib/python3.10"),
                PathBuf::from("/usr/lib/python3.10/lib-dynload"),
            ],
        };

        env1.override_empty(env2.clone());

        // Verify interpreter_stdlib_path is correctly propagated
        assert_eq!(env1.interpreter_stdlib_path, env2.interpreter_stdlib_path);
        assert_eq!(
            env1.interpreter_site_package_path,
            env2.interpreter_site_package_path
        );
    }
}
