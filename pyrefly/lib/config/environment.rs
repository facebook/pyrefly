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
use serde::Deserialize;
use serde::Serialize;
use starlark_map::small_map::SmallMap;
use tracing::error;
#[cfg(not(target_arch = "wasm32"))]
use which::which;

use crate::PythonVersion;
use crate::metadata::PythonPlatform;
use crate::util::lock::Mutex;

static INTERPRETER_ENV_REGISTRY: LazyLock<Mutex<SmallMap<PathBuf, Option<PythonEnvironment>>>> =
    LazyLock::new(|| Mutex::new(SmallMap::new()));

/// Values representing the environment of the Python interpreter.
/// These values are `None` by default, so we can tell if a config
/// overrode them, or if we should query a Python interpreter for
/// any missing values. We can't query a Python interpreter
/// on config parsing, since we also won't know if an executable
/// other than the first available on the path should be used (i.e.
/// should we always look at a venv/conda environment instead?)
#[derive(Debug, PartialEq, Eq, Deserialize, Serialize, Clone)]
pub struct PythonEnvironment {
    #[serde(default)]
    pub python_platform: Option<PythonPlatform>,

    #[serde(default)]
    pub python_version: Option<PythonVersion>,

    #[serde(default)]
    pub site_package_path: Option<Vec<PathBuf>>,
}

impl PythonEnvironment {
    const DEFAULT_INTERPRETERS: &[&str] = &["python3", "python"];

    pub fn new(
        python_platform: PythonPlatform,
        python_version: PythonVersion,
        site_package_path: Vec<PathBuf>,
    ) -> Self {
        Self {
            python_platform: Some(python_platform),
            python_version: Some(python_version),
            site_package_path: Some(site_package_path),
        }
    }

    pub fn any_empty(&self) -> bool {
        self.python_platform.is_none()
            || self.python_version.is_none()
            || self.site_package_path.is_none()
    }

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
    }

    fn get_env_from_interpreter(interpreter: &Path) -> anyhow::Result<PythonEnvironment> {
        let script = "\
import json, site, sys
platform = sys.platform
v = sys.version_info
version = '{}.{}.{}'.format(v.major, v.minor, v.micro)
packages = site.getsitepackages()
print(json.dumps({'python_platform': platform, 'python_version': version, 'site_package_path': packages}))
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

        let deserialized: PythonEnvironment = serde_json::from_str(&stdout)?;

        deserialized.python_platform.as_ref().ok_or_else(|| {
            anyhow!("Expected `python_platform` from Python interpreter query to be non-empty")
        })?;
        deserialized.python_version.as_ref().ok_or_else(|| {
            anyhow!("Expected `python_version` from Python interpreter query to be non-empty")
        })?;
        deserialized.site_package_path.as_ref().ok_or_else(|| {
            anyhow!("Expected `site_package_path` from Python interpreter query to be non-empty")
        })?;

        Ok(deserialized)
    }

    pub fn get_default_interpreter() -> Option<PathBuf> {
        // disable query with `which` on wasm
        #[cfg(not(target_arch = "wasm32"))]
        for interpreter in Self::DEFAULT_INTERPRETERS {
            if let Ok(interpreter_path) = which(interpreter) {
                return Some(interpreter_path);
            }
        }
        None
    }

    pub fn get_interpreter_env(interpreter: &Path) -> PythonEnvironment {
        INTERPRETER_ENV_REGISTRY.lock()
        .entry(interpreter.to_path_buf()).or_insert_with(move || {
            Self::get_env_from_interpreter(interpreter).inspect_err(|e| {
                error!("Failed to query interpreter, falling back to default Python environment settings\n{}", e);
            }).ok()
        }).clone().unwrap_or_default()
    }
}

impl Default for PythonEnvironment {
    /// This supplies Pyrefly's backup default values if we are unable to query
    /// an interpreter or want to have a `PythonEnvironment` in testing.
    /// Prefer to query an interpreter if possible.
    fn default() -> Self {
        Self::new(
            PythonPlatform::default(),
            PythonVersion::default(),
            Vec::new(),
        )
    }
}

impl Display for PythonEnvironment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{{python_platform: {}, python_version: {}, site_package_path: [{}]}}",
            self.python_platform
                .as_ref()
                .map_or_else(|| "None".to_owned(), |platform| platform.to_string()),
            self.python_version
                .map_or_else(|| "None".to_owned(), |version| version.to_string()),
            self.site_package_path.as_ref().map_or_else(
                || "".to_owned(),
                |path| path.iter().map(|p| p.display()).join(", ")
            )
        )
    }
}
