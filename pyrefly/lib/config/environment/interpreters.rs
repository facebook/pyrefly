/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt::Display;
use std::path::Path;
use std::path::PathBuf;
use std::sync::LazyLock;

use serde::Deserialize;
use serde::Serialize;
#[cfg(not(target_arch = "wasm32"))]
use which::which;

use crate::config::environment::active_environment::ActiveEnvironment;
use crate::config::environment::conda;
use crate::config::environment::venv;
use crate::config::util::ConfigOrigin;

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize, Clone, Default)]
#[serde(rename_all = "kebab-case")]
pub struct Interpreters {
    // TODO(connernilsen): make this mutually exclusive with venv/conda env
    /// The python executable that will be queried for `python_version`,
    /// `python_platform`, or `site_package_path` if any of the values are missing.
    #[serde(
                default,
                skip_serializing_if = "ConfigOrigin::should_skip_serializing_option",
                // TODO(connernilsen): DON'T COPY THIS TO NEW FIELDS. This is a temporary
                // alias while we migrate existing fields from snake case to kebab case.
                alias = "python_interpreter"
            )]
    pub python_interpreter: Option<ConfigOrigin<PathBuf>>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conda_environment: Option<ConfigOrigin<String>>,

    /// Should we do any querying of an interpreter?
    #[serde(
        default,
        skip_serializing_if = "crate::config::util::skip_default_false"
    )]
    pub skip_interpreter_query: bool,
}

impl Display for Interpreters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self {
                skip_interpreter_query: true,
                ..
            } => write!(f, "<interpreter query skipped>"),
            Self {
                python_interpreter: None,
                ..
            } => write!(f, "<none found successfully>"),
            Self {
                conda_environment: Some(conda),
                python_interpreter: Some(path),
                ..
            } => write!(
                f,
                "conda environment {conda} with interpreter at {}",
                path.display()
            ),
            Self {
                python_interpreter: Some(path),
                ..
            } => write!(f, "{}", path.display()),
        }
    }
}

impl Interpreters {
    const DEFAULT_INTERPRETERS: &[&str] = &["python3", "python"];

    /// Finds interpreters by searching in prioritized locations for the given project
    /// and interpreter settings.
    ///
    /// The priorities are:
    /// 1. Check for an overridden `--python-interpreter` or `--conda-environment`
    /// 2. Check for an active venv or Conda environment
    /// 3. Check for a configured `python-interpreter`
    /// 4. Check for a configured `conda-environment`
    /// 5. Check for a `venv` in the current project
    /// 6. Use an interpreter we can find on the `$PATH`
    /// 7. Give up and return an error
    pub fn find_interpreter(&self, path: Option<&Path>) -> anyhow::Result<ConfigOrigin<PathBuf>> {
        if let Some(interpreter @ ConfigOrigin::CommandLine(_)) = &self.python_interpreter {
            return Ok(interpreter.clone());
        }

        if let Some(conda_env @ ConfigOrigin::CommandLine(_)) = &self.conda_environment {
            return conda_env
                .as_deref()
                .map(conda::find_interpreter_from_env)
                .transpose_err();
        }

        if let Some(active_env) = ActiveEnvironment::find() {
            return Ok(ConfigOrigin::auto(active_env));
        }

        if let Some(interpreter) = &self.python_interpreter {
            return Ok(interpreter.clone());
        }

        if let Some(conda_env) = &self.conda_environment {
            return conda_env
                .as_deref()
                .map(conda::find_interpreter_from_env)
                .transpose_err();
        }

        if let Some(start_path) = path
            && let Some(venv) = venv::find(start_path)
        {
            return Ok(ConfigOrigin::auto(venv));
        }

        if let Some(interpreter) = Self::get_default_interpreter() {
            return Ok(ConfigOrigin::auto(interpreter.to_path_buf()));
        }

        Err(anyhow::anyhow!(
            "Python environment (version, platform, or site-package-path) has value unset, \
                but no Python interpreter could be found to query for values. Falling back to \
                Pyrefly defaults for missing values."
        ))
    }

    /// Get the first interpreter available on the path by using `which`
    /// and querying for [`Self::DEFAULT_INTERPRETERS`] in order.
    pub fn get_default_interpreter() -> Option<&'static Path> {
        static SYSTEM_INTERP: LazyLock<Option<PathBuf>> = LazyLock::new(|| {
            // disable query with `which` on wasm
            #[cfg(not(target_arch = "wasm32"))]
            for binary_name in Interpreters::DEFAULT_INTERPRETERS {
                use std::process::Command;

                let Ok(binary_path) = which(binary_name) else {
                    continue;
                };
                let mut check = Command::new(&binary_path);
                check.arg("--version");
                if let Ok(output) = check.output()
                    && output.status.success()
                {
                    return Some(binary_path);
                }
            }
            None
        });
        SYSTEM_INTERP.as_deref()
    }
}

#[cfg(test)]
mod test {
    use pyrefly_util::test_path::TestPath;
    use tempfile::TempDir;
    use tempfile::tempdir;

    use super::*;

    fn setup_test_dir() -> TempDir {
        let tempdir = tempdir().unwrap();
        let root = tempdir.path();
        let interpreter_suffix = if cfg!(windows) { ".exe" } else { "" };
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "venv",
                vec![
                    TestPath::file(&format!("python3{}", interpreter_suffix)),
                    TestPath::file("pyvenv.cfg"),
                ],
            )],
        );
        tempdir
    }

    #[test]
    fn test_find_interpreter_precedence_python_interpreter_cli_highest_priority() {
        let tempdir = setup_test_dir();

        let python_interpreter = ConfigOrigin::cli(PathBuf::from("asdf"));

        let interpreters = Interpreters {
            python_interpreter: Some(python_interpreter.clone()),
            ..Default::default()
        };

        assert_eq!(
            interpreters.find_interpreter(Some(tempdir.path())).unwrap(),
            python_interpreter
        );
    }

    #[test]
    fn test_find_interpreter_precedence_conda_cli_highest_priority() {
        let tempdir = setup_test_dir();

        // this conda environment really shouldn't be able to exist
        let conda_environment = ConfigOrigin::cli("../././".to_owned());

        let interpreters = Interpreters {
            conda_environment: Some(conda_environment),
            ..Default::default()
        };

        assert!(interpreters.find_interpreter(Some(tempdir.path())).is_err());
    }

    #[test]
    fn test_find_interpreter_precedence_conda_config() {
        let tempdir = setup_test_dir();

        // this conda environment really shouldn't be able to exist
        let conda_environment = ConfigOrigin::config("../././".to_owned());

        let interpreters = Interpreters {
            conda_environment: Some(conda_environment),
            ..Default::default()
        };

        assert!(interpreters.find_interpreter(Some(tempdir.path())).is_err());
    }

    #[test]
    fn test_find_interpreter_precedence_venv() {
        let tempdir = setup_test_dir();

        let interpreters = Interpreters::default();
        let interpreter_suffix = if cfg!(windows) { ".exe" } else { "" };

        assert_eq!(
            interpreters.find_interpreter(Some(tempdir.path())).unwrap(),
            ConfigOrigin::auto(
                tempdir
                    .path()
                    .join(format!("venv/python3{}", interpreter_suffix))
            )
        );
    }
}
