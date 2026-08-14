/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt::Display;
use std::ops::Deref;
use std::path::Path;
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::process::Command;
use std::sync::LazyLock;

#[cfg(not(target_arch = "wasm32"))]
use anyhow::Context;
use serde::Deserialize;
use serde::Serialize;
use serde_with::skip_serializing_none;
#[cfg(not(target_arch = "wasm32"))]
use which::which;
#[cfg(not(target_arch = "wasm32"))]
use which::which_in;

use crate::environment::active_environment::ActiveEnvironment;
use crate::environment::conda;
use crate::environment::environment::PythonEnvironment;
use crate::environment::venv;
use crate::util::ConfigOrigin;

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize, Clone)]
#[serde(try_from = "Vec<String>", into = "Vec<String>")]
pub(crate) struct InterpreterDiscoveryCommand(Vec<String>);

impl TryFrom<Vec<String>> for InterpreterDiscoveryCommand {
    type Error = &'static str;

    fn try_from(parts: Vec<String>) -> Result<Self, Self::Error> {
        if parts.is_empty() {
            Err("`python-interpreter-find-cmd` must contain a program")
        } else {
            Ok(Self(parts))
        }
    }
}

impl From<InterpreterDiscoveryCommand> for Vec<String> {
    fn from(command: InterpreterDiscoveryCommand) -> Self {
        command.0
    }
}

impl Deref for InterpreterDiscoveryCommand {
    type Target = [String];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Display for InterpreterDiscoveryCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.join(" "))
    }
}

#[skip_serializing_none]
#[derive(Debug, PartialEq, Eq, Deserialize, Serialize, Clone, Default)]
#[serde(rename_all = "kebab-case")]
pub struct Interpreters {
    #[serde(
        skip_serializing_if = "ConfigOrigin::should_skip_serializing_option",
        // TODO(connernilsen): DON'T COPY THIS TO NEW FIELDS. This is a temporary
        // alias while we migrate existing fields from snake case to kebab case.
        alias = "python_interpreter",
        alias = "python-interpreter",
    )]
    pub(crate) python_interpreter_path: Option<ConfigOrigin<PathBuf>>,

    /// Should we turn a generic command into a `python_interpreter` path?
    pub(crate) fallback_python_interpreter_name: Option<ConfigOrigin<String>>,

    /// Command whose stdout is the path to the Python interpreter.
    pub(crate) python_interpreter_find_cmd: Option<InterpreterDiscoveryCommand>,

    pub(crate) conda_environment: Option<ConfigOrigin<String>>,

    /// Should we do any querying of an interpreter?
    #[serde(default, skip_serializing_if = "crate::util::skip_default_false")]
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
                python_interpreter_path: None,
                python_interpreter_find_cmd: Some(cmd),
                ..
            } => write!(f, "interpreter from command `{cmd}`"),
            Self {
                python_interpreter_path: None,
                ..
            } => write!(f, "<none found successfully>"),
            Self {
                fallback_python_interpreter_name: Some(cmd),
                python_interpreter_path: Some(path),
                ..
            } => write!(
                f,
                "interpreter at path {} (from `which {cmd}`)",
                path.display(),
            ),
            Self {
                python_interpreter_find_cmd: Some(cmd),
                python_interpreter_path: Some(path),
                ..
            } => write!(
                f,
                "interpreter at path {} (from command `{cmd}`)",
                path.display(),
            ),
            Self {
                conda_environment: Some(conda),
                python_interpreter_path: Some(path),
                ..
            } => write!(
                f,
                "conda environment {conda} with interpreter at {}",
                path.display()
            ),
            Self {
                python_interpreter_path: Some(path),
                ..
            } => write!(f, "{}", path.display()),
        }
    }
}

impl Interpreters {
    const DEFAULT_INTERPRETERS: &[&str] = &["python3", "python"];

    /// Checks if any interpreter is currently set, typically used when determining
    /// if the config or CLI overrides explicitly specified a config to figure out
    /// if we should respect an IDE-supplied interpreter preference.
    pub fn is_empty(&self) -> bool {
        self.python_interpreter_path.is_none()
            && self.python_interpreter_find_cmd.is_none()
            && self.conda_environment.is_none()
            && self.fallback_python_interpreter_name.is_none()
    }

    pub fn set_lsp_python_interpreter(&mut self, interpreter: PathBuf) {
        self.python_interpreter_path = Some(ConfigOrigin::lsp(interpreter));
    }

    /// Finds interpreters by searching in prioritized locations for the given project
    /// and interpreter settings.
    ///
    /// The priorities are:
    /// 1. Check for an overridden interpreter or Conda environment from the CLI.
    /// 2. Check for a configured interpreter path, discovery command, or Conda environment.
    /// 3. Check for an IDE / LSP provided `python-interpreter`.
    /// 4. Check for an active venv or Conda environment.
    /// 5. Check for a `venv` in the current project.
    /// 6. Use an interpreter we can find on the `$PATH`.
    /// 7. Give up and return an error.
    pub(crate) fn find_interpreter(
        &self,
        path: Option<&Path>,
    ) -> anyhow::Result<ConfigOrigin<PathBuf>> {
        let python_interpreter = self.interpreter_path_or_cmd()?;
        if let Some(interpreter @ ConfigOrigin::CommandLine(_)) = python_interpreter {
            return Ok(interpreter);
        }
        if let Some(conda_env @ ConfigOrigin::CommandLine(_)) = &self.conda_environment {
            return conda_env
                .as_deref()
                .map(conda::find_interpreter_from_env)
                .transpose_err();
        }

        if let Some(interpreter @ ConfigOrigin::ConfigFile(_)) = python_interpreter {
            return Ok(interpreter);
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(command) = self.python_interpreter_find_cmd.as_deref() {
            let interpreter = Self::find_interpreter_from_command(command, path)?;
            return Ok(ConfigOrigin::auto(interpreter));
        }
        #[cfg(target_arch = "wasm32")]
        if self.python_interpreter_find_cmd.is_some() {
            return Err(anyhow::anyhow!(
                "`python-interpreter-find-cmd` is not supported on WebAssembly"
            ));
        }
        if let Some(conda_env @ ConfigOrigin::ConfigFile(_)) = &self.conda_environment {
            return conda_env
                .as_deref()
                .map(conda::find_interpreter_from_env)
                .transpose_err();
        }

        if let Some(interpreter @ ConfigOrigin::Lsp(_)) = python_interpreter {
            return Ok(interpreter);
        }

        // fallback, just in case an 'auto' interpreter or conda env is set, though
        // it shouldn't be (except in tests below)
        if let Some(interpreter) = python_interpreter {
            return Ok(interpreter);
        }
        if let Some(conda_env) = &self.conda_environment {
            return conda_env
                .as_deref()
                .map(conda::find_interpreter_from_env)
                .transpose_err();
        }

        if let Some(active_env) = ActiveEnvironment::find() {
            return Ok(ConfigOrigin::auto(active_env));
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

    fn interpreter_path_or_cmd(&self) -> anyhow::Result<Option<ConfigOrigin<PathBuf>>> {
        if self.python_interpreter_path.is_some() {
            return Ok(self.python_interpreter_path.clone());
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(cmd) = &self.fallback_python_interpreter_name {
            fn which_to_anyhow_err(cmd: &String) -> anyhow::Result<PathBuf> {
                Ok(which(cmd)?)
            }
            return Ok(Some(cmd.as_ref().map(which_to_anyhow_err).transpose_err()?));
        }
        Ok(self.python_interpreter_path.clone())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn find_interpreter_from_command(
        command_parts: &[String],
        working_directory: Option<&Path>,
    ) -> anyhow::Result<PathBuf> {
        let program = &command_parts[0];
        let args = &command_parts[1..];
        let program = match working_directory {
            Some(root) => which_in(program, std::env::var_os("PATH"), root),
            None => which(program),
        }
        .with_context(|| "Could not resolve the Python interpreter discovery command")?;

        let mut command = Command::new(program);
        command.args(args);
        if let Some(working_directory) = working_directory {
            command.current_dir(working_directory);
        }
        let output = command.output().with_context(|| {
            format!(
                "Failed to run Python interpreter discovery command `{}`",
                command_parts.join(" ")
            )
        })?;
        if !output.status.success() {
            return Err(anyhow::anyhow!(
                "Python interpreter discovery command `{}` failed with status {}: {}",
                command_parts.join(" "),
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }

        let stdout = String::from_utf8(output.stdout)
            .context("Python interpreter discovery command output was not valid UTF-8")?;
        let interpreter = stdout.trim();
        if interpreter.is_empty() {
            return Err(anyhow::anyhow!(
                "Python interpreter discovery command `{}` returned an empty path",
                command_parts.join(" ")
            ));
        }
        if interpreter.lines().count() != 1 {
            return Err(anyhow::anyhow!(
                "Python interpreter discovery command `{}` must return exactly one path",
                command_parts.join(" ")
            ));
        }

        let mut interpreter = PathBuf::from(interpreter);
        if interpreter.is_relative()
            && let Some(working_directory) = working_directory
        {
            interpreter = working_directory.join(interpreter);
        }
        Ok(interpreter)
    }

    /// Get the first executable interpreter available on the path.
    ///
    /// Query the interpreter environment as the validation step. The result is cached, so the
    /// caller's environment lookup does not spawn the same interpreter a second time.
    pub(crate) fn get_default_interpreter() -> Option<&'static Path> {
        static SYSTEM_INTERP: LazyLock<Option<PathBuf>> = LazyLock::new(|| {
            // disable query with `which` on wasm
            #[cfg(not(target_arch = "wasm32"))]
            for binary_name in Interpreters::DEFAULT_INTERPRETERS {
                if let Ok(binary_path) = which(binary_name) {
                    let (_, error) = PythonEnvironment::get_interpreter_env(&binary_path);
                    if error.is_none() {
                        return Some(binary_path);
                    }
                }
            }
            None
        });
        SYSTEM_INTERP.as_deref()
    }
}

#[cfg(test)]
mod test {
    #[cfg(windows)]
    use std::fs;

    use pyrefly_util::test_path::TestPath;
    use tempfile::TempDir;
    use tempfile::tempdir;

    use super::*;

    fn test_venv_interpreter_name() -> &'static str {
        if cfg!(windows) {
            "python.exe"
        } else {
            "python3"
        }
    }

    fn setup_test_dir() -> TempDir {
        let tempdir = tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "venv",
                vec![
                    TestPath::file(test_venv_interpreter_name()),
                    TestPath::file("pyvenv.cfg"),
                ],
            )],
        );
        tempdir
    }

    /// Produces a conda environment name that should not actually be possible in conda.
    fn fake_conda_name() -> String {
        "../././".to_owned()
    }

    fn discovery_command(parts: &[&str]) -> InterpreterDiscoveryCommand {
        InterpreterDiscoveryCommand::try_from(
            parts
                .iter()
                .map(|part| (*part).to_owned())
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }

    #[test]
    fn test_find_interpreter_precedence_cli_highest_priority() {
        let tempdir = setup_test_dir();

        let python_interpreter = ConfigOrigin::cli(PathBuf::from("asdf"));
        let conda_environment = ConfigOrigin::config("somecondaenv".to_owned());

        let interpreters = Interpreters {
            python_interpreter_path: Some(python_interpreter.clone()),
            conda_environment: Some(conda_environment.clone()),
            ..Default::default()
        };

        assert_eq!(
            interpreters.find_interpreter(Some(tempdir.path())).unwrap(),
            python_interpreter
        );

        let conda_environment = ConfigOrigin::cli(fake_conda_name());
        let interpreters = Interpreters {
            python_interpreter_path: Some(ConfigOrigin::config(PathBuf::from("asdf"))),
            conda_environment: Some(conda_environment.clone()),
            ..Default::default()
        };

        let found_interpreter = interpreters.find_interpreter(Some(tempdir.path()));
        // we check for blanket errors, since we'll either get an error that the environment
        // doesn't exist (since it can't be named that) or that conda doesn't exist, which is
        // still an indication the logic works
        assert!(found_interpreter.is_err());
    }

    #[test]
    fn test_find_interpreter_precedence_config_second_highest_priority() {
        let tempdir = setup_test_dir();

        let python_interpreter = ConfigOrigin::config(PathBuf::from("asdf"));
        let conda_environment = ConfigOrigin::lsp("somecondaenv".to_owned());

        let interpreters = Interpreters {
            python_interpreter_path: Some(python_interpreter.clone()),
            conda_environment: Some(conda_environment.clone()),
            ..Default::default()
        };

        assert_eq!(
            interpreters.find_interpreter(Some(tempdir.path())).unwrap(),
            python_interpreter
        );

        let interpreters = Interpreters {
            python_interpreter_path: Some(python_interpreter.clone()),
            python_interpreter_find_cmd: Some(discovery_command(&["does-not-exist"])),
            ..Default::default()
        };
        assert_eq!(
            interpreters.find_interpreter(Some(tempdir.path())).unwrap(),
            python_interpreter
        );

        let conda_environment = ConfigOrigin::config(fake_conda_name());
        let interpreters = Interpreters {
            python_interpreter_path: Some(ConfigOrigin::lsp(PathBuf::from("asdf"))),
            conda_environment: Some(conda_environment.clone()),
            ..Default::default()
        };

        let found_interpreter = interpreters.find_interpreter(Some(tempdir.path()));
        // we check for blanket errors, since we'll either get an error that the environment
        // doesn't exist (since it can't be named that) or that conda doesn't exist, which is
        // still an indication the logic works
        assert!(found_interpreter.is_err());
    }

    #[test]
    fn test_find_interpreter_precedence_lsp_third_highest_priority() {
        let tempdir = setup_test_dir();

        let python_interpreter = ConfigOrigin::config(PathBuf::from("asdf"));
        let conda_environment = ConfigOrigin::auto("somecondaenv".to_owned());

        let interpreters = Interpreters {
            python_interpreter_path: Some(python_interpreter.clone()),
            conda_environment: Some(conda_environment.clone()),
            ..Default::default()
        };

        assert_eq!(
            interpreters.find_interpreter(Some(tempdir.path())).unwrap(),
            python_interpreter
        );

        let conda_environment = ConfigOrigin::config(fake_conda_name());
        let interpreters = Interpreters {
            python_interpreter_path: Some(ConfigOrigin::auto(PathBuf::from("asdf"))),
            conda_environment: Some(conda_environment.clone()),
            ..Default::default()
        };

        let found_interpreter = interpreters.find_interpreter(Some(tempdir.path()));
        // we check for blanket errors, since we'll either get an error that the environment
        // doesn't exist (since it can't be named that) or that conda doesn't exist, which is
        // still an indication the logic works
        assert!(found_interpreter.is_err());
    }

    #[test]
    fn test_find_interpreter_precedence_venv() {
        let tempdir = setup_test_dir();

        let interpreters = Interpreters::default();

        unsafe {
            // clear this variable if it exists, since we can't test that in unit tests.
            // no other threads should ever test behavior around this
            std::env::remove_var(venv::ENV_VAR);
        }

        assert_eq!(
            interpreters.find_interpreter(Some(tempdir.path())).unwrap(),
            ConfigOrigin::auto(
                tempdir
                    .path()
                    .join("venv")
                    .join(test_venv_interpreter_name())
            )
        );
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn test_interpreter_find_command_resolves_relative_output() {
        let tempdir = tempdir().unwrap();

        #[cfg(unix)]
        let command = discovery_command(&["sh", "-c", "printf 'venv/bin/python\\n'"]);
        #[cfg(windows)]
        let command = discovery_command(&["cmd", "/C", "echo venv\\Scripts\\python.exe"]);

        let interpreters = Interpreters {
            python_interpreter_find_cmd: Some(command),
            conda_environment: Some(ConfigOrigin::config(fake_conda_name())),
            ..Default::default()
        };

        #[cfg(unix)]
        let expected = tempdir.path().join("venv/bin/python");
        #[cfg(windows)]
        let expected = tempdir.path().join("venv\\Scripts\\python.exe");

        assert_eq!(
            interpreters.find_interpreter(Some(tempdir.path())).unwrap(),
            ConfigOrigin::auto(expected)
        );
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn test_interpreter_find_command_uses_working_directory() {
        let tempdir = tempdir().unwrap();

        #[cfg(unix)]
        let command = discovery_command(&["sh", "-c", "pwd"]);
        #[cfg(windows)]
        let command = discovery_command(&["cmd", "/C", "cd"]);

        let interpreters = Interpreters {
            python_interpreter_find_cmd: Some(command),
            ..Default::default()
        };
        let interpreter = interpreters.find_interpreter(Some(tempdir.path())).unwrap();

        assert_eq!(
            interpreter.as_path().canonicalize().unwrap(),
            tempdir.path().canonicalize().unwrap(),
        );
    }

    #[cfg(windows)]
    #[test]
    fn test_interpreter_find_command_resolves_relative_program() {
        let tempdir = tempdir().unwrap();
        let tools = tempdir.path().join("tools");
        fs::create_dir(&tools).unwrap();
        fs::copy(which("cmd").unwrap(), tools.join("find-python.exe")).unwrap();

        let interpreters = Interpreters {
            python_interpreter_find_cmd: Some(discovery_command(&[
                "tools\\find-python.exe",
                "/C",
                "echo venv\\Scripts\\python.exe",
            ])),
            ..Default::default()
        };

        assert_eq!(
            interpreters.find_interpreter(Some(tempdir.path())).unwrap(),
            ConfigOrigin::auto(tempdir.path().join("venv\\Scripts\\python.exe")),
        );
    }

    #[cfg(unix)]
    #[test]
    fn test_interpreter_find_command_must_return_one_path() {
        let interpreters = Interpreters {
            python_interpreter_find_cmd: Some(discovery_command(&[
                "sh",
                "-c",
                "printf 'first\\nsecond\\n'",
            ])),
            ..Default::default()
        };

        let error = interpreters.find_interpreter(None).unwrap_err();
        assert!(error.to_string().contains("must return exactly one path"));
    }

    #[cfg(unix)]
    #[test]
    fn test_interpreter_find_command_reports_failure() {
        let interpreters = Interpreters {
            python_interpreter_find_cmd: Some(discovery_command(&[
                "sh",
                "-c",
                "printf 'manager failed' >&2; exit 7",
            ])),
            ..Default::default()
        };

        let error = interpreters.find_interpreter(None).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("failed with status"));
        assert!(message.contains("manager failed"));
    }

    #[test]
    fn test_interpreter_display_includes_discovery_command() {
        let interpreters = Interpreters {
            python_interpreter_path: Some(ConfigOrigin::auto(PathBuf::from("/resolved/python"))),
            python_interpreter_find_cmd: Some(discovery_command(&["poetry", "env", "info", "-e"])),
            ..Default::default()
        };

        assert_eq!(
            interpreters.to_string(),
            "interpreter at path /resolved/python (from command `poetry env info -e`)"
        );
    }
}
