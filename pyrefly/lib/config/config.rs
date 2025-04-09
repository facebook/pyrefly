/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ffi::OsStr;
use std::fmt;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use anyhow::anyhow;
use anyhow::Context;
use itertools::Itertools;
use path_absolutize::Absolutize;
use serde::Deserialize;
use serde::Serialize;
use toml::Table;
use tracing::debug;
use tracing::warn;

use crate::config::error::ErrorDisplayConfig;
use crate::config::util::ExtraConfigs;
use crate::globs::Globs;
use crate::metadata::PythonVersion;
use crate::metadata::RuntimeMetadata;
use crate::module::wildcard::ModuleWildcard;
use crate::PythonEnvironment;

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize, Clone)]
pub struct ConfigFile {
    /// Files that should be counted as sources (e.g. user-space code).
    /// NOTE: this is never replaced with CLI args in this config, but may be overridden by CLI args where used.
    #[serde(default = "ConfigFile::default_project_includes")]
    pub project_includes: Globs,

    /// Files that should be excluded as sources (e.g. user-space code). These take
    /// precedence over `project_includes`.
    /// NOTE: this is never replaced with CLI args in this config, but may be overridden by CLI args where used.
    #[serde(default = "ConfigFile::default_project_excludes")]
    pub project_excludes: Globs,

    /// analyze function body and infer return type
    #[serde(default)]
    pub skip_untyped_functions: bool,

    /// corresponds to --search-path in Args, the list of directories where imports are
    /// found (including type checked files).
    #[serde(default = "ConfigFile::default_search_path")]
    pub search_path: Vec<PathBuf>,

    // TODO(connernilsen): make this mutually exclusive with venv/conda env
    #[serde(default = "PythonEnvironment::get_default_interpreter")]
    pub python_interpreter: Option<PathBuf>,

    /// Values representing the environment of the Python interpreter
    /// (which platform, Python version, ...). When we parse, these values
    /// are set to false so we know to query the `python_interpreter` before falling
    /// back to Pyrefly's defaults.
    #[serde(flatten)]
    pub python_environment: PythonEnvironment,

    #[serde(default)]
    pub errors: ErrorDisplayConfig,

    /// String-prefix-matched names of modules from which import errors should be ignored
    /// and the module should always be replaced with `typing.Any`
    #[serde(default)]
    pub replace_imports_with_any: Vec<ModuleWildcard>,

    /// Whether to ignore type errors in generated code. By default this is disabled.
    /// Generated code is defined as code that contains the marker string `@` immediately followed by `generated`.
    #[serde(default)]
    pub ignore_errors_in_generated_code: bool,

    /// Any unknown config items
    #[serde(default, flatten)]
    pub extras: ExtraConfigs,
}

impl Default for ConfigFile {
    /// Gets a default ConfigFile, with all paths rewritten relative to cwd.
    fn default() -> ConfigFile {
        let mut result = Self::default_no_path_rewrite();
        match std::env::current_dir() {
            Ok(cwd) => {
                result.rewrite_with_path_to_config(&cwd);
            }
            Err(err) => {
                debug!(
                    "Cannot identify current dir for default config path rewriting: {}",
                    err
                );
            }
        }
        result
    }
}

impl ConfigFile {
    /// Gets a default ConfigFile, with no path rewriting. This should only be used for unit testing,
    /// since it may have strange runtime behavior. Prefer to use `ConfigFile::default()` instead.
    fn default_no_path_rewrite() -> Self {
        ConfigFile {
            search_path: Self::default_search_path(),
            python_environment: PythonEnvironment {
                python_platform: None,
                python_version: None,
                site_package_path: None,
            },
            project_includes: Self::default_project_includes(),
            project_excludes: Self::default_project_excludes(),
            skip_untyped_functions: false,
            python_interpreter: PythonEnvironment::get_default_interpreter(),
            errors: ErrorDisplayConfig::default(),
            ignore_errors_in_generated_code: false,
            extras: Self::default_extras(),
            replace_imports_with_any: Vec::new(),
        }
    }
}

impl ConfigFile {
    pub const CONFIG_FILE_NAME: &str = "pyrefly.toml";
    pub const PYPROJECT_FILE_NAME: &str = "pyproject.toml";
    pub const CONFIG_FILE_NAMES: [&str; 2] = [Self::CONFIG_FILE_NAME, Self::PYPROJECT_FILE_NAME];

    pub fn default_project_includes() -> Globs {
        Globs::new(vec!["**/*".to_owned()])
    }

    pub fn default_project_excludes() -> Globs {
        Globs::new(vec![
            "**/__pycache__/**".to_owned(),
            // match any hidden file, but don't match `.` or `..` (equivalent to regex: `\.[^/\.]{0,1}.*`)
            "**/.[!/.]*".to_owned(),
        ])
    }

    pub fn default_search_path() -> Vec<PathBuf> {
        vec![PathBuf::from("")]
    }

    pub fn default_extras() -> ExtraConfigs {
        ExtraConfigs(Table::new())
    }

    pub fn python_version(&self) -> PythonVersion {
        self.python_environment.python_version()
    }

    pub fn python_platform(&self) -> &str {
        self.python_environment.python_platform()
    }

    pub fn site_package_path(&self) -> &[PathBuf] {
        self.python_environment.site_package_path()
    }

    pub fn get_runtime_metadata(&self) -> RuntimeMetadata {
        self.python_environment.get_runtime_metadata()
    }

    pub fn default_error_config() -> ErrorDisplayConfig {
        ErrorDisplayConfig::default()
    }

    /// Configures values that must be updated *after* overwriting with CLI flag values,
    /// which should probably be everything except for `PathBuf` or `Globs` types.
    pub fn configure(&mut self) {
        let env = &mut self.python_environment;

        let env_has_empty = env.python_version.is_none()
            || env.python_platform.is_none()
            || env.site_package_path.is_none();

        if env_has_empty && let Some(interpreter) = &self.python_interpreter {
            let system_env = PythonEnvironment::get_interpreter_env(interpreter);

            if env.python_version.is_none() {
                env.python_version = system_env.python_version;
            }
            if env.python_platform.is_none() {
                env.python_platform = system_env.python_platform;
            }
            if env.site_package_path.is_none() {
                env.site_package_path = system_env.site_package_path;
            }
        } else if env_has_empty {
            warn!(
                "Python environment (version, platform, or site_package_path) has value unset, \
                but no Python interpreter could be found to query for values. Falling back to \
                Pyrefly defaults for missing values."
            )
        }
    }

    /// Rewrites any config values that must be updated *before* applying CLI flag values, namely
    /// rewriting any `PathBuf`s and `Globs` to be relative to `config_root`.
    /// We do this as a step separate from `configure()` because CLI args may override some of these
    /// values, but CLI args will always be relative to CWD, whereas config values should be relative
    /// to the config root.
    fn rewrite_with_path_to_config(&mut self, config_root: &Path) {
        // TODO(connernilsen): store root as part of config to make it easier to rewrite later on
        self.project_includes = self.project_includes.clone().from_root(config_root);
        self.search_path.iter_mut().for_each(|search_root| {
            let mut base = config_root.to_path_buf();
            base.push(search_root.as_path());
            *search_root = base;
        });
        // push config to search path to make sure we can fall back to the config directory as an import path
        // if users forget to add it
        self.search_path.push(config_root.to_path_buf());
        self.python_environment
            .site_package_path
            .iter_mut()
            .for_each(|v| {
                v.iter_mut().for_each(|site_package_path| {
                    let mut with_base = config_root.to_path_buf();
                    with_base.push(site_package_path.as_path());
                    *site_package_path = with_base;
                });
            });
        self.project_excludes = self.project_excludes.clone().from_root(config_root);
        self.python_interpreter = self
            .python_interpreter
            .as_ref()
            .map(|i| config_root.join(i));
    }

    pub fn validate(&self) {
        fn warn_on_invalid(p: &Path, field: &str) {
            match p.try_exists() {
                Ok(true) => return,
                Err(err) => {
                    debug!(
                        "Error checking for existence of path {}: {}",
                        p.display(),
                        err
                    );
                    return;
                }
                _ => (),
            }
            let p = if p == Path::new("") {
                Path::new("./")
            } else {
                p
            };
            warn!("Nonexistent `{field}` found: {}", p.display());
        }
        self.python_environment
            .site_package_path
            .as_ref()
            .inspect(|p| {
                p.iter()
                    .for_each(|p| warn_on_invalid(p, "site_package_path"))
            });
        self.search_path
            .iter()
            .for_each(|p| warn_on_invalid(p, "search_path"));
    }

    pub fn from_file(config_path: &Path, error_on_extras: bool) -> anyhow::Result<ConfigFile> {
        let config_path = config_path
            .absolutize()
            .with_context(|| format!("Path `{}` cannot be absolutized", config_path.display()))?
            .into_owned();
        let config_str = fs::read_to_string(&config_path)?;
        let mut config = if config_path.file_name() == Some(OsStr::new(&Self::PYPROJECT_FILE_NAME))
        {
            Self::parse_pyproject_toml(&config_str)
        } else {
            Self::parse_config(&config_str)
        }?;

        if error_on_extras && !config.extras.0.is_empty() {
            let extra_keys = config.extras.0.keys().join(", ");
            return Err(anyhow!("Extra keys found in config: {extra_keys}"));
        }

        if let Some(config_root) = config_path.parent() {
            config.rewrite_with_path_to_config(config_root);
        }

        Ok(config)
    }

    fn parse_config(config_str: &str) -> anyhow::Result<ConfigFile> {
        toml::from_str::<ConfigFile>(config_str).map_err(|err| anyhow::Error::msg(err.to_string()))
    }

    fn parse_pyproject_toml(config_str: &str) -> anyhow::Result<ConfigFile> {
        #[derive(Debug, Deserialize)]
        struct PyProject {
            #[serde(default)]
            pub tool: Option<Tool>,
        }

        #[derive(Debug, Deserialize)]
        struct Tool {
            #[serde(default)]
            pub pyrefly: Option<ConfigFile>,
        }

        let maybe_config = toml::from_str::<PyProject>(config_str)
            .map_err(|err| anyhow::Error::msg(err.to_string()))?
            .tool
            .and_then(|c| c.pyrefly);
        Ok(maybe_config.unwrap_or_else(ConfigFile::default))
    }
}

impl fmt::Display for ConfigFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{{project_includes: {}, project_excludes: {}, search_path: [{}], python_interpreter: {:?}, python_environment: {}, replace_imports_with_any: [{}]}}",
            self.project_includes,
            self.project_excludes,
            self.search_path.iter().map(|p| p.display()).join(", "),
            self.python_interpreter,
            self.python_environment,
            self.replace_imports_with_any
                .iter()
                .map(|p| p.as_str())
                .join(", "),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path;

    use toml::Value;

    use super::*;
    use crate::error::kind::ErrorKind;

    #[test]
    fn deserialize_pyrefly_config() {
        let config_str = r#"
            project_includes = ["tests", "./implementation"]
            project_excludes = ["tests/untyped/**"]
            skip_untyped_functions = false
            search_path = ["../.."]
            python_platform = "darwin"
            python_version = "1.2.3"
            site_package_path = ["venv/lib/python1.2.3/site-packages"]
            python_interpreter = "venv/my/python"
            replace_imports_with_any = ["fibonacci"]
            ignore_errors_in_generated_code = true
            [errors]
            assert-type = true
            bad-return = false
        "#;
        let config = ConfigFile::parse_config(config_str).unwrap();
        assert_eq!(
            config,
            ConfigFile {
                project_includes: Globs::new(vec![
                    "tests".to_owned(),
                    "./implementation".to_owned()
                ]),
                project_excludes: Globs::new(vec!["tests/untyped/**".to_owned()]),
                skip_untyped_functions: false,
                search_path: vec![PathBuf::from("../..")],
                python_environment: PythonEnvironment::new(
                    "darwin".to_owned(),
                    PythonVersion::new(1, 2, 3),
                    vec![PathBuf::from("venv/lib/python1.2.3/site-packages")],
                ),
                python_interpreter: Some(PathBuf::from("venv/my/python")),
                extras: ConfigFile::default_extras(),
                errors: ErrorDisplayConfig::new(HashMap::from_iter([
                    (ErrorKind::AssertType, true),
                    (ErrorKind::BadReturn, false)
                ])),
                ignore_errors_in_generated_code: true,
                replace_imports_with_any: vec![ModuleWildcard::new("fibonacci").unwrap()],
            },
        );
    }

    #[test]
    fn deserialize_pyrefly_config_defaults() {
        let config_str = "";
        let config = ConfigFile::parse_config(config_str).unwrap();
        assert_eq!(config, ConfigFile::default_no_path_rewrite());
    }

    #[test]
    fn deserialize_pyrefly_config_with_unknown() {
        let config_str = r#"
            laszewo = "good kids"
            python_platform = "windows"
        "#;
        let config = ConfigFile::parse_config(config_str).unwrap();
        assert_eq!(
            config.extras.0,
            Table::from_iter([("laszewo".to_owned(), Value::String("good kids".to_owned()))])
        );
    }

    #[test]
    fn deserialize_pyproject_toml() {
        let config_str = r#"
            [tool.pyrefly]
            project_includes = ["./tests", "./implementation"]
            python_platform = "darwin"
            python_version = "1.2.3"
        "#;
        let config = ConfigFile::parse_pyproject_toml(config_str).unwrap();
        assert_eq!(
            config,
            ConfigFile {
                project_includes: Globs::new(vec![
                    "./tests".to_owned(),
                    "./implementation".to_owned()
                ]),
                python_environment: PythonEnvironment {
                    python_platform: Some("darwin".to_owned()),
                    python_version: Some(PythonVersion::new(1, 2, 3)),
                    site_package_path: None,
                },
                ..ConfigFile::default_no_path_rewrite()
            }
        );
    }

    #[test]
    fn deserialize_pyproject_toml_defaults() {
        let config_str = "";
        let config = ConfigFile::parse_pyproject_toml(config_str).unwrap();
        assert_eq!(config, ConfigFile::default());
    }

    #[test]
    fn deserialize_pyproject_toml_with_unknown() {
        let config_str = r#"
            top_level = 1
            [table1]
            table1_value = 2
            [tool.pysa]
            pysa_value = 2
            [tool.pyrefly]
            python_version = "1.2.3"
        "#;
        let config = ConfigFile::parse_pyproject_toml(config_str).unwrap();
        assert_eq!(
            config,
            ConfigFile {
                python_environment: PythonEnvironment {
                    python_version: Some(PythonVersion::new(1, 2, 3)),
                    python_platform: None,
                    site_package_path: None,
                },
                ..ConfigFile::default_no_path_rewrite()
            }
        );
    }

    #[test]
    fn deserialize_pyproject_toml_without_pyrefly() {
        let config_str = "
            top_level = 1
            [table1]
            table1_value = 2
            [tool.pysa]
            pysa_value = 2
        ";
        let config = ConfigFile::parse_pyproject_toml(config_str).unwrap();
        assert_eq!(config, ConfigFile::default());
    }

    #[test]
    fn deserialize_pyproject_toml_with_unknown_in_pyrefly() {
        let config_str = r#"
            top_level = 1
            [table1]
            table1_value = 2
            [tool.pysa]
            pysa_value = 2
            [tool.pyrefly]
            python_version = "1.2.3"
            inzo = "overthinker"
        "#;
        let config = ConfigFile::parse_pyproject_toml(config_str).unwrap();
        assert_eq!(
            config.extras.0,
            Table::from_iter([("inzo".to_owned(), Value::String("overthinker".to_owned()))])
        );
    }

    #[test]
    fn test_rewrite_with_path_to_config() {
        fn with_sep(s: &str) -> String {
            s.replace("/", path::MAIN_SEPARATOR_STR)
        }
        let mut python_environment = PythonEnvironment {
            site_package_path: Some(vec![PathBuf::from("venv/lib/python1.2.3/site-packages")]),
            ..PythonEnvironment::default()
        };
        let interpreter = "venv/bin/python3".to_owned();
        let mut config = ConfigFile {
            project_includes: Globs::new(vec!["path1/**".to_owned(), "path2/path3".to_owned()]),
            project_excludes: Globs::new(vec!["tests/untyped/**".to_owned()]),
            skip_untyped_functions: false,
            search_path: vec![PathBuf::from("../..")],
            python_environment: python_environment.clone(),
            python_interpreter: Some(PathBuf::from(interpreter.clone())),
            ..Default::default()
        };

        let path_str = with_sep("path/to/my/config");
        let test_path = PathBuf::from(path_str.clone());

        let project_includes_vec = vec![
            path_str.clone() + &with_sep("/path1/**"),
            path_str.clone() + &with_sep("/path2/path3"),
        ];
        let project_excludes_vec = vec![path_str.clone() + &with_sep("/tests/untyped/**")];
        let skip_untyped_functions = false;
        let search_path = vec![test_path.join("../.."), test_path.clone()];
        python_environment.site_package_path =
            Some(vec![test_path.join("venv/lib/python1.2.3/site-packages")]);

        config.rewrite_with_path_to_config(&test_path);

        let expected_config = ConfigFile {
            project_includes: Globs::new(project_includes_vec),
            project_excludes: Globs::new(project_excludes_vec),
            skip_untyped_functions,
            search_path,
            python_environment,
            python_interpreter: Some(test_path.join(interpreter)),
            ..ConfigFile::default_no_path_rewrite()
        };
        assert_eq!(config, expected_config);
    }

    #[test]
    fn test_deserializing_unknown_error_errors() {
        let config_str = "
            [errors]
            subtronics = true
            zeds_dead = false
            GRiZ = true
        ";
        let err = ConfigFile::parse_config(config_str).unwrap_err();
        assert!(err.to_string().contains("unknown variant"));
    }
}
