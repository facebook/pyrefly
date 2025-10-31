/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::path::PathBuf;

use pyrefly_python::sys_info::PythonVersion;
use pyrefly_util::globs::Glob;
use pyrefly_util::globs::Globs;
use serde::Deserialize;
use serde_with::FromInto;
use serde_with::serde_as;

use crate::base::ConfigBase;
use crate::config::ConfigFile;
use crate::config::SubConfig;
use crate::error::ErrorDisplayConfig;
use crate::error_kind::ErrorKind;
use crate::error_kind::Severity;

/// Represents a pyright executionEnvironment.
/// pyright's ExecutionEnvironments allow you to specify a different Python environment for a subdirectory,
/// e.g. with a different Python version, search path, and platform.
/// pyrefly does not support any of that, so we only look for rule overrides.
#[derive(Clone, Debug, Deserialize)]
pub struct ExecEnv {
    pub root: String,
    #[serde(flatten)]
    pub errors: RuleOverrides,
}

impl ExecEnv {
    pub fn convert(self) -> anyhow::Result<SubConfig> {
        let settings = ConfigBase {
            errors: self.errors.to_config(),
            ..Default::default()
        };
        Ok(SubConfig {
            matches: Glob::new(self.root)?,
            settings,
        })
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct PyrightConfig {
    #[serde(rename = "include")]
    pub project_includes: Option<Globs>,
    #[serde(rename = "exclude")]
    pub project_excludes: Option<Globs>,
    #[serde(rename = "extraPaths")]
    pub search_path: Option<Vec<PathBuf>>,
    #[serde(rename = "stubPath")]
    pub stub_path: Option<PathBuf>,
    #[serde(rename = "pythonPlatform")]
    pub python_platform: Option<String>,
    #[serde(rename = "pythonVersion")]
    pub python_version: Option<PythonVersion>,
    #[serde(flatten)]
    pub errors: RuleOverrides,
    #[serde(default, rename = "executionEnvironments")]
    pub execution_environments: Vec<ExecEnv>,
}

use crate::migration::config_option_migrater::ConfigOptionMigrater;
use crate::migration::error_codes::ErrorCodes;
use crate::migration::ignore_missing_imports::IgnoreMissingImports;
use crate::migration::project_excludes::ProjectExcludes;
use crate::migration::project_includes::ProjectIncludes;
use crate::migration::python_interpreter::PythonInterpreter;
use crate::migration::python_platform::PythonPlatformConfig;
use crate::migration::python_version::PythonVersionConfig;
use crate::migration::search_path::SearchPath;
use crate::migration::site_package_path::SitePackagePath;
use crate::migration::sub_configs::SubConfigs;

impl PyrightConfig {
    pub fn parse(text: &str) -> anyhow::Result<Self> {
        Ok(serde_jsonrc::from_str::<Self>(text)?)
    }

    pub fn convert(self) -> ConfigFile {
        let mut cfg = ConfigFile::default();

        // Create a list of all config options
        let config_options: Vec<Box<dyn ConfigOptionMigrater>> = vec![
            Box::new(ProjectIncludes),
            Box::new(ProjectExcludes),
            Box::new(PythonInterpreter),
            Box::new(PythonVersionConfig),
            Box::new(PythonPlatformConfig),
            Box::new(SearchPath),
            Box::new(SitePackagePath),
            Box::new(IgnoreMissingImports),
            Box::new(ErrorCodes),
            Box::new(SubConfigs),
        ];

        // Iterate through all config options and apply them to the config
        for option in config_options {
            // Ignore errors for now, we can use this in the future if we want to print out error messages or use for logging purpose
            let _ = option.migrate_from_pyright(&self, &mut cfg);
        }

        // Pyright does not infer empty container types and unsolved type variables based on their first use.
        cfg.root.infer_with_first_use = Some(false);

        cfg
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DiagnosticLevel {
    None,
    Information,
    Warning,
    Error,
}

impl DiagnosticLevel {
    fn to_bool(&self) -> bool {
        !matches!(self, Self::None)
    }
}

impl From<DiagnosticLevel> for bool {
    fn from(value: DiagnosticLevel) -> Self {
        value.to_bool()
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum DiagnosticLevelOrBool {
    DiagnosticLevel(DiagnosticLevel),
    Bool(bool),
}

impl DiagnosticLevelOrBool {
    fn to_bool(&self) -> bool {
        match self {
            Self::DiagnosticLevel(dl) => dl.to_bool(),
            Self::Bool(b) => *b,
        }
    }
}

impl From<DiagnosticLevelOrBool> for bool {
    fn from(value: DiagnosticLevelOrBool) -> Self {
        value.to_bool()
    }
}

/// Type Check Rule Overrides are pyright's equivalent to the `errors` dict in pyrefly's configs.
/// That is, they control which "diangostic settings" are displayed to the user.
#[serde_as]
#[derive(Clone, Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
/// Rule overrides for pyright
pub struct RuleOverrides {
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    #[serde(default)]
    pub report_missing_imports: Option<bool>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    #[serde(default)]
    pub report_missing_module_source: Option<bool>,
}

impl RuleOverrides {
    /// Consume the RuleOverrides to turn it into an ErrorDisplayConfig map.
    pub fn to_config(self) -> Option<ErrorDisplayConfig> {
        let mut map = HashMap::new();
        // For each ErrorKind, there are one or more RuleOverrides fields.
        // The ErrorDisplayConfig map has an entry for an ErrorKind if at least one of the RuleOverrides for that ErrorKind is present.
        // The value of that ErrorKind's entry is found by or'ing together the present RuleOverrides.
        if let Some(import_error) = [
            self.report_missing_imports,
            self.report_missing_module_source,
        ]
        .into_iter()
        .flatten()
        .reduce(|acc, x| acc | x)
        {
            map.insert(
                ErrorKind::MissingImport,
                if import_error {
                    Severity::Error
                } else {
                    Severity::Ignore
                },
            );
        }

        if map.is_empty() {
            None
        } else {
            Some(ErrorDisplayConfig::new(map))
        }
    }
}

#[derive(thiserror::Error, Debug)]
#[error("No [tool.pyright] section found in pyproject.toml")]
pub struct PyrightNotFoundError {}

pub fn parse_pyproject_toml(raw_file: &str) -> anyhow::Result<ConfigFile> {
    #[derive(Deserialize)]
    struct Tool {
        pyright: Option<PyrightConfig>,
    }

    #[derive(Deserialize)]
    struct PyProject {
        tool: Option<Tool>,
    }

    toml::from_str::<PyProject>(raw_file)?
        .tool
        .and_then(|tool| tool.pyright)
        .ok_or(anyhow::anyhow!(PyrightNotFoundError {}))
        .map(PyrightConfig::convert)
}

#[cfg(test)]
mod tests {

    use pyrefly_python::sys_info::PythonPlatform;

    use super::*;
    use crate::environment::environment::PythonEnvironment;

    #[test]
    fn test_convert_pyright_config() -> anyhow::Result<()> {
        let raw_file = r#"
            {
                "include": [
                    "src/**/*.py",
                    "test/**/*.py"
                ],
                "exclude": [
                    "src/excluded/**/*.py"
                ],
                "extraPaths": [
                    "src/extra"
                ],
                "pythonPlatform": "Linux",
                "pythonVersion": "3.10"
            }
            "#;
        let pyr = serde_json::from_str::<PyrightConfig>(raw_file)?;
        let config = pyr.convert();
        assert_eq!(
            config,
            ConfigFile {
                project_includes: Globs::new(vec![
                    "src/**/*.py".to_owned(),
                    "test/**/*.py".to_owned()
                ])
                .unwrap(),
                project_excludes: Globs::new(vec!["src/excluded/**/*.py".to_owned()]).unwrap(),
                search_path_from_file: vec![PathBuf::from("src/extra")],
                python_environment: PythonEnvironment {
                    python_platform: Some(PythonPlatform::linux()),
                    python_version: Some(PythonVersion::new(3, 10, 0)),
                    site_package_path: None,
                    interpreter_site_package_path: config
                        .python_environment
                        .interpreter_site_package_path
                        .clone(),
                    interpreter_stdlib_path: config
                        .python_environment
                        .interpreter_stdlib_path
                        .clone(),
                },
                root: ConfigBase {
                    infer_with_first_use: Some(false),
                    ..Default::default()
                },
                ..Default::default()
            }
        );
        Ok(())
    }

    #[test]
    fn test_convert_pyright_config_with_missing_fields() -> anyhow::Result<()> {
        let raw_file = r#"
            {
                "include": [
                    "src/**/*.py",
                    "test/**/*.py"
                ],
                "pythonVersion": "3.11"
            }
            "#;
        let pyr = serde_json::from_str::<PyrightConfig>(raw_file)?;
        let config = pyr.convert();
        assert_eq!(
            config,
            ConfigFile {
                project_includes: Globs::new(vec![
                    "src/**/*.py".to_owned(),
                    "test/**/*.py".to_owned()
                ])
                .unwrap(),
                python_environment: PythonEnvironment {
                    python_version: Some(PythonVersion::new(3, 11, 0)),
                    python_platform: None,
                    site_package_path: None,
                    interpreter_site_package_path: config
                        .python_environment
                        .interpreter_site_package_path
                        .clone(),
                    interpreter_stdlib_path: config
                        .python_environment
                        .interpreter_stdlib_path
                        .clone(),
                },
                root: ConfigBase {
                    infer_with_first_use: Some(false),
                    ..Default::default()
                },
                ..Default::default()
            }
        );
        Ok(())
    }

    #[test]
    fn test_convert_from_pyproject() -> anyhow::Result<()> {
        // From https://microsoft.github.io/pyright/#/configuration?id=sample-pyprojecttoml-file
        let src = r#"[tool.pyright]
include = ["src"]
exclude = ["**/node_modules",
    "**/__pycache__",
    "src/experimental",
    "src/typestubs"
]
ignore = ["src/oldstuff"]
defineConstant = { DEBUG = true }
stubPath = "src/stubs"

reportMissingImports = "error"
reportMissingTypeStubs = false

pythonVersion = "3.6"
pythonPlatform = "Linux"

executionEnvironments = [
  { root = "src/web", pythonVersion = "3.5", pythonPlatform = "Windows", extraPaths = [ "src/service_libs" ], reportMissingImports = "warning" },
  { root = "src/sdk", pythonVersion = "3.0", extraPaths = [ "src/backend" ] },
  { root = "src/tests", extraPaths = ["src/tests/e2e", "src/sdk" ]},
  { root = "src" }
]
"#;
        parse_pyproject_toml(src).map(|_| ())
    }

    #[test]
    fn test_report_trailing_commas() -> anyhow::Result<()> {
        let raw_file = r#"
            {
                "include": [
                    "src/**/*.py",
                    "test/**/*.py",
                ],
                "pythonVersion": "3.11",
                "reportMissingImports": "none"
            }
            "#;
        let pyr = serde_jsonrc::from_str::<PyrightConfig>(raw_file)?;
        let config = pyr.convert();
        assert!(!config.project_includes.is_empty());
        Ok(())
    }
}
