/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::path::PathBuf;

use pyrefly_python::sys_info::PythonPlatform;
use pyrefly_python::sys_info::PythonVersion;
use pyrefly_util::globs::Glob;
use pyrefly_util::globs::Globs;
use serde::Deserialize;
use serde_with::FromInto;
use serde_with::serde_as;

use crate::config::base::ConfigBase;
use crate::config::config::ConfigFile;
use crate::config::config::SubConfig;
use crate::config::error::ErrorDisplayConfig;
use crate::error::kind::ErrorKind;

/// Represents a pyright executionEnvironment.
/// pyright's ExecutionEnvironments allow you to specify a different Python environment for a subdirectory,
/// e.g. with a different Python version, search path, and platform.
/// pyrefly does not support any of that, so we only look for rule overrides.
#[derive(Clone, Debug, Deserialize)]
pub struct ExecEnv {
    root: String,
    #[serde(flatten)]
    errors: RuleOverrides,
}

impl ExecEnv {
    pub fn convert(self) -> SubConfig {
        let settings = ConfigBase {
            errors: self.errors.to_config(),
            ..Default::default()
        };
        SubConfig {
            matches: Glob::new(self.root),
            settings,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct PyrightConfig {
    #[serde(rename = "include")]
    project_includes: Option<Globs>,
    #[serde(rename = "exclude")]
    project_excludes: Option<Globs>,
    #[serde(rename = "extraPaths")]
    search_path: Option<Vec<PathBuf>>,
    #[serde(rename = "pythonPlatform")]
    python_platform: Option<String>,
    #[serde(rename = "pythonVersion")]
    python_version: Option<PythonVersion>,
    #[serde(flatten)]
    errors: RuleOverrides,
    #[serde(default, rename = "executionEnvironments")]
    execution_environments: Vec<ExecEnv>,
}

impl PyrightConfig {
    pub fn convert(self) -> ConfigFile {
        let mut cfg = ConfigFile::default();
        if let Some(includes) = self.project_includes {
            cfg.project_includes = includes;
        }
        if let Some(excludes) = self.project_excludes {
            cfg.project_excludes = excludes;
        }
        if let Some(search_path) = self.search_path {
            cfg.search_path_from_file = search_path;
        }
        if let Some(platform) = self.python_platform {
            cfg.python_environment.python_platform = Some(PythonPlatform::new(&platform));
        }
        if self.python_version.is_some() {
            cfg.python_environment.python_version = self.python_version;
        }
        cfg.root.errors = self.errors.to_config();

        let sub_configs: Vec<SubConfig> = self
            .execution_environments
            .into_iter()
            .map(ExecEnv::convert)
            .collect();
        cfg.sub_configs = sub_configs;

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
        match self {
            Self::None => false,
            _ => true,
        }
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
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuleOverrides {
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    #[serde(default)]
    report_missing_imports: Option<bool>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    #[serde(default)]
    report_missing_module_source: Option<bool>,
}

impl RuleOverrides {
    /// Consume the RuleOverrides to turn it into an ErrorDisplayConfig map.
    fn to_config(self) -> Option<ErrorDisplayConfig> {
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
            map.insert(ErrorKind::ImportError, import_error);
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
    use std::path::Path;

    use super::*;
    use crate::config::environment::environment::PythonEnvironment;
    use crate::config::environment::environment::SitePackagePathSource;

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
                ]),
                project_excludes: Globs::new(vec!["src/excluded/**/*.py".to_owned()]),
                search_path_from_file: vec![PathBuf::from("src/extra")],
                python_environment: PythonEnvironment {
                    python_platform: Some(PythonPlatform::linux()),
                    python_version: Some(PythonVersion::new(3, 10, 0)),
                    site_package_path: None,
                    site_package_path_source: SitePackagePathSource::ConfigFile,
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
                ]),
                python_environment: PythonEnvironment {
                    python_version: Some(PythonVersion::new(3, 11, 0)),
                    python_platform: None,
                    site_package_path: None,
                    site_package_path_source: SitePackagePathSource::ConfigFile,
                },
                ..Default::default()
            }
        );
        Ok(())
    }

    #[test]
    fn test_report_diagnostics_with_bool() -> anyhow::Result<()> {
        let raw_file = r#"
            {
                "include": [
                    "src/**/*.py",
                    "test/**/*.py"
                ],
                "pythonVersion": "3.11",
                "reportMissingImports": false
            }
            "#;
        let pyr = serde_json::from_str::<PyrightConfig>(raw_file)?;
        let config = pyr.convert();
        assert!(
            config
                .root
                .errors
                .is_some_and(|m| !m.is_enabled(ErrorKind::ImportError))
        );
        Ok(())
    }

    #[test]
    fn test_report_diagnostics_with_level() -> anyhow::Result<()> {
        let raw_file = r#"
            {
                "include": [
                    "src/**/*.py",
                    "test/**/*.py"
                ],
                "pythonVersion": "3.11",
                "reportMissingImports": "none"
            }
            "#;
        let pyr = serde_json::from_str::<PyrightConfig>(raw_file)?;
        let config = pyr.convert();
        assert!(
            config
                .root
                .errors
                .is_some_and(|m| !m.is_enabled(ErrorKind::ImportError))
        );
        Ok(())
    }

    #[test]
    fn test_exec_env() -> anyhow::Result<()> {
        let raw_file = r#"
        {
            "include": ["src"],
            "executionEnvironments": [
                { "root": "src/web", "pythonVersion": "3.5", "pythonPlatform": "Windows", "extraPaths": [ "src/service_libs" ], "reportMissingImports": "none" },
                { "root": "src" }
            ]
        }
        "#;
        let pyr = serde_json::from_str::<PyrightConfig>(raw_file)?;
        let mut config = pyr.convert();
        config.configure();
        assert!(
            config
                .errors(Path::new("src/init.py"))
                .is_enabled(ErrorKind::ImportError)
        );
        assert!(
            !config
                .errors(Path::new("src/web/foo.py"))
                .is_enabled(ErrorKind::ImportError)
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
}
