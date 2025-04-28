/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;

use serde::Deserialize;

use crate::config::config::ConfigFile;
use crate::sys_info::PythonPlatform;
use crate::sys_info::PythonVersion;
use crate::util::globs::Globs;

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
            cfg.search_path = search_path;
        }
        if let Some(platform) = self.python_platform {
            cfg.python_environment.python_platform = Some(PythonPlatform::new(&platform));
        }
        if self.python_version.is_some() {
            cfg.python_environment.python_version = self.python_version;
        }
        cfg
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
    use super::*;
    use crate::config::environment::PythonEnvironment;

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
                search_path: vec![PathBuf::from("src/extra")],
                python_environment: PythonEnvironment {
                    python_platform: Some(PythonPlatform::linux()),
                    python_version: Some(PythonVersion::new(3, 10, 0)),
                    site_package_path: None,
                    site_package_path_from_interpreter: false,
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
                    site_package_path_from_interpreter: false,
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
}
