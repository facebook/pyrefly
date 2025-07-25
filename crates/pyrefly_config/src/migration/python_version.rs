/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::str::FromStr;

use configparser::ini::Ini;
use pyrefly_python::sys_info::PythonVersion;

use crate::config::ConfigFile;
use crate::migration::config_option_migrater::ConfigOptionMigrater;
use crate::migration::pyright::PyrightConfig;

/// Configuration option for Python version
pub struct PythonVersionConfig;

impl ConfigOptionMigrater for PythonVersionConfig {
    fn migrate_from_mypy(
        &self,
        mypy_cfg: &Ini,
        pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()> {
        // https://mypy.readthedocs.io/en/latest/config_file.html#import-discovery
        // python_version can only be set in the top level `[mypy]` global section
        let python_version = mypy_cfg.get("mypy", "python_version");

        if python_version.is_none() {
            return Err(anyhow::anyhow!("No python_version found in mypy config"));
        }

        let version = python_version.unwrap();
        pyrefly_cfg.python_environment.python_version = PythonVersion::from_str(&version).ok();
        Ok(())
    }

    fn migrate_from_pyright(
        &self,
        pyright_cfg: &PyrightConfig,
        pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()> {
        // In pyright, python version is specified in the "pythonVersion" field
        let version = match &pyright_cfg.python_version {
            Some(v) => v,
            None => return Err(anyhow::anyhow!("No python_version found in pyright config")),
        };

        pyrefly_cfg.python_environment.python_version = Some(*version);
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::migration::test_util::default_pyright_config;

    #[test]
    fn test_migrate_from_mypy() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set("mypy", "python_version", Some("3.8".to_owned()));

        let mut pyrefly_cfg = ConfigFile::default();

        let python_version_config = PythonVersionConfig;
        let _ = python_version_config.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert_eq!(
            pyrefly_cfg.python_environment.python_version,
            Some(PythonVersion::new(3, 8, 0))
        );
    }

    #[test]
    fn test_migrate_from_mypy_empty() {
        let mypy_cfg = Ini::new();

        let mut pyrefly_cfg = ConfigFile::default();
        let default_version = pyrefly_cfg.python_environment.python_version;

        let python_version_config = PythonVersionConfig;
        let _ = python_version_config.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert_eq!(
            pyrefly_cfg.python_environment.python_version,
            default_version
        );
    }

    #[test]
    fn test_migrate_from_pyright() {
        let mut pyright_cfg = default_pyright_config();
        pyright_cfg.python_version = Some(PythonVersion::new(3, 10, 0));

        let mut pyrefly_cfg = ConfigFile::default();

        let python_version_config = PythonVersionConfig;
        let result = python_version_config.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

        assert!(result.is_ok());
        assert_eq!(
            pyrefly_cfg.python_environment.python_version,
            Some(PythonVersion::new(3, 10, 0))
        );
    }

    #[test]
    fn test_migrate_from_pyright_empty() {
        let pyright_cfg = default_pyright_config();

        let mut pyrefly_cfg = ConfigFile::default();
        let default_version = pyrefly_cfg.python_environment.python_version;

        let python_version_config = PythonVersionConfig;
        let result = python_version_config.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

        assert!(result.is_err());
        assert_eq!(
            pyrefly_cfg.python_environment.python_version,
            default_version
        );
    }
}
