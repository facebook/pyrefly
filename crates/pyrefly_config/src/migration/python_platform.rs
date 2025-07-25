/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use configparser::ini::Ini;
use pyrefly_python::sys_info::PythonPlatform;

use crate::config::ConfigFile;
use crate::migration::config_option_migrater::ConfigOptionMigrater;
use crate::migration::pyright::PyrightConfig;

/// Configuration option for Python platform
pub struct PythonPlatformConfig;

impl ConfigOptionMigrater for PythonPlatformConfig {
    fn migrate_from_mypy(
        &self,
        _mypy_cfg: &Ini,
        _pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()> {
        // mypy doesn't have a field for the Python platform
        Err(anyhow::anyhow!(
            "mypy does not support platform configuration"
        ))
    }

    fn migrate_from_pyright(
        &self,
        pyright_cfg: &PyrightConfig,
        pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()> {
        // In pyright, python platform is specified in the "pythonPlatform" field
        let platform = match &pyright_cfg.python_platform {
            Some(p) => p,
            None => {
                return Err(anyhow::anyhow!(
                    "No python_platform found in pyright config"
                ));
            }
        };

        pyrefly_cfg.python_environment.python_platform = Some(PythonPlatform::new(platform));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use configparser::ini::Ini;

    use super::*;
    use crate::migration::test_util::default_pyright_config;

    #[test]
    fn test_migrate_from_mypy() {
        let mypy_cfg = Ini::new();

        let mut pyrefly_cfg = ConfigFile::default();
        let default_platform = pyrefly_cfg.python_environment.python_platform.clone();

        let python_platform_config = PythonPlatformConfig;
        let result = python_platform_config.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert!(result.is_err());
        assert_eq!(
            pyrefly_cfg.python_environment.python_platform,
            default_platform
        );
    }

    #[test]
    fn test_migrate_from_pyright_platforms() {
        // Define test cases: (platform_name, expected_platform_function)
        let test_cases = [
            ("Linux", PythonPlatform::linux()),
            ("Windows", PythonPlatform::windows()),
            ("Darwin", PythonPlatform::mac()),
        ];

        for (platform_name, expected_platform) in test_cases {
            // Create a pyright config with the platform set
            let mut pyright_cfg = default_pyright_config();
            pyright_cfg.python_platform = Some(platform_name.to_owned());

            let mut pyrefly_cfg = ConfigFile::default();

            let python_platform_config = PythonPlatformConfig;
            let result =
                python_platform_config.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

            assert!(result.is_ok());
            assert_eq!(
                pyrefly_cfg.python_environment.python_platform,
                Some(expected_platform),
                "Failed for platform: {platform_name}"
            );
        }
    }

    #[test]
    fn test_migrate_from_pyright_empty() {
        let pyright_cfg = default_pyright_config();

        let mut pyrefly_cfg = ConfigFile::default();
        let default_platform = pyrefly_cfg.python_environment.python_platform.clone();

        let python_platform_config = PythonPlatformConfig;
        let result = python_platform_config.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

        assert!(result.is_err());
        assert_eq!(
            pyrefly_cfg.python_environment.python_platform,
            default_platform
        );
    }
}
