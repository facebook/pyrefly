/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use configparser::ini::Ini;

use crate::config::ConfigFile;
use crate::migration::config_option_migrater::ConfigOptionMigrater;
use crate::migration::pyright::PyrightConfig;

/// Configuration option for site package path
pub struct SitePackagePath;

impl ConfigOptionMigrater for SitePackagePath {
    fn migrate_from_mypy(
        &self,
        _mypy_cfg: &Ini,
        _pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()> {
        Err(anyhow::anyhow!("No site package path found in mypy config"))
    }

    fn migrate_from_pyright(
        &self,
        pyright_cfg: &PyrightConfig,
        pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()> {
        // In pyright, stub path is specified in the "stubPath" field
        let stub_path = match &pyright_cfg.stub_path {
            Some(path) => path,
            None => return Err(anyhow::anyhow!("No stub path found in pyright config")),
        };

        pyrefly_cfg.python_environment.site_package_path = Some(vec![stub_path.clone()]);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::migration::test_util::default_pyright_config;

    const STUB_PATH: &str = "src/stubs";

    #[test]
    fn test_migrate_from_pyright_with_stub_path() {
        let mut pyright_cfg = default_pyright_config();
        pyright_cfg.stub_path = Some(PathBuf::from(STUB_PATH));

        let mut pyrefly_cfg = ConfigFile::default();

        let site_package_path = SitePackagePath;
        let result = site_package_path.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

        assert!(result.is_ok());
        assert_eq!(
            pyrefly_cfg.python_environment.site_package_path,
            Some(vec![PathBuf::from(STUB_PATH)])
        );
    }

    #[test]
    fn test_migrate_from_pyright_without_stub_path() {
        let pyright_cfg = default_pyright_config();

        let mut pyrefly_cfg = ConfigFile::default();
        let default_site_package_path = pyrefly_cfg.python_environment.site_package_path.clone();

        let site_package_path = SitePackagePath;
        let result = site_package_path.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

        assert!(result.is_err());
        assert_eq!(
            pyrefly_cfg.python_environment.site_package_path,
            default_site_package_path
        );
    }
}
