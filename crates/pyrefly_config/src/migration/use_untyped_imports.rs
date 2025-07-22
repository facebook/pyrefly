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

/// Configuration option for using untyped imports
pub struct UseUntypedImports;

impl ConfigOptionMigrater for UseUntypedImports {
    fn migrate_from_mypy(
        &self,
        mypy_cfg: &Ini,
        pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()> {
        // follow_untyped_imports may be used as a global or per-module setting. As a per-module setting, it's used to
        // indicate that the module should be ignored if it's untyped.
        // Pyrefly's use_untyped_imports is only a global setting.
        // We handle this by *only* checking the for the global config.
        let value = mypy_cfg.getboolcoerce("mypy", "follow_untyped_imports");
        if !matches!(value, Ok(Some(_))) {
            return Err(anyhow::anyhow!(
                "No follow_untyped_imports found in mypy config"
            ));
        }

        pyrefly_cfg.use_untyped_imports = value.unwrap().unwrap();

        Ok(())
    }

    fn migrate_from_pyright(
        &self,
        _pyright_cfg: &PyrightConfig,
        _pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()> {
        // Pyright doesn't have a direct equivalent to follow_untyped_imports
        // We'll return an error to indicate this
        Err(anyhow::anyhow!(
            "Pyright does not have a direct equivalent to follow_untyped_imports"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::migration::test_util::default_pyright_config;

    #[test]
    fn test_migrate_from_mypy_true() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set("mypy", "follow_untyped_imports", Some("True".to_owned()));

        let mut pyrefly_cfg = ConfigFile {
            use_untyped_imports: false,
            ..Default::default()
        };

        let use_untyped_imports = UseUntypedImports;
        let _ = use_untyped_imports.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert!(pyrefly_cfg.use_untyped_imports);
    }

    #[test]
    fn test_migrate_from_mypy_false() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set("mypy", "follow_untyped_imports", Some("False".to_owned()));

        let mut pyrefly_cfg = ConfigFile {
            use_untyped_imports: true,
            ..Default::default()
        };
        let use_untyped_imports = UseUntypedImports;
        let _ = use_untyped_imports.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert!(!pyrefly_cfg.use_untyped_imports);
    }

    #[test]
    fn test_migrate_from_mypy_empty() {
        let mypy_cfg = Ini::new();

        let mut pyrefly_cfg = ConfigFile::default();
        let default_value = pyrefly_cfg.use_untyped_imports;

        let use_untyped_imports = UseUntypedImports;
        let _ = use_untyped_imports.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert_eq!(pyrefly_cfg.use_untyped_imports, default_value);
    }

    #[test]
    fn test_migrate_from_mypy_ignores_per_module_setting() {
        // Create a test mypy.ini config with both global and per-module settings
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set("mypy", "follow_untyped_imports", Some("False".to_owned()));
        mypy_cfg.set(
            "mypy-this.setting.ignored",
            "follow_untyped_imports",
            Some("True".to_owned()),
        );

        let mut pyrefly_cfg = ConfigFile {
            ..Default::default()
        };

        let use_untyped_imports = UseUntypedImports;
        let _ = use_untyped_imports.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        // Verify that only the global setting was applied, and the per-module setting was ignored
        assert!(!pyrefly_cfg.use_untyped_imports);
    }

    #[test]
    fn test_migrate_from_pyright() {
        let pyright_cfg = default_pyright_config();
        let mut pyrefly_cfg = ConfigFile::default();
        let default_value = pyrefly_cfg.use_untyped_imports;

        let use_untyped_imports = UseUntypedImports;
        let result = use_untyped_imports.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

        // Pyright doesn't have a direct equivalent to follow_untyped_imports, so we expect an error
        assert!(result.is_err());
        // The value should remain unchanged
        assert_eq!(pyrefly_cfg.use_untyped_imports, default_value);
    }
}
