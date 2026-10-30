/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use configparser::ini::Ini;

use crate::base::InferReturnTypes;
use crate::config::ConfigFile;
use crate::migration::config_option_migrater::ConfigOptionMigrater;
use crate::migration::pyright::PyrightConfig;

/// Configuration option for handling untyped function definitions
pub struct UntypedDefBehaviorConfig;

impl ConfigOptionMigrater for UntypedDefBehaviorConfig {
    fn migrate_from_mypy(
        &self,
        mypy_cfg: &Ini,
        pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()> {
        // check_untyped_defs may be used as a global or per-module setting.
        // We handle this by only checking for the global config.
        let check_untyped_defs = match mypy_cfg.getboolcoerce("mypy", "check_untyped_defs") {
            Ok(Some(value)) => value,
            _ if mypy_cfg
                .getboolcoerce("mypy", "strict")
                .is_ok_and(|value| value == Some(true)) =>
            {
                true
            }
            // No explicit setting: let the preset handle the defaults.
            _ => {
                return Err(anyhow::anyhow!(
                    "No check_untyped_defs or strict found in mypy config"
                ));
            }
        };

        // mypy's check_untyped_defs controls whether bodies of unannotated functions
        // are analyzed. It never infers return types, so infer_return_types = Never.
        pyrefly_cfg.root.check_unannotated_defs = Some(check_untyped_defs);
        pyrefly_cfg.root.infer_return_types = Some(InferReturnTypes::Never);

        Ok(())
    }

    fn migrate_from_pyright(
        &self,
        _pyright_cfg: &PyrightConfig,
        _pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()> {
        // Pyright doesn't have a direct equivalent to check_untyped_defs
        // We'll return an error to indicate this
        Err(anyhow::anyhow!(
            "Pyright does not have a direct equivalent to check_untyped_defs"
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
        mypy_cfg.set("mypy", "check_untyped_defs", Some("True".to_owned()));

        let mut pyrefly_cfg = ConfigFile::default();

        let config = UntypedDefBehaviorConfig;
        let result = config.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert!(result.is_ok());
        assert_eq!(pyrefly_cfg.root.check_unannotated_defs, Some(true));
        assert_eq!(
            pyrefly_cfg.root.infer_return_types,
            Some(InferReturnTypes::Never)
        );
    }

    #[test]
    fn test_migrate_from_mypy_false() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set("mypy", "check_untyped_defs", Some("False".to_owned()));

        let mut pyrefly_cfg = ConfigFile::default();

        let config = UntypedDefBehaviorConfig;
        let result = config.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert!(result.is_ok());
        assert_eq!(pyrefly_cfg.root.check_unannotated_defs, Some(false));
        assert_eq!(
            pyrefly_cfg.root.infer_return_types,
            Some(InferReturnTypes::Never)
        );
    }

    #[test]
    fn test_migrate_from_mypy_strict() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set("mypy", "strict", Some("True".to_owned()));

        let mut pyrefly_cfg = ConfigFile::default();
        UntypedDefBehaviorConfig
            .migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg)
            .expect("strict should enable untyped body checking");

        assert_eq!(pyrefly_cfg.root.check_unannotated_defs, Some(true));
        assert_eq!(
            pyrefly_cfg.root.infer_return_types,
            Some(InferReturnTypes::Never)
        );
    }

    #[test]
    fn test_migrate_from_mypy_empty() {
        let mypy_cfg = Ini::new();

        let mut pyrefly_cfg = ConfigFile::default();

        let config = UntypedDefBehaviorConfig;
        let result = config.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        // No explicit setting: fields left as None for the preset to handle
        assert!(result.is_err());
        assert_eq!(pyrefly_cfg.root.check_unannotated_defs, None);
        assert_eq!(pyrefly_cfg.root.infer_return_types, None);
    }

    #[test]
    fn test_migrate_from_pyright() {
        let pyright_cfg = default_pyright_config();
        let mut pyrefly_cfg = ConfigFile::default();

        let config = UntypedDefBehaviorConfig;
        let result = config.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

        // Pyright doesn't have a direct equivalent to check_untyped_defs, so we expect an error
        assert!(result.is_err());
        assert_eq!(pyrefly_cfg.root.check_unannotated_defs, None);
        assert_eq!(pyrefly_cfg.root.infer_return_types, None);
    }
}
