/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use configparser::ini::Ini;

use crate::base::UntypedDefBehavior;
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
        let value = mypy_cfg.getboolcoerce("mypy", "check_untyped_defs");
        if !matches!(value, Ok(Some(_))) {
            pyrefly_cfg.root.untyped_def_behavior = Some(UntypedDefBehavior::SkipAndInferReturnAny);
            return Err(anyhow::anyhow!(
                "No check_untyped_defs found in mypy config, setting default to `skip-and-infer-return-any`"
            ));
        }

        // If check_untyped_defs is True, set untyped_def_behavior to CheckAndInferReturnAny
        // This matches Pyrefly's untyped-def-behavior = "check-and-infer-return-any"
        if value.unwrap().unwrap() {
            pyrefly_cfg.root.untyped_def_behavior =
                Some(UntypedDefBehavior::CheckAndInferReturnAny);
        }

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

        let untyped_def_behavior = UntypedDefBehaviorConfig;
        let result = untyped_def_behavior.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert!(result.is_ok());
        assert_eq!(
            pyrefly_cfg.root.untyped_def_behavior,
            Some(UntypedDefBehavior::CheckAndInferReturnAny)
        );
    }

    #[test]
    fn test_migrate_from_mypy_false() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set("mypy", "check_untyped_defs", Some("False".to_owned()));

        let mut pyrefly_cfg = ConfigFile::default();

        let untyped_def_behavior = UntypedDefBehaviorConfig;
        let result = untyped_def_behavior.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert!(result.is_ok());
        // When check_untyped_defs is False, we don't change the default behavior
        assert_eq!(pyrefly_cfg.root.untyped_def_behavior, None);
    }

    #[test]
    fn test_migrate_from_mypy_empty() {
        let mypy_cfg = Ini::new();

        let mut pyrefly_cfg = ConfigFile::default();

        let untyped_def_behavior = UntypedDefBehaviorConfig;
        let result = untyped_def_behavior.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert!(result.is_err());
        assert_eq!(
            pyrefly_cfg.root.untyped_def_behavior,
            Some(UntypedDefBehavior::SkipAndInferReturnAny)
        );
    }

    #[test]
    fn test_migrate_from_pyright() {
        let pyright_cfg = default_pyright_config();
        let mut pyrefly_cfg = ConfigFile::default();

        let untyped_def_behavior = UntypedDefBehaviorConfig;
        let result = untyped_def_behavior.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

        // Pyright doesn't have a direct equivalent to check_untyped_defs, so we expect an error
        assert!(result.is_err());
        assert_eq!(pyrefly_cfg.root.untyped_def_behavior, None);
    }
}
