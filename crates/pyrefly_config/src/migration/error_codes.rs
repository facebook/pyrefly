/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use configparser::ini::Ini;

use crate::config::ConfigFile;
use crate::migration::config_option_migrater::ConfigOptionMigrater;
use crate::migration::mypy::util;
use crate::migration::pyright::PyrightConfig;

/// Configuration option for error codes
pub struct ErrorCodes;

impl ConfigOptionMigrater for ErrorCodes {
    fn migrate_from_mypy(
        &self,
        mypy_cfg: &Ini,
        pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()> {
        // Get disable_error_code and enable_error_code from the mypy.ini file
        let disable_error_code = util::string_to_array(&mypy_cfg.get("mypy", "disable_error_code"));
        let enable_error_code = util::string_to_array(&mypy_cfg.get("mypy", "enable_error_code"));

        let error_config = util::make_error_config(disable_error_code, enable_error_code)
            .ok_or_else(|| anyhow::anyhow!("Failed to create error config"))?;
        pyrefly_cfg.root.errors = Some(error_config);
        Ok(())
    }

    fn migrate_from_pyright(
        &self,
        pyright_cfg: &PyrightConfig,
        pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()> {
        // In pyright, error settings are specified in various "report*" fields
        // The PyrightConfig struct already has a method to convert these to an ErrorDisplayConfig
        let error_config = pyright_cfg
            .errors
            .clone()
            .to_config()
            .ok_or_else(|| anyhow::anyhow!("No error settings found in pyright config"))?;

        pyrefly_cfg.root.errors = Some(error_config);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error_kind::ErrorKind;
    use crate::error_kind::Severity;
    use crate::migration::test_util::default_pyright_config;

    #[test]
    fn test_migrate_from_mypy_with_both_error_codes() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set("mypy", "disable_error_code", Some("union-attr".to_owned()));
        mypy_cfg.set("mypy", "enable_error_code", Some("attr-defined".to_owned()));

        let mut pyrefly_cfg = ConfigFile::default();

        let error_codes = ErrorCodes;
        let _ = error_codes.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert!(pyrefly_cfg.root.errors.is_some());
        let errors = pyrefly_cfg.root.errors.as_ref().unwrap();
        assert_eq!(
            errors.severity(ErrorKind::MissingAttribute),
            Severity::Error
        );
    }

    #[test]
    fn test_migrate_from_mypy_with_only_disable_codes() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set("mypy", "disable_error_code", Some("union-attr".to_owned()));

        let mut pyrefly_cfg = ConfigFile::default();

        let error_codes = ErrorCodes;
        let _ = error_codes.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert!(pyrefly_cfg.root.errors.is_some());
        let errors = pyrefly_cfg.root.errors.as_ref().unwrap();
        assert_eq!(
            errors.severity(ErrorKind::MissingAttribute),
            Severity::Ignore
        );
    }

    #[test]
    fn test_migrate_from_mypy_with_only_enable_codes() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set("mypy", "enable_error_code", Some("attr-defined".to_owned()));

        let mut pyrefly_cfg = ConfigFile::default();

        let error_codes = ErrorCodes;
        let _ = error_codes.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert!(pyrefly_cfg.root.errors.is_some());
        let errors = pyrefly_cfg.root.errors.as_ref().unwrap();
        assert_eq!(
            errors.severity(ErrorKind::MissingAttribute),
            Severity::Error
        );
    }

    #[test]
    fn test_migrate_from_mypy_with_empty_config() {
        let mypy_cfg = Ini::new();

        let mut pyrefly_cfg = ConfigFile::default();
        let default_errors = pyrefly_cfg.root.errors.clone();

        let error_codes = ErrorCodes;
        let _ = error_codes.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert_eq!(pyrefly_cfg.root.errors, default_errors);
    }

    #[test]
    fn test_migrate_from_pyright() {
        let mut pyright_cfg = default_pyright_config();
        pyright_cfg.errors.report_missing_module_source = Some(true);

        let mut pyrefly_cfg = ConfigFile::default();

        let error_codes = ErrorCodes;
        let result = error_codes.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

        assert!(result.is_ok());
        assert!(pyrefly_cfg.root.errors.is_some());
        let errors = pyrefly_cfg.root.errors.as_ref().unwrap();

        assert_eq!(errors.severity(ErrorKind::ImportError), Severity::Error);
    }

    #[test]
    fn test_migrate_from_pyright_empty() {
        let pyright_cfg = default_pyright_config();

        let mut pyrefly_cfg = ConfigFile::default();
        let default_errors = pyrefly_cfg.root.errors.clone();

        let error_codes = ErrorCodes;
        let result = error_codes.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

        // If RuleOverrides.to_config() returns None when all fields are None,
        // this should fail with an error
        assert!(result.is_err());
        assert_eq!(pyrefly_cfg.root.errors, default_errors);
    }
}
