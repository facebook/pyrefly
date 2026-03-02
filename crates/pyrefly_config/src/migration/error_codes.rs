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
use crate::migration::mypy::util::MypyErrorConfigFlags;
use crate::migration::pyright::PyrightConfig;

/// Configuration option for error codes
pub struct ErrorCodes;

impl ConfigOptionMigrater for ErrorCodes {
    fn migrate_from_mypy(
        &self,
        mypy_cfg: &Ini,
        pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()> {
        let warn_redundant_casts =
            util::get_bool_or_default(mypy_cfg, "mypy", "warn_redundant_casts");
        let disallow_untyped_defs =
            util::get_bool_or_default(mypy_cfg, "mypy", "disallow_untyped_defs");
        let disallow_incomplete_defs =
            util::get_bool_or_default(mypy_cfg, "mypy", "disallow_incomplete_defs");
        let disallow_any_generics =
            util::get_bool_or_default(mypy_cfg, "mypy", "disallow_any_generics");
        let report_deprecated_as_note =
            util::get_bool_or_default(mypy_cfg, "mypy", "report_deprecated_as_note");
        let allow_redefinitions =
            util::get_bool_or_default(mypy_cfg, "mypy", "allow_redefinitions");
        let strict = util::get_bool_or_default(mypy_cfg, "mypy", "strict");
        let mypy_flags = MypyErrorConfigFlags {
            warn_redundant_casts,
            disallow_untyped_defs,
            disallow_incomplete_defs,
            disallow_any_generics,
            report_deprecated_as_note,
            allow_redefinitions,
            strict,
        };
        let disable_error_code = util::string_to_array(&mypy_cfg.get("mypy", "disable_error_code"));
        let enable_error_code = util::string_to_array(&mypy_cfg.get("mypy", "enable_error_code"));
        let error_config =
            util::make_error_config(Some(mypy_flags), disable_error_code, enable_error_code)
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
    use configparser::ini::IniDefault;

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
    fn test_migrate_from_mypy_with_disable_codes() {
        let mut default = IniDefault::default();
        default.multiline = true;
        let mut mypy_cfg = Ini::new_from_defaults(default);

        mypy_cfg.set(
            "mypy",
            "disable_error_code",
            Some("arg-type, call-overload".to_owned()),
        );

        let mut pyrefly_cfg = ConfigFile::default();

        let error_codes = ErrorCodes;
        let _ = error_codes.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert!(pyrefly_cfg.root.errors.is_some());
        let errors = pyrefly_cfg.root.errors.as_ref().unwrap();
        assert_eq!(
            errors.severity(ErrorKind::BadArgumentType),
            Severity::Ignore
        );
        assert_eq!(
            errors.severity(ErrorKind::NoMatchingOverload),
            Severity::Ignore
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
    fn test_migrate_from_pyright_simple() {
        let mut pyright_cfg = default_pyright_config();
        pyright_cfg.errors.report_missing_module_source = Some(Severity::Error);

        let mut pyrefly_cfg = ConfigFile::default();

        let error_codes = ErrorCodes;
        let result = error_codes.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

        assert!(result.is_ok());
        assert!(pyrefly_cfg.root.errors.is_some());
        let errors = pyrefly_cfg.root.errors.as_ref().unwrap();

        assert_eq!(errors.severity(ErrorKind::MissingImport), Severity::Error);
    }

    #[test]
    fn test_migrate_from_pyright_use_max_severity() {
        let mut pyright_cfg = default_pyright_config();
        pyright_cfg.errors.report_unknown_parameter_type = Some(Severity::Error);
        pyright_cfg.errors.report_unknown_argument_type = Some(Severity::Warn);

        pyright_cfg.errors.report_possibly_unbound_variable = Some(Severity::Warn);
        pyright_cfg.errors.report_unbound_variable = Some(Severity::Error);

        let mut pyrefly_cfg = ConfigFile::default();

        let error_codes = ErrorCodes;
        let result = error_codes.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

        assert!(result.is_ok());
        assert!(pyrefly_cfg.root.errors.is_some());
        let errors = pyrefly_cfg.root.errors.as_ref().unwrap();

        assert_eq!(
            errors.severity(ErrorKind::UnannotatedParameter),
            Severity::Error
        );
        assert_eq!(errors.severity(ErrorKind::ImplicitAny), Severity::Warn);
        assert_eq!(errors.severity(ErrorKind::UnboundName), Severity::Error);
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

    #[test]
    fn test_migrate_from_mypy_with_bad_return_disabled_error_code() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set("mypy", "disable_error_code", Some("return".to_owned()));

        let mut pyrefly_cfg = ConfigFile::default();

        let error_codes = ErrorCodes;
        let _ = error_codes.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert!(pyrefly_cfg.root.errors.is_some());
        let errors = pyrefly_cfg.root.errors.as_ref().unwrap();
        assert_eq!(errors.severity(ErrorKind::BadReturn), Severity::Ignore);
    }

    #[test]
    fn test_migrate_from_mypy_with_type_arg_enabled_error_code() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set("mypy", "enable_error_code", Some("type-arg".to_owned()));

        let mut pyrefly_cfg = ConfigFile::default();

        let error_codes = ErrorCodes;
        let _ = error_codes.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert!(pyrefly_cfg.root.errors.is_some());
        let errors = pyrefly_cfg.root.errors.as_ref().unwrap();
        assert_eq!(errors.severity(ErrorKind::ImplicitAny), Severity::Error);
    }
}
