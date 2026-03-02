/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use configparser::ini::Ini;
use itertools::Itertools as _;
use pyrefly_util::globs::Glob;

use crate::base::ConfigBase;
use crate::config::ConfigFile;
use crate::config::SubConfig;
use crate::error::ErrorDisplayConfig;
use crate::migration::config_option_migrater::ConfigOptionMigrater;
use crate::migration::mypy::util;
use crate::migration::pyright::PyrightConfig;

/// Configuration option for sub-configs (per-module options)
pub struct SubConfigs;

impl ConfigOptionMigrater for SubConfigs {
    fn migrate_from_mypy(
        &self,
        mypy_cfg: &Ini,
        pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()> {
        let mut sub_configs: Vec<(String, ErrorDisplayConfig)> = vec![];

        // Check all sections for per-module options
        util::visit_ini_sections(
            mypy_cfg,
            |section_name| section_name.starts_with("mypy-"),
            |section_name, ini| {
                if let Some(stripped) = section_name.strip_prefix("mypy-") {
                    // For subconfigs, the only config that needs to be extracted is enable/disable error codes.
                    let disable_error_code =
                        util::string_to_array(&ini.get(section_name, "disable_error_code"));
                    let enable_error_code =
                        util::string_to_array(&ini.get(section_name, "enable_error_code"));

                    if disable_error_code.is_empty() && enable_error_code.is_empty() {
                        return;
                    }

                    if let Some(error_config) =
                        util::make_error_config(None, disable_error_code, enable_error_code)
                    {
                        sub_configs.push((stripped.to_owned(), error_config));
                    }
                }
            },
        );

        if sub_configs.is_empty() {
            return Err(anyhow::anyhow!("No sub configs found in mypy config"));
        }

        let sub_configs_vec = sub_configs
            .into_iter()
            .flat_map(|(section, errors)| {
                // Split the section headers into individual modules and pair them with the section's error config.
                // mypy uses module wildcards for its per-module sections, but we use globs.
                // A simple translation: turn `.` into `/` and `*` into `**`, e.g. `a.*.b` -> `a/**/b`.
                section
                    .split(",")
                    .map(|x| x.trim())
                    .filter(|x| !x.is_empty())
                    .map(|module| Glob::new(module.replace('.', "/").replace('*', "**")))
                    .collect::<Vec<_>>()
                    .into_iter()
                    .zip(std::iter::repeat(Some(errors)))
            })
            .map(|(matches, errors)| -> anyhow::Result<SubConfig> {
                Ok(SubConfig {
                    matches: matches?,
                    settings: ConfigBase {
                        errors,
                        ..Default::default()
                    },
                })
            })
            .process_results(|i| i.collect::<Vec<_>>())?;

        if sub_configs_vec.is_empty() {
            return Err(anyhow::anyhow!("No valid sub configs found in mypy config"));
        }
        pyrefly_cfg.sub_configs = sub_configs_vec;
        Ok(())
    }

    fn migrate_from_pyright(
        &self,
        pyright_cfg: &PyrightConfig,
        pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()> {
        // In pyright, sub configs are specified in the "executionEnvironments" field
        // Each execution environment has a root path and error settings
        let sub_configs: Vec<SubConfig> = pyright_cfg
            .execution_environments
            .iter()
            .map(|env| env.clone().convert())
            .process_results(|i| i.collect())?;

        if sub_configs.is_empty() {
            return Err(anyhow::anyhow!(
                "No execution environments found in pyright config"
            ));
        }

        pyrefly_cfg.sub_configs = sub_configs;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error_kind::ErrorKind;
    use crate::error_kind::Severity;
    use crate::migration::pyright::ExecEnv;
    use crate::migration::pyright::RuleOverrides;
    use crate::migration::test_util::default_pyright_config;

    #[test]
    fn test_migrate_from_mypy_with_single_module() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set(
            "mypy-app.models",
            "disable_error_code",
            Some("union-attr".to_owned()),
        );

        let mut pyrefly_cfg = ConfigFile::default();

        let sub_configs = SubConfigs;
        let _ = sub_configs.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert_eq!(pyrefly_cfg.sub_configs.len(), 1);

        let sub_config = &pyrefly_cfg.sub_configs[0];
        assert_eq!(sub_config.matches.to_string(), "app/models");

        let errors = sub_config.settings.errors.as_ref().unwrap();
        assert_eq!(
            errors.severity(ErrorKind::MissingAttribute),
            Severity::Ignore
        );
    }

    #[test]
    fn test_migrate_from_mypy_with_multiple_modules() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set(
            "mypy-app.models",
            "disable_error_code",
            Some("union-attr".to_owned()),
        );
        mypy_cfg.set(
            "mypy-app.views",
            "enable_error_code",
            Some("union-attr".to_owned()),
        );

        let mut pyrefly_cfg = ConfigFile::default();

        let sub_configs = SubConfigs;
        let _ = sub_configs.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert_eq!(pyrefly_cfg.sub_configs.len(), 2);

        // Find the sub_config for app.models
        let models_config = pyrefly_cfg
            .sub_configs
            .iter()
            .find(|c| c.matches.to_string() == "app/models")
            .unwrap();
        let models_errors = models_config.settings.errors.as_ref().unwrap();
        assert_eq!(
            models_errors.severity(ErrorKind::MissingAttribute),
            Severity::Ignore
        );

        // Find the sub_config for app.views
        let views_config = pyrefly_cfg
            .sub_configs
            .iter()
            .find(|c| c.matches.to_string() == "app/views")
            .unwrap();
        let views_errors = views_config.settings.errors.as_ref().unwrap();
        assert_eq!(
            views_errors.severity(ErrorKind::MissingAttribute),
            Severity::Error
        );
    }

    #[test]
    fn test_migrate_from_mypy_with_no_error_codes() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set("mypy-app.models", "follow_imports", Some("skip".to_owned()));

        let mut pyrefly_cfg = ConfigFile::default();
        let default_sub_configs = pyrefly_cfg.sub_configs.clone();

        let sub_configs = SubConfigs;
        let _ = sub_configs.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert_eq!(pyrefly_cfg.sub_configs, default_sub_configs);
    }

    #[test]
    fn test_migrate_from_mypy_with_comma_separated_modules() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set(
            "mypy-app.models, app.views",
            "disable_error_code",
            Some("union-attr".to_owned()),
        );

        let mut pyrefly_cfg = ConfigFile::default();

        let sub_configs = SubConfigs;
        let _ = sub_configs.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert_eq!(pyrefly_cfg.sub_configs.len(), 2);

        // Check that both modules have the same error config
        let models_config = pyrefly_cfg
            .sub_configs
            .iter()
            .find(|c| c.matches.to_string() == "app/models")
            .unwrap();
        let views_config = pyrefly_cfg
            .sub_configs
            .iter()
            .find(|c| c.matches.to_string() == "app/views")
            .unwrap();

        let models_errors = models_config.settings.errors.as_ref().unwrap();
        let views_errors = views_config.settings.errors.as_ref().unwrap();

        assert_eq!(
            models_errors.severity(ErrorKind::MissingAttribute),
            Severity::Ignore
        );
        assert_eq!(
            views_errors.severity(ErrorKind::MissingAttribute),
            Severity::Ignore
        );
    }

    #[test]
    fn test_migrate_from_mypy_with_module_wildcards() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set(
            "mypy-app.*.models",
            "disable_error_code",
            Some("union-attr".to_owned()),
        );

        let mut pyrefly_cfg = ConfigFile::default();

        let sub_configs = SubConfigs;
        let _ = sub_configs.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert_eq!(pyrefly_cfg.sub_configs.len(), 1);

        let sub_config = &pyrefly_cfg.sub_configs[0];
        // Check that the module wildcard was converted to a glob
        assert_eq!(sub_config.matches.to_string(), "app/**/models");

        let errors = sub_config.settings.errors.as_ref().unwrap();
        assert_eq!(
            errors.severity(ErrorKind::MissingAttribute),
            Severity::Ignore
        );
    }

    #[test]
    fn test_migrate_from_mypy_with_empty_config() {
        let mypy_cfg = Ini::new();

        let mut pyrefly_cfg = ConfigFile::default();
        let default_sub_configs = pyrefly_cfg.sub_configs.clone();

        let sub_configs = SubConfigs;
        let _ = sub_configs.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert_eq!(pyrefly_cfg.sub_configs, default_sub_configs);
    }

    #[test]
    fn test_migrate_from_pyright() {
        let mut pyright_cfg = default_pyright_config();

        // Create execution environments with different error settings
        let env1 = ExecEnv {
            root: "src".to_owned(),
            errors: RuleOverrides {
                report_missing_imports: Some(Severity::Ignore),
                ..Default::default()
            },
        };

        let env2 = ExecEnv {
            root: "tests".to_owned(),
            errors: RuleOverrides {
                report_missing_module_source: Some(Severity::Error),
                ..Default::default()
            },
        };

        pyright_cfg.execution_environments = vec![env1, env2];

        let mut pyrefly_cfg = ConfigFile::default();

        let sub_configs = SubConfigs;
        let result = sub_configs.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

        assert!(result.is_ok());
        assert_eq!(pyrefly_cfg.sub_configs.len(), 2);

        // Find the sub_config for src
        let src_config = pyrefly_cfg
            .sub_configs
            .iter()
            .find(|c| c.matches.to_string() == "src")
            .unwrap();

        // Find the sub_config for tests
        let tests_config = pyrefly_cfg
            .sub_configs
            .iter()
            .find(|c| c.matches.to_string() == "tests")
            .unwrap();

        // Verify that the error settings were properly migrated
        let src_errors = src_config.settings.errors.as_ref().unwrap();
        let tests_errors = tests_config.settings.errors.as_ref().unwrap();

        assert_eq!(
            src_errors.severity(ErrorKind::MissingImport),
            Severity::Ignore
        );
        assert_eq!(
            tests_errors.severity(ErrorKind::MissingImport),
            Severity::Error
        );
    }

    #[test]
    fn test_migrate_from_pyright_empty() {
        let pyright_cfg = default_pyright_config();

        let mut pyrefly_cfg = ConfigFile::default();
        let default_sub_configs = pyrefly_cfg.sub_configs.clone();

        let sub_configs = SubConfigs;
        let result = sub_configs.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

        assert!(result.is_err());
        assert_eq!(pyrefly_cfg.sub_configs, default_sub_configs);
    }
}
