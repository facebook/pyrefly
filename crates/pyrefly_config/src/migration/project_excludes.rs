/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use configparser::ini::Ini;
use pyrefly_util::globs::Globs;

use crate::config::ConfigFile;
use crate::migration::config_option_migrater::ConfigOptionMigrater;
use crate::migration::mypy::regex_converter;
use crate::migration::pyright::PyrightConfig;

/// Configuration option for project excludes
pub struct ProjectExcludes;

impl ConfigOptionMigrater for ProjectExcludes {
    fn migrate_from_mypy(
        &self,
        mypy_cfg: &Ini,
        pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()> {
        // https://mypy.readthedocs.io/en/latest/config_file.html#import-discovery
        // excludes can only be set in the top level `[mypy]` global section
        let exclude = mypy_cfg.get("mypy", "exclude");

        if exclude.is_none() {
            return Err(anyhow::anyhow!("No exclude found in mypy config"));
        }

        let exclude_regex = exclude.unwrap();
        let patterns = regex_converter::convert(&exclude_regex)?;

        if patterns.is_empty() {
            return Err(anyhow::anyhow!("No valid patterns found in exclude regex"));
        }

        pyrefly_cfg.project_excludes = Globs::new(patterns)?;
        Ok(())
    }
    fn migrate_from_pyright(
        &self,
        pyright_cfg: &PyrightConfig,
        pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()> {
        // In pyright, project excludes are specified in the "exclude" field
        if let Some(excludes) = &pyright_cfg.project_excludes {
            if excludes.is_empty() {
                return Err(anyhow::anyhow!(
                    "No project excludes found in pyright config"
                ));
            }
            pyrefly_cfg.project_excludes = excludes.clone();
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                "No project excludes found in pyright config"
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::migration::test_util::default_pyright_config;

    #[test]
    fn test_migrate_from_mypy() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set(
            "mypy",
            "exclude",
            Some("src/include/|other_src/include/".to_owned()),
        );

        let mut pyrefly_cfg = ConfigFile::default();

        let project_excludes = ProjectExcludes;
        let _ = project_excludes.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        let expected_excludes = Globs::new(vec![
            "**/src/include/".to_owned(),
            "**/other_src/include/".to_owned(),
        ])
        .unwrap();
        assert_eq!(pyrefly_cfg.project_excludes, expected_excludes);
    }

    #[test]
    fn test_migrate_from_pyright() {
        let project_excludes_globs =
            Globs::new(vec!["src/**/*.py".to_owned(), "test/**/*.py".to_owned()]).unwrap();
        let mut pyright_cfg = default_pyright_config();
        pyright_cfg.project_excludes = Some(project_excludes_globs.clone());

        let mut pyrefly_cfg = ConfigFile::default();

        let project_excludes = ProjectExcludes;
        let result = project_excludes.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

        assert!(result.is_ok());
        assert_eq!(pyrefly_cfg.project_excludes, project_excludes_globs);
    }

    #[test]
    fn test_migrate_from_pyright_empty() {
        let pyright_cfg = default_pyright_config();

        let mut pyrefly_cfg = ConfigFile::default();
        let default_excludes = pyrefly_cfg.project_excludes.clone();

        let project_excludes = ProjectExcludes;
        let result = project_excludes.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

        assert!(result.is_err());
        assert_eq!(pyrefly_cfg.project_excludes, default_excludes);
    }

    #[test]
    fn test_migrate_from_pyright_empty_globs() {
        let mut pyright_cfg = default_pyright_config();
        pyright_cfg.project_excludes = Some(Globs::new(vec![]).unwrap());

        let mut pyrefly_cfg = ConfigFile::default();
        let default_excludes = pyrefly_cfg.project_excludes.clone();

        let project_excludes = ProjectExcludes;
        let result = project_excludes.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

        assert!(result.is_err());
        assert_eq!(pyrefly_cfg.project_excludes, default_excludes);
    }
}
