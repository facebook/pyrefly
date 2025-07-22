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
use crate::migration::mypy::util;
use crate::migration::pyright::PyrightConfig;

/// Configuration option for project includes (files, packages, modules)
pub struct ProjectIncludes;

impl ConfigOptionMigrater for ProjectIncludes {
    fn migrate_from_mypy(
        &self,
        mypy_cfg: &Ini,
        pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()> {
        // https://mypy.readthedocs.io/en/latest/config_file.html#import-discovery
        // files, packages, modules can only be set in the top level `[mypy]` global section
        let files = util::string_to_array(&mypy_cfg.get("mypy", "files"));
        let packages = util::string_to_array(&mypy_cfg.get("mypy", "packages"));
        let modules = util::string_to_array(&mypy_cfg.get("mypy", "modules"));

        let includes: Vec<String> = [files, packages, modules].into_iter().flatten().collect();

        if includes.is_empty() {
            return Err(anyhow::anyhow!("No project includes found in mypy config"));
        }

        pyrefly_cfg.project_includes = Globs::new(includes);
        Ok(())
    }

    fn migrate_from_pyright(
        &self,
        pyright_cfg: &PyrightConfig,
        pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()> {
        // In pyright, project includes are specified in the "include" field
        let includes = match &pyright_cfg.project_includes {
            Some(includes) => includes,
            None => {
                return Err(anyhow::anyhow!(
                    "No project includes found in pyright config"
                ));
            }
        };
        if includes.is_empty() {
            return Err(anyhow::anyhow!(
                "Empty project includes found in pyright config"
            ));
        }

        pyrefly_cfg.project_includes = includes.clone();
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
        mypy_cfg.set("mypy", "files", Some("src,test/some_test.py".to_owned()));
        mypy_cfg.set("mypy", "packages", Some("package1,package2".to_owned()));
        mypy_cfg.set("mypy", "modules", Some("module1,module2".to_owned()));

        let mut pyrefly_cfg = ConfigFile::default();

        let project_includes = ProjectIncludes;
        let _ = project_includes.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        let expected_includes = Globs::new(vec![
            "src".to_owned(),
            "test/some_test.py".to_owned(),
            "package1".to_owned(),
            "package2".to_owned(),
            "module1".to_owned(),
            "module2".to_owned(),
        ]);
        assert_eq!(pyrefly_cfg.project_includes, expected_includes);
    }

    #[test]
    fn test_migrate_from_mypy_empty() {
        let mypy_cfg = Ini::new();

        let mut pyrefly_cfg = ConfigFile::default();
        let default_includes = pyrefly_cfg.project_includes.clone();

        let project_includes = ProjectIncludes;
        let _ = project_includes.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert_eq!(pyrefly_cfg.project_includes, default_includes);
    }

    #[test]
    fn test_migrate_from_pyright() {
        let project_includes_globs =
            Globs::new(vec!["src/**/*.py".to_owned(), "test/**/*.py".to_owned()]);
        let mut pyright_cfg = default_pyright_config();
        pyright_cfg.project_includes = Some(project_includes_globs.clone());

        let mut pyrefly_cfg = ConfigFile::default();

        let project_includes = ProjectIncludes;
        let result = project_includes.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

        assert!(result.is_ok());
        assert_eq!(pyrefly_cfg.project_includes, project_includes_globs);
    }

    #[test]
    fn test_migrate_from_pyright_empty() {
        let pyright_cfg = default_pyright_config();

        let mut pyrefly_cfg = ConfigFile::default();
        let default_includes = pyrefly_cfg.project_includes.clone();

        let project_includes = ProjectIncludes;
        let result = project_includes.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

        assert!(result.is_err());
        assert_eq!(pyrefly_cfg.project_includes, default_includes);
    }

    #[test]
    fn test_migrate_from_pyright_empty_globs() {
        let mut pyright_cfg = default_pyright_config();
        pyright_cfg.project_includes = Some(Globs::new(vec![]));

        let mut pyrefly_cfg = ConfigFile::default();
        let default_includes = pyrefly_cfg.project_includes.clone();

        let project_includes = ProjectIncludes;
        let result = project_includes.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

        assert!(result.is_err());
        assert_eq!(pyrefly_cfg.project_includes, default_includes);
    }
}
