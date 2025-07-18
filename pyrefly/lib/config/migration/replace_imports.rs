/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use configparser::ini::Ini;

use crate::config::config::ConfigFile;
use crate::config::migration::config_option_migrater::ConfigOptionMigrater;
use crate::config::migration::pyright::PyrightConfig;
use crate::config::migration::utils;
use crate::module::wildcard::ModuleWildcard;

/// Configuration option for replacing imports with Any
pub struct ReplaceImports;

impl ConfigOptionMigrater for ReplaceImports {
    fn migrate_from_mypy(
        &self,
        mypy_cfg: &Ini,
        pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()> {
        let mut replace_imports: Vec<String> = Vec::new();

        // Check all sections for ignore_missing_imports or follow_imports=skip
        utils::visit_ini_sections(
            mypy_cfg,
            |section_name| section_name.starts_with("mypy-"),
            |section_name, ini| {
                if utils::get_bool_or_default(ini, section_name, "ignore_missing_imports")
                    || ini
                        .get(section_name, "follow_imports")
                        .is_some_and(|val| val == "skip")
                {
                    replace_imports.push(section_name.to_owned());
                }
            },
        );

        if replace_imports.is_empty() {
            return Err(anyhow::anyhow!("No replace imports found in mypy config"));
        }

        // Convert to ModuleWildcard objects
        let result: Vec<ModuleWildcard> = replace_imports
            .into_iter()
            .flat_map(|x| {
                if let Some(stripped) = x.strip_prefix("mypy-") {
                    stripped
                        .split(",")
                        .filter(|x| !x.is_empty())
                        .map(|x| ModuleWildcard::new(x).ok())
                        .collect()
                } else {
                    vec![ModuleWildcard::new(&x).ok()]
                }
            })
            .flatten()
            .collect();

        if result.is_empty() {
            return Err(anyhow::anyhow!(
                "Failed to create valid module wildcards for replace imports"
            ));
        }

        pyrefly_cfg.root.replace_imports_with_any = Some(result);
        Ok(())
    }

    fn migrate_from_pyright(
        &self,
        _pyright_cfg: &PyrightConfig,
        _pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()> {
        Err(anyhow::anyhow!(
            "Pyright does not have a direct equivalent for ignore_missing_imports or follow_imports=skip"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::migration::test_utils::default_pyright_config;

    #[test]
    fn test_migrate_from_mypy_ignore_missing_imports() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set(
            "mypy-some.*.project",
            "ignore_missing_imports",
            Some("True".to_owned()),
        );

        let mut pyrefly_cfg = ConfigFile::default();

        let replace_imports = ReplaceImports;
        let _ = replace_imports.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        let expected = vec![ModuleWildcard::new("some.*.project").unwrap()];
        assert_eq!(pyrefly_cfg.root.replace_imports_with_any, Some(expected));
    }

    #[test]
    fn test_migrate_from_mypy_follow_imports_skip() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set(
            "mypy-another.project",
            "follow_imports",
            Some("skip".to_owned()),
        );

        let mut pyrefly_cfg = ConfigFile::default();

        let replace_imports = ReplaceImports;
        let _ = replace_imports.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        let expected = vec![ModuleWildcard::new("another.project").unwrap()];
        assert_eq!(pyrefly_cfg.root.replace_imports_with_any, Some(expected));
    }

    #[test]
    fn test_migrate_from_mypy_multiple_sections() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set(
            "mypy-some.*.project",
            "ignore_missing_imports",
            Some("True".to_owned()),
        );
        mypy_cfg.set(
            "mypy-another.project",
            "follow_imports",
            Some("skip".to_owned()),
        );
        mypy_cfg.set(
            "mypy-third.project",
            "follow_imports",
            Some("silent".to_owned()),
        ); // This should be ignored

        let mut pyrefly_cfg = ConfigFile::default();

        let replace_imports = ReplaceImports;
        let _ = replace_imports.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        let expected = [
            ModuleWildcard::new("some.*.project").unwrap(),
            ModuleWildcard::new("another.project").unwrap(),
        ];
        assert_eq!(
            pyrefly_cfg
                .root
                .replace_imports_with_any
                .as_ref()
                .unwrap()
                .len(),
            2
        );
        assert!(
            pyrefly_cfg
                .root
                .replace_imports_with_any
                .as_ref()
                .unwrap()
                .contains(&expected[0])
        );
        assert!(
            pyrefly_cfg
                .root
                .replace_imports_with_any
                .as_ref()
                .unwrap()
                .contains(&expected[1])
        );
    }

    #[test]
    fn test_migrate_from_mypy_comma_separated() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set(
            "mypy-module1,module2",
            "ignore_missing_imports",
            Some("True".to_owned()),
        );

        let mut pyrefly_cfg = ConfigFile::default();

        let replace_imports = ReplaceImports;
        let _ = replace_imports.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        let expected = [
            ModuleWildcard::new("module1").unwrap(),
            ModuleWildcard::new("module2").unwrap(),
        ];
        assert_eq!(
            pyrefly_cfg
                .root
                .replace_imports_with_any
                .as_ref()
                .unwrap()
                .len(),
            2
        );
        assert!(
            pyrefly_cfg
                .root
                .replace_imports_with_any
                .as_ref()
                .unwrap()
                .contains(&expected[0])
        );
        assert!(
            pyrefly_cfg
                .root
                .replace_imports_with_any
                .as_ref()
                .unwrap()
                .contains(&expected[1])
        );
    }

    #[test]
    fn test_migrate_from_mypy_empty() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set("mypy", "files", Some("src".to_owned()));
        mypy_cfg.set(
            "mypy-some.project",
            "follow_imports",
            Some("normal".to_owned()),
        );

        let mut pyrefly_cfg = ConfigFile::default();
        let default_replace_imports = pyrefly_cfg.root.replace_imports_with_any.clone();

        let replace_imports = ReplaceImports;
        let _ = replace_imports.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert_eq!(
            pyrefly_cfg.root.replace_imports_with_any,
            default_replace_imports
        );
    }

    #[test]
    fn test_migrate_from_pyright() {
        let pyright_cfg = default_pyright_config();
        let mut pyrefly_cfg = ConfigFile::default();
        let default_replace_imports = pyrefly_cfg.root.replace_imports_with_any.clone();

        let replace_imports = ReplaceImports;
        let result = replace_imports.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

        assert!(result.is_err());
        assert_eq!(
            pyrefly_cfg.root.replace_imports_with_any,
            default_replace_imports
        );
    }
}
