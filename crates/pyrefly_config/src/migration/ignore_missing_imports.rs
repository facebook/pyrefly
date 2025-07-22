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
use crate::module_wildcard::ModuleWildcard;

/// Configuration option for ignoring missing imports
pub struct IgnoreMissingImports;

impl IgnoreMissingImports {
    /// Helper function to check if a section has ignore_missing_imports=true or follow_imports=skip
    fn should_ignore_imports(&self, ini: &Ini, section_name: &str) -> bool {
        util::get_bool_or_default(ini, section_name, "ignore_missing_imports")
            || ini
                .get(section_name, "follow_imports")
                .is_some_and(|val| val == "skip")
    }
}

impl ConfigOptionMigrater for IgnoreMissingImports {
    fn migrate_from_mypy(
        &self,
        mypy_cfg: &Ini,
        pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()> {
        let mut ignore_imports: Vec<String> = Vec::new();
        let mut ignore_all_missing_imports = false;

        // Check if the default "mypy" section has ignore_missing_imports or follow_imports=skip
        if self.should_ignore_imports(mypy_cfg, "mypy") {
            ignore_all_missing_imports = true;
        }

        // Check all sections for ignore_missing_imports or follow_imports=skip
        util::visit_ini_sections(
            mypy_cfg,
            |section_name| section_name.starts_with("mypy-"),
            |section_name, ini| {
                if self.should_ignore_imports(ini, section_name) {
                    ignore_imports.push(section_name.to_owned());
                }
            },
        );

        if ignore_imports.is_empty() && !ignore_all_missing_imports {
            return Err(anyhow::anyhow!(
                "No ignore missing imports found in mypy config"
            ));
        }

        // If we have a global ignore_missing_imports, add a wildcard for all
        if ignore_all_missing_imports {
            ignore_imports.push("*".to_owned());
        }

        // Convert to ModuleWildcard objects
        let result: Vec<ModuleWildcard> = ignore_imports
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

        pyrefly_cfg.root.ignore_missing_imports = Some(result);
        pyrefly_cfg.root.replace_imports_with_any = Some(Default::default());
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
    use crate::migration::test_util::default_pyright_config;

    #[test]
    fn test_migrate_from_mypy_ignore_missing_imports() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set(
            "mypy-some.*.project",
            "ignore_missing_imports",
            Some("True".to_owned()),
        );

        let mut pyrefly_cfg = ConfigFile::default();

        let ignore_imports = IgnoreMissingImports;
        let _ = ignore_imports.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        let expected = vec![ModuleWildcard::new("some.*.project").unwrap()];
        assert_eq!(pyrefly_cfg.root.ignore_missing_imports, Some(expected));
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

        let ignore_imports = IgnoreMissingImports;
        let _ = ignore_imports.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        let expected = vec![ModuleWildcard::new("another.project").unwrap()];
        assert_eq!(pyrefly_cfg.root.ignore_missing_imports, Some(expected));
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

        let ignore_imports = IgnoreMissingImports;
        let _ = ignore_imports.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        let expected = [
            ModuleWildcard::new("some.*.project").unwrap(),
            ModuleWildcard::new("another.project").unwrap(),
        ];
        assert_eq!(
            pyrefly_cfg
                .root
                .ignore_missing_imports
                .as_ref()
                .unwrap()
                .len(),
            2
        );
        assert!(
            pyrefly_cfg
                .root
                .ignore_missing_imports
                .as_ref()
                .unwrap()
                .contains(&expected[0])
        );
        assert!(
            pyrefly_cfg
                .root
                .ignore_missing_imports
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

        let ignore_imports = IgnoreMissingImports;
        let _ = ignore_imports.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        let expected = [
            ModuleWildcard::new("module1").unwrap(),
            ModuleWildcard::new("module2").unwrap(),
        ];
        assert_eq!(
            pyrefly_cfg
                .root
                .ignore_missing_imports
                .as_ref()
                .unwrap()
                .len(),
            2
        );
        assert!(
            pyrefly_cfg
                .root
                .ignore_missing_imports
                .as_ref()
                .unwrap()
                .contains(&expected[0])
        );
        assert!(
            pyrefly_cfg
                .root
                .ignore_missing_imports
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
        let default_ignore_imports = pyrefly_cfg.root.ignore_missing_imports.clone();

        let ignore_imports = IgnoreMissingImports;
        let result = ignore_imports.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert!(result.is_err());
        assert_eq!(
            pyrefly_cfg.root.ignore_missing_imports,
            default_ignore_imports
        );
    }

    #[test]
    fn test_migrate_from_mypy_global_ignore_missing_imports() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set("mypy", "ignore_missing_imports", Some("True".to_owned()));

        let mut pyrefly_cfg = ConfigFile::default();

        let ignore_imports = IgnoreMissingImports;
        let _ = ignore_imports.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        let expected = vec![ModuleWildcard::new("*").unwrap()];
        assert_eq!(pyrefly_cfg.root.ignore_missing_imports, Some(expected));
    }

    #[test]
    fn test_migrate_from_mypy_global_follow_imports_skip() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set("mypy", "follow_imports", Some("skip".to_owned()));

        let mut pyrefly_cfg = ConfigFile::default();

        let ignore_imports = IgnoreMissingImports;
        let _ = ignore_imports.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        let expected = vec![ModuleWildcard::new("*").unwrap()];
        assert_eq!(pyrefly_cfg.root.ignore_missing_imports, Some(expected));
    }

    #[test]
    fn test_migrate_from_mypy_global_and_specific() {
        let mut mypy_cfg = Ini::new();
        // Global setting
        mypy_cfg.set("mypy", "ignore_missing_imports", Some("True".to_owned()));
        // Specific section
        mypy_cfg.set(
            "mypy-some.module",
            "ignore_missing_imports",
            Some("True".to_owned()),
        );

        let mut pyrefly_cfg = ConfigFile::default();

        let ignore_imports = IgnoreMissingImports;
        let _ = ignore_imports.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        // Should contain both the specific module and the global wildcard
        let expected = [
            ModuleWildcard::new("some.module").unwrap(),
            ModuleWildcard::new("*").unwrap(),
        ];
        assert_eq!(
            pyrefly_cfg
                .root
                .ignore_missing_imports
                .as_ref()
                .unwrap()
                .len(),
            2
        );
        assert!(
            pyrefly_cfg
                .root
                .ignore_missing_imports
                .as_ref()
                .unwrap()
                .contains(&expected[0])
        );
        assert!(
            pyrefly_cfg
                .root
                .ignore_missing_imports
                .as_ref()
                .unwrap()
                .contains(&expected[1])
        );
    }

    #[test]
    fn test_migrate_from_pyright() {
        let pyright_cfg = default_pyright_config();
        let mut pyrefly_cfg = ConfigFile::default();
        let default_ignore_imports = pyrefly_cfg.root.ignore_missing_imports.clone();

        let ignore_imports = IgnoreMissingImports;
        let result = ignore_imports.migrate_from_pyright(&pyright_cfg, &mut pyrefly_cfg);

        assert!(result.is_err());
        assert_eq!(
            pyrefly_cfg.root.ignore_missing_imports,
            default_ignore_imports
        );
    }
}
