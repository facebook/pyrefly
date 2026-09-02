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

/// Configuration options for controlling how imports are followed.
pub struct IgnoreMissingImports;

impl IgnoreMissingImports {
    fn module_wildcards(sections: Vec<String>) -> Vec<ModuleWildcard> {
        sections
            .into_iter()
            .flat_map(|section| {
                section
                    .strip_prefix("mypy-")
                    .unwrap_or(&section)
                    .split(',')
                    .filter(|module| !module.is_empty())
                    .filter_map(|module| ModuleWildcard::new(module).ok())
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

impl ConfigOptionMigrater for IgnoreMissingImports {
    fn migrate_from_mypy(
        &self,
        mypy_cfg: &Ini,
        pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()> {
        let mut ignore_missing = Vec::new();
        let mut replace_with_any = Vec::new();

        if util::get_bool_or_default(mypy_cfg, "mypy", "ignore_missing_imports") {
            ignore_missing.push("*".to_owned());
        }
        if mypy_cfg
            .get("mypy", "follow_imports")
            .is_some_and(|value| value == "skip")
        {
            replace_with_any.push("*".to_owned());
        }
        let follow_untyped_imports =
            util::get_bool_or_default(mypy_cfg, "mypy", "follow_untyped_imports");
        // A global setting contributes a "*" that covers every module, which makes
        // the narrower per-module patterns redundant. Note that this assumes that
        // per-module settings contain only positive patterns (i.e., ones that add
        // modules). We currently do not support negative patterns.
        let ignore_missing_is_global = !ignore_missing.is_empty();
        let replace_with_any_is_global = !replace_with_any.is_empty();
        // Mypy does not follow untyped imports by default. Negated per-module
        // overrides must precede the catch-all because module globs use the
        // first matching pattern.
        let mut replace_untyped_with_any = Vec::new();

        util::visit_ini_sections(
            mypy_cfg,
            |section_name| section_name.starts_with("mypy-"),
            |section_name, ini| {
                if !ignore_missing_is_global
                    && util::get_bool_or_default(ini, section_name, "ignore_missing_imports")
                {
                    ignore_missing.push(section_name.to_owned());
                }
                if !replace_with_any_is_global
                    && ini
                        .get(section_name, "follow_imports")
                        .is_some_and(|value| value == "skip")
                {
                    replace_with_any.push(section_name.to_owned());
                }
                if let Some(section_value) = ini
                    .getboolcoerce(section_name, "follow_untyped_imports")
                    .ok()
                    .flatten()
                    && section_value != follow_untyped_imports
                {
                    replace_untyped_with_any.extend(
                        section_name
                            .strip_prefix("mypy-")
                            .expect("section names were filtered to the mypy prefix")
                            .split(',')
                            .map(|module| {
                                if section_value {
                                    format!("!{module}")
                                } else {
                                    module.to_owned()
                                }
                            }),
                    );
                }
            },
        );

        if follow_untyped_imports {
            if replace_untyped_with_any.is_empty() {
                // An explicit negative catch-all overrides the legacy preset. An
                // empty list would be omitted when the generated TOML is serialized.
                replace_untyped_with_any.push("!*".to_owned());
            }
        } else if !replace_untyped_with_any.is_empty() {
            // Explicit negative module globs replace rather than extend the preset,
            // so retain the catch-all after them.
            replace_untyped_with_any.push("*".to_owned());
        }

        let ignore_missing = Self::module_wildcards(ignore_missing);
        let replace_with_any = Self::module_wildcards(replace_with_any);
        let replace_untyped_with_any = Self::module_wildcards(replace_untyped_with_any);

        if !ignore_missing.is_empty() {
            pyrefly_cfg.root.ignore_missing_imports = Some(ignore_missing);
        }
        if !replace_with_any.is_empty() {
            pyrefly_cfg.root.replace_imports_with_any = Some(replace_with_any);
        }
        if !replace_untyped_with_any.is_empty() {
            pyrefly_cfg.root.replace_untyped_imports_with_any = Some(replace_untyped_with_any);
        }
        Ok(())
    }

    fn migrate_from_pyright(
        &self,
        _pyright_cfg: &PyrightConfig,
        _pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()> {
        Err(anyhow::anyhow!(
            "Pyright does not have direct equivalents for mypy's import handling options"
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
        assert_eq!(pyrefly_cfg.root.replace_imports_with_any, Some(expected));
        assert_eq!(pyrefly_cfg.root.ignore_missing_imports, None);
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

        assert_eq!(
            pyrefly_cfg.root.ignore_missing_imports,
            Some(vec![ModuleWildcard::new("some.*.project").unwrap()])
        );
        assert_eq!(
            pyrefly_cfg.root.replace_imports_with_any,
            Some(vec![ModuleWildcard::new("another.project").unwrap()])
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
    fn test_migrate_from_mypy_uses_legacy_follow_untyped_imports_default() {
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

        assert!(result.is_ok());
        assert_eq!(
            pyrefly_cfg.root.ignore_missing_imports,
            default_ignore_imports
        );
        assert_eq!(pyrefly_cfg.root.replace_untyped_imports_with_any, None);
    }

    #[test]
    fn test_migrate_from_mypy_follows_untyped_imports_globally() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set("mypy", "follow_untyped_imports", Some("True".to_owned()));

        let mut pyrefly_cfg = ConfigFile::default();
        let result = IgnoreMissingImports.migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg);

        assert!(result.is_ok());
        assert_eq!(
            pyrefly_cfg.root.replace_untyped_imports_with_any,
            Some(vec![ModuleWildcard::new("!*").unwrap()])
        );
    }

    #[test]
    fn test_migrate_from_mypy_replaces_selected_untyped_imports() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set("mypy", "follow_untyped_imports", Some("True".to_owned()));
        mypy_cfg.set(
            "mypy-untyped.*",
            "follow_untyped_imports",
            Some("False".to_owned()),
        );

        let mut pyrefly_cfg = ConfigFile::default();
        IgnoreMissingImports
            .migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg)
            .expect("per-module follow_untyped_imports should migrate");

        assert_eq!(
            pyrefly_cfg.root.replace_untyped_imports_with_any,
            Some(vec![ModuleWildcard::new("untyped.*").unwrap()])
        );
    }

    #[test]
    fn test_migrate_from_mypy_follows_selected_untyped_imports() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set(
            "mypy-untyped.*,vendor",
            "follow_untyped_imports",
            Some("True".to_owned()),
        );

        let mut pyrefly_cfg = ConfigFile::default();
        IgnoreMissingImports
            .migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg)
            .expect("per-module follow_untyped_imports should migrate");

        assert_eq!(
            pyrefly_cfg.root.replace_untyped_imports_with_any,
            Some(vec![
                ModuleWildcard::new("!untyped.*").unwrap(),
                ModuleWildcard::new("!vendor").unwrap(),
                ModuleWildcard::new("*").unwrap(),
            ])
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
        assert_eq!(pyrefly_cfg.root.replace_imports_with_any, Some(expected));
        assert_eq!(pyrefly_cfg.root.ignore_missing_imports, None);
    }

    #[test]
    fn test_migrate_from_mypy_mixed_import_options() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set("mypy", "ignore_missing_imports", Some("True".to_owned()));
        mypy_cfg.set(
            "mypy-some.module",
            "follow_imports",
            Some("skip".to_owned()),
        );

        let mut pyrefly_cfg = ConfigFile::default();
        IgnoreMissingImports
            .migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg)
            .expect("both import options should migrate independently");

        assert_eq!(
            pyrefly_cfg.root.ignore_missing_imports,
            Some(vec![ModuleWildcard::new("*").unwrap()])
        );
        assert_eq!(
            pyrefly_cfg.root.replace_imports_with_any,
            Some(vec![ModuleWildcard::new("some.module").unwrap()])
        );
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

        IgnoreMissingImports
            .migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg)
            .expect("global and module settings should migrate");

        assert_eq!(
            pyrefly_cfg.root.ignore_missing_imports,
            Some(vec![ModuleWildcard::new("*").unwrap()])
        );
        assert_eq!(pyrefly_cfg.root.replace_imports_with_any, None);
    }

    #[test]
    fn test_migrate_from_mypy_global_follow_imports_and_specific() {
        let mut mypy_cfg = Ini::new();
        mypy_cfg.set("mypy", "follow_imports", Some("skip".to_owned()));
        mypy_cfg.set(
            "mypy-some.module",
            "follow_imports",
            Some("skip".to_owned()),
        );

        let mut pyrefly_cfg = ConfigFile::default();

        IgnoreMissingImports
            .migrate_from_mypy(&mypy_cfg, &mut pyrefly_cfg)
            .expect("global and module settings should migrate");

        assert_eq!(
            pyrefly_cfg.root.replace_imports_with_any,
            Some(vec![ModuleWildcard::new("*").unwrap()])
        );
        assert_eq!(pyrefly_cfg.root.ignore_missing_imports, None);
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
