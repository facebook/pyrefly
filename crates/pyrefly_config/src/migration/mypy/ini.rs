/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;

use anyhow::Context as _;
use configparser::ini::Ini;
use configparser::ini::IniDefault;

use crate::config::ConfigFile;
use crate::migration::config_option_migrater::ConfigOptionMigrater;
use crate::migration::error_codes::ErrorCodes;
use crate::migration::ignore_missing_imports::IgnoreMissingImports;
use crate::migration::project_excludes::ProjectExcludes;
use crate::migration::project_includes::ProjectIncludes;
use crate::migration::python_interpreter::PythonInterpreter;
use crate::migration::python_version::PythonVersionConfig;
use crate::migration::search_path::SearchPath;
use crate::migration::sub_configs::SubConfigs;
use crate::migration::untyped_def_behavior::UntypedDefBehaviorConfig;

pub fn parse_mypy_config(ini_path: &Path) -> anyhow::Result<ConfigFile> {
    let mut default = IniDefault::default();
    // Need to set this to properly parse things like the PyTorch mypy.ini file,
    // Which has a multiline `files` comment that gets parsed incorrectly without this.
    default.multiline = true;
    let mut config = Ini::new_from_defaults(default);
    config
        .load(ini_path)
        .map_err(|e| anyhow::anyhow!(e))
        .with_context(|| format!("While trying to read mypy config at {}", ini_path.display()))?;

    let mut cfg = ConfigFile::default();

    let config_options: Vec<Box<dyn ConfigOptionMigrater>> = vec![
        Box::new(ProjectIncludes),
        Box::new(ProjectExcludes),
        Box::new(PythonInterpreter),
        Box::new(PythonVersionConfig),
        Box::new(IgnoreMissingImports),
        Box::new(SearchPath),
        Box::new(ErrorCodes),
        Box::new(SubConfigs),
        Box::new(UntypedDefBehaviorConfig),
    ];

    // Iterate through all config options and apply them to the config
    for option in config_options {
        // Ignore errors for now, we can use this in the future if we want to print out error messages or use for logging purpose
        let _ = option.migrate_from_mypy(&config, &mut cfg);
    }

    // All configuration options are now processed by the individual config option migraters

    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use pyrefly_util::fs_anyhow;
    use pyrefly_util::globs::Globs;

    use super::*;

    #[test]
    fn test_run_mypy() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let input_path = tmp.path().join("mypy.ini");
        // This config is derived from the pytorch mypy.ini.
        let mypy = r#"[mypy]
files =
    src,
    other_src,
    test/some_test.py,

mypy_path = some_paths:comma,separated

unknown_option = True

exclude = src/include/|other_src/include/|src/specific/bad/file.py

[mypy-some.*.project]
ignore_missing_imports = True

[mypy-some.specific.project.subdir]
ignore_missing_imports = True

[mypy-stricter.on.this.*]
check_untyped_defs = True

[mypy-comma,separated,projects]
ignore_missing_imports = True
 "#;
        fs_anyhow::write(&input_path, mypy)?;

        let cfg = parse_mypy_config(&input_path)?;
        let project_includes = Globs::new(vec![
            "src".to_owned(),
            "other_src".to_owned(),
            "test/some_test.py".to_owned(),
        ])
        .unwrap();
        assert_eq!(cfg.project_includes, project_includes);

        assert_eq!(
            cfg.search_path_from_file,
            vec![
                PathBuf::from("some_paths"),
                PathBuf::from("comma"),
                PathBuf::from("separated"),
            ]
        );

        let expected_excludes = Globs::new(vec![
            "**/src/include/".to_owned(),
            "**/other_src/include/".to_owned(),
            "**/src/specific/bad/file.py".to_owned(),
        ])
        .unwrap();
        assert_eq!(cfg.project_excludes, expected_excludes);
        assert_eq!(cfg.root.ignore_missing_imports.unwrap().len(), 5);
        assert_eq!(cfg.root.replace_imports_with_any.unwrap().len(), 0);
        Ok(())
    }

    #[test]
    fn test_no_overwrite_missing_includes() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let input_path = tmp.path().join("mypy.ini");
        // This config is derived from the pytorch mypy.ini.
        let mypy = r#"[mypy]
unknown_option = True
"#;
        fs_anyhow::write(&input_path, mypy)?;

        let cfg = parse_mypy_config(&input_path)?;
        let default = ConfigFile::default();
        assert_eq!(cfg.project_includes, default.project_includes);
        assert_eq!(cfg.project_excludes, default.project_excludes);
        Ok(())
    }
}
