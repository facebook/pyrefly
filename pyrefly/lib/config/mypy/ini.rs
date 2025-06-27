/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::str::FromStr;

use anyhow::Context as _;
use configparser::ini::Ini;
use configparser::ini::IniDefault;
use pyrefly_python::sys_info::PythonPlatform;
use pyrefly_python::sys_info::PythonVersion;
use pyrefly_util::globs::Glob;
use pyrefly_util::globs::Globs;
use serde::Deserialize;

use crate::config::base::ConfigBase;
use crate::config::config::ConfigFile;
use crate::config::config::SubConfig;
use crate::config::environment::environment::PythonEnvironment;
use crate::config::error::ErrorDisplayConfig;
use crate::config::mypy::regex_converter;
use crate::module::wildcard::ModuleWildcard;
#[derive(Clone, Debug, Deserialize)]
pub struct MypyConfig {
    files: Option<Vec<String>>,
    packages: Option<Vec<String>>,
    modules: Option<Vec<String>>,
    #[allow(dead_code)]
    #[serde(rename = "exclude")]
    exclude_regex: Option<String>,
    #[serde(rename = "mypy_path")]
    search_path: Option<Vec<PathBuf>>,
    #[serde(rename = "platform")]
    python_platform: Option<String>,
    python_version: Option<PythonVersion>,
    #[serde(rename = "python_executable")]
    python_interpreter: Option<PathBuf>,
    #[serde(
        rename = "follow_untyped_imports",
        default = "ConfigFile::default_true"
    )]
    use_untyped_imports: bool,
    disable_error_code: Option<Vec<String>>,
    enable_error_code: Option<Vec<String>>,
}

impl MypyConfig {
    pub fn parse_mypy_config(ini_path: &Path) -> anyhow::Result<ConfigFile> {
        fn ini_string_to_array(value: &Option<String>) -> Vec<String> {
            match value {
                Some(value) => value
                    .split(',')
                    .map(|x| x.trim().to_owned())
                    .filter(|s| !s.is_empty())
                    .collect(),
                _ => Vec::new(),
            }
        }

        // getboolcoerce returns an Option<bool> to indicate if the value was set or not,
        // wrapped in a Result to indicate if the value could be parsed as a bool.
        // We assume that mypy configs are well formed, since they're already used by mypy itself,
        // so it's semantically safe to smoosh this down to an Option and unwrap_or_default it.
        fn bool_or_default(config: &Ini, section: &str, key: &str) -> bool {
            config
                .getboolcoerce(section, key)
                .ok()
                .flatten()
                .unwrap_or_default()
        }

        fn make_error_config(
            disables: Vec<String>,
            enables: Vec<String>,
        ) -> Option<ErrorDisplayConfig> {
            let mut errors = HashMap::new();
            for error_code in disables {
                errors.insert(error_code, false);
            }
            // enable_error_code overrides disable_error_code
            for error_code in enables {
                errors.insert(error_code, true);
            }
            crate::config::mypy::code_to_kind(errors)
        }

        let mut default = IniDefault::default();
        // Need to set this to properly parse things like the PyTorch mypy.ini file,
        // Which has a multiline `files` comment that gets parsed incorrectly without this.
        default.multiline = true;
        let mut config = Ini::new_from_defaults(default);
        config
            .load(ini_path)
            .map_err(|e| anyhow::anyhow!(e))
            .with_context(|| {
                format!("While trying to read mypy config at {}", ini_path.display())
            })?;

        // https://mypy.readthedocs.io/en/latest/config_file.html#import-discovery
        // files, packages, modules, mypy_path, python_executable, python_version, and excludes can only be set in the top level `[mypy]` global section
        let files: Vec<String> = ini_string_to_array(&config.get("mypy", "files"));
        let packages: Vec<String> = ini_string_to_array(&config.get("mypy", "packages")); // list of strings
        let modules: Vec<String> = ini_string_to_array(&config.get("mypy", "modules")); // list of strings
        let excludes = config.get("mypy", "exclude"); // regex
        let mypy_path = config.get("mypy", "mypy_path"); // string
        let python_executable = config.get("mypy", "python_executable");
        let python_version = config.get("mypy", "python_version");
        let disable_error_code: Vec<String> =
            ini_string_to_array(&config.get("mypy", "disable_error_code"));
        let enable_error_code: Vec<String> =
            ini_string_to_array(&config.get("mypy", "enable_error_code"));
        // follow_untyped_imports may be used as a global or per-module setting. As a per-module setting, it's used to
        // indicate that the module should be ignored if it's untyped.
        // Pyrefly's use_untyped_imports is only a global setting.
        // We handle this by *only* checking the for the global config.
        let follow_untyped_imports = config
            .getboolcoerce("mypy", "follow_untyped_imports")
            .ok()
            .flatten();

        let mut replace_imports: Vec<String> = Vec::new();
        // This is the list of mypy per-module section headers and the error configs found in those sections.
        // We'll split the headers into separate modules later and turn each one into a subconfig.
        let mut sub_configs: Vec<(String, ErrorDisplayConfig)> = vec![];
        for section in &config.sections() {
            if !section.starts_with("mypy-") {
                continue;
            }

            if bool_or_default(&config, section, "ignore_missing_imports")
                || config
                    .get(section, "follow_imports")
                    .is_some_and(|val| val == "skip")
            {
                replace_imports.push(section.to_owned());
            }

            // For subconfigs, the only config that needs to be extracted is enable/disable error codes.
            let disable_error_code: Vec<String> =
                ini_string_to_array(&config.get(section, "disable_error_code"));
            let enable_error_code: Vec<String> =
                ini_string_to_array(&config.get(section, "enable_error_code"));
            let errors = make_error_config(disable_error_code, enable_error_code);
            if let Some(errors) = errors {
                sub_configs.push((section.strip_prefix("mypy-").unwrap().to_owned(), errors));
            }
        }

        let mut cfg = ConfigFile::default();

        let project_includes = [files, packages, modules]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        if !project_includes.is_empty() {
            cfg.project_includes = Globs::new(project_includes);
        }

        if let Some(exclude_regex) = excludes {
            let patterns = regex_converter::convert(&exclude_regex)?;
            if !patterns.is_empty() {
                cfg.project_excludes = Globs::new(patterns);
            }
        }

        if let Some(python_interpreter) = python_executable {
            // TODO: Add handling for when these are virtual environments
            // Is this something we can auto detect instead of hardcoding here.
            cfg.python_interpreter = PathBuf::from_str(&python_interpreter).ok();
        }

        if let Some(version) = python_version {
            cfg.python_environment.python_version = PythonVersion::from_str(&version).ok();
        }

        if let Some(search_paths) = mypy_path {
            let value: Vec<PathBuf> = search_paths
                .split([',', ':'])
                .map(|x| x.trim().to_owned())
                .filter(|x| !x.is_empty())
                .map(PathBuf::from)
                .collect();
            cfg.search_path_from_file = value;
        }
        cfg.use_untyped_imports = follow_untyped_imports.unwrap_or(cfg.use_untyped_imports);
        cfg.root.replace_imports_with_any = replace_imports
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
            .filter(|x| x.is_some())
            .collect();

        cfg.root.errors = make_error_config(disable_error_code, enable_error_code);

        let sub_configs = sub_configs
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
            .map(|(matches, errors)| SubConfig {
                matches,
                settings: ConfigBase {
                    errors,
                    ..Default::default()
                },
            })
            .collect::<Vec<_>>();
        cfg.sub_configs = sub_configs;

        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use pyrefly_util::fs_anyhow;

    use super::*;
    use crate::error::kind::ErrorKind;

    #[test]
    fn test_run_mypy() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let input_path = tmp.path().join("mypy.ini");
        // This config is derived from the pytorch mypy.ini.
        let mypy = br#"[mypy]
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

[mypy-do.follow.*]
follow_untyped_imports = True

[mypy-comma,separated,projects]
ignore_missing_imports = True
 "#;
        fs_anyhow::write(&input_path, mypy)?;

        let cfg = MypyConfig::parse_mypy_config(&input_path)?;
        let project_includes = Globs::new(vec![
            "src".to_owned(),
            "other_src".to_owned(),
            "test/some_test.py".to_owned(),
        ]);
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
        ]);
        assert_eq!(cfg.project_excludes, expected_excludes);
        assert_eq!(cfg.replace_imports_with_any(None).len(), 5);
        assert!(cfg.use_untyped_imports);
        Ok(())
    }

    #[test]
    fn test_no_overwrite_missing_includes() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let input_path = tmp.path().join("mypy.ini");
        // This config is derived from the pytorch mypy.ini.
        let mypy = br#"[mypy]
unknown_option = True
"#;
        fs_anyhow::write(&input_path, mypy)?;

        let cfg = MypyConfig::parse_mypy_config(&input_path)?;
        let default = ConfigFile::default();
        assert_eq!(cfg.project_includes, default.project_includes);
        assert_eq!(cfg.project_excludes, default.project_excludes);
        Ok(())
    }
    #[test]
    fn test_disable_errors() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let input_path = tmp.path().join("mypy.ini");
        let src = br#"[mypy]
disable_error_code = union-attr
"#;
        fs_anyhow::write(&input_path, src)?;
        let mut cfg = MypyConfig::parse_mypy_config(&input_path)?;
        cfg.configure();
        let errors = cfg.errors(tmp.path());
        assert!(!errors.is_enabled(ErrorKind::MissingAttribute));
        Ok(())
    }

    #[test]
    fn test_replace_imports() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let input_path = tmp.path().join("mypy.ini");
        let src = br#"[mypy]
test_flag = True

# replace some.*.project with Any.
[mypy-some.*.project]
ignore_missing_imports = True

# Don't replace this one, because it's not `follow_imports = skip`.
[mypy-another.project]
follow_imports = silent
"#;
        fs_anyhow::write(&input_path, src)?;
        let cfg = MypyConfig::parse_mypy_config(&input_path)?;
        assert_eq!(
            cfg.replace_imports_with_any(None),
            vec![ModuleWildcard::new("some.*.project").unwrap(),]
        );
        Ok(())
    }

    #[test]
    fn test_subconfigs() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let input_path = tmp.path().join("mypy.ini");
        let src = br#"[mypy]
files = src

[mypy-src.*.linux]
disable_error_code = union-attr

[mypy-another.project]
follow_imports = silent
"#;
        fs_anyhow::write(&input_path, src)?;
        let mut cfg = MypyConfig::parse_mypy_config(&input_path)?;
        cfg.configure();

        assert!(
            cfg.errors(Path::new("src"))
                .is_enabled(ErrorKind::MissingAttribute)
        );
        assert!(
            !cfg.errors(Path::new("src/linux"))
                .is_enabled(ErrorKind::MissingAttribute)
        );
        Ok(())
    }

    #[test]
    fn test_no_use_untyped_imports() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let input_path = tmp.path().join("mypy.ini");
        let src = br#"[mypy]
follow_untyped_imports = False

[mypy-this.setting.ignored]
follow_untyped_imports = True
"#;
        fs_anyhow::write(&input_path, src)?;
        let mut cfg = MypyConfig::parse_mypy_config(&input_path)?;
        cfg.configure();

        assert!(!cfg.use_untyped_imports);
        Ok(())
    }
}
