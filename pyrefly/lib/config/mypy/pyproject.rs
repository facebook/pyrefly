/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::path::PathBuf;

use itertools::Itertools;
use pyrefly_python::sys_info::PythonPlatform;
use pyrefly_python::sys_info::PythonVersion;
use pyrefly_util::globs::Glob;
use pyrefly_util::globs::Globs;
use serde::Deserialize;
use serde_with::OneOrMany;
use serde_with::serde_as;

use crate::config::base::ConfigBase;
use crate::config::config::ConfigFile;
use crate::config::config::SubConfig;
use crate::config::error::ErrorDisplayConfig;
use crate::config::mypy::regex_converter;
use crate::module::wildcard::ModuleWildcard;

// A pyproject.toml Mypy config differs a bit from the INI format:
// - The [mypy] section is written as [tool.mypy]
// - The `exclude` regex can be written as an array of regexes OR a single string
// - Per-module configs go in [[tool.mypy.overrides]]
// - Any of the fields may be a string, a list of strings, or a string that needs to be split (possibly with a delimiter besides `,`.)
// This means it's hard to share code with the INI parsing implementation.

#[allow(dead_code)]
#[serde_as]
#[derive(Debug, Deserialize)]
struct ModuleSection {
    #[serde_as(as = "OneOrMany<_>")]
    module: Vec<String>,
    #[serde(default)]
    ignore_missing_imports: bool,
    #[serde(default)]
    follow_untyped_imports: bool,
    #[serde(default)]
    follow_imports: Option<String>,
    #[serde_as(as = "Option<OneOrMany<_>>")]
    #[serde(default)]
    disable_error_code: Option<Vec<String>>,
    #[serde_as(as = "Option<OneOrMany<_>>")]
    #[serde(default)]
    enable_error_code: Option<Vec<String>>,
}

#[serde_as]
#[derive(Debug, Deserialize)]
struct MypySection {
    #[serde_as(as = "Option<OneOrMany<_>>")]
    #[serde(default)]
    files: Option<Vec<String>>,
    #[serde_as(as = "Option<OneOrMany<_>>")]
    #[serde(default)]
    packages: Option<Vec<String>>,
    #[serde_as(as = "Option<OneOrMany<_>>")]
    #[serde(default)]
    modules: Option<Vec<String>>,
    #[serde_as(as = "Option<OneOrMany<_>>")]
    #[serde(rename = "exclude", default)]
    exclude_regex: Option<Vec<String>>,
    #[serde_as(as = "Option<OneOrMany<_>>")]
    #[serde(rename = "mypy_path", default)]
    search_path: Option<Vec<String>>,
    #[serde(rename = "platform")]
    python_platform: Option<String>,
    python_version: Option<PythonVersion>,
    #[serde(rename = "python_executable")]
    python_interpreter: Option<PathBuf>,
    disable_error_code: Option<Vec<String>>,
    enable_error_code: Option<Vec<String>>,
    #[serde(default)]
    follow_untyped_imports: Option<bool>,
    #[serde(default)]
    overrides: Vec<ModuleSection>,
}

#[derive(Deserialize)]
struct Tool {
    mypy: Option<MypySection>,
}

#[derive(Deserialize)]
struct PyProject {
    tool: Option<Tool>,
}

#[derive(thiserror::Error, Debug)]
#[error("No [tool.mypy] section found in pyproject.toml")]
pub struct MypyNotFoundError {}

fn split_all(strs: &[String], pat: &[char]) -> Vec<String> {
    strs.iter()
        .flat_map(|s| s.split(pat))
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>()
}

fn split_comma(strs: &[String]) -> Vec<String> {
    split_all(strs, &[','])
}

fn make_error_config(disable: Vec<String>, enable: Vec<String>) -> Option<ErrorDisplayConfig> {
    let mut errors = HashMap::new();
    for error_code in disable {
        errors.insert(error_code, false);
    }
    // enable_error_code overrides disable_error_code
    for error_code in enable {
        errors.insert(error_code, true);
    }
    crate::config::mypy::code_to_kind(errors)
}

pub fn parse_pyproject_config(raw_file: &str) -> anyhow::Result<ConfigFile> {
    let mypy = toml::from_str::<PyProject>(raw_file)?
        .tool
        .and_then(|t| t.mypy)
        .ok_or(MypyNotFoundError {})?;

    let mut cfg = ConfigFile::default();

    let project_includes = [mypy.files, mypy.packages, mypy.modules]
        .into_iter()
        .flatten()
        .flatten()
        .collect::<Vec<_>>();
    if !project_includes.is_empty() {
        let project_includes = split_comma(&project_includes);
        cfg.project_includes = Globs::new(project_includes);
    }

    if let Some(regexes) = mypy.exclude_regex {
        let mut patterns = vec![];
        for reg in regexes {
            patterns.extend(regex_converter::convert(&reg)?);
        }
        if !patterns.is_empty() {
            cfg.project_excludes = Globs::new(patterns);
        }
    }

    if let Some(paths) = mypy.search_path {
        let search_path = split_all(&paths, &[',', ':'])
            .iter()
            .map_into()
            .collect::<Vec<PathBuf>>();
        cfg.search_path_from_file = search_path;
    }

    if let Some(platform) = mypy.python_platform {
        cfg.python_environment.python_platform = Some(PythonPlatform::new(&platform));
    }
    if mypy.python_version.is_some() {
        cfg.python_environment.python_version = mypy.python_version;
    }
    if mypy.python_interpreter.is_some() {
        cfg.python_interpreter = mypy.python_interpreter;
    }

    let disable_error_code = mypy
        .disable_error_code
        .map(|d| split_comma(&d))
        .unwrap_or(vec![]);
    let enable_error_code = mypy
        .enable_error_code
        .map(|d| split_comma(&d))
        .unwrap_or(vec![]);
    cfg.root.errors = make_error_config(disable_error_code, enable_error_code);

    // follow_untyped_imports may be used as a global or per-module setting. As a per-module setting, it's used to
    // indicate that the module should be ignored if it's untyped.
    // Pyrefly's use_untyped_imports is only a global setting.
    // We handle this by *only* checking the for the global config.
    cfg.use_untyped_imports = mypy
        .follow_untyped_imports
        .unwrap_or(cfg.use_untyped_imports);

    let mut replace_imports = vec![];
    let mut sub_configs = vec![];
    for module in mypy.overrides {
        if module.ignore_missing_imports || module.follow_imports.is_some_and(|v| v == "skip") {
            replace_imports.extend(
                module
                    .module
                    .iter()
                    .filter_map(|m| ModuleWildcard::new(m).ok()),
            );
        }
        let disable = module
            .disable_error_code
            .map(|d| split_comma(&d))
            .unwrap_or(vec![]);
        let enable = module
            .enable_error_code
            .map(|d| split_comma(&d))
            .unwrap_or(vec![]);
        let errors = make_error_config(disable, enable);
        if errors.is_some() {
            sub_configs.extend(module.module.iter().map(|m| {
                let matches = Glob::new(m.replace('.', "/").replace('*', "**"));
                SubConfig {
                    matches,
                    settings: ConfigBase {
                        errors: errors.clone(),
                        ..Default::default()
                    },
                }
            }));
        }
    }

    // SubConfig supports replace_imports_with_any, but mypy's ignore_missing_imports and follow_imports=skip
    // are equivalent to the root replace_imports_with_any.
    cfg.root.replace_imports_with_any = Some(replace_imports);
    cfg.sub_configs = sub_configs;

    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::error::kind::ErrorKind;

    #[test]
    fn test_missing_mypy() -> anyhow::Result<()> {
        let src = r#"[project]
name = "test_project"
description = "A project used in a test"
"#;
        let err = parse_pyproject_config(src).unwrap_err();
        assert!(err.downcast_ref::<MypyNotFoundError>().is_some());
        Ok(())
    }

    #[test]
    fn test_missing_fields() -> anyhow::Result<()> {
        // Yes, we should never expect to encounter a [tool.mypy] with no fields in it.
        // But this test ensures that all fields are optional.
        parse_pyproject_config("[tool.mypy]\n").map(|_| ())
    }

    #[test]
    fn test_allow_splittable_strings() -> anyhow::Result<()> {
        let src = r#"[tool.mypy]
files = ["a,b", "c"]
mypy_path = "a:b,c"
"#;
        let cfg = parse_pyproject_config(src)?;
        assert_eq!(
            cfg.project_includes,
            Globs::new(vec!["a".to_owned(), "b".to_owned(), "c".to_owned()])
        );
        assert_eq!(
            cfg.search_path_from_file,
            vec![PathBuf::from("a"), PathBuf::from("b"), PathBuf::from("c")]
        );
        Ok(())
    }

    #[test]
    fn test_exclude_str_or_list() -> anyhow::Result<()> {
        let src_str = r#"[tool.mypy]
exclude = "test/|foo.py"
"#;
        let cfg_str = parse_pyproject_config(src_str)?;

        let src_list = r#"[tool.mypy]
exclude = [
    "test/",
    "foo.py",
]
"#;
        let cfg_list = parse_pyproject_config(src_list)?;
        assert_eq!(cfg_str, cfg_list);
        Ok(())
    }

    #[test]
    fn test_disable_errors() -> anyhow::Result<()> {
        let src = r#"[tool.mypy]
disable_error_code = ["union-attr"]
"#;
        let mut cfg = parse_pyproject_config(src)?;
        cfg.configure();
        let errors = cfg.errors(Path::new("."));
        assert!(!errors.is_enabled(ErrorKind::MissingAttribute));
        Ok(())
    }

    #[test]
    fn test_replace_imports() -> anyhow::Result<()> {
        let src = r#"[tool.mypy]
files = ["src/a.py"]

[[tool.mypy.overrides]]
module = [
    "a.*.b",
    "some.module"
]
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = "uses.follow"
follow_imports = "skip"

[[tool.mypy.overrides]]
module = [
    "do.not.replace",
]

"#;
        let cfg = parse_pyproject_config(src)?;
        assert_eq!(
            cfg.root.replace_imports_with_any,
            Some(vec![
                ModuleWildcard::new("a.*.b").unwrap(),
                ModuleWildcard::new("some.module").unwrap(),
                ModuleWildcard::new("uses.follow").unwrap(),
            ])
        );

        Ok(())
    }

    #[test]
    fn test_use_untyped_imports() -> anyhow::Result<()> {
        let src = r#"[tool.mypy]
packages = ["cake"]
follow_untyped_imports = false

[[tool.mypy.overrides]]
module = "a"
follow_untyped_imports = true
"#;
        let cfg = parse_pyproject_config(src)?;
        assert!(!cfg.use_untyped_imports);

        let cfg = parse_pyproject_config("[tool.mypy]\n")?;
        assert!(cfg.use_untyped_imports);
        Ok(())
    }

    #[test]
    fn test_subconfig_errors() -> anyhow::Result<()> {
        let src = r#"[tool.mypy]
files = "src"

[[tool.mypy.overrides]]
module = [
    "src.*.linux",
]
disable_error_code = "union-attr"

[[tool.mypy.overrides]]
module = "src.foo"
disable_error_code = [
    "union-attr",
    "attr-defined",
]

[[tool.mypy.overrides]]
module = "another.project"
follow_imports = "silent"
"#;
        let mut cfg = parse_pyproject_config(src)?;
        cfg.configure();
        assert!(
            cfg.errors(Path::new("src"))
                .is_enabled(ErrorKind::MissingAttribute)
        );
        assert!(
            !cfg.errors(Path::new("src/linux"))
                .is_enabled(ErrorKind::MissingAttribute)
        );
        assert!(
            !cfg.errors(Path::new("src/down/the/tree/linux"))
                .is_enabled(ErrorKind::MissingAttribute)
        );
        assert!(
            !cfg.errors(Path::new("src/foo"))
                .is_enabled(ErrorKind::MissingAttribute)
        );
        Ok(())
    }

    #[test]
    fn test_no_empty_subconfigs() -> anyhow::Result<()> {
        let src = r#"[tool.mypy]
files = "src"

[[tool.mypy.overrides]]
module = [
    "src.*.linux",
]
fake_field = "foo"
"#;
        let mut cfg = parse_pyproject_config(src)?;
        cfg.configure();
        assert!(cfg.sub_configs.is_empty());
        Ok(())
    }
}
