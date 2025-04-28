/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;

use itertools::Itertools;
use serde::Deserialize;
use serde_with::OneOrMany;
use serde_with::serde_as;

use crate::config::config::ConfigFile;
use crate::config::mypy::regex_converter;
use crate::sys_info::PythonPlatform;
use crate::sys_info::PythonVersion;
use crate::util::globs::Globs;

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
    #[allow(dead_code)]
    #[serde(default)]
    per_module: Vec<ModuleSection>,
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

pub fn parse_pyrproject_config(raw_file: &str) -> anyhow::Result<ConfigFile> {
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
        cfg.search_path = search_path;
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

    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_missing_mypy() -> anyhow::Result<()> {
        let src = r#"[project]
name = "test_project"
description = "A project used in a test"
"#;
        let err = parse_pyrproject_config(src).unwrap_err();
        assert!(err.downcast_ref::<MypyNotFoundError>().is_some());
        Ok(())
    }

    #[test]
    fn test_missing_fields() -> anyhow::Result<()> {
        // Yes, we should never expect to encounter a [tool.mypy] with no fields in it.
        // But this test ensures that all fields are optional.
        parse_pyrproject_config("[tool.mypy]\n").map(|_| ())
    }

    #[test]
    fn test_allow_splittable_strings() -> anyhow::Result<()> {
        let src = r#"[tool.mypy]
files = ["a,b", "c"]
mypy_path = "a:b,c"
"#;
        let cfg = parse_pyrproject_config(src)?;
        assert_eq!(
            cfg.project_includes,
            Globs::new(vec!["a".to_owned(), "b".to_owned(), "c".to_owned()])
        );
        assert_eq!(
            cfg.search_path,
            vec![PathBuf::from("a"), PathBuf::from("b"), PathBuf::from("c")]
        );
        Ok(())
    }

    #[test]
    fn test_exclude_str_or_list() -> anyhow::Result<()> {
        let src_str = r#"[tool.mypy]
exclude = "test/|foo.py"
"#;
        let cfg_str = parse_pyrproject_config(src_str)?;

        let src_list = r#"[tool.mypy]
exclude = [
    "test/",
    "foo.py",
]
"#;
        let cfg_list = parse_pyrproject_config(src_list)?;
        assert_eq!(cfg_str, cfg_list);
        Ok(())
    }
}
