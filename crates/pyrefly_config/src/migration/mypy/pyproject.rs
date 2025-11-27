/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use configparser::ini::Ini;
use configparser::ini::IniDefault;
use serde::Deserialize;
use serde_with::OneOrMany;
use serde_with::serde_as;

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

// A pyproject.toml Mypy config differs a bit from the INI format:
// - The [mypy] section is written as [tool.mypy]
// - The `exclude` regex can be written as an array of regexes OR a single string
// - Per-module configs go in [[tool.mypy.overrides]]
// - Any of the fields may be a string, a list of strings, or a string that needs to be split (possibly with a delimiter besides `,`.)

#[allow(dead_code)]
#[serde_as]
#[derive(Debug, Deserialize)]
struct ModuleSection {
    #[serde_as(as = "OneOrMany<_>")]
    module: Vec<String>,
    #[serde(default)]
    ignore_missing_imports: bool,
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
    python_version: Option<String>,
    #[serde(rename = "python_executable")]
    python_interpreter: Option<String>,
    #[serde_as(as = "Option<OneOrMany<_>>")]
    #[serde(default)]
    disable_error_code: Option<Vec<String>>,
    #[serde_as(as = "Option<OneOrMany<_>>")]
    #[serde(default)]
    enable_error_code: Option<Vec<String>>,
    #[serde(default)]
    ignore_missing_imports: Option<bool>,
    #[serde(default)]
    follow_imports: Option<String>,
    #[serde(default)]
    overrides: Vec<ModuleSection>,
    #[serde(default)]
    warn_redundant_casts: Option<bool>,
    #[serde(default)]
    disallow_untyped_defs: Option<bool>,
    #[serde(default)]
    disallow_incomplete_defs: Option<bool>,
    #[serde(default)]
    disallow_any_generics: Option<bool>,
    #[serde(default)]
    strict: Option<bool>,
    #[serde(default)]
    report_deprecated_as_note: Option<bool>,
    #[serde(default)]
    allow_redefinitions: Option<bool>,
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

/// Convert a pyproject.toml file to an Ini object that can be used with the existing ConfigOptionMigrater implementations
fn pyproject_to_ini(raw_file: &str) -> anyhow::Result<Ini> {
    let mypy = toml::from_str::<PyProject>(raw_file)?
        .tool
        .and_then(|t| t.mypy)
        .ok_or(MypyNotFoundError {})?;

    let mut default = IniDefault::default();
    default.multiline = true;
    let mut ini = Ini::new_from_defaults(default);

    // Add the global mypy section
    if let Some(files) = mypy.files {
        ini.set("mypy", "files", Some(join_strings(&files)));
    }
    if let Some(packages) = mypy.packages {
        ini.set("mypy", "packages", Some(join_strings(&packages)));
    }
    if let Some(modules) = mypy.modules {
        ini.set("mypy", "modules", Some(join_strings(&modules)));
    }
    if let Some(exclude_regex) = mypy.exclude_regex {
        // For exclude patterns, we need to join them with | instead of commas
        // because the regex_converter::convert function expects a regex pattern
        ini.set("mypy", "exclude", Some(exclude_regex.join("|")));
    }
    if let Some(search_path) = mypy.search_path {
        ini.set("mypy", "mypy_path", Some(join_strings(&search_path)));
    }
    if let Some(platform) = mypy.python_platform {
        ini.set("mypy", "platform", Some(platform));
    }
    if let Some(version) = mypy.python_version {
        ini.set("mypy", "python_version", Some(version));
    }
    if let Some(interpreter) = mypy.python_interpreter {
        ini.set("mypy", "python_executable", Some(interpreter));
    }
    if let Some(disable_error_code) = mypy.disable_error_code {
        ini.set(
            "mypy",
            "disable_error_code",
            Some(join_strings(&disable_error_code)),
        );
    }
    if let Some(enable_error_code) = mypy.enable_error_code {
        ini.set(
            "mypy",
            "enable_error_code",
            Some(join_strings(&enable_error_code)),
        );
    }
    if let Some(ignore_missing_imports) = mypy.ignore_missing_imports {
        ini.set(
            "mypy",
            "ignore_missing_imports",
            Some(ignore_missing_imports.to_string()),
        );
    }
    if let Some(follow_imports) = mypy.follow_imports {
        ini.set("mypy", "follow_imports", Some(follow_imports));
    }
    if let Some(warn_redundant_casts) = mypy.warn_redundant_casts {
        ini.set(
            "mypy",
            "warn_redundant_casts",
            Some(warn_redundant_casts.to_string()),
        );
    }
    if let Some(disallow_untyped_defs) = mypy.disallow_untyped_defs {
        ini.set(
            "mypy",
            "disallow_untyped_defs",
            Some(disallow_untyped_defs.to_string()),
        );
    }
    if let Some(disallow_incomplete_defs) = mypy.disallow_incomplete_defs {
        ini.set(
            "mypy",
            "disallow_incomplete_defs",
            Some(disallow_incomplete_defs.to_string()),
        );
    }
    if let Some(disallow_any_generics) = mypy.disallow_any_generics {
        ini.set(
            "mypy",
            "disallow_any_generics",
            Some(disallow_any_generics.to_string()),
        );
    }
    if let Some(strict) = mypy.strict {
        ini.set("mypy", "strict", Some(strict.to_string()));
    }
    if let Some(report_deprecated_as_note) = mypy.report_deprecated_as_note {
        ini.set(
            "mypy",
            "report_deprecated_as_note",
            Some(report_deprecated_as_note.to_string()),
        );
    }
    if let Some(allow_redefinitions) = mypy.allow_redefinitions {
        ini.set(
            "mypy",
            "allow_redefinitions",
            Some(allow_redefinitions.to_string()),
        );
    }

    // Add the per-module sections
    for module_section in mypy.overrides {
        for module_name in &module_section.module {
            let section_name = format!("mypy-{module_name}");

            if module_section.ignore_missing_imports {
                ini.set(
                    &section_name,
                    "ignore_missing_imports",
                    Some("True".to_owned()),
                );
            }
            if let Some(follow_imports) = &module_section.follow_imports {
                ini.set(
                    &section_name,
                    "follow_imports",
                    Some(follow_imports.clone()),
                );
            }
            if let Some(disable_error_code) = &module_section.disable_error_code {
                ini.set(
                    &section_name,
                    "disable_error_code",
                    Some(join_strings(disable_error_code)),
                );
            }
            if let Some(enable_error_code) = &module_section.enable_error_code {
                ini.set(
                    &section_name,
                    "enable_error_code",
                    Some(join_strings(enable_error_code)),
                );
            }
        }
    }

    Ok(ini)
}

/// Join a vector of strings with commas
fn join_strings(strings: &[String]) -> String {
    strings.join(",")
}

pub fn parse_pyproject_config(raw_file: &str) -> anyhow::Result<ConfigFile> {
    let ini = pyproject_to_ini(raw_file)?;
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
    ];

    // Iterate through all config options and apply them to the config
    for option in config_options {
        // Ignore errors for now, we can use this in the future if we want to print out error messages or use for logging purpose
        let _ = option.migrate_from_mypy(&ini, &mut cfg);
    }

    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::path::PathBuf;

    use pyrefly_util::globs::Globs;

    use super::*;
    use crate::error_kind::ErrorKind;
    use crate::error_kind::Severity;
    use crate::module_wildcard::ModuleWildcard;

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
            Globs::new(vec!["a".to_owned(), "b".to_owned(), "c".to_owned()]).unwrap()
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
        assert_eq!(
            errors.severity(ErrorKind::MissingAttribute),
            Severity::Ignore
        );
        Ok(())
    }

    #[test]
    fn test_ignore_imports() -> anyhow::Result<()> {
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
            cfg.root.ignore_missing_imports,
            Some(vec![
                ModuleWildcard::new("a.*.b").unwrap(),
                ModuleWildcard::new("some.module").unwrap(),
                ModuleWildcard::new("uses.follow").unwrap(),
            ])
        );
        assert_eq!(cfg.root.replace_imports_with_any, Some(vec![]),);

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
        assert_eq!(
            cfg.errors(Path::new("src"))
                .severity(ErrorKind::MissingAttribute),
            Severity::Error
        );
        assert_eq!(
            cfg.errors(Path::new("src/linux"))
                .severity(ErrorKind::MissingAttribute),
            Severity::Ignore
        );
        assert_eq!(
            cfg.errors(Path::new("src/down/the/tree/linux"))
                .severity(ErrorKind::MissingAttribute),
            Severity::Ignore
        );
        assert_eq!(
            cfg.errors(Path::new("src/foo"))
                .severity(ErrorKind::MissingAttribute),
            Severity::Ignore
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
