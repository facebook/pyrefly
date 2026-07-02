/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::path::PathBuf;
use std::str::FromStr;

use anyhow::Context as _;
use clap::Parser;
use pyrefly_config::base::Preset;
use pyrefly_config::error::ErrorDisplayConfig;
use pyrefly_config::error_kind::Severity;
use pyrefly_python::sys_info::PythonPlatform;
use pyrefly_python::sys_info::PythonVersion;
use pyrefly_util::fs_anyhow;
use pyrefly_util::thread_pool::ThreadCount;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;

use crate::commands::util::CommandExitStatus;

/// Arguments for Bazel-powered type checking.
#[derive(Debug, Clone, Parser)]
pub struct BazelCheckArgs {
    /// Path to Bazel input JSON.
    input_path: PathBuf,

    /// Path to output JSON file containing Pyrefly type check results.
    #[arg(long = "output", short = 'o', value_name = "FILE")]
    output_path: PathBuf,

    /// Minimum severity level for errors to be displayed.
    /// Errors below this severity will not be shown. Defaults to "error".
    #[arg(long, value_enum, default_value_t = Severity::Error)]
    min_severity: Severity,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct BazelCheckInput {
    target: BazelTargetInfo,
    check_roots: BazelCheckRoots,
    search_path: BazelSearchPath,
    #[serde(default)]
    path_overlays: Vec<BazelPathOverlay>,
    config: BazelConfig,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
struct BazelTargetInfo {
    label: String,
    workspace_name: String,
    package: String,
    name: String,
    rule_kind: String,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct BazelCheckRoots {
    #[serde(default)]
    sources: Vec<String>,
    #[serde(default)]
    stubs: Vec<String>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct BazelSearchPath {
    main_file: Option<String>,
    main_file_directory: Option<String>,
    #[serde(default)]
    explicit_imports: Vec<String>,
    workspace_name: String,
    #[serde(default)]
    python_import_all_repositories: bool,
    #[serde(default)]
    repository_roots: Vec<String>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct BazelPathOverlay {
    short_path: String,
    path: String,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct BazelConfig {
    python_version: Option<String>,
    system_platform: Option<String>,
    preset: Option<String>,
    error_severities: Option<Value>,
}

#[derive(Debug, PartialEq, Eq)]
struct ParsedBazelConfig {
    python_version: PythonVersion,
    system_platform: PythonPlatform,
    preset: Option<Preset>,
    errors: Option<ErrorDisplayConfig>,
}

impl BazelCheckInput {
    fn from_json_bytes(path: &Path, data: &[u8]) -> anyhow::Result<Self> {
        serde_json::from_slice(data)
            .with_context(|| format!("failed to parse Bazel input JSON `{}`", path.display()))
    }
}

impl BazelConfig {
    fn parse(self) -> anyhow::Result<ParsedBazelConfig> {
        let python_version = self
            .python_version
            .as_deref()
            .map(PythonVersion::from_str)
            .transpose()?
            .unwrap_or_default();
        let system_platform = self
            .system_platform
            .as_deref()
            .map(PythonPlatform::new)
            .unwrap_or_default();
        let preset = self
            .preset
            .map(|preset| {
                serde_json::from_value::<Preset>(Value::String(preset))
                    .with_context(|| "invalid Bazel input config `preset`")
            })
            .transpose()?;
        let errors = self
            .error_severities
            .map(|errors| {
                serde_json::from_value::<ErrorDisplayConfig>(errors)
                    .with_context(|| "invalid Bazel input config `error_severities`")
            })
            .transpose()?;

        Ok(ParsedBazelConfig {
            python_version,
            system_platform,
            preset,
            errors,
        })
    }
}

#[derive(Debug, Serialize)]
struct BazelDiagnostics {
    diagnostics: Vec<Value>,
}

fn read_input_file(path: &Path) -> anyhow::Result<BazelCheckInput> {
    let data = fs_anyhow::read(path)
        .with_context(|| format!("failed to read Bazel input JSON `{}`", path.display()))?;
    BazelCheckInput::from_json_bytes(path, &data)
}

fn write_output(path: &Path, diagnostics: &BazelDiagnostics) -> anyhow::Result<()> {
    let output_bytes = serde_json::to_vec(diagnostics)
        .with_context(|| "failed to serialize Bazel diagnostic JSON value to bytes")?;
    fs_anyhow::write(path, &output_bytes)
}

impl BazelCheckArgs {
    pub fn run(self, _thread_count: ThreadCount) -> anyhow::Result<CommandExitStatus> {
        match self.run_inner() {
            Ok(status) => Ok(status),
            Err(error) => {
                eprintln!("{error:?}");
                Ok(CommandExitStatus::InfraError)
            }
        }
    }

    fn run_inner(self) -> anyhow::Result<CommandExitStatus> {
        let Self {
            input_path,
            output_path,
            min_severity: _,
        } = self;
        let _config = read_input_file(&input_path)?.config.parse()?;
        write_output(
            &output_path,
            &BazelDiagnostics {
                diagnostics: Vec::new(),
            },
        )?;
        Ok(CommandExitStatus::Success)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_input_json() -> &'static str {
        r#"{
  "target": {
    "label": "//pkg:app",
    "workspace_name": "_main",
    "package": "pkg",
    "name": "app",
    "rule_kind": "py_binary"
  },
  "check_roots": {
    "sources": ["pkg/app.py"],
    "stubs": ["pkg/app.pyi"]
  },
  "search_path": {
    "main_file": "pkg/app.py",
    "main_file_directory": "pkg",
    "explicit_imports": [
      "_main/pkg",
      "rules_python++pip+pypi_311/site-packages"
    ],
    "workspace_name": "_main",
    "python_import_all_repositories": false,
    "repository_roots": ["_main"]
  },
  "path_overlays": [
    {
      "short_path": "pkg/generated.py",
      "path": "bazel-out/darwin-fastbuild/bin/pkg/generated.py"
    }
  ],
  "config": {
    "python_version": "3.12",
    "system_platform": "linux",
    "preset": null,
    "error_severities": null
  }
}"#
    }

    fn parse(json: &str) -> anyhow::Result<BazelCheckInput> {
        BazelCheckInput::from_json_bytes(Path::new("input.json"), json.as_bytes())
    }

    fn parse_config_object(config: &str) -> anyhow::Result<ParsedBazelConfig> {
        let json = format!(
            r#"{{
  "target": {{"label": "//pkg:lib", "workspace_name": "_main", "package": "pkg", "name": "lib", "rule_kind": "py_library"}},
  "check_roots": {{"sources": ["pkg/lib.py"]}},
  "search_path": {{"workspace_name": "_main"}},
  "config": {config}
}}"#
        );
        parse(&json)?.config.parse()
    }

    #[test]
    fn target_metadata_fields_deserialize() {
        let input = parse(full_input_json()).expect("full input JSON should parse");
        assert_eq!(input.target.label, "//pkg:app");
        assert_eq!(input.target.workspace_name, "_main");
        assert_eq!(input.target.package, "pkg");
        assert_eq!(input.target.name, "app");
        assert_eq!(input.target.rule_kind, "py_binary");
    }

    #[test]
    fn check_root_fields_deserialize() {
        let input = parse(full_input_json()).expect("full input JSON should parse");
        assert_eq!(input.check_roots.sources, vec!["pkg/app.py".to_owned()]);
        assert_eq!(input.check_roots.stubs, vec!["pkg/app.pyi".to_owned()]);
    }

    #[test]
    fn search_path_fields_deserialize() {
        let input = parse(full_input_json()).expect("full input JSON should parse");
        assert_eq!(input.search_path.main_file.as_deref(), Some("pkg/app.py"));
        assert_eq!(
            input.search_path.main_file_directory.as_deref(),
            Some("pkg")
        );
        assert_eq!(
            input
                .search_path
                .explicit_imports
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            vec!["_main/pkg", "rules_python++pip+pypi_311/site-packages"]
        );
        assert_eq!(input.search_path.workspace_name, "_main");
        assert!(!input.search_path.python_import_all_repositories);
        assert_eq!(input.search_path.repository_roots, vec!["_main".to_owned()]);
    }

    #[test]
    fn path_overlay_fields_deserialize() {
        let input = parse(full_input_json()).expect("full input JSON should parse");
        assert_eq!(input.path_overlays.len(), 1);
        assert_eq!(input.path_overlays[0].short_path, "pkg/generated.py");
        assert_eq!(
            input.path_overlays[0].path,
            "bazel-out/darwin-fastbuild/bin/pkg/generated.py"
        );
    }

    #[test]
    fn empty_package_and_main_file_directory_are_preserved() {
        let input = parse(
            r#"{
  "target": {"label": "//:app", "workspace_name": "_main", "package": "", "name": "app", "rule_kind": "py_binary"},
  "check_roots": {},
  "search_path": {"main_file_directory": "", "workspace_name": "_main"},
  "config": {}
}"#,
        )
        .expect("input with empty string fields should parse");
        assert_eq!(input.target.package, "");
        assert_eq!(input.search_path.main_file_directory.as_deref(), Some(""));
    }

    #[test]
    fn omitted_repeated_fields_default_to_empty() {
        let input = parse(
            r#"{
  "target": {"label": "//pkg:lib", "workspace_name": "_main", "package": "pkg", "name": "lib", "rule_kind": "py_library"},
  "check_roots": {},
  "search_path": {"workspace_name": "_main"},
  "config": {}
}"#,
        )
        .expect("input with omitted repeated fields should parse");
        assert!(input.check_roots.sources.is_empty());
        assert!(input.check_roots.stubs.is_empty());
        assert!(input.search_path.explicit_imports.is_empty());
        assert!(input.search_path.repository_roots.is_empty());
        assert!(input.path_overlays.is_empty());
    }

    #[test]
    fn omitted_optional_search_path_fields_default_to_none_or_false() {
        let input = parse(
            r#"{
  "target": {"label": "//pkg:lib", "workspace_name": "_main", "package": "pkg", "name": "lib", "rule_kind": "py_library"},
  "check_roots": {},
  "search_path": {"workspace_name": "_main"},
  "config": {}
}"#,
        )
        .expect("input with omitted optional search path fields should parse");
        assert!(input.search_path.main_file.is_none());
        assert!(input.search_path.main_file_directory.is_none());
        assert!(!input.search_path.python_import_all_repositories);
    }

    #[test]
    fn null_config_values_use_pyrefly_defaults() {
        let config = parse_config_object(
            r#"{"python_version": null, "system_platform": null, "preset": null, "error_severities": null}"#,
        );
        let config = config.expect("explicit null config values should parse");
        assert_eq!(config.python_version, PythonVersion::default());
        assert_eq!(config.system_platform, PythonPlatform::default());
        assert!(config.preset.is_none());
        assert!(config.errors.is_none());
    }

    #[test]
    fn python_version_config_parses_to_pyrefly_type() {
        let config =
            parse_config_object(r#"{"python_version": "python3.10", "system_platform": null}"#)
                .expect("valid Python version config should parse");
        assert_eq!(config.python_version, PythonVersion::new(3, 10, 0));
    }

    #[test]
    fn system_platform_config_parses_to_pyrefly_type() {
        let config = parse_config_object(r#"{"python_version": null, "system_platform": "Linux"}"#)
            .expect("valid system platform config should parse");
        assert_eq!(config.system_platform, PythonPlatform::linux());
    }

    #[test]
    fn preset_config_parses_to_pyrefly_type() {
        let config = parse_config_object(
            r#"{"python_version": null, "system_platform": null, "preset": "strict"}"#,
        )
        .expect("valid preset config should parse");
        assert_eq!(config.preset, Some(Preset::Strict));
    }

    #[test]
    fn error_severities_config_parses_to_pyrefly_type() {
        let config = parse_config_object(
            r#"{"python_version": null, "system_platform": null, "error_severities": {"missing-attribute": "warn"}}"#,
        )
        .expect("valid error severities config should parse");
        assert!(config.errors.is_some());
    }

    #[test]
    fn invalid_python_version_is_rejected() {
        let error = parse_config_object(r#"{"python_version": "python"}"#)
            .expect_err("invalid Python version should fail");
        assert!(
            error.to_string().contains("Invalid version string"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn invalid_preset_is_rejected() {
        let error =
            parse_config_object(r#"{"preset": "nope"}"#).expect_err("invalid preset should fail");
        assert!(
            format!("{error:#}").contains("invalid Bazel input config `preset`"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn invalid_error_severity_name_is_rejected() {
        let error = parse_config_object(r#"{"error_severities": {"not-a-real-error": "warn"}}"#)
            .expect_err("invalid error severity config should fail");
        assert!(
            format!("{error:#}").contains("invalid Bazel input config `error_severities`"),
            "unexpected error: {error:#}"
        );
    }
}
