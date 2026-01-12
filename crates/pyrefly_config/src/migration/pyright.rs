/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::path::PathBuf;

use pyrefly_python::sys_info::PythonVersion;
use pyrefly_util::globs::Glob;
use pyrefly_util::globs::Globs;
use serde::Deserialize;
use serde_with::FromInto;
use serde_with::serde_as;

use crate::base::ConfigBase;
use crate::config::ConfigFile;
use crate::config::SubConfig;
use crate::error::ErrorDisplayConfig;
use crate::error_kind::ErrorKind;
use crate::error_kind::Severity;

/// Represents a pyright executionEnvironment.
/// pyright's ExecutionEnvironments allow you to specify a different Python environment for a subdirectory,
/// e.g. with a different Python version, search path, and platform.
/// pyrefly does not support any of that, so we only look for rule overrides.
#[derive(Clone, Debug, Deserialize)]
pub struct ExecEnv {
    pub root: String,
    #[serde(flatten)]
    pub errors: RuleOverrides,
}

impl ExecEnv {
    pub fn convert(self) -> anyhow::Result<SubConfig> {
        let settings = ConfigBase {
            errors: self.errors.to_config(),
            ..Default::default()
        };
        Ok(SubConfig {
            matches: Glob::new(self.root)?,
            settings,
        })
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct PyrightConfig {
    #[serde(rename = "include")]
    pub project_includes: Option<Globs>,
    #[serde(rename = "exclude")]
    pub project_excludes: Option<Globs>,
    #[serde(rename = "extraPaths")]
    pub search_path: Option<Vec<PathBuf>>,
    #[serde(rename = "stubPath")]
    pub stub_path: Option<PathBuf>,
    #[serde(rename = "pythonPlatform")]
    pub python_platform: Option<String>,
    #[serde(rename = "pythonVersion")]
    pub python_version: Option<PythonVersion>,
    #[serde(flatten)]
    pub errors: RuleOverrides,
    #[serde(default, rename = "executionEnvironments")]
    pub execution_environments: Vec<ExecEnv>,
}

use crate::migration::config_option_migrater::ConfigOptionMigrater;
use crate::migration::error_codes::ErrorCodes;
use crate::migration::ignore_missing_imports::IgnoreMissingImports;
use crate::migration::project_excludes::ProjectExcludes;
use crate::migration::project_includes::ProjectIncludes;
use crate::migration::python_interpreter::PythonInterpreter;
use crate::migration::python_platform::PythonPlatformConfig;
use crate::migration::python_version::PythonVersionConfig;
use crate::migration::search_path::SearchPath;
use crate::migration::site_package_path::SitePackagePath;
use crate::migration::sub_configs::SubConfigs;

impl PyrightConfig {
    pub fn parse(text: &str) -> anyhow::Result<Self> {
        Ok(serde_jsonrc::from_str::<Self>(text)?)
    }

    pub fn convert(self) -> ConfigFile {
        let mut cfg = ConfigFile::default();

        // Create a list of all config options
        let config_options: Vec<Box<dyn ConfigOptionMigrater>> = vec![
            Box::new(ProjectIncludes),
            Box::new(ProjectExcludes),
            Box::new(PythonInterpreter),
            Box::new(PythonVersionConfig),
            Box::new(PythonPlatformConfig),
            Box::new(SearchPath),
            Box::new(SitePackagePath),
            Box::new(IgnoreMissingImports),
            Box::new(ErrorCodes),
            Box::new(SubConfigs),
        ];

        // Iterate through all config options and apply them to the config
        for option in config_options {
            // Ignore errors for now, we can use this in the future if we want to print out error messages or use for logging purpose
            let _ = option.migrate_from_pyright(&self, &mut cfg);
        }

        // Pyright does not infer empty container types and unsolved type variables based on their first use.
        cfg.root.infer_with_first_use = Some(false);

        cfg
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DiagnosticLevel {
    None,
    Information,
    Warning,
    Error,
}

impl DiagnosticLevel {
    fn to_severity(&self) -> Severity {
        match self {
            Self::None => Severity::Ignore,
            Self::Information => Severity::Info,
            Self::Warning => Severity::Warn,
            Self::Error => Severity::Error,
        }
    }
}

impl From<DiagnosticLevel> for Severity {
    fn from(value: DiagnosticLevel) -> Self {
        value.to_severity()
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum DiagnosticLevelOrBool {
    DiagnosticLevel(DiagnosticLevel),
    Bool(bool),
}

impl DiagnosticLevelOrBool {
    fn to_severity(&self) -> Severity {
        match self {
            Self::DiagnosticLevel(dl) => dl.to_severity(),
            Self::Bool(b) => {
                if *b {
                    Severity::Error
                } else {
                    Severity::Ignore
                }
            }
        }
    }
}

impl From<DiagnosticLevelOrBool> for Severity {
    fn from(value: DiagnosticLevelOrBool) -> Self {
        value.to_severity()
    }
}

/// Type Check Rule Overrides are pyright's equivalent to the `errors` dict in pyrefly's configs.
/// That is, they control which "diangostic settings" are displayed to the user.
#[serde_as]
#[derive(Clone, Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
#[serde(default)]
pub struct RuleOverrides {
    // Import rules
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_missing_imports: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_missing_module_source: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_missing_type_stubs: Option<Severity>,

    // Type annotation rules
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_invalid_type_form: Option<Severity>,

    // Abstract/instantiation rules
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_abstract_usage: Option<Severity>,

    // Type checking rules
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_argument_type: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_assert_type_failure: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_assignment_type: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_attribute_access_issue: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    #[expect(unused)]
    pub report_call_issue: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_inconsistent_overload: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_index_issue: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_invalid_type_arguments: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_no_overload_implementation: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_operator_issue: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    #[expect(unused)]
    pub report_optional_subscript: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    #[expect(unused)]
    pub report_optional_member_access: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    #[expect(unused)]
    pub report_optional_call: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    #[expect(unused)]
    pub report_optional_iterable: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    #[expect(unused)]
    pub report_optional_context_manager: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    #[expect(unused)]
    pub report_optional_operand: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_return_type: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    #[expect(unused)]
    pub report_typed_dict_not_required_access: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_private_usage: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_deprecated: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_incompatible_method_override: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_incompatible_variable_override: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_possibly_unbound_variable: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_uninitialized_instance_variable: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    #[expect(unused)]
    pub report_invalid_string_escape_sequence: Option<Severity>,

    // Unknown/implicit any rules
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_unknown_parameter_type: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_unknown_argument_type: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    #[expect(unused)]
    pub report_unknown_lambda_type: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_unknown_variable_type: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_unknown_member_type: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_missing_parameter_type: Option<Severity>,

    // Type variable rules
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_invalid_type_var_use: Option<Severity>,

    // Redundancy/unnecessary code rules
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    #[expect(unused)]
    pub report_unnecessary_is_instance: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_unnecessary_cast: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    #[expect(unused)]
    pub report_unnecessary_comparison: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    #[expect(unused)]
    pub report_unnecessary_contains: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    #[expect(unused)]
    pub report_assert_always_true: Option<Severity>,

    // Name/variable rules
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_undefined_variable: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_unbound_variable: Option<Severity>,
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    #[expect(unused)]
    pub report_unhashable: Option<Severity>,

    // Coroutine rules
    #[serde_as(as = "Option<FromInto<DiagnosticLevelOrBool>>")]
    pub report_unused_coroutine: Option<Severity>,
}

impl RuleOverrides {
    /// Consume the RuleOverrides to turn it into an ErrorDisplayConfig map.
    pub fn to_config(self) -> Option<ErrorDisplayConfig> {
        let mut map = HashMap::new();
        let mut add = |value, kind| {
            // If multiple Pyright overrides map to the same Pyrefly error
            // use the maximum severity.
            if let Some(value) = value
                && map.get(&kind).is_none_or(|x| *x < value)
            {
                map.insert(kind, value);
            }
        };
        // For each ErrorKind, there are one or more RuleOverrides fields.
        // The ErrorDisplayConfig map has an entry for an ErrorKind if at least one of the RuleOverrides for that ErrorKind is present.
        // The value of that ErrorKind's entry is found by or'ing together the present RuleOverrides.
        add(self.report_missing_imports, ErrorKind::MissingImport);
        add(self.report_missing_module_source, ErrorKind::MissingSource);
        add(self.report_missing_type_stubs, ErrorKind::UntypedImport);
        add(self.report_invalid_type_form, ErrorKind::InvalidAnnotation);
        add(self.report_abstract_usage, ErrorKind::BadInstantiation);
        add(self.report_argument_type, ErrorKind::BadArgumentType);
        add(self.report_assert_type_failure, ErrorKind::AssertType);
        add(self.report_assignment_type, ErrorKind::BadAssignment);
        add(
            self.report_attribute_access_issue,
            ErrorKind::MissingAttribute,
        );
        add(
            self.report_inconsistent_overload,
            ErrorKind::InconsistentOverload,
        );
        add(self.report_index_issue, ErrorKind::BadIndex);
        add(
            self.report_invalid_type_arguments,
            ErrorKind::BadSpecialization,
        );
        add(
            self.report_no_overload_implementation,
            ErrorKind::InvalidOverload,
        );
        add(self.report_operator_issue, ErrorKind::UnsupportedOperation);
        add(self.report_return_type, ErrorKind::BadReturn);
        add(self.report_private_usage, ErrorKind::NoAccess);
        add(self.report_deprecated, ErrorKind::Deprecated);
        add(
            self.report_incompatible_method_override,
            ErrorKind::BadOverride,
        );
        add(
            self.report_incompatible_variable_override,
            ErrorKind::BadOverride,
        );
        add(
            self.report_possibly_unbound_variable,
            ErrorKind::UnboundName,
        );
        add(
            self.report_uninitialized_instance_variable,
            ErrorKind::ImplicitlyDefinedAttribute,
        );
        add(
            self.report_unknown_parameter_type,
            ErrorKind::UnannotatedParameter,
        );
        add(
            self.report_missing_parameter_type,
            ErrorKind::UnannotatedParameter,
        );
        add(self.report_unknown_argument_type, ErrorKind::ImplicitAny);
        add(self.report_unknown_variable_type, ErrorKind::ImplicitAny);
        add(self.report_unknown_member_type, ErrorKind::ImplicitAny);
        add(self.report_invalid_type_var_use, ErrorKind::InvalidTypeVar);
        add(self.report_unnecessary_cast, ErrorKind::RedundantCast);
        add(self.report_undefined_variable, ErrorKind::UnknownName);
        add(self.report_unbound_variable, ErrorKind::UnboundName);
        add(self.report_unused_coroutine, ErrorKind::UnusedCoroutine);

        if map.is_empty() {
            None
        } else {
            Some(ErrorDisplayConfig::new(map))
        }
    }
}

#[derive(thiserror::Error, Debug)]
#[error("No [tool.pyright] section found in pyproject.toml")]
pub struct PyrightNotFoundError {}

pub fn parse_pyproject_toml(raw_file: &str) -> anyhow::Result<ConfigFile> {
    #[derive(Deserialize)]
    struct Tool {
        pyright: Option<PyrightConfig>,
    }

    #[derive(Deserialize)]
    struct PyProject {
        tool: Option<Tool>,
    }

    toml::from_str::<PyProject>(raw_file)?
        .tool
        .and_then(|tool| tool.pyright)
        .ok_or(anyhow::anyhow!(PyrightNotFoundError {}))
        .map(PyrightConfig::convert)
}

#[cfg(test)]
mod tests {

    use pyrefly_python::sys_info::PythonPlatform;

    use super::*;
    use crate::environment::environment::PythonEnvironment;

    #[test]
    fn test_convert_pyright_config() -> anyhow::Result<()> {
        let raw_file = r#"
            {
                "include": [
                    "src/**/*.py",
                    "test/**/*.py"
                ],
                "exclude": [
                    "src/excluded/**/*.py"
                ],
                "extraPaths": [
                    "src/extra"
                ],
                "pythonPlatform": "Linux",
                "pythonVersion": "3.10"
            }
            "#;
        let pyr = serde_json::from_str::<PyrightConfig>(raw_file)?;
        let config = pyr.convert();
        assert_eq!(
            config,
            ConfigFile {
                project_includes: Globs::new(vec![
                    "src/**/*.py".to_owned(),
                    "test/**/*.py".to_owned()
                ])
                .unwrap(),
                project_excludes: Globs::new(vec!["src/excluded/**/*.py".to_owned()]).unwrap(),
                search_path_from_file: vec![PathBuf::from("src/extra")],
                python_environment: PythonEnvironment {
                    python_platform: Some(PythonPlatform::linux()),
                    python_version: Some(PythonVersion::new(3, 10, 0)),
                    site_package_path: None,
                    interpreter_site_package_path: config
                        .python_environment
                        .interpreter_site_package_path
                        .clone(),
                    interpreter_stdlib_path: config
                        .python_environment
                        .interpreter_stdlib_path
                        .clone(),
                },
                root: ConfigBase {
                    infer_with_first_use: Some(false),
                    ..Default::default()
                },
                ..Default::default()
            }
        );
        Ok(())
    }

    #[test]
    fn test_convert_pyright_config_with_missing_fields() -> anyhow::Result<()> {
        let raw_file = r#"
            {
                "include": [
                    "src/**/*.py",
                    "test/**/*.py"
                ],
                "pythonVersion": "3.11"
            }
            "#;
        let pyr = serde_json::from_str::<PyrightConfig>(raw_file)?;
        let config = pyr.convert();
        assert_eq!(
            config,
            ConfigFile {
                project_includes: Globs::new(vec![
                    "src/**/*.py".to_owned(),
                    "test/**/*.py".to_owned()
                ])
                .unwrap(),
                python_environment: PythonEnvironment {
                    python_version: Some(PythonVersion::new(3, 11, 0)),
                    python_platform: None,
                    site_package_path: None,
                    interpreter_site_package_path: config
                        .python_environment
                        .interpreter_site_package_path
                        .clone(),
                    interpreter_stdlib_path: config
                        .python_environment
                        .interpreter_stdlib_path
                        .clone(),
                },
                root: ConfigBase {
                    infer_with_first_use: Some(false),
                    ..Default::default()
                },
                ..Default::default()
            }
        );
        Ok(())
    }

    #[test]
    fn test_convert_from_pyproject() -> anyhow::Result<()> {
        // From https://microsoft.github.io/pyright/#/configuration?id=sample-pyprojecttoml-file
        let src = r#"[tool.pyright]
include = ["src"]
exclude = ["**/node_modules",
    "**/__pycache__",
    "src/experimental",
    "src/typestubs"
]
ignore = ["src/oldstuff"]
defineConstant = { DEBUG = true }
stubPath = "src/stubs"

reportMissingImports = "error"
reportMissingTypeStubs = false

pythonVersion = "3.6"
pythonPlatform = "Linux"

executionEnvironments = [
  { root = "src/web", pythonVersion = "3.5", pythonPlatform = "Windows", extraPaths = [ "src/service_libs" ], reportMissingImports = "warning" },
  { root = "src/sdk", pythonVersion = "3.0", extraPaths = [ "src/backend" ] },
  { root = "src/tests", extraPaths = ["src/tests/e2e", "src/sdk" ]},
  { root = "src" }
]
"#;
        parse_pyproject_toml(src).map(|_| ())
    }

    #[test]
    fn test_report_trailing_commas() -> anyhow::Result<()> {
        let raw_file = r#"
            {
                "include": [
                    "src/**/*.py",
                    "test/**/*.py",
                ],
                "pythonVersion": "3.11",
                "reportMissingImports": "none"
            }
            "#;
        let pyr = serde_jsonrc::from_str::<PyrightConfig>(raw_file)?;
        let config = pyr.convert();
        assert!(!config.project_includes.is_empty());
        Ok(())
    }
}
