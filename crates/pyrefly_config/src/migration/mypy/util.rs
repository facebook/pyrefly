/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::path::PathBuf;

use configparser::ini::Ini;

use crate::error::ErrorDisplayConfig;
use crate::error_kind::ErrorKind;
use crate::error_kind::Severity;

/// Iterate over INI sections and apply a function to each section
///
/// # Arguments
///
/// * `ini` - The INI configuration to iterate over
/// * `section_filter` - A function that determines which sections to process
/// * `section_processor` - A function that processes each section that passes the filter
///
/// # Example
///
/// let mut result = Vec::new();
/// visit_ini_sections(
///     &mypy_cfg,
///     |section_name| section_name.starts_with("mypy-"),
///     |section_name, ini| {
///         if get_bool_or_default(ini, section_name, "ignore_missing_imports") {
///             result.push(section_name.to_owned());
///         }
///     },
/// );
pub fn visit_ini_sections<F, P>(ini: &Ini, section_filter: F, mut section_processor: P)
where
    F: Fn(&str) -> bool,
    P: FnMut(&str, &Ini),
{
    for section_name in &ini.sections() {
        if section_filter(section_name) {
            section_processor(section_name, ini);
        }
    }
}

/// Convert a comma-separated string to a vector of strings
pub fn string_to_array(value: &Option<String>) -> Vec<String> {
    match value {
        Some(value) => value
            .split(',')
            .map(|x| x.trim().to_owned())
            .filter(|s| !s.is_empty())
            .collect(),
        _ => Vec::new(),
    }
}

/// Get a boolean value from the config, with a default value if not present
pub fn get_bool_or_default(config: &Ini, section: &str, key: &str) -> bool {
    config
        .getboolcoerce(section, key)
        .ok()
        .flatten()
        .unwrap_or_default()
}

/// Convert a colon or comma-separated string to a vector of PathBufs
pub fn string_to_paths(value: &str) -> Vec<PathBuf> {
    value
        .split([',', ':'])
        .map(|x| x.trim().to_owned())
        .filter(|x| !x.is_empty())
        .map(PathBuf::from)
        .collect()
}

#[derive(Default)]
pub struct MypyErrorConfigFlags {
    pub warn_redundant_casts: bool,
    pub disallow_untyped_defs: bool,
    pub disallow_incomplete_defs: bool,
    pub disallow_any_generics: bool,
    pub strict: bool,
    pub report_deprecated_as_note: bool,
    pub allow_redefinitions: bool,
}

/// Create an error config from disable and enable error codes
pub fn make_error_config(
    mypy_error_config_flags: Option<MypyErrorConfigFlags>,
    disables: Vec<String>,
    enables: Vec<String>,
) -> Option<ErrorDisplayConfig> {
    let mut errors = HashMap::new();
    for error_code in disables {
        errors.insert(error_code, Severity::Ignore);
    }
    // enable_error_code overrides disable_error_code
    for error_code in enables {
        errors.insert(error_code, Severity::Error);
    }
    if let Some(MypyErrorConfigFlags {
        warn_redundant_casts,
        disallow_untyped_defs,
        disallow_incomplete_defs,
        disallow_any_generics,
        strict,
        report_deprecated_as_note,
        allow_redefinitions,
    }) = mypy_error_config_flags
    {
        // These severities take precedence over enable/disable
        if warn_redundant_casts || strict {
            errors.insert(ErrorKind::RedundantCast.to_string(), Severity::Warn);
        }
        if disallow_untyped_defs || disallow_incomplete_defs || strict {
            errors.insert(ErrorKind::UnannotatedParameter.to_string(), Severity::Error);
            errors.insert(ErrorKind::UnannotatedReturn.to_string(), Severity::Error);
        }
        if disallow_any_generics || strict {
            errors.insert(ErrorKind::ImplicitAny.to_string(), Severity::Error);
        }
        if report_deprecated_as_note && errors.contains_key(&ErrorKind::Deprecated.to_string()) {
            errors.insert(ErrorKind::Deprecated.to_string(), Severity::Info);
        }
        if allow_redefinitions {
            errors.insert(ErrorKind::Redefinition.to_string(), Severity::Ignore);
        }
    }
    code_to_kind(errors)
}

/// Convert mypy error codes to pyrefly ErrorKinds.
fn code_to_kind(errors: HashMap<String, Severity>) -> Option<ErrorDisplayConfig> {
    let mut map = HashMap::new();
    let mut add = |value, kind| {
        // If multiple Mypy overrides map to the same Pyrefly error
        // use the maximum severity.
        if map.get(&kind).is_none_or(|x| *x < value) {
            map.insert(kind, value);
        }
    };

    for (code, severity) in errors {
        match code.as_str() {
            "union-attr" | "attr-defined" => add(severity, ErrorKind::MissingAttribute),
            "arg-type" => add(severity, ErrorKind::BadArgumentType),
            "assignment" => add(severity, ErrorKind::BadAssignment),
            "call-arg" => add(severity, ErrorKind::BadArgumentCount),
            "call-overload" => add(severity, ErrorKind::NoMatchingOverload),
            "index" => {
                add(severity, ErrorKind::BadIndex);
                add(severity, ErrorKind::UnsupportedOperation);
            }
            "dict-item" => add(severity, ErrorKind::BadTypedDict),
            "operator" => add(severity, ErrorKind::UnsupportedOperation),
            "typeddict-unknown-key" => add(severity, ErrorKind::BadTypedDictKey),
            "typeddict-readonly-mutated" => add(severity, ErrorKind::ReadOnly),
            "name-defined" => add(severity, ErrorKind::UnknownName),
            "used-before-def" | "possibly-undefined" => add(severity, ErrorKind::UnboundName),
            "valid-type" => add(severity, ErrorKind::InvalidAnnotation),
            "type-arg" | "no-untyped-def" => add(severity, ErrorKind::ImplicitAny),
            "metaclass" => add(severity, ErrorKind::InvalidInheritance),
            "override" => add(severity, ErrorKind::BadOverride),
            "return" | "return-value" => add(severity, ErrorKind::BadReturn),
            "type-var" => add(severity, ErrorKind::BadSpecialization),
            "import" | "import-not-found" => add(severity, ErrorKind::MissingImport),
            "import-untyped" => add(severity, ErrorKind::UntypedImport),
            "abstract" => add(severity, ErrorKind::BadInstantiation),
            "no-overload-impl" => add(severity, ErrorKind::InvalidOverload),
            "unused-coroutine" | "unused-awaitable" => add(severity, ErrorKind::UnusedCoroutine),
            "top-level-await" | "await-not-async" => add(severity, ErrorKind::NotAsync),
            "assert-type" => add(severity, ErrorKind::AssertType),
            "syntax" => add(severity, ErrorKind::ParseError),
            "redundant-cast" => add(severity, ErrorKind::RedundantCast),
            "redundant-expr" | "truthy-function" | "truthy-bool" | "truthy-iterable" => {
                add(severity, ErrorKind::RedundantCondition)
            }
            "deprecated" => add(severity, ErrorKind::Deprecated),
            _ => {}
        }
    }

    if map.is_empty() {
        None
    } else {
        Some(ErrorDisplayConfig::new(map))
    }
}
