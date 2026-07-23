/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

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

/// Resolve mypy's `$MYPY_CONFIG_FILE_DIR` variable in a single migrated path.
///
/// `$MYPY_CONFIG_FILE_DIR` is mypy's reference to the directory containing the
/// config file. Pyrefly resolves migrated relative paths against that same
/// directory, so the variable is only meaningful as a *leading* reference to it:
/// we strip it (and a following `/`) from the front, leaving a relative path
/// that is absolutized against the config root later. The variable matches only
/// when it is the whole entry or immediately followed by `/`, so a longer name
/// that merely shares the prefix (e.g. `$MYPY_CONFIG_FILE_DIRECTORY/foo`) is left
/// untouched. A bare `$MYPY_CONFIG_FILE_DIR` denotes the config directory itself
/// and becomes `.` (which absolutizes back to that directory), not the empty
/// string that downstream `is_empty` filters would silently drop. Apply this per
/// path element, and only to the options mypy itself runs through `expand_path` —
/// among those pyrefly migrates, `mypy_path`, `files`, and `python_executable` —
/// never to `exclude` (a regex) or non-path options.
///
/// We strip a `/`, never a Windows `\`, on purpose. Pyrefly config paths use `/`
/// as the sole separator for cross-platform portability (see `Glob` in
/// `pyrefly_util::globs`), and mypy applies no separator normalization of its
/// own — a `\` "works" only when mypy runs on Windows, where Python's path layer
/// happens to accept it, and is broken on POSIX. So a portable, checked-in mypy
/// config that uses `$MYPY_CONFIG_FILE_DIR` already writes `/`; honoring `\` here
/// would only smuggle a separator that the rest of the migrated config rejects.
///
/// Deliberately minimal: a mid-path occurrence, the rare `${...}` form, `~`, and
/// other environment variables are left as-is. Unlike mypy (which expands them
/// per run) a migrated config is checked in, so they have no portable
/// migration-time meaning.
pub fn expand_config_file_dir(path: &str) -> String {
    let rest = match path.strip_prefix("$MYPY_CONFIG_FILE_DIR") {
        Some(rest) => rest,
        None => return path.to_owned(),
    };
    // Match only when the variable is the whole entry or followed by `/`; a bare
    // prefix on a longer name (`...DIRECTORY/foo`) is not the variable.
    let relative = match rest {
        "" => "",
        _ => match rest.strip_prefix('/') {
            Some(after) => after,
            None => return path.to_owned(),
        },
    };
    // An empty remainder is the config directory itself, i.e. `.`.
    if relative.is_empty() {
        ".".to_owned()
    } else {
        relative.to_owned()
    }
}

#[derive(Default)]
pub struct MypyErrorConfigFlags {
    pub warn_return_any: bool,
    pub warn_redundant_casts: bool,
    pub disallow_untyped_defs: bool,
    pub disallow_incomplete_defs: bool,
    pub disallow_any_generics: bool,
    pub disallow_any_explicit: bool,
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
        warn_return_any,
        warn_redundant_casts,
        disallow_untyped_defs,
        disallow_incomplete_defs,
        disallow_any_generics,
        disallow_any_explicit,
        strict,
        report_deprecated_as_note,
        allow_redefinitions,
    }) = mypy_error_config_flags
    {
        // These severities take precedence over enable/disable
        if warn_return_any || strict {
            errors.insert(ErrorKind::NoAnyReturn.to_name().to_owned(), Severity::Error);
        }
        if warn_redundant_casts || strict {
            errors.insert(
                ErrorKind::RedundantCast.to_name().to_owned(),
                Severity::Warn,
            );
        }
        if disallow_untyped_defs || disallow_incomplete_defs || strict {
            errors.insert(
                ErrorKind::ImplicitAnyParameter.to_name().to_owned(),
                Severity::Error,
            );
            errors.insert(
                ErrorKind::UnannotatedReturn.to_name().to_owned(),
                Severity::Error,
            );
        }
        if disallow_any_generics || strict {
            errors.insert(ErrorKind::ImplicitAny.to_name().to_owned(), Severity::Error);
        }
        if disallow_any_explicit {
            errors.insert(ErrorKind::ExplicitAny.to_name().to_owned(), Severity::Error);
        }
        if report_deprecated_as_note && errors.contains_key(ErrorKind::Deprecated.to_name()) {
            errors.insert(ErrorKind::Deprecated.to_name().to_owned(), Severity::Info);
        }
        if allow_redefinitions {
            errors.insert(
                ErrorKind::Redefinition.to_name().to_owned(),
                Severity::Ignore,
            );
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
            "explicit-any" => add(severity, ErrorKind::ExplicitAny),
            "operator" => add(severity, ErrorKind::UnsupportedOperation),
            "typeddict-unknown-key" => add(severity, ErrorKind::BadTypedDictKey),
            "typeddict-readonly-mutated" => add(severity, ErrorKind::ReadOnly),
            "name-defined" => add(severity, ErrorKind::UnknownName),
            "used-before-def" | "possibly-undefined" => add(severity, ErrorKind::UnboundName),
            "valid-type" => add(severity, ErrorKind::InvalidAnnotation),
            "type-arg" => add(severity, ErrorKind::ImplicitAnyTypeArgument),
            "no-untyped-def" => {
                add(severity, ErrorKind::ImplicitAnyParameter);
                add(severity, ErrorKind::UnannotatedReturn);
            }
            "metaclass" => add(severity, ErrorKind::InvalidInheritance),
            "override" => add(severity, ErrorKind::BadOverride),
            "mutable-override" => add(severity, ErrorKind::BadOverrideMutableAttribute),
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
            "name-match" => add(severity, ErrorKind::NameMismatch),
            "no-any-return" => add(severity, ErrorKind::NoAnyReturn),
            _ => {}
        }
    }

    if map.is_empty() {
        None
    } else {
        Some(ErrorDisplayConfig::new(map))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_config_file_dir() {
        // Leading variable + `/`: strip it, leaving a config-relative path.
        assert_eq!(expand_config_file_dir("$MYPY_CONFIG_FILE_DIR/src"), "src");
        // No variable: passthrough untouched.
        assert_eq!(expand_config_file_dir("src/lib"), "src/lib");
        // Bare variable (with or without trailing `/`) is the config dir → `.`.
        assert_eq!(expand_config_file_dir("$MYPY_CONFIG_FILE_DIR"), ".");
        assert_eq!(expand_config_file_dir("$MYPY_CONFIG_FILE_DIR/"), ".");
        // A longer name sharing the prefix is not the variable: leave it be.
        assert_eq!(
            expand_config_file_dir("$MYPY_CONFIG_FILE_DIRECTORY/foo"),
            "$MYPY_CONFIG_FILE_DIRECTORY/foo"
        );
    }
}
