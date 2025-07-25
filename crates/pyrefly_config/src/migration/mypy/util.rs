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

/// Create an error config from disable and enable error codes
pub fn make_error_config(
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
    code_to_kind(errors)
}

/// Convert mypy error codes to pyrefly ErrorKinds. This consumes the input map.
// One or more error codes can map to the same ErrorKind, and this must be taken into consideration when adding a new error code.
// If the error code is the only one that maps to a specific ErrorKind:
//     if let Some(value) = errors.remove("error-thing") {
//       map.insert(ErrorKind::Unknown, value);
//     }
// If multiple error codes map to the same ErrorKind:
//     if let Some(import_error) = [
//         error.remove("error-thing"),
//         error.remove("other-thing"),
//     ]
//     .into_iter()
//     .flatten()  // get rid of the ones that are None
//     .reduce(|acc, x| acc | x)  // OR them together
//     {
//         map.insert(ErrorKind::Unknown, import_error);
//     }
fn code_to_kind(mut errors: HashMap<String, Severity>) -> Option<ErrorDisplayConfig> {
    let mut map = HashMap::new();
    if let Some(value) = [errors.remove("union-attr"), errors.remove("attr-defined")]
        .into_iter()
        .flatten()
        .max()
    {
        map.insert(ErrorKind::MissingAttribute, value);
    }
    if map.is_empty() {
        None
    } else {
        Some(ErrorDisplayConfig::new(map))
    }
}
