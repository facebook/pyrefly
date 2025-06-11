/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

mod ini;
mod pyproject;
mod regex_converter;

use std::collections::HashMap;

pub use ini::MypyConfig;
pub use pyproject::parse_pyproject_config;

use crate::config::error::ErrorDisplayConfig;
use crate::error::kind::ErrorKind;

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
//         map.insert(ErrorKind::Unkown, import_error);
//     }
fn code_to_kind(mut errors: HashMap<String, bool>) -> Option<ErrorDisplayConfig> {
    let mut map = HashMap::new();
    if let Some(value) = [errors.remove("union-attr"), errors.remove("attr-defined")]
        .into_iter()
        .flatten()
        .reduce(|acc, x| acc | x)
    {
        map.insert(ErrorKind::MissingAttribute, value);
    }
    if map.is_empty() {
        None
    } else {
        Some(ErrorDisplayConfig::new(map))
    }
}
