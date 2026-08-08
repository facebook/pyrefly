/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;

/// Convert a Path to a unix string for use in pretty printing/error emission, converting all
/// (possibly Windows) separators to `/`.
pub fn path_to_unix_string(path: &Path) -> String {
    let path_str = path.to_string_lossy().into_owned();
    str_path_to_unix_string(path_str)
}

/// Convert a stringified Path to a unix string for use in pretty printing/error emission,
/// converting all (possibly Windows) separators to `/`.
pub fn str_path_to_unix_string(mut path_str: String) -> String {
    if std::path::MAIN_SEPARATOR != '/' {
        path_str = path_str.replace(std::path::MAIN_SEPARATOR, "/");
    }
    path_str
}
