/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Registry for Python standard library paths, used to filter stdlib files
//! from file watch events and state population.

use std::path::Path;
use std::path::PathBuf;
use std::sync::LazyLock;

use starlark_map::small_set::SmallSet;

use crate::lock::RwLock;

static INTERPRETER_STDLIB_PATH_REGISTRY: LazyLock<RwLock<SmallSet<PathBuf>>> =
    LazyLock::new(|| RwLock::new(SmallSet::new()));

/// Registers stdlib paths discovered from a Python interpreter.
pub fn register_stdlib_paths(paths: Vec<PathBuf>) {
    INTERPRETER_STDLIB_PATH_REGISTRY.write().extend(paths);
}

/// Returns a reference to the stdlib path registry for reading.
pub fn get_stdlib_path_registry() -> &'static LazyLock<RwLock<SmallSet<PathBuf>>> {
    &INTERPRETER_STDLIB_PATH_REGISTRY
}

/// Determines whether a file path belongs to the Python standard library.
///
/// This function checks if the given path is located within any of the configured
/// Python interpreter's standard library directories. It canonicalizes both the input
/// path and the stdlib paths for comparison to handle symlinks correctly.
pub fn is_python_stdlib_file(path: &Path) -> bool {
    let stdlib_paths = INTERPRETER_STDLIB_PATH_REGISTRY.read().clone().into_iter();
    let path_to_check = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());

    for stdlib_path in stdlib_paths {
        let stdlib_to_check = stdlib_path
            .canonicalize()
            .unwrap_or_else(|_| stdlib_path.clone());

        if path_to_check.starts_with(&stdlib_to_check) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stdlib_paths_for_mac_and_windows_paths() {
        register_stdlib_paths(vec![PathBuf::from(
            "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12",
        )]);
        register_stdlib_paths(vec![PathBuf::from(
            "/usr/local/Cellar/python@3.12/3.12.0/lib/python3.12",
        )]);

        assert!(is_python_stdlib_file(&PathBuf::from(
            "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/os.py"
        )));
        assert!(is_python_stdlib_file(&PathBuf::from(
            "/usr/local/Cellar/python@3.12/3.12.0/lib/python3.12/sys.py"
        )));
        assert!(!is_python_stdlib_file(&PathBuf::from(
            "/Users/user/my_project/main.py"
        )));

        if cfg!(windows) {
            register_stdlib_paths(vec![PathBuf::from(r"C:\Python312\Lib")]);
            register_stdlib_paths(vec![PathBuf::from(r"C:\Program Files\Python39\Lib")]);

            assert!(is_python_stdlib_file(&PathBuf::from(
                r"C:\Python312\Lib\os.py"
            )));
            assert!(is_python_stdlib_file(&PathBuf::from(
                r"C:\Program Files\Python39\Lib\pathlib.py"
            )));
            assert!(!is_python_stdlib_file(&PathBuf::from(
                r"C:\Python312\Scripts\pip.py"
            )));
            assert!(!is_python_stdlib_file(&PathBuf::from(
                r"C:\Users\user\my_project\main.py"
            )));
        }
    }
}
