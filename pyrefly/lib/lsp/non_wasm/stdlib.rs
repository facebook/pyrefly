/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;

use pyrefly_config::config::ConfigFile;
use pyrefly_util::arc_id::ArcId;

use crate::lsp::non_wasm::server::TypeErrorDisplayStatus;

#[allow(dead_code)]
pub fn is_python_stdlib_file(path: &Path) -> bool {
    let stdlib_paths =
        pyrefly_config::environment::environment::PythonEnvironment::get_interpreter_stdlib_path()
            .read()
            .clone()
            .into_iter();
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

#[allow(dead_code)]
pub fn should_show_stdlib_error(
    config: &ArcId<ConfigFile>,
    type_error_status: TypeErrorDisplayStatus,
    path: &Path,
) -> bool {
    matches!(
        type_error_status,
        TypeErrorDisplayStatus::EnabledInIdeConfig
    ) || (config.project_includes.covers(path) && !config.project_excludes.covers(path))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use pyrefly_config::environment::environment::PythonEnvironment;

    use super::*;

    #[test]
    fn test_stdlib_paths_for_mac_and_windows_paths() {
        PythonEnvironment::get_interpreter_stdlib_path()
            .write()
            .insert(PathBuf::from(
                "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12",
            ));
        PythonEnvironment::get_interpreter_stdlib_path()
            .write()
            .insert(PathBuf::from(
                "/usr/local/Cellar/python@3.12/3.12.0/lib/python3.12",
            ));

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
            PythonEnvironment::get_interpreter_stdlib_path()
                .write()
                .insert(PathBuf::from(r"C:\Python312\Lib"));
            PythonEnvironment::get_interpreter_stdlib_path()
                .write()
                .insert(PathBuf::from(r"C:\Program Files\Python39\Lib"));

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
