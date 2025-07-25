/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::path::PathBuf;
use std::sync::LazyLock;

use regex::Regex;
use walkdir::WalkDir;

/// Regex for matching a Python executable, copied from MS vscode-python. The correct pattern will be selected
/// for the user's current platform.
///
/// [posix]: https://github.com/microsoft/vscode-python/blob/b0b8aff14bfa25b5072c05e1b75581430ce532c2/src/client/pythonEnvironments/common/posixUtils.ts#L24
/// [windows]: https://github.com/microsoft/vscode-python/blob/b0b8aff14bfa25b5072c05e1b75581430ce532c2/src/client/pythonEnvironments/common/windowsUtils.ts#L32
static PYTHON_INTERPRETER_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    let pattern = if cfg!(windows) {
        r"^python(\d+(.\d+)?)?\.exe$"
    } else {
        r"^python(\d+(\.\d+)?)?$"
    };
    Regex::new(pattern).unwrap()
});

/// Walk the given start [`Path`] up to the given depth, searching for
/// [`PYTHON_INTERPRETER_REGEX`]. If an error is encountered, we return
/// an empty [`Vec`] for the given operation.
///
/// The strategy we use for this is similar to
/// [vscode-python](https://github.com/microsoft/vscode-python/blob/b44b4d442fce5c0ab1547c777b846c1ade889832/src/client/pythonEnvironments/common/environmentManagers/simplevirtualenvs.ts#L14),
/// which given an interpreter path, looks for a `pyenv.cfg` file in the
/// same or parent directory of the interpreter. It is preferred
/// to take the more deeply nested interpreter according to vscode-python.
///
/// Example: if we have `.venv/pyenv.cfg`, valid an interpreter is expected
/// to be found at either `.venv/bin/python(\d(\.\d)?)?` or `.venv/python(\d(\.\d)?)?`.
///
/// Note: we do not follow links. For `venv`s, the symlink path contains important
/// information the Python interpreter needs to proplrly execute.
pub fn walk_interpreter(start: &Path, depth: usize) -> impl Iterator<Item = PathBuf> {
    let walker = WalkDir::new(start)
        .min_depth(1)
        .max_depth(depth)
        .follow_links(false);

    fn filter_map(entry: Result<walkdir::DirEntry, walkdir::Error>) -> Option<PathBuf> {
        let entry = entry.ok()?;

        // don't handle symlinks differently here, since the symlinked path is important for
        // using `pyenv.cfg`.
        if entry.file_type().is_dir()
            || !PYTHON_INTERPRETER_REGEX.is_match(entry.file_name().to_string_lossy().as_ref())
        {
            return None;
        }
        Some(entry.path().to_path_buf())
    }

    walker.into_iter().filter_map(filter_map)
}

#[cfg(test)]
mod tests {
    use itertools::Itertools as _;
    use pyrefly_util::test_path::TestPath;

    use super::*;

    fn interp_name(version_suffix: &str) -> String {
        let windows_suffix = if cfg!(windows) { ".exe" } else { "" };
        format!("python{version_suffix}{windows_suffix}")
    }

    fn create_fs(root: &Path, interp_name: &str) {
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::file("pyrefly.toml"),
                TestPath::dir("foo", vec![TestPath::file("bar.py")]),
                TestPath::dir(
                    ".venv",
                    vec![
                        TestPath::file(interp_name),
                        TestPath::dir("bin", vec![TestPath::file(interp_name)]),
                    ],
                ),
            ],
        );
    }

    #[test]
    fn test_find_all_interpreters() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        let interp_name = interp_name("");
        create_fs(root, &interp_name);

        assert_eq!(
            walk_interpreter(root, 3).sorted().collect::<Vec<PathBuf>>(),
            vec![
                root.join(".venv/bin").join(&interp_name),
                root.join(".venv").join(&interp_name)
            ]
        );
    }

    #[test]
    fn test_find_max_depth() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        let interp_name = interp_name("");
        create_fs(root, &interp_name);

        assert_eq!(
            walk_interpreter(root, 0).collect::<Vec<PathBuf>>(),
            Vec::<PathBuf>::new()
        );
        assert_eq!(
            walk_interpreter(root, 1).collect::<Vec<PathBuf>>(),
            Vec::<PathBuf>::new()
        );
        assert_eq!(
            walk_interpreter(root, 2).collect::<Vec<PathBuf>>(),
            vec![root.join(".venv").join(&interp_name)]
        );
        assert_eq!(
            walk_interpreter(root, 3).sorted().collect::<Vec<PathBuf>>(),
            vec![
                root.join(".venv/bin").join(&interp_name),
                root.join(".venv").join(&interp_name)
            ]
        );
    }
}
