/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::path::PathBuf;

use crate::environment::finder::walk_interpreter;

const CONFIG_FILE: &str = "pyvenv.cfg";
/// How deep within a project root should we attempt to search for a valid Python executable?
/// 3 seems like a reasonable default to be able to find something in `.venv/bin/python3`.
const SEARCH_DEPTH: usize = 3;
pub const ENV_VAR: &str = "VIRTUAL_ENV";

fn has_standard_relative_config(interp: &Path) -> bool {
    interp
        .parent()
        .and_then(|p| p.parent())
        .is_some_and(|p| p.join(CONFIG_FILE).exists())
}

fn has_backup_relative_config(interp: &Path) -> bool {
    interp
        .parent()
        .is_some_and(|p| p.join(CONFIG_FILE).exists())
}

pub fn find(project_path: &Path) -> Option<PathBuf> {
    let interpreters = walk_interpreter(project_path, SEARCH_DEPTH).collect::<Vec<PathBuf>>();

    if interpreters.is_empty() {
        return None;
    }

    if let Some(first) = interpreters
        .iter()
        .find(|i| has_standard_relative_config(i))
    {
        return Some(first.to_owned());
    }

    interpreters
        .into_iter()
        .find(|i| has_backup_relative_config(i))
}

#[cfg(test)]
mod tests {
    use pyrefly_util::test_path::TestPath;

    use super::*;

    fn interp_name(version_suffix: &str) -> String {
        let windows_suffix = if cfg!(windows) { ".exe" } else { "" };
        format!("python{version_suffix}{windows_suffix}")
    }

    #[test]
    fn test_find_no_interpreters() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::file("pyrefly.toml"),
                TestPath::dir("foo", vec![TestPath::file("bar.py")]),
            ],
        );

        assert_eq!(find(root), None);
    }

    #[test]
    fn test_find_standard_venv_layout() {
        fn test(version_suffix: &str) {
            let tempdir = tempfile::tempdir().unwrap();
            let root = tempdir.path();
            let interp_name = interp_name(version_suffix);
            TestPath::setup_test_directory(
                root,
                vec![
                    TestPath::file("pyrefly.toml"),
                    TestPath::dir("foo", vec![TestPath::file("bar.py")]),
                    TestPath::dir(
                        ".venv",
                        vec![
                            TestPath::file(CONFIG_FILE),
                            TestPath::dir("bin", vec![TestPath::file(&interp_name)]),
                            // we should never find this first
                            TestPath::file(&interp_name),
                        ],
                    ),
                ],
            );

            assert_eq!(find(root), Some(root.join(".venv/bin").join(interp_name)),);
        }

        test("");
        test("3");
        test("3.8");
        test("3.12");
    }

    #[test]
    fn test_find_nonstandard_venv_layout() {
        fn test(python_version: &str) {
            let tempdir = tempfile::tempdir().unwrap();
            let root = tempdir.path();
            let interp_name = interp_name(python_version);
            TestPath::setup_test_directory(
                root,
                vec![
                    TestPath::file("pyrefly.toml"),
                    TestPath::dir("foo", vec![TestPath::file("bar.py")]),
                    TestPath::dir(
                        ".venv",
                        vec![TestPath::file(CONFIG_FILE), TestPath::file(&interp_name)],
                    ),
                ],
            );

            assert_eq!(find(root), Some(root.join(".venv").join(interp_name)),);
        }

        test("");
        test("3");
        test("3.8");
        test("3.12");
    }

    #[test]
    fn test_find_missing_config_file() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        let interp_name = interp_name("");
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::file("pyrefly.toml"),
                TestPath::dir("foo", vec![TestPath::file("bar.py")]),
                TestPath::dir(
                    ".venv",
                    vec![
                        TestPath::file(&interp_name),
                        TestPath::dir("bin", vec![TestPath::file(&interp_name)]),
                    ],
                ),
            ],
        );

        assert_eq!(find(root), None);
    }
}
