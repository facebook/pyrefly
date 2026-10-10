/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::path::PathBuf;

const CONFIG_FILE: &str = "pyvenv.cfg";
const CANDIDATE_DIRS: &[&str] = &[".venv", "venv", "env"];
pub const ENV_VAR: &str = "VIRTUAL_ENV";

/// A venv root is any directory holding a `pyvenv.cfg`.
fn is_venv_root(dir: &Path) -> bool {
    dir.join(CONFIG_FILE).exists()
}

#[cfg(windows)]
fn interpreter_path(root: &Path) -> PathBuf {
    root.join("Scripts").join("python.exe")
}

#[cfg(not(windows))]
fn interpreter_path(root: &Path) -> PathBuf {
    root.join("bin").join("python")
}

/// Every environment builder writes the platform's canonical interpreter path.
/// Version-suffixed names such as `python3` and `python3.12` are additional names
/// for that same interpreter, never the only one, so an environment lacking the
/// canonical path was modified after creation and needs a configured interpreter.
fn find_interpreter(root: &Path) -> Option<PathBuf> {
    let interpreter = interpreter_path(root);
    interpreter.is_file().then_some(interpreter)
}

/// Find the interpreter in a known virtual environment root.
pub fn find_active(root: &Path) -> Option<PathBuf> {
    if is_venv_root(root) {
        find_interpreter(root)
    } else {
        None
    }
}

/// Find an interpreter in a known venv subdirectory (`.venv`, `venv`, or `env`).
pub(crate) fn find_in_root(root: &Path) -> Option<PathBuf> {
    CANDIDATE_DIRS
        .iter()
        .map(|candidate| root.join(candidate))
        .filter(|path| is_venv_root(path))
        .find_map(|path| find_interpreter(&path))
}

#[cfg(test)]
mod tests {
    use pyrefly_util::test_path::TestPath;

    use super::*;
    use crate::environment::interpreters::Interpreters;

    fn interp_name() -> &'static str {
        if cfg!(windows) {
            "python.exe"
        } else {
            "python"
        }
    }

    fn interp_dir() -> &'static str {
        if cfg!(windows) { "Scripts" } else { "bin" }
    }

    fn interp_path(root: &Path) -> PathBuf {
        root.join(interp_dir()).join(interp_name())
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

        assert_eq!(Interpreters::find_project_interpreter(root), None);
    }

    #[test]
    fn test_find_standard_venv_layout() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::file("pyrefly.toml"),
                TestPath::dir("foo", vec![TestPath::file("bar.py")]),
                TestPath::dir(
                    ".venv",
                    vec![
                        TestPath::file(CONFIG_FILE),
                        TestPath::dir(interp_dir(), vec![TestPath::file(interp_name())]),
                        // we should never find this first
                        TestPath::file(interp_name()),
                    ],
                ),
            ],
        );

        assert_eq!(
            Interpreters::find_project_interpreter(root),
            Some(interp_path(&root.join(".venv")))
        );
    }

    #[test]
    fn test_find_env_directory() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "env",
                vec![
                    TestPath::file(CONFIG_FILE),
                    TestPath::dir(interp_dir(), vec![TestPath::file(interp_name())]),
                ],
            )],
        );

        assert_eq!(
            Interpreters::find_project_interpreter(root),
            Some(interp_path(&root.join("env")))
        );
    }

    #[cfg(unix)]
    #[test]
    fn test_find_detects_symlinked_project_venv() {
        use std::os::unix::fs::symlink;

        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        let project_root = root.join("project");
        let real_venv = root.join("real-venv");
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir(
                    "real-venv",
                    vec![
                        TestPath::file(CONFIG_FILE),
                        TestPath::dir(interp_dir(), vec![TestPath::file(interp_name())]),
                    ],
                ),
                TestPath::dir("project", vec![TestPath::file("pyrefly.toml")]),
            ],
        );
        symlink(&real_venv, project_root.join(".venv")).unwrap();

        assert_eq!(
            Interpreters::find_project_interpreter(&project_root),
            Some(interp_path(&project_root.join(".venv"))),
        );
    }

    #[test]
    fn test_find_ignores_interpreter_beside_config() {
        // No environment builder puts the interpreter next to `pyvenv.cfg`, so an
        // environment shaped like this has been modified and must be configured
        // explicitly rather than guessed at.
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::file("pyrefly.toml"),
                TestPath::dir("foo", vec![TestPath::file("bar.py")]),
                TestPath::dir(
                    ".venv",
                    vec![TestPath::file(CONFIG_FILE), TestPath::file(interp_name())],
                ),
            ],
        );

        assert_eq!(Interpreters::find_project_interpreter(root), None);
    }

    #[test]
    fn test_find_ignores_version_suffixed_interpreter() {
        // A pruned environment keeping only `python3` or `python3.12` needs a
        // configured interpreter; we do not enumerate version-suffixed names.
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        let suffix = if cfg!(windows) { ".exe" } else { "" };
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                ".venv",
                vec![
                    TestPath::file(CONFIG_FILE),
                    TestPath::dir(
                        interp_dir(),
                        vec![
                            TestPath::file(&format!("python3{suffix}")),
                            TestPath::file(&format!("python3.12{suffix}")),
                        ],
                    ),
                ],
            )],
        );

        assert_eq!(Interpreters::find_project_interpreter(root), None);
    }

    #[test]
    fn test_find_missing_config_file() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::file("pyrefly.toml"),
                TestPath::dir("foo", vec![TestPath::file("bar.py")]),
                TestPath::dir(
                    ".venv",
                    vec![TestPath::dir(
                        interp_dir(),
                        vec![TestPath::file(interp_name())],
                    )],
                ),
            ],
        );

        assert_eq!(Interpreters::find_project_interpreter(root), None);
    }

    #[test]
    fn test_find_searches_ancestor_roots() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        let project_root = root.join("project");
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir(
                    ".venv",
                    vec![
                        TestPath::file(CONFIG_FILE),
                        TestPath::dir(interp_dir(), vec![TestPath::file(interp_name())]),
                    ],
                ),
                TestPath::dir(
                    "project",
                    vec![TestPath::dir("src", vec![TestPath::file("main.py")])],
                ),
            ],
        );

        assert_eq!(
            Interpreters::find_project_interpreter(&project_root),
            Some(interp_path(&root.join(".venv"))),
        );
    }

    #[test]
    fn test_find_prefers_nearest_ancestor_root() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        let project_root = root.join("project");
        let start_path = project_root.join("src");
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir(
                    ".venv",
                    vec![
                        TestPath::file(CONFIG_FILE),
                        TestPath::dir(interp_dir(), vec![TestPath::file(interp_name())]),
                    ],
                ),
                TestPath::dir(
                    "project",
                    vec![
                        TestPath::dir(
                            ".venv",
                            vec![
                                TestPath::file(CONFIG_FILE),
                                TestPath::dir(interp_dir(), vec![TestPath::file(interp_name())]),
                            ],
                        ),
                        TestPath::dir("src", vec![TestPath::file("main.py")]),
                    ],
                ),
            ],
        );

        // Start from project/src so the search considers both ancestor environments.
        // The nearest ancestor with .venv is project/, not root/.
        assert_eq!(
            Interpreters::find_project_interpreter(&start_path),
            Some(interp_path(&project_root.join(".venv"))),
        );
    }

    #[test]
    fn test_find_active_venv_root() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::file(CONFIG_FILE),
                TestPath::dir(interp_dir(), vec![TestPath::file(interp_name())]),
            ],
        );

        assert_eq!(find_active(root), Some(interp_path(root)));
    }

    #[test]
    fn test_find_does_not_treat_ancestor_as_venv() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        let project_root = root.join("project");
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::file(CONFIG_FILE),
                TestPath::dir(interp_dir(), vec![TestPath::file(interp_name())]),
                TestPath::dir("project", vec![TestPath::file("main.py")]),
            ],
        );

        assert_eq!(Interpreters::find_project_interpreter(&project_root), None);
    }

    #[test]
    fn test_find_does_not_search_nonstandard_venv_names_at_start_path() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::file("pyrefly.toml"),
                TestPath::dir(
                    "custom-venv",
                    vec![
                        TestPath::file(CONFIG_FILE),
                        TestPath::dir(interp_dir(), vec![TestPath::file(interp_name())]),
                    ],
                ),
            ],
        );

        assert_eq!(Interpreters::find_project_interpreter(root), None);
    }

    #[test]
    fn test_find_does_not_search_nonstandard_venv_names_in_ancestors() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        let project_root = root.join("project");
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir(
                    "custom-venv",
                    vec![
                        TestPath::file(CONFIG_FILE),
                        TestPath::dir(interp_dir(), vec![TestPath::file(interp_name())]),
                    ],
                ),
                TestPath::dir("project", vec![TestPath::file("pyrefly.toml")]),
            ],
        );

        assert_eq!(Interpreters::find_project_interpreter(&project_root), None);
    }

    #[test]
    fn test_find_does_not_search_nested_subdirectories() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::file("pyrefly.toml"),
                TestPath::dir(
                    "subdir",
                    vec![TestPath::dir(
                        ".venv",
                        vec![
                            TestPath::file(CONFIG_FILE),
                            TestPath::dir(interp_dir(), vec![TestPath::file(interp_name())]),
                        ],
                    )],
                ),
            ],
        );
        assert_eq!(Interpreters::find_project_interpreter(root), None);
    }

    #[test]
    fn test_find_does_not_search_venv_beside_deep_source_tree() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::file("pyrefly.toml"),
                TestPath::dir(
                    "src",
                    vec![TestPath::dir(
                        "pkg",
                        vec![TestPath::dir("sub", vec![TestPath::file("mod.py")])],
                    )],
                ),
                TestPath::dir(
                    "my-venv",
                    vec![
                        TestPath::file(CONFIG_FILE),
                        TestPath::dir(interp_dir(), vec![TestPath::file(interp_name())]),
                    ],
                ),
            ],
        );
        assert_eq!(Interpreters::find_project_interpreter(root), None);
    }
}
