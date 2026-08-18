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
fn interpreter_candidates(root: &Path) -> [PathBuf; 2] {
    [
        root.join("Scripts").join("python.exe"),
        root.join("python.exe"),
    ]
}

#[cfg(not(windows))]
fn interpreter_candidates(root: &Path) -> [PathBuf; 4] {
    [
        root.join("bin").join("python3"),
        root.join("bin").join("python"),
        root.join("python3"),
        root.join("python"),
    ]
}

fn find_interpreter(root: &Path) -> Option<PathBuf> {
    interpreter_candidates(root)
        .into_iter()
        .find(|path| path.is_file())
}

fn find_in_root(root: &Path) -> Option<PathBuf> {
    if is_venv_root(root) {
        return find_interpreter(root);
    }

    CANDIDATE_DIRS
        .iter()
        .map(|candidate| root.join(candidate))
        .filter(|path| is_venv_root(path))
        .find_map(|path| find_interpreter(&path))
}

fn search_roots(project_path: &Path) -> impl Iterator<Item = &Path> {
    project_path
        .ancestors()
        .take_while(|path| !path.as_os_str().is_empty())
}

/// Find a virtual environment interpreter starting from `project_path`.
///
/// Search order:
/// 1. If `project_path` or a known subdir (`.venv`, `venv`, `env`) contains `pyvenv.cfg`,
///    look for an interpreter there.
/// 2. Repeat step 1 in each ancestor directory.
pub fn find(project_path: &Path) -> Option<PathBuf> {
    search_roots(project_path).find_map(find_in_root)
}

#[cfg(test)]
mod tests {
    use pyrefly_util::test_path::TestPath;

    use super::*;

    fn interp_name(version_suffix: &str) -> String {
        let windows_suffix = if cfg!(windows) { ".exe" } else { "" };
        format!("python{version_suffix}{windows_suffix}")
    }

    fn interp_dir() -> &'static str {
        if cfg!(windows) { "Scripts" } else { "bin" }
    }

    fn interp_path(root: &Path, version_suffix: &str) -> PathBuf {
        root.join(interp_dir()).join(interp_name(version_suffix))
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
                            TestPath::dir(interp_dir(), vec![TestPath::file(&interp_name)]),
                            // we should never find this first
                            TestPath::file(&interp_name),
                        ],
                    ),
                ],
            );

            assert_eq!(
                find(root),
                Some(interp_path(&root.join(".venv"), version_suffix))
            );
        }

        test("");
        #[cfg(not(windows))]
        test("3");
    }

    #[test]
    fn test_find_env_directory() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        let interp_name = interp_name("");
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "env",
                vec![
                    TestPath::file(CONFIG_FILE),
                    TestPath::dir(interp_dir(), vec![TestPath::file(&interp_name)]),
                ],
            )],
        );

        assert_eq!(find(root), Some(interp_path(&root.join("env"), "")));
    }

    #[cfg(unix)]
    #[test]
    fn test_find_detects_symlinked_project_venv() {
        use std::os::unix::fs::symlink;

        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        let interp_name = interp_name("3");
        let project_root = root.join("project");
        let real_venv = root.join("real-venv");
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir(
                    "real-venv",
                    vec![
                        TestPath::file(CONFIG_FILE),
                        TestPath::dir(interp_dir(), vec![TestPath::file(&interp_name)]),
                    ],
                ),
                TestPath::dir("project", vec![TestPath::file("pyrefly.toml")]),
            ],
        );
        symlink(&real_venv, project_root.join(".venv")).unwrap();

        assert_eq!(
            find(&project_root),
            Some(interp_path(&project_root.join(".venv"), "3")),
        );
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
        #[cfg(not(windows))]
        test("3");
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
                        TestPath::dir(interp_dir(), vec![TestPath::file(&interp_name)]),
                    ],
                ),
            ],
        );

        assert_eq!(find(root), None);
    }

    #[test]
    fn test_find_searches_ancestor_roots() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        let interp_name = interp_name("");
        let project_root = root.join("project");
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir(
                    ".venv",
                    vec![
                        TestPath::file(CONFIG_FILE),
                        TestPath::dir(interp_dir(), vec![TestPath::file(&interp_name)]),
                    ],
                ),
                TestPath::dir(
                    "project",
                    vec![TestPath::dir("src", vec![TestPath::file("main.py")])],
                ),
            ],
        );

        assert_eq!(
            find(&project_root),
            Some(interp_path(&root.join(".venv"), "")),
        );
    }

    #[test]
    fn test_find_prefers_nearest_ancestor_root() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        let interp_name = interp_name("");
        let project_root = root.join("project");
        let start_path = project_root.join("src");
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir(
                    ".venv",
                    vec![
                        TestPath::file(CONFIG_FILE),
                        TestPath::dir(interp_dir(), vec![TestPath::file(&interp_name)]),
                    ],
                ),
                TestPath::dir(
                    "project",
                    vec![
                        TestPath::dir(
                            ".venv",
                            vec![
                                TestPath::file(CONFIG_FILE),
                                TestPath::dir(interp_dir(), vec![TestPath::file(&interp_name)]),
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
            find(&start_path),
            Some(interp_path(&project_root.join(".venv"), "")),
        );
    }

    #[test]
    fn test_find_ancestor_is_venv_directory() {
        // Exercises the find_in_root early-return branch where the ancestor
        // directory itself contains pyvenv.cfg (i.e., the ancestor IS a venv).
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        let interp_name = interp_name("");
        let project_root = root.join("project");
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::file(CONFIG_FILE),
                TestPath::dir(interp_dir(), vec![TestPath::file(&interp_name)]),
                TestPath::dir(
                    "project",
                    vec![TestPath::dir("src", vec![TestPath::file("main.py")])],
                ),
            ],
        );

        assert_eq!(find(&project_root), Some(interp_path(root, "")),);
    }

    #[test]
    fn test_find_does_not_search_nonstandard_venv_names_at_start_path() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        let interp_name = interp_name("");
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::file("pyrefly.toml"),
                TestPath::dir(
                    "custom-venv",
                    vec![
                        TestPath::file(CONFIG_FILE),
                        TestPath::dir(interp_dir(), vec![TestPath::file(&interp_name)]),
                    ],
                ),
            ],
        );

        assert_eq!(find(root), None);
    }

    #[test]
    fn test_find_does_not_search_nonstandard_venv_names_in_ancestors() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        let interp_name = interp_name("");
        let project_root = root.join("project");
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir(
                    "custom-venv",
                    vec![
                        TestPath::file(CONFIG_FILE),
                        TestPath::dir(interp_dir(), vec![TestPath::file(&interp_name)]),
                    ],
                ),
                TestPath::dir("project", vec![TestPath::file("pyrefly.toml")]),
            ],
        );

        assert_eq!(find(&project_root), None);
    }

    #[test]
    fn test_find_does_not_search_nested_subdirectories() {
        let interp_name = interp_name("");

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
                            TestPath::dir(interp_dir(), vec![TestPath::file(&interp_name)]),
                        ],
                    )],
                ),
            ],
        );
        assert_eq!(find(root), None);

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
                        vec![TestPath::file(CONFIG_FILE), TestPath::file(&interp_name)],
                    )],
                ),
            ],
        );
        assert_eq!(find(root), None);
    }

    #[test]
    fn test_find_does_not_search_venv_beside_deep_source_tree() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        let interp_name = interp_name("");
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
                        TestPath::dir(interp_dir(), vec![TestPath::file(&interp_name)]),
                    ],
                ),
            ],
        );
        assert_eq!(find(root), None);
    }

    #[test]
    fn test_search_roots_skips_empty_relative_ancestor() {
        assert_eq!(
            search_roots(Path::new("project/src"))
                .map(Path::to_path_buf)
                .collect::<Vec<_>>(),
            vec![PathBuf::from("project/src"), PathBuf::from("project")],
        );
    }
}
