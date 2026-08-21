/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::path::PathBuf;

const DEFAULT_ENVIRONMENT: &str = "default";

#[cfg(windows)]
fn interpreter_candidates(environment: &Path) -> [PathBuf; 1] {
    [environment.join("python.exe")]
}

#[cfg(not(windows))]
fn interpreter_candidates(environment: &Path) -> [PathBuf; 2] {
    [
        environment.join("bin").join("python3"),
        environment.join("bin").join("python"),
    ]
}

fn find_in_workspace(workspace: &Path) -> Option<PathBuf> {
    let environment = workspace
        .join(".pixi")
        .join("envs")
        .join(DEFAULT_ENVIRONMENT);
    interpreter_candidates(&environment)
        .into_iter()
        .find(|path| path.is_file())
}

/// Find the Python interpreter in an installed Pixi workspace's default environment.
///
/// Activated Pixi environments already expose their interpreter through environment variables and
/// `PATH`. This fallback covers editors and other processes started without Pixi activation, where
/// the default environment lives at `.pixi/envs/default`.
pub fn find(project_path: &Path) -> Option<PathBuf> {
    project_path
        .ancestors()
        .take_while(|path| !path.as_os_str().is_empty())
        .find_map(find_in_workspace)
}

#[cfg(test)]
mod tests {
    use pyrefly_util::test_path::TestPath;

    use super::*;

    fn interpreter_name() -> &'static str {
        if cfg!(windows) {
            "python.exe"
        } else {
            "python3"
        }
    }

    fn environment(name: &str, include_interpreter: bool) -> TestPath {
        #[cfg(windows)]
        let contents = if include_interpreter {
            vec![TestPath::file(interpreter_name())]
        } else {
            Vec::new()
        };

        #[cfg(not(windows))]
        let contents = vec![TestPath::dir(
            "bin",
            if include_interpreter {
                vec![TestPath::file(interpreter_name())]
            } else {
                Vec::new()
            },
        )];

        TestPath::dir(name, contents)
    }

    fn setup_workspace(root: &Path, environments: Vec<TestPath>) {
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::file("pixi.toml"),
                TestPath::dir(".pixi", vec![TestPath::dir("envs", environments)]),
            ],
        );
    }

    fn default_interpreter(root: &Path) -> PathBuf {
        let environment = root.join(".pixi/envs/default");
        if cfg!(windows) {
            environment.join(interpreter_name())
        } else {
            environment.join("bin").join(interpreter_name())
        }
    }

    #[test]
    fn test_find_default_environment() {
        let tempdir = tempfile::tempdir().unwrap();
        setup_workspace(tempdir.path(), vec![environment("default", true)]);

        assert_eq!(
            find(tempdir.path()),
            Some(default_interpreter(tempdir.path()))
        );
    }

    #[test]
    fn test_find_default_environment_from_descendant() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        setup_workspace(root, vec![environment("default", true)]);
        TestPath::setup_test_directory(
            root,
            vec![TestPath::dir(
                "project",
                vec![TestPath::dir("src", vec![TestPath::file("main.py")])],
            )],
        );

        assert_eq!(
            find(&root.join("project/src")),
            Some(default_interpreter(root))
        );
    }

    #[test]
    fn test_find_ignores_named_environment_without_default() {
        let tempdir = tempfile::tempdir().unwrap();
        setup_workspace(tempdir.path(), vec![environment("dev", true)]);

        assert_eq!(find(tempdir.path()), None);
    }

    #[test]
    fn test_find_requires_installed_python() {
        let tempdir = tempfile::tempdir().unwrap();
        setup_workspace(tempdir.path(), vec![environment("default", false)]);

        assert_eq!(find(tempdir.path()), None);
    }
}
