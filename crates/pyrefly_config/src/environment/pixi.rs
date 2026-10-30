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

/// Find the Python interpreter in an installed Pixi workspace's default environment.
///
/// Activated Pixi environments already expose their interpreter through environment variables and
/// `PATH`. This fallback only inspects `.pixi/envs/default`; it does not validate the manifest or
/// resolve detached-environment locations.
pub(crate) fn find_in_workspace(workspace: &Path) -> Option<PathBuf> {
    let environment = workspace
        .join(".pixi")
        .join("envs")
        .join(DEFAULT_ENVIRONMENT);
    interpreter_candidates(&environment)
        .into_iter()
        .find(|path| path.is_file())
}

#[cfg(test)]
mod tests {
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::symlink;

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
            find_in_workspace(tempdir.path()),
            Some(default_interpreter(tempdir.path()))
        );
    }

    #[test]
    fn test_find_ignores_named_environment_without_default() {
        let tempdir = tempfile::tempdir().unwrap();
        setup_workspace(tempdir.path(), vec![environment("dev", true)]);

        assert_eq!(find_in_workspace(tempdir.path()), None);
    }

    #[test]
    fn test_find_requires_installed_python() {
        let tempdir = tempfile::tempdir().unwrap();
        setup_workspace(tempdir.path(), vec![environment("default", false)]);

        assert_eq!(find_in_workspace(tempdir.path()), None);
    }

    #[test]
    fn test_find_does_not_require_manifest() {
        let tempdir = tempfile::tempdir().unwrap();
        let interpreter = default_interpreter(tempdir.path());
        fs::create_dir_all(interpreter.parent().unwrap()).unwrap();
        fs::write(&interpreter, "").unwrap();

        assert_eq!(find_in_workspace(tempdir.path()), Some(interpreter));
    }

    #[cfg(not(windows))]
    #[test]
    fn test_find_python_fallback() {
        let tempdir = tempfile::tempdir().unwrap();
        let interpreter = tempdir.path().join(".pixi/envs/default/bin/python");
        fs::create_dir_all(interpreter.parent().unwrap()).unwrap();
        fs::write(&interpreter, "").unwrap();

        assert_eq!(find_in_workspace(tempdir.path()), Some(interpreter));
    }

    #[cfg(unix)]
    #[test]
    fn test_find_symlinked_environment() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        let environment = root.join("real-environment");
        fs::create_dir_all(environment.join("bin")).unwrap();
        fs::write(environment.join("bin/python3"), "").unwrap();
        fs::create_dir_all(root.join(".pixi/envs")).unwrap();
        symlink(&environment, root.join(".pixi/envs/default")).unwrap();

        assert_eq!(find_in_workspace(root), Some(default_interpreter(root)));
    }

    #[cfg(windows)]
    #[test]
    fn test_find_ignores_venv_layout() {
        let tempdir = tempfile::tempdir().unwrap();
        let interpreter = tempdir.path().join(".pixi/envs/default/Scripts/python.exe");
        fs::create_dir_all(interpreter.parent().unwrap()).unwrap();
        fs::write(interpreter, "").unwrap();

        assert_eq!(find_in_workspace(tempdir.path()), None);
    }
}
