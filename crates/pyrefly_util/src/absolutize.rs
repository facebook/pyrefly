/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::path::PathBuf;

use path_absolutize::Absolutize as PathAbsolutize;

pub trait Absolutize {
    fn absolutize(&self) -> PathBuf;
    fn absolutize_from(&self, base: &Path) -> PathBuf;
    fn relativize_from(&self, base: &Path) -> PathBuf;
}

impl Absolutize for Path {
    /// Absolutize the path, removing `..` and `.` components,
    /// relative to cwd.
    fn absolutize(&self) -> PathBuf {
        if let Ok(absolutized) = PathAbsolutize::absolutize(self) {
            return absolutized.into_owned();
        }

        let Ok(mut cwd) = std::env::current_dir() else {
            return self.to_path_buf();
        };
        cwd.push(self);
        cwd
    }

    /// Absolutize the path, removing `..` and `.` components,
    /// relative to `base`. A relative base is resolved against cwd.
    fn absolutize_from(&self, base: &Path) -> PathBuf {
        // The dependency's `absolutize_from` can panic with a relative base
        // when normalization removes every component (for example, `.` from `""`).
        Absolutize::absolutize(base.join(self).as_path())
    }

    /// Compute a relative path from `base` to `self`.
    /// Both paths are absolutized first so `diff_paths` always succeeds.
    fn relativize_from(&self, base: &Path) -> PathBuf {
        let abs_self = Absolutize::absolutize(self);
        let abs_base = Absolutize::absolutize(base);
        pathdiff::diff_paths(&abs_self, &abs_base).unwrap_or(abs_self)
    }
}

#[cfg(test)]
mod tests {
    use std::env::current_dir;
    use std::path::Path;

    use super::Absolutize;

    #[test]
    fn test_absolutize_from_relative_base() {
        let cwd = current_dir().unwrap();
        for (path, base, expected) in [
            ("", "", cwd.clone()),
            (".", "", cwd.clone()),
            ("..", "project", cwd.clone()),
            ("src/..", "", cwd.clone()),
            ("src/../main.py", "project", cwd.join("project/main.py")),
            ("main.py", "project/..", cwd.join("main.py")),
        ] {
            assert_eq!(
                Path::new(path).absolutize_from(Path::new(base)),
                expected,
                "path {path:?}, base {base:?}",
            );
        }
    }

    #[test]
    fn test_absolutize_from_absolute_base() {
        let cwd = current_dir().unwrap();
        let base = cwd.join("project");
        assert_eq!(Path::new("..").absolutize_from(&base), cwd);
        assert_eq!(
            Path::new("src/../main.py").absolutize_from(&base),
            base.join("main.py"),
        );
        assert_eq!(cwd.absolutize_from(&base), cwd);
        let root = cwd.ancestors().last().unwrap();
        assert_eq!(Path::new("../../..").absolutize_from(root), root);
    }
}
