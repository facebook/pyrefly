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
    /// relative to `base`.
    fn absolutize_from(&self, base: &Path) -> PathBuf {
        if let Ok(absolutized) = PathAbsolutize::absolutize_from(self, base) {
            return absolutized.into_owned();
        }

        let mut base = base.to_path_buf();
        base.push(self);
        base
    }

    /// Compute a relative path from `base` to `self`.
    /// Both paths are absolutized first so `diff_paths` always succeeds.
    fn relativize_from(&self, base: &Path) -> PathBuf {
        let abs_self = Absolutize::absolutize(self);
        let abs_base = Absolutize::absolutize(base);
        pathdiff::diff_paths(&abs_self, &abs_base).unwrap_or(abs_self)
    }
}
