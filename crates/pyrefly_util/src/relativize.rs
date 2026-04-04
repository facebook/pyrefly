/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::path::PathBuf;

pub trait Relativize {
    fn relativize_from(&self, base: &Path) -> PathBuf;
}

impl Relativize for Path {
    fn relativize_from(&self, base: &Path) -> PathBuf {
        pathdiff::diff_paths(self, base).unwrap_or_else(|| self.to_path_buf())
    }
}
