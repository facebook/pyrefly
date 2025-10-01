/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::path::PathBuf;

pub trait Includes {
    /// Given a glob pattern, return the directories that can contain files that match the pattern.
    fn roots(&self) -> Vec<PathBuf>;

    fn files(&self) -> anyhow::Result<Vec<PathBuf>>;

    fn covers(&self, path: &Path) -> bool;

    fn errors(&mut self) -> Vec<anyhow::Error>;
}
