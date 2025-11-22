/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::path::PathBuf;

/// Some kind of pattern that can be used by filesystem watchers. Some filesystem watchers
/// expect the path to be separate from the pattern part, so this enables downstream logic
/// to handle pattern components as they need.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum WatchPattern<'a> {
    /// A pattern consisting of a file or directory with no wildcards.
    File(PathBuf),
    /// A pattern consisting of a [`PathBuf`] to a root and a [`String`] containing a wildcard.
    Root(&'a Path, String),
    /// A root pattern if we need to own the path.
    OwnedRoot(PathBuf, String),
}

impl<'a> WatchPattern<'a> {
    pub fn file(path: PathBuf) -> Self {
        Self::File(path)
    }

    pub fn root(root: &'a Path, pattern: String) -> Self {
        Self::Root(root, pattern)
    }

    pub fn owned_root(root: PathBuf, pattern: String) -> Self {
        Self::OwnedRoot(root, pattern)
    }
}
