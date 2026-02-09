/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::path::MAIN_SEPARATOR_STR;
use std::path::PathBuf;

use crate::interned_path::InternedPath;

/// Some kind of pattern that can be used by filesystem watchers. Some filesystem watchers
/// expect the path to be separate from the pattern part, so this enables downstream logic
/// to handle pattern components as they need.
#[derive(PartialEq, Eq, Clone, Hash)]
pub enum WatchPattern {
    /// A pattern consisting of a file or directory with no wildcards.
    File(PathBuf),
    /// A pattern consisting of a [`PathBuf`] to a root and a [`String`] containing a wildcard.
    Root(InternedPath, String),
}

impl WatchPattern {
    pub fn file(path: PathBuf) -> Self {
        Self::File(path)
    }

    pub fn root(root: InternedPath, pattern: String) -> Self {
        Self::Root(root, pattern)
    }
}

impl Debug for WatchPattern {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::File(p) => write!(f, "{}", p.display()),
            Self::Root(r, p) => {
                write!(f, "{}{}{p}", r.display(), MAIN_SEPARATOR_STR)
            }
        }
    }
}
