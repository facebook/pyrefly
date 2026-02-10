/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use core::fmt;
use std::ops::Deref;
use std::path::Path;
use std::path::PathBuf;

use dupe::Dupe;
use equivalent::Equivalent;
use serde::Deserialize;
use serde::Serialize;
use serde::Serializer;
use static_interner::Intern;
use static_interner::Interner;

static PATH_INTERNER: Interner<PathBuf> = Interner::new();

#[derive(Debug, Hash, PartialEq, Eq)]
struct PathRef<'a>(&'a Path);

impl<'a> Equivalent<PathBuf> for PathRef<'a> {
    fn equivalent(&self, key: &PathBuf) -> bool {
        *self.0 == *key
    }
}

impl<'a> From<PathRef<'a>> for PathBuf {
    fn from(value: PathRef<'a>) -> Self {
        value.0.to_path_buf()
    }
}

#[derive(Clone, Dupe, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct InternedPath(Intern<PathBuf>);

impl Deref for InternedPath {
    type Target = PathBuf;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Serialize for InternedPath {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for InternedPath {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let path: &Path = Deserialize::deserialize(d)?;
        Ok(InternedPath::from_path(path))
    }
}

impl fmt::Debug for InternedPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Self as fmt::Display>::fmt(self, f)
    }
}

impl fmt::Display for InternedPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", (**self.0).display())
    }
}

impl InternedPath {
    pub fn new(path: PathBuf) -> Self {
        Self(PATH_INTERNER.intern(path))
    }

    pub fn from_path(path: &Path) -> Self {
        Self(PATH_INTERNER.intern(PathRef(path)))
    }
}
