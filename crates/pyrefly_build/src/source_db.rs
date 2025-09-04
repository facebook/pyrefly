/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;

use dupe::Dupe;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;
use serde::Serializer;
use static_interner::Intern;
use static_interner::Interner;

use crate::handle::Handle;

// We're interning `Target`s, since they'll be duplicated all over the place,
// and it would be nice to have something that implements `Copy`.
// We choose Interning over `Arc`, since we want to make sure all `Target`s
// with the same data (especially when deserialied) point to the same value.
static TARGET_INTERNER: Interner<String> = Interner::new();

#[derive(Debug, Clone, Dupe, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Target(Intern<String>);
impl Serialize for Target {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for Target {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s: &str = Deserialize::deserialize(deserializer)?;
        Ok(Self::from_string(s.to_owned()))
    }
}

impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Target {
    #[allow(unused)]
    pub fn from_string(x: String) -> Self {
        Target(TARGET_INTERNER.intern(x))
    }
}

/// Represents a virtual filesystem provided by a build system. A build system
/// should understand the relationship between targets and importable qualified
/// paths to the files contained in the build system.
pub trait SourceDatabase: Send + Sync + fmt::Debug {
    fn modules_to_check(&self) -> Vec<Handle>;
    fn lookup(&self, module: &ModuleName, origin: Option<&Handle>) -> Option<ModulePath>;
}
