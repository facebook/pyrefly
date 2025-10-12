/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ffi::OsStr;
use std::fmt;
use std::path::Path;
use std::path::PathBuf;

use dupe::Dupe;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModuleStyle;
use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;
use serde::Serializer;
use starlark_map::small_set::SmallSet;
use static_interner::Intern;
use static_interner::Interner;

use crate::handle::Handle;

pub mod buck_check;
pub mod map_db;
pub(crate) mod query_source_db;

// We're interning `Target`s, since they'll be duplicated all over the place,
// and it would be nice to have something that implements `Copy`.
// We choose Interning over `Arc`, since we want to make sure all `Target`s
// with the same data (especially when deserialied) point to the same value.
static TARGET_INTERNER: Interner<String> = Interner::new();

#[derive(Debug, Clone, Dupe, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Target(Intern<String>);
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
    pub fn from_string(x: String) -> Self {
        Target(TARGET_INTERNER.intern(x))
    }

    pub fn to_os_str(&self) -> &OsStr {
        OsStr::new(self.0.as_str())
    }
}

/// Represents a virtual filesystem provided by a build system. A build system
/// should understand the relationship between targets and importable qualified
/// paths to the files contained in the build system.
pub trait SourceDatabase: Send + Sync + fmt::Debug {
    /// Get the Handles for modules that should be checked. Used when targets are
    /// specified with the sourcedb.
    fn modules_to_check(&self) -> Vec<Handle>;
    /// Find the given module in the sourcedb, given the module it's originating from.
    fn lookup(
        &self,
        module: &ModuleName,
        origin: Option<&Path>,
        style_filter: Option<ModuleStyle>,
    ) -> Option<ModulePath>;
    /// Get the handle for the given module path, including its Python platform and version
    /// settings.
    fn handle_from_module_path(&self, module_path: ModulePath) -> Option<Handle>;
    /// Requeries this sourcedb if the set of files provided differs from the files
    /// previously queried for. This is a blocking operation.
    /// Returns `Err` if the shellout to the build system failed
    /// The resulting bool represents whether find caches
    /// related to this sourcedb should be invalidated.
    fn requery_source_db(&self, files: SmallSet<PathBuf>) -> anyhow::Result<bool>;
    /// The source database-related configuration files a watcher should wait for
    /// changes on. Changes to one of these returned watchfiles should force
    /// a sourcedb rebuild.
    fn get_critical_files(&self) -> SmallSet<PathBuf>;
    /// Get the target for the given [`ModulePath`], if one exists.
    fn get_target(&self, origin: Option<&Path>) -> Option<Target>;
}
