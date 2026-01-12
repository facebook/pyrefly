/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#![warn(clippy::all)]
#![allow(clippy::enum_variant_names)]
#![allow(clippy::manual_flatten)]
#![allow(clippy::match_like_matches_macro)]
#![allow(clippy::module_inception)]
#![allow(clippy::needless_lifetimes)]
#![allow(clippy::new_without_default)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::single_match)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::wrong_self_convention)]
#![deny(clippy::cloned_instead_of_copied)]
#![deny(clippy::derive_partial_eq_without_eq)]
#![deny(clippy::inefficient_to_string)]
#![deny(clippy::str_to_string)]
#![deny(clippy::trivially_copy_pass_by_ref)]
#![feature(const_type_name)]
#![feature(if_let_guard)]

use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::LazyLock;

use dupe::Dupe as _;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::arc_id::WeakArcId;
use pyrefly_util::lock::Mutex;
use serde::Deserialize;
use serde::Serialize;

pub mod handle;
pub mod source_db;
pub use source_db::SourceDatabase;
use starlark_map::small_map::SmallMap;
mod query;
#[cfg(not(target_arch = "wasm32"))]
use which::which;

use crate::query::SourceDbQuerier;
use crate::query::buck::BxlArgs;
use crate::query::buck::BxlQuerier;
use crate::query::custom::CustomQuerier;
use crate::query::custom::CustomQueryArgs;
use crate::source_db::query_source_db::QuerySourceDatabase;

/// A cache of previously loaded build systems, keyed on their project root
/// and config.
static BUILD_SYSTEM_CACHE: LazyLock<
    Mutex<
        SmallMap<
            (PathBuf, BuildSystemArgs),
            WeakArcId<Box<dyn source_db::SourceDatabase + 'static>>,
        >,
    >,
> = LazyLock::new(|| Mutex::new(SmallMap::new()));

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "kebab-case", tag = "type")]
pub enum BuildSystemArgs {
    Buck(BxlArgs),
    Custom(CustomQueryArgs),
}

impl BuildSystemArgs {
    fn is_build_system_available(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let cmd = match self {
                Self::Buck(_) => "buck2",
                Self::Custom(args) => args.command.first(),
            };
            which(cmd).is_ok()
        }
        #[cfg(target_arch = "wasm32")]
        false
    }

    fn get_repo_root(&self, cwd: &Path) -> anyhow::Result<PathBuf> {
        match self {
            Self::Buck(args) => args.get_repo_root(cwd),
            Self::Custom(args) => args.get_repo_root(cwd),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub struct BuildSystem {
    #[serde(flatten)]
    args: BuildSystemArgs,
    #[serde(default)]
    ignore_if_build_system_missing: bool,
    // TODO(connernilsen): remove once internal stubs are deprecated
    /// Are there any sources we should use before looking at the build system (like stubs)?
    #[serde(default)]
    pub search_path_prefix: Vec<PathBuf>,
}

impl BuildSystem {
    pub fn new(
        isolation_dir: Option<String>,
        extras: Option<Vec<String>>,
        ignore_if_build_system_missing: bool,
        search_path_prefix: Vec<PathBuf>,
    ) -> Self {
        let args = BuildSystemArgs::Buck(BxlArgs::new(isolation_dir, extras));
        Self {
            args,
            ignore_if_build_system_missing,
            search_path_prefix,
        }
    }

    pub fn get_source_db(
        &mut self,
        config_root: PathBuf,
    ) -> Option<anyhow::Result<ArcId<Box<dyn source_db::SourceDatabase + 'static>>>> {
        let build_system_available = self.args.is_build_system_available();
        if !build_system_available {
            if self.ignore_if_build_system_missing {
                return None;
            } else {
                return Some(Err(anyhow::anyhow!(
                    "Build system configured, but could not be found on PATH."
                )));
            }
        }

        let repo_root = match self.args.get_repo_root(&config_root) {
            Err(e) => return Some(Err(e)),
            Ok(path) => path,
        };
        let mut cache = BUILD_SYSTEM_CACHE.lock();
        let key = (repo_root.clone(), self.args.clone());
        if let Some(maybe_result) = cache.get(&key)
            && let Some(result) = maybe_result.upgrade()
        {
            return Some(Ok(result.dupe()));
        }

        for path in &mut self.search_path_prefix {
            *path = config_root.join(&path);
        }

        let querier: Arc<dyn SourceDbQuerier> = match &self.args {
            BuildSystemArgs::Buck(args) => Arc::new(BxlQuerier::new(args.clone())),
            BuildSystemArgs::Custom(args) => Arc::new(CustomQuerier::new(args.clone())),
        };
        let source_db = ArcId::new(Box::new(QuerySourceDatabase::new(
            repo_root.to_path_buf(),
            querier,
        )) as Box<dyn SourceDatabase>);
        cache.insert(key, source_db.downgrade());
        Some(Ok(source_db))
    }
}
