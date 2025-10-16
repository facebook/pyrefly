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
#![deny(clippy::string_to_string)]
#![deny(clippy::trivially_copy_pass_by_ref)]
#![feature(const_type_name)]
#![feature(if_let_guard)]

use std::path::PathBuf;
use std::sync::Arc;

use serde::Deserialize;
use serde::Serialize;

pub mod handle;
pub mod source_db;
pub use source_db::SourceDatabase;
mod query;

use crate::query::SourceDbQuerier;
use crate::query::buck::BxlArgs;
use crate::query::buck::BxlQuerier;
use crate::query::custom::CustomQuerier;
use crate::query::custom::CustomQueryArgs;
use crate::source_db::query_source_db::QuerySourceDatabase;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case", tag = "type")]
pub enum BuildSystem {
    Buck(BxlArgs),
    Custom(CustomQueryArgs),
}

impl BuildSystem {
    pub fn get_source_db(
        &self,
        config_root: PathBuf,
    ) -> Box<dyn source_db::SourceDatabase + 'static> {
        let querier: Arc<dyn SourceDbQuerier> = match &self {
            Self::Buck(args) => Arc::new(BxlQuerier::new(args.clone())),
            Self::Custom(args) => Arc::new(CustomQuerier::new(args.clone())),
        };
        Box::new(QuerySourceDatabase::new(config_root, querier))
    }
}
