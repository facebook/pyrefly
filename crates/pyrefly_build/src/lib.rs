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
#![feature(let_chains)]

use std::path::PathBuf;

use dupe::Dupe;
use serde::Deserialize;
use serde::Serialize;

pub mod buck;
pub mod handle;
pub mod map_db;
pub mod source_db;

#[derive(Debug, Clone, Dupe, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum BuildSystem {
    Buck,
}

impl BuildSystem {
    pub fn get_source_db(
        &self,
        config_root: PathBuf,
    ) -> Box<dyn source_db::SourceDatabase + 'static> {
        match &self {
            Self::Buck => Box::new(buck::bxl::BuckSourceDatabase::new(config_root)),
        }
    }
}
