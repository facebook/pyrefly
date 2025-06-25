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

//! Utility functions that are not specific to the things Pyre does.

pub mod arc_id;
pub mod args;
pub mod assert_size;
pub mod display;
pub mod events;
pub mod exclusive_lock;
pub mod forgetter;
pub mod fs_anyhow;
pub mod gas;
pub mod globs;
pub mod lined_buffer;
pub mod lock;
pub mod locked_map;
pub mod memory;
pub mod no_hash;
pub mod owner;
pub mod prelude;
pub mod recurser;
pub mod ruff_visitors;
pub mod small_set1;
pub mod task_heap;
pub mod test_path;
pub mod thread_pool;
pub mod trace;
pub mod uniques;
pub mod upgrade_lock;
pub mod upward_search;
pub mod visit;
#[cfg(not(target_arch = "wasm32"))]
pub mod watcher;
pub mod with_hash;
