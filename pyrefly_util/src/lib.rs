/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
pub mod lock;
pub mod locked_map;
pub mod memory;
pub mod no_hash;
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
