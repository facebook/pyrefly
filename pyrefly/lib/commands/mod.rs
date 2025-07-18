/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

pub mod autotype;
pub mod buck_check;
pub mod check;
pub mod config_args;
pub mod config_finder;
#[cfg(not(target_arch = "wasm32"))]
pub mod config_migration;
pub mod dump_config;
pub mod globs_and_config_getter;
#[cfg(not(target_arch = "wasm32"))]
pub mod init;
#[cfg(not(target_arch = "wasm32"))]
pub mod lsp;
#[cfg(not(target_arch = "wasm32"))]
pub mod run;
pub mod suppress;
pub mod util;
