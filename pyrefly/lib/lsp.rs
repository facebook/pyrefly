/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

pub mod module_helpers;

#[cfg(not(target_arch = "wasm32"))]
pub mod non_wasm;
pub mod wasm;
