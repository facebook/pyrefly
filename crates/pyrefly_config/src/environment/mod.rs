/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

pub(crate) mod active_environment;
pub(crate) mod conda;
#[expect(
    clippy::module_inception,
    reason = "environment is both the module group and public API"
)]
pub mod environment;
pub(crate) mod finder;
pub mod interpreters;
pub(crate) mod venv;
