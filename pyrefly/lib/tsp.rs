/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Internal TSP support shared by the native server and portable hosts.
//!
//! The Rust interface is unstable; only the transport-independent conversion
//! modules are available on WebAssembly.

#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod requests;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod server;
pub mod transaction_conversion;
pub(crate) mod type_conversion;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod type_facts;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod validation;
