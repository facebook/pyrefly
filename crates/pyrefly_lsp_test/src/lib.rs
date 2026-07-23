/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A reusable, in-process harness for driving Pyrefly's language server from
//! tests and benchmarks. [`object_model::LspInteraction`] spawns the server on a
//! thread connected via in-memory channels and exposes a `TestClient` for sending
//! requests/notifications and asserting on responses and diagnostics.
//!
//! This lives in its own crate (rather than inside the `pyrefly` test module) so
//! that both the `lsp_interaction` tests and the `pyrefly` benchmarks — separate
//! compilation units — can share it.

mod init;
pub mod object_model;

// Re-export the pyrefly LSP types that cross this harness's API boundary. The
// `lsp_interaction` tests run inline in the `pyrefly` crate's own `cfg(test)`
// build, so their `crate::` types come from a *different* instance of `pyrefly`
// than the one this harness links. Importing these from the harness guarantees
// tests construct the exact types the harness expects, rather than tripping over
// "multiple versions of crate pyrefly" mismatches.
pub use pyrefly::commands::lsp::IndexingMode;
pub use pyrefly::commands::lsp::LspArgs;
pub use pyrefly::lsp::non_wasm::protocol::Message;
pub use pyrefly::lsp::non_wasm::protocol::Notification;
pub use pyrefly::lsp::non_wasm::protocol::Request;
