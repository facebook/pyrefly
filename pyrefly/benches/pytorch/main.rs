/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! PyTorch walltime LSP benchmarks. Each benchmark lives in its own module
//! (`cold_start`, `error_propagation`) sharing the checkout harness in
//! [`common`]; this crate root just aggregates their criterion groups into one
//! binary, so a single `pytorch_bench` target builds and runs all of them.
//! Individual benchmarks are still selectable by name at runtime, e.g.
//! `cargo bench -p pyrefly --bench pytorch -- cold_start`.

mod cold_start;
mod common;
mod error_propagation;

use criterion::criterion_main;

criterion_main!(cold_start::benches, error_propagation::benches);
