/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! PyTorch walltime benchmarks. Each benchmark lives in its own module
//! (`cold_start`, `error_propagation`, `rename`, `indexed_memory`, `full_check`)
//! sharing the checkout harness in [`common`]; this crate root aggregates their
//! groups into one binary, so a single `pytorch_bench` target runs all of them.
//! Individual benchmarks are still selectable by name at runtime, e.g.
//! `cargo bench -p pyrefly --bench pytorch -- cold_start`.

mod cold_start;
mod common;
mod error_propagation;
mod full_check;
mod indexed_memory;
mod rename;

use criterion::criterion_main;

criterion_main!(
    cold_start::benches,
    error_propagation::benches,
    rename::benches,
    indexed_memory::benches,
    full_check::benches
);
