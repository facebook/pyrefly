/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! PyTorch walltime benchmarks. Each benchmark lives in its own module
//! (`cold_start`, `error_propagation`, `rename`, `indexed_memory`,
//! `workspace_symbol`, `full_check`) sharing the checkout harness in [`common`];
//! this crate root aggregates their criterion groups into one binary.
//! Individual benchmarks are still selectable by name at runtime, e.g.
//! `cargo bench -p pyrefly --bench pytorch -- cold_start`.

mod cold_start;
mod common;
mod error_propagation;
mod full_check;
mod indexed_memory;
mod rename;
mod workspace_symbol;

use criterion::criterion_main;

criterion_main!(
    cold_start::benches,
    error_propagation::benches,
    rename::benches,
    indexed_memory::benches,
    workspace_symbol::benches,
    full_check::benches
);
