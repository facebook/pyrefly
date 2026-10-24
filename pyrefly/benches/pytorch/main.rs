/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! PyTorch walltime benchmarks. Each benchmark lives in its own module
//! (`cold_start`, `error_propagation`, `full_check`, `indexed_memory`,
//! `semantic_tokens`, `workspace_symbol`) sharing the checkout harness in
//! [`common`]. This crate root aggregates their Criterion groups into one binary.
//! A single `pytorch_bench` target builds and runs all of them.
//! Individual benchmarks are still selectable by name at runtime, e.g.
//! `cargo bench -p pyrefly --bench pytorch -- cold_start`.

mod cold_start;
mod common;
mod error_propagation;
mod full_check;
mod indexed_memory;
mod semantic_tokens;
mod workspace_symbol;

use criterion::criterion_main;

criterion_main!(
    cold_start::benches,
    error_propagation::benches,
    indexed_memory::benches,
    semantic_tokens::benches,
    workspace_symbol::benches,
    full_check::benches
);
