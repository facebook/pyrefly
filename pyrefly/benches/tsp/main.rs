/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! TSP benchmarks. Each benchmark lives in its own module; this crate root
//! aggregates their criterion groups into one binary, so the single `tsp_bench`
//! target builds and runs all of them. Select one at runtime with a criterion
//! name filter, e.g. `cargo bench -p pyrefly --bench tsp -- unopened_cached`.

mod get_computed_type_unopened_cached;

use criterion::criterion_main;

criterion_main!(get_computed_type_unopened_cached::benches);
