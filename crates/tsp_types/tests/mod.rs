/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests for TSP types - parameter construction, serialization, and type validation
//!
//! This module contains tests that focus purely on tsp_types functionality:
//! - Parameter type construction and validation
//! - Serialization/deserialization round-trips
//! - Flag and enum validation
//! - Type structure validation
//!
//! These tests are separate from integration tests that require the main pyrefly crate.

pub mod get_snapshot;
