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

pub mod combine_types;
pub mod get_builtin_type;
pub mod get_diagnostics;
pub mod get_diagnostics_version;
pub mod get_docstring;
pub mod get_matching_overloads;
pub mod get_metaclass;
pub mod get_overloads;
pub mod get_repr;
pub mod get_snapshot;
pub mod get_supported_protocol_version;
pub mod get_symbols_for_node;
pub mod get_symbols_for_type;
pub mod get_type;
pub mod get_type_alias_info;
pub mod get_type_args;
pub mod resolve_import_declaration;
