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

pub mod flags_smoke;
pub mod get_builtin_type;
pub mod get_diagnostics;
pub mod get_function_parts;
pub mod get_matching_overloads;
pub mod get_overloads;
pub mod get_repr;
pub mod get_symbol;
pub mod get_symbols_for_file;
pub mod get_type;
pub mod get_type_alias_info;
pub mod get_type_attributes;
pub mod resolve_import_declaration;
pub mod search_for_type_attribute;
