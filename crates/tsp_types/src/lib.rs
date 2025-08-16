/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Type Server Protocol (TSP) type definitions and common utilities
//!
//! This crate provides the core type definitions for the Type Server Protocol,
//! including generated protocol types and common utility functions for handling
//! TSP requests and responses.

pub mod common;
pub mod protocol;

// Re-export the main types for convenience
pub use common::*;
pub use protocol::*;
