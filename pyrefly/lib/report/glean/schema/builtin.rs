/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @generated
 * Regenerate with glean/schema/gen/Glean/Schema/Gen/Rust.hs
 *  buck2 run glean/schema/gen:gen-schema -- --dir glean/schema/source --rust pyrefly/pyrefly/lib/report/glean
 */

#![allow(warnings)]
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use serde_repr::*;

use crate::report::glean::schema::*;
use crate::report::glean::facts::GleanPredicate;
// Schema version
pub const VERSION: i64 = 1;
pub const SCHEMA_ID: &str = "835e529061f1429222e801fbfe85f209";

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct Unit {}
