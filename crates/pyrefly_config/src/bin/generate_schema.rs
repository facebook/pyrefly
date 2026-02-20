/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Binary to auto-generate JSON Schema for pyrefly.toml / pyproject.toml [tool.pyrefly]
//! configuration files using the `schemars` crate.
//!
//! Usage:
//!   cargo run -p pyrefly_config --features jsonschema --bin generate_schema
//!
//! The output is written to stdout (JSON Schema Draft-07).
//! Redirect to a file to save:
//!   cargo run -p pyrefly_config --features jsonschema --bin generate_schema > schemas/pyrefly.json

use schemars::r#gen::SchemaSettings;

use pyrefly_config::config::ConfigFile;

fn main() {
    let settings = SchemaSettings::draft07().with(|s| {
        s.option_nullable = false;
        s.option_add_null_type = false;
    });
    let generator = settings.into_generator();
    let schema = generator.into_root_schema_for::<ConfigFile>();

    let json = serde_json::to_string_pretty(&schema).expect("Failed to serialize schema to JSON");
    println!("{json}");
}
