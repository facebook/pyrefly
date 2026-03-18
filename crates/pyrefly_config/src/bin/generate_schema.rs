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

use pyrefly_config::config::ConfigFile;
use schemars::r#gen::SchemaSettings;

fn main() {
    let settings = SchemaSettings::draft07().with(|s| {
        // option_nullable: When true, adds `"nullable": true` to schemas for Option<T> fields
        // (OpenAPI 3.0 style). We set this to false because JSON Schema Draft-07 doesn't use
        // the "nullable" keyword - it's an OpenAPI extension.
        s.option_nullable = false;

        // option_add_null_type: When true, Option<T> fields generate a union type like
        // `{"anyOf": [{"type": "null"}, <T's schema>]}`. We set this to false so that
        // optional fields simply omit the "required" constraint, and their schema is just
        // the inner type T's schema. This produces cleaner, more readable schemas.
        s.option_add_null_type = false;
    });
    let generator = settings.into_generator();
    let schema = generator.into_root_schema_for::<ConfigFile>();

    let json = serde_json::to_string_pretty(&schema).expect("Failed to serialize schema to JSON");
    println!("{json}");
}
