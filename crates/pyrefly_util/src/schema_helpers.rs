/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Helper functions for generating JSON schemas with schemars.

/// Generates a JSON schema for `Vec1<String>`: an array of strings with `minItems: 1`.
///
/// This is useful for fields that use `vec1::Vec1<String>` which requires at least one element.
/// Use with `#[schemars(schema_with = "pyrefly_util::schema_helpers::vec1_string_schema")]`.
#[cfg(feature = "jsonschema")]
pub fn vec1_string_schema(
    generator: &mut schemars::r#gen::SchemaGenerator,
) -> schemars::schema::Schema {
    use schemars::schema::ArrayValidation;
    use schemars::schema::InstanceType;
    use schemars::schema::SchemaObject;
    use schemars::schema::SingleOrVec;

    SchemaObject {
        instance_type: Some(InstanceType::Array.into()),
        array: Some(Box::new(ArrayValidation {
            items: Some(SingleOrVec::Single(Box::new(
                generator.subschema_for::<String>(),
            ))),
            min_items: Some(1),
            ..Default::default()
        })),
        ..Default::default()
    }
    .into()
}
