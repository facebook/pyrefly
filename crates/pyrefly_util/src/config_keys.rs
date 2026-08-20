/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/// The serialized config keys a struct accepts, used to suggest a "did you
/// mean" for unrecognized keys. Derive it with `#[derive(pyrefly_derive::ConfigKeys)]`,
/// which mirrors serde's serialized-key semantics: it honors `rename_all` and
/// per-field `rename`, drops `skip`/`skip_serializing` fields, and splices in
/// the keys and aliases of `#[serde(flatten)]` sub-structs through their own
/// `ConfigKeys` impl. The flatten catch-all field (`extras`) is excluded via
/// `#[config_keys(skip)]`.
pub trait ConfigKeys {
    /// The serialized keys of this struct, in field-declaration order.
    fn config_keys() -> Vec<&'static str>;

    /// The `#[serde(alias = "...")]` spellings this struct accepts, each paired
    /// with the canonical (resolved `rename`/`rename_all`) key it resolves to.
    /// A field may contribute several pairs when it declares multiple aliases.
    fn config_key_aliases() -> Vec<(&'static str, &'static str)>;
}
