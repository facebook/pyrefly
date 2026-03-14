/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Structured type representation for CinderX reports.
//!
//! Types are represented as a flat, deduplicated table of `TypeTableEntry`
//! values. Each entry carries a structural hash so that identical types
//! (across modules or within a single module) share the same table slot.
//!
//! The representation is designed for CinderX's static Python compiler,
//! which needs per-expression type information in a format it can consume
//! without depending on pyrefly internals.

use std::collections::HashMap;
use std::hash::Hasher;

use pyrefly_util::lined_buffer::PythonASTRange;
use serde::Serialize;
use xxhash_rust::xxh64::Xxh64;

// ---------------------------------------------------------------------------
// StructuredType: the four kinds of type entries
// ---------------------------------------------------------------------------

/// Structured type representation for the CinderX report protocol.
/// Each variant maps to a category of Python type that CinderX cares about.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind")]
pub enum StructuredType {
    /// Real Python class with qualified name and type arguments.
    /// Covers ClassType, TypedDict, and other types backed by actual Python classes.
    #[serde(rename = "class")]
    Class {
        qname: String,
        args: Vec<usize>,
        /// Optional semantic tags for class-like entries.
        ///
        /// These are classification hints derived from pyrefly metadata/MRO,
        /// not type arguments. In v0 we only emit typed-dict related tags:
        /// - `"typed_dict"`
        /// - `"partial_typed_dict"`
        ///
        /// Additional tags may be added in future versions.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        traits: Vec<String>,
    },
    /// Callable type with parameter types and return type.
    /// `defining_func` is the fully qualified name of the function that
    /// defines this callable (for dispatch purposes). Only set for
    /// `Type::Function`; `Type::Callable` is purely structural and has no
    /// identity, so it stays `None`.
    #[serde(rename = "callable")]
    Callable {
        params: Vec<usize>,
        return_type: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        defining_func: Option<String>,
    },
    /// Type operators, special forms, and other non-class type constructs.
    /// These do not have an MRO and are not real Python classes.
    #[serde(rename = "other_form")]
    OtherForm { qname: String, args: Vec<usize> },
    /// A method bound to a specific object instance.
    /// `self_type` is the receiver, `func_type` is the underlying callable.
    /// `defining_class` is the fully qualified name of the class that defines
    /// the method (for dispatch purposes).
    #[serde(rename = "bound_method")]
    BoundMethod {
        self_type: usize,
        func_type: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        defining_class: Option<String>,
    },
    /// Type variable with name and bounds.
    #[serde(rename = "variable")]
    Variable {
        name: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        bounds: Vec<usize>,
    },
    /// Literal value (used as type arg of `typing_extensions.Literal`).
    /// `promoted_type` points to the class entry for the literal's runtime type
    /// (e.g. `builtins.int` for `Literal[42]`), so CinderX can compile
    /// operations without re-deriving the type.
    #[serde(rename = "literal")]
    Literal { value: String, promoted_type: usize },
}

// ---------------------------------------------------------------------------
// TypeTableEntry: one slot in the deduplicated table
// ---------------------------------------------------------------------------

/// A single entry in the type table: structured type + structural hash.
#[derive(Debug, Clone, Serialize)]
pub struct TypeTableEntry {
    #[serde(flatten)]
    pub ty: StructuredType,
    pub hash: u64,
}

// ---------------------------------------------------------------------------
// LocatedType: pairs a source location with a type table index
// ---------------------------------------------------------------------------

/// A located type reference: source range paired with an index into the type table.
///
/// When a facet narrow is detected (e.g. `foo.x` where `foo.x` has been narrowed
/// in a conditional), `unnarrowed_type` holds the type that would be resolved
/// without the narrow and `is_narrowed_mismatch` indicates whether the narrowed
/// and unnarrowed types differ structurally.
#[derive(Debug, Clone, Serialize)]
pub struct LocatedType {
    #[serde(rename = "loc")]
    pub location: PythonASTRange,
    #[serde(rename = "type")]
    pub type_index: usize,
    /// Index into the type table for the unnarrowed type, if a facet narrow
    /// was detected on this expression.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unnarrowed_type: Option<usize>,
    /// True when the narrowed and unnarrowed types have different structural hashes.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub is_narrowed_mismatch: bool,
}

// ---------------------------------------------------------------------------
// TypeTable: builder with hash-based deduplication
// ---------------------------------------------------------------------------

/// Builder for a deduplicated type table.
///
/// Entries are inserted bottom-up (children before parents).
/// Deduplication is by structural hash: if two types produce the same
/// hash they share a single table slot.
pub(crate) struct TypeTable {
    /// The ordered list of unique entries.
    entries: Vec<TypeTableEntry>,
    /// Maps structural hash -> index in `entries` for O(1) dedup lookup.
    seen: HashMap<u64, usize>,
}

impl TypeTable {
    pub(crate) fn new() -> Self {
        Self {
            entries: Vec::new(),
            seen: HashMap::new(),
        }
    }

    /// Insert an entry if not already present (by hash).
    /// Returns the table index.
    ///
    /// TODO(stroxler): Deduplication is by hash only, not structural equality.
    /// Hash collisions (two different types mapping to the same u64) would
    /// silently merge distinct types. With 64-bit xxh64 this is astronomically
    /// unlikely in practice, but we should either add a structural equality
    /// check on collision or document this as an accepted limitation.
    pub(crate) fn insert(&mut self, ty: StructuredType, hash: u64) -> usize {
        if let Some(&idx) = self.seen.get(&hash) {
            return idx;
        }
        let idx = self.entries.len();
        self.entries.push(TypeTableEntry { ty, hash });
        self.seen.insert(hash, idx);
        idx
    }

    /// Set traits on an existing class entry. No-op if the entry is not a Class.
    #[allow(dead_code)] // Used once pending_class_traits post-processing is wired up.
    pub(crate) fn set_traits(&mut self, idx: usize, traits: Vec<String>) {
        if let StructuredType::Class {
            traits: ref mut t, ..
        } = self.entries[idx].ty
        {
            *t = traits;
        }
    }

    /// Look up the hash of an entry by index. Used when building parent hashes
    /// from child indices.
    pub(crate) fn hash_at(&self, idx: usize) -> u64 {
        self.entries[idx].hash
    }

    /// Consume the builder and return the finished entry list.
    pub(crate) fn into_entries(self) -> Vec<TypeTableEntry> {
        self.entries
    }
}

// ---------------------------------------------------------------------------
// Structural hash functions
// ---------------------------------------------------------------------------

/// Kind discriminants for structural hashing.
/// Each `StructuredType` variant gets a unique byte prefix so that
/// e.g. a class named "int" and a variable named "int" hash differently.
const HASH_KIND_CLASS: u8 = 0;
const HASH_KIND_CALLABLE: u8 = 1;
const HASH_KIND_VARIABLE: u8 = 2;
const HASH_KIND_LITERAL: u8 = 3;
const HASH_KIND_OTHER_FORM: u8 = 4;
const HASH_KIND_BOUND_METHOD: u8 = 5;

/// Compute structural hash for a class-kind type.
///
/// Traits are included in the hash because they are part of the structural
/// identity (e.g. a `TypedDict` and `PartialTypedDict` with the same qname
/// must not collide).
pub(crate) fn hash_class(qname: &str, arg_hashes: &[u64], traits: &[&str]) -> u64 {
    let mut h = Xxh64::new(0);
    h.write_u8(HASH_KIND_CLASS);
    h.write(qname.as_bytes());
    for &ah in arg_hashes {
        h.write_u64(ah);
    }
    for t in traits {
        h.write(t.as_bytes());
    }
    h.finish()
}

/// Compute structural hash for a callable-kind type.
///
/// When `defining_func` is present, it is included in the hash so that
/// two functions with the same signature but different identities get
/// distinct table entries (CinderX needs this for dispatch).
pub(crate) fn hash_callable(
    param_hashes: &[u64],
    return_hash: u64,
    defining_func: Option<&str>,
) -> u64 {
    let mut h = Xxh64::new(0);
    h.write_u8(HASH_KIND_CALLABLE);
    for &ph in param_hashes {
        h.write_u64(ph);
    }
    h.write_u64(return_hash);
    if let Some(df) = defining_func {
        h.write(df.as_bytes());
    }
    h.finish()
}

/// Compute structural hash for a variable-kind type.
pub(crate) fn hash_variable(name: &str, bound_hashes: &[u64]) -> u64 {
    let mut h = Xxh64::new(0);
    h.write_u8(HASH_KIND_VARIABLE);
    h.write(name.as_bytes());
    for &bh in bound_hashes {
        h.write_u64(bh);
    }
    h.finish()
}

/// Compute structural hash for a literal-kind type.
///
/// The promoted type's hash is included so that two literals with the same
/// string representation but different promoted types (which shouldn't happen
/// in practice, but could in theory) get distinct hashes.
pub(crate) fn hash_literal(value: &str, promoted_hash: u64) -> u64 {
    let mut h = Xxh64::new(0);
    h.write_u8(HASH_KIND_LITERAL);
    h.write(value.as_bytes());
    h.write_u64(promoted_hash);
    h.finish()
}

/// Compute structural hash for an other-form-kind type.
pub(crate) fn hash_other_form(qname: &str, arg_hashes: &[u64]) -> u64 {
    let mut h = Xxh64::new(0);
    h.write_u8(HASH_KIND_OTHER_FORM);
    h.write(qname.as_bytes());
    for &ah in arg_hashes {
        h.write_u64(ah);
    }
    h.finish()
}

/// Compute structural hash for a bound-method-kind type.
pub(crate) fn hash_bound_method(
    self_hash: u64,
    func_hash: u64,
    defining_class: Option<&str>,
) -> u64 {
    let mut h = Xxh64::new(0);
    h.write_u8(HASH_KIND_BOUND_METHOD);
    h.write_u64(self_hash);
    h.write_u64(func_hash);
    if let Some(dc) = defining_class {
        h.write(dc.as_bytes());
    }
    h.finish()
}
