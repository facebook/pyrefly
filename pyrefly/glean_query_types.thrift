/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Thrift types for deserializing Glean Angle tuple query results.
//
// When an Angle query returns a tuple like {File, Span, Digest}, Glean
// encodes it as a Thrift struct with fields named tuplefield0, tuplefield1, etc.
// (see glean/angle/Glean/Schema/Util.hs:59-62). The result is wrapped in a
// container struct with {id, optional key}.
//
// The struct definitions below mirror the Glean schema types (src.File,
// src.ByteSpan, digest.Digest) with matching field IDs so that compact
// protocol deserialization works correctly. They are inlined here to avoid
// depending on glean/schema/thrift:src-rust which has a codegen bug with
// types_split_count.

package "facebook.com/pyrefly/glean_query_types"

namespace rust pyrefly_glean_query_types

// Mirrors src.File (glean/schema/thrift/src.thrift)
struct File {
  1: i64 id;
  2: optional string key;
}

// Mirrors src.ByteSpan (glean/schema/thrift/src.thrift)
struct ByteSpan {
  1: i64 start;
  2: i64 length;
}

// Mirrors digest.Digest (glean/schema/thrift/digest.thrift)
struct Digest {
  1: string hash;
  2: i64 size;
}

/// Result container for the Angle query:
///   {File, Span, Digest} where
///     python.DeclarationWithName { declaration = Decl, name = "..." };
///     python.DeclarationUses { declaration = Decl, file = File, span = Span };
///     digest.FileDigest { file = File, digest = Digest };
struct FindReferencesResult {
  1: i64 id;
  2: optional FindReferencesResultTuple key;
}

/// The tuple fields in the same order as the query result columns.
struct FindReferencesResultTuple {
  1: File tuplefield0;
  2: ByteSpan tuplefield1;
  3: Digest tuplefield2;
}
