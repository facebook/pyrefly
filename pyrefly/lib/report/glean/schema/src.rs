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

use crate::report::glean::schema::*;

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct RangeContains {
    pub id: u64,
    pub key: Box<RangeContains_key>,
}

impl RangeContains {
    pub fn GLEAN_name() -> String {
        String::from("src.RangeContains.1")
    }

    pub fn new(fileLines: Range, contains: Range) -> Self {
        RangeContains {
            id: 0,
            key: Box::new(RangeContains_key {
                fileLines,
                contains
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct IndexFailure {
    pub id: u64,
    pub key: Box<IndexFailure_key>,
}

impl IndexFailure {
    pub fn GLEAN_name() -> String {
        String::from("src.IndexFailure.1")
    }

    pub fn new(file: File, reason: IndexFailureReason, details: String) -> Self {
        IndexFailure {
            id: 0,
            key: Box::new(IndexFailure_key {
                file,
                reason,
                details
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct FileLines {
    pub id: u64,
    pub key: Box<FileLines_key>,
}

impl FileLines {
    pub fn GLEAN_name() -> String {
        String::from("src.FileLines.1")
    }

    pub fn new(file: File, lengths: Vec<u64>, endsInNewline: bool, hasUnicodeOrTabs: bool) -> Self {
        FileLines {
            id: 0,
            key: Box::new(FileLines_key {
                file,
                lengths,
                endsInNewline,
                hasUnicodeOrTabs
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct FileLanguage {
    pub id: u64,
    pub key: Box<FileLanguage_key>,
}

impl FileLanguage {
    pub fn GLEAN_name() -> String {
        String::from("src.FileLanguage.1")
    }

    pub fn new(file: File, language: Language) -> Self {
        FileLanguage {
            id: 0,
            key: Box::new(FileLanguage_key {
                file,
                language
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct FileDigest {
    pub id: u64,
    pub key: Box<File>,
    pub value: FileDigest_value,
}

impl FileDigest {
    pub fn GLEAN_name() -> String {
        String::from("src.FileDigest.1")
    }

    pub fn new(key: File, value: FileDigest_value) -> Self {
        FileDigest {
            id: 0,
            key: Box::new(key),
            value: value,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct File {
    pub id: u64,
    pub key: Box<String>,
}

impl File {
    pub fn GLEAN_name() -> String {
        String::from("src.File.1")
    }

    pub fn new(key: String) -> Self {
        File {
            id: 0,
            key: Box::new(key),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ByteSpanContains {
    pub id: u64,
    pub key: Box<ByteSpanContains_key>,
}

impl ByteSpanContains {
    pub fn GLEAN_name() -> String {
        String::from("src.ByteSpanContains.1")
    }

    pub fn new(byteSpan: ByteSpan, contains: ByteSpan) -> Self {
        ByteSpanContains {
            id: 0,
            key: Box::new(ByteSpanContains_key {
                byteSpan,
                contains
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct RelByteSpan {
    pub offset: u64,
    pub length: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct Range {
    pub file: File,
    pub lineBegin: u64,
    pub columnBegin: u64,
    pub lineEnd: u64,
    pub columnEnd: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct RangeContains_key {
    pub fileLines: Range,
    pub contains: Range,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct PackedByteSpansGroup {
    pub length: u64,
    pub offsets: Vec<u64>,
}

pub type PackedByteSpans = Vec<PackedByteSpansGroup>;

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct Loc {
    pub file: File,
    pub line: u64,
    pub column: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub enum Language {
    Buck,
    C,
    Cpp,
    Hack,
    Haskell,
    ObjC,
    ObjCpp,
    Python,
    Thrift,
    Java,
    GraphQL,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub enum IndexFailureReason {
    CompileError,
    BuildSystemError,
    Unclassified,
    DiscoveryError,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct IndexFailure_key {
    pub file: File,
    pub reason: IndexFailureReason,
    pub details: String,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct FileLines_key {
    pub file: File,
    pub lengths: Vec<u64>,
    pub endsInNewline: bool,
    pub hasUnicodeOrTabs: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct FileLanguage_key {
    pub file: File,
    pub language: Language,
}

pub type FileDigest_value = String;

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ByteSpan {
    pub start: u64,
    pub length: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ByteSpanContains_key {
    pub byteSpan: ByteSpan,
    pub contains: ByteSpan,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct FileLocation {
    pub file: File,
    pub span: ByteSpan,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ByteRange {
    pub begin: u64,
    pub end: u64,
}
