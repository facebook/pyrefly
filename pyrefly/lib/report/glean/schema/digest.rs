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
pub struct FileDigest {
    pub id: u64,
    pub key: Box<FileDigest_key>,
}

impl FileDigest {
    pub fn GLEAN_name() -> String {
        String::from("digest.FileDigest.1")
    }

    pub fn new(file: src::File, digest: Digest) -> Self {
        FileDigest {
            id: 0,
            key: Box::new(FileDigest_key {
                file,
                digest
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct Digest {
    pub hash: String,
    pub size: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct FileDigest_key {
    pub file: src::File,
    pub digest: Digest,
}
