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

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct GenCodeSignature {
    pub id: u64,
    pub key: Box<String>,
}

impl GenCodeSignature {
    pub fn GLEAN_name() -> String {
        String::from("gencode.GenCodeSignature.1")
    }

    pub fn new(key: String) -> Self {
        GenCodeSignature {
            id: 0,
            key: Box::new(key),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct GenCodeCommand {
    pub id: u64,
    pub key: Box<String>,
}

impl GenCodeCommand {
    pub fn GLEAN_name() -> String {
        String::from("gencode.GenCodeCommand.1")
    }

    pub fn new(key: String) -> Self {
        GenCodeCommand {
            id: 0,
            key: Box::new(key),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct GenCodeClass {
    pub id: u64,
    pub key: Box<String>,
}

impl GenCodeClass {
    pub fn GLEAN_name() -> String {
        String::from("gencode.GenCodeClass.1")
    }

    pub fn new(key: String) -> Self {
        GenCodeClass {
            id: 0,
            key: Box::new(key),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct GenCodeBySource {
    pub id: u64,
    pub key: Box<GenCodeBySource_key>,
}

impl GenCodeBySource {
    pub fn GLEAN_name() -> String {
        String::from("gencode.GenCodeBySource.1")
    }

    pub fn new(source: src::File, gencode: src::File) -> Self {
        GenCodeBySource {
            id: 0,
            key: Box::new(GenCodeBySource_key {
                source,
                gencode
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct GenCode {
    pub id: u64,
    pub key: Box<GenCode_key>,
}

impl GenCode {
    pub fn GLEAN_name() -> String {
        String::from("gencode.GenCode.1")
    }

    pub fn new(file: src::File, variant: GenCodeVariant, source: Option<src::File>, command: Option<GenCodeCommand>, class_: Option<GenCodeClass>, signature: Option<GenCodeSignature>) -> Self {
        GenCode {
            id: 0,
            key: Box::new(GenCode_key {
                file,
                variant,
                source,
                command,
                class_,
                signature
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize_repr, Eq, Hash, PartialEq, Serialize_repr)]
#[repr(u8)]
pub enum GenCodeVariant {
    Full,
    Partial,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct GenCodeBySource_key {
    pub source: src::File,
    pub gencode: src::File,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct GenCode_key {
    pub file: src::File,
    pub variant: GenCodeVariant,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<src::File>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<GenCodeCommand>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub class_: Option<GenCodeClass>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<GenCodeSignature>,
}
