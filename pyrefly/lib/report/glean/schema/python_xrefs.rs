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

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct XRefsByTarget {
    pub id: u64,
    pub key: Box<XRefsByTarget_key>,
}

impl XRefsByTarget {
    pub fn new(target: python::DeclarationLocation, file: src::File, spans: Vec<src::ByteSpan>) -> Self {
        XRefsByTarget {
            id: 0,
            key: Box::new(XRefsByTarget_key {
                target,
                file,
                spans
            }),
        }
    }
}

impl GleanPredicate for XRefsByTarget {
    fn GLEAN_name() -> String {
        String::from("python.xrefs.XRefsByTarget.1")
    }

}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct XRefsByFile {
    pub id: u64,
    pub key: Box<XRefsByFile_key>,
}

impl XRefsByFile {
    pub fn new(file: src::File, xrefs: Vec<XRef>) -> Self {
        XRefsByFile {
            id: 0,
            key: Box::new(XRefsByFile_key {
                file,
                xrefs
            }),
        }
    }
}

impl GleanPredicate for XRefsByFile {
    fn GLEAN_name() -> String {
        String::from("python.xrefs.XRefsByFile.1")
    }

}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct XRefsByTarget_key {
    pub target: python::DeclarationLocation,
    pub file: src::File,
    pub spans: Vec<src::ByteSpan>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct XRefDefinitionLocation {
    pub name: python::Name,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<src::File>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct XRef {
    pub target: XRefDefinitionLocation,
    pub source: src::ByteSpan,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct XRefsByFile_key {
    pub file: src::File,
    pub xrefs: Vec<XRef>,
}
