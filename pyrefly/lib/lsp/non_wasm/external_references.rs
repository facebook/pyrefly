/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::time::Duration;

use lsp_types::Range;
use lsp_types::Url;

#[expect(dead_code)]
pub trait ExternalReferences {
    fn find_references(
        &self,
        qualified_name: &str,
        source_uri: &Url,
        timeout: Duration,
    ) -> Vec<(Url, Vec<Range>)>;
}

#[expect(dead_code)]
pub struct NoExternalReferences;

/// This struct will be used when we have no source of external references.
impl ExternalReferences for NoExternalReferences {
    fn find_references(
        &self,
        _qualified_name: &str,
        _source_uri: &Url,
        _timeout: Duration,
    ) -> Vec<(Url, Vec<Range>)> {
        Vec::new()
    }
}
