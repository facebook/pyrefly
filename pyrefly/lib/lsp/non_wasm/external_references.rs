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
pub trait ExternalReferencesProvider {
    fn find_references(
        &self,
        qualified_name: &str,
        source_uri: &Url,
        timeout: Duration,
    ) -> Vec<(Url, Vec<Range>)>;
}
