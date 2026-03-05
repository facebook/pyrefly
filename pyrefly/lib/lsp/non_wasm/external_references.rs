/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::time::Duration;

use lsp_types::Range;
use lsp_types::Url;
use pyrefly_util::telemetry::SubTaskTelemetry;

pub trait ExternalReferences: Send + Sync {
    fn find_references(
        &self,
        qualified_name: &str,
        source_uri: &Url,
        timeout: Duration,
        telemetry: Option<SubTaskTelemetry>,
    ) -> Vec<(Url, Vec<Range>)>;
}

pub struct NoExternalReferences;

impl ExternalReferences for NoExternalReferences {
    fn find_references(
        &self,
        _qualified_name: &str,
        _source_uri: &Url,
        _timeout: Duration,
        _telemetry: Option<SubTaskTelemetry>,
    ) -> Vec<(Url, Vec<Range>)> {
        Vec::new()
    }
}
