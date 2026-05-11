/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use pyrefly_build::handle::Handle;
use serde::Serialize;

use crate::state::state::Transaction;

#[derive(Serialize)]
struct Output {
    modules: BTreeMap<String, BTreeSet<String>>,
}

/// Produce a JSON string mapping each module's absolute filesystem path to the
/// sorted set of absolute filesystem paths it directly depends on.
/// Only modules in `handles` with on-disk paths are included; bundled typeshed
/// and in-memory modules are excluded.
pub fn dependency_graph(transaction: &Transaction, handles: &[Handle]) -> String {
    let graph = transaction.get_dependency_graph(handles);
    let modules: BTreeMap<String, BTreeSet<String>> = graph
        .into_iter()
        .map(|(path, deps)| {
            let deps: BTreeSet<String> = deps
                .into_iter()
                .map(|d| d.to_string_lossy().into_owned())
                .collect();
            (path.to_string_lossy().into_owned(), deps)
        })
        .collect();
    let output = Output { modules };
    serde_json::to_string_pretty(&output).unwrap()
}
