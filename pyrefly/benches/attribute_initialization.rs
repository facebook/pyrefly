/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Measure edits to a file containing a constructor with many attributes and joins.
//! Server initialization and the first check are outside the measured region.

use std::fmt::Write as _;
use std::fs;

use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use lsp_types::DidChangeTextDocumentNotification;
use lsp_types::Uri;
use pyrefly::commands::lsp::IndexingMode;
use pyrefly::commands::lsp::LspArgs;
use pyrefly_lsp_test::object_model::InitializeSettings;
use pyrefly_lsp_test::object_model::LspInteraction;
use pyrefly_lsp_test::object_model::LspInteractionArgs;
use pyrefly_util::telemetry::NoTelemetry;
use pyrefly_util::thread_pool::TEST_THREAD_COUNT;
use serde_json::json;
use tempfile::tempdir;

fn attribute_initialization_edit(c: &mut Criterion) {
    let dir = tempdir().unwrap();
    fs::write(
        dir.path().join("pyrefly.toml"),
        "[errors]\npossibly-uninitialized-attribute = 'error'\n",
    )
    .unwrap();
    let path = dir.path().join("constructor.py");
    let mut source =
        String::from("class C:\n    def __init__(self, flag: bool, values: list[int]) -> None:\n");
    for i in 0..64 {
        let _ = writeln!(source, "        self.attribute_{i}: int = 0");
    }
    for i in 0..64 {
        let _ = writeln!(
            source,
            "        for value in values:\n            if flag:\n                self.attribute_{i} = value"
        );
    }
    let with_error = format!("{source}\nresult: int = 'error'\n");
    fs::write(&path, &source).unwrap();
    let mut interaction = LspInteraction::new_with_args(LspInteractionArgs {
        args: LspArgs {
            indexing_mode: IndexingMode::None,
            workspace_indexing_limit: 50,
            build_system_blocking: false,
        },
        telemetry: Box::new(NoTelemetry),
        thread_count: TEST_THREAD_COUNT,
        thrift_remapper: None,
    });
    interaction.set_root(dir.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(json!([
                {"pyrefly": {"displayTypeErrors": "force-on"}}
            ]))),
            ..Default::default()
        })
        .unwrap();
    interaction.client.did_open("constructor.py");
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(path.clone(), 0)
        .unwrap();

    let mut version = 1;
    c.bench_function("attribute_initialization/edit_to_diagnostics", |b| {
        b.iter(|| {
            version += 1;
            let has_error = version % 2 == 0;
            let text = if has_error { &with_error } else { &source };
            interaction
                .client
                .send_notification::<DidChangeTextDocumentNotification>(json!({
                    "textDocument": {
                        "uri": Uri::from_file_path(&path).unwrap(),
                        "version": version,
                    },
                    "contentChanges": [{"text": text}],
                }));
            // Alternating the error count ensures that each edit completes.
            interaction
                .client
                .expect_publish_diagnostics_eventual_error_count(
                    path.clone(),
                    usize::from(has_error),
                )
                .unwrap();
        });
    });
}

criterion_group!(benches, attribute_initialization_edit);
criterion_main!(benches);
