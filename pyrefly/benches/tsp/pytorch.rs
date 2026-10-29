/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! TSP vs. LSP latency for the same IDE features on a slow PyTorch file.
//!
//! A TSP client such as Pylance answers IDE requests itself and asks the type
//! server only for types. Each benchmark serves one IDE feature on
//! `torch/_inductor/ir.py` (one of the slowest PyTorch modules to solve)
//! through two servers over the same pinned checkout:
//! - `lsp`: the pyrefly language server answers the LSP request.
//! - `tsp`: the pyrefly type server answers the TSP requests that Pylance sent
//!   for the same feature, replayed from `traces/<feature>.json`.
//!
//! The features, on `V.graph.sizevars.guard_or_false(...)` in `ir.py`:
//! - `cold_start`: start a fresh server, open `ir.py`, and report its
//!   diagnostics. LSP: `textDocument/diagnostic`. TSP: every request Pylance
//!   sends after opening the file until it goes quiet, mostly its background
//!   checker's sweep of the file.
//! - `hover`: edit, then hover on `guard_or_false`.
//! - `completion`: edit, then complete after `V.graph.sizevars.`.
//! - `recheck`: edit, then report the file's diagnostics. LSP:
//!   `textDocument/diagnostic`. TSP: the requests the Pylance background
//!   checker sends to re-check the file, about one `getComputedType` per
//!   expression.
//!
//! The warm features start with an edit because Pylance caches the types it
//! receives until the snapshot changes: a hover or completion that repeats
//! without an edit sends no TSP requests at all.
//!
//! In Pylance, the background checker runs at the same time as the other
//! requests and shares the server's request queue with them; these
//! benchmarks measure each request stream alone.
//!
//! `record/` holds the tool that records the traces from a real Pylance; see
//! `benches/README.md`. The shared PyTorch checkout harness lives in
//! `benches/pytorch/common.rs`.

use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use clap::Parser as _;
use criterion::BatchSize;
use criterion::Criterion;
use criterion::criterion_group;
use lsp_server::ErrorCode;
use lsp_server::RequestId;
use lsp_server::ResponseError;
use lsp_types::Uri;
use pyrefly::commands::tsp::TspArgs;
use pyrefly::commands::tsp::run_tsp;
use pyrefly::lsp::non_wasm::protocol::Message;
use pyrefly::lsp::non_wasm::protocol::Request;
use pyrefly::lsp::non_wasm::server::Connection;
use pyrefly::module::bundled::BundledStub;
use pyrefly::module::third_party::get_bundled_third_party;
use pyrefly::module::typeshed::typeshed;
use pyrefly_lsp_test::object_model::InitializeSettings;
use pyrefly_lsp_test::object_model::LspInteraction;
use pyrefly_lsp_test::object_model::LspInteractionArgs;
use pyrefly_lsp_test::object_model::TestClient;
use pyrefly_util::lock::FinishHandle;
use pyrefly_util::telemetry::NoTelemetry;
use pyrefly_util::thread_pool::ThreadCount;
use pyrefly_util::trace::init_tracing;
use serde::Deserialize;
use serde_json::Value;

use crate::common::lsp_args;
use crate::common::pytorch_root_or_skip;

const IR: &str = "torch/_inductor/ir.py";

/// The position of `guard_or_false` in
/// `return V.graph.sizevars.guard_or_false(is_aligned)` in `ir.py`
/// (0-indexed), where the traces hovered and completed. `prepare` asserts the
/// text, so drift after a pin bump fails loudly.
const LINE: u32 = 489;
const METHOD: u32 = 28;

/// The TSP requests Pylance sent for one feature, recorded by `record/` against
/// the pinned checkout. URIs start with `${ROOT}` (the checkout), `${TYPESHED}`,
/// or `${THIRD_PARTY}` (the bundled stubs that pyrefly writes to disk).
#[derive(Deserialize)]
struct Trace {
    uris: Vec<String>,
    requests: Vec<TraceRequest>,
}

/// One recorded request, without its snapshot, which the replay fills in.
#[derive(Deserialize)]
#[serde(untagged)]
enum TraceRequest {
    /// `getComputedType` of a node, the bulk of every trace, stored compactly:
    /// an index into `Trace::uris`, then the start and end line and character.
    ComputedType(usize, u32, u32, u32, u32),
    Other {
        method: String,
        params: Value,
    },
}

#[derive(Clone, Copy)]
enum Feature {
    ColdStart,
    Hover,
    Completion,
    Recheck,
}

#[derive(Clone, Copy)]
enum Protocol {
    Lsp,
    Tsp,
}

/// Load the trace of `feature` for a checkout at `root`, as `(method, params)`
/// pairs.
fn load_trace(root: &Path, feature: Feature) -> Vec<(String, Value)> {
    let typeshed_dir = typeshed()
        .and_then(|bundle| bundle.materialized_path_on_disk())
        .expect("bundled typeshed should be written to disk");
    let third_party_dir = get_bundled_third_party()
        .and_then(|bundle| bundle.materialized_path_on_disk())
        .expect("bundled third-party stubs should be written to disk");
    let uri = |path: &Path| Uri::from_file_path(path).unwrap().to_string();
    // `include_str!` takes only a literal path, so each feature names its file.
    let json = match feature {
        Feature::ColdStart => include_str!("traces/cold_start.json"),
        Feature::Hover => include_str!("traces/hover.json"),
        Feature::Completion => include_str!("traces/completion.json"),
        Feature::Recheck => include_str!("traces/recheck.json"),
    };
    let text = json
        .replace("${ROOT}", &uri(root))
        .replace("${TYPESHED}", &uri(&typeshed_dir))
        .replace("${THIRD_PARTY}", &uri(&third_party_dir));
    let trace: Trace = serde_json::from_str(&text).unwrap();
    trace
        .requests
        .into_iter()
        .map(|request| match request {
            TraceRequest::ComputedType(
                uri,
                start_line,
                start_character,
                end_line,
                end_character,
            ) => (
                "typeServer/getComputedType".to_owned(),
                serde_json::json!({
                    "arg": {
                        "uri": trace.uris[uri],
                        "range": {
                            "start": { "line": start_line, "character": start_character },
                            "end": { "line": end_line, "character": end_character },
                        },
                    },
                }),
            ),
            TraceRequest::Other { method, params } => (method, params),
        })
        .collect()
}

/// A server for the pinned checkout.
struct Session {
    interaction: LspInteraction,
    protocol: Protocol,
    ir_text: String,
    /// The snapshot the TSP client last saw, as Pylance tracks it: updated by
    /// every `getSnapshot` response.
    snapshot: i32,
    next_tsp_id: u32,
    edits: u32,
}

/// Spawn the type server the same way `LspInteraction::new_with_args` spawns
/// the language server, with the arguments Pylance passes, so that both are
/// driven by the same client.
fn spawn_tsp() -> LspInteraction {
    let ((conn_client, _client_reader), (conn_server, server_reader)) = Connection::memory();
    let finish_handle = Arc::new(FinishHandle::new());
    let finish_server = finish_handle.clone();
    // `TspArgs`'s fields are private to the `pyrefly` crate, so it is built the
    // way the CLI builds it.
    let args = TspArgs::parse_from(["pyrefly", "--indexing-mode", "none"]);
    thread::spawn(move || {
        let _ = run_tsp(
            conn_server,
            server_reader,
            args,
            &NoTelemetry,
            None,
            ThreadCount::AllThreads,
            None,
        );
        finish_server.notify_finished();
    });
    LspInteraction {
        client: TestClient::new(conn_client, finish_handle),
    }
}

/// Start a server and initialize it, without opening any file.
fn start(root: &Path, protocol: Protocol) -> Session {
    let mut interaction = match protocol {
        Protocol::Lsp => LspInteraction::new_with_args(LspInteractionArgs {
            args: lsp_args(),
            telemetry: Box::new(NoTelemetry),
            thread_count: ThreadCount::AllThreads,
            thrift_remapper: None,
        }),
        Protocol::Tsp => spawn_tsp(),
    };
    interaction.client.set_message_logging(false);
    interaction
        .client
        .set_timeouts(Duration::from_secs(120), Duration::from_secs(1800));
    interaction.set_root(root.to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            workspace_folders: Some(vec![(
                "pytorch".to_owned(),
                Uri::from_file_path(root).unwrap(),
            )]),
            ..Default::default()
        })
        .unwrap();

    let ir_text = fs::read_to_string(root.join(IR)).unwrap();
    let line = ir_text.lines().nth(LINE as usize).unwrap();
    assert!(
        line[..METHOD as usize].ends_with("V.graph.sizevars.")
            && line[METHOD as usize..].starts_with("guard_or_false("),
        "LINE/METHOD drifted in {IR}; update them and re-record the traces after a pin bump"
    );
    Session {
        interaction,
        protocol,
        ir_text,
        snapshot: 0,
        next_tsp_id: 0,
        edits: 0,
    }
}

impl Session {
    fn client(&self) -> &TestClient {
        &self.interaction.client
    }

    fn tsp_request(&mut self, method: &str, params: Value) -> Result<Value, ResponseError> {
        self.next_tsp_id += 1;
        let id = RequestId::from(format!("tsp-{}", self.next_tsp_id));
        self.client().send_message(Message::Request(Request {
            id: id.clone(),
            method: method.to_owned(),
            params,
            activity_key: None,
        }));
        let response = self
            .client()
            .expect_message(method, |message| match message {
                Message::Response(response) if response.id == id => Some(Ok(response)),
                _ => None,
            })
            .unwrap();
        match response.error {
            Some(error) => Err(error),
            None => Ok(response.result.unwrap_or(Value::Null)),
        }
    }

    /// Send one recorded request at the current snapshot. When the server
    /// reports the snapshot outdated, fetch the new one and retry, as Pylance
    /// does.
    fn replay(&mut self, method: &str, params: &Value) {
        loop {
            let mut request = params.clone();
            if let Value::Object(fields) = &mut request {
                fields.insert("snapshot".to_owned(), self.snapshot.into());
            }
            match self.tsp_request(method, request) {
                Ok(result) if method == "typeServer/getSnapshot" => {
                    self.snapshot = serde_json::from_value(result).unwrap();
                    return;
                }
                Ok(_) => return,
                Err(error) if error.code == ErrorCode::ServerCancelled as i32 => {
                    let snapshot = self
                        .tsp_request("typeServer/getSnapshot", Value::Null)
                        .unwrap();
                    self.snapshot = serde_json::from_value(snapshot).unwrap();
                }
                Err(error) => panic!("{method} {params} failed: {error:?}"),
            }
        }
    }

    /// Append a distinct comment to `ir.py`, so that every edit changes the
    /// file without moving the queried position.
    fn edit(&mut self) {
        self.edits += 1;
        let text = format!("{}\n# pyrefly bench edit {}\n", self.ir_text, self.edits);
        self.client().did_change(IR, &text);
    }

    fn serve(&mut self, feature: Feature, trace: &[(String, Value)]) {
        match feature {
            Feature::ColdStart => self.client().did_open(IR),
            _ => self.edit(),
        }
        match self.protocol {
            Protocol::Tsp => trace
                .iter()
                .for_each(|(method, params)| self.replay(method, params)),
            Protocol::Lsp => match feature {
                Feature::Hover => self
                    .client()
                    .hover(IR, LINE, METHOD)
                    .expect_hover_response_with_markup(|markup| {
                        markup.is_some_and(|markup| markup.contains("guard_or_false"))
                    })
                    .unwrap(),
                Feature::Completion => self
                    .client()
                    .completion(IR, LINE, METHOD)
                    .expect_completion_response_with(|list| {
                        list.items.iter().any(|item| item.label == "guard_or_false")
                    })
                    .unwrap(),
                Feature::ColdStart | Feature::Recheck => self
                    .client()
                    .diagnostic(IR)
                    .expect_response_with(|_| true)
                    .unwrap(),
            },
        }
    }
}

fn bench(c: &mut Criterion, name: &str, feature: Feature) {
    let Some(root) = pytorch_root_or_skip() else {
        return;
    };
    // `LspInteraction::new_with_args` initializes tracing the first time it
    // runs; doing it here first gives both servers the same logging setup.
    init_tracing(true, true);
    let trace = load_trace(&root, feature);
    let mut group = c.benchmark_group(format!("tsp/pytorch_{name}"));
    group.sample_size(10);
    for (id, protocol) in [("lsp", Protocol::Lsp), ("tsp", Protocol::Tsp)] {
        match feature {
            // Every sample starts a fresh server. The session is returned so
            // that the server shuts down outside the measurement.
            Feature::ColdStart => group.bench_function(id, |b| {
                b.iter_batched(
                    || (),
                    |()| {
                        let mut session = start(&root, protocol);
                        session.serve(feature, &trace);
                        session
                    },
                    BatchSize::PerIteration,
                )
            }),
            _ => {
                let mut session = start(&root, protocol);
                session.client().did_open(IR);
                session
                    .client()
                    .expect_publish_diagnostics_for_file(root.join(IR))
                    .unwrap();
                group.bench_function(id, |b| b.iter(|| session.serve(feature, &trace)))
            }
        };
    }
    group.finish();
}

fn cold_start(c: &mut Criterion) {
    bench(c, "cold_start", Feature::ColdStart);
}

fn hover(c: &mut Criterion) {
    bench(c, "hover", Feature::Hover);
}

fn completion(c: &mut Criterion) {
    bench(c, "completion", Feature::Completion);
}

fn recheck(c: &mut Criterion) {
    bench(c, "recheck", Feature::Recheck);
}

criterion_group!(benches, cold_start, hover, completion, recheck);
