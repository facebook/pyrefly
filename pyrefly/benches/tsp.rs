/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `tsp/get_computed_type_unopened_cached`: durable regression coverage for
//! the solve-reuse behavior added behind D118537886 ([pyrefly][PR] Commit the
//! solve behind a type query on an unopened file).
//!
//! Before that fix, every `typeServer/getComputedType` request against a file
//! the client had never opened solved the whole module from scratch (to
//! `Step::Solutions`, `Require::Everything`) and threw the result away once
//! the response was sent -- so an *identical, repeated* request on an
//! unchanged snapshot paid the full solve cost every single time. After the
//! fix, the first such request commits its solve, so later requests on the
//! same snapshot reuse it instead of re-solving.
//!
//! This benchmark drives a real TSP server (indexing disabled, so the only
//! background work is a recheck this benchmark never triggers) over the
//! plain in-process main JSON-RPC connection -- the dispatch code that
//! commits an unopened-file solve
//! (`TspServer::dispatch_tsp_request`/`handle_get_computed_type`) treats a
//! request from the main connection and one from a real "extra connection"
//! identically, so the extra connection's own transport plumbing adds
//! complexity without exercising any more of the fix's logic. It repeats the
//! same `getComputedType` request, on a stable snapshot, against one
//! never-opened, unreferenced module deliberately expensive to solve (nested
//! generic constructors -- see [`fixture_source`]). Server spawn, fixture
//! generation, and the first (necessarily cold) request all happen outside
//! Criterion's timing; only the repeated request is measured, so this
//! benchmark is exactly the steady-state behavior the fix changes.

use std::thread;
use std::thread::JoinHandle;
use std::time::Duration;

use clap::Parser as _;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use crossbeam_channel::RecvTimeoutError;
use lsp_server::RequestId;
use lsp_types::Url;
use pyrefly::commands::tsp::TspArgs;
use pyrefly::commands::tsp::run_tsp;
use pyrefly::lsp::non_wasm::protocol::Message;
use pyrefly::lsp::non_wasm::protocol::Notification;
use pyrefly::lsp::non_wasm::protocol::Request;
use pyrefly::lsp::non_wasm::protocol::Response;
use pyrefly::lsp::non_wasm::server::Connection;
use pyrefly_util::telemetry::NoTelemetry;
use pyrefly_util::thread_pool::ThreadCount;
use tempfile::TempDir;

/// Nesting depth of the constructor chain in [`fixture_source`]. Each layer
/// pushes `Base[object]` inward through a soft contextual-type mismatch,
/// which (per the identical fixture in `benches/micro.rs`'s
/// `nested_generic_constructor_soft_error`) makes every layer retry, so cost
/// grows sharply with depth. At this depth the one-time cold solve measures
/// ~200ms (vs. ~60-90µs for a reused, committed repeat) -- unmistakably
/// expensive, while the unmeasured setup still stays a small fraction of a
/// second.
const DEPTH: usize = 15;

/// Position of `result` in `result: Base[object] = ...`, the line whose
/// binding requires solving the whole nested constructor chain.
const QUERY_LINE: u32 = 12;
const QUERY_CHARACTER: u32 = 0;

/// One never-opened, unreferenced module: a `depth`-deep chain of generic
/// constructors assigned to an annotated name. Identical in shape to
/// `benches/micro.rs`'s `nested_generic_constructor_soft_error`, which
/// documents why it is expensive: each layer's contextual type mismatch is
/// soft, so the solver retries it, and retries compound with depth.
fn fixture_source(depth: usize) -> String {
    let mut expression = "Leaf(lambda x: 0)".to_owned();
    for _ in 0..depth {
        expression = format!("Box({expression})");
    }
    format!(
        "from typing import Generic, TypeVar\n\
         \n\
         T = TypeVar(\"T\", covariant=True)\n\
         \n\
         class Base(Generic[T]): ...\n\
         \n\
         class Box(Base[T], Generic[T]):\n\
         \x20   def __init__(self, value: Base[T]) -> None: ...\n\
         \n\
         class Leaf(Base[T], Generic[T]):\n\
         \x20   def __init__(self, value: T) -> None: ...\n\
         \n\
         result: Base[object] = {expression}\n"
    )
}

/// Write the fixture module (plus a `pyproject.toml` marking the project
/// root) into a fresh temp project. Returns the directory (kept alive for as
/// long as the fixture is in use) and the module's `file://` URI. The module
/// is never `didOpen`ed and nothing imports it -- this is TSP's
/// unopened-file query path, the one the fix changed.
fn write_fixture() -> (TempDir, String) {
    let dir = tempfile::tempdir().expect("create temp project for TSP bench fixture");
    std::fs::write(
        dir.path().join("pyproject.toml"),
        "[project]\nname = \"tsp-bench-fixture\"\nversion = \"1.0.0\"\n",
    )
    .expect("write pyproject.toml");
    let module_path = dir.path().join("heavy.py");
    std::fs::write(&module_path, fixture_source(DEPTH)).expect("write heavy.py");
    let uri = Url::from_file_path(&module_path)
        .expect("fixture path should be absolute")
        .to_string();
    (dir, uri)
}

/// A minimal, in-process TSP client, trimmed to what this benchmark needs
/// (the full test-only harness in `lib/test/tsp/tsp_interaction` is private
/// to the `pyrefly` crate's own test build and unreachable from an external
/// bench binary). Talks over the plain main connection: the simplest
/// portable transport that still exercises the fix's dispatch path (see the
/// module doc for why the real extra connection adds nothing here).
struct MainConn {
    sender: crossbeam_channel::Sender<Message>,
    receiver: crossbeam_channel::Receiver<Message>,
    next_id: i32,
    /// The server exits on its own once every sender to its connection is
    /// dropped; this benchmark doesn't join it, matching how the existing
    /// PyTorch benches let their server threads wind down on drop rather
    /// than blocking teardown on a join.
    #[expect(
        dead_code,
        reason = "held only to keep the server thread alive until drop; never read"
    )]
    server_thread: JoinHandle<()>,
}

impl MainConn {
    fn next_request_id(&mut self) -> RequestId {
        self.next_id += 1;
        RequestId::from(self.next_id)
    }

    fn send_request(&mut self, method: &str, params: serde_json::Value) -> RequestId {
        let id = self.next_request_id();
        self.sender
            .send(Message::Request(Request {
                id: id.clone(),
                method: method.to_owned(),
                params,
                activity_key: None,
            }))
            .expect("TSP server should still be accepting requests on the main connection");
        id
    }

    fn send_notification(&self, method: &str, params: serde_json::Value) {
        self.sender
            .send(Message::Notification(Notification {
                method: method.to_owned(),
                params,
                activity_key: None,
            }))
            .expect("TSP server should still be accepting notifications on the main connection");
    }

    /// Receive messages until the `Response` matching `expected_id` arrives,
    /// skipping notifications and any response carrying a different id.
    /// Requests are always awaited before the next is sent, so a mismatched
    /// id should never occur in practice; matching against it (rather than
    /// returning the first response seen) guards against silently
    /// attributing a server-initiated request or a stray response to the
    /// wrong caller.
    fn recv_response(&self, expected_id: &RequestId) -> Response {
        let timeout = Duration::from_secs(25);
        loop {
            match self.receiver.recv_timeout(timeout) {
                Ok(Message::Response(response)) if response.id == *expected_id => {
                    return response;
                }
                Ok(_) => continue,
                Err(RecvTimeoutError::Timeout) => {
                    panic!(
                        "timed out after {timeout:?} awaiting response {expected_id} on the main connection"
                    )
                }
                Err(RecvTimeoutError::Disconnected) => {
                    panic!("main connection closed while awaiting response {expected_id}")
                }
            }
        }
    }

    fn get_snapshot(&mut self) -> i32 {
        let id = self.send_request("typeServer/getSnapshot", serde_json::json!(null));
        let response = self.recv_response(&id);
        serde_json::from_value(response.result.expect("getSnapshot should succeed"))
            .expect("getSnapshot should return an integer")
    }

    fn get_computed_type(
        &mut self,
        uri: &str,
        line: u32,
        character: u32,
        snapshot: i32,
    ) -> Response {
        let id = self.send_request(
            "typeServer/getComputedType",
            serde_json::json!({
                "arg": {
                    "uri": uri,
                    "range": {
                        "start": { "line": line, "character": character },
                        "end": { "line": line, "character": character },
                    },
                },
                "snapshot": snapshot,
            }),
        );
        self.recv_response(&id)
    }
}

/// Spawn a fresh TSP server on an in-memory connection with indexing
/// disabled: this benchmark queries one already-known position in one
/// already-known file, so it exercises none of the find-references-style
/// features indexing exists for, and leaving it on would add unrelated
/// background work. `TspArgs`'s fields are private to the `pyrefly` crate
/// (unlike the sibling `LspArgs`), so it is built through
/// `clap::Parser::parse_from`, the same way the real CLI entry point does.
fn spawn_server() -> MainConn {
    let ((conn_server, server_reader), (conn_client, _client_reader)) = Connection::memory();
    let receiver = conn_client.channel_receiver().clone();
    let args = TspArgs::parse_from([
        "tsp-bench",
        "--indexing-mode",
        "none",
        "--workspace-indexing-limit",
        "0",
        "--transport",
        "stdio",
    ]);
    let server_thread = thread::spawn(move || {
        run_tsp(
            conn_server,
            server_reader,
            args,
            &NoTelemetry,
            None,
            ThreadCount::Inline,
            None,
        )
        .expect("TSP server should run to completion");
    });
    MainConn {
        sender: conn_client.sender,
        receiver,
        next_id: 0,
        server_thread,
    }
}

fn initialize(main: &mut MainConn) {
    let id = main.send_request(
        "initialize",
        serde_json::json!({
            "rootPath": "/",
            "processId": std::process::id(),
            "capabilities": {
                "textDocument": { "publishDiagnostics": { "relatedInformation": true } },
            },
        }),
    );
    let response = main.recv_response(&id);
    assert!(
        response.error.is_none(),
        "initialize failed: {:?}",
        response.error
    );
    main.send_notification("initialized", serde_json::json!({}));
}

/// The `declaration.name` of a `typeServer/getComputedType` result, e.g.
/// `"Base"` for `result`. Used for the correctness check: however many
/// times solve work was reused, the answer must not change.
fn declaration_name(result: &serde_json::Value) -> &str {
    result
        .get("declaration")
        .and_then(|d| d.get("name"))
        .and_then(|n| n.as_str())
        .unwrap_or_else(|| panic!("expected declaration.name in getComputedType result: {result}"))
}

fn assert_resolves_to_base(response: &Response) {
    assert!(
        response.error.is_none(),
        "getComputedType failed: {:?}",
        response.error
    );
    let result = response
        .result
        .as_ref()
        .expect("getComputedType should return a result for an unopened module");
    assert_eq!(
        declaration_name(result),
        "Base",
        "result should resolve to Base[object]"
    );
}

/// Regression test for the solve-reuse fix in
/// https://github.com/facebook/pyrefly/pull/4698: one `getComputedType`
/// request, on a stable snapshot, repeated against one never-opened,
/// unreferenced, inference-heavy module. Server spawn, fixture generation,
/// and the one necessarily-cold request that solves and commits `heavy.py`
/// all happen outside Criterion's timing; each measured iteration is
/// exactly one logical request. Nothing in this benchmark ever edits a file
/// or sends a recheck-triggering notification, so the snapshot never moves
/// and there is nothing to invalidate the commit -- this is the fix's
/// steady state.
fn get_computed_type_unopened_cached(c: &mut Criterion) {
    let (_fixture, uri) = write_fixture();
    let mut main = spawn_server();
    initialize(&mut main);
    let snapshot = main.get_snapshot();

    let first = main.get_computed_type(&uri, QUERY_LINE, QUERY_CHARACTER, snapshot);
    assert_resolves_to_base(&first);

    c.bench_function("tsp/get_computed_type_unopened_cached", |b| {
        b.iter(|| {
            let response = main.get_computed_type(&uri, QUERY_LINE, QUERY_CHARACTER, snapshot);
            assert_resolves_to_base(&response);
        });
    });
}

criterion_group!(benches, get_computed_type_unopened_cached);
criterion_main!(benches);
