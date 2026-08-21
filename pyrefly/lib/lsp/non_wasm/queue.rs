/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;

use crossbeam_channel::Receiver;
use crossbeam_channel::RecvError;
use crossbeam_channel::Select;
use crossbeam_channel::SendError;
use crossbeam_channel::Sender;
use lsp_server::RequestId;
use lsp_types::DidChangeConfigurationParams;
use lsp_types::DidChangeTextDocumentParams;
use lsp_types::DidChangeWorkspaceFoldersParams;
use lsp_types::DidCloseTextDocumentParams;
use lsp_types::DidOpenTextDocumentParams;
use lsp_types::DidSaveTextDocumentParams;
use pyrefly_util::lock::Mutex;
use pyrefly_util::telemetry::QueueName;
use pyrefly_util::telemetry::Telemetry;
use pyrefly_util::telemetry::TelemetryEvent;
use pyrefly_util::telemetry::TelemetryEventKind;
use tracing::debug;
use tracing::info;

use crate::lsp::non_wasm::protocol::Request;
use crate::lsp::non_wasm::protocol::Response;
use crate::lsp::non_wasm::server::Server;
use crate::lsp::wasm::notebook::DidChangeNotebookDocumentParams;
use crate::lsp::wasm::notebook::DidCloseNotebookDocumentParams;
use crate::lsp::wasm::notebook::DidOpenNotebookDocumentParams;
use crate::lsp::wasm::notebook::DidSaveNotebookDocumentParams;

pub enum LspEvent {
    // Part 1: Events that the server should try to handle first.
    /// Notify the server that recheck finishes, so server can revalidate all in-memory content
    /// based on the latest `State`. The included config files are configs whose find
    /// caches should be invalidated. on the next run.
    RecheckFinished,
    /// Inform the server that a request is cancelled.
    /// Server should know about this ASAP to avoid wasting time on cancelled requests.
    CancelRequest(RequestId),
    // Part 2: Events that can be queued in FIFO order and handled at a later time.
    DidOpenTextDocument(DidOpenTextDocumentParams),
    DidChangeTextDocument(DidChangeTextDocumentParams),
    DidCloseTextDocument(DidCloseTextDocumentParams),
    DidSaveTextDocument(DidSaveTextDocumentParams),
    DrainWatchedFileChanges,
    DidChangeWorkspaceFolders(DidChangeWorkspaceFoldersParams),
    DidChangeConfiguration(DidChangeConfigurationParams),
    DidOpenNotebookDocument(DidOpenNotebookDocumentParams),
    DidCloseNotebookDocument(DidCloseNotebookDocumentParams),
    DidChangeNotebookDocument(DidChangeNotebookDocumentParams),
    DidSaveNotebookDocument(DidSaveNotebookDocumentParams),
    /// Inform the server that some configs' find caches are now invalid (stored in
    /// `server.invalidated_configs`), and that a new type check must occur.
    InvalidateConfigFind,
    LspResponse(Response),
    LspRequest(Request),
    Exit,
}

/// An LSP event with the identity and enqueue time from its first queue entry.
pub struct QueuedEvent {
    id: usize,
    event: LspEvent,
    enqueued_at: Instant,
}

impl QueuedEvent {
    pub(crate) fn from_parts(id: usize, event: LspEvent, enqueued_at: Instant) -> Self {
        Self {
            id,
            event,
            enqueued_at,
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn event(&self) -> &LspEvent {
        &self.event
    }

    pub fn enqueued_at(&self) -> Instant {
        self.enqueued_at
    }

    pub fn describe(&self) -> String {
        self.event.describe()
    }

    pub fn into_parts(self) -> (usize, LspEvent, Instant) {
        (self.id, self.event, self.enqueued_at)
    }
}

impl LspEvent {
    pub fn describe(&self) -> String {
        match self {
            Self::RecheckFinished => "RecheckFinished".to_owned(),
            Self::CancelRequest(_) => "CancelRequest".to_owned(),
            Self::InvalidateConfigFind => "InvalidateConfigFind".to_owned(),
            Self::DidOpenTextDocument(_) => "DidOpenTextDocument".to_owned(),
            Self::DidChangeTextDocument(_) => "DidChangeTextDocument".to_owned(),
            Self::DidCloseTextDocument(_) => "DidCloseTextDocument".to_owned(),
            Self::DidSaveTextDocument(_) => "DidSaveTextDocument".to_owned(),
            Self::DrainWatchedFileChanges => "DidChangeWatchedFiles".to_owned(),
            Self::DidChangeWorkspaceFolders(_) => "DidChangeWorkspaceFolders".to_owned(),
            Self::DidChangeConfiguration(_) => "DidChangeConfiguration".to_owned(),
            Self::DidOpenNotebookDocument(_) => "DidOpenNotebookDocument".to_owned(),
            Self::DidCloseNotebookDocument(_) => "DidCloseNotebookDocument".to_owned(),
            Self::DidChangeNotebookDocument(_) => "DidChangeNotebookDocument".to_owned(),
            Self::DidSaveNotebookDocument(_) => "DidSaveNotebookDocument".to_owned(),
            Self::LspResponse(_) => "LspResponse".to_owned(),
            Self::LspRequest(request) => format!("LspRequest({})", request.method,),
            Self::Exit => "Exit".to_owned(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LspEventKind {
    Priority,
    /// An event that makes work triggered by preceding mutations redundant.
    Mutation,
    /// A normal-priority event that does not count as a mutation for coalescing.
    Query,
}

impl LspEvent {
    /// Whether this event is an in-editor edit to a document's content, i.e. the
    /// kind of change that should reset the inlay-hint debounce window. Opens,
    /// saves, closes, config/workspace-folder changes, watched-file drains, and
    /// client responses are deliberately excluded: they aren't the user typing,
    /// so they must not hold inlay hints back (otherwise background file churn
    /// could defer hints indefinitely while the user sits idle).
    fn is_edit(&self) -> bool {
        matches!(
            self,
            Self::DidChangeTextDocument(_) | Self::DidChangeNotebookDocument(_)
        )
    }

    fn kind(&self) -> LspEventKind {
        match self {
            Self::RecheckFinished | Self::CancelRequest(_) => LspEventKind::Priority,
            Self::DidOpenTextDocument(_)
            | Self::DidChangeTextDocument(_)
            | Self::DidCloseTextDocument(_)
            | Self::DidSaveTextDocument(_)
            | Self::DrainWatchedFileChanges
            | Self::DidChangeWorkspaceFolders(_)
            | Self::DidChangeConfiguration(_)
            | Self::DidOpenNotebookDocument(_)
            | Self::DidCloseNotebookDocument(_)
            | Self::DidSaveNotebookDocument(_)
            | Self::DidChangeNotebookDocument(_)
            | Self::InvalidateConfigFind
            | Self::Exit => LspEventKind::Mutation,
            Self::LspResponse(_) | Self::LspRequest(_) => LspEventKind::Query,
        }
    }
}

pub struct LspQueue {
    /// The next id to use for a new event.
    id: AtomicUsize,
    /// The index of the last queued mutation. `recv` uses this to tell handlers
    /// whether work triggered by an earlier mutation can be deferred. 0 = unknown.
    last_mutation: AtomicUsize,
    /// When the most recent document edit was enqueued, or `None` if no edit has
    /// happened yet. Used to debounce queries (e.g. inlay hints) that shouldn't
    /// recompute on every keystroke. Only genuine edits bump this (see
    /// [`LspEvent::is_edit`]), not every mutation, so non-typing activity doesn't
    /// hold debounced queries back. `None` means there's nothing to debounce
    /// against, so queries run immediately (e.g. right after server startup).
    last_edit_time: Mutex<Option<Instant>>,
    /// A single query request (an inlay hint) held back to debounce it, paired
    /// with the instant it becomes ready. `recv` delivers it once that instant
    /// passes if nothing else arrives first. The entry keeps the enqueue time it
    /// was first given, so queue-latency metrics reflect the full debounce wait.
    /// Only one is held at a time; see [`LspQueue::send_delayed`].
    delayed: Mutex<Option<(QueuedEvent, Instant)>>,
    normal: (
        Sender<(usize, LspEvent, Instant)>,
        Receiver<(usize, LspEvent, Instant)>,
    ),
    priority: (
        Sender<(usize, LspEvent, Instant)>,
        Receiver<(usize, LspEvent, Instant)>,
    ),
}

impl LspQueue {
    pub fn new() -> Self {
        Self {
            id: AtomicUsize::new(1),
            last_mutation: AtomicUsize::new(0),
            last_edit_time: Mutex::new(None),
            delayed: Mutex::new(None),
            normal: crossbeam_channel::unbounded(),
            priority: crossbeam_channel::unbounded(),
        }
    }

    #[allow(clippy::result_large_err)]
    pub fn send(&self, x: LspEvent) -> Result<(), SendError<LspEvent>> {
        let kind = x.kind();
        let id = self.id.fetch_add(1, Ordering::Relaxed);
        if kind == LspEventKind::Mutation {
            // This is gently dubious, as we might race condition and it might not really be the last
            // mutation. But it's good enough for now.
            self.last_mutation.store(id, Ordering::Relaxed);
        }
        if x.is_edit() {
            *self.last_edit_time.lock() = Some(Instant::now());
        }
        if kind == LspEventKind::Priority {
            self.priority
                .0
                .send((id, x, Instant::now()))
                .map_err(|x| SendError(x.0.1))
        } else {
            self.normal
                .0
                .send((id, x, Instant::now()))
                .map_err(|x| SendError(x.0.1))
        }
    }

    /// Return the next event with its original queue metadata.
    pub fn recv(&self) -> Result<QueuedEvent, RecvError> {
        // If a delayed request is held, wake once its window expires so we can
        // deliver it. The slot is only written by the same thread that calls
        // `recv` (via `send_delayed` during event processing), so it can't change
        // while we block here.
        let deadline = self.delayed.lock().as_ref().map(|(_, ready_at)| *ready_at);

        let mut event_receiver_selector = Select::new_biased();
        // Biased selector will pick the receiver with lower index over higher ones,
        // so we register priority_events_receiver first.
        let priority_receiver_index = event_receiver_selector.recv(&self.priority.1);
        let queued_events_receiver_index = event_receiver_selector.recv(&self.normal.1);

        let selected = match deadline {
            Some(deadline) => match event_receiver_selector.select_deadline(deadline) {
                Ok(selected) => selected,
                Err(_) => {
                    let (event, _ready_at) = self
                        .delayed
                        .lock()
                        .take()
                        .expect("a deadline is only set while a delayed request is held");
                    return Ok(event);
                }
            },
            None => event_receiver_selector.select(),
        };
        let (id, x, queue_time) = match selected.index() {
            i if i == priority_receiver_index => selected.recv(&self.priority.1)?,
            i if i == queued_events_receiver_index => selected.recv(&self.normal.1)?,
            _ => unreachable!(),
        };
        let last_mutation = self.last_mutation.load(Ordering::Relaxed);
        if id == last_mutation {
            self.last_mutation.store(0, Ordering::Relaxed);
        }
        Ok(QueuedEvent::from_parts(id, x, queue_time))
    }

    /// Return whether an observed mutation follows `event` in the live queue.
    /// A concurrent send can cause a false negative, but never a false positive.
    pub fn has_subsequent_mutation(&self, event: &QueuedEvent) -> bool {
        let last_mutation = self.last_mutation.load(Ordering::Relaxed);
        last_mutation > event.id
    }

    /// Hold `event` until `ready_at`, after which `recv` delivers it. This
    /// debounces queries (inlay hints) that shouldn't recompute on every
    /// keystroke. The event keeps its original enqueue time so delivery metrics
    /// include all holding periods. At most one event is held; a second call
    /// displaces and returns the previous one so the caller can respond to the
    /// superseded request.
    pub fn send_delayed(&self, event: QueuedEvent, ready_at: Instant) -> Option<QueuedEvent> {
        self.delayed
            .lock()
            .replace((event, ready_at))
            .map(|(event, ..)| event)
    }

    /// How long since the most recent document edit was enqueued, or `None` if
    /// no edit has happened yet.
    pub fn time_since_last_edit(&self) -> Option<Duration> {
        self.last_edit_time.lock().map(|t| t.elapsed())
    }
}

pub struct HeavyTask(
    Box<dyn FnOnce(&Server, &dyn Telemetry, &mut TelemetryEvent) + Send + Sync + 'static>,
);

impl HeavyTask {
    fn run(self, server: &Server, telemetry: &dyn Telemetry, telemetry_event: &mut TelemetryEvent) {
        self.0(server, telemetry, telemetry_event);
    }
}

/// A queue for heavy tasks that should be run in the background thread.
pub struct HeavyTaskQueue {
    task_sender: Sender<(HeavyTask, TelemetryEventKind, Instant)>,
    task_receiver: Receiver<(HeavyTask, TelemetryEventKind, Instant)>,
    stop_sender: Sender<()>,
    stop_receiver: Receiver<()>,
    queue_name: QueueName,
    next_task_id: AtomicUsize,
}

impl HeavyTaskQueue {
    pub fn new(queue_name: QueueName) -> Self {
        let (task_sender, task_receiver) = crossbeam_channel::unbounded();
        let (stop_sender, stop_receiver) = crossbeam_channel::unbounded();
        Self {
            task_sender,
            task_receiver,
            stop_sender,
            stop_receiver,
            queue_name,
            next_task_id: AtomicUsize::new(1),
        }
    }

    pub fn queue_task(
        &self,
        kind: TelemetryEventKind,
        f: Box<dyn FnOnce(&Server, &dyn Telemetry, &mut TelemetryEvent) + Send + Sync + 'static>,
    ) {
        self.task_sender
            .send((HeavyTask(f), kind, Instant::now()))
            .expect("Failed to queue heavy task");
        debug!("Enqueued task on {} heavy task queue", self.queue_name);
    }

    pub fn run_until_stopped(&self, server: &Server, telemetry: &dyn Telemetry) {
        let mut receiver_selector = Select::new_biased();
        // Biased selector will pick the receiver with lower index over higher ones,
        // so we register priority_events_receiver first.
        let stop_receiver_index = receiver_selector.recv(&self.stop_receiver);
        let task_receiver_index = receiver_selector.recv(&self.task_receiver);
        loop {
            let selected = receiver_selector.select();
            match selected.index() {
                i if i == stop_receiver_index => {
                    selected
                        .recv(&self.stop_receiver)
                        .expect("Failed to receive stop signal");
                    return;
                }
                i if i == task_receiver_index => {
                    let (task, kind, enqueued) = selected
                        .recv(&self.task_receiver)
                        .expect("Failed to receive heavy task");
                    debug!("Dequeued task on {} heavy task queue", self.queue_name);
                    let task_id = self.next_task_id.fetch_add(1, Ordering::Relaxed);
                    let (mut telemetry_event, queue_duration) = TelemetryEvent::new_dequeued(
                        kind,
                        enqueued,
                        server.telemetry_state(),
                        self.queue_name,
                        task_id,
                    );
                    task.run(server, telemetry, &mut telemetry_event);
                    let process_duration = telemetry_event.finish_and_record(telemetry, None);
                    info!(
                        "Ran task on {} heavy task queue. Queue time: {:.2}, task time: {:.2}",
                        self.queue_name,
                        queue_duration.as_secs_f32(),
                        process_duration.as_secs_f32()
                    );
                }
                _ => unreachable!(),
            };
        }
    }

    /// Make `run_until_stopped` exit after finishing the current task.
    pub fn stop(&self) {
        self.stop_sender.send(()).expect("Failed to stop the queue");
    }
}

#[cfg(test)]
mod tests {
    use lsp_server::RequestId;
    use lsp_types::TextDocumentItem;
    use lsp_types::Url;
    use lsp_types::VersionedTextDocumentIdentifier;

    use super::*;

    fn edit() -> LspEvent {
        LspEvent::DidChangeTextDocument(DidChangeTextDocumentParams {
            text_document: VersionedTextDocumentIdentifier {
                uri: Url::parse("file:///test.py").unwrap(),
                version: 1,
            },
            content_changes: Vec::new(),
        })
    }

    fn non_edit() -> LspEvent {
        LspEvent::DidChangeConfiguration(DidChangeConfigurationParams {
            settings: serde_json::Value::Null,
        })
    }

    fn request() -> LspEvent {
        LspEvent::LspRequest(Request {
            id: RequestId::from(1),
            method: "test/query".to_owned(),
            params: serde_json::Value::Null,
            activity_key: None,
        })
    }

    #[test]
    fn test_delayed_request_preserves_enqueue_time() {
        let queue = LspQueue::new();
        queue.send(request()).unwrap();
        let queued = queue.recv().unwrap();
        let enqueued_at = queued.enqueued_at();
        let (id, event, _) = queued.into_parts();
        let LspEvent::LspRequest(request) = event else {
            panic!("expected a request");
        };

        queue.send_delayed(
            QueuedEvent::from_parts(id, LspEvent::LspRequest(request), enqueued_at),
            Instant::now(),
        );
        let delivered_enqueue_time = queue.recv().unwrap().enqueued_at();

        assert_eq!(delivered_enqueue_time, enqueued_at);
    }

    #[test]
    fn test_time_since_last_edit_resets_only_on_edit() {
        let queue = LspQueue::new();

        // No edit has happened yet, so there is nothing to debounce against.
        assert_eq!(
            queue.time_since_last_edit(),
            None,
            "the debounce clock must be unset until the first edit"
        );

        queue.send(edit()).unwrap();

        std::thread::sleep(Duration::from_millis(30));
        let elapsed = queue.time_since_last_edit().expect("an edit was enqueued");
        assert!(
            elapsed >= Duration::from_millis(25),
            "clock should grow between edits, got {elapsed:?}"
        );

        // A non-edit mutation (config change, save, watched-file drain, ...) must
        // NOT reset the debounce clock, otherwise background activity could hold
        // inlay hints back while the user is idle.
        queue.send(non_edit()).unwrap();
        assert!(
            queue.time_since_last_edit().expect("still have an edit") >= elapsed,
            "a non-edit event must not reset the debounce clock"
        );

        // A fresh edit resets the clock back towards zero.
        queue.send(edit()).unwrap();
        assert!(
            queue.time_since_last_edit().expect("edit was enqueued") < elapsed,
            "a new edit should reset the debounce clock"
        );
    }

    #[test]
    fn test_lsp_response_does_not_supersede_did_open() {
        let queue = LspQueue::new();
        queue
            .send(LspEvent::DidOpenTextDocument(DidOpenTextDocumentParams {
                text_document: TextDocumentItem {
                    uri: Url::parse("file:///test.py").unwrap(),
                    language_id: "python".to_owned(),
                    version: 1,
                    text: "x: int = 'not an int'".to_owned(),
                },
            }))
            .unwrap();
        let response_id = RequestId::from(3);
        queue
            .send(LspEvent::LspResponse(Response::new_ok(
                response_id.clone(),
                serde_json::json!([{}]),
            )))
            .unwrap();

        let queued = queue.recv().unwrap();
        assert!(matches!(queued.event(), LspEvent::DidOpenTextDocument(_)));
        assert!(
            !queue.has_subsequent_mutation(&queued),
            "a client response must not suppress didOpen validation"
        );

        let queued = queue.recv().unwrap();
        assert!(!queue.has_subsequent_mutation(&queued));
        assert!(matches!(
            queued.event(),
            LspEvent::LspResponse(Response { id, .. }) if *id == response_id
        ));
    }
}
