/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Instant;

use crossbeam_channel::Receiver;
use crossbeam_channel::RecvError;
use crossbeam_channel::Select;
use crossbeam_channel::SendError;
use crossbeam_channel::Sender;
use dupe::Dupe;
use lsp_server::Request;
use lsp_server::RequestId;
use lsp_server::Response;
use lsp_types::DidChangeConfigurationParams;
use lsp_types::DidChangeTextDocumentParams;
use lsp_types::DidChangeWatchedFilesParams;
use lsp_types::DidChangeWorkspaceFoldersParams;
use lsp_types::DidCloseTextDocumentParams;
use lsp_types::DidOpenTextDocumentParams;
use lsp_types::DidSaveTextDocumentParams;
use tracing::debug;
use tracing::info;

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
    /// Inform the server that the given configs' find caches are now invalid, and
    /// that a new type check must occur.
    InvalidateConfigFind,
    // Part 2: Events that can be queued in FIFO order and handled at a later time.
    DidOpenTextDocument(DidOpenTextDocumentParams),
    DidChangeTextDocument(DidChangeTextDocumentParams),
    DidCloseTextDocument(DidCloseTextDocumentParams),
    DidSaveTextDocument(DidSaveTextDocumentParams),
    DidChangeWatchedFiles(DidChangeWatchedFilesParams),
    DidChangeWorkspaceFolders(DidChangeWorkspaceFoldersParams),
    DidChangeConfiguration(DidChangeConfigurationParams),
    DidOpenNotebookDocument(DidOpenNotebookDocumentParams),
    DidCloseNotebookDocument(DidCloseNotebookDocumentParams),
    DidChangeNotebookDocument(DidChangeNotebookDocumentParams),
    DidSaveNotebookDocument(DidSaveNotebookDocumentParams),
    LspResponse(Response),
    LspRequest(Request),
    Exit,
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
            Self::DidChangeWatchedFiles(_) => "DidChangeWatchedFiles".to_owned(),
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
    Mutation,
    Query,
}

impl LspEvent {
    fn kind(&self) -> LspEventKind {
        match self {
            Self::RecheckFinished | Self::CancelRequest(_) | Self::InvalidateConfigFind => {
                LspEventKind::Priority
            }
            Self::DidOpenTextDocument(_)
            | Self::DidChangeTextDocument(_)
            | Self::DidCloseTextDocument(_)
            | Self::DidSaveTextDocument(_)
            | Self::DidChangeWatchedFiles(_)
            | Self::DidChangeWorkspaceFolders(_)
            | Self::DidChangeConfiguration(_)
            | Self::LspResponse(_)
            | Self::DidOpenNotebookDocument(_)
            | Self::DidCloseNotebookDocument(_)
            | Self::DidSaveNotebookDocument(_)
            | Self::DidChangeNotebookDocument(_)
            | Self::Exit => LspEventKind::Mutation,
            Self::LspRequest(_) => LspEventKind::Query,
        }
    }
}

#[derive(Clone, Dupe)]
pub struct LspQueue(Arc<LspQueueInner>);

struct LspQueueInner {
    /// The next id to use for a new event.
    id: AtomicUsize,
    /// The index of the last event we are aware of that is a mutation. 0 = unknown.
    last_mutation: AtomicUsize,
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
        Self(Arc::new(LspQueueInner {
            id: AtomicUsize::new(1),
            last_mutation: AtomicUsize::new(0),
            normal: crossbeam_channel::unbounded(),
            priority: crossbeam_channel::unbounded(),
        }))
    }

    #[allow(clippy::result_large_err)]
    pub fn send(&self, x: LspEvent) -> Result<(), SendError<LspEvent>> {
        let kind = x.kind();
        let id = self.0.id.fetch_add(1, Ordering::Relaxed);
        if kind == LspEventKind::Mutation {
            // This is gently dubious, as we might race condition and it might not really be the last
            // mutation. But it's good enough for now.
            self.0.last_mutation.store(id, Ordering::Relaxed);
        }
        if kind == LspEventKind::Priority {
            self.0
                .priority
                .0
                .send((id, x, Instant::now()))
                .map_err(|x| SendError(x.0.1))
        } else {
            self.0
                .normal
                .0
                .send((id, x, Instant::now()))
                .map_err(|x| SendError(x.0.1))
        }
    }

    /// Return a bool indicating whether there is a subsequent mutation event in the queue,
    /// and the event itself.
    ///
    /// Due to race conditions, we might say false when there is a subsequent mutation,
    /// but we will never say true when there is not.
    pub fn recv(&self) -> Result<(bool, LspEvent, Instant), RecvError> {
        let mut event_receiver_selector = Select::new_biased();
        // Biased selector will pick the receiver with lower index over higher ones,
        // so we register priority_events_receiver first.
        let priority_receiver_index = event_receiver_selector.recv(&self.0.priority.1);
        let queued_events_receiver_index = event_receiver_selector.recv(&self.0.normal.1);

        let selected = event_receiver_selector.select();
        let (id, x, queue_time) = match selected.index() {
            i if i == priority_receiver_index => selected.recv(&self.0.priority.1)?,
            i if i == queued_events_receiver_index => selected.recv(&self.0.normal.1)?,
            _ => unreachable!(),
        };
        let mut last_mutation = self.0.last_mutation.load(Ordering::Relaxed);
        if id == last_mutation {
            self.0.last_mutation.store(0, Ordering::Relaxed);
            last_mutation = 0;
        }
        Ok((last_mutation != 0, x, queue_time))
    }
}

pub struct HeavyTask(Box<dyn FnOnce() + Send + Sync + 'static>, Instant);

struct HeavyTaskQueueInner {
    task_sender: Sender<HeavyTask>,
    task_receiver: Receiver<HeavyTask>,
    stop_sender: Sender<()>,
    stop_receiver: Receiver<()>,
    queue_name: String,
}

/// A queue for heavy tasks that should be run in the background thread.
#[derive(Clone, Dupe)]
pub struct HeavyTaskQueue(Arc<HeavyTaskQueueInner>);

impl HeavyTaskQueue {
    pub fn new(queue_name: &str) -> Self {
        let queue_name = queue_name.to_owned();
        let (task_sender, task_receiver) = crossbeam_channel::unbounded();
        let (stop_sender, stop_receiver) = crossbeam_channel::unbounded();
        Self(Arc::new(HeavyTaskQueueInner {
            task_sender,
            task_receiver,
            stop_sender,
            stop_receiver,
            queue_name,
        }))
    }

    pub fn queue_task(&self, f: Box<dyn FnOnce() + Send + Sync + 'static>) {
        self.0
            .task_sender
            .send(HeavyTask(f, Instant::now()))
            .expect("Failed to queue heavy task");
        debug!("Enqueued task on {} heavy task queue", self.0.queue_name);
    }

    pub fn run_until_stopped(&self) {
        let mut receiver_selector = Select::new_biased();
        // Biased selector will pick the receiver with lower index over higher ones,
        // so we register priority_events_receiver first.
        let stop_receiver_index = receiver_selector.recv(&self.0.stop_receiver);
        let task_receiver_index = receiver_selector.recv(&self.0.task_receiver);
        loop {
            let selected = receiver_selector.select();
            match selected.index() {
                i if i == stop_receiver_index => {
                    selected
                        .recv(&self.0.stop_receiver)
                        .expect("Failed to receive stop signal");
                    return;
                }
                i if i == task_receiver_index => {
                    let task = selected
                        .recv(&self.0.task_receiver)
                        .expect("Failed to receive heavy task");
                    let queue_duration = task.1.elapsed().as_secs_f32();
                    debug!("Dequeued task on {} heavy task queue", self.0.queue_name);
                    let task_start = Instant::now();
                    (task.0)();
                    info!(
                        "Ran task on {} heavy task queue. Queue time: {:.2}, task time: {:.2}",
                        self.0.queue_name,
                        queue_duration,
                        task_start.elapsed().as_secs_f32()
                    );
                }
                _ => unreachable!(),
            };
        }
    }

    /// Make `run_until_stopped` exit after finishing the current task.
    pub fn stop(&self) {
        self.0
            .stop_sender
            .send(())
            .expect("Failed to stop the queue");
    }
}
