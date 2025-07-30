/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

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

pub enum LspEvent {
    // Part 1: Events that the server should try to handle first.
    /// Notify the server that recheck finishes, so server can revalidate all in-memory content
    /// based on the latest `State`.
    RecheckFinished,
    /// Inform the server that a request is cancelled.
    /// Server should know about this ASAP to avoid wasting time on cancelled requests.
    CancelRequest(RequestId),
    // Part 2: Events that can be queued in FIFO order and handled at a later time.
    DidOpenTextDocument(DidOpenTextDocumentParams),
    DidChangeTextDocument(DidChangeTextDocumentParams),
    DidCloseTextDocument(DidCloseTextDocumentParams),
    DidSaveTextDocument(DidSaveTextDocumentParams),
    DidChangeWatchedFiles(DidChangeWatchedFilesParams),
    DidChangeWorkspaceFolders(DidChangeWorkspaceFoldersParams),
    DidChangeConfiguration(DidChangeConfigurationParams),
    LspResponse(Response),
    LspRequest(Request),
    Exit,
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
            Self::RecheckFinished | Self::CancelRequest(_) => LspEventKind::Priority,
            Self::DidOpenTextDocument(_)
            | Self::DidChangeTextDocument(_)
            | Self::DidCloseTextDocument(_)
            | Self::DidSaveTextDocument(_)
            | Self::DidChangeWatchedFiles(_)
            | Self::DidChangeWorkspaceFolders(_)
            | Self::DidChangeConfiguration(_)
            | Self::LspResponse(_)
            | Self::Exit => LspEventKind::Mutation,
            Self::LspRequest(_) => LspEventKind::Query,
        }
    }
}

#[derive(Clone, Dupe)]
pub struct LspQueue(Arc<LspQueueInner>);

struct LspQueueInner {
    normal: (Sender<LspEvent>, Receiver<LspEvent>),
    priority: (Sender<LspEvent>, Receiver<LspEvent>),
}

impl LspQueue {
    pub fn new() -> Self {
        Self(Arc::new(LspQueueInner {
            normal: crossbeam_channel::unbounded(),
            priority: crossbeam_channel::unbounded(),
        }))
    }

    #[allow(clippy::result_large_err)]
    pub fn send(&self, x: LspEvent) -> Result<(), SendError<LspEvent>> {
        if x.kind() == LspEventKind::Priority {
            self.0.priority.0.send(x)
        } else {
            self.0.normal.0.send(x)
        }
    }

    pub fn recv(&self) -> Result<LspEvent, RecvError> {
        let mut event_receiver_selector = Select::new_biased();
        // Biased selector will pick the receiver with lower index over higher ones,
        // so we register priority_events_receiver first.
        let priority_receiver_index = event_receiver_selector.recv(&self.0.priority.1);
        let queued_events_receiver_index = event_receiver_selector.recv(&self.0.normal.1);

        let selected = event_receiver_selector.select();
        match selected.index() {
            i if i == priority_receiver_index => selected.recv(&self.0.priority.1),
            i if i == queued_events_receiver_index => selected.recv(&self.0.normal.1),
            _ => unreachable!(),
        }
    }
}
