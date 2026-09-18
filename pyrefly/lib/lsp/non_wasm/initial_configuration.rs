/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::VecDeque;
use std::mem::take;
use std::time::Duration;
use std::time::Instant;

use crossbeam_channel::RecvError;
use crossbeam_channel::TryRecvError;
use lsp_server::RequestId;
use tracing::info;
use tracing::warn;

use crate::lsp::non_wasm::queue::LspEvent;
use crate::lsp::non_wasm::queue::LspQueue;
use crate::lsp::non_wasm::queue::QueuedEvent;

/// How long to wait for the client to answer the initial `workspace/configuration`
/// request before continuing with default settings.
const RESPONSE_DEADLINE: Duration = Duration::from_secs(10);

#[derive(Default)]
struct DeferredEvents {
    events: VecDeque<QueuedEvent>,
    last_mutation_id: Option<usize>,
}

impl DeferredEvents {
    fn push(&mut self, event: QueuedEvent) {
        if event.is_mutation() {
            self.last_mutation_id = Some(event.id());
        }
        self.events.push_back(event);
    }

    fn pop_front(&mut self) -> Option<QueuedEvent> {
        self.events.pop_front()
    }

    fn has_subsequent_mutation(&self, event: &QueuedEvent) -> bool {
        self.last_mutation_id.is_some_and(|id| id > event.id())
    }
}

enum State {
    Waiting {
        request_id: RequestId,
        started_at: Instant,
        deferred: DeferredEvents,
    },
    Applying {
        started_at: Instant,
        deferred: DeferredEvents,
    },
    Draining(DeferredEvents),
    Ready,
}

/// Delays ordinary events until the initial workspace settings are committed.
pub struct InitialConfigurationEvents<'a> {
    queue: &'a LspQueue,
    state: State,
}

impl<'a> InitialConfigurationEvents<'a> {
    pub fn new(queue: &'a LspQueue, request: Option<(RequestId, Instant)>) -> Self {
        let state = request.map_or(State::Ready, |(request_id, started_at)| State::Waiting {
            request_id,
            started_at,
            deferred: DeferredEvents::default(),
        });
        Self { queue, state }
    }

    pub fn recv(&mut self) -> Result<QueuedEvent, RecvError> {
        loop {
            let event = match &mut self.state {
                State::Draining(deferred) => match self.queue.try_recv_priority() {
                    Ok(event) => event,
                    Err(TryRecvError::Empty) => match deferred.pop_front() {
                        Some(event) => event,
                        None => {
                            self.state = State::Ready;
                            let _ = self
                                .queue
                                .send(LspEvent::InitialWorkspaceConfigurationDrained);
                            continue;
                        }
                    },
                    Err(TryRecvError::Disconnected) => {
                        unreachable!("LspQueue owns its priority sender")
                    }
                },
                State::Waiting {
                    started_at,
                    deferred,
                    ..
                } => match self
                    .queue
                    .recv_until(Some(*started_at + RESPONSE_DEADLINE))?
                {
                    Some(event) => event,
                    None => {
                        warn!(
                            "No initial workspace configuration response after {:.2}s, continuing with defaults",
                            started_at.elapsed().as_secs_f32()
                        );
                        self.state = State::Draining(take(deferred));
                        continue;
                    }
                },
                State::Applying { .. } | State::Ready => self.queue.recv()?,
            };

            match &mut self.state {
                State::Waiting {
                    request_id,
                    started_at,
                    deferred,
                } if matches!(event.event(), LspEvent::LspResponse(response) if response.id == *request_id) =>
                {
                    let started_at = *started_at;
                    let deferred = take(deferred);
                    self.state = State::Applying {
                        started_at,
                        deferred,
                    };
                    return Ok(event);
                }
                State::Applying {
                    started_at,
                    deferred,
                } if matches!(
                    event.event(),
                    LspEvent::InitialWorkspaceConfigurationApplied
                ) =>
                {
                    let started_at = *started_at;
                    let deferred = take(deferred);
                    info!(
                        "Committed initial workspace configuration after {:.2}s",
                        started_at.elapsed().as_secs_f32()
                    );
                    self.state = State::Draining(deferred);
                    continue;
                }
                State::Waiting { deferred, .. } | State::Applying { deferred, .. }
                    if !can_process_while_waiting(event.event()) =>
                {
                    deferred.push(event);
                    continue;
                }
                State::Waiting { .. }
                | State::Applying { .. }
                | State::Draining(_)
                | State::Ready => {}
            }
            return Ok(event);
        }
    }

    pub fn has_subsequent_mutation(&self, event: &QueuedEvent) -> bool {
        let deferred = match &self.state {
            State::Draining(deferred) => deferred.has_subsequent_mutation(event),
            State::Waiting { .. } | State::Applying { .. } | State::Ready => false,
        };
        deferred || self.queue.has_subsequent_mutation(event)
    }
}

fn can_process_while_waiting(event: &LspEvent) -> bool {
    matches!(
        event,
        LspEvent::RecheckFinished
            | LspEvent::CancelRequest(_)
            | LspEvent::InitialWorkspaceConfigurationApplied
            | LspEvent::LspResponse(_)
            | LspEvent::Exit
    )
}

#[cfg(test)]
mod tests {
    use lsp_types::DidChangeTextDocumentParams;
    use lsp_types::TextDocumentIdentifier;
    use lsp_types::Uri;
    use lsp_types::VersionedTextDocumentIdentifier;

    use super::*;
    use crate::lsp::non_wasm::protocol::Request;
    use crate::lsp::non_wasm::protocol::Response;

    const CONFIGURATION_REQUEST: i32 = 0;

    fn request(id: i32) -> LspEvent {
        LspEvent::LspRequest(Request {
            id: RequestId::from(id),
            method: "test/query".to_owned(),
            params: serde_json::Value::Null,
            activity_key: None,
        })
    }

    fn edit() -> LspEvent {
        LspEvent::DidChangeTextDocument(DidChangeTextDocumentParams {
            text_document: VersionedTextDocumentIdentifier {
                text_document_identifier: TextDocumentIdentifier {
                    uri: Uri::parse("file:///test.py").unwrap(),
                },
                version: 1,
            },
            content_changes: Vec::new(),
        })
    }

    fn configuration_response() -> LspEvent {
        LspEvent::LspResponse(Response::new_ok(RequestId::from(CONFIGURATION_REQUEST), ()))
    }

    fn waiting_events(queue: &LspQueue) -> InitialConfigurationEvents<'_> {
        InitialConfigurationEvents::new(
            queue,
            Some((RequestId::from(CONFIGURATION_REQUEST), Instant::now())),
        )
    }

    fn request_id(event: &QueuedEvent) -> Option<RequestId> {
        match event.event() {
            LspEvent::LspRequest(request) => Some(request.id.clone()),
            _ => None,
        }
    }

    #[test]
    fn test_held_events_replay_in_order_after_commit() {
        let queue = LspQueue::new();
        let mut events = waiting_events(&queue);
        queue.send(request(1)).unwrap();
        queue.send(edit()).unwrap();
        queue.send(configuration_response()).unwrap();
        queue.send(request(2)).unwrap();

        assert!(matches!(
            events.recv().unwrap().event(),
            LspEvent::LspResponse(_)
        ));
        // The server sends this after the configuration response commits.
        queue
            .send(LspEvent::InitialWorkspaceConfigurationApplied)
            .unwrap();

        let first = events.recv().unwrap();
        assert_eq!(request_id(&first), Some(RequestId::from(1)));
        assert!(
            events.has_subsequent_mutation(&first),
            "a held edit must make the earlier held request stale"
        );
        assert!(matches!(
            events.recv().unwrap().event(),
            LspEvent::DidChangeTextDocument(_)
        ));
        assert!(
            matches!(
                events.recv().unwrap().event(),
                LspEvent::InitialWorkspaceConfigurationDrained
            ),
            "the drain must finish before any event that was not held"
        );
        assert_eq!(
            request_id(&events.recv().unwrap()),
            Some(RequestId::from(2))
        );
    }

    #[test]
    fn test_cancellation_passes_while_waiting() {
        let queue = LspQueue::new();
        let mut events = waiting_events(&queue);
        queue.send(request(1)).unwrap();
        queue
            .send(LspEvent::CancelRequest(RequestId::from(1)))
            .unwrap();
        queue.send(configuration_response()).unwrap();

        assert!(matches!(
            events.recv().unwrap().event(),
            LspEvent::CancelRequest(_)
        ));
        assert!(matches!(
            events.recv().unwrap().event(),
            LspEvent::LspResponse(_)
        ));
    }
}
