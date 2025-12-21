/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::time::Duration;
use std::time::Instant;

use anyhow::Error;

pub trait Telemetry {
    fn record_event<T>(
        &self,
        event: TelemetryEvent,
        result: Result<T, Error>,
    ) -> (Duration, Result<T, Error>);
}
pub struct NoTelemetry;

impl Telemetry for NoTelemetry {
    fn record_event<T>(
        &self,
        event: TelemetryEvent,
        result: Result<T, Error>,
    ) -> (Duration, Result<T, Error>) {
        event.finish(result)
    }
}

pub enum TelemetryEventKind {
    LspEvent(String),
}

pub struct TelemetryEvent {
    pub kind: TelemetryEventKind,
    pub queue: Duration,
    pub start: Instant,
    pub error: Option<Error>,
    pub validate: Option<Duration>,
    pub server_has_sourcedb: bool,
}

impl TelemetryEvent {
    pub fn new_dequeued(
        kind: TelemetryEventKind,
        enqueued_at: Instant,
        server_has_sourcedb: bool,
    ) -> Self {
        let start = Instant::now();
        let queue = start - enqueued_at;
        Self {
            kind,
            queue,
            start,
            error: None,
            validate: None,
            server_has_sourcedb,
        }
    }

    pub fn set_validate_duration(&mut self, duration: Duration) {
        self.validate = Some(duration);
    }

    pub fn finish<T>(&self, result: Result<T, Error>) -> (Duration, Result<T, Error>) {
        let finish = Instant::now();
        let process = finish - self.start;
        (process, result)
    }

    pub fn finish_and_record<T>(
        self,
        telemetry: &impl Telemetry,
        result: Result<T, Error>,
    ) -> (Duration, Result<T, Error>) {
        telemetry.record_event(self, result)
    }
}
