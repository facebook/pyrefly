/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::time::Duration;
use std::time::Instant;

use anyhow::Error;
use lsp_types::Url;
use uuid::Uuid;

pub trait Telemetry: Send + Sync {
    fn record_event(&self, event: TelemetryEvent, process: Duration, error: Option<&Error>);
}
pub struct NoTelemetry;

impl Telemetry for NoTelemetry {
    fn record_event(&self, _event: TelemetryEvent, _process: Duration, _error: Option<&Error>) {}
}

pub enum TelemetryEventKind {
    LspEvent(String),
    Invalidate,
    InvalidateConfig,
    InvalidateOnClose,
    PopulateProjectFiles,
    PopulateWorkspaceFiles,
    SourceDbRebuild,
    FindFromDefinition,
}

pub struct TelemetryEvent {
    pub kind: TelemetryEventKind,
    pub queue: Duration,
    pub start: Instant,
    pub error: Option<Error>,
    pub invalidate: Option<Duration>,
    pub validate: Option<Duration>,
    pub transaction_stats: Option<TelemetryTransactionStats>,
    pub server_state: TelemetryServerState,
    pub file_stats: Option<TelemetryFileStats>,
    pub task_stats: Option<TelemetryTaskStats>,
    pub sourcedb_rebuild_stats: Option<TelemetrySourceDbRebuildStats>,
}

pub struct TelemetryFileStats {
    pub uri: Url,
    pub config_root: Option<Url>,
}

pub struct TelemetryServerState {
    pub has_sourcedb: bool,
    pub id: Uuid,
}

#[derive(Default)]
pub struct TelemetryTransactionStats {
    pub modules: usize,
    pub dirty_rdeps: usize,
    pub cycle_rdeps: usize,
    pub run_steps: usize,
    pub run_time: Duration,
    pub committed: bool,
}

pub struct TelemetryTaskStats {
    pub queue_name: &'static str,
    pub id: usize,
}

impl TelemetryTaskStats {
    pub fn new(queue_name: &'static str, id: usize) -> Self {
        Self { queue_name, id }
    }
}

#[derive(Default)]
pub struct TelemetrySourceDbRebuildStats {
    pub count: usize,
    pub files: usize,
    pub changed: bool,
    pub had_error: bool,
}

impl TelemetryEvent {
    pub fn new_dequeued(
        kind: TelemetryEventKind,
        enqueued_at: Instant,
        server_state: TelemetryServerState,
    ) -> Self {
        let start = Instant::now();
        let queue = start - enqueued_at;
        Self {
            kind,
            queue,
            start,
            error: None,
            invalidate: None,
            validate: None,
            transaction_stats: None,
            server_state,
            file_stats: None,
            task_stats: None,
            sourcedb_rebuild_stats: None,
        }
    }

    pub fn set_invalidate_duration(&mut self, duration: Duration) {
        self.invalidate = Some(duration);
    }

    pub fn set_validate_duration(&mut self, duration: Duration) {
        self.validate = Some(duration);
    }

    pub fn set_transaction_stats(&mut self, stats: TelemetryTransactionStats) {
        self.transaction_stats = Some(stats);
    }

    pub fn set_file_stats(&mut self, stats: TelemetryFileStats) {
        self.file_stats = Some(stats);
    }

    pub fn set_task_stats(&mut self, stats: TelemetryTaskStats) {
        self.task_stats = Some(stats);
    }

    pub fn set_sourcedb_rebuild_stats(&mut self, stats: TelemetrySourceDbRebuildStats) {
        self.sourcedb_rebuild_stats = Some(stats);
    }

    pub fn finish_and_record(self, telemetry: &impl Telemetry, error: Option<&Error>) -> Duration {
        let process = self.start.elapsed();
        telemetry.record_event(self, process, error);
        process
    }
}
