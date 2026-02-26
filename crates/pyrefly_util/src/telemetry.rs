/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use anyhow::Error;
use lsp_types::Url;
use serde::Deserialize;
use serde::Serialize;
use uuid::Uuid;

pub trait Telemetry: Send + Sync {
    fn record_event(&self, event: TelemetryEvent, process: Duration, error: Option<&Error>);
    fn surface(&self) -> Option<String>;
}
pub struct NoTelemetry;

impl Telemetry for NoTelemetry {
    fn record_event(&self, _event: TelemetryEvent, _process: Duration, _error: Option<&Error>) {}
    fn surface(&self) -> Option<String> {
        None
    }
}

pub enum TelemetryEventKind {
    LspEvent(String),
    CodeAction(String),
    AdHocSolve(String),
    SetMemory,
    InvalidateDisk,
    InvalidateFind,
    InvalidateEvents,
    InvalidateConfig,
    InvalidateOnClose,
    PopulateProjectFiles,
    PopulateWorkspaceFiles,
    SourceDbRebuild,
    SourceDbRebuildInstance,
    FindFromDefinition,
}

pub struct TelemetryEvent {
    pub kind: TelemetryEventKind,
    pub queue: Option<Duration>,
    pub start: Instant,
    pub invalidate: Option<Duration>,
    pub validate: Option<Duration>,
    pub transaction_stats: Option<TelemetryTransactionStats>,
    pub server_state: TelemetryServerState,
    pub file_stats: Option<TelemetryFileStats>,
    pub task_id: Option<TelemetryTaskId>,
    pub sourcedb_rebuild_stats: Option<TelemetrySourceDbRebuildStats>,
    pub sourcedb_rebuild_instance_stats: Option<TelemetrySourceDbRebuildInstanceStats>,
    pub file_watcher_stats: Option<TelemetryFileWatcherStats>,
    pub did_change_watched_files_stats: Option<TelemetryDidChangeWatchedFilesStats>,
    pub activity_key: Option<ActivityKey>,
    pub canceled: bool,
}

#[derive(Clone)]
pub struct TelemetryFileStats {
    pub uri: Url,
    pub config_root: Option<Url>,
}

#[derive(Clone)]
pub struct TelemetryServerState {
    pub has_sourcedb: bool,
    pub id: Uuid,
    /// The surface/entrypoint for the language server
    pub surface: Option<String>,
}

#[derive(Default)]
pub struct TelemetryTransactionStats {
    pub modules: usize,
    pub dirty_rdeps: usize,
    pub cycle_rdeps: usize,
    pub run_steps: usize,
    pub run_time: Duration,
    pub committed: bool,
    pub state_lock_blocked: Duration,
}

#[derive(Clone)]
pub struct TelemetryTaskId {
    pub queue_name: &'static str,
    pub id: Option<usize>,
}

impl TelemetryTaskId {
    pub fn new(queue_name: &'static str, id: Option<usize>) -> Self {
        Self { queue_name, id }
    }
}

#[derive(Default)]
pub struct TelemetryCommonSourceDbStats {
    pub files: usize,
    pub changed: bool,
    pub forced: bool,
}

#[derive(Default)]
pub struct TelemetrySourceDbRebuildStats {
    pub count: usize,
    pub had_error: bool,
    pub common: TelemetryCommonSourceDbStats,
}

#[derive(Default)]
pub struct TelemetrySourceDbRebuildInstanceStats {
    pub common: TelemetryCommonSourceDbStats,
    pub build_id: Option<String>,
    pub build_time: Option<Duration>,
    pub parse_time: Option<Duration>,
    pub process_time: Option<Duration>,
    pub raw_size: Option<usize>,
}

#[derive(Default)]
pub struct TelemetryFileWatcherStats {
    pub duration: Duration,
    pub count: usize,
}

#[derive(Default)]
pub struct TelemetryDidChangeWatchedFilesStats {
    pub created: Vec<PathBuf>,
    pub modified: Vec<PathBuf>,
    pub removed: Vec<PathBuf>,
    pub unknown: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActivityKey {
    pub id: String,
    pub name: String,
}

impl TelemetryEvent {
    pub fn new_dequeued(
        kind: TelemetryEventKind,
        enqueued_at: Instant,
        server_state: TelemetryServerState,
    ) -> (Self, Duration) {
        let start = Instant::now();
        let queue = start - enqueued_at;
        (
            Self {
                kind,
                queue: Some(queue),
                start,
                invalidate: None,
                validate: None,
                transaction_stats: None,
                server_state,
                file_stats: None,
                task_id: None,
                sourcedb_rebuild_stats: None,
                sourcedb_rebuild_instance_stats: None,
                file_watcher_stats: None,
                did_change_watched_files_stats: None,
                activity_key: None,
                canceled: false,
            },
            queue,
        )
    }

    pub fn new_task(
        kind: TelemetryEventKind,
        server_state: TelemetryServerState,
        task_id: Option<TelemetryTaskId>,
        start: Instant,
    ) -> Self {
        Self {
            kind,
            queue: None,
            start,
            invalidate: None,
            validate: None,
            transaction_stats: None,
            server_state,
            file_stats: None,
            task_id,
            sourcedb_rebuild_stats: None,
            sourcedb_rebuild_instance_stats: None,
            file_watcher_stats: None,
            did_change_watched_files_stats: None,
            activity_key: None,
            canceled: false,
        }
    }

    pub fn set_activity_key(&mut self, activity_key: Option<ActivityKey>) {
        self.activity_key = activity_key;
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

    pub fn set_task_stats(&mut self, stats: TelemetryTaskId) {
        self.task_id = Some(stats);
    }

    pub fn set_sourcedb_rebuild_stats(&mut self, stats: TelemetrySourceDbRebuildStats) {
        self.sourcedb_rebuild_stats = Some(stats);
    }

    pub fn set_sourcedb_rebuild_instance_stats(
        &mut self,
        stats: TelemetrySourceDbRebuildInstanceStats,
    ) {
        self.sourcedb_rebuild_instance_stats = Some(stats);
    }

    pub fn set_file_watcher_stats(&mut self, stats: TelemetryFileWatcherStats) {
        self.file_watcher_stats = Some(stats);
    }

    pub fn set_did_change_watched_files_stats(
        &mut self,
        stats: TelemetryDidChangeWatchedFilesStats,
    ) {
        self.did_change_watched_files_stats = Some(stats);
    }

    pub fn finish_and_record(self, telemetry: &dyn Telemetry, error: Option<&Error>) -> Duration {
        let process = self.start.elapsed();
        telemetry.record_event(self, process, error);
        process
    }
}

pub struct SubTaskTelemetry<'a> {
    telemetry: &'a dyn Telemetry,
    server_state: TelemetryServerState,
    task_stats: Option<&'a TelemetryTaskId>,
}

impl<'a> SubTaskTelemetry<'a> {
    pub fn new(
        telemetry: &'a dyn Telemetry,
        server_state: TelemetryServerState,
        task_stats: Option<&'a TelemetryTaskId>,
    ) -> Self {
        Self {
            telemetry,
            server_state,
            task_stats,
        }
    }

    pub fn new_task(&self, kind: TelemetryEventKind, start: Instant) -> TelemetryEvent {
        TelemetryEvent::new_task(
            kind,
            self.server_state.clone(),
            self.task_stats.cloned(),
            start,
        )
    }

    pub fn finish_task(&self, telemetry_event: TelemetryEvent, error: Option<&Error>) {
        telemetry_event.finish_and_record(self.telemetry, error);
    }
}
