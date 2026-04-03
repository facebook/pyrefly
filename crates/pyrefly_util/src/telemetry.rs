/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Display;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use anyhow::Error;
use dupe::Dupe;
use lsp_types::Url;
use serde::Deserialize;
use serde::Serialize;
use uuid::Uuid;

pub trait Telemetry: Send + Sync {
    fn record_event(&self, event: TelemetryEvent, process: Duration, error: Option<&Error>);
    fn surface(&self) -> Option<String>;
    fn agent_session_id(&self) -> Option<String>;
    fn agent_invocation_id(&self) -> Option<String>;
}
pub struct NoTelemetry;

impl Telemetry for NoTelemetry {
    fn record_event(&self, _event: TelemetryEvent, _process: Duration, _error: Option<&Error>) {}
    fn surface(&self) -> Option<String> {
        None
    }
    fn agent_session_id(&self) -> Option<String> {
        None
    }
    fn agent_invocation_id(&self) -> Option<String> {
        None
    }
}

#[derive(Debug, Clone, Dupe, Copy)]
pub enum QueueName {
    LspQueue,
    RecheckQueue,
    FindReferenceQueue,
    SourceDbQueue,
}

impl Display for QueueName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl QueueName {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::LspQueue => "lsp_queue",
            Self::RecheckQueue => "recheck_queue",
            Self::FindReferenceQueue => "find_reference_queue",
            Self::SourceDbQueue => "sourcedb_queue",
        }
    }
}

pub enum TelemetryEventKind {
    LspEvent(String),
    CodeAction(&'static str),
    AdHocSolve(&'static str),
    SetMemory,
    InvalidateDisk,
    InvalidateFind,
    InvalidateEvents,
    InvalidateConfig,
    InvalidateOnClose,
    PopulateProjectFiles,
    PopulateWorkspaceFiles,
    WorkspaceDiagnosticsRepopulation,
    SourceDbRebuild,
    SourceDbRebuildInstance,
    FindFromDefinition,
    ExternalReferences,
    ExternalWorkspaceSymbols,
    LspStartup,
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
    pub queue_name: QueueName,
    pub task_id: usize,
    pub sourcedb_rebuild_stats: Option<TelemetrySourceDbRebuildStats>,
    pub sourcedb_rebuild_instance_stats: Option<TelemetrySourceDbRebuildInstanceStats>,
    pub file_watcher_stats: Option<TelemetryFileWatcherStats>,
    pub did_change_watched_files_stats: Option<TelemetryDidChangeWatchedFilesStats>,
    pub external_references_stats: Option<TelemetryExternalReferencesStats>,
    pub external_workspace_symbols_stats: Option<TelemetryExternalWorkspaceSymbolsStats>,
    pub activity_key: Option<ActivityKey>,
    pub canceled: bool,
    pub empty_response_reason: Option<EmptyResponseReason>,
    /// The LSP request ID, used to correlate CancelRequest notifications with
    /// the requests they cancel.
    pub request_id: Option<String>,
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
    pub server_start_time: Instant,
    pub agent_session_id: Option<String>,
    pub agent_invocation_id: Option<String>,
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
    /// `true` when the transaction was created fresh (restore failed or no saved state),
    /// `false` when restored from saved state.
    pub fresh: bool,
    /// Number of modules dirtied by `set_memory`.
    pub set_memory_dirty: usize,
    /// Time spent in `compute_stdlib` during `run_step`.
    pub compute_stdlib_time: Duration,
    /// `true` when stdlib was already cached and computation was skipped.
    pub compute_stdlib_cached: bool,
    /// Time spent in the parallel pre-warming phase of `compute_stdlib`.
    pub compute_stdlib_prewarm_time: Duration,
    /// Number of modules in the dirty set at the start of `run_step`.
    pub run_dirty_count: usize,
    /// Number of items pushed to the todo work queue in `run_step`.
    pub run_todo_count: usize,
    /// Time spent in `work()` (the parallel solve phase) during `run_step`.
    pub run_work_time: Duration,
    /// Time spent in `spawn_many` during `search_exports`.
    pub search_exports_time: Duration,
    /// Max time a thread waited before starting work in `search_exports`.
    pub search_exports_dispatch_time: Duration,
    /// Whether the transaction was cancelled before completing.
    pub cancelled: bool,
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
    pub exit_reason: Option<String>,
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

#[derive(Default)]
pub struct TelemetryExternalWorkspaceSymbolsStats {
    pub query: String,
    pub db_name: Option<String>,
    pub result_count: usize,
    pub find_repo_ms: Option<Duration>,
    pub angle_query_ms: Option<Duration>,
}

#[derive(Default)]
pub struct TelemetryExternalReferencesStats {
    pub qualified_name: String,
    pub db_name: Option<String>,
    pub result_file_count: usize,
    pub result_span_count: usize,
    pub find_repo_ms: Option<Duration>,
    pub angle_query_ms: Option<Duration>,
    pub cas_init_error: Option<String>,
    pub resolve_locations_ms: Option<Duration>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActivityKey {
    pub id: String,
    pub name: String,
}

/// Why an LSP handler returned an empty/null response. Used for telemetry
/// to distinguish expected cases (whitespace, comments) from unexpected
/// failures (module not found, internal bugs).
#[derive(Debug, Clone)]
pub enum EmptyResponseReason {
    /// `path_for_uri` returned None — URI couldn't be resolved to a
    /// filesystem path (e.g., unsupported URI scheme, malformed URI).
    NoFilePath,
    /// Workspace has `disable_language_services = true`.
    LanguageServicesDisabled,
    /// Specific LSP method is disabled via config.
    MethodDisabled,
    /// Notebook cell not supported for this operation.
    NotebookNotSupported,

    /// `get_module_info` returned None — file may not belong to any
    /// workspace or hasn't been loaded yet.
    ModuleInfoNotFound,
    /// `get_ast` returned None — module is in the graph but AST hasn't
    /// been computed yet (startup/initial load).
    AstNotFound,
    /// `get_answers` returned None — answers haven't been computed yet
    /// for this module (should only happen during startup/initial load).
    AnswersNotFound,
    /// `get_bindings` returned None — bindings haven't been computed yet
    /// for this module (should only happen during startup/initial load).
    BindingsNotFound,
    /// `get_type_trace` returned None — the expression at the cursor
    /// doesn't have a traced type (e.g., operator on an unresolved expr).
    TypeTraceNotFound,
    /// Import resolution couldn't find the module (e.g., `import foo`
    /// where `foo` doesn't exist or isn't in the search path).
    ModuleNotFound,

    /// Cursor is on something that isn't an identifier or navigable symbol
    /// (whitespace, comments, keywords, string literals, etc.).
    NotAnIdentifier {
        /// Context about what's at the cursor, e.g. "ExprStringLiteral",
        /// "StmtIf", "operator:not", "none".
        found: String,
    },

    /// We identified the symbol but couldn't resolve its definition.
    DefinitionNotFound {
        /// The identifier text that was being looked up.
        name: String,
        /// What kind of symbol was at the cursor.
        context: DefinitionContext,
    },

    /// Definition targets found but `to_lsp_location` filtered them all
    /// out (e.g., bundled typeshed modules whose paths can't be materialized).
    BundledModuleNotMaterialized {
        /// Number of targets that were found but couldn't be materialized.
        target_count: usize,
    },
}

impl EmptyResponseReason {
    /// Snake_case variant name for the telemetry `empty_response_reason` column.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::NoFilePath => "no_file_path",
            Self::LanguageServicesDisabled => "language_services_disabled",
            Self::MethodDisabled => "method_disabled",
            Self::NotebookNotSupported => "notebook_not_supported",
            Self::ModuleInfoNotFound => "module_info_not_found",
            Self::AstNotFound => "ast_not_found",
            Self::AnswersNotFound => "answers_not_found",
            Self::BindingsNotFound => "bindings_not_found",
            Self::TypeTraceNotFound => "type_trace_not_found",
            Self::ModuleNotFound => "module_not_found",
            Self::NotAnIdentifier { .. } => "not_an_identifier",
            Self::DefinitionNotFound { .. } => "definition_not_found",
            Self::BundledModuleNotMaterialized { .. } => "bundled_module_not_materialized",
        }
    }

    /// Detail string for the telemetry `empty_response_detail` column.
    pub fn detail(&self) -> String {
        match self {
            Self::NotAnIdentifier { found } => found.clone(),
            Self::DefinitionNotFound { name, context } => {
                format!("{name}:{context}")
            }
            Self::BundledModuleNotMaterialized { target_count } => target_count.to_string(),
            _ => String::new(),
        }
    }
}

/// What kind of symbol the cursor was on when a definition lookup failed.
#[derive(Debug, Clone)]
pub enum DefinitionContext {
    /// Name used in an expression (e.g., `x` in `x + 1`).
    NameUse,
    /// Name being defined/assigned (e.g., `x` in `x = 1`).
    NameDef,
    /// Attribute access (e.g., `bar` in `foo.bar`).
    Attribute,
    /// Keyword argument (e.g., `key` in `f(key=val)`).
    KeywordArgument,
    /// Module in an import statement (e.g., `foo` in `import foo`).
    ImportedModule,
    /// Name in a from-import (e.g., `bar` in `from foo import bar`).
    ImportedName,
    /// Function/method/class definition name.
    Definition,
    /// Parameter, type parameter, exception handler, or pattern match binding.
    LocalBinding,
    /// `global`/`nonlocal` capture.
    MutableCapture,
    /// Operator with a dunder (e.g., `+` → `__add__`), but the dunder
    /// method couldn't be found on the operand type.
    Operator {
        /// The dunder name that was looked up.
        dunder: String,
    },
    /// `None` literal — couldn't resolve `NoneType` definition.
    NoneLiteral,
}

impl DefinitionContext {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::NameUse => "name_use",
            Self::NameDef => "name_def",
            Self::Attribute => "attribute",
            Self::KeywordArgument => "keyword_argument",
            Self::ImportedModule => "imported_module",
            Self::ImportedName => "imported_name",
            Self::Definition => "definition",
            Self::LocalBinding => "local_binding",
            Self::MutableCapture => "mutable_capture",
            Self::Operator { .. } => "operator",
            Self::NoneLiteral => "none_literal",
        }
    }
}

impl fmt::Display for DefinitionContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Operator { dunder } => write!(f, "operator:{dunder}"),
            other => write!(f, "{}", other.as_str()),
        }
    }
}

impl TelemetryEvent {
    pub fn new_dequeued(
        kind: TelemetryEventKind,
        enqueued_at: Instant,
        server_state: TelemetryServerState,
        queue_name: QueueName,
        task_id: usize,
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
                queue_name,
                task_id,
                sourcedb_rebuild_stats: None,
                sourcedb_rebuild_instance_stats: None,
                file_watcher_stats: None,
                did_change_watched_files_stats: None,
                external_references_stats: None,
                external_workspace_symbols_stats: None,
                activity_key: None,
                canceled: false,
                empty_response_reason: None,
                request_id: None,
            },
            queue,
        )
    }

    pub fn new_task(
        kind: TelemetryEventKind,
        server_state: TelemetryServerState,
        queue_name: QueueName,
        task_id: usize,
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
            queue_name,
            task_id,
            sourcedb_rebuild_stats: None,
            sourcedb_rebuild_instance_stats: None,
            file_watcher_stats: None,
            did_change_watched_files_stats: None,
            external_references_stats: None,
            external_workspace_symbols_stats: None,
            activity_key: None,
            canceled: false,
            empty_response_reason: None,
            request_id: None,
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

    pub fn set_external_references_stats(&mut self, stats: TelemetryExternalReferencesStats) {
        self.external_references_stats = Some(stats);
    }

    pub fn set_external_workspace_symbols_stats(
        &mut self,
        stats: TelemetryExternalWorkspaceSymbolsStats,
    ) {
        self.external_workspace_symbols_stats = Some(stats);
    }

    pub fn set_empty_response_reason(&mut self, reason: EmptyResponseReason) {
        self.empty_response_reason = Some(reason);
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
    queue_name: QueueName,
    task_id: usize,
    activity_key: Option<ActivityKey>,
    file_stats: Option<TelemetryFileStats>,
}

impl<'a> SubTaskTelemetry<'a> {
    pub fn new(telemetry: &'a dyn Telemetry, event: &TelemetryEvent) -> Self {
        Self {
            telemetry,
            server_state: event.server_state.clone(),
            queue_name: event.queue_name,
            task_id: event.task_id,
            activity_key: event.activity_key.clone(),
            file_stats: event.file_stats.clone(),
        }
    }

    pub fn new_task(&self, kind: TelemetryEventKind, start: Instant) -> TelemetryEvent {
        let mut event = TelemetryEvent::new_task(
            kind,
            self.server_state.clone(),
            self.queue_name,
            self.task_id,
            start,
        );
        event.set_activity_key(self.activity_key.clone());
        event.file_stats = self.file_stats.clone();
        event
    }

    pub fn finish_task(&self, telemetry_event: TelemetryEvent, error: Option<&Error>) {
        telemetry_event.finish_and_record(self.telemetry, error);
    }
}
