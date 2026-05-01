/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;
use std::fmt;
use std::fmt::Display;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anstream::eprintln;
use anstream::stdout;
use anyhow::Context as _;
use clap::Parser;
use clap::ValueEnum;
use dupe::Dupe as _;
use percent_encoding::AsciiSet;
use percent_encoding::CONTROLS;
use percent_encoding::utf8_percent_encode;
use pyrefly_build::handle::Handle;
use pyrefly_config::args::ConfigOverrideArgs;
use pyrefly_config::config::ConfigFile;
use pyrefly_config::config::OutputFormat;
use pyrefly_config::error_kind::ErrorKind;
use pyrefly_config::finder::ConfigError;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_name::ModuleNameWithKind;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::args::clap_env;
use pyrefly_util::display;
use pyrefly_util::display::count;
use pyrefly_util::display::number_thousands;
use pyrefly_util::events::CategorizedEvents;
use pyrefly_util::forgetter::Forgetter;
use pyrefly_util::fs_anyhow;
use pyrefly_util::includes::Includes;
use pyrefly_util::memory::MemoryUsageTrace;
use pyrefly_util::thread_pool::ThreadCount;
use pyrefly_util::watcher::Watcher;
use ruff_text_size::Ranged;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use tracing::debug;
use tracing::info;

use crate::commands::config_finder::ConfigConfigurerWrapper;
use crate::commands::files::FilesArgs;
use crate::commands::util::CommandExitStatus;
use crate::config::error_kind::Severity;
use crate::config::finder::ConfigFinder;
use crate::error::error::Error;
use crate::error::error::print_error_counts;
use crate::error::legacy::LegacyError;
use crate::error::legacy::LegacyErrors;
use crate::error::legacy::severity_to_str;
use crate::error::summarize::print_error_summary;
use crate::error::suppress;
use crate::error::suppress::CommentLocation;
use crate::error::suppress::SerializedError;
use crate::module::typeshed::stdlib_search_path;
use crate::report;
use crate::state::load::FileContents;
use crate::state::require::Require;
use crate::state::require::RequireLevels;
use crate::state::state::State;
use crate::state::state::Transaction;
use crate::state::subscriber::ProgressBarStyle;

/// Result data from a non-watch check run, used for telemetry logging.
pub struct CheckResult {
    /// CLI-visible diagnostics in the legacy JSON format, suitable for serialization.
    pub legacy_errors: Vec<LegacyError>,
    /// Number of files (modules) that were checked.
    pub checked_file_count: usize,
}

impl CheckResult {
    /// Build a `CheckResult` from the raw error list.
    fn from_errors(errors: &[Error], relative_to: &Path, checked_file_count: usize) -> Self {
        Self {
            legacy_errors: errors
                .iter()
                .map(|e| LegacyError::from_error(relative_to, e))
                .collect(),
            checked_file_count,
        }
    }
}

/// Check the given files.
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Clone, Parser)]
pub struct FullCheckArgs {
    /// Which files to check.
    #[command(flatten)]
    pub files: FilesArgs,

    /// Watch for file changes and re-check them.
    /// (Warning: This mode is highly experimental!)
    #[arg(long, conflicts_with = "check_all")]
    watch: bool,

    /// Type checking arguments and configuration
    #[command(flatten)]
    args: CheckArgs,

    /// Configuration override options
    #[command(flatten, next_help_heading = "Config Overrides")]
    pub config_override: ConfigOverrideArgs,
}

impl FullCheckArgs {
    pub async fn run(
        self,
        wrapper: Option<ConfigConfigurerWrapper>,
        thread_count: ThreadCount,
    ) -> anyhow::Result<(CommandExitStatus, Option<CheckResult>)> {
        self.config_override.validate()?;
        let (files_to_check, config_finder) = self.files.resolve(self.config_override, wrapper)?;
        run_check(
            self.args,
            self.watch,
            files_to_check,
            config_finder,
            thread_count,
        )
        .await
    }
}

/// Resolve the `--relative-to` argument to a concrete path for error reporting.
fn resolve_relative_to(relative_to: Option<&String>) -> PathBuf {
    relative_to.map_or_else(
        || std::env::current_dir().ok().unwrap_or_default(),
        |x| PathBuf::from_str(x.as_str()).unwrap(),
    )
}

async fn run_check(
    args: CheckArgs,
    watch: bool,
    files_to_check: Box<dyn Includes>,
    config_finder: ConfigFinder,
    thread_count: ThreadCount,
) -> anyhow::Result<(CommandExitStatus, Option<CheckResult>)> {
    if watch {
        let roots = files_to_check.roots();
        info!(
            "Watching for files in {}",
            display::intersperse_iter(";", || roots.iter().map(|p| p.display()))
        );
        let watcher = Watcher::notify(&roots)?;
        args.run_watch(watcher, files_to_check, config_finder, thread_count)
            .await?;
        Ok((CommandExitStatus::Success, None))
    } else {
        let (status, _, check_result) =
            args.run_once(files_to_check, config_finder, thread_count)?;
        Ok((status, Some(check_result)))
    }
}

/// Main arguments for Pyrefly type checker
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Parser, Clone)]
pub struct CheckArgs {
    /// Output related configuration options
    #[command(flatten, next_help_heading = "Output")]
    output: OutputArgs,
    /// Behavior-related configuration options
    #[command(flatten, next_help_heading = "Behavior")]
    behavior: BehaviorArgs,
}

/// Arguments for snippet checking (excludes behavior args that don't apply to snippets)
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Parser, Clone)]
pub struct SnippetCheckArgs {
    /// Python code to type check
    code: String,

    /// Explicitly set the Pyrefly configuration to use when type checking.
    /// When not set, Pyrefly will perform an upward-filesystem-walk approach to find the nearest
    /// pyrefly.toml or pyproject.toml with `tool.pyrefly` section'. If no config is found, Pyrefly exits with error.
    /// If both a pyrefly.toml and valid pyproject.toml are found, pyrefly.toml takes precedence.
    #[arg(long, short, value_name = "FILE", env = clap_env("CONFIG"))]
    config: Option<PathBuf>,

    /// Output related configuration options
    #[command(flatten, next_help_heading = "Output")]
    output: OutputArgs,
    /// Configuration override options
    #[command(flatten, next_help_heading = "Config Overrides")]
    pub config_override: ConfigOverrideArgs,
}

impl SnippetCheckArgs {
    pub async fn run(
        self,
        wrapper: Option<ConfigConfigurerWrapper>,
        thread_count: ThreadCount,
    ) -> anyhow::Result<(CommandExitStatus, Option<CheckResult>)> {
        let (_, config_finder) =
            FilesArgs::get(vec![], self.config, self.config_override, wrapper)?;
        let check_args = CheckArgs {
            output: self.output,
            behavior: BehaviorArgs {
                check_all: false,
                suppress_errors: false,
                expectations: false,
                remove_unused_ignores: false,
            },
        };
        let (status, check_result) =
            check_args.run_once_with_snippet(self.code, config_finder, thread_count)?;
        Ok((status, Some(check_result)))
    }
}

/// how/what should Pyrefly output
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Parser, Clone)]
struct OutputArgs {
    /// Write the errors to a file, instead of printing them.
    #[arg(long, short = 'o', value_name = "OUTPUT_FILE")]
    output: Option<PathBuf>,
    /// Set the error output format.
    #[arg(long, value_enum)]
    output_format: Option<OutputFormat>,
    /// Produce debugging information about the type checking process.
    #[arg(long, value_name = "OUTPUT_FILE")]
    debug_info: Option<PathBuf>,
    /// Report the memory usage of bindings.
    #[arg(long, value_name = "OUTPUT_FILE")]
    report_binding_memory: Option<PathBuf>,
    /// Report type traces.
    #[arg(long, value_name = "OUTPUT_FILE")]
    report_trace: Option<PathBuf>,
    /// Experimental: generate a JSON dependency graph of all modules to the specified file. This is unstable and should only be used for debugging.
    #[arg(long, value_name = "OUTPUT_FILE")]
    dependency_graph: Option<PathBuf>,
    /// Process each module individually to figure out how long each step takes.
    #[arg(long, value_name = "OUTPUT_FILE")]
    report_timings: Option<PathBuf>,
    /// Generate a Glean-compatible JSON file for each module
    #[arg(long, value_name = "OUTPUT_FILE")]
    report_glean: Option<PathBuf>,
    /// Generate a Pysa-compatible JSON file for each module
    #[arg(long, value_name = "OUTPUT_FILE")]
    report_pysa: Option<PathBuf>,
    /// Format for pysa report output (json or capnp)
    #[arg(long, value_enum, default_value_t = report::pysa::PysaFormat::Capnp)]
    report_pysa_format: report::pysa::PysaFormat,
    /// Generate a CinderX-format type report (experimental, internal-only).
    #[arg(long, value_name = "OUTPUT_DIR", hide = true)]
    report_cinderx: Option<PathBuf>,
    /// Also write human-readable .txt files alongside the CinderX JSON report.
    /// Each .txt file inlines type-table indices so types are fully readable without
    /// cross-referencing the JSON. Intended for debugging; mirrors view_types.py output.
    #[arg(long, hide = true)]
    cinderx_include_readable: bool,
    /// Include all transitively-imported dependency modules in the CinderX report,
    /// not just the explicitly type-checked project files.
    #[arg(long, hide = true)]
    cinderx_include_deps: bool,
    /// Count the number of each error kind. Prints the top N [default=5] errors, sorted by count, or all errors if N is 0.
    #[arg(
        long,
        default_missing_value = "5",
        require_equals = true,
        num_args = 0..=1,
        value_name = "N",
    )]
    count_errors: Option<usize>,
    /// Summarize errors by directory. The optional index argument specifies which file path segment will be used to group errors.
    /// The default index is 0. For errors in `/foo/bar/...`, this will group errors by `/foo`. If index is 1, errors will be grouped by `/foo/bar`.
    /// An index larger than the number of path segments will group by the final path element, i.e. the file name.
    #[arg(
        long,
        default_missing_value = "0",
        require_equals = true,
        num_args = 0..=1,
        value_name = "INDEX",
    )]
    summarize_errors: Option<usize>,

    /// Filter errors to show only a specific error kind (e.g., bad-assignment, missing-return, etc.).
    /// Can be passed multiple times or as a comma-separated list.
    #[arg(
        long,
        value_enum,
        value_name = "ERROR_KIND",
        hide_possible_values = true,
        value_delimiter = ','
    )]
    only: Option<Vec<ErrorKind>>,

    /// By default show a progress bar and the number of errors.
    /// Pass `--summary` to additionally show information about lines checked and time/memory,
    /// or `--summary=none` to hide the progress bar and summary line entirely.
    #[arg(
        long,
        default_missing_value = "full",
        require_equals = true,
        num_args = 0..=1,
        value_enum,
        default_value_t
    )]
    summary: Summary,

    /// Suppress the progress bar during type checking. Deprecated: use `--progress-bar=no` instead.
    #[arg(long, hide = true)]
    no_progress_bar: bool,

    /// Set the progress bar style.
    /// `interactive` (default) shows a visual progress bar.
    /// `simple` prints periodic log-style progress messages (suitable for piping or non-interactive use).
    /// `no` disables progress reporting entirely.
    #[arg(long, value_enum)]
    progress_bar: Option<ProgressBarStyle>,

    /// When specified, strip this prefix from any paths in the output.
    /// Pass "" to show absolute paths. When omitted, we will use the current working directory.
    #[arg(long)]
    relative_to: Option<String>,

    /// Path to baseline file for comparing type errors
    #[arg(long, value_name = "BASELINE_FILE")]
    baseline: Option<PathBuf>,

    /// When specified, emit a sorted/formatted JSON of the errors to the baseline file
    #[arg(long, requires("baseline"))]
    update_baseline: bool,

    /// Minimum severity level for errors to be displayed.
    /// Errors below this severity will not be shown. Defaults to "error".
    #[arg(long, value_enum)]
    min_severity: Option<Severity>,
}

impl OutputArgs {
    fn inherit_defaults_from_config(&mut self, config: &ConfigFile) {
        if self.baseline.is_none() {
            self.baseline = config.baseline.clone();
        }
        if self.output_format.is_none() {
            self.output_format = config.output_format;
        }
        if self.min_severity.is_none() {
            self.min_severity = config.min_severity;
        }
    }

    fn output_format(&self) -> OutputFormat {
        self.output_format.unwrap_or_default()
    }

    /// Resolve the effective progress bar style, taking deprecated flags into account.
    fn progress_bar_style(&self) -> ProgressBarStyle {
        if let Some(style) = &self.progress_bar {
            return style.clone();
        }
        if self.no_progress_bar || self.summary == Summary::None {
            ProgressBarStyle::No
        } else {
            ProgressBarStyle::Interactive
        }
    }
}

#[derive(Clone, Debug, ValueEnum, Default, PartialEq, Eq)]
enum Summary {
    None,
    #[default]
    Default,
    Full,
}

/// non-config type checker behavior
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Parser, Clone)]
struct BehaviorArgs {
    /// Check all reachable modules, not just the ones that are passed in explicitly on CLI positional arguments.
    #[arg(long, short = 'a')]
    check_all: bool,
    /// Suppress errors found in the input files.
    #[arg(long)]
    suppress_errors: bool,
    /// Check against any `E:` lines in the file.
    #[arg(long)]
    expectations: bool,
    /// Remove unused ignores from the input files.
    #[arg(long)]
    remove_unused_ignores: bool,
}

fn write_errors_to_file(
    format: OutputFormat,
    path: &Path,
    relative_to: &Path,
    errors: &[Error],
) -> anyhow::Result<()> {
    match format {
        OutputFormat::MinText => write_error_text_to_file(path, relative_to, errors, false),
        OutputFormat::FullText => write_error_text_to_file(path, relative_to, errors, true),
        OutputFormat::Json => write_error_json_to_file(path, relative_to, errors),
        OutputFormat::Github => write_error_github_to_file(path, errors),
        OutputFormat::OmitErrors => Ok(()),
    }
}

fn write_errors_to_console(
    format: OutputFormat,
    relative_to: &Path,
    errors: &[Error],
) -> anyhow::Result<()> {
    match format {
        OutputFormat::MinText => write_error_text_to_console(relative_to, errors, false),
        OutputFormat::FullText => write_error_text_to_console(relative_to, errors, true),
        OutputFormat::Json => write_error_json_to_console(relative_to, errors),
        OutputFormat::Github => write_error_github_to_console(errors),
        OutputFormat::OmitErrors => Ok(()),
    }
}

fn write_error_text_to_file(
    path: &Path,
    relative_to: &Path,
    errors: &[Error],
    verbose: bool,
) -> anyhow::Result<()> {
    let mut file = BufWriter::new(File::create(path)?);
    for e in errors {
        e.write_line(&mut file, relative_to, verbose)?;
    }
    file.flush()?;
    Ok(())
}

fn write_error_text_to_console(
    relative_to: &Path,
    errors: &[Error],
    verbose: bool,
) -> anyhow::Result<()> {
    for error in errors {
        error.print_colors(relative_to, verbose);
    }
    Ok(())
}

fn write_error_json(
    writer: &mut impl Write,
    relative_to: &Path,
    errors: &[Error],
) -> anyhow::Result<()> {
    let legacy_errors = LegacyErrors::from_errors(relative_to, errors);
    serde_json::to_writer_pretty(writer, &legacy_errors)?;
    Ok(())
}

fn buffered_write_error_json(
    writer: impl Write,
    relative_to: &Path,
    errors: &[Error],
) -> anyhow::Result<()> {
    let mut writer = BufWriter::new(writer);
    write_error_json(&mut writer, relative_to, errors)?;
    writer.flush()?;
    Ok(())
}

fn write_error_json_to_file(
    path: &Path,
    relative_to: &Path,
    errors: &[Error],
) -> anyhow::Result<()> {
    fn f(path: &Path, relative_to: &Path, errors: &[Error]) -> anyhow::Result<()> {
        let file = File::create(path)?;
        buffered_write_error_json(file, relative_to, errors)
    }
    f(path, relative_to, errors)
        .with_context(|| format!("while writing JSON errors to `{}`", path.display()))
}

fn write_error_json_to_console(relative_to: &Path, errors: &[Error]) -> anyhow::Result<()> {
    buffered_write_error_json(stdout(), relative_to, errors)
}

fn write_error_github(writer: &mut impl Write, errors: &[Error]) -> anyhow::Result<()> {
    for error in errors {
        if let Some(command) = github_actions_command(error) {
            writeln!(writer, "{command}")?;
        }
    }
    Ok(())
}

fn buffered_write_error_github(writer: impl Write, errors: &[Error]) -> anyhow::Result<()> {
    let mut writer = BufWriter::new(writer);
    write_error_github(&mut writer, errors)?;
    writer.flush()?;
    Ok(())
}

fn write_error_github_to_file(path: &Path, errors: &[Error]) -> anyhow::Result<()> {
    let file = File::create(path)?;
    buffered_write_error_github(file, errors)
}

fn write_error_github_to_console(errors: &[Error]) -> anyhow::Result<()> {
    buffered_write_error_github(stdout(), errors)
}

fn severity_to_github_command(severity: Severity) -> Option<&'static str> {
    let normalized = severity_to_str(severity);
    match normalized.as_str() {
        "ignore" => None,
        "warn" => Some("warning"),
        "info" => Some("notice"),
        "error" => Some("error"),
        _ => None,
    }
}

fn github_actions_command(error: &Error) -> Option<String> {
    let command = severity_to_github_command(error.severity())?;
    let range = error.display_range();
    let file = github_actions_path(error.path().as_path());
    let params = format!(
        "file={},line={},col={},endLine={},endColumn={},title={}",
        escape_workflow_property(&file),
        range.start.line_within_file().get(),
        range.start.column().get(),
        range.end.line_within_file().get(),
        range.end.column().get(),
        escape_workflow_property(&format!("Pyrefly {}", error.error_kind().to_name())),
    );
    let message = escape_workflow_data(&error.msg());
    Some(format!("::{command} {params}::{message}"))
}

const WORKFLOW_DATA_ENCODE_SET: &AsciiSet = &CONTROLS.add(b'%');
const WORKFLOW_PROPERTY_ENCODE_SET: &AsciiSet = &WORKFLOW_DATA_ENCODE_SET.add(b':').add(b',');

fn github_actions_path(path: &Path) -> String {
    let mut path_str = path.to_string_lossy().into_owned();
    if std::path::MAIN_SEPARATOR != '/' {
        path_str = path_str.replace(std::path::MAIN_SEPARATOR, "/");
    }
    path_str
}

fn escape_workflow_data(value: &str) -> String {
    utf8_percent_encode(value, WORKFLOW_DATA_ENCODE_SET).to_string()
}

fn escape_workflow_property(value: &str) -> String {
    utf8_percent_encode(value, WORKFLOW_PROPERTY_ENCODE_SET).to_string()
}

/// A data structure to facilitate the creation of handles for all the files we want to check.
pub struct Handles {
    /// A mapping from a file to all other information needed to create a `Handle`.
    /// The value type is basically everything else in `Handle` except for the file path.
    path_data: HashSet<ModulePath>,
}

impl Handles {
    pub fn new(files: Vec<PathBuf>) -> Self {
        let mut handles = Self {
            path_data: HashSet::new(),
        };
        for file in files {
            handles.path_data.insert(ModulePath::filesystem(file));
        }
        handles
    }

    pub fn all(
        &self,
        config_finder: &ConfigFinder,
    ) -> (Vec<Handle>, SmallSet<ArcId<ConfigFile>>, Vec<ConfigError>) {
        let mut configs = SmallMap::new();
        for path in &self.path_data {
            let unknown = ModuleName::unknown();
            configs
                .entry(config_finder.python_file(ModuleNameWithKind::guaranteed(unknown), path))
                .or_insert_with(SmallSet::new)
                .insert(path.dupe());
        }

        // TODO(connernilsen): wire in force logic
        let reloaded_source_dbs = ConfigFile::query_source_db(&configs, false, None).0;
        let result = configs
            .iter()
            .flat_map(|(c, files)| files.iter().map(|p| c.handle_from_module_path(p.dupe())))
            .collect();
        let reloaded_configs = configs
            .into_iter()
            .map(|x| x.0)
            .filter(|c| {
                c.source_db
                    .as_ref()
                    .is_some_and(|db| reloaded_source_dbs.contains(db))
            })
            .collect();
        (result, reloaded_configs, Vec::new())
    }

    fn update<'a>(
        &mut self,
        created_files: impl Iterator<Item = &'a PathBuf>,
        removed_files: impl Iterator<Item = &'a PathBuf>,
    ) {
        for file in created_files {
            self.path_data
                .insert(ModulePath::filesystem(file.to_path_buf()));
        }
        for file in removed_files {
            self.path_data
                .remove(&ModulePath::filesystem(file.to_path_buf()));
        }
    }
}

async fn get_watcher_events(watcher: &mut Watcher) -> anyhow::Result<CategorizedEvents> {
    loop {
        let events = CategorizedEvents::new_notify(
            watcher
                .wait()
                .await
                .context("When waiting for watched files")?,
        );
        if !events.is_empty() {
            return Ok(events);
        }
        if !events.unknown.is_empty() {
            return Err(anyhow::anyhow!(
                "Cannot handle uncategorized watcher event on paths [{}]",
                display::commas_iter(|| events.unknown.iter().map(|x| x.display()))
            ));
        }
    }
}

/// Structure accumulating timing information.
struct Timings {
    /// The overall time we started.
    start: Instant,
    list_files: Duration,
    type_check: Duration,
    report_errors: Duration,
}

impl Display for Timings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const THRESHOLD: Duration = Duration::from_millis(100);
        let total = self.start.elapsed();
        write!(f, "{}", Self::show(total))?;

        let mut steps = Vec::with_capacity(3);

        // We want to show checking if it is less than total - threshold.
        // For the others, we want to show if they exceed threshold.
        if self.type_check + THRESHOLD < total {
            steps.push(("checking", self.type_check));
        }
        if self.report_errors > THRESHOLD {
            steps.push(("reporting", self.report_errors));
        }
        if self.list_files > THRESHOLD {
            steps.push(("listing", self.list_files));
        }
        if !steps.is_empty() {
            steps.sort_by_key(|x| x.1);
            write!(
                f,
                " ({})",
                display::intersperse_iter(", ", || steps
                    .iter()
                    .rev()
                    .map(|(lbl, dur)| format!("{lbl} {}", Self::show(*dur))))
            )?;
        }
        Ok(())
    }
}

impl Timings {
    fn new() -> Self {
        Self {
            start: Instant::now(),
            list_files: Duration::ZERO,
            type_check: Duration::ZERO,
            report_errors: Duration::ZERO,
        }
    }

    fn show(x: Duration) -> String {
        format!("{:.2}s", x.as_secs_f32())
    }
}

impl CheckArgs {
    /// Run a one-shot type check. Returns the exit status, the CLI-visible errors,
    /// and a `CheckResult` suitable for telemetry logging.
    pub fn run_once(
        mut self,
        files_to_check: Box<dyn Includes>,
        config_finder: ConfigFinder,
        thread_count: ThreadCount,
    ) -> anyhow::Result<(CommandExitStatus, Vec<Error>, CheckResult)> {
        let mut timings = Timings::new();
        let list_files_start = Instant::now();
        let expanded_file_list = config_finder.checkpoint(files_to_check.files())?;
        timings.list_files = list_files_start.elapsed();
        debug!(
            "Checking {} files (listing took {})",
            expanded_file_list.len(),
            Timings::show(timings.list_files),
        );
        if expanded_file_list.is_empty() {
            return Ok((
                CommandExitStatus::Success,
                Vec::new(),
                CheckResult {
                    legacy_errors: Vec::new(),
                    checked_file_count: 0,
                },
            ));
        }

        let state = Forgetter::new(State::new(config_finder, thread_count), true);
        let handles = Handles::new(expanded_file_list);
        let require_levels = self.get_required_levels();
        let mut transaction = Forgetter::new(
            state.as_ref().new_transaction(require_levels.default, None),
            true,
        );
        let (loaded_handles, _, sourcedb_errors) = handles.all(state.as_ref().config_finder());

        // Project-level output settings can come from config when CLI flags are absent.
        if (self.output.baseline.is_none()
            || self.output.output_format.is_none()
            || self.output.min_severity.is_none())
            && let Some(handle) = loaded_handles.first()
        {
            let config = state.as_ref().config_finder().python_file(
                ModuleNameWithKind::guaranteed(handle.module()),
                handle.path(),
            );
            self.output.inherit_defaults_from_config(&config);
        }

        let checked_file_count = loaded_handles.len();
        let relative_to = resolve_relative_to(self.output.relative_to.as_ref());
        let (status, errors) = self.run_inner(
            timings,
            transaction.as_mut(),
            &loaded_handles,
            sourcedb_errors,
            require_levels.specified,
        )?;
        let check_result = CheckResult::from_errors(&errors, &relative_to, checked_file_count);
        Ok((status, errors, check_result))
    }

    pub fn run_once_with_snippet(
        mut self,
        code: String,
        config_finder: ConfigFinder,
        thread_count: ThreadCount,
    ) -> anyhow::Result<(CommandExitStatus, CheckResult)> {
        // Create a virtual module path for the snippet
        let path = PathBuf::from_str("snippet")?;
        let module_path = ModulePath::memory(path);
        let module_name = ModuleName::from_str("__main__");

        let holder = Forgetter::new(State::new(config_finder, thread_count), true);

        // Create a single handle for the virtual module
        let config = holder
            .as_ref()
            .config_finder()
            .python_file(ModuleNameWithKind::guaranteed(module_name), &module_path);
        let sys_info = config.get_sys_info();
        let handle = Handle::new(module_name, module_path.clone(), sys_info);

        // Project-level output settings can come from config when CLI flags are absent.
        if self.output.baseline.is_none()
            || self.output.output_format.is_none()
            || self.output.min_severity.is_none()
        {
            self.output.inherit_defaults_from_config(&config);
        }

        let require_levels = self.get_required_levels();
        let mut transaction = Forgetter::new(
            holder
                .as_ref()
                .new_transaction(require_levels.default, None),
            true,
        );

        // Add the snippet source to the transaction's memory
        transaction.as_mut().set_memory(vec![(
            PathBuf::from(module_path.as_path()),
            Some(Arc::new(FileContents::from_source(code))),
        )]);

        let relative_to = resolve_relative_to(self.output.relative_to.as_ref());
        let (status, errors) = self.run_inner(
            Timings::new(),
            transaction.as_mut(),
            &[handle],
            vec![],
            require_levels.specified,
        )?;
        Ok((status, CheckResult::from_errors(&errors, &relative_to, 1)))
    }

    pub async fn run_watch(
        mut self,
        mut watcher: Watcher,
        files_to_check: Box<dyn Includes>,
        config_finder: ConfigFinder,
        thread_count: ThreadCount,
    ) -> anyhow::Result<()> {
        // TODO: We currently make 1 unrealistic assumptions, which should be fixed in the future:
        // - Config search is stable across incremental runs.
        let expanded_file_list = config_finder.checkpoint(files_to_check.files())?;
        let require_levels = self.get_required_levels();
        let mut handles = Handles::new(expanded_file_list);
        let state = State::new(config_finder, thread_count);

        // Track which output settings were explicitly set on the CLI.
        let cli_provided_baseline = self.output.baseline.is_some();
        let cli_provided_min_severity = self.output.min_severity.is_some();
        let cli_provided_output_format = self.output.output_format.is_some();

        let mut transaction = state.new_committable_transaction(require_levels.default, None);
        loop {
            let timings = Timings::new();
            let (loaded_handles, reloaded_configs, sourcedb_errors) =
                handles.all(state.config_finder());

            // Inherit project-level output settings from config on every iteration
            // to pick up config file changes when the CLI did not override them.
            // Reset non-CLI-provided fields first so updated config values are applied.
            if (!cli_provided_baseline || !cli_provided_output_format || !cli_provided_min_severity)
                && let Some(handle) = loaded_handles.first()
            {
                if !cli_provided_baseline {
                    self.output.baseline = None;
                }
                if !cli_provided_output_format {
                    self.output.output_format = None;
                }
                if !cli_provided_min_severity {
                    self.output.min_severity = None;
                }
                let config = state.config_finder().python_file(
                    ModuleNameWithKind::guaranteed(handle.module()),
                    handle.path(),
                );
                self.output.inherit_defaults_from_config(&config);
            }
            let mut_transaction = transaction.as_mut();
            mut_transaction.invalidate_find_for_configs(reloaded_configs);
            let res = self.run_inner(
                timings,
                mut_transaction,
                &loaded_handles,
                sourcedb_errors,
                require_levels.specified,
            );
            state.commit_transaction(transaction, None);
            if let Err(e) = res {
                eprintln!("{e:#}");
            }
            let events = get_watcher_events(&mut watcher).await?;
            transaction = state.new_committable_transaction(
                require_levels.default,
                self.output.progress_bar_style().make_subscriber(),
            );
            let new_transaction_mut = transaction.as_mut();
            new_transaction_mut.invalidate_events(&events);
            // File addition and removal may affect the list of files/handles to check. Update
            // the handles accordingly.
            handles.update(
                events.created.iter().filter(|p| files_to_check.covers(p)),
                events.removed.iter().filter(|p| files_to_check.covers(p)),
            );
        }
    }

    fn get_required_levels(&self) -> RequireLevels {
        let retain = self.output.report_binding_memory.is_some()
            || self.output.debug_info.is_some()
            || self.output.report_trace.is_some()
            || self.output.report_glean.is_some();
        RequireLevels {
            specified: if retain {
                Require::Everything
            } else {
                Require::Errors
            },
            default: if retain {
                Require::Everything
            } else if self.behavior.check_all
                || stdlib_search_path().is_some()
                || self.output.report_pysa.is_some()
                || self.output.report_cinderx.is_some()
            {
                Require::Errors
            } else {
                Require::Exports
            },
        }
    }

    fn run_inner(
        &self,
        mut timings: Timings,
        transaction: &mut Transaction,
        handles: &[Handle],
        mut sourcedb_errors: Vec<ConfigError>,
        require: Require,
    ) -> anyhow::Result<(CommandExitStatus, Vec<Error>)> {
        let mut memory_trace = MemoryUsageTrace::start(Duration::from_secs_f32(0.1));

        if let Some(pysa_directory) = &self.output.report_pysa {
            let reporter = report::pysa::PysaReporter::new(
                pysa_directory,
                handles,
                self.output.report_pysa_format,
            )?;
            transaction.set_pysa_reporter(Some(reporter));
        }
        if let Some(cinderx_directory) = &self.output.report_cinderx {
            let cinderx_reporter = if self.output.cinderx_include_deps {
                report::cinderx::CinderxReporter::new(
                    cinderx_directory,
                    None,
                    self.output.cinderx_include_readable,
                )?
            } else {
                report::cinderx::CinderxReporter::new(
                    cinderx_directory,
                    Some(handles),
                    self.output.cinderx_include_readable,
                )?
            };
            transaction.set_cinderx_reporter(Some(cinderx_reporter));
        }

        let type_check_start = Instant::now();
        transaction.set_subscriber(self.output.progress_bar_style().make_subscriber());
        transaction.run(handles, require, None);
        transaction.set_subscriber(None);

        let loads = if self.behavior.check_all {
            transaction.get_all_errors()
        } else {
            transaction.get_errors(handles)
        };
        timings.type_check = type_check_start.elapsed();

        let report_errors_start = Instant::now();
        let mut config_errors = transaction.get_config_errors();
        config_errors.append(&mut sourcedb_errors);
        let mut config_errors_count = 0;
        for error in config_errors {
            error.print();
            if error.severity() >= Severity::Error {
                config_errors_count += 1;
            }
        }

        let relative_to = self.output.relative_to.as_ref().map_or_else(
            || std::env::current_dir().ok().unwrap_or_default(),
            |x| PathBuf::from_str(x.as_str()).unwrap(),
        );
        let output_format = self.output.output_format();

        let collected = loads.collect_errors();
        // Pass pre-collected errors to avoid redundant error collection.
        let unused_ignore_errors = loads.collect_unused_ignore_errors_for_display(&collected);
        let errors = loads.apply_baseline(
            collected,
            self.output.baseline.as_deref(),
            relative_to.as_path(),
        );
        let (directives, ordinary_errors) = if let Some(only) = &self.output.only {
            let only = only.iter().collect::<SmallSet<_>>();
            (
                errors
                    .directives
                    .into_iter()
                    .filter(|e| only.contains(&e.error_kind()))
                    .collect(),
                errors
                    .ordinary
                    .into_iter()
                    .filter(|e| only.contains(&e.error_kind()))
                    .collect(),
            )
        } else {
            (errors.directives, errors.ordinary)
        };
        let ordinary_errors: Vec<_> = if let Some(only) = &self.output.only {
            let only = only.iter().collect::<SmallSet<_>>();
            let filtered: Vec<_> = unused_ignore_errors
                .ordinary
                .into_iter()
                .filter(|e| only.contains(&e.error_kind()))
                .collect();
            ordinary_errors.into_iter().chain(filtered).collect()
        } else {
            ordinary_errors
                .into_iter()
                .chain(unused_ignore_errors.ordinary)
                .collect()
        };

        // Suppress operates on ordinary diagnostics only — directives are
        // structurally excluded since they live in `directives`, not `ordinary_errors`.
        if self.behavior.suppress_errors {
            // TODO: Deprecate this in favor of `pyrefly suppress`
            let serialized_errors: Vec<SerializedError> = ordinary_errors
                .iter()
                .filter_map(SerializedError::from_error)
                .filter(|e| !e.is_unused_ignore())
                .collect();
            suppress::suppress_errors(serialized_errors, CommentLocation::LineBefore);
        }
        if self.behavior.remove_unused_ignores {
            // TODO: Deprecate this in favor of `pyrefly suppress`
            let collected = loads.collect_errors();
            let unused_errors = loads.collect_unused_ignore_errors(&collected);
            suppress::remove_unused_ignores(unused_errors);
        }

        // Filter by minimum severity. Directives are not subject to this
        // filter — they are merged separately in the output step below.
        let min_severity = self.output.min_severity.unwrap_or(Severity::Error);
        let (ordinary_errors, hidden_errors): (Vec<_>, Vec<_>) = ordinary_errors
            .into_iter()
            .partition(|e| e.severity() >= min_severity);

        // We update the baseline file if requested, after reporting any new
        // errors using the old baseline. Directives are structurally excluded
        // — they live in `directives`, not `ordinary_errors`. The baseline only
        // tracks errors that meet the min-severity threshold.
        if self.output.update_baseline
            && let Some(baseline_path) = &self.output.baseline
        {
            let mut new_baseline = ordinary_errors.clone();
            new_baseline.extend(
                errors
                    .baseline
                    .into_iter()
                    .filter(|e| e.severity() >= min_severity),
            );
            new_baseline.sort_by_cached_key(|error| {
                (
                    error.path().to_string(),
                    error.range().start(),
                    error.range().end(),
                    error.error_kind(),
                )
            });
            write_error_json_to_file(baseline_path, relative_to.as_path(), &new_baseline)?;
        }

        // Count only ordinary errors for exit code determination. Directives
        // (e.g. reveal_type) do not contribute to the error count.
        let mut ordinary_errors_count = config_errors_count;
        for error in &ordinary_errors {
            if error.severity() >= Severity::Error {
                ordinary_errors_count += 1;
            }
        }

        // Merge directives into the display list, re-sorting by module
        // name, path, and source range so output preserves file/line
        // interleaving across modules.
        let mut output_errors = ordinary_errors;
        output_errors.extend(directives);
        output_errors.sort_by_cached_key(|e| {
            (
                e.module().name(),
                e.path().dupe(),
                e.range().start(),
                e.range().end(),
            )
        });

        if let Some(path) = &self.output.output {
            write_errors_to_file(output_format, path, relative_to.as_path(), &output_errors)?;
        } else {
            write_errors_to_console(output_format, relative_to.as_path(), &output_errors)?;
        }
        memory_trace.stop();
        if let Some(limit) = self.output.count_errors {
            print_error_counts(&output_errors, limit);
        }
        if self.output.summarize_errors.is_some() {
            print_error_summary(&output_errors);
        }
        timings.report_errors = report_errors_start.elapsed();

        if self.output.summary != Summary::None {
            let suppress_count = errors.suppressed.len();
            let mut parts = vec![count(ordinary_errors_count, "error")];
            if suppress_count > 0 {
                parts.push(format!("{} suppressed", number_thousands(suppress_count)));
            }
            if !hidden_errors.is_empty() {
                let mut hidden_warnings = 0;
                let mut hidden_info = 0;
                for e in hidden_errors {
                    match e.severity() {
                        Severity::Error => panic!("Error-level findings can never be hidden"),
                        Severity::Warn => hidden_warnings += 1,
                        Severity::Info => hidden_info += 1,
                        Severity::Ignore => {}
                    }
                }
                let mut hidden_parts = Vec::new();
                if hidden_warnings > 0 {
                    hidden_parts.push(count(hidden_warnings, "warning"));
                }
                if hidden_info > 0 {
                    hidden_parts.push(count(hidden_info, "info message"));
                }
                parts.push(format!("{} not shown", hidden_parts.join(" and ")));
            }
            if parts.len() == 1 {
                info!("{}", parts[0]);
            } else {
                info!("{} ({})", parts[0], parts[1..].join(", "));
            }
        }
        if self.output.summary == Summary::Full {
            let user_handles: HashSet<&Handle> = handles.iter().collect();
            let (user_lines, dep_lines) = transaction.split_line_count(&user_handles);
            info!(
                "{} ({}); {} ({} in your project, {} in dependencies); \
                took {timings}; memory ({})",
                count(handles.len(), "module"),
                count(
                    transaction.module_count() - handles.len(),
                    "dependent module"
                ),
                count(user_lines + dep_lines, "line"),
                count(user_lines, "line"),
                count(dep_lines, "line"),
                memory_trace.peak()
            );
        }
        if let Some(output_path) = &self.output.report_timings {
            eprintln!("Computing timing information");
            transaction.set_subscriber(self.output.progress_bar_style().make_subscriber());
            transaction.report_timings(output_path)?;
            transaction.set_subscriber(None);
        }
        if let Some(debug_info) = &self.output.debug_info {
            let is_javascript = debug_info.extension() == Some("js".as_ref());
            fs_anyhow::write(
                debug_info,
                report::debug_info::debug_info(transaction, handles, is_javascript),
            )?;
        }
        if let Some(glean) = &self.output.report_glean {
            fs_anyhow::create_dir_all(glean)?;
            for handle in handles {
                // Generate a safe filename using hash to avoid OS filename length limits
                let module_hash = blake3::hash(handle.path().to_string().as_bytes());
                fs_anyhow::write(
                    &glean.join(format!("{}.json", &module_hash)),
                    report::glean::glean(transaction, handle),
                )?;
            }
        }
        if let Some(pysa_reporter) = transaction.take_pysa_reporter() {
            report::pysa::write_project_file(&pysa_reporter, transaction, handles, &output_errors)?;
        }
        if let Some(cinderx_reporter) = transaction.take_cinderx_reporter() {
            cinderx_reporter.write_project_files(transaction)?;
        }
        if let Some(path) = &self.output.report_binding_memory {
            fs_anyhow::write(path, report::binding_memory::binding_memory(transaction))?;
        }
        if let Some(path) = &self.output.report_trace {
            fs_anyhow::write(path, report::trace::trace(transaction))?;
        }
        if let Some(path) = &self.output.dependency_graph {
            fs_anyhow::write(
                path,
                report::dependency_graph::dependency_graph(transaction, handles),
            )?;
        }
        if self.behavior.expectations {
            loads.check_against_expectations()?;
            Ok((CommandExitStatus::Success, output_errors))
        } else if ordinary_errors_count > 0 {
            Ok((CommandExitStatus::UserError, output_errors))
        } else {
            Ok((CommandExitStatus::Success, output_errors))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use pyrefly_python::module::Module;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use ruff_text_size::TextRange;
    use ruff_text_size::TextSize;
    use vec1::Vec1;
    use vec1::vec1;

    use super::*;

    fn sample_error(msg: Vec1<String>) -> Error {
        let module = Module::new(
            ModuleName::from_str("sample"),
            ModulePath::filesystem(PathBuf::from("/repo/foo.py")),
            Arc::new("x = 1\n".to_owned()),
        );
        Error::new(
            module,
            TextRange::new(TextSize::from(0), TextSize::from(1)),
            msg,
            ErrorKind::BadAssignment,
        )
    }

    #[test]
    fn github_actions_command_includes_full_path_and_metadata() {
        let cmd = github_actions_command(&sample_error(vec1!["bad".into()]))
            .expect("should emit command");
        assert!(cmd.starts_with("::error "), "{cmd}");
        assert!(
            cmd.contains("file=/repo/foo.py"),
            "full path expected, got {cmd}"
        );
        assert!(
            cmd.contains("title=Pyrefly bad-assignment"),
            "title missing, got {cmd}"
        );
        assert!(cmd.ends_with("::bad"));
    }

    #[test]
    fn github_actions_command_respects_severity_mapping() {
        let warning = sample_error(vec1!["bad".into()]).with_severity(Severity::Warn);
        let notice = sample_error(vec1!["bad".into()]).with_severity(Severity::Info);
        let ignored = sample_error(vec1!["bad".into()]).with_severity(Severity::Ignore);
        assert!(
            github_actions_command(&warning)
                .unwrap()
                .starts_with("::warning "),
            "warning severity not mapped"
        );
        assert!(
            github_actions_command(&notice)
                .unwrap()
                .starts_with("::notice "),
            "info severity not mapped"
        );
        assert!(github_actions_command(&ignored).is_none());
    }

    #[test]
    fn escape_helpers_follow_workflow_spec() {
        assert_eq!(
            escape_workflow_data("line1\nline2\r% done"),
            "line1%0Aline2%0D%25 done"
        );
        assert_eq!(escape_workflow_property("file:name,py"), "file%3Aname%2Cpy");
    }

    #[test]
    fn github_output_format_writes_commands() {
        let errors = vec![sample_error(vec1!["bad".into()])];
        let mut buf = Vec::new();
        write_error_github(&mut buf, &errors).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("::error file=/repo/foo.py"));
        assert!(output.ends_with("::bad\n"));
    }

    #[test]
    fn output_args_inherit_output_format_from_config() {
        let mut output = OutputArgs::parse_from(["pyrefly-check"]);
        let config = ConfigFile {
            output_format: Some(OutputFormat::MinText),
            ..Default::default()
        };

        output.inherit_defaults_from_config(&config);

        assert_eq!(output.output_format(), OutputFormat::MinText);
    }

    #[test]
    fn cli_output_format_overrides_config_output_format() {
        let mut output = OutputArgs::parse_from(["pyrefly-check", "--output-format", "json"]);
        let config = ConfigFile {
            output_format: Some(OutputFormat::MinText),
            ..Default::default()
        };

        output.inherit_defaults_from_config(&config);

        assert_eq!(output.output_format(), OutputFormat::Json);
    }
}
