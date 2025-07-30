/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
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
use dupe::Dupe;
use pyrefly_config::args::ConfigOverrideArgs;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::display;
use pyrefly_util::display::count;
use pyrefly_util::display::number_thousands;
use pyrefly_util::events::CategorizedEvents;
use pyrefly_util::forgetter::Forgetter;
use pyrefly_util::fs_anyhow;
use pyrefly_util::globs::FilteredGlobs;
use pyrefly_util::lined_buffer::LineNumber;
use pyrefly_util::memory::MemoryUsageTrace;
use pyrefly_util::prelude::SliceExt;
use pyrefly_util::watcher::Watcher;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use tracing::debug;
use tracing::info;

use crate::commands::files::FilesArgs;
use crate::commands::util::CommandExitStatus;
use crate::config::error_kind::Severity;
use crate::config::finder::ConfigFinder;
use crate::error::error::Error;
use crate::error::error::print_error_counts;
use crate::error::legacy::LegacyErrors;
use crate::error::summarise::print_error_summary;
use crate::error::suppress;
use crate::module::from_path::module_from_path;
use crate::module::typeshed::stdlib_search_path;
use crate::report;
use crate::state::handle::Handle;
use crate::state::require::Require;
use crate::state::state::State;
use crate::state::state::Transaction;
use crate::state::subscriber::ProgressBarSubscriber;

/// Check the given files.
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Clone, Parser)]
pub struct FullCheckArgs {
    /// Which files to check.
    #[command(flatten)]
    pub files: FilesArgs,

    /// Watch for file changes and re-check them.
    #[arg(long, conflicts_with = "check_all")]
    watch: bool,

    /// Type checking arguments and configuration
    #[command(flatten)]
    pub args: CheckArgs,
}

impl FullCheckArgs {
    pub async fn run(self, allow_forget: bool) -> anyhow::Result<CommandExitStatus> {
        self.args.config_override.validate()?;
        let (files_to_check, config_finder) = self.files.resolve(&self.args.config_override)?;
        run_check(
            self.args,
            self.watch,
            files_to_check,
            config_finder,
            allow_forget,
        )
        .await
    }
}

async fn run_check(
    args: CheckArgs,
    watch: bool,
    files_to_check: FilteredGlobs,
    config_finder: ConfigFinder,
    allow_forget: bool,
) -> anyhow::Result<CommandExitStatus> {
    if watch {
        let roots = files_to_check.roots();
        info!(
            "Watching for files in {}",
            display::intersperse_iter(";", || roots.iter().map(|p| p.display()))
        );
        let watcher = Watcher::notify(&roots)?;
        args.run_watch(watcher, files_to_check, config_finder)
            .await?;
        Ok(CommandExitStatus::Success)
    } else {
        match args.run_once(files_to_check, config_finder, allow_forget) {
            Ok((status, _)) => Ok(status),
            Err(e) => Err(e),
        }
    }
}

#[derive(Debug, Clone, ValueEnum, Default)]
enum OutputFormat {
    /// Minimal text output, one line per error
    MinText,
    #[default]
    /// Full, verbose text output
    FullText,
    /// JSON output
    Json,
    /// Only show error count, omitting individual errors
    OmitErrors,
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
    /// Configuration override options
    #[command(flatten, next_help_heading = "Config Overrides")]
    pub config_override: ConfigOverrideArgs,
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
    #[arg(long, short, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Output related configuration options
    #[command(flatten, next_help_heading = "Output")]
    output: OutputArgs,
    /// Configuration override options
    #[command(flatten, next_help_heading = "Config Overrides")]
    pub config_override: ConfigOverrideArgs,
}

impl SnippetCheckArgs {
    pub async fn run(self, allow_forget: bool) -> anyhow::Result<CommandExitStatus> {
        let (_, config_finder) = FilesArgs::get(vec![], self.config, &self.config_override)?;
        let check_args = CheckArgs {
            output: self.output,
            behavior: BehaviorArgs {
                check_all: false,
                suppress_errors: false,
                expectations: false,
                remove_unused_ignores: false,
            },
            config_override: self.config_override,
        };
        match check_args.run_once_with_snippet(self.code, config_finder, allow_forget) {
            Ok((status, _)) => Ok(status),
            Err(e) => Err(e),
        }
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
    #[arg(long, value_enum, default_value_t)]
    output_format: OutputFormat,
    /// Produce debugging information about the type checking process.
    #[arg(long, value_name = "OUTPUT_FILE")]
    debug_info: Option<PathBuf>,
    /// Report the memory usage of bindings.
    #[arg(long, value_name = "OUTPUT_FILE")]
    report_binding_memory: Option<PathBuf>,
    /// Report type traces.
    #[arg(long, value_name = "OUTPUT_FILE")]
    report_trace: Option<PathBuf>,
    /// Process each module individually to figure out how long each step takes.
    #[arg(long, value_name = "OUTPUT_FILE")]
    report_timings: Option<PathBuf>,
    /// Generate a Glean-compatible JSON file for each module
    #[arg(long, value_name = "OUTPUT_FILE")]
    report_glean: Option<PathBuf>,
    /// Generate a Pysa-compatible JSON file for each module
    #[arg(long, value_name = "OUTPUT_FILE")]
    report_pysa: Option<PathBuf>,
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

    /// By default show the number of errors. Pass `--summary` to show information about lines checked and time/memory,
    /// or `--summary=none` to hide the summary line entirely.
    #[arg(
        long,
        default_missing_value = "full",
        require_equals = true,
        num_args = 0..=1,
        value_enum,
        default_value_t
    )]
    summary: Summary,

    /// Omit the summary in the last line of the output.
    /// Deprecated: will be removed in the next release. Use `--summary=none` instead.
    #[arg(long)]
    no_summary: bool,
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

impl OutputFormat {
    fn write_error_text_to_file(
        path: &Path,
        errors: &[Error],
        verbose: bool,
    ) -> anyhow::Result<()> {
        let mut file = BufWriter::new(File::create(path)?);
        for e in errors {
            e.write_line(&mut file, verbose)?;
        }
        file.flush()?;
        Ok(())
    }

    fn write_error_text_to_console(errors: &[Error], verbose: bool) -> anyhow::Result<()> {
        for error in errors {
            error.print_colors(verbose);
        }
        Ok(())
    }

    fn write_error_json(writer: &mut impl Write, errors: &[Error]) -> anyhow::Result<()> {
        let legacy_errors = LegacyErrors::from_errors(errors);
        serde_json::to_writer_pretty(writer, &legacy_errors)?;
        Ok(())
    }

    fn buffered_write_error_json(writer: impl Write, errors: &[Error]) -> anyhow::Result<()> {
        let mut writer = BufWriter::new(writer);
        Self::write_error_json(&mut writer, errors)?;
        writer.flush()?;
        Ok(())
    }

    fn write_error_json_to_file(path: &Path, errors: &[Error]) -> anyhow::Result<()> {
        fn f(path: &Path, errors: &[Error]) -> anyhow::Result<()> {
            let file = File::create(path)?;
            OutputFormat::buffered_write_error_json(file, errors)
        }
        f(path, errors)
            .with_context(|| format!("while writing JSON errors to `{}`", path.display()))
    }

    fn write_error_json_to_console(errors: &[Error]) -> anyhow::Result<()> {
        Self::buffered_write_error_json(stdout(), errors)
    }

    fn write_errors_to_file(&self, path: &Path, errors: &[Error]) -> anyhow::Result<()> {
        match self {
            Self::MinText => Self::write_error_text_to_file(path, errors, false),
            Self::FullText => Self::write_error_text_to_file(path, errors, true),
            Self::Json => Self::write_error_json_to_file(path, errors),
            Self::OmitErrors => Ok(()),
        }
    }

    fn write_errors_to_console(&self, errors: &[Error]) -> anyhow::Result<()> {
        match self {
            Self::MinText => Self::write_error_text_to_console(errors, false),
            Self::FullText => Self::write_error_text_to_console(errors, true),
            Self::Json => Self::write_error_json_to_console(errors),
            Self::OmitErrors => Ok(()),
        }
    }
}

/// A data structure to facilitate the creation of handles for all the files we want to check.
pub struct Handles {
    /// A mapping from a file to all other information needed to create a `Handle`.
    /// The value type is basically everything else in `Handle` except for the file path.
    path_data: HashMap<PathBuf, (ModuleName, SysInfo)>,
}

impl Handles {
    pub fn new(files: Vec<PathBuf>, config_finder: &ConfigFinder) -> Self {
        let mut handles = Self {
            path_data: HashMap::new(),
        };
        for file in files {
            handles.register_file(file, config_finder);
        }
        handles
    }

    fn register_file(
        &mut self,
        path: PathBuf,
        config_finder: &ConfigFinder,
    ) -> &(ModuleName, SysInfo) {
        let module_path = ModulePath::filesystem(path.clone());
        let unknown = ModuleName::unknown();
        let config = config_finder.python_file(unknown, &module_path);

        let search_path = config.search_path();
        let module_name = module_from_path(&path, search_path).unwrap_or(unknown);

        self.path_data
            .entry(path)
            .or_insert((module_name, config.get_sys_info()))
    }

    pub fn all(&self, specified_require: Require) -> Vec<(Handle, Require)> {
        self.path_data
            .iter()
            .map(|(path, (module_name, runtime_metadata))| {
                (
                    Handle::new(
                        module_name.dupe(),
                        ModulePath::filesystem(path.to_path_buf()),
                        runtime_metadata.dupe(),
                    ),
                    specified_require,
                )
            })
            .collect()
    }

    fn update<'a>(
        &mut self,
        created_files: impl Iterator<Item = &'a PathBuf>,
        removed_files: impl Iterator<Item = &'a PathBuf>,
        config_finder: &ConfigFinder,
    ) {
        for file in created_files {
            self.register_file(file.to_path_buf(), config_finder);
        }
        for file in removed_files {
            self.path_data.remove(file);
        }
    }
}

struct RequireLevels {
    specified: Require,
    default: Require,
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
    pub fn get_handles(
        self,
        files_to_check: FilteredGlobs,
        config_finder: &ConfigFinder,
    ) -> anyhow::Result<Vec<(Handle, Require)>> {
        let handles = Handles::new(
            config_finder.checkpoint(files_to_check.files())?,
            config_finder,
        );
        Ok(handles.all(self.get_required_levels().specified))
    }

    pub fn run_once(
        self,
        files_to_check: FilteredGlobs,
        config_finder: ConfigFinder,
        allow_forget: bool,
    ) -> anyhow::Result<(CommandExitStatus, Vec<Error>)> {
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
            return Ok((CommandExitStatus::Success, Vec::new()));
        }

        let holder = Forgetter::new(State::new(config_finder), allow_forget);
        let handles = Handles::new(expanded_file_list, holder.as_ref().config_finder());
        let require_levels = self.get_required_levels();
        let mut transaction = Forgetter::new(
            holder
                .as_ref()
                .new_transaction(require_levels.default, None),
            allow_forget,
        );
        self.run_inner(
            timings,
            transaction.as_mut(),
            &handles.all(require_levels.specified),
        )
    }

    pub fn run_once_with_snippet(
        self,
        code: String,
        config_finder: ConfigFinder,
        allow_forget: bool,
    ) -> anyhow::Result<(CommandExitStatus, Vec<Error>)> {
        // Create a virtual module path for the snippet
        let path = PathBuf::from_str("snippet")?;
        let module_path = ModulePath::memory(path);
        let module_name = ModuleName::from_str("__main__");

        let holder = Forgetter::new(State::new(config_finder), allow_forget);

        // Create a single handle for the virtual module
        let sys_info = holder
            .as_ref()
            .config_finder()
            .python_file(module_name, &module_path)
            .get_sys_info();
        let handle = Handle::new(module_name, module_path.clone(), sys_info);

        let require_levels = self.get_required_levels();
        let mut transaction = Forgetter::new(
            holder
                .as_ref()
                .new_transaction(require_levels.default, None),
            allow_forget,
        );

        // Add the snippet source to the transaction's memory
        transaction.as_mut().set_memory(vec![(
            PathBuf::from(module_path.as_path()),
            Some(Arc::new(code)),
        )]);

        self.run_inner(
            Timings::new(),
            transaction.as_mut(),
            &[(handle, require_levels.specified)],
        )
    }

    pub async fn run_watch(
        self,
        mut watcher: Watcher,
        files_to_check: FilteredGlobs,
        config_finder: ConfigFinder,
    ) -> anyhow::Result<()> {
        // TODO: We currently make 1 unrealistic assumptions, which should be fixed in the future:
        // - Config search is stable across incremental runs.
        let expanded_file_list = config_finder.checkpoint(files_to_check.files())?;
        let require_levels = self.get_required_levels();
        let mut handles = Handles::new(expanded_file_list, &config_finder);
        let state = State::new(config_finder);
        let mut transaction = state.new_committable_transaction(require_levels.default, None);
        loop {
            let timings = Timings::new();
            let res = self.run_inner(
                timings,
                transaction.as_mut(),
                &handles.all(require_levels.specified),
            );
            state.commit_transaction(transaction);
            if let Err(e) = res {
                eprintln!("{e:#}");
            }
            let events = get_watcher_events(&mut watcher).await?;
            transaction = state.new_committable_transaction(
                require_levels.default,
                Some(Box::new(ProgressBarSubscriber::new())),
            );
            let new_transaction_mut = transaction.as_mut();
            new_transaction_mut.invalidate_events(&events);
            // File addition and removal may affect the list of files/handles to check. Update
            // the handles accordingly.
            handles.update(
                events.created.iter().filter(|p| files_to_check.covers(p)),
                events.removed.iter().filter(|p| files_to_check.covers(p)),
                state.config_finder(),
            );
        }
    }

    fn get_required_levels(&self) -> RequireLevels {
        let retain = self.output.report_binding_memory.is_some()
            || self.output.debug_info.is_some()
            || self.output.report_trace.is_some()
            || self.output.report_glean.is_some()
            || self.output.report_pysa.is_some();
        RequireLevels {
            specified: if retain {
                Require::Everything
            } else {
                Require::Errors
            },
            default: if retain {
                Require::Everything
            } else if self.behavior.check_all || stdlib_search_path().is_some() {
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
        handles: &[(Handle, Require)],
    ) -> anyhow::Result<(CommandExitStatus, Vec<Error>)> {
        let mut memory_trace = MemoryUsageTrace::start(Duration::from_secs_f32(0.1));

        let type_check_start = Instant::now();
        transaction.set_subscriber(Some(Box::new(ProgressBarSubscriber::new())));
        transaction.run(handles);
        transaction.set_subscriber(None);

        let loads = if self.behavior.check_all {
            transaction.get_all_errors()
        } else {
            transaction.get_errors(handles.iter().map(|(handle, _)| handle))
        };
        timings.type_check = type_check_start.elapsed();

        let report_errors_start = Instant::now();
        let config_errors = transaction.get_config_errors();
        let config_errors_count = config_errors.len();
        for error in config_errors {
            error.print();
        }

        let errors = loads.collect_errors();
        if let Some(path) = &self.output.output {
            self.output
                .output_format
                .write_errors_to_file(path, &errors.shown)?;
        } else {
            self.output
                .output_format
                .write_errors_to_console(&errors.shown)?;
        }
        memory_trace.stop();
        if let Some(limit) = self.output.count_errors {
            print_error_counts(&errors.shown, limit);
        }
        if let Some(path_index) = self.output.summarize_errors {
            print_error_summary(&errors.shown, path_index);
        }
        let mut shown_errors_count = config_errors_count;
        for error in &errors.shown {
            if error.severity() >= Severity::Error {
                shown_errors_count += 1;
            }
        }
        timings.report_errors = report_errors_start.elapsed();

        if self.output.summary != Summary::None && !self.output.no_summary {
            let ignored = errors.disabled.len() + errors.suppressed.len();
            if ignored == 0 {
                info!("{}", count(shown_errors_count, "error"))
            } else {
                info!(
                    "{} ({} ignored)",
                    count(shown_errors_count, "error"),
                    number_thousands(ignored)
                )
            };
        }
        if self.output.summary == Summary::Full {
            info!(
                "{} ({}, {}); took {timings}; memory ({})",
                count(handles.len(), "module"),
                count(
                    transaction.module_count() - handles.len(),
                    "dependent module"
                ),
                count(transaction.line_count(), "line"),
                memory_trace.peak()
            );
        }
        if let Some(output_path) = &self.output.report_timings {
            eprintln!("Computing timing information");
            transaction.set_subscriber(Some(Box::new(ProgressBarSubscriber::new())));
            transaction.report_timings(output_path)?;
            transaction.set_subscriber(None);
        }
        if let Some(debug_info) = &self.output.debug_info {
            let is_javascript = debug_info.extension() == Some("js".as_ref());
            fs_anyhow::write(
                debug_info,
                report::debug_info::debug_info(
                    transaction,
                    &handles.map(|x| x.0.dupe()),
                    is_javascript,
                ),
            )?;
        }
        if let Some(glean) = &self.output.report_glean {
            fs_anyhow::create_dir_all(glean)?;
            for (handle, _) in handles {
                fs_anyhow::write(
                    &glean.join(format!("{}.json", handle.module())),
                    report::glean::glean(transaction, handle),
                )?;
            }
        }
        if let Some(pysa_directory) = &self.output.report_pysa {
            report::pysa::write_results(pysa_directory, transaction)?;
        }
        if let Some(path) = &self.output.report_binding_memory {
            fs_anyhow::write(path, report::binding_memory::binding_memory(transaction))?;
        }
        if let Some(path) = &self.output.report_trace {
            fs_anyhow::write(path, report::trace::trace(transaction))?;
        }
        if self.behavior.suppress_errors {
            let mut errors_to_suppress: SmallMap<PathBuf, Vec<Error>> = SmallMap::new();
            let shown_errors = errors.shown.clone();
            for e in shown_errors {
                if e.severity() >= Severity::Warn
                    && let ModulePathDetails::FileSystem(path) = e.path().details()
                {
                    errors_to_suppress.entry(path.clone()).or_default().push(e);
                }
            }
            suppress::suppress_errors(&errors_to_suppress);
        }
        if self.behavior.remove_unused_ignores {
            let mut all_ignores: SmallMap<&PathBuf, SmallSet<LineNumber>> = SmallMap::new();
            for (module_path, ignore) in loads.collect_ignores() {
                if let ModulePathDetails::FileSystem(path) = module_path.details() {
                    all_ignores.insert(path, ignore.get_pyrefly_ignores());
                }
            }

            let mut suppressed_errors: SmallMap<&PathBuf, SmallSet<LineNumber>> = SmallMap::new();
            for e in &errors.suppressed {
                if e.is_ignored(false)
                    && let ModulePathDetails::FileSystem(path) = e.path().details()
                {
                    suppressed_errors
                        .entry(path)
                        .or_default()
                        .insert(e.display_range().start.line);
                }
            }

            let unused_ignores = suppress::find_unused_ignores(all_ignores, suppressed_errors);
            suppress::remove_unused_ignores(unused_ignores);
        }
        if self.behavior.expectations {
            loads.check_against_expectations()?;
            Ok((CommandExitStatus::Success, errors.shown))
        } else if shown_errors_count > 0 {
            Ok((CommandExitStatus::UserError, errors.shown))
        } else {
            Ok((CommandExitStatus::Success, errors.shown))
        }
    }
}
