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
use pyrefly_build::handle::Handle;
use pyrefly_config::args::ConfigOverrideArgs;
use pyrefly_config::config::ConfigFile;
use pyrefly_config::error_kind::ErrorKind;
use pyrefly_config::finder::ConfigError;
use pyrefly_python::module_name::ModuleName;
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
use pyrefly_util::watcher::Watcher;
use ruff_text_size::Ranged;
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
use crate::error::summarize::print_error_summary;
use crate::error::suppress;
use crate::module::typeshed::stdlib_search_path;
use crate::report;
use crate::state::load::FileContents;
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
    pub async fn run(self) -> anyhow::Result<CommandExitStatus> {
        self.config_override.validate()?;
        let (files_to_check, config_finder) = self.files.resolve(self.config_override)?;
        run_check(self.args, self.watch, files_to_check, config_finder).await
    }
}

async fn run_check(
    args: CheckArgs,
    watch: bool,
    files_to_check: Box<dyn Includes>,
    config_finder: ConfigFinder,
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
        match args.run_once(files_to_check, config_finder) {
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
    pub async fn run(self) -> anyhow::Result<CommandExitStatus> {
        let (_, config_finder) = FilesArgs::get(vec![], self.config, self.config_override)?;
        let check_args = CheckArgs {
            output: self.output,
            behavior: BehaviorArgs {
                check_all: false,
                suppress_errors: false,
                expectations: false,
                remove_unused_ignores: false,
                all: false,
            },
        };
        match check_args.run_once_with_snippet(self.code, config_finder) {
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
    /// If we are removing unused ignores, should we remove all unused ignores or only Pyrefly specific `pyrefly: ignore`s?
    #[arg(long, requires("remove_unused_ignores"))]
    all: bool,
}

impl OutputFormat {
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
        Self::write_error_json(&mut writer, relative_to, errors)?;
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
            OutputFormat::buffered_write_error_json(file, relative_to, errors)
        }
        f(path, relative_to, errors)
            .with_context(|| format!("while writing JSON errors to `{}`", path.display()))
    }

    fn write_error_json_to_console(relative_to: &Path, errors: &[Error]) -> anyhow::Result<()> {
        Self::buffered_write_error_json(stdout(), relative_to, errors)
    }

    fn write_errors_to_file(
        &self,
        path: &Path,
        relative_to: &Path,
        errors: &[Error],
    ) -> anyhow::Result<()> {
        match self {
            Self::MinText => Self::write_error_text_to_file(path, relative_to, errors, false),
            Self::FullText => Self::write_error_text_to_file(path, relative_to, errors, true),
            Self::Json => Self::write_error_json_to_file(path, relative_to, errors),
            Self::OmitErrors => Ok(()),
        }
    }

    fn write_errors_to_console(&self, relative_to: &Path, errors: &[Error]) -> anyhow::Result<()> {
        match self {
            Self::MinText => Self::write_error_text_to_console(relative_to, errors, false),
            Self::FullText => Self::write_error_text_to_console(relative_to, errors, true),
            Self::Json => Self::write_error_json_to_console(relative_to, errors),
            Self::OmitErrors => Ok(()),
        }
    }
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
                .entry(config_finder.python_file(unknown, path))
                .or_insert_with(SmallSet::new)
                .insert(path.dupe());
        }

        // TODO(connernilsen): wire in force logic
        let reloaded_source_dbs = ConfigFile::query_source_db(&configs, false).0;
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
    pub fn run_once(
        self,
        files_to_check: Box<dyn Includes>,
        config_finder: ConfigFinder,
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

        let holder = Forgetter::new(State::new(config_finder), true);
        let handles = Handles::new(expanded_file_list);
        let require_levels = self.get_required_levels();
        let mut transaction = Forgetter::new(
            holder
                .as_ref()
                .new_transaction(require_levels.default, None),
            true,
        );
        let (loaded_handles, _, sourcedb_errors) = handles.all(holder.as_ref().config_finder());
        self.run_inner(
            timings,
            transaction.as_mut(),
            &loaded_handles,
            sourcedb_errors,
            require_levels.specified,
        )
    }

    pub fn run_once_with_snippet(
        self,
        code: String,
        config_finder: ConfigFinder,
    ) -> anyhow::Result<(CommandExitStatus, Vec<Error>)> {
        // Create a virtual module path for the snippet
        let path = PathBuf::from_str("snippet")?;
        let module_path = ModulePath::memory(path);
        let module_name = ModuleName::from_str("__main__");

        let holder = Forgetter::new(State::new(config_finder), true);

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
            true,
        );

        // Add the snippet source to the transaction's memory
        transaction.as_mut().set_memory(vec![(
            PathBuf::from(module_path.as_path()),
            Some(Arc::new(FileContents::from_source(code))),
        )]);

        self.run_inner(
            Timings::new(),
            transaction.as_mut(),
            &[handle],
            vec![],
            require_levels.specified,
        )
    }

    pub async fn run_watch(
        self,
        mut watcher: Watcher,
        files_to_check: Box<dyn Includes>,
        config_finder: ConfigFinder,
    ) -> anyhow::Result<()> {
        // TODO: We currently make 1 unrealistic assumptions, which should be fixed in the future:
        // - Config search is stable across incremental runs.
        let expanded_file_list = config_finder.checkpoint(files_to_check.files())?;
        let require_levels = self.get_required_levels();
        let mut handles = Handles::new(expanded_file_list);
        let state = State::new(config_finder);
        let mut transaction = state.new_committable_transaction(require_levels.default, None);
        loop {
            let timings = Timings::new();
            let (loaded_handles, reloaded_configs, sourcedb_errors) =
                handles.all(state.config_finder());
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
                Some(Box::new(ProgressBarSubscriber::new())),
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
        handles: &[Handle],
        mut sourcedb_errors: Vec<ConfigError>,
        require: Require,
    ) -> anyhow::Result<(CommandExitStatus, Vec<Error>)> {
        let mut memory_trace = MemoryUsageTrace::start(Duration::from_secs_f32(0.1));

        let type_check_start = Instant::now();
        transaction.set_subscriber(Some(Box::new(ProgressBarSubscriber::new())));
        transaction.run(handles, require);
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

        let errors = loads
            .collect_errors_with_baseline(self.output.baseline.as_deref(), relative_to.as_path());
        let shown_errors = if let Some(only) = &self.output.only {
            let only = only.iter().collect::<SmallSet<_>>();
            errors
                .shown
                .into_iter()
                .filter(|e| only.contains(&e.error_kind()))
                .collect()
        } else {
            errors.shown
        };

        // We update the baseline file if requested, after reporting any new errors using the old baseline
        if self.output.update_baseline
            && let Some(baseline_path) = &self.output.baseline
        {
            let mut new_baseline = shown_errors.clone();
            new_baseline.extend(errors.baseline);
            new_baseline.sort_by_cached_key(|error| {
                (
                    error.path().to_string(),
                    error.range().start(),
                    error.range().end(),
                    error.error_kind(),
                )
            });
            OutputFormat::write_error_json_to_file(
                baseline_path,
                relative_to.as_path(),
                &new_baseline,
            )?;
        }

        if let Some(path) = &self.output.output {
            self.output.output_format.write_errors_to_file(
                path,
                relative_to.as_path(),
                &shown_errors,
            )?;
        } else {
            self.output
                .output_format
                .write_errors_to_console(relative_to.as_path(), &shown_errors)?;
        }
        memory_trace.stop();
        if let Some(limit) = self.output.count_errors {
            print_error_counts(&shown_errors, limit);
        }
        if self.output.summarize_errors.is_some() {
            print_error_summary(&shown_errors);
        }
        let mut shown_errors_count = config_errors_count;
        for error in &shown_errors {
            if error.severity() >= Severity::Error {
                shown_errors_count += 1;
            }
        }
        timings.report_errors = report_errors_start.elapsed();

        if self.output.summary != Summary::None {
            let suppress_count = errors.suppressed.len();
            if suppress_count == 0 {
                info!("{}", count(shown_errors_count, "error"))
            } else {
                info!(
                    "{} ({} suppressed)",
                    count(shown_errors_count, "error"),
                    number_thousands(suppress_count)
                )
            };
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
            transaction.set_subscriber(Some(Box::new(ProgressBarSubscriber::new())));
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
                let module_hash = blake3::hash(handle.module().to_string().as_bytes());
                fs_anyhow::write(
                    &glean.join(format!("{}.json", &module_hash)),
                    report::glean::glean(transaction, handle),
                )?;
            }
        }
        if let Some(pysa_directory) = &self.output.report_pysa {
            report::pysa::write_results(pysa_directory, transaction, &shown_errors)?;
        }
        if let Some(path) = &self.output.report_binding_memory {
            fs_anyhow::write(path, report::binding_memory::binding_memory(transaction))?;
        }
        if let Some(path) = &self.output.report_trace {
            fs_anyhow::write(path, report::trace::trace(transaction))?;
        }
        if self.behavior.suppress_errors {
            suppress::suppress_errors(shown_errors.clone());
        }
        if self.behavior.remove_unused_ignores {
            suppress::remove_unused_ignores(&loads, self.behavior.all);
        }
        if self.behavior.expectations {
            loads.check_against_expectations()?;
            Ok((CommandExitStatus::Success, shown_errors))
        } else if shown_errors_count > 0 {
            Ok((CommandExitStatus::UserError, shown_errors))
        } else {
            Ok((CommandExitStatus::Success, shown_errors))
        }
    }
}
