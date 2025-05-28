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
use std::time::Duration;
use std::time::Instant;

use anstream::eprintln;
use anstream::stdout;
use anyhow::Context as _;
use clap::Parser;
use clap::ValueEnum;
use dupe::Dupe;
use path_absolutize::Absolutize;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::args::clap_env;
use pyrefly_util::display;
use pyrefly_util::display::number_thousands;
use pyrefly_util::events::CategorizedEvents;
use pyrefly_util::forgetter::Forgetter;
use pyrefly_util::fs_anyhow;
use pyrefly_util::globs::FilteredGlobs;
use pyrefly_util::memory::MemoryUsageTrace;
use pyrefly_util::prelude::SliceExt;
use pyrefly_util::watcher::Watcher;
use ruff_source_file::OneIndexed;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use tracing::debug;
use tracing::info;

use crate::commands::run::CommandExitStatus;
use crate::commands::suppress;
use crate::commands::util::module_from_path;
use crate::config::config::ConfigFile;
use crate::config::config::validate_path;
use crate::config::environment::environment::SitePackagePathSource;
use crate::config::finder::ConfigError;
use crate::config::finder::ConfigFinder;
use crate::error::error::Error;
use crate::error::error::print_error_counts;
use crate::error::legacy::LegacyErrors;
use crate::error::summarise::print_error_summary;
use crate::module::bundled::stdlib_search_path;
use crate::module::ignore::SuppressionKind;
use crate::module::module_name::ModuleName;
use crate::module::module_path::ModulePath;
use crate::module::module_path::ModulePathDetails;
use crate::report;
use crate::state::handle::Handle;
use crate::state::require::Require;
use crate::state::state::State;
use crate::state::state::Transaction;
use crate::state::subscriber::ProgressBarSubscriber;
use crate::sys_info::PythonPlatform;
use crate::sys_info::PythonVersion;
use crate::sys_info::SysInfo;

#[derive(Debug, Clone, ValueEnum, Default)]
enum OutputFormat {
    #[default]
    Text,
    Json,
}

#[derive(Debug, Parser, Clone)]
pub struct Args {
    // how/what should Pyrefly output
    /// Write the errors to a file, instead of printing them.
    #[arg(long, short = 'o', env = clap_env("OUTPUT"))]
    output: Option<PathBuf>,
    #[clap(long, value_enum, default_value_t, env = clap_env("OUTPUT_FORMAT"))]
    output_format: OutputFormat,
    /// Produce debugging information about the type checking process.
    #[clap(long, env = clap_env("DEBUG_INFO"))]
    debug_info: Option<PathBuf>,
    #[clap(long, env = clap_env("REPORT_BINDING_MEMORY"))]
    report_binding_memory: Option<PathBuf>,
    #[clap(long, env = clap_env("REPORT_TRACE"))]
    report_trace: Option<PathBuf>,
    /// Process each module individually to figure out how long each step takes.
    #[clap(long, env = clap_env("REPORT_TIMINGS"))]
    report_timings: Option<PathBuf>,
    /// Generate a Glean-compatible JSON file for each module
    #[clap(long, env = clap_env("REPORT_GLEAN"))]
    report_glean: Option<PathBuf>,
    /// Count the number of each error kind. Prints the top N errors, sorted by count, or all errors if N is not specified.
    #[clap(
        long,
        default_missing_value = "5",
        require_equals = true,
        num_args = 0..=1,
        env = clap_env("COUNT_ERRORS")
    )]
    count_errors: Option<usize>,
    /// Summarize errors by directory. The optional index argument specifies which file path segment will be used to group errors.
    /// The default index is 0. For errors in `/foo/bar/...`, this will group errors by `/foo`. If index is 1, errors will be grouped by `/foo/bar`.
    /// An index larger than the number of path segments will group by the final path element, i.e. the file name.
    #[clap(
        long,
        default_missing_value = "0",
        require_equals = true,
        num_args = 0..=1,
        env = clap_env("SUMMARIZE_ERRORS")
    )]
    summarize_errors: Option<usize>,

    /// Set this flag to omit the summary in the last line of the output.
    #[clap(long, env = clap_env("NO_SUMMARY"))]
    no_summary: bool,

    // non-config type checker behavior
    /// Check all reachable modules, not just the ones that are passed in explicitly on CLI positional arguments.
    #[clap(long, short = 'a', env = clap_env("CHECK_ALL"))]
    check_all: bool,
    /// Suppress errors found in the input files.
    #[clap(long, env = clap_env("SUPPRESS_ERRORS"))]
    suppress_errors: bool,
    /// Check against any `E:` lines in the file.
    #[clap(long, env = clap_env("EXPECTATIONS"))]
    expectations: bool,
    /// Remove unused ignores from the input files.
    #[clap(long, env = clap_env("REMOVE_UNUSED_IGNORES"))]
    remove_unused_ignores: bool,

    // config overrides
    /// The list of directories where imports are imported from, including
    /// type checked files.
    #[clap(long, env = clap_env("SEARCH_PATH"))]
    search_path: Option<Vec<PathBuf>>,
    /// The Python version any `sys.version` checks should evaluate against.
    #[clap(long, env = clap_env("PYTHON_VERSION"))]
    python_version: Option<PythonVersion>,
    /// The platform any `sys.platform` checks should evaluate against.
    #[clap(long, env = clap_env("PLATFORM"))]
    python_platform: Option<PythonPlatform>,
    /// Directories containing third-party package imports, searched
    /// after first checking `search_path` and `typeshed`.
    #[clap(long, env = clap_env("SITE_PACKAGE_PATH"))]
    site_package_path: Option<Vec<PathBuf>>,
    /// The Python executable that will be queried for `python_version`
    /// `python_platform`, or `site_package_path` if any of the values are missing.
    #[clap(long, env = clap_env("PYTHON_INTERPRETER"))]
    python_interpreter: Option<PathBuf>,
}

impl OutputFormat {
    fn write_error_text_to_file(path: &Path, errors: &[Error]) -> anyhow::Result<()> {
        let mut file = BufWriter::new(File::create(path)?);
        for e in errors {
            e.write_line(&mut file)?;
        }
        file.flush()?;
        Ok(())
    }

    fn write_error_text_to_console(errors: &[Error]) -> anyhow::Result<()> {
        for error in errors {
            error.print_colors();
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
            Self::Text => Self::write_error_text_to_file(path, errors),
            Self::Json => Self::write_error_json_to_file(path, errors),
        }
    }

    fn write_errors_to_console(&self, errors: &[Error]) -> anyhow::Result<()> {
        match self {
            Self::Text => Self::write_error_text_to_console(errors),
            Self::Json => Self::write_error_json_to_console(errors),
        }
    }
}

/// A data structure to facilitate the creation of handles for all the files we want to check.
struct Handles {
    /// A mapping from a file to all other information needed to create a `Handle`.
    /// The value type is basically everything else in `Handle` except for the file path.
    path_data: HashMap<PathBuf, (ModuleName, SysInfo)>,
}

impl Handles {
    fn new(
        files: Vec<PathBuf>,
        args_search_path: &[PathBuf],
        config_finder: &ConfigFinder,
    ) -> Self {
        let mut handles = Self {
            path_data: HashMap::new(),
        };
        for file in files {
            handles.register_file(file, args_search_path, config_finder);
        }
        handles
    }

    fn register_file(
        &mut self,
        path: PathBuf,
        args_search_path: &[PathBuf],
        config_finder: &ConfigFinder,
    ) -> &(ModuleName, SysInfo) {
        let module_path = ModulePath::filesystem(path.clone());
        let unknown = ModuleName::unknown();
        let config = config_finder.python_file(unknown, &module_path);

        let search_path = args_search_path.iter().chain(config.search_path());
        let module_name = module_from_path(&path, search_path).unwrap_or(unknown);

        self.path_data
            .entry(path)
            .or_insert((module_name, config.get_sys_info()))
    }

    fn all(&self, specified_require: Require) -> Vec<(Handle, Require)> {
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
        args_search_path: &[PathBuf],
        config_finder: &ConfigFinder,
    ) {
        for file in created_files {
            self.register_file(file.to_path_buf(), args_search_path, config_finder);
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
        let events = CategorizedEvents::new(watcher.wait().await?);
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
        const THRESHOLD: Duration = Duration::from_millis(300);
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
                display::intersperse_iter("; ", || steps
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

impl Args {
    pub fn absolute_search_path(&mut self) {
        if let Some(paths) = self.search_path.as_mut() {
            for x in paths.iter_mut() {
                if let Ok(v) = x.absolutize() {
                    *x = v.into_owned();
                }
            }
        }
    }

    pub fn get_handles(
        self,
        files_to_check: FilteredGlobs,
        config_finder: &ConfigFinder,
    ) -> anyhow::Result<Vec<(Handle, Require)>> {
        let handles = Handles::new(
            checkpoint(files_to_check.files(), config_finder)?,
            self.search_path.as_deref().unwrap_or_default(),
            config_finder,
        );
        Ok(handles.all(self.get_required_levels().specified))
    }

    pub fn run_once(
        self,
        files_to_check: FilteredGlobs,
        config_finder: ConfigFinder,
        allow_forget: bool,
    ) -> anyhow::Result<CommandExitStatus> {
        let mut timings = Timings::new();
        let list_files_start = Instant::now();
        let expanded_file_list = checkpoint(files_to_check.files(), &config_finder)?;
        timings.list_files = list_files_start.elapsed();
        debug!(
            "Checking {} files (listing took {})",
            expanded_file_list.len(),
            Timings::show(timings.list_files),
        );
        if expanded_file_list.is_empty() {
            return Ok(CommandExitStatus::Success);
        }

        let holder = Forgetter::new(State::new(config_finder), allow_forget);
        let handles = Handles::new(
            expanded_file_list,
            self.search_path.as_deref().unwrap_or_default(),
            holder.as_ref().config_finder(),
        );
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

    pub async fn run_watch(
        self,
        mut watcher: Watcher,
        files_to_check: FilteredGlobs,
        config_finder: ConfigFinder,
    ) -> anyhow::Result<()> {
        // TODO: We currently make 1 unrealistic assumptions, which should be fixed in the future:
        // - Config search is stable across incremental runs.
        let expanded_file_list = checkpoint(files_to_check.files(), &config_finder)?;
        let require_levels = self.get_required_levels();
        let mut handles = Handles::new(
            expanded_file_list,
            self.search_path.as_deref().unwrap_or_default(),
            &config_finder,
        );
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
                self.search_path.as_deref().unwrap_or_default(),
                state.config_finder(),
            );
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        fn validate_arg(arg_name: &str, paths: Option<&Vec<PathBuf>>) -> anyhow::Result<()> {
            if let Some(paths) = paths {
                for path in paths {
                    validate_path(path).with_context(|| format!("Invalid {}", arg_name))?;
                }
            }
            Ok(())
        }
        validate_arg("--site-package-path", self.site_package_path.as_ref())?;
        validate_arg("--search-path", self.search_path.as_ref())?;
        Ok(())
    }

    pub fn override_config(&self, mut config: ConfigFile) -> (ArcId<ConfigFile>, Vec<ConfigError>) {
        if let Some(x) = &self.python_platform {
            config.python_environment.python_platform = Some(x.clone());
        }
        if let Some(x) = &self.python_version {
            config.python_environment.python_version = Some(*x);
        }
        if let Some(x) = &self.search_path {
            config.search_path_from_args = x.clone();
        }
        if let Some(x) = &self.site_package_path {
            config.python_environment.site_package_path = Some(x.clone());
            config.python_environment.site_package_path_source = SitePackagePathSource::CommandLine;
        }
        if let Some(x) = &self.python_interpreter {
            config.python_interpreter = Some(x.clone());
        }
        config.configure();
        let errors = config.validate();
        (ArcId::new(config), errors)
    }

    fn get_required_levels(&self) -> RequireLevels {
        let retain = self.report_binding_memory.is_some()
            || self.debug_info.is_some()
            || self.report_trace.is_some()
            || self.report_glean.is_some();
        RequireLevels {
            specified: if retain {
                Require::Everything
            } else {
                Require::Errors
            },
            default: if retain {
                Require::Everything
            } else if self.check_all || stdlib_search_path().is_some() {
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
    ) -> anyhow::Result<CommandExitStatus> {
        let mut memory_trace = MemoryUsageTrace::start(Duration::from_secs_f32(0.1));

        let type_check_start = Instant::now();
        transaction.set_subscriber(Some(Box::new(ProgressBarSubscriber::new())));
        transaction.run(handles);
        transaction.set_subscriber(None);

        let loads = if self.check_all {
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
        if let Some(path) = &self.output {
            self.output_format
                .write_errors_to_file(path, &errors.shown)?;
        } else {
            self.output_format.write_errors_to_console(&errors.shown)?;
        }
        memory_trace.stop();
        if let Some(limit) = self.count_errors {
            print_error_counts(&errors.shown, limit);
        }
        if let Some(path_index) = self.summarize_errors {
            print_error_summary(&errors.shown, path_index);
        }
        let shown_errors_count = config_errors_count + errors.shown.len();
        timings.report_errors = report_errors_start.elapsed();

        if !self.no_summary {
            info!(
                "{} errors shown, {} errors ignored, {} modules, {} transitive dependencies, {} lines, took {timings}, peak memory {}",
                number_thousands(shown_errors_count),
                number_thousands(errors.disabled.len() + errors.suppressed.len()),
                number_thousands(handles.len()),
                number_thousands(transaction.module_count() - handles.len()),
                number_thousands(transaction.line_count()),
                memory_trace.peak()
            );
        }
        if let Some(timings) = &self.report_timings {
            eprintln!("Computing timing information");
            transaction.set_subscriber(Some(Box::new(ProgressBarSubscriber::new())));
            transaction.report_timings(timings)?;
            transaction.set_subscriber(None);
        }
        if let Some(debug_info) = &self.debug_info {
            let is_javascript = debug_info.extension() == Some("js".as_ref());
            fs_anyhow::write(
                debug_info,
                report::debug_info::debug_info(
                    transaction,
                    &handles.map(|x| x.0.dupe()),
                    is_javascript,
                )
                .as_bytes(),
            )?;
        }
        if let Some(glean) = &self.report_glean {
            fs_anyhow::create_dir_all(glean)?;
            for (handle, _) in handles {
                fs_anyhow::write(
                    &glean.join(format!("{}.json", handle.module())),
                    report::glean::glean(transaction, handle).as_bytes(),
                )?;
            }
        }
        if let Some(path) = &self.report_binding_memory {
            fs_anyhow::write(
                path,
                report::binding_memory::binding_memory(transaction).as_bytes(),
            )?;
        }
        if let Some(path) = &self.report_trace {
            fs_anyhow::write(path, report::trace::trace(transaction).as_bytes())?;
        }
        if self.suppress_errors {
            let mut errors_to_suppress: SmallMap<PathBuf, Vec<Error>> = SmallMap::new();

            for e in errors.shown {
                if let ModulePathDetails::FileSystem(path) = e.path().details() {
                    errors_to_suppress.entry(path.clone()).or_default().push(e);
                }
            }
            suppress::suppress_errors(&errors_to_suppress);
        }
        if self.remove_unused_ignores {
            let mut all_ignores: SmallMap<&PathBuf, SmallSet<OneIndexed>> = SmallMap::new();
            for (module_path, ignore) in loads.collect_ignores() {
                if let ModulePathDetails::FileSystem(path) = module_path.details() {
                    all_ignores.insert(path, ignore.get_ignores(SuppressionKind::Pyrefly));
                }
            }

            let mut suppressed_errors: SmallMap<&PathBuf, SmallSet<OneIndexed>> = SmallMap::new();
            for e in &errors.suppressed {
                if e.is_ignored()
                    && let ModulePathDetails::FileSystem(path) = e.path().details()
                {
                    suppressed_errors
                        .entry(path)
                        .or_default()
                        .insert(e.source_range().start.row);
                }
            }

            let unused_ignores = suppress::find_unused_ignores(all_ignores, suppressed_errors);
            suppress::remove_unused_ignores(unused_ignores);
        }
        if self.expectations {
            loads.check_against_expectations()?;
            Ok(CommandExitStatus::Success)
        } else if shown_errors_count > 0 {
            Ok(CommandExitStatus::UserError)
        } else {
            Ok(CommandExitStatus::Success)
        }
    }
}

/// If we have an error, print all the errors that the config finder has accumulated. This is used
/// to ensure that config errors are still surfaced if we exit early.
fn checkpoint<T>(result: anyhow::Result<T>, config_finder: &ConfigFinder) -> anyhow::Result<T> {
    if result.is_err() {
        for error in config_finder.errors() {
            error.print();
        }
    }
    result
}
