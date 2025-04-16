/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context as _;
use clap::Parser;
use clap::ValueEnum;
use dupe::Dupe;
use dupe::IterDupedExt;
use starlark_map::small_map::SmallMap;
use tracing::info;

use crate::clap_env;
use crate::commands::suppress;
use crate::commands::util::module_from_path;
use crate::config::config::ConfigFile;
use crate::config::error::ErrorConfig;
use crate::config::error::ErrorConfigs;
use crate::config::util::set_if_some;
use crate::config::util::set_option_if_some;
use crate::error::error::Error;
use crate::error::error::print_error_counts;
use crate::error::error::print_errors;
use crate::error::legacy::LegacyErrors;
use crate::error::summarise::print_error_summary;
use crate::finder::ConfigFinder;
use crate::metadata::PythonPlatform;
use crate::metadata::PythonVersion;
use crate::metadata::RuntimeMetadata;
use crate::module::bundled::typeshed;
use crate::module::finder::find_module;
use crate::module::module_name::ModuleName;
use crate::module::module_path::ModulePath;
use crate::module::module_path::ModulePathDetails;
use crate::module::wildcard::ModuleWildcard;
use crate::report;
use crate::run::CommandExitStatus;
use crate::state::handle::Handle;
use crate::state::loader::FindError;
use crate::state::loader::Loader;
use crate::state::loader::LoaderId;
use crate::state::require::Require;
use crate::state::state::CommittingTransaction;
use crate::state::state::State;
use crate::state::subscriber::ProgressBarSubscriber;
use crate::util::display;
use crate::util::display::number_thousands;
use crate::util::forgetter::Forgetter;
use crate::util::fs_anyhow;
use crate::util::listing::FileList;
use crate::util::memory::MemoryUsageTrace;
use crate::util::prelude::SliceExt;
use crate::util::watcher::CategorizedEvents;
use crate::util::watcher::Watcher;

#[derive(Debug, Clone, ValueEnum, Default)]
enum OutputFormat {
    #[default]
    Text,
    Json,
}

#[derive(Debug, Parser, Clone)]
pub struct Args {
    /// Write the errors to a file, instead of printing them.
    #[arg(long, short = 'o', env = clap_env("OUTPUT"))]
    output: Option<PathBuf>,
    #[clap(long, env = clap_env("SEARCH_PATH"))]
    search_path: Option<Vec<PathBuf>>,
    #[clap(long, value_enum, default_value_t, env = clap_env("OUTPUT_FORMAT"))]
    output_format: OutputFormat,
    /// Check all reachable modules, not just the ones that are passed in explicitly on CLI positional arguments.
    #[clap(long, short = 'a', env = clap_env("CHECK_ALL"))]
    check_all: bool,
    #[clap(long, env = clap_env("PYTHON_VERSION"))]
    python_version: Option<PythonVersion>,
    #[clap(long, env = clap_env("PLATFORM"))]
    python_platform: Option<PythonPlatform>,
    #[clap(long, env = clap_env("SITE_PACKAGE_PATH"))]
    site_package_path: Option<Vec<PathBuf>>,
    /// The Python executable that will be queried for Python version, platform, or site package path info
    /// if the values are missing
    #[clap(long, env = clap_env("PYTHON_INTERPRETER"))]
    python_interpreter: Option<PathBuf>,
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
    /// Suppress errors found in the input files.
    #[clap(long, env = clap_env("SUPPRESS_ERRORS"))]
    suppress_errors: bool,
    /// Check against any `E:` lines in the file.
    #[clap(long, env = clap_env("EXPECTATIONS"))]
    expectations: bool,
    /// Whether to ignore type errors in generated code.
    /// Generated code is defined as code that contains the marker string `@` immediately followed by `generated`.
    #[clap(long, env = clap_env("IGNORE_ERRORS_IN_GENERATED_CODE"))]
    ignore_errors_in_generated_code: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct LoaderInputs {
    search_path: Vec<PathBuf>,
    site_package_path: Vec<PathBuf>,
    replace_imports_with_any: Vec<ModuleWildcard>,
}

#[derive(Debug, Clone)]
struct CheckLoader {
    loader_inputs: LoaderInputs,
}

impl Loader for CheckLoader {
    fn find_import(&self, module: ModuleName) -> Result<ModulePath, FindError> {
        if self
            .loader_inputs
            .replace_imports_with_any
            .iter()
            .any(|p| p.matches(module))
        {
            Err(FindError::Ignored)
        } else if let Some(path) = find_module(module, &self.loader_inputs.search_path) {
            Ok(path)
        } else if let Some(path) = typeshed().map_err(FindError::not_found)?.find(module) {
            Ok(path)
        } else if let Some(path) = find_module(module, &self.loader_inputs.site_package_path) {
            Ok(path)
        } else {
            Err(FindError::search_path(
                &self.loader_inputs.search_path,
                &self.loader_inputs.site_package_path,
            ))
        }
    }
}

impl OutputFormat {
    fn write_error_text_to_file(path: &Path, errors: &[Error]) -> anyhow::Result<()> {
        let mut file = BufWriter::new(File::create(path)?);
        for e in errors {
            writeln!(file, "{e}")?;
        }
        file.flush()?;
        Ok(())
    }

    fn write_error_json_to_file(path: &Path, errors: &[Error]) -> anyhow::Result<()> {
        fn f(path: &Path, errors: &[Error]) -> anyhow::Result<()> {
            let legacy_errors = LegacyErrors::from_errors(errors);
            let mut file = BufWriter::new(File::create(path)?);
            serde_json::to_writer_pretty(&mut file, &legacy_errors)?;
            Ok(file.flush()?)
        }
        f(path, errors)
            .with_context(|| format!("while writing JSON errors to `{}`", path.display()))
    }

    fn write_errors_to_file(&self, path: &Path, errors: &[Error]) -> anyhow::Result<()> {
        match self {
            Self::Text => Self::write_error_text_to_file(path, errors),
            Self::Json => Self::write_error_json_to_file(path, errors),
        }
    }
}

fn create_loader(loader_inputs: LoaderInputs) -> LoaderId {
    LoaderId::new(CheckLoader { loader_inputs })
}

/// A data structure to facilitate the creation of handles for all the files we want to check.
struct Handles {
    /// We want to have different handles to share the same loader if the corresponding files share the same search path.
    /// This field keeps track of the loaders we've created so far and what search paths they correspond to.
    loader_factory: SmallMap<LoaderInputs, LoaderId>,
    /// A mapping from a file to all other information needed to create a `Handle`.
    /// The value type is basically everything else in `Handle` except for the file path.
    path_data: HashMap<PathBuf, (ModuleName, RuntimeMetadata, LoaderId)>,
    /// A the underlying HashMap that will be used to create an `ErrorConfigs` when requested.
    module_to_error_config: HashMap<ModulePath, ErrorConfig>,
}

impl Handles {
    fn new(files: Vec<PathBuf>, config_finder: &ConfigFinder) -> Self {
        let mut handles = Self {
            loader_factory: SmallMap::new(),
            path_data: HashMap::new(),
            module_to_error_config: HashMap::new(),
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
    ) -> &(ModuleName, RuntimeMetadata, LoaderId) {
        let config = config_finder.python_file(&path);
        self.module_to_error_config.insert(
            ModulePath::filesystem(path.clone()),
            ErrorConfig::new(
                config.errors().clone(),
                config.ignore_errors_in_generated_code(),
            ),
        );
        let loader = self.get_or_register_loader(&config);
        let module_name = module_from_path(&path, &config.search_path);
        self.path_data
            .entry(path)
            .or_insert((module_name, config.get_runtime_metadata(), loader))
    }

    fn get_or_register_loader(&mut self, config: &ConfigFile) -> LoaderId {
        let key = LoaderInputs {
            search_path: config.search_path.clone(),
            site_package_path: config.site_package_path().to_owned(),
            replace_imports_with_any: config.replace_imports_with_any().to_vec(),
        };
        if let Some(loader) = self.loader_factory.get_mut(&key) {
            loader.dupe()
        } else {
            let loader = create_loader(key.clone());
            self.loader_factory.insert(key, loader.dupe());
            loader
        }
    }

    fn all(&self, specified_require: Require) -> Vec<(Handle, Require)> {
        self.path_data
            .iter()
            .map(|(path, (module_name, runtime_metadata, loader))| {
                (
                    Handle::new(
                        module_name.dupe(),
                        ModulePath::filesystem(path.to_path_buf()),
                        runtime_metadata.dupe(),
                        loader.dupe(),
                    ),
                    specified_require,
                )
            })
            .collect()
    }

    fn loaders(&self) -> Vec<LoaderId> {
        self.loader_factory.values().duped().collect()
    }

    fn error_configs(&self) -> ErrorConfigs {
        ErrorConfigs::new(self.module_to_error_config.clone())
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
            self.module_to_error_config
                .remove(&ModulePath::filesystem(file.to_path_buf()));
            // NOTE: Need to garbage-collect unreachable Loaders at some point
        }
    }
}

struct RequireLevels {
    specified: Require,
    default: Require,
}

async fn get_watcher_events(watcher: &mut impl Watcher) -> anyhow::Result<CategorizedEvents> {
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

impl Args {
    pub fn run_once(
        self,
        files_to_check: impl FileList,
        config_finder: ConfigFinder,
        allow_forget: bool,
    ) -> anyhow::Result<CommandExitStatus> {
        let expanded_file_list = files_to_check.files()?;
        if expanded_file_list.is_empty() {
            return Ok(CommandExitStatus::Success);
        }

        let mut holder = Forgetter::new(State::new(), allow_forget);
        let handles = Handles::new(expanded_file_list, &config_finder);
        let require_levels = self.get_required_levels();
        self.run_inner(
            holder.as_mut(),
            None,
            &handles.all(require_levels.specified),
            require_levels.default,
            &handles.error_configs(),
        )
    }

    pub async fn run_watch(
        self,
        mut watcher: impl Watcher,
        files_to_check: impl FileList,
        config_finder: ConfigFinder,
    ) -> anyhow::Result<()> {
        // TODO: We currently make 1 unrealistic assumptions, which should be fixed in the future:
        // - Config search is stable across incremental runs.
        let expanded_file_list = files_to_check.files()?;
        let require_levels = self.get_required_levels();
        let mut handles = Handles::new(expanded_file_list, &config_finder);
        let state = State::new();
        let mut transaction = None;
        loop {
            let res = self.run_inner(
                &state,
                transaction,
                &handles.all(require_levels.specified),
                require_levels.default,
                &handles.error_configs(),
            );
            if let Err(e) = res {
                eprintln!("{e:#}");
            }
            let events = get_watcher_events(&mut watcher).await?;
            let mut new_transaction = state.new_committable_transaction(
                require_levels.default,
                Some(Box::new(ProgressBarSubscriber::new())),
            );
            let new_transaction_mut = new_transaction.as_mut();
            new_transaction_mut.invalidate_disk(&events.modified);

            new_transaction_mut.invalidate_disk(&events.created);
            new_transaction_mut.invalidate_disk(&events.removed);
            // File addition and removal may affect the list of files/handles to check. Update
            // the handles accordingly.
            handles.update(
                events.created.iter().filter(|p| files_to_check.covers(p)),
                events.removed.iter().filter(|p| files_to_check.covers(p)),
                &config_finder,
            );
            for loader in handles.loaders() {
                new_transaction_mut.invalidate_find(&loader);
            }
            transaction = Some(new_transaction);
        }
    }

    pub fn override_config(&self, mut config: ConfigFile) -> ConfigFile {
        set_option_if_some(
            &mut config.python_environment.python_platform,
            self.python_platform.as_ref(),
        );
        set_option_if_some(
            &mut config.python_environment.python_version,
            self.python_version.as_ref(),
        );
        set_if_some(&mut config.search_path, self.search_path.as_ref());
        set_option_if_some(
            &mut config.python_environment.site_package_path,
            self.site_package_path.as_ref(),
        );
        set_option_if_some(
            &mut config.python_interpreter,
            self.python_interpreter.as_ref(),
        );
        set_option_if_some(
            &mut config.root.ignore_errors_in_generated_code,
            self.ignore_errors_in_generated_code.as_ref(),
        );
        config.configure();
        config.validate();
        config
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
            } else if self.check_all {
                Require::Errors
            } else {
                Require::Exports
            },
        }
    }

    fn run_inner(
        &self,
        state: &State,
        transaction: Option<CommittingTransaction>,
        handles: &[(Handle, Require)],
        default_require: Require,
        error_configs: &ErrorConfigs,
    ) -> anyhow::Result<CommandExitStatus> {
        let mut memory_trace = MemoryUsageTrace::start(Duration::from_secs_f32(0.1));
        let start = Instant::now();

        if let Some(transaction) = transaction {
            state.run_with_committing_transaction(transaction, handles);
        } else {
            let progress = Box::new(ProgressBarSubscriber::new());
            state.run(handles, default_require, Some(progress));
        }
        let transaction = state.transaction();
        let readable_state = transaction.readable();
        let loads = if self.check_all {
            transaction.get_loads(readable_state.handles().iter())
        } else {
            state
                .transaction()
                .get_loads(handles.iter().map(|(handle, _)| handle))
        };
        let computing = start.elapsed();
        let errors = loads.collect_errors(error_configs);
        if let Some(path) = &self.output {
            self.output_format
                .write_errors_to_file(path, &errors.shown)?;
        } else {
            print_errors(&errors.shown);
        }
        let printing = start.elapsed();
        memory_trace.stop();
        if let Some(limit) = self.count_errors {
            print_error_counts(&errors.shown, limit);
        }
        if let Some(path_index) = self.summarize_errors {
            print_error_summary(&errors.shown, path_index);
        }
        let shown_errors_count = errors.shown.len();
        info!(
            "{} errors shown, {} errors disabled, {} errors suppressed, {} modules, {} lines, took {printing:.2?} ({computing:.2?} without printing errors), peak memory {}",
            number_thousands(shown_errors_count),
            number_thousands(errors.disabled.len()),
            number_thousands(errors.suppressed.len()),
            number_thousands(readable_state.module_count()),
            number_thousands(readable_state.line_count()),
            memory_trace.peak()
        );
        if let Some(timings) = &self.report_timings {
            eprintln!("Computing timing information");
            transaction.report_timings(timings, Some(Box::new(ProgressBarSubscriber::new())))?;
        }
        if let Some(debug_info) = &self.debug_info {
            let mut output = serde_json::to_string_pretty(
                &readable_state.debug_info(&handles.map(|x| x.0.dupe()), error_configs),
            )?;
            if debug_info.extension() == Some(OsStr::new("js")) {
                output = format!("var data = {output}");
            }
            fs_anyhow::write(debug_info, output.as_bytes())?;
        }
        if let Some(glean) = &self.report_glean {
            fs_anyhow::create_dir_all(glean)?;
            for (handle, _) in handles {
                let output = serde_json::to_string_pretty(&readable_state.glean(handle))?;
                fs_anyhow::write(
                    &glean.join(format!("{}.json", handle.module())),
                    output.as_bytes(),
                )?;
            }
        }
        if let Some(path) = &self.report_binding_memory {
            fs_anyhow::write(
                path,
                report::binding_memory::binding_memory(&transaction).as_bytes(),
            )?;
        }
        if let Some(path) = &self.report_trace {
            fs_anyhow::write(path, report::trace::trace(&transaction).as_bytes())?;
        }
        if self.suppress_errors {
            let errors: SmallMap<PathBuf, Vec<Error>> = errors
                .shown
                .into_iter()
                .filter(|e| matches!(e.path().details(), ModulePathDetails::FileSystem(_)))
                .fold(SmallMap::new(), |mut acc, e| {
                    let path = PathBuf::from(e.path().to_string());
                    acc.entry(path).or_default().push(e);
                    acc
                });
            suppress::suppress_errors(&errors);
        }
        if self.expectations {
            loads.check_against_expectations(error_configs)?;
            Ok(CommandExitStatus::Success)
        } else if shown_errors_count > 0 {
            Ok(CommandExitStatus::UserError)
        } else {
            Ok(CommandExitStatus::Success)
        }
    }
}
