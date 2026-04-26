/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::path::PathBuf;
use std::str::FromStr;

use anyhow::Context as _;
use clap::Parser;
use dupe::Dupe;
use pyrefly_build::source_db::SourceDatabase;
use pyrefly_build::source_db::buck_check::BuckCheckSourceDatabase;
use pyrefly_config::base::InferReturnTypes;
use pyrefly_config::error::ErrorDisplayConfig;
use pyrefly_config::error_kind::ErrorKind;
use pyrefly_config::error_kind::Severity;
use pyrefly_python::sys_info::PythonPlatform;
use pyrefly_python::sys_info::PythonVersion;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::fs_anyhow;
use pyrefly_util::thread_pool::ThreadCount;
use ruff_text_size::Ranged;
use serde::Deserialize;
use tracing::info;

use crate::commands::util::CommandExitStatus;
use crate::config::config::ConfigFile;
use crate::config::finder::ConfigFinder;
use crate::error::error::Error;
use crate::error::legacy::LegacyErrors;
use crate::state::require::Require;
use crate::state::require::RequireLevels;
use crate::state::state::State;

/// Arguments for Buck-powered type checking.
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Clone, Parser)]
pub struct BuckCheckArgs {
    /// Path to input JSON manifest.
    input_path: PathBuf,

    /// Path to output JSON file containing Pyrefly type check results.
    #[arg(long = "output", short = 'o', value_name = "FILE")]
    output_path: Option<PathBuf>,

    /// Minimum severity level for errors to be displayed.
    /// Errors below this severity will not be shown. Defaults to "error".
    #[arg(long, value_enum)]
    min_severity: Option<Severity>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct InputFile {
    dependencies: Vec<PathBuf>,
    py_version: String,
    sources: Vec<PathBuf>,
    typeshed: Option<PathBuf>,
    system_platform: String,
}

fn read_input_file(path: &Path) -> anyhow::Result<InputFile> {
    let data = fs_anyhow::read(path)?;
    let input_file: InputFile = serde_json::from_slice(&data)
        .with_context(|| format!("failed to parse input JSON `{}`", path.display()))?;
    Ok(input_file)
}

fn compute_errors(
    sys_info: SysInfo,
    sourcedb: impl SourceDatabase + 'static,
    thread_count: ThreadCount,
) -> Vec<Error> {
    let modules_to_check = sourcedb.modules_to_check().into_iter().collect::<Vec<_>>();

    let mut config = ConfigFile::default();
    config.python_environment.python_platform = Some(sys_info.platform().clone());
    config.python_environment.python_version = Some(sys_info.version());
    config.python_environment.site_package_path = Some(Vec::new());
    config.source_db = Some(ArcId::new(Box::new(sourcedb)));
    config.interpreters.skip_interpreter_query = true;
    config.disable_search_path_heuristics = true;

    // Modifications to make it more like Pyre.
    // Should probably figure out how to move these into PACKAGE files, or put them in Pyrefly.toml.
    config.root.permissive_ignores = Some(true);
    config.root.check_unannotated_defs = Some(false);
    config.root.infer_return_types = Some(InferReturnTypes::Annotated);
    let mut error_config = ErrorDisplayConfig::default();
    error_config.set_error_severity(ErrorKind::Deprecated, Severity::Ignore);
    error_config.set_error_severity(ErrorKind::UnusedIgnore, Severity::Info);
    config.root.errors = Some(error_config);

    config.configure();
    let config = ArcId::new(config);

    let state = State::new(ConfigFinder::new_constant(config), thread_count);
    state.run(
        &modules_to_check,
        RequireLevels {
            specified: Require::Errors,
            default: Require::Exports,
        },
        None,
        None,
        None,
    );
    let transaction = state.transaction();
    let errors = transaction.get_errors(&modules_to_check);

    // Collect main errors (done once, shared with unused ignore check)
    let collected = errors.collect_errors();
    let unused = errors.collect_unused_ignore_errors_for_display(&collected);
    let mut output_errors = collected.ordinary;
    output_errors.extend(collected.directives);
    output_errors.extend(unused.ordinary);
    output_errors.sort_by_cached_key(|e| {
        (
            e.module().name(),
            e.path().dupe(),
            e.range().start(),
            e.range().end(),
        )
    });

    output_errors
}

fn write_output_to_file(path: &Path, legacy_errors: &LegacyErrors) -> anyhow::Result<()> {
    let output_bytes = serde_json::to_vec(legacy_errors)
        .with_context(|| "failed to serialize JSON value to bytes")?;
    fs_anyhow::write(path, &output_bytes)
}

fn write_output_to_stdout(legacy_errors: &LegacyErrors) -> anyhow::Result<()> {
    let contents = serde_json::to_string_pretty(legacy_errors)?;
    println!("{contents}");
    Ok(())
}

fn write_output(errors: &[Error], path: Option<&Path>) -> anyhow::Result<()> {
    let legacy_errors = LegacyErrors::from_errors(PathBuf::new().as_path(), errors);
    if let Some(path) = path {
        write_output_to_file(path, &legacy_errors)
    } else {
        write_output_to_stdout(&legacy_errors)
    }
}

impl BuckCheckArgs {
    pub fn run(self, thread_count: ThreadCount) -> anyhow::Result<CommandExitStatus> {
        let input_file = read_input_file(self.input_path.as_path())?;
        let python_version = PythonVersion::from_str(&input_file.py_version)?;
        let python_platform = PythonPlatform::new(&input_file.system_platform);
        let sys_info = SysInfo::new(python_version, python_platform);
        let sourcedb = BuckCheckSourceDatabase::from_manifest_files(
            input_file.sources.as_slice(),
            input_file.dependencies.as_slice(),
            input_file.typeshed.as_slice(),
            sys_info.dupe(),
        )?;
        let type_errors = compute_errors(sys_info, sourcedb, thread_count);
        let min_severity = self.min_severity.unwrap_or(Severity::Error);
        let displayed_errors: Vec<Error> = type_errors
            .into_iter()
            .filter(|e| e.error_kind().is_directive() || e.severity() >= min_severity)
            .collect();
        info!("Found {} type errors", displayed_errors.len());
        write_output(&displayed_errors, self.output_path.as_deref())?;
        Ok(CommandExitStatus::Success)
    }
}
