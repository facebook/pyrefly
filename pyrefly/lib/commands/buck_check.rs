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
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::sys_info::PythonPlatform;
use pyrefly_python::sys_info::PythonVersion;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::args::clap_env;
use pyrefly_util::fs_anyhow;
use pyrefly_util::prelude::VecExt;
use serde::Deserialize;
use tracing::info;

use crate::commands::run::CommandExitStatus;
use crate::config::config::ConfigFile;
use crate::config::finder::ConfigFinder;
use crate::error::error::Error;
use crate::error::legacy::LegacyErrors;
use crate::module::source_db::BuckSourceDatabase;
use crate::state::handle::Handle;
use crate::state::require::Require;
use crate::state::state::State;

/// Arguments for Buck-powered type checking.
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Clone, Parser)]
pub struct Args {
    /// Path to input JSON manifest.
    input_path: PathBuf,

    /// Path to output JSON file containing Pyrefly Pyrefly type check results.
    #[arg(long = "output", short = 'o', env = clap_env("OUTPUT_PATH"), value_name = "FILE")]
    output_path: Option<PathBuf>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct InputFile {
    dependencies: Vec<PathBuf>,
    py_version: String,
    sources: Vec<PathBuf>,
    typeshed: Option<PathBuf>,
}

fn read_input_file(path: &Path) -> anyhow::Result<InputFile> {
    let data = fs_anyhow::read(path)?;
    let input_file: InputFile = serde_json::from_slice(&data)
        .with_context(|| format!("failed to parse input JSON `{}`", path.display()))?;
    Ok(input_file)
}

fn compute_errors(sys_info: SysInfo, sourcedb: BuckSourceDatabase) -> Vec<Error> {
    let modules = sourcedb.modules_to_check();

    let mut config = ConfigFile::default();
    config.python_environment.python_platform = Some(sys_info.platform().clone());
    config.python_environment.python_version = Some(sys_info.version());
    config.python_environment.site_package_path = Some(Vec::new());
    config.custom_module_paths = sourcedb.list();
    config.configure();
    let config = ArcId::new(config);

    let modules_to_check = modules.into_map(|(name, path)| {
        (
            Handle::new(name, ModulePath::filesystem(path), sys_info.dupe()),
            Require::Errors,
        )
    });
    let state = State::new(ConfigFinder::new_constant(config));
    state.run(&modules_to_check, Require::Exports, None);
    let transaction = state.transaction();
    transaction
        .get_errors(modules_to_check.iter().map(|(handle, _)| handle))
        .collect_errors()
        .shown
}

fn write_output_to_file(path: &Path, legacy_errors: &LegacyErrors) -> anyhow::Result<()> {
    let output_bytes = serde_json::to_vec(legacy_errors)
        .with_context(|| "failed to serialize JSON value to bytes")?;
    fs_anyhow::write(path, &output_bytes)
}

fn write_output_to_stdout(legacy_errors: &LegacyErrors) -> anyhow::Result<()> {
    let contents = serde_json::to_string_pretty(legacy_errors)?;
    println!("{}", contents);
    Ok(())
}

fn write_output(errors: &[Error], path: Option<&Path>) -> anyhow::Result<()> {
    let legacy_errors = LegacyErrors::from_errors(errors);
    if let Some(path) = path {
        write_output_to_file(path, &legacy_errors)
    } else {
        write_output_to_stdout(&legacy_errors)
    }
}

impl Args {
    pub fn run(self) -> anyhow::Result<CommandExitStatus> {
        let input_file = read_input_file(self.input_path.as_path())?;
        let python_version = PythonVersion::from_str(&input_file.py_version)?;
        let sys_info = SysInfo::new(python_version, PythonPlatform::linux());
        let sourcedb = BuckSourceDatabase::from_manifest_files(
            input_file.sources.as_slice(),
            input_file.dependencies.as_slice(),
            input_file.typeshed.as_slice(),
        )?;
        let type_errors = compute_errors(sys_info, sourcedb);
        info!("Found {} type errors", type_errors.len());
        write_output(&type_errors, self.output_path.as_deref())?;
        Ok(CommandExitStatus::Success)
    }
}
