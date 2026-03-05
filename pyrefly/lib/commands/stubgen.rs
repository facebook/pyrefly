/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;

use clap::Parser;
use dupe::Dupe;
use pyrefly_config::args::ConfigOverrideArgs;
use pyrefly_util::forgetter::Forgetter;
use pyrefly_util::fs_anyhow;

use crate::commands::check::Handles;
use crate::commands::config_finder::ConfigConfigurerWrapper;
use crate::commands::files::FilesArgs;
use crate::commands::util::CommandExitStatus;
use crate::state::require::Require;
use crate::state::state::State;
use crate::stubgen::emit::emit_stub;
use crate::stubgen::extract::ExtractConfig;
use crate::stubgen::extract::extract_module_stub;

/// Generate .pyi stub files from Python source files.
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Parser, Clone)]
pub struct StubgenArgs {
    /// Which files to generate stubs for.
    #[command(flatten)]
    files: FilesArgs,

    /// Type checking arguments and configuration.
    #[command(flatten)]
    config_override: ConfigOverrideArgs,

    /// Output directory for generated .pyi files.
    #[arg(short = 'o', long, default_value = "out")]
    output_dir: PathBuf,

    /// Include private names (single leading underscore).
    #[arg(long)]
    include_private: bool,

    /// Preserve docstrings in generated stubs.
    #[arg(long)]
    include_docstrings: bool,
}

impl StubgenArgs {
    pub fn run(
        self,
        wrapper: Option<ConfigConfigurerWrapper>,
    ) -> anyhow::Result<CommandExitStatus> {
        self.config_override.validate()?;
        let (files_to_check, config_finder) = self.files.resolve(self.config_override, wrapper)?;

        let expanded_file_list = config_finder.checkpoint(files_to_check.files())?;
        let state = State::new(config_finder);
        let holder = Forgetter::new(state, false);
        let handles = Handles::new(expanded_file_list);
        let mut forgetter = Forgetter::new(
            holder.as_ref().new_transaction(Require::Everything, None),
            true,
        );
        let transaction = forgetter.as_mut();

        let (handles, _, sourcedb_errors) = handles.all(holder.as_ref().config_finder());
        if !sourcedb_errors.is_empty() {
            for error in sourcedb_errors {
                error.print();
            }
            return Err(anyhow::anyhow!("Failed to query sourcedb."));
        }

        let config = ExtractConfig {
            include_private: self.include_private,
            include_docstrings: self.include_docstrings,
        };

        // Compute common prefix for output path mirroring.
        let all_paths: Vec<PathBuf> = handles
            .iter()
            .map(|h| h.path().as_path().to_path_buf())
            .collect();
        let common_prefix = common_path_prefix(&all_paths);

        for handle in &handles {
            transaction.run(&[handle.dupe()], Require::Everything, None);

            let module_stub = extract_module_stub(transaction, handle, &config);

            if let Some(stub) = module_stub {
                let stub_text = emit_stub(&stub);

                // Compute output path.
                let source_path = handle.path().as_path();
                let relative = source_path
                    .strip_prefix(&common_prefix)
                    .unwrap_or(source_path);
                let mut output_path = self.output_dir.join(relative);
                output_path.set_extension("pyi");

                // Create parent directories.
                if let Some(parent) = output_path.parent() {
                    fs_anyhow::create_dir_all(parent)?;
                }

                fs_anyhow::write(&output_path, stub_text)?;
            }
        }

        Ok(CommandExitStatus::Success)
    }
}

/// Find the longest common directory prefix of a set of paths.
fn common_path_prefix(paths: &[PathBuf]) -> PathBuf {
    if paths.is_empty() {
        return PathBuf::new();
    }
    if paths.len() == 1 {
        return paths[0].parent().unwrap_or(&paths[0]).to_path_buf();
    }
    let mut prefix = paths[0].clone();
    // Walk up until we find a directory that is a prefix of all paths.
    if prefix.is_file() || !prefix.is_dir() {
        prefix = prefix.parent().unwrap_or(&prefix).to_path_buf();
    }
    for path in &paths[1..] {
        while !path.starts_with(&prefix) {
            if !prefix.pop() {
                return PathBuf::new();
            }
        }
    }
    prefix
}
