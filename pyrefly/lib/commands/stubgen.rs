/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;

use clap::Parser;
use pyrefly_config::args::ConfigOverrideArgs;
use pyrefly_util::args::clap_env;

use crate::commands::config_finder::ConfigConfigurerWrapper;
use crate::commands::files::FilesArgs;
use crate::commands::util::CommandExitStatus;

/// Arguments for the stubgen command, which generates PEP 484 `.pyi` stub files
/// from Python source.
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Clone, Parser)]
pub struct StubgenArgs {
    /// Which files to generate stubs for.
    #[command(flatten)]
    files: FilesArgs,

    /// Type checking arguments and configuration.
    #[command(flatten)]
    config_override: ConfigOverrideArgs,

    /// Output directory for generated `.pyi` files.
    #[arg(
        long,
        short,
        default_value = "out",
        env = clap_env("STUBGEN_OUTPUT")
    )]
    output: PathBuf,

    /// Include private names (those starting with `_`) in generated stubs.
    #[arg(long, default_value = "false")]
    include_private: bool,

    /// Continue generating stubs even if errors occur for some files.
    #[arg(long, default_value = "false")]
    ignore_errors: bool,
}

impl StubgenArgs {
    /// Resolve inputs, create the output directory, and generate stubs.
    pub fn run(
        self,
        wrapper: Option<ConfigConfigurerWrapper>,
    ) -> anyhow::Result<CommandExitStatus> {
        self.config_override.validate()?;
        let (_files_to_check, _config_finder) =
            self.files.resolve(self.config_override, wrapper)?;

        std::fs::create_dir_all(&self.output)?;
        eprintln!("Created output directory: {}", self.output.display());

        // TODO(rayahhhmed): call stub generator for each resolved file and write .pyi output.

        Ok(CommandExitStatus::Success)
    }
}
