/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use clap::Parser;
use pyrefly_config::args::ConfigOverrideArgs;
use pyrefly_config::error_kind::Severity;

use crate::commands::files::FilesArgs;
use crate::commands::util::CommandExitStatus;
use crate::error::suppress;
use crate::error::suppress::SuppressableError;

/// Suppress type errors by adding ignore comments to source files.
#[derive(Clone, Debug, Parser)]
pub struct SuppressArgs {
    /// Which files to check and suppress errors in.
    #[command(flatten)]
    files: FilesArgs,

    /// Configuration override options.
    #[command(flatten, next_help_heading = "Config Overrides")]
    config_override: ConfigOverrideArgs,
}

impl SuppressArgs {
    pub fn run(&self) -> anyhow::Result<CommandExitStatus> {
        self.config_override.validate()?;
        let (files_to_check, config_finder) =
            self.files.clone().resolve(self.config_override.clone())?;

        // Run type checking to collect errors
        let check_args =
            super::check::CheckArgs::parse_from(["check", "--output-format", "omit-errors"]);
        let (_, errors) = check_args.run_once(files_to_check, config_finder)?;

        // Convert to SuppressableErrors, filtering by severity
        let suppressable_errors: Vec<SuppressableError> = errors
            .into_iter()
            .filter(|e| e.severity() >= Severity::Warn)
            .filter_map(|e| SuppressableError::from_error(&e))
            .collect();

        // Apply suppressions
        suppress::suppress_errors(suppressable_errors);

        Ok(CommandExitStatus::Success)
    }
}
