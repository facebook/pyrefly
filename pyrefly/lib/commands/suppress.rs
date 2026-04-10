/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;

use clap::Parser;
use pyrefly_config::args::ConfigOverrideArgs;
use pyrefly_util::thread_pool::ThreadCount;

use crate::commands::check::CheckArgs;
use crate::commands::config_finder::ConfigConfigurerWrapper;
use crate::commands::files::FilesArgs;
use crate::commands::util::CommandExitStatus;
use crate::error::suppress;
use crate::error::suppress::CommentLocation;
use crate::error::suppress::SerializedError;

/// Suppress type errors by adding ignore comments to source files.
#[derive(Clone, Debug, Parser)]
pub struct SuppressArgs {
    /// Which files to check and suppress errors in.
    #[command(flatten)]
    files: FilesArgs,

    /// Configuration override options.
    #[command(flatten, next_help_heading = "Config Overrides")]
    config_override: ConfigOverrideArgs,

    /// Path to a JSON file containing errors to suppress.
    /// The JSON should be an array of objects with "path", "line", "name", and "message" fields.
    #[arg(long)]
    json: Option<PathBuf>,

    /// Remove unused ignore comments instead of adding suppressions.
    #[arg(long)]
    remove_unused: bool,

    /// Where to place suppression comments: on the line before the error
    /// (`line-before`, the default) or on the same line (`same-line`).
    #[arg(long, default_value = "line-before")]
    comment_location: CommentLocation,
}

impl SuppressArgs {
    pub fn run(
        &self,
        wrapper: Option<ConfigConfigurerWrapper>,
        thread_count: ThreadCount,
    ) -> anyhow::Result<CommandExitStatus> {
        if self.remove_unused {
            // Remove unused ignores mode
            let unused_errors: Vec<SerializedError> = if let Some(json_path) = &self.json {
                // Parse errors from JSON file, filtering for UnusedIgnore errors only
                let json_content = std::fs::read_to_string(json_path)?;
                let errors: Vec<SerializedError> = serde_json::from_str(&json_content)?;
                errors
                    .into_iter()
                    .filter(|e| e.is_unused_ignore())
                    .collect()
            } else {
                // Run type checking to collect unused ignore errors
                self.config_override.validate()?;
                let (files_to_check, config_finder) = self
                    .files
                    .clone()
                    .resolve(self.config_override.clone(), wrapper.clone())?;

                let check_args = CheckArgs::parse_from(["check", "--output-format", "omit-errors"]);
                let (_, errors, _check_result) =
                    check_args.run_once(files_to_check, config_finder, thread_count)?;

                // Convert to SerializedErrors, filtering for UnusedIgnore only
                errors
                    .into_iter()
                    .filter_map(|e| SerializedError::from_error(&e))
                    .filter(|e| e.is_unused_ignore())
                    .collect()
            };

            // Remove unused ignores
            suppress::remove_unused_ignores_from_serialized(unused_errors);
        } else {
            // Add suppressions mode (existing behavior)
            let serialized_errors: Vec<SerializedError> = if let Some(json_path) = &self.json {
                // Parse errors from JSON file, filtering out directives and UnusedIgnore errors
                let json_content = std::fs::read_to_string(json_path)?;
                let errors: Vec<SerializedError> = serde_json::from_str(&json_content)?;
                errors
                    .into_iter()
                    .filter(|e| !e.is_directive() && !e.is_unused_ignore())
                    .collect()
            } else {
                // Run type checking to collect errors
                self.config_override.validate()?;
                let (files_to_check, config_finder) = self
                    .files
                    .clone()
                    .resolve(self.config_override.clone(), wrapper)?;

                let check_args = CheckArgs::parse_from(["check", "--output-format", "omit-errors"]);
                let (_, errors, _check_result) =
                    check_args.run_once(files_to_check, config_finder, thread_count)?;

                // Convert to SerializedErrors for all user-visible errors,
                // excluding directives (e.g. reveal_type) and UnusedIgnore
                errors
                    .into_iter()
                    .filter(|e| !e.error_kind().is_directive())
                    .filter_map(|e| SerializedError::from_error(&e))
                    .filter(|e| !e.is_unused_ignore())
                    .collect()
            };

            // Apply suppressions
            suppress::suppress_errors(serialized_errors, self.comment_location);
        }

        Ok(CommandExitStatus::Success)
    }
}
