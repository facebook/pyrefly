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
use crate::error::suppress::UnusedIgnoreKind;

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

    /// Remove unused ignores instead of adding suppressions, optionally selecting `pyrefly`, `type`, or `all`.
    /// Defaults to `pyrefly` when no kind is specified.
    #[arg(
        long,
        value_enum,
        value_name = "KIND",
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "pyrefly"
    )]
    remove_unused: Option<UnusedIgnoreKind>,

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
        if let Some(kind) = self.remove_unused {
            // Remove unused ignores mode
            let unused_errors: Vec<SerializedError> = if let Some(json_path) = &self.json {
                // Parse errors from JSON file, filtering for unused suppression errors only.
                let json_content = std::fs::read_to_string(json_path)?;
                let errors: Vec<SerializedError> = serde_json::from_str(&json_content)?;
                errors
                    .into_iter()
                    .filter(|e| {
                        kind.includes_pyrefly_or_pyre() && e.is_unused_ignore()
                            || kind.includes_type() && e.is_unused_type_ignore()
                    })
                    .collect()
            } else {
                // Delegate to `check --remove-unused-ignores`, which
                // collects unused ignore errors directly (bypassing severity
                // filtering) and handles removal in one step.
                self.config_override.validate()?;
                let (files_to_check, config_finder, upsell) = self
                    .files
                    .clone()
                    .resolve(self.config_override.clone(), wrapper.clone())?;

                let remove_unused_flag = match kind {
                    UnusedIgnoreKind::Pyrefly => "--remove-unused-ignores=pyrefly",
                    UnusedIgnoreKind::Type => "--remove-unused-ignores=type",
                    UnusedIgnoreKind::All => "--remove-unused-ignores=all",
                };
                let check_args = CheckArgs::parse_from([
                    "check",
                    "--output-format",
                    "omit-errors",
                    remove_unused_flag,
                ]);
                check_args.run_once(files_to_check, config_finder, upsell, thread_count)?;
                return Ok(CommandExitStatus::Success);
            };

            // Remove unused ignores (JSON path only)
            suppress::remove_unused_ignores_from_serialized(unused_errors, kind);
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
                let (files_to_check, config_finder, upsell) = self
                    .files
                    .clone()
                    .resolve(self.config_override.clone(), wrapper)?;

                let check_args = CheckArgs::parse_from(["check", "--output-format", "omit-errors"]);
                let (_, errors, _check_result) =
                    check_args.run_once(files_to_check, config_finder, upsell, thread_count)?;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remove_unused_cli_values() {
        for (argument, expected) in [
            (None, None),
            (Some("--remove-unused"), Some(UnusedIgnoreKind::Pyrefly)),
            (
                Some("--remove-unused=pyrefly"),
                Some(UnusedIgnoreKind::Pyrefly),
            ),
            (Some("--remove-unused=type"), Some(UnusedIgnoreKind::Type)),
            (Some("--remove-unused=all"), Some(UnusedIgnoreKind::All)),
        ] {
            let args = argument.map_or_else(
                || SuppressArgs::parse_from(["suppress"]),
                |argument| SuppressArgs::parse_from(["suppress", argument]),
            );
            assert_eq!(args.remove_unused, expected);
        }
    }
}
