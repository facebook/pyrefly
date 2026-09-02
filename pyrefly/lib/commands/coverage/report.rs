/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use clap::Parser;
use pyrefly_config::args::EnvironmentArgs;
use pyrefly_util::thread_pool::ThreadCount;

use crate::commands::config_finder::ConfigConfigurerWrapper;
use crate::commands::coverage::collect::calculate_summary;
use crate::commands::coverage::collect::collect_module_reports;
use crate::commands::coverage::types::FullReport;
use crate::commands::files::FilesArgs;
use crate::commands::util::CommandExitStatus;
use crate::config::config::ConfigScope;

/// `(major, minor)` version for the report JSON schema.
const REPORT_SCHEMA_VERSION: (u32, u32) = (0, 2);

/// Generate reports from pyrefly type checking results.
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Clone, Parser)]
pub struct ReportArgs {
    /// Which files to check.
    #[command(flatten)]
    files: FilesArgs,

    #[command(flatten)]
    config_override: EnvironmentArgs,

    /// When enabled, `.py` files are skipped if a corresponding `.pyi`
    /// file is also present in the set of files to check.
    #[clap(long, default_value_t = true, action = clap::ArgAction::Set)]
    prefer_stubs: bool,

    /// Override the module name in the report output. When set, all modules
    /// use this name instead of the name derived from the file path. Useful
    /// when reporting on a single module whose canonical package name differs
    /// from its filesystem layout.
    #[clap(long)]
    module: Option<String>,

    /// Only report symbols reachable from public modules via re-export chains.
    #[clap(long)]
    public_only: bool,
}

impl ReportArgs {
    pub fn run(
        self,
        wrapper: Option<ConfigConfigurerWrapper>,
        thread_count: ThreadCount,
    ) -> anyhow::Result<CommandExitStatus> {
        self.config_override.validate()?;

        if self.public_only && self.module.is_some() {
            anyhow::bail!("--module and --public-only cannot be combined");
        }

        let (files_to_check, config_finder, _) = self.files.resolve_scoped(
            self.config_override.into(),
            wrapper,
            ConfigScope::Coverage,
        )?;
        let (module_reports, _) = collect_module_reports(
            files_to_check,
            config_finder,
            self.prefer_stubs,
            self.module,
            self.public_only,
            None,
            thread_count,
        )?;
        let full_report = FullReport {
            schema_version: format!("{}.{}", REPORT_SCHEMA_VERSION.0, REPORT_SCHEMA_VERSION.1),
            summary: calculate_summary(&module_reports),
            module_reports,
        };
        println!("{}", serde_json::to_string_pretty(&full_report)?);

        Ok(CommandExitStatus::Success)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_report_has_schema_version() {
        let full = FullReport {
            schema_version: format!("{}.{}", REPORT_SCHEMA_VERSION.0, REPORT_SCHEMA_VERSION.1),
            module_reports: Vec::new(),
            summary: calculate_summary(&[]),
        };
        let json: serde_json::Value = serde_json::to_value(&full).unwrap();
        let version = json["schema_version"].as_str().unwrap();
        let parts: Vec<&str> = version.split('.').collect();
        assert_eq!(parts.len(), 2, "schema_version must be \"major.minor\"");
        assert!(parts[0].parse::<u32>().is_ok());
        assert!(parts[1].parse::<u32>().is_ok());
    }
}
