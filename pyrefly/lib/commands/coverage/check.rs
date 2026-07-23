/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use anstream::eprintln;
use clap::Parser;
use pyrefly_config::args::ConfigOverrideArgs;
use pyrefly_config::config::OutputFormat;
use pyrefly_config::error_kind::Severity;
use pyrefly_util::display::number_thousands;
use pyrefly_util::thread_pool::ThreadCount;

use crate::commands::check::write_errors_to_console;
use crate::commands::config_finder::ConfigConfigurerWrapper;
use crate::commands::coverage::collect::collect_module_reports;
use crate::commands::coverage::types::SlotCounts;
use crate::commands::files::FilesArgs;
use crate::commands::util::CommandExitStatus;
use crate::config::config::ConfigScope;

/// Gate type-annotation coverage against a threshold.
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Clone, Parser)]
pub struct CheckArgs {
    /// Which files to check.
    #[command(flatten)]
    files: FilesArgs,

    #[command(flatten)]
    config_override: ConfigOverrideArgs,

    /// Count `Any`-resolved annotations as untyped.
    #[clap(long)]
    strict: bool,

    /// Minimum coverage percentage; exit non-zero when coverage is below it.
    #[clap(long, short = 'f', value_name = "PERCENT", default_value_t = 100.0)]
    fail_under: f64,

    /// Prefer `.pyi` stubs over `.py` files when both are present.
    #[clap(long, default_value_t = true, action = clap::ArgAction::Set)]
    prefer_stubs: bool,

    /// Only check symbols reachable from public modules via re-export chains.
    #[clap(long)]
    public_only: bool,

    /// Format for the untyped-symbol findings.
    #[arg(long, value_enum)]
    output_format: Option<OutputFormat>,
}

impl CheckArgs {
    pub fn run(
        self,
        wrapper: Option<ConfigConfigurerWrapper>,
        thread_count: ThreadCount,
    ) -> anyhow::Result<CommandExitStatus> {
        self.config_override.validate()?;
        if !(0.0..=100.0).contains(&self.fail_under) {
            anyhow::bail!(
                "--fail-under must be between 0 and 100, got {}",
                self.fail_under
            );
        }

        let (files_to_check, config_finder, _) =
            self.files
                .resolve_scoped(self.config_override, wrapper, ConfigScope::Coverage)?;
        let (module_reports, errors) = collect_module_reports(
            files_to_check,
            config_finder,
            self.prefer_stubs,
            None,
            self.public_only,
            Some(self.strict),
            thread_count,
        )?;

        let total = module_reports
            .iter()
            .fold(SlotCounts::default(), |acc, m| acc.merge(m.slots));
        let (coverage, covered, label) = if self.strict {
            (
                total.strict_coverage(),
                total.n_typed,
                "strict type coverage",
            )
        } else {
            (
                total.coverage(),
                total.n_typed + total.n_any,
                "type coverage",
            )
        };
        let summary = format!(
            "{label} {coverage:.2}% ({} of {} typable)",
            number_thousands(covered),
            number_thousands(total.n_typable),
        );

        if coverage + 1e-9 >= self.fail_under {
            eprintln!("{} {summary}", Severity::Info.painted());
            Ok(CommandExitStatus::Success)
        } else {
            let root = std::env::current_dir().unwrap_or_default();
            write_errors_to_console(self.output_format.unwrap_or_default(), &root, &errors)?;
            eprintln!(
                "{} {summary} is below the {:.2}% threshold",
                Severity::Error.painted(),
                self.fail_under
            );
            Ok(CommandExitStatus::UserError)
        }
    }
}
