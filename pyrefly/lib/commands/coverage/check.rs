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
    #[arg(long, value_enum, default_value_t)]
    output_format: OutputFormat,
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

        let root = std::env::current_dir().unwrap_or_default();
        write_errors_to_console(self.output_format, &root, &errors)?;

        if coverage + 1e-9 >= self.fail_under {
            eprintln!("{} {summary}", Severity::Info.painted());
            Ok(CommandExitStatus::Success)
        } else {
            eprintln!(
                "{} {summary} is below the {:.2}% threshold",
                Severity::Error.painted(),
                self.fail_under
            );
            Ok(CommandExitStatus::UserError)
        }
    }
}

#[cfg(test)]
mod tests {
    use pyrefly_util::thread_pool::TEST_THREAD_COUNT;
    use tempfile::TempDir;

    use super::*;

    /// Run `pyrefly coverage check` on a one-file project with the given extra args.
    fn run_check(source: &str, extra_args: &[&str]) -> anyhow::Result<CommandExitStatus> {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("pyrefly.toml"),
            "search-path = ['.']\nskip-interpreter-query = true\n",
        )
        .unwrap();
        std::fs::write(dir.path().join("main.py"), source).unwrap();
        let file = dir.path().join("main.py").display().to_string();
        let args = ["coverage-check", file.as_str()]
            .into_iter()
            .chain(extra_args.iter().copied());
        CheckArgs::parse_from(args).run(None, TEST_THREAD_COUNT)
    }

    /// Coverage exactly at the threshold passes; just below it exits with a user error.
    #[test]
    fn test_fail_under_boundary() {
        let source = "def f(x) -> int: ...\n"; // 1 of 2 slots typed
        let check = |threshold| run_check(source, &["--fail-under", threshold]).unwrap();
        assert_eq!(check("50"), CommandExitStatus::Success);
        assert_eq!(check("50.01"), CommandExitStatus::UserError);
    }

    /// A fully-typed project passes the default 100% threshold.
    #[test]
    fn test_fully_typed_passes() {
        assert_eq!(
            run_check("def f(x: int) -> int: ...\n", &[]).unwrap(),
            CommandExitStatus::Success
        );
    }

    /// `Any` annotations count as covered by default, but not under `--strict` (gh-4024).
    #[test]
    fn test_strict_any() {
        let source = "from typing import Any\n\ndef f(x: Any) -> Any: ...\n";
        assert_eq!(run_check(source, &[]).unwrap(), CommandExitStatus::Success);
        assert_eq!(
            run_check(source, &["--strict"]).unwrap(),
            CommandExitStatus::UserError
        );
    }

    /// Out-of-range `--fail-under` values are rejected before any checking happens.
    #[test]
    fn test_fail_under_out_of_range() {
        for arg in ["--fail-under=100.1", "--fail-under=-0.1"] {
            let err = run_check("", &[arg]).unwrap_err();
            assert!(err.to_string().contains("--fail-under"), "{err}");
        }
    }
}
