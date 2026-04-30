/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;
use std::ffi::OsString;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use clap::CommandFactory;
use clap::Parser;
use clap::ValueEnum;
use lsp_types::ServerInfo;
use pyrefly_util::telemetry::Telemetry;
use pyrefly_util::thread_pool::ThreadCount;

use crate::commands::config_finder::ConfigConfigurerWrapper;
use crate::commands::util::CommandExitStatus;
use crate::commands::util::CommonGlobalArgs;
use crate::lsp::non_wasm::external_provider::ExternalProvider;
use crate::lsp::non_wasm::module_helpers::PathRemapper;
use crate::lsp::non_wasm::module_helpers::ThriftRemapper;
use crate::lsp::non_wasm::server::Connection;
use crate::lsp::non_wasm::server::InitializeInfo;
use crate::lsp::non_wasm::server::MessageReader;
use crate::lsp::non_wasm::server::capabilities;
use crate::lsp::non_wasm::server::initialize_finish;
use crate::lsp::non_wasm::server::initialize_start;
use crate::lsp::non_wasm::server::lsp_loop;

/// Pyrefly's indexing strategy for open projects when performing go-to-definition
/// requests.
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub enum IndexingMode {
    /// Do not index anything. Features that depend on indexing (e.g. find-refs) will be disabled.
    None,
    /// Start indexing when opening a file that belongs to a config in the background.
    /// Indexing will happen in another thread, so that normal IDE services are not blocked.
    #[default]
    LazyNonBlockingBackground,
    /// Start indexing when opening a file that belongs to a config in the background.
    /// Indexing will happen in the main thread, so that IDE services will be blocked.
    /// However, this is useful for deterministic testing.
    LazyBlocking,
}

/// Arguments for LSP server
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Parser, Clone)]
pub struct LspArgs {
    /// Find the struct that contains this field and add the indexing mode used by the language server
    #[arg(long, value_enum, default_value_t)]
    pub indexing_mode: IndexingMode,

    /// Sets the maximum number of user files for Pyrefly to index in the workspace.
    /// Note that indexing files is a performance-intensive task.
    #[arg(long, default_value_t = if cfg!(fbcode_build) {0} else {2000})]
    pub workspace_indexing_limit: usize,

    /// Block for build system operations, only using fallback heuristics after checking
    /// an up-to-date source DB. Only useful for benchmarking.
    #[arg(long)]
    pub build_system_blocking: bool,

    /// Enable external references integration for cross-repo go-to-definition.
    #[arg(long, hide = true)]
    pub enable_external_references: bool,
}

/// Drop flags after the `lsp` subcommand that aren't declared on `LspArgs` or
/// `CommonGlobalArgs`. Lets older Pyrefly binaries tolerate flags introduced in
/// newer Pyls extensions instead of failing to start the LSP server.
pub fn filter_unrecognized_lsp_args(args: Vec<OsString>) -> Vec<OsString> {
    let Some(lsp_pos) = args.iter().position(|arg| arg == "lsp") else {
        return args;
    };

    let mut known_long: HashSet<String> = HashSet::new();
    let mut known_short: HashSet<char> = HashSet::new();
    for cmd in [LspArgs::command(), CommonGlobalArgs::command()] {
        for arg in cmd.get_arguments() {
            if let Some(long) = arg.get_long() {
                known_long.insert(long.to_owned());
            }
            if let Some(short) = arg.get_short() {
                known_short.insert(short);
            }
        }
    }

    let mut result: Vec<OsString> = args[..=lsp_pos].to_vec();
    let mut iter = args[lsp_pos + 1..].iter();
    while let Some(arg) = iter.next() {
        let arg_string = arg.to_string_lossy();
        if arg_string == "--" {
            result.push(arg.clone());
            result.extend(iter.cloned());
            break;
        }
        let keep = if let Some(flag_body) = arg_string.strip_prefix("--") {
            let flag_name = flag_body.split('=').next().unwrap_or(flag_body);
            known_long.contains(flag_name)
        } else if let Some(short_body) = arg_string.strip_prefix('-')
            && let Some(c) = short_body.chars().next()
            && short_body.chars().count() == 1
        {
            known_short.contains(&c)
        } else {
            true
        };
        if keep {
            result.push(arg.clone());
        }
    }
    result
}

/// Run LSP server with optional path remapping.
/// When a path remapper is provided, go-to-definition will use the remapped
/// paths for URIs, allowing navigation to source files instead of installed
/// package files.
pub fn run_lsp(
    connection: Connection,
    mut reader: MessageReader,
    args: LspArgs,
    server_info: Option<ServerInfo>,
    path_remapper: Option<PathRemapper>,
    thrift_remapper: Option<ThriftRemapper>,
    telemetry: &impl Telemetry,
    external_references: Arc<dyn ExternalProvider>,
    wrapper: Option<ConfigConfigurerWrapper>,
    thread_count: ThreadCount,
) -> anyhow::Result<()> {
    let lsp_start_time = Instant::now();
    if let Some(initialize_info) =
        initialize_connection(&connection, &mut reader, args.indexing_mode, server_info)?
    {
        lsp_loop(
            connection,
            reader,
            initialize_info,
            args.indexing_mode,
            args.workspace_indexing_limit,
            args.build_system_blocking,
            path_remapper,
            thrift_remapper,
            telemetry,
            external_references,
            wrapper,
            thread_count,
            lsp_start_time,
        )?;
    }
    Ok(())
}

fn initialize_connection(
    connection: &Connection,
    reader: &mut MessageReader,
    indexing_mode: IndexingMode,
    server_info: Option<ServerInfo>,
) -> anyhow::Result<Option<InitializeInfo>> {
    let Some((id, initialize_info)) = initialize_start(&connection.sender, reader)? else {
        return Ok(None);
    };
    let capabilities = capabilities(indexing_mode, &initialize_info.params);
    if !initialize_finish(&connection.sender, reader, id, capabilities, server_info)? {
        return Ok(None);
    }
    Ok(Some(initialize_info))
}

impl LspArgs {
    /// Run LSP with optional path remapping.
    /// When a path remapper is provided, go-to-definition will navigate to
    /// remapped source files instead of installed package files.
    pub fn run(
        self,
        version: &str,
        path_remapper: Option<PathRemapper>,
        thrift_remapper: Option<ThriftRemapper>,
        telemetry: &impl Telemetry,
        external_references: Arc<dyn ExternalProvider>,
        wrapper: Option<ConfigConfigurerWrapper>,
        thread_count: ThreadCount,
    ) -> anyhow::Result<CommandExitStatus> {
        // Note that we must have our logging only write out to stderr.
        eprintln!("starting generic LSP server");

        // Create the transport. Includes the stdio (stdin and stdout) versions but this could
        // also be implemented to use sockets or HTTP.
        let (connection, reader, io_threads) = Connection::stdio();

        let server_info = ServerInfo {
            name: "pyrefly-lsp".to_owned(),
            version: Some(version.to_owned()),
        };

        run_lsp(
            connection,
            reader,
            self,
            Some(server_info),
            path_remapper,
            thrift_remapper,
            telemetry,
            external_references,
            wrapper,
            thread_count,
        )?;
        io_threads.join()?;
        // We have shut down gracefully.
        // Use writeln! instead of eprintln! to avoid panicking if stderr is closed.
        // This can happen, for example, when stderr is connected to an LSP client which
        // closes the connection before Pyrefly language server exits.
        let _ = writeln!(std::io::stderr(), "shutting down server");
        Ok(CommandExitStatus::Success)
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use super::*;

    fn os(args: &[&str]) -> Vec<OsString> {
        args.iter().map(OsString::from).collect()
    }

    #[test]
    fn filter_preserves_args_when_not_lsp() {
        let args = os(&["pyrefly", "check", "--bogus-flag", "file.py"]);
        let result = filter_unrecognized_lsp_args(args.clone());
        assert_eq!(result, args);
    }

    #[test]
    fn filter_preserves_known_lsp_flags() {
        let args = os(&["pyrefly", "lsp", "--enable-external-references"]);
        let result = filter_unrecognized_lsp_args(args.clone());
        assert_eq!(result, args);
    }

    #[test]
    fn filter_strips_unknown_flag() {
        let args = os(&["pyrefly", "lsp", "--some-future-flag"]);
        let result = filter_unrecognized_lsp_args(args);
        assert_eq!(result, os(&["pyrefly", "lsp"]));
    }

    #[test]
    fn filter_strips_unknown_but_keeps_known() {
        let args = os(&[
            "pyrefly",
            "lsp",
            "--enable-external-references",
            "--unknown-flag",
            "--indexing-mode",
            "none",
        ]);
        let result = filter_unrecognized_lsp_args(args);
        assert_eq!(
            result,
            os(&[
                "pyrefly",
                "lsp",
                "--enable-external-references",
                "--indexing-mode",
                "none"
            ])
        );
    }

    #[test]
    fn filter_strips_unknown_flag_with_equals_value() {
        let args = os(&[
            "pyrefly",
            "lsp",
            "--unknown=value",
            "--enable-external-references",
        ]);
        let result = filter_unrecognized_lsp_args(args);
        assert_eq!(
            result,
            os(&["pyrefly", "lsp", "--enable-external-references"])
        );
    }

    #[test]
    fn filter_preserves_known_short_flag() {
        let args = os(&["pyrefly", "lsp", "-v"]);
        let result = filter_unrecognized_lsp_args(args.clone());
        assert_eq!(result, args);
    }

    #[test]
    fn filter_strips_unknown_short_flag() {
        let args = os(&["pyrefly", "lsp", "-x", "-v"]);
        let result = filter_unrecognized_lsp_args(args);
        assert_eq!(result, os(&["pyrefly", "lsp", "-v"]));
    }

    #[test]
    fn filter_preserves_short_flag_value() {
        let args = os(&["pyrefly", "lsp", "-j", "4"]);
        let result = filter_unrecognized_lsp_args(args.clone());
        assert_eq!(result, args);
    }

    #[test]
    fn filter_preserves_args_before_lsp() {
        let args = os(&["pyrefly", "--verbose", "lsp", "--unknown"]);
        let result = filter_unrecognized_lsp_args(args);
        assert_eq!(result, os(&["pyrefly", "--verbose", "lsp"]));
    }

    #[test]
    fn filter_preserves_everything_after_double_dash() {
        let args = os(&[
            "pyrefly",
            "lsp",
            "--unknown-flag",
            "--",
            "--another-unknown",
            "-x",
            "positional",
        ]);
        let result = filter_unrecognized_lsp_args(args);
        assert_eq!(
            result,
            os(&[
                "pyrefly",
                "lsp",
                "--",
                "--another-unknown",
                "-x",
                "positional"
            ])
        );
    }
}
