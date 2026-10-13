/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::bail;
use clap::Parser;
use lsp_types::ServerInfo;
use pyrefly_config::config::ConfigFile;
use pyrefly_config::config::ConfigSource;
use pyrefly_config::finder::ConfigError;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::telemetry::Telemetry;
use pyrefly_util::thread_pool::ThreadCount;

use crate::commands::config_finder::ConfigConfigurer;
use crate::commands::config_finder::ConfigConfigurerWrapper;
use crate::commands::lsp::IndexingMode;
use crate::commands::util::CommandExitStatus;
use crate::lsp::non_wasm::external_provider::NoExternalProvider;
use crate::lsp::non_wasm::queue::LspQueue;
use crate::lsp::non_wasm::server::Connection;
use crate::lsp::non_wasm::server::InitializeInfo;
use crate::lsp::non_wasm::server::MessageReader;
use crate::lsp::non_wasm::server::Server;
use crate::lsp::non_wasm::server::initialize_finish;
use crate::lsp::non_wasm::server::initialize_start;
use crate::lsp::non_wasm::workspace::ServerMode;
use crate::tsp::server::tsp_capabilities;
use crate::tsp::server::tsp_loop;

/// Arguments for TSP server
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Parser, Clone)]
pub struct TspArgs {
    /// Find the struct that contains this field and add the indexing mode used by the language server
    #[arg(long, value_enum, default_value_t)]
    pub(crate) indexing_mode: IndexingMode,
    /// Sets the maximum number of user files for Pyrefly to index in the workspace.
    /// Note that indexing files is a performance-intensive task.
    #[arg(long, default_value_t = if cfg!(fbcode_build) {0} else {2000})]
    pub(crate) workspace_indexing_limit: usize,
    /// Selects the transport for the main JSON-RPC connection.
    /// Use `stdio` (default) or `ipc://<name>` for a local socket / named pipe.
    #[arg(long, default_value = "stdio")]
    pub(crate) transport: String,
    /// Use this config file for every file instead of discovering the nearest one.
    /// Lets a batch client such as a linter choose its own import resolution, for
    /// example explicit search paths in place of a slow build-system query.
    #[arg(long)]
    pub(crate) config: Option<PathBuf>,
}

/// Substitutes one explicit config for whatever config discovery found.
struct ExplicitConfig {
    config: ConfigFile,
    root: PathBuf,
    inner: Arc<dyn ConfigConfigurer>,
}

impl ConfigConfigurer for ExplicitConfig {
    fn configure(
        &self,
        _root: Option<&Path>,
        _discovered: ConfigFile,
        _discovered_errors: Vec<ConfigError>,
    ) -> (ArcId<ConfigFile>, Vec<ConfigError>) {
        // Relative paths in the explicit config are relative to its own directory.
        // Its parse errors were reported once at startup, so none are passed on.
        self.inner
            .configure(Some(&self.root), self.config.clone(), Vec::new())
    }
}

/// Wraps `wrapper` so that every config lookup yields the config at `path`.
///
/// The substitution is outermost, so `wrapper` and the inner configurer both see
/// the explicit config and its root, never the discovered one. Fails unless `path`
/// is a pyrefly config: a missing, unparsable, or marker-only file would otherwise
/// fall back to an auto-resolved config, which is what `--config` exists to avoid.
fn explicit_config_wrapper(
    path: &Path,
    wrapper: Option<ConfigConfigurerWrapper>,
) -> anyhow::Result<ConfigConfigurerWrapper> {
    let (config, errors) = ConfigFile::from_file(path);
    errors.iter().for_each(ConfigError::print);
    if !matches!(config.source, ConfigSource::File(_)) {
        bail!(
            "`--config {}` did not provide a pyrefly config",
            path.display()
        );
    }
    let root = path.parent().map_or_else(PathBuf::new, Path::to_path_buf);
    Ok(Arc::new(move |inner| -> Arc<dyn ConfigConfigurer> {
        let inner = match &wrapper {
            Some(outer) => outer(inner),
            None => inner,
        };
        Arc::new(ExplicitConfig {
            config: config.clone(),
            root: root.clone(),
            inner,
        })
    }))
}

pub fn run_tsp(
    connection: Connection,
    mut reader: MessageReader,
    args: TspArgs,
    telemetry: &impl Telemetry,
    wrapper: Option<ConfigConfigurerWrapper>,
    thread_count: ThreadCount,
    server_version: Option<String>,
) -> anyhow::Result<()> {
    let wrapper = match &args.config {
        Some(path) => Some(explicit_config_wrapper(path, wrapper)?),
        None => wrapper,
    };
    if let Some(initialize_info) = initialize_tsp_connection(
        &connection,
        &mut reader,
        args.indexing_mode,
        server_version.clone(),
    )? {
        // Create an LSP server instance for the TSP server to use.
        let lsp_queue = LspQueue::new();
        let surface = telemetry.surface();
        let agent_session_id = telemetry.agent_session_id();
        let agent_invocation_id = telemetry.agent_invocation_id();
        let lsp_server = Server::new(
            connection,
            lsp_queue,
            initialize_info.params.clone(),
            initialize_info.supports_diagnostic_markdown,
            args.indexing_mode,
            args.workspace_indexing_limit,
            false,
            ServerMode::TypeServer,
            surface,
            agent_session_id,
            agent_invocation_id,
            None, // No path remapping for TSP
            None, // No thrift remapping for TSP
            Arc::new(NoExternalProvider),
            wrapper,
            thread_count,
            Instant::now(),
            server_version,
        );

        // Reuse the existing lsp_loop but with TSP initialization
        tsp_loop(lsp_server, reader, initialize_info, telemetry)?;
    }
    Ok(())
}

fn initialize_tsp_connection(
    connection: &Connection,
    reader: &mut MessageReader,
    indexing_mode: IndexingMode,
    server_version: Option<String>,
) -> anyhow::Result<Option<InitializeInfo>> {
    let Some((id, initialize_info)) = initialize_start(&connection.sender, reader)? else {
        return Ok(None);
    };
    let capabilities = tsp_capabilities(indexing_mode, &initialize_info.params);
    let server_info = ServerInfo {
        name: "pyrefly-tsp".to_owned(),
        version: server_version,
    };
    if !initialize_finish(
        &connection.sender,
        reader,
        id,
        capabilities,
        Some(server_info),
    )? {
        return Ok(None);
    }
    Ok(Some(initialize_info))
}

impl TspArgs {
    pub fn run(
        self,
        telemetry: &impl Telemetry,
        wrapper: Option<ConfigConfigurerWrapper>,
        thread_count: ThreadCount,
        server_version: Option<String>,
    ) -> anyhow::Result<CommandExitStatus> {
        // Note that we must have our logging only write out to stderr.
        eprintln!("starting TSP server");

        let (connection, reader, io_threads) = Connection::from_transport(&self.transport)?;

        run_tsp(
            connection,
            reader,
            self,
            telemetry,
            wrapper,
            thread_count,
            server_version,
        )?;
        io_threads.join()?;
        // We have shut down gracefully.
        // Use writeln! instead of eprintln! to avoid panicking if stderr is closed.
        // This can happen, for example, when stderr is connected to an LSP client which
        // closes the connection before Pyrefly language server exits.
        let _ = writeln!(std::io::stderr(), "shutting down TSP server");
        Ok(CommandExitStatus::Success)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::Mutex;

    use tempfile::TempDir;

    use super::*;

    type Seen = Arc<Mutex<Vec<(Option<PathBuf>, ConfigSource)>>>;

    /// Records the root and source it is given, then delegates to `inner` (or
    /// returns the config when it is innermost).
    struct Recorder {
        seen: Seen,
        inner: Option<Arc<dyn ConfigConfigurer>>,
    }

    impl ConfigConfigurer for Recorder {
        fn configure(
            &self,
            root: Option<&Path>,
            config: ConfigFile,
            errors: Vec<ConfigError>,
        ) -> (ArcId<ConfigFile>, Vec<ConfigError>) {
            self.seen
                .lock()
                .unwrap()
                .push((root.map(Path::to_path_buf), config.source.clone()));
            match &self.inner {
                Some(inner) => inner.configure(root, config, errors),
                None => (ArcId::new(config), errors),
            }
        }
    }

    #[test]
    fn test_outer_wrapper_sees_the_explicit_config_and_root() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("pyrefly.toml");
        fs::write(&path, "search-path = [\"libs\"]\n").unwrap();
        let seen = Seen::default();
        let outer_seen = seen.clone();
        let outer: ConfigConfigurerWrapper = Arc::new(move |inner| {
            Arc::new(Recorder {
                seen: outer_seen.clone(),
                inner: Some(inner),
            })
        });
        let innermost = Arc::new(Recorder {
            seen: seen.clone(),
            inner: None,
        });

        let configurer = explicit_config_wrapper(&path, Some(outer)).unwrap()(innermost);
        configurer.configure(
            Some(Path::new("/discovered")),
            ConfigFile::default(),
            Vec::new(),
        );

        let expected = (Some(dir.path().to_path_buf()), ConfigSource::File(path));
        assert_eq!(
            *seen.lock().unwrap(),
            vec![expected.clone(), expected],
            "the outer wrapper and the inner configurer should both see the explicit config"
        );
    }

    #[test]
    fn test_rejects_a_path_that_is_not_a_pyrefly_config() {
        let dir = TempDir::new().unwrap();
        let tool_only = dir.path().join("pyproject.toml");
        fs::write(&tool_only, "[tool.ruff]\nline-length = 88\n").unwrap();
        for path in [dir.path().join("missing.toml"), tool_only] {
            let error = explicit_config_wrapper(&path, None)
                .err()
                .unwrap_or_else(|| panic!("expected `{}` to be rejected", path.display()));
            assert!(
                error
                    .to_string()
                    .contains("did not provide a pyrefly config"),
                "unexpected error for `{}`: {error}",
                path.display()
            );
        }
    }
}
