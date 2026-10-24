/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::io::Write;
use std::path::PathBuf;

use anyhow::Context as _;
use clap::Args;
use clap::Parser;
use clap::Subcommand;
use dupe::Dupe;
use lsp_types::CallHierarchyIncomingCall;
use lsp_types::CallHierarchyItem;
use lsp_types::CallHierarchyOutgoingCall;
use lsp_types::Location;
use lsp_types::Position;
use lsp_types::Url;
use pyrefly_build::handle::Handle;
use pyrefly_config::args::ConfigOverrideArgs;
use pyrefly_python::module::TextRangeWithModule;
use pyrefly_util::absolutize::Absolutize as _;
use pyrefly_util::thread_pool::ThreadCount;
use serde::Serialize;
use serde_json::Value;

use crate::commands::check::Handles;
use crate::commands::config_finder::ConfigConfigurerWrapper;
use crate::commands::files::FilesArgs;
use crate::commands::util::CommandExitStatus;
use crate::lsp::non_wasm::call_hierarchy::find_function_at_position_in_ast;
use crate::lsp::non_wasm::call_hierarchy::prepare_call_hierarchy_item;
use crate::lsp::non_wasm::call_hierarchy::transform_incoming_calls;
use crate::lsp::non_wasm::call_hierarchy::transform_outgoing_calls;
use crate::lsp::wasm::hover::get_hover;
use crate::state::lsp::FindPreference;
use crate::state::lsp::ImportBehavior;
use crate::state::require::Require;
use crate::state::state::State;

/// Query semantic information about a source position.
#[derive(Clone, Debug, Parser)]
pub struct QueryArgs {
    /// Semantic query to run.
    #[command(subcommand)]
    query: Query,

    /// Explicitly set the Pyrefly configuration to use.
    #[arg(long, short, global = true, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Configuration override options.
    #[command(flatten, next_help_heading = "Config Overrides")]
    config_override: ConfigOverrideArgs,
}

/// Semantic queries supported by the command-line interface.
#[derive(Clone, Debug, Subcommand)]
enum Query {
    /// Show hover information at a source position.
    Hover(QueryPosition),

    /// Find references to the symbol at a source position.
    FindReferences {
        /// Source position to query.
        #[command(flatten)]
        position: QueryPosition,

        /// Include the symbol's declaration in the results.
        #[arg(long)]
        include_declaration: bool,
    },

    /// Show incoming and outgoing calls for the callable at a source position.
    CallHierarchy(QueryPosition),
}

/// A one-based source position.
#[derive(Args, Clone, Debug)]
struct QueryPosition {
    /// Python file containing the position.
    file: PathBuf,

    /// One-based line number.
    line: u32,

    /// One-based UTF-16 column number, matching editor and LSP positions.
    column: u32,
}

impl QueryPosition {
    /// Convert the user-facing one-based position to an LSP position.
    fn to_lsp_position(&self) -> anyhow::Result<Position> {
        let line = self
            .line
            .checked_sub(1)
            .context("line must be greater than zero")?;
        let character = self
            .column
            .checked_sub(1)
            .context("column must be greater than zero")?;
        Ok(Position::new(line, character))
    }
}

/// JSON representation of a complete call hierarchy query.
#[derive(Serialize)]
struct CallHierarchyResult {
    /// The callable resolved from the requested source position.
    item: CallHierarchyItem,
    /// Functions that call the resolved callable.
    incoming: Vec<CallHierarchyIncomingCall>,
    /// Functions called by the resolved callable.
    outgoing: Vec<CallHierarchyOutgoingCall>,
}

impl QueryArgs {
    /// Run the query and write one JSON value to stdout.
    pub fn run(
        self,
        wrapper: Option<ConfigConfigurerWrapper>,
        thread_count: ThreadCount,
    ) -> anyhow::Result<CommandExitStatus> {
        let result = self.query(wrapper, thread_count)?;
        let stdout = std::io::stdout();
        let mut stdout = stdout.lock();
        serde_json::to_writer(&mut stdout, &result)?;
        writeln!(stdout)?;
        Ok(CommandExitStatus::Success)
    }

    /// Load the project and execute a semantic query.
    fn query(
        self,
        wrapper: Option<ConfigConfigurerWrapper>,
        thread_count: ThreadCount,
    ) -> anyhow::Result<Value> {
        self.config_override.validate()?;
        let position = match &self.query {
            Query::Hover(position) | Query::CallHierarchy(position) => position,
            Query::FindReferences { position, .. } => position,
        };
        let target_path = position.file.absolutize();
        let lsp_position = position.to_lsp_position()?;

        let (files_to_check, config_finder, _) =
            FilesArgs::get(Vec::new(), self.config, self.config_override, wrapper)?;
        let mut files = config_finder
            .checkpoint(files_to_check.files_iter())?
            .collect::<Vec<_>>();
        files.push(target_path.clone());
        let handles = Handles::new(files);
        let (handles, _, sourcedb_errors) = handles.all(&config_finder);
        if !sourcedb_errors.is_empty() {
            for error in sourcedb_errors {
                error.print();
            }
            return Err(anyhow::anyhow!("failed to query sourcedb"));
        }
        let target = handles
            .iter()
            .find(|handle| handle.path().as_path() == target_path)
            .cloned()
            .with_context(|| format!("could not resolve `{}`", target_path.display()))?;

        let state = State::new(config_finder, thread_count);
        let mut transaction = state.new_committable_transaction(Require::Everything, None);
        transaction
            .as_mut()
            .run(&handles, Require::Everything, None);
        state.commit_transaction(transaction, None);

        let mut transaction = state.cancellable_transaction();
        let module = transaction
            .as_ref()
            .get_module_info(&target)
            .with_context(|| format!("could not load `{}`", target_path.display()))?;
        let position = module.from_lsp_position(lsp_position, None);

        match self.query {
            Query::Hover(_) => Ok(serde_json::to_value(get_hover(
                transaction.as_ref(),
                &target,
                position,
                false,
            ))?),
            Query::FindReferences {
                include_declaration,
                ..
            } => {
                let definition = transaction
                    .as_ref()
                    .find_definition(
                        &target,
                        position,
                        FindPreference {
                            import_behavior: ImportBehavior::StopAtRenamedImports,
                            ..Default::default()
                        },
                    )
                    .map_err(|reason| {
                        anyhow::anyhow!("no definition found at the requested position: {reason:?}")
                    })?
                    .into_vec()
                    .into_iter()
                    .next()
                    .expect("find_definition always returns a nonempty Vec1");
                let references = transaction
                    .as_mut()
                    .find_global_references_from_definition(
                        *target.sys_info(),
                        definition.metadata,
                        TextRangeWithModule::new(definition.module, definition.definition_range),
                        include_declaration,
                    )
                    .map_err(|_| anyhow::anyhow!("reference query was cancelled"))?;
                let mut locations = Vec::new();
                for (module, ranges) in references {
                    let uri = Url::from_file_path(module.path().as_path()).map_err(|()| {
                        anyhow::anyhow!("reference in `{}` does not have a file URL", module.path())
                    })?;
                    locations.extend(
                        ranges
                            .into_iter()
                            .map(|range| Location::new(uri.clone(), module.to_lsp_range(range))),
                    );
                }
                locations.sort_by(|left, right| {
                    (&left.uri, left.range.start.line, left.range.start.character).cmp(&(
                        &right.uri,
                        right.range.start.line,
                        right.range.start.character,
                    ))
                });
                Ok(serde_json::to_value(locations)?)
            }
            Query::CallHierarchy(_) => {
                let definition = transaction
                    .as_ref()
                    .find_definition(&target, position, FindPreference::default())
                    .map_err(|reason| {
                        anyhow::anyhow!("no definition found at the requested position: {reason:?}")
                    })?
                    .into_vec()
                    .into_iter()
                    .next()
                    .expect("find_definition always returns a nonempty Vec1");
                let definition_handle = Handle::new(
                    definition.module.name(),
                    definition.module.path().dupe(),
                    target.sys_info().dupe(),
                );
                let ast = transaction
                    .as_ref()
                    .get_ast(&definition_handle)
                    .context("definition AST is unavailable")?;
                let function =
                    find_function_at_position_in_ast(&ast, definition.definition_range.start())
                        .context("the requested symbol is not a function or method")?;
                let uri = Url::from_file_path(definition.module.path().as_path())
                    .map_err(|()| anyhow::anyhow!("definition does not have a file URL"))?;
                let item = prepare_call_hierarchy_item(function, &definition.module, uri.clone());
                let incoming = transaction
                    .find_global_incoming_calls_from_function_definition(
                        *target.sys_info(),
                        definition.metadata,
                        &TextRangeWithModule::new(
                            definition.module.dupe(),
                            definition.definition_range,
                        ),
                    )
                    .map_err(|_| anyhow::anyhow!("incoming call query was cancelled"))?;
                let outgoing = transaction
                    .find_global_outgoing_calls_from_function_definition(
                        &definition_handle,
                        definition.definition_range.start(),
                    )
                    .map_err(|_| anyhow::anyhow!("outgoing call query was cancelled"))?;
                let mut incoming = transform_incoming_calls(incoming, None);
                let mut outgoing = transform_outgoing_calls(outgoing, &definition.module, &uri);
                incoming.sort_by_cached_key(|call| {
                    serde_json::to_string(call)
                        .expect("LSP call hierarchy results are always serializable")
                });
                outgoing.sort_by_cached_key(|call| {
                    serde_json::to_string(call)
                        .expect("LSP call hierarchy results are always serializable")
                });
                Ok(serde_json::to_value(CallHierarchyResult {
                    item,
                    incoming,
                    outgoing,
                })?)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use pyrefly_config::args::ConfigOverrideArgs;
    use pyrefly_util::thread_pool::TEST_THREAD_COUNT;

    use super::*;

    fn args(query: Query, config: PathBuf) -> QueryArgs {
        QueryArgs {
            query,
            config: Some(config),
            config_override: ConfigOverrideArgs::default(),
        }
    }

    #[test]
    fn queries_hover_references_and_call_hierarchy_as_json() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let config = root.path().join("pyrefly.toml");
        let callee = root.path().join("callee.py");
        let caller = root.path().join("caller.py");
        fs::write(
            &config,
            "project_includes = [\"*.py\"]\nproject_excludes = []\n",
        )?;
        fs::write(&callee, "def target(x: int) -> str:\n    return str(x)\n")?;
        fs::write(
            caller,
            "from callee import target\n\ndef caller() -> str:\n    return target(1)\n",
        )?;
        let position = || QueryPosition {
            file: callee.clone(),
            line: 1,
            column: 5,
        };

        let hover =
            args(Query::Hover(position()), config.clone()).query(None, TEST_THREAD_COUNT)?;
        assert!(hover.to_string().contains("(x: int) -> str"));

        let references = args(
            Query::FindReferences {
                position: position(),
                include_declaration: true,
            },
            config.clone(),
        )
        .query(None, TEST_THREAD_COUNT)?;
        assert_eq!(references.as_array().unwrap().len(), 3);

        let hierarchy =
            args(Query::CallHierarchy(position()), config).query(None, TEST_THREAD_COUNT)?;
        assert_eq!(hierarchy["item"]["name"], "target");
        assert_eq!(hierarchy["incoming"][0]["from"]["name"], "caller");
        Ok(())
    }

    #[test]
    fn positions_are_one_based() {
        let zero_line = QueryPosition {
            file: "test.py".into(),
            line: 0,
            column: 1,
        };
        assert!(zero_line.to_lsp_position().is_err());
        let position = QueryPosition {
            file: "test.py".into(),
            line: 2,
            column: 3,
        };
        assert_eq!(position.to_lsp_position().unwrap(), Position::new(1, 2));
    }
}
