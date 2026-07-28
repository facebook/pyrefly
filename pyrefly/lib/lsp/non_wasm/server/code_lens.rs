/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Server orchestration for runnable and reference-count CodeLens providers.

mod reference_count;
mod runnable;

use lsp_server::ErrorCode;
use lsp_server::RequestId;
use lsp_types::CodeLens;
use lsp_types::CodeLensParams;
use lsp_types::request::CodeLensRequest;
use lsp_types::request::Request as _;
use pyrefly_python::module::TextRangeWithModule;
use pyrefly_util::telemetry::ActivityKey;
use pyrefly_util::telemetry::EmptyResponseReason;

use super::Server;
use crate::commands::lsp::IndexingMode;
use crate::lsp::non_wasm::protocol::Message;
use crate::lsp::non_wasm::protocol::Response;
use crate::state::lsp::FindDefinitionItemWithDocstring;
use crate::state::lsp::FindPreference;
use crate::state::lsp::ImportBehavior;
use crate::state::lsp::ReferenceOptions;
use crate::state::state::Transaction;

impl Server {
    /// Return eager runnable lenses and unresolved reference-count lenses for a Python file.
    pub(super) fn code_lens(
        &self,
        transaction: &Transaction<'_>,
        params: CodeLensParams,
    ) -> Result<Option<Vec<CodeLens>>, EmptyResponseReason> {
        let uri = &params.text_document.uri;
        if self.open_notebook_cells.read().contains_key(uri) {
            return Ok(Some(Vec::new()));
        }

        let handle = self.make_handle_if_enabled(uri, Some(CodeLensRequest::METHOD))?;
        let mut lenses = runnable::code_lenses(self, transaction, &handle, uri)?;
        if self.indexing_mode != IndexingMode::None {
            lenses.extend(reference_count::unresolved_code_lenses(
                transaction,
                &handle,
                uri,
            )?);
        }
        Ok(Some(lenses))
    }

    /// Resolve one visible reference-count lens on the references queue.
    pub(super) fn resolve_code_lens<'a>(
        &'a self,
        request_id: RequestId,
        transaction: &Transaction<'a>,
        lens: &CodeLens,
        activity_key: Option<ActivityKey>,
    ) -> Result<(), EmptyResponseReason> {
        let data = match reference_count::resolve_data(lens) {
            Ok(data) => data,
            Err(message) => {
                self.connection.send(Message::Response(Response::new_err(
                    request_id,
                    ErrorCode::InvalidParams as i32,
                    message,
                )));
                return Ok(());
            }
        };
        if self.open_notebook_cells.read().contains_key(&data.uri) {
            return Err(EmptyResponseReason::NotebookNotSupported);
        }

        let handle = self.make_handle_if_enabled(&data.uri, Some(CodeLensRequest::METHOD))?;
        let path_remapper = self.path_remapper.clone();
        let source_uri = data.uri;
        let source_uri_for_response = source_uri.clone();
        let lens = lens.clone();
        self.async_find_from_definition_helper(
            request_id,
            transaction,
            handle,
            &source_uri,
            lens.range.start,
            FindPreference {
                import_behavior: ImportBehavior::StopAtRenamedImports,
                ..Default::default()
            },
            activity_key,
            move |transaction, handle, definition, _telemetry, _telemetry_event| {
                let FindDefinitionItemWithDocstring {
                    metadata,
                    definition_range,
                    module,
                    ..
                } = definition;
                Ok(transaction.find_global_references_from_definition(
                    *handle.sys_info(),
                    metadata,
                    TextRangeWithModule::new(module, definition_range),
                    ReferenceOptions::all(false),
                )?)
            },
            move |results| {
                reference_count::resolve_code_lens(
                    lens,
                    &source_uri_for_response,
                    results,
                    path_remapper.as_ref(),
                )
            },
        )
    }
}
