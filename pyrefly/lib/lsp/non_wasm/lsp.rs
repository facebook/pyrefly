/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Utilities to deal with LSP details.

use lsp_server::Notification;
use lsp_server::Request;
use lsp_server::RequestId;
use lsp_server::Response;
use lsp_server::ResponseError;
use lsp_types::TextDocumentContentChangeEvent;
use ruff_source_file::LineIndex;
use ruff_source_file::OneIndexed;
use ruff_source_file::SourceLocation;
use serde::de::DeserializeOwned;

pub fn as_notification<T>(x: &Notification) -> Option<Result<T::Params, serde_json::Error>>
where
    T: lsp_types::notification::Notification,
    T::Params: DeserializeOwned,
{
    if x.method == T::METHOD {
        match serde_json::from_value(x.params.clone()) {
            Ok(params) => Some(Ok(params)),
            Err(err) => Some(Err(err)),
        }
    } else {
        None
    }
}

pub fn as_request<T>(x: &Request) -> Option<Result<T::Params, serde_json::Error>>
where
    T: lsp_types::request::Request,
    T::Params: DeserializeOwned,
{
    if x.method == T::METHOD {
        match serde_json::from_value(x.params.clone()) {
            Ok(params) => Some(Ok(params)),
            Err(err) => Some(Err(err)),
        }
    } else {
        None
    }
}

pub fn as_request_response_pair<T>(
    request: &Request,
    response: &Response,
) -> Option<(T::Params, T::Result)>
where
    T: lsp_types::request::Request,
    T::Params: DeserializeOwned,
{
    if response.id != request.id {
        return None;
    }
    let params = as_request::<T>(request)?.ok()?;
    let result = serde_json::from_value(response.result.clone()?).unwrap_or_else(|err| {
        panic!(
            "Invalid response\n  method: {}\n response:{:?}\n, response error:{:?}\n, error: {}\n",
            request.method, response.result, response.error, err
        )
    });
    Some((params, result))
}

/// Create a new `Notification` object with the correct name from the given params.
pub fn new_notification<T>(params: T::Params) -> Notification
where
    T: lsp_types::notification::Notification,
{
    Notification {
        method: T::METHOD.to_owned(),
        params: serde_json::to_value(&params).unwrap(),
    }
}

pub fn new_response<T>(id: RequestId, params: anyhow::Result<T>) -> Response
where
    T: serde::Serialize,
{
    match params {
        Ok(params) => Response {
            id,
            result: Some(serde_json::to_value(params).unwrap()),
            error: None,
        },
        Err(e) => Response {
            id,
            result: None,
            error: Some(ResponseError {
                code: 0,
                message: format!("{e:#?}"),
                data: None,
            }),
        },
    }
}

pub fn apply_change_events(original: &str, changes: Vec<TextDocumentContentChangeEvent>) -> String {
    /// Convert lsp_types::Position to usize index for a given text.
    fn position_to_usize(
        position: lsp_types::Position,
        index: &LineIndex,
        source_text: &str,
    ) -> usize {
        let source_location = SourceLocation {
            line: OneIndexed::from_zero_indexed(position.line as usize),
            character_offset: OneIndexed::from_zero_indexed(position.character as usize),
        };
        let text_size = index.offset(
            source_location,
            source_text,
            ruff_source_file::PositionEncoding::Utf16,
        );
        text_size.to_usize()
    }

    let mut result = original.to_owned();
    for change in changes {
        let TextDocumentContentChangeEvent { range, text, .. } = change;
        // If no range is given, we can full text replace.
        match range {
            None => result = text,
            Some(range) => {
                let index = LineIndex::from_source_text(&result);
                let start = position_to_usize(range.start, &index, &result);
                let end = position_to_usize(range.end, &index, &result);
                result.replace_range(start..end, &text);
            }
        }
    }
    result
}
