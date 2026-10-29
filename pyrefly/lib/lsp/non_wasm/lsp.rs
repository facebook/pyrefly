/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Utilities to deal with LSP details.

use lsp_server::RequestId;
use lsp_server::ResponseError;
use serde::de::DeserializeOwned;

use crate::lsp::non_wasm::protocol::Notification;
use crate::lsp::non_wasm::protocol::Request;
use crate::lsp::non_wasm::protocol::Response;

pub fn as_notification<T>(x: &Notification) -> Option<Result<T::Params, serde_json::Error>>
where
    T: lsp_types::Notification,
    T::Params: DeserializeOwned,
{
    if x.method == T::METHOD.as_str() {
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
    T: lsp_types::Request,
    T::Params: DeserializeOwned,
{
    if x.method == T::METHOD.as_str() {
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
) -> Option<(T::Params, Result<T::Result, serde_json::Error>)>
where
    T: lsp_types::Request,
    T::Params: DeserializeOwned,
    T::Result: DeserializeOwned,
{
    if response.id != request.id {
        return None;
    }
    let params = as_request::<T>(request)?.ok()?;
    let result = response.result.clone().map(serde_json::from_value)?;
    Some((params, result))
}

/// Create a new `Notification` object with the correct name from the given params.
pub fn new_notification<T>(params: T::Params) -> Notification
where
    T: lsp_types::Notification,
{
    Notification {
        method: T::METHOD.as_str().to_owned(),
        params: serde_json::to_value(&params).unwrap(),
        activity_key: None,
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
