/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// To support IDE telemetry, Meta internally passes along an "activityKey" field with LSP requests and
// notifications. The 3rd party "lsp-server" crate doesn't know about this field, so we need to define
// out own message types and LSP protocol implementation.

use std::io;

use lsp_server::RequestId;
use lsp_server::ResponseError;
use pyrefly_util::telemetry::ActivityKey;
use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, Clone)]
pub enum Message {
    Request(Request),
    Response(Response),
    Notification(Notification),
}

#[derive(Debug, Clone)]
pub struct Request {
    pub id: RequestId,
    pub method: String,
    pub params: serde_json::Value,
    pub activity_key: Option<ActivityKey>,
}

#[derive(Debug, Clone)]
pub struct Response {
    pub id: RequestId,
    pub result: Option<serde_json::Value>,
    pub error: Option<ResponseError>,
}

#[derive(Debug, Clone)]
pub struct Notification {
    pub method: String,
    pub params: serde_json::Value,
    pub activity_key: Option<ActivityKey>,
}

impl From<Request> for Message {
    fn from(request: Request) -> Message {
        Message::Request(request)
    }
}

impl From<Response> for Message {
    fn from(response: Response) -> Message {
        Message::Response(response)
    }
}

impl From<Notification> for Message {
    fn from(notification: Notification) -> Message {
        Message::Notification(notification)
    }
}

impl Response {
    pub fn new_ok<R: Serialize>(id: RequestId, result: R) -> Response {
        Response {
            id,
            result: Some(serde_json::to_value(result).unwrap()),
            error: None,
        }
    }
    pub fn new_err(id: RequestId, code: i32, message: String) -> Response {
        let error = ResponseError {
            code,
            message,
            data: None,
        };
        Response {
            id,
            result: None,
            error: Some(error),
        }
    }
}

const JSONRPC_2_0: &str = "2.0";
const CONTENT_LENGTH: &[u8] = b"Content-Length: ";

fn unexpected_eof(error: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> io::Error {
    io::Error::new(io::ErrorKind::UnexpectedEof, error)
}

fn invalid_data(error: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error)
}

fn default_jsonrpc() -> &'static str {
    JSONRPC_2_0
}

#[derive(Serialize, Deserialize)]
pub struct JsonRpcMessage<'a> {
    #[serde(default = "default_jsonrpc")]
    jsonrpc: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<RequestId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    method: Option<String>,
    #[serde(default = "serde_json::Value::default")]
    #[serde(skip_serializing_if = "serde_json::Value::is_null")]
    params: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<ResponseError>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "activityKey")]
    activity_key: Option<ActivityKey>,
}

impl<'a> JsonRpcMessage<'a> {
    fn to_message(self) -> Option<Message> {
        let Self {
            jsonrpc: _,
            id,
            method,
            params,
            result,
            error,
            activity_key,
        } = self;
        match (id, method) {
            (Some(id), Some(method)) => Some(Message::Request(Request {
                id,
                method,
                params,
                activity_key,
            })),
            (Some(id), None) => Some(Message::Response(Response { id, result, error })),
            (None, Some(method)) => Some(Message::Notification(Notification {
                method,
                params,
                activity_key,
            })),
            (None, None) => None,
        }
    }

    pub fn from_message(msg: Message) -> Self {
        match msg {
            Message::Request(Request {
                id,
                method,
                params,
                activity_key,
            }) => JsonRpcMessage {
                jsonrpc: JSONRPC_2_0,
                id: Some(id),
                method: Some(method),
                params,
                result: None,
                error: None,
                activity_key,
            },
            Message::Response(Response { id, result, error }) => JsonRpcMessage {
                jsonrpc: JSONRPC_2_0,
                id: Some(id),
                method: None,
                params: serde_json::Value::Null,
                result,
                error,
                activity_key: None,
            },
            Message::Notification(Notification {
                method,
                params,
                activity_key,
            }) => JsonRpcMessage {
                jsonrpc: JSONRPC_2_0,
                id: None,
                method: Some(method),
                params,
                result: None,
                error: None,
                activity_key,
            },
        }
    }
}

/// Read a single LSP message, following the LSP protocol described in [1].
///
/// We first read headers, separated by \r\n. We require a Content-Length header. Other headers are currently
/// ignored. Returns the message or None if the input reaches EOF at a message boundary. Returns an error if
/// the input reaches EOF in the middle of a message, or if the input data is not a valid LSP message.
///
/// [1]: https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#baseProtocol
pub fn read_lsp_message(mut r: impl io::BufRead) -> io::Result<Option<Message>> {
    let mut buf: Vec<u8> = Vec::new();
    let mut size = None;

    loop {
        buf.clear();
        // Note: read_until internally retries on EINTR.
        let len = r.read_until(b'\n', &mut buf)?;

        if len == 0 {
            return Ok(None); // End of stream
        }

        if buf[len - 1] != b'\n' {
            return Err(unexpected_eof("failed to read header"));
        }

        if len == 1 || buf[len - 2] != b'\r' {
            return Err(invalid_data("header must end with \\r\\n"));
        }

        if len == 2 {
            break; // end of headers
        }

        if len - 2 >= CONTENT_LENGTH.len()
            && buf[..CONTENT_LENGTH.len()].eq_ignore_ascii_case(CONTENT_LENGTH)
        {
            let value = str::from_utf8(&buf[CONTENT_LENGTH.len()..len - 2])
                .map_err(|e| invalid_data(format!("invalid content-length: {e}")))?;

            size = Some(
                str::parse::<usize>(value)
                    .map_err(|e| invalid_data(format!("invalid content-length: {e}")))?,
            );
        }
    }

    let Some(size) = size else {
        return Err(invalid_data("missing required content-length header"));
    };

    // Read the entire body at once.
    // Note: read_exact internally retries on EINTR.
    // Note: read_exact returns UnexpectedEof if the end-of-file is reached before all bytes are read.
    buf.resize(size, 0);
    r.read_exact(&mut buf)?;

    let body = str::from_utf8(&buf).map_err(|e| invalid_data(format!("invalid body: {e}")))?;

    let msg_raw = serde_json::from_str::<JsonRpcMessage>(body)
        .map_err(|e| invalid_data(format!("invalid body: {e}")))?;

    let Some(msg) = msg_raw.to_message() else {
        return Err(invalid_data("invalid body: missing id or method"));
    };

    Ok(Some(msg))
}

pub fn write_lsp_message(mut w: impl io::Write, msg: Message) -> io::Result<()> {
    let str = serde_json::to_string(&JsonRpcMessage::from_message(msg))?;
    write!(w, "Content-Length: {}\r\n\r\n", str.len())?;
    w.write_all(str.as_bytes())?;
    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_expected_eof_empty() {
        assert!(read_lsp_message(io::empty()).unwrap().is_none());
    }

    #[test]
    fn test_unexpected_eof_partial_header() {
        let msg: &[u8] = b"abc";
        let e = read_lsp_message(msg).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn test_invalid_header_empty_newline() {
        let msg: &[u8] = b"\n"; // should be \r\n
        let e = read_lsp_message(msg).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_invalid_header_eol() {
        let msg: &[u8] = b"abc\n"; // should be \r\n
        let e = read_lsp_message(msg).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_invalid_missing_content_length() {
        let msg: &[u8] = b"Content-Type: application/vscode-jsonrpc; charset=utf-8\r\n\r\n";
        let e = read_lsp_message(msg).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_content_length_invalid_utf8() {
        let msg: &[u8] = b"Content-Length: \x80\r\n";
        let e = read_lsp_message(msg).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_content_length_not_a_number() {
        let msg: &[u8] = b"Content-Length: foo\r\n";
        let e = read_lsp_message(msg).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_unexpected_eof_partial_body() {
        let msg: &[u8] = b"Content-Length: 10\r\n\r\nabc";
        let e = read_lsp_message(msg).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn test_body_invalid_utf8() {
        let msg: &[u8] = b"Content-Length: 1\r\n\r\n\x80";
        let e = read_lsp_message(msg).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_invalid_incomplete_message() {
        let body = serde_json::to_string(&json!({
            "jsonrpc": JSONRPC_2_0,
        }))
        .unwrap();
        let msg = format!("Content-Length: {}\r\n\r\n{}", body.len(), body);
        let e = read_lsp_message(msg.as_bytes()).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_valid_request() {
        let body = serde_json::to_string(&json!({
            "jsonrpc": JSONRPC_2_0,
            "id": 1,
            "method": "foo",
            "params": { "bar": "baz" },
        }))
        .unwrap();
        let msg = format!("Content-Length: {}\r\n\r\n{}", body.len(), body);
        let msg = read_lsp_message(msg.as_bytes()).unwrap().unwrap();
        let Message::Request(Request {
            id,
            method,
            params,
            activity_key,
        }) = msg
        else {
            panic!("unexpected message")
        };

        assert_eq!(id, RequestId::from(1));
        assert_eq!(method, "foo");
        assert_eq!(params, json!({ "bar": "baz" }));
        assert_eq!(activity_key, None);
    }

    #[test]
    fn test_valid_response() {
        let body = serde_json::to_string(&json!({
            "jsonrpc": JSONRPC_2_0,
            "id": 1,
            "result": { "foo": "bar" },
        }))
        .unwrap();
        let msg = format!("Content-Length: {}\r\n\r\n{}", body.len(), body);
        let msg = read_lsp_message(msg.as_bytes()).unwrap().unwrap();
        let Message::Response(Response { id, result, error }) = msg else {
            panic!("unexpected message")
        };

        assert_eq!(id, RequestId::from(1));
        assert_eq!(result, Some(json!({ "foo": "bar" })));
        assert!(error.is_none());
    }

    #[test]
    fn test_valid_notification() {
        let body = serde_json::to_string(&json!({
            "jsonrpc": JSONRPC_2_0,
            "method": "foo",
            "params": { "bar": "baz" },
        }))
        .unwrap();
        let msg = format!("Content-Length: {}\r\n\r\n{}", body.len(), body);
        let msg = read_lsp_message(msg.as_bytes()).unwrap().unwrap();
        let Message::Notification(Notification {
            method,
            params,
            activity_key,
        }) = msg
        else {
            panic!("unexpected message")
        };

        assert_eq!(method, "foo");
        assert_eq!(params, json!({ "bar": "baz" }));
        assert_eq!(activity_key, None);
    }

    #[test]
    fn test_accept_missing_jsonrpc_field() {
        let body = serde_json::to_string(&json!({
            "id": 1,
            "method": "foo",
            "params": { "bar": "baz" },
        }))
        .unwrap();
        let msg = format!("Content-Length: {}\r\n\r\n{}", body.len(), body);
        let msg = read_lsp_message(msg.as_bytes()).unwrap().unwrap();
        let Message::Request(Request {
            id,
            method,
            params,
            activity_key,
        }) = msg
        else {
            panic!("unexpected message")
        };

        assert_eq!(id, RequestId::from(1));
        assert_eq!(method, "foo");
        assert_eq!(params, json!({ "bar": "baz" }));
        assert_eq!(activity_key, None);
    }

    #[test]
    fn test_activity_key() {
        let body = serde_json::to_string(&json!({
            "jsonrpc": JSONRPC_2_0,
            "id": 1,
            "method": "foo",
            "activityKey": { "id": "bar", "name": "baz" },
        }))
        .unwrap();
        let msg = format!("Content-Length: {}\r\n\r\n{}", body.len(), body);
        let msg = read_lsp_message(msg.as_bytes()).unwrap().unwrap();
        let Message::Request(x) = msg else {
            panic!("unexpected message")
        };
        let activity_key = x.activity_key.unwrap();
        assert_eq!(activity_key.id, "bar");
        assert_eq!(activity_key.name, "baz");
    }

    #[test]
    fn test_roundtrip_request() {
        let mut msg: Vec<u8> = Vec::new();
        write_lsp_message(
            &mut msg,
            Message::Request(Request {
                id: RequestId::from(1),
                method: "foo".to_owned(),
                params: json!({ "bar": "baz" }),
                activity_key: None,
            }),
        )
        .unwrap();

        let msg = read_lsp_message(msg.as_slice()).unwrap().unwrap();
        let Message::Request(Request {
            id,
            method,
            params,
            activity_key,
        }) = msg
        else {
            panic!("unexpected message")
        };

        assert_eq!(id, RequestId::from(1));
        assert_eq!(method, "foo");
        assert_eq!(params, json!({ "bar": "baz" }));
        assert_eq!(activity_key, None);
    }

    #[test]
    fn test_roundtrip_response() {
        let mut msg: Vec<u8> = Vec::new();
        write_lsp_message(
            &mut msg,
            Message::Response(Response {
                id: RequestId::from(1),
                result: Some(json!({"foo": "bar"})),
                error: None,
            }),
        )
        .unwrap();

        let msg = read_lsp_message(msg.as_slice()).unwrap().unwrap();
        let Message::Response(Response { id, result, error }) = msg else {
            panic!("unexpected message")
        };

        assert_eq!(id, RequestId::from(1));
        assert_eq!(result, Some(json!({"foo": "bar"})));
        assert!(error.is_none());
    }

    #[test]
    fn test_roundtrip_notification() {
        let mut msg: Vec<u8> = Vec::new();
        write_lsp_message(
            &mut msg,
            Message::Notification(Notification {
                method: "foo".to_owned(),
                params: json!({ "bar": "baz" }),
                activity_key: None,
            }),
        )
        .unwrap();

        let msg = read_lsp_message(msg.as_slice()).unwrap().unwrap();
        let Message::Notification(Notification {
            method,
            params,
            activity_key,
        }) = msg
        else {
            panic!("unexpected message")
        };

        assert_eq!(method, "foo");
        assert_eq!(params, json!({ "bar": "baz" }));
        assert_eq!(activity_key, None);
    }

    #[test]
    fn test_serialize_params() {
        let msg = serde_json::to_value(&JsonRpcMessage {
            jsonrpc: JSONRPC_2_0,
            id: Some(RequestId::from(1)),
            method: Some("foo".to_owned()),
            params: json!({"bar": "baz"}),
            result: None,
            error: None,
            activity_key: None,
        })
        .unwrap();
        assert_eq!(
            msg,
            json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "foo",
                "params": {"bar": "baz"},
            })
        );
    }

    #[test]
    fn test_serialize_request_no_params() {
        let msg = serde_json::to_value(&JsonRpcMessage {
            jsonrpc: JSONRPC_2_0,
            id: Some(RequestId::from(1)),
            method: Some("foo".to_owned()),
            params: serde_json::Value::Null,
            result: None,
            error: None,
            activity_key: None,
        })
        .unwrap();
        assert_eq!(
            msg,
            json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "foo",
            })
        );
    }

    #[test]
    fn test_serialize_response() {
        let msg = serde_json::to_value(&JsonRpcMessage {
            jsonrpc: JSONRPC_2_0,
            id: Some(RequestId::from(1)),
            method: None,
            params: serde_json::Value::Null,
            result: Some(json!({"foo": "bar"})),
            error: None,
            activity_key: None,
        })
        .unwrap();
        assert_eq!(
            msg,
            json!({
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"foo": "bar"},
            })
        );
    }

    #[test]
    fn test_serialize_notification() {
        let msg = serde_json::to_value(&JsonRpcMessage {
            jsonrpc: JSONRPC_2_0,
            id: None,
            method: Some("foo".to_owned()),
            params: json!({"bar": "baz"}),
            result: None,
            error: None,
            activity_key: None,
        })
        .unwrap();
        assert_eq!(
            msg,
            json!({
                "jsonrpc": "2.0",
                "method": "foo",
                "params": {"bar": "baz"},
            })
        );
    }

    #[test]
    fn test_serialize_notification_no_params() {
        let msg = serde_json::to_value(&JsonRpcMessage {
            jsonrpc: JSONRPC_2_0,
            id: None,
            method: Some("foo".to_owned()),
            params: serde_json::Value::Null,
            result: None,
            error: None,
            activity_key: None,
        })
        .unwrap();
        assert_eq!(
            msg,
            json!({
                "jsonrpc": "2.0",
                "method": "foo",
            })
        );
    }
}
