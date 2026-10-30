/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::slice;

use lsp_types::CompletionItem;
use lsp_types::CompletionItemTextEdit;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use ruff_python_ast::Expr;
use ruff_text_size::TextSize;

use crate::state::lsp::ImportFormat;
use crate::state::require::Require;
use crate::state::state::State;
use crate::test::util::TestEnv;
use crate::test::util::extract_cursors_for_test;

mod association;
mod exports;
mod review_regressions;

pub(super) fn framework_env() -> TestEnv {
    let mut env = TestEnv::new();
    for (module, path, code) in [
        ("starlette", "starlette/__init__.pyi", ""),
        (
            "starlette.applications",
            "starlette/applications.pyi",
            "class Starlette: ...",
        ),
        (
            "starlette.testclient",
            "starlette/testclient.pyi",
            r#"
from typing import Any
class TestClient:
    def __init__(self, app: object) -> None: ...
    def get(self, url: str, **kwargs: Any) -> Any: ...
    def post(self, url: str, **kwargs: Any) -> Any: ...
    def websocket_connect(self, url: str, **kwargs: Any) -> Any: ...
"#,
        ),
        (
            "fastapi.applications",
            "fastapi/applications.pyi",
            r#"
from typing import Callable
from starlette.applications import Starlette
class FastAPI(Starlette):
    def get[F](self, path: str) -> Callable[[F], F]: ...
    def post[F](self, path: str) -> Callable[[F], F]: ...
    def api_route[F](self, path: str, methods: list[str] | None = None) -> Callable[[F], F]: ...
    def websocket[F](self, path: str) -> Callable[[F], F]: ...
    def include_router(self, router: object, prefix: str = "") -> None: ...
"#,
        ),
        (
            "fastapi",
            "fastapi/__init__.pyi",
            "from fastapi.applications import FastAPI as FastAPI",
        ),
        (
            "fastapi.testclient",
            "fastapi/testclient.pyi",
            "from starlette.testclient import TestClient as TestClient",
        ),
    ] {
        env.add_with_path(module, path, code);
    }
    env
}

fn state_for(code: &str) -> (State, Handle) {
    let mut env = framework_env();
    env.add("main", code);
    let (state, handles) = env.to_state();
    (state, handles("main"))
}

pub(super) fn report_with_framework(
    code: &str,
    report: impl Fn(&State, &Handle, TextSize) -> String,
) -> String {
    let (state, handle) = state_for(code);
    extract_cursors_for_test(code)
        .into_iter()
        .map(|position| report(&state, &handle, position))
        .collect::<Vec<_>>()
        .join("\n")
}

fn position(code: &str, marker: &str, offset: usize) -> TextSize {
    TextSize::try_from(code.rfind(marker).expect("cursor marker exists") + offset).unwrap()
}

fn endpoint_items(state: &State, handle: &Handle, position: TextSize) -> Vec<CompletionItem> {
    state
        .transaction()
        .completion(handle, position, ImportFormat::Absolute, false, None)
        .into_iter()
        .filter(|item| {
            item.detail
                .as_ref()
                .is_some_and(|detail| detail.contains("endpoint"))
        })
        .collect()
}

fn apply_completion(state: &State, handle: &Handle, code: &str, item: &CompletionItem) -> String {
    let Some(CompletionItemTextEdit::TextEdit(edit)) = &item.text_edit else {
        panic!("Endpoint completion must have a replacement edit");
    };
    let module = state.transaction().get_module_info(handle).unwrap();
    let range = module.from_lsp_range(edit.range, None);
    let mut updated = code.to_owned();
    updated.replace_range(
        range.start().to_usize()..range.end().to_usize(),
        &edit.new_text,
    );
    updated
}

fn definition_names(state: &State, handle: &Handle, position: TextSize) -> Vec<String> {
    state
        .transaction()
        .goto_definition(handle, position)
        .unwrap_or_default()
        .into_iter()
        .map(|definition| definition.module.code_at(definition.range).to_owned())
        .collect()
}

const APP: &str = r#"
from fastapi import FastAPI
from fastapi.testclient import TestClient
app = FastAPI()
@app.get("/users")
def handler() -> None: pass
"#;

#[test]
fn endpoint_partial_literal_edit() {
    let code = format!("{APP}\nclient = TestClient(app)\nclient.get(\"/us\")\n");
    let (state, handle) = state_for(&code);
    let items = endpoint_items(&state, &handle, position(&code, "/us\"", 3));
    let item = items.iter().find(|item| item.label == "/users").unwrap();
    assert!(item.insert_text.is_none());
    let updated = apply_completion(&state, &handle, &code, item);
    assert!(updated.ends_with("client.get(\"/users\")\n"));
}

#[test]
fn endpoint_quoted_literal_edit() {
    let code =
        APP.replace("/users", "/say\\\"hello") + "\nclient = TestClient(app)\nclient.get(\"\")\n";
    let (state, handle) = state_for(&code);
    let items = endpoint_items(&state, &handle, position(&code, "client.get(\"", 12));
    let item = items.first().unwrap();
    let updated = apply_completion(&state, &handle, &code, item);
    let Expr::Call(call) =
        Ast::parse_expr(updated.lines().last().unwrap(), TextSize::new(0)).unwrap()
    else {
        panic!("Expected client call");
    };
    let Expr::StringLiteral(literal) = &call.arguments.args[0] else {
        panic!("Expected URL literal");
    };
    assert_eq!(literal.value.to_string(), "/say\"hello");
}

#[test]
fn endpoint_definition_argument_boundary() {
    let code = format!(
        "{APP}\nclient = TestClient(app)\nclient.get(\"/users\", timeout=None)\ncontrol = None\n"
    );
    let (state, handle) = state_for(&code);
    assert_eq!(
        definition_names(&state, &handle, position(&code, "timeout=None", 10)),
        definition_names(&state, &handle, position(&code, "control = None", 12)),
    );
}

#[test]
fn endpoint_receiver_scope_collision() {
    let code = format!(
        "{APP}\ndef setup() -> None:\n    client = TestClient(app)\ndef lookup(client: dict[str, int]) -> int | None:\n    return client.get(\"/users\")\n"
    );
    let (state, handle) = state_for(&code);
    let cursor = position(&code, "/users\"", 2);
    assert!(endpoint_items(&state, &handle, cursor).is_empty());
    assert!(definition_names(&state, &handle, cursor).is_empty());
}

#[test]
fn endpoint_receiver_reassigned_to_dictionary() {
    let code = format!("{APP}\nclient = TestClient(app)\nclient = {{}}\nclient.get(\"/users\")\n");
    let (state, handle) = state_for(&code);
    let cursor = position(&code, "/users\"", 2);
    assert!(endpoint_items(&state, &handle, cursor).is_empty());
    assert!(definition_names(&state, &handle, cursor).is_empty());
}

#[test]
fn endpoint_client_application_identity() {
    let code = r#"
from fastapi import FastAPI
from fastapi.testclient import TestClient
first = FastAPI()
second = FastAPI()
@first.get("/one")
def one() -> None: pass
@second.get("/two")
def two() -> None: pass
client = TestClient(first)
client.get("")
"#;
    let (state, handle) = state_for(code);
    let labels = endpoint_items(&state, &handle, position(code, "client.get(\"", 12))
        .into_iter()
        .map(|item| item.label)
        .collect::<Vec<_>>();
    assert_eq!(labels, vec!["/one"]);
}

#[test]
fn endpoint_definition_without_retained_route_ast() {
    let mut env = framework_env();
    env.add("app", APP);
    let client = "from app import app\nfrom fastapi.testclient import TestClient\nclient = TestClient(app)\nclient.get(\"/users\")\n";
    env.add("client", client);
    let (state, handles) = env
        .with_default_require_level(Require::Exports)
        .with_run_require(Require::Errors)
        .to_state();
    let app_handle = handles("app");
    let client_handle = handles("client");
    let mut transaction = state.new_committable_transaction(Require::Exports, None);
    transaction
        .as_mut()
        .run(slice::from_ref(&client_handle), Require::Everything, None);
    state.commit_transaction(transaction, None);
    assert!(state.transaction().get_ast(&app_handle).is_none());
    assert_eq!(
        definition_names(&state, &client_handle, position(client, "/users\"", 2)),
        vec!["handler"],
    );
}

#[test]
fn endpoint_definition_names_are_source_ranges() {
    let code = format!("{APP}\nclient = TestClient(app)\nclient.get(\"/users\")\n");
    let (state, handle) = state_for(&code);
    let definitions = state
        .transaction()
        .goto_definition(&handle, position(&code, "/users\"", 2))
        .unwrap();
    assert_eq!(definitions.len(), 1);
    let definition = &definitions[0];
    assert_eq!(definition.module.code_at(definition.range), "handler");
    assert_eq!(definition.range.len(), TextSize::new(7));
}
