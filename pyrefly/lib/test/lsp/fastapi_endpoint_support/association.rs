/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use super::*;

#[test]
fn endpoint_registry_decorator_is_not_application_route() {
    let code = format!(
        r#"{APP}
from typing import Callable
class Registry:
    def get[F](self, name: str) -> Callable[[F], F]:
        return lambda value: value
registry = Registry()
@registry.get("/not-http")
def ordinary() -> None: pass
client = TestClient(app)
client.get("")
"#
    );
    let (state, handle) = state_for(&code);
    let labels = endpoint_items(&state, &handle, position(&code, "client.get(\"", 12))
        .into_iter()
        .map(|item| item.label)
        .collect::<Vec<_>>();
    assert_eq!(labels, vec!["/users"]);
}

#[test]
fn endpoint_router_path_has_unknown_application_prefix() {
    let mut env = framework_env();
    env.add_with_path(
        "fastapi.routing",
        "fastapi/routing.pyi",
        r#"
from typing import Callable
class APIRouter:
    def __init__(self, prefix: str = "") -> None: ...
    def get[F](self, path: str) -> Callable[[F], F]: ...
"#,
    );
    let code = r#"
from fastapi import FastAPI
from fastapi.routing import APIRouter
from fastapi.testclient import TestClient
router = APIRouter(prefix="/api")
@router.get("/users")
def handler() -> None: pass
app = FastAPI()
app.include_router(router)
client = TestClient(app)
client.get("")
"#;
    env.add("main", code);
    let (state, handles) = env.to_state();
    let handle = handles("main");
    let labels = endpoint_items(&state, &handle, position(code, "client.get(\"", 12))
        .into_iter()
        .map(|item| item.label)
        .collect::<Vec<_>>();
    assert!(labels.is_empty());
}

#[test]
fn endpoint_client_and_application_aliases() {
    let code = format!(
        "{APP}\napp_alias = app\nConstructor = TestClient\noriginal = Constructor(app_alias)\nclient = original\nclient.get(\"/users\")\n"
    );
    let (state, handle) = state_for(&code);
    assert_eq!(
        definition_names(&state, &handle, position(&code, "/users\"", 2)),
        vec!["handler"]
    );
}

#[test]
fn endpoint_literal_prefix_ignores_text_after_cursor() {
    let code = format!("{APP}\nclient = TestClient(app)\nclient.get(\"/users/suffix\")\n");
    let (state, handle) = state_for(&code);
    let items = endpoint_items(&state, &handle, position(&code, "/users/suffix", 2));
    let item = items.iter().find(|item| item.label == "/users").unwrap();
    let updated = apply_completion(&state, &handle, &code, item);
    assert!(updated.ends_with("client.get(\"/users\")\n"));
}

#[test]
fn endpoint_raw_and_multiline_literal_edits() {
    for call in ["client.get(r\"/us\")\n", "client.get(\"\"\"/us\"\"\")\n"] {
        let code = format!("{APP}\nclient = TestClient(app)\n{call}");
        let (state, handle) = state_for(&code);
        let items = endpoint_items(&state, &handle, position(&code, "/us", 3));
        let item = items.iter().find(|item| item.label == "/users").unwrap();
        let updated = apply_completion(&state, &handle, &code, item);
        assert!(updated.ends_with("client.get(\"/users\")\n"));
    }
}
