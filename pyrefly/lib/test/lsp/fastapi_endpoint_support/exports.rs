/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use super::*;

const APPLICATIONS: &str = r#"
from fastapi import FastAPI
from fastapi.testclient import TestClient
first = FastAPI()
second = FastAPI()
@first.get("/old")
def old() -> None: pass
@second.get("/new")
def new() -> None: pass
"#;

fn imported_labels(definitions: &str, request: &str) -> Vec<String> {
    let mut env = framework_env();
    env.add("services", definitions);
    env.add("bridge", "from services import client as client\n");
    env.add("test_clients", request);
    let (state, handles) = env
        .with_default_require_level(Require::Everything)
        .with_run_require(Require::Everything)
        .to_state();
    let handle = handles("test_clients");
    endpoint_items(&state, &handle, position(request, "client.get(\"", 12))
        .into_iter()
        .map(|item| item.label)
        .collect()
}

#[test]
fn endpoint_imported_client_reassignment() {
    let source =
        format!("{APPLICATIONS}\nclient = TestClient(first)\nclient = TestClient(second)\n");
    let labels = imported_labels(&source, "from services import client\nclient.get(\"\")\n");
    assert!(labels.is_empty());
}

#[test]
fn endpoint_reexported_client_reassignment() {
    let source =
        format!("{APPLICATIONS}\nclient = TestClient(first)\nclient = TestClient(second)\n");
    let labels = imported_labels(&source, "from bridge import client\nclient.get(\"\")\n");
    assert!(labels.is_empty());
}

#[test]
fn endpoint_imported_application_reassignment() {
    let source = format!("{APPLICATIONS}\napp = first\napp = second\n");
    let labels = imported_labels(
        &source,
        "from services import app\nfrom fastapi.testclient import TestClient\nclient = TestClient(app)\nclient.get(\"\")\n",
    );
    assert!(labels.is_empty());
}

#[test]
fn endpoint_imported_conditional_client() {
    let source = format!(
        "{APPLICATIONS}\nflag = bool(input())\nif flag:\n    client = TestClient(first)\nelse:\n    client = TestClient(second)\n"
    );
    let labels = imported_labels(&source, "from services import client\nclient.get(\"\")\n");
    assert!(labels.is_empty());
}

#[test]
fn endpoint_function_local_client_positive_control() {
    let source = format!(
        "{APP}\ndef test_users() -> None:\n    client = TestClient(app)\n    client.get(\"/users\")\n"
    );
    let (state, handle) = state_for(&source);
    let cursor = position(&source, "/users\"", 2);
    assert_eq!(definition_names(&state, &handle, cursor), vec!["handler"]);
    assert!(
        endpoint_items(&state, &handle, cursor)
            .iter()
            .any(|item| item.label == "/users")
    );
}
