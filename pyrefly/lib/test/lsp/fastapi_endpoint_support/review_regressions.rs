/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use super::*;

#[test]
fn endpoint_conditional_client_ownership() {
    let code = format!(
        r#"{APP}
other = FastAPI()
@other.get("/other")
def other_handler() -> None: pass
def probe(flag: bool) -> None:
    if flag:
        client = TestClient(app)
    else:
        client = TestClient(other)
    client.get("")
"#
    );
    let (state, handle) = state_for(&code);
    assert!(endpoint_items(&state, &handle, position(&code, "client.get(\"", 12)).is_empty());
}

#[test]
fn endpoint_loop_client_ownership() {
    let code = format!(
        r#"{APP}
other = FastAPI()
def probe(flags: list[bool]) -> None:
    client = TestClient(app)
    for flag in flags:
        client = TestClient(other)
    client.get("")
"#
    );
    let (state, handle) = state_for(&code);
    assert!(endpoint_items(&state, &handle, position(&code, "client.get(\"", 12)).is_empty());
}

#[test]
fn endpoint_unknown_methods_are_not_unrestricted() {
    let code = APP.replace(
        "@app.get(\"/users\")",
        "verbs = [\"POST\"]\n@app.api_route(\"/users\", methods=verbs)",
    ) + "\nclient = TestClient(app)\nclient.get(\"/users\")\n";
    let (state, handle) = state_for(&code);
    let cursor = position(&code, "/users\"", 2);
    assert!(definition_names(&state, &handle, cursor).is_empty());
}

#[test]
fn endpoint_angle_brackets_are_literal_path_text() {
    let code = APP.replace("/users", "/items/<name>")
        + "\nclient = TestClient(app)\nclient.get(\"/items/alice\")\n";
    let (state, handle) = state_for(&code);
    assert!(definition_names(&state, &handle, position(&code, "/items/alice", 3)).is_empty());
}

#[test]
fn endpoint_editor_filter_text_includes_literal_opener() {
    for (call, typed) in [
        ("client.get(\"/us\")\n", "\"/us"),
        ("client.get('/us')\n", "'/us"),
        ("client.get(r\"/us\")\n", "r\"/us"),
    ] {
        let code = format!("{APP}\nclient = TestClient(app)\n{call}");
        let (state, handle) = state_for(&code);
        let items = endpoint_items(&state, &handle, position(&code, "/us", 3));
        let item = items.iter().find(|item| item.label == "/users").unwrap();
        assert!(item.filter_text.as_ref().unwrap().starts_with(typed));
    }
}

#[test]
fn endpoint_multiline_literal_edit_is_not_single_line() {
    let code = format!("{APP}\nclient = TestClient(app)\nclient.get(\"\"\"/us\n\"\"\")\n");
    let (state, handle) = state_for(&code);
    assert!(endpoint_items(&state, &handle, position(&code, "/us", 3)).is_empty());
}

#[test]
fn endpoint_internal_empty_path_segments_are_significant() {
    let code =
        APP.replace("/users", "/a//b") + "\nclient = TestClient(app)\nclient.get(\"/a/b\")\n";
    let (state, handle) = state_for(&code);
    assert!(definition_names(&state, &handle, position(&code, "/a/b", 2)).is_empty());
}
