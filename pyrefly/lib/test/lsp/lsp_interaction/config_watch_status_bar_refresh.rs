/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Coverage for refreshing the V2 status-bar payload when a `pyrefly.toml` is
//! added to or removed from an already-open workspace.

use std::fs;
use std::time::Duration;

use lsp_types::Url;
use lsp_types::notification::Notification as _;
use pyrefly_lsp_test::Message;
use pyrefly_lsp_test::object_model::InitializeSettings;
use pyrefly_lsp_test::object_model::LspInteraction;
use pyrefly_lsp_test::object_model::LspInteractionArgs;
use pyrefly_lsp_test::object_model::TestTelemetry;
use pyrefly_util::telemetry::TelemetryEventKind;
use pyrefly_util::telemetry::TelemetryInvalidateFindReason;
use serde_json::json;
use tempfile::TempDir;

use crate::lsp::non_wasm::type_error_display_status::TypeErrorDisplayStatusChangedNotification;
use crate::lsp::non_wasm::type_error_display_status::TypeErrorDisplayStatusRequest;

/// Request the current status for `uri` and assert its `label` field.
fn expect_label(interaction: &LspInteraction, uri: &Url, expected: Option<&str>) {
    let id = interaction
        .client
        .send_request::<TypeErrorDisplayStatusRequest>(json!({ "uri": uri }))
        .id()
        .clone();
    interaction
        .client
        .expect_message("typeErrorDisplayStatus response", |msg| match msg {
            Message::Response(r) if r.id == id => {
                let result = r.result.unwrap();
                assert_eq!(
                    result.get("label"),
                    Some(&json!(expected)),
                    "unexpected label, got: {result}"
                );
                Some(Ok(()))
            }
            _ => None,
        })
        .unwrap();
}

/// Adding a `pyrefly.toml` to an open workspace should push a status refresh so the
/// client drops the "Basic" (no-config) label without needing an unrelated trigger.
#[test]
fn test_adding_pyrefly_toml_pushes_status_refresh_notification() {
    let root = TempDir::new().unwrap();
    fs::write(root.path().join("main.py"), "x: int = 1\n").unwrap();

    let telemetry = TestTelemetry::new();
    let telemetry_events = telemetry.subscribe();
    let mut interaction = LspInteraction::new_with_args(LspInteractionArgs {
        telemetry: Box::new(telemetry),
        ..Default::default()
    });
    interaction.set_root(root.path().to_path_buf());
    let scope_uri = Url::from_file_path(root.path()).unwrap();
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            file_watch: true,
            initialization_options: Some(json!({
                "pyrefly": {
                    "typeErrorDisplayStatusVersion": "v2",
                    "pushTypeErrorDisplayStatus": true,
                }
            })),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("main.py");
    let uri = Url::from_file_path(root.path().join("main.py")).unwrap();

    // No `pyrefly.toml` exists yet, so the workspace is running on the synthesized
    // "no nearby config" preset.
    expect_label(&interaction, &uri, Some("Basic"));

    fs::write(root.path().join("pyrefly.toml"), "").unwrap();
    interaction.client.file_created("pyrefly.toml");

    // Wait for the watcher-driven invalidation to finish, so the follow-up checks
    // below can't race ahead of it.
    loop {
        let event = telemetry_events
            .recv_timeout(Duration::from_secs(30))
            .unwrap();
        if matches!(event.event.kind, TelemetryEventKind::InvalidateFind)
            && matches!(
                event.event.invalidate_find_reason,
                Some(TelemetryInvalidateFindReason::WatcherEvents)
            )
        {
            break;
        }
    }

    interaction
        .client
        .expect_message(
            "typeErrorDisplayStatusChanged notification after pyrefly.toml appeared",
            |msg| match msg {
                Message::Notification(n)
                    if n.method == TypeErrorDisplayStatusChangedNotification::METHOD =>
                {
                    Some(Ok(()))
                }
                _ => None,
            },
        )
        .unwrap();

    // The config was invalidated, so a fresh request reflects the new pyrefly.toml.
    expect_label(&interaction, &uri, None);

    interaction.shutdown().unwrap();
}

/// Removing a `pyrefly.toml` from an open workspace should push a status refresh so
/// the client picks up the "Basic" (no-config) label without an unrelated trigger.
#[test]
fn test_removing_pyrefly_toml_pushes_status_refresh_notification() {
    let root = TempDir::new().unwrap();
    fs::write(root.path().join("main.py"), "x: int = 1\n").unwrap();
    let config_path = root.path().join("pyrefly.toml");
    fs::write(&config_path, "").unwrap();

    let telemetry = TestTelemetry::new();
    let telemetry_events = telemetry.subscribe();
    let mut interaction = LspInteraction::new_with_args(LspInteractionArgs {
        telemetry: Box::new(telemetry),
        ..Default::default()
    });
    interaction.set_root(root.path().to_path_buf());
    let scope_uri = Url::from_file_path(root.path()).unwrap();
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            file_watch: true,
            initialization_options: Some(json!({
                "pyrefly": {
                    "typeErrorDisplayStatusVersion": "v2",
                    "pushTypeErrorDisplayStatus": true,
                }
            })),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("main.py");
    let uri = Url::from_file_path(root.path().join("main.py")).unwrap();

    // A `pyrefly.toml` already exists, so the status is silent (no onboarding nudge).
    expect_label(&interaction, &uri, None);

    fs::remove_file(&config_path).unwrap();
    interaction.client.file_deleted("pyrefly.toml");

    loop {
        let event = telemetry_events
            .recv_timeout(Duration::from_secs(30))
            .unwrap();
        if matches!(event.event.kind, TelemetryEventKind::InvalidateFind)
            && matches!(
                event.event.invalidate_find_reason,
                Some(TelemetryInvalidateFindReason::WatcherEvents)
            )
        {
            break;
        }
    }

    interaction
        .client
        .expect_message(
            "typeErrorDisplayStatusChanged notification after pyrefly.toml was removed",
            |msg| match msg {
                Message::Notification(n)
                    if n.method == TypeErrorDisplayStatusChangedNotification::METHOD =>
                {
                    Some(Ok(()))
                }
                _ => None,
            },
        )
        .unwrap();

    // The config was invalidated, so a fresh request reflects the missing pyrefly.toml.
    expect_label(&interaction, &uri, Some("Basic"));

    interaction.shutdown().unwrap();
}
