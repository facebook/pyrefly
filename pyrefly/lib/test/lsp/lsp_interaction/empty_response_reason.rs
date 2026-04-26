/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests that [`EmptyResponseReason`] telemetry is set correctly on LSP
//! handlers that return empty/null responses.

use std::sync::Arc;
use std::time::Duration;

use lsp_types::Url;
use pyrefly::commands::lsp::IndexingMode;
use pyrefly::commands::lsp::LspArgs;
use pyrefly_util::telemetry::EmptyResponseReason;
use pyrefly_util::telemetry::TelemetryEventKind;
use serde_json::json;

use crate::object_model::InitializeSettings;
use crate::object_model::LspInteraction;
use crate::object_model::RecordedTelemetryEvent;
use crate::object_model::TestTelemetry;
use crate::util::get_test_files_root;

const RECV_TIMEOUT: Duration = Duration::from_secs(30);

/// Wait for the next LspEvent matching the given method substring,
/// skipping non-LspEvent events and LspEvents for other methods.
fn wait_for_lsp_event(
    rx: &crossbeam_channel::Receiver<Arc<RecordedTelemetryEvent>>,
    method_substr: &str,
) -> Arc<RecordedTelemetryEvent> {
    loop {
        let event = rx.recv_timeout(RECV_TIMEOUT).unwrap_or_else(|_| {
            panic!("timed out waiting for LspEvent containing '{method_substr}'")
        });
        if let TelemetryEventKind::LspEvent(ref name) = event.event.kind {
            if name.contains(method_substr) {
                return event;
            }
        }
    }
}

fn default_args() -> LspArgs {
    LspArgs {
        indexing_mode: IndexingMode::None,
        workspace_indexing_limit: 50,
        build_system_blocking: false,
        enable_external_references: false,
    }
}

#[test]
fn test_successful_definition_has_no_reason() {
    let test_files_root = get_test_files_root();
    let root_path = test_files_root.path().join("basic");
    let scope_uri = Url::from_file_path(&root_path).unwrap();

    let telemetry = TestTelemetry::new();
    let rx = telemetry.subscribe();

    let mut interaction = LspInteraction::new_with_args(default_args(), telemetry, None, None);
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            configuration: Some(None),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_open("foo.py");

    // Go to definition on `Bar` (line 6, col 16) — this should succeed
    interaction
        .client
        .definition("foo.py", 6, 16)
        .expect_response(json!({
            "uri": Url::from_file_path(root_path.join("bar.py")).unwrap().to_string(),
            "range": {
                "start": {"line": 6, "character": 6},
                "end": {"line": 6, "character": 9}
            }
        }))
        .unwrap();

    let event = wait_for_lsp_event(&rx, "definition");
    assert!(
        event.event.empty_response_reason.is_none(),
        "expected no empty_response_reason for successful definition, got: {:?}",
        event.event.empty_response_reason,
    );
}

#[test]
fn test_language_services_disabled_sets_reason() {
    let test_files_root = get_test_files_root();
    let root_path = test_files_root.path().join("basic");
    let scope_uri = Url::from_file_path(&root_path).unwrap();

    let telemetry = TestTelemetry::new();
    let rx = telemetry.subscribe();

    let mut interaction = LspInteraction::new_with_args(default_args(), telemetry, None, None);
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
            configuration: Some(None),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_open("foo.py");

    // Disable language services
    interaction.client.did_change_configuration();
    interaction
        .client
        .expect_configuration_request(Some(vec![&scope_uri]))
        .expect("Failed to receive configuration request")
        .send_configuration_response(json!([{"pyrefly": {"disableLanguageServices": true}}]));

    // Go to definition — should get null response with LanguageServicesDisabled reason
    interaction
        .client
        .definition("foo.py", 6, 16)
        .expect_response(json!(null))
        .expect("Failed to receive expected response");

    let event = wait_for_lsp_event(&rx, "definition");
    assert!(
        matches!(
            event.event.empty_response_reason,
            Some(EmptyResponseReason::LanguageServicesDisabled)
        ),
        "expected LanguageServicesDisabled, got: {:?}",
        event.event.empty_response_reason,
    );
}

#[test]
fn test_method_disabled_sets_reason() {
    let test_files_root = get_test_files_root();
    let root_path = test_files_root.path().join("basic");
    let scope_uri = Url::from_file_path(&root_path).unwrap();

    let telemetry = TestTelemetry::new();
    let rx = telemetry.subscribe();

    let mut interaction = LspInteraction::new_with_args(default_args(), telemetry, None, None);
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
            configuration: Some(None),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_open("foo.py");

    // Disable only hover
    interaction.client.did_change_configuration();
    interaction
        .client
        .expect_configuration_request(Some(vec![&scope_uri]))
        .expect("Failed to receive configuration request")
        .send_configuration_response(json!([{
            "pyrefly": {
                "disabledLanguageServices": {
                    "hover": true
                }
            }
        }]));

    // Hover — should get null response with MethodDisabled reason
    interaction
        .client
        .hover("foo.py", 6, 17)
        .expect_response(json!(null))
        .expect("Failed to receive expected response");

    let event = wait_for_lsp_event(&rx, "hover");
    assert!(
        matches!(
            event.event.empty_response_reason,
            Some(EmptyResponseReason::MethodDisabled)
        ),
        "expected MethodDisabled, got: {:?}",
        event.event.empty_response_reason,
    );
}

#[test]
fn test_not_an_identifier_sets_reason() {
    let test_files_root = get_test_files_root();
    let root_path = test_files_root.path().join("basic");
    let scope_uri = Url::from_file_path(&root_path).unwrap();

    let telemetry = TestTelemetry::new();
    let rx = telemetry.subscribe();

    let mut interaction = LspInteraction::new_with_args(default_args(), telemetry, None, None);
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            configuration: Some(None),
            ..Default::default()
        })
        .expect("Failed to initialize");

    interaction.client.did_open("foo.py");

    // Go to definition on a comment line (line 0, col 2 = "# Copyright...")
    interaction
        .client
        .definition("foo.py", 0, 2)
        .expect_response(json!(null))
        .expect("Failed to receive expected response");

    let event = wait_for_lsp_event(&rx, "definition");
    assert!(
        matches!(
            event.event.empty_response_reason,
            Some(EmptyResponseReason::NotAnIdentifier { .. })
        ),
        "expected NotAnIdentifier, got: {:?}",
        event.event.empty_response_reason,
    );
}

#[test]
fn test_definition_not_found_sets_reason() {
    let test_files_root = get_test_files_root();
    let root_path = test_files_root.path().join("basic");
    let scope_uri = Url::from_file_path(&root_path).unwrap();

    let telemetry = TestTelemetry::new();
    let rx = telemetry.subscribe();

    let mut interaction = LspInteraction::new_with_args(default_args(), telemetry, None, None);
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            configuration: Some(None),
            ..Default::default()
        })
        .expect("Failed to initialize");

    // Open a file with an undefined name
    let undef_uri = Url::from_file_path(root_path.join("undef.py")).unwrap();
    interaction
        .client
        .did_open_uri(&undef_uri, "python", "undefined_name\n");

    // Go to definition on `undefined_name` — name exists but can't be resolved
    interaction
        .client
        .definition("undef.py", 0, 5)
        .expect_response(json!(null))
        .expect("Failed to receive expected response");

    let event = wait_for_lsp_event(&rx, "definition");
    assert!(
        matches!(
            event.event.empty_response_reason,
            Some(EmptyResponseReason::DefinitionNotFound { .. })
        ),
        "expected DefinitionNotFound, got: {:?}",
        event.event.empty_response_reason,
    );
    if let Some(EmptyResponseReason::DefinitionNotFound { ref name, .. }) =
        event.event.empty_response_reason
    {
        assert_eq!(name, "undefined_name");
    }
}

#[test]
fn test_definition_not_found_attribute_context() {
    let test_files_root = get_test_files_root();
    let root_path = test_files_root.path().join("basic");
    let scope_uri = Url::from_file_path(&root_path).unwrap();

    let telemetry = TestTelemetry::new();
    let rx = telemetry.subscribe();

    let mut interaction = LspInteraction::new_with_args(default_args(), telemetry, None, None);
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
            configuration: Some(None),
            ..Default::default()
        })
        .expect("Failed to initialize");

    // Open a file where an attribute access fails to resolve
    let attr_uri = Url::from_file_path(root_path.join("attr_undef.py")).unwrap();
    interaction
        .client
        .did_open_uri(&attr_uri, "python", "x: int = 1\nx.nonexistent\n");

    // Go to definition on `nonexistent` (line 1, col 5)
    interaction
        .client
        .definition("attr_undef.py", 1, 5)
        .expect_response(json!(null))
        .expect("Failed to receive expected response");

    let event = wait_for_lsp_event(&rx, "definition");
    assert!(
        matches!(
            event.event.empty_response_reason,
            Some(EmptyResponseReason::DefinitionNotFound { .. })
        ),
        "expected DefinitionNotFound, got: {:?}",
        event.event.empty_response_reason,
    );
    if let Some(EmptyResponseReason::DefinitionNotFound {
        ref name,
        ref context,
    }) = event.event.empty_response_reason
    {
        assert_eq!(name, "nonexistent");
        assert_eq!(context.as_str(), "attribute");
    }
}
