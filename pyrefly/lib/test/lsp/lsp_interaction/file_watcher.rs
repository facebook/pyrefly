/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;
use std::fs;
use std::path::MAIN_SEPARATOR;
use std::path::Path;
use std::time::Duration;

use lsp_types::RegistrationParams;
use lsp_types::Url;
use lsp_types::notification::DidChangeConfiguration;
use lsp_types::request::RegisterCapability;
use lsp_types::request::Request as _;
use lsp_types::request::UnregisterCapability;
use pyrefly_lsp_test::IndexingMode;
use pyrefly_lsp_test::LspArgs;
use pyrefly_lsp_test::Message;
use pyrefly_lsp_test::object_model::InitializeSettings;
use pyrefly_lsp_test::object_model::LspInteraction;
use pyrefly_lsp_test::object_model::LspInteractionArgs;
use pyrefly_lsp_test::object_model::LspMessageError;
use pyrefly_lsp_test::object_model::TestTelemetry;
use pyrefly_util::telemetry::TelemetryEventKind;
use pyrefly_util::telemetry::TelemetryInvalidateFindReason;
use serde::Deserialize;
use serde_json::json;
use tempfile::TempDir;

use crate::test::lsp::lsp_interaction::util::get_test_files_root;

fn path_to_lsp_glob(path: &Path) -> String {
    path.to_string_lossy().replace(MAIN_SEPARATOR, "/")
}

/// Consume the next watched-files registration request and return its ID and glob
/// patterns. Fails if the server sends an unregistration instead, so tests can assert
/// that exact watchers are never unregistered.
pub fn expect_watched_files(
    interaction: &LspInteraction,
) -> Result<(String, HashSet<String>), LspMessageError> {
    let params: RegistrationParams = interaction.client.expect_message(
        &format!("Request {}", RegisterCapability::METHOD),
        |msg| match msg {
            Message::Request(request) if request.method == RegisterCapability::METHOD => {
                Some(Ok(serde_json::from_value(request.params).unwrap()))
            }
            Message::Request(request) if request.method == UnregisterCapability::METHOD => {
                Some(Err(LspMessageError::Custom {
                    description: "unexpected watcher unregistration".to_owned(),
                }))
            }
            _ => None,
        },
    )?;
    assert_eq!(params.registrations.len(), 1);
    let registration = params
        .registrations
        .into_iter()
        .next()
        .expect("watched-files request should contain one registration");
    assert_eq!(registration.method, "workspace/didChangeWatchedFiles");
    #[derive(Deserialize)]
    struct Pattern {
        #[serde(rename = "globPattern")]
        glob_pattern: String,
    }
    #[derive(Deserialize)]
    struct Options {
        watchers: Vec<Pattern>,
    }
    let options = registration
        .register_options
        .expect("watched-files registration should include register_options");
    let options: Options = serde_json::from_value(options)
        .expect("watched-files register_options should contain a watcher list");
    let patterns = options
        .watchers
        .into_iter()
        .map(|watcher| watcher.glob_pattern)
        .collect();
    Ok((registration.id, patterns))
}

/// Initialize a test interaction with file watcher enabled.
/// Returns the TempDir (to keep it alive) and the interaction after consuming
/// the initial file watcher registration.
fn setup_file_watcher_test() -> (TempDir, LspInteraction) {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());

    let scope_uri = Url::from_file_path(root.path()).unwrap();
    interaction
        .initialize(InitializeSettings {
            workspace_folders: Some(vec![("test".to_owned(), scope_uri.clone())]),
            file_watch: true,
            ..Default::default()
        })
        .unwrap();

    (root, interaction)
}

/// Test that file watcher registration happens even when no specific patterns are watched.
/// This ensures the server always registers file watchers after initialization.
#[test]
fn test_file_watcher_registered_on_initialization() {
    let (_root, interaction) = setup_file_watcher_test();

    interaction.shutdown().unwrap();
}

/// Test that incremental pattern additions only send register (no unregister first)
/// when the change is small enough.
#[test]
fn test_incremental_pattern_addition() {
    let (_root, interaction) = setup_file_watcher_test();

    // Opening a new file with a new extension shouldn't trigger full re-watch
    // Just an incremental register for new patterns
    interaction.client.did_open("text_document.py");
    let (_, text_document_watched) = expect_watched_files(&interaction).unwrap();

    interaction
        .client
        .did_open("imports_builtins/imports_builtins.py");

    // We only watch new files, even though some similar files should be watched.
    let (_, builtins_watched) = expect_watched_files(&interaction).unwrap();
    assert!(text_document_watched.is_disjoint(&builtins_watched));

    interaction
        .client
        .did_open("imports_builtins/site-packages/typing.py");

    // Opening a file already covered by a watched pattern adds nothing, so no watcher
    // request is sent. Verify the diagnostic response arrives without one.
    let diagnostic = interaction
        .client
        .diagnostic("imports_builtins/site-packages/typing.py");
    let diagnostic_id = diagnostic.id().clone();
    interaction
        .client
        .expect_message(
            "diagnostic response without another watcher request",
            |msg| match msg {
                Message::Request(request)
                    if request.method == RegisterCapability::METHOD
                        || request.method == UnregisterCapability::METHOD =>
                {
                    Some(Err(LspMessageError::Custom {
                        description: "opening an already-covered file sent a watcher request"
                            .to_owned(),
                    }))
                }
                Message::Response(response) if response.id == diagnostic_id => Some(Ok(())),
                _ => None,
            },
        )
        .unwrap();

    interaction.shutdown().unwrap();
}

/// Verifies that an explicit config path is watched and reloaded after a file change.
#[test]
fn test_absolute_explicit_config_watches_and_reloads() {
    let root = TempDir::new().unwrap();
    let config_path = root.path().join("project.settings");
    fs::write(&config_path, "disable-type-errors-in-ide = true\n").unwrap();
    fs::write(root.path().join("source.py"), "x: int = 'bad'\n").unwrap();

    let telemetry = TestTelemetry::new();
    let telemetry_events = telemetry.subscribe();
    let mut interaction = LspInteraction::new_with_args(LspInteractionArgs {
        telemetry: Box::new(telemetry),
        ..Default::default()
    });
    interaction.set_root(root.path().to_path_buf());
    let scope_uri = Url::from_file_path(root.path()).unwrap();
    let settings = InitializeSettings {
        workspace_folders: Some(vec![("test".to_owned(), scope_uri)]),
        file_watch: true,
        initialization_options: Some(json!({"pyrefly": {"configPath": config_path}})),
        ..Default::default()
    };
    interaction
        .client
        .send_initialize(interaction.client.get_initialize_params(&settings));
    interaction.client.expect_any_message().unwrap();
    interaction.client.send_initialized();

    let (root_registration, _) = expect_watched_files(&interaction).unwrap();
    assert_eq!(root_registration, "FILEWATCHER");
    let (config_registration, watched) = expect_watched_files(&interaction).unwrap();
    assert!(config_registration.starts_with("FILEWATCHER-EXACT-"));
    assert_eq!(watched, HashSet::from([path_to_lsp_glob(&config_path)]));
    interaction.client.did_open("source.py");
    interaction
        .client
        .diagnostic("source.py")
        .expect_response(json!({"items": [], "kind": "full"}))
        .unwrap();

    fs::write(&config_path, "disable-type-errors-in-ide = false\n").unwrap();
    interaction.client.file_modified("project.settings");
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
        .diagnostic("source.py")
        .expect_response_with(|result| {
            serde_json::to_value(result).unwrap()["items"]
                .as_array()
                .is_some_and(|items| items.len() == 1)
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

/// Verifies that replacing the explicit config registers a new exact watcher, keeps the
/// previous one (exact registrations are never unregistered), and that a stale event for
/// the previous config path no longer forces a config reload.
#[test]
fn test_replaced_explicit_config_keeps_old_watcher_and_ignores_stale_event() {
    let root = TempDir::new().unwrap();
    let config_a = root.path().join("a.settings");
    let config_b = root.path().join("b.settings");
    fs::write(&config_a, "disable-type-errors-in-ide = false\n").unwrap();
    fs::write(&config_b, "disable-type-errors-in-ide = true\n").unwrap();
    fs::write(root.path().join("source.py"), "x: int = 'bad'\n").unwrap();

    let telemetry = TestTelemetry::new();
    let telemetry_events = telemetry.subscribe();
    let mut interaction = LspInteraction::new_with_args(LspInteractionArgs {
        telemetry: Box::new(telemetry),
        ..Default::default()
    });
    interaction.set_root(root.path().to_path_buf());
    let settings = InitializeSettings {
        file_watch: true,
        initialization_options: Some(json!({"pyrefly": {"configPath": config_a}})),
        ..Default::default()
    };
    interaction
        .client
        .send_initialize(interaction.client.get_initialize_params(&settings));
    interaction.client.expect_any_message().unwrap();
    interaction.client.send_initialized();

    let (root_registration, _) = expect_watched_files(&interaction).unwrap();
    assert_eq!(root_registration, "FILEWATCHER");
    let (registration_a, watched_a) = expect_watched_files(&interaction).unwrap();
    assert!(registration_a.starts_with("FILEWATCHER-EXACT-"));
    assert_eq!(watched_a, HashSet::from([path_to_lsp_glob(&config_a)]));

    interaction
        .client
        .send_notification::<DidChangeConfiguration>(json!({
            "settings": {"python": {"pyrefly": {"configPath": config_b}}}
        }));
    let (registration_b, watched_b) = expect_watched_files(&interaction).unwrap();
    assert!(registration_b.starts_with("FILEWATCHER-EXACT-"));
    assert_ne!(registration_a, registration_b);
    assert_eq!(watched_b, HashSet::from([path_to_lsp_glob(&config_b)]));
    loop {
        let event = telemetry_events
            .recv_timeout(Duration::from_secs(30))
            .unwrap();
        if matches!(event.event.kind, TelemetryEventKind::InvalidateConfig) {
            break;
        }
    }

    interaction.client.did_open("source.py");
    interaction
        .client
        .diagnostic("source.py")
        .expect_response(json!({"items": [], "kind": "full"}))
        .unwrap();

    fs::write(&config_b, "disable-type-errors-in-ide = false\n").unwrap();
    interaction.client.file_modified("a.settings");
    loop {
        let event = telemetry_events
            .recv_timeout(Duration::from_secs(30))
            .unwrap();
        if matches!(event.event.kind, TelemetryEventKind::InvalidateFind) {
            break;
        }
    }
    interaction
        .client
        .diagnostic("source.py")
        .expect_response(json!({"items": [], "kind": "full"}))
        .unwrap();

    interaction.shutdown().unwrap();
}

/// Verifies that an edit to an explicit config path re-registers the file watchers, so a
/// search path that the edit adds is watched.
#[test]
fn test_explicit_config_edit_rewatches_added_search_path() {
    let root = TempDir::new().unwrap();
    let search_path = TempDir::new().unwrap();
    let config_path = root.path().join("project.settings");
    fs::write(&config_path, "skip-interpreter-query = true\n").unwrap();
    fs::write(root.path().join("source.py"), "x: int = 1\n").unwrap();

    let telemetry = TestTelemetry::new();
    let telemetry_events = telemetry.subscribe();
    let mut interaction = LspInteraction::new_with_args(LspInteractionArgs {
        telemetry: Box::new(telemetry),
        ..Default::default()
    });
    interaction.set_root(root.path().to_path_buf());
    let settings = InitializeSettings {
        file_watch: true,
        initialization_options: Some(json!({"pyrefly": {"configPath": config_path}})),
        ..Default::default()
    };
    interaction
        .client
        .send_initialize(interaction.client.get_initialize_params(&settings));
    interaction.client.expect_any_message().unwrap();
    interaction.client.send_initialized();

    let (root_registration, _) = expect_watched_files(&interaction).unwrap();
    assert_eq!(root_registration, "FILEWATCHER");
    let (config_registration, _) = expect_watched_files(&interaction).unwrap();
    assert!(config_registration.starts_with("FILEWATCHER-EXACT-"));

    interaction.client.did_open("source.py");
    interaction
        .client
        .diagnostic("source.py")
        .expect_response(json!({"items": [], "kind": "full"}))
        .unwrap();

    // JSON string escaping matches TOML basic strings, so a Windows separator survives.
    let search_path_value = serde_json::to_string(&search_path.path().to_string_lossy()).unwrap();
    fs::write(
        &config_path,
        format!("skip-interpreter-query = true\nsearch-path = [{search_path_value}]\n"),
    )
    .unwrap();
    interaction.client.file_modified("project.settings");
    loop {
        let event = telemetry_events
            .recv_timeout(Duration::from_secs(30))
            .unwrap();
        if matches!(event.event.kind, TelemetryEventKind::InvalidateFind) {
            break;
        }
    }

    let (rewatch_registration, watched) = expect_watched_files(&interaction).unwrap();
    assert_eq!(rewatch_registration, "FILEWATCHER");
    let search_path_glob = path_to_lsp_glob(search_path.path());
    assert!(
        watched
            .iter()
            .any(|pattern| pattern.starts_with(&search_path_glob)),
        "the added search path should be watched, got {watched:?}"
    );

    interaction.shutdown().unwrap();
}

/// Test that multiple consecutive DidChangeWatchedFiles notifications are
/// eventually processed. This simulates a burst of file system events (e.g., git
/// checkout) where many files change at once. The first two notifications are
/// for files that didn't change on disk (noise).
#[test]
fn test_consecutive_file_watcher_events() {
    let root = get_test_files_root();
    let root_path = root.path().join("streaming");
    let mut interaction = LspInteraction::new_with_args(LspInteractionArgs {
        args: LspArgs {
            indexing_mode: IndexingMode::LazyBlocking,
            ..LspInteractionArgs::default().args
        },
        ..Default::default()
    });
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
            )),
            workspace_folders: Some(vec![(
                "streaming".to_owned(),
                Url::from_file_path(root_path.clone()).unwrap(),
            )]),
            file_watch: true,
            ..Default::default()
        })
        .unwrap();

    let b_path = root_path.join("b.py");
    let c_path = root_path.join("c.py");

    interaction.client.did_open("c.py");
    interaction.client.did_open("b.py");
    interaction
        .client
        .expect_file_watcher_register()
        .expect("Register file watcher for b");

    std::fs::write(&b_path, "").unwrap();

    // Send multiple DidChangeWatchedFiles notifications in rapid succession.
    // The first few are noise (those files didn't change on disk); only the
    // last notification (b.py) carries a real change.
    interaction.client.file_modified("a.py");
    interaction.client.file_modified("a.py");
    interaction.client.file_modified("a.py");
    interaction.client.file_modified("a.py");
    interaction.client.file_modified("a.py");
    interaction.client.file_modified("b.py");

    // Verify that b.py was re-read from disk and d now shows the type error.
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(c_path.clone(), 1)
        .expect("Failed to receive diagnostics after file watcher events");

    interaction.shutdown().unwrap();
}

/// A modified dependency lockfile invalidates cached missing-import results.
#[test]
fn test_uv_lock_modification_refreshes_import_resolution() {
    let root = TempDir::new().expect("create test directory");
    let site_packages = root.path().join("site-packages");
    fs::create_dir(&site_packages).expect("create site-packages");
    fs::write(
        root.path().join("pyrefly.toml"),
        "skip-interpreter-query = true\nsite-package-path = [\"site-packages\"]\n",
    )
    .expect("write config");
    fs::write(
        root.path().join("main.py"),
        "from dependency import value\nanswer: int = value\n",
    )
    .expect("write source");
    fs::write(root.path().join("uv.lock"), "revision = 1\n").expect("write lockfile");

    let root_path = root.path().to_path_buf();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root_path.clone());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
            )),
            workspace_folders: Some(vec![(
                "uv-project".to_owned(),
                Url::from_file_path(&root_path).unwrap(),
            )]),
            file_watch: true,
            ..Default::default()
        })
        .expect("initialize");

    interaction.client.did_open("main.py");
    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(root_path.join("main.py"), 1)
        .expect("missing dependency should produce a diagnostic");
    interaction
        .client
        .expect_file_watcher_register()
        .expect("register site-package watcher")
        .send_response(json!(null));

    fs::write(site_packages.join("dependency.py"), "value: int = 1\n").expect("install dependency");
    fs::write(root.path().join("uv.lock"), "revision = 2\n").expect("update lockfile");
    interaction.client.file_modified("uv.lock");

    interaction
        .client
        .expect_publish_diagnostics_eventual_error_count(root_path.join("main.py"), 0)
        .expect("modified uv.lock should refresh import resolution");

    interaction.shutdown().expect("shutdown");
}
