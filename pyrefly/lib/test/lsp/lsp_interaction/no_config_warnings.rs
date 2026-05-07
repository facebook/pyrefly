/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::DocumentDiagnosticReport;
use lsp_types::DocumentDiagnosticReportResult;

use crate::object_model::InitializeSettings;
use crate::object_model::LspInteraction;
use crate::util::get_test_files_root;

/// With no config file, the resolver synthesizes a `Basic` preset which
/// includes `missing-import` and `unknown-name` in its explicit
/// error-severity list. The preset itself decides per-kind severities,
/// so these errors appear at `Error` severity in projects without a
/// Pyrefly config, the same as in projects with one (the difference is
/// in *which* kinds Basic surfaces, not their severity).
#[test]
fn test_no_config_missing_import_is_error() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("no_config_warnings.py");

    interaction
        .client
        .diagnostic("no_config_warnings.py")
        .expect_response_with(|response| {
            let DocumentDiagnosticReportResult::Report(report) = response else {
                return false;
            };
            let DocumentDiagnosticReport::Full(full) = report else {
                return false;
            };
            let items = &full.full_document_diagnostic_report.items;
            let has_missing_import_error = items.iter().any(|item| {
                item.code
                    == Some(lsp_types::NumberOrString::String(
                        "missing-import".to_owned(),
                    ))
                    && item.severity == Some(lsp_types::DiagnosticSeverity::ERROR)
            });
            let has_unknown_name_error = items.iter().any(|item| {
                item.code == Some(lsp_types::NumberOrString::String("unknown-name".to_owned()))
                    && item.severity == Some(lsp_types::DiagnosticSeverity::ERROR)
            });
            has_missing_import_error && has_unknown_name_error
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

/// With no config file, syntax errors should still appear as Error (severity 1).
#[test]
fn test_no_config_syntax_error_is_error() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("no_config_syntax_error.py");

    interaction
        .client
        .diagnostic("no_config_syntax_error.py")
        .expect_response_with(|response| {
            let DocumentDiagnosticReportResult::Report(report) = response else {
                return false;
            };
            let DocumentDiagnosticReport::Full(full) = report else {
                return false;
            };
            let items = &full.full_document_diagnostic_report.items;
            items.iter().any(|item| {
                item.code == Some(lsp_types::NumberOrString::String("parse-error".to_owned()))
                    && item.severity == Some(lsp_types::DiagnosticSeverity::ERROR)
            })
        })
        .unwrap();

    interaction.shutdown().unwrap();
}

/// With a config file present, the same missing import should appear as Error (severity 1),
/// not Warning, since the user has opted in to full type checking.
#[test]
fn test_with_config_missing_import_is_error() {
    let test_files_root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(test_files_root.path().join("tests_requiring_config"));
    interaction
        .initialize(InitializeSettings {
            configuration: Some(None),
            ..Default::default()
        })
        .unwrap();

    interaction.client.did_open("no_config_warnings.py");

    interaction
        .client
        .diagnostic("no_config_warnings.py")
        .expect_response_with(|response| {
            let DocumentDiagnosticReportResult::Report(report) = response else {
                return false;
            };
            let DocumentDiagnosticReport::Full(full) = report else {
                return false;
            };
            let items = &full.full_document_diagnostic_report.items;
            let has_missing_import_error = items.iter().any(|item| {
                item.code
                    == Some(lsp_types::NumberOrString::String(
                        "missing-import".to_owned(),
                    ))
                    && item.severity == Some(lsp_types::DiagnosticSeverity::ERROR)
            });
            let has_unknown_name_error = items.iter().any(|item| {
                item.code == Some(lsp_types::NumberOrString::String("unknown-name".to_owned()))
                    && item.severity == Some(lsp_types::DiagnosticSeverity::ERROR)
            });
            has_missing_import_error && has_unknown_name_error
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
