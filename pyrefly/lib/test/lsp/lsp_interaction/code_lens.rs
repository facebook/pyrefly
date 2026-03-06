/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::CodeLens;
use lsp_types::Url;
use lsp_types::request::CodeLensRequest;
use serde_json::Value;
use serde_json::json;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_code_lens_for_tests_and_main() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    let test_root = root.path().join("code_lens");
    interaction.set_root(test_root.clone());
    interaction
        .initialize(InitializeSettings::default())
        .unwrap();

    interaction.client.did_open("main_and_tests.py");

    let path = test_root.join("main_and_tests.py");
    let uri = Url::from_file_path(&path).unwrap();

    interaction
        .client
        .send_request::<CodeLensRequest>(json!({
            "textDocument": {
                "uri": uri.to_string()
            },
        }))
        .expect_response_with(|response: Option<Vec<CodeLens>>| {
            let Some(lenses) = response else {
                return false;
            };

            let mut has_run = false;
            let mut pytest_test = false;
            let mut unittest_test = false;
            let mut top_level_test = false;
            for lens in lenses {
                let Some(command) = lens.command else {
                    continue;
                };
                if command.command == "pyrefly.runMain" && command.title == "Run" {
                    has_run |= lens.range.start.line == 17;
                }
                if command.command == "pyrefly.runTest" && command.title == "Test" {
                    let args = command.arguments.unwrap_or_default();
                    let Some(Value::Object(obj)) = args.get(0) else {
                        continue;
                    };
                    let is_unittest = obj
                        .get("isUnittest")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                    let test_name = obj.get("testName").and_then(Value::as_str);
                    let class_name = obj.get("className").and_then(Value::as_str);
                    match (is_unittest, class_name, test_name) {
                        (false, Some("TestPytest"), Some("test_method")) => pytest_test = true,
                        (true, Some("MyTestCase"), Some("test_unittest")) => unittest_test = true,
                        (false, None, Some("test_top_level")) => top_level_test = true,
                        _ => {}
                    }
                }
            }

            has_run && pytest_test && unittest_test && top_level_test
        })
        .unwrap();

    interaction.shutdown().unwrap();
}
