/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests for the TSP `initialize` response

use crate::test::tsp::tsp_interaction::object_model::InitializeSettings;
use crate::test::tsp::tsp_interaction::object_model::TspInteraction;

#[test]
fn test_tsp_initialize_includes_server_info() {
    // Unlike some LSP clients, TSP clients rely on serverInfo to identify the
    // server they're talking to, so the initialize response must include it.
    let mut tsp = TspInteraction::new();

    let params = tsp
        .server
        .get_initialize_params(&InitializeSettings::default());
    tsp.server.send_initialize(params);
    let response = tsp.client.receive_response_skip_notifications();
    let result = response
        .result
        .expect("initialize response should have a result");

    assert_eq!(
        result["serverInfo"]["name"],
        serde_json::json!("pyrefly-tsp")
    );

    tsp.server.send_initialized();
    tsp.shutdown();
}
