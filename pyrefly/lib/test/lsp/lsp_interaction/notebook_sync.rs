/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_server::RequestId;
use lsp_server::Response;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_notebook_publish_diagnostics() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(Some(
            serde_json::json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
        )),
        ..Default::default()
    });
    interaction.open_notebook("notebook.ipynb", vec!["z: str = ''\nz = 1"]);

    let cell_uri = interaction.cell_uri("notebook.ipynb", "cell1");
    interaction
        .client
        .expect_publish_diagnostics_exact_uri(&cell_uri, 1);

    interaction.close_notebook("notebook.ipynb");
    interaction
        .client
        .expect_publish_diagnostics_exact_uri(&cell_uri, 0);

    interaction.shutdown();
}

#[test]
fn test_notebook_did_open() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(Some(
            serde_json::json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
        )),
        ..Default::default()
    });

    // Send did open notification
    interaction.open_notebook(
        "notebook.ipynb",
        vec!["x: int = 1", "y: str = \"foo\"", "z: str = ''\nz = 1"],
    );

    // Cell 1 doesn't have any errors
    interaction.diagnostic_for_cell("notebook.ipynb", "cell1");
    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({"items": [], "kind": "full"})),
        error: None,
    });
    // Cell 3 has an error
    interaction.diagnostic_for_cell("notebook.ipynb", "cell3");
    interaction.client.expect_response(Response {
        id: RequestId::from(3),
        result: Some(serde_json::json!({
            "items": [{
                "code": "bad-assignment",
                "codeDescription": {
                    "href": "https://pyrefly.org/en/docs/error-kinds/#bad-assignment"
                },
                "message": "`Literal[1]` is not assignable to variable `z` with type `str`",
                "range": {
                    "start": {"line": 1, "character": 4},
                    "end": {"line": 1, "character": 5}
                },
                "severity": 1,
                "source": "Pyrefly"
            }],
            "kind": "full"
        })),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_notebook_did_change_cell_contents() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(Some(
            serde_json::json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
        )),
        ..Default::default()
    });

    interaction.open_notebook(
        "notebook.ipynb",
        vec!["x: int = 1", "y: str = \"foo\"", "z: str = ''\nz = 1"],
    );

    // Change the contents of cell 3 to fix the type error
    let cell_uri = interaction.cell_uri("notebook.ipynb", "cell3");
    interaction.change_notebook(
        "notebook.ipynb",
        2,
        serde_json::json!({
            "cells": {
                "textContent": [{
                    "document": {
                        "uri": cell_uri,
                        "version": 2
                    },
                    "changes": [{
                        "text": "z: str = ''\nz = 'fixed'"
                    }]
                }]
            }
        }),
    );

    // Cell 3 should now have no errors
    interaction.diagnostic_for_cell("notebook.ipynb", "cell3");
    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({"items": [], "kind": "full"})),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_notebook_did_change_swap_cells() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(Some(
            serde_json::json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
        )),
        ..Default::default()
    });

    interaction.open_notebook("notebook.ipynb", vec!["x: int = 1", "z = x"]);

    // Swap cells 1 and 2
    let cell2_uri = interaction.cell_uri("notebook.ipynb", "cell2");

    // Step 1: Delete cell 2
    interaction.change_notebook(
        "notebook.ipynb",
        2,
        serde_json::json!({
            "cells": {
                "structure": {
                    "array": {
                        "start": 1,
                        "deleteCount": 1,
                        "cells": null
                    },
                    "didClose": [{
                        "uri": cell2_uri
                    }]
                }
            }
        }),
    );

    // Step 2: Insert cell 2 at position 0
    interaction.change_notebook(
        "notebook.ipynb",
        3,
        serde_json::json!({
            "cells": {
                "structure": {
                    "array": {
                        "start": 0,
                        "deleteCount": 0,
                        "cells": [{
                            "kind": 2,
                            "document": cell2_uri
                        }]
                    },
                    "didOpen": [{
                        "uri": cell2_uri,
                        "languageId": "python",
                        "version": 1,
                        "text": "z = x"
                    }]
                }
            }
        }),
    );

    // Cell 1 should have no errors
    interaction.diagnostic_for_cell("notebook.ipynb", "cell1");
    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({"items": [], "kind": "full"})),
        error: None,
    });

    // Cell 2 should have an error
    interaction.diagnostic_for_cell("notebook.ipynb", "cell2");
    interaction.client.expect_response(Response {
        id: RequestId::from(3),
        result: Some(serde_json::json!({
            "items": [{
                "code": "unbound-name",
                "codeDescription": {
                    "href": "https://pyrefly.org/en/docs/error-kinds/#unbound-name"
                },
                "message": "`x` is uninitialized",
                "range": {
                    "start": {"line": 0, "character": 4},
                    "end": {"line": 0, "character": 5}
                },
                "severity": 1,
                "source": "Pyrefly"
            }],
            "kind": "full"
        })),
        error: None,
    });

    interaction.shutdown();
}

#[test]
fn test_notebook_did_change_delete_cell() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction.initialize(InitializeSettings {
        configuration: Some(Some(
            serde_json::json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
        )),
        ..Default::default()
    });

    // Open notebook with same contents as test_notebook_did_open
    interaction.open_notebook(
        "notebook.ipynb",
        vec!["x: int = 1", "y: str = \"foo\"", "z = y"],
    );

    // Delete cell 2 (the middle cell)
    let cell2_uri = interaction.cell_uri("notebook.ipynb", "cell2");

    interaction.change_notebook(
        "notebook.ipynb",
        2,
        serde_json::json!({
            "cells": {
                "structure": {
                    "array": {
                        "start": 1,
                        "deleteCount": 1,
                        "cells": null
                    },
                    "didClose": [{
                        "uri": cell2_uri
                    }]
                }
            }
        }),
    );

    // Cell 1 should still have no errors
    interaction.diagnostic_for_cell("notebook.ipynb", "cell1");
    interaction.client.expect_response(Response {
        id: RequestId::from(2),
        result: Some(serde_json::json!({"items": [], "kind": "full"})),
        error: None,
    });

    // Cell 2 does not exist, should still have no errors
    interaction.diagnostic_for_cell("notebook.ipynb", "cell2");
    interaction.client.expect_response(Response {
        id: RequestId::from(3),
        result: Some(serde_json::json!({"items": [], "kind": "full"})),
        error: None,
    });

    // Cell 3 should still have the error
    interaction.diagnostic_for_cell("notebook.ipynb", "cell3");
    interaction.client.expect_response(Response {
        id: RequestId::from(4),
        result: Some(serde_json::json!({
            "items": [{
                "code": "unknown-name",
                "codeDescription": {
                    "href": "https://pyrefly.org/en/docs/error-kinds/#unknown-name"
                },
                "message": "Could not find name `y`",
                "range": {
                    "start": {"line": 0, "character": 4},
                    "end": {"line": 0, "character": 5}
                },
                "severity": 1,
                "source": "Pyrefly"
            }],
            "kind": "full"
        })),
        error: None,
    });

    interaction.shutdown();
}
