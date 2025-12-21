/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::Url;
use lsp_types::notification::DidChangeTextDocument;
use lsp_types::notification::DidCloseTextDocument;
use serde_json::json;

use crate::test::lsp::lsp_interaction::object_model::InitializeSettings;
use crate::test::lsp::lsp_interaction::object_model::LspInteraction;
use crate::test::lsp::lsp_interaction::util::get_test_files_root;

#[test]
fn test_notebook_publish_diagnostics() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
            )),
            ..Default::default()
        })
        .unwrap();
    interaction.open_notebook("notebook.ipynb", vec!["z: str = ''\nz = 1"]);

    let cell_uri = interaction.cell_uri("notebook.ipynb", "cell1");
    interaction
        .client
        .expect_publish_diagnostics_uri(&cell_uri, 1)
        .unwrap();

    interaction.close_notebook("notebook.ipynb");

    interaction
        .client
        .expect_publish_diagnostics_uri(&cell_uri, 0)
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_notebook_did_open() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
            )),
            ..Default::default()
        })
        .unwrap();

    // Send did open notification
    interaction.open_notebook(
        "notebook.ipynb",
        vec!["x: int = 1", "y: str = \"foo\"", "z: str = ''\nz = 1"],
    );

    // Cell 1 doesn't have any errors
    interaction
        .diagnostic_for_cell("notebook.ipynb", "cell1")
        .expect_response(json!({"items": [], "kind": "full"}))
        .unwrap();
    // Cell 3 has an error
    interaction
        .diagnostic_for_cell("notebook.ipynb", "cell3")
        .expect_response(json!({
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
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_notebook_completion_parse_error() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
            )),
            ..Default::default()
        })
        .unwrap();

    interaction.open_notebook("notebook.ipynb", vec!["x: int = 1\nx.", "x: int = 1\nx."]);

    interaction
        .diagnostic_for_cell("notebook.ipynb", "cell1")
        .expect_response(json!({"items": [
            {
                "code": "missing-attribute",
                "codeDescription": {
                    "href": "https://pyrefly.org/en/docs/error-kinds/#missing-attribute"
                },
                "message": "Object of class `int` has no attribute ``",
                "range": {
                    "end": {"character": 2, "line": 1},
                    "start": {"character": 0, "line": 1}
                },
                "severity": 1,
                "source": "Pyrefly"
            },
            {
                "code": "parse-error",
                "codeDescription": {
                    "href": "https://pyrefly.org/en/docs/error-kinds/#parse-error"
                },
                "message": "Parse error: Expected an identifier",
                "range": {
                    "end": {"character": 0, "line": 2}, // This is actually 0:0 in the "next" cell
                    "start": {"character": 2, "line": 1}
                },
                "severity": 1,
                "source": "Pyrefly"
            }
        ], "kind": "full"}))
        .unwrap();

    interaction
        .diagnostic_for_cell("notebook.ipynb", "cell2")
        .expect_response(json!({"items": [
    {
        "code": "missing-attribute",
        "codeDescription": {
            "href": "https://pyrefly.org/en/docs/error-kinds/#missing-attribute"
        },
        "message": "Object of class `int` has no attribute ``",
        "range": {
            "end": {"character": 2, "line": 1},
            "start": {"character": 0, "line": 1}
        },
        "severity": 1,
        "source": "Pyrefly"
    },
    {
        "code": "parse-error",
        "codeDescription": {
            "href": "https://pyrefly.org/en/docs/error-kinds/#parse-error"
        },
        "message": "Parse error: Expected an identifier",
        "range": {
            "end": {"character": 0, "line": 2}, // This is actually 0:0 in the "next" cell
            "start": {"character": 2, "line": 1}
        },
        "severity": 1,
        "source": "Pyrefly"
    }
        ], "kind": "full"}))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_notebook_did_change_cell_contents() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
            )),
            ..Default::default()
        })
        .unwrap();

    interaction.open_notebook(
        "notebook.ipynb",
        vec!["x: int = 1", "y: str = \"foo\"", "z: str = ''\nz = 1"],
    );

    // Change the contents of cell 3 to fix the type error
    let cell_uri = interaction.cell_uri("notebook.ipynb", "cell3");
    interaction.change_notebook(
        "notebook.ipynb",
        2,
        json!({
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
    interaction
        .diagnostic_for_cell("notebook.ipynb", "cell3")
        .expect_response(json!({"items": [], "kind": "full"}))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_notebook_did_change_swap_cells() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
            )),
            ..Default::default()
        })
        .unwrap();

    interaction.open_notebook("notebook.ipynb", vec!["x: int = 1", "z = x"]);

    // Swap cells 1 and 2
    let cell2_uri = interaction.cell_uri("notebook.ipynb", "cell2");

    // Step 1: Delete cell 2
    interaction.change_notebook(
        "notebook.ipynb",
        2,
        json!({
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
        json!({
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
    interaction
        .diagnostic_for_cell("notebook.ipynb", "cell1")
        .expect_response(json!({"items": [], "kind": "full"}))
        .unwrap();

    // Cell 2 should have an error
    interaction
        .diagnostic_for_cell("notebook.ipynb", "cell2")
        .expect_response(json!({
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
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

#[test]
fn test_notebook_did_change_delete_cell() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
            )),
            ..Default::default()
        })
        .unwrap();

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
        json!({
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
    interaction
        .diagnostic_for_cell("notebook.ipynb", "cell1")
        .expect_response(json!({"items": [], "kind": "full"}))
        .unwrap();

    // Cell 2 does not exist, should still have no errors
    interaction
        .diagnostic_for_cell("notebook.ipynb", "cell2")
        .expect_response(json!({"items": [], "kind": "full"}))
        .unwrap();

    // Cell 3 should still have the error
    interaction
        .diagnostic_for_cell("notebook.ipynb", "cell3")
        .expect_response(json!({
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
        }))
        .unwrap();

    interaction.shutdown().unwrap();
}

/// Test that opening a file as a notebook with a non-.ipynb extension works correctly.
/// This tests support for formats like Marimo that use .py files as notebooks.
#[test]
fn test_notebook_non_ipynb_extension() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
            )),
            ..Default::default()
        })
        .unwrap();

    // Open a .py file as a notebook (simulating Marimo or similar)
    // The key test is that the notebook operations work even with .py extension
    interaction.open_notebook("marimo_notebook.py", vec!["x: int = 1", "z: str = ''\nz = 1"]);

    // Cell 1 should have no errors
    interaction
        .diagnostic_for_cell("marimo_notebook.py", "cell1")
        .expect_response(json!({"items": [], "kind": "full"}))
        .unwrap();

    // Cell 2 should have the type error
    interaction
        .diagnostic_for_cell("marimo_notebook.py", "cell2")
        .expect_response(json!({
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
        }))
        .unwrap();

    interaction.close_notebook("marimo_notebook.py");
    interaction.shutdown().unwrap();
}

/// Test that when a file is already open as a notebook, text document operations
/// (didOpen, didChange, didClose) do not affect the notebook state.
/// This is important for formats like Marimo where the same .py file could
/// potentially receive both notebook and text document events.
#[test]
fn test_notebook_text_open_conflict() {
    let root = get_test_files_root();
    let mut interaction = LspInteraction::new();
    interaction.set_root(root.path().to_path_buf());
    interaction
        .initialize(InitializeSettings {
            configuration: Some(Some(
                json!([{"pyrefly": {"displayTypeErrors": "force-on"}}]),
            )),
            ..Default::default()
        })
        .unwrap();

    // First, open as a notebook
    interaction.open_notebook("conflict_test.py", vec!["x: int = 1", "y: str = ''\ny = 1"]);

    // Verify notebook diagnostics work
    interaction
        .diagnostic_for_cell("conflict_test.py", "cell2")
        .expect_response(json!({
            "items": [{
                "code": "bad-assignment",
                "codeDescription": {
                    "href": "https://pyrefly.org/en/docs/error-kinds/#bad-assignment"
                },
                "message": "`Literal[1]` is not assignable to variable `y` with type `str`",
                "range": {
                    "start": {"line": 1, "character": 4},
                    "end": {"line": 1, "character": 5}
                },
                "severity": 1,
                "source": "Pyrefly"
            }],
            "kind": "full"
        }))
        .unwrap();

    // Now try to open the same file as a text document (should be ignored)
    let notebook_path = root.path().join("conflict_test.py");
    let notebook_uri = Url::from_file_path(&notebook_path).unwrap();
    interaction
        .client
        .did_open_uri(&notebook_uri, "python", "# This should be ignored\nx: str = 1");

    // Try to send a textDocument/didChange for the notebook file (should be ignored)
    interaction.client.send_notification::<DidChangeTextDocument>(json!({
        "textDocument": {
            "uri": notebook_uri.to_string(),
            "version": 2
        },
        "contentChanges": [{
            "text": "# Changed content that should be ignored"
        }]
    }));

    // The notebook should still work - cell diagnostics should still be available
    // If the text operations had affected the notebook, this would fail
    interaction
        .diagnostic_for_cell("conflict_test.py", "cell1")
        .expect_response(json!({"items": [], "kind": "full"}))
        .unwrap();

    interaction
        .diagnostic_for_cell("conflict_test.py", "cell2")
        .expect_response(json!({
            "items": [{
                "code": "bad-assignment",
                "codeDescription": {
                    "href": "https://pyrefly.org/en/docs/error-kinds/#bad-assignment"
                },
                "message": "`Literal[1]` is not assignable to variable `y` with type `str`",
                "range": {
                    "start": {"line": 1, "character": 4},
                    "end": {"line": 1, "character": 5}
                },
                "severity": 1,
                "source": "Pyrefly"
            }],
            "kind": "full"
        }))
        .unwrap();

    // Try to send a textDocument/didClose for the notebook file (should be ignored)
    // The notebook should remain open and functional
    interaction.client.send_notification::<DidCloseTextDocument>(json!({
        "textDocument": {
            "uri": notebook_uri.to_string()
        }
    }));

    // Verify the notebook is still functional after the ignored didClose
    interaction
        .diagnostic_for_cell("conflict_test.py", "cell1")
        .expect_response(json!({"items": [], "kind": "full"}))
        .unwrap();

    // Now properly close the notebook
    interaction.close_notebook("conflict_test.py");
    interaction.shutdown().unwrap();
}
