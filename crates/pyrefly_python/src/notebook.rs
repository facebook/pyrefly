/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Jupyter notebook parsing support.
//
// This module extracts Python code from `.ipynb` files (Jupyter notebooks)
// and converts them into a single Python source file that can be analyzed
// by the type checker.

use anyhow::Context;
use anyhow::Result;
use serde::Deserialize;

/// Represents a Jupyter notebook cell
#[derive(Debug, Deserialize)]
struct NotebookCell {
    cell_type: String,
    source: NotebookSource,
}

/// Source can be either a string or an array of strings
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum NotebookSource {
    String(String),
    Array(Vec<String>),
}

/// Minimal representation of a Jupyter notebook
#[derive(Debug, Deserialize)]
struct Notebook {
    cells: Vec<NotebookCell>,
}

/// Extracts Python code from a Jupyter notebook JSON string.
///
/// This function:
/// - Parses the notebook JSON
/// - Extracts only code cells (ignoring markdown cells)
/// - Concatenates all code into a single Python source string
/// - Adds cell markers as comments for debugging
///
/// # Arguments
/// * `content` - The raw JSON content of a `.ipynb` file
///
/// # Returns
/// A single Python source string containing all code cells, or an error if parsing fails
pub fn extract_python_from_notebook(content: &str) -> Result<String> {
    let notebook: Notebook =
        serde_json::from_str(content).context("Failed to parse notebook JSON")?;

    let mut python_code = String::new();
    let mut code_cell_count = 0;

    for cell in notebook.cells.iter() {
        if cell.cell_type == "code" {
            code_cell_count += 1;
            // Add a comment marker for each cell
            python_code.push_str(&format!("# Cell {}\n", code_cell_count));

            // Extract the source code
            match &cell.source {
                NotebookSource::String(s) => {
                    python_code.push_str(s.as_str());
                }
                NotebookSource::Array(lines) => {
                    for line in lines {
                        python_code.push_str(line.as_str());
                    }
                }
            }

            // Add spacing between cells
            if !python_code.ends_with('\n') {
                python_code.push('\n');
            }
            python_code.push('\n');
        }
    }

    Ok(python_code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_python_from_notebook() {
        let notebook_json = r##"{
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["def hello():\n", "    return 'world'\n"]
                },
                {
                    "cell_type": "markdown",
                    "source": ["This is a markdown cell"]
                },
                {
                    "cell_type": "code",
                    "source": "x = 5"
                }
            ]
        }"##;

        let result = extract_python_from_notebook(notebook_json).unwrap();

        assert!(result.contains("# Cell 1"));
        assert!(result.contains("def hello():"));
        assert!(result.contains("return 'world'"));
        assert!(!result.contains("This is a markdown cell"));
        assert!(result.contains("# Cell 2"));
        assert!(result.contains("x = 5"));
    }

    #[test]
    fn test_extract_python_from_empty_notebook() {
        let notebook_json = r#"{"cells": []}"#;
        let result = extract_python_from_notebook(notebook_json).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_extract_python_with_invalid_json() {
        let invalid_json = "not valid json";
        let result = extract_python_from_notebook(invalid_json);
        assert!(result.is_err());
    }
}
