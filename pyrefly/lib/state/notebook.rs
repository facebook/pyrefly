/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use lsp_types::Url;
use ruff_notebook::Cell;
use ruff_notebook::Notebook;
use starlark_map::small_map::SmallMap;

use crate::lsp::wasm::notebook::NotebookDocument;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LspNotebook {
    ruff_notebook: Arc<Notebook>,
    notebook_document: NotebookDocument,
    // Notebook cells have Urls of unspecified format
    cell_url_to_index: SmallMap<Url, usize>,
    cell_index_to_url: Vec<Url>,
}

impl LspNotebook {
    pub fn new(ruff_notebook: Notebook, notebook_document: NotebookDocument) -> Self {
        let mut cell_url_to_index = SmallMap::new();
        let mut cell_index_to_url = Vec::new();
        for (idx, cell) in notebook_document.cells.iter().enumerate() {
            cell_url_to_index.insert(cell.document.clone(), idx);
            cell_index_to_url.push(cell.document.clone());
        }
        Self {
            ruff_notebook: Arc::new(ruff_notebook),
            notebook_document,
            cell_url_to_index,
            cell_index_to_url,
        }
    }

    pub fn notebook_document(&self) -> &NotebookDocument {
        &self.notebook_document
    }

    /// Returns the code-cell index for a cell URL, i.e. the index among only
    /// code cells. This matches the indexing used by `Notebook::cell_offsets()`.
    /// Returns `None` if the URL is not found or the cell is not a code cell.
    ///
    /// Note: this uses `Cell::is_code_cell()` rather than ruff's internal
    /// `is_valid_python_code_cell()` (which also excludes cell-magic and
    /// non-Python cells) because the latter is not public. In practice the
    /// two filters agree for typical Python notebooks.
    pub fn get_code_cell_index(&self, cell_url: &Url) -> Option<usize> {
        let all_cells_idx = *self.cell_url_to_index.get(cell_url)?;
        let cells = self.ruff_notebook.cells();
        if !cells.get(all_cells_idx)?.is_code_cell() {
            return None;
        }
        let code_cell_index = cells[..all_cells_idx]
            .iter()
            .filter(|c| c.is_code_cell())
            .count();
        Some(code_cell_index)
    }

    pub fn get_code_cell_url(&self, cell_index: usize) -> Option<&Url> {
        self.cell_index_to_url.get(cell_index)
    }

    pub fn code_cell_urls(&self) -> &Vec<Url> {
        &self.cell_index_to_url
    }

    pub fn get_cell_contents(&self, cell_url: &Url) -> Option<String> {
        let idx = *self.cell_url_to_index.get(cell_url)?;
        let cell = self.ruff_notebook.cells().get(idx)?;
        if let Cell::Code(cell) = cell {
            Some(cell.source.to_string())
        } else {
            None
        }
    }

    pub fn ruff_notebook(&self) -> &Arc<Notebook> {
        &self.ruff_notebook
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use lsp_types::Url;

    use super::*;
    use crate::lsp::wasm::notebook::NotebookCell;
    use crate::lsp::wasm::notebook::NotebookCellKind;
    use crate::lsp::wasm::notebook::NotebookDocument;

    #[test]
    fn test_get_code_cell_index_returns_code_cell_index() {
        let cell0_url = Url::parse("vscode-notebook-cell://notebook/cell0").unwrap();
        let cell1_url = Url::parse("vscode-notebook-cell://notebook/cell1").unwrap();
        let cell2_url = Url::parse("vscode-notebook-cell://notebook/cell2").unwrap();

        let notebook_doc = NotebookDocument {
            uri: Url::parse("file:///notebook.ipynb").unwrap(),
            notebook_type: "jupyter-notebook".to_owned(),
            version: 1,
            metadata: None,
            cells: vec![
                NotebookCell {
                    kind: NotebookCellKind::Code,
                    document: cell0_url.clone(),
                    metadata: None,
                    execution_summary: None,
                },
                NotebookCell {
                    kind: NotebookCellKind::Markup,
                    document: cell1_url.clone(),
                    metadata: None,
                    execution_summary: None,
                },
                NotebookCell {
                    kind: NotebookCellKind::Code,
                    document: cell2_url.clone(),
                    metadata: None,
                    execution_summary: None,
                },
            ],
        };

        let mut cell_content = HashMap::new();
        cell_content.insert(cell0_url.clone(), "x = 1".to_owned());
        cell_content.insert(cell1_url.clone(), "# heading".to_owned());
        cell_content.insert(cell2_url.clone(), "y = 2".to_owned());

        let ruff_notebook = notebook_doc
            .clone()
            .to_ruff_notebook(&cell_content)
            .unwrap();
        let lsp_notebook = LspNotebook::new(ruff_notebook, notebook_doc);

        // First code cell (all-cells index 0) → code-cell index 0
        assert_eq!(lsp_notebook.get_code_cell_index(&cell0_url), Some(0));
        // Markdown cell → None (not a code cell)
        assert_eq!(lsp_notebook.get_code_cell_index(&cell1_url), None);
        // Second code cell (all-cells index 2) → code-cell index 1
        assert_eq!(lsp_notebook.get_code_cell_index(&cell2_url), Some(1));
    }
}
