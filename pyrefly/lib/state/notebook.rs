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

    pub fn get_cell_index(&self, cell_url: &Url) -> Option<usize> {
        self.cell_url_to_index.get(cell_url).copied()
    }

    pub fn get_cell_url(&self, cell_index: usize) -> Option<&Url> {
        self.cell_index_to_url.get(cell_index)
    }

    pub fn cell_urls(&self) -> &Vec<Url> {
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
