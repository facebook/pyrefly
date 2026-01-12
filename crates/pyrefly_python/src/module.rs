/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Debug;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::lined_buffer::DisplayPos;
use pyrefly_util::lined_buffer::DisplayRange;
use pyrefly_util::lined_buffer::LinedBuffer;
use ruff_notebook::Notebook;
use ruff_python_ast::PySourceType;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use starlark_map::small_set::SmallSet;

use crate::ignore::Ignore;
use crate::ignore::Tool;
use crate::module_name::ModuleName;
use crate::module_path::ModulePath;

pub static GENERATED_TOKEN: &str = concat!("@", "generated");

/// Information about a module, notably its name, path, and contents.
#[derive(Clone, Dupe, PartialEq, Eq, Hash)]
pub struct Module(ArcId<ModuleInner>);

#[derive(Debug, Clone)]
struct ModuleInner {
    name: ModuleName,
    path: ModulePath,
    ignore: Ignore,
    is_generated: bool,
    contents: LinedBuffer,
    notebook: Option<Arc<Notebook>>,
}

impl Debug for Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Module")
            .field("name", &self.0.name)
            .field("path", &self.0.path)
            .finish_non_exhaustive()
    }
}

impl Module {
    /// Create a new Module. Will NOT read the `path`, but use the value from `contents` instead.
    pub fn new(name: ModuleName, path: ModulePath, contents: Arc<String>) -> Self {
        let ignore = Ignore::new(&contents);
        let is_generated = contents.contains(GENERATED_TOKEN);
        let contents = LinedBuffer::new(contents);
        Self(ArcId::new(ModuleInner {
            name,
            path,
            ignore,
            is_generated,
            contents,
            notebook: None,
        }))
    }

    pub fn new_notebook(name: ModuleName, path: ModulePath, notebook: Arc<Notebook>) -> Self {
        let contents: Arc<String> = Arc::from(notebook.source_code().to_owned());
        let ignore = Ignore::new(&contents);
        let is_generated = contents.contains(GENERATED_TOKEN);
        let contents = LinedBuffer::new(contents);
        Self(ArcId::new(ModuleInner {
            name,
            path,
            ignore,
            is_generated,
            contents,
            notebook: Some(notebook),
        }))
    }

    pub fn lined_buffer(&self) -> &LinedBuffer {
        &self.0.contents
    }

    pub fn line_count(&self) -> usize {
        // By default we count the empty lines, but sometimes to get stats
        // we might need to only count the non-empty/non-comment lines.
        const COUNT_EMPTY_LINES: bool = true;
        if COUNT_EMPTY_LINES {
            self.0.contents.line_count()
        } else {
            self.0
                .contents
                .lines()
                .filter(|x| {
                    let res = x.trim_start();
                    !res.is_empty() && !res.starts_with('#')
                })
                .count()
        }
    }

    pub fn display_range(&self, range: TextRange) -> DisplayRange {
        self.0.contents.display_range(range, self.notebook())
    }

    pub fn display_pos(&self, offset: TextSize) -> DisplayPos {
        self.0.contents.display_pos(offset, self.notebook())
    }

    pub fn to_lsp_range(&self, x: TextRange) -> lsp_types::Range {
        self.lined_buffer().to_lsp_range(x, self.notebook())
    }

    pub fn to_lsp_position(&self, x: TextSize) -> lsp_types::Position {
        self.lined_buffer().to_lsp_position(x, self.notebook())
    }

    /// If the module is a notebook, take an input position relative to the concatenated contents
    /// and return the index of the corresponding notebook cell.
    pub fn to_cell_for_lsp(&self, x: TextSize) -> Option<usize> {
        self.lined_buffer().to_cell_for_lsp(x, self.notebook())
    }

    /// Translates an LSP position to a text size.
    /// For notebooks, the input position is relative to a notebook cell and the output
    /// position is relative to the concatenated contents of the notebook.
    pub fn from_lsp_position(
        &self,
        position: lsp_types::Position,
        notebook_cell: Option<usize>,
    ) -> TextSize {
        self.lined_buffer().from_lsp_position(
            position,
            notebook_cell.map(|c| (self.notebook().unwrap(), c)),
        )
    }

    /// Translates an LSP position to a text range.
    /// For notebooks, the input range is relative to a notebook cell and the output
    /// position is range to the concatenated contents of the notebook.
    pub fn from_lsp_range(
        &self,
        position: lsp_types::Range,
        notebook_cell: Option<usize>,
    ) -> TextRange {
        self.lined_buffer().from_lsp_range(
            position,
            notebook_cell.map(|c| (self.notebook().unwrap(), c)),
        )
    }

    pub fn code_at(&self, range: TextRange) -> &str {
        self.0.contents.code_at(range)
    }

    pub fn path(&self) -> &ModulePath {
        &self.0.path
    }

    pub fn is_notebook(&self) -> bool {
        self.0.notebook.is_some()
    }

    pub fn is_generated(&self) -> bool {
        self.0.is_generated
    }

    pub fn contents(&self) -> &Arc<String> {
        self.0.contents.contents()
    }

    pub fn name(&self) -> ModuleName {
        self.0.name
    }

    pub fn is_ignored(
        &self,
        source_range: &DisplayRange,
        error_kind: &str,
        enabled_ignores: &SmallSet<Tool>,
    ) -> bool {
        self.0.ignore.is_ignored(
            source_range.start.line_within_file(),
            error_kind,
            enabled_ignores,
        )
    }

    pub fn ignore(&self) -> &Ignore {
        &self.0.ignore
    }

    pub fn notebook(&self) -> Option<&Notebook> {
        self.0.notebook.as_deref()
    }

    pub fn source_type(&self) -> PySourceType {
        if self.is_notebook() {
            PySourceType::Ipynb
        } else {
            PySourceType::Python
        }
    }
}

#[derive(Debug, Clone)]
pub struct TextRangeWithModule {
    pub module: Module,
    pub range: TextRange,
}

impl TextRangeWithModule {
    pub fn new(module: Module, range: TextRange) -> Self {
        Self { module, range }
    }
}
