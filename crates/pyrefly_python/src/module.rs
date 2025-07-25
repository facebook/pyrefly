/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use dupe::Dupe;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::lined_buffer::DisplayPos;
use pyrefly_util::lined_buffer::DisplayRange;
use pyrefly_util::lined_buffer::LinedBuffer;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::ignore::Ignore;
use crate::module_name::ModuleName;
use crate::module_path::ModulePath;

pub static GENERATED_TOKEN: &str = concat!("@", "generated");

/// Information about a module, notably its name, path, and contents.
#[derive(Debug, Clone, Dupe, PartialEq, Eq, Hash)]
pub struct Module(ArcId<ModuleInner>);

#[derive(Debug, Clone)]
struct ModuleInner {
    name: ModuleName,
    path: ModulePath,
    ignore: Ignore,
    is_generated: bool,
    contents: LinedBuffer,
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
        self.0.contents.display_range(range)
    }

    pub fn display_pos(&self, offset: TextSize) -> DisplayPos {
        self.0.contents.display_pos(offset)
    }

    pub fn code_at(&self, range: TextRange) -> &str {
        self.0.contents.code_at(range)
    }

    pub fn path(&self) -> &ModulePath {
        &self.0.path
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
        permissive_ignores: bool,
    ) -> bool {
        self.0.ignore.is_ignored(
            source_range.start.line,
            source_range.end.line,
            error_kind,
            permissive_ignores,
        )
    }

    pub fn ignore(&self) -> &Ignore {
        &self.0.ignore
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
