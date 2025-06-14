/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Display;
use std::ops::Range;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_util::arc_id::ArcId;
use ruff_python_ast::ModModule;
use ruff_source_file::LineColumn;
use ruff_source_file::LineIndex;
use ruff_source_file::OneIndexed;
use ruff_source_file::PositionEncoding;
use ruff_source_file::SourceLocation;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use vec1::vec1;

use crate::error::collector::ErrorCollector;
use crate::error::kind::ErrorKind;
use crate::module::ignore::Ignore;
use crate::module::module_name::ModuleName;
use crate::module::module_path::ModulePath;
use crate::ruff::ast::Ast;
use crate::sys_info::PythonVersion;

pub static GENERATED_TOKEN: &str = concat!("@", "generated");

#[derive(Debug, Clone, Ord, PartialOrd, PartialEq, Eq, Hash, Default)]
pub struct SourceRange {
    pub start: LineColumn,
    pub end: LineColumn,
}

impl Display for SourceRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.start.line == self.end.line {
            if self.start.column == self.end.column {
                write!(f, "{}:{}", self.start.line, self.start.column)
            } else {
                write!(
                    f,
                    "{}:{}-{}",
                    self.start.line, self.start.column, self.end.column
                )
            }
        } else {
            write!(
                f,
                "{}:{}-{}:{}",
                self.start.line, self.start.column, self.end.line, self.end.column
            )
        }
    }
}

/// Information about a module, notably its name, path, and contents.
#[derive(Debug, Clone, Dupe, PartialEq, Eq, Hash)]
pub struct ModuleInfo(ArcId<ModuleInfoInner>);

#[derive(Debug, Clone)]
struct ModuleInfoInner {
    name: ModuleName,
    path: ModulePath,
    index: LineIndex,
    ignore: Ignore,
    is_generated: bool,
    contents: Arc<String>,
}

impl ModuleInfo {
    /// Create a new ModuleInfo. Will NOT read the `path`, but use the value from `contents` instead.
    pub fn new(name: ModuleName, path: ModulePath, contents: Arc<String>) -> Self {
        let index = LineIndex::from_source_text(&contents);
        let ignore = Ignore::new(&contents);
        let is_generated = contents.contains(GENERATED_TOKEN);
        Self(ArcId::new(ModuleInfoInner {
            name,
            path,
            index,
            ignore,
            is_generated,
            contents,
        }))
    }

    pub fn len(&self) -> usize {
        self.0.contents.len()
    }

    pub fn line_count(&self) -> usize {
        // By default we count the empty lines, but sometimes to get stats
        // we might need to only count the non-empty/non-comment lines.
        const COUNT_EMPTY_LINES: bool = true;
        if COUNT_EMPTY_LINES {
            self.0.index.line_count()
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

    pub fn source_range(&self, range: TextRange) -> SourceRange {
        SourceRange {
            start: self.line_column(range.start()),
            end: self.line_column(range.end()),
        }
    }

    pub fn line_column(&self, offset: TextSize) -> LineColumn {
        assert!(
            offset.to_usize() <= self.len(),
            "Module {}({}): offset out of range, expected {} <= {}",
            self.0.name,
            self.0.path,
            offset.to_usize(),
            self.len()
        );
        self.0.index.line_column(offset, &self.0.contents)
    }

    pub fn code_at(&self, range: TextRange) -> &str {
        match self.0.contents.get(Range::<usize>::from(range)) {
            Some(code) => code,
            None => panic!(
                "Module {}({}): `range` is invalid, got {range:?}, but file is {} bytes long",
                self.0.name,
                self.0.path,
                self.0.contents.len()
            ),
        }
    }

    pub fn path(&self) -> &ModulePath {
        &self.0.path
    }

    pub fn is_generated(&self) -> bool {
        self.0.is_generated
    }

    pub fn contents(&self) -> &Arc<String> {
        &self.0.contents
    }

    pub fn parse(&self, version: PythonVersion, errors: &ErrorCollector) -> ModModule {
        let (module, parse_errors) = Ast::parse(self.contents());
        for err in parse_errors {
            errors.add(
                err.location,
                ErrorKind::ParseError,
                None,
                vec1![format!("Parse error: {}", err.error)],
            );
        }
        SemanticSyntaxContext::new(self.contents(), version, errors).visit(&module);
        module
    }

    pub fn name(&self) -> ModuleName {
        self.0.name
    }

    pub fn to_text_size(&self, line: u32, column: u32) -> TextSize {
        self.0.index.offset(
            SourceLocation {
                line: OneIndexed::from_zero_indexed(line as usize),
                character_offset: OneIndexed::from_zero_indexed(column as usize),
            },
            &self.0.contents,
            PositionEncoding::Utf32,
        )
    }

    pub fn to_text_range(&self, source_range: &SourceRange) -> TextRange {
        TextRange::new(
            self.to_text_size(
                source_range.start.line.to_zero_indexed() as u32,
                source_range.start.column.to_zero_indexed() as u32,
            ),
            self.to_text_size(
                source_range.end.line.to_zero_indexed() as u32,
                source_range.end.column.to_zero_indexed() as u32,
            ),
        )
    }

    /// Gets the content from the beginning of start_line to the end of end_line.
    pub fn content_in_line_range(&self, start_line: OneIndexed, end_line: OneIndexed) -> &str {
        debug_assert!(start_line <= end_line);
        let start = self.0.index.line_start(start_line, &self.0.contents);
        let end = self.0.index.line_end(end_line, &self.0.contents);
        &self.0.contents[start.to_usize()..end.to_usize()]
    }

    pub fn is_ignored(&self, source_range: &SourceRange) -> bool {
        // Extend the range of the error to include comment lines before it.
        // This makes it so that the preceding ignore could "see through" comments.
        let start_line = {
            let mut start_line = source_range.start.line;
            while let Some(earlier_line) = start_line.checked_sub(OneIndexed::MIN) {
                let earlier_line_content = &self
                    .content_in_line_range(earlier_line, earlier_line)
                    .trim();
                if Ignore::get_suppression_kind(earlier_line_content).is_some() {
                    break;
                } else if earlier_line_content.starts_with('#') {
                    start_line = earlier_line;
                } else {
                    break;
                }
            }
            start_line
        };
        self.0.ignore.is_ignored(start_line, source_range.end.line)
    }

    pub fn ignore(&self) -> &Ignore {
        &self.0.ignore
    }
}

#[derive(Debug, Clone)]
pub struct TextRangeWithModuleInfo {
    pub module_info: ModuleInfo,
    pub range: TextRange,
}

impl TextRangeWithModuleInfo {
    pub fn new(module_info: ModuleInfo, range: TextRange) -> Self {
        Self { module_info, range }
    }
}

pub struct SemanticSyntaxContext<'me> {
    content: &'me str,
    version: ruff_python_ast::PythonVersion,
    errors: &'me ErrorCollector,
}

impl<'me> SemanticSyntaxContext<'me> {
    pub fn new(content: &'me str, version: PythonVersion, errors: &'me ErrorCollector) -> Self {
        Self {
            content,
            version: ruff_python_ast::PythonVersion {
                major: version.major as u8,
                minor: version.minor as u8,
            },
            errors,
        }
    }

    pub fn visit(&self, module: &ModModule) {
        let mut checker = ruff_python_parser::semantic_errors::SemanticSyntaxChecker::new();
        module.body.iter().for_each(|stmt| {
            checker.visit_stmt(stmt, self);
        });
    }
}

impl<'me> ruff_python_parser::semantic_errors::SemanticSyntaxContext
    for SemanticSyntaxContext<'me>
{
    fn python_version(&self) -> ruff_python_ast::PythonVersion {
        self.version
    }

    fn report_semantic_error(
        &self,
        error: ruff_python_parser::semantic_errors::SemanticSyntaxError,
    ) {
        self.errors.add(
            error.range,
            ErrorKind::InvalidSyntax,
            None,
            vec1![error.to_string()],
        );
    }

    fn future_annotations_or_stub(&self) -> bool {
        false
    }

    fn source(&self) -> &str {
        self.content
    }

    fn global(&self, _: &str) -> Option<TextRange> {
        // TODO: Properly implement this
        None
    }

    fn in_async_context(&self) -> bool {
        // TODO: Properly implement this
        true
    }

    fn in_await_allowed_context(&self) -> bool {
        // TODO: Properly implement this
        true
    }

    fn in_yield_allowed_context(&self) -> bool {
        // TODO: Properly implement this
        true
    }

    fn in_sync_comprehension(&self) -> bool {
        // TODO: Properly implement this
        false
    }

    fn in_module_scope(&self) -> bool {
        // TODO: Properly implement this
        false
    }

    fn in_function_scope(&self) -> bool {
        // TODO: Properly implement this
        true
    }

    fn in_generator_scope(&self) -> bool {
        // TODO: Properly implement this
        false
    }

    fn in_notebook(&self) -> bool {
        // TODO: Properly implement this
        false
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ruff_source_file::LineColumn;

    use super::*;

    #[test]
    fn test_module_info_unicode() {
        // Test with a mix of ASCII, accented characters, and emoji
        let contents =
            "def greet(name: str) -> str:\n    return f\"Bonjour {name}! 👋 Café? ☕\"\n# done\n";
        let module_info = ModuleInfo::new(
            ModuleName::from_str("module_info_test_suite"),
            ModulePath::filesystem("test.py".into()),
            Arc::new(contents.to_owned()),
        );

        assert_eq!(module_info.line_count(), 4);

        let range = |l1, c1, l2, c2| SourceRange {
            start: LineColumn {
                line: OneIndexed::from_zero_indexed(l1),
                column: OneIndexed::from_zero_indexed(c1),
            },
            end: LineColumn {
                line: OneIndexed::from_zero_indexed(l2),
                column: OneIndexed::from_zero_indexed(c2),
            },
        };

        assert_eq!(
            module_info.code_at(module_info.to_text_range(&range(1, 4, 2, 0))),
            "return f\"Bonjour {name}! 👋 Café? ☕\"\n"
        );

        assert_eq!(
            module_info.code_at(module_info.to_text_range(&range(1, 29, 1, 36))),
            "👋 Café?"
        );
        assert_eq!(
            module_info.code_at(module_info.to_text_range(&range(2, 2, 2, 4))),
            "do"
        );
    }
}
