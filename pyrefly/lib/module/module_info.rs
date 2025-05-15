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
use ruff_python_ast::ModModule;
use ruff_source_file::LineIndex;
use ruff_source_file::OneIndexed;
use ruff_source_file::SourceLocation;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

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
    pub start: SourceLocation,
    pub end: SourceLocation,
}

impl Display for SourceRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.start.row == self.end.row {
            if self.start.column == self.end.column {
                write!(f, "{}:{}", self.start.row, self.start.column)
            } else {
                write!(
                    f,
                    "{}:{}-{}",
                    self.start.row, self.start.column, self.end.column
                )
            }
        } else {
            write!(
                f,
                "{}:{}-{}:{}",
                self.start.row, self.start.column, self.end.row, self.end.column
            )
        }
    }
}

/// Information about a module, notably its name, path, and contents.
#[derive(Debug, Clone, Dupe)]
pub struct ModuleInfo(Arc<ModuleInfoInner>);

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
        Self(Arc::new(ModuleInfoInner {
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
            start: self.source_location(range.start()),
            end: self.source_location(range.end()),
        }
    }

    pub fn source_location(&self, offset: TextSize) -> SourceLocation {
        assert!(
            offset.to_usize() <= self.len(),
            "Module {}({}): offset out of range, expected {} <= {}",
            self.0.name,
            self.0.path,
            offset.to_usize(),
            self.len()
        );
        self.0.index.source_location(offset, &self.0.contents)
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
                format!("Parse error: {}", err.error),
                ErrorKind::ParseError,
                None,
            );
        }
        SemanticSyntaxContext::new(version, errors).visit(&module);
        module
    }

    pub fn name(&self) -> ModuleName {
        self.0.name
    }

    pub fn to_text_size(&self, line: u32, column: u32) -> TextSize {
        self.0.index.offset(
            OneIndexed::from_zero_indexed(line as usize),
            OneIndexed::from_zero_indexed(column as usize),
            &self.0.contents,
        )
    }

    pub fn is_ignored(&self, source_range: &SourceRange, msg: &str) -> bool {
        self.0.ignore.is_ignored(source_range, msg)
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
    version: ruff_python_ast::PythonVersion,
    errors: &'me ErrorCollector,
}

impl<'me> SemanticSyntaxContext<'me> {
    pub fn new(version: PythonVersion, errors: &'me ErrorCollector) -> Self {
        Self {
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
    fn seen_docstring_boundary(&self) -> bool {
        false
    }

    fn python_version(&self) -> ruff_python_ast::PythonVersion {
        self.version
    }

    fn report_semantic_error(
        &self,
        error: ruff_python_parser::semantic_errors::SemanticSyntaxError,
    ) {
        self.errors
            .add(error.range, error.to_string(), ErrorKind::InvalidSyntax, None);
    }
}
