/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;
use std::sync::Arc;

use dupe::Dupe;
use lsp_types::CompletionItem;
use lsp_types::CompletionItemKind;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::prelude::VecExt;
use ruff_source_file::LineColumn;
use serde::Serialize;

use crate::config::config::ConfigFile;
use crate::config::finder::ConfigFinder;
use crate::module::module_info::SourceRange;
use crate::module::module_name::ModuleName;
use crate::module::module_path::ModulePath;
use crate::state::handle::Handle;
use crate::state::require::Require;
use crate::state::state::State;
use crate::sys_info::SysInfo;

#[derive(Serialize)]
pub struct Position {
    #[serde(rename(serialize = "column"))]
    pub column: i32,
    #[serde(rename(serialize = "lineNumber"))]
    pub line: i32,
}

impl Position {
    fn new(position: LineColumn) -> Self {
        Self {
            line: position.line.to_zero_indexed() as i32 + 1,
            column: position.column.to_zero_indexed() as i32 + 1,
        }
    }
}

#[derive(Serialize)]
pub struct Range {
    #[serde(rename(serialize = "startLineNumber"))]
    pub start_line: i32,
    #[serde(rename(serialize = "startColumn"))]
    pub start_col: i32,
    #[serde(rename(serialize = "endLineNumber"))]
    pub end_line: i32,
    #[serde(rename(serialize = "endColumn"))]
    pub end_col: i32,
}

impl Range {
    fn new(range: SourceRange) -> Self {
        Self {
            start_line: range.start.line.to_zero_indexed() as i32 + 1,
            start_col: range.start.column.to_zero_indexed() as i32 + 1,
            end_line: range.end.line.to_zero_indexed() as i32 + 1,
            end_col: range.end.column.to_zero_indexed() as i32 + 1,
        }
    }
}

#[derive(Serialize)]
pub struct Diagnostic {
    #[serde(rename(serialize = "startLineNumber"))]
    pub start_line: i32,
    #[serde(rename(serialize = "startColumn"))]
    pub start_col: i32,
    #[serde(rename(serialize = "endLineNumber"))]
    pub end_line: i32,
    #[serde(rename(serialize = "endColumn"))]
    pub end_col: i32,
    pub message: String,
    pub kind: String,
    pub severity: i32,
}

#[derive(Serialize)]
pub struct TypeQueryContent {
    language: String,
    value: String,
}

#[derive(Serialize)]
pub struct TypeQueryResult {
    contents: Vec<TypeQueryContent>,
}

#[derive(Serialize)]
pub struct AutoCompletionItem {
    label: String,
    detail: Option<String>,
    kind: Option<CompletionItemKind>,
    #[serde(rename(serialize = "sortText"))]
    sort_text: Option<String>,
}

#[derive(Serialize)]
pub struct InlayHint {
    label: String,
    position: Position,
}

pub struct Playground {
    state: State,
    handle: Handle,
}

impl Playground {
    pub fn new() -> Self {
        let mut config = ConfigFile::default();
        config.python_environment.set_empty_to_default();
        config.configure();
        let config = ArcId::new(config);

        let state = State::new(ConfigFinder::new_constant(config.dupe()));
        let handle = Handle::new(
            ModuleName::from_str("test"),
            ModulePath::memory(PathBuf::from("test.py")),
            SysInfo::default(),
        );
        let mut me = Self { state, handle };
        me.update_source("".to_owned());
        me
    }

    pub fn update_source(&mut self, source: String) {
        let source = Arc::new(source);
        let mut transaction = self
            .state
            .new_committable_transaction(Require::Exports, None);
        transaction
            .as_mut()
            .set_memory(vec![(PathBuf::from("test.py"), Some(source))]);
        self.state.run_with_committing_transaction(
            transaction,
            &[(self.handle.dupe(), Require::Everything)],
        );
    }

    pub fn get_errors(&self) -> Vec<Diagnostic> {
        self.state
            .transaction()
            .get_errors([&self.handle])
            .collect_errors()
            .shown
            .into_iter()
            .map(|e| {
                let range = e.source_range();
                Diagnostic {
                    start_line: range.start.line.to_zero_indexed() as i32 + 1,
                    start_col: range.start.column.to_zero_indexed() as i32 + 1,
                    end_line: range.end.line.to_zero_indexed() as i32 + 1,
                    end_col: range.end.column.to_zero_indexed() as i32 + 1,
                    message: e.msg().to_owned(),
                    kind: e.error_kind().to_name().to_owned(),
                    severity: 8,
                }
            })
            .collect()
    }

    pub fn query_type(&mut self, line: i32, column: i32) -> Option<TypeQueryResult> {
        let handle = self.handle.dupe();
        let transaction = self.state.transaction();
        transaction
            .get_module_info(&handle)
            .map(|info| info.to_text_size((line - 1) as u32, (column - 1) as u32))
            .and_then(|position| transaction.get_type_at(&handle, position))
            .map(|t| t.to_string())
            .map(|result| TypeQueryResult {
                contents: vec![TypeQueryContent {
                    language: "python".to_owned(),
                    value: result,
                }],
            })
    }

    pub fn goto_definition(&mut self, line: i32, column: i32) -> Option<Range> {
        let handle = self.handle.dupe();
        let transaction = self.state.transaction();
        transaction
            .get_module_info(&handle)
            .map(|info| info.to_text_size((line - 1) as u32, (column - 1) as u32))
            .and_then(|position| transaction.goto_definition(&handle, position))
            .map(|range_with_mod_info| {
                Range::new(
                    range_with_mod_info
                        .module_info
                        .source_range(range_with_mod_info.range),
                )
            })
    }

    pub fn autocomplete(&mut self, line: i32, column: i32) -> Vec<AutoCompletionItem> {
        let handle = self.handle.dupe();
        let transaction = self.state.transaction();
        transaction
            .get_module_info(&handle)
            .map(|info| info.to_text_size((line - 1) as u32, (column - 1) as u32))
            .map_or(Vec::new(), |position| {
                transaction.completion(&handle, position)
            })
            .into_iter()
            .map(
                |CompletionItem {
                     label,
                     detail,
                     sort_text,
                     kind,
                     ..
                 }| AutoCompletionItem {
                    label,
                    detail,
                    kind,
                    sort_text,
                },
            )
            .collect::<Vec<_>>()
    }

    pub fn inlay_hint(&mut self) -> Vec<InlayHint> {
        let handle = self.handle.dupe();
        let transaction = self.state.transaction();
        transaction
            .get_module_info(&handle)
            .zip(transaction.inlay_hints(&handle))
            .map(|(info, hints)| {
                hints.into_map(|(position, label)| {
                    let position = Position::new(info.line_column(position));
                    InlayHint { label, position }
                })
            })
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::kind::ErrorKind;

    #[test]
    fn test_regular_import() {
        let mut state = Playground::new();
        let expected_errors: Vec<String> = Vec::new();

        state.update_source("from typing import *".to_owned());

        assert_eq!(
            state
                .get_errors()
                .into_iter()
                .map(|x| x.message)
                .collect::<Vec<_>>(),
            expected_errors,
        );
    }

    #[test]
    fn test_invalid_import() {
        let mut state = Playground::new();
        state.update_source("from t".to_owned());
        let expected_errors = &[
            "Could not find import of `t`\n  No search path or site package path",
            "Parse error: Expected 'import', found newline",
        ];
        let expected_error_kinds = &[ErrorKind::ImportError, ErrorKind::ParseError];

        assert_eq!(&state.get_errors().into_map(|x| x.message), expected_errors);

        assert_eq!(
            state.get_errors().into_map(|x| x.kind),
            expected_error_kinds.map(|k| k.to_name()),
        );
    }
}
