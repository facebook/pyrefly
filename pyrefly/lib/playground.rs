/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;

use dupe::Dupe;
use lsp_types::CompletionItem;
use lsp_types::CompletionItemKind;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::lined_buffer::DisplayPos;
use pyrefly_util::lined_buffer::DisplayRange;
use pyrefly_util::lined_buffer::LineNumber;
use pyrefly_util::prelude::VecExt;
use ruff_text_size::TextSize;
use serde::Serialize;

use crate::config::config::ConfigFile;
use crate::config::error_kind::Severity;
use crate::config::finder::ConfigFinder;
use crate::state::handle::Handle;
use crate::state::require::Require;
use crate::state::state::State;
use crate::state::state::Transaction;

#[derive(Serialize)]
pub struct Position {
    #[serde(rename(serialize = "column"))]
    pub column: i32,
    #[serde(rename(serialize = "lineNumber"))]
    pub line: i32,
}

impl Position {
    pub fn new(line: i32, column: i32) -> Self {
        Self { line, column }
    }

    fn from_display_pos(position: DisplayPos) -> Self {
        Self {
            line: position.line.get() as i32,
            column: position.column.get() as i32,
        }
    }

    // This should always succeed, but we are being convervative
    fn to_display_pos(&self) -> Option<DisplayPos> {
        Some(DisplayPos {
            line: LineNumber::new(u32::try_from(self.line).ok()?)?,
            column: NonZeroU32::new(u32::try_from(self.column).ok()?)?,
        })
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
    fn new(range: DisplayRange) -> Self {
        Self {
            start_line: range.start.line.get() as i32,
            start_col: range.start.column.get() as i32,
            end_line: range.end.line.get() as i32,
            end_col: range.end.column.get() as i32,
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
    pub message_header: String,
    pub message_details: String,
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
        config.interpreters.skip_interpreter_query = true;
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
            .into_map(|e| {
                let range = e.display_range();
                Diagnostic {
                    start_line: range.start.line.get() as i32,
                    start_col: range.start.column.get() as i32,
                    end_line: range.end.line.get() as i32,
                    end_col: range.end.column.get() as i32,
                    message_header: e.msg_header().to_owned(),
                    message_details: e.msg_details().unwrap_or("").to_owned(),
                    kind: e.error_kind().to_name().to_owned(),
                    // Severity values defined here: https://microsoft.github.io/monaco-editor/typedoc/enums/MarkerSeverity.html
                    severity: match e.error_kind().default_severity() {
                        Severity::Error => 8,
                        Severity::Warn => 4,
                        Severity::Info => 2,
                        Severity::Ignore => 1, // Ignored errors shouldn't be in `CollectedErrors.shown`
                    },
                }
            })
    }

    fn to_text_size(&self, transaction: &Transaction, pos: Position) -> Option<TextSize> {
        let info = transaction.get_module_info(&self.handle)?;
        Some(info.lined_buffer().from_display_pos(pos.to_display_pos()?))
    }

    pub fn query_type(&self, pos: Position) -> Option<TypeQueryResult> {
        let transaction = self.state.transaction();
        let position = self.to_text_size(&transaction, pos)?;
        let t = transaction.get_type_at(&self.handle, position)?;
        Some(TypeQueryResult {
            contents: vec![TypeQueryContent {
                language: "python".to_owned(),
                value: t.to_string(),
            }],
        })
    }

    pub fn goto_definition(&mut self, pos: Position) -> Option<Range> {
        let transaction = self.state.transaction();
        let position = self.to_text_size(&transaction, pos)?;
        // TODO: Support goto multiple definitions
        transaction
            .goto_definition(&self.handle, position)
            .into_iter()
            .next()
            .map(|r| Range::new(r.module.display_range(r.range)))
    }

    pub fn autocomplete(&self, pos: Position) -> Vec<AutoCompletionItem> {
        let transaction = self.state.transaction();
        self.to_text_size(&transaction, pos)
            .map_or(Vec::new(), |position| {
                transaction.completion(&self.handle, position)
            })
            .into_map(
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
    }

    pub fn inlay_hint(&self) -> Vec<InlayHint> {
        let transaction = self.state.transaction();
        transaction
            .get_module_info(&self.handle)
            .zip(transaction.inlay_hints(&self.handle))
            .map(|(info, hints)| {
                hints.into_map(|(position, label)| {
                    let position = Position::from_display_pos(info.display_pos(position));
                    InlayHint { label, position }
                })
            })
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::error_kind::ErrorKind;

    #[test]
    fn test_regular_import() {
        let mut state = Playground::new();
        let expected_errors: Vec<String> = Vec::new();

        state.update_source("from typing import *".to_owned());

        assert_eq!(
            state
                .get_errors()
                .into_iter()
                .map(|x| x.message_header)
                .collect::<Vec<_>>(),
            expected_errors,
        );
    }

    #[test]
    fn test_invalid_import() {
        let mut state = Playground::new();
        state.update_source("from t".to_owned());
        let expected_headers = &[
            "Could not find import of `t`",
            "Parse error: Expected 'import', found newline",
        ];
        let expected_details = &["  No search path or site package path", ""];
        let expected_error_kinds = &[ErrorKind::ImportError, ErrorKind::ParseError];

        assert_eq!(
            &state.get_errors().into_map(|x| x.message_header),
            expected_headers
        );
        assert_eq!(
            &state.get_errors().into_map(|x| x.message_details),
            expected_details
        );

        assert_eq!(
            state.get_errors().into_map(|x| x.kind),
            expected_error_kinds.map(|k| k.to_name()),
        );
    }
}
