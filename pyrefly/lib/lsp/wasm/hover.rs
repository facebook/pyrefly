/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::Hover;
use lsp_types::HoverContents;
use lsp_types::MarkupContent;
use lsp_types::MarkupKind;
use lsp_types::Url;
use pyrefly_build::handle::Handle;
use pyrefly_python::docstring::Docstring;
use pyrefly_python::ignore::Ignore;
use pyrefly_python::ignore::find_comment_start_in_line;
use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_types::types::Type;
use pyrefly_util::lined_buffer::LineNumber;
use ruff_text_size::TextSize;
use starlark_map::small_set::SmallSet;

use crate::error::error::Error;
use crate::state::lsp::FindDefinitionItemWithDocstring;
use crate::state::lsp::FindPreference;
use crate::state::state::Transaction;

/// Gets all suppressed errors that overlap with the given line.
///
/// This function filters the suppressed errors for a specific handle to find
/// only those that affect the line where a suppression applies.
fn get_suppressed_errors_for_line(
    transaction: &Transaction,
    handle: &Handle,
    suppression_line: LineNumber,
    ignore: &Ignore,
) -> Vec<Error> {
    let errors = transaction.get_errors(std::iter::once(handle));
    let suppressed = errors.collect_errors().suppressed;
    // Filter errors that overlap with the suppression line
    suppressed
        .into_iter()
        .filter(|error| {
            let range = error.display_range();
            ignore.is_ignored_by_suppression_line(
                suppression_line,
                range.start.line_within_file(),
                range.end.line_within_file(),
                error.error_kind().to_name(),
                false,
            )
        })
        .collect()
}

/// Formats suppressed errors into a hover response with markdown.
///
/// The format varies based on the number of errors:
/// - No errors: Shows a message that no errors are suppressed
/// - Single error: Shows the error kind and message
/// - Multiple errors: Shows a bulleted list of all suppressed errors
fn format_suppressed_errors_hover(errors: Vec<Error>) -> Hover {
    let content = if errors.is_empty() {
        "**No errors suppressed by this ignore**\n\n_The ignore comment may have an incorrect error code or there may be no errors on this line._".to_owned()
    } else if errors.len() == 1 {
        let err = &errors[0];
        format!(
            "**Suppressed Error**\n\n`{}`: {}",
            err.error_kind().to_name(),
            err.msg()
        )
    } else {
        let mut content = "**Suppressed Errors**\n\n".to_owned();
        for err in &errors {
            content.push_str(&format!(
                "- `{}`: {}\n",
                err.error_kind().to_name(),
                err.msg()
            ));
        }
        content
    };

    Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: content,
        }),
        range: None,
    }
}

pub struct HoverValue {
    pub kind: Option<SymbolKind>,
    pub name: Option<String>,
    pub type_: Type,
    pub docstring: Option<Docstring>,
}

impl HoverValue {
    #[cfg(not(target_arch = "wasm32"))]
    fn format_symbol_def_locations(t: &Type) -> Option<String> {
        let mut tracked_def_locs = SmallSet::new();
        t.universe(&mut |t| tracked_def_locs.extend(t.qname()));
        let linked_names = tracked_def_locs
            .into_iter()
            .filter_map(|qname| {
                if let Ok(mut url) = Url::from_file_path(qname.module_path().as_path()) {
                    let start_pos = qname.module().display_range(qname.range()).start;
                    if let Some(cell) = start_pos.cell() {
                        url.set_fragment(Some(&format!(
                            "{},L{},{}",
                            cell.get(),
                            start_pos.line_within_cell().get(),
                            start_pos.column()
                        )));
                    } else {
                        url.set_fragment(Some(&format!(
                            "L{},{}",
                            start_pos.line_within_file().get(),
                            start_pos.column()
                        )));
                    }
                    Some(format!("[{}]({})", qname.id(), url))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(" | ");

        if linked_names.is_empty() {
            None
        } else {
            Some(format!("\n\nGo to {linked_names}"))
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn format_symbol_def_locations(t: &Type) -> Option<String> {
        None
    }

    pub fn format(&self) -> Hover {
        let docstring_formatted = self
            .docstring
            .as_ref()
            .map(|docstring| docstring.resolve())
            .map_or_else(
                || "".to_owned(),
                |content| format!("\n---\n{}", content.trim()),
            );
        let kind_formatted = self.kind.map_or("".to_owned(), |kind| {
            format!("{} ", kind.display_for_hover())
        });
        let name_formatted = self
            .name
            .as_ref()
            .map_or("".to_owned(), |s| format!("{s}: "));
        let symbol_def_formatted =
            HoverValue::format_symbol_def_locations(&self.type_).unwrap_or("".to_owned());

        Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: format!(
                    "```python\n{}{}{}\n```{}{}",
                    kind_formatted,
                    name_formatted,
                    self.type_.as_hover_string(),
                    docstring_formatted,
                    symbol_def_formatted
                ),
            }),
            range: None,
        }
    }
}

pub fn get_hover(
    transaction: &Transaction<'_>,
    handle: &Handle,
    position: TextSize,
) -> Option<Hover> {
    // Handle hovering over an ignore comment
    if let Some(module) = transaction.get_module_info(handle) {
        let display_pos = module.display_pos(position);
        let line_text = module.lined_buffer().content_in_line_range(
            display_pos.line_within_file(),
            display_pos.line_within_file(),
        );
        // Find comment start in the current line and check if cursor is at or after the comment
        if let Some(comment_offset) = find_comment_start_in_line(line_text)
            && display_pos.column().get() >= comment_offset as u32
        {
            // If the comment appears on its own line, check the next line for suppressed errors
            // Otherwise, check the current line
            let suppression_line = if line_text.trim().starts_with("#") {
                display_pos.line_within_file().increment()
            } else {
                display_pos.line_within_file()
            };
            if module.ignore().get(&suppression_line).is_some() {
                let suppressed_errors = get_suppressed_errors_for_line(
                    transaction,
                    handle,
                    suppression_line,
                    module.ignore(),
                );
                return Some(format_suppressed_errors_hover(suppressed_errors));
            }
        }
    }

    // Otherwise, fall through to the existing type hover logic
    let type_ = transaction.get_type_at(handle, position)?;
    let (kind, name, docstring_range, module) = if let Some(FindDefinitionItemWithDocstring {
        metadata,
        definition_range: definition_location,
        module,
        docstring_range,
    }) = transaction
        .find_definition(
            handle,
            position,
            &FindPreference {
                prefer_pyi: false,
                ..Default::default()
            },
        )
        // TODO: handle more than 1 definition
        .into_iter()
        .next()
    {
        (
            metadata.symbol_kind(),
            Some(module.code_at(definition_location).to_owned()),
            docstring_range,
            Some(module),
        )
    } else {
        (None, None, None, None)
    };

    let docstring = if let (Some(docstring), Some(module)) = (docstring_range, module) {
        Some(Docstring(docstring, module))
    } else {
        None
    };

    Some(
        HoverValue {
            kind,
            name,
            type_,
            docstring,
        }
        .format(),
    )
}
