/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use export::docstring::Docstring;
use lsp_types::Hover;
use lsp_types::HoverContents;
use lsp_types::MarkupContent;
use lsp_types::MarkupKind;
use lsp_types::Url;
use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_types::types::Type;
use ruff_text_size::TextSize;
use starlark_map::small_set::SmallSet;

use crate::export;
use crate::state::handle::Handle;
use crate::state::lsp::FindDefinitionItemWithDocstring;
use crate::state::state::Transaction;

pub struct HoverValue {
    pub kind: Option<SymbolKind>,
    pub name: Option<String>,
    pub type_: Type,
    pub docstring: Option<Docstring>,
}

impl HoverValue {
    fn format_symbol_def_locations(t: &Type) -> Option<String> {
        let mut tracked_def_locs = SmallSet::new();
        t.universe(&mut |t| tracked_def_locs.extend(t.qname()));
        let linked_names = tracked_def_locs
            .into_iter()
            .filter_map(|qname| {
                if let Ok(mut url) = Url::from_file_path(qname.module_path().as_path()) {
                    let start_pos = qname.module().display_range(qname.range()).start;
                    url.set_fragment(Some(&format!(
                        "L{},{}",
                        start_pos.line.get(),
                        start_pos.column
                    )));
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
            Some(format!("\n---\nGo to {linked_names}"))
        }
    }

    pub fn format(&self) -> Hover {
        let docstring_formatted = self.docstring.clone().map_or("".to_owned(), |docstring| {
            format!("\n---\n{}", docstring.as_string().trim())
        });
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
                    self.type_,
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
    let type_ = transaction.get_type_at(handle, position)?;
    let (kind, name, docstring) = if let Some(FindDefinitionItemWithDocstring {
        metadata,
        location,
        docstring,
    }) = transaction
        .find_definition(handle, position, true)
        // TODO: handle more than 1 definition
        .into_iter()
        .next()
    {
        (
            metadata.symbol_kind(),
            Some(location.module.code_at(location.range).to_owned()),
            docstring,
        )
    } else {
        (None, None, None)
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
