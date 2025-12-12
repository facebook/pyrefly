/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use lsp_types::Hover;
use lsp_types::HoverContents;
use lsp_types::MarkupContent;
use lsp_types::MarkupKind;
use lsp_types::Url;
use pyrefly_build::handle::Handle;
use pyrefly_python::docstring::Docstring;
use pyrefly_python::docstring::parse_parameter_documentation;
use pyrefly_python::ignore::Ignore;
use pyrefly_python::ignore::Tool;
use pyrefly_python::ignore::find_comment_start_in_line;
use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_types::callable::Callable;
use pyrefly_types::callable::Param;
use pyrefly_types::callable::ParamList;
use pyrefly_types::callable::Params;
use pyrefly_types::callable::Required;
use pyrefly_types::display::LspDisplayMode;
use pyrefly_types::types::Type;
use pyrefly_util::lined_buffer::LineNumber;
use ruff_python_ast::Stmt;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::alt::answers_solver::AnswersSolver;
use crate::error::error::Error;
use crate::lsp::module_helpers::collect_symbol_def_paths;
use crate::state::lsp::DefinitionMetadata;
use crate::state::lsp::FindDefinitionItemWithDocstring;
use crate::state::lsp::FindPreference;
use crate::state::lsp::IdentifierContext;
use crate::state::state::Transaction;
use crate::state::state::TransactionHandle;

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
                &Tool::default_enabled(),
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

fn position_is_in_docstring(
    transaction: &Transaction<'_>,
    handle: &Handle,
    position: TextSize,
) -> bool {
    let Some(ast) = transaction.get_ast(handle) else {
        return false;
    };
    fn body_contains_docstring(body: &[Stmt], position: TextSize) -> bool {
        if let Some(range) = Docstring::range_from_stmts(body)
            && range.contains_inclusive(position)
        {
            return true;
        }
        for stmt in body {
            match stmt {
                Stmt::FunctionDef(func) => {
                    if body_contains_docstring(func.body.as_slice(), position) {
                        return true;
                    }
                }
                Stmt::ClassDef(class_def) => {
                    if body_contains_docstring(class_def.body.as_slice(), position) {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }
    body_contains_docstring(ast.body.as_slice(), position)
}

pub struct HoverValue {
    pub kind: Option<SymbolKind>,
    pub name: Option<String>,
    pub type_: Type,
    pub docstring: Option<Docstring>,
    pub parameter_doc: Option<(String, String)>,
    pub display: Option<String>,
    pub show_go_to_links: bool,
}

impl HoverValue {
    #[cfg(not(target_arch = "wasm32"))]
    fn format_symbol_def_locations(t: &Type) -> Option<String> {
        let symbol_paths = collect_symbol_def_paths(t);
        let linked_names = symbol_paths
            .into_iter()
            .filter_map(|(qname, file_path)| {
                if let Ok(mut url) = Url::from_file_path(&file_path) {
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
        let parameter_doc_formatted =
            self.parameter_doc
                .as_ref()
                .map_or("".to_owned(), |(name, doc)| {
                    let prefix = if self.docstring.is_some() {
                        "\n\n---\n"
                    } else {
                        "\n---\n"
                    };
                    let cleaned = doc.trim().replace('\n', "  \n");
                    format!("{prefix}**Parameter `{}`**\n{}", name, cleaned)
                });
        let kind_formatted = self.kind.map_or_else(
            || {
                if self.type_.is_toplevel_callable() {
                    "(function) ".to_owned()
                } else {
                    String::new()
                }
            },
            |kind| format!("{} ", kind.display_for_hover()),
        );
        let name_formatted = self
            .name
            .as_ref()
            .map_or("".to_owned(), |s| format!("{s}: "));
        let symbol_def_formatted = if self.show_go_to_links {
            HoverValue::format_symbol_def_locations(&self.type_).unwrap_or("".to_owned())
        } else {
            String::new()
        };
        let type_display = self.display.clone().unwrap_or_else(|| {
            self.type_
                .as_lsp_string_with_fallback_name(self.name.as_deref(), LspDisplayMode::Hover)
        });

        Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: format!(
                    "```python\n{}{}{}\n```{}{}{}",
                    kind_formatted,
                    name_formatted,
                    type_display,
                    docstring_formatted,
                    parameter_doc_formatted,
                    symbol_def_formatted
                ),
            }),
            range: None,
        }
    }
}

fn collect_typed_dict_fields_for_hover<'a>(
    solver: &AnswersSolver<TransactionHandle<'a>>,
    ty: &Type,
) -> Option<Vec<(Name, Type, Required)>> {
    match ty {
        Type::Unpack(inner) => match inner.as_ref() {
            Type::TypedDict(typed_dict) => {
                let fields = solver.type_order().typed_dict_kw_param_info(typed_dict);
                if fields.is_empty() {
                    None
                } else {
                    Some(fields)
                }
            }
            _ => None,
        },
        _ => None,
    }
}

fn expand_callable_kwargs_for_hover<'a>(
    solver: &AnswersSolver<TransactionHandle<'a>>,
    callable: &mut Callable,
) {
    if let Params::List(param_list) = &mut callable.params {
        let mut expanded = Vec::with_capacity(param_list.len());
        let mut changed = false;
        for param in param_list.items() {
            if let Param::Kwargs(_, ty) = param
                && let Some(fields) = collect_typed_dict_fields_for_hover(solver, ty)
            {
                changed = true;
                for (field_name, field_type, required) in fields {
                    expanded.push(Param::KwOnly(field_name, field_type, required));
                }
            }
            expanded.push(param.clone());
        }
        if changed {
            *param_list = ParamList::new(expanded);
        }
    }
}

/// If we can't determine a symbol name via go-to-definition, fall back to what the
/// type metadata knows about the callable. This primarily handles third-party stubs
/// where we only have typeshed information.
fn fallback_hover_name_from_type(type_: &Type) -> Option<String> {
    let name = type_.check_toplevel_func_metadata(&|meta| {
        Some(meta.kind.function_name().into_owned().to_string())
    });
    if name.is_some() {
        return name;
    }
    // Recurse through Type wrapper
    if let Type::Type(inner) = type_ {
        return fallback_hover_name_from_type(inner);
    }
    None
}

/// Extract the identifier under the cursor so we can label hover results
/// even when go-to-definition fails.
fn identifier_text_at(
    transaction: &Transaction<'_>,
    handle: &Handle,
    position: TextSize,
) -> Option<String> {
    transaction
        .identifier_at(handle, position)
        .map(|id| id.identifier.id.to_string())
}

pub fn get_hover(
    transaction: &Transaction<'_>,
    handle: &Handle,
    position: TextSize,
    show_go_to_links: bool,
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

    if position_is_in_docstring(transaction, handle, position) {
        return None;
    }

    // Otherwise, fall through to the existing type hover logic
    let type_ = transaction.get_type_at(handle, position)?;
    let fallback_name_from_type = fallback_hover_name_from_type(&type_);
    let (kind, name, docstring_range, module) = if let Some(FindDefinitionItemWithDocstring {
        metadata,
        definition_range: definition_location,
        module,
        docstring_range,
        display_name,
    }) = transaction
        .find_definition(
            handle,
            position,
            FindPreference {
                prefer_pyi: false,
                ..Default::default()
            },
        )
        // TODO: handle more than 1 definition
        .into_iter()
        .next()
    {
        let mut kind = metadata.symbol_kind();
        if matches!(kind, Some(SymbolKind::Attribute)) && type_.is_toplevel_callable() {
            kind = Some(SymbolKind::Method);
        }
        let name = {
            let snippet = module.code_at(definition_location);
            if snippet.chars().any(|c| !c.is_whitespace()) {
                Some(snippet.to_owned())
            } else if let Some(name) = display_name.clone() {
                Some(name)
            } else {
                fallback_name_from_type.clone()
            }
        };
        (kind, name, docstring_range, Some(module))
    } else {
        (None, fallback_name_from_type, None, None)
    };

    let name = name.or_else(|| identifier_text_at(transaction, handle, position));

    let name_for_display = name.clone();
    let type_display = transaction.ad_hoc_solve(handle, {
        let mut cloned = type_.clone();
        move |solver| {
            cloned.visit_toplevel_callable_mut(|c| expand_callable_kwargs_for_hover(&solver, c));
            cloned.as_lsp_string_with_fallback_name(
                name_for_display.as_deref(),
                LspDisplayMode::Hover,
            )
        }
    });

    let docstring = if let (Some(docstring), Some(module)) = (docstring_range, module) {
        Some(Docstring(docstring, module))
    } else {
        None
    };

    let mut parameter_doc = keyword_argument_documentation(transaction, handle, position)
        .and_then(|(name, doc)| (!doc.trim().is_empty()).then_some((name, doc)));

    if parameter_doc.is_none()
        && let Some(FindDefinitionItemWithDocstring {
            metadata: DefinitionMetadata::Variable(Some(SymbolKind::Parameter)),
            definition_range,
            module,
            ..
        }) = transaction
            .find_definition(handle, position, FindPreference::default())
            .into_iter()
            .next()
    {
        let name_str = module.code_at(definition_range);
        let name = Name::new(name_str);
        if let Some(doc) =
            parameter_definition_documentation(transaction, handle, definition_range, &name)
        {
            parameter_doc = Some(doc);
        }
    }

    Some(
        HoverValue {
            kind,
            name,
            type_,
            docstring,
            parameter_doc,
            display: type_display,
            show_go_to_links,
        }
        .format(),
    )
}

fn keyword_argument_documentation(
    transaction: &Transaction<'_>,
    handle: &Handle,
    position: TextSize,
) -> Option<(String, String)> {
    let identifier = transaction.identifier_at(handle, position)?;
    if !matches!(identifier.context, IdentifierContext::KeywordArgument(_)) {
        return None;
    }
    let (_, _, _, callee_range) = transaction.get_callables_from_call(handle, position)?;
    let docs = parameter_documentation_for_callee(transaction, handle, callee_range)?;
    let name = identifier.identifier.id.to_string();
    docs.get(name.as_str()).cloned().map(|doc| (name, doc))
}

fn parameter_definition_documentation(
    transaction: &Transaction<'_>,
    handle: &Handle,
    definition_range: TextRange,
    name: &Name,
) -> Option<(String, String)> {
    let ast = transaction.get_ast(handle)?;
    let module = transaction.get_module_info(handle)?;

    let func = ast
        .body
        .iter()
        .filter_map(|stmt| match stmt {
            ruff_python_ast::Stmt::FunctionDef(func) => Some(func),
            _ => None,
        })
        .find(|func| func.range.contains_inclusive(definition_range.start()))?;

    let doc_range = Docstring::range_from_stmts(func.body.as_slice())?;
    let docs = parse_parameter_documentation(module.code_at(doc_range));
    let key = name.as_str();
    docs.get(key).cloned().map(|doc| (key.to_owned(), doc))
}

fn parameter_documentation_for_callee(
    transaction: &Transaction<'_>,
    handle: &Handle,
    callee_range: TextRange,
) -> Option<HashMap<String, String>> {
    let position = callee_range.start();
    let docstring = transaction
        .find_definition(
            handle,
            position,
            FindPreference {
                prefer_pyi: false,
                ..Default::default()
            },
        )
        .into_iter()
        .find_map(|item| {
            item.docstring_range
                .map(|range| (range, item.module.clone()))
        })
        .or_else(|| {
            transaction
                .find_definition(handle, position, FindPreference::default())
                .into_iter()
                .find_map(|item| {
                    item.docstring_range
                        .map(|range| (range, item.module.clone()))
                })
        })?;
    let (range, module) = docstring;
    let docs = parse_parameter_documentation(module.code_at(range));
    if docs.is_empty() { None } else { Some(docs) }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use pyrefly_python::module::Module;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use pyrefly_types::callable::Callable;
    use pyrefly_types::callable::FuncFlags;
    use pyrefly_types::callable::FuncId;
    use pyrefly_types::callable::FuncMetadata;
    use pyrefly_types::callable::Function;
    use pyrefly_types::callable::FunctionKind;
    use ruff_python_ast::name::Name;

    use super::*;

    fn make_function_type(module_name: &str, func_name: &str) -> Type {
        let module = Module::new(
            ModuleName::from_str(module_name),
            ModulePath::filesystem(PathBuf::from(format!("{module_name}.pyi"))),
            Arc::new(String::new()),
        );
        let metadata = FuncMetadata {
            kind: FunctionKind::Def(Box::new(FuncId {
                module,
                cls: None,
                name: Name::new(func_name),
            })),
            flags: FuncFlags::default(),
        };
        Type::Function(Box::new(Function {
            signature: Callable::ellipsis(Type::None),
            metadata,
        }))
    }

    #[test]
    fn fallback_uses_function_metadata() {
        let ty = make_function_type("numpy", "arange");
        let fallback = fallback_hover_name_from_type(&ty);
        assert_eq!(fallback.as_deref(), Some("arange"));
    }

    #[test]
    fn fallback_recurses_through_type_wrapper() {
        let ty = Type::Type(Box::new(make_function_type("pkg.subpkg", "run")));
        let fallback = fallback_hover_name_from_type(&ty);
        assert_eq!(fallback.as_deref(), Some("run"));
    }
}
