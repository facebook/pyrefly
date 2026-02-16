/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every SPELL

use std::collections::HashMap;

use lsp_types::Hover;
use lsp_types::HoverContents;
use lsp_types::MarkupContent;
use lsp_types::MarkupKind;
use lsp_types::Url;
use pyrefly_build::handle::Handle;
use pyrefly_graph::index::Idx;
use pyrefly_python::ast::Ast;
use pyrefly_python::docstring::Docstring;
use pyrefly_python::docstring::parse_parameter_documentation;
use pyrefly_python::ignore::Ignore;
use pyrefly_python::ignore::Tool;
use pyrefly_python::ignore::find_comment_start_in_line;
use pyrefly_python::module::Module;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_types::callable::Callable;
use pyrefly_types::callable::FunctionKind;
use pyrefly_types::callable::Param;
use pyrefly_types::callable::ParamList;
use pyrefly_types::callable::Params;
use pyrefly_types::callable::Required;
use pyrefly_types::display::LspDisplayMode;
use pyrefly_types::types::Type;
use pyrefly_util::lined_buffer::LineNumber;
use pyrefly_util::visit::Visit;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::ExprContext;
use ruff_python_ast::Stmt;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use starlark_map::small_set::SmallSet;

use crate::alt::answers_solver::AnswersSolver;
use crate::binding::binding::Binding;
use crate::binding::binding::FirstUse;
use crate::binding::binding::Key;
use crate::binding::bindings::Bindings;
use crate::binding::narrow::AtomicNarrowOp;
use crate::binding::narrow::NarrowOp;
use crate::error::error::Error;
use crate::lsp::module_helpers::collect_symbol_def_paths;
use crate::lsp::wasm::signature_help::is_constructor_call;
use crate::lsp::wasm::signature_help::override_constructor_return_type;
use crate::state::lsp::DefinitionMetadata;
use crate::state::lsp::FindDefinitionItemWithDocstring;
use crate::state::lsp::FindPreference;
use crate::state::lsp::IdentifierContext;
use crate::state::state::Transaction;
use crate::state::state::TransactionHandle;

pub struct HoverValue {
    pub kind: Option<SymbolKind>,
    pub name: Option<String>,
    pub type_: Type,
    pub docstring: Option<Docstring>,
    pub parameter_doc: Option<(String, String)>,
    pub type_sources: Vec<String>,
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

    fn resolve_symbol_kind(&self) -> Option<SymbolKind> {
        match self.kind {
            Some(SymbolKind::Attribute) if self.type_.is_toplevel_callable() => self
                .type_
                .visit_toplevel_func_metadata(&|meta| match &meta.kind {
                    FunctionKind::Def(func) if func.cls.is_some() => Some(SymbolKind::Method),
                    _ => Some(SymbolKind::Function),
                })
                .unwrap_or(SymbolKind::Method)
                .into(),
            Some(other) => Some(other),
            None => None,
        }
    }

    pub fn format(&self) -> Hover {
        let docstring_formatted = match self.docstring.as_ref().map(|d| d.resolve()) {
            Some(content) => format!("\n---\n{}", content.trim()),
            None => String::new(),
        };
        let parameter_doc_formatted =
            self.parameter_doc
                .as_ref()
                .map_or(String::new(), |(name, doc)| {
                    let prefix = if self.docstring.is_some() {
                        "\n\n---\n"
                    } else {
                        "\n---\n"
                    };
                    let cleaned = doc.trim().replace('\n', "  \n");
                    format!("{prefix}**Parameter `{}`**\n{}", name, cleaned)
                });
        let kind_formatted = self
            .resolve_symbol_kind()
            .map(|kind| format!("{} ", kind.display_for_hover()))
            .or_else(|| {
                if self.type_.is_toplevel_callable() {
                    Some("(function) ".to_owned())
                } else {
                    None
                }
            })
            .unwrap_or_default();
        let name_formatted = self
            .name
            .as_ref()
            .map(|s| format!("{s}: "))
            .unwrap_or_default();
        let symbol_def_formatted = if self.show_go_to_links {
            HoverValue::format_symbol_def_locations(&self.type_).unwrap_or_default()
        } else {
            String::new()
        };
        let type_source_formatted = if self.type_sources.is_empty() {
            String::new()
        } else {
            let mut section = String::from("\n---\n**Type source**\n");
            for source in &self.type_sources {
                section.push_str("- ");
                section.push_str(source);
                section.push('\n');
            }
            section
        };
        let type_display = self.display.clone().unwrap_or_else(|| {
            self.type_
                .as_lsp_string_with_fallback_name(self.name.as_deref(), LspDisplayMode::Hover)
        });

        Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: format!(
                    "```python\n{}{}{}\n```{}{}{}{}",
                    kind_formatted,
                    name_formatted,
                    type_display,
                    type_source_formatted,
                    docstring_formatted,
                    parameter_doc_formatted,
                    symbol_def_formatted
                ),
            }),
            range: None,
        }
    }
}

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

/// If we can't determine a symbol name via go-to-definition, fall back to what the
/// type metadata knows about the callable. This primarily handles third-party stubs
/// where we only have typeshed information.
fn fallback_hover_name_from_type(type_: &Type) -> Option<String> {
    let name = type_.visit_toplevel_func_metadata(&|meta| Some(meta.kind.function_name()));
    if let Some(name) = name {
        return Some(name.to_string());
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

fn format_type_source_location(module: &Module, range: TextRange) -> String {
    format!("{}", module.display_pos(range.start()))
}

fn format_code_snippet(module: &Module, range: TextRange) -> Option<String> {
    if range.is_empty() {
        return None;
    }
    let snippet = module.code_at(range);
    let cleaned = snippet
        .trim()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace('`', "'");
    if cleaned.is_empty() {
        return None;
    }
    const MAX_LEN: usize = 80;
    let mut truncated = cleaned;
    if truncated.len() > MAX_LEN {
        truncated.truncate(MAX_LEN.saturating_sub(3));
        truncated.push_str("...");
    }
    Some(truncated)
}

fn format_inline_code(snippet: &str) -> String {
    format!("`{snippet}`")
}

fn format_narrow_op(module: &Module, base_name: &Name, op: &NarrowOp) -> Option<String> {
    fn subject_with_facet(
        base_name: &Name,
        facet: Option<&crate::binding::narrow::FacetSubject>,
    ) -> String {
        match facet {
            Some(facet) => format!("{base_name}{}", facet.chain),
            None => base_name.to_string(),
        }
    }

    fn expr_snippet(module: &Module, expr: &ruff_python_ast::Expr) -> Option<String> {
        format_code_snippet(module, expr.range())
    }

    fn args_snippet(module: &Module, args: &ruff_python_ast::Arguments) -> Option<String> {
        format_code_snippet(module, args.range())
    }

    match op {
        NarrowOp::Atomic(facet, atomic) => {
            let subject = subject_with_facet(base_name, facet.as_ref());
            let formatted = match atomic {
                AtomicNarrowOp::Is(expr) => {
                    Some(format!("{subject} is {}", expr_snippet(module, expr)?))
                }
                AtomicNarrowOp::IsNot(expr) => {
                    Some(format!("{subject} is not {}", expr_snippet(module, expr)?))
                }
                AtomicNarrowOp::Eq(expr) => {
                    Some(format!("{subject} == {}", expr_snippet(module, expr)?))
                }
                AtomicNarrowOp::NotEq(expr) => {
                    Some(format!("{subject} != {}", expr_snippet(module, expr)?))
                }
                AtomicNarrowOp::IsInstance(expr, _) => Some(format!(
                    "isinstance({subject}, {})",
                    expr_snippet(module, expr)?
                )),
                AtomicNarrowOp::IsNotInstance(expr, _) => Some(format!(
                    "not isinstance({subject}, {})",
                    expr_snippet(module, expr)?
                )),
                AtomicNarrowOp::IsSubclass(expr) => Some(format!(
                    "issubclass({subject}, {})",
                    expr_snippet(module, expr)?
                )),
                AtomicNarrowOp::IsNotSubclass(expr) => Some(format!(
                    "not issubclass({subject}, {})",
                    expr_snippet(module, expr)?
                )),
                AtomicNarrowOp::TypeEq(expr) => Some(format!(
                    "type({subject}) == {}",
                    expr_snippet(module, expr)?
                )),
                AtomicNarrowOp::TypeNotEq(expr) => Some(format!(
                    "type({subject}) != {}",
                    expr_snippet(module, expr)?
                )),
                AtomicNarrowOp::In(expr) => {
                    Some(format!("{subject} in {}", expr_snippet(module, expr)?))
                }
                AtomicNarrowOp::NotIn(expr) => {
                    Some(format!("{subject} not in {}", expr_snippet(module, expr)?))
                }
                AtomicNarrowOp::HasAttr(attr) => Some(format!("hasattr({subject}, \"{attr}\")")),
                AtomicNarrowOp::NotHasAttr(attr) => {
                    Some(format!("not hasattr({subject}, \"{attr}\")"))
                }
                AtomicNarrowOp::GetAttr(attr, default) => {
                    let default_snippet =
                        default.as_ref().and_then(|expr| expr_snippet(module, expr));
                    Some(match default_snippet {
                        Some(default_snippet) => {
                            format!("getattr({subject}, \"{attr}\", {default_snippet})")
                        }
                        None => format!("getattr({subject}, \"{attr}\")"),
                    })
                }
                AtomicNarrowOp::NotGetAttr(attr, default) => {
                    let default_snippet =
                        default.as_ref().and_then(|expr| expr_snippet(module, expr));
                    Some(match default_snippet {
                        Some(default_snippet) => {
                            format!("not getattr({subject}, \"{attr}\", {default_snippet})")
                        }
                        None => format!("not getattr({subject}, \"{attr}\")"),
                    })
                }
                AtomicNarrowOp::HasKey(key) => Some(format!("\"{key}\" in {subject}")),
                AtomicNarrowOp::NotHasKey(key) => Some(format!("\"{key}\" not in {subject}")),
                AtomicNarrowOp::LenEq(expr) => {
                    Some(format!("len({subject}) == {}", expr_snippet(module, expr)?))
                }
                AtomicNarrowOp::LenNotEq(expr) => {
                    Some(format!("len({subject}) != {}", expr_snippet(module, expr)?))
                }
                AtomicNarrowOp::LenGt(expr) => {
                    Some(format!("len({subject}) > {}", expr_snippet(module, expr)?))
                }
                AtomicNarrowOp::LenGte(expr) => {
                    Some(format!("len({subject}) >= {}", expr_snippet(module, expr)?))
                }
                AtomicNarrowOp::LenLt(expr) => {
                    Some(format!("len({subject}) < {}", expr_snippet(module, expr)?))
                }
                AtomicNarrowOp::LenLte(expr) => {
                    Some(format!("len({subject}) <= {}", expr_snippet(module, expr)?))
                }
                AtomicNarrowOp::IsSequence => Some(format!("isinstance({subject}, Sequence)")),
                AtomicNarrowOp::IsNotSequence => {
                    Some(format!("not isinstance({subject}, Sequence)"))
                }
                AtomicNarrowOp::IsMapping => Some(format!("isinstance({subject}, Mapping)")),
                AtomicNarrowOp::IsNotMapping => Some(format!("not isinstance({subject}, Mapping)")),
                AtomicNarrowOp::Call(expr, arguments) => {
                    let func_snippet = expr_snippet(module, expr)?;
                    let args_snippet = args_snippet(module, arguments).unwrap_or_default();
                    Some(format!("{func_snippet}{args_snippet}"))
                }
                AtomicNarrowOp::NotCall(expr, arguments) => {
                    let func_snippet = expr_snippet(module, expr)?;
                    let args_snippet = args_snippet(module, arguments).unwrap_or_default();
                    Some(format!("not {func_snippet}{args_snippet}"))
                }
                AtomicNarrowOp::IsTruthy => Some(subject),
                AtomicNarrowOp::IsFalsy => Some(format!("not {subject}")),
                AtomicNarrowOp::TypeGuard(_, arguments) => Some(format!(
                    "TypeGuard{}",
                    args_snippet(module, arguments).unwrap_or_default()
                )),
                AtomicNarrowOp::NotTypeGuard(_, arguments) => Some(format!(
                    "not TypeGuard{}",
                    args_snippet(module, arguments).unwrap_or_default()
                )),
                AtomicNarrowOp::TypeIs(_, arguments) => Some(format!(
                    "TypeIs{}",
                    args_snippet(module, arguments).unwrap_or_default()
                )),
                AtomicNarrowOp::NotTypeIs(_, arguments) => Some(format!(
                    "not TypeIs{}",
                    args_snippet(module, arguments).unwrap_or_default()
                )),
                AtomicNarrowOp::Placeholder => None,
            };
            formatted
        }
        NarrowOp::And(ops) => {
            let parts = ops
                .iter()
                .filter_map(|op| format_narrow_op(module, base_name, op))
                .collect::<Vec<_>>();
            if parts.is_empty() {
                None
            } else {
                Some(parts.join(" and "))
            }
        }
        NarrowOp::Or(ops) => {
            let parts = ops
                .iter()
                .filter_map(|op| format_narrow_op(module, base_name, op))
                .collect::<Vec<_>>();
            if parts.is_empty() {
                None
            } else {
                Some(parts.join(" or "))
            }
        }
    }
}

fn narrow_source_for_key(
    bindings: &Bindings,
    module: &Module,
    start_idx: Idx<Key>,
) -> Option<String> {
    let mut current = start_idx;
    let mut seen = SmallSet::new();
    loop {
        if seen.contains(&current) {
            return None;
        }
        seen.insert(current);
        match bindings.get(current) {
            Binding::Forward(next) => current = *next,
            Binding::Narrow(_, op, _) => {
                let key = bindings.idx_to_key(current);
                let Key::Narrow(name, op_range, _) = key else {
                    return None;
                };
                let location = format_type_source_location(module, *op_range);
                let mut msg = format!("Narrowed by condition at {location}");
                if let Some(snippet) = format_narrow_op(module, name, op)
                    .or_else(|| format_code_snippet(module, *op_range))
                {
                    msg.push_str(&format!(": {}", format_inline_code(&snippet)));
                }
                return Some(msg);
            }
            _ => return None,
        }
    }
}

fn definition_short_identifier(bindings: &Bindings, key: &Key) -> Option<ShortIdentifier> {
    let mut current = bindings.key_to_idx(key);
    let mut seen = SmallSet::new();
    loop {
        if seen.contains(&current) {
            return None;
        }
        seen.insert(current);
        let current_key = bindings.idx_to_key(current);
        if let Key::Definition(short_identifier) = current_key {
            return Some(short_identifier.clone());
        }
        match bindings.get(current) {
            Binding::Forward(next)
            | Binding::Narrow(next, ..)
            | Binding::CompletedPartialType(next, ..)
            | Binding::PartialTypeWithUpstreamsCompleted(next, ..)
            | Binding::LoopPhi(next, ..) => current = *next,
            Binding::Phi(_, branches) if !branches.is_empty() => {
                current = branches[0].value_key;
            }
            _ => return None,
        }
    }
}

fn first_use_source_for_key(bindings: &Bindings, module: &Module, key: &Key) -> Option<String> {
    let Some(def_identifier) = definition_short_identifier(bindings, key) else {
        return None;
    };
    let key = Key::CompletedPartialType(def_identifier);
    if !bindings.is_valid_key(&key) {
        return None;
    }
    let idx = bindings.key_to_idx(&key);
    match bindings.get(idx) {
        Binding::CompletedPartialType(_, FirstUse::UsedBy(use_idx)) => {
            let use_range = bindings.idx_to_key(*use_idx).range();
            let location = format_type_source_location(module, use_range);
            let mut msg = format!("Inferred from first use at {location}");
            if let Some(snippet) = format_code_snippet(module, use_range) {
                msg.push_str(&format!(": {}", format_inline_code(&snippet)));
            }
            Some(msg)
        }
        _ => None,
    }
}

fn type_sources_for_hover(
    transaction: &Transaction<'_>,
    handle: &Handle,
    position: TextSize,
) -> Vec<String> {
    let Some(bindings) = transaction.get_bindings(handle) else {
        return Vec::new();
    };
    let Some(module) = transaction.get_module_info(handle) else {
        return Vec::new();
    };
    let Some(identifier_with_context) = transaction.identifier_at(handle, position) else {
        return Vec::new();
    };
    let key = match identifier_with_context.context {
        IdentifierContext::Expr(expr_context) => match expr_context {
            ExprContext::Store => {
                Key::Definition(ShortIdentifier::new(&identifier_with_context.identifier))
            }
            ExprContext::Load | ExprContext::Del | ExprContext::Invalid => {
                Key::BoundName(ShortIdentifier::new(&identifier_with_context.identifier))
            }
        },
        _ => return Vec::new(),
    };
    if !bindings.is_valid_key(&key) {
        return Vec::new();
    }
    let idx = bindings.key_to_idx(&key);
    let mut sources = Vec::new();
    if let Some(narrow_source) = narrow_source_for_key(&bindings, &module, idx) {
        sources.push(narrow_source);
    }
    if let Some(first_use_source) = first_use_source_for_key(&bindings, &module, &key) {
        sources.push(first_use_source);
    }
    sources
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

/// Check if the cursor position is on the `in` keyword within a for loop or comprehension.
/// Returns Some(iterable_range) if found, None otherwise.
fn in_keyword_in_iteration_at(
    transaction: &Transaction<'_>,
    handle: &Handle,
    position: TextSize,
) -> Option<TextRange> {
    let ast = transaction.get_ast(handle)?;

    for node in Ast::locate_node(&ast, position) {
        // Extract target end and iter range from for statements and comprehensions.
        // In valid Python syntax, the region between target and iter contains only
        // whitespace and the `in` keyword, so a position check is sufficient.
        let (target_end, iter_range) = match node {
            AnyNodeRef::StmtFor(s) => (s.target.range().end(), s.iter.range()),
            AnyNodeRef::Comprehension(c) => (c.target.range().end(), c.iter.range()),
            _ => continue,
        };
        if position >= target_end && position < iter_range.start() {
            return Some(iter_range);
        }
    }
    None
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

    // Check if hovering over `in` keyword in for loop or comprehension. These `in`s are different
    // from using `in` as a binary comparison operator and therefore needs some special handling.
    if let Some(iterable_range) = in_keyword_in_iteration_at(transaction, handle, position)
        && let Some(iterable_type) = transaction.get_type_at(handle, iterable_range.start())
    {
        return Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: format!(
                    "```python\n(keyword) in\n```\n---\nIteration over `{}`",
                    iterable_type
                ),
            }),
            range: None,
        });
    }

    // Otherwise, fall through to the existing type hover logic
    let mut type_ = transaction.get_type_at(handle, position)?;

    // Helper function to check if we're hovering over a callee and get its range
    let find_callee_range_at_position = || -> Option<TextRange> {
        use ruff_python_ast::Expr;
        let mod_module = transaction.get_ast(handle)?;
        let mut result = None;
        mod_module.visit(&mut |expr: &Expr| {
            if let Expr::Call(call) = expr {
                // Check if position is within the callee (func) range
                if call.func.range().contains(position) {
                    result = Some(call.func.range());
                }
            }
        });
        result
    };

    // Check both: hovering in arguments area OR hovering over the callee itself
    let callee_range_opt = transaction
        .get_callables_from_call(handle, position)
        .map(|(_, _, _, range)| range)
        .or_else(find_callee_range_at_position);

    if let Some(callee_range) = callee_range_opt {
        let is_constructor = transaction
            .get_answers(handle)
            .and_then(|ans| ans.get_type_trace(callee_range))
            .is_some_and(is_constructor_call);
        if is_constructor && let Some(new_type) = override_constructor_return_type(type_.clone()) {
            type_ = new_type;
        }
    }

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
        let kind = metadata.symbol_kind();
        let name = {
            let snippet = module.code_at(definition_location);
            if snippet.chars().any(|c| !c.is_whitespace()) {
                Some(snippet.to_owned())
            } else if let Some(name) = display_name.clone() {
                Some(name)
            } else {
                fallback_name_from_type
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
            type_sources: type_sources_for_hover(transaction, handle, position),
            display: type_display,
            show_go_to_links,
        }
        .format(),
    )
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
    use pyrefly_types::heap::TypeHeap;
    use ruff_python_ast::name::Name;

    use super::*;

    fn make_function_type(heap: &TypeHeap, module_name: &str, func_name: &str) -> Type {
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
        heap.mk_function(Function {
            signature: Callable::ellipsis(heap.mk_none()),
            metadata,
        })
    }

    #[test]
    fn fallback_uses_function_metadata() {
        let heap = TypeHeap::new();
        let ty = make_function_type(&heap, "numpy", "arange");
        let fallback = fallback_hover_name_from_type(&ty);
        assert_eq!(fallback.as_deref(), Some("arange"));
    }

    #[test]
    fn fallback_recurses_through_type_wrapper() {
        let heap = TypeHeap::new();
        let ty = heap.mk_type(make_function_type(&heap, "pkg.subpkg", "run"));
        let fallback = fallback_hover_name_from_type(&ty);
        assert_eq!(fallback.as_deref(), Some("run"));
    }
}
