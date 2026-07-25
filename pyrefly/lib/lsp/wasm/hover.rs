/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every SPELL

use std::collections::HashMap;
use std::sync::LazyLock;

use lsp_types::Hover;
use lsp_types::HoverContents;
use lsp_types::MarkupContent;
use lsp_types::MarkupKind;
use lsp_types::Range;
use lsp_types::Url;
use pyrefly_build::handle::Handle;
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
use pyrefly_types::class::Class;
use pyrefly_types::class::ClassType;
use pyrefly_types::display::LspDisplayMode;
use pyrefly_types::type_var::Variance;
use pyrefly_types::types::Type;
use pyrefly_util::absolutize::Absolutize as _;
use pyrefly_util::lined_buffer::LineNumber;
use pyrefly_util::visit::Visit;
use regex::Regex;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use vec1::Vec1;

use crate::alt::answers_solver::AnswersSolver;
use crate::binding::binding::Key;
use crate::error::error::Error;
use crate::lsp::module_helpers::collect_symbol_def_paths;
use crate::lsp::module_helpers::to_real_path;
use crate::lsp::wasm::signature_help::CallInfo;
use crate::lsp::wasm::signature_help::is_constructor_call;
use crate::lsp::wasm::signature_help::override_constructor_return_type;
use crate::lsp::wasm::type_source::set_display_pos_fragment;
use crate::lsp::wasm::type_source::type_sources_for_hover;
use crate::state::lsp::DefinitionMetadata;
use crate::state::lsp::FindDefinitionItemWithDocstring;
use crate::state::lsp::FindPreference;
use crate::state::lsp::IdentifierContext;
use crate::state::lsp::IdentifierWithContext;
use crate::state::lsp::attribute_symbol_kind_from_type;
use crate::state::state::Transaction;
use crate::state::state::TransactionHandle;

/// Matches Sphinx cross-references like `:meth:`target``, `:class:`MyClass``, etc.
/// The role name is captured but ignored — all roles resolve uniformly.
static SPHINX_REFERENCE_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r":([a-zA-Z0-9_-]+):`([^`]+)`").expect("invalid regex"));

pub struct HoverValue {
    pub kind: Option<SymbolKind>,
    pub name: Option<String>,
    pub type_: Type,
    pub range: Option<Range>,
    pub docstring: Option<Docstring>,
    pub parameter_doc: Option<(String, String)>,
    pub type_sources: Vec<String>,
    pub display: Option<String>,
    pub show_go_to_links: bool,
}

/// Hover contents and whether another verbosity request can reveal more detail.
///
/// Hover verbosity currently has two states: named nested unions or fully expanded unions.
pub struct HoverResult {
    pub hover: Hover,
    pub can_increase_verbosity: bool,
}

/// Display knobs for a hover request. `verbosity_level > 0` expands named nested unions.
#[derive(Debug, Clone, Copy)]
pub struct HoverOptions {
    pub show_go_to_links: bool,
    pub verbosity_level: usize,
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
                    set_display_pos_fragment(&mut url, start_pos);
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
            Some(SymbolKind::Attribute) => Some(attribute_symbol_kind_from_type(&self.type_)),
            Some(other) => Some(other),
            None => None,
        }
    }

    /// Replace `:role:`target`` patterns in docstring text with markdown links or inline code.
    /// On non-WASM, attempts to resolve targets as same-class attributes to clickable links.
    /// On WASM, all targets become inline code since file:// URLs aren't supported.
    fn resolve_sphinx_references(
        text: String,
        transaction: &Transaction,
        handle: &Handle,
        context_type: &Type,
    ) -> String {
        SPHINX_REFERENCE_PATTERN
            .replace_all(&text, |caps: &regex::Captures| {
                let target = &caps[2];
                Self::format_sphinx_target(target, transaction, handle, context_type)
            })
            .into_owned()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn format_sphinx_target(
        target: &str,
        transaction: &Transaction,
        handle: &Handle,
        context_type: &Type,
    ) -> String {
        Self::try_resolve_sphinx_target(target, transaction, handle, context_type)
            .map(|url| format!("[{target}]({url})"))
            .unwrap_or_else(|| format!("`{target}`"))
    }

    #[cfg(target_arch = "wasm32")]
    fn format_sphinx_target(
        target: &str,
        _transaction: &Transaction,
        _handle: &Handle,
        _context_type: &Type,
    ) -> String {
        format!("`{target}`")
    }

    /// Resolve a Sphinx target to a file URL by looking up the attribute on the
    /// enclosing class. Only supports unqualified same-class references.
    #[cfg(not(target_arch = "wasm32"))]
    fn try_resolve_sphinx_target(
        target: &str,
        transaction: &Transaction,
        handle: &Handle,
        context_type: &Type,
    ) -> Option<String> {
        if target.contains('.') {
            return None;
        }

        // For methods, search in parent class; for constructors, use the return type
        let search_type = context_type
            .visit_toplevel_func_metadata(&|meta| {
                if let FunctionKind::Def(func) = &meta.kind
                    && let Some(class) = &func.cls
                {
                    Some(Type::ClassType(ClassType::new(
                        class.clone(),
                        Default::default(),
                    )))
                } else {
                    None
                }
            })
            .or_else(|| {
                if let Type::Callable(callable) = context_type
                    && let Type::ClassType(_) = callable.ret
                {
                    Some(callable.ret.clone())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| context_type.clone());

        let defs = transaction
            .find_attribute_definition_for_base_type(
                handle,
                FindPreference::default(),
                search_type,
                &Name::new(target),
            )
            .ok()?;

        let def = defs.into_vec().into_iter().next()?;
        let file_path = to_real_path(def.module.path())
            .unwrap_or_else(|| def.module.path().as_path().to_path_buf());
        let abs_path = file_path.absolutize();
        let mut url = Url::from_file_path(&abs_path).ok()?;
        set_display_pos_fragment(
            &mut url,
            def.module.display_range(def.definition_range).start,
        );
        Some(url.to_string())
    }

    pub fn format(&self, transaction: &Transaction, handle: &Handle) -> Hover {
        let docstring_formatted = match &self.docstring {
            Some(docstring) => {
                let content = docstring.resolve();
                let resolved_content =
                    Self::resolve_sphinx_references(content, transaction, handle, &self.type_);
                format!("\n---\n{}", resolved_content.trim())
            }
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
            range: self.range,
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
            // Check both this kind's name and any parent kind's name,
            // so that e.g. `ignore[bad-override]` shows suppressed
            // `bad-override-mutable-attribute` errors on hover.
            error.error_kind().suppression_names().any(|name| {
                ignore.is_ignored_by_suppression_line(
                    suppression_line,
                    range.start.line_within_file(),
                    range.end.line_within_file(),
                    name,
                    &Tool::default_enabled(),
                )
            })
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

fn position_is_in_docstring(ast: Option<&ModModule>, position: TextSize) -> bool {
    let Some(ast) = ast else {
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
                Stmt::FunctionDef(func)
                    if body_contains_docstring(func.body.as_slice(), position) =>
                {
                    return true;
                }
                Stmt::ClassDef(class_def)
                    if body_contains_docstring(class_def.body.as_slice(), position) =>
                {
                    return true;
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

/// Given the position, if it corresponds to a class-scoped PEP695 type var declaration
/// return the class. This only applies to the declaration of the type var, not to
/// any usages.
fn get_owner_class_of_pep695_type_parameter_at(
    id: &IdentifierWithContext,
    transaction: &Transaction<'_>,
    handle: &Handle,
    position: TextSize,
) -> Option<Class> {
    if !matches!(id.context, IdentifierContext::TypeParameter) {
        return None;
    }
    let module = transaction.get_ast(handle)?;
    let owner = Ast::locate_node(&module, position)
        .into_iter()
        .rev()
        .find_map(|node| match node {
            AnyNodeRef::StmtClassDef(class_def)
                if class_def
                    .type_params
                    .as_ref()
                    .is_some_and(|type_params| type_params.range().contains(position)) =>
            {
                Some(class_def.name.clone())
            }
            _ => None,
        });
    let key = Key::Definition(ShortIdentifier::new(&owner?));
    match transaction.get_type_for_display(handle, &key)? {
        Type::ClassDef(class) => Some(class),
        _ => None,
    }
}

fn display_variance_for_hover(variance: Variance) -> &'static str {
    match variance {
        Variance::Covariant => "covariant",
        Variance::Contravariant => "contravariant",
        // Bivariant is an internal result, it's treated as invariant.
        Variance::Invariant | Variance::Bivariant => "invariant",
    }
}

fn type_parameter_hover_display(
    solver: &AnswersSolver<TransactionHandle<'_>>,
    type_: &Type,
    owner: &Class,
) -> Option<String> {
    let quantified = match type_ {
        Type::Quantified(q) | Type::QuantifiedValue(q) => q,
        _ => return None,
    };
    let variance = solver.type_order().get_variance_from_class(owner);
    Some(format!(
        "{}@{} ({})",
        quantified.name(),
        owner.name(),
        display_variance_for_hover(variance.get(quantified.name()))
    ))
}

fn class_display_type(solver: &AnswersSolver<TransactionHandle<'_>>, type_: &Type) -> Option<Type> {
    let enum_class = match type_ {
        Type::ClassDef(cls) => Some(cls),
        Type::ClassType(cls) => Some(cls.class_object()),
        Type::Type(t) => match &**t {
            Type::ClassType(cls) => Some(cls.class_object()),
            _ => None,
        },
        _ => None,
    };
    if let Some(cls) = enum_class
        && solver.get_metadata_for_class(cls).is_enum()
    {
        let members: Vec<Type> = solver
            .get_enum_members(cls)
            .into_iter()
            .map(|lit| lit.to_implicit_type())
            .collect();
        return Some(if members.is_empty() {
            type_.clone()
        } else {
            solver.heap.mk_union(members)
        });
    }

    let mut constructor = match type_ {
        Type::ClassDef(cls) if !solver.get_metadata_for_class(cls).is_typed_dict() => Some(
            solver
                .type_order()
                .constructor_to_callable(&solver.promote_nontypeddict_silently_to_classtype(cls)),
        ),
        Type::Type(t) => match &**t {
            Type::ClassType(cls) => Some(solver.type_order().constructor_to_callable(cls)),
            _ => None,
        },
        _ => None,
    }?;
    constructor.transform_toplevel_callable(|c| expand_callable_kwargs_for_hover(solver, c));
    Some(solver.for_display(constructor))
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
        .map(Vec1::into_vec)
        .unwrap_or_default()
        .into_iter()
        .find_map(|item| {
            item.docstring_range
                .map(|range| (range, item.module.clone()))
        })
        .or_else(|| {
            transaction
                .find_definition(handle, position, FindPreference::default())
                .map(Vec1::into_vec)
                .unwrap_or_default()
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
    let CallInfo { callee_range, .. } = transaction.get_callables_from_call(handle, position)?;
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
fn in_keyword_in_iteration_at(ast: Option<&ModModule>, position: TextSize) -> Option<TextRange> {
    let ast = ast?;

    for node in Ast::locate_node(ast, position) {
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

/// Hover contents when the cursor is on an ignore comment covering suppressed errors.
fn ignore_comment_hover(
    transaction: &Transaction<'_>,
    handle: &Handle,
    module: Option<&Module>,
    position: TextSize,
) -> Option<HoverResult> {
    let module = module?;
    let display_pos = module.display_pos(position);
    let line_text = module.lined_buffer().content_in_line_range(
        display_pos.line_within_file(),
        display_pos.line_within_file(),
    );
    let comment_offset = find_comment_start_in_line(line_text)?;
    if display_pos.column().get() < comment_offset as u32 {
        return None;
    }
    // A comment on its own line suppresses errors on the next line; otherwise this line.
    let suppression_line = if line_text.trim().starts_with("#") {
        display_pos.line_within_file().increment()
    } else {
        display_pos.line_within_file()
    };
    module.ignore().get(&suppression_line)?;
    let suppressed_errors =
        get_suppressed_errors_for_line(transaction, handle, suppression_line, module.ignore());
    Some(HoverResult {
        hover: format_suppressed_errors_hover(suppressed_errors),
        can_increase_verbosity: false,
    })
}

/// Hover contents for the `in` keyword of a for-loop or comprehension, which is distinct
/// from `in` used as a binary comparison operator and needs special handling.
fn in_keyword_hover(
    transaction: &Transaction<'_>,
    handle: &Handle,
    ast: Option<&ModModule>,
    position: TextSize,
) -> Option<HoverResult> {
    let iterable_range = in_keyword_in_iteration_at(ast, position)?;
    let iterable_type = transaction.get_type_at_for_display(handle, iterable_range.start())?;
    Some(HoverResult {
        hover: Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: format!(
                    "```python\n(keyword) in\n```\n---\nIteration over `{iterable_type}`"
                ),
            }),
            range: None,
        },
        can_increase_verbosity: false,
    })
}

/// Resolve the type under the cursor, coercing it to a callable when hovering over a
/// call's callee (or to the constructor's return type for a constructor call).
fn resolve_hovered_type(
    transaction: &Transaction<'_>,
    handle: &Handle,
    ast: Option<&ModModule>,
    position: TextSize,
) -> Option<Type> {
    let mut type_ = transaction
        .subscript_operator_type_at(handle, position)
        .or_else(|| transaction.get_type_at_for_display(handle, position))
        .or_else(|| transaction.operator_type_at(handle, position))?;

    // Find the innermost call whose callee (func) encloses the cursor, returning the
    // callee's range and whether the cursor is on the callee's own name — the attribute
    // in `a.b()`, or the whole callee otherwise. A receiver like `a` in `a.b()` is inside
    // the callee range but not on the name, so hovering it must not coerce its type.
    let callee_at_position = || -> Option<(TextRange, bool)> {
        use ruff_python_ast::Expr;
        let ast = ast?;
        let mut result = None;
        ast.visit(&mut |expr: &Expr| {
            if let Expr::Call(call) = expr
                && call.func.range().contains(position)
            {
                let on_callee_name = match &*call.func {
                    Expr::Attribute(attr) => attr.attr.range(),
                    _ => call.func.range(),
                }
                .contains(position);
                result = Some((call.func.range(), on_callee_name));
            }
        });
        result
    };

    // Prefer the enclosing call found from the argument list; only walk the AST for a
    // callee hover when the cursor is not inside an argument. Hovering inside arguments
    // is never "on the callee", so coercion stays disabled there.
    let (callee_range_opt, hovering_over_callee) =
        match transaction.get_callables_from_call(handle, position) {
            Some(info) => (Some(info.callee_range), false),
            None => match callee_at_position() {
                Some((range, on_name)) => (Some(range), on_name),
                None => (None, false),
            },
        };

    if let Some(callee_range) = callee_range_opt {
        let is_constructor = transaction
            .get_answers(handle)
            .and_then(|ans| ans.get_type_trace(callee_range))
            .is_some_and(is_constructor_call);
        if is_constructor && let Some(new_type) = override_constructor_return_type(type_.clone()) {
            type_ = new_type;
        } else if hovering_over_callee {
            type_ = transaction.coerce_type_to_callable(handle, type_);
        }
    }
    Some(type_)
}

/// Resolve parameter documentation for the symbol under the cursor: keyword-argument docs
/// at a call site, otherwise the docstring-derived docs for a parameter definition.
fn resolve_hover_parameter_doc(
    transaction: &Transaction<'_>,
    handle: &Handle,
    position: TextSize,
) -> Option<(String, String)> {
    if let Some(doc) = keyword_argument_documentation(transaction, handle, position)
        .and_then(|(name, doc)| (!doc.trim().is_empty()).then_some((name, doc)))
    {
        return Some(doc);
    }

    if let Some(FindDefinitionItemWithDocstring {
        metadata: DefinitionMetadata::Variable(Some(SymbolKind::Parameter)),
        definition_range,
        module,
        ..
    }) = transaction
        .find_definition(handle, position, FindPreference::default())
        .map(Vec1::into_vec)
        .unwrap_or_default()
        .into_iter()
        .next()
    {
        let name = Name::new(module.code_at(definition_range));
        return parameter_definition_documentation(transaction, handle, definition_range, &name);
    }

    None
}

pub fn get_hover(
    transaction: &Transaction<'_>,
    handle: &Handle,
    position: TextSize,
    show_go_to_links: bool,
) -> Option<Hover> {
    get_hover_with_verbosity(
        transaction,
        handle,
        position,
        HoverOptions {
            show_go_to_links,
            verbosity_level: 0,
        },
    )
    .map(|result| result.hover)
}

/// Build hover contents and report whether the compact type display can be expanded.
pub fn get_hover_with_verbosity(
    transaction: &Transaction<'_>,
    handle: &Handle,
    position: TextSize,
    options: HoverOptions,
) -> Option<HoverResult> {
    let module_info = transaction.get_module_info(handle);

    if let Some(result) = ignore_comment_hover(transaction, handle, module_info.as_ref(), position)
    {
        return Some(result);
    }

    let ast = transaction.get_ast(handle);

    if position_is_in_docstring(ast.as_deref(), position) {
        return None;
    }

    if let Some(result) = in_keyword_hover(transaction, handle, ast.as_deref(), position) {
        return Some(result);
    }

    let type_ = resolve_hovered_type(transaction, handle, ast.as_deref(), position)?;

    // `a and b and c` is a single flat BoolOp, so hovering any operator in the
    // chain highlights the whole expression.
    let range = ast
        .as_deref()
        .zip(module_info.as_ref())
        .and_then(|(ast, module_info)| {
            Ast::locate_node(ast, position)
                .into_iter()
                .find(|node| node.as_expr_ref().is_some())
                .and_then(|node| match node {
                    AnyNodeRef::ExprBoolOp(bool_op) => {
                        Some(module_info.to_lsp_range(bool_op.range()))
                    }
                    _ => None,
                })
        });

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
        .map(Vec1::into_vec)
        .unwrap_or_default()
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
    let hover_identifier = transaction.identifier_at(handle, position);
    let type_parameter_owner_class = hover_identifier.as_ref().and_then(|id| {
        get_owner_class_of_pep695_type_parameter_at(id, transaction, handle, position)
    });
    let show_constructor = kind == Some(SymbolKind::Class)
        && hover_identifier
            .as_ref()
            .is_some_and(|id| !matches!(id.context, IdentifierContext::ClassDef { .. }));
    let (type_display, can_increase_verbosity) =
        match transaction.ad_hoc_solve(handle, "hover_display", {
            let mut cloned = type_.clone();
            move |solver| {
                let unions_expanded = options.verbosity_level > 0;
                if let Some(owner) = &type_parameter_owner_class
                    && let Some(display) = type_parameter_hover_display(&solver, &cloned, owner)
                {
                    // Type parameter displays (e.g. `T@Foo (covariant)`) never contain unions.
                    return (display, false);
                }
                // Named unions can be surfaced by the class/kwargs transforms, so pick the
                // type we're about to render first.
                let display_type = if show_constructor
                    && let Some(class_type) = class_display_type(&solver, &cloned)
                {
                    class_type
                } else {
                    cloned.transform_toplevel_callable(|c| {
                        expand_callable_kwargs_for_hover(&solver, c)
                    });
                    cloned
                };
                let render = |expand| {
                    display_type.as_lsp_string_with_fallback_name_and_expanded_unions(
                        name_for_display.as_deref(),
                        LspDisplayMode::Hover,
                        expand,
                    )
                };
                let rendered = render(unions_expanded);
                // Offer "+" only when expanding actually changes the rendered type. Deriving
                // this from the renderer itself (rather than a structural type walk) means the
                // affordance can never advertise an expansion that produces identical output.
                let can_increase_verbosity = !unions_expanded && rendered != render(true);
                (rendered, can_increase_verbosity)
            }
        }) {
            Some((display, can_increase)) => (Some(display), can_increase),
            None => (None, false),
        };

    let docstring = if let (Some(docstring), Some(module)) = (docstring_range, module) {
        Some(Docstring(docstring, module))
    } else {
        None
    };

    let parameter_doc = resolve_hover_parameter_doc(transaction, handle, position);

    Some(HoverResult {
        hover: HoverValue {
            kind,
            name,
            type_,
            range,
            docstring,
            parameter_doc,
            type_sources: type_sources_for_hover(transaction, handle, position),
            display: type_display,
            show_go_to_links: options.show_go_to_links,
        }
        .format(transaction, handle),
        can_increase_verbosity,
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use pyrefly_python::module::Module;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use pyrefly_types::callable::Callable;
    use pyrefly_types::callable::FuncMetadata;
    use pyrefly_types::callable::Function;
    use pyrefly_types::heap::TypeHeap;
    use ruff_python_ast::name::Name;

    use super::*;

    fn make_function_type(heap: &TypeHeap, module_name: &str, func_name: &str) -> Type {
        let module = Module::new(
            ModuleName::from_str(module_name),
            ModulePath::filesystem(PathBuf::from(format!("{module_name}.pyi"))),
            Arc::new(String::new()),
        );
        let metadata = FuncMetadata::def(&module, None, Name::new(func_name));
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
