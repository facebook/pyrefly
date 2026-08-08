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
use ruff_python_ast::Identifier;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::name::Name;
use ruff_python_ast::token::Token;
use ruff_python_ast::token::TokenKind;
use ruff_python_ast::token::Tokens;
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
                let symbol = meta.kind.to_func_symbol()?;
                let class = symbol.cls.as_ref()?;
                Some(Type::ClassType(ClassType::new(
                    class.clone(),
                    Default::default(),
                )))
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

fn simple_python_string_literal_contents(snippet: &str) -> Option<&str> {
    let mut rest = snippet.trim();
    while let Some(ch) = rest.chars().next()
        && matches!(ch, 'b' | 'B' | 'f' | 'F' | 'r' | 'R' | 'u' | 'U')
    {
        rest = &rest[ch.len_utf8()..];
    }
    let quote = if rest.starts_with("'''") {
        "'''"
    } else if rest.starts_with("\"\"\"") {
        "\"\"\""
    } else if rest.starts_with('\'') {
        "'"
    } else if rest.starts_with('"') {
        "\""
    } else {
        return None;
    };
    if rest.len() < quote.len() * 2 || !rest.ends_with(quote) {
        return None;
    }
    Some(&rest[quote.len()..rest.len() - quote.len()])
}

fn is_ascii_identifier_like(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn hover_name_from_definition_snippet(
    snippet: &str,
    display_name: Option<&str>,
    fallback_name: Option<String>,
) -> Option<String> {
    let snippet = snippet.trim();
    let fallback_name = display_name.map(ToOwned::to_owned).or(fallback_name);
    if snippet.is_empty() {
        return fallback_name;
    }
    if let Some(literal_contents) = simple_python_string_literal_contents(snippet) {
        return fallback_name.or_else(|| {
            if is_ascii_identifier_like(literal_contents) {
                Some(literal_contents.to_owned())
            } else {
                Some(snippet.to_owned())
            }
        });
    }
    Some(snippet.to_owned())
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
    let typed_dict = ty.unpacked_typed_dict()?;
    let fields = solver.type_order().typed_dict_kw_param_info(typed_dict);
    if fields.is_empty() {
        None
    } else {
        Some(fields)
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

fn keyword_argument_identifier(
    transaction: &Transaction<'_>,
    handle: &Handle,
    position: TextSize,
) -> Option<Identifier> {
    let identifier = transaction.identifier_at(handle, position)?;
    matches!(identifier.context, IdentifierContext::KeywordArgument(_))
        .then_some(identifier.identifier)
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

fn keyword_token_at(
    transaction: &Transaction<'_>,
    handle: &Handle,
    module: &Module,
    position: TextSize,
) -> Option<Token> {
    fn find(tokens: &Tokens, position: TextSize) -> Option<Token> {
        tokens
            .at_offset(position)
            .find(|token| token.kind().is_keyword())
    }

    let parsed = transaction.get_parsed_module(handle)?;
    if let Some(tokens) = parsed.tokens() {
        find(&tokens, position)
    } else {
        let (parsed, _, _) = Ast::parse_with_version(
            module.contents(),
            handle.sys_info().version(),
            module.source_type(),
        );
        find(parsed.tokens(), position)
    }
}

/// Soft keywords remain valid identifiers outside their specific grammar constructs.
fn soft_keyword_is_syntax(token: Token, ast: &ModModule) -> bool {
    let position = token.range().start();
    Ast::locate_node(ast, position)
        .into_iter()
        .any(|node| match (token.kind(), node) {
            (TokenKind::Match, AnyNodeRef::StmtMatch(stmt)) => {
                position < stmt.subject.range().start()
            }
            (TokenKind::Case, AnyNodeRef::MatchCase(case)) => {
                position < case.pattern.range().start()
            }
            (TokenKind::Lazy, AnyNodeRef::StmtImport(stmt)) => {
                stmt.is_lazy && position == stmt.range().start()
            }
            (TokenKind::Lazy, AnyNodeRef::StmtImportFrom(stmt)) => {
                stmt.is_lazy && position == stmt.range().start()
            }
            (TokenKind::Type, AnyNodeRef::StmtTypeAlias(stmt)) => {
                position < stmt.name.range().start()
            }
            _ => false,
        })
}

fn keyword_documentation(
    token: Token,
    module: &Module,
    ast: &ModModule,
) -> (&'static str, &'static str) {
    let compound = "https://docs.python.org/3/reference/compound_stmts.html";
    let position = token.range().start();
    let nodes = Ast::locate_node(ast, position);
    let source = module.contents();
    let before = source
        .get(..usize::from(token.range().start()))
        .expect("token range is within the module")
        .trim_end();
    let after = source
        .get(usize::from(token.range().end())..)
        .expect("token range is within the module")
        .trim_start();
    let preceded_by = |keyword: &str| {
        before.strip_suffix(keyword).is_some_and(|prefix| {
            prefix
                .chars()
                .next_back()
                .is_none_or(|c| !c.is_alphanumeric() && c != '_')
        })
    };
    let followed_by = |keyword: &str| {
        after.strip_prefix(keyword).is_some_and(|suffix| {
            suffix
                .chars()
                .next()
                .is_none_or(|c| !c.is_alphanumeric() && c != '_')
        })
    };
    match token.kind() {
        TokenKind::False => (
            "The false value of the `bool` type.",
            "https://docs.python.org/3/reference/expressions.html#literals",
        ),
        TokenKind::None => (
            "The singleton value used to represent the absence of a value.",
            "https://docs.python.org/3/reference/expressions.html#literals",
        ),
        TokenKind::True => (
            "The true value of the `bool` type.",
            "https://docs.python.org/3/reference/expressions.html#literals",
        ),
        TokenKind::And => (
            "Evaluates the right operand only when the left operand is true, then returns one of the operands.",
            "https://docs.python.org/3/reference/expressions.html#boolean-operations",
        ),
        TokenKind::Or => (
            "Evaluates the right operand only when the left operand is false, then returns one of the operands.",
            "https://docs.python.org/3/reference/expressions.html#boolean-operations",
        ),
        TokenKind::Not if followed_by("in") => (
            "Negates a membership test, producing `True` when the left operand is not a member of the right operand.",
            "https://docs.python.org/3/reference/expressions.html#membership-test-operations",
        ),
        TokenKind::Not if preceded_by("is") => (
            "Completes the `is not` operator, which tests whether two references point to different objects.",
            "https://docs.python.org/3/reference/expressions.html#is-not",
        ),
        TokenKind::Not => (
            "Negates the truth value of its operand.",
            "https://docs.python.org/3/reference/expressions.html#boolean-operations",
        ),
        TokenKind::In if preceded_by("not") => (
            "Tests whether the left operand is not a member of the right operand.",
            "https://docs.python.org/3/reference/expressions.html#membership-test-operations",
        ),
        TokenKind::In => (
            "Tests whether the left operand is a member of the right operand.",
            "https://docs.python.org/3/reference/expressions.html#membership-test-operations",
        ),
        TokenKind::Is if followed_by("not") => (
            "Tests whether two references point to different objects.",
            "https://docs.python.org/3/reference/expressions.html#is-not",
        ),
        TokenKind::Is => (
            "Tests whether two references point to the same object.",
            "https://docs.python.org/3/reference/expressions.html#is-not",
        ),
        TokenKind::Lambda => (
            "Creates an anonymous function whose body is a single expression.",
            "https://docs.python.org/3/reference/expressions.html#lambdas",
        ),
        TokenKind::Yield
            if nodes
                .iter()
                .any(|node| matches!(node, AnyNodeRef::ExprYieldFrom(_))) =>
        {
            (
                "Delegates part of a generator's operation to another iterable with `yield from`.",
                "https://docs.python.org/3/reference/expressions.html#yield-expressions",
            )
        }
        TokenKind::Yield => (
            "Suspends a generator and produces a value for its caller.",
            "https://docs.python.org/3/reference/expressions.html#yield-expressions",
        ),
        TokenKind::Await => (
            "Suspends the current coroutine until the awaitable completes.",
            "https://docs.python.org/3/reference/expressions.html#await-expression",
        ),
        TokenKind::Assert => (
            "Checks a condition and raises `AssertionError` when it is false. Assertions may be disabled with optimization.",
            "https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement",
        ),
        TokenKind::Break => (
            "Exits the nearest enclosing loop and skips that loop's `else` clause.",
            "https://docs.python.org/3/reference/simple_stmts.html#the-break-statement",
        ),
        TokenKind::Continue => (
            "Skips the rest of the current loop iteration and proceeds with the next one.",
            "https://docs.python.org/3/reference/simple_stmts.html#the-continue-statement",
        ),
        TokenKind::Del => (
            "Deletes a name, item, slice, or attribute according to the target.",
            "https://docs.python.org/3/reference/simple_stmts.html#the-del-statement",
        ),
        TokenKind::From
            if nodes
                .iter()
                .any(|node| matches!(node, AnyNodeRef::ExprYieldFrom(_))) =>
        {
            (
                "Delegates part of a generator's operation to another iterable.",
                "https://docs.python.org/3/reference/expressions.html#yield-expressions",
            )
        }
        TokenKind::From
            if nodes
                .iter()
                .any(|node| matches!(node, AnyNodeRef::StmtRaise(_))) =>
        {
            (
                "Introduces the explicit cause of an exception, or suppresses its context with `from None`.",
                "https://docs.python.org/3/reference/simple_stmts.html#the-raise-statement",
            )
        }
        TokenKind::From => (
            "Introduces the module in a `from ... import ...` statement.",
            "https://docs.python.org/3/reference/simple_stmts.html#the-import-statement",
        ),
        TokenKind::Import
            if nodes.iter().any(|node| {
                matches!(node, AnyNodeRef::StmtImport(stmt) if stmt.is_lazy)
                    || matches!(node, AnyNodeRef::StmtImportFrom(stmt) if stmt.is_lazy)
            }) =>
        {
            (
                "Binds names through an import whose module loading is deferred until first use.",
                "https://docs.python.org/3.15/reference/simple_stmts.html#lazy-imports",
            )
        }
        TokenKind::Import => (
            "Loads a module and binds selected names, or binds the module itself.",
            "https://docs.python.org/3/reference/simple_stmts.html#the-import-statement",
        ),
        TokenKind::Global => (
            "Makes listed names refer to bindings in the module's global scope.",
            "https://docs.python.org/3/reference/simple_stmts.html#the-global-statement",
        ),
        TokenKind::Nonlocal => (
            "Makes listed names refer to existing bindings in the nearest enclosing function scope.",
            "https://docs.python.org/3/reference/simple_stmts.html#the-nonlocal-statement",
        ),
        TokenKind::Pass => (
            "Performs no operation; it is a placeholder where syntax requires a statement.",
            "https://docs.python.org/3/reference/simple_stmts.html#the-pass-statement",
        ),
        TokenKind::Raise => nodes
            .iter()
            .find_map(|node| match node {
                AnyNodeRef::StmtRaise(stmt) if stmt.cause.is_some() => Some((
                    "Raises an exception with an explicit cause set or suppressed by `from`.",
                    "https://docs.python.org/3/reference/simple_stmts.html#the-raise-statement",
                )),
                AnyNodeRef::StmtRaise(stmt) if stmt.exc.is_none() => Some((
                    "Re-raises the exception currently being handled.",
                    "https://docs.python.org/3/reference/simple_stmts.html#the-raise-statement",
                )),
                AnyNodeRef::StmtRaise(_) => Some((
                    "Raises the specified exception.",
                    "https://docs.python.org/3/reference/simple_stmts.html#the-raise-statement",
                )),
                _ => None,
            })
            .unwrap_or((
                "Raises an exception, or re-raises the exception currently being handled.",
                "https://docs.python.org/3/reference/simple_stmts.html#the-raise-statement",
            )),
        TokenKind::Return => (
            "Leaves the current function and supplies its result to the caller.",
            "https://docs.python.org/3/reference/simple_stmts.html#the-return-statement",
        ),
        TokenKind::Type => (
            "Declares a type alias. `type` is a soft keyword in this context.",
            "https://docs.python.org/3/reference/simple_stmts.html#the-type-statement",
        ),
        TokenKind::Lazy => (
            "Marks an import as potentially lazy, deferring module loading until the imported name is first used. `lazy` is a soft keyword added in Python 3.15.",
            "https://docs.python.org/3.15/reference/simple_stmts.html#lazy-imports",
        ),
        TokenKind::As => nodes
            .iter()
            .find_map(|node| match node {
                AnyNodeRef::Alias(_) => Some((
                    "Binds an imported module or name under an alternate name.",
                    "https://docs.python.org/3/reference/simple_stmts.html#the-import-statement",
                )),
                AnyNodeRef::ExceptHandlerExceptHandler(_) => Some((
                    "Binds the handled exception to a name for the duration of the handler.",
                    "https://docs.python.org/3/reference/compound_stmts.html#except-clause",
                )),
                AnyNodeRef::WithItem(_) => Some((
                    "Binds the value returned by a context manager's enter method.",
                    "https://docs.python.org/3/reference/compound_stmts.html#the-with-statement",
                )),
                AnyNodeRef::PatternMatchAs(_) => Some((
                    "Binds the value matched by a pattern to a name.",
                    "https://docs.python.org/3/reference/compound_stmts.html#the-match-statement",
                )),
                _ => None,
            })
            .unwrap_or((
                "Binds a value under a name in the surrounding construct.",
                compound,
            )),
        TokenKind::Async
            if nodes
                .iter()
                .any(|node| matches!(node, AnyNodeRef::Comprehension(_))) =>
        {
            (
                "Marks a comprehension clause as asynchronous.",
                "https://docs.python.org/3/reference/expressions.html#displays-for-lists-sets-and-dictionaries",
            )
        }
        TokenKind::Async
            if nodes
                .iter()
                .any(|node| matches!(node, AnyNodeRef::StmtFor(_))) =>
        {
            (
                "Marks a `for` statement as asynchronous, iterating with the asynchronous iteration protocol.",
                "https://docs.python.org/3/reference/compound_stmts.html#the-async-for-statement",
            )
        }
        TokenKind::Async
            if nodes
                .iter()
                .any(|node| matches!(node, AnyNodeRef::StmtWith(_))) =>
        {
            (
                "Marks a `with` statement as asynchronous, using asynchronous context managers.",
                "https://docs.python.org/3/reference/compound_stmts.html#the-async-with-statement",
            )
        }
        TokenKind::Async => (
            "Marks a function definition as asynchronous.",
            "https://docs.python.org/3/reference/compound_stmts.html#coroutine-function-definition",
        ),
        TokenKind::Class => (
            "Creates a class object by executing the class body in a new namespace.",
            "https://docs.python.org/3/reference/compound_stmts.html#class-definitions",
        ),
        TokenKind::Def => (
            "Defines a function and binds the resulting function object to a name.",
            "https://docs.python.org/3/reference/compound_stmts.html#function-definitions",
        ),
        TokenKind::Elif => (
            "Tests another condition when all preceding conditions in the `if` statement were false.",
            "https://docs.python.org/3/reference/compound_stmts.html#the-if-statement",
        ),
        TokenKind::Else => nodes
            .iter()
            .find_map(|node| match node {
                AnyNodeRef::ExprIf(_) => Some((
                    "Introduces the value selected when a conditional expression's condition is false.",
                    "https://docs.python.org/3/reference/expressions.html#conditional-expressions",
                )),
                AnyNodeRef::StmtIf(_) => Some((
                    "Introduces the suite selected when all preceding conditions are false.",
                    "https://docs.python.org/3/reference/compound_stmts.html#the-if-statement",
                )),
                AnyNodeRef::StmtFor(_) => Some((
                    "Introduces the suite run after iteration is exhausted, but not after `break`.",
                    "https://docs.python.org/3/reference/compound_stmts.html#the-for-statement",
                )),
                AnyNodeRef::StmtWhile(_) => Some((
                    "Introduces the suite run when the condition becomes false, but not after `break`.",
                    "https://docs.python.org/3/reference/compound_stmts.html#the-while-statement",
                )),
                AnyNodeRef::StmtTry(_) => Some((
                    "Introduces the suite run when the `try` suite completes without an exception or control-flow exit.",
                    "https://docs.python.org/3/reference/compound_stmts.html#the-try-statement",
                )),
                _ => None,
            })
            .unwrap_or((
                "Introduces a fallback branch in the surrounding construct.",
                compound,
            )),
        TokenKind::Except
            if nodes
                .iter()
                .any(|node| matches!(node, AnyNodeRef::StmtTry(stmt) if stmt.is_star)) =>
        {
            (
                "Handles matching subgroups from an exception group independently with `except*`.",
                "https://docs.python.org/3/reference/compound_stmts.html#except-clause",
            )
        }
        TokenKind::Except => (
            "Handles an exception raised by the associated `try` suite when its type matches.",
            "https://docs.python.org/3/reference/compound_stmts.html#except-clause",
        ),
        TokenKind::Finally => (
            "Runs a cleanup suite when control leaves the associated `try` statement, including during exceptions and returns.",
            "https://docs.python.org/3/reference/compound_stmts.html#finally-clause",
        ),
        TokenKind::For
            if nodes
                .iter()
                .any(|node| matches!(node, AnyNodeRef::Comprehension(_))) =>
        {
            (
                "Introduces an iteration clause in a comprehension.",
                "https://docs.python.org/3/reference/expressions.html#displays-for-lists-sets-and-dictionaries",
            )
        }
        TokenKind::For => (
            "Iterates over an iterable, assigning each item to the target. Its `else` suite runs after exhaustion, but not after `break`.",
            "https://docs.python.org/3/reference/compound_stmts.html#the-for-statement",
        ),
        TokenKind::If => nodes
            .iter()
            .find_map(|node| match node {
                AnyNodeRef::ExprIf(_) => Some((
                    "Selects one of two values according to a condition.",
                    "https://docs.python.org/3/reference/expressions.html#conditional-expressions",
                )),
                AnyNodeRef::Comprehension(_) => Some((
                    "Filters items in a comprehension according to a condition.",
                    "https://docs.python.org/3/reference/expressions.html#displays-for-lists-sets-and-dictionaries",
                )),
                AnyNodeRef::MatchCase(_) => Some((
                    "Introduces a guard that must succeed after a `case` pattern matches.",
                    "https://docs.python.org/3/reference/compound_stmts.html#guards",
                )),
                AnyNodeRef::StmtIf(_) => Some((
                    "Conditionally selects a suite.",
                    "https://docs.python.org/3/reference/compound_stmts.html#the-if-statement",
                )),
                _ => None,
            })
            .unwrap_or((
                "Conditionally selects a suite or value.",
                "https://docs.python.org/3/reference/compound_stmts.html#the-if-statement",
            )),
        TokenKind::Match => (
            "Evaluates a subject and selects the first `case` whose pattern and optional guard succeed. `match` is a soft keyword.",
            "https://docs.python.org/3/reference/compound_stmts.html#the-match-statement",
        ),
        TokenKind::Case => (
            "Introduces a pattern and optional guard within a `match` statement. `case` is a soft keyword.",
            "https://docs.python.org/3/reference/compound_stmts.html#the-match-statement",
        ),
        TokenKind::Try => (
            "Runs a protected suite with optional exception handlers, an `else` suite, and cleanup in `finally`.",
            "https://docs.python.org/3/reference/compound_stmts.html#the-try-statement",
        ),
        TokenKind::While => (
            "Repeats a suite while its condition is true. Its `else` suite runs when the condition becomes false, but not after `break`.",
            "https://docs.python.org/3/reference/compound_stmts.html#the-while-statement",
        ),
        TokenKind::With => (
            "Runs a suite under one or more context managers, ensuring their exit methods are called.",
            "https://docs.python.org/3/reference/compound_stmts.html#the-with-statement",
        ),
        _ => unreachable!("keyword hover only requests documented Python keywords"),
    }
}

fn format_keyword_documentation(token: Token, module: &Module, ast: &ModModule) -> String {
    let (documentation, reference) = keyword_documentation(token, module, ast);
    format!("{documentation}\n\n[Python language reference]({reference})")
}

fn keyword_hover(
    transaction: &Transaction<'_>,
    handle: &Handle,
    module: Option<&Module>,
    ast: Option<&ModModule>,
    position: TextSize,
) -> Option<HoverResult> {
    let module = module?;
    let ast = ast?;
    let token = keyword_token_at(transaction, handle, module, position)?;
    // Keep the existing type-aware hovers for boolean, identity, and membership operators.
    if matches!(
        token.kind(),
        TokenKind::And | TokenKind::In | TokenKind::Is | TokenKind::Not | TokenKind::Or
    ) {
        return None;
    }
    if token.kind().is_soft_keyword() && !soft_keyword_is_syntax(token, ast) {
        return None;
    }
    let documentation = format_keyword_documentation(token, module, ast);
    let keyword = module.code_at(token.range());
    Some(HoverResult {
        hover: Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: format!("```python\n(keyword) {keyword}\n```\n---\n{documentation}"),
            }),
            range: Some(module.to_lsp_range(token.range())),
        },
        can_increase_verbosity: false,
    })
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
    let ast = ast?;
    let iterable_range = in_keyword_in_iteration_at(Some(ast), position)?;
    let iterable_type = transaction.get_type_at_for_display(handle, iterable_range.start())?;
    let (documentation, reference) = if Ast::locate_node(ast, position)
        .iter()
        .any(|node| matches!(node, AnyNodeRef::Comprehension(_)))
    {
        (
            "Separates the target and iterable in a comprehension's iteration clause.",
            "https://docs.python.org/3/reference/expressions.html#displays-for-lists-sets-and-dictionaries",
        )
    } else {
        (
            "Separates the target and iterable in a `for` statement. The iterable is evaluated once, then its items are assigned to the target one by one.",
            "https://docs.python.org/3/reference/compound_stmts.html#the-for-statement",
        )
    };
    Some(HoverResult {
        hover: Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: format!(
                    "```python\n(keyword) in\n```\n---\nIteration over `{iterable_type}`\n\n{documentation}\n\n[Python language reference]({reference})"
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

    if let Some(result) = keyword_hover(
        transaction,
        handle,
        module_info.as_ref(),
        ast.as_deref(),
        position,
    ) {
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
    let keyword_argument_identifier = keyword_argument_identifier(transaction, handle, position);
    let definition = transaction
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
        .next()
        .filter(|item| {
            keyword_argument_identifier
                .as_ref()
                .is_none_or(|identifier| {
                    item.module.code_at(item.definition_range) == identifier.id.as_str()
                })
        });
    let (kind, name, docstring_range, module) = if let Some(FindDefinitionItemWithDocstring {
        metadata,
        definition_range: definition_location,
        module,
        docstring_range,
        display_name,
    }) = definition
    {
        let kind = metadata.symbol_kind();
        let name = hover_name_from_definition_snippet(
            module.code_at(definition_location),
            display_name.as_deref(),
            fallback_name_from_type,
        );
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

    let mut hover = HoverValue {
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
    .format(transaction, handle);

    if let Some((module, ast, token)) =
        module_info
            .as_ref()
            .zip(ast.as_deref())
            .and_then(|(module, ast)| {
                keyword_token_at(transaction, handle, module, position)
                    .filter(|token| {
                        matches!(
                            token.kind(),
                            TokenKind::And
                                | TokenKind::In
                                | TokenKind::Is
                                | TokenKind::Not
                                | TokenKind::Or
                        )
                    })
                    .map(|token| (module, ast, token))
            })
    {
        let keyword = module.code_at(token.range());
        let documentation = format_keyword_documentation(token, module, ast);
        let HoverContents::Markup(contents) = &mut hover.contents else {
            unreachable!("type hover always uses markdown contents")
        };
        contents.value.push_str(&format!(
            "\n---\n```python\n(keyword) {keyword}\n```\n{documentation}"
        ));
    }

    Some(HoverResult {
        hover,
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
    use pyrefly_types::function::FuncMetadata;
    use pyrefly_types::function::Function;
    use pyrefly_types::heap::TypeHeap;
    use ruff_python_ast::name::Name;

    use super::*;

    fn make_function_type(heap: &TypeHeap, module_name: &str, func_name: &str) -> Type {
        let module = Module::new(
            ModuleName::from_str(module_name),
            ModulePath::filesystem(PathBuf::from(format!("{module_name}.pyi"))),
            Arc::new(String::new()),
        );
        let metadata = FuncMetadata::synthesized(&module, None, Name::new(func_name));
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

    #[test]
    fn hover_name_prefers_display_name_over_quoted_definition_snippet() {
        let name = hover_name_from_definition_snippet("'array'", Some("array"), None);
        assert_eq!(name.as_deref(), Some("array"));
    }

    #[test]
    fn hover_name_uses_fallback_name_over_quoted_definition_snippet() {
        let name =
            hover_name_from_definition_snippet("  'array'  ", None, Some("array".to_owned()));
        assert_eq!(name.as_deref(), Some("array"));
    }

    #[test]
    fn hover_name_keeps_source_snippet_for_normal_definitions() {
        let name = hover_name_from_definition_snippet(
            "external_function",
            Some("array"),
            Some("array".to_owned()),
        );
        assert_eq!(name.as_deref(), Some("external_function"));
    }
}
