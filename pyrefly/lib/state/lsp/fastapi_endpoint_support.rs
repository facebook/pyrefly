/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! FastAPI/Starlette TestClient URL completion and handler navigation.
//!
//! This is an LSP convenience for application tests, not additional FastAPI type checking.
//! For a typed, directly constructed TestClient associated with one known application, URL
//! literals receive method-specific path suggestions and go-to-definition finds the handler:
//!
//! ```python
//! app = FastAPI()
//! @app.get("/users")
//! def users(): ...
//! client = TestClient(app)
//! client.get("/us")     # Completion offers /users.
//! client.get("/users")  # Go-to-definition navigates to users.
//! ```
//!
//! Resolved types establish framework provenance; feature-local binding summaries establish
//! application identity and decorator ownership. Source-versioned summaries work even when
//! the route module's AST is not retained. Completion replaces a complete single-line literal
//! with properly escaped text, preserving the editor's filtering prefix.
//!
//! Only direct application decorators with literal paths and known HTTP methods are described.
//! Router prefixes/mounts, TestClient factories or custom constructors, URL-scope overrides,
//! ambiguous bindings/exports and dynamic method sets are omitted instead of guessed. Handler
//! matching supports literal segments and whole-segment, untyped `{name}` parameters.
//!
//! Framework rules remain in this module and its submodules. Shared Pyrefly state only owns the
//! cache; the core LSP adds completion/definition hooks. This feature executes no HTTP requests
//! and changes neither inference nor diagnostics.

use std::collections::BTreeMap;

use dupe::Dupe;
use lsp_types::CompletionItem;
use lsp_types::CompletionItemKind;
use lsp_types::CompletionItemTextEdit;
use lsp_types::TextEdit;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::module::Module;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_python::module_path::ModuleStyle;
use pyrefly_python::symbol_kind::SymbolKind;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprList;
use ruff_python_ast::ExprSet;
use ruff_python_ast::ExprStringLiteral;
use ruff_python_ast::ExprTuple;
use ruff_python_ast::Keyword;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use serde_json::to_string;
use vec1::Vec1;

use super::DefinitionMetadata;
use super::FindDefinitionItemWithDocstring;
use crate::lsp::wasm::completion::RankedCompletion;
use crate::state::state::Transaction;

mod association;
mod association_summary;
mod cache;
mod provenance;

pub(crate) use cache::EndpointModuleCache;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EndpointMethod {
    Get,
    Post,
    Put,
    Delete,
    Patch,
    Options,
    Head,
    Trace,
    WebSocket,
}

impl EndpointMethod {
    fn from_attr(attr: &str) -> Option<Self> {
        match attr {
            "get" => Some(Self::Get),
            "post" => Some(Self::Post),
            "put" => Some(Self::Put),
            "delete" => Some(Self::Delete),
            "patch" => Some(Self::Patch),
            "options" => Some(Self::Options),
            "head" => Some(Self::Head),
            "trace" => Some(Self::Trace),
            "websocket" => Some(Self::WebSocket),
            _ => None,
        }
    }

    fn from_literal(value: &str) -> Option<Self> {
        Self::from_attr(&value.to_ascii_lowercase())
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Put => "PUT",
            Self::Delete => "DELETE",
            Self::Patch => "PATCH",
            Self::Options => "OPTIONS",
            Self::Head => "HEAD",
            Self::Trace => "TRACE",
            Self::WebSocket => "WEBSOCKET",
        }
    }
}

struct EndpointDefinition {
    path: String,
    methods: Vec<EndpointMethod>,
    module: Module,
    definition_range: TextRange,
    docstring_range: Option<TextRange>,
}

struct EndpointCallContext {
    receiver: Expr,
    literal: ExprStringLiteral,
    method: EndpointMethod,
}

struct EndpointDecoratorSpec {
    path: String,
    methods: Vec<EndpointMethod>,
}

fn endpoint_decorator_spec(expr: &Expr) -> Option<EndpointDecoratorSpec> {
    let Expr::Call(call) = expr else {
        return None;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    if call
        .arguments
        .keywords
        .iter()
        .any(|keyword| keyword.arg.is_none())
    {
        return None;
    }
    let path = match call.arguments.args.as_ref() {
        [Expr::StringLiteral(literal)] if keyword_arg_expr(call, &["path"]).is_none() => {
            literal.value.to_string()
        }
        [] => string_literal_kw_arg(call, &["path"])?.value.to_string(),
        _ => return None,
    };
    let callee = attribute.attr.as_str();
    let methods = if let Some(method) = EndpointMethod::from_attr(callee) {
        vec![method]
    } else if matches!(callee, "route" | "api_route") {
        match keyword_arg_expr(call, &["methods"]) {
            Some(expression) => methods_from_expr(expression)?,
            None => vec![EndpointMethod::Get],
        }
    } else {
        return None;
    };
    Some(EndpointDecoratorSpec { path, methods })
}

fn methods_from_expr(expr: &Expr) -> Option<Vec<EndpointMethod>> {
    let elements = match expr {
        Expr::List(ExprList { elts, .. })
        | Expr::Tuple(ExprTuple { elts, .. })
        | Expr::Set(ExprSet { elts, .. }) => elts,
        _ => return None,
    };
    if elements.is_empty() {
        return None;
    }
    elements
        .iter()
        .map(|element| {
            let Expr::StringLiteral(literal) = element else {
                return None;
            };
            EndpointMethod::from_literal(&literal.value.to_string())
        })
        .collect()
}

fn keyword_arg_expr<'a>(call: &'a ExprCall, names: &[&str]) -> Option<&'a Expr> {
    call.arguments
        .keywords
        .iter()
        .find_map(|Keyword { arg, value, .. }| {
            arg.as_ref()
                .filter(|identifier| names.contains(&identifier.id.as_str()))
                .map(|_| value)
        })
}

fn string_literal_kw_arg(call: &ExprCall, names: &[&str]) -> Option<ExprStringLiteral> {
    match keyword_arg_expr(call, names)? {
        Expr::StringLiteral(lit) => Some(lit.clone()),
        _ => None,
    }
}

fn endpoint_call_context(
    transaction: &Transaction,
    handle: &Handle,
    nodes: &[AnyNodeRef<'_>],
    position: TextSize,
) -> Option<EndpointCallContext> {
    for node in nodes {
        let AnyNodeRef::ExprCall(call) = node else {
            continue;
        };
        let Expr::Attribute(attr) = call.func.as_ref() else {
            continue;
        };
        let Some(method) = EndpointMethod::from_attr(attr.attr.as_str()) else {
            continue;
        };
        let literal = match call.arguments.args.first() {
            Some(Expr::StringLiteral(literal)) => literal.clone(),
            _ => match string_literal_kw_arg(call, &["url"]) {
                Some(literal) => literal,
                None => continue,
            },
        };
        let mut parts = literal.value.iter();
        let Some(part) = parts.next() else {
            continue;
        };
        if parts.next().is_some() || !part.content_range().contains_inclusive(position) {
            continue;
        }
        if provenance::is_test_client(transaction, handle, attr.value.range()) {
            return Some(EndpointCallContext {
                receiver: (*attr.value).clone(),
                literal,
                method,
            });
        }
    }
    None
}

/// Decode only the typed prefix; the edit replaces the complete literal.
fn literal_prefix(
    module: &Module,
    literal: &ExprStringLiteral,
    position: TextSize,
) -> Option<String> {
    let part = literal.value.iter().next()?;
    let opener = module.code_at(TextRange::new(
        part.range().start(),
        part.content_range().start(),
    ));
    let quote = opener.find(['\'', '"'])?;
    let prefix = module.code_at(TextRange::new(part.range().start(), position));
    let closed_prefix = format!("{prefix}{}", &opener[quote..]);
    let Expr::StringLiteral(prefix) = Ast::parse_expr(&closed_prefix, TextSize::new(0)).ok()?
    else {
        return None;
    };
    Some(prefix.value.to_string())
}

fn normalize_path(path: &str) -> &str {
    path.split('?').next().unwrap_or(path)
}

fn split_segments(path: &str) -> Vec<&str> {
    let path = normalize_path(path);
    path.strip_suffix('/').unwrap_or(path).split('/').collect()
}

fn is_param_segment(segment: &str) -> bool {
    segment
        .strip_prefix('{')
        .and_then(|segment| segment.strip_suffix('}'))
        .is_some_and(|parameter| !parameter.is_empty() && !parameter.contains([':', '{', '}']))
}

fn path_matches(pattern: &str, candidate: &str) -> bool {
    let pattern_segments = split_segments(pattern);
    let candidate_segments = split_segments(candidate);
    pattern_segments.len() == candidate_segments.len()
        && pattern_segments
            .iter()
            .zip(candidate_segments)
            .all(|(pattern, candidate)| {
                if is_param_segment(pattern) {
                    !candidate.is_empty()
                } else {
                    !pattern.contains(['{', '}']) && *pattern == candidate
                }
            })
}

fn method_matches(methods: &[EndpointMethod], request: EndpointMethod) -> bool {
    methods.contains(&request)
}

impl Transaction<'_> {
    fn endpoints_for_application(
        &self,
        handle: &Handle,
        receiver: &Expr,
    ) -> Vec<EndpointDefinition> {
        let Some((module, application)) =
            association::application_for_client(self, handle, receiver)
        else {
            return Vec::new();
        };
        if module.path().style() == ModuleStyle::Interface {
            return Vec::new();
        }
        match module.path().details() {
            ModulePathDetails::FileSystem(_) | ModulePathDetails::Memory(_) => {}
            _ => return Vec::new(),
        }
        let summary = self.endpoint_module_cache().get(&module, None);
        summary
            .routes
            .iter()
            .filter(|route| route.application == application)
            .map(|route| EndpointDefinition {
                path: route.spec.path.clone(),
                methods: route.spec.methods.clone(),
                module: module.dupe(),
                definition_range: route.definition_range,
                docstring_range: route.docstring_range,
            })
            .collect()
    }

    pub(crate) fn add_endpoint_completions(
        &self,
        handle: &Handle,
        nodes: &[AnyNodeRef<'_>],
        position: TextSize,
        completions: &mut Vec<RankedCompletion>,
    ) {
        let Some(context) = endpoint_call_context(self, handle, nodes, position) else {
            return;
        };
        let Some(module_info) = self.get_module_info(handle) else {
            return;
        };
        let edit_range = module_info.to_lsp_range(context.literal.range());
        if edit_range.start.line != edit_range.end.line {
            return;
        }
        let Some(prefix) = literal_prefix(&module_info, &context.literal, position) else {
            return;
        };
        let typed_text =
            module_info.code_at(TextRange::new(context.literal.range().start(), position));
        let mut suggestions = BTreeMap::new();
        for endpoint in self.endpoints_for_application(handle, &context.receiver) {
            if method_matches(&endpoint.methods, context.method)
                && endpoint.path.starts_with(&prefix)
            {
                suggestions
                    .entry(endpoint.path)
                    .or_insert_with(|| format!("{} endpoint", context.method.as_str()));
            }
        }
        for (label, detail) in suggestions {
            let new_text = to_string(&label).expect("Serializing an endpoint string cannot fail");
            let filter_text = format!("{typed_text}{}", &label[prefix.len()..]);
            completions.push(RankedCompletion::new(CompletionItem {
                label,
                filter_text: Some(filter_text),
                detail: Some(detail),
                kind: Some(CompletionItemKind::Value),
                text_edit: Some(CompletionItemTextEdit::TextEdit(TextEdit {
                    range: edit_range,
                    new_text,
                })),
                ..Default::default()
            }));
        }
    }

    pub(crate) fn find_definition_for_endpoint_literal(
        &self,
        handle: &Handle,
        nodes: &[AnyNodeRef<'_>],
        position: TextSize,
    ) -> Option<Vec1<FindDefinitionItemWithDocstring>> {
        let context = endpoint_call_context(self, handle, nodes, position)?;
        if !context
            .literal
            .value
            .iter()
            .next()?
            .content_range()
            .contains(position)
        {
            return None;
        }
        let request_path = context.literal.value.to_string();
        Vec1::try_from_vec(
            self.endpoints_for_application(handle, &context.receiver)
                .into_iter()
                .filter(|endpoint| {
                    method_matches(&endpoint.methods, context.method)
                        && path_matches(&endpoint.path, &request_path)
                })
                .map(|endpoint| FindDefinitionItemWithDocstring {
                    metadata: DefinitionMetadata::VariableOrAttribute(Some(SymbolKind::Function)),
                    module: endpoint.module,
                    definition_range: endpoint.definition_range,
                    docstring_range: endpoint.docstring_range,
                    display_name: None,
                })
                .collect(),
        )
        .ok()
    }
}
