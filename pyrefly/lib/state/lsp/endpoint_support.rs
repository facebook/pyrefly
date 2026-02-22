/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::BTreeMap;

use dupe::Dupe;
use lsp_types::CompletionItem;
use lsp_types::CompletionItemKind;
use pyrefly_build::handle::Handle;
use pyrefly_python::docstring::Docstring;
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
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::visitor::Visitor;
use ruff_python_ast::visitor::walk_stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use super::DefinitionMetadata;
use super::FindDefinitionItemWithDocstring;
use crate::state::state::Transaction;

const ENDPOINT_ROUTE_DECORATORS: [&str; 3] = ["route", "api_route", "websocket"];

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
        match attr.to_ascii_lowercase().as_str() {
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
        Self::from_attr(value)
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

#[derive(Clone)]
struct EndpointDefinition {
    path: String,
    methods: Option<Vec<EndpointMethod>>,
    module: Module,
    definition_range: TextRange,
    docstring_range: Option<TextRange>,
}

struct EndpointCallContext {
    literal: ExprStringLiteral,
    method: EndpointMethod,
    in_string_literal: bool,
}

#[derive(Clone)]
struct EndpointDecoratorSpec {
    path: String,
    methods: Option<Vec<EndpointMethod>>,
}

struct EndpointCollector {
    module: Module,
    endpoints: Vec<EndpointDefinition>,
}

impl EndpointCollector {
    fn new(module: Module) -> Self {
        Self {
            module,
            endpoints: Vec::new(),
        }
    }

    fn push_endpoint(&mut self, func: &StmtFunctionDef, spec: EndpointDecoratorSpec) {
        self.endpoints.push(EndpointDefinition {
            path: spec.path,
            methods: spec.methods,
            module: self.module.dupe(),
            definition_range: func.name.range(),
            docstring_range: Docstring::range_from_stmts(&func.body),
        });
    }

    fn collect_from_function(&mut self, func: &StmtFunctionDef) {
        for decorator in &func.decorator_list {
            if let Some(spec) = endpoint_decorator_spec(&decorator.expression) {
                self.push_endpoint(func, spec);
            }
        }
    }
}

impl<'a> Visitor<'a> for EndpointCollector {
    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        if let Stmt::FunctionDef(func) = stmt {
            self.collect_from_function(func);
        }
        walk_stmt(self, stmt);
    }
}

fn endpoint_decorator_spec(expr: &Expr) -> Option<EndpointDecoratorSpec> {
    let Expr::Call(call) = expr else {
        return None;
    };
    let callee = call_callee_name(call)?;
    if let Some(method) = EndpointMethod::from_attr(callee) {
        let path = endpoint_path_from_call(call)?;
        return Some(EndpointDecoratorSpec {
            path,
            methods: Some(vec![method]),
        });
    }
    if ENDPOINT_ROUTE_DECORATORS
        .iter()
        .any(|name| name.eq_ignore_ascii_case(callee))
    {
        let path = endpoint_path_from_call(call)?;
        let methods = endpoint_methods_from_call(call).or_else(|| {
            if keyword_arg_expr(call, &["methods", "method"]).is_some() {
                None
            } else {
                Some(vec![EndpointMethod::Get])
            }
        });
        return Some(EndpointDecoratorSpec { path, methods });
    }
    None
}

fn call_callee_name(call: &ExprCall) -> Option<&str> {
    match call.func.as_ref() {
        Expr::Attribute(attr) => Some(attr.attr.as_str()),
        Expr::Name(name) => Some(name.id.as_str()),
        Expr::Call(call) => call_callee_name(call),
        _ => None,
    }
}

fn endpoint_path_from_call(call: &ExprCall) -> Option<String> {
    if let Some(Expr::StringLiteral(lit)) = call.arguments.args.first() {
        return Some(literal_value(lit));
    }
    string_literal_kw_arg(call, &["path", "url", "route"]).map(|lit| literal_value(&lit))
}

fn endpoint_methods_from_call(call: &ExprCall) -> Option<Vec<EndpointMethod>> {
    let expr = keyword_arg_expr(call, &["methods", "method"])?;
    methods_from_expr(expr)
}

fn methods_from_expr(expr: &Expr) -> Option<Vec<EndpointMethod>> {
    match expr {
        Expr::StringLiteral(lit) => {
            EndpointMethod::from_literal(&lit.value.to_string()).map(|method| vec![method])
        }
        Expr::List(ExprList { elts, .. })
        | Expr::Tuple(ExprTuple { elts, .. })
        | Expr::Set(ExprSet { elts, .. }) => {
            let mut methods = Vec::new();
            for elt in elts {
                let Expr::StringLiteral(lit) = elt else {
                    continue;
                };
                if let Some(method) = EndpointMethod::from_literal(&lit.value.to_string()) {
                    methods.push(method);
                }
            }
            if methods.is_empty() {
                None
            } else {
                Some(methods)
            }
        }
        _ => None,
    }
}

fn keyword_arg_expr<'a>(call: &'a ExprCall, names: &[&str]) -> Option<&'a Expr> {
    for Keyword { arg, value, .. } in &call.arguments.keywords {
        let Some(identifier) = arg else {
            continue;
        };
        if !names
            .iter()
            .any(|name| identifier.id.as_str().eq_ignore_ascii_case(name))
        {
            continue;
        }
        return Some(value);
    }
    None
}

fn string_literal_kw_arg(call: &ExprCall, names: &[&str]) -> Option<ExprStringLiteral> {
    let expr = keyword_arg_expr(call, names)?;
    if let Expr::StringLiteral(lit) = expr {
        return Some(lit.clone());
    }
    None
}

fn literal_value(lit: &ExprStringLiteral) -> String {
    lit.value.to_string()
}

fn string_literal_priority(position: TextSize, range: TextRange) -> (u8, TextSize) {
    if range.contains(position) {
        (0, TextSize::from(0))
    } else if position < range.start() {
        (1, range.start() - position)
    } else {
        (2, position - range.end())
    }
}

fn endpoint_literal_in_call(call: &ExprCall, position: TextSize) -> Option<ExprStringLiteral> {
    let mut best: Option<(u8, TextSize, ExprStringLiteral)> = None;
    let mut candidates = Vec::new();
    if let Some(Expr::StringLiteral(lit)) = call.arguments.args.first() {
        candidates.push(lit.clone());
    }
    for name in ["url", "path", "route"] {
        if let Some(lit) = string_literal_kw_arg(call, &[name]) {
            candidates.push(lit);
        }
    }
    for literal in candidates {
        let (priority, dist) = string_literal_priority(position, literal.range());
        let should_update = match &best {
            Some((best_prio, best_dist, _)) => {
                priority < *best_prio || (priority == *best_prio && dist < *best_dist)
            }
            None => true,
        };
        if should_update {
            best = Some((priority, dist, literal));
            if priority == 0 && dist == TextSize::from(0) {
                break;
            }
        }
    }
    best.map(|(_, _, literal)| literal)
}

fn endpoint_call_context(
    module: &ruff_python_ast::ModModule,
    position: TextSize,
) -> Option<EndpointCallContext> {
    let nodes = pyrefly_python::ast::Ast::locate_node(module, position);
    let mut best: Option<(u8, TextSize, EndpointCallContext)> = None;
    for node in nodes {
        let AnyNodeRef::ExprCall(call) = node else {
            continue;
        };
        let method = EndpointMethod::from_attr(call_callee_name(call)?)?;
        let literal = endpoint_literal_in_call(call, position)?;
        let range = literal.range();
        let (priority, dist) = string_literal_priority(position, range);
        let should_update = match &best {
            Some((best_prio, best_dist, _)) => {
                priority < *best_prio || (priority == *best_prio && dist < *best_dist)
            }
            None => true,
        };
        if should_update {
            best = Some((
                priority,
                dist,
                EndpointCallContext {
                    literal,
                    method,
                    in_string_literal: range.contains(position),
                },
            ));
            if priority == 0 && dist == TextSize::from(0) {
                break;
            }
        }
    }
    best.map(|(_, _, ctx)| ctx)
}

fn normalize_path(path: &str) -> &str {
    path.split('?').next().unwrap_or(path)
}

fn split_segments(path: &str) -> Vec<&str> {
    let trimmed = normalize_path(path).trim_end_matches('/');
    if trimmed.is_empty() {
        return Vec::new();
    }
    trimmed
        .split('/')
        .filter(|segment| !segment.is_empty())
        .collect()
}

fn is_param_segment(segment: &str) -> bool {
    (segment.starts_with('{') && segment.ends_with('}'))
        || (segment.starts_with('<') && segment.ends_with('>'))
}

fn path_matches(pattern: &str, candidate: &str) -> bool {
    let pattern_segments = split_segments(pattern);
    let candidate_segments = split_segments(candidate);
    if pattern_segments.len() != candidate_segments.len() {
        return false;
    }
    for (pattern_segment, candidate_segment) in pattern_segments.iter().zip(candidate_segments) {
        if is_param_segment(pattern_segment) {
            continue;
        }
        if pattern_segment != &candidate_segment {
            return false;
        }
    }
    true
}

fn method_matches(methods: &Option<Vec<EndpointMethod>>, request: EndpointMethod) -> bool {
    match methods {
        Some(methods) => methods.contains(&request),
        None => true,
    }
}

impl<'a> Transaction<'a> {
    fn collect_endpoint_definitions(&self) -> Vec<EndpointDefinition> {
        let mut endpoints = Vec::new();
        for handle in self.handles() {
            let Some(module) = self.get_module_info(&handle) else {
                continue;
            };
            if module.path().style() == ModuleStyle::Interface {
                continue;
            }
            match module.path().details() {
                ModulePathDetails::FileSystem(_)
                | ModulePathDetails::Memory(_)
                | ModulePathDetails::Namespace(_) => {}
                _ => continue,
            }
            let Some(ast) = self.get_ast(&handle) else {
                continue;
            };
            let mut collector = EndpointCollector::new(module);
            for stmt in &ast.body {
                collector.visit_stmt(stmt);
            }
            endpoints.extend(collector.endpoints);
        }
        endpoints
    }

    pub(crate) fn add_endpoint_completions(
        &self,
        _handle: &Handle,
        module: &ruff_python_ast::ModModule,
        position: TextSize,
        completions: &mut Vec<CompletionItem>,
    ) {
        let Some(context) = endpoint_call_context(module, position) else {
            return;
        };
        let literal_range = context.literal.range();
        let allowance = TextSize::from(4);
        let lower_bound = literal_range
            .start()
            .checked_sub(allowance)
            .unwrap_or_else(|| TextSize::new(0));
        if position < lower_bound || position > literal_range.end() {
            return;
        }
        let prefix = literal_value(&context.literal);
        let mut suggestions: BTreeMap<String, String> = BTreeMap::new();
        for endpoint in self.collect_endpoint_definitions() {
            if !method_matches(&endpoint.methods, context.method) {
                continue;
            }
            if !prefix.is_empty() && !endpoint.path.starts_with(&prefix) {
                continue;
            }
            suggestions
                .entry(endpoint.path)
                .or_insert_with(|| format!("{} endpoint", context.method.as_str()));
        }
        if suggestions.is_empty() {
            return;
        }
        for (label, detail) in suggestions {
            let insert_text = if context.in_string_literal {
                label.clone()
            } else {
                format!("\"{}\"", label)
            };
            completions.push(CompletionItem {
                label,
                detail: Some(detail),
                kind: Some(CompletionItemKind::VALUE),
                insert_text: Some(insert_text),
                ..Default::default()
            });
        }
    }

    pub(crate) fn find_definition_for_endpoint_literal(
        &self,
        handle: &Handle,
        position: TextSize,
    ) -> Option<Vec<FindDefinitionItemWithDocstring>> {
        let module = self.get_ast(handle)?;
        let context = endpoint_call_context(module.as_ref(), position)?;
        let request_path = literal_value(&context.literal);
        let mut results = Vec::new();
        for endpoint in self.collect_endpoint_definitions() {
            if !method_matches(&endpoint.methods, context.method) {
                continue;
            }
            if !path_matches(&endpoint.path, &request_path) {
                continue;
            }
            results.push(FindDefinitionItemWithDocstring {
                metadata: DefinitionMetadata::VariableOrAttribute(Some(SymbolKind::Function)),
                module: endpoint.module,
                definition_range: endpoint.definition_range,
                docstring_range: endpoint.docstring_range,
                display_name: None,
            });
        }
        if results.is_empty() {
            None
        } else {
            Some(results)
        }
    }
}
