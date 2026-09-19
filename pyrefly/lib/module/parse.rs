/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::ast::Ast;
use pyrefly_python::ignore::Ignore;
use pyrefly_python::sys_info::PythonVersion;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_python_ast::PySourceType;
use ruff_python_ast::token::Tokens;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;

/// How deeply expressions may nest before we refuse to analyze a module.
///
/// Every pass over the AST recurses once per level of expression nesting, so the
/// depth a module can survive is bounded by the thread's stack. The parser has
/// its own recursion limit, but it builds operator and postfix chains
/// iteratively — `a + b + c ...` nests one level per operand while the parser
/// stays flat — so it does not bound the depth of the tree it produces. The
/// shallowest crash measured on a 10 MiB stack is a chain of about 1500 binary
/// operators; this limit keeps a wide margin under that.
const MAX_EXPRESSION_DEPTH: usize = 1000;

/// The range of an expression nested deeper than `MAX_EXPRESSION_DEPTH`, if the
/// module contains one. Walks iteratively, since a recursive walk is the thing
/// being guarded against.
fn overly_nested_expression(x: &ModModule) -> Option<TextRange> {
    let mut found = None;
    let mut stack: Vec<(&Expr, usize)> = Vec::new();
    // `visit` yields the expressions written directly in each statement, so each
    // one is the root of a distinct subtree and the walk stays linear overall.
    x.visit(&mut |e: &Expr| {
        if found.is_some() {
            return;
        }
        stack.push((e, 1));
        while let Some((e, depth)) = stack.pop() {
            if depth > MAX_EXPRESSION_DEPTH {
                found = Some(e.range());
                stack.clear();
                return;
            }
            e.recurse(&mut |child| stack.push((child, depth + 1)));
        }
    });
    found
}

pub fn module_parse(
    contents: &str,
    version: PythonVersion,
    source_type: PySourceType,
    errors: &ErrorCollector,
    keep_tokens: bool,
) -> (ModModule, Option<Tokens>, Ignore) {
    let (parsed, parse_errors, unsupported_syntax_errors) =
        Ast::parse_with_version(contents, version, source_type);
    for err in parse_errors {
        errors
            .error_builder(
                err.location,
                ErrorKind::ParseError,
                format!("Parse error: {}", err.error),
            )
            .emit();
    }
    for err in unsupported_syntax_errors {
        errors
            .error_builder(err.range, ErrorKind::InvalidSyntax, format!("{err}"))
            .emit();
    }

    let ignore = Ignore::from_tokens(contents, parsed.tokens());
    let tokens = if keep_tokens {
        Some(parsed.tokens().clone())
    } else {
        None
    };

    let mut module = parsed.into_syntax();
    if let Some(range) = overly_nested_expression(&module) {
        errors
            .error_builder(
                range,
                ErrorKind::ParseError,
                "Parse error: Source is too deeply nested".to_owned(),
            )
            .emit();
        // Analyzing the module would recurse to this depth and overflow the
        // stack, so there is nothing further we can say about it.
        module.body.clear();
    }

    (module, tokens, ignore)
}
