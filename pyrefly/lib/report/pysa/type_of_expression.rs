/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use pyrefly_util::visit::Visit;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprCall;
use ruff_text_size::Ranged;

use crate::report::pysa::context::ModuleContext;
use crate::report::pysa::location::PysaLocation;
use crate::report::pysa::types::PysaType;

struct VisitorContext<'a> {
    module_context: &'a ModuleContext<'a>,
    type_of_expression: &'a mut HashMap<PysaLocation, PysaType>,
}

/// Export the type of a single expression, if it has one.
fn maybe_export_type(e: &Expr, context: &mut VisitorContext) {
    let range = e.range();
    if let Some(type_) = context
        .module_context
        .answers_context
        .answers
        .get_type_trace(range)
    {
        // An expression may match multiple patterns (e.g., a Name node that is
        // also a call argument). The type is the same, so skip duplicates.
        context
            .type_of_expression
            .entry(PysaLocation::from_text_range(
                range,
                &context.module_context.answers_context.module_info,
            ))
            .or_insert_with(|| PysaType::from_type(&type_, context.module_context));
    }
}

/// We only export types for expressions that Pysa needs:
/// - `Expr::Name`: simple variable references (e.g. `x`)
/// - `Expr::Attribute`: the base of an attribute access (e.g. type of `x` in `x.foo`)
/// - `Expr::Call`: each positional and keyword argument
///
/// We still recurse into all child expressions so nested occurrences are found.
fn visit_expression(e: &Expr, context: &mut VisitorContext) {
    match e {
        Expr::Name(_) => maybe_export_type(e, context),
        Expr::Attribute(ExprAttribute { value, .. }) => maybe_export_type(value, context),
        Expr::Call(ExprCall { arguments, .. }) => {
            for arg in &arguments.args {
                maybe_export_type(arg, context);
            }
            for keyword in &arguments.keywords {
                maybe_export_type(&keyword.value, context);
            }
        }
        _ => {}
    }

    e.recurse(&mut |e| visit_expression(e, context));
}

pub fn export_type_of_expressions(context: &ModuleContext) -> HashMap<PysaLocation, PysaType> {
    let mut type_of_expression = HashMap::new();
    let mut visitor_context = VisitorContext {
        module_context: context,
        type_of_expression: &mut type_of_expression,
    };

    for stmt in &context.answers_context.ast.body {
        stmt.recurse(&mut |e| visit_expression(e, &mut visitor_context));
    }

    type_of_expression
}
