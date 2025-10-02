/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use pyrefly_util::visit::Visit;
use ruff_python_ast::Expr;
use ruff_text_size::Ranged;

use crate::report::pysa::context::ModuleContext;
use crate::report::pysa::location::PysaLocation;
use crate::report::pysa::types::PysaType;

struct VisitorContext<'a> {
    module_context: &'a ModuleContext<'a>,
    type_of_expression: &'a mut HashMap<PysaLocation, PysaType>,
}

fn visit_expression(e: &Expr, context: &mut VisitorContext) {
    let range = e.range();

    // If the expression has a type, export it.
    if let Some(type_) = context.module_context.answers.get_type_trace(range) {
        let display_range = context.module_context.module_info.display_range(range);

        assert!(
            context
                .type_of_expression
                .insert(
                    PysaLocation::new(display_range),
                    PysaType::from_type(&type_, context.module_context)
                )
                .is_none(),
            "Found expressions with the same location"
        );
    }

    e.recurse(&mut |e| visit_expression(e, context));
}

pub fn export_type_of_expressions(context: &ModuleContext) -> HashMap<PysaLocation, PysaType> {
    let mut type_of_expression = HashMap::new();
    let mut visitor_context = VisitorContext {
        module_context: context,
        type_of_expression: &mut type_of_expression,
    };

    for stmt in &context.ast.body {
        stmt.recurse(&mut |e| visit_expression(e, &mut visitor_context));
    }

    type_of_expression
}
