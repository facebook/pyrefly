/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ruff_python_ast::Expr;
use ruff_python_ast::name::Name;

use crate::binding::bindings::BindingsBuilder;

const PRIMARY_KEY: Name = Name::new_static("primary_key");

impl<'a> BindingsBuilder<'a> {
    /// Detect if a field has `primary_key=True` set. This will be used to support Django models with custom primary keys.
    pub fn extract_django_primary_key(&self, e: &Expr) -> bool {
        let Some(call) = e.as_call_expr() else {
            return false;
        };
        for keyword in &call.arguments.keywords {
            if let Some(arg_name) = &keyword.arg
                && arg_name.as_str() == PRIMARY_KEY.as_str()
                && let Expr::BooleanLiteral(bl) = &keyword.value
            {
                return bl.value;
            }
        }

        false
    }
}
