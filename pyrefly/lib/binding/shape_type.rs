/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ruff_python_ast::Expr;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::binding::bindings::BindingsBuilder;
use crate::binding::expr::Usage;
use crate::export::special::SpecialExport;

#[derive(Clone, Debug)]
pub enum TypeParameterBound {
    Ordinary(Expr),
    ShapeFlag {
        domain: Option<Expr>,
        range: TextRange,
    },
}

impl BindingsBuilder<'_> {
    /// Record binding dependencies for a type parameter bound, including the
    /// shape-specific handling required by `shape_extensions.Flag`.
    pub(super) fn record_type_parameter_bound(
        &mut self,
        bound_expr: &mut Expr,
        usage: &mut Usage,
    ) -> TypeParameterBound {
        match bound_expr {
            Expr::Subscript(subscript) if self.is_shape_flag(&subscript.value) => {
                self.ensure_expr(&mut subscript.value, usage);
                self.ensure_type_with_usage(&mut subscript.slice, None, usage);
                TypeParameterBound::ShapeFlag {
                    domain: Some((*subscript.slice).clone()),
                    range: subscript.range,
                }
            }
            marker if self.is_shape_flag(marker) => {
                let range = marker.range();
                self.ensure_expr(marker, usage);
                TypeParameterBound::ShapeFlag {
                    domain: None,
                    range,
                }
            }
            bound => {
                self.ensure_type_with_usage(bound, None, usage);
                TypeParameterBound::Ordinary(bound.clone())
            }
        }
    }

    fn is_shape_flag(&self, expr: &Expr) -> bool {
        if let Expr::Name(name) = expr
            && SpecialExport::new(&name.id) == Some(SpecialExport::Flag)
            && SpecialExport::Flag.defined_in(self.module_info.name())
        {
            return self.scopes.current_binding_is_module_binding(&name.id)
                && matches!(
                    self.scopes.binding_idx_for_name(&name.id),
                    Some((idx, _)) if self.binding_is_class_def(idx)
                );
        }
        self.as_special_export(expr) == Some(SpecialExport::Flag)
    }
}
