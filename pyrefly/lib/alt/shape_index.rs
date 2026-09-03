/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Solving policy for the experimental `shape_extensions.Index` restriction.

use pyrefly_types::callable::Param;
use pyrefly_types::callable::Required;
use pyrefly_types::shape_index::lower_index_type;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::types::TParams;
use pyrefly_types::types::Type;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::shape_extension::direct_function_parameter_sources;
use crate::binding::binding::FunctionDefData;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;

impl<Ans: LookupAnswer> AnswersSolver<'_, '_, Ans> {
    pub(crate) fn validate_shape_index_type_parameter_default(
        &self,
        name: &Name,
        default: &Type,
        range: TextRange,
        restriction: &Restriction,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        if !restriction.is_index() {
            return None;
        }
        // Tuple type expressions infer as `type[...]`; defaults store the corresponding value.
        let default = match default {
            Type::Type(inner) => inner.as_ref(),
            _ => default,
        };
        if lower_index_type(default).is_valid() {
            Some(default.clone())
        } else {
            self.error(
                errors,
                range,
                ErrorKind::InvalidTypeVar,
                format!(
                    "Default for `Index` type parameter `{name}` is not a valid index value: `{default}`"
                ),
            );
            Some(self.heap.mk_any_error())
        }
    }

    pub(crate) fn validate_shape_index_function_parameters(
        &self,
        stmt: &FunctionDefData,
        params: &[Param],
        tparams: &TParams,
        errors: &ErrorCollector,
    ) {
        for tparam in tparams
            .iter()
            .filter(|tparam| tparam.restriction().is_index())
        {
            let sources = direct_function_parameter_sources(stmt, params, tparam);
            if sources.len() != 1 {
                self.error(
                    errors,
                    stmt.name.range(),
                    ErrorKind::InvalidTypeVar,
                    format!(
                        "`Index` type parameter `{}` must directly annotate exactly one function parameter, found {}",
                        tparam.name(),
                        sources.len(),
                    ),
                );
                continue;
            }
            let (source_index, source_range, unpacked) = sources[0];
            if unpacked {
                self.error(
                    errors,
                    source_range,
                    ErrorKind::InvalidTypeVar,
                    format!(
                        "`Index` type parameter `{}` cannot bind an unpacked parameter",
                        tparam.name(),
                    ),
                );
                continue;
            }
            let runtime_default = match &params[source_index] {
                Param::PosOnly(_, _, Required::Optional(default))
                | Param::Pos(_, _, Required::Optional(default))
                | Param::KwOnly(_, _, Required::Optional(default)) => default.as_ref(),
                _ => None,
            };
            if let Some(default) = runtime_default
                && !lower_index_type(&default.ty).is_valid()
            {
                self.error(
                    errors,
                    source_range,
                    ErrorKind::InvalidTypeVar,
                    format!(
                        "Default for parameter binding `Index` type parameter `{}` is not a valid index value: `{}`",
                        tparam.name(),
                        default.ty,
                    ),
                );
            }
        }
    }
}
