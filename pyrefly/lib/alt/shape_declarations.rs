/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Type parameters from the dimensions a `@shape_vars` or `@static_jaxtyping`
//! declaration introduces.

use std::sync::Arc;

use dupe::Dupe;
use pyrefly_types::quantified::Quantified;
use pyrefly_types::types::TParams;
use pyrefly_types::types::TParamsSource;
use pyrefly_util::visit::Visit;
use ruff_text_size::TextRange;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::binding::shape_type::ShapeDeclaration;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::types::types::Type;

impl<'ctx, 'answer, Ans: LookupAnswer> AnswersSolver<'ctx, 'answer, Ans> {
    /// Report each dimension a definition declares whose name is already a type
    /// parameter of the definition or of its enclosing class. The two would be
    /// separate variables that print identically, and an annotation naming one of
    /// them would silently mean the other.
    ///
    /// A declaration reserves its names just as a native type-parameter list does.
    /// Collisions are reported even when the signature does not use the declared
    /// name, so adding a local annotation cannot change whether the definition
    /// itself is valid.
    fn check_declared_dimension_names(
        &self,
        tparams: &[Quantified],
        class_tparams: Option<&Arc<TParams>>,
        declaration: &ShapeDeclaration,
        name_range: TextRange,
        errors: &ErrorCollector,
    ) {
        for dim in declaration.dims() {
            let existing = tparams
                .iter()
                .find(|param| param.name() == dim.name())
                .map(|param| (param, "this definition"))
                .or_else(|| {
                    class_tparams.and_then(|tparams| {
                        tparams
                            .iter()
                            .find(|param| param.name() == dim.name())
                            .map(|param| (param, "the enclosing class"))
                    })
                });
            if let Some((existing, owner)) = existing {
                self.error(
                    errors,
                    name_range,
                    ErrorKind::InvalidTypeVar,
                    format!(
                        "`{}` is declared by `{}` and is already a type parameter of \
                         {owner}. Rename one of them: the two would be separate \
                         variables spelled the same way",
                        existing.name(),
                        declaration.kind().decorator(),
                    ),
                );
            }
        }
    }

    /// Validate the type parameters of a class, with the dimensions it declares
    /// appended.
    ///
    /// Declared dimensions always come after a class's own parameters, so the
    /// argument order of a generic does not depend on whether those were written
    /// as PEP 695 syntax or as a `Generic[...]` base, which reach this point by
    /// different routes.
    pub fn finalize_class_tparams(
        &self,
        name_range: TextRange,
        mut tparams: Vec<Quantified>,
        errors: &ErrorCollector,
    ) -> TParams {
        if let Some(declaration) = self.bindings().shape_declarations.declared_by(name_range) {
            self.check_declared_dimension_names(&tparams, None, declaration, name_range, errors);
            tparams.extend(declaration.dims().iter().cloned());
        }
        self.validated_tparams(name_range, tparams, TParamsSource::Class, errors)
    }

    /// Add the shape variables a function declares to its type parameters.
    ///
    /// Only declared dimensions the signature actually mentions become callable
    /// parameters. A dimension used only in the body remains a rigid symbol scoped
    /// to this definition; it is deliberately not an inference variable that each
    /// call would instantiate independently. A nested definition may capture that
    /// symbol, just as it may capture a native type parameter from this definition.
    pub fn collect_declared_dimensions(
        &self,
        callable: &impl Visit<Type>,
        tparams: &Arc<TParams>,
        class_tparams: Option<&Arc<TParams>>,
        name_range: TextRange,
        errors: &ErrorCollector,
    ) -> Arc<TParams> {
        let Some(declaration) = self.bindings().shape_declarations.declared_by(name_range) else {
            return tparams.dupe();
        };
        let declared = declaration.dims();
        debug_assert!(
            declared.iter().all(|dim| dim.default().is_none()),
            "shape declarations cannot specify defaults"
        );

        let mut used = Vec::new();
        let mut collect = |q: &Quantified| {
            if declared.contains(q) && !used.contains(q) {
                used.push(q.clone());
            }
        };
        // Dimensions inside an `IntTuple` are stored as `Int::Symbolic(Type)`, so use
        // the semantic type-variable traversal rather than a structural one. A
        // quantified owned by a nested `Forall` belongs to that generic value and
        // must not be hoisted into this callable.
        callable.visit(&mut |ty: &Type| ty.for_each_free_quantified(&mut collect));

        self.check_declared_dimension_names(
            tparams.as_vec(),
            class_tparams,
            declaration,
            name_range,
            errors,
        );
        // Declared dimensions come last, in declaration order rather than the order
        // the signature happens to mention them in.
        let mut params: Vec<_> = tparams.as_vec().to_vec();
        params.extend(declared.iter().filter(|dim| used.contains(dim)).cloned());
        if params.len() == tparams.as_vec().len() {
            // Nothing was bound, so the existing parameters stand as they were
            // already validated. Running them through again would repeat any
            // diagnostic they carry.
            return tparams.dupe();
        }
        // Existing parameters were already validated, and declared dimensions
        // never have defaults, so only the boundary between the two can be invalid.
        if let Some(previous) = tparams.iter().last()
            && previous.default().is_some()
        {
            self.error(
                errors,
                name_range,
                ErrorKind::InvalidTypeVar,
                format!(
                    "Type parameter `{}` without a default cannot follow type parameter `{}` with a default",
                    params[tparams.len()].name(),
                    previous.name()
                ),
            );
        }
        Arc::new(TParams::new(params))
    }
}
