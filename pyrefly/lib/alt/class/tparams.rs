/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ops::Deref;
use std::sync::Arc;

use dupe::Dupe;
use ruff_python_ast::Identifier;
use ruff_python_ast::TypeParams;
use ruff_text_size::Ranged;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::solve::TypeFormContext;
use crate::binding::base_class::BaseClass;
use crate::binding::binding::KeyLegacyTypeParam;
use crate::binding::binding::KeyTParams;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorInfo;
use crate::graph::index::Idx;
use crate::types::class::Class;
use crate::types::types::TParams;
use crate::types::types::TParamsSource;
use crate::types::types::Type;

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    /// Calculate class type parameters in the case where we were able to predetermine
    /// syntactically (from looking at the base class expressions) that there are no legacy type variables.
    pub fn calculate_class_tparams_no_legacy(
        &self,
        name: &Identifier,
        scoped_type_params: Option<&TypeParams>,
        errors: &ErrorCollector,
    ) -> Arc<TParams> {
        let scoped_tparams = self.scoped_type_params(scoped_type_params);
        self.validated_tparams(name.range, scoped_tparams, TParamsSource::Class, errors)
    }

    pub fn calculate_class_tparams(
        &self,
        name: &Identifier,
        scoped_type_params: Option<&TypeParams>,
        bases: &[BaseClass],
        legacy: &[Idx<KeyLegacyTypeParam>],
        errors: &ErrorCollector,
    ) -> Arc<TParams> {
        let scoped_tparams = self.scoped_type_params(scoped_type_params);
        let legacy_tparams = legacy
            .iter()
            .filter_map(|key| self.get_idx(*key).deref().parameter().cloned())
            .collect::<SmallSet<_>>();
        let legacy_map = legacy_tparams
            .iter()
            .map(|p| (p.quantified.clone(), p))
            .collect::<SmallMap<_, _>>();
        let lookup_tparam = |t: &Type| {
            let (q, kind) = match t {
                Type::Unpack(t) => (t.as_quantified(), "TypeVarTuple"),
                _ => (t.as_quantified(), "type variable"),
            };
            if q.is_none() && !t.is_error() {
                self.error(
                    errors,
                    name.range,
                    ErrorInfo::Kind(ErrorKind::InvalidTypeVar),
                    format!("Expected a {kind}, got `{}`", self.for_display(t.clone())),
                );
            }
            q.and_then(|q| {
                let p = legacy_map.get(&q);
                if p.is_none() {
                    self.error(
                        errors,
                        name.range,
                        ErrorInfo::Kind(ErrorKind::InvalidTypeVar),
                        "Redundant type parameter declaration".to_owned(),
                    );
                }
                p.map(|x| (*x).clone())
            })
        };

        // TODO(stroxler): There are a lot of checks, such as that `Generic` only appears once
        // and no non-type-vars are used, that we can more easily detect in a dedicated class
        // validation step that validates all the bases. We are deferring these for now.
        let mut generic_tparams = SmallSet::new();
        let mut protocol_tparams = SmallSet::new();
        for base in bases.iter() {
            match base {
                BaseClass::Generic(xs, ..) => {
                    for x in xs {
                        let ty = self.expr_untype(x, TypeFormContext::GenericBase, errors);
                        if let Some(p) = lookup_tparam(&ty)
                            && !generic_tparams.insert(p)
                        {
                            self.error(
                                errors,
                                x.range(),
                                ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                                format!(
                                    "Duplicated type parameter declaration `{}`",
                                    self.module().display(x)
                                ),
                            );
                        }
                    }
                }
                BaseClass::Protocol(xs, ..) => {
                    for x in xs {
                        let ty = self.expr_untype(x, TypeFormContext::GenericBase, errors);
                        if let Some(p) = lookup_tparam(&ty)
                            && !protocol_tparams.insert(p)
                        {
                            self.error(
                                errors,
                                x.range(),
                                ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                                format!(
                                    "Duplicated type parameter declaration `{}`",
                                    self.module().display(x),
                                ),
                            );
                        }
                    }
                }
                _ => {}
            }
        }
        if !generic_tparams.is_empty() && !protocol_tparams.is_empty() {
            self.error(
                errors,
                name.range,
                ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                format!(
                    "Class `{}` specifies type parameters in both `Generic` and `Protocol` bases",
                    name.id,
                ),
            );
        }
        // Initialized the tparams: combine scoped and explicit type parameters
        let mut tparams = SmallSet::new();
        tparams.extend(scoped_tparams);
        tparams.extend(generic_tparams);
        tparams.extend(protocol_tparams);
        // Handle implicit tparams: if a Quantified was bound at this scope and is not yet
        // in tparams, we add it. These will be added in left-to-right order.
        let implicit_tparams_okay = tparams.is_empty();
        for p in legacy_tparams.iter() {
            if !tparams.contains(p) {
                if !implicit_tparams_okay {
                    self.error(errors,
                        name.range,
                        ErrorInfo::Kind(ErrorKind::InvalidTypeVar),
                        format!(
                            "Class `{}` uses type variables not specified in `Generic` or `Protocol` base",
                            name.id,
                        ),
                    );
                }
                tparams.insert(p.clone());
            }
        }

        // Convert our set of `TParam`s into a `TParams` object, which will also perform
        // some additional validation that isn't specific to classes.
        self.validated_tparams(
            name.range,
            tparams.into_iter().collect(),
            TParamsSource::Class,
            errors,
        )
    }

    pub fn get_class_tparams(&self, class: &Class) -> Arc<TParams> {
        match class.precomputed_tparams() {
            Some(tparams) => tparams.dupe(),
            None => self
                .get_from_class(class, &KeyTParams(class.index()))
                .unwrap_or_default(),
        }
    }
}
