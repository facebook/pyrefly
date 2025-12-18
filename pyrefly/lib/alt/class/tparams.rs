/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ops::Deref;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_graph::index::Idx;
use pyrefly_python::short_identifier::ShortIdentifier;
use ruff_python_ast::Identifier;
use ruff_python_ast::TypeParams;
use ruff_text_size::Ranged;
use starlark_map::Hashed;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::solve::TypeFormContext;
use crate::binding::base_class::BaseClassGeneric;
use crate::binding::base_class::BaseClassGenericKind;
use crate::binding::binding::Binding;
use crate::binding::binding::Key;
use crate::binding::binding::KeyLegacyTypeParam;
use crate::binding::binding::KeyTParams;
use crate::binding::binding::TypeParameter;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorInfo;
use crate::types::class::Class;
use crate::types::class::ClassDefIndex;
use crate::types::quantified::Quantified;
use crate::types::type_var::PreInferenceVariance;
use crate::types::type_var::Restriction;
use crate::types::types::TParam;
use crate::types::types::TParams;
use crate::types::types::TParamsSource;
use crate::types::types::Type;

struct PlaceholderClassTParamsGuard<'a, 'b, Ans: LookupAnswer> {
    solver: &'a AnswersSolver<'b, Ans>,
    class_idx: ClassDefIndex,
}

impl<'a, 'b, Ans: LookupAnswer> PlaceholderClassTParamsGuard<'a, 'b, Ans> {
    fn new(solver: &'a AnswersSolver<'b, Ans>, class_idx: ClassDefIndex) -> Self {
        Self { solver, class_idx }
    }
}

impl<'a, 'b, Ans: LookupAnswer> Drop for PlaceholderClassTParamsGuard<'a, 'b, Ans> {
    fn drop(&mut self) {
        self.solver.remove_placeholder_class_tparams(self.class_idx);
    }
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    /// Calculate class type parameters in the case where we were able to predetermine
    /// syntactically (from looking at the base class expressions) that there are no legacy type variables.
    pub fn calculate_class_tparams_no_legacy(
        &self,
        def_index: ClassDefIndex,
        name: &Identifier,
        scoped_type_params: Option<&TypeParams>,
        errors: &ErrorCollector,
    ) -> Arc<TParams> {
        self.with_placeholder_class_tparams(def_index, scoped_type_params, || {
            let scoped_tparams = self.scoped_type_params(scoped_type_params, errors);
            self.validated_tparams(name.range, scoped_tparams, TParamsSource::Class, errors)
        })
    }

    pub fn calculate_class_tparams(
        &self,
        def_index: ClassDefIndex,
        name: &Identifier,
        scoped_type_params: Option<&TypeParams>,
        generic_bases: &[BaseClassGeneric],
        legacy: &[Idx<KeyLegacyTypeParam>],
        errors: &ErrorCollector,
    ) -> Arc<TParams> {
        self.with_placeholder_class_tparams(def_index, scoped_type_params, || {
            let scoped_tparams = self.scoped_type_params(scoped_type_params, errors);
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
            for generic_base in generic_bases.iter() {
                for x in generic_base.args.iter() {
                    let ty = self.expr_untype(x, TypeFormContext::GenericBase, errors);
                    if let Some(p) = lookup_tparam(&ty) {
                        let inserted = match generic_base.kind {
                            BaseClassGenericKind::Generic => generic_tparams.insert(p),
                            BaseClassGenericKind::Protocol => protocol_tparams.insert(p),
                        };
                        if !inserted {
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

            self.validated_tparams(
                name.range,
                tparams.into_iter().collect(),
                TParamsSource::Class,
                errors,
            )
        })
    }

    pub fn get_class_tparams(&self, class: &Class) -> Arc<TParams> {
        if let Some(placeholder) = self.placeholder_class_tparams(class.index()) {
            return placeholder;
        }
        match class.precomputed_tparams() {
            Some(tparams) => tparams.dupe(),
            None => self
                .get_from_class(class, &KeyTParams(class.index()))
                .unwrap_or_default(),
        }
    }

    fn with_placeholder_class_tparams<R>(
        &self,
        def_index: ClassDefIndex,
        scoped_type_params: Option<&TypeParams>,
        f: impl FnOnce() -> R,
    ) -> R {
        if let Some(scoped) =
            scoped_type_params.and_then(|params| self.build_placeholder_class_tparams(params))
        {
            self.insert_placeholder_class_tparams(def_index, scoped);
            let guard = PlaceholderClassTParamsGuard::new(self, def_index);
            let result = f();
            drop(guard);
            result
        } else {
            f()
        }
    }

    fn build_placeholder_class_tparams(
        &self,
        scoped_type_params: &TypeParams,
    ) -> Option<Arc<TParams>> {
        if scoped_type_params.type_params.is_empty() {
            return None;
        }
        let mut params = Vec::new();
        for raw_param in scoped_type_params.type_params.iter() {
            let name = raw_param.name();
            let key = Key::Definition(ShortIdentifier::new(name));
            let Some(idx) = self.bindings().key_to_idx_hashed_opt(Hashed::new(&key)) else {
                continue;
            };
            if let Binding::TypeParameter(box TypeParameter {
                name, unique, kind, ..
            }) = self.bindings().get(idx)
            {
                let quantified = Quantified::new(
                    *unique,
                    name.clone(),
                    *kind,
                    None,
                    Restriction::Unrestricted,
                );
                params.push(TParam {
                    quantified,
                    variance: PreInferenceVariance::PUndefined,
                });
            }
        }
        if params.is_empty() {
            None
        } else {
            Some(Arc::new(TParams::new(params)))
        }
    }
}
