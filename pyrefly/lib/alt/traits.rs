/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::class::class_field::ClassField;
use crate::alt::class::variance_inference::VarianceMap;
use crate::alt::types::class_bases::ClassBases;
use crate::alt::types::class_metadata::ClassMetadata;
use crate::alt::types::class_metadata::ClassMro;
use crate::alt::types::class_metadata::ClassSynthesizedFields;
use crate::alt::types::decorated_function::UndecoratedFunction;
use crate::alt::types::legacy_lookup::LegacyTypeParameterLookup;
use crate::alt::types::yields::YieldFromResult;
use crate::alt::types::yields::YieldResult;
use crate::binding::binding::AnnotationTarget;
use crate::binding::binding::AnnotationWithTarget;
use crate::binding::binding::Binding;
use crate::binding::binding::BindingAnnotation;
use crate::binding::binding::BindingClass;
use crate::binding::binding::BindingClassBaseType;
use crate::binding::binding::BindingClassField;
use crate::binding::binding::BindingClassMetadata;
use crate::binding::binding::BindingClassMro;
use crate::binding::binding::BindingClassSynthesizedFields;
use crate::binding::binding::BindingConsistentOverrideCheck;
use crate::binding::binding::BindingDecoratedFunction;
use crate::binding::binding::BindingExpect;
use crate::binding::binding::BindingExport;
use crate::binding::binding::BindingLegacyTypeParam;
use crate::binding::binding::BindingTParams;
use crate::binding::binding::BindingUndecoratedFunction;
use crate::binding::binding::BindingVariance;
use crate::binding::binding::BindingYield;
use crate::binding::binding::BindingYieldFrom;
use crate::binding::binding::EmptyAnswer;
use crate::binding::binding::Initialized;
use crate::binding::binding::Key;
use crate::binding::binding::KeyAnnotation;
use crate::binding::binding::KeyClass;
use crate::binding::binding::KeyClassBaseType;
use crate::binding::binding::KeyClassField;
use crate::binding::binding::KeyClassMetadata;
use crate::binding::binding::KeyClassMro;
use crate::binding::binding::KeyClassSynthesizedFields;
use crate::binding::binding::KeyConsistentOverrideCheck;
use crate::binding::binding::KeyDecoratedFunction;
use crate::binding::binding::KeyExpect;
use crate::binding::binding::KeyExport;
use crate::binding::binding::KeyLegacyTypeParam;
use crate::binding::binding::KeyTParams;
use crate::binding::binding::KeyUndecoratedFunction;
use crate::binding::binding::KeyVariance;
use crate::binding::binding::KeyYield;
use crate::binding::binding::KeyYieldFrom;
use crate::binding::binding::Keyed;
use crate::binding::binding::NoneIfRecursive;
use crate::error::collector::ErrorCollector;
use crate::types::annotation::Annotation;
use crate::types::class::Class;
use crate::types::type_info::TypeInfo;
use crate::types::types::TParams;
use crate::types::types::Type;
use crate::types::types::Var;

pub trait Solve<Ans: LookupAnswer>: Keyed {
    /// Solve the binding.
    /// Note that the key (`Self`) is not provided, as the result of a binding should
    /// not depend on the key it was bound to.
    fn solve(
        answers: &AnswersSolver<Ans>,
        binding: &Self::Value,
        errors: &ErrorCollector,
    ) -> Arc<Self::Answer>;

    /// We have reached a recursive solve of this binding.
    /// Create a sentinel value to store information about it.
    fn create_recursive(_answers: &AnswersSolver<Ans>, _binding: &Self::Value) -> Var {
        Var::ZERO
    }

    /// We hit a recursive case, so promote the recursive value into an answer that needs to be
    /// sufficient for now.
    fn promote_recursive(x: Var) -> Self::Answer;

    /// We solved a binding, but during its execution we gave some people back a recursive value.
    /// Record that recursive value along with the answer.
    fn record_recursive(
        _answers: &AnswersSolver<Ans>,
        _range: TextRange,
        _answer: &Arc<Self::Answer>,
        _recursive: Var,
        _errors: &ErrorCollector,
    ) {
    }
}

impl<Ans: LookupAnswer> Solve<Ans> for Key {
    fn solve(
        answers: &AnswersSolver<Ans>,
        binding: &Binding,
        errors: &ErrorCollector,
    ) -> Arc<TypeInfo> {
        answers.solve_binding(binding, errors)
    }

    fn create_recursive(answers: &AnswersSolver<Ans>, binding: &Self::Value) -> Var {
        answers.create_recursive(binding)
    }

    fn promote_recursive(x: Var) -> Self::Answer {
        TypeInfo::of_ty(Type::Var(x))
    }

    fn record_recursive(
        answers: &AnswersSolver<Ans>,
        range: TextRange,
        answer: &Arc<TypeInfo>,
        recursive: Var,
        errors: &ErrorCollector,
    ) {
        answers.record_recursive(range, answer.ty().clone(), recursive, errors);
    }
}

impl<Ans: LookupAnswer> Solve<Ans> for KeyExpect {
    fn solve(
        answers: &AnswersSolver<Ans>,
        binding: &BindingExpect,
        errors: &ErrorCollector,
    ) -> Arc<EmptyAnswer> {
        answers.solve_expectation(binding, errors)
    }

    fn promote_recursive(_: Var) -> Self::Answer {
        EmptyAnswer
    }
}

impl<Ans: LookupAnswer> Solve<Ans> for KeyConsistentOverrideCheck {
    fn solve(
        answers: &AnswersSolver<Ans>,
        binding: &BindingConsistentOverrideCheck,
        errors: &ErrorCollector,
    ) -> Arc<EmptyAnswer> {
        answers.solve_consistent_override_check(binding, errors)
    }

    fn promote_recursive(_: Var) -> Self::Answer {
        EmptyAnswer
    }
}

impl<Ans: LookupAnswer> Solve<Ans> for KeyExport {
    fn solve(
        answers: &AnswersSolver<Ans>,
        binding: &BindingExport,
        errors: &ErrorCollector,
    ) -> Arc<Type> {
        Arc::new(answers.solve_binding(&binding.0, errors).arc_clone_ty())
    }

    fn create_recursive(answers: &AnswersSolver<Ans>, binding: &Self::Value) -> Var {
        answers.create_recursive(&binding.0)
    }

    fn promote_recursive(x: Var) -> Self::Answer {
        Type::Var(x)
    }

    fn record_recursive(
        answers: &AnswersSolver<Ans>,
        range: TextRange,
        answer: &Arc<Type>,
        recursive: Var,
        errors: &ErrorCollector,
    ) {
        answers.record_recursive(range, answer.as_ref().clone(), recursive, errors);
    }
}

impl<Ans: LookupAnswer> Solve<Ans> for KeyDecoratedFunction {
    fn solve(
        answers: &AnswersSolver<Ans>,
        binding: &BindingDecoratedFunction,
        errors: &ErrorCollector,
    ) -> Arc<Type> {
        answers.solve_decorated_function(binding, errors)
    }

    fn promote_recursive(_: Var) -> Self::Answer {
        // TODO(samgoldman) I'm not sure this really makes sense. These bindings should never
        // be recursive, but this definition is required.
        Type::any_implicit()
    }
}

impl<Ans: LookupAnswer> Solve<Ans> for KeyUndecoratedFunction {
    fn solve(
        answers: &AnswersSolver<Ans>,
        binding: &BindingUndecoratedFunction,
        errors: &ErrorCollector,
    ) -> Arc<UndecoratedFunction> {
        answers.solve_undecorated_function(binding, errors)
    }

    fn promote_recursive(_: Var) -> Self::Answer {
        // This shouldn't happen
        UndecoratedFunction::recursive()
    }
}

impl<Ans: LookupAnswer> Solve<Ans> for KeyClass {
    fn solve(
        answers: &AnswersSolver<Ans>,
        binding: &BindingClass,
        errors: &ErrorCollector,
    ) -> Arc<NoneIfRecursive<Class>> {
        answers.solve_class(binding, errors)
    }

    fn promote_recursive(_: Var) -> Self::Answer {
        NoneIfRecursive(None)
    }
}

impl<Ans: LookupAnswer> Solve<Ans> for KeyTParams {
    fn solve(
        answers: &AnswersSolver<Ans>,
        binding: &BindingTParams,
        errors: &ErrorCollector,
    ) -> Arc<TParams> {
        answers.solve_tparams(binding, errors)
    }

    fn promote_recursive(_: Var) -> Self::Answer {
        TParams::default()
    }
}

impl<Ans: LookupAnswer> Solve<Ans> for KeyClassBaseType {
    fn solve(
        answers: &AnswersSolver<Ans>,
        binding: &BindingClassBaseType,
        errors: &ErrorCollector,
    ) -> Arc<ClassBases> {
        answers.solve_class_base_type(binding, errors)
    }

    fn promote_recursive(_: Var) -> Self::Answer {
        ClassBases::default()
    }
}

impl<Ans: LookupAnswer> Solve<Ans> for KeyClassField {
    fn solve(
        answers: &AnswersSolver<Ans>,
        binding: &BindingClassField,
        errors: &ErrorCollector,
    ) -> Arc<ClassField> {
        answers.solve_class_field(binding, errors)
    }

    fn promote_recursive(_: Var) -> Self::Answer {
        // TODO(stroxler) Revisit the recursive handling, which needs changes in the plumbing
        // to work correctly; what we have here is a fallback to permissive gradual typing.
        ClassField::recursive()
    }
}

impl<Ans: LookupAnswer> Solve<Ans> for KeyClassSynthesizedFields {
    fn solve(
        answers: &AnswersSolver<Ans>,
        binding: &BindingClassSynthesizedFields,
        errors: &ErrorCollector,
    ) -> Arc<ClassSynthesizedFields> {
        answers.solve_class_synthesized_fields(errors, binding)
    }

    fn promote_recursive(_: Var) -> Self::Answer {
        ClassSynthesizedFields::default()
    }
}

impl<Ans: LookupAnswer> Solve<Ans> for KeyVariance {
    fn solve(
        answers: &AnswersSolver<Ans>,
        binding: &BindingVariance,
        _errors: &ErrorCollector,
    ) -> Arc<VarianceMap> {
        answers.solve_variance_binding(binding)
    }

    fn promote_recursive(_: Var) -> Self::Answer {
        VarianceMap::default()
    }
}

impl<Ans: LookupAnswer> Solve<Ans> for KeyAnnotation {
    fn solve(
        answers: &AnswersSolver<Ans>,
        binding: &BindingAnnotation,
        errors: &ErrorCollector,
    ) -> Arc<AnnotationWithTarget> {
        answers.solve_annotation(binding, errors)
    }

    fn promote_recursive(_: Var) -> Self::Answer {
        AnnotationWithTarget {
            target: AnnotationTarget::Assign(Name::default(), Initialized::Yes),
            annotation: Annotation::default(),
        }
    }
}

impl<Ans: LookupAnswer> Solve<Ans> for KeyClassMetadata {
    fn solve(
        answers: &AnswersSolver<Ans>,
        binding: &BindingClassMetadata,
        errors: &ErrorCollector,
    ) -> Arc<ClassMetadata> {
        answers.solve_class_metadata(binding, errors)
    }

    fn promote_recursive(_: Var) -> Self::Answer {
        ClassMetadata::recursive()
    }
}

impl<Ans: LookupAnswer> Solve<Ans> for KeyClassMro {
    fn solve(
        answers: &AnswersSolver<Ans>,
        binding: &BindingClassMro,
        errors: &ErrorCollector,
    ) -> Arc<ClassMro> {
        answers.solve_class_mro(binding, errors)
    }

    fn promote_recursive(_: Var) -> Self::Answer {
        ClassMro::recursive()
    }
}

impl<Ans: LookupAnswer> Solve<Ans> for KeyLegacyTypeParam {
    fn solve(
        answers: &AnswersSolver<Ans>,
        binding: &BindingLegacyTypeParam,
        _errors: &ErrorCollector,
    ) -> Arc<LegacyTypeParameterLookup> {
        answers.solve_legacy_tparam(binding)
    }

    fn promote_recursive(_: Var) -> Self::Answer {
        LegacyTypeParameterLookup::NotParameter(Type::any_implicit())
    }
}

impl<Ans: LookupAnswer> Solve<Ans> for KeyYield {
    fn solve(
        answers: &AnswersSolver<Ans>,
        binding: &BindingYield,
        errors: &ErrorCollector,
    ) -> Arc<YieldResult> {
        answers.solve_yield(binding, errors)
    }

    fn promote_recursive(_: Var) -> Self::Answer {
        // In practice, we should never have recursive bindings with yield.
        YieldResult::recursive()
    }
}

impl<Ans: LookupAnswer> Solve<Ans> for KeyYieldFrom {
    fn solve(
        answers: &AnswersSolver<Ans>,
        binding: &BindingYieldFrom,
        errors: &ErrorCollector,
    ) -> Arc<YieldFromResult> {
        answers.solve_yield_from(binding, errors)
    }

    fn promote_recursive(_: Var) -> Self::Answer {
        // In practice, we should never have recursive bindings with yield from.
        YieldFromResult::recursive()
    }
}
