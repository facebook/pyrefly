/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::iter;
use std::mem;
use std::sync::Arc;

use dupe::Dupe;
use itertools::Either;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprSubscript;
use ruff_python_ast::TypeParam;
use ruff_python_ast::TypeParams;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::ordered_set::OrderedSet;
use starlark_map::small_map::Entry;
use starlark_map::small_map::SmallMap;

use crate::alt::answers::AnswersSolver;
use crate::alt::answers::LookupAnswer;
use crate::alt::callable::CallArg;
use crate::alt::class::class_field::ClassField;
use crate::alt::types::class_metadata::ClassMetadata;
use crate::alt::types::class_metadata::ClassSynthesizedFields;
use crate::alt::types::decorated_function::DecoratedFunction;
use crate::alt::types::legacy_lookup::LegacyTypeParameterLookup;
use crate::alt::types::yields::YieldFromResult;
use crate::alt::types::yields::YieldResult;
use crate::binding::binding::AnnotationStyle;
use crate::binding::binding::AnnotationTarget;
use crate::binding::binding::AnnotationWithTarget;
use crate::binding::binding::Binding;
use crate::binding::binding::BindingAnnotation;
use crate::binding::binding::BindingClass;
use crate::binding::binding::BindingClassField;
use crate::binding::binding::BindingClassMetadata;
use crate::binding::binding::BindingClassSynthesizedFields;
use crate::binding::binding::BindingExpect;
use crate::binding::binding::BindingFunction;
use crate::binding::binding::BindingLegacyTypeParam;
use crate::binding::binding::BindingYield;
use crate::binding::binding::BindingYieldFrom;
use crate::binding::binding::EmptyAnswer;
use crate::binding::binding::ExprOrBinding;
use crate::binding::binding::FunctionStubOrImpl;
use crate::binding::binding::Initialized;
use crate::binding::binding::IsAsync;
use crate::binding::binding::Key;
use crate::binding::binding::KeyExport;
use crate::binding::binding::KeyFunction;
use crate::binding::binding::LastStmt;
use crate::binding::binding::NoneIfRecursive;
use crate::binding::binding::RaisedException;
use crate::binding::binding::SizeExpectation;
use crate::binding::binding::SuperStyle;
use crate::binding::binding::TypeParameter;
use crate::binding::binding::UnpackedPosition;
use crate::binding::narrow::identifier_and_chain_for_property;
use crate::binding::narrow::identifier_and_chain_prefix_for_property;
use crate::dunder;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorContext;
use crate::error::context::TypeCheckContext;
use crate::error::context::TypeCheckKind;
use crate::error::kind::ErrorKind;
use crate::module::short_identifier::ShortIdentifier;
use crate::ruff::ast::Ast;
use crate::types::annotation::Annotation;
use crate::types::annotation::Qualifier;
use crate::types::callable::Function;
use crate::types::callable::FunctionKind;
use crate::types::callable::Param;
use crate::types::callable::ParamList;
use crate::types::callable::Required;
use crate::types::class::Class;
use crate::types::class::ClassType;
use crate::types::display::TypeDisplayContext;
use crate::types::literal::Lit;
use crate::types::module::Module;
use crate::types::param_spec::ParamSpec;
use crate::types::quantified::Quantified;
use crate::types::quantified::QuantifiedInfo;
use crate::types::quantified::QuantifiedKind;
use crate::types::tuple::Tuple;
use crate::types::type_info::TypeInfo;
use crate::types::type_var::Restriction;
use crate::types::type_var::TypeVar;
use crate::types::type_var::Variance;
use crate::types::type_var_tuple::TypeVarTuple;
use crate::types::types::AnyStyle;
use crate::types::types::CalleeKind;
use crate::types::types::Forallable;
use crate::types::types::SuperObj;
use crate::types::types::TParamInfo;
use crate::types::types::TParams;
use crate::types::types::Type;
use crate::types::types::TypeAlias;
use crate::types::types::TypeAliasStyle;
use crate::util::prelude::SliceExt;
use crate::util::visit::VisitMut;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TypeFormContext {
    /// Expression in a base class list
    BaseClassList,
    /// Variable annotation in a class
    ClassVarAnnotation,
    /// Argument to a function such as cast, assert_type, or TypeVar
    FunctionArgument,
    /// Arguments to Generic[] or Protocol[]
    GenericBase,
    /// Parameter annotation for a function
    ParameterAnnotation,
    ParameterArgsAnnotation,
    ParameterKwargsAnnotation,
    ReturnAnnotation,
    /// Type argument for a generic
    TypeArgument,
    /// Type argument for the return position of a Callable type
    TypeArgumentCallableReturn,
    /// Type argument for the parameters list of a Callable type or a tuple
    TupleOrCallableParam,
    /// Constraints or upper bound for type variables
    TypeVarConstraint,
    /// Default values for each kind of type variable
    TypeVarDefault,
    ParamSpecDefault,
    TypeVarTupleDefault,
    /// A type being aliased
    TypeAlias,
    /// Variable annotation outside of a class definition
    /// Is the variable assigned a value here?
    VarAnnotation(Initialized),
}

pub enum Iterable {
    OfType(Type),
    FixedLen(Vec<Type>),
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub fn solve_legacy_tparam(
        &self,
        binding: &BindingLegacyTypeParam,
    ) -> Arc<LegacyTypeParameterLookup> {
        match self.get_idx(binding.0).ty() {
            Type::Type(box Type::TypeVar(x)) => {
                let q = Quantified::type_var(
                    x.qname().id().clone(),
                    self.uniques,
                    x.default().cloned(),
                    x.restriction().clone(),
                );
                Arc::new(LegacyTypeParameterLookup::Parameter(TParamInfo {
                    quantified: q,
                    variance: x.variance(),
                }))
            }
            Type::Type(box Type::TypeVarTuple(x)) => {
                let q = Quantified::type_var_tuple(
                    x.qname().id().clone(),
                    self.uniques,
                    x.default().cloned(),
                );
                Arc::new(LegacyTypeParameterLookup::Parameter(TParamInfo {
                    quantified: q,
                    variance: Some(Variance::Invariant),
                }))
            }
            Type::Type(box Type::ParamSpec(x)) => {
                let q = Quantified::param_spec(
                    x.qname().id().clone(),
                    self.uniques,
                    x.default().cloned(),
                );
                Arc::new(LegacyTypeParameterLookup::Parameter(TParamInfo {
                    quantified: q,
                    variance: Some(Variance::Invariant),
                }))
            }
            ty => Arc::new(LegacyTypeParameterLookup::NotParameter(ty.clone())),
        }
    }

    pub fn solve_mro(
        &self,
        binding: &BindingClassMetadata,
        errors: &ErrorCollector,
    ) -> Arc<ClassMetadata> {
        let BindingClassMetadata {
            def: k,
            bases,
            keywords,
            decorators,
            is_new_type,
            special_base,
        } = binding;
        let metadata = match &self.get_idx(*k).0 {
            None => ClassMetadata::recursive(),
            Some(cls) => self.class_metadata_of(
                cls,
                bases,
                keywords,
                decorators,
                *is_new_type,
                special_base,
                errors,
            ),
        };
        Arc::new(metadata)
    }

    pub fn solve_annotation(
        &self,
        binding: &BindingAnnotation,
        errors: &ErrorCollector,
    ) -> Arc<AnnotationWithTarget> {
        match binding {
            BindingAnnotation::AnnotateExpr(target, x, class_key) => {
                let type_form_context = target.type_form_context();
                let mut ann = self.expr_annotation(x, type_form_context, errors);
                if let Some(class_key) = class_key
                    && let Some(ty) = &mut ann.ty
                {
                    let class = &*self.get_idx(*class_key);
                    if let Some(cls) = &class.0 {
                        ty.subst_self_special_form_mut(&Type::SelfType(cls.as_class_type()));
                    }
                }
                Arc::new(AnnotationWithTarget {
                    target: target.clone(),
                    annotation: ann,
                })
            }
            BindingAnnotation::Type(target, x) => Arc::new(AnnotationWithTarget {
                target: target.clone(),
                annotation: Annotation::new_type(x.clone()),
            }),
        }
    }

    pub fn is_subset_eq(&self, got: &Type, want: &Type) -> bool {
        self.solver().is_subset_eq(got, want, self.type_order())
    }

    fn expr_qualifier(
        &self,
        x: &Expr,
        type_form_context: TypeFormContext,
        errors: &ErrorCollector,
    ) -> Option<Qualifier> {
        let ty = match x {
            Expr::Name(_) | Expr::Attribute(_) => Some(self.expr_infer(x, errors)),
            _ => None,
        };
        if let Some(Type::Type(box Type::SpecialForm(special))) = ty {
            let qualifier = special.to_qualifier();
            match qualifier {
                Some(
                    Qualifier::ClassVar
                    | Qualifier::ReadOnly
                    | Qualifier::NotRequired
                    | Qualifier::Required,
                ) if type_form_context != TypeFormContext::ClassVarAnnotation => {
                    self.error(
                        errors,
                        x.range(),
                        ErrorKind::InvalidAnnotation,
                        None,
                        format!("{} is only allowed inside a class body.", special),
                    );
                    None
                }
                Some(Qualifier::Final)
                    if !matches!(
                        type_form_context,
                        TypeFormContext::ClassVarAnnotation | TypeFormContext::VarAnnotation(_),
                    ) =>
                {
                    self.error(
                        errors,
                        x.range(),
                        ErrorKind::InvalidAnnotation,
                        None,
                        format!(
                            "{} is only allowed on a class or local variable annotation.",
                            special
                        ),
                    );
                    None
                }
                Some(Qualifier::TypeAlias)
                    if !matches!(
                        type_form_context,
                        TypeFormContext::VarAnnotation(_) | TypeFormContext::ClassVarAnnotation
                    ) =>
                {
                    self.error(
                        errors,
                        x.range(),
                        ErrorKind::InvalidAnnotation,
                        None,
                        "TypeAlias is only allowed on variable annotations.".to_owned(),
                    );
                    None
                }
                _ => qualifier,
            }
        } else {
            None
        }
    }

    fn is_valid_annotation(x: &Expr, errors: &ErrorCollector) -> bool {
        // Note that this function only checks for correct syntax.
        // Semantic validation (e.g. that `typing.Self` is used in a class
        // context, or that a string evaluates to a proper type expression) is
        // handled elsewhere.
        // See https://typing.readthedocs.io/en/latest/spec/annotations.html#type-and-annotation-expressions
        let problem = match x {
            Expr::Name(..)
            | Expr::BinOp(ruff_python_ast::ExprBinOp {
                op: ruff_python_ast::Operator::BitOr,
                ..
            })
            | Expr::Named(..)
            | Expr::StringLiteral(..)
            | Expr::NoneLiteral(..)
            | Expr::Attribute(..)
            | Expr::Starred(..) => return true,
            Expr::Subscript(s) => match *s.value {
                Expr::Name(..)
                | Expr::BinOp(ruff_python_ast::ExprBinOp {
                    op: ruff_python_ast::Operator::BitOr,
                    ..
                })
                | Expr::Named(..)
                | Expr::StringLiteral(..)
                | Expr::NoneLiteral(..)
                | Expr::Attribute(..) => return true,
                _ => "invalid subscript expression",
            },
            Expr::Call(..) => "function call",
            Expr::Lambda(..) => "lambda definition",
            Expr::List(..) => "list literal",
            Expr::NumberLiteral(..) => "number literal",
            Expr::Tuple(..) => "tuple literal",
            Expr::Dict(..) => "dict literal",
            Expr::ListComp(..) => "list comprehension",
            Expr::If(..) => "if expression",
            Expr::BooleanLiteral(..) => "bool literal",
            Expr::BoolOp(..) => "boolean operation",
            Expr::FString(..) => "f-string",
            Expr::UnaryOp(..) => "unary operation",
            // There are many Expr variants. Not all of them are likely to be used
            // in annotations, even accidentally. We can add branches for specific
            // expression constructs if desired.
            _ => "expression",
        };
        errors.add(
            x.range(),
            format!("{problem} cannot be used in annotations"),
            ErrorKind::InvalidAnnotation,
            None,
        );
        false
    }

    fn expr_annotation(
        &self,
        x: &Expr,
        type_form_context: TypeFormContext,
        errors: &ErrorCollector,
    ) -> Annotation {
        if !Self::is_valid_annotation(x, errors) {
            return Annotation::new_type(Type::any_error());
        }
        match x {
            _ if let Some(qualifier) = self.expr_qualifier(x, type_form_context, errors) => {
                match qualifier {
                    Qualifier::TypeAlias | Qualifier::ClassVar => {}
                    // A local variable annotated assignment is only allowed to have an un-parameterized
                    // Final annotation if it's initialized with a value
                    Qualifier::Final
                        if !matches!(
                            type_form_context,
                            TypeFormContext::VarAnnotation(Initialized::No)
                        ) => {}
                    _ => {
                        self.error(
                            errors,
                            x.range(),
                            ErrorKind::InvalidAnnotation,
                            None,
                            format!("Expected a type argument for `{}`", qualifier,),
                        );
                    }
                }
                Annotation {
                    qualifiers: vec![qualifier],
                    ty: None,
                }
            }
            Expr::Subscript(x)
                if let unpacked_slice = Ast::unpack_slice(&x.slice)
                    && !unpacked_slice.is_empty()
                    && let Some(qualifier) =
                        self.expr_qualifier(&x.value, type_form_context, errors) =>
            {
                if qualifier == Qualifier::Annotated {
                    // TODO: we may want to preserve the extra annotation info for `Annotated` in the future
                    if unpacked_slice.len() < 2 {
                        self.error(
                            errors,
                            x.range(),
                            ErrorKind::InvalidAnnotation,
                            None,
                            "`Annotated` needs at least one piece of metadata in addition to the type".to_owned(),
                        );
                    }
                } else if unpacked_slice.len() != 1 {
                    self.error(
                        errors,
                        x.range(),
                        ErrorKind::InvalidAnnotation,
                        None,
                        format!(
                            "Expected 1 type argument for {}, got {}",
                            qualifier,
                            unpacked_slice.len()
                        ),
                    );
                }
                let mut ann = self.expr_annotation(&unpacked_slice[0], type_form_context, errors);
                if qualifier == Qualifier::ClassVar && ann.get_type().any(|x| x.is_type_variable())
                {
                    self.error(
                        errors,
                        unpacked_slice[0].range(),
                        ErrorKind::InvalidAnnotation,
                        None,
                        "`ClassVar` arguments may not contain any type variables".to_owned(),
                    );
                }
                if qualifier == Qualifier::Final && ann.is_class_var() {
                    self.error(
                        errors,
                        unpacked_slice[0].range(),
                        ErrorKind::InvalidAnnotation,
                        None,
                        "`ClassVar` may not be nested inside `Final`".to_owned(),
                    );
                }
                ann.qualifiers.insert(0, qualifier);
                ann
            }
            _ => {
                let ann_ty = self.expr_untype(x, type_form_context, errors);
                if let Type::SpecialForm(special_form) = ann_ty
                    && !special_form.is_valid_unparameterized_annotation(type_form_context)
                {
                    self.error(
                        errors,
                        x.range(),
                        ErrorKind::InvalidAnnotation,
                        None,
                        format!("Expected a type argument for `{}`", special_form),
                    );
                }
                Annotation::new_type(ann_ty)
            }
        }
    }

    /// Given an `iterable` type, determine the iteration type; this is the type
    /// of `x` if we were to loop using `for x in iterable`.
    pub fn iterate(
        &self,
        iterable: &Type,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Vec<Iterable> {
        // Use the iterable protocol interfaces to determine the iterable type.
        // Special cases like Tuple should be intercepted first.
        let context = || ErrorContext::Iteration(self.for_display(iterable.clone()));
        match iterable {
            Type::ClassType(cls) if let Some(elts) = self.named_tuple_element_types(cls) => {
                vec![Iterable::FixedLen(elts.clone())]
            }
            Type::Tuple(Tuple::Concrete(elts)) => vec![Iterable::FixedLen(elts.clone())],
            Type::Var(v) if let Some(_guard) = self.recurser.recurse(*v) => {
                self.iterate(&self.solver().force_var(*v), range, errors)
            }
            Type::Union(ts) => ts
                .iter()
                .flat_map(|t| self.iterate(t, range, errors))
                .collect(),
            _ => {
                let ty = self
                    .unwrap_iterable(iterable)
                    .or_else(|| {
                        let int_ty = self.stdlib.int().clone().to_type();
                        let arg = CallArg::Type(&int_ty, range);
                        self.call_magic_dunder_method(
                            iterable,
                            &dunder::GETITEM,
                            range,
                            &[arg],
                            &[],
                            errors,
                            Some(&context),
                        )
                    })
                    .unwrap_or_else(|| {
                        self.error(
                            errors,
                            range,
                            ErrorKind::NotIterable,
                            None,
                            context().format(),
                        )
                    });
                vec![Iterable::OfType(ty)]
            }
        }
    }

    /// Given a type, determine the async iteration type; this is the type
    /// of `x` if we were to loop using `async for x in iterable`.
    pub fn async_iterate(
        &self,
        iterable: &Type,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Vec<Iterable> {
        match iterable {
            Type::Var(v) if let Some(_guard) = self.recurser.recurse(*v) => {
                self.async_iterate(&self.solver().force_var(*v), range, errors)
            }
            _ => {
                let context = || ErrorContext::AsyncIteration(self.for_display(iterable.clone()));
                let ty = self.unwrap_async_iterable(iterable).unwrap_or_else(|| {
                    self.error(
                        errors,
                        range,
                        ErrorKind::NotIterable,
                        None,
                        context().format(),
                    )
                });
                vec![Iterable::OfType(ty)]
            }
        }
    }

    fn check_is_exception(
        &self,
        x: &Expr,
        range: TextRange,
        allow_none: bool,
        errors: &ErrorCollector,
    ) {
        let actual_type = self.expr_infer(x, errors);
        if allow_none && actual_type.is_none() {
            return;
        }
        let base_exception_class = self.stdlib.base_exception();
        let base_exception_class_type = Type::ClassDef(base_exception_class.class_object().dupe());
        let base_exception_type = base_exception_class.clone().to_type();
        let expected_types = vec![base_exception_type, base_exception_class_type];
        if !self.is_subset_eq(&actual_type, &Type::Union(expected_types)) {
            self.error(
                errors,
                range,
                ErrorKind::InvalidInheritance,
                None,
                format!(
                    "Expression `{}` has type `{}` which does not derive from BaseException",
                    self.module_info().display(x),
                    self.for_display(actual_type),
                ),
            );
        }
    }

    fn tvars_to_tparams_for_type_alias(
        &self,
        ty: &mut Type,
        seen_type_vars: &mut SmallMap<TypeVar, Quantified>,
        seen_type_var_tuples: &mut SmallMap<TypeVarTuple, Quantified>,
        seen_param_specs: &mut SmallMap<ParamSpec, Quantified>,
        tparams: &mut Vec<TParamInfo>,
    ) {
        match ty {
            Type::Union(ts) => {
                for t in ts.iter_mut() {
                    self.tvars_to_tparams_for_type_alias(
                        t,
                        seen_type_vars,
                        seen_type_var_tuples,
                        seen_param_specs,
                        tparams,
                    );
                }
            }
            Type::ClassType(cls) => {
                for t in cls.targs_mut().as_mut() {
                    self.tvars_to_tparams_for_type_alias(
                        t,
                        seen_type_vars,
                        seen_type_var_tuples,
                        seen_param_specs,
                        tparams,
                    );
                }
            }
            Type::Callable(box callable)
            | Type::Function(box Function {
                signature: callable,
                metadata: _,
            }) => {
                let mut visit = |t: &mut Type| {
                    self.tvars_to_tparams_for_type_alias(
                        t,
                        seen_type_vars,
                        seen_type_var_tuples,
                        seen_param_specs,
                        tparams,
                    )
                };
                callable.recurse_mut(&mut visit);
            }
            Type::Concatenate(box prefix, box pspec) => {
                for t in prefix {
                    self.tvars_to_tparams_for_type_alias(
                        t,
                        seen_type_vars,
                        seen_type_var_tuples,
                        seen_param_specs,
                        tparams,
                    )
                }
                self.tvars_to_tparams_for_type_alias(
                    pspec,
                    seen_type_vars,
                    seen_type_var_tuples,
                    seen_param_specs,
                    tparams,
                )
            }
            Type::Tuple(tuple) => {
                let mut visit = |t: &mut Type| {
                    self.tvars_to_tparams_for_type_alias(
                        t,
                        seen_type_vars,
                        seen_type_var_tuples,
                        seen_param_specs,
                        tparams,
                    )
                };
                tuple.recurse_mut(&mut visit);
            }
            Type::TypeVar(ty_var) => {
                let q = match seen_type_vars.entry(ty_var.dupe()) {
                    Entry::Occupied(e) => e.get().clone(),
                    Entry::Vacant(e) => {
                        let q = Quantified::type_var(
                            ty_var.qname().id().clone(),
                            self.uniques,
                            ty_var.default().cloned(),
                            ty_var.restriction().clone(),
                        );
                        e.insert(q.clone());
                        tparams.push(TParamInfo {
                            quantified: q.clone(),
                            variance: ty_var.variance(),
                        });
                        q
                    }
                };
                *ty = Type::Quantified(q);
            }
            Type::TypeVarTuple(ty_var_tuple) => {
                let q = match seen_type_var_tuples.entry(ty_var_tuple.dupe()) {
                    Entry::Occupied(e) => e.get().clone(),
                    Entry::Vacant(e) => {
                        let q = Quantified::type_var_tuple(
                            ty_var_tuple.qname().id().clone(),
                            self.uniques,
                            ty_var_tuple.default().cloned(),
                        );
                        e.insert(q.clone());
                        tparams.push(TParamInfo {
                            quantified: q.clone(),
                            variance: Some(Variance::Invariant),
                        });
                        q
                    }
                };
                *ty = Type::Quantified(q);
            }
            Type::ParamSpec(param_spec) => {
                let q = match seen_param_specs.entry(param_spec.dupe()) {
                    Entry::Occupied(e) => e.get().clone(),
                    Entry::Vacant(e) => {
                        let q = Quantified::param_spec(
                            param_spec.qname().id().clone(),
                            self.uniques,
                            param_spec.default().cloned(),
                        );
                        e.insert(q.clone());
                        tparams.push(TParamInfo {
                            quantified: q.clone(),
                            variance: Some(Variance::Invariant),
                        });
                        q
                    }
                };
                *ty = Type::Quantified(q);
            }
            Type::Unpack(box t) => self.tvars_to_tparams_for_type_alias(
                t,
                seen_type_vars,
                seen_type_var_tuples,
                seen_param_specs,
                tparams,
            ),
            _ => {}
        }
    }

    fn as_type_alias(
        &self,
        name: &Name,
        style: TypeAliasStyle,
        ty: Type,
        expr: &Expr,
        errors: &ErrorCollector,
    ) -> Type {
        let range = expr.range();
        if !Self::is_valid_annotation(expr, errors) {
            return Type::any_error();
        }
        let untyped = self.untype_opt(ty.clone(), range);
        let mut ty = if let Type::ClassDef(cls) = ty {
            // TODO: should we be promoting this or making a Forall type?
            self.promote(&cls, range)
        } else if let Some(untyped) = untyped {
            let validated =
                self.validate_type_form(untyped, range, TypeFormContext::TypeAlias, errors);
            if validated.is_error() {
                return validated;
            }
            validated
        } else {
            self.error(
                errors,
                range,
                ErrorKind::TypeAliasError,
                None,
                format!("Expected `{name}` to be a type alias, got {ty}"),
            );
            return Type::any_error();
        };
        let mut seen_type_vars = SmallMap::new();
        let mut seen_type_var_tuples = SmallMap::new();
        let mut seen_param_specs = SmallMap::new();
        let mut tparams = Vec::new();
        self.tvars_to_tparams_for_type_alias(
            &mut ty,
            &mut seen_type_vars,
            &mut seen_type_var_tuples,
            &mut seen_param_specs,
            &mut tparams,
        );
        let ta = TypeAlias::new(name.clone(), Type::type_form(ty), style);
        Forallable::TypeAlias(ta).forall(self.type_params(range, tparams, errors))
    }

    fn context_value_enter(
        &self,
        context_manager_type: &Type,
        kind: IsAsync,
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) -> Type {
        match kind {
            IsAsync::Sync => self.call_method_or_error(
                context_manager_type,
                &dunder::ENTER,
                range,
                &[],
                &[],
                errors,
                context,
            ),
            IsAsync::Async => match self.unwrap_awaitable(&self.call_method_or_error(
                context_manager_type,
                &dunder::AENTER,
                range,
                &[],
                &[],
                errors,
                context,
            )) {
                Some(ty) => ty,
                None => self.error(
                    errors,
                    range,
                    ErrorKind::AsyncError,
                    context,
                    format!("Expected `{}` to be async", dunder::AENTER),
                ),
            },
        }
    }

    fn context_value_exit(
        &self,
        context_manager_type: &Type,
        kind: IsAsync,
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) -> Type {
        let base_exception_class_type =
            Type::type_form(self.stdlib.base_exception().clone().to_type());
        let arg1 = Type::Union(vec![base_exception_class_type, Type::None]);
        let arg2 = Type::Union(vec![
            self.stdlib.base_exception().clone().to_type(),
            Type::None,
        ]);
        let arg3 = Type::Union(vec![
            self.stdlib.traceback_type().clone().to_type(),
            Type::None,
        ]);
        let exit_arg_types = [
            CallArg::Type(&arg1, range),
            CallArg::Type(&arg2, range),
            CallArg::Type(&arg3, range),
        ];
        match kind {
            IsAsync::Sync => self.call_method_or_error(
                context_manager_type,
                &kind.context_exit_dunder(),
                range,
                &exit_arg_types,
                &[],
                errors,
                context,
            ),
            IsAsync::Async => match self.unwrap_awaitable(&self.call_method_or_error(
                context_manager_type,
                &kind.context_exit_dunder(),
                range,
                &exit_arg_types,
                &[],
                errors,
                context,
            )) {
                Some(ty) => ty,
                None => self.error(
                    errors,
                    range,
                    ErrorKind::AsyncError,
                    context,
                    format!("Expected `{}` to be async", dunder::AEXIT),
                ),
            },
        }
    }

    fn context_value(
        &self,
        context_manager_type: &Type,
        kind: IsAsync,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        self.distribute_over_union(context_manager_type, |context_manager_type| {
            let context =
                || ErrorContext::BadContextManager(self.for_display(context_manager_type.clone()));
            let enter_type =
                self.context_value_enter(context_manager_type, kind, range, errors, Some(&context));
            let exit_type =
                self.context_value_exit(context_manager_type, kind, range, errors, Some(&context));
            self.check_type(
                &Type::Union(vec![self.stdlib.bool().clone().to_type(), Type::None]),
                &exit_type,
                range,
                errors,
                &|| TypeCheckContext {
                    kind: TypeCheckKind::MagicMethodReturn(
                        self.for_display(context_manager_type.clone()),
                        kind.context_exit_dunder(),
                    ),
                    context: Some(context()),
                },
            );
            // TODO: `exit_type` may also affect exceptional control flow, which is yet to be supported:
            // https://typing.readthedocs.io/en/latest/spec/exceptions.html#context-managers
            enter_type
        })
    }

    pub fn scoped_type_params(
        &self,
        x: Option<&TypeParams>,
        errors: &ErrorCollector,
    ) -> Vec<TParamInfo> {
        match x {
            Some(x) => {
                fn get_quantified(t: &Type) -> Quantified {
                    match t {
                        Type::Type(box Type::Quantified(q)) => q.clone(),
                        _ => unreachable!(),
                    }
                }
                let mut type_var_tuple_count = 0;
                let mut params = Vec::new();
                for raw_param in x.type_params.iter() {
                    if matches!(raw_param, TypeParam::TypeVarTuple(_)) {
                        if type_var_tuple_count == 1 {
                            self.error(
                                errors,
                                raw_param.range(),
                                ErrorKind::InvalidTypeVarTuple,
                                None,
                                "There cannot be more than one TypeVarTuple type parameter"
                                    .to_owned(),
                            );
                        }
                        type_var_tuple_count += 1;
                    }
                    let name = raw_param.name();
                    let quantified =
                        get_quantified(self.get(&Key::Definition(ShortIdentifier::new(name))).ty());
                    params.push(TParamInfo {
                        quantified,
                        variance: None,
                    });
                }
                params
            }
            None => Vec::new(),
        }
    }

    pub fn type_params(
        &self,
        range: TextRange,
        info: Vec<TParamInfo>,
        errors: &ErrorCollector,
    ) -> TParams {
        let validated_tparams = TParams::new(info);
        for error in validated_tparams.errors {
            self.error(errors, range, ErrorKind::InvalidTypeVar, None, error);
        }
        validated_tparams.tparams
    }

    pub fn solve_binding(&self, binding: &Binding, errors: &ErrorCollector) -> Arc<TypeInfo> {
        // Special case for forward, as we don't want to re-expand the type
        if let Binding::Forward(fwd) = binding {
            return self.get_idx(*fwd);
        }
        let mut type_info = self.binding_to_type_info(binding, errors);
        type_info.visit_mut(&mut |ty| {
            self.expand_type_mut(ty);
        });
        Arc::new(type_info)
    }

    pub fn expand_type_mut(&self, ty: &mut Type) {
        // Replace any solved recursive variables with their answers.
        // We call self.unions() to simplify cases like
        // v = @1 | int, @1 = int.
        self.solver().expand_mut(ty);
        if let Type::Union(tys) = ty {
            *ty = self.unions(mem::take(tys));
        }
    }

    pub fn solve_expectation(
        &self,
        binding: &BindingExpect,
        errors: &ErrorCollector,
    ) -> Arc<EmptyAnswer> {
        match binding {
            BindingExpect::TypeCheckExpr(box x) => {
                self.expr_infer(x, errors);
            }
            BindingExpect::Delete(box x) => match x {
                Expr::Name(_) => {
                    self.expr_infer(x, errors);
                }
                Expr::Attribute(attr) => {
                    let base = self.expr_infer(&attr.value, errors);
                    self.check_attr_delete(
                        &base,
                        &attr.attr.id,
                        attr.range,
                        errors,
                        None,
                        "Answers::solve_expectation::Delete",
                    );
                }
                Expr::Subscript(x) => {
                    let base = self.expr_infer(&x.value, errors);
                    let slice_ty = self.expr_infer(&x.slice, errors);
                    match (&base, &slice_ty) {
                        (Type::TypedDict(typed_dict), Type::Literal(Lit::Str(field_name))) => {
                            if let Some(field) =
                                self.typed_dict_field(typed_dict, &Name::new(field_name))
                            {
                                if field.read_only || field.required {
                                    self.error(
                                        errors,
                                        x.slice.range(),
                                        ErrorKind::DeleteError,
                                        None,
                                        format!(
                                            "Key `{}` in TypedDict `{}` may not be deleted",
                                            field_name,
                                            typed_dict.name(),
                                        ),
                                    );
                                }
                            } else {
                                self.error(
                                    errors,
                                    x.slice.range(),
                                    ErrorKind::TypedDictKeyError,
                                    None,
                                    format!(
                                        "TypedDict `{}` does not have key `{}`",
                                        typed_dict.name(),
                                        field_name
                                    ),
                                );
                            }
                        }
                        (_, _) => {
                            self.call_method_or_error(
                                &base,
                                &dunder::DELITEM,
                                x.range,
                                &[CallArg::Type(&slice_ty, x.slice.range())],
                                &[],
                                errors,
                                Some(&|| ErrorContext::DelItem(self.for_display(base.clone()))),
                            );
                        }
                    }
                }
                _ => {
                    self.error(
                        errors,
                        x.range(),
                        ErrorKind::DeleteError,
                        None,
                        "Invalid target for `del`".to_owned(),
                    );
                }
            },
            BindingExpect::UnpackedLength(b, range, expect) => {
                let iterable_ty = self.get_idx(*b);
                let iterables = self.iterate(iterable_ty.ty(), *range, errors);
                for iterable in iterables {
                    match iterable {
                        Iterable::OfType(_) => {}
                        Iterable::FixedLen(ts) => {
                            let error = match expect {
                                SizeExpectation::Eq(n) => {
                                    if ts.len() == *n {
                                        None
                                    } else {
                                        match n {
                                            1 => Some(format!("{n} value")),
                                            _ => Some(format!("{n} values")),
                                        }
                                    }
                                }
                                SizeExpectation::Ge(n) => {
                                    if ts.len() >= *n {
                                        None
                                    } else {
                                        Some(format!("{n}+ values"))
                                    }
                                }
                            };
                            match error {
                                Some(expectation) => {
                                    self.error(
                                        errors,
                                        *range,
                                        ErrorKind::BadUnpacking,
                                        None,
                                        format!(
                                            "Cannot unpack {} (of size {}) into {}",
                                            iterable_ty,
                                            ts.len(),
                                            expectation,
                                        ),
                                    );
                                }
                                None => {}
                            }
                        }
                    }
                }
            }
            BindingExpect::CheckRaisedException(RaisedException::WithoutCause(exc)) => {
                self.check_is_exception(exc, exc.range(), false, errors);
            }
            BindingExpect::CheckRaisedException(RaisedException::WithCause(box (exc, cause))) => {
                self.check_is_exception(exc, exc.range(), false, errors);
                self.check_is_exception(cause, cause.range(), true, errors);
            }
            BindingExpect::Redefinition {
                new,
                existing,
                name,
            } => {
                let ann_new = self.get_idx(*new);
                let ann_existing = self.get_idx(*existing);
                if let Some(t_new) = ann_new.ty(self.stdlib)
                    && let Some(t_existing) = ann_existing.ty(self.stdlib)
                    && t_new != t_existing
                {
                    let t_new = self.for_display(t_new.clone());
                    let t_existing = self.for_display(t_existing.clone());
                    let ctx = TypeDisplayContext::new(&[&t_new, &t_existing]);
                    self.error(
                        errors,
                        self.bindings().idx_to_key(*new).range(),
                        ErrorKind::AnnotationMismatch,
                        None,
                        format!(
                            "`{}` cannot be annotated with `{}`, it is already defined with type `{}`",
                            name,
                            ctx.display(&t_new),
                            ctx.display(&t_existing),
                        ),
                    );
                }
            }
        }
        Arc::new(EmptyAnswer)
    }

    pub fn solve_class(
        &self,
        cls: &BindingClass,
        errors: &ErrorCollector,
    ) -> Arc<NoneIfRecursive<Class>> {
        let cls = match cls {
            BindingClass::ClassDef(x) => self.class_definition(
                x.index,
                &x.def,
                x.fields.clone(),
                &x.bases,
                &x.legacy_tparams,
                errors,
            ),
            BindingClass::FunctionalClassDef(index, x, fields) => {
                self.functional_class_definition(*index, x, fields)
            }
        };
        Arc::new(NoneIfRecursive(Some(cls)))
    }

    pub fn solve_class_field(
        &self,
        field: &BindingClassField,
        errors: &ErrorCollector,
    ) -> Arc<ClassField> {
        let field = match &self.get_idx(field.class).0 {
            None => ClassField::recursive(),
            Some(class) => {
                let annotation = field.annotation.map(|a| self.get_idx(a));
                self.calculate_class_field(
                    &field.name,
                    &field.value,
                    annotation.as_deref().map(|annot| &annot.annotation),
                    &field.initial_value,
                    class,
                    field.is_function_without_return_annotation,
                    field.range,
                    errors,
                )
            }
        };
        Arc::new(field)
    }

    pub fn solve_class_synthesized_fields(
        &self,
        fields: &BindingClassSynthesizedFields,
    ) -> Arc<ClassSynthesizedFields> {
        let fields = match &self.get_idx(fields.0).0 {
            None => ClassSynthesizedFields::default(),
            Some(cls) => self
                .get_typed_dict_synthesized_fields(cls)
                .or_else(|| self.get_dataclass_synthesized_fields(cls))
                .or_else(|| self.get_named_tuple_synthesized_fields(cls))
                .or_else(|| self.get_new_type_synthesized_fields(cls))
                .unwrap_or_default(),
        };
        Arc::new(fields)
    }

    /// Get the class that attribute lookup on `super(cls, obj)` should be done on.
    /// This is the class above `cls` in `obj`'s MRO.
    fn get_super_lookup_class(&self, cls: &Class, obj: &ClassType) -> Option<ClassType> {
        let mut lookup_cls = None;
        let metadata = self.get_metadata_for_class(obj.class_object());
        let mut found = false;
        for ancestor in iter::once(obj).chain(metadata.ancestors(self.stdlib)) {
            if ancestor.class_object() == cls {
                found = true;
                // Handle the corner case of `ancestor` being `object` (and
                // therefore having no ancestor of its own).
                lookup_cls = Some(ancestor);
            } else if found {
                lookup_cls = Some(ancestor);
                break;
            }
        }
        lookup_cls.cloned()
    }

    fn solve_super_binding(
        &self,
        style: &SuperStyle,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        match style {
            SuperStyle::ExplicitArgs(cls_binding, obj_binding) => {
                match self.get_idx(*cls_binding).ty() {
                    Type::Any(style) => style.propagate(),
                    cls_type @ Type::ClassDef(cls) => {
                        let make_super_instance = |obj_cls, super_obj: &dyn Fn() -> SuperObj| {
                            let lookup_cls = self.get_super_lookup_class(cls, obj_cls);
                            lookup_cls.map_or_else (
                                || {
                                    let cls_type = self.for_display(cls_type.clone());
                                    self.error(
                                        errors,
                                        range,
                                        ErrorKind::InvalidSuperCall,
                                        None,
                                        format!(
                                            "Illegal `super({cls_type}, {obj_cls})` call: `{obj_cls}` is not an instance or subclass of `{cls_type}`"
                                        ),
                                    )
                                },
                                |lookup_cls| {
                                    Type::SuperInstance(Box::new((lookup_cls, super_obj())))
                                }
                            )
                        };
                        match self.get_idx(*obj_binding).ty() {
                            Type::Any(style) => style.propagate(),
                            Type::ClassType(obj_cls) => make_super_instance(obj_cls, &|| SuperObj::Instance(obj_cls.clone())),
                            Type::Type(box Type::ClassType(obj_cls)) => make_super_instance(obj_cls, &|| SuperObj::Class(obj_cls.class_object().dupe())),
                            Type::ClassDef(obj_cls) => {
                                let obj_type = obj_cls.as_class_type();
                                make_super_instance(&obj_type, &|| SuperObj::Class(obj_cls.dupe()))
                            }
                            Type::SelfType(obj_cls) => {
                                make_super_instance(obj_cls, &|| SuperObj::Instance(obj_cls.clone()))
                            }
                            Type::Type(box Type::SelfType(obj_cls)) => {
                                make_super_instance(obj_cls, &|| SuperObj::Class(obj_cls.class_object().dupe()))
                            }
                            t => {
                                self.error(
                                    errors,
                                    range,
                                    ErrorKind::InvalidArgument,
                                    None,
                                    format!("Expected second argument to `super` to be a class object or instance, got `{}`", self.for_display(t.clone())),
                                )
                            }
                        }
                    }
                    t => self.error(
                        errors,
                        range,
                        ErrorKind::InvalidArgument,
                        None,
                        format!(
                            "Expected first argument to `super` to be a class object, got `{}`",
                            self.for_display(t.clone())
                        ),
                    ),
                }
            }
            SuperStyle::ImplicitArgs(self_binding, method) => {
                match &self.get_idx(*self_binding).0 {
                    Some(obj_cls) => {
                        let obj_type = obj_cls.as_class_type();
                        let lookup_cls = self.get_super_lookup_class(obj_cls, &obj_type).unwrap();
                        let obj = if method.id == dunder::NEW {
                            // __new__ is special: it's the only static method in which the
                            // no-argument form of super is allowed.
                            SuperObj::Class(obj_cls.dupe())
                        } else {
                            let method_ty = self.get(&KeyFunction(ShortIdentifier::new(method)));
                            if method_ty.metadata.flags.is_staticmethod {
                                return self.error(
                                    errors,
                                    range,
                                    ErrorKind::InvalidSuperCall,
                                    None,
                                    "`super` call with no arguments is not valid inside a staticmethod".to_owned(),
                                );
                            } else if method_ty.metadata.flags.is_classmethod {
                                SuperObj::Class(obj_cls.dupe())
                            } else {
                                SuperObj::Instance(obj_type)
                            }
                        };
                        Type::SuperInstance(Box::new((lookup_cls, obj)))
                    }
                    None => Type::any_implicit(),
                }
            }
            SuperStyle::Any => Type::any_implicit(),
        }
    }

    pub fn validate_type_var_default(
        &self,
        name: &Name,
        kind: QuantifiedKind,
        default: &Type,
        range: TextRange,
        restriction: &Restriction,
        errors: &ErrorCollector,
    ) -> Type {
        if default.is_error() {
            return default.clone();
        }
        match restriction {
            // Default must be a subtype of the upper bound
            Restriction::Bound(bound_ty) => {
                if !self
                    .solver()
                    .is_subset_eq(default, bound_ty, self.type_order())
                {
                    self.error(
                        errors,
                        range,
                        kind.error_kind(),
                        None,
                        format!(
                            "Expected default `{}` of `{}` to be assignable to the upper bound of `{}`",
                            default,
                            name,
                            bound_ty,
                        ),
                    );
                    return Type::any_error();
                }
            }
            Restriction::Constraints(constraints) => {
                // Default must exactly match one of the constraints
                if !constraints
                    .iter()
                    .any(|c| self.is_subset_eq(c, default) && self.is_subset_eq(default, c))
                {
                    let formatted_constraints = constraints
                        .iter()
                        .map(|x| format!("`{}`", x))
                        .collect::<Vec<_>>()
                        .join(", ");
                    self.error(
                        errors,
                        range,
                        kind.error_kind(),
                        None,
                        format!(
                            "Expected default `{}` of `{}` to be one of the following constraints: {}",
                            default,
                            name,
                            formatted_constraints,
                        ),
                    );
                    return Type::any_error();
                }
            }
            Restriction::Unrestricted => {}
        };
        match kind {
            QuantifiedKind::ParamSpec => {
                if default.is_kind_param_spec() {
                    default.clone()
                } else {
                    self.error(
                        errors,
                        range,
                        ErrorKind::InvalidParamSpec,
                        None,
                        format!("Default for ParamSpec must be a parameter list, `...`, or another ParamSpec, got `{}`", default),
                    );
                    Type::any_error()
                }
            }
            QuantifiedKind::TypeVarTuple => {
                if let Type::Unpack(box inner) = default
                    && (matches!(inner, Type::Tuple(_)) || inner.is_kind_type_var_tuple())
                {
                    inner.clone()
                } else {
                    self.error(
                        errors,
                        range,
                        ErrorKind::InvalidTypeVarTuple,
                        None,
                        format!("Default for TypeVarTuple must be an unpacked tuple form or another TypeVarTuple, got `{}`", default),
                    );
                    Type::any_error()
                }
            }
            QuantifiedKind::TypeVar => {
                if default.is_kind_param_spec() || default.is_kind_type_var_tuple() {
                    self.error(
                        errors,
                        range,
                        ErrorKind::InvalidTypeVar,
                        None,
                        format!(
                            "Default for TypeVar may not be a TypeVarTuple or ParamSpec, got `{}`",
                            default
                        ),
                    );
                    Type::any_error()
                } else {
                    default.clone()
                }
            }
        }
    }

    fn binding_to_type_info(&self, binding: &Binding, errors: &ErrorCollector) -> TypeInfo {
        match binding {
            Binding::Forward(k) => self.get_idx(*k).arc_clone(),
            Binding::Narrow(k, op, range) => {
                self.narrow(self.get_idx(*k).as_ref(), op, *range, errors)
            }
            Binding::Phi(ks) => {
                if ks.len() == 1 {
                    self.get_idx(*ks.first().unwrap()).arc_clone()
                } else {
                    let type_infos = ks
                        .iter()
                        .filter_map(|k| {
                            let t: Arc<TypeInfo> = self.get_idx(*k);
                            // Filter out all `@overload`-decorated types except the one that
                            // accumulates all signatures into a Type::Overload.
                            if matches!(t.ty(), Type::Overload(_)) || !t.ty().is_overload() {
                                Some(t.arc_clone())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>();
                    TypeInfo::join(type_infos, &|ts| self.unions(ts))
                }
            }
            Binding::Default(default, binding) => {
                // We force the default first so that if we hit a recursive case it is already available
                self.get_idx(*default);
                self.binding_to_type_info(binding, errors)
            }
            Binding::AssignToAttribute(box (attr, got)) => {
                let base = self.expr_infer(&attr.value, errors);
                let narrowed = self.check_assign_to_attribute_and_infer_narrow(
                    &base,
                    &attr.attr.id,
                    got,
                    attr.range,
                    errors,
                );
                if let Some((identifier, chain)) =
                    identifier_and_chain_for_property(&Expr::Attribute(attr.clone()))
                {
                    let mut type_info = self
                        .get(&Key::Usage(ShortIdentifier::new(&identifier)))
                        .arc_clone();
                    type_info.update_for_assignment(chain.properties(), narrowed);
                    type_info
                } else if let Some((identifier, properties)) =
                    identifier_and_chain_prefix_for_property(&Expr::Attribute(attr.clone()))
                {
                    // If the chain contains an unknown subscript index, we clear narrowing for
                    // all indexes of its parent.
                    let mut type_info = self
                        .get(&Key::Usage(ShortIdentifier::new(&identifier)))
                        .arc_clone();
                    type_info.invalidate_all_indexes_for_assignment(&properties);
                    type_info
                } else {
                    // Placeholder: in this case, we're assigning to an anonymous base and the
                    // type info will not propagate anywhere.
                    TypeInfo::of_ty(Type::never())
                }
            }
            Binding::AssignToSubscript(box (subscript, value)) => {
                let value_ty = match value {
                    ExprOrBinding::Expr(e) => self.expr_infer(e, errors),
                    ExprOrBinding::Binding(b) => self.solve_binding(b, errors).arc_clone_ty(),
                };
                // If we can't assign to this subscript, then we don't narrow the type
                let narrowed = if self.check_assign_to_subscript(subscript, &value_ty, errors)
                    == Type::any_error()
                {
                    None
                } else {
                    Some(value_ty)
                };
                if let Some((identifier, chain)) =
                    identifier_and_chain_for_property(&Expr::Subscript(subscript.clone()))
                {
                    let mut type_info = self
                        .get(&Key::Usage(ShortIdentifier::new(&identifier)))
                        .arc_clone();
                    type_info.update_for_assignment(chain.properties(), narrowed);
                    type_info
                } else if let Some((identifier, properties)) =
                    identifier_and_chain_prefix_for_property(&Expr::Subscript(subscript.clone()))
                {
                    // If the chain contains an unknown subscript index, we clear narrowing for
                    // all indexes of its parent.
                    let mut type_info = self
                        .get(&Key::Usage(ShortIdentifier::new(&identifier)))
                        .arc_clone();
                    type_info.invalidate_all_indexes_for_assignment(&properties);
                    type_info
                } else {
                    // Placeholder: in this case, we're assigning to an anonymous base and the
                    // type info will not propagate anywhere.
                    TypeInfo::of_ty(Type::never())
                }
            }
            _ => {
                // All other Bindings model `Type` level operations where we do not
                // propagate any attribute narrows.
                TypeInfo::of_ty(self.binding_to_type(binding, errors))
            }
        }
    }

    fn check_assign_to_subscript(
        &self,
        subscript: &ExprSubscript,
        value: &Type,
        errors: &ErrorCollector,
    ) -> Type {
        let base = self.expr_infer(&subscript.value, errors);
        let slice_ty = self.expr_infer(&subscript.slice, errors);
        match (&base, &slice_ty) {
            (Type::TypedDict(typed_dict), Type::Literal(Lit::Str(field_name))) => {
                if let Some(field) = self.typed_dict_field(typed_dict, &Name::new(field_name)) {
                    if field.read_only {
                        self.error(
                            errors,
                            subscript.slice.range(),
                            ErrorKind::ReadOnly,
                            None,
                            format!(
                                "Key `{}` in TypedDict `{}` is read-only",
                                field_name,
                                typed_dict.name(),
                            ),
                        )
                    } else if !self.is_subset_eq(value, &field.ty) {
                        self.error(
                            errors,
                            subscript.range(),
                            ErrorKind::BadAssignment,
                            None,
                            format!("Expected {}, got {}", field.ty, value),
                        )
                    } else {
                        Type::None
                    }
                } else {
                    self.error(
                        errors,
                        subscript.slice.range(),
                        ErrorKind::TypedDictKeyError,
                        None,
                        format!(
                            "TypedDict `{}` does not have key `{}`",
                            typed_dict.name(),
                            field_name
                        ),
                    )
                }
            }
            (_, _) => self.call_method_or_error(
                &base,
                &dunder::SETITEM,
                subscript.range,
                &[
                    CallArg::Type(&slice_ty, subscript.slice.range()),
                    // use the subscript's location
                    CallArg::Type(value, subscript.range),
                ],
                &[],
                errors,
                Some(&|| ErrorContext::SetItem(self.for_display(base.clone()))),
            ),
        }
    }

    fn check_implicit_return_against_annotation(
        &self,
        implicit_return: Arc<TypeInfo>,
        annotation: &Type,
        is_async: bool,
        is_generator: bool,
        has_explicit_returns: bool,
        range: TextRange,
        errors: &ErrorCollector,
    ) {
        if is_async && is_generator {
            if self.decompose_async_generator(annotation).is_none() {
                self.error(
                    errors,
                    range,
                    ErrorKind::BadReturn,
                    None,
                    "Async generator function should return `AsyncGenerator`".to_owned(),
                );
            }
        } else if is_generator {
            if let Some((_, _, return_ty)) = self.decompose_generator(annotation) {
                self.check_type(&return_ty, implicit_return.ty(), range, errors, &|| {
                    TypeCheckContext::of_kind(TypeCheckKind::ImplicitFunctionReturn(
                        has_explicit_returns,
                    ))
                });
            } else {
                self.error(
                    errors,
                    range,
                    ErrorKind::BadReturn,
                    None,
                    "Generator function should return `Generator`".to_owned(),
                );
            }
        } else {
            self.check_type(annotation, implicit_return.ty(), range, errors, &|| {
                TypeCheckContext::of_kind(TypeCheckKind::ImplicitFunctionReturn(
                    has_explicit_returns,
                ))
            });
        }
    }

    fn may_be_implicit_type_alias(ty: &Type) -> bool {
        fn check_type_form(ty: &Type, allow_none: bool) -> bool {
            // TODO(stroxler, rechen): Do we want to include Type::ClassDef(_)
            // when there is no annotation, so that `mylist = list` is treated
            // like a value assignment rather than a type alias?
            match ty {
                Type::Type(_) => true,
                Type::None if allow_none => true,
                Type::Union(members) => {
                    for member in members {
                        // `None` can be part of an implicit type alias if it's
                        // part of a union. In other words, we treat
                        // `x = T | None` as a type alias, but not `x = None`
                        if !check_type_form(member, true) {
                            return false;
                        }
                    }
                    true
                }
                _ => false,
            }
        }
        check_type_form(ty, false)
    }

    fn binding_to_type(&self, binding: &Binding, errors: &ErrorCollector) -> Type {
        match binding {
            Binding::Forward(..)
            | Binding::Default(..)
            | Binding::Phi(..)
            | Binding::Narrow(..)
            | Binding::AssignToAttribute(..)
            | Binding::AssignToSubscript(..) => {
                // These forms require propagating attribute narrowing information, so they
                // are handled in `binding_to_type_info`
                self.binding_to_type_info(binding, errors).into_ty()
            }
            Binding::Expr(ann, e) => match ann {
                Some(k) => {
                    let annot = self.get_idx(*k);
                    let tcc: &dyn Fn() -> TypeCheckContext = &|| {
                        TypeCheckContext::of_kind(TypeCheckKind::from_annotation_target(
                            &annot.target,
                        ))
                    };
                    if annot.annotation.is_final() {
                        self.error(
                            errors,
                            e.range(),
                            ErrorKind::BadAssignment,
                            None,
                            "Assignment target is marked final".to_owned(),
                        );
                    }
                    self.expr(e, annot.ty(self.stdlib).as_ref().map(|t| (t, tcc)), errors)
                }
                None => {
                    // TODO(stroxler): propagate attribute narrows here
                    self.expr(e, None, errors)
                }
            },
            Binding::PatternMatchMapping(mapping_key, binding_key) => {
                // TODO: check that value is a mapping
                // TODO: check against duplicate keys (optional)
                let key_ty = self.expr_infer(mapping_key, errors);
                let binding = self.get_idx(*binding_key);
                let arg = CallArg::Type(&key_ty, mapping_key.range());
                self.call_method_or_error(
                    binding.ty(),
                    &dunder::GETITEM,
                    mapping_key.range(),
                    &[arg],
                    &[],
                    errors,
                    None,
                )
            }
            Binding::PatternMatchClassPositional(_, idx, key, range) => {
                // TODO: check that value matches class
                // TODO: check against duplicate keys (optional)
                let binding = self.get_idx(*key);
                let context =
                    || ErrorContext::MatchPositional(self.for_display(binding.ty().clone()));
                let match_args = self
                    .attr_infer(
                        &binding,
                        &dunder::MATCH_ARGS,
                        *range,
                        errors,
                        Some(&context),
                    )
                    .into_ty();
                match match_args {
                    Type::Tuple(Tuple::Concrete(ts)) => {
                        if *idx < ts.len() {
                            if let Some(Type::Literal(Lit::Str(box attr_name))) = ts.get(*idx) {
                                self.attr_infer(
                                    &binding,
                                    &Name::new(attr_name),
                                    *range,
                                    errors,
                                    Some(&context),
                                )
                                .into_ty()
                            } else {
                                self.error(
                                    errors,
                                    *range,
                                    ErrorKind::MatchError,
                                    Some(&context),
                                    format!(
                                        "Expected literal string in `__match_args__`, got {}",
                                        ts[*idx]
                                    ),
                                )
                            }
                        } else {
                            self.error(
                                errors,
                                *range,
                                ErrorKind::MatchError,
                                Some(&context),
                                format!("Index {idx} out of range for `__match_args__`"),
                            )
                        }
                    }
                    Type::Any(AnyStyle::Error) => match_args,
                    _ => self.error(
                        errors,
                        *range,
                        ErrorKind::MatchError,
                        Some(&context),
                        format!(
                            "Expected concrete tuple for `__match_args__`, got {}",
                            match_args
                        ),
                    ),
                }
            }
            Binding::PatternMatchClassKeyword(_, attr, key) => {
                // TODO: check that value matches class
                // TODO: check against duplicate keys (optional)
                let binding = self.get_idx(*key);
                self.attr_infer(&binding, &attr.id, attr.range, errors, None)
                    .into_ty()
            }
            Binding::NameAssign(name, annot_key, expr) => {
                let (has_type_alias_qualifier, ty) = match annot_key.as_ref() {
                    Some((style, k)) => {
                        let annot = self.get_idx(*k);
                        let tcc: &dyn Fn() -> TypeCheckContext = &|| {
                            TypeCheckContext::of_kind(match style {
                                AnnotationStyle::Direct => TypeCheckKind::AnnAssign,
                                AnnotationStyle::Forwarded => {
                                    TypeCheckKind::AnnotatedName(name.clone())
                                }
                            })
                        };
                        if annot.annotation.is_final() && *style == AnnotationStyle::Forwarded {
                            self.error(
                                errors,
                                expr.range(),
                                ErrorKind::BadAssignment,
                                None,
                                format!("`{}` is marked final", name),
                            );
                        }
                        let annot_ty = annot.ty(self.stdlib);
                        let hint = annot_ty.as_ref().map(|t| (t, tcc));
                        let expr_ty = self.expr(expr, hint, errors);
                        let ty = if *style == AnnotationStyle::Direct {
                            // For direct assignments, user-provided annotation takes
                            // precedence over inferred expr type.
                            annot_ty.unwrap_or(expr_ty)
                        } else {
                            // For forwarded assignment, user-provided annotation is treated
                            // as just an upper-bound hint.
                            expr_ty
                        };
                        (
                            Some(annot.annotation.qualifiers.contains(&Qualifier::TypeAlias)),
                            ty,
                        )
                    }
                    None => (None, self.expr(expr, None, errors)),
                };
                match (has_type_alias_qualifier, &ty) {
                    (Some(true), _) => {
                        self.as_type_alias(name, TypeAliasStyle::LegacyExplicit, ty, expr, errors)
                    }
                    (None, ty_ref) if Self::may_be_implicit_type_alias(ty_ref) => {
                        self.as_type_alias(name, TypeAliasStyle::LegacyImplicit, ty, expr, errors)
                    }
                    _ => ty,
                }
            }
            Binding::TypeVar(ann, name, x) => {
                let ty = Type::type_form(self.typevar_from_call(name.clone(), x, errors).to_type());
                if let Some(k) = ann
                    && let AnnotationWithTarget {
                        target,
                        annotation:
                            Annotation {
                                ty: Some(want),
                                qualifiers: _,
                            },
                    } = &*self.get_idx(*k)
                {
                    self.check_and_return_type(want, ty, x.range, errors, &|| {
                        TypeCheckContext::of_kind(TypeCheckKind::from_annotation_target(target))
                    })
                } else {
                    ty
                }
            }
            Binding::ParamSpec(ann, name, x) => {
                let ty =
                    Type::type_form(self.paramspec_from_call(name.clone(), x, errors).to_type());
                if let Some(k) = ann
                    && let AnnotationWithTarget {
                        target,
                        annotation:
                            Annotation {
                                ty: Some(want),
                                qualifiers: _,
                            },
                    } = &*self.get_idx(*k)
                {
                    self.check_and_return_type(want, ty, x.range, errors, &|| {
                        TypeCheckContext::of_kind(TypeCheckKind::from_annotation_target(target))
                    })
                } else {
                    ty
                }
            }
            Binding::TypeVarTuple(ann, name, x) => {
                let ty = Type::type_form(
                    self.typevartuple_from_call(name.clone(), x, errors)
                        .to_type(),
                );
                if let Some(k) = ann
                    && let AnnotationWithTarget {
                        target,
                        annotation:
                            Annotation {
                                ty: Some(want),
                                qualifiers: _,
                            },
                    } = &*self.get_idx(*k)
                {
                    self.check_and_return_type(want, ty, x.range, errors, &|| {
                        TypeCheckContext::of_kind(TypeCheckKind::from_annotation_target(target))
                    })
                } else {
                    ty
                }
            }
            Binding::ReturnType(box x) => {
                let is_generator = !x.yields.is_empty();
                let implicit_return = self.get_idx(x.implicit_return);
                if let Some((range, annot)) = &x.annot {
                    // TODO: A return type annotation like `Final` is invalid in this context.
                    // It will result in an implicit Any type, which is reasonable, but we should
                    // at least error here.
                    let ty = self.get_idx(*annot).annotation.get_type().clone();
                    // If the function body is stubbed out or if the function is decorated with
                    // `@abstractmethod`, we blindly accept the return type annotation.
                    if x.stub_or_impl != FunctionStubOrImpl::Stub
                        && !x.decorators.iter().any(|k| {
                            let decorator = self.get_idx(*k);
                            match decorator.ty().callee_kind() {
                                Some(CalleeKind::Function(FunctionKind::AbstractMethod)) => true,
                                _ => false,
                            }
                        })
                    {
                        self.check_implicit_return_against_annotation(
                            implicit_return,
                            &ty,
                            x.is_async,
                            is_generator,
                            !x.returns.is_empty(),
                            *range,
                            errors,
                        );
                    }
                    ty
                } else {
                    let returns = x.returns.iter().map(|k| self.get_idx(*k).arc_clone_ty());
                    // TODO: It should always be a no-op to include a `Type::Never` in unions, but
                    // `simple::test_solver_variables` fails if we do, because `solver::unions` does
                    // `is_subset_eq` to force free variables, causing them to be equated to
                    // `Type::Never` instead of becoming `Type::Any`.
                    let return_ty = if implicit_return.ty().is_never() {
                        self.unions(returns.collect())
                    } else {
                        self.unions(
                            returns
                                .chain(iter::once(implicit_return.arc_clone_ty()))
                                .collect(),
                        )
                    };
                    if is_generator {
                        let yield_ty = self.unions(
                            x.yields
                                .iter()
                                .map(|x| match x {
                                    Either::Left(k) => self.get_idx(*k).yield_ty.clone(),
                                    Either::Right(k) => self.get_idx(*k).yield_ty.clone(),
                                })
                                .collect(),
                        );
                        if x.is_async {
                            self.stdlib
                                .async_generator(yield_ty, Type::any_implicit())
                                .to_type()
                        } else {
                            self.stdlib
                                .generator(yield_ty, Type::any_implicit(), return_ty)
                                .to_type()
                        }
                    } else {
                        // Do *not* modify the return type for a (non-generator) async def, we want the return type
                        // here to match what would be annotated. We will handle creating an awaitable type when
                        // binding the function def.
                        return_ty
                    }
                }
            }
            Binding::ReturnExplicit(x) => {
                let annot = x.annot.map(|k| self.get_idx(k));
                let hint = annot.as_ref().and_then(|ann| ann.ty(self.stdlib));

                if let Some(expr) = &x.expr {
                    if x.is_async && x.is_generator {
                        self.expr_infer(expr, errors);
                        self.error(
                            errors,
                            expr.range(),
                            ErrorKind::BadReturn,
                            None,
                            "Return statement with value is not allowed in async generator"
                                .to_owned(),
                        )
                    } else if x.is_generator {
                        let hint =
                            hint.and_then(|ty| self.decompose_generator(&ty).map(|(_, _, r)| r));
                        let tcc: &dyn Fn() -> TypeCheckContext =
                            &|| TypeCheckContext::of_kind(TypeCheckKind::ExplicitFunctionReturn);
                        self.expr(expr, hint.as_ref().map(|t| (t, tcc)), errors)
                    } else if matches!(hint, Some(Type::TypeGuard(_) | Type::TypeIs(_))) {
                        let hint = Some(Type::ClassType(self.stdlib.bool().clone()));
                        let tcc: &dyn Fn() -> TypeCheckContext =
                            &|| TypeCheckContext::of_kind(TypeCheckKind::TypeGuardReturn);
                        self.expr(expr, hint.as_ref().map(|t| (t, tcc)), errors)
                    } else {
                        let tcc: &dyn Fn() -> TypeCheckContext =
                            &|| TypeCheckContext::of_kind(TypeCheckKind::ExplicitFunctionReturn);
                        self.expr(expr, hint.as_ref().map(|t| (t, tcc)), errors)
                    }
                } else {
                    Type::None
                }
            }
            Binding::ReturnImplicit(x) => {
                // Would context have caught something:
                // https://typing.python.org/en/latest/spec/exceptions.html#context-managers.
                let context_catch = |x: &Type| -> bool {
                    match x {
                        Type::Literal(Lit::Bool(b)) => *b,
                        Type::ClassType(cls) => cls == self.stdlib.bool(),
                        _ => false, // Default to assuming exceptions are not suppressed
                    }
                };

                if self.module_info().path().is_interface() {
                    Type::any_implicit() // .pyi file, functions don't have bodies
                } else if x.last_exprs.as_ref().is_some_and(|xs| {
                    xs.iter().all(|(last, k)| {
                        let e = self.get_idx(*k);
                        match last {
                            LastStmt::Expr => e.ty().is_never(),
                            LastStmt::With(kind) => {
                                let res = self.context_value_exit(
                                    e.ty(),
                                    *kind,
                                    TextRange::default(),
                                    &self.error_swallower(),
                                    None,
                                );
                                !context_catch(&res)
                            }
                        }
                    })
                }) {
                    Type::never()
                } else {
                    Type::None
                }
            }
            Binding::ExceptionHandler(box ann, is_star) => {
                let base_exception_type = self.stdlib.base_exception().clone().to_type();
                let base_exception_group_any_type = if *is_star {
                    // Only query for `BaseExceptionGroup` if we see an `except*` handler (which
                    // was introduced in Python3.11).
                    // We can't unconditionally query for `BaseExceptionGroup` until Python3.10
                    // is out of its EOL period.
                    let res = self
                        .stdlib
                        .base_exception_group(Type::Any(AnyStyle::Implicit))
                        .map(|x| x.to_type());
                    if res.is_none() {
                        self.error(
                            errors,
                            ann.range(),
                            ErrorKind::Unsupported,
                            None,
                            "`expect*` is unsupported until Python 3.11".to_owned(),
                        );
                    }
                    res
                } else {
                    None
                };
                let check_exception_type = |exception_type: Type, range| {
                    let exception = self.untype(exception_type, range, errors);
                    self.check_type(&base_exception_type, &exception, range, errors, &|| {
                        TypeCheckContext::of_kind(TypeCheckKind::ExceptionClass)
                    });
                    if let Some(base_exception_group_any_type) =
                        base_exception_group_any_type.as_ref()
                        && !exception.is_any()
                        && self.is_subset_eq(&exception, base_exception_group_any_type)
                    {
                        self.error(
                            errors,
                            range,
                            ErrorKind::InvalidInheritance,
                            None,
                            "Exception handler annotation in `except*` clause may not extend `BaseExceptionGroup`".to_owned());
                    }
                    exception
                };
                let exceptions = match ann {
                    // if the exception classes are written as a tuple literal, use each annotation's position for error reporting
                    Expr::Tuple(tup) => tup
                        .elts
                        .iter()
                        .map(|e| check_exception_type(self.expr_infer(e, errors), e.range()))
                        .collect(),
                    _ => {
                        let exception_types = self.expr_infer(ann, errors);
                        match exception_types {
                            Type::Tuple(Tuple::Concrete(ts)) => ts
                                .into_iter()
                                .map(|t| check_exception_type(t, ann.range()))
                                .collect(),
                            Type::Tuple(Tuple::Unbounded(box t)) => {
                                vec![check_exception_type(t, ann.range())]
                            }
                            _ => vec![check_exception_type(exception_types, ann.range())],
                        }
                    }
                };
                let exceptions = self.unions(exceptions);
                if *is_star && let Some(t) = self.stdlib.exception_group(exceptions.clone()) {
                    t.to_type()
                } else {
                    exceptions
                }
            }
            Binding::AugAssign(ann, x) => self.augassign_infer(*ann, x, errors),
            Binding::IterableValue(ann, e, is_async) => {
                let ty = ann.map(|k| self.get_idx(k));
                let tcc: &dyn Fn() -> TypeCheckContext = &|| {
                    let (name, annot_type) = {
                        match &ty {
                            None => (None, None),
                            Some(t) => (
                                match &t.target {
                                    AnnotationTarget::Assign(name, _)
                                    | AnnotationTarget::ClassMember(name) => Some(name.clone()),
                                    _ => None,
                                },
                                t.ty(self.stdlib).clone(),
                            ),
                        }
                    };
                    TypeCheckContext::of_kind(TypeCheckKind::IterationVariableMismatch(
                        name.unwrap_or_else(|| Name::new_static("_")),
                        self.for_display(annot_type.unwrap_or_else(Type::any_implicit)),
                    ))
                };
                let iterables = if is_async.is_async() {
                    let hint = ty.clone().and_then(|x| {
                        x.ty(self.stdlib)
                            .map(|ty| self.stdlib.async_iterable(ty.clone()).to_type())
                    });
                    self.async_iterate(
                        &self.expr(e, hint.as_ref().map(|t| (t, tcc)), errors),
                        e.range(),
                        errors,
                    )
                } else {
                    let hint = ty.clone().and_then(|x| {
                        x.ty(self.stdlib)
                            .map(|ty| self.stdlib.iterable(ty.clone()).to_type())
                    });
                    self.iterate(
                        &self.expr(e, hint.as_ref().map(|t| (t, tcc)), errors),
                        e.range(),
                        errors,
                    )
                };
                let mut values = Vec::new();
                for iterable in iterables {
                    match iterable {
                        Iterable::OfType(ty) => values.push(ty),
                        Iterable::FixedLen(ts) => values.extend(ts),
                    }
                }
                self.unions(values)
            }
            Binding::ContextValue(ann, e, range, kind) => {
                let context_manager = self.get_idx(*e);
                let context_value = self.context_value(context_manager.ty(), *kind, *range, errors);
                let ty = ann.map(|k| self.get_idx(k));
                match ty
                    .as_ref()
                    .and_then(|x| x.ty(self.stdlib).map(|t| (t, &x.target)))
                {
                    Some((ty, target)) => {
                        self.check_and_return_type(&ty, context_value, *range, errors, &|| {
                            TypeCheckContext::of_kind(TypeCheckKind::from_annotation_target(target))
                        })
                    }
                    None => context_value,
                }
            }
            Binding::UnpackedValue(b, range, pos) => {
                let iterables = self.iterate(self.get_idx(*b).ty(), *range, errors);
                let mut values = Vec::new();
                for iterable in iterables {
                    values.push(match iterable {
                        Iterable::OfType(ty) => match pos {
                            UnpackedPosition::Index(_) | UnpackedPosition::ReverseIndex(_) => ty,
                            UnpackedPosition::Slice(_, _) => self.stdlib.list(ty).to_type(),
                        },
                        Iterable::FixedLen(ts) => {
                            match pos {
                                UnpackedPosition::Index(i) | UnpackedPosition::ReverseIndex(i) => {
                                    let idx = if matches!(pos, UnpackedPosition::Index(_)) {
                                        Some(*i)
                                    } else {
                                        ts.len().checked_sub(*i)
                                    };
                                    if let Some(idx) = idx
                                        && let Some(element) = ts.get(idx)
                                    {
                                        element.clone()
                                    } else {
                                        // We'll report this error when solving for Binding::UnpackedLength.
                                        Type::any_error()
                                    }
                                }
                                UnpackedPosition::Slice(i, j) => {
                                    let start = *i;
                                    let end = ts.len().checked_sub(*j);
                                    if let Some(end) = end
                                        && end >= start
                                        && let Some(items) = ts.get(start..end)
                                    {
                                        let elem_ty = self.unions(items.to_vec());
                                        self.stdlib.list(elem_ty).to_type()
                                    } else {
                                        // We'll report this error when solving for Binding::UnpackedLength.
                                        Type::any_error()
                                    }
                                }
                            }
                        }
                    })
                }
                self.unions(values)
            }
            &Binding::Function(idx, mut pred, class_meta) => {
                self.solve_function_binding(idx, &mut pred, class_meta.as_ref(), errors)
            }
            Binding::Import(m, name) => self
                .get_from_module(*m, None, &KeyExport(name.clone()))
                .arc_clone(),
            Binding::ClassDef(x, decorators) => match &self.get_idx(*x).0 {
                None => Type::any_implicit(),
                Some(cls) => {
                    let mut ty = Type::ClassDef(cls.dupe());
                    for x in decorators.iter().rev() {
                        ty = self.apply_decorator(*x, ty, errors)
                    }
                    ty
                }
            },
            Binding::AnnotatedType(ann, val) => match &self.get_idx(*ann).ty(self.stdlib) {
                Some(ty) => (*ty).clone(),
                None => self.binding_to_type(val, errors),
            },
            Binding::Type(x) => x.clone(),
            Binding::StrType => self.stdlib.str().clone().to_type(),
            Binding::TypeParameter(box TypeParameter {
                name,
                unique,
                kind,
                bound,
                default,
                constraints,
            }) => {
                let restriction = if let Some(bound) = bound {
                    let bound_ty =
                        self.expr_untype(bound, TypeFormContext::TypeVarConstraint, errors);
                    Restriction::Bound(bound_ty)
                } else if let Some((constraints, range)) = constraints {
                    if constraints.len() < 2 {
                        self.error(
                            errors,
                            *range,
                            ErrorKind::InvalidTypeVar,
                            None,
                            format!(
                                "Expected at least 2 constraints in TypeVar `{}`, got {}",
                                name,
                                constraints.len(),
                            ),
                        );
                        Restriction::Unrestricted
                    } else {
                        let constraint_tys = constraints.map(|constraint| {
                            self.expr_untype(constraint, TypeFormContext::TypeVarConstraint, errors)
                        });
                        Restriction::Constraints(constraint_tys)
                    }
                } else {
                    Restriction::Unrestricted
                };
                let mut default_ty = None;
                if let Some(default_expr) = default {
                    let default = self.expr_untype(
                        default_expr,
                        kind.type_form_context_for_default(),
                        errors,
                    );
                    default_ty = Some(self.validate_type_var_default(
                        name,
                        *kind,
                        &default,
                        default_expr.range(),
                        &restriction,
                        errors,
                    ));
                }
                Type::type_form(
                    Quantified::new(
                        *unique,
                        QuantifiedInfo {
                            name: name.clone(),
                            kind: *kind,
                            default: default_ty,
                            restriction,
                        },
                    )
                    .to_type(),
                )
            }
            Binding::Module(m, path, prev) => {
                let prev = prev
                    .as_ref()
                    .and_then(|x| self.get_idx(*x).ty().as_module().cloned());
                match prev {
                    Some(prev) if prev.path() == path => prev.add_module(*m).to_type(),
                    _ => {
                        if path.len() == 1 {
                            Type::Module(Module::new(
                                path[0].clone(),
                                OrderedSet::from_iter([(*m)]),
                            ))
                        } else {
                            assert_eq!(&m.components(), path);
                            Type::Module(Module::new_as(*m))
                        }
                    }
                }
            }
            Binding::CheckLegacyTypeParam(key, range_if_scoped_params_exist) => {
                match &*self.get_idx(*key) {
                    LegacyTypeParameterLookup::Parameter(p) => {
                        // This class or function has scoped (PEP 695) type parameters. Mixing legacy-style parameters is an error.
                        if let Some(r) = range_if_scoped_params_exist {
                            self.error(
                                errors,
                                *r,
                                ErrorKind::InvalidTypeVar,
                                None,
                                format!(
                                    "Type parameter {} is not included in the type parameter list",
                                    self.module_info()
                                        .display(&self.bindings().idx_to_key(*key).0)
                                ),
                            );
                        }
                        Type::type_form(p.quantified.clone().to_type())
                    }
                    LegacyTypeParameterLookup::NotParameter(ty) => ty.clone(),
                }
            }
            Binding::ScopedTypeAlias(name, params, expr) => {
                let ty = self.expr_infer(expr, errors);
                let ta = self.as_type_alias(name, TypeAliasStyle::Scoped, ty, expr, errors);
                match ta {
                    Type::Forall(..) => self.error(
                        errors,
                        expr.range(),
                        ErrorKind::InvalidTypeVar,
                        None,
                        format!("Type parameters used in `{name}` but not declared"),
                    ),
                    Type::TypeAlias(ta) => {
                        let params_range = params.as_ref().map_or(expr.range(), |x| x.range);
                        Forallable::TypeAlias(ta).forall(self.type_params(
                            params_range,
                            self.scoped_type_params(params.as_ref(), errors),
                            errors,
                        ))
                    }
                    _ => ta,
                }
            }
            Binding::Decorator(expr) => self.expr_infer(expr, errors),
            Binding::LambdaParameter(var) => var.to_type(),
            Binding::FunctionParameter(param) => {
                match param {
                    Either::Left(key) => {
                        let annotation = self.get_idx(*key);
                        annotation.ty(self.stdlib).clone().unwrap_or_else(|| {
                            // This annotation isn't valid. It's something like `: Final` that doesn't
                            // have enough information to create a real type.
                            Type::any_implicit()
                        })
                    }
                    Either::Right((var, function_idx, _)) => {
                        // Force the function binding to be evaluated, if it hasn't already.
                        // Solving the function will also force the Var type to some concrete type,
                        // and this must happen first so the Var can not interact with other types.
                        self.get_idx(*function_idx);
                        var.to_type()
                    }
                }
            }
            Binding::SuperInstance(style, range) => self.solve_super_binding(style, *range, errors),
        }
    }

    pub fn solve_function(
        &self,
        x: &BindingFunction,
        errors: &ErrorCollector,
    ) -> Arc<DecoratedFunction> {
        self.function_definition(
            &x.def,
            x.stub_or_impl,
            x.class_key.as_ref(),
            &x.decorators,
            &x.legacy_tparams,
            errors,
        )
    }

    pub fn solve_yield(&self, x: &BindingYield, errors: &ErrorCollector) -> Arc<YieldResult> {
        match x {
            BindingYield::Yield(annot, x) => {
                // TODO: Keep track of whether the function is async in the binding, decompose hint
                // appropriately instead of just trying both.
                let annot = annot.map(|k| self.get_idx(k));
                let hint = annot
                    .as_ref()
                    .and_then(|x| x.ty(self.stdlib))
                    .and_then(|ty| {
                        if let Some((yield_ty, send_ty, _)) = self.decompose_generator(&ty) {
                            Some((yield_ty, send_ty))
                        } else {
                            self.decompose_async_generator(&ty)
                        }
                    });
                if let Some((yield_hint, send_ty)) = hint {
                    let yield_ty = if let Some(expr) = x.value.as_ref() {
                        self.expr(
                            expr,
                            Some((&yield_hint, &|| {
                                TypeCheckContext::of_kind(TypeCheckKind::YieldValue)
                            })),
                            errors,
                        )
                    } else {
                        self.check_and_return_type(
                            &yield_hint,
                            Type::None,
                            x.range,
                            errors,
                            &|| TypeCheckContext::of_kind(TypeCheckKind::UnexpectedBareYield),
                        )
                    };
                    Arc::new(YieldResult { yield_ty, send_ty })
                } else {
                    let yield_ty = if let Some(expr) = x.value.as_ref() {
                        self.expr_infer(expr, errors)
                    } else {
                        Type::None
                    };
                    let send_ty = Type::any_implicit();
                    Arc::new(YieldResult { yield_ty, send_ty })
                }
            }
            BindingYield::Invalid(x) => {
                if let Some(expr) = x.value.as_ref() {
                    self.expr_infer(expr, errors);
                }
                self.error(
                    errors,
                    x.range,
                    ErrorKind::InvalidYield,
                    None,
                    "Invalid `yield` outside of a function".to_owned(),
                );
                Arc::new(YieldResult::any_error())
            }
        }
    }

    pub fn solve_yield_from(
        &self,
        x: &BindingYieldFrom,
        errors: &ErrorCollector,
    ) -> Arc<YieldFromResult> {
        match x {
            BindingYieldFrom::YieldFrom(annot, x) => {
                // TODO: Error if the function is async
                let annot = annot.map(|k| self.get_idx(k));
                let want = annot.as_ref().and_then(|x| x.ty(self.stdlib));

                let mut ty = self.expr_infer(&x.value, errors);
                let res = if let Some(generator) = self.unwrap_generator(&ty) {
                    YieldFromResult::from_generator(generator)
                } else if let Some(yield_ty) = self.unwrap_iterable(&ty) {
                    // Promote the type to a generator for the check below to succeed.
                    // Per PEP-380, if None is sent to the delegating generator, the
                    // iterator's __next__() method is called, so promote to a generator
                    // with a `None` send type.
                    // TODO: This might cause confusing type errors.
                    ty = self
                        .stdlib
                        .generator(yield_ty.clone(), Type::None, Type::None)
                        .to_type();
                    YieldFromResult::from_iterable(yield_ty)
                } else {
                    ty = self.error(
                        errors,
                        x.range,
                        ErrorKind::InvalidYield,
                        None,
                        format!("yield from value must be iterable, got `{ty}`"),
                    );
                    YieldFromResult::any_error()
                };
                if let Some(want) = want {
                    self.check_type(&want, &ty, x.range, errors, &|| {
                        TypeCheckContext::of_kind(TypeCheckKind::YieldFrom)
                    });
                }
                Arc::new(res)
            }
            BindingYieldFrom::Invalid(x) => {
                self.expr_infer(&x.value, errors);
                self.error(
                    errors,
                    x.range,
                    ErrorKind::InvalidYield,
                    None,
                    "Invalid `yield from` outside of a function".to_owned(),
                );
                Arc::new(YieldFromResult::any_error())
            }
        }
    }

    /// Unwraps a type, originally evaluated as a value, so that it can be used as a type annotation.
    /// For example, in `def f(x: int): ...`, we evaluate `int` as a value, getting its type as
    /// `type[int]`, then call `untype(type[int])` to get the `int` annotation.
    fn untype(&self, ty: Type, range: TextRange, errors: &ErrorCollector) -> Type {
        let mut ty = ty;
        if let Type::Forall(forall) = ty {
            // A generic type alias with no type arguments is OK if all the type params have defaults
            let targs = self.check_and_create_targs(
                &forall.body.name(),
                &forall.tparams,
                Vec::new(),
                range,
                errors,
            );
            let param_map = forall
                .tparams
                .quantified()
                .cloned()
                .zip(targs.as_slice().iter().cloned())
                .collect::<SmallMap<_, _>>();
            ty = forall.body.as_type().subst(&param_map)
        };
        if let Some(t) = self.untype_opt(ty.clone(), range) {
            t
        } else {
            self.error(
                errors,
                range,
                ErrorKind::NotAType,
                None,
                format!(
                    "Expected a type form, got instance of `{}`",
                    self.for_display(ty),
                ),
            )
        }
    }

    pub fn untype_opt(&self, ty: Type, range: TextRange) -> Option<Type> {
        match self.canonicalize_all_class_types(ty, range) {
            Type::Union(xs) if !xs.is_empty() => {
                let mut ts = Vec::new();
                for x in xs {
                    let t = self.untype_opt(x, range)?;
                    ts.push(t);
                }
                Some(self.unions(ts))
            }
            Type::Var(v) if let Some(_guard) = self.recurser.recurse(v) => {
                self.untype_opt(self.solver().force_var(v), range)
            }
            Type::Type(box t) => Some(t),
            Type::None => Some(Type::None), // Both a value and a type
            Type::Ellipsis => Some(Type::Ellipsis), // A bit weird because of tuples, so just promote it
            Type::Any(style) => Some(style.propagate()),
            Type::TypeAlias(ta) => self.untype_opt(ta.as_type(), range),
            t @ Type::Unpack(
                box Type::Tuple(_) | box Type::TypeVarTuple(_) | box Type::Quantified(_),
            ) => Some(t),
            Type::Unpack(box Type::Var(v)) if let Some(_guard) = self.recurser.recurse(v) => {
                self.untype_opt(Type::Unpack(Box::new(self.solver().force_var(v))), range)
            }
            _ => None,
        }
    }

    pub fn validate_type_form(
        &self,
        ty: Type,
        range: TextRange,
        type_form_context: TypeFormContext,
        errors: &ErrorCollector,
    ) -> Type {
        if type_form_context != TypeFormContext::ParameterKwargsAnnotation
            && matches!(ty, Type::Unpack(box Type::TypedDict(_)))
        {
            return self.error(
                errors,
                range,
                ErrorKind::InvalidAnnotation,
                None,
                "Unpack with a TypedDict is only allowed in a **kwargs annotation.".to_owned(),
            );
        }
        if type_form_context != TypeFormContext::ParameterKwargsAnnotation
            && matches!(ty, Type::Kwargs(_))
        {
            return self.error(
                errors,
                range,
                ErrorKind::InvalidAnnotation,
                None,
                "ParamSpec **kwargs is only allowed in a **kwargs annotation.".to_owned(),
            );
        }
        if type_form_context != TypeFormContext::ParameterArgsAnnotation
            && matches!(ty, Type::Args(_))
        {
            return self.error(
                errors,
                range,
                ErrorKind::InvalidAnnotation,
                None,
                "ParamSpec *args is only allowed in an *args annotation.".to_owned(),
            );
        }
        if !matches!(
            type_form_context,
            TypeFormContext::ParameterArgsAnnotation
                | TypeFormContext::ParameterKwargsAnnotation
                | TypeFormContext::TypeArgument
                | TypeFormContext::TupleOrCallableParam
                | TypeFormContext::GenericBase
                | TypeFormContext::TypeVarTupleDefault
        ) && matches!(ty, Type::Unpack(_))
        {
            return self.error(
                errors,
                range,
                ErrorKind::InvalidAnnotation,
                None,
                "Unpack is not allowed in this context.".to_owned(),
            );
        }
        if !matches!(
            type_form_context,
            TypeFormContext::TypeArgument
                | TypeFormContext::GenericBase
                | TypeFormContext::ParamSpecDefault
        ) && matches!(
            ty,
            Type::Concatenate(_, _) | Type::ParamSpecValue(_) | Type::ParamSpec(_)
        ) {
            return self.error(
                errors,
                range,
                ErrorKind::InvalidAnnotation,
                None,
                format!("{} is not allowed in this context.", ty),
            );
        }
        if !matches!(
            type_form_context,
            TypeFormContext::TupleOrCallableParam | TypeFormContext::TypeArgument
        ) && matches!(ty, Type::TypeVarTuple(_))
        {
            return self.error(
                errors,
                range,
                ErrorKind::InvalidAnnotation,
                None,
                "TypeVarTuple must be unpacked.".to_owned(),
            );
        }
        if let Type::SpecialForm(special_form) = ty
            && !special_form.is_valid_unparameterized_annotation(type_form_context)
        {
            self.error(
                errors,
                range,
                ErrorKind::InvalidAnnotation,
                None,
                format!("Expected a type argument for `{}`", special_form),
            );
        }
        if let Type::Quantified(quantified) = &ty {
            if quantified.is_param_spec()
                && !matches!(
                    type_form_context,
                    TypeFormContext::TypeArgument
                        | TypeFormContext::GenericBase
                        | TypeFormContext::ParamSpecDefault
                )
            {
                return self.error(
                    errors,
                    range,
                    ErrorKind::InvalidAnnotation,
                    None,
                    "ParamSpec is not allowed in this context.".to_owned(),
                );
            }
            // We check tuple/callable/generic type arguments separately, so exclude those
            // to avoid emitting duplicate errors.
            if quantified.is_type_var_tuple()
                && !matches!(
                    type_form_context,
                    TypeFormContext::TupleOrCallableParam | TypeFormContext::TypeArgument
                )
            {
                return self.error(
                    errors,
                    range,
                    ErrorKind::InvalidAnnotation,
                    None,
                    "TypeVarTuple must be unpacked.".to_owned(),
                );
            }
        }
        if type_form_context == TypeFormContext::TypeVarConstraint && ty.any(Type::is_type_variable)
        {
            return self.error(
                errors,
                range,
                ErrorKind::InvalidAnnotation,
                None,
                "Type variable bounds and constraints must be concrete".to_owned(),
            );
        }
        ty
    }

    pub fn expr_untype(
        &self,
        x: &Expr,
        type_form_context: TypeFormContext,
        errors: &ErrorCollector,
    ) -> Type {
        let result = match x {
            Expr::List(x)
                if matches!(
                    type_form_context,
                    TypeFormContext::TypeArgument | TypeFormContext::ParamSpecDefault
                ) =>
            {
                let elts: Vec<Param> = x
                    .elts
                    .iter()
                    .map(|x| {
                        Param::PosOnly(
                            self.expr_untype(x, type_form_context, errors),
                            Required::Required,
                        )
                    })
                    .collect();
                Type::ParamSpecValue(ParamList::new(elts))
            }
            _ => self.untype(self.expr_infer(x, errors), x.range(), errors),
        };
        self.validate_type_form(result, x.range(), type_form_context, errors)
    }
}
