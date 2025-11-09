/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::iter;
use std::ops::Deref;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_python::ast::Ast;
use pyrefly_python::dunder;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::facet::FacetKind;
use pyrefly_types::type_info::JoinStyle;
use pyrefly_types::typed_dict::ExtraItems;
use pyrefly_types::typed_dict::TypedDict;
use pyrefly_util::prelude::SliceExt;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprBinOp;
use ruff_python_ast::ExprSubscript;
use ruff_python_ast::TypeParams;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::ordered_set::OrderedSet;
use starlark_map::small_map::Entry;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use vec1::vec1;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::callable::CallArg;
use crate::alt::class::class_field::ClassField;
use crate::alt::class::variance_inference::VarianceMap;
use crate::alt::types::abstract_class::AbstractClassMembers;
use crate::alt::types::class_bases::ClassBases;
use crate::alt::types::class_metadata::ClassMetadata;
use crate::alt::types::class_metadata::ClassMro;
use crate::alt::types::class_metadata::ClassSynthesizedFields;
use crate::alt::types::decorated_function::UndecoratedFunction;
use crate::alt::types::legacy_lookup::LegacyTypeParameterLookup;
use crate::alt::types::yields::YieldFromResult;
use crate::alt::types::yields::YieldResult;
use crate::alt::unwrap::HintRef;
use crate::binding::binding::AnnAssignHasValue;
use crate::binding::binding::AnnotationStyle;
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
use crate::binding::binding::BindingLegacyTypeParam;
use crate::binding::binding::BindingTParams;
use crate::binding::binding::BindingUndecoratedFunction;
use crate::binding::binding::BindingVariance;
use crate::binding::binding::BindingYield;
use crate::binding::binding::BindingYieldFrom;
use crate::binding::binding::EmptyAnswer;
use crate::binding::binding::ExprOrBinding;
use crate::binding::binding::FirstUse;
use crate::binding::binding::FunctionParameter;
use crate::binding::binding::FunctionStubOrImpl;
use crate::binding::binding::IsAsync;
use crate::binding::binding::Key;
use crate::binding::binding::KeyExport;
use crate::binding::binding::KeyLegacyTypeParam;
use crate::binding::binding::KeyUndecoratedFunction;
use crate::binding::binding::LastStmt;
use crate::binding::binding::LinkedKey;
use crate::binding::binding::NoneIfRecursive;
use crate::binding::binding::RaisedException;
use crate::binding::binding::ReturnTypeKind;
use crate::binding::binding::SizeExpectation;
use crate::binding::binding::SuperStyle;
use crate::binding::binding::TypeParameter;
use crate::binding::binding::UnpackedPosition;
use crate::binding::narrow::identifier_and_chain_for_expr;
use crate::binding::narrow::identifier_and_chain_prefix_for_expr;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorContext;
use crate::error::context::ErrorInfo;
use crate::error::context::TypeCheckContext;
use crate::error::context::TypeCheckKind;
use crate::error::style::ErrorStyle;
use crate::export::special::SpecialExport;
use crate::graph::index::Idx;
use crate::solver::solver::SubsetError;
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
use crate::types::module::ModuleType;
use crate::types::param_spec::ParamSpec;
use crate::types::quantified::Quantified;
use crate::types::quantified::QuantifiedKind;
use crate::types::special_form::SpecialForm;
use crate::types::tuple::Tuple;
use crate::types::type_info::TypeInfo;
use crate::types::type_var::PreInferenceVariance;
use crate::types::type_var::Restriction;
use crate::types::type_var::TypeVar;
use crate::types::type_var_tuple::TypeVarTuple;
use crate::types::types::AnyStyle;
use crate::types::types::CalleeKind;
use crate::types::types::Forallable;
use crate::types::types::SuperObj;
use crate::types::types::TParam;
use crate::types::types::TParams;
use crate::types::types::TParamsSource;
use crate::types::types::Type;
use crate::types::types::TypeAlias;
use crate::types::types::TypeAliasStyle;
use crate::types::types::Var;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TypeFormContext {
    /// Expression in a base class list
    BaseClassList,
    /// Keyword in a class definition - `class C(some_keyword=SomeValue): ...`
    ClassKeyword,
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
    /// Type argument for `builtins.type`
    TypeArgumentForType,
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
    VarAnnotation(AnnAssignHasValue),
}

impl TypeFormContext {
    pub fn quantified_kind_default(x: QuantifiedKind) -> Self {
        match x {
            QuantifiedKind::TypeVar => TypeFormContext::TypeVarDefault,
            QuantifiedKind::ParamSpec => TypeFormContext::ParamSpecDefault,
            QuantifiedKind::TypeVarTuple => TypeFormContext::TypeVarTupleDefault,
        }
    }

    /// Is this special form valid as an un-parameterized annotation anywhere?
    pub fn is_valid_unparameterized_annotation(self, x: SpecialForm) -> bool {
        match x {
            SpecialForm::Protocol | SpecialForm::TypedDict => {
                matches!(self, TypeFormContext::BaseClassList)
            }
            SpecialForm::TypeAlias => matches!(
                self,
                TypeFormContext::TypeAlias | TypeFormContext::VarAnnotation(AnnAssignHasValue::Yes)
            ),
            SpecialForm::Final => matches!(
                self,
                TypeFormContext::VarAnnotation(AnnAssignHasValue::Yes)
                    | TypeFormContext::ClassVarAnnotation
            ),
            SpecialForm::LiteralString
            | SpecialForm::Never
            | SpecialForm::NoReturn
            | SpecialForm::Type
            | SpecialForm::SelfType => true,
            _ => false,
        }
    }
}

#[derive(Debug)]
pub enum Iterable {
    OfType(Type),
    FixedLen(Vec<Type>),
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub fn solve_legacy_tparam(
        &self,
        binding: &BindingLegacyTypeParam,
    ) -> Arc<LegacyTypeParameterLookup> {
        let maybe_parameter = match binding {
            BindingLegacyTypeParam::ParamKeyed(k) => self.get_idx(*k),
            BindingLegacyTypeParam::ModuleKeyed(k, attr) => {
                let module = self.get_idx(*k);
                // Errors in attribute lookup are reported elsewhere, so we provide dummy values
                // for arguments related to error reporting.
                self.attr_infer(
                    &module,
                    attr,
                    TextRange::default(),
                    &self.error_swallower(),
                    None,
                )
                .into()
            }
        };
        match maybe_parameter.ty() {
            Type::TypeVar(x) => {
                let q = Quantified::type_var(
                    x.qname().id().clone(),
                    self.uniques,
                    x.default().cloned(),
                    x.restriction().clone(),
                );
                Arc::new(LegacyTypeParameterLookup::Parameter(TParam {
                    quantified: q,
                    variance: x.variance(),
                }))
            }
            Type::TypeVarTuple(x) => {
                let q = Quantified::type_var_tuple(
                    x.qname().id().clone(),
                    self.uniques,
                    x.default().cloned(),
                );
                Arc::new(LegacyTypeParameterLookup::Parameter(TParam {
                    quantified: q,
                    variance: PreInferenceVariance::PInvariant,
                }))
            }
            Type::ParamSpec(x) => {
                let q = Quantified::param_spec(
                    x.qname().id().clone(),
                    self.uniques,
                    x.default().cloned(),
                );
                Arc::new(LegacyTypeParameterLookup::Parameter(TParam {
                    quantified: q,
                    variance: PreInferenceVariance::PInvariant,
                }))
            }
            ty => Arc::new(LegacyTypeParameterLookup::NotParameter(ty.clone())),
        }
    }

    pub fn solve_class_metadata(
        &self,
        binding: &BindingClassMetadata,
        errors: &ErrorCollector,
    ) -> Arc<ClassMetadata> {
        let BindingClassMetadata {
            class_idx: k,
            bases,
            keywords,
            decorators,
            is_new_type,
            pydantic_config_dict,
            django_primary_key_field,
        } = binding;
        let metadata = match &self.get_idx(*k).0 {
            None => ClassMetadata::recursive(),
            Some(cls) => self.class_metadata_of(
                cls,
                bases,
                keywords,
                decorators,
                *is_new_type,
                pydantic_config_dict,
                django_primary_key_field.as_ref(),
                errors,
            ),
        };
        Arc::new(metadata)
    }

    pub fn solve_class_mro(
        &self,
        binding: &BindingClassMro,
        errors: &ErrorCollector,
    ) -> Arc<ClassMro> {
        let mro = match &self.get_idx(binding.class_idx).0 {
            None => ClassMro::recursive(),
            Some(cls) => self.calculate_class_mro(cls, errors),
        };
        Arc::new(mro)
    }

    pub fn solve_abstract_members(
        &self,
        cls: &Class,
        errors: &ErrorCollector,
    ) -> Arc<AbstractClassMembers> {
        let metadata = self.get_metadata_for_class(cls);
        let abstract_members = self.calculate_abstract_members(cls);
        let unimplemented = abstract_members.unimplemented_abstract_methods();
        if !unimplemented.is_empty() {
            let members = unimplemented
                .iter()
                .map(|member| format!("`{member}`"))
                .collect::<Vec<_>>()
                .join(", ");
            if !metadata.is_protocol() && metadata.is_final() {
                self.error(
                    errors,
                    cls.range(),
                    ErrorInfo::Kind(ErrorKind::BadClassDefinition),
                    format!(
                        "Final class `{}` cannot have unimplemented abstract members: {}",
                        cls.name(),
                        members
                    ),
                );
            } else if !metadata.is_protocol()
                && !metadata.is_new_type()
                && !metadata.is_explicitly_abstract()
            {
                self.error(
                    errors,
                    cls.range(),
                    ErrorInfo::Kind(ErrorKind::ImplicitAbstractClass),
                    format!(
                        "Class `{}` has unimplemented abstract members: {}",
                        cls.name(),
                        members
                    ),
                );
            }
        }
        Arc::new(abstract_members)
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
                        ty.subst_self_special_form_mut(&Type::SelfType(
                            self.as_class_type_unchecked(cls),
                        ));
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

    /// Check that got is assignable to want
    pub fn is_subset_eq(&self, got: &Type, want: &Type) -> bool {
        self.is_subset_eq_with_reason(got, want).is_ok()
    }

    pub fn is_subset_eq_with_reason(&self, got: &Type, want: &Type) -> Result<(), SubsetError> {
        self.solver().is_subset_eq(got, want, self.type_order())
    }

    /// Check that got and want are consistent with each other
    pub fn is_equal(&self, got: &Type, want: &Type) -> bool {
        self.solver().is_equal(got, want, self.type_order()).is_ok()
    }

    pub fn expr_class_keyword(&self, x: &Expr, errors: &ErrorCollector) -> Annotation {
        // For now, we happen to know that ReadOnly is the only qualifier we support here, so we can
        // make some simplifying assumptions about what patterns we need to match. We swallow
        // errors from expr_qualifier() because expr_infer will produce the same errors anyway.
        match x {
            Expr::Subscript(x)
                if let Some(qualifier) = self.expr_qualifier(
                    &x.value,
                    TypeFormContext::ClassKeyword,
                    &self.error_swallower(),
                ) =>
            {
                Annotation {
                    qualifiers: vec![qualifier],
                    ty: Some(self.expr_infer(&x.slice, errors)),
                }
            }
            _ => Annotation::new_type(self.expr_infer(x, errors)),
        }
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
                Some(Qualifier::ClassVar | Qualifier::NotRequired | Qualifier::Required)
                    if type_form_context != TypeFormContext::ClassVarAnnotation =>
                {
                    self.error(
                        errors,
                        x.range(),
                        ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                        format!("`{special}` is only allowed inside a class body"),
                    );
                    None
                }
                Some(Qualifier::ReadOnly)
                    if !matches!(
                        type_form_context,
                        TypeFormContext::ClassVarAnnotation | TypeFormContext::ClassKeyword
                    ) =>
                {
                    self.error(
                        errors,
                        x.range(),
                        ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                        format!("`{special}` is only allowed inside a class body or class keyword"),
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
                        ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                        format!(
                            "`{special}` is only allowed on a class or local variable annotation"
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
                        ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                        "`TypeAlias` is only allowed on variable annotations".to_owned(),
                    );
                    None
                }
                _ => qualifier,
            }
        } else if let Some(ty) = ty
            && let Type::ClassDef(cls) = &ty
            && cls.has_toplevel_qname("dataclasses", "InitVar")
        {
            Some(Qualifier::InitVar)
        } else {
            None
        }
    }

    fn has_valid_annotation_syntax(&self, x: &Expr, errors: &ErrorCollector) -> bool {
        // Note that this function only checks for correct syntax.
        // Semantic validation (e.g. that `typing.Self` is used in a class
        // context, or that a string evaluates to a proper type expression) is
        // handled elsewhere.
        // See https://typing.readthedocs.io/en/latest/spec/annotations.html#type-and-annotation-expressions
        let problem = match x {
            Expr::Name(..)
            | Expr::BinOp(ExprBinOp {
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
                | Expr::BinOp(ExprBinOp {
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
            Expr::TString(..) => "t-string",
            Expr::UnaryOp(..) => "unary operation",
            Expr::BinOp(ExprBinOp { op, .. }) => &format!("binary operation `{}`", op.as_str()),
            // There are many Expr variants. Not all of them are likely to be used
            // in annotations, even accidentally. We can add branches for specific
            // expression constructs if desired.
            _ => "expression",
        };
        self.error(
            errors,
            x.range(),
            ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
            format!("{problem} cannot be used in annotations"),
        );
        false
    }

    fn expr_annotation(
        &self,
        x: &Expr,
        type_form_context: TypeFormContext,
        errors: &ErrorCollector,
    ) -> Annotation {
        if !self.has_valid_annotation_syntax(x, errors) {
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
                            TypeFormContext::VarAnnotation(AnnAssignHasValue::No)
                        ) => {}
                    _ => {
                        self.error(
                            errors,
                            x.range(),
                            ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                            format!("Expected a type argument for `{qualifier}`"),
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
                            ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                            "`Annotated` needs at least one piece of metadata in addition to the type".to_owned(),
                        );
                    }
                } else if unpacked_slice.len() != 1 {
                    self.error(
                        errors,
                        x.range(),
                        ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                        format!(
                            "Expected 1 type argument for `{}`, got {}",
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
                        ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                        "`ClassVar` arguments may not contain any type variables".to_owned(),
                    );
                }
                if qualifier == Qualifier::Final && ann.is_class_var() {
                    self.error(
                        errors,
                        unpacked_slice[0].range(),
                        ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                        "`ClassVar` may not be nested inside `Final`".to_owned(),
                    );
                }
                if (qualifier == Qualifier::Required
                    && ann.qualifiers.contains(&Qualifier::NotRequired))
                    || (qualifier == Qualifier::NotRequired
                        && ann.qualifiers.contains(&Qualifier::Required))
                {
                    self.error(
                        errors,
                        x.range(),
                        ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                        "Cannot combine `Required` and `NotRequired` for a TypedDict field"
                            .to_owned(),
                    );
                }
                if qualifier != Qualifier::Annotated && ann.qualifiers.contains(&qualifier) {
                    self.error(
                        errors,
                        x.range(),
                        ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                        format!("Duplicate qualifier `{qualifier}`"),
                    );
                } else {
                    ann.qualifiers.insert(0, qualifier);
                }
                ann
            }
            _ => {
                let ann_ty = self.expr_untype(x, type_form_context, errors);
                if let Type::SpecialForm(special_form) = ann_ty
                    && !type_form_context.is_valid_unparameterized_annotation(special_form)
                {
                    if special_form.can_be_subscripted() {
                        self.error(
                            errors,
                            x.range(),
                            ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                            format!("Expected a type argument for `{special_form}`"),
                        );
                    } else {
                        self.error(
                            errors,
                            x.range(),
                            ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                            format!("`{special_form}` is not allowed in this context"),
                        );
                    }
                }
                Annotation::new_type(ann_ty)
            }
        }
    }

    /// Given an `iterable` type, determine the iteration type; this is the type
    /// of `x` if we were to loop using `for x in iterable`.
    ///
    /// Returns a Vec of length 1 unless the iterable is a union, in which case the
    /// caller must handle each case.
    pub fn iterate(
        &self,
        iterable: &Type,
        range: TextRange,
        errors: &ErrorCollector,
        orig_context: Option<&dyn Fn() -> ErrorContext>,
    ) -> Vec<Iterable> {
        // Use the iterable protocol interfaces to determine the iterable type.
        // Special cases like Tuple should be intercepted first.
        let context = || {
            orig_context.map_or_else(
                || ErrorContext::Iteration(self.for_display(iterable.clone())),
                |ctx| ctx(),
            )
        };
        match iterable {
            Type::ClassType(cls) if let Some(Tuple::Concrete(elts)) = self.as_tuple(cls) => {
                vec![Iterable::FixedLen(elts.clone())]
            }
            Type::Tuple(Tuple::Concrete(elts)) => vec![Iterable::FixedLen(elts.clone())],
            Type::Var(v) if let Some(_guard) = self.recurse(*v) => {
                self.iterate(&self.solver().force_var(*v), range, errors, orig_context)
            }
            Type::Union(ts) => ts
                .iter()
                .flat_map(|t| self.iterate(t, range, errors, orig_context))
                .collect(),
            _ => {
                let ty = self
                    .unwrap_iterable(iterable)
                    .or_else(|| {
                        let int_ty = self.stdlib.int().clone().to_type();
                        let arg = CallArg::ty(&int_ty, range);
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
                            ErrorInfo::Kind(ErrorKind::NotIterable),
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
            Type::Var(v) if let Some(_guard) = self.recurse(*v) => {
                self.async_iterate(&self.solver().force_var(*v), range, errors)
            }
            _ => {
                let context = || ErrorContext::AsyncIteration(self.for_display(iterable.clone()));
                let ty = self.unwrap_async_iterable(iterable).unwrap_or_else(|| {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::NotIterable),
                        context().format(),
                    )
                });
                vec![Iterable::OfType(ty)]
            }
        }
    }

    pub fn get_produced_type(&self, iterables: Vec<Iterable>) -> Type {
        let mut produced_types = Vec::new();
        for iterable in iterables {
            match iterable {
                Iterable::OfType(t) => produced_types.push(t),
                Iterable::FixedLen(ts) => produced_types.extend(ts),
            }
        }
        self.unions(produced_types)
    }

    fn check_is_exception(
        &self,
        x: &Expr,
        range: TextRange,
        allow_none: bool,
        errors: &ErrorCollector,
    ) {
        let actual_type = self.expr_infer(x, errors);
        let base_exception_class = self.stdlib.base_exception();
        let base_exception_class_type = Type::ClassDef(base_exception_class.class_object().dupe());
        let base_exception_type = base_exception_class.clone().to_type();
        let mut expected_types = vec![base_exception_type, base_exception_class_type];
        let mut expected = "`BaseException`";
        if allow_none {
            expected_types.push(Type::None);
            expected = "`BaseException` or `None`"
        }
        if !self.is_subset_eq(&actual_type, &Type::Union(expected_types)) {
            self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::BadRaise),
                format!(
                    "Expression `{}` has type `{}`, expected {}",
                    self.module().display(x),
                    self.for_display(actual_type),
                    expected,
                ),
            );
        }
    }

    fn tvars_to_tparams_for_type_alias_type(
        &self,
        exprs: &Vec<Expr>,
        seen_type_vars: &mut SmallMap<TypeVar, Quantified>,
        seen_type_var_tuples: &mut SmallMap<TypeVarTuple, Quantified>,
        seen_param_specs: &mut SmallMap<ParamSpec, Quantified>,
        tparams: &mut Vec<TParam>,
        errors: &ErrorCollector,
    ) {
        for expr in exprs {
            let ty = self.expr_infer(expr, errors);
            let ty = self.untype(ty, expr.range(), errors);
            if ty == Type::any_error() {
                continue;
            }
            match ty {
                Type::TypeVar(ty_var) => {
                    match seen_type_vars.entry(ty_var.dupe()) {
                        Entry::Occupied(_) => {
                            self.error(
                                errors,
                                expr.range(),
                                ErrorInfo::Kind(ErrorKind::InvalidTypeAlias),
                                format!("Duplicate type variable `{}`", ty_var.qname().id()),
                            );
                        }
                        Entry::Vacant(e) => {
                            let q = Quantified::type_var(
                                ty_var.qname().id().clone(),
                                self.uniques,
                                ty_var.default().cloned(),
                                ty_var.restriction().clone(),
                            );
                            e.insert(q.clone());
                            tparams.push(TParam {
                                quantified: q.clone(),
                                variance: ty_var.variance(),
                            });
                        }
                    };
                }
                Type::TypeVarTuple(ty_var_tuple) => {
                    match seen_type_var_tuples.entry(ty_var_tuple.dupe()) {
                        Entry::Occupied(_) => {
                            self.error(
                                errors,
                                expr.range(),
                                ErrorInfo::Kind(ErrorKind::InvalidTypeAlias),
                                format!("Duplicate type variable `{}`", ty_var_tuple.qname().id()),
                            );
                        }
                        Entry::Vacant(e) => {
                            let q = Quantified::type_var_tuple(
                                ty_var_tuple.qname().id().clone(),
                                self.uniques,
                                ty_var_tuple.default().cloned(),
                            );
                            e.insert(q.clone());
                            tparams.push(TParam {
                                quantified: q.clone(),
                                variance: PreInferenceVariance::PInvariant,
                            });
                        }
                    };
                }
                Type::ParamSpec(param_spec) => {
                    match seen_param_specs.entry(param_spec.dupe()) {
                        Entry::Occupied(_) => {
                            self.error(
                                errors,
                                expr.range(),
                                ErrorInfo::Kind(ErrorKind::InvalidTypeAlias),
                                format!("Duplicate type variable `{}`", param_spec.qname().id()),
                            );
                        }
                        Entry::Vacant(e) => {
                            let q = Quantified::param_spec(
                                param_spec.qname().id().clone(),
                                self.uniques,
                                param_spec.default().cloned(),
                            );
                            e.insert(q.clone());
                            tparams.push(TParam {
                                quantified: q.clone(),
                                variance: PreInferenceVariance::PInvariant,
                            });
                        }
                    };
                }
                _ => {
                    self.error(
                        errors,
                        expr.range(),
                        ErrorInfo::Kind(ErrorKind::InvalidTypeAlias),
                        format!("Expected a type variable, got `{}`", self.for_display(ty),),
                    );
                }
            }
        }
    }

    fn tvars_to_tparams_for_type_alias(
        &self,
        ty: &mut Type,
        seen_type_vars: &mut SmallMap<TypeVar, Quantified>,
        seen_type_var_tuples: &mut SmallMap<TypeVarTuple, Quantified>,
        seen_param_specs: &mut SmallMap<ParamSpec, Quantified>,
        tparams: &mut Vec<TParam>,
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
            Type::Concatenate(prefix, pspec) => {
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
                        tparams.push(TParam {
                            quantified: q.clone(),
                            variance: ty_var.variance(),
                        });
                        q
                    }
                };
                *ty = q.to_type();
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
                        tparams.push(TParam {
                            quantified: q.clone(),
                            variance: PreInferenceVariance::PInvariant,
                        });
                        q
                    }
                };
                *ty = q.to_type();
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
                        tparams.push(TParam {
                            quantified: q.clone(),
                            variance: PreInferenceVariance::PInvariant,
                        });
                        q
                    }
                };
                *ty = q.to_type();
            }
            Type::Unpack(t) => self.tvars_to_tparams_for_type_alias(
                t,
                seen_type_vars,
                seen_type_var_tuples,
                seen_param_specs,
                tparams,
            ),
            Type::Type(t) => self.tvars_to_tparams_for_type_alias(
                t,
                seen_type_vars,
                seen_type_var_tuples,
                seen_param_specs,
                tparams,
            ),
            _ => {}
        }
    }

    /// `typealiastype_tparams` refers specifically to the elements of the tuple literal passed to the `TypeAliasType` constructor
    /// For all other kinds of type aliases, it should be `None`.
    ///
    /// When present, we visit those types first to determine the `TParams` for this alias, and any
    /// type variables when we subsequently visit the aliased type are considered out of scope.
    ///
    /// `legacy_tparams` refers to the type parameters collected in the bindings phase. It is only populated if we know for sure
    /// that this is actually a type alias, like when a variable assignment is annotated with `TypeAlias`
    fn as_type_alias(
        &self,
        name: &Name,
        style: TypeAliasStyle,
        ty: Type,
        expr: &Expr,
        typealiastype_tparams: Option<Vec<Expr>>,
        legacy_tparams: &Option<Box<[Idx<KeyLegacyTypeParam>]>>,
        errors: &ErrorCollector,
    ) -> Type {
        let range = expr.range();
        if !self.has_valid_annotation_syntax(expr, errors) {
            return Type::any_error();
        }
        let untyped = self.untype_opt(ty.clone(), range, errors);
        let mut ty = if let Some(untyped) = untyped {
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
                ErrorInfo::Kind(ErrorKind::InvalidTypeAlias),
                format!("Expected `{name}` to be a type alias, got `{ty}`"),
            );
            return Type::any_error();
        };
        let mut seen_type_vars = SmallMap::new();
        let mut seen_type_var_tuples = SmallMap::new();
        let mut seen_param_specs = SmallMap::new();
        let mut tparams = Vec::new();
        let mut tparams_for_type_alias_type = None;
        if let Some(type_params) = &typealiastype_tparams {
            self.tvars_to_tparams_for_type_alias_type(
                type_params,
                &mut seen_type_vars,
                &mut seen_type_var_tuples,
                &mut seen_param_specs,
                &mut tparams,
                errors,
            );
            tparams_for_type_alias_type = Some(tparams.len());
        }
        if let Some(legacy_tparams) = legacy_tparams {
            tparams = legacy_tparams
                .iter()
                .filter_map(|key| self.get_idx(*key).deref().parameter().cloned())
                .collect();
        } else {
            self.tvars_to_tparams_for_type_alias(
                &mut ty,
                &mut seen_type_vars,
                &mut seen_type_var_tuples,
                &mut seen_param_specs,
                &mut tparams,
            );
        }
        if let Some(n) = tparams_for_type_alias_type {
            for extra_tparam in tparams.iter().skip(n) {
                errors.add(
                    expr.range(),
                    ErrorInfo::Kind(ErrorKind::InvalidTypeAlias),
                    vec1![
                        format!(
                            "Type variable `{}` is out of scope for this `TypeAliasType`",
                            extra_tparam.name()
                        ),
                        format!(
                            "Type parameters must be passed as a tuple literal to the `type_params` argument",
                        )
                    ],
                );
            }
        }
        // Extract Annotated metadata; skip the first element since that's the type and collect the rest of the vector
        let annotated_metadata = match expr {
            Expr::Subscript(s)
                if matches!(
                    self.expr_qualifier(&s.value, TypeFormContext::TypeAlias, errors),
                    Some(Qualifier::Annotated)
                ) =>
            {
                Ast::unpack_slice(&s.slice)
                    .iter()
                    .skip(1)
                    .map(|e| self.expr_infer(e, &self.error_swallower()))
                    .collect()
            }
            _ => Vec::new(),
        };

        let ta = TypeAlias::new(name.clone(), Type::type_form(ty), style, annotated_metadata);

        Forallable::TypeAlias(ta).forall(self.validated_tparams(
            range,
            tparams,
            TParamsSource::TypeAlias,
            errors,
        ))
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
                    ErrorInfo::new(ErrorKind::NotAsync, context),
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
        // Call `__exit__` or `__aexit__` and unwrap the results if async, swallowing any errors from the call itself
        let call_exit = |exit_arg_types, swallow_errors| match kind {
            IsAsync::Sync => self.call_method_or_error(
                context_manager_type,
                &kind.context_exit_dunder(),
                range,
                exit_arg_types,
                &[],
                swallow_errors,
                context,
            ),
            IsAsync::Async => match self.unwrap_awaitable(&self.call_method_or_error(
                context_manager_type,
                &kind.context_exit_dunder(),
                range,
                exit_arg_types,
                &[],
                swallow_errors,
                context,
            )) {
                Some(ty) => ty,
                // We emit this error directly, since it's different from type checking the arguments
                None => self.error(
                    errors,
                    range,
                    ErrorInfo::new(ErrorKind::NotAsync, context),
                    format!("Expected `{}` to be async", dunder::AEXIT),
                ),
            },
        };
        let base_exception_class_type =
            Type::type_form(self.stdlib.base_exception().clone().to_type());
        let arg1 = base_exception_class_type;
        let arg2 = self.stdlib.base_exception().clone().to_type();
        let arg3 = self.stdlib.traceback_type().clone().to_type();
        let exit_with_error_args = [
            CallArg::ty(&arg1, range),
            CallArg::ty(&arg2, range),
            CallArg::ty(&arg3, range),
        ];
        let exit_ok_args = [
            CallArg::ty(&Type::None, range),
            CallArg::ty(&Type::None, range),
            CallArg::ty(&Type::None, range),
        ];
        let exit_with_error_errors =
            ErrorCollector::new(errors.module().clone(), ErrorStyle::Delayed);
        let exit_with_ok_errors = ErrorCollector::new(errors.module().clone(), ErrorStyle::Delayed);
        let error_args_result = call_exit(&exit_with_error_args, &exit_with_error_errors);
        let ok_args_result = call_exit(&exit_ok_args, &exit_with_ok_errors);
        // If the call only has one error we can directly forward it
        // If there is more than one error, we emit a generic error instead of emitting one error for each mismatched argument
        if exit_with_error_errors.len() <= 1 {
            errors.extend(exit_with_error_errors);
        } else {
            self.error(
                errors,
                range,
                ErrorInfo::new(ErrorKind::BadContextManager, context),
                format!("`{}` must be callable with the argument types (type[BaseException], BaseException, TracebackType)", kind.context_exit_dunder()),
            );
        }
        if exit_with_ok_errors.len() <= 1 {
            errors.extend(exit_with_ok_errors);
        } else {
            self.error(
                errors,
                range,
                ErrorInfo::new(ErrorKind::BadContextManager, context),
                format!(
                    "`{}` must be callable with the argument types (None, None, None)",
                    kind.context_exit_dunder()
                ),
            );
        }
        self.union(error_args_result, ok_args_result)
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
                &exit_type,
                &Type::optional(self.stdlib.bool().clone().to_type()),
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

    pub fn scoped_type_params(&self, x: Option<&TypeParams>) -> Vec<TParam> {
        match x {
            Some(x) => {
                fn get_quantified(t: &Type) -> Quantified {
                    match t {
                        Type::QuantifiedValue(q) => (**q).clone(),
                        _ => unreachable!(),
                    }
                }
                let mut params = Vec::new();
                for raw_param in x.type_params.iter() {
                    let name = raw_param.name();
                    let quantified =
                        get_quantified(self.get(&Key::Definition(ShortIdentifier::new(name))).ty());
                    params.push(TParam {
                        quantified,
                        variance: PreInferenceVariance::PUndefined,
                    });
                }
                params
            }
            None => Vec::new(),
        }
    }

    fn validate_type_params(
        &self,
        range: TextRange,
        tparams: &[TParam],
        source: TParamsSource,
        errors: &ErrorCollector,
    ) {
        let mut last_tparam: Option<&TParam> = None;
        let mut seen = SmallSet::new();
        let mut typevartuple = None;
        let mut typevartuple_count = 0;
        for tparam in tparams {
            if let Some(p) = last_tparam
                && p.quantified.default().is_some()
            {
                // Check for missing default
                if tparam.quantified.default().is_none() {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::InvalidTypeVar),
                        format!(
                            "Type parameter `{}` without a default cannot follow type parameter `{}` with a default",
                            tparam.quantified.name(),
                            p.name()
                        )
                    );
                }
            }
            if let Some(default) = tparam.quantified.default() {
                let mut out_of_scope_names = Vec::new();
                default.universe(&mut |t| {
                    let name = match t {
                        Type::TypeVar(t) => t.qname().id(),
                        Type::TypeVarTuple(t) => t.qname().id(),
                        Type::ParamSpec(p) => p.qname().id(),
                        _ => return,
                    };
                    if !seen.contains(name) {
                        out_of_scope_names.push(name);
                    }
                });
                if !out_of_scope_names.is_empty() {
                    self.error(errors, range, ErrorInfo::Kind(ErrorKind::InvalidTypeVar), format!(
                        "Default of type parameter `{}` refers to out-of-scope type parameter{} {}",
                        tparam.quantified.name(),
                        if out_of_scope_names.len() != 1 {
                            "s"
                        } else {
                            ""
                        },
                        out_of_scope_names.map(|n| format!("`{n}`")).join(", "),
                    ));
                }
                if tparam.quantified.is_type_var()
                    && let Some(tvt) = &typevartuple
                {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::InvalidTypeVar),
                        format!(
                            "TypeVar `{}` with a default cannot follow TypeVarTuple `{}`",
                            tparam.quantified.name(),
                            tvt
                        ),
                    );
                }
            }
            seen.insert(tparam.quantified.name().clone());
            if tparam.quantified.is_type_var_tuple() {
                typevartuple = Some(tparam.quantified.name().clone());
                typevartuple_count += 1;
            }
            last_tparam = Some(tparam);
        }
        if typevartuple_count > 1
            && matches!(source, TParamsSource::Class | TParamsSource::TypeAlias)
        {
            self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::InvalidTypeVarTuple),
                format!("Type parameters for {source} may not have more than one TypeVarTuple")
                    .to_owned(),
            );
        }
    }

    pub fn validated_tparams(
        &self,
        range: TextRange,
        tparams: Vec<TParam>,
        source: TParamsSource,
        errors: &ErrorCollector,
    ) -> Arc<TParams> {
        self.validate_type_params(range, &tparams, source, errors);
        Arc::new(TParams::new(tparams))
    }

    pub fn solve_binding(&self, binding: &Binding, errors: &ErrorCollector) -> Arc<TypeInfo> {
        // Special case for forward, as we don't want to re-expand the type
        if let Binding::Forward(fwd) = binding {
            return self.get_idx(*fwd);
        }
        let mut type_info = self.binding_to_type_info(binding, errors);
        type_info.visit_mut(&mut |ty| {
            if !matches!(
                binding,
                Binding::NameAssign(..) | Binding::PartialTypeWithUpstreamsCompleted(..)
            ) {
                self.pin_all_placeholder_types(ty);
            }
            self.expand_vars_mut(ty);
        });
        Arc::new(type_info)
    }

    /// Force the outermost type, without deep-forcing. Without this, narrowing behavior
    /// is unpredictable and has undesirable behavior particularly in loop recursion.
    pub fn force_for_narrowing(&self, ty: &Type) -> Type {
        match ty {
            Type::Var(v) => self.force_for_narrowing(&self.solver().force_var(*v)),
            _ => ty.clone(),
        }
    }

    pub fn expand_vars_mut(&self, ty: &mut Type) {
        // Replace any solved recursive variables with their answers.
        self.solver().expand_vars_mut(ty);
    }

    fn check_del_typed_dict_field(
        &self,
        typed_dict: &Name,
        field_name: Option<&Name>,
        read_only: bool,
        required: bool,
        range: TextRange,
        errors: &ErrorCollector,
    ) {
        if read_only || required {
            let maybe_field_name = if let Some(field_name) = field_name {
                format!(" `{field_name}`")
            } else {
                "".to_owned()
            };
            self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::UnsupportedDelete),
                format!("Key{maybe_field_name} in TypedDict `{typed_dict}` may not be deleted"),
            );
        }
    }

    fn check_del_typed_dict_literal_key(
        &self,
        typed_dict: &TypedDict,
        field_name: &Name,
        range: TextRange,
        errors: &ErrorCollector,
    ) {
        let (read_only, required) =
            if let Some(field) = self.typed_dict_field(typed_dict, field_name) {
                (field.is_read_only(), field.required)
            } else if let ExtraItems::Extra(extra) =
                self.typed_dict_extra_items(typed_dict.class_object())
            {
                (extra.read_only, false)
            } else {
                self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::BadTypedDictKey),
                    format!(
                        "TypedDict `{}` does not have key `{}`",
                        typed_dict.name(),
                        field_name
                    ),
                );
                return;
            };
        self.check_del_typed_dict_field(
            typed_dict.name(),
            Some(field_name),
            read_only,
            required,
            range,
            errors,
        )
    }

    pub fn solve_expectation(
        &self,
        binding: &BindingExpect,
        errors: &ErrorCollector,
    ) -> Arc<EmptyAnswer> {
        match binding {
            BindingExpect::TypeCheckExpr(x) => {
                self.expr_infer(x, errors);
            }
            BindingExpect::TypeCheckBaseClassExpr(x) => {
                self.expr_untype(x, TypeFormContext::BaseClassList, errors);
            }
            BindingExpect::Bool(x) => {
                let ty = self.expr_infer(x, errors);
                self.check_dunder_bool_is_callable(&ty, x.range(), errors);
                self.check_redundant_condition(&ty, x.range(), errors);
            }
            BindingExpect::UnpackedLength(b, range, expect) => {
                let iterable_ty = self.get_idx(*b);
                let iterables = self.iterate(iterable_ty.ty(), *range, errors, None);
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
                                        ErrorInfo::Kind(ErrorKind::BadUnpacking),
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
                        ErrorInfo::Kind(ErrorKind::AnnotationMismatch),
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

    pub fn solve_consistent_override_check(
        &self,
        binding: &BindingConsistentOverrideCheck,
        errors: &ErrorCollector,
    ) -> Arc<EmptyAnswer> {
        if let Some(cls) = &self.get_idx(binding.class_key).0 {
            let class_bases = self.get_base_types_for_class(cls);
            for (name, field) in self.get_class_field_map(cls).iter() {
                self.check_consistent_override_for_field(
                    cls,
                    name,
                    field.as_ref(),
                    class_bases.as_ref(),
                    errors,
                );
            }

            // If we are inheriting from multiple base types, we should
            // check whether the multiple inheritance is consistent
            if class_bases.as_ref().base_type_count() > 1 {
                self.check_consistent_multiple_inheritance(cls, errors);
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
                x.def_index,
                &x.def,
                &x.parent,
                x.fields.clone(),
                x.tparams_require_binding,
                errors,
            ),
            BindingClass::FunctionalClassDef(def_index, x, parent, fields) => {
                self.functional_class_definition(*def_index, x, parent, fields)
            }
        };
        Arc::new(NoneIfRecursive(Some(cls)))
    }

    pub fn solve_tparams(&self, binding: &BindingTParams, errors: &ErrorCollector) -> Arc<TParams> {
        self.calculate_class_tparams(
            &binding.name,
            binding.scoped_type_params.as_deref(),
            &binding.generic_bases,
            &binding.legacy_tparams,
            errors,
        )
    }

    pub fn solve_class_base_type(
        &self,
        binding: &BindingClassBaseType,
        errors: &ErrorCollector,
    ) -> Arc<ClassBases> {
        let class_bases = match &self.get_idx(binding.class_idx).0 {
            None => ClassBases::recursive(),
            Some(cls) => self.class_bases_of(cls, &binding.bases, binding.is_new_type, errors),
        };
        Arc::new(class_bases)
    }

    pub fn solve_class_field(
        &self,
        field: &BindingClassField,
        errors: &ErrorCollector,
    ) -> Arc<ClassField> {
        let functional_class_def = matches!(
            self.bindings().get(field.class_idx),
            BindingClass::FunctionalClassDef(_, _, _, _)
        );
        let field = match &self.get_idx(field.class_idx).0 {
            None => ClassField::recursive(),
            Some(class) => self.calculate_class_field(
                class,
                &field.name,
                field.range,
                &field.definition,
                functional_class_def,
                errors,
            ),
        };
        Arc::new(field)
    }

    pub fn solve_class_synthesized_fields(
        &self,
        errors: &ErrorCollector,
        fields: &BindingClassSynthesizedFields,
    ) -> Arc<ClassSynthesizedFields> {
        let fields = match &self.get_idx(fields.0).0 {
            None => ClassSynthesizedFields::default(),
            Some(cls) => {
                let mut fields = ClassSynthesizedFields::default();
                if let Some(new_fields) = self.get_typed_dict_synthesized_fields(cls) {
                    fields = fields.combine(new_fields);
                }
                if let Some(new_fields) = self.get_dataclass_synthesized_fields(cls, errors) {
                    fields = fields.combine(new_fields);
                }
                if let Some(new_fields) = self.get_named_tuple_synthesized_fields(cls) {
                    fields = fields.combine(new_fields);
                }
                if let Some(new_fields) = self.get_new_type_synthesized_fields(cls) {
                    fields = fields.combine(new_fields);
                }
                if let Some(new_fields) = self.get_total_ordering_synthesized_fields(errors, cls) {
                    fields = fields.combine(new_fields);
                }
                if let Some(new_fields) = self.get_django_enum_synthesized_fields(cls) {
                    fields = fields.combine(new_fields);
                }
                if let Some(new_fields) = self.get_django_model_synthesized_fields(cls) {
                    fields = fields.combine(new_fields);
                }
                fields
            }
        };
        Arc::new(fields)
    }

    // TODO zeina: After doing the full implementation, look into extracting fields and
    // base types from existing bindings
    pub fn solve_variance_binding(&self, variance_info: &BindingVariance) -> Arc<VarianceMap> {
        let class_idx = variance_info.class_key;
        let class = self.get_idx(class_idx);

        if let Some(class) = &class.0 {
            self.variance_map(class)
        } else {
            Arc::new(VarianceMap::default())
        }
    }

    /// Get the class that attribute lookup on `super(cls, obj)` should be done on.
    /// This is the class above `cls` in `obj`'s MRO.
    fn get_super_lookup_class(&self, cls: &Class, obj: &ClassType) -> Option<ClassType> {
        let mut lookup_cls = None;
        let mro = self.get_mro_for_class(obj.class_object());
        let mut found = false;
        for ancestor in iter::once(obj).chain(mro.ancestors(self.stdlib)) {
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
                                        ErrorInfo::Kind(ErrorKind::InvalidSuperCall),
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
                            Type::Type(box Type::ClassType(obj_cls)) => {
                                make_super_instance(obj_cls, &|| SuperObj::Class(obj_cls.clone()))
                            }
                            Type::ClassDef(obj_cls) => {
                                let obj_type = self.type_order().as_class_type_unchecked(obj_cls);
                                make_super_instance(&obj_type, &|| SuperObj::Class(obj_type.clone()))
                            }
                            Type::SelfType(obj_cls) => {
                                make_super_instance(obj_cls, &|| SuperObj::Instance(obj_cls.clone()))
                            }
                            Type::Type(box Type::SelfType(obj_cls)) => {
                                make_super_instance(obj_cls, &|| SuperObj::Class(obj_cls.clone()))
                            }
                            t => {
                                self.error(
                                    errors,
                                    range,
                                    ErrorInfo::Kind(ErrorKind::InvalidArgument),
                                    format!("Expected second argument to `super` to be a class object or instance, got `{}`", self.for_display(t.clone())),
                                )
                            }
                        }
                    }
                    t => self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::InvalidArgument),
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
                        let obj_type = self.as_class_type_unchecked(obj_cls);
                        let lookup_cls = self.get_super_lookup_class(obj_cls, &obj_type).unwrap();
                        let obj = if method.id == dunder::NEW {
                            // __new__ is special: it's the only static method in which the
                            // no-argument form of super is allowed.
                            SuperObj::Class(obj_type.clone())
                        } else {
                            let method_ty =
                                self.get(&KeyUndecoratedFunction(ShortIdentifier::new(method)));
                            if method_ty.metadata.flags.is_staticmethod {
                                return self.error(
                                    errors,
                                    range,
                                    ErrorInfo::Kind(ErrorKind::InvalidSuperCall),
                                    "`super` call with no arguments is not valid inside a staticmethod".to_owned(),
                                );
                            } else if method_ty.metadata.flags.is_classmethod {
                                SuperObj::Class(obj_type.clone())
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
        pub fn quantified_error<'a>(kind: QuantifiedKind) -> ErrorInfo<'a> {
            ErrorInfo::Kind(match kind {
                QuantifiedKind::TypeVar => ErrorKind::InvalidTypeVar,
                QuantifiedKind::ParamSpec => ErrorKind::InvalidParamSpec,
                QuantifiedKind::TypeVarTuple => ErrorKind::InvalidTypeVarTuple,
            })
        }

        if default.is_error() {
            return default.clone();
        }
        match restriction {
            // Default must be a subtype of the upper bound
            Restriction::Bound(bound_ty) => {
                if !self.is_subset_eq(default, bound_ty) {
                    self.error(
                        errors,
                        range,
                        quantified_error(kind),
                        format!(
                            "Expected default `{default}` of `{name}` to be assignable to the upper bound of `{bound_ty}`",
                        ),
                    );
                    return Type::any_error();
                }
            }
            Restriction::Constraints(constraints) => {
                // Default must exactly match one of the constraints
                if !constraints.iter().any(|c| self.is_equal(c, default)) {
                    let formatted_constraints = constraints
                        .iter()
                        .map(|x| format!("`{x}`"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    self.error(
                        errors,
                        range,
                        quantified_error(kind),
                        format!(
                            "Expected default `{default}` of `{name}` to be one of the following constraints: {formatted_constraints}"
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
                        ErrorInfo::Kind(ErrorKind::InvalidParamSpec),
                        format!("Default for `ParamSpec` must be a parameter list, `...`, or another `ParamSpec`, got `{default}`"),
                    );
                    Type::any_error()
                }
            }
            QuantifiedKind::TypeVarTuple => {
                if let Type::Unpack(inner) = default
                    && (matches!(&**inner, Type::Tuple(_)) || inner.is_kind_type_var_tuple())
                {
                    (**inner).clone()
                } else {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::InvalidTypeVarTuple),
                        format!("Default for `TypeVarTuple` must be an unpacked tuple form or another `TypeVarTuple`, got `{default}`"),
                    );
                    Type::any_error()
                }
            }
            QuantifiedKind::TypeVar => {
                if default.is_kind_param_spec() || default.is_kind_type_var_tuple() {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::InvalidTypeVar),
                        format!( "Default for `TypeVar` may not be a `TypeVarTuple` or `ParamSpec`, got `{default}`"),
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
            Binding::Phi(join_style, ks) => {
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
                    TypeInfo::join(
                        type_infos,
                        &|ts| self.unions(ts),
                        &|got, want| self.is_subset_eq(got, want),
                        join_style.map(|idx| self.get_idx(*idx)),
                    )
                }
            }
            Binding::LoopPhi(default, ks) => {
                // We force the default first so that if we hit a recursive case it is already available
                self.get_idx(*default);
                // Then solve the phi like a regular Phi binding
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
                    TypeInfo::join(
                        type_infos,
                        &|ts| self.unions(ts),
                        &|got, want| self.is_subset_eq(got, want),
                        JoinStyle::SimpleMerge,
                    )
                }
            }
            Binding::AssignToAttribute(attr, got) => {
                // NOTE: Deterministic pinning of placeholder types based on first use relies on an
                // invariant: if `got` is used in the binding for a class field, we must always solve
                // that `ClassField` binding *before* analyzing `got`.
                //
                // This should be the case since contextual typing requires working out the class field
                // type information first, but is difficult to see from a skim.
                let base = self.expr_infer(&attr.value, errors);
                let narrowed = self.check_assign_to_attribute_and_infer_narrow(
                    &base,
                    &attr.attr.id,
                    got,
                    attr.range,
                    errors,
                );
                if let Some((identifier, chain)) =
                    identifier_and_chain_for_expr(&Expr::Attribute(attr.clone()))
                {
                    // Note that the value we are doing `self.get` on is the same one we did in infer_expr, which is a bit sad.
                    // But avoiding the duplicate get/clone would require us to duplicate some of infer_expr here, which might
                    // fall out of sync.
                    let mut type_info = self
                        .get(&Key::BoundName(ShortIdentifier::new(&identifier)))
                        .arc_clone();
                    type_info.update_for_assignment(chain.facets(), narrowed);
                    type_info
                } else if let Some((identifier, facets)) =
                    identifier_and_chain_prefix_for_expr(&Expr::Attribute(attr.clone()))
                {
                    // If the chain contains an unknown subscript index, we clear narrowing for
                    // all indexes of its parent.
                    let mut type_info = self
                        .get(&Key::BoundName(ShortIdentifier::new(&identifier)))
                        .arc_clone();
                    type_info.invalidate_all_indexes_for_assignment(&facets);
                    type_info
                } else {
                    // Placeholder: in this case, we're assigning to an anonymous base and the
                    // type info will not propagate anywhere.
                    TypeInfo::of_ty(Type::never())
                }
            }
            Binding::AssignToSubscript(subscript, value) => {
                // If we can't assign to this subscript, then we don't narrow the type
                let assigned_ty = self.check_assign_to_subscript(subscript, value, errors);
                let narrowed = if assigned_ty.is_any() {
                    None
                } else {
                    Some(assigned_ty)
                };
                if let Some((identifier, chain)) =
                    identifier_and_chain_for_expr(&Expr::Subscript(subscript.clone()))
                {
                    let mut type_info = self
                        .get(&Key::BoundName(ShortIdentifier::new(&identifier)))
                        .arc_clone();
                    type_info.update_for_assignment(chain.facets(), narrowed);
                    type_info
                } else if let Some((identifier, facets)) =
                    identifier_and_chain_prefix_for_expr(&Expr::Subscript(subscript.clone()))
                {
                    // If the chain contains an unknown subscript index, we clear narrowing for
                    // all indexes of its parent.
                    let mut type_info = self
                        .get(&Key::BoundName(ShortIdentifier::new(&identifier)))
                        .arc_clone();
                    type_info.invalidate_all_indexes_for_assignment(&facets);
                    type_info
                } else {
                    // Placeholder: in this case, we're assigning to an anonymous base and the
                    // type info will not propagate anywhere.
                    TypeInfo::of_ty(Type::never())
                }
            }
            Binding::PossibleLegacyTParam(key, range_if_scoped_params_exist) => {
                let ty = match &*self.get_idx(*key) {
                    LegacyTypeParameterLookup::Parameter(p) => {
                        // This class or function has scoped (PEP 695) type parameters. Mixing legacy-style parameters is an error.
                        if let Some(r) = range_if_scoped_params_exist {
                            self.error(
                                errors,
                                *r,
                                ErrorInfo::Kind(ErrorKind::InvalidTypeVar),
                                format!(
                                    "Type parameter {} is not included in the type parameter list",
                                    self.module().display(&self.bindings().idx_to_key(*key).0)
                                ),
                            );
                        }
                        p.quantified.clone().to_value()
                    }
                    LegacyTypeParameterLookup::NotParameter(ty) => ty.clone(),
                };
                match self.bindings().get(*key) {
                    BindingLegacyTypeParam::ModuleKeyed(idx, attr) => {
                        // `idx` points to a module whose `attr` attribute may be a legacy type
                        // variable that needs to be replaced with a QuantifiedValue. Since the
                        // ModuleKeyed binding is for the module itself, we use the mechanism for
                        // attribute ("facet") type narrowing to change the type that will be
                        // produced when `attr` is accessed.
                        let module = (*self.get_idx(*idx)).clone();
                        if matches!(ty, Type::QuantifiedValue(_)) {
                            module.with_narrow(&vec1![FacetKind::Attribute((**attr).clone())], ty)
                        } else {
                            module
                        }
                    }
                    BindingLegacyTypeParam::ParamKeyed(_) => TypeInfo::of_ty(ty),
                }
            }
            _ => {
                // All other Bindings model `Type` level operations where we do not
                // propagate any attribute narrows.
                TypeInfo::of_ty(self.binding_to_type(binding, errors))
            }
        }
    }

    fn check_assign_to_typed_dict_field(
        &self,
        typed_dict: &Name,
        field_name: Option<&Name>,
        field_ty: &Type,
        read_only: bool,
        value: &ExprOrBinding,
        key_range: TextRange,
        assign_range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        if read_only {
            let key = if let Some(field_name) = field_name {
                format!("Key `{field_name}`")
            } else {
                "`extra_items`".to_owned()
            };
            self.error(
                errors,
                key_range,
                ErrorInfo::Kind(ErrorKind::ReadOnly),
                format!("{key} in TypedDict `{typed_dict}` is read-only"),
            )
        } else {
            let context =
                &|| TypeCheckContext::of_kind(TypeCheckKind::TypedDictKey(field_name.cloned()));
            match value {
                ExprOrBinding::Expr(e) => self.expr(e, Some((field_ty, context)), errors),
                ExprOrBinding::Binding(b) => {
                    let binding_ty = self.solve_binding(b, errors).arc_clone_ty();
                    self.check_and_return_type(binding_ty, field_ty, assign_range, errors, context)
                }
            }
        }
    }

    fn check_assign_to_typed_dict_literal_subscript(
        &self,
        typed_dict: &TypedDict,
        field_name: &Name,
        value: &ExprOrBinding,
        key_range: TextRange,
        assign_range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        let (field_ty, read_only) =
            if let Some(field) = self.typed_dict_field(typed_dict, field_name) {
                let read_only = field.is_read_only();
                (field.ty, read_only)
            } else if let ExtraItems::Extra(extra) =
                self.typed_dict_extra_items(typed_dict.class_object())
            {
                (extra.ty, extra.read_only)
            } else {
                return self.error(
                    errors,
                    key_range,
                    ErrorInfo::Kind(ErrorKind::BadTypedDictKey),
                    format!(
                        "TypedDict `{}` does not have key `{}`",
                        typed_dict.name(),
                        field_name
                    ),
                );
            };
        self.check_assign_to_typed_dict_field(
            typed_dict.name(),
            Some(field_name),
            &field_ty,
            read_only,
            value,
            key_range,
            assign_range,
            errors,
        )
    }

    fn check_assign_to_subscript(
        &self,
        subscript: &ExprSubscript,
        value: &ExprOrBinding,
        errors: &ErrorCollector,
    ) -> Type {
        let base = self.expr_infer(&subscript.value, errors);
        let slice_ty = self.expr_infer(&subscript.slice, errors);
        self.distribute_over_union(&base, |base| {
            self.distribute_over_union(&slice_ty, |key| {
                match (base, key) {
                    (Type::TypedDict(typed_dict), Type::Literal(Lit::Str(field_name))) => {
                        let field_name = Name::new(field_name);
                        self.check_assign_to_typed_dict_literal_subscript(
                            typed_dict,
                            &field_name,
                            value,
                            subscript.slice.range(),
                            subscript.range(),
                            errors,
                        )
                    }
                    (Type::TypedDict(typed_dict), Type::ClassType(cls))
                        if cls.is_builtin("str")
                            && let Some(field_ty) =
                                self.get_typed_dict_value_type_as_builtins_dict(typed_dict) =>
                    {
                        self.check_assign_to_typed_dict_field(
                            typed_dict.name(),
                            None,
                            &field_ty,
                            false,
                            value,
                            subscript.slice.range(),
                            subscript.range(),
                            errors,
                        )
                    }
                    (_, _) => {
                        let call_setitem = |value_arg| {
                            self.call_method_or_error(
                                base,
                                &dunder::SETITEM,
                                subscript.range,
                                &[CallArg::ty(key, subscript.slice.range()), value_arg],
                                &[],
                                errors,
                                Some(&|| ErrorContext::SetItem(self.for_display(base.clone()))),
                            )
                        };
                        match value {
                            ExprOrBinding::Expr(e) => {
                                call_setitem(CallArg::expr(e));
                                // We already emit errors for `e` during `call_method_or_error`
                                self.expr_infer(
                                    e,
                                    &ErrorCollector::new(
                                        errors.module().clone(),
                                        ErrorStyle::Never,
                                    ),
                                )
                            }
                            ExprOrBinding::Binding(b) => {
                                let binding_ty = self.solve_binding(b, errors).arc_clone_ty();
                                // Use the subscript's location
                                call_setitem(CallArg::ty(&binding_ty, subscript.range));
                                binding_ty
                            }
                        }
                    }
                }
            })
        })
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
                    ErrorInfo::Kind(ErrorKind::BadReturn),
                    "Async generator function should return `AsyncGenerator`".to_owned(),
                );
            }
        } else if is_generator {
            if let Some((_, _, return_ty)) = self.decompose_generator(annotation) {
                self.check_type(implicit_return.ty(), &return_ty, range, errors, &|| {
                    TypeCheckContext::of_kind(TypeCheckKind::ImplicitFunctionReturn(
                        has_explicit_returns,
                    ))
                });
            } else {
                self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::BadReturn),
                    "Generator function should return `Generator`".to_owned(),
                );
            }
        } else {
            self.check_type(implicit_return.ty(), annotation, range, errors, &|| {
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
                Type::Type(_) | Type::TypeVar(_) | Type::ParamSpec(_) | Type::TypeVarTuple(_) => {
                    true
                }
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

    // Given a type, force all `Vars` that indicate placeholder types
    // (everything that isn't either an answer or a Recursive var).
    fn pin_all_placeholder_types(&self, ty: &mut Type) {
        // Expand the type, in case unexpanded `Vars` are hiding further `Var`s that
        // need to be pinned.
        self.solver().expand_vars_mut(ty);
        // Collect all the vars we may need to pin
        fn f(t: &Type, vars: &mut Vec<Var>) {
            match t {
                Type::Var(v) => vars.push(*v),
                _ => t.recurse(&mut |t| f(t, vars)),
            }
        }
        let mut vars = vec![];
        f(ty, &mut vars);
        // Pin all relevant vars
        for var in vars {
            self.solver().pin_placeholder_type(var);
        }
    }

    fn return_type_from_annotation(
        &self,
        annotated_ty: Type,
        is_async: bool,
        is_generator: bool,
    ) -> Type {
        if is_async && !is_generator {
            self.stdlib
                .coroutine(Type::any_implicit(), Type::any_implicit(), annotated_ty)
                .to_type()
        } else {
            annotated_ty
        }
    }

    fn binding_to_type(&self, binding: &Binding, errors: &ErrorCollector) -> Type {
        match binding {
            Binding::Forward(..)
            | Binding::Phi(..)
            | Binding::LoopPhi(..)
            | Binding::Narrow(..)
            | Binding::AssignToAttribute(..)
            | Binding::AssignToSubscript(..)
            | Binding::PossibleLegacyTParam(..) => {
                // These forms require propagating attribute narrowing information, so they
                // are handled in `binding_to_type_info`
                self.binding_to_type_info(binding, errors).into_ty()
            }
            Binding::SelfTypeLiteral(class_key, r) => {
                if let Some(cls) = &self.get_idx(*class_key).as_ref().0 {
                    match self.instantiate(cls) {
                        Type::ClassType(class_type) => Type::type_form(Type::SelfType(class_type)),
                        ty => self.error(
                            errors,
                            *r,
                            ErrorInfo::Kind(ErrorKind::InvalidSelfType),
                            format!(
                                "Cannot apply `typing.Self` to non-class-instance type `{}`",
                                self.for_display(ty)
                            ),
                        ),
                    }
                } else {
                    self.error(
                        errors,
                        *r,
                        ErrorInfo::Kind(ErrorKind::InvalidSelfType),
                        "Could not resolve the class for `typing.Self` (may indicate unexpected recursion resolving types)".to_owned(),
                    )
                }
            }
            Binding::CompletedPartialType(unpinned_idx, first_use) => {
                // Calculate the first use for its side-effects (it might pin `Var`s)
                match first_use {
                    FirstUse::UsedBy(idx) => {
                        self.get_idx(*idx);
                    }
                    FirstUse::Undetermined | FirstUse::DoesNotPin => {}
                }
                self.get_idx(*unpinned_idx).arc_clone().into_ty()
            }
            Binding::PartialTypeWithUpstreamsCompleted(raw_idx, first_used_by) => {
                // Force all of the upstream `Pin`s for which was the first use. This ensures
                // that any `Var` in the result originated directly from `raw_idx`.
                for idx in first_used_by {
                    self.get_idx(*idx);
                }
                self.get_idx(*raw_idx).arc_clone().into_ty()
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
                            ErrorInfo::Kind(ErrorKind::BadAssignment),
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
            Binding::StmtExpr(e, special_export) => {
                let result = self.expr(e, None, errors);
                if *special_export != Some(SpecialExport::AssertType)
                    && let Type::ClassType(cls) = &result
                    && self.is_coroutine(&result)
                    && !self.extends_any(cls.class_object())
                {
                    self.error(
                        errors,
                        e.range(),
                        ErrorInfo::Kind(ErrorKind::UnusedCoroutine),
                        "Result of async function call is unused. Did you forget to `await`?"
                            .to_owned(),
                    );
                }
                result
            }
            Binding::MultiTargetAssign(ann, idx, range) => {
                let type_info = self.get_idx(*idx);
                let ty = type_info.ty();
                if let Some(ann_idx) = ann {
                    let annot = self.get_idx(*ann_idx);
                    if annot.annotation.is_final() {
                        self.error(
                            errors,
                            *range,
                            ErrorInfo::Kind(ErrorKind::BadAssignment),
                            "Assignment target is marked final".to_owned(),
                        );
                    }
                    if let Some(annot_ty) = annot.ty(self.stdlib)
                        && !self.is_subset_eq(ty, &annot_ty)
                    {
                        self.error(
                            errors,
                            *range,
                            ErrorInfo::Kind(ErrorKind::BadAssignment),
                            format!(
                                "Wrong type for assignment, expected `{}` and got `{}`",
                                &annot_ty, ty
                            ),
                        );
                        return annot_ty;
                    }
                }
                ty.clone()
            }
            Binding::PatternMatchMapping(mapping_key, binding_key) => {
                // TODO: check that value is a mapping
                // TODO: check against duplicate keys (optional)
                let key_ty = self.expr_infer(mapping_key, errors);
                let binding = self.get_idx(*binding_key);
                let arg = CallArg::ty(&key_ty, mapping_key.range());
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
                            if let Some(Type::Literal(Lit::Str(attr_name))) = ts.get(*idx) {
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
                                    ErrorInfo::Context(&context),
                                    format!(
                                        "Expected literal string in `__match_args__`, got `{}`",
                                        ts[*idx]
                                    ),
                                )
                            }
                        } else {
                            self.error(
                                errors,
                                *range,
                                ErrorInfo::Context(&context),
                                format!("Index {idx} out of range for `__match_args__`"),
                            )
                        }
                    }
                    Type::Any(AnyStyle::Error) => match_args,
                    _ => {
                        self.error(
                            errors,
                            *range,
                            ErrorInfo::Context(&context),
                            format!(
                                "Expected concrete tuple for `__match_args__`, got `{match_args}`",
                            ),
                        )
                    }
                }
            }
            Binding::PatternMatchClassKeyword(_, attr, key) => {
                // TODO: check that value matches class
                // TODO: check against duplicate keys (optional)
                let binding = self.get_idx(*key);
                self.attr_infer(&binding, &attr.id, attr.range, errors, None)
                    .into_ty()
            }
            Binding::NameAssign(name, annot_key, expr, legacy_tparams) => {
                let (has_type_alias_qualifier, ty) = match annot_key.as_ref() {
                    // First infer the type as a normal value
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
                                ErrorInfo::Kind(ErrorKind::BadAssignment),
                                format!("`{name}` is marked final"),
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
                // Then, handle the possibility that we need to treat the type as a type alias
                match has_type_alias_qualifier {
                    Some(true) => self.as_type_alias(
                        name,
                        TypeAliasStyle::LegacyExplicit,
                        ty,
                        expr,
                        None,
                        legacy_tparams,
                        errors,
                    ),
                    None if Self::may_be_implicit_type_alias(&ty)
                        && self.has_valid_annotation_syntax(expr, &self.error_swallower()) =>
                    {
                        self.as_type_alias(
                            name,
                            TypeAliasStyle::LegacyImplicit,
                            ty,
                            expr,
                            None,
                            legacy_tparams,
                            errors,
                        )
                    }
                    _ => ty,
                }
            }
            Binding::TypeVar(ann, name, x) => {
                let ty = self.typevar_from_call(name.clone(), x, errors).to_type();
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
                    self.check_and_return_type(ty, want, x.range, errors, &|| {
                        TypeCheckContext::of_kind(TypeCheckKind::from_annotation_target(target))
                    })
                } else {
                    ty
                }
            }
            Binding::ParamSpec(ann, name, x) => {
                let ty = self.paramspec_from_call(name.clone(), x, errors).to_type();
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
                    self.check_and_return_type(ty, want, x.range, errors, &|| {
                        TypeCheckContext::of_kind(TypeCheckKind::from_annotation_target(target))
                    })
                } else {
                    ty
                }
            }
            Binding::TypeVarTuple(ann, name, x) => {
                let ty = self
                    .typevartuple_from_call(name.clone(), x, errors)
                    .to_type();
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
                    self.check_and_return_type(ty, want, x.range, errors, &|| {
                        TypeCheckContext::of_kind(TypeCheckKind::from_annotation_target(target))
                    })
                } else {
                    ty
                }
            }
            Binding::ReturnType(x) => {
                match &x.kind {
                    ReturnTypeKind::ShouldValidateAnnotation {
                        range,
                        annotation,
                        stub_or_impl,
                        decorators,
                        implicit_return,
                        is_generator,
                        has_explicit_return,
                    } => {
                        // TODO: A return type annotation like `Final` is invalid in this context.
                        // It will result in an implicit Any type, which is reasonable, but we should
                        // at least error here.
                        let ty = self.get_idx(*annotation).annotation.get_type().clone();
                        // If the function body is stubbed out or if the function is decorated with
                        // `@abstractmethod`, we blindly accept the return type annotation.
                        if *stub_or_impl != FunctionStubOrImpl::Stub
                            && !decorators.iter().any(|k| {
                                let decorator = self.get_idx(*k);
                                match decorator.ty().callee_kind() {
                                    Some(CalleeKind::Function(FunctionKind::AbstractMethod)) => {
                                        true
                                    }
                                    _ => false,
                                }
                            })
                        {
                            let implicit_return = self.get_idx(*implicit_return);
                            self.check_implicit_return_against_annotation(
                                implicit_return,
                                &ty,
                                x.is_async,
                                *is_generator,
                                *has_explicit_return,
                                *range,
                                errors,
                            );
                        }
                        self.return_type_from_annotation(ty, x.is_async, *is_generator)
                    }
                    ReturnTypeKind::ShouldTrustAnnotation {
                        annotation,
                        is_generator,
                    } => {
                        // TODO: A return type annotation like `Final` is invalid in this context.
                        // It will result in an implicit Any type, which is reasonable, but we should
                        // at least error here.
                        let ty = self.get_idx(*annotation).annotation.get_type().clone();
                        self.return_type_from_annotation(ty, x.is_async, *is_generator)
                    }
                    ReturnTypeKind::ShouldReturnAny { is_generator } => self
                        .return_type_from_annotation(
                            Type::any_implicit(),
                            x.is_async,
                            *is_generator,
                        ),
                    ReturnTypeKind::ShouldInferType {
                        returns,
                        implicit_return,
                        yields,
                        yield_froms,
                    } => {
                        let is_generator = !(yields.is_empty() && yield_froms.is_empty());
                        let returns = returns.iter().map(|k| self.get_idx(*k).arc_clone_ty());
                        let implicit_return = self.get_idx(*implicit_return);
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
                            let yield_ty = self.unions({
                                let yield_tys =
                                    yields.iter().map(|idx| self.get_idx(*idx).yield_ty.clone());
                                let yield_from_tys = yield_froms
                                    .iter()
                                    .map(|idx| self.get_idx(*idx).yield_ty.clone());
                                yield_tys.chain(yield_from_tys).collect()
                            });
                            if x.is_async {
                                self.stdlib
                                    .async_generator(yield_ty, Type::any_implicit())
                                    .to_type()
                            } else {
                                self.stdlib
                                    .generator(yield_ty, Type::any_implicit(), return_ty)
                                    .to_type()
                            }
                        } else if x.is_async {
                            self.stdlib
                                .coroutine(Type::any_implicit(), Type::any_implicit(), return_ty)
                                .to_type()
                        } else {
                            return_ty
                        }
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
                            ErrorInfo::Kind(ErrorKind::BadReturn),
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

                if self.module().path().is_interface() {
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
            Binding::ExceptionHandler(ann, is_star) => {
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
                            ErrorInfo::Kind(ErrorKind::Unsupported),
                            "`expect*` is unsupported until Python 3.11".to_owned(),
                        );
                    }
                    res
                } else {
                    None
                };
                let check_exception_type = |exception_type: Type, range| {
                    let exception = self.untype(exception_type, range, errors);
                    self.check_type(&exception, &base_exception_type, range, errors, &|| {
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
                            ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                            "Exception handler annotation in `except*` clause may not extend `BaseExceptionGroup`".to_owned());
                    }
                    exception
                };
                let exceptions = match &**ann {
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
                            Type::Tuple(Tuple::Unbounded(t)) => {
                                vec![check_exception_type(*t, ann.range())]
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
                    let infer_hint = ty.clone().and_then(|x| {
                        x.ty(self.stdlib)
                            .map(|ty| self.stdlib.async_iterable(ty.clone()).to_type())
                    });
                    let iterable = self.expr_infer_with_hint(
                        e,
                        infer_hint.as_ref().map(HintRef::soft),
                        errors,
                    );
                    self.async_iterate(&iterable, e.range(), errors)
                } else {
                    let infer_hint = ty.clone().and_then(|x| {
                        x.ty(self.stdlib)
                            .map(|ty| self.stdlib.iterable(ty.clone()).to_type())
                    });
                    let iterable = self.expr_infer_with_hint(
                        e,
                        infer_hint.as_ref().map(HintRef::soft),
                        errors,
                    );
                    self.iterate(&iterable, e.range(), errors, None)
                };
                let value = self.get_produced_type(iterables);
                let check_hint = ty.clone().and_then(|x| x.ty(self.stdlib));
                if let Some(check_hint) = check_hint {
                    self.check_and_return_type(value, &check_hint, e.range(), errors, tcc)
                } else {
                    value
                }
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
                        self.check_and_return_type(context_value, &ty, *range, errors, &|| {
                            TypeCheckContext::of_kind(TypeCheckKind::from_annotation_target(target))
                        })
                    }
                    None => context_value,
                }
            }
            Binding::UnpackedValue(ann, to_unpack, range, pos) => {
                let iterables = self.iterate(self.get_idx(*to_unpack).ty(), *range, errors, None);
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
                let got = self.unions(values);
                if let Some(want) = ann
                    .map(|idx| self.get_idx(idx))
                    .and_then(|ann| ann.ty(self.stdlib))
                {
                    self.check_type(&got, &want, *range, errors, &|| {
                        TypeCheckContext::of_kind(TypeCheckKind::UnpackedAssign)
                    });
                }
                got
            }
            &Binding::Function(idx, mut pred, class_meta) => {
                let def = self.get_decorated_function(idx);
                self.solve_function_binding(def, &mut pred, class_meta.as_ref(), errors)
            }
            Binding::Import(m, name, _aliased) => self
                .get_from_export(*m, None, &KeyExport(name.clone()))
                .arc_clone(),
            Binding::ClassDef(x, decorators) => match &self.get_idx(*x).0 {
                None => Type::any_implicit(),
                Some(cls) => {
                    let mut ty = Type::ClassDef(cls.dupe());
                    for x in decorators.iter().rev() {
                        let decorator = self.get_idx(*x).arc_clone_ty();
                        let range = self.bindings().idx_to_key(*x).range();
                        ty = self.apply_decorator(decorator, ty, range, errors)
                    }
                    ty
                }
            },
            Binding::AnnotatedType(ann, val) => match &self.get_idx(*ann).ty(self.stdlib) {
                Some(ty) => (*ty).clone(),
                None => self.binding_to_type(val, errors),
            },
            Binding::Type(x) => x.clone(),
            Binding::Global(global) => global.as_type(self.stdlib),
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
                            ErrorInfo::Kind(ErrorKind::InvalidTypeVar),
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
                        TypeFormContext::quantified_kind_default(*kind),
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
                Quantified::new(*unique, name.clone(), *kind, default_ty, restriction).to_value()
            }
            Binding::Module(m, path, prev) => {
                let prev = prev
                    .as_ref()
                    .and_then(|x| self.get_idx(*x).ty().as_module().cloned());
                match prev {
                    Some(prev) if prev.parts() == path => prev.add_module(*m).to_type(),
                    _ => {
                        if path.len() == 1 {
                            Type::Module(ModuleType::new(
                                path[0].clone(),
                                OrderedSet::from_iter([(*m)]),
                            ))
                        } else {
                            assert_eq!(&m.components(), path);
                            Type::Module(ModuleType::new_as(*m))
                        }
                    }
                }
            }
            Binding::ScopedTypeAlias(name, params, expr) => {
                let ty = self.expr_infer(expr, errors);
                let ta =
                    self.as_type_alias(name, TypeAliasStyle::Scoped, ty, expr, None, &None, errors);
                match ta {
                    Type::Forall(..) => self.error(
                        errors,
                        expr.range(),
                        ErrorInfo::Kind(ErrorKind::InvalidTypeVar),
                        format!("Type parameters used in `{name}` but not declared"),
                    ),
                    Type::TypeAlias(ta) => {
                        let params_range = params.as_ref().map_or(expr.range(), |x| x.range);
                        Forallable::TypeAlias(*ta).forall(self.validated_tparams(
                            params_range,
                            self.scoped_type_params(params.as_ref()),
                            TParamsSource::TypeAlias,
                            errors,
                        ))
                    }
                    _ => ta,
                }
            }
            Binding::TypeAliasType(ann, name, x) => {
                let Some((expr, type_param_exprs)) =
                    self.typealiastype_from_call(name.clone(), x, errors)
                else {
                    return Type::any_error();
                };
                let ty = self.expr_infer(&expr, errors);
                let ta = self.as_type_alias(
                    &name.id,
                    TypeAliasStyle::Scoped,
                    ty,
                    &expr,
                    Some(type_param_exprs),
                    &None,
                    errors,
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
                    self.check_and_return_type(ta.clone(), want, x.range, errors, &|| {
                        TypeCheckContext::of_kind(TypeCheckKind::from_annotation_target(target))
                    })
                } else {
                    ta
                }
            }
            Binding::Decorator(expr) => self.expr_infer(expr, errors),
            Binding::LambdaParameter(var) => var.to_type(),
            Binding::FunctionParameter(param) => {
                let finalize = |target: &AnnotationTarget, ty| match target {
                    AnnotationTarget::ArgsParam(_) => Type::Tuple(Tuple::unbounded(ty)),
                    AnnotationTarget::KwargsParam(_) => self
                        .stdlib
                        .dict(self.stdlib.str().clone().to_type(), ty)
                        .to_type(),
                    _ => ty,
                };
                match param {
                    FunctionParameter::Annotated(key) => {
                        let annotation = self.get_idx(*key);
                        annotation.ty(self.stdlib).clone().unwrap_or_else(|| {
                            // This annotation isn't valid. It's something like `: Final` that doesn't
                            // have enough information to create a real type.
                            finalize(&annotation.target, Type::any_implicit())
                        })
                    }
                    FunctionParameter::Unannotated(var, function_idx, target) => {
                        // It's important that we force the undecorated function binding before reading
                        // from this var. Solving the undecorated function binding pins the type of the var,
                        // either to a concrete type or to any. Without this we can have non-determinism
                        // where the reader can observe an unresolved var or a resolved type, depending on
                        // the order of solved bindings.
                        self.get_idx(*function_idx);
                        let ty = self.solver().force_var(*var);
                        finalize(target, ty)
                    }
                }
            }
            Binding::SuperInstance(style, range) => self.solve_super_binding(style, *range, errors),
            // For first-usage-based type inference, we occasionally just want a way to force
            // some other `K::Value` type in order to deterministically pin `Var`s introduced by a definition.
            Binding::UsageLink(linked_key) => {
                match linked_key {
                    LinkedKey::Yield(idx) => {
                        self.get_idx(*idx);
                    }
                    LinkedKey::YieldFrom(idx) => {
                        self.get_idx(*idx);
                    }
                    LinkedKey::Expect(idx) => {
                        self.get_idx(*idx);
                    }
                }
                // Produce a placeholder type; it will not be used.
                Type::None
            }
            Binding::Delete(x) => self.check_del_statement(x, errors),
        }
    }

    pub fn solve_decorated_function(
        &self,
        x: &BindingDecoratedFunction,
        errors: &ErrorCollector,
    ) -> Arc<Type> {
        let b = self.bindings().get(x.undecorated_idx);
        let def = self.get_idx(x.undecorated_idx);
        self.decorated_function_type(&def, &b.def, errors)
    }

    pub fn solve_undecorated_function(
        &self,
        x: &BindingUndecoratedFunction,
        errors: &ErrorCollector,
    ) -> Arc<UndecoratedFunction> {
        self.undecorated_function(
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
                            Type::None,
                            &yield_hint,
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
                    ErrorInfo::Kind(ErrorKind::InvalidYield),
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
            BindingYieldFrom::YieldFrom(annot, is_async, x) => {
                if is_async.is_async() {
                    self.error(
                        errors,
                        x.range,
                        ErrorInfo::Kind(ErrorKind::InvalidYield),
                        "Invalid `yield from` in async function".to_owned(),
                    );
                }
                let annot = annot.map(|k| self.get_idx(k));
                let want = annot
                    .as_ref()
                    .and_then(|x| x.ty(self.stdlib))
                    .and_then(|ty| self.decompose_generator(&ty));

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
                    ty = if is_async.is_async() {
                        // We already errored above.
                        Type::any_error()
                    } else {
                        self.error(
                            errors,
                            x.range,
                            ErrorInfo::Kind(ErrorKind::InvalidYield),
                            format!(
                                "yield from value must be iterable, got `{}`",
                                self.for_display(ty)
                            ),
                        )
                    };
                    YieldFromResult::any_error()
                };
                if let Some((want_yield, want_send, _)) = want {
                    // We don't need to be compatible with the expected generator return type.
                    let want = self
                        .stdlib
                        .generator(want_yield, want_send, Type::any_implicit())
                        .to_type();
                    self.check_type(&ty, &want, x.range, errors, &|| {
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
                    ErrorInfo::Kind(ErrorKind::InvalidYield),
                    "Invalid `yield from` outside of a function".to_owned(),
                );
                Arc::new(YieldFromResult::any_error())
            }
        }
    }

    /// Unwraps a type, originally evaluated as a value, so that it can be used as a type annotation.
    /// For example, in `def f(x: int): ...`, we evaluate `int` as a value, getting its type as
    /// `type[int]`, then call `untype(type[int])` to get the `int` annotation.
    pub fn untype(&self, ty: Type, range: TextRange, errors: &ErrorCollector) -> Type {
        if let Some(t) = self.untype_opt(ty.clone(), range, errors) {
            t
        } else {
            self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::NotAType),
                format!(
                    "Expected a type form, got instance of `{}`",
                    self.for_display(ty),
                ),
            )
        }
    }

    pub fn untype_opt(
        &self,
        mut ty: Type,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        if let Type::Forall(forall) = ty {
            ty = self.promote_forall(*forall, range);
        };
        match self.canonicalize_all_class_types(ty, range, errors) {
            Type::Union(xs) if !xs.is_empty() => {
                let mut ts = Vec::new();
                for x in xs {
                    let t = self.untype_opt(x, range, errors)?;
                    ts.push(t);
                }
                Some(self.unions(ts))
            }
            Type::Var(v) if let Some(_guard) = self.recurse(v) => {
                self.untype_opt(self.solver().force_var(v), range, errors)
            }
            ty @ (Type::TypeVar(_)
            | Type::ParamSpec(_)
            | Type::TypeVarTuple(_)
            | Type::Args(_)
            | Type::Kwargs(_)) => Some(ty),
            Type::Type(t) => Some(*t),
            Type::None => Some(Type::None), // Both a value and a type
            Type::Ellipsis => Some(Type::Ellipsis), // A bit weird because of tuples, so just promote it
            Type::Any(style) => Some(style.propagate()),
            Type::TypeAlias(ta) => self.untype_opt(ta.as_type(), range, errors),
            t @ Type::Unpack(
                box Type::Tuple(_) | box Type::TypeVarTuple(_) | box Type::Quantified(_),
            ) => Some(t),
            Type::Unpack(box Type::Var(v)) if let Some(_guard) = self.recurse(v) => self
                .untype_opt(
                    Type::Unpack(Box::new(self.solver().force_var(v))),
                    range,
                    errors,
                ),
            Type::QuantifiedValue(q) => Some(q.to_type()),
            Type::ArgsValue(q) => Some(Type::Args(q)),
            Type::KwargsValue(q) => Some(Type::Kwargs(q)),
            _ => None,
        }
    }

    // Approximate the result of calling `type()` on something of type T
    // In many cases the result is just type[T] with generics erased, but sometimes
    // we'll fall back to builtins.type. We can add more cases here as-needed.
    pub fn type_of(&self, ty: Type) -> Type {
        match ty {
            Type::ClassType(cls) | Type::SelfType(cls) => {
                Type::ClassDef(cls.class_object().clone())
            }
            Type::Literal(lit) => {
                Type::ClassDef(lit.general_class_type(self.stdlib).class_object().clone())
            }
            Type::LiteralString => Type::ClassDef(self.stdlib.str().class_object().clone()),
            Type::None => Type::ClassDef(self.stdlib.none_type().class_object().clone()),
            Type::Tuple(_) => Type::ClassDef(self.stdlib.tuple_object().clone()),
            Type::TypedDict(_) | Type::PartialTypedDict(_) => {
                Type::ClassDef(self.stdlib.dict_object().clone())
            }
            Type::Union(xs) if !xs.is_empty() => {
                let mut ts = Vec::new();
                for x in xs {
                    let t = self.type_of(x);
                    ts.push(t);
                }
                self.unions(ts)
            }
            Type::TypeAlias(ta) => self.type_of(ta.as_type()),
            Type::Any(style) => Type::type_form(style.propagate()),
            Type::ClassDef(cls) => Type::type_form(Type::ClassType(
                self.get_metadata_for_class(&cls)
                    .metaclass(self.stdlib)
                    .clone(),
            )),
            _ => self.stdlib.builtins_type().clone().to_type(),
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
                ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                "`Unpack` with a `TypedDict` is only allowed in a **kwargs annotation".to_owned(),
            );
        }
        if type_form_context == TypeFormContext::ParameterKwargsAnnotation
            && matches!(ty, Type::Unpack(ref inner) if !matches!(**inner, Type::TypedDict(_)))
        {
            return self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                "`Unpack` in **kwargs annotation must be used only with a `TypedDict`".to_owned(),
            );
        }
        if type_form_context != TypeFormContext::ParameterKwargsAnnotation
            && matches!(ty, Type::Kwargs(_))
        {
            return self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                "`ParamSpec` **kwargs is only allowed in a **kwargs annotation".to_owned(),
            );
        }
        if type_form_context != TypeFormContext::ParameterArgsAnnotation
            && matches!(ty, Type::Args(_))
        {
            return self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                "`ParamSpec` *args is only allowed in an *args annotation".to_owned(),
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
                ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                "`Unpack` is not allowed in this context".to_owned(),
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
                ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                format!("`{ty}` is not allowed in this context"),
            );
        }
        if !matches!(
            type_form_context,
            TypeFormContext::TupleOrCallableParam | TypeFormContext::TypeArgument
        ) && matches!(ty, Type::TypeVarTuple(_))
        {
            // Determine whether we're simply missing an `Unpack[...]` or the TypeVarTuple isn't allowed at all in this context.
            let tmp_collector = self.error_collector();
            self.validate_type_form(
                Type::Unpack(Box::new(ty)),
                range,
                type_form_context,
                &tmp_collector,
            );
            if tmp_collector.is_empty() {
                return self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                    "`TypeVarTuple` must be unpacked".to_owned(),
                );
            } else {
                return self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                    "`TypeVarTuple` is not allowed in this context".to_owned(),
                );
            }
        }
        if let Type::SpecialForm(special_form) = ty
            && !type_form_context.is_valid_unparameterized_annotation(special_form)
        {
            if special_form.can_be_subscripted() {
                self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                    format!("Expected a type argument for `{special_form}`"),
                );
            } else {
                self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                    format!("`{special_form}` is not allowed in this context"),
                );
            }
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
                    ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                    "`ParamSpec` is not allowed in this context".to_owned(),
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
                    ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                    "`TypeVarTuple` must be unpacked".to_owned(),
                );
            }
        }
        if type_form_context == TypeFormContext::TypeVarConstraint && ty.any(Type::is_type_variable)
        {
            return self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                "Type variable bounds and constraints must be concrete".to_owned(),
            );
        }
        ty
    }

    /// Type check a delete expression, including ensuring that the target of the
    /// delete is legal.
    fn check_del_statement(&self, delete_target: &Expr, errors: &ErrorCollector) -> Type {
        match delete_target {
            Expr::Name(_) => {
                self.expr_infer(delete_target, errors);
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
                        let field_name = Name::new(field_name);
                        self.check_del_typed_dict_literal_key(
                            typed_dict,
                            &field_name,
                            x.slice.range(),
                            errors,
                        );
                    }
                    (Type::TypedDict(typed_dict), Type::ClassType(cls))
                        if cls.is_builtin("str")
                            && self
                                .get_typed_dict_value_type_as_builtins_dict(typed_dict)
                                .is_some() =>
                    {
                        self.check_del_typed_dict_field(
                            typed_dict.name(),
                            None,
                            false,
                            false,
                            x.slice.range(),
                            errors,
                        )
                    }
                    (_, _) => {
                        self.call_method_or_error(
                            &base,
                            &dunder::DELITEM,
                            x.range,
                            &[CallArg::ty(&slice_ty, x.slice.range())],
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
                    delete_target.range(),
                    ErrorInfo::Kind(ErrorKind::UnsupportedDelete),
                    "Invalid target for `del`".to_owned(),
                );
            }
        }
        // This is a fallback in case a variable is defined *only* by a `del` - we'll use `Any` as
        // the type for reads (i.e. `BoundName` / `Forward` key/binding pairs) in that case.
        Type::any_implicit()
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
                            None,
                            self.expr_untype(x, type_form_context, errors),
                            Required::Required,
                        )
                    })
                    .collect();
                Type::ParamSpecValue(ParamList::new(elts))
            }
            _ => {
                let inferred_ty = self.expr_infer(x, errors);
                // Check if this is a scoped type alias in base class context
                // We do this check here instead of `validate_type_form` because it
                // substitutes type aliases with the aliased type
                if type_form_context == TypeFormContext::BaseClassList
                    && let Type::TypeAlias(ta) = &inferred_ty
                    && ta.style == TypeAliasStyle::Scoped
                {
                    return self.error(
                                errors,
                                x.range(),
                                ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                                format!(
                                    "Cannot use scoped type alias `{}` as a base class. Use a legacy type alias instead: `{}: TypeAlias = {}`",
                                    ta.name,
                                    ta.name,
                                    self.for_display(ta.as_type())
                                ),
                            );
                }
                self.untype(inferred_ty, x.range(), errors)
            }
        };
        self.validate_type_form(result, x.range(), type_form_context, errors)
    }
}
