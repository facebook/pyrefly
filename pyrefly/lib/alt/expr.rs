/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cell::LazyCell;
use std::fmt;
use std::fmt::Display;

use dupe::Dupe;
use itertools::Either;
use itertools::Itertools;
use pyrefly_python::ast::Ast;
use pyrefly_python::dunder;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::callable::FunctionKind;
use pyrefly_types::dimension::SizeExpr;
use pyrefly_types::dimension::canonicalize;
use pyrefly_types::literal::LitStyle;
use pyrefly_types::tensor::IndexOp;
use pyrefly_types::tensor::TensorShape;
use pyrefly_types::tensor::TensorType;
use pyrefly_types::tensor::index_shape_int;
use pyrefly_types::tensor::index_shape_multi;
use pyrefly_types::tensor::index_shape_slice;
use pyrefly_types::tensor::index_shape_tensor;
use pyrefly_types::typed_dict::AnonymousTypedDictInner;
use pyrefly_types::typed_dict::ExtraItems;
use pyrefly_types::typed_dict::TypedDict;
use pyrefly_types::typed_dict::TypedDictField;
use pyrefly_types::types::Union;
use pyrefly_util::owner::Owner;
use pyrefly_util::prelude::SliceExt;
use pyrefly_util::prelude::VecExt;
use pyrefly_util::suggest::best_suggestion;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Arguments;
use ruff_python_ast::BoolOp;
use ruff_python_ast::Comprehension;
use ruff_python_ast::DictItem;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprBinOp;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprGenerator;
use ruff_python_ast::ExprNumberLiteral;
use ruff_python_ast::ExprSlice;
use ruff_python_ast::ExprStarred;
use ruff_python_ast::ExprStringLiteral;
use ruff_python_ast::ExprTuple;
use ruff_python_ast::Identifier;
use ruff_python_ast::Keyword;
use ruff_python_ast::Number;
use ruff_python_ast::Operator;
use ruff_python_ast::StringLiteralValue;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::Hashed;
use vec1::Vec1;
use vec1::vec1;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::callable::CallArg;
use crate::alt::nn_module_specials::is_nn_module_dict;
use crate::alt::solve::TypeFormContext;
use crate::alt::unwrap::Hint;
use crate::alt::unwrap::HintRef;
use crate::binding::binding::Key;
use crate::binding::binding::KeyYield;
use crate::binding::binding::KeyYieldFrom;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorContext;
use crate::error::context::ErrorInfo;
use crate::error::context::TypeCheckContext;
use crate::error::context::TypeCheckKind;
use crate::types::callable::Param;
use crate::types::callable::ParamList;
use crate::types::callable::Params;
use crate::types::callable::Required;
use crate::types::class::Class;
use crate::types::facet::FacetKind;
use crate::types::lit_int::LitInt;
use crate::types::literal::Lit;
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
use crate::types::types::Type;

#[derive(Debug, Clone, Copy)]
pub enum TypeOrExpr<'a> {
    /// Bundles a `Type` with a `TextRange`, allowing us to give good errors.
    Type(&'a Type, TextRange),
    Expr(&'a Expr),
}

impl Ranged for TypeOrExpr<'_> {
    fn range(&self) -> TextRange {
        match self {
            TypeOrExpr::Type(_, range) => *range,
            TypeOrExpr::Expr(expr) => expr.range(),
        }
    }
}

static ANONYMOUS_TYPED_DICT_MAX_ITEMS: usize = 20;

impl<'a> TypeOrExpr<'a> {
    pub fn infer<Ans: LookupAnswer>(
        self,
        solver: &AnswersSolver<Ans>,
        errors: &ErrorCollector,
    ) -> Type {
        match self {
            TypeOrExpr::Type(ty, _) => ty.clone(),
            TypeOrExpr::Expr(x) => solver.expr_infer(x, errors),
        }
    }

    pub fn transform<Ans: LookupAnswer>(
        &self,
        solver: &AnswersSolver<Ans>,
        errors: &ErrorCollector,
        owner: &'a Owner<Type>,
        transformation: impl Fn(&Type) -> Type,
    ) -> (Self, bool) {
        let ty = self.infer(solver, errors);
        let transformed = transformation(&ty);
        let changed = ty != transformed;
        (
            TypeOrExpr::Type(owner.push(transformed), self.range()),
            changed,
        )
    }
}

#[derive(Debug, Clone)]
enum ConditionRedundantReason {
    /// The boolean indicates whether it's equivalent to True
    IntLiteral(bool),
    StrLiteral(bool),
    BytesLiteral(bool),
    /// Class name + member name
    EnumLiteral(Name, Name),
    Function(ModuleName, FunctionKind),
    Class(Name),
}

impl ConditionRedundantReason {
    fn equivalent_boolean(&self) -> Option<bool> {
        match self {
            ConditionRedundantReason::Function(..) | ConditionRedundantReason::Class(..) => {
                Some(true)
            }
            ConditionRedundantReason::IntLiteral(b)
            | ConditionRedundantReason::StrLiteral(b)
            | ConditionRedundantReason::BytesLiteral(b) => Some(*b),
            ConditionRedundantReason::EnumLiteral(..) => None,
        }
    }

    fn description(&self) -> String {
        match self {
            ConditionRedundantReason::IntLiteral(..) => {
                "Integer literal used as condition".to_owned()
            }
            ConditionRedundantReason::StrLiteral(..) => {
                "String literal used as condition".to_owned()
            }
            ConditionRedundantReason::BytesLiteral(..) => {
                "Bytes literal used as condition".to_owned()
            }
            ConditionRedundantReason::EnumLiteral(class_name, member_name) => {
                format!("Enum literal `{class_name}.{member_name}` used as condition")
            }
            ConditionRedundantReason::Function(module_name, func_id) => {
                format!(
                    "Function object `{}` used as condition",
                    func_id.format(module_name.dupe())
                )
            }
            ConditionRedundantReason::Class(name) => {
                format!("Class name `{name}` used as condition")
            }
        }
    }
}

impl Display for ConditionRedundantReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}. It's equivalent to {}",
            self.description(),
            match self.equivalent_boolean() {
                Some(true) => "`True`",
                Some(false) => "`False`",
                None => "a boolean literal",
            }
        )
    }
}

static MAX_TUPLE_LENGTH: usize = 256;

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    /// Infer a type for an expression, with an optional type hint that influences the inferred type.
    /// The inferred type is also checked against the hint.
    pub fn expr(
        &self,
        x: &Expr,
        check: Option<(&Type, &dyn Fn() -> TypeCheckContext)>,
        errors: &ErrorCollector,
    ) -> Type {
        self.expr_type_info(x, check, errors).into_ty()
    }

    /// Like expr(), but errors from the infer and check steps are recorded to separate error collectors.
    pub fn expr_with_separate_check_errors(
        &self,
        x: &Expr,
        check: Option<(&Type, &ErrorCollector, &dyn Fn() -> TypeCheckContext)>,
        errors: &ErrorCollector,
    ) -> Type {
        self.expr_type_info_with_separate_check_errors(x, check, errors)
            .into_ty()
    }

    /// Infer a type for an expression.
    pub fn expr_infer(&self, x: &Expr, errors: &ErrorCollector) -> Type {
        self.expr_infer_type_info_with_hint(x, None, errors)
            .into_ty()
    }

    /// Infer a type for an expression, with an optional type hint that influences the inferred type.
    /// Unlike expr(), the inferred type is not checked against the hint.
    pub fn expr_infer_with_hint(
        &self,
        x: &Expr,
        hint: Option<HintRef>,
        errors: &ErrorCollector,
    ) -> Type {
        self.expr_infer_type_info_with_hint(x, hint, errors)
            .into_ty()
    }

    /// Like expr_infer_with_hint(), but returns a TypeInfo that includes narrowing information.
    pub fn expr_infer_type_info_with_hint(
        &self,
        x: &Expr,
        hint: Option<HintRef>,
        errors: &ErrorCollector,
    ) -> TypeInfo {
        if let Some(self_type_annotation) = self.intercept_typing_self_use(x) {
            return self_type_annotation;
        }
        let res = match x {
            Expr::Name(x) => {
                if Ast::is_synthesized_empty_name(x) {
                    TypeInfo::of_ty(self.heap.mk_any_error())
                } else {
                    self.get(&Key::BoundName(ShortIdentifier::expr_name(x)))
                        .arc_clone()
                }
            }
            Expr::Attribute(x) => {
                let base = self.expr_infer_type_info_with_hint(&x.value, None, errors);
                self.record_external_attribute_definition_index(
                    base.ty(),
                    x.attr.id(),
                    x.attr.range,
                );
                let attr_type = self.attr_infer(&base, &x.attr.id, x.range, errors, None);
                if base.ty().is_literal_string() {
                    match attr_type.ty() {
                        Type::BoundMethod(method) => attr_type
                            .clone()
                            .with_ty(method.with_bound_object(base.ty().clone()).as_type()),
                        _ => attr_type,
                    }
                } else {
                    attr_type
                }
            }
            Expr::Subscript(x) => {
                // TODO: We don't deal properly with hint here, we should.
                let base = self.expr_infer_type_info_with_hint(&x.value, None, errors);
                self.subscript_infer(&base, &x.slice, x.range(), errors)
            }
            Expr::Named(x) => match &*x.target {
                Expr::Name(name) if !Ast::is_synthesized_empty_name(name) => self
                    .get(&Key::Definition(ShortIdentifier::expr_name(name)))
                    .arc_clone(),
                _ => self.expr_infer_type_info_with_hint(&x.value, hint, errors),
            },
            // All other expressions operate at the `Type` level only, so we avoid the overhead of
            // wrapping and unwrapping `TypeInfo` by computing the result as a `Type` and only wrapping
            // at the end.
            _ => TypeInfo::of_ty(self.expr_infer_type_no_trace(x, hint, errors)),
        };
        // Check for deprecation
        self.check_for_deprecated_call(res.ty(), x.range(), errors);
        self.record_type_trace(x.range(), res.ty());
        res
    }

    fn expr_type_info(
        &self,
        x: &Expr,
        check: Option<(&Type, &dyn Fn() -> TypeCheckContext)>,
        errors: &ErrorCollector,
    ) -> TypeInfo {
        self.expr_type_info_with_separate_check_errors(
            x,
            check.map(|(ty, tcc)| (ty, errors, tcc)),
            errors,
        )
    }

    fn expr_type_info_with_separate_check_errors(
        &self,
        x: &Expr,
        check: Option<(&Type, &ErrorCollector, &dyn Fn() -> TypeCheckContext)>,
        errors: &ErrorCollector,
    ) -> TypeInfo {
        match check {
            Some((hint, hint_errors, tcc)) if !hint.is_any() => {
                let got = self.expr_infer_type_info_with_hint(
                    x,
                    Some(HintRef::new(hint, Some(hint_errors))),
                    errors,
                );
                self.check_and_return_type_info(got, hint, x.range(), hint_errors, tcc)
            }
            _ => self.expr_infer_type_info_with_hint(x, None, errors),
        }
    }

    /// This function should not be used directly: we want every expression to record a type trace,
    /// and that is handled in expr_infer_type_info_with_hint. This function should *only* be called
    /// via expr_infer_type_info_with_hint.
    fn expr_infer_type_no_trace(
        &self,
        x: &Expr,
        hint: Option<HintRef>,
        errors: &ErrorCollector,
    ) -> Type {
        match x {
            Expr::Name(..) | Expr::Attribute(..) | Expr::Named(..) | Expr::Subscript(..) => {
                // These cases are required to preserve attribute narrowing information. But anyone calling
                // this function only needs the Type, so we can just pull it out.
                self.expr_infer_type_info_with_hint(x, hint, errors)
                    .into_ty()
            }
            Expr::If(x) => {
                let condition_type = self.expr_infer(&x.test, errors);
                let body_type = self
                    .expr_infer_type_info_with_hint(&x.body, hint, errors)
                    .into_ty();
                let orelse_type = self
                    .expr_infer_type_info_with_hint(&x.orelse, hint, errors)
                    .into_ty();
                self.check_dunder_bool_is_callable(&condition_type, x.range(), errors);
                self.check_redundant_condition(&condition_type, x.range(), errors);
                match self.as_bool(&condition_type, x.test.range(), errors) {
                    Some(true) => body_type,
                    Some(false) => orelse_type,
                    None => self.union(body_type, orelse_type),
                }
            }
            Expr::BoolOp(x) => self.boolop(&x.values, x.op, hint, errors),
            Expr::BinOp(x) => self.binop_infer(x, hint, errors),
            Expr::UnaryOp(x) => self.unop_infer(x, errors),
            Expr::Lambda(lambda) => {
                let param_vars = if let Some(parameters) = &lambda.parameters {
                    parameters
                        .iter_non_variadic_params()
                        .map(|x| (&x.name().id, self.bindings().get_lambda_param(x.name())))
                        .collect()
                } else {
                    Vec::new()
                };
                // Pass any contextual information to the parameter bindings used in the lambda body as a side
                // effect, by setting an answer for the vars created at binding time.
                let return_hint = hint.and_then(|hint| self.decompose_lambda(hint, &param_vars));

                let mut params = param_vars.into_map(|(name, var)| {
                    Param::Pos(
                        name.clone(),
                        self.solver().force_var(var),
                        Required::Required,
                    )
                });
                if let Some(parameters) = &lambda.parameters {
                    params.extend(parameters.vararg.iter().map(|x| {
                        Param::VarArg(
                            Some(x.name.id.clone()),
                            self.solver()
                                .force_var(self.bindings().get_lambda_param(&x.name)),
                        )
                    }));
                    params.extend(parameters.kwarg.iter().map(|x| {
                        Param::Kwargs(
                            Some(x.name.id.clone()),
                            self.solver()
                                .force_var(self.bindings().get_lambda_param(&x.name)),
                        )
                    }));
                }
                let params = Params::List(ParamList::new(params));
                if let Some(hint) = hint {
                    // Ensure no param vars are pinned to unfinished Variable::Quantified.
                    // Since lambda parameters are unannotated, the specialization errors can be ignored.
                    let _specialization_errors = self.solver().finish_all_quantified(hint.ty());
                }
                let ret = self.expr_infer_type_no_trace(
                    &lambda.body,
                    return_hint.as_ref().map(|hint| hint.as_ref()),
                    errors,
                );
                let (yield_keys, yield_from_keys) = self.bindings().lambda_yield_keys(lambda.range);
                let ret = if !(yield_keys.is_empty() && yield_from_keys.is_empty()) {
                    let yield_ty = self.unions(
                        yield_keys
                            .iter()
                            .map(|idx| self.get_idx(*idx).yield_ty.clone())
                            .chain(
                                yield_from_keys
                                    .iter()
                                    .map(|idx| self.get_idx(*idx).yield_ty.clone()),
                            )
                            .collect(),
                    );
                    self.stdlib
                        .generator(yield_ty, self.heap.mk_any_implicit(), ret)
                        .to_type()
                } else {
                    ret
                };
                self.heap.mk_callable(params, ret)
            }
            Expr::Tuple(x) => self.tuple_infer(x, hint, errors),
            Expr::List(x) => {
                let elt_hint = hint.and_then(|ty| self.decompose_list(ty));
                if x.is_empty() {
                    let elem_ty = elt_hint.map_or_else(
                        || {
                            self.solver()
                                .fresh_partial_contained(self.uniques, x.range)
                                .to_type(self.heap)
                        },
                        |hint| hint.to_type(),
                    );
                    self.heap.mk_class_type(self.stdlib.list(elem_ty))
                } else {
                    let elem_tys = self.elts_infer(&x.elts, elt_hint, errors);
                    self.heap
                        .mk_class_type(self.stdlib.list(self.unions(elem_tys)))
                }
            }
            Expr::Dict(x) => self.dict_infer(&x.items, hint, x.range, errors),
            Expr::Set(x) => {
                let elem_hint = hint.and_then(|ty| self.decompose_set(ty));
                if x.is_empty() {
                    let elem_ty = elem_hint.map_or_else(
                        || {
                            self.solver()
                                .fresh_partial_contained(self.uniques, x.range)
                                .to_type(self.heap)
                        },
                        |hint| hint.to_type(),
                    );
                    self.heap.mk_class_type(self.stdlib.set(elem_ty))
                } else {
                    let elem_tys = self.elts_infer(&x.elts, elem_hint, errors);
                    self.heap
                        .mk_class_type(self.stdlib.set(self.unions(elem_tys)))
                }
            }
            Expr::ListComp(x) => {
                let elem_hint = hint.and_then(|ty| self.decompose_list(ty));
                self.ifs_infer(&x.generators, errors);
                let elem_ty = self.expr_infer_with_hint_promote(
                    &x.elt,
                    elem_hint.as_ref().map(|hint| hint.as_ref()),
                    errors,
                );
                self.heap.mk_class_type(self.stdlib.list(elem_ty))
            }
            Expr::SetComp(x) => {
                let elem_hint = hint.and_then(|ty| self.decompose_set(ty));
                self.ifs_infer(&x.generators, errors);
                let elem_ty = self.expr_infer_with_hint_promote(
                    &x.elt,
                    elem_hint.as_ref().map(|hint| hint.as_ref()),
                    errors,
                );
                self.heap.mk_class_type(self.stdlib.set(elem_ty))
            }
            Expr::DictComp(x) => {
                let (key_hint, value_hint) =
                    hint.map_or((None, None), |ty| self.decompose_dict(ty));
                self.ifs_infer(&x.generators, errors);
                let key_ty = self.expr_infer_with_hint_promote(
                    &x.key,
                    key_hint.as_ref().map(|hint| hint.as_ref()),
                    errors,
                );
                let value_ty = self.expr_infer_with_hint_promote(
                    &x.value,
                    value_hint.as_ref().map(|hint| hint.as_ref()),
                    errors,
                );
                self.heap.mk_class_type(self.stdlib.dict(key_ty, value_ty))
            }
            Expr::Generator(x) => {
                let yield_hint = hint.and_then(|hint| self.decompose_generator_yield(hint));
                self.ifs_infer(&x.generators, errors);
                let yield_ty = self
                    .expr_infer_type_info_with_hint(
                        &x.elt,
                        yield_hint.as_ref().map(|hint| hint.as_ref()),
                        errors,
                    )
                    .into_ty();
                if self.generator_expr_is_async(x) {
                    self.heap
                        .mk_class_type(self.stdlib.async_generator(yield_ty, self.heap.mk_none()))
                } else {
                    let none = self.heap.mk_none();
                    self.heap
                        .mk_class_type(self.stdlib.generator(yield_ty, none.clone(), none))
                }
            }
            Expr::Await(x) => {
                let awaiting_ty = self.expr_infer(&x.value, errors);
                self.distribute_over_union(&awaiting_ty, |ty| match self.unwrap_awaitable(ty) {
                    Some(ty) => ty,
                    None => self.error(
                        errors,
                        x.range,
                        ErrorInfo::Kind(ErrorKind::NotAsync),
                        ErrorContext::Await(self.for_display(ty.clone())).format(),
                    ),
                })
            }
            Expr::Yield(x) => self.get(&KeyYield(x.range)).send_ty.clone(),
            Expr::YieldFrom(x) => self.get(&KeyYieldFrom(x.range)).return_ty.clone(),
            Expr::Compare(x) => self.compare_infer(x, errors),
            Expr::Call(x) => {
                let callee_ty = self.expr_infer(&x.func, errors);
                if let Some(d) = self.call_to_dict(&callee_ty, &x.arguments) {
                    self.dict_infer(&d, hint, x.range, errors)
                } else if let Some((obj_ty, key)) =
                    self.is_dict_get_with_literal(&x.func, &x.arguments, errors)
                {
                    obj_ty
                        .at_facet(&FacetKind::Key(key.to_string()), || {
                            self.expr_call_infer(x, callee_ty.clone(), hint, errors)
                        })
                        .into_ty()
                } else {
                    self.expr_call_infer(x, callee_ty, hint, errors)
                }
            }
            Expr::FString(x) => {
                // Ensure we detect type errors in f-string expressions.
                let mut all_literal_strings = true;
                x.visit(&mut |x| {
                    let fstring_expr_ty = self.expr_infer(x, errors);
                    if !fstring_expr_ty.is_literal_string() {
                        all_literal_strings = false;
                    }
                });
                match Lit::from_fstring(x) {
                    Some(lit) => lit.to_implicit_type(),
                    _ if all_literal_strings => self.heap.mk_literal_string(LitStyle::Implicit),
                    _ => self.heap.mk_class_type(self.stdlib.str().clone()),
                }
            }
            Expr::TString(x) => {
                x.visit(&mut |x| {
                    self.expr_infer(x, errors);
                });
                if let Some(template) = self.stdlib.template() {
                    self.heap.mk_class_type(template.clone())
                } else {
                    self.error(
                        errors,
                        x.range,
                        ErrorInfo::Kind(ErrorKind::InvalidSyntax),
                        "t-strings are only available in Python 3.14+".to_owned(),
                    )
                }
            }
            Expr::StringLiteral(x) => match Lit::from_string_literal(x) {
                Some(lit) => lit.to_implicit_type(),
                None => self.heap.mk_literal_string(LitStyle::Implicit),
            },
            Expr::BytesLiteral(x) => Lit::from_bytes_literal(x).to_implicit_type(),
            Expr::NumberLiteral(x) => match &x.value {
                Number::Int(x) => Lit::from_int(x).to_implicit_type(),
                Number::Float(_) => self.heap.mk_class_type(self.stdlib.float().clone()),
                Number::Complex { .. } => self.heap.mk_class_type(self.stdlib.complex().clone()),
            },
            Expr::BooleanLiteral(x) => Lit::from_boolean_literal(x).to_implicit_type(),
            Expr::NoneLiteral(_) => self.heap.mk_none(),
            Expr::EllipsisLiteral(_) => self.heap.mk_ellipsis(),
            Expr::Starred(ExprStarred { value, .. }) => {
                let ty = self.expr_untype(value, TypeFormContext::TypeArgument, errors);
                self.heap.mk_unpack(ty)
            }
            Expr::Slice(x) => {
                let elt_exprs = [x.lower.as_ref(), x.upper.as_ref(), x.step.as_ref()];
                let elts = elt_exprs
                    .iter()
                    .filter_map(|e| e.map(|e| self.expr_infer(e, errors)))
                    .collect::<Vec<_>>();
                self.specialize(&self.stdlib.slice_class_object(), elts, x.range(), errors)
            }
            Expr::IpyEscapeCommand(x) => {
                if self.module().is_notebook() {
                    self.heap.mk_any_implicit()
                } else {
                    self.error(
                        errors,
                        x.range,
                        ErrorInfo::Kind(ErrorKind::Unsupported),
                        "IPython escapes are not supported".to_owned(),
                    )
                }
            }
        }
    }

    fn expr_infer_with_hint_promote(
        &self,
        x: &Expr,
        hint: Option<HintRef>,
        errors: &ErrorCollector,
    ) -> Type {
        let ty = self.expr_infer_with_hint(x, hint, errors);
        if let Some(want) = hint
            && self.is_subset_eq(&ty, want.ty())
        {
            want.ty().clone()
        } else {
            ty.promote_implicit_literals(self.stdlib)
        }
    }

    /// Check whether a type corresponds to a deprecated function or method, and if so, log a deprecation warning.
    fn check_for_deprecated_call(&self, ty: &Type, range: TextRange, errors: &ErrorCollector) {
        let Some(deprecation) = ty.function_deprecation() else {
            return;
        };
        let deprecated_function = ty
            .to_func_kind()
            .map(|func_kind| func_kind.format(self.module().name()));
        if let Some(deprecated_function) = deprecated_function {
            errors.add(
                range,
                ErrorInfo::Kind(ErrorKind::Deprecated),
                deprecation.as_error_message(format!("`{deprecated_function}` is deprecated")),
            );
        }
    }

    fn tuple_infer(&self, x: &ExprTuple, hint: Option<HintRef>, errors: &ErrorCollector) -> Type {
        let owner = Owner::new();
        let has_hint = hint.is_some();
        let (hint_ts, default_hint) = if let Some(hint) = &hint {
            let (tuples, nontuples) = self.split_tuple_hint(hint.ty());
            // Combine hints from multiple tuples.
            let mut element_hints: Vec<Vec1<&Type>> = Vec::new();
            let mut default_hint = Vec::new();
            for tuple in tuples {
                let (cur_element_hints, cur_default_hint) = self.tuple_to_element_hints(tuple);
                if let Some(cur_default_hint) = cur_default_hint {
                    // Use the default hint for any elements that this tuple doesn't provide per-element hints for.
                    for ts in element_hints.iter_mut().skip(cur_element_hints.len()) {
                        ts.push(cur_default_hint);
                    }
                    default_hint.push(cur_default_hint);
                }
                for (i, element_hint) in cur_element_hints.into_iter().enumerate() {
                    if i < element_hints.len() {
                        element_hints[i].push(element_hint);
                    } else {
                        element_hints.push(vec1![element_hint]);
                    }
                }
            }
            if !nontuples.is_empty() {
                // The non-tuple options may contain a type like Sequence[T] that provides an additional default hint.
                // TODO: we filter out top-level Vars to prevent premature pinning
                // (https://github.com/facebook/pyrefly/issues/105), but this also prevents us from picking up hints
                // from Quantified restrictions. See test::contextual::test_sequence_hint_in_typevar_bound.
                let nontuple_hint = self.unions(
                    nontuples
                        .into_iter()
                        .filter(|t| !matches!(t, Type::Var(_)))
                        .cloned()
                        .collect(),
                );
                let nontuple_element_hint =
                    self.decompose_tuple(HintRef::new(&nontuple_hint, hint.errors()));
                if let Some(nontuple_element_hint) = nontuple_element_hint {
                    let nontuple_element_hint = owner.push(nontuple_element_hint.to_type());
                    for ts in element_hints.iter_mut() {
                        ts.push(nontuple_element_hint);
                    }
                    default_hint.push(nontuple_element_hint);
                }
            }
            (
                element_hints.into_map(|ts| self.types_to_hint(ts, hint.errors(), &owner)),
                Vec1::try_from_vec(default_hint)
                    .ok()
                    .map(|ts| self.types_to_hint(ts, hint.errors(), &owner)),
            )
        } else {
            (Vec::new(), None)
        };
        let mut prefix = Vec::new();
        let mut unbounded = Vec::new();
        let mut suffix = Vec::new();
        let mut hint_ts_iter = hint_ts.into_iter();
        let mut encountered_invalid_star = false;
        for elt in x.elts.iter() {
            match elt {
                Expr::Starred(ExprStarred { value, .. }) => {
                    let ty = self.expr_infer(value, errors);
                    match ty {
                        Type::Tuple(Tuple::Concrete(elts)) => {
                            if unbounded.is_empty() {
                                if !elts.is_empty() {
                                    hint_ts_iter.nth(elts.len() - 1);
                                }
                                prefix.extend(elts);
                            } else {
                                suffix.extend(elts)
                            }
                        }
                        Type::Tuple(Tuple::Unpacked(box (pre, middle, suff)))
                            if unbounded.is_empty() =>
                        {
                            prefix.extend(pre);
                            suffix.extend(suff);
                            unbounded.push(middle);
                            hint_ts_iter.nth(usize::MAX);
                        }
                        _ => {
                            if let Some(iterable_ty) = self.unwrap_iterable(&ty) {
                                if !unbounded.is_empty() {
                                    unbounded
                                        .push(self.heap.mk_unbounded_tuple(self.unions(suffix)));
                                    suffix = Vec::new();
                                }
                                unbounded.push(self.heap.mk_unbounded_tuple(iterable_ty));
                                hint_ts_iter.nth(usize::MAX);
                            } else {
                                self.error(
                                    errors,
                                    x.range(),
                                    ErrorInfo::Kind(ErrorKind::NotIterable),
                                    format!("Expected an iterable, got `{}`", self.for_display(ty)),
                                );
                                encountered_invalid_star = true;
                                hint_ts_iter.nth(usize::MAX); // TODO: missing test
                            }
                        }
                    }
                }
                _ => {
                    let ty = self.expr_infer_type_no_trace(
                        elt,
                        if unbounded.is_empty() {
                            hint_ts_iter.next().or(default_hint)
                        } else {
                            None
                        },
                        errors,
                    );
                    if unbounded.is_empty() {
                        prefix.push(ty)
                    } else {
                        suffix.push(ty)
                    }
                }
            }
        }
        if encountered_invalid_star {
            // We already produced the type error, and we can't really roll up a suitable outermost type here.
            // TODO(stroxler): should we really be producing a `tuple[Any]` here? We do at least know *something* about the type!
            self.heap.mk_any_error()
        } else {
            match unbounded.as_slice() {
                [] => {
                    if !has_hint && prefix.len() > MAX_TUPLE_LENGTH {
                        self.heap.mk_unbounded_tuple(self.heap.mk_any_implicit())
                    } else {
                        self.heap.mk_concrete_tuple(prefix)
                    }
                }
                [middle] => self.heap.mk_unpacked_tuple(prefix, middle.clone(), suffix),
                // We can't precisely model unpacking two unbounded iterables, so we'll keep any
                // concrete prefix and suffix elements and merge everything in between into an unbounded tuple
                _ => {
                    let middle_types: Vec<Type> = unbounded
                        .iter()
                        .map(|t| {
                            self.unwrap_iterable(t)
                                .unwrap_or_else(|| self.heap.mk_any_implicit())
                        })
                        .collect();
                    self.heap.mk_unpacked_tuple(
                        prefix,
                        self.heap.mk_unbounded_tuple(self.unions(middle_types)),
                        suffix,
                    )
                }
            }
        }
    }

    fn split_tuple_hint<'b>(&self, hint: &'b Type) -> (Vec<&'b Tuple>, Vec<&'b Type>) {
        match hint {
            Type::Tuple(tuple) => (vec![tuple], Vec::new()),
            Type::Union(box Union { members, .. }) => members.iter().partition_map(|t| match t {
                Type::Tuple(tuple) => Either::Left(tuple),
                _ => Either::Right(t),
            }),
            _ => (Vec::new(), vec![hint]),
        }
    }

    fn tuple_to_element_hints<'b>(&self, tup: &'b Tuple) -> (Vec<&'b Type>, Option<&'b Type>) {
        match tup {
            Tuple::Concrete(elts) => (elts.iter().collect(), None),
            Tuple::Unpacked(box (prefix, _, _)) => {
                // TODO: We should also contextually type based on the middle and suffix
                (prefix.iter().collect(), None)
            }
            Tuple::Unbounded(elt) => (Vec::new(), Some(elt)),
        }
    }

    fn types_to_hint<'b>(
        &self,
        ts: Vec1<&'b Type>,
        errors: Option<&'b ErrorCollector>,
        owner: &'b Owner<Type>,
    ) -> HintRef<'b, 'b> {
        if ts.len() == 1 {
            let (t, _) = ts.split_off_first();
            HintRef::new(t, errors)
        } else {
            HintRef::new(
                owner.push(self.unions(ts.into_iter().cloned().collect())),
                errors,
            )
        }
    }

    fn dict_infer(
        &self,
        items: &[DictItem],
        hint: Option<HintRef>,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        let flattened_items = Ast::flatten_dict_items(items);
        let hints = hint.as_ref().map_or(Vec::new(), |hint| match hint.ty() {
            Type::Union(box Union { members: ts, .. }) => ts
                .iter()
                .map(|ty| HintRef::new(ty, hint.errors()))
                .collect(),
            _ => vec![*hint],
        });
        for hint in hints.iter() {
            let (typed_dict, is_update) = match hint.ty() {
                Type::TypedDict(td) => (td, false),
                Type::PartialTypedDict(td) => (td, true),
                _ => continue,
            };
            let check_errors = self.error_collector();
            let item_errors = self.error_collector();
            self.check_dict_items_against_typed_dict(
                &flattened_items,
                typed_dict,
                is_update,
                range,
                &check_errors,
                &item_errors,
            );

            // We use the TypedDict hint if it successfully matched or if there is only one hint, unless
            // this is a "soft" type hint, in which case we don't want to raise any check errors.
            if check_errors.is_empty()
                || hints.len() == 1
                    && hint
                        .errors()
                        .inspect(|errors| errors.extend(check_errors))
                        .is_some()
            {
                errors.extend(item_errors);
                return (*hint.ty()).clone();
            }
        }
        // Note that we don't need to filter out the TypedDict options here; any non-`dict` options
        // are ignored when decomposing the hint.
        self.dict_items_infer(range, flattened_items, hint, errors)
    }

    /// Infers a type for a dictionary literal with the specified items & an optional contextual hint
    /// In order to preserve information about heterogeneous key/value types, we will infer an anonymous
    /// typed dict if the following conditions are met:
    /// - there cannot already be a contextual hint
    /// - all the keys must be string literals
    /// - there cannot be any unpackings
    /// - the dict cannot be empty
    fn dict_items_infer(
        &self,
        range: TextRange,
        items: Vec<&DictItem>,
        hint: Option<HintRef>,
        errors: &ErrorCollector,
    ) -> Type {
        let (key_hint, value_hint) = hint.map_or((None, None), |ty| self.decompose_dict(ty));
        if items.is_empty() {
            let key_ty = key_hint.map_or_else(
                || {
                    self.solver()
                        .fresh_partial_contained(self.uniques, range)
                        .to_type(self.heap)
                },
                |ty| ty.to_type(),
            );
            let value_ty = value_hint.map_or_else(
                || {
                    self.solver()
                        .fresh_partial_contained(self.uniques, range)
                        .to_type(self.heap)
                },
                |ty| ty.to_type(),
            );
            self.heap.mk_class_type(self.stdlib.dict(key_ty, value_ty))
        } else {
            let mut typed_dict_fields = Vec::new();
            let can_create_anonymous_typed_dict = hint.is_none()
                && items.len() <= ANONYMOUS_TYPED_DICT_MAX_ITEMS
                && items.iter().all(|item| {
                    item.key
                        .as_ref()
                        .is_some_and(|k| k.as_string_literal_expr().is_some())
                });
            let mut key_tys = Vec::new();
            let mut value_tys = Vec::new();
            let mut has_type_mismatch = false;
            let check_items = hint
                .filter(|hint| hint.errors().is_some())
                .is_some_and(|hint| !matches!(hint.ty(), Type::Union(_)));
            let infer_with_hint = |expr: &Expr, hint: Option<&Hint>, errors: &ErrorCollector| {
                let ty = self.expr_infer_with_hint(expr, hint.map(|hint| hint.as_ref()), errors);
                match hint {
                    Some(hint) => {
                        let matches_hint = self.is_subset_eq(&ty, hint.ty());
                        let ty = if matches_hint {
                            hint.ty().clone()
                        } else {
                            ty.promote_implicit_literals(self.stdlib)
                        };
                        (ty, matches_hint)
                    }
                    None => (ty.promote_implicit_literals(self.stdlib), true),
                }
            };
            items.iter().for_each(|x| match &x.key {
                Some(key) => {
                    let (key_t, key_matches_hint) = infer_with_hint(key, key_hint.as_ref(), errors);
                    let (value_t, value_matches_hint) =
                        infer_with_hint(&x.value, value_hint.as_ref(), errors);
                    if check_items
                        && let Some(hint) = hint
                        && let Some(check_errors) = hint.errors()
                    {
                        let tcc: &dyn Fn() -> TypeCheckContext =
                            &|| TypeCheckContext::of_kind(TypeCheckKind::DictLiteralItem);
                        if let Some(key_hint) = &key_hint
                            && !key_matches_hint
                        {
                            self.check_type(&key_t, key_hint.ty(), key.range(), check_errors, tcc);
                            has_type_mismatch = true;
                        }
                        if let Some(value_hint) = &value_hint
                            && !value_matches_hint
                        {
                            self.check_type(
                                &value_t,
                                value_hint.ty(),
                                x.value.range(),
                                check_errors,
                                tcc,
                            );
                            has_type_mismatch = true;
                        }
                    }
                    if !key_t.is_error() {
                        key_tys.push(key_t);
                    }
                    if !value_t.is_error() {
                        if can_create_anonymous_typed_dict
                            && let Some(string_lit) = key.as_string_literal_expr()
                        {
                            let key_name = Name::new(string_lit.value.to_str());
                            typed_dict_fields.push((
                                key_name,
                                TypedDictField {
                                    ty: if value_t.is_none() {
                                        self.heap.mk_union(vec![
                                            self.heap.mk_none(),
                                            self.solver()
                                                .fresh_partial_contained(
                                                    self.uniques,
                                                    x.value.range(),
                                                )
                                                .to_type(self.heap),
                                        ])
                                    } else {
                                        value_t.clone()
                                    },
                                    required: false,
                                    read_only_reason: None,
                                },
                            ));
                        }
                        value_tys.push(value_t);
                    }
                }
                None => {
                    let ty = self.expr_infer(&x.value, errors);
                    if let Some((key_t, value_t)) = self.unwrap_mapping(&ty) {
                        let key_matches_hint = key_hint
                            .as_ref()
                            .is_some_and(|key_hint| self.is_subset_eq(&key_t, key_hint.ty()));
                        let value_matches_hint = value_hint
                            .as_ref()
                            .is_some_and(|value_hint| self.is_subset_eq(&value_t, value_hint.ty()));
                        if check_items
                            && let Some(hint) = hint
                            && let Some(check_errors) = hint.errors()
                        {
                            let tcc: &dyn Fn() -> TypeCheckContext =
                                &|| TypeCheckContext::of_kind(TypeCheckKind::DictLiteralItem);
                            if let Some(key_hint) = &key_hint
                                && !key_matches_hint
                            {
                                self.check_type(
                                    &key_t,
                                    key_hint.ty(),
                                    x.value.range(),
                                    check_errors,
                                    tcc,
                                );
                                has_type_mismatch = true;
                            }
                            if let Some(value_hint) = &value_hint
                                && !value_matches_hint
                            {
                                self.check_type(
                                    &value_t,
                                    value_hint.ty(),
                                    x.value.range(),
                                    check_errors,
                                    tcc,
                                );
                                has_type_mismatch = true;
                            }
                        }
                        if !key_t.is_error() {
                            if let Some(key_hint) = &key_hint
                                && self.is_subset_eq(&key_t, key_hint.ty())
                            {
                                key_tys.push(key_hint.ty().clone());
                            } else {
                                key_tys.push(key_t);
                            }
                        }
                        if !value_t.is_error() {
                            if let Some(value_hint) = &value_hint
                                && self.is_subset_eq(&value_t, value_hint.ty())
                            {
                                value_tys.push(value_hint.ty().clone());
                            } else {
                                value_tys.push(value_t);
                            }
                        }
                    } else {
                        self.error(
                            errors,
                            x.value.range(),
                            ErrorInfo::Kind(ErrorKind::InvalidArgument),
                            format!("Expected a mapping, got {}", self.for_display(ty)),
                        );
                    }
                }
            });
            if has_type_mismatch
                && let Some(hint) = hint
                && hint.errors().is_some()
            {
                return hint.ty().clone();
            }
            if can_create_anonymous_typed_dict && !typed_dict_fields.is_empty() {
                return self.heap.mk_typed_dict(TypedDict::Anonymous(Box::new(
                    AnonymousTypedDictInner {
                        fields: typed_dict_fields,
                        value_type: self.unions(value_tys),
                    },
                )));
            }
            if key_tys.is_empty() {
                key_tys.push(self.heap.mk_any_error())
            }
            if value_tys.is_empty() {
                value_tys.push(self.heap.mk_any_error())
            }
            let key_ty = self.unions(key_tys);
            let value_ty = self.unions(value_tys);
            self.heap.mk_class_type(self.stdlib.dict(key_ty, value_ty))
        }
    }

    /// If this is a `dict` call that can be converted to an equivalent dict literal (e.g., `dict(x=1)` => `{'x': 1}`),
    /// return the items in the converted dict.
    fn call_to_dict(&self, callee_ty: &Type, args: &Arguments) -> Option<Vec<DictItem>> {
        if !matches!(callee_ty, Type::ClassDef(class) if class.is_builtin("dict")) {
            return None;
        }
        if !args.args.is_empty() {
            // The positional args could contain expressions that are convertible to dict literals,
            // but this is a less common pattern, so we defer supporting it for now.
            return None;
        }
        Some(args.keywords.map(|kw| {
            DictItem {
                key: kw
                    .arg
                    .as_ref()
                    .map(|id| Ast::str_expr(id.as_str(), id.range)),
                value: kw.value.clone(),
            }
        }))
    }

    // Is this a call to `dict.get` with a single string literal argument
    fn is_dict_get_with_literal(
        &self,
        func: &Expr,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<(TypeInfo, StringLiteralValue)> {
        let Expr::Attribute(attr_expr) = func else {
            return None;
        };
        if attr_expr.attr.id.as_str() != "get" {
            return None;
        }
        if args.args.len() != 1 {
            return None;
        }
        let Expr::StringLiteral(ExprStringLiteral { value: key, .. }) = &args.args[0] else {
            return None;
        };
        let obj_ty = self.expr_infer_type_info_with_hint(&attr_expr.value, None, errors);
        if self.is_dict_like(obj_ty.ty()) {
            Some((obj_ty, key.clone()))
        } else {
            None
        }
    }

    // Is this type a `TypedDict` or subtype of `dict`, but not `Any`?
    pub fn is_dict_like(&self, ty: &Type) -> bool {
        if ty.is_any() {
            return false;
        }
        if ty.is_typed_dict() {
            return true;
        }
        let dict_type = self.heap.mk_class_type(
            self.stdlib
                .dict(self.heap.mk_any_implicit(), self.heap.mk_any_implicit()),
        );
        self.is_subset_eq(ty, &dict_type)
    }

    /// Determine the boolean behavior of a type:
    /// - `Some(true)` or `Some(false)` when it is known to be statically truthy
    ///   or falsey (as determined by some baked in rules for literals
    ///   and looking at the `__bool__` method, if it is present).
    /// - `None` if it's truthiness is not statically known.
    pub fn as_bool(&self, ty: &Type, range: TextRange, errors: &ErrorCollector) -> Option<bool> {
        if let Type::TypedDict(td) = ty {
            // If a TypedDict has ANY required keys, it can never be empty.
            // Therefore, it is always Truthy.
            if self
                .typed_dict_fields(td)
                .values()
                .any(|field| field.required)
            {
                return Some(true);
            }
        }
        ty.as_bool().or_else(|| {
            // If the object defines `__bool__`, we can check if it returns a statically known value
            if self
                .type_of_magic_dunder_attr(ty, &dunder::BOOL, range, errors, None, "as_bool", true)?
                .is_never()
            {
                return None;
            };
            self.call_method_or_error(ty, &dunder::BOOL, range, &[], &[], errors, None)
                .as_bool()
        })
    }

    // Helper method for inferring the type of a boolean operation over a sequence of values.
    fn boolop(
        &self,
        values: &[Expr],
        op: BoolOp,
        hint: Option<HintRef>,
        errors: &ErrorCollector,
    ) -> Type {
        let target = match op {
            BoolOp::And => false,
            BoolOp::Or => true,
        };
        let should_shortcircuit =
            |t: &Type, r: TextRange| self.as_bool(t, r, errors) == Some(target);
        let should_discard = |t: &Type, r: TextRange| self.as_bool(t, r, errors) == Some(!target);

        let mut t_acc = self.heap.mk_never();
        // Separate accumulator for soft hints - uses un-narrowed types.
        // The narrowing of bool/int/str to literals is for the result type of the boolop,
        // not for contextual typing of subsequent expressions.
        let mut hint_acc: Option<Type> = None;
        let last_index = values.len() - 1;
        for (i, value) in values.iter().enumerate() {
            // If there isn't a hint for the overall expression, use the preceding branches as a "soft" hint
            // for the next one. Most useful for expressions like `optional_list or []`.
            let hint = hint.or_else(|| hint_acc.as_ref().map(HintRef::soft));
            let mut t = self.expr_infer_with_hint(value, hint, errors);
            self.expand_vars_mut(&mut t);
            // If this is not the last entry, we have to make a type-dependent decision and also narrow the
            // result; both operations require us to force `Var` first or they become unpredictable.
            if i < last_index {
                t = self.force_for_narrowing(&t, value.range(), errors);
            }
            if i < last_index && should_shortcircuit(&t, value.range()) {
                t_acc = self.union(t_acc, t);
                break;
            }
            for t in t.into_unions() {
                // If we reach the last value, we should always keep it.
                if i == last_index || !should_discard(&t, value.range()) {
                    // Accumulate un-narrowed type for hints
                    hint_acc = Some(match hint_acc {
                        None => t.clone(),
                        Some(acc) => self.union(acc, t.clone()),
                    });
                    // Narrow the type for the result of the boolop
                    let t = if i != last_index
                        && t == self.heap.mk_class_type(self.stdlib.bool().clone())
                    {
                        Lit::Bool(target).to_implicit_type()
                    } else if i != last_index
                        && t == self.heap.mk_class_type(self.stdlib.int().clone())
                        && !target
                    {
                        LitInt::new(0).to_implicit_type()
                    } else if i != last_index
                        && t == self.heap.mk_class_type(self.stdlib.str().clone())
                        && !target
                    {
                        Lit::Str(Default::default()).to_implicit_type()
                    } else {
                        t
                    };
                    t_acc = self.union(t_acc, t)
                }
            }
        }
        t_acc
    }

    /// Infers types for `if` clauses in the given comprehensions.
    /// This is for error detection only; the types are not used.
    fn ifs_infer(&self, comps: &[Comprehension], errors: &ErrorCollector) {
        for comp in comps {
            for if_clause in comp.ifs.iter() {
                let ty = self.expr_infer(if_clause, errors);
                self.check_redundant_condition(&ty, if_clause.range(), errors);
            }
        }
    }

    /// If a comprehension contains `async for` clauses, or if it contains
    /// `await` expressions or other asynchronous comprehensions anywhere except
    /// the iterable expression in the leftmost `for` clause, it is treated as an `AsyncGenerator`
    fn generator_expr_is_async(&self, generator: &ExprGenerator) -> bool {
        if Ast::contains_await(&generator.elt) {
            return true;
        }
        for (idx, comp) in generator.generators.iter().enumerate() {
            if comp.is_async
                || (idx != 0 && Ast::contains_await(&comp.iter))
                || Ast::contains_await(&comp.target)
                || comp.ifs.iter().any(Ast::contains_await)
            {
                return true;
            }
        }
        false
    }

    pub fn attr_infer_for_type(
        &self,
        base: &Type,
        attr_name: &Name,
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) -> Type {
        self.type_of_attr_get(
            base,
            attr_name,
            range,
            errors,
            context,
            "Expr::attr_infer_for_type",
        )
    }

    pub fn attr_infer(
        &self,
        base: &TypeInfo,
        attr_name: &Name,
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) -> TypeInfo {
        TypeInfo::at_facet(base, &FacetKind::Attribute(attr_name.clone()), || {
            self.attr_infer_for_type(base.ty(), attr_name, range, errors, context)
        })
    }

    pub fn subscript_infer(
        &self,
        base: &TypeInfo,
        slice: &Expr,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> TypeInfo {
        if let Expr::NumberLiteral(ExprNumberLiteral {
            value: Number::Int(idx),
            ..
        }) = slice
            && let Some(idx) = idx.as_usize()
        {
            TypeInfo::at_facet(base, &FacetKind::Index(idx), || {
                self.subscript_infer_for_type(base.ty(), slice, range, errors)
            })
        } else if let Expr::StringLiteral(ExprStringLiteral { value, .. }) = slice {
            TypeInfo::at_facet(base, &FacetKind::Key(value.to_string()), || {
                self.subscript_infer_for_type(base.ty(), slice, range, errors)
            })
        } else {
            let swallower = self.error_swallower();
            match self.expr_infer(slice, &swallower) {
                Type::Literal(ref lit) if let Lit::Str(value) = &lit.value => {
                    TypeInfo::at_facet(base, &FacetKind::Key(value.to_string()), || {
                        self.subscript_infer_for_type(base.ty(), slice, range, errors)
                    })
                }
                _ => {
                    TypeInfo::of_ty(self.subscript_infer_for_type(base.ty(), slice, range, errors))
                }
            }
        }
    }

    /// When interpreted as static types (as opposed to when accounting for runtime
    /// behavior when used as values), `Type::ClassDef(cls)` is equivalent to
    /// `Type::Type(box Type::ClassType(cls, default_targs(cls)))` where `default_targs(cls)`
    /// is the result of looking up the class `tparams` and synthesizing default `targs` that
    /// are gradual if needed (e.g. `list` is treated as `list[Any]` when used as an annotation).
    ///
    /// This function canonicalizes to `Type::ClassType` or `Type::TypedDict`
    pub fn canonicalize_all_class_types(
        &self,
        ty: Type,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        ty.transform(&mut |ty| match ty {
            Type::SpecialForm(SpecialForm::Tuple) => {
                Self::add_implicit_any_error(errors, range, "class `tuple`".to_owned(), None);
                *ty = self.heap.mk_unbounded_tuple(self.heap.mk_any_implicit());
            }
            Type::SpecialForm(SpecialForm::Callable) => {
                Self::add_implicit_any_error(errors, range, "class `Callable`".to_owned(), None);
                *ty = self.heap.mk_callable_ellipsis(self.heap.mk_any_implicit())
            }
            Type::SpecialForm(SpecialForm::Type) => {
                Self::add_implicit_any_error(errors, range, "class `type`".to_owned(), None);
                *ty = self.heap.mk_type_form(self.heap.mk_any_implicit())
            }
            Type::ClassDef(cls) => {
                if cls.is_builtin("tuple") {
                    Self::add_implicit_any_error(errors, range, "class `tuple`".to_owned(), None);
                    *ty = self
                        .heap
                        .mk_type_form(self.heap.mk_unbounded_tuple(self.heap.mk_any_implicit()));
                } else if cls.is_builtin("type") {
                    // `type`` is equivalent to `type[Any]`. As a result, the class def itself
                    // has type `type[type[Any]]`.
                    *ty = self
                        .heap
                        .mk_type_form(self.heap.mk_type_form(self.heap.mk_any_implicit()));
                } else if cls.has_toplevel_qname("typing", "Any") {
                    *ty = self.heap.mk_type_form(self.heap.mk_any_explicit())
                } else {
                    // All other classes (including Tensor) get promoted and wrapped in type_form
                    *ty = self.heap.mk_type_form(self.promote(cls, range, errors));
                }
            }
            Type::ClassType(cls) if cls.is_builtin("type") => {
                *ty = self.heap.mk_type_form(self.heap.mk_any_implicit());
            }
            _ => {}
        })
    }

    fn literal_bool_infer(&self, x: &Expr, errors: &ErrorCollector) -> bool {
        let ty = self.expr_infer(x, errors);
        match ty {
            Type::Literal(lit) if let Lit::Bool(b) = lit.value => b,
            _ => {
                self.error(
                    errors,
                    x.range(),
                    ErrorInfo::Kind(ErrorKind::InvalidLiteral),
                    format!(
                        "Expected literal `True` or `False`, got `{}`",
                        self.for_display(ty)
                    ),
                );
                false
            }
        }
    }

    pub fn typevar_from_call(
        &self,
        name: Identifier,
        x: &ExprCall,
        errors: &ErrorCollector,
    ) -> TypeVar {
        let mut arg_name = false;
        let mut restriction = None;
        let mut default = None;
        let mut variance = None;

        let check_name_arg = |arg: &Expr| {
            if let Expr::StringLiteral(lit) = arg {
                if lit.value.to_str() != name.id.as_str() {
                    self.error(
                        errors,
                        x.range,
                        ErrorInfo::Kind(ErrorKind::InvalidTypeVar),
                        format!(
                            "TypeVar must be assigned to a variable named `{}`",
                            lit.value.to_str()
                        ),
                    );
                }
            } else {
                self.error(
                    errors,
                    arg.range(),
                    ErrorInfo::Kind(ErrorKind::InvalidTypeVar),
                    "Expected first argument of TypeVar to be a string literal".to_owned(),
                );
            }
        };

        let mut try_set_variance = |kw: &Keyword, v: PreInferenceVariance| {
            if self.literal_bool_infer(&kw.value, errors) {
                if variance.is_some() {
                    self.error(
                        errors,
                        kw.range,
                        ErrorInfo::Kind(ErrorKind::InvalidTypeVar),
                        "Contradictory variance specifications".to_owned(),
                    );
                } else {
                    variance = Some(v);
                }
            }
        };

        let mut iargs = x.arguments.args.iter();
        if let Some(arg) = iargs.next() {
            check_name_arg(arg);
            arg_name = true;
        }

        let constraints: Vec<Type> = iargs
            .map(|arg| self.expr_untype(arg, TypeFormContext::TypeVarConstraint, errors))
            .collect();
        if !constraints.is_empty() {
            restriction = Some(Restriction::Constraints(constraints));
        }

        for kw in &x.arguments.keywords {
            match &kw.arg {
                Some(id) => match id.id.as_str() {
                    "bound" => {
                        let bound =
                            self.expr_untype(&kw.value, TypeFormContext::TypeVarConstraint, errors);
                        if restriction.is_some() {
                            self.error(
                                errors,
                                kw.range,
                                ErrorInfo::Kind(ErrorKind::InvalidTypeVar),
                                "TypeVar cannot have both constraints and bound".to_owned(),
                            );
                            restriction = Some(Restriction::Unrestricted);
                        } else {
                            restriction = Some(Restriction::Bound(bound));
                        }
                    }
                    "default" => {
                        default = Some((
                            self.expr_untype(&kw.value, TypeFormContext::TypeVarDefault, errors),
                            kw.value.range(),
                        ))
                    }
                    "covariant" => try_set_variance(kw, PreInferenceVariance::Covariant),
                    "contravariant" => try_set_variance(kw, PreInferenceVariance::Contravariant),
                    "invariant" => try_set_variance(kw, PreInferenceVariance::Invariant),
                    "infer_variance" => try_set_variance(kw, PreInferenceVariance::Undefined),
                    "name" => {
                        if arg_name {
                            self.error(
                                errors,
                                kw.range,
                                ErrorInfo::Kind(ErrorKind::InvalidTypeVar),
                                "Multiple values for argument `name`".to_owned(),
                            );
                        } else {
                            check_name_arg(&kw.value);
                            arg_name = true;
                        }
                    }
                    _ => {
                        self.error(
                            errors,
                            kw.range,
                            ErrorInfo::Kind(ErrorKind::InvalidTypeVar),
                            format!("Unexpected keyword argument `{}` to TypeVar", id.id),
                        );
                    }
                },
                _ => {
                    self.error(
                        errors,
                        kw.range,
                        ErrorInfo::Kind(ErrorKind::InvalidTypeVar),
                        "Cannot pass unpacked keyword arguments to TypeVar".to_owned(),
                    );
                }
            }
        }

        if !arg_name {
            self.error(
                errors,
                x.range,
                ErrorInfo::Kind(ErrorKind::InvalidTypeVar),
                "Missing `name` argument".to_owned(),
            );
        }
        // If we ended up with a single constraint, emit an error and treat as unrestricted.
        if let Some(Restriction::Constraints(cs)) = &restriction
            && cs.len() < 2
        {
            self.error(
                errors,
                x.range,
                ErrorInfo::Kind(ErrorKind::InvalidTypeVar),
                format!(
                    "Expected at least 2 constraints in TypeVar `{}`, got {}",
                    name.id,
                    cs.len(),
                ),
            );
            restriction = Some(Restriction::Unrestricted);
        }
        let restriction = restriction.unwrap_or(Restriction::Unrestricted);
        let mut default_value = None;
        if let Some((default_ty, default_range)) = default {
            default_value = Some(self.validate_type_var_default(
                &name.id,
                QuantifiedKind::TypeVar,
                &default_ty,
                default_range,
                &restriction,
                errors,
            ));
        }

        let variance = variance.unwrap_or(PreInferenceVariance::Invariant);

        TypeVar::new(
            name,
            self.module().dupe(),
            restriction,
            default_value,
            variance,
        )
    }

    pub fn paramspec_from_call(
        &self,
        name: Identifier,
        x: &ExprCall,
        errors: &ErrorCollector,
    ) -> ParamSpec {
        // TODO: check and complain on extra args, keywords
        let mut arg_name = false;

        let check_name_arg = |arg: &Expr| {
            if let Expr::StringLiteral(lit) = arg {
                if lit.value.to_str() != name.id.as_str() {
                    self.error(
                        errors,
                        x.range,
                        ErrorInfo::Kind(ErrorKind::InvalidParamSpec),
                        format!(
                            "ParamSpec must be assigned to a variable named `{}`",
                            lit.value.to_str()
                        ),
                    );
                }
            } else {
                self.error(
                    errors,
                    arg.range(),
                    ErrorInfo::Kind(ErrorKind::InvalidParamSpec),
                    "Expected first argument of ParamSpec to be a string literal".to_owned(),
                );
            }
        };

        if let Some(arg) = x.arguments.args.first() {
            check_name_arg(arg);
            arg_name = true;
        }
        let mut default = None;
        for kw in &x.arguments.keywords {
            match &kw.arg {
                Some(id) => match id.id.as_str() {
                    "name" => {
                        if arg_name {
                            self.error(
                                errors,
                                kw.range,
                                ErrorInfo::Kind(ErrorKind::InvalidParamSpec),
                                "Multiple values for argument `name`".to_owned(),
                            );
                        } else {
                            check_name_arg(&kw.value);
                            arg_name = true;
                        }
                    }
                    "default" => {
                        default = Some((
                            self.expr_untype(&kw.value, TypeFormContext::ParamSpecDefault, errors),
                            kw.range(),
                        ));
                    }
                    _ => {
                        self.error(
                            errors,
                            kw.range,
                            ErrorInfo::Kind(ErrorKind::InvalidParamSpec),
                            format!("Unexpected keyword argument `{}` to ParamSpec", id.id),
                        );
                    }
                },
                _ => {
                    self.error(
                        errors,
                        kw.range,
                        ErrorInfo::Kind(ErrorKind::InvalidParamSpec),
                        "Cannot pass unpacked keyword arguments to ParamSpec".to_owned(),
                    );
                }
            }
        }

        if !arg_name {
            self.error(
                errors,
                x.range,
                ErrorInfo::Kind(ErrorKind::InvalidParamSpec),
                "Missing `name` argument".to_owned(),
            );
        }
        let mut default_value = None;
        if let Some((default_ty, default_range)) = default {
            default_value = Some(self.validate_type_var_default(
                &name.id,
                QuantifiedKind::ParamSpec,
                &default_ty,
                default_range,
                &Restriction::Unrestricted,
                errors,
            ));
        }
        ParamSpec::new(name, self.module().dupe(), default_value)
    }

    pub fn typevartuple_from_call(
        &self,
        name: Identifier,
        x: &ExprCall,
        errors: &ErrorCollector,
    ) -> TypeVarTuple {
        let mut arg_name = false;
        let check_name_arg = |arg: &Expr| {
            if let Expr::StringLiteral(lit) = arg {
                if lit.value.to_str() != name.id.as_str() {
                    self.error(
                        errors,
                        x.range,
                        ErrorInfo::Kind(ErrorKind::InvalidTypeVarTuple),
                        format!(
                            "TypeVarTuple must be assigned to a variable named `{}`",
                            lit.value.to_str()
                        ),
                    );
                }
            } else {
                self.error(
                    errors,
                    arg.range(),
                    ErrorInfo::Kind(ErrorKind::InvalidTypeVarTuple),
                    "Expected first argument of TypeVarTuple to be a string literal".to_owned(),
                );
            }
        };
        if let Some(arg) = x.arguments.args.first() {
            check_name_arg(arg);
            arg_name = true;
        }
        if let Some(arg) = x.arguments.args.get(1) {
            self.error(
                errors,
                arg.range(),
                ErrorInfo::Kind(ErrorKind::InvalidTypeVarTuple),
                "Unexpected positional argument to TypeVarTuple".to_owned(),
            );
        }
        let mut default = None;
        for kw in &x.arguments.keywords {
            match &kw.arg {
                Some(id) => match id.id.as_str() {
                    "name" => {
                        if arg_name {
                            self.error(
                                errors,
                                kw.range,
                                ErrorInfo::Kind(ErrorKind::InvalidTypeVarTuple),
                                "Multiple values for argument `name`".to_owned(),
                            );
                        } else {
                            check_name_arg(&kw.value);
                            arg_name = true;
                        }
                    }
                    "default" => {
                        default = Some((
                            self.expr_untype(
                                &kw.value,
                                TypeFormContext::TypeVarTupleDefault,
                                errors,
                            ),
                            kw.range(),
                        ));
                    }
                    _ => {
                        self.error(
                            errors,
                            kw.range,
                            ErrorInfo::Kind(ErrorKind::InvalidTypeVarTuple),
                            format!("Unexpected keyword argument `{}` to TypeVarTuple", id.id),
                        );
                    }
                },
                _ => {
                    self.error(
                        errors,
                        kw.range,
                        ErrorInfo::Kind(ErrorKind::InvalidTypeVarTuple),
                        "Cannot pass unpacked keyword arguments to TypeVarTuple".to_owned(),
                    );
                }
            }
        }
        if !arg_name {
            self.error(
                errors,
                x.range,
                ErrorInfo::Kind(ErrorKind::InvalidTypeVarTuple),
                "Missing `name` argument".to_owned(),
            );
        }
        let mut default_value = None;
        if let Some((default_ty, default_range)) = default {
            default_value = Some(self.validate_type_var_default(
                &name.id,
                QuantifiedKind::TypeVarTuple,
                &default_ty,
                default_range,
                &Restriction::Unrestricted,
                errors,
            ));
        }
        TypeVarTuple::new(name, self.module().dupe(), default_value)
    }

    /// Helper to infer element types for a list or set.
    fn elts_infer(
        &self,
        elts: &[Expr],
        elt_hint: Option<Hint>,
        errors: &ErrorCollector,
    ) -> Vec<Type> {
        let star_hint = LazyCell::new(|| {
            elt_hint.as_ref().map(|hint| {
                hint.as_ref()
                    .map_ty(|ty| self.heap.mk_class_type(self.stdlib.iterable(ty.clone())))
            })
        });
        elts.map(|x| match x {
            Expr::Starred(ExprStarred { value, .. }) => {
                let unpacked_ty = self.expr_infer_with_hint_promote(
                    value,
                    star_hint.as_ref().map(|hint| hint.as_ref()),
                    errors,
                );
                if let Some(iterable_ty) = self.unwrap_iterable(&unpacked_ty) {
                    iterable_ty
                } else {
                    self.error(
                        errors,
                        x.range(),
                        ErrorInfo::Kind(ErrorKind::NotIterable),
                        format!(
                            "Expected an iterable, got `{}`",
                            self.for_display(unpacked_ty)
                        ),
                    )
                }
            }
            _ => self.expr_infer_with_hint_promote(
                x,
                elt_hint.as_ref().map(|hint| hint.as_ref()),
                errors,
            ),
        })
    }

    fn intercept_typing_self_use(&self, x: &Expr) -> Option<TypeInfo> {
        match x {
            Expr::Name(..) | Expr::Attribute(..) => {
                let key = Key::SelfTypeLiteral(x.range());
                let self_type_form = self.get_hashed_opt(Hashed::new(&key))?;
                Some(self_type_form.arc_clone())
            }
            _ => None,
        }
    }

    fn is_enum_class_type(&self, ty: &Type) -> bool {
        match ty {
            Type::ClassType(cls) | Type::SelfType(cls) => {
                self.has_superclass(cls.class_object(), self.stdlib.enum_class().class_object())
            }
            Type::Union(box Union {
                members: variants, ..
            }) => variants
                .iter()
                .all(|variant| self.is_enum_class_type(variant)),
            _ => false,
        }
    }

    fn is_restricted_to_enum_class_def_type(&self, quantified: &Quantified) -> bool {
        match quantified.restriction() {
            Restriction::Unrestricted => false,
            Restriction::Bound(bound) => self.is_enum_class_type(bound),
            Restriction::Constraints(constraints) => {
                !constraints.is_empty()
                    && constraints
                        .iter()
                        .all(|constraint| self.is_enum_class_type(constraint))
            }
        }
    }

    pub fn subscript_infer_for_type(
        &self,
        base: &Type,
        slice: &Expr,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        let xs = Ast::unpack_slice(slice);
        self.distribute_over_union(base, |base| {
            let mut base = base.clone();
            if let Type::Var(v) = base {
                base = self.solver().force_var(v);
            }
            if matches!(&base, Type::ClassDef(t) if t.name() == "tuple") {
                base = self.heap.mk_type_form(self.heap.mk_special_form(SpecialForm::Tuple));
            }
            if let Type::Intersect(x) = base {
                // TODO: Handle subscription of intersections properly.
                base = x.1;
            }
            match base {
                Type::Forall(forall) => {
                    let tys =
                        xs.map(|x| self.expr_untype(x, TypeFormContext::TypeArgument, errors));
                    self.specialize_forall(*forall, tys, range, errors)
                }
                // Note that we have to check for `builtins.type` by name here because this code runs
                // when we're bootstrapping the stdlib and don't have access to class objects yet.
                Type::ClassDef(cls) if cls.is_builtin("type") => {
                    let targ = match xs.len() {
                        // This causes us to treat `type[list]` as equivalent to `type[list[Any]]`,
                        // which may or may not be what we want.
                        1 => self.expr_untype(&xs[0], TypeFormContext::TypeArgumentForType, errors),
                        _ => self.error(
                            errors,
                            range,
                            ErrorInfo::Kind(ErrorKind::BadSpecialization),
                            format!("Expected 1 type argument for `type`, got {}", xs.len()),
                        ),
                    };
                    // TODO: Validate that `targ` refers to a "valid in-scope class or TypeVar"
                    // (https://typing.readthedocs.io/en/latest/spec/annotations.html#type-and-annotation-expressions)
                    self.heap.mk_type_form(self.heap.mk_type_form(targ))
                }
                // TODO: pyre_extensions.PyreReadOnly is a non-standard type system extension that marks read-only
                // objects. We don't support it yet.
                Type::ClassDef(cls)
                    if cls.has_toplevel_qname("pyre_extensions", "PyreReadOnly")
                        || cls.has_toplevel_qname("pyre_extensions", "ReadOnly") =>
                {
                    match xs.len() {
                        1 => self.expr_infer(&xs[0], errors),
                        _ => self.error(
                            errors,
                            range,
                            ErrorInfo::Kind(ErrorKind::BadSpecialization),
                            format!(
                                "Expected 1 type argument for `PyreReadOnly`, got {}",
                                xs.len()
                            ),
                        ),
                    }
                }
                // Tensor type parsing: Tensor[2, 3] syntax
                Type::ClassDef(ref cls) if self.is_tensor_class(cls) => {
                    Type::type_form(self.parse_tensor_type(cls, xs, errors))
                }
                // Dim type parsing: Dim[3], Dim[N], Dim[N+1] syntax
                Type::ClassDef(ref cls) if self.is_symint_class(cls) => {
                    self.parse_symint_type(xs, range, errors)
                }
                Type::ClassDef(ref cls)
                    if let Expr::StringLiteral(ExprStringLiteral { value: key, .. }) = slice
                        && self.get_enum_from_class(cls).is_some() =>
                {
                    if let Some(member) = self.get_enum_member(cls, &Name::new(key.to_str())) {
                        member.to_implicit_type()
                    } else {
                        self.error(
                            errors,
                            slice.range(),
                            ErrorInfo::Kind(ErrorKind::BadIndex),
                            format!(
                                "Enum `{}` does not have a member named `{}`",
                                cls.name(),
                                key.to_str()
                            ),
                        )
                    }
                }
                Type::ClassDef(ref cls) if self.get_enum_from_class(cls).is_some() => {
                    if self.is_subset_eq(
                        &self.expr(slice, None, errors),
                        &self.heap.mk_class_type(self.stdlib.str().clone()),
                    ) {
                        self.heap.mk_class_type(self.as_class_type_unchecked(cls))
                    } else {
                        self.error(
                            errors,
                            slice.range(),
                            ErrorInfo::Kind(ErrorKind::BadIndex),
                            format!("Enum `{}` can only be indexed by strings", cls.name()),
                        )
                    }
                }
                Type::ClassDef(cls) => {
                    let metadata = self.get_metadata_for_class(&cls);
                    let class_ty = Type::ClassDef(cls.dupe());
                    let allow_dunder_lookup = self.get_class_tparams(&cls).is_empty()
                        && !metadata.has_base_any()
                        && !metadata.is_new_type();
                    let class_getitem_result = if allow_dunder_lookup {
                        let class_ty = self.heap.mk_class_def(cls.dupe());
                        // TODO(stroxler): Add a new API, similar to `type_of_attr_get` but returning a
                        // LookupResult or an Optional type, that we could use here to avoid the double lookup.
                        if self.has_attr(&class_ty, &dunder::CLASS_GETITEM) {
                            let cls_value = self.promote_silently(&cls);
                            let call_args = [CallArg::ty(&cls_value, range), CallArg::expr(slice)];
                            Some(self.call_method_or_error(
                                &class_ty,
                                &dunder::CLASS_GETITEM,
                                range,
                                &call_args,
                                &[],
                                errors,
                                Some(&|| ErrorContext::Index(self.for_display(class_ty.clone()))),
                            ))
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    let metaclass_getitem_result =
                        if class_getitem_result.is_none() && allow_dunder_lookup {
                            self.call_magic_dunder_method(
                                &class_ty,
                                &dunder::GETITEM,
                                range,
                                &[CallArg::expr(slice)],
                                &[],
                                errors,
                                Some(&|| ErrorContext::Index(self.for_display(class_ty.clone()))),
                            )
                        } else {
                            None
                        };
                    if let Some(result) = class_getitem_result.or(metaclass_getitem_result) {
                        result
                    } else {
                        self.heap.mk_type_form(self.specialize(
                            &cls,
                            xs.map(|x| self.expr_untype(x, TypeFormContext::TypeArgument, errors)),
                            range,
                            errors,
                        ))
                    }
                }
                Type::Type(box Type::Quantified(quantified)) if quantified.is_type_var() => {
                    let quantified = *quantified;
                    let base_display_ty =
                        self.heap.mk_type(self.heap.mk_quantified(quantified.clone()));
                    if self.is_restricted_to_enum_class_def_type(&quantified) {
                        if self.is_subset_eq(
                            &self.expr(slice, None, errors),
                            &self.heap.mk_class_type(self.stdlib.str().clone()),
                        ) {
                            quantified.to_type(self.heap)
                        } else {
                            self.error(
                                errors,
                                slice.range(),
                                ErrorInfo::Kind(ErrorKind::BadIndex),
                                format!(
                                    "Enum type `{}` can only be indexed by strings",
                                    self.for_display(base_display_ty)
                                ),
                            )
                        }
                    } else {
                        self.error(
                            errors,
                            range,
                            ErrorInfo::Kind(ErrorKind::UnsupportedOperation),
                            format!(
                                "`{}` is not subscriptable",
                                self.for_display(base_display_ty)
                            ),
                        )
                    }
                }
                Type::Type(inner) if self.is_enum_class_type(inner.as_ref()) => {
                    let base_display_ty = self.heap.mk_type_form((*inner).clone());
                    let enum_value_ty = *inner;
                    if self.is_subset_eq(
                        &self.expr(slice, None, errors),
                        &self.heap.mk_class_type(self.stdlib.str().clone()),
                    ) {
                        enum_value_ty
                    } else {
                        self.error(
                            errors,
                            slice.range(),
                            ErrorInfo::Kind(ErrorKind::BadIndex),
                            format!(
                                "Enum type `{}` can only be indexed by strings",
                                self.for_display(base_display_ty)
                            ),
                        )
                    }
                }
                Type::Type(box Type::SpecialForm(special)) => {
                    self.apply_special_form(special, slice, range, errors)
                }
                Type::Tuple(ref tuple) => self.infer_tuple_subscript(
                    tuple.clone(),
                    slice,
                    range,
                    errors,
                    Some(&|| ErrorContext::Index(self.for_display(base.clone()))),
                ),
                Type::Any(style) => style.propagate(),
                Type::Literal(ref lit) if let Lit::Bytes(ref bytes) = lit.value => self.subscript_bytes_literal(
                    bytes,
                    slice,
                    errors,
                    range,
                    Some(&|| ErrorContext::Index(self.for_display(base.clone()))),
                ),
                Type::LiteralString(_) if xs.len() <= 3 => {
                    // We could have a more precise type here, but this matches Pyright.
                    self.heap.mk_class_type(self.stdlib.str().clone())
                }
                Type::Literal(ref lit) if let Lit::Str(ref value) = lit.value && xs.len() <= 3 => {
                    let base_ty = Lit::Str(value.clone()).to_implicit_type();
                    let context = || ErrorContext::Index(self.for_display(base_ty.clone()));
                    self.subscript_str_literal(
                        value.as_str(),
                        &base_ty,
                        slice,
                        errors,
                        range,
                        Some(&context),
                    )
                }
                Type::Args(_) => {
                    let tuple = Tuple::Unbounded(Box::new(
                        self.heap.mk_class_type(self.stdlib.object().clone()),
                    ));
                    self.infer_tuple_subscript(
                        tuple,
                        slice,
                        range,
                        errors,
                        Some(&|| ErrorContext::Index(self.for_display(base.clone()))),
                    )
                }
                Type::Kwargs(_) => {
                    let kwargs_ty = self.heap.mk_class_type(self.stdlib.dict(
                        self.heap.mk_class_type(self.stdlib.str().clone()),
                        self.heap.mk_class_type(self.stdlib.object().clone()),
                    ));
                    self.call_method_or_error(
                        &kwargs_ty,
                        &dunder::GETITEM,
                        range,
                        &[CallArg::expr(slice)],
                        &[],
                        errors,
                        Some(&|| ErrorContext::Index(self.for_display(base.clone()))),
                    )
                }
                // Tensor indexing: tensor[0] reduces dimensionality
                Type::Tensor(ref tensor_type) => {
                    self.infer_tensor_index(tensor_type, slice, range, errors)
                }
                // Shapeless tensor as ClassType: use tensor indexing logic
                // e.g., x: Tensor then x[0] should still work
                Type::ClassType(ref cls) if self.is_tensor_class(cls.class_object()) => {
                    // Extract shape dimensions from the ClassType's type arguments
                    // E.g., Tensor[10, 20] has targs [10, 20]
                    let targs = cls.targs().as_slice();

                    match targs {
                        [] | [Type::Tuple(Tuple::Unbounded(box Type::Any(_)))] => {
                            // Shapeless tensor class - create shapeless TensorType and use tensor indexing
                            let tensor_type = TensorType::shapeless(cls.clone());
                            self.infer_tensor_index(&tensor_type, slice, range, errors)
                        }
                        _ => {
                            // Build TensorShape from type arguments
                            let shape_dims: Vec<Type> = targs.to_vec();
                            let tensor_shape = TensorShape::from_types(shape_dims);

                            // Create TensorType with the class as base_class
                            let tensor_type = TensorType::new(cls.clone(), tensor_shape);
                            self.infer_tensor_index(&tensor_type, slice, range, errors)
                        }
                    }
                }
                Type::ClassType(ref cls) | Type::SelfType(ref cls)
                    if let Some(tuple) = self.as_tuple(cls) =>
                {
                    self.infer_tuple_subscript(
                        tuple,
                        slice,
                        range,
                        errors,
                        Some(&|| ErrorContext::Index(self.for_display(base.clone()))),
                    )
                }
                // Special handling for nn.ModuleDict with TypedDict type argument
                Type::ClassType(ref cls) if is_nn_module_dict(cls) => {
                    self.try_nn_module_dict_index(cls, &base, slice, range, errors)
                }
                Type::ClassType(_) | Type::SelfType(_) => self.call_method_or_error(
                    &base,
                    &dunder::GETITEM,
                    range,
                    &[CallArg::expr(slice)],
                    &[],
                    errors,
                    Some(&|| ErrorContext::Index(self.for_display(base.clone()))),
                ),
                Type::Quantified(ref q) if q.is_type_var() && q.restriction().is_restricted() => {
                    self.call_method_or_error(
                        &base,
                        &dunder::GETITEM,
                        range,
                        &[CallArg::expr(slice)],
                        &[],
                        errors,
                        Some(&|| ErrorContext::Index(self.for_display(base.clone()))),
                    )
                }
                Type::TypedDict(typed_dict) => {
                    let key_ty = self.expr_infer(slice, errors);
                    // Don't warn on anonymous typed dicts
                    let warn_on_not_required_access = matches!(typed_dict, TypedDict::TypedDict(_));
                    self.distribute_over_union(&key_ty, |ty| match ty {
                        Type::Literal(lit) if let Lit::Str(field_name) = &lit.value => {
                            let fields = self.typed_dict_fields(&typed_dict);
                            let key_name = Name::new(field_name);
                            if let Some(field) = fields.get(&key_name) {
                                if warn_on_not_required_access && !field.required {
                                    errors.add(
                                        slice.range(),
                                        ErrorInfo::Kind(ErrorKind::NotRequiredKeyAccess),
                                        vec1![format!(
                                            "TypedDict key `{}` may be absent",
                                            key_name
                                        ),
                                        format!(
                                            "Hint: guard this access with `'{}' in obj` or `obj.get('{}')`",
                                            key_name, key_name
                                        )],
                                    );
                                }
                                field.ty.clone()
                            } else if let ExtraItems::Extra(extra) =
                                self.typed_dict_extra_items(&typed_dict)
                            {
                                extra.ty
                            } else {
                                let mut msg = vec1![format!(
                                    "TypedDict `{}` does not have key `{}`",
                                    typed_dict.name(),
                                    field_name
                                )];
                                if let Some(suggestion) = best_suggestion(
                                    &key_name,
                                    fields.keys().map(|candidate| (candidate, 0usize)),
                                ) {
                                    msg.push(format!("Did you mean `{suggestion}`?"));
                                }
                                errors.add(
                                    slice.range(),
                                    ErrorInfo::Kind(ErrorKind::BadTypedDictKey),
                                    msg,
                                );
                                self.heap.mk_any_error()
                            }
                        }
                        _ => {
                            if self.is_subset_eq(
                                ty,
                                &self.heap.mk_class_type(self.stdlib.str().clone()),
                            )
                                && !matches!(
                                    self.typed_dict_extra_items(&typed_dict),
                                    ExtraItems::Default
                                )
                            {
                                self.get_typed_dict_value_type(&typed_dict)
                            } else {
                                self.error(
                                    errors,
                                    slice.range(),
                                    ErrorInfo::Kind(ErrorKind::BadTypedDictKey),
                                    format!(
                                        "Invalid key for TypedDict `{}`, got `{}`",
                                        typed_dict.name(),
                                        self.for_display(ty.clone())
                                    ),
                                )
                            }
                        }
                    })
                }
                Type::UntypedAlias(ta) => self.subscript_infer_for_type(&self.untype_alias(&ta), slice, range, errors),
                t => self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::UnsupportedOperation),
                    format!("`{}` is not subscriptable", self.for_display(t)),
                ),
            }
        })
    }

    /// Handle tensor indexing operations
    /// - Integer index: reduces dimensionality by 1 (removes first dimension)
    /// - Slice: preserves dimensionality (keeps all dimensions)
    fn infer_tensor_index(
        &self,
        tensor_type: &TensorType,
        index: &Expr,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        // Convert a slice bound expression to a dimension type.
        let to_dim = |expr: &Expr| -> Type {
            let ty = self.expr_infer(expr, errors);
            match ty {
                Type::Literal(ref lit) if let Some(val) = lit.value.as_index_i64() => {
                    self.heap.mk_size(SizeExpr::Literal(val))
                }
                Type::Dim(ref inner_ty) => (**inner_ty).clone(),
                Type::Quantified(_) | Type::Size(_) => ty.clone(),
                _ => Type::any_implicit(),
            }
        };

        // Classify a non-slice, non-ellipsis index expression into an IndexOp.
        // Returns None to bail to shapeless for unclassifiable indices.
        let classify_index_expr = |expr: &Expr| -> Option<IndexOp> {
            let idx_ty = self.expr_infer(expr, errors);
            if let Type::Tuple(ref tuple) = idx_ty {
                return match tuple {
                    Tuple::Concrete(elems) => Some(IndexOp::Fancy(Some(elems.len() as i64))),
                    _ => None,
                };
            }
            if let Type::ClassType(ref cls) = idx_ty
                && cls.has_qname("builtins", "list")
            {
                return Some(IndexOp::Fancy(None));
            }
            let is_int = matches!(&idx_ty, Type::Literal(lit) if lit.value.as_index_i64().is_some())
                || matches!(&idx_ty, Type::ClassType(cls) if cls.is_builtin("int"))
                || matches!(&idx_ty, Type::Dim(_));
            if is_int { Some(IndexOp::Int) } else { None }
        };

        // Classify any index expression (including slices) into an IndexOp.
        let classify = |expr: &Expr| -> Option<IndexOp> {
            match expr {
                Expr::Slice(ExprSlice { lower, upper, .. }) => {
                    let start = lower.as_ref().map(|e| to_dim(e));
                    let stop = upper.as_ref().map(|e| to_dim(e));
                    Some(IndexOp::Slice { start, stop })
                }
                _ => classify_index_expr(expr),
            }
        };

        match index {
            // Slice operation: tensor[start:stop]
            Expr::Slice(ExprSlice { lower, upper, .. }) => {
                let start = lower.as_ref().map(|e| to_dim(e));
                let stop = upper.as_ref().map(|e| to_dim(e));
                match index_shape_slice(&tensor_type.shape, start, stop) {
                    Ok(shape) => TensorType::new(tensor_type.base_class.clone(), shape).to_type(),
                    Err(err) => self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::BadIndex),
                        err.to_string(),
                    ),
                }
            }
            // Bare ellipsis: tensor[...] - preserves entire shape
            Expr::EllipsisLiteral(_) => tensor_type.clone().to_type(),
            // Tuple index: tensor[:, -1, :] - apply each index to corresponding dimension
            Expr::Tuple(ExprTuple { elts, .. }) => {
                // Check for ellipsis and validate at most one
                let mut ellipsis_pos: Option<usize> = None;
                for (i, elt) in elts.iter().enumerate() {
                    if matches!(elt, Expr::EllipsisLiteral(_)) {
                        if ellipsis_pos.is_some() {
                            return self.error(
                                errors,
                                range,
                                ErrorInfo::Kind(ErrorKind::BadIndex),
                                "Multiple ellipsis not allowed in tensor index".to_owned(),
                            );
                        }
                        ellipsis_pos = Some(i);
                    }
                }

                // Split indices at ellipsis into pre and post groups
                let (pre_exprs, post_exprs) = match ellipsis_pos {
                    Some(pos) => (&elts[..pos], &elts[pos + 1..]),
                    None => (&elts[..], &elts[0..0]),
                };

                // Classify all index expressions into IndexOps
                let pre_ops: Option<Vec<IndexOp>> = pre_exprs.iter().map(&classify).collect();
                let post_ops: Option<Vec<IndexOp>> = post_exprs.iter().map(classify).collect();
                let (Some(pre_ops), Some(post_ops)) = (pre_ops, post_ops) else {
                    return TensorType::shapeless(tensor_type.base_class.clone()).to_type();
                };

                match index_shape_multi(
                    &tensor_type.shape,
                    &pre_ops,
                    &post_ops,
                    ellipsis_pos.is_some(),
                ) {
                    Ok(shape) => TensorType::new(tensor_type.base_class.clone(), shape).to_type(),
                    Err(err) => self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::BadIndex),
                        err.to_string(),
                    ),
                }
            }
            // Integer index, tensor index, or other
            _ => {
                let idx_type = self.expr_infer(index, errors);
                let is_int_index = matches!(&idx_type, Type::Literal(lit) if lit.value.as_index_i64().is_some())
                    || matches!(&idx_type, Type::ClassType(cls) if cls.is_builtin("int"));

                if is_int_index {
                    match index_shape_int(&tensor_type.shape) {
                        Ok(shape) => {
                            TensorType::new(tensor_type.base_class.clone(), shape).to_type()
                        }
                        Err(err) => self.error(
                            errors,
                            range,
                            ErrorInfo::Kind(ErrorKind::BadIndex),
                            err.to_string(),
                        ),
                    }
                } else if let Type::Tensor(ref idx_tensor) = idx_type {
                    // Tensor indexing: tensor[index_tensor] replaces first dim with index shape
                    let TensorShape::Concrete(idx_dims) = &idx_tensor.shape else {
                        return TensorType::shapeless(tensor_type.base_class.clone()).to_type();
                    };
                    match index_shape_tensor(&tensor_type.shape, idx_dims) {
                        Ok(shape) => {
                            TensorType::new(tensor_type.base_class.clone(), shape).to_type()
                        }
                        Err(err) => self.error(
                            errors,
                            range,
                            ErrorInfo::Kind(ErrorKind::BadIndex),
                            err.to_string(),
                        ),
                    }
                } else {
                    // Unknown index type - return shapeless
                    TensorType::shapeless(tensor_type.base_class.clone()).to_type()
                }
            }
        }
    }

    /// Check if a class is a tensor class (torch.Tensor)
    fn is_tensor_class(&self, cls: &Class) -> bool {
        cls.has_toplevel_qname("torch", "Tensor")
    }

    /// Check if a class is a Dim class (torch_shapes.Dim)
    fn is_symint_class(&self, cls: &Class) -> bool {
        cls.has_toplevel_qname("torch_shapes", "Dim")
    }

    /// Parse a single dimension expression (recursive helper)
    fn parse_dimension_expr(&self, expr: &Expr, errors: &ErrorCollector) -> Option<Type> {
        match expr {
            // String literals are not valid dimensions
            Expr::StringLiteral(_) => {
                self.error(
                    errors,
                    expr.range(),
                    ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                    "String literals are not valid tensor dimensions".to_owned(),
                );
                None
            }
            // Number literal: concrete dimension
            Expr::NumberLiteral(ExprNumberLiteral { value, .. }) => match value {
                Number::Int(int_val) => {
                    if let Some(value) = int_val.as_i64() {
                        // Allow any integer value during parsing - validation happens later
                        // This allows expressions like N + 0 where 0 is part of an expression
                        Some(self.heap.mk_size(SizeExpr::literal(value)))
                    } else {
                        self.error(
                            errors,
                            expr.range(),
                            ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                            "Tensor shape dimension too large".to_owned(),
                        );
                        None
                    }
                }
                _ => {
                    self.error(
                        errors,
                        expr.range(),
                        ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                        "Tensor shape dimensions must be integers, not floats or complex numbers"
                            .to_owned(),
                    );
                    None
                }
            },
            // Name expression: could be a type variable
            Expr::Name(_) => {
                let expr_type = self.expr_infer(expr, errors);

                match &expr_type {
                    Type::QuantifiedValue(q) => Some(Type::Quantified(q.clone())),
                    Type::ClassDef(cls) if cls.has_toplevel_qname("typing", "Any") => {
                        // typing.Any in a type annotation position (e.g., Tensor[16, Any])
                        // Use Explicit since the user wrote Any explicitly
                        Some(Type::Any(AnyStyle::Explicit))
                    }
                    _ => {
                        self.error(
                            errors,
                            expr.range(),
                            ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                            format!(
                                "Tensor shape dimensions must be integer literals or type variables, got `{}`",
                                self.for_display(expr_type)
                            ),
                        );
                        None
                    }
                }
            }
            // Binary operations: N + M, N * M, etc.
            Expr::BinOp(ExprBinOp {
                left, op, right, ..
            }) => {
                let left_dim = self.parse_dimension_expr(left, errors)?;
                let right_dim = self.parse_dimension_expr(right, errors)?;

                match op {
                    Operator::Add => Some(self.heap.mk_size(SizeExpr::add(left_dim, right_dim))),
                    Operator::Sub => Some(self.heap.mk_size(SizeExpr::sub(left_dim, right_dim))),
                    Operator::Mult => Some(self.heap.mk_size(SizeExpr::mul(left_dim, right_dim))),
                    Operator::FloorDiv => {
                        Some(self.heap.mk_size(SizeExpr::floor_div(left_dim, right_dim)))
                    }
                    _ => {
                        self.error(
                            errors,
                            expr.range(),
                            ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                            format!(
                                "Unsupported operator `{}` in tensor shape dimension",
                                op.as_str()
                            ),
                        );
                        None
                    }
                }
            }
            // Anything else is an error
            _ => {
                let expr_type = self.expr_infer(expr, errors);
                self.error(
                    errors,
                    expr.range(),
                    ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                    format!(
                        "Tensor shape dimensions must be positive integer literals, string literals, type variables, or expressions, got `{}`",
                        self.for_display(expr_type)
                    ),
                );
                None
            }
        }
    }

    /// Parse a list of dimension expressions, simplifying and validating each one.
    /// Returns None if any dimension fails to parse or is non-positive.
    fn parse_dimension_list(&self, args: &[Expr], errors: &ErrorCollector) -> Option<Vec<Type>> {
        let mut dims = Vec::new();
        for arg in args {
            if let Some(dim) = self.parse_dimension_expr(arg, errors) {
                let simplified = canonicalize(dim);

                // Validate that literal dimensions are positive
                if let Type::Size(SizeExpr::Literal(value)) = &simplified
                    && value <= &0
                {
                    self.error(
                        errors,
                        arg.range(),
                        ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                        format!("Tensor shape dimension must be positive, got {}", value),
                    );
                    return None;
                }

                dims.push(simplified);
            } else {
                return None;
            }
        }
        Some(dims)
    }

    /// Check if a type is a valid TypeVarTuple (either directly or wrapped in Unpack).
    /// Returns the unwrapped type if valid, None otherwise.
    fn unwrap_type_var_tuple(ty: &Type) -> Option<Type> {
        match ty {
            Type::TypeVarTuple(_) => Some(ty.clone()),
            Type::Quantified(q) if q.kind() == QuantifiedKind::TypeVarTuple => Some(ty.clone()),
            Type::Unpack(inner) => Self::unwrap_type_var_tuple(inner),
            _ => None,
        }
    }

    /// Parse Tensor[2, 3] or Tensor["batch", 2, 3] or Tensor[N + M, K] or Tensor[2, *Shape, 4] into a TensorType
    fn parse_tensor_type(&self, cls: &Class, shape_args: &[Expr], errors: &ErrorCollector) -> Type {
        // Check if any argument is a starred expression (unpacked TypeVarTuple)
        let star = shape_args
            .iter()
            .enumerate()
            .find(|(_, arg)| matches!(arg, Expr::Starred(_)));

        if let Some((star_idx, Expr::Starred(ExprStarred { value, .. }))) = star {
            // Handle variadic shape: Tensor[2, *Shape, 4]
            // Verify there's only one starred expression
            if let Some(second) = shape_args[star_idx + 1..]
                .iter()
                .find(|arg| matches!(arg, Expr::Starred(_)))
            {
                self.error(
                    errors,
                    second.range(),
                    ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                    "Tensor shape can have at most one unpacked TypeVarTuple".to_owned(),
                );
                return Type::any_error();
            }

            // Parse prefix and suffix dimensions
            let Some(prefix) = self.parse_dimension_list(&shape_args[..star_idx], errors) else {
                return Type::any_error();
            };
            let Some(suffix) = self.parse_dimension_list(&shape_args[star_idx + 1..], errors)
            else {
                return Type::any_error();
            };

            // Parse the starred expression
            let middle_ty = self.expr_untype(value, TypeFormContext::TypeArgument, errors);

            // Verify and unwrap TypeVarTuple
            let Some(middle_ty) = Self::unwrap_type_var_tuple(&middle_ty) else {
                self.error(
                    errors,
                    value.range(),
                    ErrorInfo::Kind(ErrorKind::InvalidAnnotation),
                    format!(
                        "Unpacked type in Tensor shape must be a TypeVarTuple, got `{}`",
                        self.for_display(middle_ty)
                    ),
                );
                return Type::any_error();
            };

            // Create the base class type
            let base_class = self.promote_nontypeddict_silently_to_classtype(cls);

            // Create variadic tensor shape
            let tensor_shape = TensorShape::unpacked(prefix, middle_ty, suffix);
            let tensor_type = TensorType::new(base_class, tensor_shape);

            return tensor_type.to_type();
        }

        // No starred expression - parse as concrete shape
        let Some(dims) = self.parse_dimension_list(shape_args, errors) else {
            return Type::any_error();
        };

        // Create the base class type (with default type arguments if needed)
        let base_class = self.promote_nontypeddict_silently_to_classtype(cls);

        // Create the tensor type
        let tensor_shape = TensorShape::from_types(dims);
        let tensor_type = TensorType::new(base_class, tensor_shape);

        tensor_type.to_type()
    }

    /// Parse Dim[3], Dim[N], Dim[N+1] into Type::Dim(...)
    fn parse_symint_type(&self, args: &[Expr], range: TextRange, errors: &ErrorCollector) -> Type {
        // Dim takes exactly one argument
        if args.len() != 1 {
            self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::BadSpecialization),
                format!("Expected 1 type argument for `Dim`, got {}", args.len()),
            );
            return Type::any_error();
        }

        // Parse, simplify, and validate the dimension
        let Some(dims) = self.parse_dimension_list(args, errors) else {
            return Type::any_error();
        };

        // Wrap in Type::Dim(...)
        self.heap
            .mk_type_form(self.heap.mk_dim(dims.into_iter().next().unwrap()))
    }

    /// Return the reason why we think `ty` is suspicious to use as a branching condition
    fn get_condition_redundant_reason(&self, ty: &Type) -> Option<ConditionRedundantReason> {
        match ty {
            Type::Literal(lit) if let Lit::Bool(_) = lit.value => None,
            Type::Literal(lit) if let Lit::Int(i) = &lit.value => {
                Some(ConditionRedundantReason::IntLiteral(i.as_bool()))
            }
            Type::Literal(lit) if let Lit::Str(s) = &lit.value => {
                Some(ConditionRedundantReason::StrLiteral(!s.is_empty()))
            }
            Type::Literal(lit) if let Lit::Bytes(s) = &lit.value => {
                Some(ConditionRedundantReason::BytesLiteral(!s.is_empty()))
            }
            Type::Literal(lit) if let Lit::Enum(e) = &lit.value => {
                Some(ConditionRedundantReason::EnumLiteral(
                    e.class.class_object().name().clone(),
                    e.member.clone(),
                ))
            }
            Type::Function(f) => Some(ConditionRedundantReason::Function(
                self.module().name(),
                f.metadata.kind.clone(),
            )),
            Type::Overload(f) => Some(ConditionRedundantReason::Function(
                self.module().name(),
                f.metadata.kind.clone(),
            )),
            Type::BoundMethod(f) => Some(ConditionRedundantReason::Function(
                self.module().name(),
                f.func.metadata().kind.clone(),
            )),
            Type::ClassDef(cls) => Some(ConditionRedundantReason::Class(cls.name().clone())),
            _ => None,
        }
    }

    pub fn check_redundant_condition(
        &self,
        condition_type: &Type,
        range: TextRange,
        errors: &ErrorCollector,
    ) {
        if let Some(reason) = self.get_condition_redundant_reason(condition_type) {
            self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::RedundantCondition),
                format!("{reason}"),
            );
        }
    }
}
