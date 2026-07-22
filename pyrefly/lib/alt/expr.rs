/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cell::LazyCell;
use std::fmt;
use std::fmt::Display;
use std::slice;

use dupe::Dupe;
use itertools::Either;
use itertools::Itertools;
use pyrefly_python::ast::Ast;
use pyrefly_python::dunder;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::nesting_context::NestingContext;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::callable::FunctionKind;
use pyrefly_types::data_frame::DataFrameKind;
use pyrefly_types::data_frame::SchemaCompleteness;
use pyrefly_types::dimension::Int;
use pyrefly_types::dimension::canonicalize;
use pyrefly_types::dimension::gradual_size;
use pyrefly_types::dimension::int_type_is_provably_negative;
use pyrefly_types::literal::LitStyle;
use pyrefly_types::shaped_array::IndexOp;
use pyrefly_types::shaped_array::IntTuple;
use pyrefly_types::shaped_array::IntTupleView;
use pyrefly_types::shaped_array::ShapedArrayType;
use pyrefly_types::shaped_array::index_shape_int;
use pyrefly_types::shaped_array::index_shape_multi;
use pyrefly_types::shaped_array::index_shape_slice;
use pyrefly_types::shaped_array::index_shape_tensor;
use pyrefly_types::shaped_array::shape_to_tuple_carrier;
use pyrefly_types::shaped_array::tuple_carrier_to_shape;
use pyrefly_types::shaped_array::type_to_dim;
use pyrefly_types::typed_dict::AnonymousTypedDictInner;
use pyrefly_types::typed_dict::ExtraItems;
use pyrefly_types::typed_dict::TypedDict;
use pyrefly_types::typed_dict::TypedDictField;
use pyrefly_types::types::Forallable;
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
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprBinOp;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprGenerator;
use ruff_python_ast::ExprList;
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
use ruff_python_ast::UnaryOp;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::Hashed;
use starlark_map::small_map::SmallMap;
use vec1::Vec1;
use vec1::vec1;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::answers_solver::TypeCheckOptions;
use crate::alt::callable::CallArg;
use crate::alt::class::typed_dict::TypedDictErrorKind;
use crate::alt::nn_module_specials::is_nn_module_dict;
use crate::alt::solve::TypeFormContext;
use crate::alt::solve::UntypeContext;
use crate::alt::unwrap::HintRef;
use crate::binding::binding::Binding;
use crate::binding::binding::Key;
use crate::binding::binding::KeyYield;
use crate::binding::binding::KeyYieldFrom;
use crate::binding::binding::LambdaParamId;
use crate::binding::narrow::AtomicNarrowOp;
use crate::binding::narrow::int_from_slice;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorContext;
use crate::error::context::TypeCheckContext;
use crate::solver::solver::CallContext;
use crate::types::callable::DefaultValue;
use crate::types::callable::Param;
use crate::types::callable::ParamList;
use crate::types::callable::Params;
use crate::types::callable::Required;
use crate::types::class::Class;
use crate::types::class::ClassType;
use crate::types::facet::FacetKind;
use crate::types::literal::Lit;
use crate::types::param_spec::ParamSpec;
use crate::types::quantified::Quantified;
use crate::types::quantified::QuantifiedKind;
use crate::types::sentinel::Sentinel;
use crate::types::special_form::SpecialForm;
use crate::types::tuple::Tuple;
use crate::types::type_info::TypeInfo;
use crate::types::type_var::PreInferenceVariance;
use crate::types::type_var::Restriction;
use crate::types::type_var::TypeVar;
use crate::types::type_var_tuple::TypeVarTuple;
use crate::types::types::AnyStyle;
use crate::types::types::Type;
use crate::types::types::Var;

#[derive(Debug, Clone, Copy)]
pub enum TypeOrExpr<'a> {
    /// Bundles a `Type` with a `TextRange`, allowing us to give good errors.
    Type(&'a Type, TextRange),
    Expr(&'a Expr),
}

/// Where a dimension expression appears, which controls whether a plain
/// `TypeVar` is accepted. Shape arithmetic (e.g. `N + 1`) needs the
/// symbolic-integer semantics of an `IntVar`, so an operand of an arithmetic
/// expression must be an `IntVar`; a dimension used on its own accepts any type
/// variable kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DimensionExprContext {
    /// A dimension written directly, e.g. the `N` in `Tensor[N, 3]`. Any type
    /// variable kind is allowed.
    Bare,
    /// An operand of shape arithmetic, e.g. the `N` in `Tensor[N + 1]`. Only a
    /// `IntVar` is allowed; a plain `TypeVar` is rejected.
    Arithmetic,
}

impl DimensionExprContext {
    fn error_context(self) -> &'static str {
        match self {
            Self::Bare => "as a shape dimension",
            Self::Arithmetic => "in shape arithmetic",
        }
    }
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

pub struct ExprOptions<'a, 'b> {
    errors: &'a ErrorCollector,
    expectation: ExprExpectation<'a, 'b>,
}

enum ExprExpectation<'a, 'b> {
    Infer(Option<HintRef<'a, 'b>>),
    Check {
        want: &'b Type,
        errors: &'a ErrorCollector,
        context: &'a dyn Fn() -> TypeCheckContext,
        call_context: Option<&'a CallContext>,
    },
}

impl<'a, 'b> ExprOptions<'a, 'b> {
    pub fn infer(errors: &'a ErrorCollector, hint: Option<HintRef<'a, 'b>>) -> Self {
        Self {
            errors,
            expectation: ExprExpectation::Infer(hint),
        }
    }

    pub fn check(
        want: &'b Type,
        errors: &'a ErrorCollector,
        check_errors: &'a ErrorCollector,
        context: &'a dyn Fn() -> TypeCheckContext,
        call_context: Option<&'a CallContext>,
    ) -> Self {
        Self {
            errors,
            expectation: ExprExpectation::Check {
                want,
                errors: check_errors,
                context,
                call_context,
            },
        }
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
    /// Instance of a class that defines neither `__bool__` nor `__len__`, so always truthy
    InstanceAlwaysTruthy(Name),
}

impl ConditionRedundantReason {
    fn equivalent_boolean(&self) -> Option<bool> {
        match self {
            ConditionRedundantReason::Function(..)
            | ConditionRedundantReason::Class(..)
            | ConditionRedundantReason::InstanceAlwaysTruthy(..) => Some(true),
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
            ConditionRedundantReason::InstanceAlwaysTruthy(name) => {
                format!("Instance of `{name}` used as condition")
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

pub(crate) const MAX_TUPLE_LENGTH: usize = 256;

fn is_integer_index_scalar_type(ty: &Type) -> bool {
    match ty {
        Type::Literal(lit) => matches!(lit.value, Lit::Int(_)),
        Type::ClassType(cls) => cls.is_builtin("int"),
        Type::Int(_) => true,
        Type::Union(union) => {
            !union.members.is_empty() && union.members.iter().all(is_integer_index_scalar_type)
        }
        _ => false,
    }
}

fn classify_shaped_array_index_type(ty: &Type) -> Option<IndexOp> {
    match ty {
        Type::None => Some(IndexOp::NewAxis),
        Type::ShapedArray(index) => {
            let shape_index = index.tuple_carrier_shape_arg_index()?;
            let targs = index.base_class.targs().as_slice();
            targs
                .get(shape_index)
                .expect("registered shape index must reference a type argument");
            let mut scalar_types = targs
                .iter()
                .enumerate()
                .filter_map(|(i, ty)| (i != shape_index).then_some(ty));
            let scalar_type = scalar_types.next()?;
            if scalar_types.next().is_some() || !is_integer_index_scalar_type(scalar_type) {
                return None;
            }
            index
                .shape()
                .as_concrete()
                .map(|dims| IndexOp::ShapedArrayIndex(dims.to_vec()))
        }
        Type::Tuple(Tuple::Concrete(elements))
            if elements.iter().all(is_integer_index_scalar_type) =>
        {
            Some(IndexOp::Fancy(Int::Literal(elements.len() as i64)))
        }
        Type::Tuple(Tuple::Unbounded(element)) if is_integer_index_scalar_type(element) => {
            Some(IndexOp::Fancy(Int::Int))
        }
        Type::ClassType(cls) if cls.has_qname("builtins", "list") => match cls.targs().as_slice() {
            [element] if is_integer_index_scalar_type(element) => Some(IndexOp::Fancy(Int::Int)),
            _ => None,
        },
        _ if is_integer_index_scalar_type(ty) => Some(IndexOp::Int),
        _ => None,
    }
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    fn synthesized_functional_class_type(&self, call: &ExprCall) -> Option<Type> {
        let anon_key = Key::Anon(call.range);
        let idx = self
            .bindings()
            .key_to_idx_hashed_opt(Hashed::new(&anon_key))?;
        matches!(self.bindings().get(idx), Binding::ClassDef(..))
            .then(|| self.get_hashed(Hashed::new(&anon_key)).ty().clone())
    }

    /// Infer a type for an expression, with an optional type hint that influences the inferred type.
    /// The inferred type is also checked against the hint.
    /// Convenience wrapper around `expr_with_options`.
    pub fn expr_check(
        &self,
        x: &Expr,
        check: Option<(&Type, &dyn Fn() -> TypeCheckContext)>,
        errors: &ErrorCollector,
    ) -> Type {
        let options = match check {
            Some((want, context)) => ExprOptions::check(want, errors, errors, context, None),
            None => ExprOptions::infer(errors, None),
        };
        self.expr_with_options(x, options).into_ty()
    }

    /// Infer a type for an expression. Convenience wrapper around `expr_with_options`.
    pub fn expr_infer(&self, x: &Expr, errors: &ErrorCollector) -> Type {
        self.expr_with_options(x, ExprOptions::infer(errors, None))
            .into_ty()
    }

    /// Infer a type for an expression, with an optional type hint that influences the inferred type.
    /// Convenience wrapper around `expr_with_options`.
    pub fn expr_infer_with_hint(
        &self,
        x: &Expr,
        hint: Option<HintRef>,
        errors: &ErrorCollector,
    ) -> Type {
        self.expr_with_options(x, ExprOptions::infer(errors, hint))
            .into_ty()
    }

    /// Infer a type for an expression, with options to influence the inference and control whether
    /// and how the type is checked against an expected type.
    pub fn expr_with_options(&self, x: &Expr, options: ExprOptions) -> TypeInfo {
        match options.expectation {
            ExprExpectation::Check {
                want,
                errors,
                context,
                call_context,
            } if !want.is_any() => {
                let got =
                    self.expr_infer_impl(x, Some(HintRef::new(want, Some(errors))), options.errors);
                let check_options = match call_context {
                    Some(call_context) => {
                        TypeCheckOptions::new(errors, context).with_call_context(call_context)
                    }
                    None => TypeCheckOptions::new(errors, context),
                };
                if self.check_type_with_options(got.ty(), want, x.range(), check_options) {
                    got
                } else {
                    got.with_ty(want.clone())
                }
            }
            ExprExpectation::Check { .. } => self.expr_infer_impl(x, None, options.errors),
            ExprExpectation::Infer(hint) => self.expr_infer_impl(x, hint, options.errors),
        }
    }

    /// The core logic for inferring a type for an expression.
    /// Returns a TypeInfo that includes narrowing information.
    fn expr_infer_impl(
        &self,
        x: &Expr,
        hint: Option<HintRef>,
        errors: &ErrorCollector,
    ) -> TypeInfo {
        let res = match x {
            Expr::Name(x) => {
                if Ast::is_synthesized_empty_name(x) {
                    TypeInfo::of_ty(self.heap.mk_any_error())
                } else {
                    let result = self
                        .get(&Key::BoundName(ShortIdentifier::expr_name(x)))
                        .arc_clone();
                    // Complements PromoteForward for seeded captures.
                    if self.bindings().should_promote_at_range(x.range) {
                        result.map_ty(|ty| ty.promote_shallow_implicit_literals(self.stdlib))
                    } else {
                        result
                    }
                }
            }
            Expr::Attribute(x) => {
                let base = self.expr_infer_impl(&x.value, None, errors);
                self.attr_access_infer(x, &base, errors)
            }
            Expr::Subscript(x) => {
                // TODO: We don't deal properly with hint here, we should.
                let base = self.expr_infer_impl(&x.value, None, errors);
                self.subscript_infer(&base, &x.slice, x.range(), errors)
            }
            Expr::Named(x) => match &*x.target {
                Expr::Name(name) if !Ast::is_synthesized_empty_name(name) => self
                    .get(&Key::Definition(ShortIdentifier::expr_name(name)))
                    .arc_clone(),
                _ => self.expr_infer_impl(&x.value, hint, errors),
            },
            // All other expressions operate at the `Type` level only, so we avoid the overhead of
            // wrapping and unwrapping `TypeInfo` by computing the result as a `Type` and only wrapping
            // at the end.
            _ => TypeInfo::of_ty(self.expr_infer_impl_helper(x, hint, errors)),
        };
        // Check for deprecation
        self.check_for_deprecated_call(res.ty(), x.range(), errors);
        self.record_type_trace(x.range(), res.ty());
        res
    }

    /// This function should not be used directly: we want every expression to record a type trace,
    /// and that is handled in expr_infer_impl. This function should *only* be called via expr_infer_impl.
    fn expr_infer_impl_helper(
        &self,
        x: &Expr,
        hint: Option<HintRef>,
        errors: &ErrorCollector,
    ) -> Type {
        match x {
            Expr::Name(..) | Expr::Attribute(..) | Expr::Named(..) | Expr::Subscript(..) => {
                // These cases are required to preserve attribute narrowing information. But anyone calling
                // this function only needs the Type, so we can just pull it out.
                self.expr_infer_impl(x, hint, errors).into_ty()
            }
            Expr::If(x) => {
                let condition_type = self.expr_infer(&x.test, errors);
                let body_type = self.expr_infer_impl(&x.body, hint, errors).into_ty();
                let orelse_type = self.expr_infer_impl(&x.orelse, hint, errors).into_ty();
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
                let param_ids = if let Some(parameters) = &lambda.parameters {
                    parameters
                        .iter_non_variadic_params()
                        .map(|x| (&x.name().id, self.bindings().get_lambda_param_id(x.name())))
                        .collect()
                } else {
                    Vec::new()
                };
                let param_default_tys: Vec<Option<Type>> = match &lambda.parameters {
                    Some(parameters) => parameters
                        .iter_non_variadic_params()
                        .map(|p| p.default.as_deref().map(|d| self.expr_infer(d, errors)))
                        .collect(),
                    None => Vec::new(),
                };
                let callable = self.callable_infer_with_hint(
                    hint,
                    errors,
                    |cur_hint, callable_errors| {
                        let param_vars = self.allocate_lambda_param_vars(&param_ids);

                        // Pass any contextual information to the parameter bindings used in the lambda body as a side
                        // effect, by setting an answer for the vars created at binding time.
                        let return_hint =
                            cur_hint.and_then(|hint| self.decompose_lambda(hint, &param_vars));

                        // For each parameter that has a default value but whose Var is not
                        // constrained by a contextual hint, constrain the Var to the
                        // (promoted) type of the default.
                        for ((_, var), default_ty) in param_vars.iter().zip(&param_default_tys) {
                            if let Some(default_ty) = default_ty
                                && matches!(self.solver().expand_unwrap(*var), Type::Var(_))
                            {
                                let mut resolved = default_ty.clone();
                                self.solver().expand_with_bounds(&mut resolved);
                                let promoted = resolved
                                    .with_literal_style(LitStyle::Implicit)
                                    .promote_implicit_literals(self.stdlib);
                                // A `None` default almost always denotes an optional value,
                                // so infer `Any | None` to keep the parameter permissive
                                // rather than strictly `None`.
                                let inferred = if promoted.is_none() {
                                    self.union(self.heap.mk_any_implicit(), promoted)
                                } else {
                                    promoted
                                };
                                let _ = self.is_subset_eq(&inferred, &var.to_type(self.heap));
                            }
                        }

                        let mut params: Vec<Param> = param_vars
                            .into_iter()
                            .zip(&param_default_tys)
                            .map(|((name, var), default_ty)| {
                                let ty = self.solver().force_var(var);
                                let required = match default_ty {
                                    Some(default_ty) => {
                                        Required::Optional(Some(DefaultValue::new(
                                            default_ty
                                                .clone()
                                                .with_literal_style(LitStyle::Explicit),
                                        )))
                                    }
                                    None => Required::Required,
                                };
                                Param::Pos(name.clone(), ty, required)
                            })
                            .collect();
                        if let Some(parameters) = &lambda.parameters {
                            params.extend(parameters.vararg.iter().map(|x| {
                                let var = self.solver().fresh_unwrap(self.uniques);
                                self.set_lambda_param_var(
                                    self.bindings().get_lambda_param_id(&x.name),
                                    var,
                                );
                                Param::Varargs(
                                    Some(x.name.id.clone()),
                                    self.solver().force_var(var),
                                )
                            }));
                            params.extend(parameters.kwarg.iter().map(|x| {
                                let var = self.solver().fresh_unwrap(self.uniques);
                                self.set_lambda_param_var(
                                    self.bindings().get_lambda_param_id(&x.name),
                                    var,
                                );
                                Param::Kwargs(Some(x.name.id.clone()), self.solver().force_var(var))
                            }));
                        }
                        let params = Params::List(ParamList::new(params));
                        if let Some(hint) = cur_hint {
                            // Ensure no param vars are pinned to unfinished Variable::Quantified.
                            // Since lambda parameters are unannotated, the specialization errors can be ignored.
                            let _specialization_errors =
                                self.solver().finish_all_quantified(hint, self.type_order());
                        }
                        let ret = self.expr_infer_impl_helper(
                            &lambda.body,
                            HintRef::with_ty_opt(hint, return_hint.as_ref()),
                            callable_errors,
                        );
                        let (yield_keys, yield_from_keys) =
                            self.bindings().lambda_yield_keys(lambda.range);
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
                    },
                    |callable| callable,
                );
                if let Type::Callable(c) = &callable {
                    let is_implicit_any = |t: &Type| matches!(t, Type::Any(AnyStyle::Implicit));
                    // Collect the AST parameters in the same order the callable's params
                    // were built above (non-variadic, then vararg, then kwarg), so we can
                    // point each error at the specific parameter by name and range.
                    let mut ast_params: Vec<(&Name, TextRange)> = Vec::new();
                    if let Some(parameters) = &lambda.parameters {
                        for p in parameters.iter_non_variadic_params() {
                            ast_params.push((&p.name().id, p.name().range()));
                        }
                        if let Some(vararg) = &parameters.vararg {
                            ast_params.push((&vararg.name.id, vararg.name.range()));
                        }
                        if let Some(kwarg) = &parameters.kwarg {
                            ast_params.push((&kwarg.name.id, kwarg.name.range()));
                        }
                    }
                    if let Params::List(params) = &c.params {
                        for (param, (name, range)) in params.items().iter().zip(&ast_params) {
                            if is_implicit_any(param.as_type()) {
                                self.error(
                                    errors,
                                    *range,
                                    ErrorKind::ImplicitAnyLambda,
                                    format!("Type of lambda parameter `{name}` is unknown"),
                                );
                            }
                        }
                    }
                    if is_implicit_any(&c.ret) {
                        self.error(
                            errors,
                            lambda.body.range(),
                            ErrorKind::ImplicitAnyLambda,
                            "Return type of lambda is unknown".to_owned(),
                        );
                    }
                }
                callable
            }
            Expr::Tuple(x) => self.tuple_infer(x, hint, errors),
            Expr::List(x) => self.infer_with_decomposed_hint(
                hint,
                |hint| self.decompose_list(hint),
                |elt_hint| {
                    if x.is_empty() {
                        let elem_ty = elt_hint.unwrap_or_else(|| {
                            self.solver()
                                .fresh_partial_contained(self.uniques, x.range)
                                .to_type(self.heap)
                        });
                        self.heap.mk_class_type(self.stdlib.list(elem_ty))
                    } else {
                        let elem_tys = self.elts_infer(
                            &x.elts,
                            HintRef::with_ty_opt(hint, elt_hint.as_ref()),
                            errors,
                        );
                        self.heap
                            .mk_class_type(self.stdlib.list(self.unions(elem_tys)))
                    }
                },
            ),
            Expr::Dict(x) => self.dict_infer(&x.items, hint, x.range, errors),
            Expr::Set(x) => self.infer_with_decomposed_hint(
                hint,
                |hint| self.decompose_set(hint),
                |elem_hint| {
                    if x.is_empty() {
                        let elem_ty = elem_hint.unwrap_or_else(|| {
                            self.solver()
                                .fresh_partial_contained(self.uniques, x.range)
                                .to_type(self.heap)
                        });
                        self.heap.mk_class_type(self.stdlib.set(elem_ty))
                    } else {
                        let elem_tys = self.elts_infer(
                            &x.elts,
                            HintRef::with_ty_opt(hint, elem_hint.as_ref()),
                            errors,
                        );
                        self.heap
                            .mk_class_type(self.stdlib.set(self.unions(elem_tys)))
                    }
                },
            ),
            Expr::ListComp(x) => self.infer_with_decomposed_hint(
                hint,
                |hint| self.decompose_list(hint),
                |elem_hint| {
                    self.ifs_infer(&x.generators, errors);
                    let elem_ty = self.expr_infer_with_hint_promote(
                        &x.elt,
                        HintRef::with_ty_opt(hint, elem_hint.as_ref()),
                        errors,
                    );
                    self.heap.mk_class_type(self.stdlib.list(elem_ty))
                },
            ),
            Expr::SetComp(x) => self.infer_with_decomposed_hint(
                hint,
                |hint| self.decompose_set(hint),
                |elem_hint| {
                    self.ifs_infer(&x.generators, errors);
                    let elem_ty = self.expr_infer_with_hint_promote(
                        &x.elt,
                        HintRef::with_ty_opt(hint, elem_hint.as_ref()),
                        errors,
                    );
                    self.heap.mk_class_type(self.stdlib.set(elem_ty))
                },
            ),
            Expr::DictComp(x) => self.infer_with_decomposed_hint(
                hint,
                |hint| {
                    let (key_hint, value_hint) = self.decompose_dict(hint);
                    if key_hint.is_none() && value_hint.is_none() {
                        None
                    } else {
                        Some((key_hint, value_hint))
                    }
                },
                |hints| {
                    let (key_hint, value_hint) = hints.unwrap_or_default();
                    let key_hint = key_hint.as_ref().and_then(|key_hint| {
                        hint.as_ref()
                            .map(|hint| HintRef::new(key_hint, hint.errors()))
                    });
                    let value_hint = value_hint.as_ref().and_then(|value_hint| {
                        hint.as_ref()
                            .map(|hint| HintRef::new(value_hint, hint.errors()))
                    });
                    self.ifs_infer(&x.generators, errors);
                    // `key` is only `None` for a syntactically invalid dict comprehension
                    // (parser error recovery); the parser already reports the syntax error.
                    let key_ty = match &x.key {
                        Some(key) => self.expr_infer_with_hint_promote(key, key_hint, errors),
                        None => self.heap.mk_any_error(),
                    };
                    let value_ty = self.expr_infer_with_hint_promote(&x.value, value_hint, errors);
                    self.heap.mk_class_type(self.stdlib.dict(key_ty, value_ty))
                },
            ),
            Expr::Generator(x) => self.infer_with_decomposed_hint(
                hint,
                |hint| self.decompose_generator(hint).map(|(y, _, _)| y),
                |yield_hint| {
                    self.ifs_infer(&x.generators, errors);
                    let yield_ty = self
                        .expr_infer_impl(
                            &x.elt,
                            HintRef::with_ty_opt(hint, yield_hint.as_ref()),
                            errors,
                        )
                        .into_ty();
                    if self.generator_expr_is_async(x) {
                        self.heap.mk_class_type(
                            self.stdlib.async_generator(yield_ty, self.heap.mk_none()),
                        )
                    } else {
                        let none = self.heap.mk_none();
                        self.heap
                            .mk_class_type(self.stdlib.generator(yield_ty, none.clone(), none))
                    }
                },
            ),
            Expr::Await(x) => {
                let awaiting_ty = self.expr_infer(&x.value, errors);
                self.distribute_over_union(&awaiting_ty, |ty| match self.unwrap_awaitable(ty) {
                    Some(ty) => ty,
                    None => self.error(
                        errors,
                        x.range,
                        ErrorKind::NotAsync,
                        ErrorContext::Await(self.for_display(ty.clone())).format(),
                    ),
                })
            }
            Expr::Yield(x) => self.get(&KeyYield(x.range)).send_ty.clone(),
            Expr::YieldFrom(x) => self.get(&KeyYieldFrom(x.range)).return_ty.clone(),
            Expr::Compare(x) => self.compare_infer(x, errors),
            Expr::Call(x) => {
                if let Some(ty) = self.synthesized_functional_class_type(x) {
                    return ty;
                }
                // Infer a method call's receiver once. A Polars column-algebra transform is a
                // pure function of the receiver schema and is dispatched here; any other
                // receiver is reused for ordinary callee inference rather than inferred again.
                let callee_ty = if let Expr::Attribute(func) = &*x.func {
                    let base = self.expr_infer_impl(&func.value, None, errors);
                    if let Some(ty) = self.polars_select(base.ty(), func, &x.arguments, errors) {
                        return ty;
                    }
                    if let Some(ty) = self.polars_drop(base.ty(), func, &x.arguments, errors) {
                        return ty;
                    }
                    if let Some(ty) = self.polars_rename(base.ty(), func, &x.arguments, errors) {
                        return ty;
                    }
                    if let Some(ty) =
                        self.polars_with_columns(base.ty(), func, &x.arguments, errors)
                    {
                        return ty;
                    }
                    if let Some(ty) =
                        self.polars_row_transform(base.ty(), func, &x.arguments, errors)
                    {
                        return ty;
                    }
                    let attr = self.attr_access_infer(func, &base, errors);
                    // Reusing `base` bypasses `expr_infer_impl`, so record the callee's type trace
                    // and deprecation check here as that path would for any other expression.
                    self.check_for_deprecated_call(attr.ty(), func.range(), errors);
                    self.record_type_trace(func.range(), attr.ty());
                    attr.into_ty()
                } else {
                    self.expr_infer(&x.func, errors)
                };
                self.check_pytorch_tensor_item_call(x, &callee_ty, errors);
                self.check_pytorch_tensor_cuda_call(x, &callee_ty, errors);
                self.check_pytorch_print_tensor(x, &callee_ty, errors);
                self.check_pytorch_redundant_to_call(x, &callee_ty, errors);
                if let Some(d) = self.call_to_dict(&callee_ty, &x.arguments) {
                    self.dict_infer(&d, hint, x.range, errors)
                } else if let Some(ty) = self
                    .anonymous_typed_dict_get_or_setdefault_with_literal(
                        &x.func,
                        &x.arguments,
                        "get",
                        errors,
                    )
                    .or_else(|| {
                        self.anonymous_typed_dict_get_or_setdefault_with_literal(
                            &x.func,
                            &x.arguments,
                            "setdefault",
                            errors,
                        )
                    })
                {
                    ty
                } else if let Some(ty) =
                    self.anonymous_typed_dict_pop_with_literal(&x.func, &x.arguments, errors)
                {
                    ty
                } else if let Some((obj_ty, key_expr, key)) =
                    self.is_dict_get_with_literal(&x.func, &x.arguments, errors)
                {
                    let facet = FacetKind::Key(key.to_string());
                    if obj_ty.has_value_less_presence(&facet) {
                        self.subscript_infer_for_type_with_key_present(
                            obj_ty.ty(),
                            key_expr,
                            x.range,
                            errors,
                            true,
                        )
                    } else {
                        obj_ty
                            .at_facet(&facet, || {
                                self.expr_call_infer(x, callee_ty.clone(), hint, errors)
                            })
                            .into_ty()
                    }
                } else {
                    self.expr_call_infer(x, callee_ty, hint, errors)
                }
            }
            Expr::FString(x) => {
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
                        ErrorKind::InvalidSyntax,
                        "t-strings are only available in Python 3.14+".to_owned(),
                    )
                }
            }
            Expr::StringLiteral(x) => match Lit::from_string_literal(x) {
                Some(lit) => lit.to_implicit_type(),
                None => self.heap.mk_literal_string(LitStyle::Implicit),
            },
            Expr::BytesLiteral(x) => match Lit::from_bytes_literal(x) {
                Some(lit) => lit.to_implicit_type(),
                None => self.heap.mk_class_type(self.stdlib.bytes().clone()),
            },
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
                        ErrorKind::Unsupported,
                        "IPython escapes are not supported".to_owned(),
                    )
                }
            }
        }
    }

    /// Convenience function to call `expr_infer_with_hint` and promote literals in the result
    fn expr_infer_with_hint_promote(
        &self,
        x: &Expr,
        hint: Option<HintRef>,
        errors: &ErrorCollector,
    ) -> Type {
        let ty = self.expr_infer_with_hint(x, hint, errors);
        if let Some(want) = hint {
            // Optimization: delay Type cloning until absolutely necessary.
            if let &[want] = &want.types() {
                if self.is_subset_eq(&ty, want) {
                    return want.clone();
                }
            } else {
                let want = Type::union(want.types().to_vec());
                if self.is_subset_eq(&ty, &want) {
                    return want;
                }
            }
        }
        ty.promote_implicit_literals(self.stdlib)
    }

    /// Check whether a type corresponds to a deprecated function or method, and if so, log a deprecation warning.
    fn check_for_deprecated_call(&self, ty: &Type, range: TextRange, errors: &ErrorCollector) {
        if ty.property_metadata().is_some() {
            // This prevents misfiring deprecation warnings on property setters and deleters.
            return;
        }
        let Some(deprecation) = ty.function_deprecation() else {
            return;
        };
        let deprecated_function = ty
            .to_func_kind()
            .map(|func_kind| func_kind.format(self.module().name()));
        if let Some(deprecated_function) = deprecated_function {
            let header = format!("`{deprecated_function}` is deprecated");
            let detail = deprecation.as_error_detail();
            let mut builder = errors.error_builder(range, ErrorKind::Deprecated, header);
            if let Some(detail) = detail {
                builder = builder.with_detail(detail);
            }
            builder.emit();
        }
    }

    /// Warn when `.item()` is called on a `torch.Tensor`. This forces GPU→CPU
    /// synchronization, stalling the training loop until all pending GPU ops finish.
    fn check_pytorch_tensor_item_call(
        &self,
        x: &ExprCall,
        callee_ty: &Type,
        errors: &ErrorCollector,
    ) {
        let Expr::Attribute(attr_expr) = &*x.func else {
            return;
        };
        if attr_expr.attr.id.as_str() != "item" {
            return;
        }
        if !x.arguments.is_empty() {
            return;
        }
        // Extract the receiver type from the already-resolved BoundMethod
        // rather than re-inferring the base expression.
        if matches!(callee_ty, Type::BoundMethod(bm) if Self::is_pytorch_tensor_type(&bm.obj)) {
            errors
                .error_builder(
                    x.range(),
                    ErrorKind::PytorchEfficiencyLintItemCall,
                    "`Tensor.item()` causes implicit GPU-to-CPU synchronization".to_owned(),
                )
                .with_detail(
                    "This call blocks until all pending GPU operations complete, \
                     which can reduce GPU utilization from >90% to under 50%. \
                     Consider `tensor[0]` for scalar tensors, accumulate values \
                     on the GPU with `torch.sum()`, or defer `.item()` to outside \
                     the training loop."
                        .to_owned(),
                )
                .emit();
        }
    }

    /// Warn when `.cuda()` is called on a `torch.Tensor`. This hard-codes the
    /// target device; `.to(device)` is preferred for device-agnostic code.
    fn check_pytorch_tensor_cuda_call(
        &self,
        x: &ExprCall,
        callee_ty: &Type,
        errors: &ErrorCollector,
    ) {
        let Expr::Attribute(attr_expr) = &*x.func else {
            return;
        };
        if attr_expr.attr.id.as_str() != "cuda" {
            return;
        }
        if !x.arguments.is_empty() {
            return;
        }
        if matches!(callee_ty, Type::BoundMethod(bm) if Self::is_pytorch_tensor_type(&bm.obj)) {
            errors
                .error_builder(
                    x.range(),
                    ErrorKind::PytorchEfficiencyLintCudaCall,
                    "`Tensor.cuda()` hard-codes the target device".to_owned(),
                )
                .with_detail(
                    "Use `.to(device)` instead so your code works on any \
                     accelerator (CUDA, XPU, MPS, etc.). For example: \
                     `tensor.to(device)` where `device` is set at the top of \
                     your script."
                        .to_owned(),
                )
                .emit();
        }
    }

    /// Warn when a `torch.Tensor` is passed to `print()`. This triggers
    /// `__repr__`, which forces GPU→CPU synchronization.
    fn check_pytorch_print_tensor(&self, x: &ExprCall, _callee_ty: &Type, errors: &ErrorCollector) {
        let Expr::Name(name) = &*x.func else {
            return;
        };
        if name.id.as_str() != "print" {
            return;
        }
        for arg in &x.arguments.args {
            // Only check simple name references to avoid re-inferring complex
            // expressions (which could produce duplicate diagnostics).
            let Expr::Name(_) = arg else {
                continue;
            };
            let arg_ty = self.expr_infer(arg, errors);
            if Self::is_pytorch_tensor_type(&arg_ty) {
                errors
                    .error_builder(
                        arg.range(),
                        ErrorKind::PytorchEfficiencyLintPrintTensor,
                        "printing a `Tensor` causes implicit GPU-to-CPU synchronization".to_owned(),
                    )
                    .with_detail(
                        "The `print()` call triggers `Tensor.__repr__()`, which \
                         transfers data from GPU to CPU and blocks until all pending \
                         GPU operations complete. Use `print(tensor.shape)` to inspect \
                         metadata without synchronizing, or guard with \
                         `if DEBUG: print(tensor)`."
                            .to_owned(),
                    )
                    .emit();
            }
        }
    }

    /// Warn when `.to(device)` is called on a tensor returned by a factory function
    /// like `torch.zeros()` that already accepts a `device=` parameter. Passing
    /// `device=` directly avoids allocating on CPU and then copying to the target device.
    fn check_pytorch_redundant_to_call(
        &self,
        x: &ExprCall,
        callee_ty: &Type,
        errors: &ErrorCollector,
    ) {
        let Expr::Attribute(attr_expr) = &*x.func else {
            return;
        };
        if attr_expr.attr.id.as_str() != "to" {
            return;
        }
        if x.arguments.is_empty() {
            return;
        }
        if !matches!(callee_ty, Type::BoundMethod(bm) if Self::is_pytorch_tensor_type(&bm.obj)) {
            return;
        }
        let Expr::Call(base_call) = &*attr_expr.value else {
            return;
        };
        let Expr::Attribute(factory_attr) = &*base_call.func else {
            return;
        };
        let factory_name = factory_attr.attr.id.as_str();
        const TENSOR_FACTORIES: &[&str] = &[
            "zeros",
            "ones",
            "empty",
            "randn",
            "rand",
            "full",
            "arange",
            "linspace",
            "logspace",
            "eye",
            "zeros_like",
            "ones_like",
            "empty_like",
            "randn_like",
            "rand_like",
            "full_like",
        ];
        if !TENSOR_FACTORIES.contains(&factory_name) {
            return;
        }
        let Expr::Name(module_name) = &*factory_attr.value else {
            return;
        };
        if module_name.id.as_str() != "torch" {
            return;
        }
        // Don't fire if the factory already has `device=` — the `.to()` is
        // likely a dtype cast (e.g., `torch.randn(..., device="cuda").to(torch.bfloat16)`).
        let factory_has_device = base_call
            .arguments
            .keywords
            .iter()
            .any(|kw| kw.arg.as_ref().is_some_and(|id| id.as_str() == "device"));
        if factory_has_device {
            return;
        }
        errors
            .error_builder(
                x.range(),
                ErrorKind::PytorchEfficiencyLintRedundantToCall,
                format!(
                    "`torch.{factory_name}(...).to(device)` creates the tensor on CPU \
                     first, then copies it"
                ),
            )
            .with_detail(format!(
                "Pass `device=` directly to `torch.{factory_name}()` \
                 to create the tensor on the target device and avoid a redundant copy. \
                 For example: `torch.{factory_name}(..., device=device)`"
            ))
            .emit();
    }

    fn tuple_infer(&self, x: &ExprTuple, hint: Option<HintRef>, errors: &ErrorCollector) -> Type {
        let owner = Owner::new();
        let (hint_ts, default_hint) = if let Some(hint) = &hint {
            let (tuples, nontuples) = self.split_tuple_hint(*hint);
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
                // The Var filter is needed for performance, not correctness. Without it, we get a
                // significant slowdown in pytorch incremental edit time. Note that this filtering
                // technically causes us to lose an opportunity for contextual typing: if the var
                // was created from a Quantified with an upper bound, we could use the upper bound
                // as a hint. However, no other type checker does this.
                let nontuple_hint = self.unions(
                    nontuples
                        .into_iter()
                        .filter(|t| !matches!(t, Type::Var(_)))
                        .cloned()
                        .collect(),
                );
                let nontuple_element_hints = self
                    .decompose_hint(HintRef::soft(&nontuple_hint), |hint| {
                        self.decompose_tuple(hint)
                    });
                for nontuple_element_hint in nontuple_element_hints {
                    let nontuple_element_hint = owner.push(nontuple_element_hint);
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
                        Type::Tuple(Tuple::Unpacked(f)) if unbounded.is_empty() => {
                            let (pre, middle, suff) = *f;
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
                                    ErrorKind::NotIterable,
                                    format!("Expected an iterable, got `{}`", self.for_display(ty)),
                                );
                                encountered_invalid_star = true;
                                hint_ts_iter.nth(usize::MAX); // TODO: missing test
                            }
                        }
                    }
                }
                _ => {
                    let ty = self.expr_infer_with_hint(
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
                    if hint.is_none() && prefix.len() > MAX_TUPLE_LENGTH {
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

    fn split_tuple_hint<'b>(&self, hint: HintRef<'_, 'b>) -> (Vec<&'b Tuple>, Vec<&'b Type>) {
        hint.types().iter().partition_map(|t| match t {
            Type::Tuple(tuple) => Either::Left(tuple),
            _ => Either::Right(t),
        })
    }

    fn tuple_to_element_hints<'b>(&self, tup: &'b Tuple) -> (Vec<&'b Type>, Option<&'b Type>) {
        match tup {
            Tuple::Concrete(elts) => (elts.iter().collect(), None),
            Tuple::Unpacked(f) => {
                // TODO: We should also contextually type based on the middle and suffix
                (f.0.iter().collect(), None)
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
        if let Some(hint) = hint {
            for hint_ty in hint.types() {
                let (typed_dict, is_update) = match hint_ty {
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
                    || hint.types().len() == 1
                        && hint
                            .errors()
                            .inspect(|errors| errors.extend(check_errors))
                            .is_some()
                {
                    errors.extend(item_errors);
                    return hint_ty.clone();
                }
            }
        }
        // Note that we don't need to filter out the TypedDict options here; any non-`dict` options
        // are ignored when decomposing the hint.
        self.dict_items_infer(range, flattened_items, hint, errors)
    }

    /// Infers a type for a dictionary literal with the specified items & an optional contextual hint
    /// In order to preserve information about heterogeneous key/value types, we will infer an anonymous
    /// typed dict if the following conditions are met:
    /// - there cannot already be a contextual hint, unless it is a bare partial placeholder and at
    ///   least one literal value still contains an unpinned placeholder var (for example `[]` or
    ///   `{}`). This lets `{"start": d, "tasks": []}` form an anonymous TypedDict so the open
    ///   container can be pinned by later use, while still letting plain accumulator patterns like
    ///   `d[k] = {"x": 1}` widen normally.
    /// - all the keys must be string literals
    /// - any unpacked value is also an anonymous typed dict
    /// - the dict cannot be empty
    fn dict_items_infer(
        &self,
        range: TextRange,
        items: Vec<&DictItem>,
        hint: Option<HintRef>,
        errors: &ErrorCollector,
    ) -> Type {
        self.infer_with_decomposed_hint(
            hint,
            |hint| {
                // A partial union member carries no structural information for dict decomposition.
                // The lone-bare-partial case is handled later when deciding whether to form an
                // anonymous TypedDict.
                if self.solver().is_partial(hint) {
                    return None;
                }
                let (key_hint, value_hint) = self.decompose_dict(hint);
                if key_hint.is_none() && value_hint.is_none() {
                    None
                } else {
                    Some((key_hint, value_hint))
                }
            },
            |hints| {
                let (key_hint, value_hint) = hints.unwrap_or_default();
                self.dict_items_infer_inner(range, &items, hint, key_hint, value_hint, errors)
            },
        )
    }

    fn dict_items_infer_inner(
        &self,
        range: TextRange,
        items: &[&DictItem],
        hint: Option<HintRef>,
        key_hint: Option<Type>,
        value_hint: Option<Type>,
        errors: &ErrorCollector,
    ) -> Type {
        if items.is_empty() {
            let key_ty = key_hint.unwrap_or_else(|| {
                self.solver()
                    .fresh_partial_contained(self.uniques, range)
                    .to_type(self.heap)
            });
            let value_ty = value_hint.unwrap_or_else(|| {
                self.solver()
                    .fresh_partial_contained(self.uniques, range)
                    .to_type(self.heap)
            });
            self.heap.mk_class_type(self.stdlib.dict(key_ty, value_ty))
        } else {
            // Use a map to track fields by name so later fields override earlier ones
            let mut typed_dict_fields_map: SmallMap<Name, TypedDictField> = SmallMap::new();
            let bare_partial_hint = matches!(hint, Some(hint) if matches!(hint.types(), [ty] if self.solver().is_partial(ty)));
            // We can create an anonymous typed dict if there's no hint, the size is reasonable,
            // and all keys are string literals. A bare partial hint from first-use inference is
            // also allowed so heterogeneous literals like `{"start": d, "tasks": []}` can first
            // form an anonymous TypedDict before the outer container pins their shape. Unpackings
            // are resolved later - we only allow them if all unpackings resolve to anonymous typed
            // dicts.
            let mut can_create_anonymous_typed_dict = (hint.is_none() || bare_partial_hint)
                && items.len() <= ANONYMOUS_TYPED_DICT_MAX_ITEMS
                && items.iter().all(|item| {
                    item.key.is_none()
                        || item
                            .key
                            .as_ref()
                            .is_some_and(|k| k.as_string_literal_expr().is_some())
                });
            let has_non_none_value = items
                .iter()
                .any(|x| x.key.is_some() && !x.value.is_none_literal_expr());
            let mut key_tys = Vec::new();
            let mut value_tys = Vec::new();
            items.iter().for_each(|x| match &x.key {
                Some(key) => {
                    let key_t = self.expr_infer_with_hint_promote(
                        key,
                        key_hint.as_ref().and_then(|key_hint| {
                            hint.as_ref()
                                .map(|hint| HintRef::new(key_hint, hint.errors()))
                        }),
                        errors,
                    );
                    let value_t = self.expr_infer_with_hint_promote(
                        &x.value,
                        value_hint.as_ref().and_then(|value_hint| {
                            hint.as_ref()
                                .map(|hint| HintRef::new(value_hint, hint.errors()))
                        }),
                        errors,
                    );
                    if !key_t.is_error() {
                        key_tys.push(key_t);
                    }
                    if can_create_anonymous_typed_dict
                        && let Some(string_lit) = key.as_string_literal_expr()
                    {
                        let key_name = Name::new(string_lit.value.to_str());
                        typed_dict_fields_map.insert(
                            key_name,
                            TypedDictField {
                                ty: if value_t.is_none() && !has_non_none_value {
                                    self.unions(vec![
                                        self.heap.mk_none(),
                                        self.solver()
                                            .fresh_partial_contained(self.uniques, x.value.range())
                                            .to_type(self.heap),
                                    ])
                                } else {
                                    value_t.clone()
                                },
                                required: false,
                                read_only_reason: None,
                            },
                        );
                    }
                    if !value_t.is_error() {
                        value_tys.push(value_t);
                    }
                }
                None => {
                    let ty = self.expr_infer(&x.value, errors);
                    // If the unpacked value is an anonymous typed dict, merge its fields.
                    // Later fields override earlier ones with the same name.
                    if can_create_anonymous_typed_dict
                        && let Type::TypedDict(TypedDict::Anonymous(inner)) = &ty
                    {
                        key_tys.push(self.stdlib.str().clone().to_type());
                        for (name, field) in inner.fields.iter() {
                            typed_dict_fields_map.insert(name.clone(), field.clone());
                            if !field.ty.is_error() {
                                value_tys.push(field.ty.clone());
                            }
                        }
                    } else if let Some((key_t, value_t)) = self.unwrap_mapping(&ty) {
                        // Non-anonymous-typed-dict unpacking disables anonymous typed dict creation
                        can_create_anonymous_typed_dict = false;
                        if !key_t.is_error() {
                            if let Some(key_hint) = &key_hint
                                && self.is_subset_eq(&key_t, key_hint)
                            {
                                key_tys.push(key_hint.clone());
                            } else {
                                key_tys.push(key_t);
                            }
                        }
                        if !value_t.is_error() {
                            if let Some(value_hint) = &value_hint
                                && self.is_subset_eq(&value_t, value_hint)
                            {
                                value_tys.push(value_hint.clone());
                            } else {
                                value_tys.push(value_t);
                            }
                        }
                    } else {
                        can_create_anonymous_typed_dict = false;
                        self.error(
                            errors,
                            x.value.range(),
                            ErrorKind::InvalidArgument,
                            format!("Expected a mapping, got {}", self.for_display(ty)),
                        );
                    }
                }
            });
            let any_field_has_open_placeholder = typed_dict_fields_map.values().any(|field| {
                field
                    .ty
                    .collect_maybe_placeholder_vars()
                    .iter()
                    .any(|v| self.solver().var_is_partial(*v))
            });
            if can_create_anonymous_typed_dict
                && !typed_dict_fields_map.is_empty()
                && typed_dict_fields_map.len() <= ANONYMOUS_TYPED_DICT_MAX_ITEMS
                && (!bare_partial_hint || any_field_has_open_placeholder)
            {
                let typed_dict_fields: Vec<_> = typed_dict_fields_map.into_iter().collect();
                return self.heap.mk_typed_dict(TypedDict::Anonymous(Box::new(
                    AnonymousTypedDictInner {
                        fields: typed_dict_fields,
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

    /// If `func(args)` is a `.<method>("<literal>", ...)` call, return the receiver's
    /// type, the key expression, and the literal key. Callers apply their own
    /// receiver/arg-count constraints.
    fn dict_method_literal_key<'b>(
        &self,
        func: &Expr,
        args: &'b Arguments,
        method: &str,
        errors: &ErrorCollector,
    ) -> Option<(TypeInfo, &'b Expr, &'b StringLiteralValue)> {
        let Expr::Attribute(attr_expr) = func else {
            return None;
        };
        if attr_expr.attr.id.as_str() != method {
            return None;
        }
        let key_expr = args.args.first()?;
        let Expr::StringLiteral(ExprStringLiteral { value: key, .. }) = key_expr else {
            return None;
        };
        let obj_ty = self.expr_infer_impl(&attr_expr.value, None, errors);
        Some((obj_ty, key_expr, key))
    }

    // Is this a call to `dict.get` with a single string literal argument
    fn is_dict_get_with_literal<'b>(
        &self,
        func: &Expr,
        args: &'b Arguments,
        errors: &ErrorCollector,
    ) -> Option<(TypeInfo, &'b Expr, StringLiteralValue)> {
        if args.args.len() != 1 {
            return None;
        }
        let (obj_ty, key_expr, key) = self.dict_method_literal_key(func, args, "get", errors)?;
        self.is_dict_like(obj_ty.ty())
            .then(|| (obj_ty, key_expr, key.clone()))
    }

    /// `.get`/`.setdefault` on an anonymous TypedDict with a literal key. Both yield the
    /// value if present, else `None`/the default, so the result is `field.ty | None`, or
    /// `field.ty | default`. (`.setdefault` also inserts the key; the result type is the same.)
    fn anonymous_typed_dict_get_or_setdefault_with_literal(
        &self,
        func: &Expr,
        args: &Arguments,
        method: &str,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        if !args.keywords.is_empty() || args.args.len() > 2 {
            return None;
        }
        let (obj_ty, _key_expr, key) = self.dict_method_literal_key(func, args, method, errors)?;
        let Type::TypedDict(td @ TypedDict::Anonymous(_)) = obj_ty.ty() else {
            return None;
        };
        let field = self.typed_dict_field(td, &Name::new(key.to_str()))?;
        // A presence-narrowed key (e.g. after `if "x" in d:`) is known to be present, so the
        // value cannot be `None` and any default is unreachable.
        if obj_ty.has_value_less_presence(&FacetKind::Key(key.to_string())) {
            return Some(field.ty);
        }
        let result = if let Some(default) = args.args.get(1) {
            self.union(
                field.ty,
                self.expr_infer(default, errors)
                    .promote_implicit_literals(self.stdlib),
            )
        } else {
            self.heap.mk_optional(field.ty)
        };
        Some(
            obj_ty
                .at_facet(&FacetKind::Key(key.to_string()), || result.clone())
                .into_ty(),
        )
    }

    /// `.pop` on an anonymous TypedDict with a literal key. Unlike `.get`, a missing key
    /// raises `KeyError` instead of returning `None`, so the result is `field.ty`, or
    /// `field.ty | default` when a default is given.
    fn anonymous_typed_dict_pop_with_literal(
        &self,
        func: &Expr,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        if !args.keywords.is_empty() || args.args.len() > 2 {
            return None;
        }
        let (obj_ty, _key_expr, key) = self.dict_method_literal_key(func, args, "pop", errors)?;
        let Type::TypedDict(td @ TypedDict::Anonymous(_)) = obj_ty.ty() else {
            return None;
        };
        let field = self.typed_dict_field(td, &Name::new(key.to_str()))?;
        // A presence-narrowed key is known to be present, so any default is unreachable.
        if obj_ty.has_value_less_presence(&FacetKind::Key(key.to_string())) {
            return Some(field.ty);
        }
        let result = if let Some(default) = args.args.get(1) {
            self.union(
                field.ty,
                self.expr_infer(default, errors)
                    .promote_implicit_literals(self.stdlib),
            )
        } else {
            field.ty
        };
        Some(
            obj_ty
                .at_facet(&FacetKind::Key(key.to_string()), || result.clone())
                .into_ty(),
        )
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
        // `target` is the truthiness that causes short-circuiting: `and` short-circuits on
        // falsy values, `or` on truthy values.
        //
        // `result_narrow` is used to narrow all but the last operand to values that could actually be
        // returned as the result — for `and` that means the falsy subset, and vice versa for `or`.
        // For example: `X and Y` only returns `X` if it is falsy, so the returned type is `IsFalsy(X) | Y`
        let (target, result_narrow) = match op {
            BoolOp::And => (false, AtomicNarrowOp::IsFalsy),
            BoolOp::Or => (true, AtomicNarrowOp::IsTruthy),
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
            self.expand_mut(&mut t);
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
                    let t = if i != last_index {
                        self.atomic_narrow(&t, &result_narrow, value.range(), errors)
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

    /// Infer an attribute access from its already-inferred base. Factored from the
    /// `Expr::Attribute` arm so a method call can infer its receiver once and reuse it.
    fn attr_access_infer(
        &self,
        x: &ExprAttribute,
        base: &TypeInfo,
        errors: &ErrorCollector,
    ) -> TypeInfo {
        self.record_external_attribute_definition_index(base.ty(), x.attr.id(), x.attr.range);
        let attr_type = self.attr_infer(base, &x.attr.id, x.range, errors, None);
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
        if let Some(idx) = int_from_slice(slice) {
            TypeInfo::at_facet(base, &FacetKind::Index(idx), || {
                self.subscript_infer_for_type(base.ty(), slice, range, errors)
            })
        } else if let Expr::StringLiteral(ExprStringLiteral { value, .. }) = slice {
            self.subscript_infer_for_key_facet(
                base,
                FacetKind::Key(value.to_string()),
                slice,
                range,
                errors,
            )
        } else {
            let swallower = self.error_swallower();
            match self.expr_infer(slice, &swallower) {
                Type::Literal(ref lit) if let Lit::Str(value) = &lit.value => {
                    let facet = FacetKind::Key(value.to_string());
                    self.subscript_infer_for_key_facet(base, facet, slice, range, errors)
                }
                _ => {
                    TypeInfo::of_ty(self.subscript_infer_for_type(base.ty(), slice, range, errors))
                }
            }
        }
    }

    /// Resolve a string-key subscript taking into account whether the key is definitely known to be present.
    fn subscript_infer_for_key_facet(
        &self,
        base: &TypeInfo,
        facet: FacetKind,
        slice: &Expr,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> TypeInfo {
        if base.has_value_less_presence(&facet) {
            TypeInfo::of_ty(self.subscript_infer_for_type_with_key_present(
                base.ty(),
                slice,
                range,
                errors,
                true,
            ))
        } else {
            TypeInfo::at_facet(base, &facet, || {
                self.subscript_infer_for_type(base.ty(), slice, range, errors)
            })
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
                *ty = self.heap.mk_type_of(self.heap.mk_any_implicit())
            }
            Type::ClassDef(cls) => {
                if cls.is_builtin("tuple") {
                    Self::add_implicit_any_error(errors, range, "class `tuple`".to_owned(), None);
                    *ty = self
                        .heap
                        .mk_type_of(self.heap.mk_unbounded_tuple(self.heap.mk_any_implicit()));
                } else if cls.is_builtin("type") {
                    // `type`` is equivalent to `type[Any]`. As a result, the class def itself
                    // has type `type[type[Any]]`.
                    *ty = self
                        .heap
                        .mk_type_of(self.heap.mk_type_of(self.heap.mk_any_implicit()));
                } else if cls.has_toplevel_qname("typing", "Any") {
                    *ty = self.heap.mk_type_of(self.heap.mk_any_explicit())
                } else if cls.has_toplevel_qname("typing", "NamedTuple") {
                    // When `NamedTuple` is used as a type annotation (e.g. TypeVar bound),
                    // resolve to `NamedTupleFallback` — the class that actually appears in
                    // the MRO of user-defined NamedTuple subclasses.
                    *ty = self.heap.mk_type_of(
                        self.heap
                            .mk_class_type(self.stdlib.named_tuple_fallback().clone()),
                    );
                } else {
                    // All other classes (including Tensor) get promoted and wrapped in type_form
                    *ty = self.heap.mk_type_of(self.promote(cls, range, errors));
                }
            }
            Type::ClassType(cls) if cls.is_builtin("type") => {
                *ty = self.heap.mk_type_of(self.heap.mk_any_implicit());
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
                    ErrorKind::InvalidLiteral,
                    format!(
                        "Expected literal `True` or `False`, got `{}`",
                        self.for_display(ty)
                    ),
                );
                false
            }
        }
    }

    pub fn sentinel_from_call(
        &self,
        assignment_name: Identifier,
        nesting_context: NestingContext,
        x: &ExprCall,
        errors: &ErrorCollector,
    ) -> Sentinel {
        let mut sentinel_name = assignment_name;
        let mut iargs = x.arguments.args.iter();
        if let Some(arg) = iargs.next() {
            if let Expr::StringLiteral(lit) = arg {
                sentinel_name = Identifier::new(lit.value.to_str(), lit.range());
            } else {
                self.error(
                    errors,
                    arg.range(),
                    ErrorKind::InvalidSentinel,
                    "Expected first argument of sentinel to be a string literal".to_owned(),
                );
            }
        } else {
            self.error(
                errors,
                x.range,
                ErrorKind::InvalidSentinel,
                "Sentinel requires a name as the first argument".to_owned(),
            );
        }
        if let Some(arg) = iargs.next() {
            let args_range_end = x.arguments.args.last().map(|arg| arg.range().end());
            let range = TextRange::new(
                arg.range().start(),
                // args_range_end should never be None as it should only be None if there are
                // no args, but no reason not to have a default here anyway.
                args_range_end.unwrap_or_else(|| arg.range().end()),
            );
            self.error(
                errors,
                range,
                ErrorKind::InvalidSentinel,
                "Sentinel only takes one positional argument".to_owned(),
            );
        }

        for kw in &x.arguments.keywords {
            match &kw.arg {
                Some(id) => match id.id.as_str() {
                    "repr" => {
                        let got = self.expr_infer(&kw.value, errors);
                        if !self
                            .is_subset_eq(&got, &self.heap.mk_class_type(self.stdlib.str().clone()))
                        {
                            self.error(
                                errors,
                                kw.range,
                                ErrorKind::InvalidSentinel,
                                format!("Invalid type for sentinel `repr` {got}"),
                            );
                        }
                    }
                    _ => {
                        self.error(
                            errors,
                            kw.range,
                            ErrorKind::InvalidSentinel,
                            format!("Unexpected keyword argument `{}` to sentinel", id.id),
                        );
                    }
                },
                _ => {
                    self.error(
                        errors,
                        kw.range,
                        ErrorKind::InvalidSentinel,
                        "Cannot pass unpacked keyword arguments to sentinel".to_owned(),
                    );
                }
            }
        }

        Sentinel::new(sentinel_name, nesting_context, self.module().dupe())
    }

    pub fn typevar_from_call(
        &self,
        name: Identifier,
        x: &ExprCall,
        kind: QuantifiedKind,
        errors: &ErrorCollector,
    ) -> TypeVar {
        let construct = kind.to_string();
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
                        ErrorKind::InvalidTypeVar,
                        format!(
                            "{construct} must be assigned to a variable named `{}`",
                            lit.value.to_str()
                        ),
                    );
                }
            } else {
                self.error(
                    errors,
                    arg.range(),
                    ErrorKind::InvalidTypeVar,
                    format!("Expected first argument of {construct} to be a string literal"),
                );
            }
        };

        let mut try_set_variance = |kw: &Keyword, v: PreInferenceVariance| {
            if self.literal_bool_infer(&kw.value, errors) {
                if variance.is_some() {
                    self.error(
                        errors,
                        kw.range,
                        ErrorKind::InvalidTypeVar,
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
                                ErrorKind::InvalidTypeVar,
                                format!("{construct} cannot have both constraints and bound"),
                            );
                            restriction = Some(Restriction::Unrestricted);
                        } else {
                            restriction = Some(Restriction::Bound(bound));
                        }
                    }
                    "default" => {
                        default = Some((
                            self.expr_untype(
                                &kw.value,
                                TypeFormContext::quantified_kind_default(kind),
                                errors,
                            ),
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
                                ErrorKind::InvalidTypeVar,
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
                            ErrorKind::InvalidTypeVar,
                            format!("Unexpected keyword argument `{}` to {construct}", id.id),
                        );
                    }
                },
                _ => {
                    self.error(
                        errors,
                        kw.range,
                        ErrorKind::InvalidTypeVar,
                        format!("Cannot pass unpacked keyword arguments to {construct}"),
                    );
                }
            }
        }

        if !arg_name {
            self.error(
                errors,
                x.range,
                ErrorKind::InvalidTypeVar,
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
                ErrorKind::InvalidTypeVar,
                format!(
                    "Expected at least 2 constraints in {construct} `{}`, got {}",
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
                kind,
                &default_ty,
                default_range,
                &restriction,
                errors,
            ));
        }

        let variance = variance.unwrap_or(PreInferenceVariance::Invariant);

        TypeVar::new_with_kind(
            name,
            self.module().dupe(),
            kind,
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
                        ErrorKind::InvalidParamSpec,
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
                    ErrorKind::InvalidParamSpec,
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
                                ErrorKind::InvalidParamSpec,
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
                            ErrorKind::InvalidParamSpec,
                            format!("Unexpected keyword argument `{}` to ParamSpec", id.id),
                        );
                    }
                },
                _ => {
                    self.error(
                        errors,
                        kw.range,
                        ErrorKind::InvalidParamSpec,
                        "Cannot pass unpacked keyword arguments to ParamSpec".to_owned(),
                    );
                }
            }
        }

        if !arg_name {
            self.error(
                errors,
                x.range,
                ErrorKind::InvalidParamSpec,
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
                        ErrorKind::InvalidTypeVarTuple,
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
                    ErrorKind::InvalidTypeVarTuple,
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
                ErrorKind::InvalidTypeVarTuple,
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
                                ErrorKind::InvalidTypeVarTuple,
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
                            ErrorKind::InvalidTypeVarTuple,
                            format!("Unexpected keyword argument `{}` to TypeVarTuple", id.id),
                        );
                    }
                },
                _ => {
                    self.error(
                        errors,
                        kw.range,
                        ErrorKind::InvalidTypeVarTuple,
                        "Cannot pass unpacked keyword arguments to TypeVarTuple".to_owned(),
                    );
                }
            }
        }
        if !arg_name {
            self.error(
                errors,
                x.range,
                ErrorKind::InvalidTypeVarTuple,
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
        elt_hint: Option<HintRef>,
        errors: &ErrorCollector,
    ) -> Vec<Type> {
        let star_hint = LazyCell::new(|| {
            elt_hint.map(|hint| {
                Type::union(
                    hint.types()
                        .map(|hint| self.heap.mk_class_type(self.stdlib.iterable(hint.clone()))),
                )
            })
        });
        elts.map(|x| match x {
            Expr::Starred(ExprStarred { value, .. }) => {
                let unpacked_ty = self.expr_infer_with_hint_promote(
                    value,
                    HintRef::with_ty_opt(elt_hint, star_hint.as_ref()),
                    errors,
                );
                if let Some(iterable_ty) = self.unwrap_iterable(&unpacked_ty) {
                    iterable_ty
                } else {
                    self.error(
                        errors,
                        x.range(),
                        ErrorKind::NotIterable,
                        format!(
                            "Expected an iterable, got `{}`",
                            self.for_display(unpacked_ty)
                        ),
                    )
                }
            }
            _ => self.expr_infer_with_hint_promote(x, elt_hint, errors),
        })
    }

    fn is_enum_class_type(&self, ty: &Type) -> bool {
        match ty {
            Type::ClassType(cls) | Type::SelfType(cls) => {
                self.has_superclass(cls.class_object(), self.stdlib.enum_class().class_object())
            }
            Type::Union(f) => f
                .members
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
        self.subscript_infer_for_type_with_key_present(base, slice, range, errors, false)
    }

    fn subscript_infer_for_type_with_key_present(
        &self,
        base: &Type,
        slice: &Expr,
        range: TextRange,
        errors: &ErrorCollector,
        key_present: bool, // true if the key is definitely known to be present
    ) -> Type {
        let xs = Ast::unpack_slice(slice);
        self.distribute_over_union(base, |base| {
            let mut base = base.clone();
            if let Type::Var(v) = base {
                base = self.solver().force_var(v);
            }
            if matches!(&base, Type::ClassDef(t) if t.name() == "tuple") {
                base = self.heap.mk_type_of(self.heap.mk_special_form(SpecialForm::Tuple));
            }
            if let Type::Intersect(x) = base {
                // TODO: Handle subscription of intersections properly.
                base = x.1;
            }
            match base {
                Type::Forall(forall) => {
                    if matches!(forall.body, Forallable::TypeAlias(_)) {
                        let tys =
                            self.parse_type_args_for_tparams(xs, forall.tparams.as_vec(), errors);
                        self.specialize_forall(*forall, tys, range, errors)
                    } else {
                        let name = forall.body.name();
                        self.error(
                            errors,
                            range,
                            ErrorKind::UnsupportedOperation,
                            format!("`{}` is not subscriptable", name.as_ref().as_str()),
                        )
                    }
                }
                // Note that we have to check for `builtins.type` by name here because this code runs
                // when we're bootstrapping the stdlib and don't have access to class objects yet.
                Type::ClassDef(cls) if cls.is_builtin("type") => {
                    let (arguments, _) = match slice {
                            Expr::Tuple(x) => (x.elts.as_slice(), x.parenthesized),
                            _ => (slice::from_ref(slice), false),
                    };
                    self.apply_unary_special_form("type".to_owned(), arguments, range, TypeFormContext::TypeArgumentForType, errors, |arg| self.heap.mk_type_of(arg))
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
                            ErrorKind::BadSpecialization,
                            format!(
                                "Expected 1 type argument for `PyreReadOnly`, got {}",
                                xs.len()
                            ),
                        ),
                    }
                }
                // Shaped-array type parsing for registered array classes.
                Type::ClassDef(ref cls) if self.is_shaped_array_class(cls) => {
                    Type::type_of(self.parse_registered_shaped_array_type(cls, xs, range, errors))
                }
                Type::ClassDef(ref cls) if self.is_int_tuple_class(cls) => {
                    self.parse_int_tuple_type(xs, errors)
                }
                Type::ClassDef(ref cls) if self.is_int_class(cls) => {
                    self.parse_int_type(xs, range, errors)
                }
                Type::ClassDef(ref cls)
                    if cls.has_toplevel_qname("shape_extensions", "ProxyMethod") =>
                {
                    self.proxy_method_subscript_infer(cls, xs, range, errors)
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
                            ErrorKind::BadIndex,
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
                        &self.expr_check(slice, None, errors),
                        &self.heap.mk_class_type(self.stdlib.str().clone()),
                    ) {
                        self.heap.mk_class_type(self.as_class_type_unchecked(cls))
                    } else {
                        self.error(
                            errors,
                            slice.range(),
                            ErrorKind::BadIndex,
                            format!("Enum `{}` can only be indexed by strings", cls.name()),
                        )
                    }
                }
                Type::ClassDef(cls) => self.class_subscript_infer(&cls, slice, xs, range, errors),
                Type::Type(f) if matches!(&*f, Type::Quantified(q) if q.is_type_var()) => {
                    // Repeated match because pattern guards cannot move out of bindings.
                    let Type::Quantified(quantified) = *f else { unreachable!("guarded by matches! above") };
                    let quantified = *quantified;
                    let base_display_ty =
                        self.heap.mk_type(self.heap.mk_quantified(quantified.clone()));
                    if self.is_restricted_to_enum_class_def_type(&quantified) {
                        if self.is_subset_eq(
                            &self.expr_check(slice, None, errors),
                            &self.heap.mk_class_type(self.stdlib.str().clone()),
                        ) {
                            quantified.to_type(self.heap)
                        } else {
                            self.error(
                                errors,
                                slice.range(),
                                ErrorKind::BadIndex,
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
                            ErrorKind::UnsupportedOperation,
                            format!(
                                "`{}` is not subscriptable",
                                self.for_display(base_display_ty)
                            ),
                        )
                    }
                }
                Type::Type(inner) if self.is_enum_class_type(inner.as_ref()) => {
                    let base_display_ty = self.heap.mk_type_of((*inner).clone());
                    let enum_value_ty = *inner;
                    if self.is_subset_eq(
                        &self.expr_check(slice, None, errors),
                        &self.heap.mk_class_type(self.stdlib.str().clone()),
                    ) {
                        enum_value_ty
                    } else {
                        self.error(
                            errors,
                            slice.range(),
                            ErrorKind::BadIndex,
                            format!(
                                "Enum type `{}` can only be indexed by strings",
                                self.for_display(base_display_ty)
                            ),
                        )
                    }
                }
                Type::Type(f) if let Type::SpecialForm(special) = *f => {
                    self.apply_special_form(special, slice, range, errors)
                }
                Type::Tuple(ref tuple) => self.infer_tuple_subscript(
                    tuple.clone(),
                    slice,
                    range,
                    errors,
                    Some(&|| ErrorContext::Index(self.for_display(base.clone()))),
                ),
                Type::IntTuple(ref int_tuple) => self.infer_int_tuple_subscript(
                    int_tuple,
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
                Type::ShapedArray(ref shaped_array_type) => {
                    self.infer_shaped_array_index(shaped_array_type, slice, range, errors)
                }
                // Shaped arrays that have not gone through annotation
                // canonicalization still use tensor indexing logic.
                Type::ClassType(ref cls) if self.is_shaped_array_class(cls.class_object()) => {
                    let shaped_array_type = self.shaped_array_classtype_to_shaped_array_type(cls);
                    self.infer_shaped_array_index(&shaped_array_type, slice, range, errors)
                }
                Type::ClassType(ref cls) | Type::SelfType(ref cls)
                    if let Some(tuple) = self.as_tuple(cls)
                        && !self.class_overrides_tuple_getitem(cls) =>
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
                Type::DataFrame(schema) => {
                    if let Expr::StringLiteral(key) = slice
                        && schema.completeness == SchemaCompleteness::Complete
                    {
                        let name = key.value.to_str();
                        if !schema.columns.iter().any(|(c, _)| c.as_str() == name) {
                            errors
                                .error_builder(
                                    slice.range(),
                                    ErrorKind::UnknownColumn,
                                    format!("Column `{name}` is not in the DataFrame schema"),
                                )
                                .emit();
                        }
                    } else if let Expr::List(ExprList { elts, .. }) = slice
                        && schema.kind == DataFrameKind::Polars
                        && let Some(narrowed) =
                            self.polars_select_columns(&schema, elts, errors)
                    {
                        return narrowed;
                    }
                    self.subscript_infer_for_type_with_key_present(
                        &schema.underlying_type(),
                        slice,
                        range,
                        errors,
                        key_present,
                    )
                }
                Type::Quantified(ref q) if q.is_type_var() && q.restriction().is_restricted() => {
                    match q.restriction() {
                        Restriction::Bound(bound) => self
                            .subscript_infer_for_type_with_key_present(
                                bound,
                                slice,
                                range,
                                errors,
                                key_present,
                            ),
                        Restriction::Constraints(constraints) => {
                            self.unions(constraints.map(|constraint| {
                                self.subscript_infer_for_type_with_key_present(
                                    constraint,
                                    slice,
                                    range,
                                    errors,
                                    key_present,
                                )
                            }))
                        }
                        Restriction::Unrestricted => {
                            unreachable!("restricted TypeVar cannot be unrestricted")
                        }
                    }
                }
                Type::TypedDict(typed_dict) => {
                    let key_ty = self.expr_infer(slice, errors);
                    // Don't warn on anonymous typed dicts
                    let warn_on_not_required_access = matches!(typed_dict, TypedDict::TypedDict(_));
                    self.distribute_over_union(&key_ty, |ty| match ty {
                        Type::Literal(lit) if let Lit::Str(field_name) = &lit.value => {
                            let key_name = Name::new(field_name);
                            if let Some(field) = self.typed_dict_field(&typed_dict, &key_name) {
                                if warn_on_not_required_access && !field.required && !key_present {
                                    errors
                                        .error_builder(
                                            slice.range(),
                                            ErrorKind::NotRequiredKeyAccess,
                                            format!(
                                                "TypedDict key `{}` may be absent",
                                                key_name
                                            ),
                                        )
                                        .with_detail(format!(
                                            "Hint: guard this access with `'{}' in obj` or `obj.get('{}')`",
                                            key_name, key_name
                                        ))
                                        .emit();
                                }
                                field.ty.clone()
                            } else {
                                match self.typed_dict_extra_items(&typed_dict) {
                                    ExtraItems::Extra(extra) => extra.ty,
                                    extra_items if key_present => {
                                        extra_items.extra_item(self.stdlib).ty
                                    }
                                    _ => {
                                        let mut builder = errors.error_builder(
                                            slice.range(),
                                            typed_dict.key_error_kind(),
                                            format!(
                                                "{} does not have key `{field_name}`",
                                                typed_dict.label()
                                            ),
                                        );
                                        let fields = self.typed_dict_fields(&typed_dict);
                                        if let Some(suggestion) = best_suggestion(
                                            &key_name,
                                            fields.keys().map(|candidate| (candidate, 0usize)),
                                        ) {
                                            builder = builder.with_detail(format!(
                                                "Did you mean `{suggestion}`?"
                                            ));
                                        }
                                        builder.emit();
                                        self.heap.mk_any_error()
                                    }
                                }
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
                                    typed_dict.key_error_kind(),
                                    format!(
                                        "Invalid key for {}, got `{}`",
                                        typed_dict.label(),
                                        self.for_display(ty.clone())
                                    ),
                                )
                            }
                        }
                    })
                }
                Type::UntypedAlias(ta) => self.subscript_infer_for_type_with_key_present(
                    &self.untype_alias(&ta),
                    slice,
                    range,
                    errors,
                    key_present,
                ),
                t => self.error(
                    errors,
                    range,
                    ErrorKind::UnsupportedOperation,
                    format!("`{}` is not subscriptable", self.for_display(t)),
                ),
            }
        })
    }

    /// Handle tensor indexing operations
    /// - Integer index: reduces dimensionality by 1 (removes first dimension)
    /// - Slice: preserves dimensionality (keeps all dimensions)
    fn infer_shaped_array_index(
        &self,
        shaped_array_type: &ShapedArrayType,
        index: &Expr,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        // Convert a slice bound expression to a dimension type.
        // For unary negation (-expr), we preserve the Mul(-1, ...) wrapper
        // without canonicalizing, so adjust_negative can detect negative bounds
        // even after the distributive law would otherwise distribute -1 across sums.
        let to_dim = |expr: &Expr| -> Int {
            // Detect syntactic unary minus: -(inner)
            if let Expr::UnaryOp(x) = expr
                && x.op == UnaryOp::USub
            {
                let inner_ty = self.expr_infer(&x.operand, errors);
                let inner_dim = match type_to_dim(&inner_ty) {
                    Some(Int::Literal(val)) => {
                        // Literal negation: just negate the value directly
                        return Int::Literal(-val);
                    }
                    Some(dim) => dim,
                    None => return Int::Int,
                };
                // Wrap in Mul(-1, ...) WITHOUT canonicalizing.
                // This preserves the structural signal for adjust_negative.
                // The final canonicalization happens in `IntTuple`.
                return Int::Mul(Box::new(Int::Literal(-1)), Box::new(inner_dim));
            }
            let ty = self.expr_infer(expr, errors);
            type_to_dim(&ty).unwrap_or(Int::Int)
        };

        let classify = |expr: &Expr, inferred: Option<&Type>| -> Option<IndexOp> {
            match expr {
                Expr::Slice(ExprSlice {
                    lower, upper, step, ..
                }) => {
                    let start = lower.as_ref().map(|e| to_dim(e));
                    let stop = upper.as_ref().map(|e| to_dim(e));
                    let step_val = step.as_ref().map(|e| to_dim(e));
                    Some(IndexOp::Slice {
                        start,
                        stop,
                        step: step_val,
                    })
                }
                Expr::List(ExprList { elts, .. })
                    if elts.iter().all(|elt| !matches!(elt, Expr::Starred(_)))
                        && elts.iter().all(|elt| {
                            matches!(
                                classify_shaped_array_index_type(&self.expr_infer(elt, errors)),
                                Some(IndexOp::Int)
                            )
                        }) =>
                {
                    Some(IndexOp::Fancy(Int::Literal(elts.len() as i64)))
                }
                _ => match inferred {
                    Some(ty) => classify_shaped_array_index_type(ty),
                    None => classify_shaped_array_index_type(&self.expr_infer(expr, errors)),
                },
            }
        };

        match index {
            // Slice operation: tensor[start:stop:step]
            Expr::Slice(ExprSlice {
                lower, upper, step, ..
            }) => {
                let start = lower.as_ref().map(|e| to_dim(e));
                let stop = upper.as_ref().map(|e| to_dim(e));
                let step_val = step.as_ref().map(|e| to_dim(e));
                match index_shape_slice(&shaped_array_type.shape(), start, stop, step_val) {
                    Ok(shape) => self
                        .shaped_array_with_shape(shaped_array_type, shape)
                        .to_type(),
                    Err(err) => self.error(errors, range, ErrorKind::BadIndex, err.to_string()),
                }
            }
            // Bare ellipsis: tensor[...] - preserves entire shape
            Expr::EllipsisLiteral(_) => shaped_array_type.clone().to_type(),
            // None index: tensor[None] - inserts a new dimension of size 1 at the front
            Expr::NoneLiteral(_) => {
                match index_shape_multi(&shaped_array_type.shape(), &[IndexOp::NewAxis], &[], false)
                {
                    Ok(shape) => self
                        .shaped_array_with_shape(shaped_array_type, shape)
                        .to_type(),
                    Err(err) => self.error(errors, range, ErrorKind::BadIndex, err.to_string()),
                }
            }
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
                                ErrorKind::BadIndex,
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
                let pre_ops: Option<Vec<IndexOp>> =
                    pre_exprs.iter().map(|expr| classify(expr, None)).collect();
                let post_ops: Option<Vec<IndexOp>> =
                    post_exprs.iter().map(|expr| classify(expr, None)).collect();
                let (Some(pre_ops), Some(post_ops)) = (pre_ops, post_ops) else {
                    return self.shaped_array_shapeless(shaped_array_type).to_type();
                };

                match index_shape_multi(
                    &shaped_array_type.shape(),
                    &pre_ops,
                    &post_ops,
                    ellipsis_pos.is_some(),
                ) {
                    Ok(shape) => self
                        .shaped_array_with_shape(shaped_array_type, shape)
                        .to_type(),
                    Err(err) => self.error(errors, range, ErrorKind::BadIndex, err.to_string()),
                }
            }
            _ => {
                let index_ty = self.expr_infer(index, errors);
                if let Type::Tuple(tuple) = &index_ty {
                    let Tuple::Concrete(elements) = tuple else {
                        return self.shaped_array_shapeless(shaped_array_type).to_type();
                    };
                    let Some(ops) = elements
                        .iter()
                        .map(classify_shaped_array_index_type)
                        .collect::<Option<Vec<_>>>()
                    else {
                        return self.shaped_array_shapeless(shaped_array_type).to_type();
                    };
                    return match index_shape_multi(&shaped_array_type.shape(), &ops, &[], false) {
                        Ok(shape) => self
                            .shaped_array_with_shape(shaped_array_type, shape)
                            .to_type(),
                        Err(err) => self.error(errors, range, ErrorKind::BadIndex, err.to_string()),
                    };
                }

                match classify(index, Some(&index_ty)) {
                    Some(IndexOp::Int) => match index_shape_int(&shaped_array_type.shape()) {
                        Ok(shape) => self
                            .shaped_array_with_shape(shaped_array_type, shape)
                            .to_type(),
                        Err(err) => self.error(errors, range, ErrorKind::BadIndex, err.to_string()),
                    },
                    Some(IndexOp::ShapedArrayIndex(idx_dims)) => {
                        match index_shape_tensor(&shaped_array_type.shape(), &idx_dims) {
                            Ok(shape) => self
                                .shaped_array_with_shape(shaped_array_type, shape)
                                .to_type(),
                            Err(err) => {
                                self.error(errors, range, ErrorKind::BadIndex, err.to_string())
                            }
                        }
                    }
                    Some(op @ IndexOp::Fancy(_)) | Some(op @ IndexOp::NewAxis) => {
                        match index_shape_multi(&shaped_array_type.shape(), &[op], &[], false) {
                            Ok(shape) => self
                                .shaped_array_with_shape(shaped_array_type, shape)
                                .to_type(),
                            Err(err) => {
                                self.error(errors, range, ErrorKind::BadIndex, err.to_string())
                            }
                        }
                    }
                    Some(IndexOp::Slice { .. }) => {
                        unreachable!("slice indices are handled before generic index dispatch")
                    }
                    None => self.shaped_array_shapeless(shaped_array_type).to_type(),
                }
            }
        }
    }

    fn is_pytorch_tensor_type(ty: &Type) -> bool {
        fn is_torch_tensor_class(cls: &ClassType) -> bool {
            let module = cls.class_object().module_name();
            let module = module.as_str();
            cls.name().as_str() == "Tensor" && matches!(module, "torch" | "torch._tensor")
        }

        match ty {
            Type::ClassType(cls) => is_torch_tensor_class(cls),
            Type::ShapedArray(shaped_array) => is_torch_tensor_class(&shaped_array.base_class),
            _ => false,
        }
    }

    /// Check if a class should use shaped-array type parsing.
    fn is_shaped_array_class(&self, cls: &Class) -> bool {
        self.shaped_array_shape_for_class(cls).is_some()
    }

    pub(crate) fn shaped_array_shape_arg_index(&self, cls: &ClassType) -> Option<usize> {
        let shape_param = self.shaped_array_shape_for_class_type(cls)?;
        self.get_class_tparams(cls.class_object())
            .iter()
            .position(|param| param == &shape_param)
    }

    pub(crate) fn shaped_array_shape_arg(&self, cls: &ClassType) -> Option<Type> {
        let shape_idx = self.shaped_array_shape_arg_index(cls)?;
        let mut shape_arg = cls.targs().as_slice().get(shape_idx)?.clone();
        self.expand_mut(&mut shape_arg);
        Some(shape_arg)
    }

    pub(crate) fn shaped_array_shape_arg_to_shape(&self, shape_arg: &Type) -> Option<IntTuple> {
        IntTuple::from_shape_arg_type(shape_arg)
            .or_else(|| tuple_carrier_to_shape(shape_arg))
            .or_else(|| {
                let upper_bound = match shape_arg {
                    Type::Quantified(q) if q.is_type_var() => q.upper_bound(self.stdlib, self.heap),
                    Type::TypeVar(tv) => tv.upper_bound(self.stdlib, self.heap),
                    _ => return None,
                };
                let int_type = self.stdlib.int().clone().to_type();
                Self::is_int_tuple_carrier_bound(&upper_bound, &int_type)
                    .then(|| IntTuple::unpacked(Vec::new(), shape_arg.clone(), Vec::new()))
            })
    }

    pub(crate) fn shaped_array_classtype_to_shaped_array_type(
        &self,
        cls: &ClassType,
    ) -> ShapedArrayType {
        // Derive the index and argument from a single metadata lookup rather
        // than re-resolving through `shaped_array_shape_arg_index`/`_arg`, which
        // would force the (expensive) class shape metadata two more times.
        let shape_param = self
            .shaped_array_shape_for_class_type(cls)
            .expect("registered shaped-array class should have shape metadata");
        let shape_idx = self
            .get_class_tparams(cls.class_object())
            .iter()
            .position(|param| param == &shape_param)
            .expect("shaped-array metadata should refer to a class type parameter");
        let mut shape_arg = cls
            .targs()
            .as_slice()
            .get(shape_idx)
            .expect("class type should have an argument for each type parameter")
            .clone();
        self.expand_mut(&mut shape_arg);
        match shape_param.kind() {
            QuantifiedKind::TypeVar | QuantifiedKind::IntVar => {
                let shape = self
                    .shaped_array_shape_arg_to_shape(&shape_arg)
                    .unwrap_or_else(IntTuple::shapeless);
                let mut base_class = cls.clone();
                let shape_arg = base_class
                    .targs_mut()
                    .as_mut()
                    .get_mut(shape_idx)
                    .expect("class type should have an argument for each type parameter");
                *shape_arg = shape.to_shape_arg_type();
                ShapedArrayType::new(base_class, shape).with_tuple_carrier_shape_arg(shape_idx)
            }
            QuantifiedKind::TypeVarTuple => unreachable!(
                "shaped-array metadata validation rejects TypeVarTuple shape parameters"
            ),
            QuantifiedKind::ParamSpec => {
                unreachable!("shaped-array metadata validation rejects ParamSpec shape parameters")
            }
        }
    }

    /// Build a shaped-array type with a new semantic shape.
    ///
    /// Registered arrays store their shape in the metadata-selected class type
    /// argument; unregistered arrays store it inline. Non-shape class arguments
    /// such as `DType` are preserved.
    pub(crate) fn shaped_array_with_shape(
        &self,
        tensor: &ShapedArrayType,
        shape: IntTuple,
    ) -> ShapedArrayType {
        match self.shaped_array_shape_for_class_type(&tensor.base_class) {
            Some(shape_param) => match shape_param.kind() {
                QuantifiedKind::TypeVar | QuantifiedKind::IntVar => {
                    let shape_idx = self
                        .shaped_array_shape_arg_index(&tensor.base_class)
                        .expect("shaped-array metadata should refer to a class type parameter");
                    let mut tensor = tensor.clone();
                    // A registered shaped-array class stores its shape in the
                    // carrier argument at `shape_idx`, so `TupleCarrier` is the
                    // coherent style regardless of the input's prior style (e.g. a
                    // stale `Unknown`): we normalize it to match where the shape
                    // now actually lives.
                    tensor.set_tuple_carrier_shape_arg(shape_idx);
                    tensor.set_shape(shape);
                    tensor
                }
                QuantifiedKind::TypeVarTuple => unreachable!(
                    "shaped-array metadata validation rejects TypeVarTuple shape parameters"
                ),
                QuantifiedKind::ParamSpec => {
                    unreachable!(
                        "shaped-array metadata validation rejects ParamSpec shape parameters"
                    )
                }
            },
            None => {
                // A `TupleCarrier` shape lives in a registered class argument, so
                // such a tensor always takes the `Some` branch above; only inline
                // arrays (no registration metadata) reach here. Assert that
                // invariant, since `new` produces an inline shape and would
                // otherwise silently drop a carrier index -- which participates in
                // `ShapedArrayType` identity / `Eq` / `Hash`.
                assert!(
                    tensor.tuple_carrier_shape_arg_index().is_none(),
                    "a tuple-carrier shaped array reached the unregistered-class branch"
                );
                ShapedArrayType::new(tensor.base_class.clone(), shape).with_syntax(tensor.syntax)
            }
        }
    }

    /// Build a shapeless shaped-array type while keeping the raw tuple carrier
    /// coherent. A plain `ShapedArrayType::shapeless` would leave the old carrier
    /// (e.g. an unknown-rank `S`) on `base_class`, so `.shape` would stale-read the
    /// pre-operation shape. Routing through `shaped_array_with_shape` rewrites the
    /// carrier to the shapeless form too.
    fn shaped_array_shapeless(&self, tensor: &ShapedArrayType) -> ShapedArrayType {
        self.shaped_array_with_shape(tensor, IntTuple::shapeless())
    }

    /// Check if a class is a Int class (shape_extensions.Int)
    fn is_int_class(&self, cls: &Class) -> bool {
        cls.has_toplevel_qname("shape_extensions", "Int")
    }

    /// Check if a class is the shape arithmetic wrapper (shape_extensions.D)
    fn is_shape_arith_wrapper_class(&self, cls: &Class) -> bool {
        cls.has_toplevel_qname("shape_extensions", "D")
    }

    /// Parse a single dimension expression (recursive helper).
    ///
    /// A dimension expression is one element of a tensor shape. For example, in
    /// `Tensor[Batch, Channels + 1, 3]` the dimension expressions are the type
    /// variable `Batch`, the arithmetic expression `Channels + 1`, and the
    /// integer literal `3`. Returns the `Type` the dimension resolves to, or
    /// `None` (after emitting an error) if it is not a valid dimension.
    fn parse_dimension_expr(&self, expr: &Expr, errors: &ErrorCollector) -> Option<Type> {
        self.parse_dimension_expr_with_context(expr, errors, DimensionExprContext::Bare)
    }

    fn parse_dimension_expr_with_context(
        &self,
        expr: &Expr,
        errors: &ErrorCollector,
        context: DimensionExprContext,
    ) -> Option<Type> {
        // shape_extensions.D[...] and D(...) are runtime-only wrappers that
        // let Python evaluate arithmetic on PEP 695 type variables.
        match expr {
            Expr::Subscript(x) => {
                let base = self.expr_infer(&x.value, errors);
                if let Type::ClassDef(ref cls) = base
                    && self.is_shape_arith_wrapper_class(cls)
                {
                    return self.parse_dimension_expr_with_context(&x.slice, errors, context);
                }
            }
            Expr::Call(ExprCall {
                func, arguments, ..
            }) => {
                let callee = self.expr_infer(func, errors);
                if let Type::ClassDef(ref cls) = callee
                    && self.is_shape_arith_wrapper_class(cls)
                {
                    if arguments.args.len() == 1 && arguments.keywords.is_empty() {
                        return self.parse_dimension_expr_with_context(
                            &arguments.args[0],
                            errors,
                            context,
                        );
                    }
                    self.error(
                        errors,
                        expr.range(),
                        ErrorKind::InvalidAnnotation,
                        if arguments.keywords.is_empty() {
                            format!(
                                "Expected 1 positional argument for `D`, got {}",
                                arguments.args.len()
                            )
                        } else {
                            format!(
                                "`D` accepts exactly 1 positional argument and no keyword arguments, got {} positional and {} keyword",
                                arguments.args.len(),
                                arguments.keywords.len()
                            )
                        },
                    );
                    return None;
                }
            }
            _ => {}
        }

        match expr {
            // String literals are not valid dimensions
            Expr::StringLiteral(_) => {
                self.error(
                    errors,
                    expr.range(),
                    ErrorKind::InvalidAnnotation,
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
                        Some(self.heap.mk_int(Int::literal(value)))
                    } else {
                        self.error(
                            errors,
                            expr.range(),
                            ErrorKind::InvalidAnnotation,
                            "Tensor shape dimension too large".to_owned(),
                        );
                        None
                    }
                }
                _ => {
                    self.error(
                        errors,
                        expr.range(),
                        ErrorKind::InvalidAnnotation,
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
                    Type::ClassDef(cls) if cls.has_toplevel_qname("typing", "Any") => {
                        // typing.Any in a type annotation position (e.g., Tensor[16, Any])
                        // Use Explicit since the user wrote Any explicitly
                        Some(Type::Any(AnyStyle::Explicit))
                    }
                    Type::ClassDef(cls) if cls.is_builtin("int") => Some(gradual_size()),
                    _ => {
                        match self.untype_opt_with_context(
                            expr_type.clone(),
                            expr.range(),
                            errors,
                            UntypeContext::SymbolicInt(context.error_context()),
                        ) {
                            Some(Type::Quantified(q)) if q.kind() == QuantifiedKind::IntVar => {
                                Some(Type::Quantified(q))
                            }
                            Some(ty @ Type::TypeVar(_)) => Some(ty),
                            Some(ty) if ty.is_error() => Some(ty),
                            _ => {
                                self.error(
                                    errors,
                                    expr.range(),
                                    ErrorKind::InvalidAnnotation,
                                    format!(
                                        "Tensor shape dimensions must be integer literals or type variables, got `{}`",
                                        self.for_display(expr_type)
                                    ),
                                );
                                None
                            }
                        }
                    }
                }
            }
            // Unary negation: -N, -1, -(N + 1), etc.
            Expr::UnaryOp(x) if x.op == UnaryOp::USub => {
                let inner = self.parse_dimension_expr_with_context(
                    &x.operand,
                    errors,
                    DimensionExprContext::Arithmetic,
                )?;
                Some(
                    self.heap
                        .mk_int(Int::sub(self.heap.mk_int(Int::Literal(0)), inner)),
                )
            }
            // Binary operations: N + M, N * M, etc.
            Expr::BinOp(ExprBinOp {
                left, op, right, ..
            }) => {
                let make_int = match op {
                    Operator::Add => Int::add,
                    Operator::Sub => Int::sub,
                    Operator::Mult => Int::mul,
                    Operator::FloorDiv => Int::floor_div,
                    Operator::Pow => Int::pow,
                    _ => {
                        self.error(
                            errors,
                            expr.range(),
                            ErrorKind::InvalidAnnotation,
                            format!(
                                "Unsupported operator `{}` in tensor shape dimension",
                                op.as_str()
                            ),
                        );
                        return None;
                    }
                };
                let left_dim = self.parse_dimension_expr_with_context(
                    left,
                    errors,
                    DimensionExprContext::Arithmetic,
                )?;
                let right_dim = self.parse_dimension_expr_with_context(
                    right,
                    errors,
                    DimensionExprContext::Arithmetic,
                )?;
                if *op == Operator::Pow {
                    let right_dim_canon = canonicalize(right_dim.clone());
                    if int_type_is_provably_negative(&right_dim)
                        || int_type_is_provably_negative(&right_dim_canon)
                    {
                        self.error(
                            errors,
                            expr.range(),
                            ErrorKind::InvalidAnnotation,
                            "Tensor shape exponent must not be negative".to_owned(),
                        );
                        return None;
                    }
                    return Some(canonicalize(
                        self.heap.mk_int(Int::pow(left_dim, right_dim_canon)),
                    ));
                }
                Some(self.heap.mk_int(make_int(left_dim, right_dim)))
            }
            // Anything else is an error
            _ => {
                let expr_type = self.expr_infer(expr, errors);
                self.error(
                    errors,
                    expr.range(),
                    ErrorKind::InvalidAnnotation,
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
    pub(super) fn parse_dimension_list(
        &self,
        args: &[Expr],
        errors: &ErrorCollector,
    ) -> Option<Vec<Type>> {
        let mut dims = Vec::new();
        for arg in args {
            if let Some(dim) = self.parse_dimension_expr(arg, errors) {
                let simplified = canonicalize(dim);

                // Validate that literal dimensions are positive
                if let Type::Int(Int::Literal(value)) = &simplified
                    && value <= &0
                {
                    self.error(
                        errors,
                        arg.range(),
                        ErrorKind::InvalidAnnotation,
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

    pub fn parse_assert_shape_expr(
        &self,
        expr: &Expr,
        errors: &ErrorCollector,
    ) -> Option<IntTuple> {
        match expr {
            Expr::Tuple(ExprTuple { elts, .. }) => self
                .parse_dimension_list(elts, errors)
                .map(IntTuple::from_types),
            _ => {
                self.error(
                    errors,
                    expr.range(),
                    ErrorKind::BadArgumentType,
                    "Second argument to `assert_shape` must be a tuple of tensor dimensions"
                        .to_owned(),
                );
                None
            }
        }
    }

    fn proxy_method_subscript_infer(
        &self,
        cls: &Class,
        xs: &[Expr],
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        let target_ty = match xs {
            [Expr::StringLiteral(lit)] => match Lit::from_string_literal(lit) {
                Some(lit) => lit.to_explicit_type(),
                None => {
                    self.error(
                        errors,
                        lit.range(),
                        ErrorKind::InvalidAnnotation,
                        "`ProxyMethod` target must be a string literal".to_owned(),
                    );
                    self.heap.mk_any_error()
                }
            },
            [arg] => {
                self.error(
                    errors,
                    arg.range(),
                    ErrorKind::InvalidAnnotation,
                    "`ProxyMethod` target must be a string literal".to_owned(),
                );
                self.heap.mk_any_error()
            }
            _ => {
                self.error(
                    errors,
                    range,
                    ErrorKind::InvalidAnnotation,
                    "`ProxyMethod` requires exactly one string literal target".to_owned(),
                );
                self.heap.mk_any_error()
            }
        };
        self.heap
            .mk_type_of(self.specialize(cls, vec![target_ty], range, errors))
    }

    fn class_subscript_infer(
        &self,
        cls: &Class,
        slice: &Expr,
        xs: &[Expr],
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        let metadata = self.get_metadata_for_class(cls);
        let class_ty = Type::ClassDef(cls.dupe());
        let allow_dunder_lookup = self.get_class_tparams(cls).is_empty()
            && !metadata.has_base_any()
            && !metadata.is_new_type();
        let class_getitem_result = if allow_dunder_lookup {
            let class_ty = self.heap.mk_class_def(cls.dupe());
            // TODO(stroxler): Add a new API, similar to `type_of_attr_get` but returning a
            // LookupResult or an Optional type, that we could use here to avoid the double lookup.
            if self.has_attr(&class_ty, &dunder::CLASS_GETITEM) {
                Some(self.call_method_or_error(
                    &class_ty,
                    &dunder::CLASS_GETITEM,
                    range,
                    &[CallArg::expr(slice)],
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
        let metaclass_getitem_result = if class_getitem_result.is_none() && allow_dunder_lookup {
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
            let targs = self.parse_class_type_args(cls, xs, errors);
            self.heap
                .mk_type_of(self.specialize(cls, targs, range, errors))
        }
    }

    fn parse_class_type_args(
        &self,
        cls: &Class,
        args: &[Expr],
        errors: &ErrorCollector,
    ) -> Vec<Type> {
        let tparams = self.get_class_tparams(cls);
        self.parse_type_args_for_tparams(args, tparams.as_vec(), errors)
    }

    fn parse_type_args_for_tparams(
        &self,
        args: &[Expr],
        tparams_vec: &[Quantified],
        errors: &ErrorCollector,
    ) -> Vec<Type> {
        if !self.solver().tensor_shapes {
            return args.map(|arg| self.expr_untype(arg, TypeFormContext::TypeArgument, errors));
        }
        let variadic_idx = tparams_vec
            .iter()
            .position(|param| param.is_type_var_tuple());
        let int_type = self.stdlib.int().clone().to_type();
        let param_for_arg = |idx: usize| {
            if let Some(variadic_idx) = variadic_idx {
                let suffix_len = tparams_vec.len() - variadic_idx - 1;
                if idx < variadic_idx {
                    tparams_vec.get(idx)
                } else if idx + suffix_len < args.len() {
                    tparams_vec.get(variadic_idx)
                } else {
                    tparams_vec.get(tparams_vec.len() - (args.len() - idx))
                }
            } else {
                tparams_vec.get(idx)
            }
        };
        args.iter()
            .enumerate()
            .map(|(idx, arg)| {
                if let Some(param) = param_for_arg(idx) {
                    if !matches!(arg, Expr::Starred(_)) && param.kind() == QuantifiedKind::IntVar {
                        return self
                            .parse_dimension_expr(arg, errors)
                            .unwrap_or_else(Type::any_error);
                    }
                    if param.kind() == QuantifiedKind::TypeVar
                        && let Expr::List(ExprList { elts, .. }) = arg
                        && Self::is_int_tuple_carrier_bound(
                            &param.upper_bound(self.stdlib, self.heap),
                            &int_type,
                        )
                    {
                        return self
                            .parse_int_tuple_shape_args(elts, errors)
                            .map(|shape| shape.to_shape_arg_type())
                            .unwrap_or_else(Type::any_error);
                    }
                }
                self.expr_untype(arg, TypeFormContext::TypeArgument, errors)
            })
            .collect()
    }

    /// Returns whether `ty` is the normalized upper bound for a bare `IntTuple`
    /// carrier `TypeVar`.
    ///
    /// Other tuple bounds are ordinary type bounds and must not enable compact
    /// shape-list parsing.
    fn is_int_tuple_carrier_bound(ty: &Type, int_type: &Type) -> bool {
        match ty {
            Type::IntTuple(_) => true,
            Type::Tuple(Tuple::Unbounded(inner)) => inner.as_ref() == int_type,
            _ => false,
        }
    }

    /// Returns whether `ty` can legally be the argument inside `Elements[...]`.
    ///
    /// Valid carriers are concrete tuple types, type aliases (which normalize to
    /// tuples), and `TypeVar`s whose upper bound is an `IntTuple` (i.e., a tuple type).
    fn is_int_tuple_elements_carrier(&self, ty: &Type) -> bool {
        let upper_bound = match ty {
            Type::Tuple(_) | Type::IntTuple(_) | Type::UntypedAlias(_) => return true,
            Type::Quantified(q) if q.is_type_var() => q.upper_bound(self.stdlib, self.heap),
            Type::TypeVar(tv) => tv.upper_bound(self.stdlib, self.heap),
            _ => return false,
        };
        let int_type = self.stdlib.int().clone().to_type();
        Self::is_int_tuple_carrier_bound(&upper_bound, &int_type)
    }

    fn is_shape_elements_class(&self, cls: &Class) -> bool {
        cls.has_toplevel_qname("shape_extensions", "Elements")
    }

    /// Parse `Elements[S]` in `*Elements[S]`, returning the bare `S` carrier.
    ///
    /// `Elements` is the conceptual inverse of `tuple[Unpack[Ts]]`: whereas
    /// `tuple[Unpack[Ts]]` wraps a `TypeVarTuple` into a concrete tuple type,
    /// `Elements[S]` extracts the element sequence from an `IntTuple` carrier `S`.
    /// This fills a gap in the typing spec — there is no standard way to decompose
    /// a variadic carrier without a `TypeVarTuple` — letting callers write
    /// `Array[[*Elements[S], OUT], DType]` instead of needing a `TypeVarTuple`.
    fn parse_int_tuple_elements_projection(
        &self,
        value: &Expr,
        errors: &ErrorCollector,
    ) -> Result<Option<Type>, ()> {
        let Expr::Subscript(subscript) = value else {
            return Ok(None);
        };
        let base = self.expr_infer(&subscript.value, errors);
        let Type::ClassDef(ref cls) = base else {
            return Ok(None);
        };
        if !self.is_shape_elements_class(cls) {
            return Ok(None);
        }

        match Ast::unpack_slice(&subscript.slice) {
            [arg] => {
                let carrier = self.expr_untype(arg, TypeFormContext::TypeArgument, errors);
                match carrier {
                    Type::IntTuple(shape) => match shape.view() {
                        IntTupleView::Concrete(_) => Ok(Some(shape_to_tuple_carrier(&shape))),
                        IntTupleView::Gradual => Ok(Some(self.bare_int_tuple_carrier())),
                        IntTupleView::Unpacked { .. } => {
                            self.error(
                                errors,
                                arg.range(),
                                ErrorKind::InvalidAnnotation,
                                "`Elements[...]` only supports concrete `IntTuple[...]` values or shape carriers"
                                    .to_owned(),
                            );
                            Err(())
                        }
                    },
                    carrier if self.is_int_tuple_elements_carrier(&carrier) => Ok(Some(carrier)),
                    carrier => {
                        self.error(
                            errors,
                            arg.range(),
                            ErrorKind::InvalidAnnotation,
                            format!(
                                "`Elements[...]` requires an `IntTuple` carrier, got `{}`",
                                self.for_display(carrier)
                            ),
                        );
                        Err(())
                    }
                }
            }
            args => {
                self.error(
                    errors,
                    subscript.slice.range(),
                    ErrorKind::BadSpecialization,
                    format!(
                        "Expected 1 type argument for `Elements`, got {}",
                        args.len()
                    ),
                );
                Err(())
            }
        }
    }

    /// Return whether a tuple-carrier shape contains an unbounded tuple segment.
    fn has_unbounded_tuple_carrier(ty: &Type) -> bool {
        match ty {
            Type::Tuple(Tuple::Unbounded(_)) => true,
            Type::Tuple(Tuple::Unpacked(unpacked)) => {
                let (_, middle, _) = &**unpacked;
                Self::has_unbounded_tuple_carrier(middle)
            }
            Type::Unpack(inner) => Self::has_unbounded_tuple_carrier(inner),
            _ => false,
        }
    }

    fn parse_int_tuple_shape_args(
        &self,
        args: &[Expr],
        errors: &ErrorCollector,
    ) -> Option<IntTuple> {
        let star = args
            .iter()
            .enumerate()
            .find(|(_, arg)| matches!(arg, Expr::Starred(_)));

        if let Some((star_idx, Expr::Starred(ExprStarred { value, .. }))) = star {
            if let Some(second) = args[star_idx + 1..]
                .iter()
                .find(|arg| matches!(arg, Expr::Starred(_)))
            {
                self.error(
                    errors,
                    second.range(),
                    ErrorKind::InvalidAnnotation,
                    "`IntTuple` can have at most one unpacked shape carrier".to_owned(),
                );
                return None;
            }

            let prefix = self.parse_dimension_list(&args[..star_idx], errors)?;
            let suffix = self.parse_dimension_list(&args[star_idx + 1..], errors)?;
            let middle_ty = match self.parse_int_tuple_elements_projection(value, errors) {
                Ok(Some(middle_ty)) => middle_ty,
                Ok(None) => {
                    let got = self.expr_untype(value, TypeFormContext::TypeArgument, errors);
                    self.error(
                        errors,
                        value.range(),
                        ErrorKind::InvalidAnnotation,
                        format!(
                            "Unpacked type in `IntTuple` must use `Elements[...]`, got `{}`",
                            self.for_display(got)
                        ),
                    );
                    return None;
                }
                Err(()) => return None,
            };
            if let Type::Tuple(Tuple::Concrete(middle)) = middle_ty {
                let dims = prefix.into_iter().chain(middle).chain(suffix).collect();
                return Some(IntTuple::from_types(dims));
            }
            return Some(IntTuple::unpacked_from_types(prefix, middle_ty, suffix));
        }

        self.parse_dimension_list(args, errors)
            .map(IntTuple::from_types)
    }

    /// Parse a registered shaped-array annotation.
    ///
    /// The registered shape parameter is a single ordinary type argument that
    /// carries a tuple (e.g. `ndarray[Shape, DType]`). We specialize the class
    /// normally and project the carrier into a shape via
    /// `shaped_array_classtype_to_shaped_array_type`.
    fn parse_registered_shaped_array_type(
        &self,
        cls: &Class,
        args: &[Expr],
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        let shape_param = self
            .shaped_array_shape_for_class(cls)
            .expect("registered shaped-array class should have shape metadata");

        let tparams = self.get_class_tparams(cls);
        let shape_idx = tparams
            .iter()
            .position(|param| param == &shape_param)
            .expect("shaped-array metadata should refer to a class type parameter");
        match shape_param.kind() {
            QuantifiedKind::TypeVar | QuantifiedKind::IntVar => {}
            QuantifiedKind::TypeVarTuple => unreachable!(
                "shaped-array metadata validation rejects TypeVarTuple shape parameters"
            ),
            QuantifiedKind::ParamSpec => {
                unreachable!("shaped-array metadata validation rejects ParamSpec shape parameters")
            }
        }
        let validate_shape_slot = shape_idx < args.len() && args.len() <= tparams.len();
        let shape_param_accepts_int_tuple = matches!(
            shape_param.upper_bound(self.stdlib, self.heap),
            Type::IntTuple(_)
        );
        let shape_validation_arg = |carrier: &Type| {
            if shape_param_accepts_int_tuple {
                self.heap.mk_int_tuple(IntTuple::shapeless())
            } else {
                carrier.clone()
            }
        };
        let mut shape_arg_carrier = None;
        let class_targs = args
            .iter()
            .enumerate()
            .map(|(i, arg)| match arg {
                Expr::List(ExprList { elts, .. }) if i == shape_idx => {
                    match self.parse_int_tuple_shape_args(elts, errors) {
                        Some(shape) => {
                            let carrier = shape_to_tuple_carrier(&shape);
                            shape_arg_carrier = Some(shape.to_shape_arg_type());
                            shape_validation_arg(&carrier)
                        }
                        None => Type::any_error(),
                    }
                }
                _ => {
                    if i == shape_idx
                        && let Type::ClassDef(cls) = self.expr_infer(arg, &self.error_swallower())
                        && self.is_int_tuple_class(&cls)
                    {
                        let carrier = self.bare_int_tuple_carrier();
                        shape_arg_carrier = Some(IntTuple::shapeless().to_shape_arg_type());
                        shape_validation_arg(&carrier)
                    } else {
                        match self.expr_untype(arg, TypeFormContext::TypeArgument, errors) {
                            Type::IntTuple(shape) if i == shape_idx => {
                                let carrier = if shape.is_shapeless() {
                                    self.bare_int_tuple_carrier()
                                } else {
                                    shape_to_tuple_carrier(&shape)
                                };
                                shape_arg_carrier = Some(shape.to_shape_arg_type());
                                shape_validation_arg(&carrier)
                            }
                            ty => {
                                if validate_shape_slot
                                    && i == shape_idx
                                    && Self::has_unbounded_tuple_carrier(&ty)
                                {
                                    self.error(
                                        errors,
                                        arg.range(),
                                        ErrorKind::InvalidAnnotation,
                                        "Unbounded tuple types cannot be used as shaped-array shape carriers"
                                            .to_owned(),
                                    );
                                    Type::any_error()
                                } else if i == shape_idx && matches!(ty, Type::Tuple(_)) {
                                    if let Some(shape) = tuple_carrier_to_shape(&ty) {
                                        shape_arg_carrier = Some(shape.to_shape_arg_type());
                                        shape_validation_arg(&ty)
                                    } else {
                                        self.error(
                                            errors,
                                            arg.range(),
                                            ErrorKind::InvalidAnnotation,
                                            format!(
                                                "Invalid shaped-array shape carrier `{}`",
                                                self.for_display(ty)
                                            ),
                                        );
                                        Type::any_error()
                                    }
                                } else {
                                    ty
                                }
                            }
                        }
                    }
                }
            })
            .collect();
        let mut base_class =
            self.specialize_nontypeddict_to_classtype(cls, class_targs, range, errors);
        if let Some(carrier) = shape_arg_carrier
            && let Some(shape_arg) = base_class.targs_mut().as_mut().get_mut(shape_idx)
        {
            *shape_arg = carrier;
        }
        self.shaped_array_classtype_to_shaped_array_type(&base_class)
            .to_type()
    }

    fn parse_int_tuple_type(&self, args: &[Expr], errors: &ErrorCollector) -> Type {
        let Some(shape) = self.parse_int_tuple_shape_args(args, errors) else {
            return self.heap.mk_type_of(Type::any_error());
        };
        self.heap.mk_type_of(self.heap.mk_int_tuple(shape))
    }

    fn parse_single_int_type(
        &self,
        spelling: &str,
        args: &[Expr],
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        if args.len() != 1 {
            self.error(
                errors,
                range,
                ErrorKind::BadSpecialization,
                format!(
                    "Expected 1 type argument for `{}`, got {}",
                    spelling,
                    args.len()
                ),
            );
            return Type::any_error();
        }

        let Some(dims) = self.parse_dimension_list(args, errors) else {
            return Type::any_error();
        };
        let dim = dims.into_iter().next().expect(
            "parse_dimension_list returns a non-empty list for a single validated argument",
        );
        // `Dim[Any]`/`Size[Any]` desugar to plain `Any` since it's maximally gradual.
        if matches!(dim, Type::Any(_)) {
            return dim;
        }
        let Some(symint) = Int::from_type(&dim) else {
            unreachable!("Int::from_type failed on non-Any dimension: {:?}", dim);
        };
        let size = canonicalize(self.heap.mk_int(symint));
        self.heap.mk_type_of(size)
    }

    /// Parse Int[3], Int[N], Int[N+1] into `Type::Int(...)`.
    fn parse_int_type(&self, args: &[Expr], range: TextRange, errors: &ErrorCollector) -> Type {
        self.parse_single_int_type("Int", args, range, errors)
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
            ty if let Some(kind) = ty.to_func_kind() => Some(ConditionRedundantReason::Function(
                self.module().name(),
                kind.clone(),
            )),
            Type::ClassDef(cls) => Some(ConditionRedundantReason::Class(cls.name().clone())),
            Type::ClassType(ct) => {
                let cls = ct.class_object();
                // Skip warning for `object` itself and for abstract/protocol types:
                // a variable typed as `Hashable`, `Iterable`, etc. may hold a concrete
                // instance that defines `__bool__` or `__len__` at runtime.
                let metadata = self.get_metadata_for_class(cls);
                let is_abstract =
                    cls.is_builtin("object") || metadata.is_protocol() || metadata.extends_abc();
                // Skip warning for classes coming from stubs. Stub-only classes often have
                // dynamic runtime behavior (e.g. `datetime`, `asyncio.Future`, `Lock`,
                // sqlalchemy `Session`) that the stubs don't model, and the idiomatic
                // `if x:` None-guard pattern is widespread in real-world code.
                let is_from_stub = cls.module_path().is_interface();
                // Skip warning for dataclasses. These are commonly used as plain data
                // containers and `if obj:` is frequently a defensive pattern; the
                // warning would create excessive noise for little benefit.
                let is_dataclass = metadata.dataclass_metadata().is_some();
                if !is_abstract
                    && !is_from_stub
                    && !is_dataclass
                    && self.class_instances_always_truthy(cls)
                {
                    Some(ConditionRedundantReason::InstanceAlwaysTruthy(
                        cls.name().clone(),
                    ))
                } else {
                    None
                }
            }
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
                ErrorKind::RedundantCondition,
                format!("{reason}"),
            );
        }
    }
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    fn allocate_lambda_param_vars<'b>(
        &self,
        param_ids: &[(&'b Name, LambdaParamId)],
    ) -> Vec<(&'b Name, Var)> {
        param_ids
            .iter()
            .map(|(name, id)| {
                let var = self.solver().fresh_unwrap(self.uniques);
                self.set_lambda_param_var(*id, var);
                (*name, var)
            })
            .collect()
    }
}
