/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::mem;

use dupe::Dupe as _;
use pyrefly_graph::index::Idx;
use pyrefly_python::ast::Ast;
use pyrefly_python::docstring::Docstring;
use pyrefly_python::dunder;
use pyrefly_python::nesting_context::NestingContext;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::prelude::VecExt;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Decorator;
use ruff_python_ast::ExceptHandler;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::Identifier;
use ruff_python_ast::Parameters;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtExpr;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::StmtRaise;
use ruff_python_ast::StmtReturn;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;

use crate::binding::binding::AnnotationTarget;
use crate::binding::binding::Binding;
use crate::binding::binding::BindingAnnotation;
use crate::binding::binding::BindingDecoratedFunction;
use crate::binding::binding::BindingUndecoratedFunction;
use crate::binding::binding::BindingYield;
use crate::binding::binding::BindingYieldFrom;
use crate::binding::binding::FunctionStubOrImpl;
use crate::binding::binding::IsAsync;
use crate::binding::binding::Key;
use crate::binding::binding::KeyAnnotation;
use crate::binding::binding::KeyClass;
use crate::binding::binding::KeyDecorator;
use crate::binding::binding::KeyLegacyTypeParam;
use crate::binding::binding::KeyUndecoratedFunction;
use crate::binding::binding::LastStmt;
use crate::binding::binding::MethodSelfKind;
use crate::binding::binding::ReturnExplicit;
use crate::binding::binding::ReturnImplicit;
use crate::binding::binding::ReturnType;
use crate::binding::binding::ReturnTypeKind;
use crate::binding::bindings::BindingsBuilder;
use crate::binding::bindings::LegacyTParamCollector;
use crate::binding::expr::Usage;
use crate::binding::scope::FlowStyle;
use crate::binding::scope::InstanceAttribute;
use crate::binding::scope::Scope;
use crate::binding::scope::UnusedParameter;
use crate::binding::scope::UnusedVariable;
use crate::binding::scope::YieldsAndReturns;
use crate::config::base::UntypedDefBehavior;
use crate::export::special::SpecialExport;
use crate::types::types::Type;

struct Decorators {
    has_no_type_check: bool,
    is_overload: bool,
    is_abstract_method: bool,
    is_override: bool,
    is_classmethod: bool,
    decorators: Box<[Idx<KeyDecorator>]>,
}

pub struct SelfAssignments {
    pub method_name: Name,
    pub instance_attributes: SmallMap<Name, InstanceAttribute>,
}

/// Determine whether a function definition is annotated.
/// Used in the `untyped-def-behavior = "skip-and-infer-returns-any"` mode.
fn is_annotated<T>(returns: &Option<T>, params: &Parameters) -> bool {
    if returns.is_some() {
        return true;
    }
    for p in params.iter() {
        if p.annotation().is_some() {
            return true;
        }
    }
    false
}
struct SelfAttrNames<'a> {
    self_name: &'a Name,
    names: SmallMap<Name, TextRange>,
}

impl<'a> SelfAttrNames<'a> {
    fn expr_lvalue(&mut self, x: &Expr) {
        match x {
            Expr::Attribute(x) => {
                if let Expr::Name(v) = x.value.as_ref()
                    && &v.id == self.self_name
                    && !self.names.contains_key(&x.attr.id)
                {
                    self.names.insert(x.attr.id.clone(), x.attr.range());
                }
            }
            Expr::Tuple(x) => {
                for x in &x.elts {
                    self.expr_lvalue(x);
                }
            }

            Expr::List(x) => {
                for x in &x.elts {
                    self.expr_lvalue(x);
                }
            }
            _ => {}
        }
    }

    fn stmt(&mut self, x: &Stmt) {
        match x {
            Stmt::Assign(x) => {
                for e in x.targets.iter() {
                    self.expr_lvalue(e);
                }
            }
            Stmt::AnnAssign(x) => {
                self.expr_lvalue(x.target.as_ref());
            }
            _ => {}
        }
        x.recurse(&mut |x| self.stmt(x))
    }

    /// Given an unannotated method (it is the caller's responsibility to
    /// check these conditions), traverse the body to find the name and range
    /// of `self.<attr>` assignments.
    fn find(
        func_name: &Identifier,
        parameters: &mut Box<Parameters>,
        body: Vec<Stmt>,
    ) -> Option<SelfAssignments> {
        let self_name = if let Some(p) = parameters.iter_non_variadic_params().next() {
            &p.parameter.name.id
        } else {
            return None;
        };
        let mut finder = SelfAttrNames {
            self_name,
            names: SmallMap::new(),
        };
        for x in body.iter() {
            finder.stmt(x);
        }
        let instance_attributes = finder
            .names
            .into_iter()
            .map(|(n, r)| {
                (
                    n,
                    InstanceAttribute(
                        super::binding::ExprOrBinding::Binding(Binding::Type(Type::any_implicit())),
                        None,
                        r,
                        MethodSelfKind::Instance,
                    ),
                )
            })
            .collect();
        Some(SelfAssignments {
            method_name: func_name.id.clone(),
            instance_attributes,
        })
    }
}

impl<'a> BindingsBuilder<'a> {
    fn parameters(
        &mut self,
        x: &mut Parameters,
        undecorated_idx: Idx<KeyUndecoratedFunction>,
        class_key: Option<Idx<KeyClass>>,
        method_self_kind: MethodSelfKind,
    ) {
        let mut self_name = None;
        for x in x.iter_non_variadic_params() {
            if class_key.is_some() && self_name.is_none() {
                self_name = Some(x.parameter.name.clone());
            }
            self.bind_function_param(
                AnnotationTarget::Param(x.parameter.name.id.clone()),
                &x.parameter,
                undecorated_idx,
                class_key,
                false,
            );
        }
        if let Some(args) = &x.vararg {
            self.bind_function_param(
                AnnotationTarget::ArgsParam(args.name.id.clone()),
                args,
                undecorated_idx,
                class_key,
                true,
            );
        }
        if let Some(kwargs) = &x.kwarg {
            self.bind_function_param(
                AnnotationTarget::KwargsParam(kwargs.name.id.clone()),
                kwargs,
                undecorated_idx,
                class_key,
                true,
            );
        }
        self.scopes
            .set_self_name_if_applicable(self_name, method_self_kind);
    }

    fn to_return_annotation_with_range(
        &mut self,
        mut x: Expr,
        func_name: &Identifier,
        class_key: Option<Idx<KeyClass>>,
        tparams_builder: &mut Option<LegacyTParamCollector>,
    ) -> (TextRange, Idx<KeyAnnotation>) {
        self.ensure_type(&mut x, tparams_builder);
        (
            x.range(),
            self.insert_binding(
                KeyAnnotation::ReturnAnnotation(ShortIdentifier::new(func_name)),
                BindingAnnotation::AnnotateExpr(
                    AnnotationTarget::Return(func_name.id.clone()),
                    x,
                    class_key,
                ),
            ),
        )
    }

    fn function_header(
        &mut self,
        x: &mut StmtFunctionDef,
        func_name: &Identifier,
        class_key: Option<Idx<KeyClass>>,
        usage: &mut Usage,
    ) -> (
        Option<(TextRange, Idx<KeyAnnotation>)>,
        Vec<Idx<KeyLegacyTypeParam>>,
    ) {
        let tparams = x
            .type_params
            .as_mut()
            .map(|tparams| self.type_params(tparams));

        let mut legacy = Some(LegacyTParamCollector::new(tparams.is_some()));

        // We need to bind all the parameters expressions _after_ the type params, but before the parameter names,
        // which might shadow some types.
        for (param, default) in Ast::parameters_iter_mut(&mut x.parameters) {
            self.ensure_type_opt(param.annotation.as_deref_mut(), &mut legacy);
            if let Some(default) = default {
                self.ensure_expr_opt(default.as_deref_mut(), usage);
            }
        }

        let return_ann_with_range = mem::take(&mut x.returns)
            .map(|e| self.to_return_annotation_with_range(*e, func_name, class_key, &mut legacy));

        let legacy_tparam_collector = legacy.unwrap();
        self.add_name_definitions(&legacy_tparam_collector);
        let legacy_tparams = legacy_tparam_collector.lookup_keys();
        (return_ann_with_range, legacy_tparams)
    }

    /// Handle creating a scope and binding the function body.
    ///
    /// Note that are some aspects of function analysis (such as implicit return analysis) that also depend on the
    /// function body but are not handled here.
    fn function_body_scope(
        &mut self,
        parameters: &mut Box<Parameters>,
        body: Vec<Stmt>,
        range: TextRange,
        func_name: &Identifier,
        parent: &NestingContext,
        undecorated_idx: Idx<KeyUndecoratedFunction>,
        class_key: Option<Idx<KeyClass>>,
        is_async: bool,
        method_self_kind: MethodSelfKind,
    ) -> (
        YieldsAndReturns,
        Option<SelfAssignments>,
        Vec<UnusedParameter>,
        Vec<UnusedVariable>,
    ) {
        self.scopes
            .push_function_scope(range, func_name, class_key.is_some(), is_async);
        self.parameters(parameters, undecorated_idx, class_key, method_self_kind);
        self.init_static_scope(&body, false);
        self.stmts(
            body,
            &NestingContext::function(ShortIdentifier::new(func_name), parent.dupe()),
        );
        let (yields_and_returns, self_assignments, unused_parameters, unused_variables) =
            self.scopes.pop_function_scope();
        (
            yields_and_returns,
            self_assignments,
            unused_parameters,
            unused_variables,
        )
    }

    fn unchecked_function_body_scope(
        &mut self,
        parameters: &mut Box<Parameters>,
        body: Vec<Stmt>,
        range: TextRange,
        func_name: &Identifier,
        undecorated_idx: Idx<KeyUndecoratedFunction>,
        class_key: Option<Idx<KeyClass>>,
        is_async: bool,
        method_self_kind: MethodSelfKind,
    ) -> Option<SelfAssignments> {
        // Push a scope to create the parameter keys (but do nothing else with it).
        self.scopes
            .push_function_scope(range, func_name, class_key.is_some(), is_async);
        self.parameters(parameters, undecorated_idx, class_key, method_self_kind);
        self.scopes.pop();
        // If we are in a class, use a simple visitor to find `self.<attr>` assignments.
        if class_key.is_some() {
            SelfAttrNames::find(func_name, parameters, body)
        } else {
            None
        }
    }

    /// Compute a `Key::ReturnImplicit` / `Binding::ReturnImplicit` for the given function body.
    ///
    /// This function must not be called unless the function body statements will be bound;
    /// it relies on that binding to ensure we don't have a dangling `Idx<Key>` (which could lead
    /// to a panic).
    fn implicit_return(&mut self, body: &[Stmt], func_name: &Identifier) -> Idx<Key> {
        let last_exprs = function_last_expressions(body, self.sys_info).map(|x| {
            x.into_map(|(last, x)| (last, self.last_statement_idx_for_implicit_return(last, x)))
                .into_boxed_slice()
        });
        self.insert_binding(
            Key::ReturnImplicit(ShortIdentifier::new(func_name)),
            Binding::ReturnImplicit(ReturnImplicit { last_exprs }),
        )
    }

    /// Handles both checking yield / return expressions and binding the return type.
    fn analyze_return_type(
        &mut self,
        func_name: &Identifier,
        is_async: bool,
        yields_and_returns: YieldsAndReturns,
        return_ann_with_range: Option<(TextRange, Idx<KeyAnnotation>)>,
        implicit_return: Option<Idx<Key>>,
        should_infer_return_type: bool,
        stub_or_impl: FunctionStubOrImpl,
        decorators: Box<[Idx<KeyDecorator>]>,
    ) {
        let is_generator =
            !(yields_and_returns.yields.is_empty() && yields_and_returns.yield_froms.is_empty());
        let return_ann = return_ann_with_range.as_ref().map(|(_, key)| *key);

        // Collect the keys of explicit returns.
        let return_keys = yields_and_returns
            .returns
            .into_map(|(idx, x)| {
                self.insert_binding_idx(
                    idx,
                    Binding::ReturnExplicit(ReturnExplicit {
                        annot: return_ann,
                        expr: x.value,
                        is_generator,
                        is_async,
                        range: x.range,
                    }),
                )
            })
            .into_boxed_slice();

        // Collect the keys of yield expressions.
        let yield_keys = yields_and_returns
            .yields
            .into_map(|(idx, x)| self.insert_binding_idx(idx, BindingYield::Yield(return_ann, x)))
            .into_boxed_slice();
        let yield_from_keys = yields_and_returns
            .yield_froms
            .into_map(|(idx, x)| {
                self.insert_binding_idx(
                    idx,
                    BindingYieldFrom::YieldFrom(return_ann, IsAsync::new(is_async), x),
                )
            })
            .into_boxed_slice();

        let return_type_binding = {
            let kind = match (return_ann_with_range, implicit_return) {
                (Some((range, annotation)), Some(implicit_return)) => {
                    // We have an explicit return annotation and we want to validate it.
                    ReturnTypeKind::ShouldValidateAnnotation {
                        range,
                        annotation,
                        stub_or_impl,
                        decorators,
                        implicit_return,
                        is_generator: !(yield_keys.is_empty() && yield_from_keys.is_empty()),
                        has_explicit_return: !return_keys.is_empty(),
                    }
                }
                (Some((_, annotation)), None) => {
                    // We have an explicit return annotation and we just want to trust it.
                    ReturnTypeKind::ShouldTrustAnnotation {
                        annotation,
                        is_generator: !(yield_keys.is_empty() && yield_from_keys.is_empty()),
                    }
                }
                (None, Some(implicit_return)) if should_infer_return_type => {
                    // We don't have an explicit return annotation, but we want to infer it.
                    ReturnTypeKind::ShouldInferType {
                        returns: return_keys,
                        implicit_return,
                        yields: yield_keys,
                        yield_froms: yield_from_keys,
                    }
                }
                (None, _) => {
                    // We don't have an explicit return annotation, or we don't want to infer return type.
                    // Just treat the return type as `Any`.
                    ReturnTypeKind::ShouldReturnAny {
                        is_generator: !(yield_keys.is_empty() && yield_from_keys.is_empty()),
                    }
                }
            };
            Binding::ReturnType(Box::new(ReturnType { kind, is_async }))
        };
        self.insert_binding(
            Key::ReturnType(ShortIdentifier::new(func_name)),
            return_type_binding,
        );
    }

    fn mark_as_returns_any(&mut self, func_name: &Identifier) {
        self.insert_binding(
            Key::ReturnType(ShortIdentifier::new(func_name)),
            // TODO(grievejia): traverse the function body and calculate the `is_generator` flag, then
            // use ReturnTypeKind::ShouldReturnAny to get more precision here.
            Binding::Type(Type::any_implicit()),
        );
    }

    fn decorators(&mut self, decorator_list: Vec<Decorator>, usage: &mut Usage) -> Decorators {
        let mut is_overload = false;
        let mut is_override = false;
        let mut has_no_type_check = false;
        let mut is_abstract_method = false;
        let mut is_classmethod = false;
        for d in &decorator_list {
            let special_export = self.as_special_export(&d.expression);
            is_overload = is_overload || matches!(special_export, Some(SpecialExport::Overload));
            is_override = is_override || matches!(special_export, Some(SpecialExport::Override));
            is_abstract_method = is_abstract_method
                || matches!(
                    special_export,
                    Some(SpecialExport::AbstractMethod | SpecialExport::AbstractClassMethod)
                );
            has_no_type_check =
                has_no_type_check || matches!(special_export, Some(SpecialExport::NoTypeCheck));
            is_classmethod = is_classmethod
                || matches!(
                    special_export,
                    Some(SpecialExport::ClassMethod | SpecialExport::AbstractClassMethod)
                );
        }
        let decorators = self
            .ensure_and_bind_decorators(decorator_list, usage)
            .into_boxed_slice();
        Decorators {
            has_no_type_check,
            is_overload,
            is_abstract_method,
            is_override,
            is_classmethod,
            decorators,
        }
    }

    fn function_body(
        &mut self,
        parameters: &mut Box<Parameters>,
        body: Vec<Stmt>,
        decorators: &Decorators,
        range: TextRange,
        is_async: bool,
        return_ann_with_range: Option<(TextRange, Idx<KeyAnnotation>)>,
        func_name: &Identifier,
        parent: &NestingContext,
        undecorated_idx: Idx<KeyUndecoratedFunction>,
        class_key: Option<Idx<KeyClass>>,
    ) -> (FunctionStubOrImpl, Option<SelfAssignments>) {
        let stub_or_impl = if (body.first().is_some_and(is_docstring)
            && decorators.is_abstract_method)
            || is_ellipse(&body)
            || (body.first().is_some_and(is_docstring) && decorators.is_overload)
        {
            FunctionStubOrImpl::Stub
        } else {
            FunctionStubOrImpl::Impl
        };

        let body_is_trivial = match body.as_slice() {
            [Stmt::Pass(_)] => true,
            // raise NotImplementedError(...)
            [
                Stmt::Raise(StmtRaise {
                    exc: Some(box Expr::Call(ExprCall { box func, .. })),
                    ..
                }),
            ] if self.as_special_export(func) == Some(SpecialExport::NotImplementedError) => true,
            // return NotImplemented
            [
                Stmt::Return(StmtReturn {
                    value: Some(box val),
                    ..
                }),
            ] if self.as_special_export(val) == Some(SpecialExport::NotImplemented) => true,
            _ => false,
        };
        let should_report_unused_parameters = stub_or_impl == FunctionStubOrImpl::Impl
            && !body_is_trivial
            && !decorators.is_overload
            && !decorators.is_override
            && !decorators.is_abstract_method
            && !is_ellipse(&body);
        let method_self_kind = if class_key.is_some()
            && (decorators.is_classmethod
                || func_name.id == dunder::INIT_SUBCLASS
                || func_name.id == dunder::NEW)
        {
            MethodSelfKind::Class
        } else {
            MethodSelfKind::Instance
        };

        let self_assignments = if decorators.has_no_type_check
            || (self.untyped_def_behavior == UntypedDefBehavior::SkipAndInferReturnAny
                && !is_annotated(&return_ann_with_range, parameters))
        {
            self.mark_as_returns_any(func_name);
            self.unchecked_function_body_scope(
                parameters,
                body,
                range,
                func_name,
                undecorated_idx,
                class_key,
                is_async,
                method_self_kind,
            )
        } else {
            match self.untyped_def_behavior {
                UntypedDefBehavior::SkipAndInferReturnAny => {
                    let (yields_and_returns, self_assignments, unused_parameters, unused_variables) =
                        self.function_body_scope(
                            parameters,
                            body,
                            range,
                            func_name,
                            parent,
                            undecorated_idx,
                            class_key,
                            is_async,
                            method_self_kind,
                        );
                    if should_report_unused_parameters {
                        self.record_unused_parameters(unused_parameters);
                        self.record_unused_variables(unused_variables);
                    }
                    self.analyze_return_type(
                        func_name,
                        is_async,
                        yields_and_returns,
                        return_ann_with_range,
                        None,
                        false, // this disables return type inference
                        stub_or_impl,
                        decorators.decorators.clone(),
                    );
                    self_assignments
                }
                UntypedDefBehavior::CheckAndInferReturnAny => {
                    let implicit_return = self.implicit_return(&body, func_name);
                    let (yields_and_returns, self_assignments, unused_parameters, unused_variables) =
                        self.function_body_scope(
                            parameters,
                            body,
                            range,
                            func_name,
                            parent,
                            undecorated_idx,
                            class_key,
                            is_async,
                            method_self_kind,
                        );
                    if should_report_unused_parameters {
                        self.record_unused_parameters(unused_parameters);
                        self.record_unused_variables(unused_variables);
                    }
                    self.analyze_return_type(
                        func_name,
                        is_async,
                        yields_and_returns,
                        return_ann_with_range,
                        Some(implicit_return),
                        false, // this disables return type inference
                        stub_or_impl,
                        decorators.decorators.clone(),
                    );
                    self_assignments
                }
                UntypedDefBehavior::CheckAndInferReturnType => {
                    let implicit_return = self.implicit_return(&body, func_name);
                    let (yields_and_returns, self_assignments, unused_parameters, unused_variables) =
                        self.function_body_scope(
                            parameters,
                            body,
                            range,
                            func_name,
                            parent,
                            undecorated_idx,
                            class_key,
                            is_async,
                            method_self_kind,
                        );
                    if should_report_unused_parameters {
                        self.record_unused_parameters(unused_parameters);
                        self.record_unused_variables(unused_variables);
                    }
                    self.analyze_return_type(
                        func_name,
                        is_async,
                        yields_and_returns,
                        return_ann_with_range,
                        Some(implicit_return),
                        true,
                        stub_or_impl,
                        decorators.decorators.clone(),
                    );
                    self_assignments
                }
            }
        };

        (stub_or_impl, self_assignments)
    }

    pub fn function_def(&mut self, mut x: StmtFunctionDef, parent: &NestingContext) {
        let func_name = x.name.clone();
        let mut def_idx =
            self.declare_current_idx(Key::Definition(ShortIdentifier::new(&func_name)));

        let undecorated_idx =
            self.idx_for_promise(KeyUndecoratedFunction(ShortIdentifier::new(&func_name)));

        // Get preceding function definition, if any. Used for building an overload type.
        let (function_idx, pred_idx) = self.create_function_index(&func_name);

        let (class_key, metadata_key) = match self.scopes.current_class_and_metadata_keys() {
            Some((class_key, metadata_key)) => (Some(class_key), Some(metadata_key)),
            _ => (None, None),
        };

        self.scopes.push(Scope::annotation(x.range));
        let (return_ann_with_range, legacy_tparams) =
            self.function_header(&mut x, &func_name, class_key, def_idx.usage());

        let decorators = self.decorators(mem::take(&mut x.decorator_list), def_idx.usage());

        let docstring_range = Docstring::range_from_stmts(x.body.as_slice());
        let (stub_or_impl, self_assignments) = self.function_body(
            &mut x.parameters,
            mem::take(&mut x.body),
            &decorators,
            x.range,
            x.is_async,
            return_ann_with_range,
            &func_name,
            parent,
            undecorated_idx,
            class_key,
        );

        // Pop the annotation scope to get back to the parent scope, and handle this
        // case where we need to track assignments to `self` from methods.
        self.scopes.pop();
        self.scopes
            .record_self_assignments_if_applicable(self_assignments);
        let undecorated_idx = self.insert_binding_idx(
            undecorated_idx,
            BindingUndecoratedFunction {
                def: x,
                stub_or_impl,
                class_key,
                decorators: decorators.decorators,
                legacy_tparams: legacy_tparams.into_boxed_slice(),
                module_style: self.module_info.path().style(),
            },
        );

        self.insert_binding_idx(
            function_idx,
            BindingDecoratedFunction {
                undecorated_idx,
                successor: None,
                docstring_range,
            },
        );

        self.bind_current_as(
            &func_name,
            def_idx,
            Binding::Function(function_idx, pred_idx, metadata_key),
            FlowStyle::FunctionDef(function_idx, return_ann_with_range.is_some()),
        );
    }
}

/// Given the body of a function, what are the potential expressions that
/// could be the last ones to be executed, where the function then falls off the end.
///
/// * Return None to say there are branches that fall off the end always.
/// * Return Some([]) to say that we can never reach the end (e.g. always return, raise)
/// * Return Some(xs) to say this set might be the last expression.
fn function_last_expressions<'a>(
    x: &'a [Stmt],
    sys_info: &SysInfo,
) -> Option<Vec<(LastStmt, &'a Expr)>> {
    fn f<'a>(sys_info: &SysInfo, x: &'a [Stmt], res: &mut Vec<(LastStmt, &'a Expr)>) -> Option<()> {
        match x.last()? {
            Stmt::Expr(x) => res.push((LastStmt::Expr, &x.value)),
            Stmt::Return(_) | Stmt::Raise(_) => {}
            Stmt::Assert(x) if sys_info.evaluate_bool(&x.test) == Some(false) => {}
            Stmt::With(x) => {
                let kind = IsAsync::new(x.is_async);
                for y in &x.items {
                    res.push((LastStmt::With(kind), &y.context_expr));
                }
                f(sys_info, &x.body, res)?;
            }
            Stmt::While(x) => {
                // Infinite loops with no breaks cannot fall through
                if sys_info.evaluate_bool(&x.test) != Some(true) {
                    return None;
                }
                let mut has_break = false;
                fn f(stmt: &Stmt, res: &mut bool) {
                    match stmt {
                        Stmt::Break(_) => {
                            *res = true;
                        }
                        Stmt::While(_) | Stmt::For(_) => {}
                        _ => stmt.recurse(&mut |stmt| f(stmt, res)),
                    }
                }
                x.body.visit(&mut |stmt| f(stmt, &mut has_break));
                if has_break {
                    return None;
                }
            }
            Stmt::For(x) => {
                let mut has_break = false;
                fn f(stmt: &Stmt, res: &mut bool) {
                    match stmt {
                        Stmt::Break(_) => {
                            *res = true;
                        }
                        Stmt::While(_) | Stmt::For(_) => {}
                        _ => stmt.recurse(&mut |stmt| f(stmt, res)),
                    }
                }
                x.body.visit(&mut |stmt| f(stmt, &mut has_break));
                if has_break {
                    return None;
                }
            }
            Stmt::If(x) => {
                let mut last_test = None;
                for (test, body) in sys_info.pruned_if_branches(x) {
                    last_test = test;
                    f(sys_info, body, res)?;
                }
                if last_test.is_some() {
                    // The final `if` can fall through, so the `if` itself might be the last statement.
                    return None;
                }
            }
            Stmt::Try(x) => {
                // If final body is not empty, _and_ contains a return statement,
                // process it.
                if !x.finalbody.is_empty()
                    && x.finalbody
                        .iter()
                        .any(|stmt| matches!(stmt, Stmt::Return(_)))
                {
                    f(sys_info, &x.finalbody, res)?;
                } else {
                    if x.orelse.is_empty() {
                        f(sys_info, &x.body, res)?;
                    } else {
                        f(sys_info, &x.orelse, res)?;
                    }
                    for handler in &x.handlers {
                        match handler {
                            ExceptHandler::ExceptHandler(x) => f(sys_info, &x.body, res)?,
                        }
                    }
                    // If we don't have a matching handler, we raise an exception, which is fine.
                }
            }
            Stmt::Match(x) => {
                let mut exhaustive = false;
                for case in x.cases.iter() {
                    f(sys_info, &case.body, res)?;
                    if case.pattern.is_wildcard() || case.pattern.is_irrefutable() {
                        exhaustive = true;
                        break;
                    }
                }
                if !exhaustive {
                    return None;
                }
            }
            _ => return None,
        }
        Some(())
    }

    let mut res = Vec::new();
    f(sys_info, x, &mut res)?;
    Some(res)
}

fn is_docstring(x: &Stmt) -> bool {
    match x {
        Stmt::Expr(StmtExpr { value, .. }) => value.is_string_literal_expr(),
        _ => false,
    }
}

fn is_ellipse(x: &[Stmt]) -> bool {
    match x.iter().find(|x| !is_docstring(x)) {
        Some(Stmt::Expr(StmtExpr { value, .. })) => value.is_ellipsis_literal_expr(),
        _ => false,
    }
}
