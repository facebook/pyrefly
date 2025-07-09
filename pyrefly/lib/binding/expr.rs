/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::ast::Ast;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::Arguments;
use ruff_python_ast::AtomicNodeIndex;
use ruff_python_ast::BoolOp;
use ruff_python_ast::Comprehension;
use ruff_python_ast::Decorator;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprBoolOp;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprLambda;
use ruff_python_ast::ExprNoneLiteral;
use ruff_python_ast::ExprSubscript;
use ruff_python_ast::ExprYield;
use ruff_python_ast::ExprYieldFrom;
use ruff_python_ast::Identifier;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::Hashed;
use starlark_map::small_set::SmallSet;

use crate::binding::binding::Binding;
use crate::binding::binding::BindingYield;
use crate::binding::binding::BindingYieldFrom;
use crate::binding::binding::IsAsync;
use crate::binding::binding::Key;
use crate::binding::binding::KeyYield;
use crate::binding::binding::KeyYieldFrom;
use crate::binding::binding::LinkedKey;
use crate::binding::binding::SuperStyle;
use crate::binding::bindings::BindingsBuilder;
use crate::binding::bindings::LegacyTParamBuilder;
use crate::binding::bindings::LookupError;
use crate::binding::bindings::LookupKind;
use crate::binding::narrow::AtomicNarrowOp;
use crate::binding::narrow::NarrowOps;
use crate::binding::scope::Flow;
use crate::binding::scope::Scope;
use crate::binding::scope::ScopeClass;
use crate::binding::scope::ScopeKind;
use crate::error::kind::ErrorKind;
use crate::export::special::SpecialExport;
use crate::graph::index::Idx;
use crate::module::short_identifier::ShortIdentifier;
use crate::types::callable::unexpected_keyword;
use crate::types::types::Type;

/// Looking up names in an expression requires knowing the identity of the binding
/// we are computing for usage tracking.
///
/// There are some cases - particularly in type declaration contexts like annotations,
/// type variable declarations, and match patterns - that we want to skip for usage
/// tracking.
#[derive(Debug)]
pub enum Usage {
    /// I am a usage to create a `Binding`.
    CurrentIdx(Idx<Key>, SmallSet<Idx<Key>>),
    /// I am a usage that will appear in a narrowing operation (including a
    /// match pattern). We don't allow pinning in this case:
    /// - It is generally not useful (narrowing operations don't usually pin types)
    /// - Because narrowing introduces duplicate expressions, it is difficult
    ///   to ensure unpinned Vars cannot leak into the binding graph and cause
    ///   nondeterminism.
    Narrowing,
    /// I'm a usage in some context (a type variable declaration, an annotation,
    /// a cast, etc) where we are dealing with static types. I will not pin
    /// any placeholder types.
    StaticTypeInformation,
}

enum TestAssertion {
    AssertTrue,
    AssertFalse,
    AssertIsNone,
    AssertIsNotNone,
    AssertIsInstance,
    AssertNotIsInstance,
    AssertIs,
    AssertIsNot,
    AssertEqual,
    AssertNotEqual,
    AssertIn,
    AssertNotIn,
}

impl TestAssertion {
    pub fn to_narrow_ops(&self, builder: &BindingsBuilder, args: &[Expr]) -> Option<NarrowOps> {
        match self {
            Self::AssertTrue if let Some(arg0) = args.first() => {
                Some(NarrowOps::from_expr(builder, Some(arg0)))
            }
            Self::AssertFalse if let Some(arg0) = args.first() => {
                Some(NarrowOps::from_expr(builder, Some(arg0)).negate())
            }
            Self::AssertIsNone if let Some(arg0) = args.first() => {
                Some(NarrowOps::from_single_narrow_op(
                    arg0,
                    AtomicNarrowOp::Is(Expr::NoneLiteral(ExprNoneLiteral {
                        node_index: AtomicNodeIndex::dummy(),
                        range: TextRange::default(),
                    })),
                    arg0.range(),
                ))
            }
            Self::AssertIsNotNone if let Some(arg0) = args.first() => {
                Some(NarrowOps::from_single_narrow_op(
                    arg0,
                    AtomicNarrowOp::IsNot(Expr::NoneLiteral(ExprNoneLiteral {
                        node_index: AtomicNodeIndex::dummy(),
                        range: TextRange::default(),
                    })),
                    arg0.range(),
                ))
            }
            Self::AssertIsInstance
                if let Some(arg0) = args.first()
                    && let Some(arg1) = args.get(1) =>
            {
                Some(NarrowOps::from_single_narrow_op(
                    arg0,
                    AtomicNarrowOp::IsInstance(arg1.clone()),
                    arg0.range(),
                ))
            }
            Self::AssertNotIsInstance
                if let Some(arg0) = args.first()
                    && let Some(arg1) = args.get(1) =>
            {
                Some(NarrowOps::from_single_narrow_op(
                    arg0,
                    AtomicNarrowOp::IsNotInstance(arg1.clone()),
                    arg0.range(),
                ))
            }
            Self::AssertEqual
                if let Some(arg0) = args.first()
                    && let Some(arg1) = args.get(1) =>
            {
                Some(NarrowOps::from_single_narrow_op(
                    arg0,
                    AtomicNarrowOp::Eq(arg1.clone()),
                    arg0.range(),
                ))
            }
            Self::AssertNotEqual
                if let Some(arg0) = args.first()
                    && let Some(arg1) = args.get(1) =>
            {
                Some(NarrowOps::from_single_narrow_op(
                    arg0,
                    AtomicNarrowOp::NotEq(arg1.clone()),
                    arg0.range(),
                ))
            }
            Self::AssertIs
                if let Some(arg0) = args.first()
                    && let Some(arg1) = args.get(1) =>
            {
                Some(NarrowOps::from_single_narrow_op(
                    arg0,
                    AtomicNarrowOp::Is(arg1.clone()),
                    arg0.range(),
                ))
            }
            Self::AssertIsNot
                if let Some(arg0) = args.first()
                    && let Some(arg1) = args.get(1) =>
            {
                Some(NarrowOps::from_single_narrow_op(
                    arg0,
                    AtomicNarrowOp::IsNot(arg1.clone()),
                    arg0.range(),
                ))
            }
            Self::AssertIn
                if let Some(arg0) = args.first()
                    && let Some(arg1) = args.get(1) =>
            {
                Some(NarrowOps::from_single_narrow_op(
                    arg0,
                    AtomicNarrowOp::In(arg1.clone()),
                    arg0.range(),
                ))
            }
            Self::AssertNotIn
                if let Some(arg0) = args.first()
                    && let Some(arg1) = args.get(1) =>
            {
                Some(NarrowOps::from_single_narrow_op(
                    arg0,
                    AtomicNarrowOp::NotIn(arg1.clone()),
                    arg0.range(),
                ))
            }
            _ => None,
        }
    }
}

impl<'a> BindingsBuilder<'a> {
    /// Given a name appearing in an expression, create a `Usage` key for that
    /// name at the current location. The binding will indicate how to compute
    /// the type if we found that name in scope; if we do not find the name we
    /// record an error and fall back to `Any`.
    ///
    /// This function is the the core scope lookup logic for binding creation.
    pub fn ensure_name(
        &mut self,
        name: &Identifier,
        value: Result<Binding, LookupError>,
    ) -> Idx<Key> {
        let key = Key::BoundName(ShortIdentifier::new(name));
        if name.is_empty() {
            // We only get empty identifiers if Ruff has done error correction,
            // so there must be a parse error.
            //
            // Occasionally Ruff might give out the same Identifier twice in an error.
            //
            // We still need to produce a `Key` here just to be safe, because other
            // code may rely on all `Identifier`s having `Usage` keys and we could panic
            // in an IDE setting if we don't ensure this is the case.
            return self.insert_binding_overwrite(key, Binding::Type(Type::any_error()));
        }
        match value {
            Ok(value) => {
                if !self.module_info.path().is_interface() {
                    // Don't check flow for global/nonlocal lookups
                    if let Some(error_message) = self
                        .scopes
                        .get_flow_style(&name.id)
                        .uninitialized_error_message(name)
                    {
                        self.error(name.range, ErrorKind::UnboundName, None, error_message);
                    }
                }
                self.insert_binding(key, value)
            }
            Err(error) => {
                // Record a type error and fall back to `Any`.
                self.error(
                    name.range,
                    ErrorKind::UnknownName,
                    None,
                    error.message(name),
                );
                self.insert_binding(key, Binding::Type(Type::any_error()))
            }
        }
    }

    fn bind_comprehensions(
        &mut self,
        range: TextRange,
        comprehensions: &mut [Comprehension],
        usage: &mut Usage,
    ) {
        self.scopes.push(Scope::comprehension(range));
        for comp in comprehensions {
            // Resolve the type of the iteration value *before* binding the target of the iteration.
            // This is necessary so that, e.g. `[x for x in x]` correctly uses the outer scope for
            // the `in x` lookup.
            self.ensure_expr(&mut comp.iter, usage);
            let iterable_value_idx = self.insert_binding(
                Key::Anon(comp.iter.range()),
                Binding::IterableValue(None, comp.iter.clone(), IsAsync::new(comp.is_async)),
            );
            self.scopes.add_lvalue_to_current_static(&comp.target);
            // A comprehension target cannot be annotated, so it is safe to ignore the
            // annotation (which is None) and just use a `Forward` here.
            self.bind_target_no_expr(&mut comp.target, &|_ann_is_none| {
                Binding::Forward(iterable_value_idx)
            });
            for x in comp.ifs.iter_mut() {
                self.ensure_expr(x, &mut Usage::Narrowing);
                let narrow_ops = NarrowOps::from_expr(self, Some(x));
                self.bind_narrow_ops(&narrow_ops, comp.range);
            }
        }
    }

    pub fn bind_lambda(&mut self, lambda: &mut ExprLambda, usage: &mut Usage) {
        self.scopes.push(Scope::function(lambda.range));
        if let Some(parameters) = &lambda.parameters {
            for x in parameters {
                self.bind_lambda_param(x.name());
            }
        }
        self.ensure_expr(&mut lambda.body, usage);
        // Pyrefly currently does not support `yield` in lambdas, but we cannot drop them
        // entirely or we will panic at solve time.
        //
        // TODO: We should properly handle `yield` and `yield from`; lambdas can be generators.
        // One example of this is in the standard library, in `_collections_abc.pyi`:
        // https://github.com/python/cpython/blob/965662ee4a986605b60da470d9e7c1e9a6f922b3/Lib/_collections_abc.py#L92
        let (yields_and_returns, _) = self.scopes.pop_function_scope();
        for (idx, y) in yields_and_returns.yields {
            self.insert_binding_idx(idx, BindingYield::Invalid(y));
        }
        for (idx, y) in yields_and_returns.yield_froms {
            self.insert_binding_idx(idx, BindingYieldFrom::Invalid(y));
        }
    }

    /// Helper to clean up an expression that does type narrowing. We merge flows for the narrowing
    /// operation and its negation, so that narrowing is limited to the body of the expression but
    /// newly defined names persist.
    fn negate_and_merge_flow(
        &mut self,
        base: Flow,
        ops: &NarrowOps,
        orelse: Option<&mut Expr>,
        range: TextRange,
        usage: &mut Usage,
    ) {
        let if_branch = self.scopes.replace_current_flow(base);
        self.bind_narrow_ops(&ops.negate(), range);
        self.ensure_expr_opt(orelse, usage);
        // Swap them back again, to make sure that the merge order is if, then else
        let else_branch = self.scopes.replace_current_flow(if_branch);
        self.merge_branches_into_current(vec![else_branch], range);
    }

    fn enclosing_class_name(&self) -> Option<&Identifier> {
        for scope in self.scopes.iter_rev() {
            if let ScopeKind::Class(ScopeClass { name, .. }) = &scope.kind {
                return Some(name);
            }
        }
        None
    }

    // We want to special-case `self.assertXXX()` methods in unit tests.
    // The logic is intentionally syntax-based as we want to avoid checking whether the base type
    // is `unittest.TestCase` on every single method invocation.
    fn as_assert_in_test(&self, func: &Expr) -> Option<TestAssertion> {
        if let Some(class_name) = self.enclosing_class_name() {
            let class_name_str = class_name.as_str();
            if !(class_name_str.contains("test") || class_name_str.contains("Test")) {
                return None;
            }
            match func {
                Expr::Attribute(ExprAttribute { value, attr, .. })
                    if let Expr::Name(base_name) = &**value
                        && base_name.id.as_str() == "self" =>
                {
                    match attr.id.as_str() {
                        "assertTrue" => Some(TestAssertion::AssertTrue),
                        "assertFalse" => Some(TestAssertion::AssertFalse),
                        "assertIsNone" => Some(TestAssertion::AssertIsNone),
                        "assertIsNotNone" => Some(TestAssertion::AssertIsNotNone),
                        "assertIsInstance" => Some(TestAssertion::AssertIsInstance),
                        "assertNotIsInstance" => Some(TestAssertion::AssertNotIsInstance),
                        "assertIs" => Some(TestAssertion::AssertIs),
                        "assertIsNot" => Some(TestAssertion::AssertIsNot),
                        "assertEqual" => Some(TestAssertion::AssertEqual),
                        "assertNotEqual" => Some(TestAssertion::AssertNotEqual),
                        "assertIn" => Some(TestAssertion::AssertIn),
                        "assertNotIn" => Some(TestAssertion::AssertNotIn),
                        _ => None,
                    }
                }
                _ => None,
            }
        } else {
            None
        }
    }

    fn record_yield(&mut self, mut x: ExprYield) {
        let mut yield_link = self.declare_current_idx(Key::UsageLink(x.range));
        let idx = self.idx_for_promise(KeyYield(x.range));
        self.ensure_expr_opt(x.value.as_deref_mut(), yield_link.usage());
        if let Err(oops_top_level) = self.scopes.record_or_reject_yield(idx, x) {
            self.insert_binding_idx(idx, BindingYield::Invalid(oops_top_level));
        }
        self.insert_binding_current(yield_link, Binding::UsageLink(LinkedKey::Yield(idx)));
    }

    fn record_yield_from(&mut self, mut x: ExprYieldFrom) {
        let mut yield_from_link = self.declare_current_idx(Key::UsageLink(x.range));
        let idx = self.idx_for_promise(KeyYieldFrom(x.range));
        self.ensure_expr(&mut x.value, yield_from_link.usage());
        if let Err(oops_top_level) = self.scopes.record_or_reject_yield_from(idx, x) {
            self.insert_binding_idx(idx, BindingYieldFrom::Invalid(oops_top_level));
        }
        self.insert_binding_current(
            yield_from_link,
            Binding::UsageLink(LinkedKey::YieldFrom(idx)),
        );
    }

    /// Execute through the expr, ensuring every name has a binding.
    pub fn ensure_expr(&mut self, x: &mut Expr, usage: &mut Usage) {
        match x {
            Expr::If(x) => {
                // Ternary operation. We treat it like an if/else statement.
                let base = self.scopes.clone_current_flow();
                self.ensure_expr(&mut x.test, &mut Usage::Narrowing);
                let narrow_ops = NarrowOps::from_expr(self, Some(&x.test));
                self.bind_narrow_ops(&narrow_ops, x.body.range());
                self.ensure_expr(&mut x.body, usage);
                let range = x.range();
                self.negate_and_merge_flow(base, &narrow_ops, Some(&mut x.orelse), range, usage);
            }
            Expr::BoolOp(ExprBoolOp {
                node_index: _,
                range,
                op,
                values,
            }) => {
                let base = self.scopes.clone_current_flow();
                let mut narrow_ops = NarrowOps::new();
                for value in values {
                    self.bind_narrow_ops(&narrow_ops, value.range());
                    self.ensure_expr(value, &mut Usage::Narrowing);
                    let new_narrow_ops = NarrowOps::from_expr(self, Some(value));
                    match op {
                        BoolOp::And => {
                            // Every subsequent value is evaluated only if all previous values were truthy.
                            narrow_ops.and_all(new_narrow_ops);
                        }
                        BoolOp::Or => {
                            // Every subsequent value is evaluated only if all previous values were falsy.
                            narrow_ops.and_all(new_narrow_ops.negate());
                        }
                    }
                }
                self.negate_and_merge_flow(base, &narrow_ops, None, *range, usage);
            }
            Expr::Call(ExprCall {
                node_index: _,
                range: _,
                func,
                arguments,
            }) if self.as_special_export(func) == Some(SpecialExport::AssertType)
                && arguments.args.len() > 1 =>
            {
                // Handle forward references in the second argument to an assert_type call
                self.ensure_expr(func, usage);
                for (i, arg) in arguments.args.iter_mut().enumerate() {
                    if i == 1 {
                        self.ensure_type(arg, &mut None);
                    } else {
                        self.ensure_expr(arg, usage);
                    }
                }
                for kw in arguments.keywords.iter_mut() {
                    self.ensure_expr(&mut kw.value, usage);
                }
            }
            Expr::Call(ExprCall {
                node_index: _,
                range: _,
                func,
                arguments,
            }) if self.as_special_export(func) == Some(SpecialExport::Cast)
                && !arguments.is_empty() =>
            {
                // Handle forward references in the first argument to a cast call
                self.ensure_expr(func, usage);
                if let Some(arg) = arguments.args.first_mut() {
                    self.ensure_type(arg, &mut None)
                }
                for arg in arguments.args.iter_mut().skip(1) {
                    self.ensure_expr(arg, usage);
                }
                for kw in arguments.keywords.iter_mut() {
                    if let Some(id) = &kw.arg
                        && id.as_str() == "typ"
                    {
                        self.ensure_type(&mut kw.value, &mut None);
                    } else {
                        self.ensure_expr(&mut kw.value, usage);
                    }
                }
            }
            Expr::Call(ExprCall {
                node_index: _,
                range,
                func,
                arguments:
                    Arguments {
                        node_index: _,
                        range: _,
                        args: posargs,
                        keywords,
                    },
            }) if self.as_special_export(func) == Some(SpecialExport::Super) => {
                self.ensure_expr(func, usage);
                for kw in keywords {
                    self.ensure_expr(&mut kw.value, usage);
                    unexpected_keyword(
                        &|msg| self.error(*range, ErrorKind::UnexpectedKeyword, None, msg),
                        "super",
                        kw,
                    );
                }
                let nargs = posargs.len();
                let style = if nargs == 0 {
                    let mut method_name = None;
                    let mut class_key = None;
                    for scope in self.scopes.iter_rev() {
                        match &scope.kind {
                            ScopeKind::Method(method_scope) => {
                                method_name = Some(method_scope.name.clone());
                            }
                            ScopeKind::Class(class_scope) if method_name.is_some() => {
                                class_key = Some(class_scope.indices.class_idx);
                                break;
                            }
                            _ => {}
                        }
                    }
                    match (class_key, method_name) {
                        (Some(class_idx), Some(method)) => {
                            SuperStyle::ImplicitArgs(class_idx, method)
                        }
                        _ => {
                            self.error(
                                *range,
                                ErrorKind::InvalidSuperCall,
                                None,
                                "`super` call with no arguments is valid only inside a method"
                                    .to_owned(),
                            );
                            SuperStyle::Any
                        }
                    }
                } else if nargs == 2 {
                    let mut bind = |expr: &mut Expr| {
                        self.ensure_expr(expr, usage);
                        self.insert_binding(
                            Key::Anon(expr.range()),
                            Binding::Expr(None, expr.clone()),
                        )
                    };
                    let cls_key = bind(&mut posargs[0]);
                    let obj_key = bind(&mut posargs[1]);
                    SuperStyle::ExplicitArgs(cls_key, obj_key)
                } else {
                    if nargs != 1 {
                        // Calling super() with one argument is technically legal: https://stackoverflow.com/a/30190341.
                        // This is a very niche use case, and we don't support it aside from not erroring.
                        self.error(
                            *range,
                            ErrorKind::InvalidSuperCall,
                            None,
                            format!("`super` takes at most 2 arguments, got {}", nargs),
                        );
                    }
                    for arg in posargs {
                        self.ensure_expr(arg, usage);
                    }
                    SuperStyle::Any
                };
                self.insert_binding(
                    Key::SuperInstance(*range),
                    Binding::SuperInstance(style, *range),
                );
            }
            Expr::Call(ExprCall {
                node_index: _,
                range,
                func,
                arguments,
            }) if let Some(test_assert) = self.as_assert_in_test(func)
                && let Some(narrow_op) = test_assert.to_narrow_ops(self, &arguments.args) =>
            {
                self.ensure_expr(func, usage);
                for arg in arguments.args.iter_mut() {
                    self.ensure_expr(arg, &mut Usage::Narrowing);
                }
                for kw in arguments.keywords.iter_mut() {
                    self.ensure_expr(&mut kw.value, usage);
                }
                self.bind_narrow_ops(&narrow_op, *range);
            }
            Expr::Named(x) => {
                // For scopes defined in terms of Definitions, we should normally already have the name in Static, but
                // we still need this for comprehensions, whose scope is defined on-the-fly.
                self.scopes.add_lvalue_to_current_static(&x.target);
                self.bind_target_with_expr(&mut x.target, &mut x.value, &|expr, ann| {
                    Binding::Expr(ann, expr.clone())
                });
            }
            Expr::Lambda(x) => {
                self.bind_lambda(x, usage);
            }
            Expr::ListComp(x) => {
                self.bind_comprehensions(x.range, &mut x.generators, usage);
                self.ensure_expr(&mut x.elt, usage);
                self.scopes.pop();
            }
            Expr::SetComp(x) => {
                self.bind_comprehensions(x.range, &mut x.generators, usage);
                self.ensure_expr(&mut x.elt, usage);
                self.scopes.pop();
            }
            Expr::DictComp(x) => {
                self.bind_comprehensions(x.range, &mut x.generators, usage);
                self.ensure_expr(&mut x.key, usage);
                self.ensure_expr(&mut x.value, usage);
                self.scopes.pop();
            }
            Expr::Generator(x) => {
                self.bind_comprehensions(x.range, &mut x.generators, usage);
                self.ensure_expr(&mut x.elt, usage);
                self.scopes.pop();
            }
            Expr::Call(ExprCall { func, .. })
                if matches!(
                    self.as_special_export(func),
                    Some(SpecialExport::Exit | SpecialExport::Quit | SpecialExport::OsExit)
                ) =>
            {
                x.recurse_mut(&mut |x| self.ensure_expr(x, usage));
                // Control flow doesn't proceed after sys.exit(), exit(), quit(), or os._exit().
                self.scopes.mark_flow_termination();
            }
            Expr::Name(x) => {
                let name = Ast::expr_name_identifier(x.clone());
                let binding = self
                    .lookup_name_usage(Hashed::new(&name.id), usage)
                    .map(Binding::Forward);
                self.ensure_name(&name, binding);
            }
            Expr::Yield(x) => {
                self.record_yield(x.clone());
            }
            Expr::YieldFrom(x) => {
                self.record_yield_from(x.clone());
            }
            _ => {
                x.recurse_mut(&mut |x| self.ensure_expr(x, usage));
            }
        }
    }

    /// Execute through the expr, ensuring every name has a binding.
    pub fn ensure_expr_opt(&mut self, x: Option<&mut Expr>, usage: &mut Usage) {
        if let Some(x) = x {
            self.ensure_expr(x, usage);
        }
    }

    /// Execute through the expr, ensuring every name has a binding.
    pub fn ensure_type(&mut self, x: &mut Expr, tparams_builder: &mut Option<LegacyTParamBuilder>) {
        // We do not treat static types as usage for the purpose of first-usage-based type inference.
        let static_type_usage = &mut Usage::StaticTypeInformation;
        match x {
            Expr::Name(x) => {
                let name = Ast::expr_name_identifier(x.clone());
                let binding = match tparams_builder {
                    Some(legacy) => legacy
                        .intercept_lookup(self, &name)
                        .ok_or(LookupError::NotFound),
                    None => self
                        .lookup_name(Hashed::new(&name.id), LookupKind::Regular)
                        .map(Binding::Forward),
                };
                self.ensure_name(&name, binding);
            }
            Expr::Subscript(ExprSubscript { value, .. })
                if self.as_special_export(value) == Some(SpecialExport::Literal) =>
            {
                // Don't go inside a literal, since you might find strings which are really strings, not string-types
                self.ensure_expr(x, static_type_usage);
            }
            Expr::Subscript(ExprSubscript { value, slice, .. })
                if self.as_special_export(value) == Some(SpecialExport::Annotated)
                    && matches!(&**slice, Expr::Tuple(tup) if !tup.is_empty()) =>
            {
                // Only go inside the first argument to Annotated, the rest are non-type metadata.
                self.ensure_type(&mut *value, tparams_builder);
                // We can't bind a mut box in the guard (sadly), so force unwrapping it here
                let tup = slice.as_tuple_expr_mut().unwrap();
                self.ensure_type(&mut tup.elts[0], tparams_builder);
                for e in tup.elts[1..].iter_mut() {
                    self.ensure_expr(e, static_type_usage);
                }
            }
            Expr::StringLiteral(literal) if let Some(literal) = literal.as_single_part_string() => {
                match Ast::parse_type_literal(literal) {
                    Ok(expr) => {
                        *x = expr;
                        // TODO: Remember if we have already done a parse_type_literal, so we could properly
                        // reject annotations of the form `"'T'"`.
                        self.ensure_type(x, tparams_builder);
                    }
                    Err(e) => {
                        self.error(
                            literal.range,
                            ErrorKind::ParseError,
                            None,
                            format!("Could not parse type string: {}, got {e}", literal.value),
                        );
                    }
                }
            }
            // Bind the lambda so we don't crash on undefined parameter names.
            Expr::Lambda(_) => self.ensure_expr(x, static_type_usage),
            // Bind the call so we generate all expected bindings. See
            // test::class_super::test_super_in_base_classes for an example of a SuperInstance
            // binding that we crash looking for if we don't do this.
            Expr::Call(_) => self.ensure_expr(x, static_type_usage),
            _ => x.recurse_mut(&mut |x| self.ensure_type(x, tparams_builder)),
        }
    }

    /// Execute through the expr, ensuring every name has a binding.
    pub fn ensure_type_opt(
        &mut self,
        x: Option<&mut Expr>,
        tparams_builder: &mut Option<LegacyTParamBuilder>,
    ) {
        if let Some(x) = x {
            self.ensure_type(x, tparams_builder);
        }
    }

    pub fn ensure_and_bind_decorators(
        &mut self,
        decorators: Vec<Decorator>,
        usage: &mut Usage,
    ) -> Vec<Idx<Key>> {
        let mut decorator_keys = Vec::with_capacity(decorators.len());
        for mut x in decorators {
            self.ensure_expr(&mut x.expression, usage);
            let k = self.insert_binding(Key::Anon(x.range), Binding::Decorator(x.expression));
            decorator_keys.push(k);
        }
        decorator_keys
    }

    pub fn ensure_and_bind_decorators_with_ranges(
        &mut self,
        decorators: Vec<Decorator>,
        usage: &mut Usage,
    ) -> Vec<(Idx<Key>, TextRange)> {
        let mut decorator_keys_with_ranges = Vec::with_capacity(decorators.len());
        for mut x in decorators {
            self.ensure_expr(&mut x.expression, usage);
            let range = x.range();
            let k = self.insert_binding(Key::Anon(x.range), Binding::Decorator(x.expression));
            decorator_keys_with_ranges.push((k, range));
        }
        decorator_keys_with_ranges
    }
}
