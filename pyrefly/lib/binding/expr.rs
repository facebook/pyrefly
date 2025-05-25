/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::mem;

use itertools::Either;
use ruff_python_ast::Arguments;
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
use ruff_python_ast::Identifier;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::binding::binding::Binding;
use crate::binding::binding::IsAsync;
use crate::binding::binding::Key;
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
use crate::dunder;
use crate::error::kind::ErrorKind;
use crate::export::special::SpecialExport;
use crate::graph::index::Idx;
use crate::module::short_identifier::ShortIdentifier;
use crate::ruff::ast::Ast;
use crate::types::callable::unexpected_keyword;
use crate::types::types::Type;
use crate::util::visit::VisitMut;

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
                        range: TextRange::default(),
                    })),
                    arg0.range(),
                ))
            }
            Self::AssertIsNotNone if let Some(arg0) = args.first() => {
                Some(NarrowOps::from_single_narrow_op(
                    arg0,
                    AtomicNarrowOp::IsNot(Expr::NoneLiteral(ExprNoneLiteral {
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
        let key = Key::Usage(ShortIdentifier::new(name));
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
                        self.error(name.range, error_message, ErrorKind::UnboundName);
                    }
                }
                self.insert_binding(key, value)
            }
            Err(_) if name.id == dunder::FILE || name.id == dunder::NAME => {
                self.insert_binding(key, Binding::StrType)
            }
            Err(_) if name.id == dunder::DEBUG => self.insert_binding(key, Binding::BoolType),
            Err(_) if name.id == dunder::DOC => self.insert_binding(
                key,
                if self.has_docstring {
                    Binding::StrType
                } else {
                    Binding::Type(Type::None)
                },
            ),
            Err(error) => {
                // Record a type error and fall back to `Any`.
                self.error(name.range, error.message(name), ErrorKind::UnknownName);
                self.insert_binding(key, Binding::Type(Type::any_error()))
            }
        }
    }

    fn bind_comprehensions(&mut self, range: TextRange, comprehensions: &mut [Comprehension]) {
        self.scopes.push(Scope::comprehension(range));
        for comp in comprehensions {
            self.scopes.current_mut().stat.expr_lvalue(&comp.target);
            let make_binding =
                |ann| Binding::IterableValue(ann, comp.iter.clone(), IsAsync::new(comp.is_async));
            self.bind_target(&mut comp.target, &make_binding);
            for x in comp.ifs.iter() {
                let narrow_ops = NarrowOps::from_expr(self, Some(x));
                self.bind_narrow_ops(&narrow_ops, comp.range);
            }
        }
    }

    pub fn bind_lambda(&mut self, lambda: &ExprLambda) {
        self.scopes.push(Scope::function(lambda.range));
        if let Some(parameters) = &lambda.parameters {
            for x in parameters {
                self.bind_lambda_param(x.name());
            }
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
    ) {
        let if_branch = mem::take(&mut self.scopes.current_mut().flow);
        self.scopes.current_mut().flow = base;
        self.bind_narrow_ops(&ops.negate(), range);
        self.ensure_expr_opt(orelse);
        let else_branch = mem::take(&mut self.scopes.current_mut().flow);
        self.scopes.current_mut().flow = self.merge_flow(vec![if_branch, else_branch], range);
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
                Expr::Attribute(ExprAttribute {
                    value: box Expr::Name(base_name),
                    attr,
                    ..
                }) if base_name.id.as_str() == "self" => match attr.id.as_str() {
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
                },
                _ => None,
            }
        } else {
            None
        }
    }

    /// Execute through the expr, ensuring every name has a binding.
    pub fn ensure_expr(&mut self, x: &mut Expr) {
        let new_scope = match x {
            Expr::If(x) => {
                // Ternary operation. We treat it like an if/else statement.
                let base = self.scopes.current().flow.clone();
                self.ensure_expr(&mut x.test);
                let narrow_ops = NarrowOps::from_expr(self, Some(&x.test));
                self.bind_narrow_ops(&narrow_ops, x.body.range());
                self.ensure_expr(&mut x.body);
                let range = x.range();
                self.negate_and_merge_flow(base, &narrow_ops, Some(&mut x.orelse), range);
                return;
            }
            Expr::BoolOp(ExprBoolOp { range, op, values }) => {
                let base = self.scopes.current().flow.clone();
                let mut narrow_ops = NarrowOps::new();
                for value in values {
                    self.bind_narrow_ops(&narrow_ops, value.range());
                    self.ensure_expr(value);
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
                self.negate_and_merge_flow(base, &narrow_ops, None, *range);
                return;
            }
            Expr::Call(ExprCall {
                range: _,
                func,
                arguments,
            }) if self.as_special_export(func) == Some(SpecialExport::AssertType)
                && arguments.args.len() > 1 =>
            {
                // Handle forward references in the second argument to an assert_type call
                self.ensure_expr(func);
                for (i, arg) in arguments.args.iter_mut().enumerate() {
                    if i == 1 {
                        self.ensure_type(arg, &mut None);
                    } else {
                        self.ensure_expr(arg);
                    }
                }
                for kw in arguments.keywords.iter_mut() {
                    self.ensure_expr(&mut kw.value);
                }
                return;
            }
            Expr::Call(ExprCall {
                range: _,
                func,
                arguments,
            }) if self.as_special_export(func) == Some(SpecialExport::Cast)
                && !arguments.is_empty() =>
            {
                // Handle forward references in the first argument to a cast call
                self.ensure_expr(func);
                if let Some(arg) = arguments.args.first_mut() {
                    self.ensure_type(arg, &mut None)
                }
                for arg in arguments.args.iter_mut().skip(1) {
                    self.ensure_expr(arg);
                }
                for kw in arguments.keywords.iter_mut() {
                    if let Some(id) = &kw.arg
                        && id.as_str() == "typ"
                    {
                        self.ensure_type(&mut kw.value, &mut None);
                    } else {
                        self.ensure_expr(&mut kw.value);
                    }
                }
                return;
            }
            Expr::Call(ExprCall {
                range,
                func,
                arguments:
                    Arguments {
                        range: _,
                        args: posargs,
                        keywords,
                    },
            }) if self.as_special_export(func) == Some(SpecialExport::Super) => {
                self.ensure_expr(func);
                for kw in keywords {
                    self.ensure_expr(&mut kw.value);
                    unexpected_keyword(
                        &|msg| self.error(*range, msg, ErrorKind::UnexpectedKeyword),
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
                                "`super` call with no arguments is valid only inside a method"
                                    .to_owned(),
                                ErrorKind::InvalidSuperCall,
                            );
                            SuperStyle::Any
                        }
                    }
                } else if nargs == 2 {
                    let mut bind = |expr: &mut Expr| {
                        self.ensure_expr(expr);
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
                            format!("`super` takes at most 2 arguments, got {}", nargs),
                            ErrorKind::InvalidSuperCall,
                        );
                    }
                    for arg in posargs {
                        self.ensure_expr(arg);
                    }
                    SuperStyle::Any
                };
                self.insert_binding(
                    Key::SuperInstance(*range),
                    Binding::SuperInstance(style, *range),
                );
                return;
            }
            Expr::Call(ExprCall {
                range,
                func,
                arguments,
            }) if let Some(test_assert) = self.as_assert_in_test(func)
                && let Some(narrow_op) = test_assert.to_narrow_ops(self, &arguments.args) =>
            {
                self.ensure_expr(func);
                for arg in arguments.args.iter_mut() {
                    self.ensure_expr(arg);
                }
                for kw in arguments.keywords.iter_mut() {
                    self.ensure_expr(&mut kw.value);
                }
                self.bind_narrow_ops(&narrow_op, *range);
                return;
            }
            Expr::Named(x) => {
                self.scopes.current_mut().stat.expr_lvalue(&x.target);
                let make_binding = |ann| Binding::Expr(ann, (*x.value).clone());
                self.bind_target(&mut x.target, &make_binding);
                self.ensure_expr(&mut x.value);
                return;
            }
            Expr::Call(ExprCall {
                range: _,
                func,
                arguments: _,
            }) if matches!(
                self.as_special_export(func),
                Some(SpecialExport::Exit | SpecialExport::Quit | SpecialExport::OsExit)
            ) =>
            {
                // Control flow doesn't proceed after sys.exit(), exit(), quit(), or os._exit().
                self.scopes.current_mut().flow.no_next = true;
                false
            }
            Expr::Name(x) => {
                let name = Ast::expr_name_identifier(x.clone());
                let binding = self
                    .lookup_name(&name.id, LookupKind::Regular)
                    .map(Binding::Forward);
                self.ensure_name(&name, binding);
                false
            }
            Expr::ListComp(x) => {
                self.bind_comprehensions(x.range, &mut x.generators);
                true
            }
            Expr::SetComp(x) => {
                self.bind_comprehensions(x.range, &mut x.generators);
                true
            }
            Expr::DictComp(x) => {
                self.bind_comprehensions(x.range, &mut x.generators);
                true
            }
            Expr::Generator(x) => {
                self.bind_comprehensions(x.range, &mut x.generators);
                true
            }
            Expr::Lambda(x) => {
                self.bind_lambda(x);
                true
            }
            Expr::Yield(x) => {
                self.function_yields_and_returns
                    .last_mut()
                    .yields
                    .push(Either::Left(x.clone()));
                false
            }
            Expr::YieldFrom(x) => {
                self.function_yields_and_returns
                    .last_mut()
                    .yields
                    .push(Either::Right(x.clone()));
                false
            }
            _ => false,
        };
        x.recurse_mut(&mut |x| self.ensure_expr(x));
        if new_scope {
            self.scopes.pop();
        }
    }

    /// Execute through the expr, ensuring every name has a binding.
    pub fn ensure_expr_opt(&mut self, x: Option<&mut Expr>) {
        if let Some(x) = x {
            self.ensure_expr(x);
        }
    }

    /// Execute through the expr, ensuring every name has a binding.
    pub fn ensure_type(&mut self, x: &mut Expr, tparams_builder: &mut Option<LegacyTParamBuilder>) {
        match x {
            Expr::Name(x) => {
                let name = Ast::expr_name_identifier(x.clone());
                let binding = match tparams_builder {
                    Some(legacy) => legacy
                        .intercept_lookup(self, &name)
                        .ok_or(LookupError::NotFound),
                    None => self
                        .lookup_name(&name.id, LookupKind::Regular)
                        .map(Binding::Forward),
                };
                self.ensure_name(&name, binding);
            }
            Expr::Subscript(ExprSubscript { value, .. })
                if self.as_special_export(value) == Some(SpecialExport::Literal) =>
            {
                // Don't go inside a literal, since you might find strings which are really strings, not string-types
                self.ensure_expr(x);
            }
            Expr::Subscript(ExprSubscript {
                value,
                slice: box Expr::Tuple(tup),
                ..
            }) if self.as_special_export(value) == Some(SpecialExport::Annotated)
                && !tup.is_empty() =>
            {
                // Only go inside the first argument to Annotated, the rest are non-type metadata.
                self.ensure_type(&mut *value, tparams_builder);
                self.ensure_type(&mut tup.elts[0], tparams_builder);
                for e in tup.elts[1..].iter_mut() {
                    self.ensure_expr(e);
                }
            }
            Expr::StringLiteral(literal) => match Ast::parse_type_literal(literal) {
                Ok(expr) => {
                    *x = expr;
                    // TODO: Remember if we have already done a parse_type_literal, so we could properly
                    // reject annotations of the form `"'T'"`.
                    self.ensure_type(x, tparams_builder);
                }
                Err(e) => {
                    self.error(
                        literal.range,
                        format!(
                            "Could not parse type string: {}, got {e}",
                            literal.value.to_str()
                        ),
                        ErrorKind::ParseError,
                    );
                }
            },
            // Bind the lambda so we don't crash on undefined parameter names.
            Expr::Lambda(_) => self.ensure_expr(x),
            // Bind the call so we generate all expected bindings. See
            // test::class_super::test_super_in_base_classes for an example of a SuperInstance
            // binding that we crash looking for if we don't do this.
            Expr::Call(_) => self.ensure_expr(x),
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

    pub fn ensure_and_bind_decorators(&mut self, decorators: Vec<Decorator>) -> Vec<Idx<Key>> {
        let mut decorator_keys = Vec::with_capacity(decorators.len());
        for mut x in decorators {
            self.ensure_expr(&mut x.expression);
            let k = self.insert_binding(Key::Anon(x.range), Binding::Decorator(x.expression));
            decorator_keys.push(k);
        }
        decorator_keys
    }
}
