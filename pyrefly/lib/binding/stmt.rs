/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_graph::index::Idx;
use pyrefly_python::ast::Ast;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::nesting_context::NestingContext;
use pyrefly_python::short_identifier::ShortIdentifier;
use ruff_python_ast::Arguments;
use ruff_python_ast::AtomicNodeIndex;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprList;
use ruff_python_ast::ExprName;
use ruff_python_ast::ExprNumberLiteral;
use ruff_python_ast::ExprSet;
use ruff_python_ast::ExprTuple;
use ruff_python_ast::Identifier;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtAssign;
use ruff_python_ast::StmtExpr;
use ruff_python_ast::StmtImportFrom;
use ruff_python_ast::StmtReturn;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::small_set::SmallSet;

use crate::binding::binding::AnnAssignHasValue;
use crate::binding::binding::AnnotationTarget;
use crate::binding::binding::Binding;
use crate::binding::binding::BindingAnnotation;
use crate::binding::binding::BindingExpect;
use crate::binding::binding::ExprOrBinding;
use crate::binding::binding::IsAsync;
use crate::binding::binding::Key;
use crate::binding::binding::KeyAnnotation;
use crate::binding::binding::KeyExpect;
use crate::binding::binding::LinkedKey;
use crate::binding::binding::NarrowUseLocation;
use crate::binding::binding::RaisedException;
use crate::binding::bindings::BindingsBuilder;
use crate::binding::expr::Usage;
use crate::binding::narrow::NarrowOps;
use crate::binding::scope::FlowStyle;
use crate::binding::scope::LoopExit;
use crate::binding::scope::Scope;
use crate::config::error_kind::ErrorKind;
use crate::error::context::ErrorInfo;
use crate::export::definitions::MutableCaptureKind;
use crate::export::exports::Export;
use crate::export::exports::ExportLocation;
use crate::export::exports::Exports;
use crate::export::special::SpecialExport;
use crate::state::loader::FindError;
use crate::state::loader::FindingOrError;
use crate::types::alias::resolve_typeshed_alias;
use crate::types::special_form::SpecialForm;
use crate::types::types::Type;

/// Checks if an iterable expression is guaranteed to be non-empty and thus
/// the for-loop body will definitely execute at least once.
///
/// Returns true for:
/// - `range(N)` where N is a positive integer literal
/// - Non-empty list literals like `[1, 2, 3]`
/// - Non-empty tuple literals like `(1, 2, 3)`
/// - Non-empty set literals like `{1, 2, 3}`
fn is_definitely_nonempty_iterable(iter: &Expr) -> bool {
    match iter {
        // Check for range(N) where N is a positive integer literal
        Expr::Call(ExprCall {
            func, arguments, ..
        }) => {
            // Check if the function is `range` with a single argument and no keywords
            if let Expr::Name(ExprName { id, .. }) = &**func
                && id.as_str() == "range"
                && arguments.keywords.is_empty()
                && let [arg] = &*arguments.args
            {
                // range(stop) - positive stop means at least one iteration
                // range(start, stop) - we only handle range(stop) for simplicity
                if let Expr::NumberLiteral(ExprNumberLiteral { value, .. }) = arg
                    && let Some(n) = value.as_int().and_then(|i| i.as_i64())
                {
                    return n > 0;
                }
                // Also handle negative literals like range(-5) which iterate 0 times
                if let Expr::UnaryOp(unary) = arg
                    && matches!(unary.op, ruff_python_ast::UnaryOp::USub)
                {
                    // range(-N) always iterates 0 times
                    return false;
                }
            }
            false
        }
        // Check for non-empty list literals
        Expr::List(ExprList { elts, .. }) => !elts.is_empty(),
        // Check for non-empty tuple literals
        Expr::Tuple(ExprTuple { elts, .. }) => !elts.is_empty(),
        // Check for non-empty set literals
        Expr::Set(ExprSet { elts, .. }) => !elts.is_empty(),
        _ => false,
    }
}

impl<'a> BindingsBuilder<'a> {
    fn assert(&mut self, assert_range: TextRange, mut test: Expr, msg: Option<Expr>) {
        let test_range = test.range();
        self.ensure_expr(&mut test, &mut Usage::Narrowing(None));
        let narrow_ops = NarrowOps::from_expr(self, Some(&test));
        let static_test = self.sys_info.evaluate_bool(&test);
        self.insert_binding(Key::Anon(test_range), Binding::Expr(None, test));
        if let Some(mut msg_expr) = msg {
            let mut base = self.scopes.clone_current_flow();
            // Negate the narrowing of the test expression when typechecking
            // the error message, since we know the assertion was false
            let negated_narrow_ops = narrow_ops.negate();
            self.bind_narrow_ops(
                &negated_narrow_ops,
                NarrowUseLocation::Span(msg_expr.range()),
                &Usage::Narrowing(None),
            );
            let mut msg = self.declare_current_idx(Key::UsageLink(msg_expr.range()));
            self.ensure_expr(&mut msg_expr, msg.usage());
            let idx = self.insert_binding(
                KeyExpect(msg_expr.range()),
                BindingExpect::TypeCheckExpr(msg_expr),
            );
            self.insert_binding_current(msg, Binding::UsageLink(LinkedKey::Expect(idx)));
            self.scopes.swap_current_flow_with(&mut base);
        };
        self.bind_narrow_ops(
            &narrow_ops,
            NarrowUseLocation::Span(assert_range),
            &Usage::Narrowing(None),
        );
        if let Some(false) = static_test {
            self.scopes.mark_flow_termination();
        }
    }

    fn bind_unimportable_names(&mut self, x: &StmtImportFrom, as_error: bool) {
        let any = if as_error {
            Type::any_error()
        } else {
            Type::any_explicit()
        };
        for x in &x.names {
            if &x.name != "*" {
                let asname = x.asname.as_ref().unwrap_or(&x.name);
                // We pass None as imported_from, since we are really faking up a local error definition
                self.bind_definition(asname, Binding::Type(any.clone()), FlowStyle::Other);
            }
        }
    }

    /// Bind a special assignment where we do not want the usage tracking or placeholder var pinning
    /// used for normal assignments.
    ///
    /// Used for legacy type variables and for `_Alias()` assignments in `typing` that
    /// we redirect to hard-coded alternative bindings.
    fn bind_legacy_type_var_or_typing_alias(
        &mut self,
        name: &ExprName,
        make_binding: impl FnOnce(Option<Idx<KeyAnnotation>>) -> Binding,
    ) {
        let assigned = self.declare_current_idx(Key::Definition(ShortIdentifier::expr_name(name)));
        let ann = self.bind_current(&name.id, &assigned, FlowStyle::Other);
        let binding = make_binding(ann);
        self.insert_binding_current(assigned, binding);
    }

    fn assign_type_var(&mut self, name: &ExprName, call: &mut ExprCall) {
        // Type var declarations are static types only; skip them for first-usage type inference.
        let static_type_usage = &mut Usage::StaticTypeInformation;
        self.ensure_expr(&mut call.func, static_type_usage);
        let mut iargs = call.arguments.args.iter_mut();
        if let Some(expr) = iargs.next() {
            self.ensure_expr(expr, static_type_usage);
        }
        // The constraints (i.e., any positional arguments after the first)
        // and some keyword arguments are types.
        for arg in iargs {
            self.ensure_type(arg, &mut None);
        }
        for kw in call.arguments.keywords.iter_mut() {
            if let Some(id) = &kw.arg
                && (id.id == "bound" || id.id == "default")
            {
                self.ensure_type(&mut kw.value, &mut None);
            } else {
                self.ensure_expr(&mut kw.value, static_type_usage);
            }
        }
        self.bind_legacy_type_var_or_typing_alias(name, |ann| {
            Binding::TypeVar(
                ann,
                Ast::expr_name_identifier(name.clone()),
                Box::new(call.clone()),
            )
        })
    }

    fn ensure_type_var_tuple_and_param_spec_args(&mut self, call: &mut ExprCall) {
        // Type var declarations are static types only; skip them for first-usage type inference.
        let static_type_usage = &mut Usage::StaticTypeInformation;
        self.ensure_expr(&mut call.func, static_type_usage);
        for arg in call.arguments.args.iter_mut() {
            self.ensure_expr(arg, static_type_usage);
        }
        for kw in call.arguments.keywords.iter_mut() {
            if let Some(id) = &kw.arg
                && id.id == "default"
            {
                self.ensure_type(&mut kw.value, &mut None);
            } else {
                self.ensure_expr(&mut kw.value, static_type_usage);
            }
        }
    }

    fn assign_param_spec(&mut self, name: &ExprName, call: &mut ExprCall) {
        self.ensure_type_var_tuple_and_param_spec_args(call);
        self.bind_legacy_type_var_or_typing_alias(name, |ann| {
            Binding::ParamSpec(
                ann,
                Ast::expr_name_identifier(name.clone()),
                Box::new(call.clone()),
            )
        })
    }

    fn assign_type_var_tuple(&mut self, name: &ExprName, call: &mut ExprCall) {
        self.ensure_type_var_tuple_and_param_spec_args(call);
        self.bind_legacy_type_var_or_typing_alias(name, |ann| {
            Binding::TypeVarTuple(
                ann,
                Ast::expr_name_identifier(name.clone()),
                Box::new(call.clone()),
            )
        })
    }

    fn ensure_type_alias_type_args(&mut self, call: &mut ExprCall) {
        // Type var declarations are static types only; skip them for first-usage type inference.
        let static_type_usage = &mut Usage::StaticTypeInformation;
        self.ensure_expr(&mut call.func, static_type_usage);
        let mut iargs = call.arguments.args.iter_mut();
        // The first argument is the name
        if let Some(expr) = iargs.next() {
            self.ensure_expr(expr, static_type_usage);
        }
        // The second argument is the type
        if let Some(expr) = iargs.next() {
            self.ensure_type(expr, &mut None);
        }
        // There shouldn't be any other positional arguments
        for arg in iargs {
            self.ensure_expr(arg, static_type_usage);
        }
        for kw in call.arguments.keywords.iter_mut() {
            if let Some(id) = &kw.arg
                && id.id == "type_params"
                && let Expr::Tuple(type_params) = &mut kw.value
            {
                for type_param in type_params.elts.iter_mut() {
                    self.ensure_type(type_param, &mut None);
                }
            } else {
                self.ensure_expr(&mut kw.value, static_type_usage);
            }
        }
    }

    fn assign_type_alias_type(&mut self, name: &ExprName, call: &mut ExprCall) {
        self.ensure_type_alias_type_args(call);
        self.bind_legacy_type_var_or_typing_alias(name, |ann| {
            Binding::TypeAliasType(
                ann,
                Ast::expr_name_identifier(name.clone()),
                Box::new(call.clone()),
            )
        })
    }

    /// Bind the annotation in an `AnnAssign`
    pub fn bind_annotation(
        &mut self,
        name: &Identifier,
        annotation: &mut Expr,
        is_initialized: AnnAssignHasValue,
    ) -> Idx<KeyAnnotation> {
        let ann_key = KeyAnnotation::Annotation(ShortIdentifier::new(name));
        self.ensure_type(annotation, &mut None);
        let ann_val = if let Some(special) = SpecialForm::new(&name.id, annotation) {
            // Special case `_: SpecialForm` declarations (this mainly affects some names declared in `typing.pyi`)
            BindingAnnotation::Type(
                AnnotationTarget::Assign(name.id.clone(), AnnAssignHasValue::Yes),
                special.to_type(),
            )
        } else {
            BindingAnnotation::AnnotateExpr(
                if self.scopes.in_class_body() {
                    AnnotationTarget::ClassMember(name.id.clone())
                } else {
                    AnnotationTarget::Assign(name.id.clone(), is_initialized)
                },
                annotation.clone(),
                None,
            )
        };
        self.insert_binding(ann_key, ann_val)
    }

    /// Record a return statement for later analysis if we are in a function body, and mark
    /// that the flow has terminated.
    ///
    /// If this is the top level, report a type error about the invalid return
    /// and also create a binding to ensure we type check the expression.
    fn record_return(&mut self, mut x: StmtReturn) {
        // PEP 765: Disallow return in finally block (Python 3.14+)
        if self.sys_info.version().at_least(3, 14) && self.scopes.in_finally() {
            self.error(
                x.range(),
                ErrorInfo::Kind(ErrorKind::InvalidSyntax),
                "`return` in a `finally` block will silence exceptions".to_owned(),
            );
        }
        let mut ret = self.declare_current_idx(Key::ReturnExplicit(x.range()));
        self.ensure_expr_opt(x.value.as_deref_mut(), ret.usage());
        if let Err((ret, oops_top_level)) = self.scopes.record_or_reject_return(ret, x) {
            match oops_top_level.value {
                Some(v) => self.insert_binding_current(ret, Binding::Expr(None, *v)),
                None => self.insert_binding_current(ret, Binding::Type(Type::None)),
            };
            self.error(
                oops_top_level.range,
                ErrorInfo::Kind(ErrorKind::InvalidSyntax),
                "Invalid `return` outside of a function".to_owned(),
            );
        }
        self.scopes.mark_flow_termination();
    }

    fn find_error(&self, error: &FindError, range: TextRange) {
        let Some(kind) = error.kind() else {
            return;
        };
        let (ctx, msg) = error.display();
        self.error_multiline(range, ErrorInfo::new(kind, ctx.as_deref()), msg);
    }

    /// Evaluate the statements and update the bindings.
    /// Every statement should end up in the bindings, perhaps with a location that is never used.
    pub fn stmt(&mut self, x: Stmt, parent: &NestingContext) {
        self.with_semantic_checker(|semantic, context| semantic.visit_stmt(&x, context));

        match x {
            Stmt::FunctionDef(x) => {
                self.function_def(x, parent);
            }
            Stmt::ClassDef(x) => self.class_def(x, parent),
            Stmt::Return(x) => {
                self.record_return(x);
            }
            Stmt::Delete(mut x) => {
                for target in &mut x.targets {
                    let mut delete_idx = self.declare_current_idx(Key::Delete(target.range()));
                    if let Expr::Name(name) = target {
                        self.ensure_expr_name(name, delete_idx.usage());
                        self.scopes.mark_as_deleted(&name.id);
                    } else {
                        self.ensure_expr(target, delete_idx.usage());
                    }
                    self.insert_binding_current(delete_idx, Binding::Delete(target.clone()));
                }
            }
            Stmt::Assign(ref x)
                if let [Expr::Name(name)] = x.targets.as_slice()
                    && let Some((module, forward)) =
                        resolve_typeshed_alias(self.module_info.name(), &name.id, &x.value) =>
            {
                // This hook is used to treat certain names defined in `typing.pyi` as `_Alias()`
                // assignments "as if" they were imports of the aliased name.
                //
                // For example, we treat `typing.List` as if it were an import of `builtins.list`.
                self.bind_legacy_type_var_or_typing_alias(name, |_| {
                    Binding::Import(module, forward, None)
                })
            }
            Stmt::Assign(mut x) => {
                if let [Expr::Name(name)] = x.targets.as_slice() {
                    if let Expr::Call(call) = &mut *x.value
                        && let Some(special) = self.as_special_export(&call.func)
                    {
                        match special {
                            SpecialExport::TypeVar => {
                                self.assign_type_var(name, call);
                                return;
                            }
                            SpecialExport::ParamSpec => {
                                self.assign_param_spec(name, call);
                                return;
                            }
                            SpecialExport::TypeAliasType => {
                                self.assign_type_alias_type(name, call);
                                return;
                            }
                            SpecialExport::TypeVarTuple => {
                                self.assign_type_var_tuple(name, call);
                                return;
                            }
                            SpecialExport::Enum
                            | SpecialExport::IntEnum
                            | SpecialExport::StrEnum => {
                                if let Some((arg_name, members)) =
                                    call.arguments.args.split_first_mut()
                                {
                                    self.synthesize_enum_def(
                                        name,
                                        parent,
                                        &mut call.func,
                                        arg_name,
                                        members,
                                    );
                                    return;
                                }
                            }
                            SpecialExport::TypedDict => {
                                if let Some((arg_name, members)) =
                                    call.arguments.args.split_first_mut()
                                {
                                    self.synthesize_typed_dict_def(
                                        name,
                                        parent,
                                        &mut call.func,
                                        arg_name,
                                        members,
                                        &mut call.arguments.keywords,
                                    );
                                    return;
                                }
                            }
                            SpecialExport::TypingNamedTuple => {
                                if let Some((arg_name, members)) =
                                    call.arguments.args.split_first_mut()
                                {
                                    self.synthesize_typing_named_tuple_def(
                                        name,
                                        parent,
                                        &mut call.func,
                                        arg_name,
                                        members,
                                    );
                                    return;
                                }
                            }
                            SpecialExport::CollectionsNamedTuple => {
                                if let Some((arg_name, members)) =
                                    call.arguments.args.split_first_mut()
                                {
                                    self.synthesize_collections_named_tuple_def(
                                        name,
                                        parent,
                                        &mut call.func,
                                        arg_name,
                                        members,
                                        &mut call.arguments.keywords,
                                    );
                                    return;
                                }
                            }
                            SpecialExport::NewType => {
                                if let [new_type_name, base] = &mut *call.arguments.args {
                                    self.synthesize_typing_new_type(
                                        name,
                                        parent,
                                        new_type_name,
                                        base,
                                    );
                                    return;
                                }
                            }
                            _ => {}
                        }
                    }
                    self.bind_single_name_assign(
                        &Ast::expr_name_identifier(name.clone()),
                        x.value,
                        None,
                    );
                } else {
                    self.bind_targets_with_value(&mut x.targets, &mut x.value);
                }
            }
            Stmt::AnnAssign(mut x) => match *x.target {
                Expr::Name(name) => {
                    let name = Ast::expr_name_identifier(name);
                    // We have to handle the value carefully because the annotation, class field, and
                    // binding do not all treat `...` exactly the same:
                    // - an annotation key and a class field treat `...` as initializing, but only in stub files
                    // - we skip the `NameAssign` if we are in a stub and the value is `...`
                    let (value, maybe_ellipses) = if let Some(value) = x.value {
                        // Treat a name as initialized, but skip actually checking the value, if we are assigning `...` in a stub.
                        if self.module_info.path().is_interface()
                            && matches!(&*value, Expr::EllipsisLiteral(_))
                        {
                            (None, Some(*value))
                        } else {
                            (Some(value), None)
                        }
                    } else {
                        (None, None)
                    };
                    let ann_idx = self.bind_annotation(
                        &name,
                        &mut x.annotation,
                        match (&value, &maybe_ellipses) {
                            (None, None) => AnnAssignHasValue::No,
                            _ => AnnAssignHasValue::Yes,
                        },
                    );
                    let canonical_ann_idx = match value {
                        Some(value) => self.bind_single_name_assign(
                            &name,
                            value,
                            Some((&x.annotation, ann_idx)),
                        ),
                        None => self.bind_definition(
                            &name,
                            Binding::AnnotatedType(
                                ann_idx,
                                Box::new(Binding::Type(Type::any_implicit())),
                            ),
                            if self.scopes.in_class_body() {
                                FlowStyle::ClassField {
                                    initial_value: maybe_ellipses,
                                }
                            } else {
                                // A flow style might be already set for the name, e.g. if it was defined
                                // already. Otherwise it is uninitialized.
                                self.scopes
                                    .current_flow_style(&name.id)
                                    .unwrap_or(FlowStyle::Uninitialized)
                            },
                        ),
                    };
                    // This assignment gets checked with the provided annotation. But if there exists a prior
                    // annotation, we might be invalidating it unless the annotations are the same. Insert a
                    // check that in that case the annotations match.
                    if let Some(ann) = canonical_ann_idx {
                        self.insert_binding(
                            KeyExpect(name.range),
                            BindingExpect::Redefinition {
                                new: ann_idx,
                                existing: ann,
                                name: name.id.clone(),
                            },
                        );
                    }
                }
                Expr::Attribute(attr) => {
                    let attr_name = attr.attr.id.clone();
                    self.ensure_type(&mut x.annotation, &mut None);
                    let ann_key = self.insert_binding(
                        KeyAnnotation::AttrAnnotation(x.annotation.range()),
                        BindingAnnotation::AnnotateExpr(
                            AnnotationTarget::ClassMember(attr_name.clone()),
                            *x.annotation,
                            None,
                        ),
                    );
                    let value = match x.value {
                        Some(mut assigned) => {
                            self.bind_attr_assign(attr.clone(), &mut assigned, |v, _| {
                                ExprOrBinding::Expr(v.clone())
                            })
                        }
                        _ => ExprOrBinding::Binding(Binding::Type(Type::any_implicit())),
                    };
                    if !self
                        .scopes
                        .record_self_attr_assign(&attr, value.clone(), Some(ann_key))
                    {
                        self.error(
                             x.range,
                             ErrorInfo::Kind(ErrorKind::BadAssignment),
                             format!(
                                "Type cannot be declared in assignment to non-self attribute `{}.{}`",
                                self.module_info.display(&attr.value),
                                attr_name,
                            ),

                         );
                    }
                }
                mut target => {
                    if matches!(&target, Expr::Subscript(..)) {
                        // Note that for Expr::Subscript Python won't fail at runtime,
                        // but Mypy and Pyright both error here, so let's do the same.
                        self.error(
                            x.annotation.range(),
                            ErrorInfo::Kind(ErrorKind::InvalidSyntax),
                            "Subscripts should not be annotated".to_owned(),
                        );
                    }
                    // Try and continue as much as we can, by throwing away the type or just binding to error
                    match x.value {
                        Some(value) => self.stmt(
                            Stmt::Assign(StmtAssign {
                                node_index: AtomicNodeIndex::default(),
                                range: x.range,
                                targets: vec![target],
                                value,
                            }),
                            parent,
                        ),
                        None => {
                            self.bind_target_no_expr(&mut target, &|_| {
                                Binding::Type(Type::any_error())
                            });
                        }
                    }
                }
            },
            Stmt::AugAssign(mut x) => {
                match x.target.as_ref() {
                    Expr::Name(name) => {
                        let mut assigned = self
                            .declare_current_idx(Key::Definition(ShortIdentifier::expr_name(name)));
                        // Make sure the name is already initialized - it's current value is part of AugAssign semantics.
                        self.ensure_expr_name(name, assigned.usage());
                        self.ensure_expr(&mut x.value, assigned.usage());
                        let ann = self.bind_current(&name.id, &assigned, FlowStyle::Other);
                        let binding = Binding::AugAssign(ann, x.clone());
                        self.insert_binding_current(assigned, binding);
                    }
                    Expr::Attribute(attr) => {
                        let mut x_cloned = x.clone();
                        self.bind_attr_assign(attr.clone(), &mut x.value, move |expr, ann| {
                            x_cloned.value = Box::new(expr.clone());
                            ExprOrBinding::Binding(Binding::AugAssign(ann, x_cloned))
                        });
                    }
                    Expr::Subscript(subscr) => {
                        let mut x_cloned = x.clone();
                        self.bind_subscript_assign(
                            subscr.clone(),
                            &mut x.value,
                            move |expr, ann| {
                                x_cloned.value = Box::new(expr.clone());
                                ExprOrBinding::Binding(Binding::AugAssign(ann, x_cloned))
                            },
                        );
                    }
                    illegal_target => {
                        // Most structurally invalid targets become errors in the parser, which we propagate so there
                        // is no need for duplicate errors. But we do want to catch unbound names (which the parser
                        // will not catch)
                        //
                        // We don't track first-usage in this context, since we won't analyze the usage anyway.
                        let mut e = illegal_target.clone();
                        self.ensure_expr(&mut e, &mut Usage::StaticTypeInformation);
                    }
                }
            }
            Stmt::TypeAlias(mut x) => {
                if !self.scopes.in_module_or_class_top_level() {
                    self.error(
                        x.range,
                        ErrorInfo::Kind(ErrorKind::InvalidSyntax),
                        "`type` statement is not allowed in this context".to_owned(),
                    );
                }
                if let Expr::Name(name) = *x.name {
                    // Create a new scope for the type alias type parameters
                    self.scopes.push(Scope::type_alias(x.range));
                    if let Some(params) = &mut x.type_params {
                        self.type_params(params);
                    }
                    self.ensure_type(&mut x.value, &mut None);
                    // Pop the type alias scope before binding the definition
                    self.scopes.pop();
                    let binding = Binding::ScopedTypeAlias(
                        name.id.clone(),
                        x.type_params.map(|x| *x),
                        x.value,
                    );
                    self.bind_definition(
                        &Ast::expr_name_identifier(name),
                        binding,
                        FlowStyle::Other,
                    );
                } else {
                    self.error(
                        x.range,
                        ErrorInfo::Kind(ErrorKind::InvalidSyntax),
                        "Invalid assignment target".to_owned(),
                    );
                }
            }
            Stmt::For(mut x) => {
                if x.is_async && !self.scopes.is_in_async_def() {
                    self.error(
                        x.range(),
                        ErrorInfo::Kind(ErrorKind::InvalidSyntax),
                        "`async for` can only be used inside an async function".to_owned(),
                    );
                }
                let mut loop_header_targets = SmallSet::new();
                Ast::expr_lvalue(&x.target, &mut |name| {
                    loop_header_targets.insert(name.id.clone());
                });
                // Check if the iterable is definitely non-empty before binding
                // (must be done before x.iter is moved)
                let loop_definitely_runs = is_definitely_nonempty_iterable(&x.iter);
                self.bind_target_with_expr(&mut x.target, &mut x.iter, &|expr, ann| {
                    Binding::IterableValue(ann, expr.clone(), IsAsync::new(x.is_async))
                });
                // Note that we set up the loop *after* the header is fully bound, because the
                // loop iterator is only evaluated once before the loop begins. But the loop header
                // targets - which get re-bound each iteration - are excluded from the loop Phi logic.
                self.setup_loop(x.range, &loop_header_targets);
                self.stmts(x.body, parent);
                self.teardown_loop(
                    x.range,
                    &NarrowOps::new(),
                    x.orelse,
                    parent,
                    false,
                    loop_definitely_runs,
                );
            }
            Stmt::While(mut x) => {
                self.setup_loop(x.range, &SmallSet::new());
                // Note that it is important we ensure *after* we set up the loop, so that both the
                // narrowing and type checking are aware that the test might be impacted by changes
                // made in the loop (e.g. if we reassign the test variable).
                // Typecheck the test condition during solving.
                self.ensure_expr(&mut x.test, &mut Usage::Narrowing(None));
                let is_while_true = self.sys_info.evaluate_bool(&x.test) == Some(true);
                let narrow_ops = NarrowOps::from_expr(self, Some(&x.test));
                self.bind_narrow_ops(
                    &narrow_ops,
                    NarrowUseLocation::Span(x.range),
                    &Usage::Narrowing(None),
                );
                self.insert_binding(KeyExpect(x.test.range()), BindingExpect::Bool(*x.test));
                self.stmts(x.body, parent);
                // For while True: loops, the loop body definitely runs at least once
                self.teardown_loop(
                    x.range,
                    &narrow_ops,
                    x.orelse,
                    parent,
                    is_while_true,
                    is_while_true,
                );
            }
            Stmt::If(x) => {
                let mut exhaustive = false;
                self.start_fork(x.range);
                // Type narrowing operations that are carried over from one branch to the next. For example, in:
                //   if x is None:
                //     pass
                //   else:
                //     pass
                // x is bound to Narrow(x, Is(None)) in the if branch, and the negation, Narrow(x, IsNot(None)),
                // is carried over to the else branch.
                let mut negated_prev_ops = NarrowOps::new();
                for (range, mut test, body) in Ast::if_branches_owned(x) {
                    self.start_branch();
                    self.bind_narrow_ops(
                        &negated_prev_ops,
                        NarrowUseLocation::Start(range),
                        &Usage::Narrowing(None),
                    );
                    // If there is no test, it's an `else` clause and `this_branch_chosen` will be true.
                    let this_branch_chosen = match &test {
                        None => Some(true),
                        Some(x) => self.sys_info.evaluate_bool(x),
                    };
                    self.ensure_expr_opt(test.as_mut(), &mut Usage::Narrowing(None));
                    let new_narrow_ops = if this_branch_chosen == Some(false) {
                        // Skip the body in this case - it typically means a check (e.g. a sys version,
                        // platform, or TYPE_CHECKING check) where the body is not statically analyzable.
                        self.abandon_branch();
                        continue;
                    } else {
                        NarrowOps::from_expr(self, test.as_ref())
                    };
                    if let Some(test_expr) = test {
                        // Typecheck the test condition during solving.
                        self.insert_binding(
                            KeyExpect(test_expr.range()),
                            BindingExpect::Bool(test_expr),
                        );
                    }
                    self.bind_narrow_ops(
                        &new_narrow_ops,
                        NarrowUseLocation::Span(range),
                        &Usage::Narrowing(None),
                    );
                    negated_prev_ops.and_all(new_narrow_ops.negate());
                    self.stmts(body, parent);
                    self.finish_branch();
                    if this_branch_chosen == Some(true) {
                        exhaustive = true;
                        break; // We definitely picked this branch if we got here, nothing below is reachable.
                    }
                }
                if exhaustive {
                    self.finish_exhaustive_fork();
                } else {
                    self.finish_non_exhaustive_fork(&negated_prev_ops);
                }
            }
            Stmt::With(x) => {
                if x.is_async && !self.scopes.is_in_async_def() {
                    self.error(
                        x.range(),
                        ErrorInfo::Kind(ErrorKind::InvalidSyntax),
                        "`async with` can only be used inside an async function".to_owned(),
                    );
                }
                let kind = IsAsync::new(x.is_async);
                for mut item in x.items {
                    let item_range = item.range();
                    let expr_range = item.context_expr.range();
                    let mut context = self.declare_current_idx(Key::ContextExpr(expr_range));
                    self.ensure_expr(&mut item.context_expr, context.usage());
                    let context_idx = self
                        .insert_binding_current(context, Binding::Expr(None, item.context_expr));
                    if let Some(mut opts) = item.optional_vars {
                        let make_binding =
                            |ann| Binding::ContextValue(ann, context_idx, expr_range, kind);
                        self.bind_target_no_expr(&mut opts, &make_binding);
                    } else {
                        self.insert_binding(
                            Key::Anon(item_range),
                            Binding::ContextValue(None, context_idx, expr_range, kind),
                        );
                    }
                }
                self.stmts(x.body, parent);
            }
            Stmt::Match(x) => {
                self.stmt_match(x, parent);
            }
            Stmt::Raise(x) => {
                if let Some(mut exc) = x.exc {
                    let mut current = self.declare_current_idx(Key::UsageLink(x.range));
                    self.ensure_expr(&mut exc, current.usage());
                    let raised = if let Some(mut cause) = x.cause {
                        self.ensure_expr(&mut cause, current.usage());
                        RaisedException::WithCause(Box::new((*exc, *cause)))
                    } else {
                        RaisedException::WithoutCause(*exc)
                    };
                    let idx = self.insert_binding(
                        KeyExpect(x.range),
                        BindingExpect::CheckRaisedException(raised),
                    );
                    self.insert_binding_current(
                        current,
                        Binding::UsageLink(LinkedKey::Expect(idx)),
                    );
                } else {
                    // If there's no exception raised, don't bother checking the cause.
                }
                self.scopes.mark_flow_termination();
            }
            Stmt::Try(x) => {
                self.start_fork_and_branch(x.range);

                // We branch before the body, conservatively assuming that any statement can fail
                // entry -> try -> else -> finally
                //   |                     ^
                //   ----> handler --------|

                self.stmts(x.body, parent);
                self.stmts(x.orelse, parent);
                self.finish_branch();

                for h in x.handlers {
                    self.start_branch();
                    let range = h.range();
                    let h = h.except_handler().unwrap(); // Only one variant for now
                    match (&h.name, h.type_) {
                        (Some(name), Some(mut type_)) => {
                            let mut handler = self
                                .declare_current_idx(Key::Definition(ShortIdentifier::new(name)));
                            self.ensure_expr(&mut type_, handler.usage());
                            self.bind_current_as(
                                name,
                                handler,
                                Binding::ExceptionHandler(type_, x.is_star),
                                FlowStyle::Other,
                            );
                        }
                        (None, Some(mut type_)) => {
                            let mut handler = self.declare_current_idx(Key::Anon(range));
                            self.ensure_expr(&mut type_, handler.usage());
                            self.insert_binding_current(
                                handler,
                                Binding::ExceptionHandler(type_, x.is_star),
                            );
                        }
                        (Some(name), None) => {
                            // Must be a syntax error. But make sure we bind name to something.
                            let handler = self
                                .declare_current_idx(Key::Definition(ShortIdentifier::new(name)));
                            self.bind_current_as(
                                name,
                                handler,
                                Binding::Type(Type::any_error()),
                                FlowStyle::Other,
                            );
                        }
                        (None, None) => {}
                    }

                    self.stmts(h.body, parent);

                    if let Some(name) = &h.name {
                        // Handle the implicit delete Python performs at the end of the `except` clause.
                        //
                        // Note that because there is no scoping, even if the name was defined above the
                        // try/except, it will be unbound below whenever that name was used for a handler.
                        //
                        // https://docs.python.org/3/reference/compound_stmts.html#except-clause
                        self.scopes.mark_as_deleted(&name.id);
                    }

                    self.finish_branch();
                }

                self.finish_exhaustive_fork();
                self.scopes.enter_finally();
                self.stmts(x.finalbody, parent);
                self.scopes.exit_finally();
            }
            Stmt::Assert(x) => {
                self.assert(x.range(), *x.test, x.msg.map(|m| *m));
            }
            Stmt::Import(x) => {
                for x in x.names {
                    let m = ModuleName::from_name(&x.name.id);
                    if let Some(error) = self.lookup.get(m).error() {
                        self.find_error(&error, x.range);
                    }
                    match x.asname {
                        Some(asname) => {
                            self.scopes.register_import(&asname);
                            self.bind_definition(
                                &asname,
                                Binding::Module(m, m.components(), None),
                                FlowStyle::ImportAs(m),
                            );
                        }
                        None => {
                            let first = m.first_component();
                            let module_key = self.scopes.existing_module_import_at(&first);
                            let key = self.insert_binding(
                                Key::Import(first.clone(), x.name.range),
                                Binding::Module(m, vec![first.clone()], module_key),
                            );
                            // Register the import using the first component (e.g., "os" from "os.path")
                            // since that's the name that gets bound and used in code
                            self.scopes.register_import(&Identifier {
                                node_index: x.name.node_index.clone(),
                                id: first.clone(),
                                range: x.name.range,
                            });
                            self.bind_name(&first, key, FlowStyle::MergeableImport(m));
                        }
                    }
                }
            }
            Stmt::ImportFrom(x) => {
                if let Some(m) = self.module_info.name().new_maybe_relative(
                    self.module_info.path().is_init(),
                    x.level,
                    x.module.as_ref().map(|x| &x.id),
                ) {
                    match self.lookup.get(m) {
                        FindingOrError::Finding(module_exports) => {
                            if let Some(error) = module_exports.error {
                                self.find_error(&error, x.range);
                            }
                            self.bind_module_exports(x, m, module_exports.finding);
                        }
                        FindingOrError::Error(error) => {
                            self.find_error(&error, x.range);
                            self.bind_unimportable_names(&x, error.kind().is_some());
                        }
                    }
                } else {
                    self.error(
                        x.range,
                        ErrorInfo::Kind(ErrorKind::MissingImport),
                        format!(
                            "Could not resolve relative import `{}`",
                            ".".repeat(x.level as usize)
                        ),
                    );
                    self.bind_unimportable_names(&x, true);
                }
            }
            Stmt::Global(x) => {
                for name in x.names {
                    self.declare_mutable_capture(&name, MutableCaptureKind::Global);
                }
            }
            Stmt::Nonlocal(x) => {
                for name in x.names {
                    self.declare_mutable_capture(&name, MutableCaptureKind::Nonlocal);
                }
            }
            Stmt::Expr(StmtExpr {
                range: expr_range,
                value:
                    box Expr::Call(ExprCall {
                        range: call_range,
                        func: box Expr::Name(name),
                        arguments:
                            Arguments {
                                range: _,
                                keywords: _,
                                args,
                                ..
                            },
                        ..
                    }),
                ..
            }) if name.id.as_str() == "prod_assert" && (args.len() == 1 || args.len() == 2) => {
                let (test, msg) = if args.len() == 1 {
                    (args[0].clone(), None)
                } else if args.len() == 2 {
                    (args[0].clone(), Some(args[1].clone()))
                } else {
                    unreachable!("args.len() can only be 1 or 2")
                };
                self.insert_binding(Key::StmtExpr(expr_range), Binding::Type(Type::None));
                self.assert(call_range, test, msg);
            }
            Stmt::Expr(mut x) => {
                let mut current = self.declare_current_idx(Key::StmtExpr(x.value.range()));
                self.ensure_expr(&mut x.value, current.usage());
                let special_export = if let Expr::Call(ExprCall { func, .. }) = &*x.value {
                    self.as_special_export(func)
                } else {
                    None
                };
                self.insert_binding_current(current, Binding::StmtExpr(*x.value, special_export));
                if special_export == Some(SpecialExport::PytestNoReturn) {
                    self.scopes.mark_flow_termination();
                }
            }
            Stmt::Pass(_) => { /* no-op */ }
            Stmt::Break(x) => {
                // PEP 765: Disallow break in finally block if not inside a nested loop
                if self.sys_info.version().at_least(3, 14)
                    && self.scopes.in_finally()
                    && !self.scopes.loop_protects_from_finally_exit()
                {
                    self.error(
                        x.range,
                        ErrorInfo::Kind(ErrorKind::InvalidSyntax),
                        "`break` in a `finally` block will silence exceptions".to_owned(),
                    );
                }
                self.add_loop_exitpoint(LoopExit::Break);
            }
            Stmt::Continue(x) => {
                // PEP 765: Disallow continue in finally block if not inside a nested loop
                if self.sys_info.version().at_least(3, 14)
                    && self.scopes.in_finally()
                    && !self.scopes.loop_protects_from_finally_exit()
                {
                    self.error(
                        x.range,
                        ErrorInfo::Kind(ErrorKind::InvalidSyntax),
                        "`continue` in a `finally` block will silence exceptions".to_owned(),
                    );
                }
                self.add_loop_exitpoint(LoopExit::Continue);
            }
            Stmt::IpyEscapeCommand(x) => {
                if self.module_info.is_notebook() {
                    // No-op
                } else {
                    self.error(
                        x.range,
                        ErrorInfo::Kind(ErrorKind::Unsupported),
                        "IPython escapes are not supported".to_owned(),
                    )
                }
            }
        }
    }

    fn bind_module_exports(&mut self, x: StmtImportFrom, m: ModuleName, module_exports: Exports) {
        let exported = module_exports.exports(self.lookup);
        for x in x.names {
            if &x.name == "*" {
                for name in module_exports.wildcard(self.lookup).iter_hashed() {
                    let key = Key::Import(name.into_key().clone(), x.range);
                    let val = if exported.contains_key_hashed(name) {
                        Binding::Import(m, name.into_key().clone(), None)
                    } else {
                        self.error(
                            x.range,
                            ErrorInfo::Kind(ErrorKind::MissingModuleAttribute),
                            format!("Could not import `{name}` from `{m}`"),
                        );
                        Binding::Type(Type::any_error())
                    };
                    let key = self.insert_binding(key, val);
                    // Register the imported name from wildcard imports
                    self.scopes.register_import_with_star(&Identifier {
                        node_index: AtomicNodeIndex::default(),
                        id: name.into_key().clone(),
                        range: x.range,
                    });
                    self.bind_name(
                        name.key(),
                        key,
                        FlowStyle::Import(m, name.into_key().clone()),
                    );
                }
            } else {
                let original_name_range = if x.asname.is_some() {
                    Some(x.name.range)
                } else {
                    None
                };
                let asname = x.asname.unwrap_or_else(|| x.name.clone());
                // A `from x import y` statement is ambiguous; if `x` is a package with
                // an `__init__.py` file, then it might import the name `y` from the
                // module `x` defined by the `__init__.py` file, or it might import a
                // submodule `x.y` of the package `x`.
                //
                // If both are present, generally we prefer the name defined in `x`,
                // but there is an exception: if we are already looking at the
                // `__init__` module of `x`, we always prefer the submodule.
                let val = if (self.module_info.name() != m) && exported.contains_key(&x.name.id) {
                    if let Some(ExportLocation::ThisModule(Export {
                        deprecation: Some(deprecation),
                        ..
                    })) = exported.get(&x.name.id)
                    {
                        let msg =
                            deprecation.as_error_message(format!("`{}` is deprecated", x.name));
                        self.error_multiline(x.range, ErrorInfo::Kind(ErrorKind::Deprecated), msg);
                    }
                    Binding::Import(m, x.name.id.clone(), original_name_range)
                } else {
                    let x_as_module_name = m.append(&x.name.id);
                    let (finding, error) = match self.lookup.get(x_as_module_name) {
                        FindingOrError::Finding(finding) => (true, finding.error),
                        FindingOrError::Error(error) => (false, Some(error)),
                    };
                    let error = error.is_some_and(|e| matches!(e, FindError::NotFound(..)));
                    if error {
                        self.error(
                            x.range,
                            ErrorInfo::Kind(ErrorKind::MissingModuleAttribute),
                            format!("Could not import `{}` from `{m}`", x.name.id),
                        );
                    }
                    if finding {
                        Binding::Module(x_as_module_name, x_as_module_name.components(), None)
                    } else if error {
                        Binding::Type(Type::any_error())
                    } else {
                        Binding::Type(Type::any_explicit())
                    }
                };
                // __future__ imports have side effects even if not explicitly used,
                // so we skip the unused import check for them.
                if m == ModuleName::future() {
                    self.scopes.register_future_import(&asname);
                } else {
                    self.scopes.register_import(&asname);
                }
                self.bind_definition(&asname, val, FlowStyle::Import(m, x.name.id));
            }
        }
    }
}
