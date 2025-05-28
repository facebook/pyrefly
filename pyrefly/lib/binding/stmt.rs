/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprName;
use ruff_python_ast::Identifier;
use ruff_python_ast::Keyword;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtAssign;
use ruff_python_ast::StmtImportFrom;
use ruff_python_ast::StmtReturn;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::Hashed;

use crate::binding::binding::AnnotationStyle;
use crate::binding::binding::AnnotationTarget;
use crate::binding::binding::Binding;
use crate::binding::binding::BindingAnnotation;
use crate::binding::binding::BindingExpect;
use crate::binding::binding::ExprOrBinding;
use crate::binding::binding::Initialized;
use crate::binding::binding::IsAsync;
use crate::binding::binding::Key;
use crate::binding::binding::KeyAnnotation;
use crate::binding::binding::KeyExpect;
use crate::binding::binding::RaisedException;
use crate::binding::bindings::BindingsBuilder;
use crate::binding::bindings::LookupKind;
use crate::binding::bindings::MutableCaptureLookupKind;
use crate::binding::narrow::NarrowOps;
use crate::binding::scope::FlowStyle;
use crate::binding::scope::LoopExit;
use crate::error::kind::ErrorKind;
use crate::export::special::SpecialExport;
use crate::graph::index::Idx;
use crate::module::module_name::ModuleName;
use crate::module::short_identifier::ShortIdentifier;
use crate::ruff::ast::Ast;
use crate::state::loader::FindError;
use crate::types::alias::resolve_typeshed_alias;
use crate::types::special_form::SpecialForm;
use crate::types::types::Type;

impl<'a> BindingsBuilder<'a> {
    fn bind_unimportable_names(&mut self, x: &StmtImportFrom) {
        for x in &x.names {
            if &x.name != "*" {
                let asname = x.asname.as_ref().unwrap_or(&x.name);
                // We pass None as imported_from, since we are really faking up a local error definition
                self.bind_definition(asname, Binding::Type(Type::any_error()), FlowStyle::Other);
            }
        }
    }

    // Check that the variable name in a functional definition matches the first argument string
    fn check_functional_definition_name(&mut self, name: &Name, arg: &Expr) {
        if let Expr::StringLiteral(x) = arg {
            if x.value.to_str() != name.as_str() {
                self.error(
                    arg.range(),
                    format!("Expected string literal \"{}\"", name),
                    ErrorKind::InvalidArgument,
                );
            }
        } else {
            self.error(
                arg.range(),
                format!("Expected string literal \"{}\"", name),
                ErrorKind::InvalidArgument,
            );
        }
    }

    fn assign_type_var(&mut self, name: &ExprName, call: &mut ExprCall) {
        self.ensure_expr(&mut call.func);
        let mut iargs = call.arguments.args.iter_mut();
        if let Some(expr) = iargs.next() {
            self.ensure_expr(expr);
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
                self.ensure_expr(&mut kw.value);
            }
        }
        self.bind_assign(name, |ann| {
            Binding::TypeVar(
                ann,
                Ast::expr_name_identifier(name.clone()),
                Box::new(call.clone()),
            )
        })
    }

    fn ensure_type_var_tuple_and_param_spec_args(&mut self, call: &mut ExprCall) {
        self.ensure_expr(&mut call.func);
        for arg in call.arguments.args.iter_mut() {
            self.ensure_expr(arg);
        }
        for kw in call.arguments.keywords.iter_mut() {
            if let Some(id) = &kw.arg
                && id.id == "default"
            {
                self.ensure_type(&mut kw.value, &mut None);
            } else {
                self.ensure_expr(&mut kw.value);
            }
        }
    }

    fn assign_param_spec(&mut self, name: &ExprName, call: &mut ExprCall) {
        self.ensure_type_var_tuple_and_param_spec_args(call);
        self.bind_assign(name, |ann| {
            Binding::ParamSpec(
                ann,
                Ast::expr_name_identifier(name.clone()),
                Box::new(call.clone()),
            )
        })
    }

    fn assign_type_var_tuple(&mut self, name: &ExprName, call: &mut ExprCall) {
        self.ensure_type_var_tuple_and_param_spec_args(call);
        self.bind_assign(name, |ann| {
            Binding::TypeVarTuple(
                ann,
                Ast::expr_name_identifier(name.clone()),
                Box::new(call.clone()),
            )
        })
    }

    fn assign_enum(
        &mut self,
        name: &ExprName,
        func: &mut Expr,
        arg_name: &mut Expr,
        members: &mut [Expr],
    ) {
        self.ensure_expr(func);
        self.ensure_expr(arg_name);
        for arg in &mut *members {
            self.ensure_expr(arg);
        }
        self.check_functional_definition_name(&name.id, arg_name);
        self.synthesize_enum_def(
            Ast::expr_name_identifier(name.clone()),
            func.clone(),
            members,
        );
    }

    fn assign_typed_dict(
        &mut self,
        name: &ExprName,
        func: &mut Expr,
        arg_name: &Expr,
        args: &mut [Expr],
        keywords: &mut [Keyword],
    ) {
        self.ensure_expr(func);
        self.check_functional_definition_name(&name.id, arg_name);
        self.synthesize_typed_dict_def(
            Ast::expr_name_identifier(name.clone()),
            func.clone(),
            args,
            keywords,
        );
    }

    fn assign_typing_named_tuple(
        &mut self,
        name: &ExprName,
        func: &mut Expr,
        arg_name: &Expr,
        members: &[Expr],
    ) {
        self.ensure_expr(func);
        self.check_functional_definition_name(&name.id, arg_name);
        self.synthesize_typing_named_tuple_def(
            Ast::expr_name_identifier(name.clone()),
            func.clone(),
            members,
        );
    }

    fn assign_collections_named_tuple(
        &mut self,
        name: &ExprName,
        func: &mut Expr,
        arg_name: &Expr,
        members: &mut [Expr],
        keywords: &mut [Keyword],
    ) {
        self.ensure_expr(func);
        self.check_functional_definition_name(&name.id, arg_name);
        self.synthesize_collections_named_tuple_def(
            Ast::expr_name_identifier(name.clone()),
            members,
            keywords,
        );
    }

    fn assign_new_type(&mut self, name: &ExprName, new_type_name: &mut Expr, base: &mut Expr) {
        self.ensure_expr(new_type_name);
        self.check_functional_definition_name(&name.id, new_type_name);
        self.ensure_type(base, &mut None);
        self.synthesize_typing_new_type(Ast::expr_name_identifier(name.clone()), base.clone());
    }

    pub fn ensure_mutable_name(&mut self, x: &ExprName) -> Idx<Key> {
        let name = Ast::expr_name_identifier(x.clone());
        let binding = self
            .lookup_name(&name.id, LookupKind::Mutable)
            .map(Binding::Forward);
        self.ensure_name(&name, binding)
    }

    fn define_nonlocal_name(&mut self, name: &Identifier) {
        let key = Key::Definition(ShortIdentifier::new(name));
        let binding =
            match self.lookup_mutable_captured_name(&name.id, MutableCaptureLookupKind::Nonlocal) {
                Ok(found) => Binding::Forward(found),
                Err(error) => {
                    self.error(name.range, error.message(name), ErrorKind::UnknownName);
                    Binding::Type(Type::any_error())
                }
            };
        self.insert_binding(key, binding);
    }

    fn define_global_name(&mut self, name: &Identifier) {
        let key = Key::Definition(ShortIdentifier::new(name));
        let binding =
            match self.lookup_mutable_captured_name(&name.id, MutableCaptureLookupKind::Global) {
                Ok(found) => Binding::Forward(found),
                Err(error) => {
                    self.error(name.range, error.message(name), ErrorKind::UnknownName);
                    Binding::Type(Type::any_error())
                }
            };
        self.insert_binding(key, binding);
    }

    /// If someone does `x = C["test"]`, that might be a type alias, it might not.
    /// Use this heuristic to detect things that are definitely type aliases.
    fn is_definitely_type_alias_rhs(&mut self, x: &Expr) -> bool {
        match x {
            Expr::Subscript(x) => matches!(
                self.as_special_export(&x.value),
                Some(SpecialExport::Union | SpecialExport::Optional)
            ),
            _ => false,
        }
    }

    pub fn bind_name_assign(&mut self, name: &ExprName, mut value: Box<Expr>) {
        let idx = self.idx_for_promise(Key::Definition(ShortIdentifier::expr_name(name)));
        if self.is_definitely_type_alias_rhs(value.as_ref()) {
            self.ensure_type(&mut value, &mut None);
        } else {
            self.ensure_expr(&mut value);
        }
        let style = if self.scopes.in_class_body() {
            FlowStyle::ClassField {
                initial_value: Some((*value).clone()),
            }
        } else {
            FlowStyle::Other
        };
        let (ann, default) = self.bind_key(&name.id, idx, style);
        let mut binding = Binding::NameAssign(
            name.id.clone(),
            ann.map(|k| (AnnotationStyle::Forwarded, k)),
            value,
        );
        if let Some(default) = default {
            binding = Binding::Default(default, Box::new(binding));
        }
        self.insert_binding_idx(idx, binding);
    }

    /// Record a return statement for later analysis if we are in a function body, and mark
    /// that the flow has terminated.
    ///
    /// If this is the top level, report a type error about the invalid return
    /// and also create a binding to ensure we type check the expression.
    fn record_return(&mut self, mut x: StmtReturn) {
        self.ensure_expr_opt(x.value.as_deref_mut());
        if let Err(oops_top_level) = self.scopes.record_or_reject_return(x) {
            if let Some(x) = oops_top_level.value {
                self.insert_binding(Key::Anon(x.range()), Binding::Expr(None, *x));
            }
            self.error(
                oops_top_level.range,
                "Invalid `return` outside of a function".to_owned(),
                ErrorKind::BadReturn,
            );
        }
        self.scopes.mark_flow_termination();
    }

    /// Evaluate the statements and update the bindings.
    /// Every statement should end up in the bindings, perhaps with a location that is never used.
    pub fn stmt(&mut self, x: Stmt) {
        match x {
            Stmt::FunctionDef(x) => {
                self.function_def(x);
            }
            Stmt::ClassDef(x) => self.class_def(x),
            Stmt::Return(x) => {
                self.record_return(x);
            }
            Stmt::Delete(mut x) => {
                for target in &mut x.targets {
                    self.insert_binding(
                        KeyExpect(target.range()),
                        BindingExpect::Delete(Box::new(target.clone())),
                    );
                    if let Expr::Name(name) = target {
                        let idx = self.ensure_mutable_name(name);
                        self.scopes.update_flow_info(
                            Hashed::new(&name.id),
                            idx,
                            Some(FlowStyle::Uninitialized),
                        );
                    } else {
                        self.ensure_expr(target);
                    }
                }
            }
            Stmt::Assign(ref x)
                if let [Expr::Name(name)] = x.targets.as_slice()
                    && let Some((module, forward)) =
                        resolve_typeshed_alias(self.module_info.name(), &name.id, &x.value) =>
            {
                self.bind_assign(name, |_| Binding::Import(module, forward))
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
                                    self.assign_enum(name, &mut call.func, arg_name, members);
                                    return;
                                }
                            }
                            SpecialExport::TypedDict => {
                                if let Some((arg_name, members)) =
                                    call.arguments.args.split_first_mut()
                                {
                                    self.assign_typed_dict(
                                        name,
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
                                    self.assign_typing_named_tuple(
                                        name,
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
                                    self.assign_collections_named_tuple(
                                        name,
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
                                    self.assign_new_type(name, new_type_name, base);
                                    return;
                                }
                            }
                            _ => {}
                        }
                    }
                    self.bind_name_assign(name, x.value)
                } else {
                    self.bind_targets_with_value(&mut x.targets, &mut x.value);
                }
            }
            Stmt::AugAssign(mut x) => {
                self.ensure_expr(&mut x.value);
                let make_binding = |ann| Binding::AugAssign(ann, x.clone());
                match x.target.as_ref() {
                    Expr::Name(name) => {
                        self.ensure_mutable_name(name);
                        self.bind_assign(name, make_binding);
                    }
                    Expr::Attribute(x) => {
                        let make_assigned_value = &|ann| ExprOrBinding::Binding(make_binding(ann));
                        self.bind_attr_assign(x.clone(), make_assigned_value);
                    }
                    Expr::Subscript(x) => {
                        let make_assigned_value = &|ann| ExprOrBinding::Binding(make_binding(ann));
                        self.bind_subscript_assign(x.clone(), make_assigned_value);
                    }
                    illegal_target => {
                        // Most structurally invalid targets become errors in the parser, which we propagate so there
                        // is no need for duplicate errors. But we do want to catch unbound names (which the parser
                        // will not catch)
                        let mut e = illegal_target.clone();
                        self.ensure_expr(&mut e);
                    }
                }
            }
            Stmt::AnnAssign(mut x) => match *x.target {
                Expr::Name(name) => {
                    let name = Ast::expr_name_identifier(name);
                    let ann_key = KeyAnnotation::Annotation(ShortIdentifier::new(&name));
                    let in_class_body = self.scopes.in_class_body();
                    self.ensure_type(&mut x.annotation, &mut None);
                    let ann_val = if let Some(special) = SpecialForm::new(&name.id, &x.annotation) {
                        BindingAnnotation::Type(
                            AnnotationTarget::Assign(name.id.clone(), Initialized::Yes),
                            special.to_type(),
                        )
                    } else {
                        BindingAnnotation::AnnotateExpr(
                            if in_class_body {
                                AnnotationTarget::ClassMember(name.id.clone())
                            } else {
                                AnnotationTarget::Assign(
                                    name.id.clone(),
                                    if x.value.is_some() {
                                        Initialized::Yes
                                    } else {
                                        Initialized::No
                                    },
                                )
                            },
                            *x.annotation.clone(),
                            None,
                        )
                    };
                    let ann_key = self.insert_binding(ann_key, ann_val);
                    let flow_style = if in_class_body {
                        let initial_value = x.value.as_deref().cloned();
                        FlowStyle::ClassField { initial_value }
                    } else if x.value.is_some() {
                        FlowStyle::Other
                    } else {
                        FlowStyle::Uninitialized
                    };
                    let binding_value = if let Some(value) = x.value {
                        // Treat a name as initialized, but skip actually checking the value, if we are assigning `...` in a stub.
                        if self.module_info.path().is_interface()
                            && matches!(&*value, Expr::EllipsisLiteral(_))
                        {
                            None
                        } else {
                            Some(value)
                        }
                    } else {
                        None
                    };
                    let binding = if let Some(mut value) = binding_value {
                        // Handle forward references in explicit type aliases.
                        if self.as_special_export(&x.annotation) == Some(SpecialExport::TypeAlias) {
                            self.ensure_type(&mut value, &mut None);
                        } else {
                            self.ensure_expr(&mut value);
                        }
                        Binding::NameAssign(
                            name.id.clone(),
                            Some((AnnotationStyle::Direct, ann_key)),
                            value,
                        )
                    } else {
                        Binding::AnnotatedType(
                            ann_key,
                            Box::new(Binding::Type(Type::any_implicit())),
                        )
                    };
                    if let Some(ann) = self.bind_definition(&name, binding, flow_style)
                        && ann != ann_key
                    {
                        self.insert_binding(
                            KeyExpect(name.range),
                            BindingExpect::Redefinition {
                                new: ann_key,
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
                        Some(mut v) => {
                            self.ensure_expr(&mut v);
                            let make_value = |_| ExprOrBinding::Expr(*v);
                            self.bind_attr_assign(attr.clone(), make_value)
                        }
                        _ => ExprOrBinding::Binding(Binding::Type(Type::any_implicit())),
                    };
                    if !self
                        .scopes
                        .record_self_attr_assign(&attr, value.clone(), Some(ann_key))
                    {
                        self.error(
                             x.range,
                             format!(
                                 "Type cannot be declared in assignment to non-self attribute `{}.{}`",
                                 self.module_info.display(&attr.value),
                                 attr_name,
                             ),
                             ErrorKind::BadAssignment,
                         );
                    }
                }
                mut target => {
                    if matches!(&target, Expr::Subscript(..)) {
                        // Note that for Expr::Subscript Python won't fail at runtime,
                        // but Mypy and Pyright both error here, so let's do the same.
                        self.error(
                            x.annotation.range(),
                            "Subscripts should not be annotated".to_owned(),
                            ErrorKind::InvalidSyntax,
                        );
                    }
                    // Try and continue as much as we can, by throwing away the type or just binding to error
                    match x.value {
                        Some(value) => self.stmt(Stmt::Assign(StmtAssign {
                            range: x.range,
                            targets: vec![target],
                            value,
                        })),
                        None => {
                            self.bind_target(&mut target, &|_| Binding::Type(Type::any_error()));
                        }
                    }
                }
            },
            Stmt::TypeAlias(mut x) => {
                if let Expr::Name(name) = *x.name {
                    if let Some(params) = &mut x.type_params {
                        self.type_params(params);
                    }
                    self.ensure_type(&mut x.value, &mut None);
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
                        "Invalid assignment target".to_owned(),
                        ErrorKind::InvalidSyntax,
                    );
                }
            }
            Stmt::For(mut x) => {
                self.ensure_expr(&mut x.iter);
                self.setup_loop(x.range, &NarrowOps::new());
                let make_binding =
                    |ann| Binding::IterableValue(ann, *x.iter.clone(), IsAsync::new(x.is_async));
                self.bind_target(&mut x.target, &make_binding);
                self.stmts(x.body);
                self.teardown_loop(x.range, &NarrowOps::new(), x.orelse);
            }
            Stmt::While(mut x) => {
                let narrow_ops = NarrowOps::from_expr(self, Some(&x.test));
                self.setup_loop(x.range, &narrow_ops);
                self.ensure_expr(&mut x.test);
                let range = x.test.range();
                self.insert_binding(Key::Anon(range), Binding::Expr(None, *x.test.clone()));
                // Typecheck the test condition during solving.
                self.insert_binding(
                    KeyExpect(range),
                    BindingExpect::Bool(Box::new(*x.test), range),
                );
                self.stmts(x.body);
                self.teardown_loop(x.range, &narrow_ops, x.orelse);
            }
            Stmt::If(x) => {
                let range = x.range;
                let mut exhaustive = false;
                let mut branches = Vec::new();
                // Type narrowing operations that are carried over from one branch to the next. For example, in:
                //   if x is None:
                //     pass
                //   else:
                //     pass
                // x is bound to Narrow(x, Is(None)) in the if branch, and the negation, Narrow(x, IsNot(None)),
                // is carried over to the else branch.
                let mut negated_prev_ops = NarrowOps::new();
                let mut implicit_else = true;
                for (range, test, body) in Ast::if_branches_owned(x) {
                    let this_branch_chosen = self.sys_info.evaluate_bool_opt(test.as_ref());
                    if this_branch_chosen == Some(false) {
                        continue; // We definitely won't pick this branch
                    }
                    self.bind_narrow_ops(&negated_prev_ops, range);
                    let mut base = self.scopes.clone_current_flow();
                    let new_narrow_ops = NarrowOps::from_expr(self, test.as_ref());
                    if let Some(mut e) = test {
                        self.ensure_expr(&mut e);
                        self.insert_binding(Key::Anon(e.range()), Binding::Expr(None, e.clone()));
                        // Typecheck the test condition during solving.
                        self.insert_binding(
                            KeyExpect(e.range()),
                            BindingExpect::Bool(Box::new(e.clone()), range),
                        );
                    } else {
                        implicit_else = false;
                    }
                    self.bind_narrow_ops(&new_narrow_ops, range);
                    negated_prev_ops.and_all(new_narrow_ops.negate());
                    self.stmts(body);
                    self.scopes.swap_current_flow_with(&mut base);
                    branches.push(base);
                    if this_branch_chosen == Some(true) {
                        exhaustive = true;
                        break; // We definitely picked this branch if we got here, nothing below is reachable.
                    }
                }
                // If the conditions are exhaustive, then we only need to merge the branches.
                //
                // Otherwise, we need to merge branches with `base` (which was
                // the flow above the `If`) because the if might be skipped
                // entirely.
                if exhaustive {
                    self.set_current_flow_to_merged_branches(branches, range);
                } else {
                    if implicit_else {
                        // If there is no explicit else branch, we still want to merge the negated ops
                        // from the previous branches into the flow env.
                        // Note, using a default use_range is OK. The range is only needed to make the
                        // key distinct from other keys.
                        self.bind_narrow_ops(&negated_prev_ops, TextRange::default());
                    }
                    self.merge_branches_into_current(branches, range);
                }
            }
            Stmt::With(x) => {
                let kind = IsAsync::new(x.is_async);
                for mut item in x.items {
                    self.ensure_expr(&mut item.context_expr);
                    let item_range = item.range();
                    let expr_range = item.context_expr.range();
                    let context_idx = self.insert_binding(
                        Key::ContextExpr(expr_range),
                        Binding::Expr(None, item.context_expr),
                    );
                    if let Some(mut opts) = item.optional_vars {
                        let make_binding =
                            |ann| Binding::ContextValue(ann, context_idx, expr_range, kind);
                        self.bind_target(&mut opts, &make_binding);
                    } else {
                        self.insert_binding(
                            Key::Anon(item_range),
                            Binding::ContextValue(None, context_idx, expr_range, kind),
                        );
                    }
                }
                self.stmts(x.body);
            }
            Stmt::Match(x) => {
                self.stmt_match(x);
            }
            Stmt::Raise(x) => {
                if let Some(mut exc) = x.exc {
                    self.ensure_expr(&mut exc);
                    let raised = if let Some(mut cause) = x.cause {
                        self.ensure_expr(&mut cause);
                        RaisedException::WithCause(Box::new((*exc, *cause)))
                    } else {
                        RaisedException::WithoutCause(*exc)
                    };
                    self.insert_binding(
                        KeyExpect(x.range),
                        BindingExpect::CheckRaisedException(raised),
                    );
                } else {
                    // If there's no exception raised, don't bother checking the cause.
                }
                self.scopes.mark_flow_termination();
            }
            Stmt::Try(x) => {
                let range = x.range;
                let mut branches = Vec::new();
                let mut base = self.scopes.clone_current_flow();

                // We branch before the body, conservatively assuming that any statement can fail
                // entry -> try -> else -> finally
                //   |                     ^
                //   ----> handler --------|

                self.stmts(x.body);
                self.stmts(x.orelse);
                self.scopes.swap_current_flow_with(&mut base);
                branches.push(base);

                for h in x.handlers {
                    base = self.scopes.clone_current_flow();
                    let range = h.range();
                    let h = h.except_handler().unwrap(); // Only one variant for now
                    if let Some(name) = h.name
                        && let Some(mut type_) = h.type_
                    {
                        self.ensure_expr(&mut type_);
                        self.bind_definition(
                            &name,
                            Binding::ExceptionHandler(type_, x.is_star),
                            FlowStyle::Other,
                        );
                    } else if let Some(mut type_) = h.type_ {
                        self.ensure_expr(&mut type_);
                        self.insert_binding(
                            Key::Anon(range),
                            Binding::ExceptionHandler(type_, x.is_star),
                        );
                    }
                    self.stmts(h.body);
                    self.scopes.swap_current_flow_with(&mut base);
                    branches.push(base);
                }

                self.set_current_flow_to_merged_branches(branches, range);
                self.stmts(x.finalbody);
            }
            Stmt::Assert(mut x) => {
                self.ensure_expr(&mut x.test);
                self.bind_narrow_ops(&NarrowOps::from_expr(self, Some(&x.test)), x.range);
                self.insert_binding(Key::Anon(x.test.range()), Binding::Expr(None, *x.test));
                if let Some(mut msg_expr) = x.msg {
                    self.ensure_expr(&mut msg_expr);
                    self.insert_binding(
                        KeyExpect(msg_expr.range()),
                        BindingExpect::TypeCheckExpr(Box::new(*msg_expr)),
                    );
                };
            }
            Stmt::Import(x) => {
                for x in x.names {
                    let m = ModuleName::from_name(&x.name.id);
                    if let Err(err @ FindError::NotFound(..)) = self.lookup.get(m) {
                        self.error(x.range, err.display(), ErrorKind::ImportError);
                    }
                    match x.asname {
                        Some(asname) => {
                            self.bind_definition(
                                &asname,
                                Binding::Module(m, m.components(), None),
                                FlowStyle::ImportAs(m),
                            );
                        }
                        None => {
                            let first = m.first_component();
                            let flow_info = self.scopes.current().flow.info.get(&first);
                            let module_key = match flow_info {
                                Some(flow_info)
                                    if matches!(flow_info.style, FlowStyle::MergeableImport(_)) =>
                                {
                                    Some(flow_info.key)
                                }
                                _ => None,
                            };
                            let key = self.insert_binding(
                                Key::Import(first.clone(), x.name.range),
                                Binding::Module(m, vec![first.clone()], module_key),
                            );
                            self.bind_key(&first, key, FlowStyle::MergeableImport(m));
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
                        Ok(module_exports) => {
                            let exported = module_exports.exports(self.lookup);
                            for x in x.names {
                                if &x.name == "*" {
                                    for name in module_exports.wildcard(self.lookup).iter_hashed() {
                                        let key = Key::Import(name.into_key().clone(), x.range);
                                        let val = if exported.contains_key_hashed(name) {
                                            Binding::Import(m, name.into_key().clone())
                                        } else {
                                            self.error(
                                                x.range,
                                                format!("Could not import `{name}` from `{m}`"),
                                                ErrorKind::MissingModuleAttribute,
                                            );
                                            Binding::Type(Type::any_error())
                                        };
                                        let key = self.insert_binding(key, val);
                                        self.bind_key(
                                            name.key(),
                                            key,
                                            FlowStyle::Import(m, name.into_key().clone()),
                                        );
                                    }
                                } else {
                                    let asname = x.asname.unwrap_or_else(|| x.name.clone());
                                    // A `from x import y` statement is ambiguous; if `x` is a package with
                                    // an `__init__.py` file, then it might import the name `y` from the
                                    // module `x` defined by the `__init__.py` file, or it might import a
                                    // submodule `x.y` of the package `x`.
                                    //
                                    // If both are present, generally we prefer the name defined in `x`,
                                    // but there is an exception: if we are already looking at the
                                    // `__init__` module of `x`, we always prefer the submodule.
                                    let val = if (self.module_info.name() != m)
                                        && exported.contains_key(&x.name.id)
                                    {
                                        Binding::Import(m, x.name.id.clone())
                                    } else {
                                        let x_as_module_name = m.append(&x.name.id);
                                        if self.lookup.get(x_as_module_name).is_ok() {
                                            Binding::Module(
                                                x_as_module_name,
                                                x_as_module_name.components(),
                                                None,
                                            )
                                        } else {
                                            self.error(
                                                x.range,
                                                format!(
                                                    "Could not import `{}` from `{m}`",
                                                    x.name.id
                                                ),
                                                ErrorKind::MissingModuleAttribute,
                                            );
                                            Binding::Type(Type::any_error())
                                        }
                                    };
                                    self.bind_definition(
                                        &asname,
                                        val,
                                        FlowStyle::Import(m, x.name.id),
                                    );
                                }
                            }
                        }
                        Err(FindError::Ignored) => self.bind_unimportable_names(&x),
                        Err(
                            err @ (FindError::NoPyTyped
                            | FindError::NoSource(_)
                            | FindError::NotFound(..)),
                        ) => {
                            self.error(x.range, err.display(), ErrorKind::ImportError);
                            self.bind_unimportable_names(&x);
                        }
                    }
                } else {
                    self.error(
                        x.range,
                        format!(
                            "Could not resolve relative import `{}`",
                            ".".repeat(x.level as usize)
                        ),
                        ErrorKind::ImportError,
                    );
                    self.bind_unimportable_names(&x);
                }
            }
            Stmt::Global(x) => {
                for name in x.names {
                    self.define_global_name(&name);
                }
            }
            Stmt::Nonlocal(x) => {
                for name in x.names {
                    self.define_nonlocal_name(&name);
                }
            }
            Stmt::Expr(mut x) => {
                self.ensure_expr(&mut x.value);
                self.insert_binding(
                    Key::StmtExpr(x.value.range()),
                    Binding::Expr(None, *x.value),
                );
            }
            Stmt::Pass(_) => { /* no-op */ }
            Stmt::Break(x) => {
                self.add_loop_exitpoint(LoopExit::Break, x.range);
            }
            Stmt::Continue(x) => {
                self.add_loop_exitpoint(LoopExit::Continue, x.range);
            }
            Stmt::IpyEscapeCommand(x) => self.error(
                x.range,
                "IPython escapes are not supported".to_owned(),
                ErrorKind::Unsupported,
            ),
        }
    }
}
