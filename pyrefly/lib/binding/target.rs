/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprName;
use ruff_python_ast::ExprSubscript;
use ruff_text_size::TextRange;
use starlark_map::Hashed;

use crate::binding::binding::Binding;
use crate::binding::binding::BindingExpect;
use crate::binding::binding::ExprOrBinding;
use crate::binding::binding::Key;
use crate::binding::binding::KeyAnnotation;
use crate::binding::binding::KeyExpect;
use crate::binding::binding::SizeExpectation;
use crate::binding::binding::UnpackedPosition;
use crate::binding::bindings::BindingsBuilder;
use crate::binding::bindings::LookupKind;
use crate::binding::narrow::identifier_and_chain_prefix_for_expr;
use crate::binding::scope::FlowStyle;
use crate::error::kind::ErrorKind;
use crate::graph::index::Idx;
use crate::module::short_identifier::ShortIdentifier;

impl<'a> BindingsBuilder<'a> {
    /// Bind one level of an unpacked LHS target, for example in `x, (y, [*z]), q = foo`
    /// - one level handles `x`, `(y, [*z])`, and `q`
    /// - another level (called recursively via `bind_target`) handles `y` and `[*z]`
    ///
    /// Each potentially-recursive call results in two "levels" bindings:
    /// - A single `Key::Unpack` -> binding pair from whatever the value we are unpacking is
    /// - A `Binding::UnpackedValue` for each unpacked entry; these get passed back to
    ///   `bind_target` and used in different ways depending on the context.
    ///
    /// A few notes on how this interacts with `bind_target`:
    /// - We never contextually type unpacks, we do the unpacking at type level
    ///   for simplicity (for now).
    /// - Due to how `bind_target` works:
    ///   - the `Binding::UnpackedValue` will include an `ann` to check against if and only if
    ///     the individual unpack target is a name.
    ///   - If the unpack target is an attribute or subscript, we will instead pass it along
    ///     to the normal attribute-set or `__setitem__` call logic.
    fn bind_unpacking(
        &mut self,
        elts: &mut [Expr],
        make_binding: &dyn Fn(Option<Idx<KeyAnnotation>>) -> Binding,
        range: TextRange,
    ) {
        // We are going to use this binding many times, so compute it once.
        let idx_of_unpack = self.insert_binding(Key::Unpack(range), make_binding(None));

        // An unpacking has zero or one splats (starred expressions).
        let mut splat = false;
        let len = elts.len();
        for (i, e) in elts.iter_mut().enumerate() {
            match e {
                Expr::Starred(e) => {
                    splat = true;
                    // Counts how many elements are after the splat.
                    let j = len - i - 1;
                    let make_nested_binding = |ann| {
                        Binding::UnpackedValue(
                            ann,
                            idx_of_unpack,
                            range,
                            UnpackedPosition::Slice(i, j),
                        )
                    };
                    self.bind_target(&mut e.value, &make_nested_binding);
                }
                _ => {
                    let unpacked_position = if splat {
                        // If we've encountered a splat, we no longer know how many values have been consumed
                        // from the front, but we know how many are left at the back.
                        UnpackedPosition::ReverseIndex(len - i)
                    } else {
                        UnpackedPosition::Index(i)
                    };
                    let make_nested_binding =
                        |ann| Binding::UnpackedValue(ann, idx_of_unpack, range, unpacked_position);
                    self.bind_target(e, &make_nested_binding);
                }
            }
        }
        let expect = if splat {
            SizeExpectation::Ge(elts.len() - 1)
        } else {
            SizeExpectation::Eq(elts.len())
        };
        self.insert_binding(
            KeyExpect(range),
            BindingExpect::UnpackedLength(idx_of_unpack, range, expect),
        );
    }

    // Create a binding to verify that an attribute assignment is valid and
    // potentially narrow (or invalidate narrows on) the name assigned to.
    //
    // Return the value of the attribute assignment (as an ExprOrBinding);
    // this might be used to record self-attribute assignments.
    pub fn bind_attr_assign(
        &mut self,
        mut attr: ExprAttribute,
        make_assigned_value: impl FnOnce(Option<Idx<KeyAnnotation>>) -> ExprOrBinding,
    ) -> ExprOrBinding {
        self.ensure_expr(&mut attr.value);
        if let Some((identifier, _)) =
            identifier_and_chain_prefix_for_expr(&Expr::Attribute(attr.clone()))
        {
            let idx = self.idx_for_promise(Key::PropertyAssign(ShortIdentifier::new(&identifier)));
            let value = make_assigned_value(None);
            self.insert_binding_idx(
                idx,
                Binding::AssignToAttribute(Box::new((attr, value.clone()))),
            );
            let name = Hashed::new(&identifier.id);
            if self.lookup_name_hashed(name, LookupKind::Regular).is_ok() {
                self.scopes.update_flow_info(name, idx, None);
            }
            value
        } else {
            let idx = self.idx_for_promise(Key::Anon(attr.range));
            let value = make_assigned_value(None);
            self.insert_binding_idx(
                idx,
                Binding::AssignToAttribute(Box::new((attr, value.clone()))),
            );
            value
        }
    }

    // Create a binding to verify that a subscript assignment is valid and
    // potentially narrow (or invalidate narrows on) the name assigned to.
    pub fn bind_subscript_assign(
        &mut self,
        mut subscript: ExprSubscript,
        make_assigned_value: impl FnOnce(Option<Idx<KeyAnnotation>>) -> ExprOrBinding,
    ) {
        self.ensure_expr(&mut subscript.slice);
        self.ensure_expr(&mut subscript.value);
        if let Some((identifier, _)) =
            identifier_and_chain_prefix_for_expr(&Expr::Subscript(subscript.clone()))
        {
            let idx = self.idx_for_promise(Key::PropertyAssign(ShortIdentifier::new(&identifier)));
            let value = make_assigned_value(None);
            self.insert_binding_idx(
                idx,
                Binding::AssignToSubscript(Box::new((subscript, value))),
            );
            let name = Hashed::new(&identifier.id);
            if self.lookup_name_hashed(name, LookupKind::Regular).is_ok() {
                self.scopes.update_flow_info(name, idx, None);
            }
        } else {
            let idx = self.idx_for_promise(Key::Anon(subscript.range));
            let value = make_assigned_value(None);
            self.insert_binding_idx(
                idx,
                Binding::AssignToSubscript(Box::new((subscript, value))),
            );
        }
    }

    /// Bind the LHS of a target in a syntactic form (e.g. assignments, variables
    /// bound in a `for`` loop header, variables defined by a `with` statement header).
    ///
    /// The `target` is the LHS. It is an `Expr`, but in fact only a handful of forms
    /// are legal because targets can only be names, attributes, subscripts, or unpackngs. An
    /// example target illustrating all of the cases is `(x.y, d["k"], [z, *w, q])`
    ///
    /// The `make_binding` function is a callback to the caller, who is responsible for constructing
    /// a binding that provides the value of the RHS. To handle cases where the type of the LHS
    /// is restricted, it takes an optional `KeyAnnotation` which should be the annotation for the
    /// target when one is available.
    ///
    /// The `value` argument is only provided when handling top-level assignment targets;
    /// it enables contextual typing. At the moment it is only used in the attribute case (because
    /// the other cases instead rely on `make_binding` to handle contextual typing, which works
    /// when the form is not an unpacking but results in false negatives when it is).
    fn bind_target_impl(
        &mut self,
        target: &mut Expr,
        make_assigned_value: &dyn Fn(Option<Idx<KeyAnnotation>>) -> ExprOrBinding,
    ) {
        let make_binding = &|ann| match make_assigned_value(ann) {
            ExprOrBinding::Expr(e) => Binding::Expr(ann, e),
            ExprOrBinding::Binding(b) => b,
        };
        match target {
            Expr::Name(name) => {
                self.bind_assign(name, make_binding);
            }
            Expr::Attribute(x) => {
                let attr_value = self.bind_attr_assign(x.clone(), make_assigned_value);
                // If this is a self-assignment, record it because we may use it to infer
                // the existence of an instance-only attribute.
                self.scopes.record_self_attr_assign(x, attr_value, None);
            }
            Expr::Subscript(x) => {
                self.bind_subscript_assign(x.clone(), make_assigned_value);
            }
            Expr::Tuple(tup) => {
                self.bind_unpacking(&mut tup.elts, make_binding, tup.range);
            }
            Expr::List(lst) => {
                self.bind_unpacking(&mut lst.elts, make_binding, lst.range);
            }
            Expr::Starred(x) => {
                self.error(
                    x.range,
                    "Starred assignment target must be in a list or tuple".to_owned(),
                    ErrorKind::InvalidSyntax,
                );
                self.bind_target_impl(&mut x.value, make_assigned_value);
            }
            illegal_target => {
                // Most structurally invalid targets become errors in the parser, which we propagate so there
                // is no need for duplicate errors. But we do want to catch unbound names (which the parser
                // will not catch)
                self.ensure_expr(illegal_target);
            }
        }
    }

    pub fn bind_target(
        &mut self,
        target: &mut Expr,
        make_binding: &dyn Fn(Option<Idx<KeyAnnotation>>) -> Binding,
    ) {
        // TODO(stroxler): Clean this up: we're wrapping the binding and then just unwrapping it later.
        // Forcing all callers to produce an `ExprOrBinding` will also help us improve contextual typing.
        let make_assigned_value = &|ann| ExprOrBinding::Binding(make_binding(ann));
        self.bind_target_impl(target, make_assigned_value);
    }

    /// Similar to `bind_target`, but specifically for assignments:
    /// - Handles multi-target assignment
    /// - Takes the value as an `Expr` rather than a `make_binding` callaback, which enables
    ///   better contextual typing in cases where the assignment might actually invoke
    ///   a method (like descriptor attribute assigns and `__setitem__` calls).
    pub fn bind_targets_with_value(&mut self, targets: &mut Vec<Expr>, value: &mut Expr) {
        self.ensure_expr(value);
        let make_assigned_value = &|_| ExprOrBinding::Expr(value.clone());
        for target in targets {
            self.bind_target_impl(target, make_assigned_value);
        }
    }

    pub fn bind_assign(
        &mut self,
        name: &ExprName,
        make_binding: impl FnOnce(Option<Idx<KeyAnnotation>>) -> Binding,
    ) {
        let idx = self.idx_for_promise(Key::Definition(ShortIdentifier::expr_name(name)));
        let (ann, default) = self.bind_key(&name.id, idx, FlowStyle::Other);
        let mut binding = make_binding(ann);
        if let Some(default) = default {
            binding = Binding::Default(default, Box::new(binding));
        }
        self.insert_binding_idx(idx, binding);
    }
}
