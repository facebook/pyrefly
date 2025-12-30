/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_graph::index::Idx;
use pyrefly_python::ast::Ast;
use pyrefly_python::short_identifier::ShortIdentifier;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprBinOp;
use ruff_python_ast::ExprName;
use ruff_python_ast::ExprSubscript;
use ruff_python_ast::Identifier;
use ruff_python_ast::Operator;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::Hashed;

use crate::binding::binding::AnnotationStyle;
use crate::binding::binding::Binding;
use crate::binding::binding::BindingExpect;
use crate::binding::binding::ExprOrBinding;
use crate::binding::binding::FirstUse;
use crate::binding::binding::Key;
use crate::binding::binding::KeyAnnotation;
use crate::binding::binding::KeyExpect;
use crate::binding::binding::MethodSelfKind;
use crate::binding::binding::SizeExpectation;
use crate::binding::binding::UnpackedPosition;
use crate::binding::bindings::BindingsBuilder;
use crate::binding::bindings::LegacyTParamCollector;
use crate::binding::expr::Usage;
use crate::binding::narrow::identifier_and_chain_prefix_for_expr;
use crate::binding::scope::FlowStyle;
use crate::binding::scope::NameReadInfo;
use crate::export::special::SpecialExport;

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
        mut assigned: Option<&mut Expr>,
        make_binding: &dyn Fn(Option<&Expr>, Option<Idx<KeyAnnotation>>) -> Binding,
        range: TextRange,
        ensure_assigned: bool,
    ) {
        // Compute a binding for the RHS at this level of unpacking.
        // - If there is only one unpacking, this will be the actual RHS
        // - But if there are many layers of unpacking it might be an UnpackedValue pointing further up the tree
        //
        // The bindings directly used when binding components of LHS will be
        // `UnpackedValue` bindings that point at this.
        //
        // For example, in `x, (y, z) = foo()`:
        // - We will get two different `Key::Unpacked` bindings, one for the
        //   entire RHS and another one, pointing at the first one, for `(y, z)`.
        // - We will also get three `Key::Definition` bindings, one each for `x`, `y`, and `z`.
        let mut unpacked = self.declare_current_idx(Key::Unpack(range));
        if ensure_assigned && let Some(assigned) = &mut assigned {
            self.ensure_expr(assigned, unpacked.usage())
        }
        let unpack_idx =
            self.insert_binding_current(unpacked, make_binding(assigned.as_deref(), None));

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
                            unpack_idx,
                            range,
                            UnpackedPosition::Slice(i, j),
                        )
                    };
                    self.bind_target_no_expr(&mut e.value, &make_nested_binding);
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
                        |ann| Binding::UnpackedValue(ann, unpack_idx, range, unpacked_position);
                    self.bind_target_no_expr(e, &make_nested_binding);
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
            BindingExpect::UnpackedLength(unpack_idx, range, expect),
        );
    }

    /// Narrow a name to `Idx` if the name is defined in the current scope stack. Used
    /// to handle attribute and subscript assignment narrows, which we want to allow whenever
    /// the name was defined, but we don't want them to cause us to treat nonexistent names
    /// as defined downstream.
    fn narrow_if_name_is_defined(&mut self, identifier: Identifier, narrowed_idx: Idx<Key>) {
        let name = Hashed::new(&identifier.id);
        let name_is_defined = !matches!(
            self.scopes.look_up_name_for_read(name),
            NameReadInfo::NotFound,
        );
        if name_is_defined {
            self.scopes.narrow_in_current_flow(name, narrowed_idx);
        }
    }

    /// Create a binding to verify that an attribute assignment is valid and
    /// potentially narrow (or invalidate narrows on) the name assigned to.
    ///
    /// Return the value of the attribute assignment (as an ExprOrBinding);
    /// this might be used to record self-attribute assignments.
    pub fn bind_attr_assign_impl(
        &mut self,
        mut attr: ExprAttribute,
        mut assigned: Option<&mut Expr>,
        make_assigned_value: impl FnOnce(Option<&Expr>, Option<Idx<KeyAnnotation>>) -> ExprOrBinding,
        ensure_assigned: bool,
    ) -> ExprOrBinding {
        let narrowing_identifier =
            identifier_and_chain_prefix_for_expr(&Expr::Attribute(attr.clone()))
                .map(|(identifier, _)| identifier);
        let mut user = if let Some(identifier) = &narrowing_identifier {
            self.declare_current_idx(Key::FacetAssign(ShortIdentifier::new(identifier)))
        } else {
            self.declare_current_idx(Key::Anon(attr.range))
        };
        let allow_assign_to_final =
            self.scopes
                .method_that_sets_attr(&attr)
                .is_some_and(|method| {
                    method.recognized_attribute_defining_method
                        && matches!(method.instance_or_class, MethodSelfKind::Instance)
                });
        self.ensure_expr(&mut attr.value, user.usage());
        if ensure_assigned && let Some(assigned) = &mut assigned {
            self.ensure_expr(assigned, user.usage());
        }
        let value = make_assigned_value(assigned.as_deref(), None);
        let idx = self.insert_binding_current(
            user,
            Binding::AssignToAttribute {
                attr,
                value: Box::new(value.clone()),
                allow_assign_to_final,
            },
        );
        if let Some(identifier) = narrowing_identifier {
            self.narrow_if_name_is_defined(identifier, idx);
        }
        value
    }

    pub fn bind_attr_assign(
        &mut self,
        attr: ExprAttribute,
        assigned: &mut Expr,
        make_assigned_value: impl FnOnce(&Expr, Option<Idx<KeyAnnotation>>) -> ExprOrBinding,
    ) -> ExprOrBinding {
        self.bind_attr_assign_impl(
            attr,
            Some(assigned),
            |expr, ann| make_assigned_value(expr.unwrap(), ann),
            true,
        )
    }

    /// Create a binding to verify that a subscript assignment is valid and
    /// potentially narrow (or invalidate narrows on) the name assigned to.
    pub fn bind_subscript_assign_impl(
        &mut self,
        mut subscript: ExprSubscript,
        mut assigned: Option<&mut Expr>,
        make_assigned_value: impl FnOnce(Option<&Expr>, Option<Idx<KeyAnnotation>>) -> ExprOrBinding,
        ensure_assigned: bool,
    ) {
        let narrowing_identifier =
            identifier_and_chain_prefix_for_expr(&Expr::Subscript(subscript.clone()))
                .map(|(identifier, _)| identifier);
        let mut user = if let Some(identifier) = &narrowing_identifier {
            self.declare_current_idx(Key::FacetAssign(ShortIdentifier::new(identifier)))
        } else {
            self.declare_current_idx(Key::Anon(subscript.range))
        };
        self.ensure_expr(&mut subscript.slice, user.usage());
        self.ensure_expr(&mut subscript.value, user.usage());
        if ensure_assigned && let Some(assigned) = &mut assigned {
            self.ensure_expr(assigned, user.usage());
        }
        let value = make_assigned_value(assigned.as_deref(), None);
        let idx = self
            .insert_binding_current(user, Binding::AssignToSubscript(subscript, Box::new(value)));
        if let Some(identifier) = narrowing_identifier {
            self.narrow_if_name_is_defined(identifier, idx);
        }
    }

    pub fn bind_subscript_assign(
        &mut self,
        subscript: ExprSubscript,
        assigned: &mut Expr,
        make_assigned_value: impl FnOnce(&Expr, Option<Idx<KeyAnnotation>>) -> ExprOrBinding,
    ) {
        self.bind_subscript_assign_impl(
            subscript,
            Some(assigned),
            |expr, ann| make_assigned_value(expr.unwrap(), ann),
            true,
        )
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
        mut assigned: Option<&mut Expr>,
        make_assigned_value: &dyn Fn(Option<&Expr>, Option<Idx<KeyAnnotation>>) -> ExprOrBinding,
        ensure_assigned: bool,
    ) {
        let binding_of = |v, ann| match v {
            ExprOrBinding::Expr(e) => Binding::Expr(ann, e),
            ExprOrBinding::Binding(b) => b,
        };
        match target {
            Expr::Name(name) => {
                self.bind_target_name(
                    name,
                    assigned,
                    |expr, ann| binding_of(make_assigned_value(expr, ann), ann),
                    ensure_assigned,
                );
            }
            Expr::Attribute(x) => {
                let attr_value = self.bind_attr_assign_impl(
                    x.clone(),
                    assigned,
                    make_assigned_value,
                    ensure_assigned,
                );
                // If this is a self-assignment, record it because we may use it to infer
                // the existence of an instance-only attribute.
                self.scopes.record_self_attr_assign(x, attr_value, None);
            }
            Expr::Subscript(x) => {
                self.bind_subscript_assign_impl(
                    x.clone(),
                    assigned,
                    make_assigned_value,
                    ensure_assigned,
                );
            }
            Expr::Tuple(tup) => {
                self.bind_unpacking(
                    &mut tup.elts,
                    assigned,
                    &|expr, ann| binding_of(make_assigned_value(expr, ann), ann),
                    tup.range,
                    ensure_assigned,
                );
            }
            Expr::List(lst) => {
                self.bind_unpacking(
                    &mut lst.elts,
                    assigned,
                    &|expr, ann| binding_of(make_assigned_value(expr, ann), ann),
                    lst.range,
                    ensure_assigned,
                );
            }
            Expr::Starred(x) => {
                self.bind_target_impl(&mut x.value, assigned, make_assigned_value, ensure_assigned);
            }
            illegal_target => {
                // Most structurally invalid targets become errors in the parser, which we propagate so there
                // is no need for duplicate errors. But we do want to catch unbound names (which the parser
                // will not catch).
                //
                // We ignore such names for first-usage-tracking purposes, since
                // we are not going to analyze the code at all.
                self.ensure_expr(illegal_target, &mut Usage::StaticTypeInformation);

                // Make sure the RHS is properly bound, so that we can report errors there.
                let mut user = self.declare_current_idx(Key::Anon(illegal_target.range()));
                if ensure_assigned && let Some(assigned) = &mut assigned {
                    self.ensure_expr(assigned, user.usage());
                }
                let binding = binding_of(make_assigned_value(assigned.as_deref(), None), None);
                self.insert_binding_current(user, binding);
            }
        }
    }

    pub fn bind_target_with_expr(
        &mut self,
        target: &mut Expr,
        assigned: &mut Expr,
        make_binding: &dyn Fn(&Expr, Option<Idx<KeyAnnotation>>) -> Binding,
    ) {
        self.bind_target_impl(
            target,
            Some(assigned),
            &|expr, ann| ExprOrBinding::Binding(make_binding(expr.unwrap(), ann)),
            true,
        );
    }

    pub fn bind_target_no_expr(
        &mut self,
        target: &mut Expr,
        make_binding: &dyn Fn(Option<Idx<KeyAnnotation>>) -> Binding,
    ) {
        self.bind_target_impl(
            target,
            None,
            &|_, ann| ExprOrBinding::Binding(make_binding(ann)),
            false,
        );
    }

    /// Similar to `bind_target`, but specifically for assignments that are *not* to a single name
    /// - Handles attribute and subscript assignment. If there is only one target
    ///   we will use contextual typing.
    /// - Handles multi-target assignment; in this case we will create an anonymous
    ///   binding to type check the RHS non-contextually, and then type check each
    ///   assignment - this is needed both for usage tracking and to avoid overly
    ///   aggressive contextual typing.
    pub fn bind_targets_with_value(&mut self, targets: &mut [Expr], value: &mut Expr) {
        match targets {
            [] => {}
            [target] => {
                self.bind_target_impl(
                    target,
                    Some(value),
                    &|expr, _| ExprOrBinding::Expr(expr.unwrap().clone()),
                    true,
                );
            }
            _ => {
                let mut user = self.declare_current_idx(Key::Anon(value.range()));
                self.ensure_expr(value, user.usage());
                let rhs_idx = self.insert_binding_current(user, Binding::Expr(None, value.clone()));
                for target in targets.iter_mut() {
                    let range = target.range();
                    self.bind_target_impl(
                        target,
                        None,
                        &|_, ann| {
                            ExprOrBinding::Binding(Binding::MultiTargetAssign(ann, rhs_idx, range))
                        },
                        false,
                    );
                }
            }
        }
    }

    /// Given a function that produces a binding from an ensured expression:
    /// - Ensure the expression, if there is one we are supposed to ensure
    /// - Update the bindings table and flow info to note that:
    ///   - the name is now bound to a `Key::Definition` + the computed binding
    ///   - the flow style is `FlowStyle::Other`
    fn bind_target_name(
        &mut self,
        name: &ExprName,
        mut assigned: Option<&mut Expr>,
        make_binding: impl FnOnce(Option<&Expr>, Option<Idx<KeyAnnotation>>) -> Binding,
        ensure_assigned: bool,
    ) {
        let mut user = self.declare_current_idx(Key::Definition(ShortIdentifier::expr_name(name)));
        if ensure_assigned && let Some(assigned) = &mut assigned {
            self.ensure_expr(assigned, user.usage());
        }
        let ann = if !Ast::is_synthesized_empty_name(name) {
            self.bind_current(&name.id, &user, FlowStyle::Other)
        } else {
            // Parser error recovery can synthesize walrus targets with empty identifiers.
            // Pretend the name binding succeeded so we still analyze the RHS, but skip
            // putting an entry into the flow-sensitive scope to avoid panics later.
            None
        };
        let binding = make_binding(assigned.as_deref(), ann);
        self.insert_binding_current(user, binding);
    }

    /// Handle single assignment: this is closely related to `bind_target_name`, but
    /// handles additional concerns (such as type alias logic) that don't apply to
    /// other target name assignments.
    ///
    /// It is used for
    /// - single-name `Assign` statements
    /// - for `AnnAssign` when there is a value assigned
    ///
    /// This is the only scenario where Pyrefly allows placeholder types to leak
    /// into a binding result as `Var`s. To do that, we create two separate bindings:
    /// the `Key::Definition` binding, which potentially contains placeholder type `Var`s
    /// in its result, and a `Key::PinnedDefinition` binding which will attempt to
    /// infer types based on the first use, and force them to default types otherwise.
    ///
    /// The pinned definition is the one that goes into scopes, and normal name lookups
    /// will see that - only a first use binding may see the raw, unpinned result.
    pub fn bind_single_name_assign(
        &mut self,
        name: &Identifier,
        mut value: Box<Expr>,
        direct_ann: Option<(&Expr, Idx<KeyAnnotation>)>,
    ) -> Option<Idx<KeyAnnotation>> {
        let identifier = ShortIdentifier::new(name);
        let mut user = self.declare_current_idx(Key::Definition(identifier));
        let pinned_idx = self.idx_for_promise(Key::CompletedPartialType(identifier));
        let is_definitely_type_alias = if let Some((e, _)) = direct_ann
            && self.as_special_export(e) == Some(SpecialExport::TypeAlias)
        {
            true
        } else {
            self.is_definitely_type_alias_rhs(value.as_ref())
        };
        let mut tparams = None;
        if is_definitely_type_alias {
            let mut legacy = Some(LegacyTParamCollector::new(false));
            self.ensure_type(&mut value, &mut legacy);
            if let Some(collector) = legacy {
                tparams = Some(collector.lookup_keys().into_boxed_slice());
            }
        } else {
            self.ensure_expr(&mut value, user.usage());
        }
        let style = if self.scopes.in_class_body() {
            FlowStyle::ClassField {
                initial_value: Some((*value).clone()),
            }
        } else {
            self.scopes.register_variable(name);
            FlowStyle::Other
        };
        let canonical_ann = self.bind_name(&name.id, pinned_idx, style);
        let ann = match direct_ann {
            Some((_, idx)) => Some((AnnotationStyle::Direct, idx)),
            None => canonical_ann.map(|idx| (AnnotationStyle::Forwarded, idx)),
        };
        let binding = Binding::NameAssign {
            name: name.id.clone(),
            annotation: ann,
            expr: value,
            legacy_tparams: tparams,
            is_in_function_scope: self.scopes.in_function_scope(),
        };
        // Record the raw assignment
        let (first_used_by, def_idx) = user.decompose();
        let def_idx = self.insert_binding_idx(def_idx, binding);
        // If this is a first use, add a binding that will eliminate any placeholder types coming from upstream.
        let unpinned_idx = if first_used_by.is_empty() {
            def_idx
        } else {
            self.insert_binding(
                Key::PartialTypeWithUpstreamsCompleted(identifier),
                Binding::PartialTypeWithUpstreamsCompleted(
                    def_idx,
                    first_used_by.into_iter().collect(),
                ),
            )
        };
        // Insert the Pin binding that will pin any types, potentially after evaluating the first downstream use.
        self.insert_binding_idx(
            pinned_idx,
            Binding::CompletedPartialType(unpinned_idx, FirstUse::Undetermined),
        );
        canonical_ann
    }

    /// If someone does `x = C["test"]`, that might be a type alias, it might not.
    /// Use this heuristic to detect things that are definitely type aliases.
    fn is_definitely_type_alias_rhs(&mut self, x: &Expr) -> bool {
        match x {
            Expr::Subscript(x) => matches!(
                self.as_special_export(&x.value),
                Some(
                    SpecialExport::Union
                        | SpecialExport::Optional
                        | SpecialExport::Annotated
                        | SpecialExport::Callable
                        | SpecialExport::BuiltinsDict
                        | SpecialExport::TypingDict
                        | SpecialExport::BuiltinsList
                        | SpecialExport::TypingList
                        | SpecialExport::BuiltinsTuple
                        | SpecialExport::TypingTuple
                        | SpecialExport::BuiltinsType
                        | SpecialExport::TypingType
                )
            ),
            Expr::BinOp(ExprBinOp {
                left,
                op: Operator::BitOr,
                right,
                ..
            }) => {
                self.is_definitely_type_alias_rhs(left) || self.is_definitely_type_alias_rhs(right)
            }
            _ => false,
        }
    }
}
