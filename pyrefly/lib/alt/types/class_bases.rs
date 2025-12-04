/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;

use dupe::Dupe;
use pyrefly_derive::TypeEq;
use pyrefly_derive::VisitMut;
use pyrefly_python::ast::Ast;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::class::ClassType;
use pyrefly_types::special_form::SpecialForm;
use pyrefly_types::typed_dict::TypedDict;
use pyrefly_util::display::commas_iter;
use ruff_python_ast::Expr;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::solve::TypeFormContext;
use crate::binding::base_class::BaseClass;
use crate::binding::base_class::BaseClassExpr;
use crate::binding::binding::Key;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorInfo;
use crate::error::style::ErrorStyle;
use crate::types::class::Class;
use crate::types::tuple::Tuple;
use crate::types::types::Type;

/// Information computed from the full base types of a class
/// - The base types themselves
/// - Some additional metadata derived from the bases
///
/// This is intended to be used for any downstream computation that needs to inspect the full types
/// (in particular, the targs of generic bases) of the bases of a class. If only the class objects are
/// needed, query `ClassMetadata` instead since that one doesn't require calculating the full types.
///
/// The reason this is tracked separately from `ClassMetadata` is to avoid the possiblity of
/// cycles when type arguments of the base classes may depend on the class itself.
#[derive(Debug, Clone, TypeEq, PartialEq, Eq, VisitMut, Default)]
pub struct ClassBases {
    /// The direct base types in the base class list
    base_types: Box<[ClassType]>,
    /// The first tuple ancestor, if there is one in the inheritance tree.
    ///
    /// This is recursively computed, not just a direct tuple base. We throw an error and keep only
    /// the first tuple ancestor if there are multiple (unless one of them is `tuple[Any, ...]` in which
    /// case we prefer the more precise type).
    tuple_ancestor: Option<Tuple>,
    /// Is this a pydantic strict model? Part of ClassBases because computation for this involves matching base class ASTs.
    pub has_pydantic_strict_metadata: bool,
}

impl ClassBases {
    pub fn recursive() -> Self {
        Self {
            base_types: Box::new([]),
            tuple_ancestor: None,
            has_pydantic_strict_metadata: false,
        }
    }

    pub fn tuple_ancestor(&self) -> Option<&Tuple> {
        self.tuple_ancestor.as_ref()
    }

    pub fn iter(&self) -> impl Iterator<Item = &ClassType> {
        self.base_types.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.base_types.is_empty()
    }

    pub fn base_type_count(&self) -> usize {
        self.base_types.len()
    }
}

impl fmt::Display for ClassBases {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ClassBases({})", commas_iter(|| self.base_types.iter()))
    }
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    // This is a version of `subscript_infer_for_type` with very restricted capability, in order to avoid cyclic dependencies
    fn base_class_subscript_infer(
        &self,
        mut base: Type,
        slice: &Expr,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> (Type, bool) {
        if let Type::Var(v) = base {
            base = self.solver().force_var(v);
        }
        if matches!(&base, Type::ClassDef(t) if t.name() == "tuple") {
            base = Type::type_form(Type::SpecialForm(SpecialForm::Tuple));
        }
        let mut has_strict = false;
        let arguments_untype = |slice: &Expr, has_strict: &mut bool| {
            Ast::unpack_slice(slice)
                .iter()
                .map(|x| match BaseClassExpr::from_expr(x) {
                    Some(base_expr) => {
                        let (ty, arg_has_strict) = self.base_class_expr_untype(
                            &base_expr,
                            TypeFormContext::TypeArgument,
                            errors,
                        );
                        if arg_has_strict {
                            *has_strict = true;
                        }
                        ty
                    }
                    None => self.expr_untype(x, TypeFormContext::TypeArgument, errors),
                })
                .collect::<Vec<_>>()
        };
        let result = match base {
            Type::Forall(forall) => {
                let tys = arguments_untype(slice, &mut has_strict);
                self.specialize_forall_in_base_class(*forall, tys, range, errors)
            }
            Type::ClassDef(cls) => Type::type_form(self.specialize_in_base_class(
                &cls,
                arguments_untype(slice, &mut has_strict),
                range,
                errors,
            )),
            Type::Type(box Type::SpecialForm(special)) => {
                self.apply_special_form(special, slice, range, errors)
            }
            Type::Any(style) => style.propagate(),
            t => self.error(
                errors,
                range,
                ErrorInfo::Kind(ErrorKind::UnsupportedOperation),
                format!(
                    "`{}` is not a subscriptable type on base class list",
                    self.for_display(t)
                ),
            ),
        };
        (result, has_strict)
    }

    fn base_class_expr_infer(&self, expr: &BaseClassExpr, errors: &ErrorCollector) -> (Type, bool) {
        match expr {
            BaseClassExpr::Name(x) => (
                self.get(&Key::BoundName(ShortIdentifier::expr_name(x)))
                    .arc_clone_ty(),
                false,
            ),
            BaseClassExpr::Attribute { value, attr, range } => {
                let (base, has_strict) = self.base_class_expr_infer(value, errors);
                (
                    self.attr_infer_for_type(&base, &attr.id, *range, errors, None),
                    has_strict,
                )
            }
            BaseClassExpr::Subscript {
                value,
                slice,
                range,
            } => {
                let (base_ty, has_strict_from_value) = self.base_class_expr_infer(value, errors);
                let (result_ty, has_strict_from_subscript) =
                    self.base_class_subscript_infer(base_ty, slice, *range, errors);
                (
                    result_ty,
                    has_strict_from_value || has_strict_from_subscript,
                )
            }
        }
    }

    fn is_type_alias_with_pydantic_strict_metadata(&self, ty: &Type) -> bool {
        matches!(ty, Type::TypeAlias(ta) if ta.annotated_metadata()
            .iter()
            .any(|metadata| self.is_pydantic_strict_metadata(metadata)))
    }

    /// Get the untyped form (in other words, the instance type, after applying
    /// any type arguments) for a base class.
    ///
    /// Also return whether the base class implies pydantic strict metadata through
    /// a type alias. This is used to handle inheriting from `RootModel[X]` in some cases.
    fn base_class_expr_untype(
        &self,
        base_expr: &BaseClassExpr,
        type_form_context: TypeFormContext,
        errors: &ErrorCollector,
    ) -> (Type, bool) {
        let range = base_expr.range();
        let (inferred_ty, has_strict_from_infer) = self.base_class_expr_infer(base_expr, errors);
        let has_pydantic_strict_metadata =
            self.is_type_alias_with_pydantic_strict_metadata(&inferred_ty) || has_strict_from_infer;
        let ty = self.untype(inferred_ty, range, errors);
        (
            self.validate_type_form(ty, range, type_form_context, errors),
            has_pydantic_strict_metadata,
        )
    }

    pub fn class_bases_of(
        &self,
        cls: &Class,
        bases: &[BaseClass],
        is_new_type: bool,
        errors: &ErrorCollector,
    ) -> ClassBases {
        // Make sure errors in base class expr are not reported during expr_untype -- they'll be checked in
        // another binding.
        let fake_error_collector = ErrorCollector::new(self.module().dupe(), ErrorStyle::Never);
        let mut has_pydantic_strict_metadata = false;

        let base_types_with_ranges = bases
            .iter()
            .filter_map(|x| match x {
                BaseClass::BaseClassExpr(x) => {
                    let (ty, base_has_strict) = self.base_class_expr_untype(
                        x,
                        TypeFormContext::BaseClassList,
                        &fake_error_collector,
                    );
                    if base_has_strict {
                        has_pydantic_strict_metadata = true;
                    }
                    Some((ty, x.range()))
                }
                BaseClass::NamedTuple(..) => Some((
                    self.stdlib.named_tuple_fallback().clone().to_type(),
                    x.range(),
                )),
                BaseClass::InvalidExpr(..) | BaseClass::TypedDict(..) | BaseClass::Generic(..) => {
                    None
                }
            })
            .collect::<Vec<_>>();

        let mut tuple_ancestor = base_types_with_ranges.iter().find_map(|(ty, _)| {
            if let Type::Tuple(tuple) = ty {
                Some(tuple.clone())
            } else {
                None
            }
        });

        let base_type_base_and_range = base_types_with_ranges
            .into_iter()
            .filter_map(|base_type_and_range| {
                // Return Ok() if the base class is valid, or Err() if it is not.
                match base_type_and_range {
                    (Type::ClassType(c), range) => {
                        let bases = self.get_base_types_for_class(c.class_object());
                        // Propagate has_strict from parent class
                        if bases.has_pydantic_strict_metadata {
                            has_pydantic_strict_metadata = true;
                        }
                        Some((c, bases, range))
                    }
                    (Type::Tuple(tuple), range) => {
                        let class_ty = self.erase_tuple_type(tuple);
                        let bases = self.get_base_types_for_class(class_ty.class_object());
                        Some((class_ty, bases, range))
                    }
                    (Type::TypedDict(typed_dict), range) => {
                        if is_new_type {
                            // Error will be reported in `class_metadata_of`
                            None
                        } else {
                            // HACK HACK HACK - TypedDict instances behave very differently from instances of other
                            // classes, so we don't represent them as ClassType in normal typechecking logic. However,
                            // class ancestors are represented as ClassType all over the code base, and changing this
                            // would be quite painful. So we convert TypedDict to ClassType in this one spot. Please do
                            // not do this anywhere else.
                            match typed_dict {
                                TypedDict::TypedDict(inner) => Some((
                                    ClassType::new(
                                        inner.class_object().dupe(),
                                        inner.targs().clone(),
                                    ),
                                    self.get_base_types_for_class(inner.class_object()),
                                    range,
                                )),
                                TypedDict::Anonymous(_) => {
                                    // Anonymous TypedDict cannot be used as a base class
                                    None
                                }
                            }
                        }
                    }
                    (_, _) => None,
                }
            })
            .collect::<Vec<_>>();

        let mut base_class_types = base_type_base_and_range
            .into_iter()
            .map(|(base_class_type, base_class_bases, range)| {
                if is_new_type
                    && base_class_type.targs().as_slice().iter().any(|ty| {
                        ty.any(|ty| {
                            matches!(
                                ty,
                                Type::TypeVar(_) | Type::TypeVarTuple(_) | Type::ParamSpec(_)
                            )
                        })
                    })
                {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::InvalidArgument),
                        "Second argument to NewType cannot be an unbound generic".to_owned(),
                    );
                }
                if let Some(base_tuple_ancestor) = base_class_bases.tuple_ancestor() {
                    if let Some(existing_tuple_ancestor) = &tuple_ancestor {
                        if existing_tuple_ancestor.is_any_tuple() {
                            tuple_ancestor = Some(base_tuple_ancestor.clone());
                        } else if !base_tuple_ancestor.is_any_tuple()
                            && base_tuple_ancestor != existing_tuple_ancestor
                        {
                            self.error(
                                errors,
                                range,
                                ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                                format!(
                                    "Cannot extend multiple incompatible tuples: `{}` and `{}`",
                                    self.for_display(Type::Tuple(existing_tuple_ancestor.clone())),
                                    self.for_display(Type::Tuple(base_tuple_ancestor.clone())),
                                ),
                            );
                        }
                    } else {
                        tuple_ancestor = Some(base_tuple_ancestor.clone());
                    }
                }
                base_class_type
            })
            .collect::<Vec<_>>();

        let metadata = self.get_metadata_for_class(cls);
        if metadata.is_typed_dict() && base_class_types.is_empty() {
            // This is a "fallback" class that contains attributes that are available on all TypedDict subclasses.
            // Note that this also makes those attributes available on *instances* of said subclasses; this is
            // desirable for methods but problematic for fields like `__total__` that should be available on the class
            // but not the instance. For now, we make all fields available on both classes and instances.
            base_class_types.push(self.stdlib.typed_dict_fallback().clone());
        }

        ClassBases {
            base_types: base_class_types.into_boxed_slice(),
            tuple_ancestor,
            has_pydantic_strict_metadata,
        }
    }
}
