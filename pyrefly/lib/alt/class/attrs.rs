/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Solving-phase support for the `attrs`/`attr` library (the third-party precursor to and superset
//! of `@dataclass`). attrs classes are recognized as a [`DataclassKind::Attrs`] during binding; this
//! module implements the behaviors that diverge from plain dataclasses, including: init-parameter
//! renaming (stripping leading underscores), the `auto_attribs` defaults per decorator flavor,
//! `eq`/`order`/`cmp` validation, `on_setattr`/`setters.frozen` read-only detection,
//! `converters.optional`/`converters.pipe` handling, `@x.default` decorator return-type checks, `@x.converter`
//! decorator input types, and the `assoc`/`fields`/`fields_dict` runtime helpers.

use std::sync::Arc;

use dupe::Dupe;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::typed_dict::AnonymousTypedDictInner;
use pyrefly_types::typed_dict::TypedDict;
use pyrefly_types::typed_dict::TypedDictField;
use ruff_python_ast::Arguments;
use ruff_python_ast::Expr;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::Hashed;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::callable::CallArg;
use crate::alt::callable::CallKeyword;
use crate::alt::class::class_field::DataclassMember;
use crate::alt::class::class_metadata::TransformDataclass;
use crate::alt::types::class_metadata::ClassMetadata;
use crate::alt::types::class_metadata::DataclassKind;
use crate::alt::types::class_metadata::DataclassMetadata;
use crate::alt::unwrap::HintRef;
use crate::binding::binding::Key;
use crate::binding::binding::KeyDecorator;
use crate::binding::binding::KeyExport;
use crate::binding::binding::KeyUndecoratedFunction;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::types::callable::FunctionKind;
use crate::types::callable::Param;
use crate::types::class::Class;
use crate::types::class::ClassType;
use crate::types::keywords::DataclassFieldKeywords;
use crate::types::keywords::DataclassKeywords;
use crate::types::keywords::TypeMap;
use crate::types::literal::Lit;
use crate::types::tuple::Tuple;
use crate::types::types::CalleeKind;
use crate::types::types::Type;

/// Suppresses the assignment check against a field's declared type. Required-ness instead uses
/// binding-phase identity; this type-based path covers generic assignments with no binding context.
pub(crate) fn is_attrs_nothing(ty: &Type) -> bool {
    let class = match ty {
        Type::Literal(lit) if let Lit::Enum(e) = &lit.value => &e.class,
        Type::ClassType(c) => c,
        _ => return false,
    };
    class.has_qname("attr", "_Nothing") || class.has_qname("attrs", "_Nothing")
}

/// Recognizes `attr.setters.frozen` / `attrs.setters.frozen` (an `on_setattr` hook that makes an
/// attribute read-only) by the function's definition identity, which is stable across re-exports.
pub(crate) fn is_attrs_setters_frozen(ty: &Type) -> bool {
    if let Type::Function(f) = ty
        && let FunctionKind::Def(id) = &f.metadata.kind
    {
        id.name.as_str() == "frozen"
            && matches!(id.module.name().as_str(), "attr.setters" | "attrs.setters")
    } else {
        false
    }
}

/// Recognizes `attr.setters.pipe` / `attrs.setters.pipe`, the combinator that runs several
/// `on_setattr` hooks in sequence. A `pipe(...)` containing `setters.frozen` still freezes the
/// attribute, so we look through it at its arguments.
pub(crate) fn is_attrs_setters_pipe(ty: &Type) -> bool {
    if let Type::Function(f) = ty
        && let FunctionKind::Def(id) = &f.metadata.kind
    {
        id.name.as_str() == "pipe"
            && matches!(id.module.name().as_str(), "attr.setters" | "attrs.setters")
    } else {
        false
    }
}

/// Whether a resolved `on_setattr` type freezes the attribute: the `frozen` hook, or any element of
/// a tuple/list/union of hooks.
fn attrs_type_is_frozen_setter(ty: &Type) -> bool {
    if is_attrs_setters_frozen(ty) {
        return true;
    }
    match ty {
        Type::Tuple(Tuple::Concrete(elts)) => elts.iter().any(attrs_type_is_frozen_setter),
        Type::Tuple(Tuple::Unbounded(elt)) => attrs_type_is_frozen_setter(elt),
        Type::Tuple(Tuple::Unpacked(unpacked)) => {
            let (prefix, middle, suffix) = &**unpacked;
            prefix.iter().any(attrs_type_is_frozen_setter)
                || attrs_type_is_frozen_setter(middle)
                || suffix.iter().any(attrs_type_is_frozen_setter)
        }
        Type::ClassType(ct) if ct.is_builtin("list") => ct
            .targs()
            .as_slice()
            .first()
            .is_some_and(attrs_type_is_frozen_setter),
        Type::Union(u) => u.members.iter().any(attrs_type_is_frozen_setter),
        _ => false,
    }
}

/// The `__init__`/`__replace__` parameter name attrs derives for a field. attrs strips leading
/// underscores from private fields (`_x` -> `x`).
pub(crate) enum AttrsInitName {
    /// Use this stripped name instead of the field name.
    Renamed(Name),
    /// The stripped name collides with another field; keep the field name and report it.
    Collision(Name),
    /// Not an attrs class, or no leading underscore: use the field name unchanged.
    Unchanged,
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub(crate) fn attrs_init_param_name(
        &self,
        cls: &Class,
        name: &Name,
        dataclass: &DataclassMetadata,
    ) -> AttrsInitName {
        if !matches!(dataclass.kind, DataclassKind::Attrs { .. }) {
            return AttrsInitName::Unchanged;
        }
        let mangled = if name.as_str().starts_with("__") && !name.as_str().ends_with("__") {
            let defining_class = self
                .get_class_member_with_defining_class(cls, name)
                .map_or_else(|| cls.dupe(), |member| member.defining_class);
            let class_name = defining_class.name().as_str().trim_start_matches('_');
            (!class_name.is_empty()).then(|| format!("_{class_name}{name}"))
        } else {
            None
        };
        let effective = mangled.as_deref().unwrap_or_else(|| name.as_str());
        let stripped = effective.trim_start_matches('_');
        if stripped.is_empty() || stripped.len() == name.as_str().len() {
            return AttrsInitName::Unchanged;
        }
        let stripped = Name::new(stripped);
        if dataclass.fields.contains(&stripped) {
            AttrsInitName::Collision(stripped)
        } else {
            AttrsInitName::Renamed(stripped)
        }
    }

    pub(crate) fn is_attrs_class(
        &self,
        dataclass_from_dataclass_transform: &Option<TransformDataclass>,
        bases_with_metadata: &[(Class, Arc<ClassMetadata>)],
    ) -> bool {
        let has_attrs_field_specifiers = dataclass_from_dataclass_transform
            .as_ref()
            .is_some_and(|t| Self::field_specifiers_reference_attrs(&t.field_specifiers));
        let has_attrs_base = bases_with_metadata
            .iter()
            .any(|(_, metadata)| metadata.is_attrs_class());
        has_attrs_field_specifiers || has_attrs_base
    }

    pub(crate) fn field_specifiers_reference_attrs(field_specifiers: &[CalleeKind]) -> bool {
        field_specifiers.iter().any(|callee| {
            matches!(callee,
                CalleeKind::Function(FunctionKind::Def(id))
                    if id.module.name() == ModuleName::attr()
                        || id.module.name() == ModuleName::attrs()
            )
        })
    }

    /// The default `auto_attribs` for an attrs decorator that doesn't set it, based on the decorator's name:
    /// - `attr.s`/`attrs`/`attributes` -> `False`
    /// - `@attr.dataclass` -> `True`.
    /// - `define`/`frozen`/`mutable` -> `None`
    ///   The behavior for None is: try `True` and falls back
    ///   to `False` when a field is assigned `attr.ib()`/`field()` with no annotation.
    pub(crate) fn attrs_default_auto_attribs(
        &self,
        cls: &Class,
        decorator_range: TextRange,
        order_default: bool,
    ) -> bool {
        let Some(idx) = self
            .bindings()
            .key_to_idx_hashed_opt(Hashed::new(&KeyDecorator(decorator_range)))
        else {
            // Can't recover the decorator name; fall back to the transform default.
            return !order_default;
        };
        let binding = self.bindings().get::<KeyDecorator>(idx);
        match binding.trailing_name.as_ref().map(Name::as_str) {
            Some("s" | "attrs" | "attributes") => false,
            Some("dataclass") => true,
            Some("define" | "mutable" | "frozen") => !self.get_class_fields(cls).is_some_and(|f| {
                f.class_body_fields()
                    .any(|name| f.is_attrs_field_specifier(name) && !f.is_field_annotated(name))
            }),
            // Unknown decorator: attrs sets `order_default` only on its classic
            // decorators, so it stands in for "classic" here.
            _ => !order_default,
        }
    }

    pub(crate) fn check_attrs_default_decorator_return_types(
        &self,
        cls: &Class,
        dataclass: &DataclassMetadata,
        errors: &ErrorCollector,
    ) {
        let Some(fields) = self.get_class_fields(cls) else {
            return;
        };
        for name in dataclass.fields.iter() {
            let Some(method_range) = fields.attrs_default_decorator_method_range(name) else {
                continue;
            };
            let DataclassMember::Field(field, field_flags) = self.get_dataclass_member(cls, name)
            else {
                continue;
            };
            if field_flags.converter_param.is_some() {
                continue;
            }
            // The decorated method's member type is `Any`, so read its return type directly.
            let return_ty = self
                .get(&Key::ReturnType(ShortIdentifier::from_text_range(
                    method_range,
                )))
                .arc_clone_ty();
            let field_ty = field.value.ty();
            if !self.is_subset_eq(&return_ty, &field_ty) {
                let range = fields
                    .field_decl_range(name)
                    .expect("a field with a default-decorator spec is tracked in the field map");
                self.error(
                    errors,
                    range,
                    ErrorKind::BadClassDefinition,
                    format!(
                        "Return type `{return_ty}` of the `@{name}.default` method is not assignable to field `{name}` of type `{field_ty}`"
                    ),
                );
            }
        }
    }

    /// The `__init__` input type from a `@<field>.converter` method. attrs invokes it as
    /// `converter(self, field, value)`, so the `value` (third positional) parameter's type is the
    /// converter's input; falls back to `Any` when that parameter is absent or unannotated.
    pub(crate) fn attrs_converter_decorator_param(&self, method_range: TextRange) -> Type {
        let func = self.get(&KeyUndecoratedFunction(ShortIdentifier::from_text_range(
            method_range,
        )));
        func.params
            .iter()
            .filter_map(|p| match p {
                Param::PosOnly(_, ty, _) | Param::Pos(_, ty, _) => Some(ty.clone()),
                _ => None,
            })
            .nth(2)
            .unwrap_or_else(|| self.heap.mk_any_implicit())
    }

    /// attrs rejects two `eq`/`order`/`cmp` combinations at runtime (`ValueError`), on both the
    /// class decorator and the field specifier: `cmp` mixed with `eq`/`order`, and `order=True`
    /// with `eq=False` (ordering requires equality). A non-bool `eq` (e.g. a key callable) is
    /// truthy, so only a literal `False` triggers the second rule.
    pub(crate) fn validate_attrs_eq_order_cmp(
        &self,
        kws: &TypeMap,
        range: TextRange,
        errors: &ErrorCollector,
    ) {
        if kws.is_set(&DataclassKeywords::CMP)
            && (kws.is_set(&DataclassKeywords::EQ) || kws.is_set(&DataclassKeywords::ORDER))
        {
            self.error(
                errors,
                range,
                ErrorKind::BadClassDefinition,
                "Cannot mix `cmp` with `eq` or `order`".to_owned(),
            );
        }
        if kws.get_bool(&DataclassKeywords::EQ) == Some(false)
            && kws.get_bool(&DataclassKeywords::ORDER) == Some(true)
        {
            self.error(
                errors,
                range,
                ErrorKind::BadClassDefinition,
                "`order` cannot be True when `eq` is False".to_owned(),
            );
        }
    }

    /// Validate an `attr.assoc`/`attrs.assoc` call: unlike `evolve`, it keys on actual attribute
    /// names (no init-alias renaming) and accepts `init=False` fields, so we build a fresh signature.
    pub(crate) fn call_attrs_assoc(
        &self,
        cls: &ClassType,
        rest_args: &[CallArg],
        kws: &[CallKeyword],
        callee_range: TextRange,
        arg_range: TextRange,
        hint: Option<HintRef>,
        errors: &ErrorCollector,
    ) -> Type {
        let class_obj = cls.class_object();
        let metadata = self.get_metadata_for_class(class_obj);
        let dataclass = metadata
            .dataclass_metadata()
            .expect("assoc target is a validated attrs class");
        let kw_only = self.compute_kw_only_fields_by_class(class_obj);
        let sub = cls.targs().substitution();
        let strict_default = dataclass.kws.strict;
        let params = self
            .iter_fields(class_obj, dataclass, false, &kw_only)
            .into_iter()
            .map(|(name, field, flags)| {
                self.as_param(
                    &field,
                    &name,
                    true,
                    true,
                    flags.strict.unwrap_or(strict_default),
                    flags.converter_param.clone(),
                    &|t| sub.substitute_into(t),
                    errors,
                )
            })
            .collect();
        let assoc_ty = self.synthesized_method(
            class_obj,
            Name::new_static("assoc"),
            params,
            Type::ClassType(cls.clone()),
        );
        self.freeform_call_infer(
            assoc_ty,
            rest_args,
            kws,
            callee_range,
            arg_range,
            hint,
            errors,
        )
    }

    /// `attr.fields(C)` / `attr.fields_dict(C)`: return a field-aware type and reject a non-attrs
    /// class argument (matching attrs' runtime `NotAnAttrsClassError`). `fields_dict` returns an
    /// anonymous `TypedDict {name: Attribute[t]}`; `fields` keeps the stub's declared return for now.
    pub(crate) fn call_attrs_fields(
        &self,
        func_name: &Name,
        fields_ty: &Type,
        args: &[CallArg],
        kws: &[CallKeyword],
        callee_range: TextRange,
        arg_range: TextRange,
        hint: Option<HintRef>,
        errors: &ErrorCollector,
    ) -> Type {
        if let [CallArg::Arg(obj_arg)] = args
            && kws.is_empty()
        {
            // Keep the `ClassType` when present so a generic `type[C[int]]` substitutes its targs.
            let (cls, class_type) = match obj_arg.infer(self, errors) {
                Type::ClassDef(cls) => (Some(cls), None),
                Type::Type(inner) => match *inner {
                    Type::ClassType(c) => (Some(c.class_object().clone()), Some(c)),
                    _ => (None, None),
                },
                _ => (None, None),
            };
            if let Some(cls) = cls {
                let metadata = self.get_metadata_for_class(&cls);
                let dataclass = metadata.dataclass_metadata();
                if let Some(dataclass) = dataclass
                    && matches!(dataclass.kind, DataclassKind::Attrs { .. })
                {
                    if func_name.as_str() == "fields_dict"
                        && let Some(td) =
                            self.attrs_fields_dict_type(&cls, dataclass, class_type.as_ref())
                    {
                        return td;
                    }
                } else if !metadata.is_protocol() {
                    // `type[AttrsInstance]` (a Protocol) is the canonical "any attrs class"
                    // annotation, so accept Protocols; non-class arguments are left to the stub.
                    self.error(
                        errors,
                        arg_range,
                        ErrorKind::BadArgumentType,
                        format!("Argument to `{func_name}()` is not an attrs class"),
                    );
                    return self.heap.mk_any_explicit();
                }
            }
        }
        self.freeform_call_infer(
            fields_ty.clone(),
            args,
            kws,
            callee_range,
            arg_range,
            hint,
            errors,
        )
    }

    /// `attr.fields_dict(C)` maps each field name to its `Attribute[T]`, modeled as an anonymous
    /// `TypedDict`. Over 100 fields, returns `None` to fall back to the stub's `dict`.
    fn attrs_fields_dict_type(
        &self,
        cls: &Class,
        dataclass: &DataclassMetadata,
        class_type: Option<&ClassType>,
    ) -> Option<Type> {
        const MAX_FIELDS: usize = 100;
        let attribute_class = self.attrs_attribute_class()?;
        let kw_only = self.compute_kw_only_fields_by_class(cls);
        let raw_fields = self.iter_fields(cls, dataclass, false, &kw_only);
        if raw_fields.len() > MAX_FIELDS {
            return None;
        }
        let sub = class_type.map(|c| c.targs().substitution());
        let swallow = self.error_swallower();
        let fields = raw_fields
            .into_iter()
            .map(|(name, field, _)| {
                let ty = match &sub {
                    Some(sub) => sub.substitute_into(field.ty()),
                    None => field.ty(),
                };
                let attr_ty =
                    self.specialize(&attribute_class, vec![ty], TextRange::default(), &swallow);
                (
                    name,
                    TypedDictField {
                        ty: attr_ty,
                        required: true,
                        read_only_reason: None,
                    },
                )
            })
            .collect();
        Some(
            self.heap
                .mk_typed_dict(TypedDict::Anonymous(Box::new(AnonymousTypedDictInner {
                    fields,
                }))),
        )
    }

    /// Resolve the `attr.Attribute` / `attrs.Attribute` class from the stubs, if available.
    fn attrs_attribute_class(&self) -> Option<Class> {
        let name = Name::new_static("Attribute");
        for module in [ModuleName::attr(), ModuleName::attrs()] {
            if self.exports.export_exists(module, &name)
                && let Type::ClassDef(cls) = self
                    .get_from_export(module, None, &KeyExport(name.clone()))
                    .as_ref()
            {
                return Some(cls.clone());
            }
        }
        None
    }

    /// Whether an attrs `on_setattr` argument makes the attribute read-only. Possible forms:
    /// - a single hook like `setters.frozen`
    /// - a list or tuple of hooks
    /// - multiple hooks passed to `setters.pipe`
    pub(crate) fn on_setattr_is_frozen(&self, expr: &Expr) -> bool {
        // `pipe(...)`'s return type doesn't record which hooks it composed, so it must be inspected
        // by expression; every other form is decided from the inferred type.
        if let Expr::Call(call) = expr
            && is_attrs_setters_pipe(&self.expr_infer(&call.func, &self.error_swallower()))
        {
            return call
                .arguments
                .args
                .iter()
                .any(|e| self.on_setattr_is_frozen(e));
        }
        attrs_type_is_frozen_setter(&self.expr_infer(expr, &self.error_swallower()))
    }

    /// The `__init__` input for an `attr.converters`/`attrs.converters` combinator `converter=`:
    /// `optional(c)` adds `None` to `c`'s input; `pipe(c1, ...)` uses `c1`'s (it runs first);
    /// `default_if_none(...)` passes non-`None` values through, so it is `annotated_field_ty | None`.
    /// `None` for any other `converter=`.
    ///
    /// `annotated_field_ty` is the field's declared type annotation, if the field has one.
    pub(crate) fn attrs_converters_combinator_param(
        &self,
        args: &Arguments,
        annotated_field_ty: Option<&Type>,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let kw = args.keywords.iter().find(|kw| {
            kw.arg
                .as_ref()
                .is_some_and(|n| n.id == DataclassFieldKeywords::CONVERTER)
        })?;
        let Expr::Call(call) = &kw.value else {
            return None;
        };
        // These converters are overloaded, so match on the resolved definition identity via
        // `callee_kind` (which looks through overloads) rather than a bare `Type::Function`.
        let Some(CalleeKind::Function(FunctionKind::Def(id))) =
            self.expr_infer(&call.func, errors).callee_kind()
        else {
            return None;
        };
        if !matches!(
            id.module.name().as_str(),
            "attr.converters" | "attrs.converters"
        ) {
            return None;
        }
        let first_input =
            |hint| {
                Some(self.get_converter_param(
                    &self.expr_infer(call.arguments.args.first()?, errors),
                    hint,
                ))
            };
        match id.name.as_str() {
            // `optional(c)` feeds `c`'s output straight to the field, so its type parameters solve
            // against the annotation.
            "optional" => Some(self.union(first_input(annotated_field_ty)?, self.heap.mk_none())),
            // In `pipe(c1, ...)`, `c1`'s output feeds the next converter, not the field, so the
            // annotation must not solve `c1`'s type parameters (only the last output must match it).
            "pipe" => first_input(None),
            "default_if_none" => Some(self.union(annotated_field_ty?.clone(), self.heap.mk_none())),
            _ => None,
        }
    }
}
