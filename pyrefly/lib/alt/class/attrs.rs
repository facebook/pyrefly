/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use pyrefly_python::module_name::ModuleName;
use pyrefly_python::short_identifier::ShortIdentifier;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::Hashed;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::class::class_field::DataclassMember;
use crate::alt::class::class_metadata::TransformDataclass;
use crate::alt::types::class_metadata::ClassMetadata;
use crate::alt::types::class_metadata::DataclassMetadata;
use crate::binding::binding::Key;
use crate::binding::binding::KeyDecorator;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::types::callable::FunctionKind;
use crate::types::class::Class;
use crate::types::keywords::DataclassKeywords;
use crate::types::keywords::TypeMap;
use crate::types::literal::Lit;
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

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
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
}
