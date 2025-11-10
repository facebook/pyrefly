/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use pyrefly_python::module_name::ModuleName;
use pyrefly_types::callable::Callable;
use pyrefly_types::callable::FuncMetadata;
use pyrefly_types::callable::Function;
use pyrefly_types::callable::ParamList;
use pyrefly_types::class::Class;
use pyrefly_types::literal::Lit;
use pyrefly_types::tuple::Tuple;
use pyrefly_types::types::Type;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::class::class_field::WithDefiningClass;
use crate::alt::class::enums::VALUE_PROP;
use crate::alt::types::class_metadata::ClassSynthesizedField;
use crate::alt::types::class_metadata::ClassSynthesizedFields;
use crate::binding::binding::KeyExport;
use crate::types::simplify::unions;

/// Django stubs use this attribute to specify the Python type that a field should infer to
const DJANGO_PRIVATE_GET_TYPE: Name = Name::new_static("_pyi_private_get_type");
const CHOICES: Name = Name::new_static("choices");
const LABEL: Name = Name::new_static("label");
const LABELS: Name = Name::new_static("labels");
const VALUES: Name = Name::new_static("values");
const ID: Name = Name::new_static("id");
const PK: Name = Name::new_static("pk");
const AUTO_FIELD: Name = Name::new_static("AutoField");
const FOREIGN_KEY: Name = Name::new_static("ForeignKey");
const NULL: Name = Name::new_static("null");

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub fn get_django_field_type(
        &self,
        ty: &Type,
        class: &Class,
        field_name: Option<&Name>,
        initial_value_expr: Option<&Expr>,
    ) -> Option<Type> {
        match ty {
            Type::ClassType(cls)
                if cls.has_qname(ModuleName::django_utils_functional().as_str(), "_Getter") =>
            {
                cls.targs().as_slice().first().cloned()
            }
            Type::ClassType(cls) => self.get_django_field_type_from_class(
                cls.class_object(),
                class,
                field_name,
                initial_value_expr,
            ),
            Type::ClassDef(cls) => {
                self.get_django_field_type_from_class(cls, class, field_name, initial_value_expr)
            }
            Type::Union(union) => {
                let transformed: Vec<_> = union
                    .iter()
                    .map(|variant| {
                        self.get_django_field_type(variant, class, field_name, initial_value_expr)
                            .unwrap_or_else(|| variant.clone())
                    })
                    .collect();

                if transformed != union.to_vec() {
                    Some(unions(transformed))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn get_django_field_type_from_class(
        &self,
        field: &Class,
        class: &Class,
        field_name: Option<&Name>,
        initial_value_expr: Option<&Expr>,
    ) -> Option<Type> {
        if !(self.get_metadata_for_class(class).is_django_model()
            && self.inherits_from_django_field(field))
        {
            return None;
        }

        // Check if this is a ForeignKey field
        if self.is_foreign_key_field(field)
            && field_name.is_some()
            && let Some(e) = initial_value_expr
            && let Some(call_expr) = e.as_call_expr()
            && let Some(to_expr) = call_expr.arguments.args.first()
        {
            // Resolve the expression to a type and convert to instance type
            let related_model_type = self.resolve_foreign_key_target(to_expr, class)?;

            // If nullable, union with None
            if self.is_django_field_nullable(call_expr) {
                return Some(self.union(related_model_type, Type::None));
            } else {
                return Some(related_model_type);
            }
        }

        // Default: use _pyi_private_get_type from the field class
        self.get_class_member(field, &DJANGO_PRIVATE_GET_TYPE)
            .map(|member| member.value.ty())
    }

    /// Check if a class inherits from Django's Field class
    fn inherits_from_django_field(&self, cls: &Class) -> bool {
        self.get_mro_for_class(cls)
            .ancestors(self.stdlib)
            .any(|ancestor| {
                ancestor.has_qname(ModuleName::django_models_fields().as_str(), "Field")
            })
    }

    fn resolve_foreign_key_target(&self, to_expr: &Expr, class: &Class) -> Option<Type> {
        // Extract the model name from the expression
        let model_name = match to_expr {
            // Direct name reference. Ex: ForeignKey(Reporter, ...)
            Expr::Name(name_expr) => name_expr.id.clone(),
            // TODO: handle self references and forward references
            _ => return None,
        };

        // Look up the model in the current module and convert to instance type
        let export_key = KeyExport(model_name);
        let related_model_type =
            self.get_from_export(class.module_name(), Some(class.module_path()), &export_key);
        Some(self.class_def_to_instance_type(&related_model_type))
    }

    fn class_def_to_instance_type(&self, ty: &Type) -> Type {
        if let Type::ClassDef(class) = ty {
            self.instantiate(class)
        } else {
            ty.clone()
        }
    }

    pub fn is_foreign_key_field(&self, field: &Class) -> bool {
        field.has_toplevel_qname(
            ModuleName::django_models_fields_related().as_str(),
            FOREIGN_KEY.as_str(),
        )
    }

    pub fn get_django_enum_synthesized_fields(
        &self,
        cls: &Class,
    ) -> Option<ClassSynthesizedFields> {
        let metadata = self.get_metadata_for_class(cls);
        let enum_metadata = metadata.enum_metadata()?;
        if !enum_metadata.is_django {
            return None;
        }

        let enum_members = self.get_enum_members(cls);

        let mut label_types: Vec<Type> = enum_members
            .iter()
            .filter_map(|lit| {
                if let Lit::Enum(lit_enum) = lit
                    && let Type::Tuple(Tuple::Concrete(elements)) = &lit_enum.ty
                    && elements.len() >= 2
                {
                    Some(
                        elements[elements.len() - 1]
                            .clone()
                            .promote_literals(self.stdlib),
                    )
                } else {
                    None
                }
            })
            .collect();

        if label_types.is_empty() || label_types.len() < enum_members.len() {
            // Members without a custom label type have default label type str.
            label_types.push(self.stdlib.str().clone().to_type());
        }

        // Also include the type of __empty__ field if it exists, since it contributes to label types
        let empty_name = Name::new_static("__empty__");
        let has_empty = if let Some(WithDefiningClass { value, .. }) =
            self.get_class_member(cls, &empty_name)
        {
            label_types.push(value.ty());
            true
        } else {
            false
        };

        let label_type = self.unions(label_types);

        let base_value_attr = self.get_enum_or_instance_attribute(
            &self.as_class_type_unchecked(cls),
            &metadata,
            &VALUE_PROP,
        );
        let base_value_type = base_value_attr
            .and_then(|attr| {
                self.resolve_get_class_attr(
                    attr,
                    TextRange::default(),
                    &self.error_swallower(),
                    None,
                )
                .ok()
            })
            .unwrap_or_else(Type::any_implicit);

        // if value is optional, make the type optional
        let values_type = if has_empty {
            self.union(base_value_type.clone(), Type::None)
        } else {
            base_value_type
        };

        let mut fields = SmallMap::new();

        let field_specs = [
            (LABELS, self.stdlib.list(label_type.clone()).to_type()),
            (LABEL, self.property(cls, LABEL, label_type.clone())),
            (VALUES, self.stdlib.list(values_type.clone()).to_type()),
            (
                CHOICES,
                self.stdlib
                    .list(Type::Tuple(Tuple::Concrete(vec![values_type, label_type])))
                    .to_type(),
            ),
        ];

        for (name, ty) in field_specs {
            fields.insert(name, ClassSynthesizedField::new(ty));
        }

        Some(ClassSynthesizedFields::new(fields))
    }

    fn property(&self, cls: &Class, name: Name, ty: Type) -> Type {
        let signature = Callable::list(ParamList::new(vec![self.class_self_param(cls, false)]), ty);
        let mut metadata = FuncMetadata::def(self.module().dupe(), cls.dupe(), name);
        metadata.flags.is_property_getter = true;
        Type::Function(Box::new(Function {
            signature,
            metadata,
        }))
    }

    /// Get the primary key field type for a Django model.
    /// Returns a tuple of (pk_type, has_custom_pk) where has_custom_pk indicates
    /// whether the model has a custom primary key field defined.
    fn get_pk_field_type(&self, model: &Class) -> Option<(Type, bool)> {
        let metadata = self.get_metadata_for_class(model);

        if let Some(pk_field_name) = metadata
            .django_model_metadata()
            .and_then(|dm| dm.custom_primary_key_field.as_ref())
        {
            let instance_type = self.as_class_type_unchecked(model).to_type();
            let pk_type = self.attr_infer_for_type(
                &instance_type,
                pk_field_name,
                TextRange::default(),
                &self.error_swallower(),
                None,
            );
            Some((pk_type, true))
        } else {
            // No custom pk, use default AutoField type
            let auto_field_export = KeyExport(AUTO_FIELD);
            let auto_field_type =
                self.get_from_export(ModuleName::django_models_fields(), None, &auto_field_export);
            self.get_django_field_type(&auto_field_type, model, None, None)
                .map(|ty| (ty, false))
        }
    }

    fn is_django_field_nullable(&self, call_expr: &ExprCall) -> bool {
        call_expr.arguments.keywords.iter().any(|keyword| {
            keyword
                .arg
                .as_ref()
                .is_some_and(|name| name.as_str() == NULL.as_str())
                && matches!(
                    &keyword.value,
                    Expr::BooleanLiteral(bool_lit) if bool_lit.value
                )
        })
    }

    /// Returns the primary key type of the related model.
    fn get_foreign_key_id_type(&self, cls: &Class, field_name: &Name) -> Option<Type> {
        // Get the class field
        let class_field = self.get_field_from_current_class_only(cls, field_name)?;

        // Check if this is a ForeignKey field using the cached metadata
        if !class_field.is_foreign_key() {
            return None;
        }

        // Get the related model type from the field
        let ty = class_field.ty();
        let (related_cls, is_foreign_key_nullable) = match ty {
            Type::Union(union) => {
                // Nullable foreign key: extract the class type from the union
                let cls = union.iter().find_map(|variant| match variant {
                    Type::ClassType(cls) => Some(cls.clone()),
                    _ => None,
                })?;
                (cls, true)
            }
            Type::ClassType(cls) => (cls, false),
            _ => return None,
        };

        // Get the pk type from the related model and make it nullable if needed
        let (pk_type, _) = self.get_pk_field_type(related_cls.class_object())?;
        if is_foreign_key_nullable {
            Some(self.union(pk_type, Type::None))
        } else {
            Some(pk_type)
        }
    }

    pub fn get_django_model_synthesized_fields(
        &self,
        cls: &Class,
    ) -> Option<ClassSynthesizedFields> {
        let metadata = self.get_metadata_for_class(cls);
        if !metadata.is_django_model() {
            return None;
        }

        let mut fields = SmallMap::new();

        if let Some((pk_type, has_custom_pk)) = self.get_pk_field_type(cls) {
            if !has_custom_pk {
                // No custom pk, so synthesize an id field
                fields.insert(ID, ClassSynthesizedField::new(pk_type.clone()));
            }
            fields.insert(PK, ClassSynthesizedField::new(pk_type));
        }

        // Synthesize `<field_name>_id` fields for ForeignKey fields
        for field_name in cls.fields() {
            if let Some(fk_id_type) = self.get_foreign_key_id_type(cls, field_name) {
                let id_field_name = Name::new(format!("{}_id", field_name));
                fields.insert(id_field_name, ClassSynthesizedField::new(fk_id_type));
            }
        }

        Some(ClassSynthesizedFields::new(fields))
    }
}
