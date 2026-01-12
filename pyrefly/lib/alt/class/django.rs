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
use pyrefly_types::callable::PropertyMetadata;
use pyrefly_types::callable::PropertyRole;
use pyrefly_types::class::Class;
use pyrefly_types::literal::Lit;
use pyrefly_types::tuple::Tuple;
use pyrefly_types::types::Type;
use pyrefly_types::types::Union;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprStringLiteral;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::class::class_field::ClassField;
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
const MANY_TO_MANY_FIELD: Name = Name::new_static("ManyToManyField");
const MODEL: Name = Name::new_static("Model");
const MANYRELATEDMANAGER: Name = Name::new_static("ManyRelatedManager");

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
            Type::Union(box Union { members: union, .. }) => {
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

        let base_type = if field_name.is_some()
            && let Some(e) = initial_value_expr
            && let Some(call_expr) = e.as_call_expr()
            && let Some(to_expr) = call_expr.arguments.args.first()
            && let Some(model_type) = self.resolve_target(to_expr, class)
        {
            if self.is_foreign_key_field(field) {
                Some(model_type)
            } else if self.is_many_to_many_field(field) {
                return self.get_manager_type(model_type);
            } else {
                None
            }
        } else {
            None
        };

        let base_type = base_type.or_else(|| {
            self.get_class_member(field, &DJANGO_PRIVATE_GET_TYPE)
                .map(|field| field.ty())
        })?;

        if let Some(e) = initial_value_expr
            && let Some(call_expr) = e.as_call_expr()
            && self.is_django_field_nullable(call_expr)
        {
            Some(self.union(base_type, Type::None))
        } else {
            Some(base_type)
        }
    }

    /// Check if a class inherits from Django's Field class
    fn inherits_from_django_field(&self, cls: &Class) -> bool {
        self.get_mro_for_class(cls)
            .ancestors(self.stdlib)
            .any(|ancestor| {
                ancestor.has_qname(ModuleName::django_models_fields().as_str(), "Field")
            })
    }

    // Get ManyRelatedManager class from django stubs
    fn get_manager_type(&self, target_model_type: Type) -> Option<Type> {
        let django_related_module = ModuleName::django_models_fields_related_descriptors();
        let django_related_module_exports = self.exports.get(django_related_module).finding()?;
        let manager_class_type = if django_related_module_exports
            .exports(self.exports)
            .contains_key(&MANYRELATEDMANAGER)
        {
            self.get_from_export(django_related_module, None, &KeyExport(MANYRELATEDMANAGER))
        } else {
            return None;
        };

        // Extract the Class from ClassDef
        let manager_class = match manager_class_type.as_ref() {
            Type::ClassDef(cls) => cls,
            _ => return None,
        };

        // Get Model class for the through parameter
        let model_class =
            self.get_from_export(ModuleName::django_models(), None, &KeyExport(MODEL));

        let model_instance_type = self.class_def_to_instance_type(&model_class);

        // Create type arguments vector: [TargetModel, Model]
        let targs_vec = vec![target_model_type, model_instance_type];

        // Use specialize to create ManyRelatedManager for the specific classes we defined
        let manager_type = self.specialize(
            manager_class,
            targs_vec,
            TextRange::default(),
            &self.error_swallower(),
        );

        Some(manager_type)
    }

    fn resolve_target(&self, to_expr: &Expr, class: &Class) -> Option<Type> {
        match to_expr {
            // Use expr_infer to resolve the name in the current scope
            Expr::Name(_) => {
                let model_type = self.expr_infer(to_expr, &self.error_swallower());
                Some(self.class_def_to_instance_type(&model_type))
            }
            Expr::StringLiteral(ExprStringLiteral { value, .. }) => {
                if value.to_str() == "self" {
                    Some(self.instantiate(class))
                } else {
                    // Handle forward reference - look up the model by name in the current module
                    // This requires that the model class is imported or defined in the current module
                    let class_name = Name::new(value.to_str());
                    let module_name = class.module_name();

                    if self
                        .exports
                        .get(module_name)
                        .finding()?
                        .exports(self.exports)
                        .contains_key(&class_name)
                    {
                        let model_type =
                            self.get_from_export(module_name, None, &KeyExport(class_name));
                        Some(self.class_def_to_instance_type(&model_type))
                    } else {
                        None
                    }
                }
            }
            // we may have to extend this function to handle different kinds of fields in the future
            _ => None,
        }
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

    pub fn is_many_to_many_field(&self, field: &Class) -> bool {
        field.has_toplevel_qname(
            ModuleName::django_models_fields_related().as_str(),
            MANY_TO_MANY_FIELD.as_str(),
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
        let has_empty = if let Some(field) = self.get_class_member(cls, &empty_name) {
            label_types.push(field.ty());
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
                    &VALUE_PROP,
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
                    .list(Type::concrete_tuple(vec![values_type, label_type]))
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
        metadata.flags.property_metadata = Some(PropertyMetadata {
            role: PropertyRole::Getter,
            getter: Type::any_error(),
            setter: None,
            has_deleter: false,
        });
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

    /// Check if a Django field has a `choices` argument.
    pub fn has_django_field_choices(&self, call_expr: &ExprCall) -> bool {
        call_expr.arguments.keywords.iter().any(|keyword| {
            keyword
                .arg
                .as_ref()
                .is_some_and(|name| name.as_str() == CHOICES.as_str())
        })
    }

    /// Create a get_FOO_display method signature for a field with choices.
    /// The method takes self and returns str.
    fn get_display_method(&self, cls: &Class, method_name: &Name) -> ClassSynthesizedField {
        let params = vec![self.class_self_param(cls, false)];
        let ret = self.stdlib.str().clone().to_type();
        ClassSynthesizedField::new(Type::Function(Box::new(Function {
            signature: Callable::list(ParamList::new(params), ret),
            metadata: FuncMetadata::def(self.module().dupe(), cls.dupe(), method_name.clone()),
        })))
    }

    /// Returns the primary key type of the related model.
    fn get_foreign_key_id_type(&self, class_field: &ClassField) -> Option<Type> {
        // Check if this is a ForeignKey field using the cached metadata
        if !class_field.is_foreign_key() {
            return None;
        }

        // Get the related model type from the field
        let ty = class_field.ty();
        let (related_cls, is_foreign_key_nullable) = match ty {
            Type::Union(box Union { members: union, .. }) => {
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
        // and `get_<field_name>_display()` methods for fields with choices
        for field_name in cls.fields() {
            let Some(class_field) = self.get_field_from_current_class_only(cls, field_name) else {
                continue;
            };
            if let Some(fk_id_type) = self.get_foreign_key_id_type(&class_field) {
                let id_field_name = Name::new(format!("{}_id", field_name));
                fields.insert(id_field_name, ClassSynthesizedField::new(fk_id_type));
            }
            if class_field.has_choices() {
                let method_name = Name::new(format!("get_{}_display", field_name));
                fields.insert(
                    method_name.clone(),
                    self.get_display_method(cls, &method_name),
                );
            }
        }

        Some(ClassSynthesizedFields::new(fields))
    }
}
