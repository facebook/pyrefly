/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use dupe::Dupe;
use pyrefly_config::error_kind::ErrorKind;
use pyrefly_graph::index::Idx;
use pyrefly_python::dunder;
use pyrefly_python::module_name::ModuleName;
use pyrefly_types::annotation::Annotation;
use pyrefly_types::callable::Callable;
use pyrefly_types::callable::FuncMetadata;
use pyrefly_types::callable::Function;
use pyrefly_types::callable::Param;
use pyrefly_types::callable::ParamList;
use pyrefly_types::callable::Required;
use pyrefly_types::keywords::DataclassFieldKeywords;
use pyrefly_types::lit_int::LitInt;
use pyrefly_types::literal::Lit;
use pyrefly_types::types::Union;
use ruff_python_ast::Expr;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::solve::TypeFormContext;
use crate::alt::types::class_metadata::ClassMetadata;
use crate::alt::types::class_metadata::ClassSynthesizedField;
use crate::alt::types::class_metadata::DataclassMetadata;
use crate::alt::types::pydantic::PydanticConfig;
use crate::alt::types::pydantic::PydanticModelKind;
use crate::alt::types::pydantic::PydanticModelKind::RootModel;
use crate::alt::types::pydantic::PydanticValidationFlags;
use crate::binding::binding::BindingAnnotation;
use crate::binding::binding::KeyAnnotation;
use crate::binding::pydantic::EXTRA;
use crate::binding::pydantic::FROZEN;
use crate::binding::pydantic::FROZEN_DEFAULT;
use crate::binding::pydantic::PydanticConfigDict;
use crate::binding::pydantic::ROOT;
use crate::binding::pydantic::STRICT;
use crate::binding::pydantic::STRICT_DEFAULT;
use crate::binding::pydantic::VALIDATE_BY_ALIAS;
use crate::binding::pydantic::VALIDATE_BY_NAME;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorInfo;
use crate::types::class::Class;
use crate::types::types::Type;

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub fn get_pydantic_root_model_type_via_mro(
        &self,
        class: &Class,
        metadata: &ClassMetadata,
    ) -> Option<(Type, bool)> {
        if !matches!(metadata.pydantic_model_kind(), Some(RootModel)) {
            return None;
        }

        let has_strict = self
            .get_base_types_for_class(class)
            .has_pydantic_strict_metadata;

        let mro = self.get_mro_for_class(class);
        for base_type in mro.ancestors_no_object() {
            if base_type.has_qname(ModuleName::pydantic_root_model().as_str(), "RootModel") {
                let targs = base_type.targs().as_slice();
                if let Some(root_type) = targs.last().cloned() {
                    return Some((root_type, has_strict));
                }
            }
        }

        None
    }

    pub fn get_pydantic_root_model_init(
        &self,
        cls: &Class,
        root_model_type: Type,
        has_strict: bool,
    ) -> ClassSynthesizedField {
        let (root_requiredness, root_model_type) =
            if root_model_type.is_any() || matches!(root_model_type, Type::Quantified(_)) {
                (Required::Optional(None), root_model_type)
            } else if has_strict {
                (Required::Required, root_model_type)
            } else {
                (Required::Required, Type::any_explicit())
            };
        let root_param = Param::Pos(ROOT, root_model_type, root_requiredness);
        let params = vec![self.class_self_param(cls, false), root_param];
        let ty = Type::Function(Box::new(Function {
            signature: Callable::list(ParamList::new(params), Type::None),
            metadata: FuncMetadata::def(self.module().dupe(), cls.dupe(), dunder::INIT),
        }));
        ClassSynthesizedField::new(ty)
    }

    pub fn get_pydantic_root_model_class_field_type(
        &self,
        cls: &Class,
        attr_name: &Name,
    ) -> Option<Type> {
        if !cls.has_toplevel_qname(ModuleName::pydantic_root_model().as_str(), "RootModel")
            || *attr_name != dunder::INIT
        {
            return None;
        }
        let tparams = self.get_class_tparams(cls);
        // `RootModel` should always have a type parameter unless we're working with a broken copy
        // of Pydantic.
        let tparam = tparams.iter().next()?;
        let root_model_type = Type::Quantified(Box::new(tparam.quantified.clone()));
        Some(
            self.get_pydantic_root_model_init(cls, root_model_type, false)
                .inner
                .ty(),
        )
    }

    pub fn is_pydantic_strict_metadata(&self, ty: &Type) -> bool {
        match ty {
            Type::ClassType(cls) => cls.has_qname(ModuleName::pydantic_types().as_str(), "Strict"),
            _ => false,
        }
    }

    /// Helper function to find inherited keyword values from parent dataclass metadata
    fn find_inherited_keyword_value<T>(
        &self,
        bases_with_metadata: &[(Class, Arc<ClassMetadata>)],
        extractor: impl Fn(&DataclassMetadata) -> T,
    ) -> Option<T> {
        bases_with_metadata
            .iter()
            .find_map(|(_, metadata)| metadata.dataclass_metadata().map(&extractor))
    }

    /// Check if a type is a RootModel or subclass of RootModel, and if so, recursively extract all inner types.
    /// Returns Some(root_type) if the type is a RootModel, None otherwise.
    /// For unions containing multiple RootModels, extracts and returns a union of all their inner types.
    /// Recursively expands nested RootModels (e.g., RootModel[RootModel[int]] expands to RootModel[int] | int).
    pub fn extract_root_model_inner_type(&self, ty: &Type) -> Option<Type> {
        match ty {
            Type::Union(box Union { members: types, .. }) => {
                let root_types: Vec<Type> = types
                    .iter()
                    .filter_map(|t| self.extract_root_model_inner_type(t))
                    .collect();

                if root_types.is_empty() {
                    None
                } else {
                    Some(self.unions(root_types))
                }
            }
            Type::ClassType(cls) => {
                if cls.has_qname(ModuleName::pydantic_root_model().as_str(), "RootModel") {
                    let targs = cls.targs().as_slice();
                    let root_type = targs.last().cloned().unwrap_or_else(Type::any_implicit);
                    if let Some(nested_root_type) = self.extract_root_model_inner_type(&root_type) {
                        return Some(self.union(root_type.clone(), nested_root_type));
                    }
                    return Some(root_type);
                }

                let metadata = self.get_metadata_for_class(cls.class_object());
                if matches!(metadata.pydantic_model_kind(), Some(RootModel))
                    && let Some((root_type, _)) =
                        self.get_pydantic_root_model_type_via_mro(cls.class_object(), &metadata)
                {
                    // Recursively expand if the inner type is also a RootModel
                    // Return union of immediate inner type AND recursive expansion
                    if let Some(nested_root_type) = self.extract_root_model_inner_type(&root_type) {
                        return Some(self.union(root_type.clone(), nested_root_type));
                    }
                    Some(root_type)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub fn pydantic_config(
        &self,
        bases_with_metadata: &[(Class, Arc<ClassMetadata>)],
        pydantic_config_dict: &PydanticConfigDict,
        keywords: &[(Name, Annotation)],
        errors: &ErrorCollector,
        range: TextRange,
    ) -> Option<PydanticConfig> {
        let has_pydantic_base_model_base_class =
            bases_with_metadata.iter().any(|(base_class_object, _)| {
                base_class_object.has_toplevel_qname(ModuleName::pydantic().as_str(), "BaseModel")
            });

        let has_pydantic_base_settings_base_class =
            bases_with_metadata.iter().any(|(base_class_object, _)| {
                base_class_object
                    .has_toplevel_qname(ModuleName::pydantic_settings().as_str(), "BaseSettings")
            });

        let is_pydantic_base_model = has_pydantic_base_model_base_class
            || bases_with_metadata
                .iter()
                .any(|(_, metadata)| metadata.is_pydantic_base_model());

        if !is_pydantic_base_model {
            return None;
        }

        let has_pydantic_root_model_base_class =
            bases_with_metadata.iter().any(|(base_class_object, _)| {
                base_class_object
                    .has_toplevel_qname(ModuleName::pydantic_root_model().as_str(), "RootModel")
            });

        let has_base_settings_kind = bases_with_metadata.iter().any(|(_, metadata)| {
            matches!(
                metadata.pydantic_model_kind(),
                Some(PydanticModelKind::BaseSettings)
            )
        });

        let has_root_model_kind = bases_with_metadata.iter().any(|(_, metadata)| {
            matches!(
                metadata.pydantic_model_kind(),
                Some(PydanticModelKind::RootModel)
            )
        });

        let pydantic_model_kind = if has_pydantic_root_model_base_class || has_root_model_kind {
            PydanticModelKind::RootModel
        } else if has_pydantic_base_settings_base_class || has_base_settings_kind {
            PydanticModelKind::BaseSettings
        } else {
            PydanticModelKind::BaseModel
        };

        let PydanticConfigDict {
            frozen,
            extra,
            strict,
            validate_by_name,
            validate_by_alias,
        } = pydantic_config_dict;

        // Note: class keywords take precedence over ConfigDict keywords.
        // But another design choice is to error if there is a conflict. We can consider this design for v2.

        let default_flags = PydanticValidationFlags::default();
        let validation_flags = PydanticValidationFlags {
            validate_by_name: self.get_bool_config_value(
                &VALIDATE_BY_NAME,
                keywords,
                *validate_by_name,
                bases_with_metadata,
                |dm| dm.init_defaults.init_by_name,
                default_flags.validate_by_name,
            ),
            validate_by_alias: self.get_bool_config_value(
                &VALIDATE_BY_ALIAS,
                keywords,
                *validate_by_alias,
                bases_with_metadata,
                |dm| dm.init_defaults.init_by_alias,
                default_flags.validate_by_alias,
            ),
        };

        // Here, "ignore" and "allow" translate to true, while "forbid" translates to false.
        // With no keyword, the default is "true" and I default to "false" on a wrong keyword.
        // If we were to consider type narrowing in the "allow" case, we would need to propagate more data
        // and narrow downstream. We are not following the narrowing approach in v1 though, but should discuss it
        // for v2.
        let extra = match keywords.iter().find(|(name, _)| name == &EXTRA) {
            Some((_, ann)) => match ann.get_type() {
                Type::Literal(Lit::Str(s)) => match s.as_str() {
                    "allow" | "ignore" => true,
                    "forbid" => false,
                    _ => {
                        self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::InvalidLiteral),
                    "Invalid value for `extra`. Expected one of 'allow', 'ignore', or 'forbid'"
                        .to_owned(),
                );
                        true
                    }
                },
                _ => {
                    self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::InvalidLiteral),
                        "Invalid value for `extra`. Expected one of 'allow', 'ignore', or 'forbid'"
                            .to_owned(),
                    );
                    true
                }
            },
            None => {
                // No "extra" keyword in the class-level keywords,
                // so check if configdict has it, otherwise inherit from base classes
                if let Some(configdict_extra) = extra {
                    *configdict_extra
                } else {
                    // Check for inherited extra configuration from base classes
                    self.find_inherited_keyword_value(bases_with_metadata, |dm| dm.kws.extra)
                        .unwrap_or(true) // Default to true (ignore) if no base class has extra config
                }
            }
        };

        let frozen = self.get_bool_config_value(
            &FROZEN,
            keywords,
            *frozen,
            bases_with_metadata,
            |dm| dm.kws.frozen,
            FROZEN_DEFAULT,
        );

        let strict = self.get_bool_config_value(
            &STRICT,
            keywords,
            *strict,
            bases_with_metadata,
            |dm| dm.kws.strict,
            STRICT_DEFAULT,
        );

        Some(PydanticConfig {
            frozen,
            validation_flags,
            extra,
            strict,
            pydantic_model_kind,
        })
    }

    fn get_bool_config_value(
        &self,
        name: &Name,
        keywords: &[(Name, Annotation)],
        value_from_config_dict: Option<bool>,
        bases_with_metadata: &[(Class, Arc<ClassMetadata>)],
        extract_from_metadata: impl Fn(&DataclassMetadata) -> bool,
        default: bool,
    ) -> bool {
        // explicit keyword > explicit ConfigDict value > inherited > default
        self.extract_bool_flag(keywords, name)
            .unwrap_or(value_from_config_dict.unwrap_or_else(|| {
                self.find_inherited_keyword_value(bases_with_metadata, extract_from_metadata)
                    .unwrap_or(default)
            }))
    }

    fn extract_bool_flag(&self, keywords: &[(Name, Annotation)], key: &Name) -> Option<bool> {
        keywords
            .iter()
            .find(|(name, _)| name == key)
            .and_then(|(_, ann)| ann.get_type().as_bool())
    }

    pub fn check_pydantic_range_constraints(
        &self,
        field_name: &Name,
        field_ty: &Type,
        keywords: &DataclassFieldKeywords,
        range: TextRange,
        errors: &ErrorCollector,
    ) {
        // Note: the subset check here is too conservative when it comes to modeling runtime behavior
        // we want to check if the bound_val is coercible to the annotation type at runtime.
        // statically, this could be a challenge, which is why we go with this more conservative approach for now.
        for (bound_val, label) in [
            (&keywords.gt, "gt"),
            (&keywords.lt, "lt"),
            (&keywords.ge, "ge"),
            (&keywords.le, "le"),
        ] {
            let Some(val) = bound_val else { continue };
            if !self.is_subset_eq(val, field_ty) {
                self.error(
                        errors,
                        range,
                        ErrorInfo::Kind(ErrorKind::BadArgumentType),
                        format!(
                            "Pydantic `{label}` value is of type `{}` but the field is annotated with `{}`",
                            self.for_display(val.clone()),
                            self.for_display(field_ty.clone())
                        ),
                    );
            }
        }
        self.check_pydantic_range_default(field_name, keywords, range, errors);
    }

    fn check_pydantic_range_default(
        &self,
        field_name: &Name,
        keywords: &DataclassFieldKeywords,
        range: TextRange,
        errors: &ErrorCollector,
    ) {
        fn int_literal_from_type(ty: &Type) -> Option<&LitInt> {
            // We only currently enforce range constraints for literal defaults, so carve out
            // the `Literal[int]` case and ignore everything else.
            match ty {
                Type::Literal(Lit::Int(lit)) => Some(lit),
                _ => None,
            }
        }
        let Some(default_ty) = &keywords.default else {
            return;
        };
        let Some(value_lit) = int_literal_from_type(default_ty) else {
            return;
        };
        let emit_violation = |label: &str, constraint_ty: &Type| {
            let Some(constraint_lit) = int_literal_from_type(constraint_ty) else {
                return;
            };
            let comparison = value_lit.cmp(constraint_lit);
            let violates = match label {
                "gt" => !matches!(comparison, std::cmp::Ordering::Greater),
                "ge" => matches!(comparison, std::cmp::Ordering::Less),
                "lt" => !matches!(comparison, std::cmp::Ordering::Less),
                "le" => matches!(comparison, std::cmp::Ordering::Greater),
                _ => false,
            };
            if violates {
                self.error(
                    errors,
                    range,
                    ErrorInfo::Kind(ErrorKind::BadArgumentType),
                    format!(
                        "Default value `{}` violates Pydantic `{}` constraint `{}` for field `{}`",
                        self.for_display(default_ty.clone()),
                        label,
                        self.for_display(constraint_ty.clone()),
                        field_name
                    ),
                );
            }
        };

        if let Some(gt) = &keywords.gt {
            emit_violation("gt", gt);
        }
        if let Some(ge) = &keywords.ge {
            emit_violation("ge", ge);
        }
        if let Some(lt) = &keywords.lt {
            emit_violation("lt", lt);
        }
        if let Some(le) = &keywords.le {
            emit_violation("le", le);
        }
    }

    /// Extract Pydantic Field metadata from an annotation binding.
    /// This handles the Pydantic-specific pattern where fields can be declared as:
    /// `field: Annotated[some_type, Field(some_keyword=<value>)]`
    pub fn extract_pydantic_field_from_annotation(
        &self,
        annot: Idx<KeyAnnotation>,
        metadata: &ClassMetadata,
    ) -> Option<DataclassFieldKeywords> {
        let dm = metadata.dataclass_metadata()?;
        if !metadata.is_pydantic_base_model() {
            return None;
        }
        if let BindingAnnotation::AnnotateExpr(_, annotation_expr, _) = self.bindings().get(annot) {
            let metadata_items = self.get_annotated_metadata(
                annotation_expr,
                TypeFormContext::ClassVarAnnotation,
                &self.error_swallower(),
            );
            // Look through metadata items and find a Field(...) call, then extract its keywords
            for metadata_item in &metadata_items {
                if let Expr::Call(call) = metadata_item
                    && let Some(keywords) = self.compute_dataclass_field_initialization(call, dm)
                {
                    return Some(keywords);
                }
            }
        }
        None
    }
}
