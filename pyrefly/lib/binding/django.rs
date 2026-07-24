/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ruff_python_ast::Expr;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;

use crate::binding::binding::ClassFieldDefinition;
use crate::binding::binding::ExprOrBinding;
use crate::binding::bindings::BindingsBuilder;

const PRIMARY_KEY: Name = Name::new_static("primary_key");
const FOREIGN_KEY: Name = Name::new_static("ForeignKey");
const ONE_TO_ONE_FIELD: Name = Name::new_static("OneToOneField");
const CHOICES: Name = Name::new_static("choices");

/// Django-specific field information detected during binding phase.
#[derive(Clone, Debug, Default)]
pub struct DjangoFieldInfo {
    /// The name of the field that has primary_key=True, if any.
    pub primary_key_field: Option<Name>,
    /// Names of ForeignKey and OneToOneField fields.
    pub foreign_key_like_fields: Vec<Name>,
    /// Names of fields with choices=...
    pub fields_with_choices: Vec<Name>,
}

impl<'a> BindingsBuilder<'a> {
    /// Extract Django field information from class body field definitions.
    /// Scans all fields assigned in the class body for Django-specific patterns
    /// (primary_key, ForeignKey, choices).
    pub fn extract_django_fields_from_class_body(
        &self,
        field_definitions: &SmallMap<Name, (ClassFieldDefinition, TextRange)>,
    ) -> DjangoFieldInfo {
        let mut primary_key_field = None;
        let mut foreign_key_like_fields = Vec::new();
        let mut fields_with_choices = Vec::new();
        for (name, (definition, _range)) in field_definitions.iter() {
            if let ClassFieldDefinition::AssignedInBody { value, .. } = definition
                && let ExprOrBinding::Expr(e) = value.as_ref()
                && let Some(call) = e.as_call_expr()
            {
                if let Some(constructor_name) = match &*call.func {
                    Expr::Name(name) => Some(name.id()),
                    Expr::Attribute(attr) => Some(attr.attr.id()),
                    _ => None,
                } && (*constructor_name == FOREIGN_KEY || *constructor_name == ONE_TO_ONE_FIELD)
                {
                    foreign_key_like_fields.push(name.clone());
                }

                for keyword in &call.arguments.keywords {
                    let Some(arg_name) = &keyword.arg else {
                        continue;
                    };
                    let arg_name = arg_name.id();
                    if *arg_name == PRIMARY_KEY {
                        // Detect if a field has `primary_key=True` set.
                        if let Expr::BooleanLiteral(bl) = &keyword.value
                            && bl.value
                        {
                            primary_key_field = Some(name.clone());
                        }
                    } else if *arg_name == CHOICES {
                        fields_with_choices.push(name.clone());
                    }
                }
            }
        }
        DjangoFieldInfo {
            primary_key_field,
            foreign_key_like_fields,
            fields_with_choices,
        }
    }
}
