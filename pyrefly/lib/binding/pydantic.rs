/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ruff_python_ast::Expr;
use ruff_python_ast::name::Name;
use starlark_map::Hashed;

use crate::binding::bindings::BindingsBuilder;
use crate::export::special::SpecialExport;

// special pydantic constants
pub const FROZEN_DEFAULT: bool = false;
pub const VALIDATION_ALIAS: Name = Name::new_static("validation_alias");
pub const VALIDATE_BY_NAME: Name = Name::new_static("validate_by_name");
pub const VALIDATE_BY_ALIAS: Name = Name::new_static("validate_by_alias");
pub const GT: Name = Name::new_static("gt");
pub const LT: Name = Name::new_static("lt");
pub const GE: Name = Name::new_static("ge");
pub const ROOT: Name = Name::new_static("root");
pub const STRICT: Name = Name::new_static("strict");
const FROZEN: Name = Name::new_static("frozen");
const EXTRA: Name = Name::new_static("extra");

// Pydantic metadata that we will later extend to include more fields
// This is different than the PydanticMetadata that goes into the class metadata itself.
// TODO Zeina: look into if we want to store the expr itself or the boolean. Right now,
// this maps 1:1 to PydanticMetadata structure we encounter in the answers phase,
// but this will likely change as we add more fields.
#[derive(Debug, Clone, Default)]
pub struct PydanticMetadataBinding {
    pub frozen: Option<bool>,
    pub extra: Option<bool>,
    pub validate_by_name: bool,
    pub validate_by_alias: bool,
}

impl<'a> BindingsBuilder<'a> {
    // The goal of this function is to extract pydantic metadata (https://docs.pydantic.dev/latest/concepts/models/) from expressions.
    // TODO: Consider propagating the entire expression instead of the value
    // in case it is aliased.
    pub fn extract_pydantic_config_dict_metadata(
        &self,
        e: &Expr,
        name: &Hashed<Name>,
        pydantic_frozen: &mut Option<bool>,
        pydantic_extra: &mut Option<bool>,
        pydantic_validate_by_name: &mut bool,
        pydantic_validate_by_alias: &mut bool,
    ) {
        if name.as_str() == "model_config"
            && let Some(call) = e.as_call_expr()
            && let Some(special) = self.as_special_export(&call.func)
            && special == SpecialExport::PydanticConfigDict
        {
            for kw in &call.arguments.keywords {
                if let Some(arg_name) = &kw.arg
                    && arg_name.id == FROZEN
                    && let Expr::BooleanLiteral(bl) = &kw.value
                {
                    *pydantic_frozen = Some(bl.value);
                }

                if let Some(arg_name) = &kw.arg
                    && arg_name.id == EXTRA
                {
                    let config_dict_extra = kw.value.clone().string_literal_expr();
                    *pydantic_extra = match config_dict_extra {
                        Some(extra) => {
                            let val = extra.value.to_str();

                            if val == "allow" || val == "ignore" {
                                Some(true)
                            } else if val == "forbid" {
                                Some(false)
                            } else {
                                None
                            }
                        }
                        None => None,
                    };
                }

                if let Some(arg_name) = &kw.arg
                    && arg_name.id == VALIDATE_BY_NAME
                    && let Expr::BooleanLiteral(bl) = &kw.value
                {
                    *pydantic_validate_by_name = bl.value;
                }

                if let Some(arg_name) = &kw.arg
                    && arg_name.id == VALIDATE_BY_ALIAS
                    && let Expr::BooleanLiteral(bl) = &kw.value
                {
                    *pydantic_validate_by_alias = bl.value;
                }
            }
        }
    }

    pub fn make_pydantic_metadata(
        &self,
        frozen: Option<bool>,
        extra: Option<bool>,
        validate_by_name: bool,
        validate_by_alias: bool,
    ) -> PydanticMetadataBinding {
        PydanticMetadataBinding {
            frozen,
            extra,
            validate_by_name,
            validate_by_alias,
        }
    }
}
