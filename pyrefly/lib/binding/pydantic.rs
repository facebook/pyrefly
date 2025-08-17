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
const FROZEN_DEFAULT: bool = false;
const VALIDATION_ALIAS: Name = Name::new_static("validation_alias");
const FROZEN: Name = Name::new_static("frozen");

// Pydantic metadata that we will later extend to include more fields
// This is different than the PydanticMetadata that goes into the class metadata itself.
// TODO Zeina: look into if we want to store the expr itself or the boolean. Right now,
// this maps 1:1 to PydanticMetadata structure we encounter in the answers phase,
// but this will likely change as we add more fields.
#[derive(Debug, Clone, Default)]
pub struct PydanticMetadataBinding {
    pub frozen: bool,
    pub validation_alias: Option<Box<str>>,
}

impl<'a> BindingsBuilder<'a> {
    // The goal of this function is to extract pydantic metadata (https://docs.pydantic.dev/latest/concepts/models/) from expressions.
    // TODO: Consider propagating the entire expression instead of the value
    // in case it is aliased.
    pub fn extract_frozen_pydantic_metadata(
        &self,
        e: &Expr,
        name: Hashed<&Name>,
        pydantic_frozen: &mut Option<bool>,
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
            }
        }
    }

    pub fn extract_validation_alias(&self, e: &Expr, validation_alias: &mut Option<Box<str>>) {
        if let Some(call) = e.as_call_expr()
            && let Some(special) = self.as_special_export(&call.func)
            && special == SpecialExport::PydanticField
        {
            for kw in &call.arguments.keywords {
                if let Some(arg_name) = &kw.arg
                    && arg_name.id == VALIDATION_ALIAS
                    && let Some(str_lit) = kw.value.as_string_literal_expr()
                {
                    let val = str_lit.value.to_str();
                    *validation_alias = Some(val.to_owned().into_boxed_str());
                }
            }
        }
    }

    // TODO Zeina: We should expect to extend this beyond the frozen data.
    pub fn make_pydantic_metadata(
        &self,
        frozen: Option<bool>,
        validation_alias: Option<Box<str>>,
    ) -> PydanticMetadataBinding {
        PydanticMetadataBinding {
            frozen: frozen.unwrap_or(FROZEN_DEFAULT),
            validation_alias,
        }
    }
}
