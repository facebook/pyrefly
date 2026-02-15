/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

pub(crate) mod convert_star_import;
pub(crate) mod extract_field;
pub(crate) mod extract_function;
mod extract_shared;
pub(crate) mod extract_superclass;
pub(crate) mod extract_variable;
pub(crate) mod generate_code;
pub(crate) mod inline_method;
pub(crate) mod inline_parameter;
pub(crate) mod inline_variable;
pub(crate) mod introduce_parameter;
pub(crate) mod invert_boolean;
pub(crate) mod move_members;
pub(crate) mod move_module;
pub(crate) mod redundant_cast;
pub(crate) mod types;
pub(crate) mod use_function;
