/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#![cfg(test)]

mod abstract_methods;
mod assign;
mod attribute_narrow;
mod attributes;
mod callable;
mod calls;
mod class_keywords;
mod class_overrides;
mod class_subtyping;
mod class_super;
mod constructors;
mod contextual;
mod cycles;
mod dataclass_transform;
mod dataclasses;
mod decorators;
mod delayed_inference;
mod descriptors;
mod dict;
mod django;
mod enums;
mod flow_branching;
mod flow_looping;
mod generic_basic;
mod generic_legacy;
mod generic_restrictions;
mod generic_sub;
mod imports;
mod incremental;
mod inference;
mod literal;
mod lsp;
mod mro;
mod named_tuple;
mod narrow;
mod natural;
mod new_type;
mod operators;
mod overload;
mod pandas;
mod paramspec;
mod pattern_match;
mod perf;
mod protocol;
mod pydantic;
mod pysa;
mod redundant_cast;
mod returns;
mod scope;
mod semantic_syntax_errors;
mod simple;
mod state;
mod subscript_narrow;
mod suppression;
mod sys_info;
mod tsp;
mod tuple;
mod type_alias;
mod type_var_tuple;
mod typed_dict;
mod typing_self;
mod unnecessary_comparison;
mod untyped_def_behaviors;
pub mod util;
mod var_resolution;
mod variance_inference;
mod with;
mod yields;
