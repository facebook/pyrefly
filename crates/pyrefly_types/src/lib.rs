/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#![warn(clippy::all)]
#![allow(clippy::match_like_matches_macro)]
#![allow(clippy::needless_lifetimes)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::single_match)]
#![allow(clippy::type_complexity)]
#![allow(clippy::new_without_default)]
#![deny(clippy::cloned_instead_of_copied)]
#![deny(clippy::derive_partial_eq_without_eq)]
#![deny(clippy::inefficient_to_string)]
#![deny(clippy::mem_replace_option_with_some)]
#![deny(clippy::str_to_string)]
#![deny(clippy::trivially_copy_pass_by_ref)]
#![feature(box_patterns)]
#![feature(if_let_guard)]

pub mod alias;
pub mod annotation;
pub mod callable;
pub mod class;
pub mod display;
pub mod equality;
pub mod facet;
pub mod globals;
pub mod keywords;
pub mod lit_int;
pub mod literal;
pub mod module;
pub mod param_spec;
pub mod quantified;
pub mod read_only;
pub mod simplify;
pub mod special_form;
pub mod stdlib;
pub mod tuple;
pub mod type_info;
pub mod type_output;
pub mod type_var;
pub mod type_var_tuple;
pub mod typed_dict;
pub mod types;
