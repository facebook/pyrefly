/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A closed or open collection of named integer `**kwargs` for einops-like APIs.

use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use ruff_python_ast::name::Name;

use crate::dimension::Int;

#[derive(Debug, Clone, PartialEq, Eq, TypeEq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut)]
pub struct NamedInt {
    pub name: Name,
    pub value: Int,
    pub required: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, TypeEq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut)]
pub struct NamedInts {
    entries: Box<[NamedInt]>,
    open: bool,
}

impl NamedInts {
    pub fn new(mut entries: Vec<NamedInt>, open: bool) -> Self {
        entries.sort_by(|left, right| left.name.cmp(&right.name));
        entries.dedup_by(|right, left| {
            if left.name != right.name {
                return false;
            }
            left.required |= right.required;
            if left.value != right.value {
                left.value = Int::Int;
            }
            true
        });
        Self {
            entries: entries.into_boxed_slice(),
            open,
        }
    }

    pub fn entries(&self) -> &[NamedInt] {
        &self.entries
    }

    pub fn is_open(&self) -> bool {
        self.open
    }
}
