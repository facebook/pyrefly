/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Global variables defined at the top level of a module.
//!
//! We do not include `__doc__` as that has a type that changes based on other variables.

use std::iter;

use ruff_python_ast::name::Name;

use super::stdlib::Stdlib;
use super::types::Type;

#[derive(Debug, Clone)]
pub struct Global {
    name: Name,
    ty: fn(&Stdlib) -> Type,
}

fn dict_str_any(stdlib: &Stdlib) -> Type {
    stdlib
        .dict(stdlib.str().clone().to_type(), Type::any_explicit())
        .clone()
        .to_type()
}

const GLOBALS: &[Global] = &[
    Global::new("__annotations__", dict_str_any),
    Global::new("__builtins__", |_| Type::any_explicit()),
    Global::new("__cached__", |stdlib| stdlib.str().clone().to_type()),
    Global::new("__debug__", |stdlib| stdlib.bool().clone().to_type()),
    Global::new("__dict__", dict_str_any),
    Global::new("__file__", |stdlib| stdlib.str().clone().to_type()),
    Global::new("__loader__", |_| Type::any_explicit()),
    Global::new("__name__", |stdlib| stdlib.str().clone().to_type()),
    Global::new("__package__", |stdlib| {
        Type::optional(stdlib.str().clone().to_type())
    }),
    Global::new("__path__", |stdlib| {
        stdlib
            .mutable_sequence(stdlib.str().clone().to_type())
            .to_type()
    }),
    Global::new("__spec__", |_| Type::any_explicit()),
];

impl Global {
    const fn new(name: &'static str, ty: fn(&Stdlib) -> Type) -> Self {
        Self {
            name: Name::new_static(name),
            ty,
        }
    }

    pub fn globals(has_docstring: bool) -> impl Iterator<Item = Global> {
        GLOBALS
            .iter()
            .cloned()
            .chain(iter::once(Self::doc(has_docstring)))
    }

    #[allow(dead_code)]
    pub fn from_name(name: &Name) -> Option<Global> {
        if name.starts_with("__") && name.ends_with("__") {
            GLOBALS.iter().find(|x| &x.name == name).cloned()
        } else {
            None
        }
    }

    pub fn name(&self) -> &Name {
        &self.name
    }

    pub fn as_type(&self, stdlib: &Stdlib) -> Type {
        (self.ty)(stdlib)
    }

    pub fn doc(has_docstring: bool) -> Self {
        if has_docstring {
            Self::new("__doc__", |stdlib| stdlib.str().clone().to_type())
        } else {
            Self::new("__doc__", |_| Type::None)
        }
    }
}
