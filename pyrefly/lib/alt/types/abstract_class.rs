/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use pyrefly_derive::TypeEq;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::name::Name;
use starlark_map::small_set::SmallSet;

use crate::types::types::Type;

#[derive(Clone, Debug, TypeEq, PartialEq, Eq)]
pub struct AbstractClassMembers {
    pub unimplemented_abstract_methods: SmallSet<Name>,
}
impl VisitMut<Type> for AbstractClassMembers {
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {
        // No types to visit in this struct
    }
}

impl Display for AbstractClassMembers {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "AbstractClassCheck({:?})",
            self.unimplemented_abstract_methods
        )
    }
}

impl AbstractClassMembers {
    pub fn new(unimplemented_abstract_methods: SmallSet<Name>) -> AbstractClassMembers {
        AbstractClassMembers {
            unimplemented_abstract_methods,
        }
    }

    pub fn recursive() -> Self {
        AbstractClassMembers {
            unimplemented_abstract_methods: SmallSet::new(),
        }
    }

    pub fn unimplemented_abstract_methods(&self) -> &SmallSet<Name> {
        &self.unimplemented_abstract_methods
    }
}
