/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Clone_;
use dupe::Copy_;
use dupe::Dupe_;
use ruff_python_ast::name::Name;
use starlark_map::small_map::SmallMap;

use crate::alt::answers::AnswersSolver;
use crate::alt::answers::LookupAnswer;
use crate::alt::classes::Attribute;
use crate::alt::classes::ClassField;
use crate::types::class::Class;
use crate::types::class::ClassType;
use crate::types::stdlib::Stdlib;

/// `TypeOrder` provides a minimal API allowing `Subset` to request additional
/// information about types that may be required for solving bindings
///
/// This is needed for cases like the nominal type order and structural types where
/// the `Type` object itself does not contain enough information to determine
/// subset relations.
#[derive(Clone_, Copy_, Dupe_)]
pub struct TypeOrder<'a, Ans: LookupAnswer>(&'a AnswersSolver<'a, Ans>);

impl<'a, Ans: LookupAnswer> TypeOrder<'a, Ans> {
    pub fn new(solver: &'a AnswersSolver<'a, Ans>) -> Self {
        Self(solver)
    }

    pub fn stdlib(self) -> &'a Stdlib {
        self.0.stdlib
    }

    pub fn has_superclass(self, got: &Class, want: &Class) -> bool {
        self.0.has_superclass(got, want)
    }

    pub fn as_superclass(self, class: &ClassType, want: &Class) -> Option<ClassType> {
        self.0.as_superclass(class, want)
    }

    pub fn has_metaclass(self, cls: &Class, metaclass: &ClassType) -> bool {
        let metadata = self.0.get_metadata_for_class(cls);
        match metadata.metaclass() {
            Some(m) => *m == *metaclass,
            None => *metaclass == self.stdlib().builtins_type(),
        }
    }

    pub fn is_protocol(self, cls: &Class) -> bool {
        self.0.get_metadata_for_class(cls).is_protocol()
    }

    pub fn get_all_members(self, cls: &Class) -> SmallMap<Name, (ClassField, Class)> {
        self.0.get_all_members(cls)
    }

    pub fn get_instance_attribute(self, cls: &ClassType, name: &Name) -> Option<Attribute> {
        self.0.get_instance_attribute(cls, name)
    }
}
