/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Display;
use std::hash::Hash;

use dupe::Dupe;
use pyrefly_python::module::Module;
use pyrefly_python::nesting_context::NestingContext;
use pyrefly_python::qname::QName;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::Identifier;

use crate::equality::TypeEq;
use crate::equality::TypeEqCtx;
use crate::heap::TypeHeap;
use crate::types::Type;

/// Used to represent Sentinel calls. Each Sentinel is unique, so use the ArcId to separate them.
#[derive(Clone, Dupe, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct Sentinel(ArcId<QName>);

impl Visit<Type> for Sentinel {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

impl VisitMut<Type> for Sentinel {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}

impl Display for Sentinel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.id())
    }
}

impl Sentinel {
    pub fn new(name: Identifier, nesting_context: NestingContext, module: Module) -> Self {
        Self(ArcId::new(QName::new(name, nesting_context, module)))
    }

    pub fn qname(&self) -> &QName {
        &self.0
    }

    pub fn to_type(&self, heap: &TypeHeap) -> Type {
        heap.mk_sentinel(self.dupe())
    }

    pub fn type_eq_inner(&self, other: &Self, ctx: &mut TypeEqCtx) -> bool {
        self.0.type_eq(&other.0, ctx)
    }
}
