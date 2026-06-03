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
use pyrefly_derive::TypeEq;
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

#[derive(Clone, Copy, Dupe, Debug, PartialEq, Eq, Hash, Ord, PartialOrd, TypeEq)]
pub enum SentinelKind {
    Builtins,
    TypingExtensions,
}

impl SentinelKind {
    pub fn name(&self) -> &str {
        match self {
            Self::Builtins => "sentinel",
            Self::TypingExtensions => "Sentinel",
        }
    }
}

/// Used to represent Sentinel calls. Each Sentinel is unique, so use the ArcId to separate them.
#[derive(Clone, Dupe, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct Sentinel(ArcId<SentinelInner>);

// This is a lie, we do have types in the bound position
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
        write!(f, "{}", self.0.qname.id())
    }
}

#[derive(Debug, PartialEq, TypeEq, Eq, Ord, PartialOrd)]
struct SentinelInner {
    qname: QName,
    kind: SentinelKind,
}

impl Sentinel {
    pub fn new(name: Identifier, module: Module, kind: SentinelKind) -> Self {
        Self(ArcId::new(SentinelInner {
            kind,
            // TODO: properly take parent from caller of new()
            qname: QName::new(name, NestingContext::toplevel(), module),
        }))
    }

    pub fn kind(&self) -> SentinelKind {
        self.0.kind
    }

    pub fn qname(&self) -> &QName {
        &self.0.qname
    }

    pub fn to_type(&self, heap: &TypeHeap) -> Type {
        heap.mk_sentinel(self.dupe())
    }

    pub fn type_eq_inner(&self, other: &Self, ctx: &mut TypeEqCtx) -> bool {
        self.0.type_eq(&other.0, ctx)
    }
}
