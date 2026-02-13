/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Display;

use pyrefly_derive::TypeEq;
use ruff_python_ast::ExprName;
use ruff_python_ast::name::Name;
use vec1::Vec1;

/// The idea of "facet narrowing" is that for attribute narrowing, index narrowing,
/// and some other cases we maintain a tree of "facets" (things like attributes, etc)
/// for which we have narrowed types and we'll use these both for narrowing and for
/// reading along "facet chains".
///
/// For example if I write
/// `if x.y is not None and x.z is not None and x.y[0]["w"] is not None: ...`
/// then we'll wind up with two facet chains narrowed in our tree, one at
///   [Attribute(y), Index(0), Key("w")]
/// and another at
///   [Attribute(z)]
#[derive(Debug, Clone, PartialEq, Eq, TypeEq, Hash)]
pub enum FacetKind {
    Attribute(Name),
    Index(usize),
    Key(String),
}

impl Display for FacetKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Attribute(name) => write!(f, ".{name}"),
            Self::Index(idx) => write!(f, "[{idx}]"),
            Self::Key(key) => write!(f, "[\"{key}\"]"),
        }
    }
}

impl FacetKind {
    pub fn invalidate_on_unknown_assignment(&self) -> bool {
        match self {
            Self::Attribute(_) => false,
            Self::Index(_) | Self::Key(_) => true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FacetChain(pub Box<Vec1<FacetKind>>);

impl FacetChain {
    pub fn new(chain: Vec1<FacetKind>) -> Self {
        Self(Box::new(chain))
    }

    pub fn facets(&self) -> &Vec1<FacetKind> {
        match self {
            Self(chain) => chain,
        }
    }
}

impl Display for FacetChain {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for facet in self.0.iter() {
            write!(f, "{facet}")?;
        }
        Ok(())
    }
}

// This is like `FacetKind`, but it can also represent subscripts that are arbitrary names with unknown types
// `VariableSubscript` may resolve to a `FacetKind::Index`, `FacetKind::Key`, or nothing at all
// depending on the type of the variable it contains
#[derive(Debug, Clone, PartialEq)]
pub enum UnresolvedFacetKind {
    Attribute(Name),
    Index(usize),
    Key(String),
    VariableSubscript(ExprName),
}

impl Display for UnresolvedFacetKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Attribute(name) => write!(f, ".{name}"),
            Self::Index(idx) => write!(f, "[{idx}]"),
            Self::Key(key) => write!(f, "[\"{key}\"]"),
            Self::VariableSubscript(name) => write!(f, "[{}]", name.id),
        }
    }
}

// This is like `FacetChain`, but it can also represent subscripts that are arbitrary names with unknown types
// It gets resolved to `FacetChain` if all names in the chain resolve to literal int or string types
#[derive(Clone, Debug)]
pub struct UnresolvedFacetChain(pub Box<Vec1<UnresolvedFacetKind>>);

impl UnresolvedFacetChain {
    pub fn new(chain: Vec1<UnresolvedFacetKind>) -> Self {
        Self(Box::new(chain))
    }

    pub fn facets(&self) -> &Vec1<UnresolvedFacetKind> {
        match self {
            Self(chain) => chain,
        }
    }
}

impl Display for UnresolvedFacetChain {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for facet in self.0.iter() {
            write!(f, "{facet}")?;
        }
        Ok(())
    }
}
