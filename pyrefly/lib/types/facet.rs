/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Display;

use pyrefly_derive::TypeEq;
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
            Self::Attribute(name) => write!(f, ".{}", name),
            Self::Index(idx) => write!(f, "[{}]", idx),
            Self::Key(key) => write!(f, "[\"{}\"]", key),
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
            write!(f, "{}", facet)?;
        }
        Ok(())
    }
}
