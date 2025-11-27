/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;

use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;

use crate::type_output::TypeOutput;
use crate::types::Type;

/*
Eventually this will have to be generalized enough to handle at least four cases:

1. the gradually-typed tuple tuple[Any, ...]
2. normal tuples as are handled here
3. variadic tuples with a splatted typevartuple variable
4. indefinite-length tuples tuple[int, ...] (whose length is supposed to be treated soundly, not gradually, IIRC)
*/

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum Tuple {
    // tuple[t1, t2]
    Concrete(Vec<Type>),
    // tuple[t1, ...]
    Unbounded(Box<Type>),
    // tuple[t1, t2, *t3, t4, t5], where t3 must be a type var tuple or unbounded tuple
    Unpacked(Box<(Vec<Type>, Type, Vec<Type>)>),
}

impl Default for Tuple {
    fn default() -> Self {
        Self::Concrete(Vec::new())
    }
}

impl Tuple {
    // Check if this is tuple[Any, ...]
    pub fn is_any_tuple(&self) -> bool {
        match self {
            Self::Unbounded(ty) => ty.is_any(),
            _ => false,
        }
    }

    pub fn unpacked(prefix: Vec<Type>, middle: Type, suffix: Vec<Type>) -> Tuple {
        if prefix.is_empty()
            && suffix.is_empty()
            && let Type::Tuple(tuple) = middle
        {
            return tuple;
        }
        Self::Unpacked(Box::new((prefix, middle, suffix)))
    }

    pub fn fmt_with_type<O: TypeOutput>(
        &self,
        output: &mut O,
        write_type: &impl Fn(&Type, &mut O) -> fmt::Result,
    ) -> fmt::Result {
        output.write_str("tuple")?;
        output.write_str("[")?;
        match self {
            Self::Concrete(elts) => {
                if elts.is_empty() {
                    output.write_str("()")?;
                } else {
                    for (i, elt) in elts.iter().enumerate() {
                        if i > 0 {
                            output.write_str(", ")?;
                        }
                        write_type(elt, output)?;
                    }
                }
            }
            Self::Unbounded(ty) => {
                write_type(ty, output)?;
                output.write_str(", ...")?;
            }
            Self::Unpacked(box (prefix, unpacked, suffix)) => {
                for (i, ty) in prefix.iter().enumerate() {
                    if i > 0 {
                        output.write_str(", ")?;
                    }
                    write_type(ty, output)?;
                }
                if !prefix.is_empty() {
                    output.write_str(", ")?;
                }
                output.write_str("*")?;
                write_type(unpacked, output)?;
                if !suffix.is_empty() {
                    output.write_str(", ")?;
                    for (i, ty) in suffix.iter().enumerate() {
                        if i > 0 {
                            output.write_str(", ")?;
                        }
                        write_type(ty, output)?;
                    }
                }
            }
        }
        output.write_str("]")
    }
}
