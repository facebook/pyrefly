/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Element-dtype representation for Series values.
//!
//! A `SeriesSchema` records the element dtype of an otherwise-opaque Series
//! instance, so a column read can return a typed Series. Every type-machinery
//! site delegates to `underlying`.

use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;

use crate::class::ClassType;
use crate::polars_dtype::PolarsDType;
use crate::types::Type;

/// A Series instance carrying its element dtype.
///
/// There is no `kind` field: a Series has no column-algebra transforms to gate,
/// and its library is derivable from `underlying`'s qname when needed.
#[derive(
    Debug, PartialOrd, Ord, Clone, Eq, PartialEq, Hash, Visit, VisitMut, TypeEq
)]
pub struct SeriesSchema {
    /// The opaque Series class instance (e.g. `pl.Series`). All behavior
    /// delegates here.
    pub underlying: ClassType,
    /// The dtype of the Series elements.
    pub dtype: PolarsDType,
}

impl SeriesSchema {
    pub fn to_type(self) -> Type {
        Type::Series(Box::new(self))
    }

    /// The underlying instance as a `Type`, for delegating behavior to it.
    pub fn underlying_type(&self) -> Type {
        Type::ClassType(self.underlying.clone())
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use pyrefly_python::module::Module;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use pyrefly_python::nesting_context::NestingContext;
    use ruff_python_ast::Identifier;
    use ruff_python_ast::name::Name;
    use ruff_text_size::TextRange;
    use ruff_text_size::TextSize;

    use super::*;
    use crate::class::Class;
    use crate::class::ClassDefIndex;
    use crate::class::ClassType;
    use crate::types::TArgs;

    fn underlying_class() -> ClassType {
        let module = Module::new(
            ModuleName::from_str("polars"),
            ModulePath::filesystem(PathBuf::from("polars")),
            Arc::new("fake module contents".to_owned()),
        );
        ClassType::new(
            Class::new(
                ClassDefIndex(0),
                Identifier::new(Name::new("Series"), TextRange::empty(TextSize::new(0))),
                NestingContext::toplevel(),
                module,
                None,
                false,
            ),
            TArgs::default(),
        )
    }

    fn series(dtype: PolarsDType) -> SeriesSchema {
        SeriesSchema {
            underlying: underlying_class(),
            dtype,
        }
    }

    #[test]
    fn display_shows_the_element_dtype() {
        let s = series(PolarsDType::Int64).to_type();
        assert_eq!(format!("{s}"), "Series[Int64]");
    }

    #[test]
    fn strip_replaces_the_schema_with_its_class() {
        let underlying = Type::ClassType(underlying_class());
        assert_eq!(
            series(PolarsDType::Int64).to_type().strip_library_schemas(),
            underlying
        );
        // The strip recurses into nested positions and leaves other types untouched.
        let optional = Type::optional(series(PolarsDType::String).to_type());
        assert_eq!(optional.strip_library_schemas(), Type::optional(underlying));
    }

    #[test]
    fn traversal_preserves_underlying() {
        // The dtype is a plain `PolarsDType`, not a `Type`, so traversal reaches only `underlying`.
        let s = series(PolarsDType::Int64).to_type();
        let Type::Series(schema) = s else {
            unreachable!("to_type produces the Series variant")
        };
        assert_eq!(schema.underlying, underlying_class());
        assert_eq!(schema.dtype, PolarsDType::Int64);
    }
}
