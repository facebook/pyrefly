/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Column-schema representation for DataFrame values.
//!
//! A `DataFrameSchema` projects the per-column names and types out of an
//! otherwise-opaque DataFrame instance. Every type-machinery site delegates to
//! `underlying`.

use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use ruff_python_ast::name::Name;

use crate::class::ClassType;
use crate::polars_dtype::PolarsDType;
use crate::types::Type;

/// Whether `columns` captures every column of the DataFrame or only a known
/// subset. A subset arises when a construction argument can't be resolved
/// statically (e.g. a spread or a non-literal column key).
#[derive(
    Debug, PartialOrd, Ord, Clone, Eq, PartialEq, Hash, Visit, VisitMut, TypeEq
)]
pub enum SchemaCompleteness {
    Complete,
    Partial,
}

/// Which library produced the DataFrame. Only Polars frames get the column transforms,
/// since pandas `drop` and `rename` act on rows rather than columns.
#[derive(
    Debug, PartialOrd, Ord, Clone, Eq, PartialEq, Hash, Visit, VisitMut, TypeEq
)]
pub enum DataFrameKind {
    Polars,
    Pandas,
}

/// A DataFrame instance with an inferred column schema.
///
/// `columns` is an order-sensitive `Vec` and every trait is derived, so column
/// order is part of the type's identity.
#[derive(
    Debug, PartialOrd, Ord, Clone, Eq, PartialEq, Hash, Visit, VisitMut, TypeEq
)]
pub struct DataFrameSchema {
    /// The opaque DataFrame class instance (e.g. `pl.DataFrame`). All behavior
    /// delegates here.
    pub underlying: ClassType,
    /// Columns in definition order, each with its Polars dtype.
    pub columns: Vec<(Name, PolarsDType)>,
    pub completeness: SchemaCompleteness,
    pub kind: DataFrameKind,
}

impl DataFrameSchema {
    pub fn to_type(self) -> Type {
        Type::DataFrame(Box::new(self))
    }

    /// The underlying instance as a `Type`, for delegating behavior to it.
    pub fn underlying_type(&self) -> Type {
        Type::ClassType(self.underlying.clone())
    }

    /// Whether a column with this name exists in the schema.
    pub fn has_column(&self, name: &Name) -> bool {
        self.columns.iter().any(|(c, _)| c == name)
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hash;
    use std::hash::Hasher;
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
    use crate::equality::TypeEq;
    use crate::equality::TypeEqCtx;
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
                Identifier::new(Name::new("DataFrame"), TextRange::empty(TextSize::new(0))),
                NestingContext::toplevel(),
                module,
                None,
                false,
            ),
            TArgs::default(),
        )
    }

    fn col(name: &str, dtype: PolarsDType) -> (Name, PolarsDType) {
        (Name::new(name), dtype)
    }

    fn schema(
        columns: Vec<(Name, PolarsDType)>,
        completeness: SchemaCompleteness,
    ) -> DataFrameSchema {
        DataFrameSchema {
            underlying: underlying_class(),
            columns,
            completeness,
            kind: DataFrameKind::Polars,
        }
    }

    fn hash_of(schema: &DataFrameSchema) -> u64 {
        let mut hasher = DefaultHasher::new();
        schema.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn partial_schema_display_shows_trailing_marker() {
        let df = schema(
            vec![col("a", PolarsDType::Int64)],
            SchemaCompleteness::Partial,
        )
        .to_type();
        assert_eq!(
            format!("{df}"),
            "DataFrame[a: Int64, ...]",
            "a Partial schema shows known columns plus a trailing marker for the unknown rest"
        );
    }

    #[test]
    fn column_order_is_part_of_identity() {
        let ab = schema(
            vec![col("a", PolarsDType::Int64), col("b", PolarsDType::String)],
            SchemaCompleteness::Complete,
        );
        let ba = schema(
            vec![col("b", PolarsDType::String), col("a", PolarsDType::Int64)],
            SchemaCompleteness::Complete,
        );

        // Reordered columns are a distinct type under every relation.
        assert_ne!(ab, ba);
        assert_ne!(hash_of(&ab), hash_of(&ba));
        assert_ne!(ab.cmp(&ba), Ordering::Equal);
        assert!(!ab.type_eq(&ba, &mut TypeEqCtx::default()));

        // Identical columns in the same order are equal under every relation.
        let ab2 = schema(
            vec![col("a", PolarsDType::Int64), col("b", PolarsDType::String)],
            SchemaCompleteness::Complete,
        );
        assert_eq!(ab, ab2);
        assert_eq!(hash_of(&ab), hash_of(&ab2));
        assert_eq!(ab.cmp(&ab2), Ordering::Equal);
        assert!(ab.type_eq(&ab2, &mut TypeEqCtx::default()));
    }

    #[test]
    fn completeness_is_part_of_identity() {
        let complete = schema(
            vec![col("a", PolarsDType::Int64)],
            SchemaCompleteness::Complete,
        );
        let partial = schema(
            vec![col("a", PolarsDType::Int64)],
            SchemaCompleteness::Partial,
        );
        assert_ne!(complete, partial);
        assert!(!complete.type_eq(&partial, &mut TypeEqCtx::default()));
    }

    #[test]
    fn kind_is_part_of_identity() {
        let cols = || vec![col("a", PolarsDType::Int64)];
        let polars = schema(cols(), SchemaCompleteness::Complete);
        let pandas = DataFrameSchema {
            kind: DataFrameKind::Pandas,
            ..schema(cols(), SchemaCompleteness::Complete)
        };
        assert_ne!(polars, pandas);
        assert!(!polars.type_eq(&pandas, &mut TypeEqCtx::default()));
    }

    #[test]
    fn traversal_preserves_underlying() {
        // Columns are Polars dtypes, not `Type`s, so type traversal reaches only `underlying`.
        let df = schema(
            vec![col("a", PolarsDType::Int64)],
            SchemaCompleteness::Complete,
        )
        .to_type();
        let Type::DataFrame(s) = df else {
            unreachable!("to_type produces the DataFrame variant")
        };
        assert_eq!(s.underlying, underlying_class());
        assert_eq!(s.columns[0].1, PolarsDType::Int64);
    }
}
