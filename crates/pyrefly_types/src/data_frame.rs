/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Column schemas attached to otherwise-opaque DataFrame instances.

use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use ruff_python_ast::name::Name;

use crate::class::ClassType;
use crate::polars_dtype::PolarsDType;
use crate::types::Type;

/// Whether `columns` is exhaustive.
#[derive(
    Debug, PartialOrd, Ord, Clone, Copy, Eq, PartialEq, Hash, Visit, VisitMut, TypeEq
)]
pub enum SchemaCompleteness {
    Complete,
    Partial,
}

impl SchemaCompleteness {
    pub fn combine(self, other: Self) -> Self {
        match (self, other) {
            (Self::Complete, Self::Complete) => Self::Complete,
            _ => Self::Partial,
        }
    }
}

#[derive(
    Debug, PartialOrd, Ord, Clone, Copy, Eq, PartialEq, Hash, Visit, VisitMut, TypeEq
)]
pub enum DataFrameKind {
    Polars,
    Pandas,
}

/// A DataFrame whose ordered columns are part of its type identity.
#[derive(
    Debug, PartialOrd, Ord, Clone, Eq, PartialEq, Hash, Visit, VisitMut, TypeEq
)]
pub struct DataFrameSchema {
    pub underlying: ClassType,
    pub columns: Vec<(Name, PolarsDType)>,
    pub completeness: SchemaCompleteness,
    pub kind: DataFrameKind,
}

impl DataFrameSchema {
    pub fn to_type(self) -> Type {
        Type::DataFrame(Box::new(self))
    }

    pub fn underlying_type(&self) -> Type {
        Type::ClassType(self.underlying.clone())
    }

    pub fn is_complete(&self) -> bool {
        self.completeness == SchemaCompleteness::Complete
    }

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
    use crate::class::PrecomputedTParams;
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
                PrecomputedTParams::NotGeneric,
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

        assert_ne!(ab, ba);
        assert_ne!(hash_of(&ab), hash_of(&ba));
        assert_ne!(ab.cmp(&ba), Ordering::Equal);
        assert!(!ab.type_eq(&ba, &mut TypeEqCtx::default()));

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
    fn combining_completeness_requires_both_schemas_to_be_complete() {
        assert_eq!(
            SchemaCompleteness::Complete.combine(SchemaCompleteness::Complete),
            SchemaCompleteness::Complete
        );
        assert_eq!(
            SchemaCompleteness::Complete.combine(SchemaCompleteness::Partial),
            SchemaCompleteness::Partial
        );
        assert_eq!(
            SchemaCompleteness::Partial.combine(SchemaCompleteness::Complete),
            SchemaCompleteness::Partial
        );
        assert_eq!(
            SchemaCompleteness::Partial.combine(SchemaCompleteness::Partial),
            SchemaCompleteness::Partial
        );
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
    fn strip_library_schemas_replaces_every_schema_with_its_class() {
        let underlying = Type::ClassType(underlying_class());

        for (columns, completeness) in [
            (
                vec![col("a", PolarsDType::Int64)],
                SchemaCompleteness::Complete,
            ),
            (vec![], SchemaCompleteness::Complete),
            (
                vec![col("a", PolarsDType::Int64)],
                SchemaCompleteness::Partial,
            ),
        ] {
            let stripped = schema(columns, completeness)
                .to_type()
                .strip_library_schemas();
            assert_eq!(stripped, underlying);
        }

        let pandas = DataFrameSchema {
            kind: DataFrameKind::Pandas,
            ..schema(
                vec![col("a", PolarsDType::Int64)],
                SchemaCompleteness::Complete,
            )
        }
        .to_type();
        assert_eq!(pandas.strip_library_schemas(), underlying);

        let optional = Type::optional(
            schema(
                vec![col("a", PolarsDType::Int64)],
                SchemaCompleteness::Complete,
            )
            .to_type(),
        );
        assert_eq!(
            optional.strip_library_schemas(),
            Type::optional(underlying.clone())
        );
        assert_eq!(underlying.clone().strip_library_schemas(), underlying);
    }

    #[test]
    fn traversal_preserves_underlying() {
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
