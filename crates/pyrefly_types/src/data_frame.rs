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
    /// Columns in definition order.
    pub columns: Vec<(Name, Type)>,
    pub completeness: SchemaCompleteness,
}

impl DataFrameSchema {
    pub fn to_type(self) -> Type {
        Type::DataFrame(Box::new(self))
    }

    /// The underlying instance as a `Type`, for delegating behavior to it.
    pub fn underlying_type(&self) -> Type {
        Type::ClassType(self.underlying.clone())
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
    use starlark_map::small_map::SmallMap;

    use super::*;
    use crate::class::Class;
    use crate::class::ClassDefIndex;
    use crate::class::ClassType;
    use crate::equality::TypeEq;
    use crate::equality::TypeEqCtx;
    use crate::quantified::AnchorIndex;
    use crate::quantified::Quantified;
    use crate::quantified::QuantifiedIdentity;
    use crate::quantified::QuantifiedKind;
    use crate::quantified::QuantifiedOrigin;
    use crate::type_var::PreInferenceVariance;
    use crate::type_var::Restriction;
    use crate::types::TArgs;

    fn class_type(module: &str, name: &str) -> ClassType {
        let module = Module::new(
            ModuleName::from_str(module),
            ModulePath::filesystem(PathBuf::from(module)),
            Arc::new("fake module contents".to_owned()),
        );
        ClassType::new(
            Class::new(
                ClassDefIndex(0),
                Identifier::new(Name::new(name), TextRange::empty(TextSize::new(0))),
                NestingContext::toplevel(),
                module,
                None,
            ),
            TArgs::default(),
        )
    }

    fn class_ty(module: &str, name: &str) -> Type {
        Type::ClassType(class_type(module, name))
    }

    fn underlying_class() -> ClassType {
        class_type("polars", "DataFrame")
    }

    fn quantified(name: &str) -> Quantified {
        Quantified::new(
            QuantifiedIdentity::new(
                ModuleName::from_str("__test__"),
                AnchorIndex::first(TextRange::default()),
                QuantifiedOrigin::Pep695,
            ),
            Name::new(name),
            QuantifiedKind::TypeVar,
            None,
            Restriction::Unrestricted,
            PreInferenceVariance::Invariant,
        )
    }

    fn col(name: &str, ty: Type) -> (Name, Type) {
        (Name::new(name), ty)
    }

    fn schema(columns: Vec<(Name, Type)>, completeness: SchemaCompleteness) -> DataFrameSchema {
        DataFrameSchema {
            underlying: underlying_class(),
            columns,
            completeness,
        }
    }

    fn hash_of(schema: &DataFrameSchema) -> u64 {
        let mut hasher = DefaultHasher::new();
        schema.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn partial_schema_display_delegates_to_underlying() {
        let df = schema(
            vec![col("a", class_ty("builtins", "int"))],
            SchemaCompleteness::Partial,
        )
        .to_type();
        let shown = format!("{df}");
        assert!(
            !shown.contains('['),
            "a Partial DataFrame renders as its underlying instance with no column list, got `{shown}`"
        );
    }

    #[test]
    fn column_order_is_part_of_identity() {
        let ab = schema(
            vec![
                col("a", class_ty("builtins", "int")),
                col("b", class_ty("builtins", "str")),
            ],
            SchemaCompleteness::Complete,
        );
        let ba = schema(
            vec![
                col("b", class_ty("builtins", "str")),
                col("a", class_ty("builtins", "int")),
            ],
            SchemaCompleteness::Complete,
        );

        // Reordered columns are a distinct type under every relation.
        assert_ne!(ab, ba);
        assert_ne!(hash_of(&ab), hash_of(&ba));
        assert_ne!(ab.cmp(&ba), Ordering::Equal);
        assert!(!ab.type_eq(&ba, &mut TypeEqCtx::default()));

        // Identical columns in the same order are equal under every relation.
        let ab2 = schema(
            vec![
                col("a", class_ty("builtins", "int")),
                col("b", class_ty("builtins", "str")),
            ],
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
            vec![col("a", class_ty("builtins", "int"))],
            SchemaCompleteness::Complete,
        );
        let partial = schema(
            vec![col("a", class_ty("builtins", "int"))],
            SchemaCompleteness::Partial,
        );
        assert_ne!(complete, partial);
        assert!(!complete.type_eq(&partial, &mut TypeEqCtx::default()));
    }

    #[test]
    fn traversal_reaches_underlying_and_column_types() {
        let q = quantified("T");
        let df = schema(
            vec![col("a", Type::Quantified(Box::new(q.clone())))],
            SchemaCompleteness::Complete,
        )
        .to_type();

        // forall: the quantified inside a column type is discoverable.
        let mut found_quantified = false;
        df.for_each_quantified(&mut |x| found_quantified |= *x == q);
        assert!(
            found_quantified,
            "for_each_quantified should reach column types"
        );

        // subst (rides on visit_mut): the column quantified is rewritten, the
        // underlying class is preserved.
        let replacement = class_ty("builtins", "int");
        let mut mp = SmallMap::new();
        mp.insert(&q, &replacement);
        let Type::DataFrame(substituted) = df.subst(&mp) else {
            unreachable!("subst preserves the DataFrame variant")
        };
        assert_eq!(substituted.columns[0].1, replacement);
        assert_eq!(substituted.underlying, underlying_class());
    }
}
