/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Polars dtypes and their runtime supertype relation.

use std::fmt;

use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use ruff_python_ast::name::Name;
use vec1::Vec1;

use crate::literal::write_escaped_string;

/// A non-recursive Polars dtype used as a leaf in composite column dtypes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum PolarsScalarDType {
    Boolean,
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    UInt128,
    Float32,
    Float64,
    String,
    Binary,
    Date,
    Datetime,
    Duration,
    Time,
    Null,
    Object,
}

impl PolarsScalarDType {
    const POLARS_NAMED_DTYPES: [Self; 20] = [
        Self::Boolean,
        Self::Int8,
        Self::Int16,
        Self::Int32,
        Self::Int64,
        Self::Int128,
        Self::UInt8,
        Self::UInt16,
        Self::UInt32,
        Self::UInt64,
        Self::UInt128,
        Self::Float32,
        Self::Float64,
        Self::String,
        Self::Binary,
        Self::Date,
        Self::Datetime,
        Self::Duration,
        Self::Time,
        Self::Object,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Boolean => "Boolean",
            Self::Int8 => "Int8",
            Self::Int16 => "Int16",
            Self::Int32 => "Int32",
            Self::Int64 => "Int64",
            Self::Int128 => "Int128",
            Self::UInt8 => "UInt8",
            Self::UInt16 => "UInt16",
            Self::UInt32 => "UInt32",
            Self::UInt64 => "UInt64",
            Self::UInt128 => "UInt128",
            Self::Float32 => "Float32",
            Self::Float64 => "Float64",
            Self::String => "String",
            Self::Binary => "Binary",
            Self::Date => "Date",
            Self::Datetime => "Datetime",
            Self::Duration => "Duration",
            Self::Time => "Time",
            Self::Null => "Null",
            Self::Object => "Object",
        }
    }

    fn signed_width(self) -> Option<u16> {
        match self {
            Self::Int8 => Some(8),
            Self::Int16 => Some(16),
            Self::Int32 => Some(32),
            Self::Int64 => Some(64),
            Self::Int128 => Some(128),
            _ => None,
        }
    }

    fn unsigned_width(self) -> Option<u16> {
        match self {
            Self::UInt8 => Some(8),
            Self::UInt16 => Some(16),
            Self::UInt32 => Some(32),
            Self::UInt64 => Some(64),
            Self::UInt128 => Some(128),
            _ => None,
        }
    }

    fn signed_of_width(width: u16) -> Self {
        match width {
            8 => Self::Int8,
            16 => Self::Int16,
            32 => Self::Int32,
            64 => Self::Int64,
            128 => Self::Int128,
            other => unreachable!("unexpected signed int width {other}"),
        }
    }

    fn is_float(self) -> bool {
        matches!(self, Self::Float32 | Self::Float64)
    }

    fn is_numeric(self) -> bool {
        self.signed_width().is_some() || self.unsigned_width().is_some() || self.is_float()
    }

    fn supertype(self, other: Self) -> Option<Self> {
        use PolarsScalarDType::*;

        if self == other {
            return Some(self);
        }
        if self == Null {
            return Some(other);
        }
        if other == Null {
            return Some(self);
        }
        if self == Boolean && other.is_numeric() {
            return Some(other);
        }
        if other == Boolean && self.is_numeric() {
            return Some(self);
        }
        if !self.is_numeric() || !other.is_numeric() {
            return None;
        }
        if self.is_float() || other.is_float() {
            return Some(Self::float_supertype(self, other));
        }
        if let (Some(a), Some(b)) = (self.signed_width(), other.signed_width()) {
            return Some(Self::signed_of_width(a.max(b)));
        }
        if let (Some(a), Some(b)) = (self.unsigned_width(), other.unsigned_width()) {
            return Some(match a.max(b) {
                8 => UInt8,
                16 => UInt16,
                32 => UInt32,
                64 => UInt64,
                128 => UInt128,
                other => unreachable!("unexpected unsigned int width {other}"),
            });
        }

        let signed_unsigned = match (self.signed_width(), other.unsigned_width()) {
            (Some(s), Some(u)) => Some((s, u)),
            _ => match (other.signed_width(), self.unsigned_width()) {
                (Some(s), Some(u)) => Some((s, u)),
                _ => None,
            },
        };
        if let Some((s, u)) = signed_unsigned {
            return Some(match u {
                8 | 16 | 32 => Self::signed_of_width(s.max(u * 2)),
                64 if s < 128 => Float64,
                64 | 128 => Int128,
                other => unreachable!("unexpected unsigned int width {other}"),
            });
        }
        None
    }

    fn float_supertype(a: Self, b: Self) -> Self {
        use PolarsScalarDType::*;

        if a == Float64 || b == Float64 {
            return Float64;
        }
        let other = if a == Float32 { b } else { a };
        match other {
            Float32 | Boolean | Int8 | Int16 | UInt8 | UInt16 => Float32,
            _ => Float64,
        }
    }
}

/// The dimensions of a Polars array dtype.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum PolarsArrayShape {
    /// Dimensions known at type-checking time.
    Known(Vec1<usize>),
    /// Dimensions that cannot be determined statically.
    Unknown,
}

/// A column dtype. `Unknown` represents a column whose dtype cannot be determined.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[derive(TypeEq)]
pub enum PolarsDType {
    Scalar(PolarsScalarDType),
    List(Box<PolarsDType>),
    Array {
        element: Box<PolarsDType>,
        shape: PolarsArrayShape,
    },
    Struct(Vec<(Name, PolarsDType)>),
    Unknown,
}

// Dtypes recursively contain other dtypes, but never pyrefly `Type` values.
impl<To> pyrefly_util::visit::Visit<To> for PolarsDType {
    const RECURSE_CONTAINS: bool = false;

    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a To)) {}
}

impl<To> pyrefly_util::visit::VisitMut<To> for PolarsDType {
    const RECURSE_CONTAINS: bool = false;

    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut To)) {}
}

#[expect(
    non_upper_case_globals,
    reason = "associated constants mirror Polars dtype names"
)]
impl PolarsDType {
    pub const Boolean: Self = Self::Scalar(PolarsScalarDType::Boolean);
    pub const Int8: Self = Self::Scalar(PolarsScalarDType::Int8);
    pub const Int16: Self = Self::Scalar(PolarsScalarDType::Int16);
    pub const Int32: Self = Self::Scalar(PolarsScalarDType::Int32);
    pub const Int64: Self = Self::Scalar(PolarsScalarDType::Int64);
    pub const Int128: Self = Self::Scalar(PolarsScalarDType::Int128);
    pub const UInt8: Self = Self::Scalar(PolarsScalarDType::UInt8);
    pub const UInt16: Self = Self::Scalar(PolarsScalarDType::UInt16);
    pub const UInt32: Self = Self::Scalar(PolarsScalarDType::UInt32);
    pub const UInt64: Self = Self::Scalar(PolarsScalarDType::UInt64);
    pub const UInt128: Self = Self::Scalar(PolarsScalarDType::UInt128);
    pub const Float32: Self = Self::Scalar(PolarsScalarDType::Float32);
    pub const Float64: Self = Self::Scalar(PolarsScalarDType::Float64);
    pub const String: Self = Self::Scalar(PolarsScalarDType::String);
    pub const Binary: Self = Self::Scalar(PolarsScalarDType::Binary);
    pub const Date: Self = Self::Scalar(PolarsScalarDType::Date);
    pub const Datetime: Self = Self::Scalar(PolarsScalarDType::Datetime);
    pub const Duration: Self = Self::Scalar(PolarsScalarDType::Duration);
    pub const Time: Self = Self::Scalar(PolarsScalarDType::Time);
    pub const Null: Self = Self::Scalar(PolarsScalarDType::Null);
    pub const Object: Self = Self::Scalar(PolarsScalarDType::Object);

    pub fn from_polars_name(name: &str) -> Option<Self> {
        PolarsScalarDType::POLARS_NAMED_DTYPES
            .into_iter()
            .find(|dtype| dtype.name() == name)
            .map(Self::Scalar)
    }

    fn scalar(&self) -> Option<PolarsScalarDType> {
        match self {
            Self::Scalar(dtype) => Some(*dtype),
            _ => None,
        }
    }

    pub fn is_float(&self) -> bool {
        self.scalar().is_some_and(PolarsScalarDType::is_float)
    }

    pub fn is_integer(&self) -> bool {
        self.scalar()
            .is_some_and(|dtype| dtype.signed_width().is_some() || dtype.unsigned_width().is_some())
    }

    pub fn is_signed_int(&self) -> bool {
        self.scalar()
            .is_some_and(|dtype| dtype.signed_width().is_some())
    }

    /// `UInt128` is capped at `i128::MAX`, which still contains every representable literal.
    pub fn int_bounds(&self) -> Option<(i128, i128)> {
        use PolarsScalarDType::*;

        Some(match self.scalar()? {
            Int8 => (i8::MIN as i128, i8::MAX as i128),
            Int16 => (i16::MIN as i128, i16::MAX as i128),
            Int32 => (i32::MIN as i128, i32::MAX as i128),
            Int64 => (i64::MIN as i128, i64::MAX as i128),
            Int128 => (i128::MIN, i128::MAX),
            UInt8 => (0, u8::MAX as i128),
            UInt16 => (0, u16::MAX as i128),
            UInt32 => (0, u32::MAX as i128),
            UInt64 => (0, u64::MAX as i128),
            UInt128 => (0, i128::MAX),
            _ => return None,
        })
    }

    /// The modeled subset of `polars-core`'s `get_supertype` relation.
    pub fn supertype(self, other: Self) -> Option<Self> {
        if self == other {
            return Some(self);
        }
        if self == Self::Unknown || other == Self::Unknown {
            return Some(Self::Unknown);
        }
        if self == Self::Null {
            return Some(other);
        }
        if other == Self::Null {
            return Some(self);
        }

        match (self, other) {
            (Self::Scalar(left), Self::Scalar(right)) => left.supertype(right).map(Self::Scalar),
            _ => None,
        }
    }

    pub fn is_numeric(&self) -> bool {
        self.scalar().is_some_and(PolarsScalarDType::is_numeric)
    }
}

impl fmt::Display for PolarsDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar(dtype) => f.write_str(dtype.name()),
            Self::List(element) => write!(f, "List({element})"),
            Self::Array { element, shape } => {
                write!(f, "Array({element}, shape=")?;
                match shape {
                    PolarsArrayShape::Known(dimensions) => {
                        f.write_str("(")?;
                        for (index, dimension) in dimensions.iter().enumerate() {
                            if index > 0 {
                                f.write_str(", ")?;
                            }
                            write!(f, "{dimension}")?;
                        }
                        if dimensions.len() == 1 {
                            f.write_str(",")?;
                        }
                        f.write_str(")")?;
                    }
                    PolarsArrayShape::Unknown => f.write_str("?")?,
                }
                f.write_str(")")
            }
            Self::Struct(fields) => {
                f.write_str("Struct({")?;
                for (index, (name, dtype)) in fields.iter().enumerate() {
                    if index > 0 {
                        f.write_str(", ")?;
                    }
                    write_escaped_string(name.as_str(), f, true)?;
                    write!(f, ": {dtype}")?;
                }
                f.write_str("})")
            }
            Self::Unknown => f.write_str("Unknown"),
        }
    }
}

#[cfg(test)]
mod tests {
    use ruff_python_ast::name::Name;
    use vec1::vec1;

    use super::PolarsArrayShape;
    use super::PolarsDType;

    #[test]
    fn display_uses_polars_names() {
        assert_eq!(PolarsDType::Int64.to_string(), "Int64");
        assert_eq!(
            PolarsDType::List(Box::new(PolarsDType::Int64)).to_string(),
            "List(Int64)"
        );
        assert_eq!(
            PolarsDType::Array {
                element: Box::new(PolarsDType::String),
                shape: PolarsArrayShape::Known(vec1![2]),
            }
            .to_string(),
            "Array(String, shape=(2,))"
        );
        assert_eq!(
            PolarsDType::Struct(vec![
                (Name::new("id"), PolarsDType::Int64),
                (
                    Name::new("tags"),
                    PolarsDType::List(Box::new(PolarsDType::String)),
                ),
            ])
            .to_string(),
            "Struct({'id': Int64, 'tags': List(String)})"
        );
        assert_eq!(
            PolarsDType::Struct(vec![(Name::new("quote'and\nnewline"), PolarsDType::String)])
                .to_string(),
            "Struct({'quote\\'and\\nnewline': String})"
        );
    }

    #[test]
    fn polars_names_round_trip() {
        for scalar in super::PolarsScalarDType::POLARS_NAMED_DTYPES {
            let dtype = PolarsDType::Scalar(scalar);
            assert_eq!(PolarsDType::from_polars_name(scalar.name()), Some(dtype));
        }
        assert_eq!(PolarsDType::from_polars_name("Null"), None);
        assert_eq!(PolarsDType::from_polars_name("Unknown"), None);
        assert_eq!(PolarsDType::from_polars_name("not-a-dtype"), None);
    }

    #[test]
    fn supertype_is_reflexive() {
        for dtype in [
            PolarsDType::Int8,
            PolarsDType::Int64,
            PolarsDType::UInt32,
            PolarsDType::Float32,
            PolarsDType::Float64,
            PolarsDType::Boolean,
            PolarsDType::String,
            PolarsDType::Date,
        ] {
            assert_eq!(dtype.clone().supertype(dtype.clone()), Some(dtype));
        }
    }

    #[test]
    fn supertype_signed_widens_to_larger() {
        assert_eq!(
            PolarsDType::Int8.supertype(PolarsDType::Int16),
            Some(PolarsDType::Int16)
        );
        assert_eq!(
            PolarsDType::Int8.supertype(PolarsDType::Int32),
            Some(PolarsDType::Int32)
        );
        assert_eq!(
            PolarsDType::Int8.supertype(PolarsDType::Int64),
            Some(PolarsDType::Int64)
        );
        assert_eq!(
            PolarsDType::Int64.supertype(PolarsDType::Int8),
            Some(PolarsDType::Int64)
        );
    }

    #[test]
    fn supertype_unsigned_widens_to_larger() {
        assert_eq!(
            PolarsDType::UInt8.supertype(PolarsDType::UInt32),
            Some(PolarsDType::UInt32)
        );
        assert_eq!(
            PolarsDType::UInt64.supertype(PolarsDType::UInt16),
            Some(PolarsDType::UInt64)
        );
    }

    #[test]
    fn supertype_signed_and_unsigned_promote_to_next_signed() {
        assert_eq!(
            PolarsDType::Int8.supertype(PolarsDType::UInt8),
            Some(PolarsDType::Int16)
        );
        assert_eq!(
            PolarsDType::Int8.supertype(PolarsDType::UInt16),
            Some(PolarsDType::Int32)
        );
        assert_eq!(
            PolarsDType::Int8.supertype(PolarsDType::UInt32),
            Some(PolarsDType::Int64)
        );
        assert_eq!(
            PolarsDType::Int64.supertype(PolarsDType::UInt8),
            Some(PolarsDType::Int64)
        );
        assert_eq!(
            PolarsDType::Int16.supertype(PolarsDType::UInt16),
            Some(PolarsDType::Int32)
        );
    }

    #[test]
    fn supertype_uint64_promotes_to_float64_below_int128() {
        assert_eq!(
            PolarsDType::Int8.supertype(PolarsDType::UInt64),
            Some(PolarsDType::Float64)
        );
        assert_eq!(
            PolarsDType::Int64.supertype(PolarsDType::UInt64),
            Some(PolarsDType::Float64)
        );
    }

    #[test]
    fn supertype_int128_holds_wider_unsigned() {
        assert_eq!(
            PolarsDType::Int128.supertype(PolarsDType::UInt64),
            Some(PolarsDType::Int128)
        );
        assert_eq!(
            PolarsDType::Int128.supertype(PolarsDType::UInt32),
            Some(PolarsDType::Int128)
        );
        assert_eq!(
            PolarsDType::Int8.supertype(PolarsDType::UInt128),
            Some(PolarsDType::Int128)
        );
        assert_eq!(
            PolarsDType::Int64.supertype(PolarsDType::UInt128),
            Some(PolarsDType::Int128)
        );
        assert_eq!(
            PolarsDType::Int128.supertype(PolarsDType::UInt128),
            Some(PolarsDType::Int128)
        );
    }

    #[test]
    fn supertype_int_and_float() {
        assert_eq!(
            PolarsDType::Int8.supertype(PolarsDType::Float32),
            Some(PolarsDType::Float32)
        );
        assert_eq!(
            PolarsDType::Int16.supertype(PolarsDType::Float32),
            Some(PolarsDType::Float32)
        );
        assert_eq!(
            PolarsDType::Int32.supertype(PolarsDType::Float32),
            Some(PolarsDType::Float64)
        );
        assert_eq!(
            PolarsDType::Int64.supertype(PolarsDType::Float32),
            Some(PolarsDType::Float64)
        );
        assert_eq!(
            PolarsDType::Int8.supertype(PolarsDType::Float64),
            Some(PolarsDType::Float64)
        );
        assert_eq!(
            PolarsDType::Float32.supertype(PolarsDType::Float64),
            Some(PolarsDType::Float64)
        );
    }

    #[test]
    fn supertype_bool_joins_numeric_tower() {
        assert_eq!(
            PolarsDType::Boolean.supertype(PolarsDType::Int32),
            Some(PolarsDType::Int32)
        );
        assert_eq!(
            PolarsDType::Boolean.supertype(PolarsDType::Float64),
            Some(PolarsDType::Float64)
        );
        assert_eq!(
            PolarsDType::Int8.supertype(PolarsDType::Boolean),
            Some(PolarsDType::Int8)
        );
    }

    #[test]
    fn supertype_null_takes_the_other() {
        assert_eq!(
            PolarsDType::Null.supertype(PolarsDType::Int32),
            Some(PolarsDType::Int32)
        );
        assert_eq!(
            PolarsDType::String.supertype(PolarsDType::Null),
            Some(PolarsDType::String)
        );
    }

    #[test]
    fn supertype_unrelated_is_none() {
        for (left, right) in [
            (PolarsDType::Int64, PolarsDType::String),
            (PolarsDType::String, PolarsDType::Binary),
            (PolarsDType::Date, PolarsDType::Int64),
            (PolarsDType::Float32, PolarsDType::String),
            (PolarsDType::Float64, PolarsDType::Date),
            (PolarsDType::Float32, PolarsDType::Binary),
            (PolarsDType::Boolean, PolarsDType::String),
        ] {
            assert_eq!(left.supertype(right), None);
        }
    }
}
