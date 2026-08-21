/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Polars scalar dtypes and their runtime supertype relation.

use std::fmt;

use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;

/// A scalar column dtype. `Unknown` represents a column whose dtype cannot be determined.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum PolarsDType {
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
    Unknown,
}

impl PolarsDType {
    const POLARS_NAMED_DTYPES: [Self; 19] = [
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
    ];

    pub fn from_polars_name(name: &str) -> Option<Self> {
        Self::POLARS_NAMED_DTYPES
            .into_iter()
            .find(|dtype| dtype.name() == name)
    }

    pub fn name(&self) -> &'static str {
        match self {
            PolarsDType::Boolean => "Boolean",
            PolarsDType::Int8 => "Int8",
            PolarsDType::Int16 => "Int16",
            PolarsDType::Int32 => "Int32",
            PolarsDType::Int64 => "Int64",
            PolarsDType::Int128 => "Int128",
            PolarsDType::UInt8 => "UInt8",
            PolarsDType::UInt16 => "UInt16",
            PolarsDType::UInt32 => "UInt32",
            PolarsDType::UInt64 => "UInt64",
            PolarsDType::UInt128 => "UInt128",
            PolarsDType::Float32 => "Float32",
            PolarsDType::Float64 => "Float64",
            PolarsDType::String => "String",
            PolarsDType::Binary => "Binary",
            PolarsDType::Date => "Date",
            PolarsDType::Datetime => "Datetime",
            PolarsDType::Duration => "Duration",
            PolarsDType::Time => "Time",
            PolarsDType::Null => "Null",
            PolarsDType::Unknown => "Unknown",
        }
    }

    fn signed_width(&self) -> Option<u16> {
        match self {
            PolarsDType::Int8 => Some(8),
            PolarsDType::Int16 => Some(16),
            PolarsDType::Int32 => Some(32),
            PolarsDType::Int64 => Some(64),
            PolarsDType::Int128 => Some(128),
            _ => None,
        }
    }

    fn unsigned_width(&self) -> Option<u16> {
        match self {
            PolarsDType::UInt8 => Some(8),
            PolarsDType::UInt16 => Some(16),
            PolarsDType::UInt32 => Some(32),
            PolarsDType::UInt64 => Some(64),
            PolarsDType::UInt128 => Some(128),
            _ => None,
        }
    }

    fn signed_of_width(width: u16) -> PolarsDType {
        match width {
            8 => PolarsDType::Int8,
            16 => PolarsDType::Int16,
            32 => PolarsDType::Int32,
            64 => PolarsDType::Int64,
            128 => PolarsDType::Int128,
            other => unreachable!("unexpected signed int width {other}"),
        }
    }

    pub fn is_float(&self) -> bool {
        matches!(self, PolarsDType::Float32 | PolarsDType::Float64)
    }

    pub fn is_integer(&self) -> bool {
        self.signed_width().is_some() || self.unsigned_width().is_some()
    }

    pub fn is_signed_int(&self) -> bool {
        self.signed_width().is_some()
    }

    /// `UInt128` is capped at `i128::MAX`, which still contains every representable literal.
    pub fn int_bounds(&self) -> Option<(i128, i128)> {
        use PolarsDType::*;
        Some(match self {
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
    pub fn supertype(self, other: PolarsDType) -> Option<PolarsDType> {
        use PolarsDType::*;
        if self == other {
            return Some(self);
        }
        if self == Unknown || other == Unknown {
            return Some(Unknown);
        }
        if self == Null {
            return Some(other);
        }
        if other == Null {
            return Some(self);
        }
        if self == Boolean && (other.is_numeric()) {
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
        // Polars caps a UInt128 pair at Int128, while UInt64 promotes to Float64 without Int128.
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

    pub fn is_numeric(&self) -> bool {
        self.signed_width().is_some() || self.unsigned_width().is_some() || self.is_float()
    }

    fn float_supertype(a: PolarsDType, b: PolarsDType) -> PolarsDType {
        use PolarsDType::*;
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

impl fmt::Display for PolarsDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::PolarsDType;
    use super::PolarsDType::*;

    #[test]
    fn display_uses_polars_names() {
        assert_eq!(Int64.to_string(), "Int64");
        assert_eq!(Float64.to_string(), "Float64");
        assert_eq!(UInt8.to_string(), "UInt8");
        assert_eq!(String.to_string(), "String");
        assert_eq!(Datetime.to_string(), "Datetime");
    }

    #[test]
    fn polars_names_round_trip() {
        for dtype in PolarsDType::POLARS_NAMED_DTYPES {
            assert_eq!(PolarsDType::from_polars_name(dtype.name()), Some(dtype));
        }
        assert_eq!(PolarsDType::from_polars_name(Null.name()), None);
        assert_eq!(PolarsDType::from_polars_name(Unknown.name()), None);
        assert_eq!(PolarsDType::from_polars_name("not-a-dtype"), None);
    }

    #[test]
    fn supertype_is_reflexive() {
        for t in [Int8, Int64, UInt32, Float32, Float64, Boolean, String, Date] {
            assert_eq!(
                t.clone().supertype(t.clone()),
                Some(t.clone()),
                "{t} with itself"
            );
        }
    }

    #[test]
    fn supertype_signed_widens_to_larger() {
        assert_eq!(Int8.supertype(Int16), Some(Int16));
        assert_eq!(Int8.supertype(Int32), Some(Int32));
        assert_eq!(Int8.supertype(Int64), Some(Int64));
        assert_eq!(Int64.supertype(Int8), Some(Int64));
    }

    #[test]
    fn supertype_unsigned_widens_to_larger() {
        assert_eq!(UInt8.supertype(UInt32), Some(UInt32));
        assert_eq!(UInt64.supertype(UInt16), Some(UInt64));
    }

    #[test]
    fn supertype_signed_and_unsigned_promote_to_next_signed() {
        assert_eq!(Int8.supertype(UInt8), Some(Int16));
        assert_eq!(Int8.supertype(UInt16), Some(Int32));
        assert_eq!(Int8.supertype(UInt32), Some(Int64));
        assert_eq!(Int64.supertype(UInt8), Some(Int64));
        assert_eq!(Int16.supertype(UInt16), Some(Int32));
    }

    #[test]
    fn supertype_uint64_promotes_to_float64_below_int128() {
        assert_eq!(Int8.supertype(UInt64), Some(Float64));
        assert_eq!(Int64.supertype(UInt64), Some(Float64));
    }

    #[test]
    fn supertype_int128_holds_wider_unsigned() {
        assert_eq!(Int128.supertype(UInt64), Some(Int128));
        assert_eq!(Int128.supertype(UInt32), Some(Int128));
        assert_eq!(Int8.supertype(UInt128), Some(Int128));
        assert_eq!(Int64.supertype(UInt128), Some(Int128));
        assert_eq!(Int128.supertype(UInt128), Some(Int128));
    }

    #[test]
    fn supertype_int_and_float() {
        assert_eq!(Int8.supertype(Float32), Some(Float32));
        assert_eq!(Int16.supertype(Float32), Some(Float32));
        assert_eq!(Int32.supertype(Float32), Some(Float64));
        assert_eq!(Int64.supertype(Float32), Some(Float64));
        assert_eq!(Int8.supertype(Float64), Some(Float64));
        assert_eq!(Float32.supertype(Float64), Some(Float64));
    }

    #[test]
    fn supertype_bool_joins_numeric_tower() {
        assert_eq!(Boolean.supertype(Int32), Some(Int32));
        assert_eq!(Boolean.supertype(Float64), Some(Float64));
        assert_eq!(Int8.supertype(Boolean), Some(Int8));
    }

    #[test]
    fn supertype_null_takes_the_other() {
        assert_eq!(Null.supertype(Int32), Some(Int32));
        assert_eq!(String.supertype(Null), Some(String));
    }

    #[test]
    fn supertype_unrelated_is_none() {
        assert_eq!(Int64.supertype(String), None);
        assert_eq!(String.supertype(Binary), None);
        assert_eq!(Date.supertype(Int64), None);
        assert_eq!(Float32.supertype(String), None);
        assert_eq!(Float64.supertype(Date), None);
        assert_eq!(Float32.supertype(Binary), None);
        assert_eq!(Boolean.supertype(String), None);
    }
}
