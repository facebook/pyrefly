/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::convert::Infallible;
use std::fmt;
use std::fmt::Display;
use std::str::FromStr;
use std::sync::Arc;

use dupe::Dupe;
use itertools::Itertools;
use parse_display::Display;
use pyrefly_util::prelude::SliceExt;
use pyrefly_util::with_hash::WithHash;
use regex::Match;
use regex::Regex;
use ruff_python_ast::BoolOp;
use ruff_python_ast::CmpOp;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprBooleanLiteral;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprNumberLiteral;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtIf;
use ruff_python_ast::UnaryOp;
use serde::Deserialize;
use serde::Serialize;
use serde::de;
use serde::de::Visitor;

use crate::ast::Ast;

#[derive(Debug, Clone, Copy, Dupe, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct PythonVersion {
    pub major: u32,
    pub minor: u32,
    pub micro: u32,
}

impl Default for PythonVersion {
    fn default() -> Self {
        Self {
            major: 3,
            minor: 13,
            micro: 0,
        }
    }
}

impl FromStr for PythonVersion {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        let version_pattern = Regex::new(r"(\d+)(\.(\d+))?(\.(\d+))?").unwrap();
        let captures = version_pattern
            .captures(s)
            .ok_or_else(|| anyhow::anyhow!("Invalid version string: {s}."))?;

        fn extract_number(capture: Option<Match>, default: u32) -> anyhow::Result<u32> {
            capture.map_or(Ok(default), |capture| {
                let capture_str = capture.as_str();
                let number = capture_str
                    .parse::<u32>()
                    .map_err(|_| anyhow::anyhow!("Invalid version number {capture_str}"))?;
                Ok(number)
            })
        }

        let def = Self::default();
        let major = extract_number(captures.get(1), def.major)?;
        let minor_default = if major == def.major { def.minor } else { 0 };
        let minor = extract_number(captures.get(3), minor_default)?;
        let micro_default = if major == def.major && minor == def.minor {
            def.micro
        } else {
            0
        };
        let micro = extract_number(captures.get(5), micro_default)?;
        Ok(Self {
            major,
            minor,
            micro,
        })
    }
}

impl Display for PythonVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            major,
            minor,
            micro,
        } = self;
        write!(f, "{major}.{minor}.{micro}")
    }
}

impl TryFrom<String> for PythonVersion {
    type Error = anyhow::Error;

    fn try_from(value: String) -> anyhow::Result<Self> {
        PythonVersion::from_str(&value[..])
    }
}

impl<'de> Deserialize<'de> for PythonVersion {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct PythonVersionVisitor;

        impl<'de> Visitor<'de> for PythonVersionVisitor {
            type Value = PythonVersion;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("Python version")
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
                PythonVersion::from_str(value).map_err(|e| de::Error::custom(e.to_string()))
            }
        }

        deserializer.deserialize_string(PythonVersionVisitor)
    }
}

impl Serialize for PythonVersion {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl PythonVersion {
    pub fn new(major: u32, minor: u32, micro: u32) -> Self {
        Self {
            major,
            minor,
            micro,
        }
    }

    pub fn at_least(self, major: u32, minor: u32) -> bool {
        self >= PythonVersion::new(major, minor, 0)
    }
}

/// The platform on which Python is running, e.g. "linux", "darwin", "win32".
/// See <https://docs.python.org/3/library/sys.html#sys.platform> for examples.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Display)]
pub struct PythonPlatform(String);

impl FromStr for PythonPlatform {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::new(s))
    }
}

impl Default for PythonPlatform {
    fn default() -> Self {
        PythonPlatform::linux()
    }
}

impl PythonPlatform {
    pub fn new(platform: &str) -> Self {
        // Try and normalise common names, particularly those that Pyright allows (All, Linux, Darwin, Windows)
        match platform {
            "Linux" | "linux" | "All" => Self::linux(),
            "Darwin" | "darwin" | "mac" | "macos" => Self::mac(),
            "Windows" | "windows" | "win32" | "Win32" => Self::windows(),
            _ => Self(platform.to_owned()),
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn linux() -> Self {
        Self("linux".to_owned())
    }

    pub fn windows() -> Self {
        Self("win32".to_owned())
    }

    pub fn mac() -> Self {
        Self("darwin".to_owned())
    }
}

/// Information available from the Python library `sys`, namely
/// `version` and `platform`.
#[derive(Clone, Dupe, Debug, PartialEq, Eq, Hash, Default)]
pub struct SysInfo(Arc<WithHash<SysInfoInner>>);

#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
struct SysInfoInner {
    version: PythonVersion,
    platform: PythonPlatform,
}

impl SysInfo {
    pub fn new(version: PythonVersion, platform: PythonPlatform) -> Self {
        Self(Arc::new(WithHash::new(SysInfoInner { version, platform })))
    }

    pub fn version(&self) -> PythonVersion {
        self.0.key().version
    }

    pub fn platform(&self) -> &PythonPlatform {
        &self.0.key().platform
    }
}

#[derive(Debug, PartialEq, PartialOrd)]
enum Value {
    Tuple(Vec<Value>),
    String(String),
    Int(i64),
    Bool(bool),
    /// I know what the value evaluates to when considered truthy, but not it's precise outcome.
    /// We make sure below that it never compares equal to itself
    Truthiness(bool),
}

impl Value {
    fn to_bool(&self) -> bool {
        match self {
            Value::Bool(x) => *x,
            Value::Truthiness(x) => *x,
            Value::Int(x) => *x != 0,
            Value::String(x) => !x.is_empty(),
            Value::Tuple(x) => !x.is_empty(),
        }
    }

    fn same_type(&self, other: &Value) -> bool {
        match (self, other) {
            (Value::Tuple(_), Value::Tuple(_)) => true,
            (Value::String(_), Value::String(_)) => true,
            (Value::Int(_), Value::Int(_)) => true,
            (Value::Bool(_), Value::Bool(_)) => true,
            (Value::Truthiness(_), Value::Truthiness(_)) => false, // We don't know if they are the same ype
            _ => false,
        }
    }

    fn compare(&self, op: CmpOp, other: &Value) -> Option<bool> {
        if !self.same_type(other) {
            return None; // Someone got confused, or we are working with Truthiness
        }
        Some(match op {
            CmpOp::Eq => self == other,
            CmpOp::NotEq => self != other,
            CmpOp::Lt => self < other,
            CmpOp::LtE => self <= other,
            CmpOp::Gt => self > other,
            CmpOp::GtE => self >= other,
            _ => return None,
        })
    }
}

impl SysInfo {
    /// Return true/false if we can statically evaluate it, and None if we can't.
    pub fn evaluate_bool(&self, x: &Expr) -> Option<bool> {
        Some(self.evaluate(x)?.to_bool())
    }

    /// Version of `evaluate_bool` where `None` means no test (thus always statically true).
    pub fn evaluate_bool_opt(&self, x: Option<&Expr>) -> Option<bool> {
        match x {
            None => Some(true),
            Some(x) => self.evaluate_bool(x),
        }
    }

    fn is_type_checking_constant_name(x: &str) -> bool {
        x == "TYPE_CHECKING" || x == "TYPE_CHECKING_WITH_PYREFLY"
    }

    fn evaluate(&self, x: &Expr) -> Option<Value> {
        match x {
            Expr::Compare(x) if x.ops.len() == 1 && x.comparators.len() == 1 => Some(Value::Bool(
                self.evaluate(&x.left)?
                    .compare(x.ops[0], &self.evaluate(&x.comparators[0])?)?,
            )),
            Expr::Attribute(ExprAttribute { value, attr, .. })
                if let Expr::Name(name) = &**value
                    && &name.id == "sys" =>
            {
                match attr.as_str() {
                    "platform" => Some(Value::String(self.0.platform.as_str().to_owned())),
                    "version_info" => Some(Value::Tuple(vec![
                        Value::Int(self.0.version.major as i64),
                        Value::Int(self.0.version.minor as i64),
                    ])),
                    _ => None,
                }
            }
            Expr::Name(name) if Self::is_type_checking_constant_name(name.id()) => {
                Some(Value::Bool(true))
            }
            Expr::Attribute(ExprAttribute {
                // We support TYPE_CHECKING regardless of which import (or reimport) it is from.
                value,
                attr,
                ..
            }) if value.is_name_expr() && Self::is_type_checking_constant_name(attr.as_str()) => {
                Some(Value::Bool(true))
            }
            Expr::Call(ExprCall {
                func, arguments, ..
            }) if let Expr::Attribute(ExprAttribute { value, attr, .. }) = &**func
                && attr.as_str() == "startswith"
                && arguments.keywords.is_empty()
                && let [arg] = &*arguments.args
                && let Some(Value::String(x)) = self.evaluate(value)
                && let Some(Value::String(y)) = self.evaluate(arg) =>
            {
                Some(Value::Bool(x.starts_with(&y)))
            }
            Expr::Tuple(x) => Some(Value::Tuple(
                x.elts.try_map(|x| self.evaluate(x).ok_or(())).ok()?,
            )),
            Expr::NumberLiteral(ExprNumberLiteral { value: i, .. }) => {
                Some(Value::Int(i.as_int()?.as_i64()?))
            }
            Expr::BooleanLiteral(ExprBooleanLiteral { value: b, .. }) => Some(Value::Bool(*b)),
            Expr::StringLiteral(x) => Some(Value::String(x.value.to_str().to_owned())),
            Expr::BoolOp(x) => match x.op {
                BoolOp::And => {
                    let mut res = Some(Value::Bool(true));
                    for x in &x.values {
                        match self.evaluate(x) {
                            None => res = None,
                            Some(x) => match (x.to_bool(), res.is_none()) {
                                (false, false) => return Some(x),
                                (false, true) => return Some(Value::Truthiness(false)),
                                (true, false) => res = Some(x),
                                (true, true) => res = None,
                            },
                        }
                    }
                    res
                }
                BoolOp::Or => {
                    let mut res = Some(Value::Bool(false));
                    for x in &x.values {
                        match self.evaluate(x) {
                            None => res = None,
                            Some(x) => match (x.to_bool(), res.is_none()) {
                                (false, false) => res = Some(x),
                                (false, true) => res = None,
                                (true, false) => return Some(x),
                                (true, true) => return Some(Value::Truthiness(true)),
                            },
                        }
                    }
                    res
                }
            },
            Expr::UnaryOp(x) => match x.op {
                UnaryOp::Not => {
                    let v = self.evaluate(&x.operand)?;
                    Some(Value::Bool(!v.to_bool()))
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// Like `Ast::if_branches`, but skips branch that statically evaluate to `false`,
    /// and stops if any branch evaluates to `true`.
    pub fn pruned_if_branches<'a, 'b: 'a>(
        &'a self,
        x: &'b StmtIf,
    ) -> impl Iterator<Item = (Option<&'b Expr>, &'b [Stmt])> + 'a {
        Ast::if_branches(x)
            .map(|(test, body)| {
                let b = self.evaluate_bool_opt(test);
                (b, if b == Some(true) { None } else { test }, body)
            })
            .filter(|x| x.0 != Some(false))
            .map(|x| (x.1, x.2))
            .take_while_inclusive(|x| x.0.is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_py_version() {
        assert_eq!(
            PythonVersion::from_str("3").unwrap(),
            PythonVersion::new(3, 13, 0)
        );
        assert_eq!(
            PythonVersion::from_str("3.8").unwrap(),
            PythonVersion::new(3, 8, 0)
        );
        assert_eq!(
            PythonVersion::from_str("3.8.6").unwrap(),
            PythonVersion::new(3, 8, 6)
        );
        assert_eq!(
            PythonVersion::from_str("3.10.2").unwrap(),
            PythonVersion::new(3, 10, 2)
        );
        assert_eq!(
            PythonVersion::from_str("4").unwrap(),
            PythonVersion::new(4, 0, 0)
        );
        assert_eq!(
            PythonVersion::from_str("4.14").unwrap(),
            PythonVersion::new(4, 14, 0)
        );
        assert_eq!(
            PythonVersion::from_str("python3.10").unwrap(),
            PythonVersion::new(3, 10, 0)
        );
        assert_eq!(
            PythonVersion::from_str("cinder.3.8").unwrap(),
            PythonVersion::new(3, 8, 0)
        );
        assert_eq!(
            PythonVersion::from_str("3.10.cinder").unwrap(),
            PythonVersion::new(3, 10, 0)
        );
        assert!(PythonVersion::from_str("").is_err());
        assert!(PythonVersion::from_str("abc").is_err());
    }

    #[test]
    fn test_version_string_deserializing() {
        #[derive(Debug, Deserialize, PartialEq, Eq)]
        struct Output {
            version: PythonVersion,
        }
        assert_eq!(
            toml::from_str::<Output>("version = '3'").expect("Failed to parse"),
            Output {
                version: PythonVersion::new(3, 13, 0)
            }
        );
        assert_eq!(
            toml::from_str::<Output>("version = '3.10'").expect("Failed to parse"),
            Output {
                version: PythonVersion::new(3, 10, 0)
            }
        );
        assert_eq!(
            toml::from_str::<Output>("version = '3.10.2'").expect("Failed to parse"),
            Output {
                version: PythonVersion::new(3, 10, 2)
            }
        );
        assert!(toml::from_str::<Output>("version = 'abcd'").is_err());
        assert!(toml::from_str::<Output>("version = '5000000000'").is_err());
    }

    #[test]
    fn test_tuple_lexicographical_compare() {
        fn assert_compare(op: CmpOp, x: &[i64], y: &[i64]) {
            let lhs = Value::Tuple(x.map(|x| Value::Int(*x)));
            let rhs = Value::Tuple(y.map(|x| Value::Int(*x)));
            assert_eq!(lhs.compare(op, &rhs), Some(true));
        }

        assert_compare(CmpOp::Eq, &[], &[]);
        assert_compare(CmpOp::Lt, &[], &[1]);
        assert_compare(CmpOp::Eq, &[1], &[1]);
        assert_compare(CmpOp::Lt, &[1], &[1, 2]);
        assert_compare(CmpOp::Lt, &[1], &[2]);
        assert_compare(CmpOp::Lt, &[1], &[2, 3]);
        assert_compare(CmpOp::Lt, &[1, 2], &[1, 2, 3]);
        assert_compare(CmpOp::Gt, &[1, 3], &[1, 2, 3]);
    }
}
