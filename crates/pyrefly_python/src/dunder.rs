/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ruff_python_ast::CmpOp;
use ruff_python_ast::name::Name;

pub const MAGIC_METHOD_NAMES: &[&str] = &[
    "__abs__",
    "__add__",
    "__aenter__",
    "__aexit__",
    "__aiter__",
    "__anext__",
    "__and__",
    "__ascii__",
    "__await__",
    "__bool__",
    "__bytes__",
    "__call__",
    "__ceil__",
    "__class_getitem__",
    "__complex__",
    "__contains__",
    "__copy__",
    "__deepcopy__",
    "__del__",
    "__delete__",
    "__delattr__",
    "__delitem__",
    "__dir__",
    "__divmod__",
    "__enter__",
    "__eq__",
    "__exit__",
    "__float__",
    "__floor__",
    "__floordiv__",
    "__format__",
    "__ge__",
    "__get__",
    "__getattr__",
    "__getattribute__",
    "__getitem__",
    "__getinitargs__",
    "__getnewargs__",
    "__getnewargs_ex__",
    "__getstate__",
    "__gt__",
    "__hash__",
    "__iadd__",
    "__iand__",
    "__ifloordiv__",
    "__ilshift__",
    "__imatmul__",
    "__imod__",
    "__imul__",
    "__index__",
    "__init__",
    "__init_subclass__",
    "__instancecheck__",
    "__int__",
    "__invert__",
    "__ior__",
    "__ipow__",
    "__irshift__",
    "__isub__",
    "__iter__",
    "__itruediv__",
    "__ixor__",
    "__le__",
    "__len__",
    "__length_hint__",
    "__lshift__",
    "__lt__",
    "__matmul__",
    "__missing__",
    "__mod__",
    "__mul__",
    "__ne__",
    "__neg__",
    "__new__",
    "__next__",
    "__or__",
    "__pos__",
    "__pow__",
    "__prepare__",
    "__radd__",
    "__rand__",
    "__rdivmod__",
    "__reduce__",
    "__reduce_ex__",
    "__repr__",
    "__reversed__",
    "__rfloordiv__",
    "__rlshift__",
    "__rmatmul__",
    "__rmod__",
    "__rmul__",
    "__ror__",
    "__round__",
    "__rpow__",
    "__rrshift__",
    "__rshift__",
    "__rsub__",
    "__rtruediv__",
    "__rxor__",
    "__set__",
    "__set_name__",
    "__setattr__",
    "__setitem__",
    "__setstate__",
    "__sizeof__",
    "__slots__",
    "__str__",
    "__sub__",
    "__subclasscheck__",
    "__subclasshook__",
    "__truediv__",
    "__trunc__",
    "__xor__",
];

pub const AENTER: Name = Name::new_static("__aenter__");
pub const AEXIT: Name = Name::new_static("__aexit__");
pub const AITER: Name = Name::new_static("__aiter__");
pub const ANEXT: Name = Name::new_static("__anext__");
pub const ALL: Name = Name::new_static("__all__");
pub const ASCII: Name = Name::new_static("__ascii__");
pub const BOOL: Name = Name::new_static("__bool__");
pub const CALL: Name = Name::new_static("__call__");
pub const CLASS_GETITEM: Name = Name::new_static("__class_getitem__");
pub const CONTAINS: Name = Name::new_static("__contains__");
pub const DATACLASS_FIELDS: Name = Name::new_static("__dataclass_fields__");
pub const DELATTR: Name = Name::new_static("__delattr__");
pub const DELITEM: Name = Name::new_static("__delitem__");
pub const DOC: Name = Name::new_static("__doc__");
pub const ENTER: Name = Name::new_static("__enter__");
pub const EQ: Name = Name::new_static("__eq__");
pub const EXIT: Name = Name::new_static("__exit__");
pub const FORMAT: Name = Name::new_static("__format__");
pub const GE: Name = Name::new_static("__ge__");
pub const GET: Name = Name::new_static("__get__");
pub const GETATTR: Name = Name::new_static("__getattr__");
pub const GETATTRIBUTE: Name = Name::new_static("__getattribute__");
pub const GETITEM: Name = Name::new_static("__getitem__");
pub const GT: Name = Name::new_static("__gt__");
pub const HASH: Name = Name::new_static("__hash__");
pub const INIT: Name = Name::new_static("__init__");
pub const INIT_SUBCLASS: Name = Name::new_static("__init_subclass__");
pub const INVERT: Name = Name::new_static("__invert__");
pub const ITER: Name = Name::new_static("__iter__");
pub const LE: Name = Name::new_static("__le__");
pub const LT: Name = Name::new_static("__lt__");
pub const MATCH_ARGS: Name = Name::new_static("__match_args__");
pub const NE: Name = Name::new_static("__ne__");
pub const NEG: Name = Name::new_static("__neg__");
pub const NEW: Name = Name::new_static("__new__");
pub const NEXT: Name = Name::new_static("__next__");
pub const POS: Name = Name::new_static("__pos__");
pub const POST_INIT: Name = Name::new_static("__post_init__");
pub const REPLACE: Name = Name::new_static("__replace__");
pub const REPR: Name = Name::new_static("__repr__");
pub const SET: Name = Name::new_static("__set__");
pub const SETATTR: Name = Name::new_static("__setattr__");
pub const SETITEM: Name = Name::new_static("__setitem__");
pub const SLOTS: Name = Name::new_static("__slots__");
pub const STR: Name = Name::new_static("__str__");

pub const RICH_CMPS: &[Name] = &[LT, LE, EQ, NE, GT, GE];
/// Rich comparison methods supplied by the `functools.total_ordering` decorator
pub const RICH_CMPS_TOTAL_ORDERING: &[Name] = &[LT, LE, GT, GE];

/// Returns the associated dunder if `op` corresponds to a "rich comparison method":
/// https://docs.python.org/3/reference/datamodel.html#object.__lt__.
pub fn rich_comparison_dunder(op: CmpOp) -> Option<Name> {
    let name = match op {
        CmpOp::Lt => LT,
        CmpOp::LtE => LE,
        CmpOp::Eq => EQ,
        CmpOp::NotEq => NE,
        CmpOp::Gt => GT,
        CmpOp::GtE => GE,
        _ => return None,
    };
    Some(name)
}

/// Returns the fallback dunder if `op` corresponds to a "rich comparison method":
/// https://docs.python.org/3/reference/datamodel.html#object.__lt__.
pub fn rich_comparison_fallback(op: CmpOp) -> Option<Name> {
    let name = match op {
        CmpOp::Lt => GT,
        CmpOp::LtE => GE,
        CmpOp::Eq => NE,
        CmpOp::NotEq => EQ,
        CmpOp::Gt => LT,
        CmpOp::GtE => LE,
        _ => return None,
    };
    Some(name)
}
