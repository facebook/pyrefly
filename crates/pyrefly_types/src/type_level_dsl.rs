/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;

use crate::dimension::ShapeError;
use crate::shaped_array::IntTuple;
use crate::shaped_array::broadcast_shapes;
use crate::shaped_array::tuple_carrier_to_shape;
use crate::types::Type;

/// A deferred type-level DSL invocation held until a function-call return boundary.
#[derive(Debug, Clone, PartialEq, Eq, TypeEq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut)]
pub struct TypeLevelDslCall {
    pub function: TypeLevelDslFunction,
    pub args: Vec<Type>,
    pub result_schema: TypeLevelDslResultSchema,
}

/// The identity of a type-level DSL operation.
#[derive(Debug, Clone, PartialEq, Eq, TypeEq, PartialOrd, Ord, Hash)]
pub struct TypeLevelDslFunction {
    pub name: &'static str,
}

/// The type domain produced when a DSL call must fall back to a gradual result.
#[derive(Debug, Clone, PartialEq, Eq, TypeEq, PartialOrd, Ord, Hash)]
pub enum TypeLevelDslResultSchema {
    IntTuple,
}

impl Visit<Type> for TypeLevelDslFunction {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

impl VisitMut<Type> for TypeLevelDslFunction {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}

impl Visit<Type> for TypeLevelDslResultSchema {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

impl VisitMut<Type> for TypeLevelDslResultSchema {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}

impl TypeLevelDslCall {
    /// Constructs a native two-argument broadcast call.
    pub fn broadcast(args: Vec<Type>) -> Self {
        Self {
            function: TypeLevelDslFunction { name: "broadcast" },
            args,
            result_schema: TypeLevelDslResultSchema::IntTuple,
        }
    }

    /// Returns the gradual result for a call whose precise value cannot be determined.
    pub fn fallback(&self) -> Type {
        match &self.result_schema {
            TypeLevelDslResultSchema::IntTuple => IntTuple::shapeless().to_shape_arg_type(),
        }
    }

    /// Evaluates the native call, reporting incompatible concrete shapes.
    pub fn evaluate(&self) -> Result<Type, ShapeError> {
        match self.function.name {
            "broadcast" => {
                let [left, right] = self.args.as_slice() else {
                    unreachable!("native broadcast DSL calls are constructed with two arguments");
                };
                let Some(left) =
                    IntTuple::from_shape_arg_type(left).or_else(|| tuple_carrier_to_shape(left))
                else {
                    return Ok(self.fallback());
                };
                let Some(right) =
                    IntTuple::from_shape_arg_type(right).or_else(|| tuple_carrier_to_shape(right))
                else {
                    return Ok(self.fallback());
                };
                broadcast_shapes(&left, &right).map(|shape| shape.to_shape_arg_type())
            }
            name => unreachable!("unknown type-level DSL function `{name}`"),
        }
    }
}
