/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::Ordering;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::Expr;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::dimension::Int;
use crate::dimension::ShapeError;
use crate::dimension::gradual_size;
use crate::equality::TypeEq as TypeEqTrait;
use crate::equality::TypeEqCtx;
use crate::shaped_array::IntTuple;
use crate::shaped_array::broadcast_shapes;
use crate::shaped_array::tuple_carrier_to_shape;
use crate::types::Type;

#[derive(Debug, Clone, Copy, PartialEq, Eq, TypeEq, PartialOrd, Ord, Hash)]
pub enum TypeShapeDslDomain {
    Int,
    IntTuple,
}

impl Visit<Type> for TypeShapeDslDomain {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

impl VisitMut<Type> for TypeShapeDslDomain {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}

#[derive(Debug, Clone)]
pub struct TypeShapeDslDefinitionError {
    pub range: TextRange,
    pub message: &'static str,
}

/// An owned function AST whose restricted declaration syntax has been validated.
/// Future evaluation may interpret `definition` relying on these invariants.
#[derive(Debug, Clone)]
pub struct ValidatedTypeShapeDslFunction {
    definition: Arc<StmtFunctionDef>,
    parameter_name: Name,
}

// The AST is executable program state, not a derived cache, so its identity must participate in
// incremental equality. Aliases within one module generation share this `Arc`; reparsing an edited
// definition creates a new allocation and invalidates every dependent call result. In particular,
// this must not be wrapped in `IdentityIgnored` like the derived V1 helper-closure cache.
impl PartialEq for ValidatedTypeShapeDslFunction {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.definition, &other.definition)
    }
}

impl Eq for ValidatedTypeShapeDslFunction {}

impl Hash for ValidatedTypeShapeDslFunction {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.definition) as *const () as usize).hash(state);
    }
}

// This ordering is a process-local tie-breaker required by type nodes that derive `Ord`; it must
// not be used for stable output. Comparing the same identity as equality keeps `cmp` consistent
// with the pointer-based `Eq` above while distinguishing reparsed definitions.
impl PartialOrd for ValidatedTypeShapeDslFunction {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ValidatedTypeShapeDslFunction {
    fn cmp(&self, other: &Self) -> Ordering {
        let self_ptr = Arc::as_ptr(&self.definition) as *const () as usize;
        let other_ptr = Arc::as_ptr(&other.definition) as *const () as usize;
        self_ptr.cmp(&other_ptr)
    }
}

impl Visit<Type> for ValidatedTypeShapeDslFunction {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

impl VisitMut<Type> for ValidatedTypeShapeDslFunction {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}

impl Visit<Type> for Arc<ValidatedTypeShapeDslFunction> {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

impl VisitMut<Type> for Arc<ValidatedTypeShapeDslFunction> {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}

impl TypeEqTrait for ValidatedTypeShapeDslFunction {
    fn type_eq(&self, other: &Self, _ctx: &mut TypeEqCtx) -> bool {
        self == other
    }
}

impl ValidatedTypeShapeDslFunction {
    pub fn try_new(
        definition: StmtFunctionDef,
        is_top_level: bool,
    ) -> Result<Self, TypeShapeDslDefinitionError> {
        if !is_top_level {
            return Err(TypeShapeDslDefinitionError {
                range: definition.name.range(),
                message: "must decorate a top-level function",
            });
        }
        if definition.is_async {
            return Err(TypeShapeDslDefinitionError {
                range: definition.name.range(),
                message: "does not support async functions",
            });
        }
        if definition.type_params.is_some() {
            return Err(TypeShapeDslDefinitionError {
                range: definition.name.range(),
                message: "does not support type parameters",
            });
        }
        let parameters = &definition.parameters;
        if !parameters.posonlyargs.is_empty()
            || parameters.args.len() != 1
            || !parameters.kwonlyargs.is_empty()
            || parameters.vararg.is_some()
            || parameters.kwarg.is_some()
        {
            return Err(TypeShapeDslDefinitionError {
                range: parameters.range,
                message: "requires exactly one ordinary positional parameter",
            });
        }
        let parameter = &parameters.args[0];
        if parameter.default.is_some() {
            return Err(TypeShapeDslDefinitionError {
                range: parameter.range,
                message: "does not support parameter defaults",
            });
        }
        let [Stmt::Return(return_stmt)] = definition.body.as_slice() else {
            return Err(TypeShapeDslDefinitionError {
                range: definition.name.range(),
                message: "body must contain exactly `return <parameter>`",
            });
        };
        let Some(Expr::Name(returned_name)) = return_stmt.value.as_deref() else {
            return Err(TypeShapeDslDefinitionError {
                range: return_stmt.range,
                message: "return value must be the bare parameter name",
            });
        };
        if returned_name.id != parameter.parameter.name.id {
            return Err(TypeShapeDslDefinitionError {
                range: returned_name.range,
                message: "returned name must match the parameter name",
            });
        }
        Ok(Self {
            parameter_name: parameter.parameter.name.id.clone(),
            definition: Arc::new(definition),
        })
    }

    pub fn parameter_name(&self) -> &Name {
        &self.parameter_name
    }

    pub fn name(&self) -> &Name {
        &self.definition.name.id
    }

    pub fn parameter_annotation_range(&self) -> TextRange {
        self.definition.parameters.args[0]
            .parameter
            .annotation
            .as_ref()
            .map_or_else(|| self.definition.parameters.args[0].range(), |x| x.range())
    }

    pub fn has_parameter_annotation(&self) -> bool {
        self.definition.parameters.args[0]
            .parameter
            .annotation
            .is_some()
    }

    pub fn return_annotation_range(&self) -> TextRange {
        self.definition
            .returns
            .as_ref()
            .map_or_else(|| self.definition.name.range(), |x| x.range())
    }

    pub fn has_return_annotation(&self) -> bool {
        self.definition.returns.is_some()
    }
}

/// A deferred type-level DSL invocation held until a function-call return boundary.
#[derive(Debug, Clone, PartialEq, Eq, TypeEq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut)]
pub struct TypeLevelDslCall {
    pub(crate) function: TypeLevelDslFunction,
    pub(crate) args: Vec<Type>,
}

/// The identity of a type-level DSL operation.
#[derive(Debug, Clone, PartialEq, Eq, TypeEq, PartialOrd, Ord, Hash)]
pub enum TypeLevelDslFunction {
    Broadcast,
    UserDefined {
        function: Arc<ValidatedTypeShapeDslFunction>,
        domain: TypeShapeDslDomain,
    },
}

#[derive(Debug, Clone)]
enum DslValue {
    Int(Int),
    IntTuple(IntTuple),
}

impl Visit<Type> for TypeLevelDslFunction {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

impl VisitMut<Type> for TypeLevelDslFunction {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}

impl TypeLevelDslCall {
    /// Constructs a native two-argument broadcast call.
    pub fn broadcast(args: Vec<Type>) -> Self {
        Self {
            function: TypeLevelDslFunction::Broadcast,
            args,
        }
    }

    pub fn user_defined(
        function: Arc<ValidatedTypeShapeDslFunction>,
        domain: TypeShapeDslDomain,
        arg: Type,
    ) -> Self {
        Self {
            function: TypeLevelDslFunction::UserDefined { function, domain },
            args: vec![arg],
        }
    }

    pub fn function_name(&self) -> &str {
        match &self.function {
            TypeLevelDslFunction::Broadcast => "broadcast",
            TypeLevelDslFunction::UserDefined { function, .. } => function.name().as_str(),
        }
    }

    pub fn result_domain(&self) -> TypeShapeDslDomain {
        match &self.function {
            TypeLevelDslFunction::Broadcast => TypeShapeDslDomain::IntTuple,
            TypeLevelDslFunction::UserDefined { domain, .. } => *domain,
        }
    }

    /// Returns the gradual result for a call whose precise value cannot be determined.
    pub fn fallback(&self) -> Type {
        match self.result_domain() {
            TypeShapeDslDomain::Int => gradual_size(),
            TypeShapeDslDomain::IntTuple => IntTuple::shapeless().to_shape_arg_type(),
        }
    }

    /// Evaluates the call, reporting incompatible concrete shapes.
    pub fn evaluate(&self) -> Result<Type, ShapeError> {
        match &self.function {
            TypeLevelDslFunction::Broadcast => {
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
            TypeLevelDslFunction::UserDefined { function, domain } => {
                let [arg] = self.args.as_slice() else {
                    unreachable!("validated identity DSL calls are constructed with one argument");
                };
                Ok(DslValue::from_type(arg, *domain)
                    .map(|value| function.evaluate(value).into_type())
                    .unwrap_or_else(|| self.fallback()))
            }
        }
    }
}

impl ValidatedTypeShapeDslFunction {
    fn evaluate(&self, value: DslValue) -> DslValue {
        // Validation currently admits only the identity program.
        value
    }
}

impl DslValue {
    fn from_type(ty: &Type, domain: TypeShapeDslDomain) -> Option<Self> {
        match domain {
            TypeShapeDslDomain::Int => Int::from_type(ty).map(Self::Int),
            TypeShapeDslDomain::IntTuple => IntTuple::from_shape_arg_type(ty).map(Self::IntTuple),
        }
    }

    fn into_type(self) -> Type {
        match self {
            Self::Int(value) => Type::Int(value),
            Self::IntTuple(value) => value.to_shape_arg_type(),
        }
    }
}
