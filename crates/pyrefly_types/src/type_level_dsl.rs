/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::BoolOp;
use ruff_python_ast::CmpOp;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::Number;
use ruff_python_ast::Operator;
use ruff_python_ast::Parameters;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtAssign;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::StmtIf;
use ruff_python_ast::StmtReturn;
use ruff_python_ast::UnaryOp;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::dimension::Int;
use crate::dimension::ShapeError;
use crate::dimension::canonicalize;
use crate::dimension::gradual_size;
use crate::equality::TypeEq as TypeEqTrait;
use crate::equality::TypeEqCtx;
use crate::literal::Lit;
use crate::shaped_array::IntTuple;
use crate::shaped_array::IntTupleView;
use crate::shaped_array::broadcast_shapes;
use crate::shaped_array::tuple_carrier_to_shape;
use crate::type_var::FlagDomain;
use crate::type_var::FlagMember;
use crate::types::Type;

#[derive(Debug, Clone, Copy, PartialEq, Eq, TypeEq, PartialOrd, Ord, Hash)]
pub enum TypeShapeDslDomain {
    Int,
    IntTuple,
}

impl TypeShapeDslDomain {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Int => "Int",
            Self::IntTuple => "IntTuple",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, TypeEq, PartialOrd, Ord, Hash)]
pub enum TypeShapeDslInputDomain {
    Value(TypeShapeDslDomain),
    Flag(FlagDomain),
}

impl fmt::Display for TypeShapeDslInputDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Value(domain) => f.write_str(domain.as_str()),
            Self::Flag(domain) => write!(f, "Flag[{domain}]"),
        }
    }
}

/// A validated DSL definition paired with its resolved input and result domains.
#[derive(Debug, Clone, PartialEq, Eq, TypeEq, PartialOrd, Ord, Hash)]
pub struct ResolvedTypeShapeDslFunction {
    definition: Arc<ValidatedTypeShapeDslFunction>,
    parameter_domains: Vec<TypeShapeDslInputDomain>,
    result_domain: TypeShapeDslDomain,
}

impl ResolvedTypeShapeDslFunction {
    pub fn try_new(
        definition: Arc<ValidatedTypeShapeDslFunction>,
        parameter_domains: Vec<TypeShapeDslInputDomain>,
        result_domain: TypeShapeDslDomain,
    ) -> Option<Self> {
        if definition.parsed.parameter_count() != parameter_domains.len() {
            return None;
        }
        Some(Self {
            definition,
            parameter_domains,
            result_domain,
        })
    }

    pub fn name(&self) -> &Name {
        self.definition.name()
    }

    pub fn parameter_name(&self, index: usize) -> &Name {
        self.definition.parameter_name(index)
    }

    pub fn parameter_domains(&self) -> &[TypeShapeDslInputDomain] {
        &self.parameter_domains
    }

    pub fn result_domain(&self) -> TypeShapeDslDomain {
        self.result_domain
    }
}

impl Visit<Type> for TypeShapeDslDomain {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

impl VisitMut<Type> for TypeShapeDslDomain {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}

impl Visit<Type> for TypeShapeDslInputDomain {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

impl VisitMut<Type> for TypeShapeDslInputDomain {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}

impl Visit<Type> for ResolvedTypeShapeDslFunction {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

impl VisitMut<Type> for ResolvedTypeShapeDslFunction {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}

impl Visit<Type> for Arc<ResolvedTypeShapeDslFunction> {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

impl VisitMut<Type> for Arc<ResolvedTypeShapeDslFunction> {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}

#[derive(Debug, Clone)]
pub struct TypeShapeDslDefinitionError {
    pub range: TextRange,
    pub message: &'static str,
}

/// A type-level shape DSL declaration whose envelope was validated during binding.
#[derive(Debug, Clone)]
pub struct ParsedTypeShapeDslFunction {
    definition: Arc<StmtFunctionDef>,
}

/// An owned function AST whose restricted declaration syntax and body have been validated.
/// Future evaluation may interpret the definition relying on these invariants.
///
/// Identity is derived from the parsed program's pointer identity plus the resolved metadata. The
/// latter is required because resolving an intrinsic depends on imports outside this AST, so an
/// unedited declaration whose gradual constructor now resolves to a different domain is unequal.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ValidatedTypeShapeDslFunction {
    parsed: ParsedTypeShapeDslFunction,
    // These source-keyed facts are validation invariants for the retained AST, not a body IR.
    returns: Vec<TypeShapeDslReturn>,
    conditions: Vec<TypeShapeDslCondition>,
    expressions: Vec<TypeShapeDslExpression>,
    assignments: Vec<TypeShapeDslAssignment>,
    /// The number of lexical slots the body needs: one per parameter, then one per local.
    slot_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeShapeDslReturn {
    statement_range: TextRange,
    value_range: TextRange,
    kind: TypeShapeDslReturnKind,
}

impl TypeShapeDslReturn {
    pub fn range(&self) -> TextRange {
        self.value_range
    }

    pub fn kind(&self) -> &TypeShapeDslReturnKind {
        &self.kind
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeShapeDslCondition {
    range: TextRange,
    kind: TypeShapeDslConditionKind,
}

impl TypeShapeDslCondition {
    pub fn range(&self) -> TextRange {
        self.range
    }

    pub fn kind(&self) -> &TypeShapeDslConditionKind {
        &self.kind
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeShapeDslExpression {
    range: TextRange,
    kind: TypeShapeDslExpressionKind,
}

impl TypeShapeDslExpression {
    pub fn range(&self) -> TextRange {
        self.range
    }

    pub fn kind(&self) -> &TypeShapeDslExpressionKind {
        &self.kind
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TypeShapeDslAssignment {
    range: TextRange,
    slot: usize,
}

/// The validated source of a type-level shape DSL function's return value. Resolving this depends
/// on more than the AST, so it participates in `ValidatedTypeShapeDslFunction` identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TypeShapeDslReturnKind {
    /// Return the parameter at the given zero-based position.
    Parameter(usize),
    Local {
        slot: usize,
        domain: TypeShapeDslDomain,
    },
    AliasedParameter {
        slot: usize,
        parameters: Box<[usize]>,
    },
    /// Return the broadcast of two shape parameters.
    Broadcast {
        left_slot: usize,
        right_slot: usize,
        left_parameters: Box<[usize]>,
        right_parameters: Box<[usize]>,
    },
    /// Return an arithmetic expression over two integer flag parameters.
    IntFlagArithmetic {
        left: usize,
        op: TypeShapeDslArithmeticOp,
        right: usize,
    },
    /// Evaluate a validated value expression from the retained AST.
    Expression,
    /// Return an invalid shape computation with a source-provided message.
    Invalid,
    /// Return the gradual value for the function's declared result domain.
    Gradual(TypeShapeDslDomain),
}

/// The arithmetic a validated dimension or Flag expression applies. Reached through
/// `TypeShapeDslReturnKind` and `TypeShapeDslExpressionKind`, so it shares their identity
/// requirements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TypeShapeDslArithmeticOp {
    Add,
    Subtract,
}

/// A closed, canonical operation the DSL recognizes by callable identity rather than by spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeShapeDslIntrinsic {
    Broadcast,
    Gradual(TypeShapeDslDomain),
    IsConcreteInt,
    IntTuple,
    Invalid,
    Len,
}

/// What a validated DSL value expression computes. Like `TypeShapeDslReturnKind` this depends on
/// intrinsic resolution, so it participates in `ValidatedTypeShapeDslFunction` identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TypeShapeDslExpressionKind {
    DimensionSlot {
        slot: usize,
        parameter_origins: Option<Box<[usize]>>,
    },
    DimensionLiteral(Option<i64>),
    IntTupleIndex {
        shape: usize,
        parameter_origins: Option<Box<[usize]>>,
        index: Option<i64>,
    },
    DimensionTuple,
    IntTupleConstructor,
    Slot(usize),
}

/// What a validated DSL condition tests. Like `TypeShapeDslReturnKind` this depends on intrinsic
/// resolution, so it participates in `ValidatedTypeShapeDslFunction` identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TypeShapeDslConditionKind {
    Equal(usize, usize),
    LessThan(usize, usize),
    IsConcreteInt {
        slot: usize,
        parameter_origins: Option<Box<[usize]>>,
    },
    FlagIntLessThanLiteral {
        parameter: usize,
        literal: Option<i64>,
    },
    LengthEqualLiteral {
        shape: usize,
        literal: Option<i64>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DslStaticKind {
    UnknownParameters(Box<[usize]>),
    Dimension,
}

impl DslStaticKind {
    fn parameter_origins(&self) -> Option<&[usize]> {
        match self {
            Self::UnknownParameters(parameters) => Some(parameters),
            _ => None,
        }
    }

    fn join(self, other: Self) -> Option<Self> {
        match (self, other) {
            (left, right) if left == right => Some(left),
            (Self::UnknownParameters(left), Self::UnknownParameters(right)) => {
                let mut parameters = left.into_vec();
                parameters.extend(right);
                parameters.sort_unstable();
                parameters.dedup();
                Some(Self::UnknownParameters(parameters.into_boxed_slice()))
            }
            _ => None,
        }
    }
}

enum IntegerLiteral {
    NotLiteral,
    Unrepresentable,
    Value(i64),
}

impl IntegerLiteral {
    fn into_value(self) -> Result<Option<i64>, ()> {
        match self {
            Self::NotLiteral => Err(()),
            Self::Unrepresentable => Ok(None),
            Self::Value(value) => Ok(Some(value)),
        }
    }
}

fn integer_literal(expr: &Expr) -> IntegerLiteral {
    // TODO: Preserve the sign of out-of-`i64` thresholds instead of falling back to gradual.
    match expr {
        Expr::NumberLiteral(number) => match &number.value {
            Number::Int(value) => value
                .as_i64()
                .map_or(IntegerLiteral::Unrepresentable, IntegerLiteral::Value),
            _ => IntegerLiteral::NotLiteral,
        },
        Expr::UnaryOp(unary) if matches!(unary.op, UnaryOp::UAdd | UnaryOp::USub) => {
            let Expr::NumberLiteral(number) = unary.operand.as_ref() else {
                return IntegerLiteral::NotLiteral;
            };
            let Number::Int(value) = &number.value else {
                return IntegerLiteral::NotLiteral;
            };
            if unary.op == UnaryOp::UAdd {
                value
                    .as_i64()
                    .map_or(IntegerLiteral::Unrepresentable, IntegerLiteral::Value)
            } else {
                value
                    .as_i64()
                    .and_then(i64::checked_neg)
                    .or_else(|| (value.as_u64() == Some(i64::MAX as u64 + 1)).then_some(i64::MIN))
                    .map_or(IntegerLiteral::Unrepresentable, IntegerLiteral::Value)
            }
        }
        _ => IntegerLiteral::NotLiteral,
    }
}

#[derive(Clone)]
struct DslValidationFlow {
    assigned: Vec<bool>,
    maybe_assigned: Vec<bool>,
    kinds: Vec<DslStaticKind>,
}

struct DslValidator<'a, F> {
    parameters: &'a Parameters,
    intrinsic: &'a F,
    returns: Vec<TypeShapeDslReturn>,
    conditions: Vec<TypeShapeDslCondition>,
    expressions: Vec<TypeShapeDslExpression>,
    assignments: Vec<TypeShapeDslAssignment>,
    slots: HashMap<Name, usize>,
    declared_local_kinds: Vec<Option<DslStaticKind>>,
}

impl<'a, F: Fn(&Expr) -> Option<TypeShapeDslIntrinsic>> DslValidator<'a, F> {
    fn new(parameters: &'a Parameters, intrinsic: &'a F) -> (Self, DslValidationFlow) {
        let mut slots = HashMap::new();
        let mut kinds = Vec::new();
        for (index, parameter) in parameters.args.iter().enumerate() {
            slots.insert(parameter.parameter.name.id.clone(), index);
            kinds.push(DslStaticKind::UnknownParameters(Box::new([index])));
        }
        let assigned = vec![true; kinds.len()];
        let maybe_assigned = assigned.clone();
        (
            Self {
                parameters,
                intrinsic,
                returns: Vec::new(),
                conditions: Vec::new(),
                expressions: Vec::new(),
                assignments: Vec::new(),
                slots,
                declared_local_kinds: vec![None; kinds.len()],
            },
            DslValidationFlow {
                assigned,
                maybe_assigned,
                kinds,
            },
        )
    }

    fn intrinsic(&self, expression: &Expr) -> Option<TypeShapeDslIntrinsic> {
        if expression_root_name(expression).is_some_and(|name| self.slots.contains_key(name)) {
            None
        } else {
            (self.intrinsic)(expression)
        }
    }

    fn normalize_flow(&self, flow: &mut DslValidationFlow) {
        assert_eq!(
            flow.assigned.len(),
            flow.maybe_assigned.len(),
            "DSL validation flow assignment facts must stay aligned"
        );
        assert_eq!(
            flow.assigned.len(),
            flow.kinds.len(),
            "DSL validation flow kinds must stay aligned with assignment facts"
        );
        while flow.kinds.len() < self.declared_local_kinds.len() {
            let kind = self.declared_local_kinds[flow.kinds.len()]
                .clone()
                .expect("DSL locals must have a declared kind before flow normalization");
            flow.assigned.push(false);
            flow.maybe_assigned.push(false);
            flow.kinds.push(kind);
        }
    }

    fn slot(
        &self,
        expression: &Expr,
        flow: &DslValidationFlow,
    ) -> Result<usize, TypeShapeDslDefinitionError> {
        let Expr::Name(name) = expression else {
            return Err(TypeShapeDslDefinitionError {
                range: expression.range(),
                message: "value must be a bare parameter or local name",
            });
        };
        let Some(&slot) = self.slots.get(&name.id) else {
            return Err(TypeShapeDslDefinitionError {
                range: name.range,
                message: "local value must be assigned before use",
            });
        };
        if !flow.assigned.get(slot).copied().unwrap_or(false) {
            return Err(TypeShapeDslDefinitionError {
                range: name.range,
                message: "local value must be definitely assigned before use",
            });
        }
        Ok(slot)
    }

    fn validate_dimension(
        &mut self,
        expression: &Expr,
        flow: &DslValidationFlow,
    ) -> Result<(), TypeShapeDslDefinitionError> {
        let kind = match expression {
            Expr::Name(_) => {
                let slot = self.slot(expression, flow)?;
                match &flow.kinds[slot] {
                    DslStaticKind::UnknownParameters(parameters) => {
                        TypeShapeDslExpressionKind::DimensionSlot {
                            slot,
                            parameter_origins: Some(parameters.clone()),
                        }
                    }
                    DslStaticKind::Dimension => TypeShapeDslExpressionKind::DimensionSlot {
                        slot,
                        parameter_origins: None,
                    },
                }
            }
            Expr::NumberLiteral(_) => {
                let Ok(literal) = integer_literal(expression).into_value() else {
                    return Err(TypeShapeDslDefinitionError {
                        range: expression.range(),
                        message: "dimension literal must be an integer",
                    });
                };
                TypeShapeDslExpressionKind::DimensionLiteral(literal)
            }
            Expr::UnaryOp(unary) if matches!(unary.op, UnaryOp::UAdd | UnaryOp::USub) => {
                let Ok(literal) = integer_literal(expression).into_value() else {
                    return Err(TypeShapeDslDefinitionError {
                        range: expression.range(),
                        message: "dimension literal must be an integer",
                    });
                };
                TypeShapeDslExpressionKind::DimensionLiteral(literal)
            }
            Expr::UnaryOp(_) => {
                return Err(TypeShapeDslDefinitionError {
                    range: expression.range(),
                    message: "dimension literal supports only unary `+` or `-`",
                });
            }
            Expr::Subscript(subscript) => {
                let shape = self.slot(&subscript.value, flow)?;
                let parameter_origins = match &flow.kinds[shape] {
                    DslStaticKind::UnknownParameters(parameters) => Some(parameters.clone()),
                    _ => {
                        return Err(TypeShapeDslDefinitionError {
                            range: subscript.value.range(),
                            message: "indexed dimension source must be an `IntTuple` value",
                        });
                    }
                };
                let Ok(index) = integer_literal(&subscript.slice).into_value() else {
                    return Err(TypeShapeDslDefinitionError {
                        range: subscript.slice.range(),
                        message: "`IntTuple` index must be an integer literal",
                    });
                };
                TypeShapeDslExpressionKind::IntTupleIndex {
                    shape,
                    parameter_origins,
                    index,
                }
            }
            _ => {
                return Err(TypeShapeDslDefinitionError {
                    range: expression.range(),
                    message: "`IntTuple` elements must be dimensions, integer literals, or indexed IntTuple values",
                });
            }
        };
        self.expressions.push(TypeShapeDslExpression {
            range: expression.range(),
            kind,
        });
        Ok(())
    }

    fn validate_int_tuple_constructor(
        &mut self,
        call: &ExprCall,
        flow: &DslValidationFlow,
    ) -> Result<(), TypeShapeDslDefinitionError> {
        if call.arguments.args.len() != 1 || !call.arguments.keywords.is_empty() {
            return Err(TypeShapeDslDefinitionError {
                range: call.arguments.range,
                message: "`dsl.IntTuple` requires exactly one positional tuple argument",
            });
        }
        let Expr::Tuple(tuple) = &call.arguments.args[0] else {
            return Err(TypeShapeDslDefinitionError {
                range: call.arguments.args[0].range(),
                message: "`dsl.IntTuple` argument must be a fixed tuple",
            });
        };
        for element in &tuple.elts {
            if matches!(element, Expr::Starred(_)) {
                return Err(TypeShapeDslDefinitionError {
                    range: element.range(),
                    message: "`dsl.IntTuple` does not support starred elements",
                });
            }
            self.validate_dimension(element, flow)?;
        }
        self.expressions.push(TypeShapeDslExpression {
            range: tuple.range,
            kind: TypeShapeDslExpressionKind::DimensionTuple,
        });
        self.expressions.push(TypeShapeDslExpression {
            range: call.range,
            kind: TypeShapeDslExpressionKind::IntTupleConstructor,
        });
        Ok(())
    }

    fn validate_assignment_value(
        &mut self,
        expression: &Expr,
        flow: &DslValidationFlow,
    ) -> Result<DslStaticKind, TypeShapeDslDefinitionError> {
        match expression {
            Expr::Name(_) => {
                let slot = self.slot(expression, flow)?;
                self.expressions.push(TypeShapeDslExpression {
                    range: expression.range(),
                    kind: TypeShapeDslExpressionKind::Slot(slot),
                });
                Ok(flow.kinds[slot].clone())
            }
            Expr::Subscript(_) => {
                self.validate_dimension(expression, flow)?;
                Ok(DslStaticKind::Dimension)
            }
            _ => Err(TypeShapeDslDefinitionError {
                range: expression.range(),
                message: "local assignment value is not supported by the type-level DSL",
            }),
        }
    }

    fn assign(
        &mut self,
        statement: &StmtAssign,
        flow: &mut DslValidationFlow,
    ) -> Result<(), TypeShapeDslDefinitionError> {
        let [Expr::Name(target)] = statement.targets.as_slice() else {
            return Err(TypeShapeDslDefinitionError {
                range: statement.range,
                message: "body supports only `if` and `return`; local assignment requires exactly one bare name target",
            });
        };
        let kind = self.validate_assignment_value(&statement.value, flow)?;
        let slot = if let Some(&slot) = self.slots.get(&target.id) {
            self.normalize_flow(flow);
            if flow.maybe_assigned[slot] {
                return Err(TypeShapeDslDefinitionError {
                    range: target.range,
                    message: if slot < self.parameters.args.len() {
                        "type-level DSL parameters are immutable and cannot be assigned"
                    } else {
                        "type-level DSL locals are immutable and cannot be reassigned"
                    },
                });
            }
            slot
        } else {
            let slot = self.declared_local_kinds.len();
            self.slots.insert(target.id.clone(), slot);
            self.declared_local_kinds.push(Some(kind.clone()));
            self.normalize_flow(flow);
            slot
        };
        flow.assigned[slot] = true;
        flow.maybe_assigned[slot] = true;
        flow.kinds[slot] = kind;
        self.assignments.push(TypeShapeDslAssignment {
            range: statement.range,
            slot,
        });
        Ok(())
    }

    fn validate_condition(
        &mut self,
        condition: &Expr,
        flow: &DslValidationFlow,
    ) -> Result<(DslValidationFlow, DslValidationFlow), TypeShapeDslDefinitionError> {
        if let Expr::BoolOp(bool_op) = condition {
            if bool_op.op != BoolOp::And {
                return Err(TypeShapeDslDefinitionError {
                    range: bool_op.range,
                    message: "condition supports only the boolean operator `and`",
                });
            }
            for value in &bool_op.values {
                self.validate_condition(value, flow)?;
            }
            return Ok((flow.clone(), flow.clone()));
        }

        if let Expr::Call(call) = condition
            && self.intrinsic(&call.func) == Some(TypeShapeDslIntrinsic::IsConcreteInt)
        {
            if call.arguments.args.len() != 1
                || !call.arguments.keywords.is_empty()
                || matches!(call.arguments.args.first(), Some(Expr::Starred(_)))
            {
                return Err(TypeShapeDslDefinitionError {
                    range: call.arguments.range,
                    message: "`is_concrete_int` condition requires exactly one positional argument",
                });
            }
            let slot = self.slot(&call.arguments.args[0], flow)?;
            let parameter_origins = flow.kinds[slot]
                .parameter_origins()
                .map(<[usize]>::to_vec)
                .map(Vec::into_boxed_slice);
            let kind = TypeShapeDslConditionKind::IsConcreteInt {
                slot,
                parameter_origins,
            };
            self.conditions.push(TypeShapeDslCondition {
                range: call.range,
                kind,
            });
            return Ok((flow.clone(), flow.clone()));
        }

        let Expr::Compare(compare) = condition else {
            return Err(TypeShapeDslDefinitionError {
                range: condition.range(),
                message: "condition supports only `is_concrete_int`, `and`, `==`, and `<`",
            });
        };
        if compare.ops.len() == 1
            && compare.ops[0] == CmpOp::Eq
            && compare.comparators.len() == 1
            && let Expr::Call(call) = compare.left.as_ref()
            && self.intrinsic(&call.func) == Some(TypeShapeDslIntrinsic::Len)
        {
            if call.arguments.args.len() != 1
                || !call.arguments.keywords.is_empty()
                || matches!(call.arguments.args.first(), Some(Expr::Starred(_)))
            {
                return Err(TypeShapeDslDefinitionError {
                    range: call.arguments.range,
                    message: "`len` requires exactly one positional argument",
                });
            }
            let shape = self.slot(&call.arguments.args[0], flow)?;
            let Some(parameters) = flow.kinds[shape].parameter_origins() else {
                return Err(TypeShapeDslDefinitionError {
                    range: call.arguments.args[0].range(),
                    message: "`len` argument must name an `IntTuple` parameter or immutable alias",
                });
            };
            let [shape] = parameters else {
                return Err(TypeShapeDslDefinitionError {
                    range: call.arguments.args[0].range(),
                    message: "`len` argument must have one parameter origin",
                });
            };
            let Ok(literal) = integer_literal(&compare.comparators[0]).into_value() else {
                return Err(TypeShapeDslDefinitionError {
                    range: compare.comparators[0].range(),
                    message: "`len` comparison requires an integer literal",
                });
            };
            self.conditions.push(TypeShapeDslCondition {
                range: compare.range,
                kind: TypeShapeDslConditionKind::LengthEqualLiteral {
                    shape: *shape,
                    literal,
                },
            });
            return Ok((flow.clone(), flow.clone()));
        }
        if compare.ops.len() != 1
            || !matches!(compare.ops[0], CmpOp::Eq | CmpOp::Lt)
            || compare.comparators.len() != 1
        {
            return Err(TypeShapeDslDefinitionError {
                range: compare.range,
                message: "comparison must be exactly `<Int parameter> == <Int parameter>`, `<Int parameter> < <Int parameter>`, or `<int parameter> < <integer literal>`",
            });
        }
        let op = compare.ops[0];
        let right = &compare.comparators[0];
        if !matches!(compare.left.as_ref(), Expr::Name(_)) {
            return Err(TypeShapeDslDefinitionError {
                range: compare.left.range(),
                message: "left comparison operand must name a parameter",
            });
        }
        let left_slot = self.slot(&compare.left, flow)?;
        let Some(left_parameters) = flow.kinds[left_slot].parameter_origins() else {
            return Err(TypeShapeDslDefinitionError {
                range: compare.left.range(),
                message: "left comparison operand must name a parameter or immutable alias",
            });
        };
        let [left] = left_parameters else {
            return Err(TypeShapeDslDefinitionError {
                range: compare.left.range(),
                message: "left comparison operand must have one parameter origin",
            });
        };
        let kind = if let Expr::Name(_) = right {
            let right_slot = self.slot(right, flow)?;
            let Some(right_parameters) = flow.kinds[right_slot].parameter_origins() else {
                return Err(TypeShapeDslDefinitionError {
                    range: right.range(),
                    message: "right comparison operand must name a parameter or immutable alias",
                });
            };
            let [right] = right_parameters else {
                return Err(TypeShapeDslDefinitionError {
                    range: right.range(),
                    message: "right comparison operand must have one parameter origin",
                });
            };
            match op {
                CmpOp::Eq => TypeShapeDslConditionKind::Equal(*left, *right),
                CmpOp::Lt => TypeShapeDslConditionKind::LessThan(*left, *right),
                _ => unreachable!("validated comparison operator is equality or less-than"),
            }
        } else if op == CmpOp::Lt {
            let Ok(literal) = integer_literal(right).into_value() else {
                return Err(TypeShapeDslDefinitionError {
                    range: right.range(),
                    message: "right `<` operand must name a parameter or be an integer literal",
                });
            };
            TypeShapeDslConditionKind::FlagIntLessThanLiteral {
                parameter: *left,
                literal,
            }
        } else {
            return Err(TypeShapeDslDefinitionError {
                range: right.range(),
                message: "condition operands must name parameters or immutable aliases",
            });
        };
        self.conditions.push(TypeShapeDslCondition {
            range: compare.range,
            kind,
        });
        Ok((flow.clone(), flow.clone()))
    }

    fn validate_return(
        &mut self,
        return_stmt: &StmtReturn,
        flow: &DslValidationFlow,
    ) -> Result<(), TypeShapeDslDefinitionError> {
        let kind = match return_stmt.value.as_deref() {
            Some(returned @ Expr::Attribute(_))
                if matches!(
                    self.intrinsic(returned),
                    Some(TypeShapeDslIntrinsic::Gradual(_))
                ) =>
            {
                return Err(TypeShapeDslDefinitionError {
                    range: returned.range(),
                    message: "gradual return must be called",
                });
            }
            Some(returned @ Expr::Name(name)) => {
                let returned_intrinsic = self.intrinsic(returned);
                if matches!(returned_intrinsic, Some(TypeShapeDslIntrinsic::Gradual(_))) {
                    return Err(TypeShapeDslDefinitionError {
                        range: name.range,
                        message: "gradual return must be called",
                    });
                } else {
                    let Some(&slot) = self.slots.get(&name.id) else {
                        return Err(TypeShapeDslDefinitionError {
                            range: name.range,
                            message: "returned name must match a parameter name or definitely assigned local",
                        });
                    };
                    if !flow.assigned.get(slot).copied().unwrap_or(false) {
                        return Err(TypeShapeDslDefinitionError {
                            range: name.range,
                            message: "local value must be definitely assigned before use",
                        });
                    }
                    if slot < self.parameters.args.len() {
                        TypeShapeDslReturnKind::Parameter(slot)
                    } else {
                        let domain = match &flow.kinds[slot] {
                            DslStaticKind::Dimension => TypeShapeDslDomain::Int,
                            DslStaticKind::UnknownParameters(parameters) => {
                                self.returns.push(TypeShapeDslReturn {
                                    statement_range: return_stmt.range,
                                    value_range: returned.range(),
                                    kind: TypeShapeDslReturnKind::AliasedParameter {
                                        slot,
                                        parameters: parameters.clone(),
                                    },
                                });
                                return Ok(());
                            }
                        };
                        TypeShapeDslReturnKind::Local { slot, domain }
                    }
                }
            }
            Some(Expr::Call(call)) => match self.intrinsic(&call.func) {
                Some(TypeShapeDslIntrinsic::Gradual(domain)) => {
                    if !call.arguments.args.is_empty() || !call.arguments.keywords.is_empty() {
                        return Err(TypeShapeDslDefinitionError {
                            range: call.arguments.range,
                            message: "gradual return does not accept arguments",
                        });
                    }
                    TypeShapeDslReturnKind::Gradual(domain)
                }
                Some(TypeShapeDslIntrinsic::Broadcast) => {
                    if call.arguments.args.len() != 2 || !call.arguments.keywords.is_empty() {
                        return Err(TypeShapeDslDefinitionError {
                            range: call.arguments.range,
                            message: "`broadcast` requires exactly two positional arguments",
                        });
                    }
                    let (Expr::Name(_), Expr::Name(_)) =
                        (&call.arguments.args[0], &call.arguments.args[1])
                    else {
                        return Err(TypeShapeDslDefinitionError {
                            range: call.arguments.range,
                            message: "`broadcast` arguments must be bare parameter names",
                        });
                    };
                    let left = self.slot(&call.arguments.args[0], flow)?;
                    let right = self.slot(&call.arguments.args[1], flow)?;
                    let (Some(left_parameters), Some(right_parameters)) = (
                        flow.kinds[left].parameter_origins(),
                        flow.kinds[right].parameter_origins(),
                    ) else {
                        return Err(TypeShapeDslDefinitionError {
                            range: call.arguments.range,
                            message: "`broadcast` arguments must be IntTuple parameters or immutable aliases of them",
                        });
                    };
                    TypeShapeDslReturnKind::Broadcast {
                        left_slot: left,
                        right_slot: right,
                        left_parameters: left_parameters.to_vec().into_boxed_slice(),
                        right_parameters: right_parameters.to_vec().into_boxed_slice(),
                    }
                }
                Some(TypeShapeDslIntrinsic::IntTuple) => {
                    self.validate_int_tuple_constructor(call, flow)?;
                    TypeShapeDslReturnKind::Expression
                }
                Some(TypeShapeDslIntrinsic::Invalid) => {
                    if call.arguments.args.len() != 1
                        || !call.arguments.keywords.is_empty()
                        || !matches!(call.arguments.args.first(), Some(Expr::StringLiteral(_)))
                    {
                        return Err(TypeShapeDslDefinitionError {
                            range: call.arguments.range,
                            message: "`dsl.Invalid` requires exactly one positional string literal",
                        });
                    }
                    TypeShapeDslReturnKind::Invalid
                }
                None | Some(_) => {
                    return Err(TypeShapeDslDefinitionError {
                        range: return_stmt.range,
                        message: "return value must be a bare parameter name, a gradual return, `broadcast(...)`, or an exact `Int +/- Flag[int]` arithmetic expression; it may also be `dsl.Invalid(...)` or `dsl.IntTuple(...)`",
                    });
                }
            },
            Some(Expr::BinOp(binop)) => {
                let (Some(left), Some(right)) = (
                    parameter_index(self.parameters, &binop.left),
                    parameter_index(self.parameters, &binop.right),
                ) else {
                    return Err(TypeShapeDslDefinitionError {
                        range: binop.range,
                        message: "arithmetic return operands must be bare parameter names",
                    });
                };
                let op = match binop.op {
                    Operator::Add => TypeShapeDslArithmeticOp::Add,
                    Operator::Sub => TypeShapeDslArithmeticOp::Subtract,
                    _ => {
                        return Err(TypeShapeDslDefinitionError {
                            range: binop.range,
                            message: "arithmetic return supports only `Int parameter + Flag[int] parameter` or `Int parameter - Flag[int] parameter`",
                        });
                    }
                };
                TypeShapeDslReturnKind::IntFlagArithmetic { left, op, right }
            }
            _ => {
                return Err(TypeShapeDslDefinitionError {
                    range: return_stmt.range,
                    message: "return value must be a bare parameter name, a gradual return, `broadcast(...)`, or an exact `Int +/- Flag[int]` arithmetic expression; it may also be `dsl.Invalid(...)` or `dsl.IntTuple(...)`",
                });
            }
        };
        self.returns.push(TypeShapeDslReturn {
            statement_range: return_stmt.range,
            value_range: return_stmt
                .value
                .as_deref()
                .map_or(return_stmt.range, |value| value.range()),
            kind,
        });
        Ok(())
    }

    fn merge_flows(
        &self,
        flows: Vec<DslValidationFlow>,
        range: TextRange,
    ) -> Result<Option<DslValidationFlow>, TypeShapeDslDefinitionError> {
        let mut flows = flows.into_iter();
        let Some(mut result) = flows.next() else {
            return Ok(None);
        };
        for mut flow in flows {
            self.normalize_flow(&mut result);
            self.normalize_flow(&mut flow);
            for slot in 0..result.assigned.len() {
                result.assigned[slot] &= flow.assigned[slot];
                result.maybe_assigned[slot] |= flow.maybe_assigned[slot];
                if result.assigned[slot] {
                    let Some(kind) = result.kinds[slot].clone().join(flow.kinds[slot].clone())
                    else {
                        return Err(TypeShapeDslDefinitionError {
                            range,
                            message: "all continuing branch assignments to a local must have the same value domain",
                        });
                    };
                    result.kinds[slot] = kind;
                }
            }
        }
        Ok(Some(result))
    }

    fn validate_if(
        &mut self,
        if_stmt: &StmtIf,
        flow: &DslValidationFlow,
    ) -> Result<Option<DslValidationFlow>, TypeShapeDslDefinitionError> {
        let (when_true, mut when_false) = self.validate_condition(&if_stmt.test, flow)?;
        let mut continuing = Vec::new();
        if let Some(flow) = self.validate_suite(&if_stmt.body, when_true)? {
            continuing.push(flow);
        }
        let mut has_else = false;
        for clause in &if_stmt.elif_else_clauses {
            if let Some(test) = &clause.test {
                let (when_true, next_false) = self.validate_condition(test, &when_false)?;
                if let Some(flow) = self.validate_suite(&clause.body, when_true)? {
                    continuing.push(flow);
                }
                when_false = next_false;
            } else {
                has_else = true;
                if let Some(flow) = self.validate_suite(&clause.body, when_false.clone())? {
                    continuing.push(flow);
                }
            }
        }
        if !has_else {
            continuing.push(when_false);
        }
        self.merge_flows(continuing, if_stmt.range)
    }

    fn validate_suite(
        &mut self,
        suite: &[Stmt],
        mut flow: DslValidationFlow,
    ) -> Result<Option<DslValidationFlow>, TypeShapeDslDefinitionError> {
        let mut can_continue = true;
        for statement in suite {
            if !can_continue {
                return Err(TypeShapeDslDefinitionError {
                    range: statement.range(),
                    message: "statement is unreachable",
                });
            }
            match statement {
                Stmt::Assign(assign) => self.assign(assign, &mut flow)?,
                Stmt::Return(return_stmt) => {
                    self.validate_return(return_stmt, &flow)?;
                    can_continue = false;
                }
                Stmt::If(if_stmt) => match self.validate_if(if_stmt, &flow)? {
                    Some(merged) => flow = merged,
                    None => can_continue = false,
                },
                _ => {
                    return Err(TypeShapeDslDefinitionError {
                        range: statement.range(),
                        message: "body supports only `if` and `return`, plus supported immutable local assignments",
                    });
                }
            }
        }
        Ok(can_continue.then_some(flow))
    }
}

fn parameter_index(parameters: &Parameters, expr: &Expr) -> Option<usize> {
    let Expr::Name(name) = expr else {
        return None;
    };
    parameters
        .args
        .iter()
        .position(|parameter| parameter.parameter.name.id == name.id)
}

// The AST is executable program state, not a derived cache, so its identity must participate in
// incremental equality. Aliases within one module generation share this `Arc`; reparsing an edited
// definition creates a new allocation and invalidates every dependent call result. In particular,
// this must not be wrapped in `IdentityIgnored` like the derived V1 helper-closure cache.
impl PartialEq for ParsedTypeShapeDslFunction {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.definition, &other.definition)
    }
}

impl Eq for ParsedTypeShapeDslFunction {}

impl Hash for ParsedTypeShapeDslFunction {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.definition) as *const () as usize).hash(state);
    }
}

// This ordering is a process-local tie-breaker required by type nodes that derive `Ord`; it must
// not be used for stable output. Comparing the same identity as equality keeps `cmp` consistent
// with the pointer-based `Eq` above while distinguishing reparsed definitions.
impl PartialOrd for ParsedTypeShapeDslFunction {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ParsedTypeShapeDslFunction {
    fn cmp(&self, other: &Self) -> Ordering {
        let self_ptr = Arc::as_ptr(&self.definition) as *const () as usize;
        let other_ptr = Arc::as_ptr(&other.definition) as *const () as usize;
        self_ptr.cmp(&other_ptr)
    }
}

impl Visit<Type> for ParsedTypeShapeDslFunction {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

impl VisitMut<Type> for ParsedTypeShapeDslFunction {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}

impl Visit<Type> for Arc<ParsedTypeShapeDslFunction> {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

impl VisitMut<Type> for Arc<ParsedTypeShapeDslFunction> {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}

impl TypeEqTrait for ParsedTypeShapeDslFunction {
    fn type_eq(&self, other: &Self, _ctx: &mut TypeEqCtx) -> bool {
        self == other
    }
}

// `TextRange` has no total order, so the resolved metadata is ordered by its offsets. Like the
// pointer ordering on the parsed program this is a process-local tie-breaker required by type
// nodes that derive `Ord`, and it stays consistent with the derived `Eq` above.
impl PartialOrd for ValidatedTypeShapeDslFunction {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ValidatedTypeShapeDslFunction {
    fn cmp(&self, other: &Self) -> Ordering {
        fn offsets(range: TextRange) -> (TextSize, TextSize) {
            (range.start(), range.end())
        }
        self.parsed
            .cmp(&other.parsed)
            .then_with(|| {
                self.returns
                    .iter()
                    .map(|x| {
                        (
                            offsets(x.statement_range),
                            offsets(x.value_range),
                            x.kind.clone(),
                        )
                    })
                    .cmp(other.returns.iter().map(|x| {
                        (
                            offsets(x.statement_range),
                            offsets(x.value_range),
                            x.kind.clone(),
                        )
                    }))
            })
            .then_with(|| {
                self.conditions
                    .iter()
                    .map(|x| (offsets(x.range), x.kind.clone()))
                    .cmp(
                        other
                            .conditions
                            .iter()
                            .map(|x| (offsets(x.range), x.kind.clone())),
                    )
            })
            .then_with(|| {
                self.expressions
                    .iter()
                    .map(|x| (offsets(x.range), x.kind.clone()))
                    .cmp(
                        other
                            .expressions
                            .iter()
                            .map(|x| (offsets(x.range), x.kind.clone())),
                    )
            })
            .then_with(|| {
                self.assignments
                    .iter()
                    .map(|x| (offsets(x.range), x.slot))
                    .cmp(other.assignments.iter().map(|x| (offsets(x.range), x.slot)))
            })
            .then_with(|| self.slot_count.cmp(&other.slot_count))
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

fn expression_root_name(expr: &Expr) -> Option<&Name> {
    match expr {
        Expr::Name(name) => Some(&name.id),
        Expr::Attribute(attribute) => expression_root_name(&attribute.value),
        _ => None,
    }
}

impl ParsedTypeShapeDslFunction {
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
            || parameters.args.is_empty()
            || !parameters.kwonlyargs.is_empty()
            || parameters.vararg.is_some()
            || parameters.kwarg.is_some()
        {
            return Err(TypeShapeDslDefinitionError {
                range: parameters.range,
                message: "supports only ordinary positional parameters and requires at least one",
            });
        }
        for (index, parameter) in parameters.args.iter().enumerate() {
            if parameters.args[..index]
                .iter()
                .any(|previous| previous.parameter.name.id == parameter.parameter.name.id)
            {
                return Err(TypeShapeDslDefinitionError {
                    range: parameter.parameter.name.range(),
                    message: "parameter names must be unique",
                });
            }
        }
        if let Some(parameter) = parameters
            .args
            .iter()
            .find(|parameter| parameter.default.is_some())
        {
            return Err(TypeShapeDslDefinitionError {
                range: parameter.range,
                message: "does not support parameter defaults",
            });
        }
        Ok(Self {
            definition: Arc::new(definition),
        })
    }

    /// Validate the body: its lexical slots, control flow, and every resolved execution fact
    /// evaluation later replays against the retained AST.
    pub fn validate(
        &self,
        intrinsic: impl Fn(&Expr) -> Option<TypeShapeDslIntrinsic>,
    ) -> Result<ValidatedTypeShapeDslFunction, TypeShapeDslDefinitionError> {
        let parameters = &self.definition.parameters;
        // `DslValidator::intrinsic` suppresses resolution for any name bound by a slot, which
        // seeds with every parameter, so shadowing needs no separate check here.
        let (mut validator, flow) = DslValidator::new(parameters, &intrinsic);
        if validator
            .validate_suite(&self.definition.body, flow)?
            .is_some()
        {
            return Err(TypeShapeDslDefinitionError {
                range: self.definition.name.range(),
                message: "every control-flow path must return",
            });
        }
        let DslValidator {
            returns,
            conditions,
            expressions,
            assignments,
            declared_local_kinds,
            ..
        } = validator;
        let slot_count = declared_local_kinds.len();
        Ok(ValidatedTypeShapeDslFunction {
            parsed: self.clone(),
            returns,
            conditions,
            expressions,
            assignments,
            slot_count,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.definition.parameters.args.len()
    }

    pub fn parameter_name(&self, index: usize) -> &Name {
        &self.definition.parameters.args[index].parameter.name.id
    }

    pub fn name(&self) -> &Name {
        &self.definition.name.id
    }

    pub fn parameter_annotation_range(&self, index: usize) -> TextRange {
        self.definition.parameters.args[index]
            .parameter
            .annotation
            .as_ref()
            .map_or_else(
                || self.definition.parameters.args[index].range(),
                |x| x.range(),
            )
    }

    pub fn has_parameter_annotation(&self, index: usize) -> bool {
        self.definition.parameters.args[index]
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

impl ValidatedTypeShapeDslFunction {
    pub fn name(&self) -> &Name {
        self.parsed.name()
    }

    pub fn parameter_name(&self, index: usize) -> &Name {
        self.parsed.parameter_name(index)
    }

    pub fn returns(&self) -> impl Iterator<Item = TypeShapeDslReturn> + '_ {
        self.returns.iter().cloned()
    }

    pub fn conditions(&self) -> impl Iterator<Item = TypeShapeDslCondition> + '_ {
        self.conditions.iter().cloned()
    }

    pub fn expressions(&self) -> impl Iterator<Item = TypeShapeDslExpression> + '_ {
        self.expressions.iter().cloned()
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
    UserDefined(Arc<ResolvedTypeShapeDslFunction>),
}

#[derive(Debug, Clone)]
enum DslValue {
    Unknown,
    Dimension(Int),
    Shape(IntTuple),
    FlagInt(i64),
    DimensionTuple(Vec<Int>),
}

#[derive(Clone)]
enum DslOutcome {
    Value(DslValue),
    ExplicitGradual,
    Invalid(ShapeError),
}

struct DslEnvironment {
    parameter_count: usize,
    slots: Vec<DslValue>,
}

#[derive(Clone, Copy)]
enum DslCondition {
    True,
    False,
    Unknown,
}

enum DslControlFlow {
    Continue,
    Return(DslOutcome),
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

    pub fn user_defined(function: Arc<ResolvedTypeShapeDslFunction>, args: Vec<Type>) -> Self {
        assert_eq!(
            args.len(),
            function.parameter_domains().len(),
            "type-level DSL arguments must align with the resolved function"
        );
        Self {
            function: TypeLevelDslFunction::UserDefined(function),
            args,
        }
    }

    pub fn function_name(&self) -> &str {
        match &self.function {
            TypeLevelDslFunction::Broadcast => "broadcast",
            TypeLevelDslFunction::UserDefined(function) => function.name().as_str(),
        }
    }

    pub fn result_domain(&self) -> TypeShapeDslDomain {
        match &self.function {
            TypeLevelDslFunction::Broadcast => TypeShapeDslDomain::IntTuple,
            TypeLevelDslFunction::UserDefined(function) => function.result_domain(),
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
        let project = |outcome| match outcome {
            DslOutcome::Value(DslValue::Unknown) => Ok(self.fallback()),
            DslOutcome::Value(value) => Ok(value.into_type()),
            DslOutcome::ExplicitGradual => Ok(self.fallback()),
            DslOutcome::Invalid(error) => Err(error),
        };
        match &self.function {
            TypeLevelDslFunction::Broadcast => {
                let [left, right] = self.args.as_slice() else {
                    unreachable!("native broadcast DSL calls are constructed with two arguments");
                };
                project(evaluate_broadcast(
                    &DslValue::from_shape_type(left),
                    &DslValue::from_shape_type(right),
                ))
            }
            TypeLevelDslFunction::UserDefined(function) => {
                project(function.definition.evaluate(&self.args, function))
            }
        }
    }
}

impl ValidatedTypeShapeDslFunction {
    fn evaluate(&self, args: &[Type], signature: &ResolvedTypeShapeDslFunction) -> DslOutcome {
        let parameter_count = self.parsed.parameter_count();
        assert_eq!(
            parameter_count,
            signature.parameter_domains().len(),
            "validated type-level DSL AST must align with its signature"
        );
        assert_eq!(
            args.len(),
            parameter_count,
            "type-level DSL values must align with validated parameters"
        );
        let mut slots = args
            .iter()
            .zip(signature.parameter_domains())
            .map(|(argument, domain)| lower_parameter(argument, *domain))
            .collect::<Vec<_>>();
        slots.resize(self.slot_count, DslValue::Unknown);
        let mut environment = DslEnvironment {
            parameter_count: args.len(),
            slots,
        };
        match self.evaluate_suite(&self.parsed.definition.body, &mut environment, signature) {
            DslControlFlow::Return(result) => result,
            DslControlFlow::Continue => {
                unreachable!("validated type-level DSL function cannot fall through")
            }
        }
    }

    fn evaluate_suite(
        &self,
        suite: &[Stmt],
        environment: &mut DslEnvironment,
        signature: &ResolvedTypeShapeDslFunction,
    ) -> DslControlFlow {
        for statement in suite {
            match statement {
                Stmt::Assign(assign) => {
                    // Restricted DSL bodies keep these source-keyed validation tables small;
                    // retaining source order also keeps their diagnostic and inspection APIs simple.
                    let slot = self
                        .assignments
                        .iter()
                        .find_map(|assignment| {
                            (assignment.range == assign.range).then_some(assignment.slot)
                        })
                        .expect("validated assignment must have indexed-storage metadata");
                    match self.evaluate_expression(&assign.value, environment) {
                        DslOutcome::Value(value) => environment.assign(slot, value),
                        DslOutcome::ExplicitGradual => {
                            unreachable!("validated assignment expression cannot return gradual")
                        }
                        DslOutcome::Invalid(error) => {
                            return DslControlFlow::Return(DslOutcome::Invalid(error));
                        }
                    }
                }
                Stmt::Return(return_stmt) => {
                    let kind = self
                        .returns
                        .iter()
                        .find_map(|return_| {
                            (return_.statement_range == return_stmt.range)
                                .then(|| return_.kind.clone())
                        })
                        .expect("validated return statement must have validation metadata");
                    return DslControlFlow::Return(match kind {
                        TypeShapeDslReturnKind::Parameter(return_index) => {
                            assert_eq!(
                                signature.parameter_domains()[return_index],
                                TypeShapeDslInputDomain::Value(signature.result_domain()),
                                "validated parameter return domain must match its result domain"
                            );
                            DslOutcome::Value(environment.value(return_index).clone())
                        }
                        TypeShapeDslReturnKind::Local { slot, .. } => {
                            DslOutcome::Value(environment.value(slot).clone())
                        }
                        TypeShapeDslReturnKind::AliasedParameter { slot, .. } => {
                            DslOutcome::Value(environment.value(slot).clone())
                        }
                        TypeShapeDslReturnKind::IntFlagArithmetic { left, op, right } => {
                            let (Some(left), Some(right)) =
                                (environment.dimension(left), environment.flag_int(right))
                            else {
                                return DslControlFlow::Return(DslOutcome::Value(
                                    DslValue::Unknown,
                                ));
                            };
                            let result = match (left, op) {
                                (Int::Literal(left), TypeShapeDslArithmeticOp::Add) => {
                                    left.checked_add(right).map(Int::Literal)
                                }
                                (Int::Literal(left), TypeShapeDslArithmeticOp::Subtract) => {
                                    left.checked_sub(right).map(Int::Literal)
                                }
                                (_, TypeShapeDslArithmeticOp::Subtract) if right == i64::MIN => {
                                    None
                                }
                                (left, TypeShapeDslArithmeticOp::Add) => literal_offset(left)
                                    .and_then(|offset| {
                                        offset.checked_add(right).map(|_| {
                                            Int::add(
                                                Type::Int(left.clone()),
                                                Type::Int(Int::Literal(right)),
                                            )
                                        })
                                    }),
                                (left, TypeShapeDslArithmeticOp::Subtract) => literal_offset(left)
                                    .and_then(|offset| {
                                        offset.checked_sub(right).map(|_| {
                                            Int::sub(
                                                Type::Int(left.clone()),
                                                Type::Int(Int::Literal(right)),
                                            )
                                        })
                                    }),
                            };
                            result
                                .and_then(|result| Int::from_type(&canonicalize(Type::Int(result))))
                                .map_or(DslOutcome::Value(DslValue::Unknown), |result| {
                                    DslOutcome::Value(DslValue::Dimension(result))
                                })
                        }
                        TypeShapeDslReturnKind::Broadcast {
                            left_slot,
                            right_slot,
                            ..
                        } => evaluate_broadcast(
                            environment.value(left_slot),
                            environment.value(right_slot),
                        ),
                        TypeShapeDslReturnKind::Expression => {
                            let expression = return_stmt
                                .value
                                .as_deref()
                                .expect("validated expression return has a value");
                            self.evaluate_expression(expression, environment)
                        }
                        TypeShapeDslReturnKind::Invalid => {
                            let Some(Expr::Call(call)) = return_stmt.value.as_deref() else {
                                unreachable!("validated Invalid return is a call")
                            };
                            let Some(Expr::StringLiteral(message)) = call.arguments.args.first()
                            else {
                                unreachable!("validated Invalid return has a string message")
                            };
                            DslOutcome::Invalid(ShapeError::ShapeComputation {
                                message: message.value.to_str().to_owned(),
                            })
                        }
                        TypeShapeDslReturnKind::Gradual(domain) => {
                            assert_eq!(
                                domain,
                                signature.result_domain(),
                                "validated explicit gradual DSL return domain must match its result domain"
                            );
                            DslOutcome::ExplicitGradual
                        }
                    });
                }
                Stmt::If(if_stmt) => match self.evaluate_if(if_stmt, environment, signature) {
                    DslControlFlow::Continue => {}
                    result @ DslControlFlow::Return(_) => return result,
                },
                _ => unreachable!(
                    "validated type-level DSL suite contains only assignments, if, and return"
                ),
            }
        }
        DslControlFlow::Continue
    }

    fn evaluate_if(
        &self,
        if_stmt: &StmtIf,
        environment: &mut DslEnvironment,
        signature: &ResolvedTypeShapeDslFunction,
    ) -> DslControlFlow {
        match self.evaluate_condition(&if_stmt.test, environment) {
            Err(error) => return DslControlFlow::Return(DslOutcome::Invalid(error)),
            Ok(DslCondition::True) => {
                return self.evaluate_suite(&if_stmt.body, environment, signature);
            }
            Ok(DslCondition::False) => {}
            Ok(DslCondition::Unknown) => {
                return DslControlFlow::Return(DslOutcome::Value(DslValue::Unknown));
            }
        }
        for clause in &if_stmt.elif_else_clauses {
            match &clause.test {
                Some(test) => match self.evaluate_condition(test, environment) {
                    Err(error) => return DslControlFlow::Return(DslOutcome::Invalid(error)),
                    Ok(DslCondition::True) => {
                        return self.evaluate_suite(&clause.body, environment, signature);
                    }
                    Ok(DslCondition::False) => {}
                    Ok(DslCondition::Unknown) => {
                        return DslControlFlow::Return(DslOutcome::Value(DslValue::Unknown));
                    }
                },
                None => return self.evaluate_suite(&clause.body, environment, signature),
            }
        }
        DslControlFlow::Continue
    }

    fn expression_kind(&self, expression: &Expr) -> TypeShapeDslExpressionKind {
        self.expressions
            .iter()
            .find_map(|metadata| {
                (metadata.range == expression.range()).then(|| metadata.kind.clone())
            })
            .expect("validated DSL value expression must have validation metadata")
    }

    fn evaluate_expression(&self, expression: &Expr, environment: &DslEnvironment) -> DslOutcome {
        match self.expression_kind(expression) {
            TypeShapeDslExpressionKind::DimensionSlot { slot, .. } => {
                DslOutcome::Value(environment.value(slot).clone())
            }
            TypeShapeDslExpressionKind::DimensionLiteral(literal) => literal
                .map_or(DslOutcome::Value(DslValue::Unknown), |literal| {
                    DslOutcome::Value(DslValue::Dimension(Int::Literal(literal)))
                }),
            TypeShapeDslExpressionKind::IntTupleIndex { shape, index, .. } => {
                let Some(index) = index else {
                    return DslOutcome::Value(DslValue::Unknown);
                };
                let shape = match environment.value(shape) {
                    DslValue::Shape(shape) => shape,
                    DslValue::Unknown => return DslOutcome::Value(DslValue::Unknown),
                    _ => {
                        unreachable!("validated IntTuple index parameter evaluates to a shape")
                    }
                };
                let IntTupleView::Concrete(shape) = shape.view() else {
                    return DslOutcome::Value(DslValue::Unknown);
                };
                let length = shape.len() as i128;
                let index = i128::from(index);
                let index = if index < 0 { index + length } else { index };
                if index < 0 || index >= length {
                    return DslOutcome::Invalid(ShapeError::ShapeComputation {
                        message: "IntTuple index out of bounds".to_owned(),
                    });
                }
                DslOutcome::Value(DslValue::Dimension(shape[index as usize].clone()))
            }
            TypeShapeDslExpressionKind::DimensionTuple => {
                let Expr::Tuple(tuple) = expression else {
                    unreachable!("validated dimension tuple expression is a tuple")
                };
                let mut dimensions = Vec::with_capacity(tuple.elts.len());
                let mut unknown = false;
                for element in &tuple.elts {
                    match self.evaluate_expression(element, environment) {
                        DslOutcome::Value(DslValue::Dimension(dimension)) => {
                            dimensions.push(dimension)
                        }
                        DslOutcome::Value(DslValue::Unknown) => unknown = true,
                        DslOutcome::ExplicitGradual => {
                            unreachable!("validated value expression cannot return gradual")
                        }
                        invalid @ DslOutcome::Invalid(_) => return invalid,
                        DslOutcome::Value(_) => {
                            unreachable!("validated IntTuple element produces an Int value")
                        }
                    }
                }
                if unknown {
                    DslOutcome::Value(DslValue::Unknown)
                } else {
                    DslOutcome::Value(DslValue::DimensionTuple(dimensions))
                }
            }
            TypeShapeDslExpressionKind::IntTupleConstructor => {
                let Expr::Call(call) = expression else {
                    unreachable!("validated IntTuple constructor expression is a call")
                };
                match self.evaluate_expression(&call.arguments.args[0], environment) {
                    DslOutcome::Value(DslValue::DimensionTuple(dimensions)) => {
                        DslOutcome::Value(DslValue::Shape(IntTuple::new(dimensions)))
                    }
                    DslOutcome::Value(DslValue::Unknown) => DslOutcome::Value(DslValue::Unknown),
                    DslOutcome::ExplicitGradual => {
                        unreachable!("validated value expression cannot return gradual")
                    }
                    invalid @ DslOutcome::Invalid(_) => invalid,
                    DslOutcome::Value(_) => {
                        unreachable!("validated IntTuple constructor receives dimensions")
                    }
                }
            }
            TypeShapeDslExpressionKind::Slot(slot) => {
                DslOutcome::Value(environment.value(slot).clone())
            }
        }
    }

    fn evaluate_condition(
        &self,
        condition: &Expr,
        environment: &DslEnvironment,
    ) -> Result<DslCondition, ShapeError> {
        if let Expr::BoolOp(bool_op) = condition {
            if bool_op.op != BoolOp::And {
                unreachable!("validated boolean condition uses only `and`");
            }
            let mut unknown = false;
            for value in &bool_op.values {
                let value = match self.evaluate_condition(value, environment) {
                    Ok(value) => value,
                    // An unknown prefix may short-circuit before this operand for a concrete
                    // instantiation, so an error here is not deterministic.
                    Err(_error) if unknown => return Ok(DslCondition::Unknown),
                    Err(error) => return Err(error),
                };
                match value {
                    DslCondition::False => return Ok(DslCondition::False),
                    DslCondition::Unknown => unknown = true,
                    DslCondition::True => {}
                }
            }
            return Ok(if unknown {
                DslCondition::Unknown
            } else {
                DslCondition::True
            });
        }

        let kind = self
            .conditions
            .iter()
            .find_map(|metadata| {
                (metadata.range == condition.range()).then(|| metadata.kind.clone())
            })
            .expect("validated atomic condition must have validation metadata");
        Ok(match kind {
            TypeShapeDslConditionKind::IsConcreteInt { slot, .. } => {
                match environment.value(slot) {
                    DslValue::Dimension(Int::Literal(_)) => DslCondition::True,
                    // Symbolic and explicit-gradual `Int` values are definitively non-concrete.
                    DslValue::Dimension(_) => DslCondition::False,
                    // An admitted argument we cannot read as an `Int` is gradual, so it must fall
                    // back rather than take the false branch and produce a precise result.
                    DslValue::Unknown => DslCondition::Unknown,
                    _ => unreachable!("validated is_concrete_int operand is an Int dimension"),
                }
            }
            TypeShapeDslConditionKind::LessThan(left, right) => {
                if left == right {
                    DslCondition::False
                } else {
                    match (environment.dimension(left), environment.dimension(right)) {
                        (Some(left), Some(right)) if left == right && !matches!(left, Int::Int) => {
                            DslCondition::False
                        }
                        (Some(Int::Literal(left)), Some(Int::Literal(right))) if left < right => {
                            DslCondition::True
                        }
                        (Some(Int::Literal(_)), Some(Int::Literal(_))) => DslCondition::False,
                        _ => DslCondition::Unknown,
                    }
                }
            }
            TypeShapeDslConditionKind::Equal(left, right) => {
                if left == right {
                    DslCondition::True
                } else {
                    match (environment.dimension(left), environment.dimension(right)) {
                        (Some(Int::Literal(left)), Some(Int::Literal(right))) => {
                            if left == right {
                                DslCondition::True
                            } else {
                                DslCondition::False
                            }
                        }
                        (Some(Int::Int), _) | (_, Some(Int::Int)) => DslCondition::Unknown,
                        (Some(left), Some(right)) if left == right => DslCondition::True,
                        _ => DslCondition::Unknown,
                    }
                }
            }
            TypeShapeDslConditionKind::FlagIntLessThanLiteral { parameter, literal } => {
                match (environment.flag_int(parameter), literal) {
                    (Some(left), Some(right)) if left < right => DslCondition::True,
                    (Some(_), Some(_)) => DslCondition::False,
                    _ => DslCondition::Unknown,
                }
            }
            TypeShapeDslConditionKind::LengthEqualLiteral { shape, literal } => {
                let shape = match environment.value(shape) {
                    DslValue::Shape(shape) => shape,
                    DslValue::Unknown => return Ok(DslCondition::Unknown),
                    _ => unreachable!("validated len operand evaluates to an IntTuple"),
                };
                let IntTupleView::Concrete(shape) = shape.view() else {
                    return Ok(DslCondition::Unknown);
                };
                match literal {
                    Some(literal) if i128::from(literal) == shape.len() as i128 => {
                        DslCondition::True
                    }
                    Some(_) => DslCondition::False,
                    None => DslCondition::Unknown,
                }
            }
        })
    }
}

fn evaluate_broadcast(left: &DslValue, right: &DslValue) -> DslOutcome {
    let (DslValue::Shape(left), DslValue::Shape(right)) = (left, right) else {
        if matches!(left, DslValue::Unknown) || matches!(right, DslValue::Unknown) {
            return DslOutcome::Value(DslValue::Unknown);
        }
        unreachable!("validated broadcast operands evaluate to shapes")
    };
    match broadcast_shapes(left, right) {
        Ok(shape) => DslOutcome::Value(DslValue::Shape(shape)),
        Err(error) => DslOutcome::Invalid(error),
    }
}

fn literal_offset(value: &Int) -> Option<i64> {
    match value {
        Int::Literal(value) => Some(*value),
        Int::Add(left, right) => literal_offset(left)?.checked_add(literal_offset(right)?),
        Int::Sub(left, right) => literal_offset(left)?.checked_sub(literal_offset(right)?),
        _ => Some(0),
    }
}

fn flag_int_literal(ty: &Type) -> Option<i64> {
    if !FlagDomain::of(FlagMember::Int).accepts(ty) {
        return None;
    }
    // Symbolic shape integers satisfy `Flag[int]`, but DSL flag operations inspect only concrete
    // runtime values. Generic substitution does not re-evaluate a call that already fell back.
    match ty {
        Type::Int(Int::Literal(value)) => Some(*value),
        Type::Literal(literal) => match &literal.value {
            Lit::Int(value) => value.as_i64(),
            _ => unreachable!("`Flag[int]` accepts only integer literals"),
        },
        _ => None,
    }
}

fn lower_parameter(ty: &Type, domain: TypeShapeDslInputDomain) -> DslValue {
    match domain {
        TypeShapeDslInputDomain::Value(domain) => DslValue::from_type(ty, domain),
        TypeShapeDslInputDomain::Flag(domain) => {
            if domain != FlagDomain::of(FlagMember::Int) {
                return DslValue::Unknown;
            }
            flag_int_literal(ty).map_or(DslValue::Unknown, DslValue::FlagInt)
        }
    }
}

impl DslValue {
    fn from_type(ty: &Type, domain: TypeShapeDslDomain) -> Self {
        match domain {
            TypeShapeDslDomain::Int => Int::from_type(ty).map_or(Self::Unknown, Self::Dimension),
            TypeShapeDslDomain::IntTuple => Self::from_shape_type(ty),
        }
    }

    fn from_shape_type(ty: &Type) -> Self {
        IntTuple::from_shape_arg_type(ty)
            .or_else(|| tuple_carrier_to_shape(ty))
            .map_or(Self::Unknown, Self::Shape)
    }

    fn into_type(self) -> Type {
        match self {
            Self::Dimension(value) => Type::Int(value),
            Self::Shape(value) => value.to_shape_arg_type(),
            Self::Unknown => unreachable!("unknown DSL values project through the fallback"),
            Self::FlagInt(_) | Self::DimensionTuple(_) => {
                unreachable!("intermediate DSL values cannot be returned directly")
            }
        }
    }
}

impl DslEnvironment {
    fn value(&self, slot: usize) -> &DslValue {
        &self.slots[slot]
    }

    fn dimension(&self, parameter: usize) -> Option<&Int> {
        match &self.slots[parameter] {
            DslValue::Dimension(value) => Some(value),
            DslValue::Unknown => None,
            _ => unreachable!("validated dimension parameter has the Int domain"),
        }
    }

    fn flag_int(&self, parameter: usize) -> Option<i64> {
        match &self.slots[parameter] {
            DslValue::FlagInt(value) => Some(*value),
            DslValue::Unknown => None,
            _ => unreachable!("validated Flag[int] parameter has the Flag[int] value domain"),
        }
    }

    fn assign(&mut self, slot: usize, value: DslValue) {
        assert!(
            slot >= self.parameter_count,
            "validated assignment cannot target a parameter slot"
        );
        self.slots[slot] = value;
    }
}
