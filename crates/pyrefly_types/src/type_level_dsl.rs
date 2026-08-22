/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::Ordering;
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

/// The input domains accepted by type-level shape DSL functions.
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

/// A syntax-validated DSL definition paired with its resolved parameter domains.
#[derive(Debug, Clone, PartialEq, Eq, TypeEq, PartialOrd, Ord, Hash)]
pub struct ResolvedTypeShapeDslFunction {
    definition: Arc<ValidatedTypeShapeDslFunction>,
    parameter_domains: Vec<TypeShapeDslInputDomain>,
    result_domain: TypeShapeDslDomain,
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

/// What a validated DSL body returns. Resolving this depends on more than the AST, so it
/// participates in `ValidatedTypeShapeDslFunction` identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TypeShapeDslReturnKind {
    Parameter(usize),
    Broadcast {
        left: usize,
        right: usize,
    },
    IntFlagArithmetic {
        left: usize,
        op: TypeShapeDslArithmeticOp,
        right: usize,
    },
    Expression,
    Invalid,
    Gradual(TypeShapeDslDomain),
}

/// Arithmetic supported between an `Int` value and a `Flag[int]` value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TypeShapeDslArithmeticOp {
    Add,
    Subtract,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeShapeDslReturn {
    statement_range: TextRange,
    value_range: TextRange,
    kind: TypeShapeDslReturnKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TypeShapeDslExpressionKind {
    DimensionParameter(usize),
    DimensionLiteral(Option<i64>),
    IntTupleIndex { shape: usize, index: Option<i64> },
    DimensionTuple,
    IntTupleConstructor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeShapeDslExpression {
    range: TextRange,
    kind: TypeShapeDslExpressionKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TypeShapeDslConditionKind {
    Equal(usize, usize),
    IsConcreteInt(usize),
    LessThan(usize, usize),
    FlagIntLessThanLiteral {
        parameter: usize,
        literal: Option<i64>,
    },
    LengthEqualLiteral {
        shape: usize,
        literal: Option<i64>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeShapeDslCondition {
    range: TextRange,
    kind: TypeShapeDslConditionKind,
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum DslStatementFlow {
    Continue,
    Return,
}

struct DslValidator<'a, F> {
    parameters: &'a Parameters,
    resolve_intrinsic: &'a F,
    returns: Vec<TypeShapeDslReturn>,
    conditions: Vec<TypeShapeDslCondition>,
    expressions: Vec<TypeShapeDslExpression>,
}

impl<'a, F: Fn(&Expr) -> Option<TypeShapeDslIntrinsic>> DslValidator<'a, F> {
    fn new(parameters: &'a Parameters, resolve_intrinsic: &'a F) -> Self {
        Self {
            parameters,
            resolve_intrinsic,
            returns: Vec::new(),
            conditions: Vec::new(),
            expressions: Vec::new(),
        }
    }

    fn validate_block(
        &mut self,
        block: &[Stmt],
    ) -> Result<DslStatementFlow, TypeShapeDslDefinitionError> {
        let mut flow = DslStatementFlow::Continue;
        for statement in block {
            if flow == DslStatementFlow::Return {
                return Err(TypeShapeDslDefinitionError {
                    range: statement.range(),
                    message: "statement is unreachable",
                });
            }
            flow = self.validate_statement(statement)?;
        }
        Ok(flow)
    }

    fn validate_statement(
        &mut self,
        statement: &Stmt,
    ) -> Result<DslStatementFlow, TypeShapeDslDefinitionError> {
        match statement {
            Stmt::Return(return_stmt) => {
                self.validate_return(return_stmt)?;
                Ok(DslStatementFlow::Return)
            }
            Stmt::If(if_stmt) => {
                self.validate_if(if_stmt)?;
                Ok(DslStatementFlow::Continue)
            }
            _ => Err(TypeShapeDslDefinitionError {
                range: statement.range(),
                message: "body supports only `if` and `return`",
            }),
        }
    }

    fn validate_dimension_expression(
        &mut self,
        expression: &Expr,
    ) -> Result<(), TypeShapeDslDefinitionError> {
        let kind = match expression {
            Expr::Name(_) => {
                let Some(parameter) = parameter_index(self.parameters, expression) else {
                    return Err(TypeShapeDslDefinitionError {
                        range: expression.range(),
                        message: "dimension name must match a parameter name",
                    });
                };
                TypeShapeDslExpressionKind::DimensionParameter(parameter)
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
                let Some(shape) = parameter_index(self.parameters, &subscript.value) else {
                    return Err(TypeShapeDslDefinitionError {
                        range: subscript.value.range(),
                        message: "indexed value must name an `IntTuple` parameter",
                    });
                };
                let Ok(index) = integer_literal(&subscript.slice).into_value() else {
                    return Err(TypeShapeDslDefinitionError {
                        range: subscript.slice.range(),
                        message: "`IntTuple` index must be an integer literal",
                    });
                };
                TypeShapeDslExpressionKind::IntTupleIndex { shape, index }
            }
            _ => {
                return Err(TypeShapeDslDefinitionError {
                    range: expression.range(),
                    message: "`IntTuple` elements must be Int parameters, integer literals, or indexed IntTuple parameters",
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
    ) -> Result<(), TypeShapeDslDefinitionError> {
        if call.arguments.args.len() != 1
            || !call.arguments.keywords.is_empty()
            || matches!(call.arguments.args.first(), Some(Expr::Starred(_)))
        {
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
            self.validate_dimension_expression(element)?;
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

    fn validate_return(
        &mut self,
        return_stmt: &StmtReturn,
    ) -> Result<(), TypeShapeDslDefinitionError> {
        let kind = match return_stmt.value.as_deref() {
            Some(returned @ Expr::Attribute(_))
                if matches!(
                    (self.resolve_intrinsic)(returned),
                    Some(TypeShapeDslIntrinsic::Gradual(_))
                ) =>
            {
                return Err(TypeShapeDslDefinitionError {
                    range: returned.range(),
                    message: "gradual return must be called",
                });
            }
            Some(returned @ Expr::Name(returned_name)) => {
                if matches!(
                    (self.resolve_intrinsic)(returned),
                    Some(TypeShapeDslIntrinsic::Gradual(_))
                ) {
                    return Err(TypeShapeDslDefinitionError {
                        range: returned_name.range,
                        message: "gradual return must be called",
                    });
                }
                let Some(index) = parameter_index(self.parameters, returned) else {
                    return Err(TypeShapeDslDefinitionError {
                        range: returned_name.range,
                        message: "returned name must match a parameter name",
                    });
                };
                TypeShapeDslReturnKind::Parameter(index)
            }
            Some(Expr::Call(call)) => match (self.resolve_intrinsic)(&call.func) {
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
                    let (Some(left), Some(right)) = (
                        parameter_index(self.parameters, &call.arguments.args[0]),
                        parameter_index(self.parameters, &call.arguments.args[1]),
                    ) else {
                        return Err(TypeShapeDslDefinitionError {
                            range: call.arguments.range,
                            message: "`broadcast` arguments must be bare parameter names",
                        });
                    };
                    TypeShapeDslReturnKind::Broadcast { left, right }
                }
                Some(TypeShapeDslIntrinsic::IntTuple) => {
                    self.validate_int_tuple_constructor(call)?;
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
                Some(TypeShapeDslIntrinsic::IsConcreteInt | TypeShapeDslIntrinsic::Len) | None => {
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
                .expect("validated return must have a value")
                .range(),
            kind,
        });
        Ok(())
    }

    fn validate_if(&mut self, if_stmt: &StmtIf) -> Result<(), TypeShapeDslDefinitionError> {
        if !if_stmt.elif_else_clauses.is_empty() {
            return Err(TypeShapeDslDefinitionError {
                range: if_stmt.range,
                message: "does not support `else` or `elif`",
            });
        }
        self.validate_condition(&if_stmt.test)?;
        // Without an `else`, this statement may fall through even if its body returns.
        self.validate_block(&if_stmt.body)?;
        Ok(())
    }

    fn validate_condition(&mut self, condition: &Expr) -> Result<(), TypeShapeDslDefinitionError> {
        if let Expr::BoolOp(bool_op) = condition {
            if bool_op.op != BoolOp::And {
                return Err(TypeShapeDslDefinitionError {
                    range: bool_op.range,
                    message: "condition supports only the boolean operator `and`",
                });
            }
            for value in &bool_op.values {
                self.validate_condition(value)?;
            }
            return Ok(());
        }

        let kind = self.validate_atomic_condition(condition)?;
        self.conditions.push(TypeShapeDslCondition {
            range: condition.range(),
            kind,
        });
        Ok(())
    }

    fn validate_atomic_condition(
        &self,
        condition: &Expr,
    ) -> Result<TypeShapeDslConditionKind, TypeShapeDslDefinitionError> {
        match condition {
            Expr::Compare(compare) => {
                if compare.ops.len() == 1
                    && compare.ops[0] == CmpOp::Eq
                    && compare.comparators.len() == 1
                    && let Expr::Call(call) = compare.left.as_ref()
                    && (self.resolve_intrinsic)(&call.func) == Some(TypeShapeDslIntrinsic::Len)
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
                    let Some(shape) = parameter_index(self.parameters, &call.arguments.args[0])
                    else {
                        return Err(TypeShapeDslDefinitionError {
                            range: call.arguments.args[0].range(),
                            message: "`len` argument must name an `IntTuple` parameter",
                        });
                    };
                    let Ok(literal) = integer_literal(&compare.comparators[0]).into_value() else {
                        return Err(TypeShapeDslDefinitionError {
                            range: compare.comparators[0].range(),
                            message: "`len` comparison requires an integer literal",
                        });
                    };
                    return Ok(TypeShapeDslConditionKind::LengthEqualLiteral { shape, literal });
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
                let Some(left) = parameter_index(self.parameters, &compare.left) else {
                    return Err(TypeShapeDslDefinitionError {
                        range: compare.left.range(),
                        message: "left comparison operand must name a parameter",
                    });
                };
                let right = &compare.comparators[0];
                Ok(
                    match (compare.ops[0], parameter_index(self.parameters, right)) {
                        (CmpOp::Eq, Some(right)) => TypeShapeDslConditionKind::Equal(left, right),
                        (CmpOp::Lt, Some(right)) => {
                            TypeShapeDslConditionKind::LessThan(left, right)
                        }
                        (CmpOp::Lt, None) => {
                            let literal = match integer_literal(right) {
                                IntegerLiteral::NotLiteral => {
                                    return Err(TypeShapeDslDefinitionError {
                                        range: right.range(),
                                        message: "right `<` operand must name a parameter or be an integer literal",
                                    });
                                }
                                IntegerLiteral::Unrepresentable => None,
                                IntegerLiteral::Value(value) => Some(value),
                            };
                            TypeShapeDslConditionKind::FlagIntLessThanLiteral {
                                parameter: left,
                                literal,
                            }
                        }
                        (CmpOp::Eq, None) => {
                            return Err(TypeShapeDslDefinitionError {
                                range: right.range(),
                                message: "condition operands must name parameters",
                            });
                        }
                        _ => unreachable!("validated comparison operator is equality or less-than"),
                    },
                )
            }
            Expr::Call(call)
                if (self.resolve_intrinsic)(&call.func)
                    == Some(TypeShapeDslIntrinsic::IsConcreteInt) =>
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
                let Some(parameter) = parameter_index(self.parameters, &call.arguments.args[0])
                else {
                    return Err(TypeShapeDslDefinitionError {
                        range: call.arguments.args[0].range(),
                        message: "`is_concrete_int` argument must name a parameter",
                    });
                };
                Ok(TypeShapeDslConditionKind::IsConcreteInt(parameter))
            }
            _ => Err(TypeShapeDslDefinitionError {
                range: condition.range(),
                message: "condition supports only `is_concrete_int`, `and`, `==`, and `<`",
            }),
        }
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

    /// Validate the body, which so far may only return a parameter or a gradual intrinsic.
    pub fn validate_body(
        &self,
        resolve_intrinsic: impl Fn(&Expr) -> Option<TypeShapeDslIntrinsic>,
    ) -> Result<ValidatedTypeShapeDslFunction, TypeShapeDslDefinitionError> {
        let mut validator = DslValidator::new(&self.definition.parameters, &resolve_intrinsic);
        if validator.validate_block(&self.definition.body)? != DslStatementFlow::Return {
            return Err(TypeShapeDslDefinitionError {
                range: self.definition.name.range(),
                message: "every control-flow path must return",
            });
        }
        Ok(ValidatedTypeShapeDslFunction {
            parsed: self.clone(),
            returns: validator.returns,
            conditions: validator.conditions,
            expressions: validator.expressions,
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
                    .map(|x| (offsets(x.statement_range), offsets(x.value_range), x.kind))
                    .cmp(
                        other
                            .returns
                            .iter()
                            .map(|x| (offsets(x.statement_range), offsets(x.value_range), x.kind)),
                    )
            })
            .then_with(|| {
                self.conditions
                    .iter()
                    .map(|x| (offsets(x.range), x.kind))
                    .cmp(other.conditions.iter().map(|x| (offsets(x.range), x.kind)))
            })
            .then_with(|| {
                self.expressions
                    .iter()
                    .map(|x| (offsets(x.range), x.kind))
                    .cmp(other.expressions.iter().map(|x| (offsets(x.range), x.kind)))
            })
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
    fn parameter_count(&self) -> usize {
        self.parsed.parameter_count()
    }

    pub fn name(&self) -> &Name {
        self.parsed.name()
    }

    pub fn parameter_name(&self, index: usize) -> &Name {
        self.parsed.parameter_name(index)
    }

    pub fn returns(&self) -> impl Iterator<Item = TypeShapeDslReturn> + '_ {
        self.returns.iter().copied()
    }

    pub fn conditions(&self) -> impl Iterator<Item = TypeShapeDslCondition> + '_ {
        self.conditions.iter().copied()
    }

    pub fn expressions(&self) -> impl Iterator<Item = TypeShapeDslExpression> + '_ {
        self.expressions.iter().copied()
    }
}

impl TypeShapeDslReturn {
    pub fn range(self) -> TextRange {
        self.value_range
    }

    pub fn kind(self) -> TypeShapeDslReturnKind {
        self.kind
    }
}

impl TypeShapeDslCondition {
    pub fn range(self) -> TextRange {
        self.range
    }

    pub fn kind(self) -> TypeShapeDslConditionKind {
        self.kind
    }
}

impl TypeShapeDslExpression {
    pub fn range(self) -> TextRange {
        self.range
    }

    pub fn kind(self) -> TypeShapeDslExpressionKind {
        self.kind
    }
}

impl ResolvedTypeShapeDslFunction {
    pub fn try_new(
        definition: Arc<ValidatedTypeShapeDslFunction>,
        parameter_domains: Vec<TypeShapeDslInputDomain>,
        result_domain: TypeShapeDslDomain,
    ) -> Option<Self> {
        if definition.parameter_count() != parameter_domains.len() {
            return None;
        }
        let resolved = Self {
            definition,
            parameter_domains,
            result_domain,
        };
        if !resolved
            .definition
            .returns()
            .all(|return_| resolved.return_domain(return_.kind()) == Some(result_domain))
        {
            return None;
        }
        Some(resolved)
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

    fn return_domain(&self, kind: TypeShapeDslReturnKind) -> Option<TypeShapeDslDomain> {
        match kind {
            TypeShapeDslReturnKind::Parameter(index) => match self.parameter_domains[index] {
                TypeShapeDslInputDomain::Value(domain) => Some(domain),
                TypeShapeDslInputDomain::Flag(_) => None,
            },
            TypeShapeDslReturnKind::Broadcast { .. } => Some(TypeShapeDslDomain::IntTuple),
            TypeShapeDslReturnKind::IntFlagArithmetic { .. } => Some(TypeShapeDslDomain::Int),
            TypeShapeDslReturnKind::Expression => Some(TypeShapeDslDomain::IntTuple),
            TypeShapeDslReturnKind::Invalid => Some(self.result_domain),
            TypeShapeDslReturnKind::Gradual(domain) => Some(domain),
        }
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
    Int(Int),
    IntTuple(IntTuple),
    DimensionTuple(Vec<Int>),
}

enum DslEvaluation {
    Value(DslValue),
    ExplicitGradual,
    AutomaticFallback,
}

enum DslCondition {
    True,
    False,
    Unknown,
}

enum DslControlFlow {
    Continue,
    Return(DslEvaluation),
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
        match &self.function {
            TypeLevelDslFunction::Broadcast => {
                let [left, right] = self.args.as_slice() else {
                    unreachable!("native broadcast DSL calls are constructed with two arguments");
                };
                Ok(match evaluate_broadcast(left, right)? {
                    DslEvaluation::Value(value) => value.into_type(),
                    DslEvaluation::AutomaticFallback => self.fallback(),
                    DslEvaluation::ExplicitGradual => {
                        unreachable!("native broadcast does not return explicit gradual")
                    }
                })
            }
            TypeLevelDslFunction::UserDefined(function) => {
                Ok(match function.evaluate(&self.args)? {
                    DslEvaluation::Value(value) => value.into_type(),
                    DslEvaluation::ExplicitGradual | DslEvaluation::AutomaticFallback => {
                        self.fallback()
                    }
                })
            }
        }
    }
}

impl ResolvedTypeShapeDslFunction {
    fn evaluate(&self, args: &[Type]) -> Result<DslEvaluation, ShapeError> {
        assert_eq!(
            args.len(),
            self.parameter_domains.len(),
            "type-level DSL values must align with resolved parameters"
        );
        match self.evaluate_block(&self.definition.parsed.definition.body, args)? {
            DslControlFlow::Return(result) => Ok(result),
            DslControlFlow::Continue => {
                unreachable!("validated type-level DSL function cannot fall through")
            }
        }
    }

    fn evaluate_block(&self, block: &[Stmt], args: &[Type]) -> Result<DslControlFlow, ShapeError> {
        for statement in block {
            match statement {
                Stmt::Return(return_stmt) => {
                    let kind = self
                        .definition
                        .returns
                        .iter()
                        .find_map(|return_| {
                            (return_.statement_range == return_stmt.range).then_some(return_.kind)
                        })
                        .expect("validated return statement must have validation metadata");
                    return Ok(DslControlFlow::Return(match kind {
                        TypeShapeDslReturnKind::Parameter(index) => {
                            let TypeShapeDslInputDomain::Value(domain) =
                                self.parameter_domains[index]
                            else {
                                unreachable!("resolved Flag parameters cannot be returned")
                            };
                            assert_eq!(
                                domain,
                                self.result_domain(),
                                "resolved parameter return domain must match the result domain"
                            );
                            DslValue::from_type(&args[index], domain)
                                .map_or(DslEvaluation::AutomaticFallback, DslEvaluation::Value)
                        }
                        TypeShapeDslReturnKind::IntFlagArithmetic { left, op, right } => {
                            assert_eq!(
                                self.result_domain(),
                                TypeShapeDslDomain::Int,
                                "resolved arithmetic return must produce Int"
                            );
                            assert_eq!(
                                self.parameter_domains[left],
                                TypeShapeDslInputDomain::Value(TypeShapeDslDomain::Int),
                                "resolved arithmetic left operand must be Int"
                            );
                            assert_eq!(
                                self.parameter_domains[right],
                                TypeShapeDslInputDomain::Flag(FlagDomain::of(FlagMember::Int)),
                                "resolved arithmetic right operand must be Flag[int]"
                            );
                            let (Some(left), Some(right)) =
                                (Int::from_type(&args[left]), flag_int_literal(&args[right]))
                            else {
                                return Ok(DslControlFlow::Return(
                                    DslEvaluation::AutomaticFallback,
                                ));
                            };
                            let result = match op {
                                TypeShapeDslArithmeticOp::Add => {
                                    Int::add(Type::Int(left), Type::Int(Int::Literal(right)))
                                }
                                TypeShapeDslArithmeticOp::Subtract => {
                                    Int::sub(Type::Int(left), Type::Int(Int::Literal(right)))
                                }
                            };
                            Int::from_type(&canonicalize(Type::Int(result)))
                                .map_or(DslEvaluation::AutomaticFallback, |result| {
                                    DslEvaluation::Value(DslValue::Int(result))
                                })
                        }
                        TypeShapeDslReturnKind::Broadcast { left, right } => {
                            assert_eq!(
                                self.result_domain(),
                                TypeShapeDslDomain::IntTuple,
                                "resolved broadcast return must produce IntTuple"
                            );
                            assert_eq!(
                                self.parameter_domains[left],
                                TypeShapeDslInputDomain::Value(TypeShapeDslDomain::IntTuple),
                                "resolved broadcast left operand must be IntTuple"
                            );
                            assert_eq!(
                                self.parameter_domains[right],
                                TypeShapeDslInputDomain::Value(TypeShapeDslDomain::IntTuple),
                                "resolved broadcast right operand must be IntTuple"
                            );
                            evaluate_broadcast(&args[left], &args[right])?
                        }
                        TypeShapeDslReturnKind::Expression => {
                            let expression = return_stmt
                                .value
                                .as_deref()
                                .expect("validated expression return has a value");
                            self.evaluate_expression(expression, args)?
                        }
                        TypeShapeDslReturnKind::Invalid => {
                            let Some(Expr::Call(call)) = return_stmt.value.as_deref() else {
                                unreachable!("validated Invalid return is a call")
                            };
                            let Some(Expr::StringLiteral(message)) = call.arguments.args.first()
                            else {
                                unreachable!("validated Invalid return has a string message")
                            };
                            return Err(ShapeError::ShapeComputation {
                                message: message.value.to_str().to_owned(),
                            });
                        }
                        TypeShapeDslReturnKind::Gradual(_) => DslEvaluation::ExplicitGradual,
                    }));
                }
                Stmt::If(if_stmt) => match self.evaluate_condition(&if_stmt.test, args) {
                    DslCondition::True => match self.evaluate_block(&if_stmt.body, args)? {
                        DslControlFlow::Continue => {}
                        result @ DslControlFlow::Return(_) => return Ok(result),
                    },
                    DslCondition::False => {}
                    DslCondition::Unknown => {
                        return Ok(DslControlFlow::Return(DslEvaluation::AutomaticFallback));
                    }
                },
                _ => unreachable!("validated type-level DSL block contains only if and return"),
            }
        }
        Ok(DslControlFlow::Continue)
    }

    fn expression_kind(&self, expression: &Expr) -> TypeShapeDslExpressionKind {
        self.definition
            .expressions
            .iter()
            .find_map(|metadata| (metadata.range == expression.range()).then_some(metadata.kind))
            .expect("validated DSL value expression must have validation metadata")
    }

    fn evaluate_expression(
        &self,
        expression: &Expr,
        args: &[Type],
    ) -> Result<DslEvaluation, ShapeError> {
        Ok(match self.expression_kind(expression) {
            TypeShapeDslExpressionKind::DimensionParameter(parameter) => {
                DslValue::from_type(&args[parameter], TypeShapeDslDomain::Int)
                    .map_or(DslEvaluation::AutomaticFallback, DslEvaluation::Value)
            }
            TypeShapeDslExpressionKind::DimensionLiteral(literal) => literal
                .map_or(DslEvaluation::AutomaticFallback, |literal| {
                    DslEvaluation::Value(DslValue::Int(Int::Literal(literal)))
                }),
            TypeShapeDslExpressionKind::IntTupleIndex { shape, index } => {
                let Some(index) = index else {
                    return Ok(DslEvaluation::AutomaticFallback);
                };
                let Some(shape) = IntTuple::from_shape_arg_type(&args[shape]) else {
                    return Ok(DslEvaluation::AutomaticFallback);
                };
                let IntTupleView::Concrete(shape) = shape.view() else {
                    return Ok(DslEvaluation::AutomaticFallback);
                };
                let length = shape.len() as i128;
                let index = i128::from(index);
                let index = if index < 0 { index + length } else { index };
                if index < 0 || index >= length {
                    return Err(ShapeError::ShapeComputation {
                        message: "IntTuple index out of bounds".to_owned(),
                    });
                }
                DslEvaluation::Value(DslValue::Int(shape[index as usize].clone()))
            }
            TypeShapeDslExpressionKind::DimensionTuple => {
                let Expr::Tuple(tuple) = expression else {
                    unreachable!("validated dimension tuple expression is a tuple")
                };
                let mut dimensions = Vec::with_capacity(tuple.elts.len());
                let mut unknown = false;
                for element in &tuple.elts {
                    match self.evaluate_expression(element, args)? {
                        DslEvaluation::Value(DslValue::Int(dimension)) => {
                            dimensions.push(dimension)
                        }
                        DslEvaluation::AutomaticFallback | DslEvaluation::ExplicitGradual => {
                            unknown = true;
                        }
                        DslEvaluation::Value(_) => {
                            unreachable!("validated IntTuple element produces an Int value")
                        }
                    }
                }
                if unknown {
                    DslEvaluation::AutomaticFallback
                } else {
                    DslEvaluation::Value(DslValue::DimensionTuple(dimensions))
                }
            }
            TypeShapeDslExpressionKind::IntTupleConstructor => {
                let Expr::Call(call) = expression else {
                    unreachable!("validated IntTuple constructor expression is a call")
                };
                match self.evaluate_expression(&call.arguments.args[0], args)? {
                    DslEvaluation::Value(DslValue::DimensionTuple(dimensions)) => {
                        DslEvaluation::Value(DslValue::IntTuple(IntTuple::new(dimensions)))
                    }
                    DslEvaluation::AutomaticFallback | DslEvaluation::ExplicitGradual => {
                        DslEvaluation::AutomaticFallback
                    }
                    DslEvaluation::Value(_) => {
                        unreachable!("validated IntTuple constructor receives dimensions")
                    }
                }
            }
        })
    }

    fn evaluate_condition(&self, condition: &Expr, args: &[Type]) -> DslCondition {
        if let Expr::BoolOp(bool_op) = condition {
            let mut result = DslCondition::True;
            for value in &bool_op.values {
                match self.evaluate_condition(value, args) {
                    DslCondition::False => return DslCondition::False,
                    DslCondition::Unknown => result = DslCondition::Unknown,
                    DslCondition::True => {}
                }
            }
            return result;
        }

        let kind = self
            .definition
            .conditions
            .iter()
            .find_map(|condition_metadata| {
                (condition_metadata.range == condition.range()).then_some(condition_metadata.kind)
            })
            .expect("validated atomic condition must have validation metadata");
        match kind {
            TypeShapeDslConditionKind::IsConcreteInt(parameter) => {
                match Int::from_type(&args[parameter]) {
                    Some(Int::Literal(_)) => DslCondition::True,
                    Some(_) => DslCondition::False,
                    None => DslCondition::Unknown,
                }
            }
            TypeShapeDslConditionKind::LessThan(left, right) => {
                if left == right {
                    DslCondition::False
                } else {
                    match (Int::from_type(&args[left]), Int::from_type(&args[right])) {
                        (Some(Int::Int), _) | (_, Some(Int::Int)) => DslCondition::Unknown,
                        (Some(left), Some(right)) if left == right => DslCondition::False,
                        (Some(Int::Literal(left)), Some(Int::Literal(right))) if left < right => {
                            DslCondition::True
                        }
                        (Some(Int::Literal(_)), Some(Int::Literal(_))) => DslCondition::False,
                        _ => DslCondition::Unknown,
                    }
                }
            }
            TypeShapeDslConditionKind::Equal(left, right) => {
                // Reflexive equality is true even when the parameter itself is gradual.
                if left == right {
                    DslCondition::True
                } else {
                    match (Int::from_type(&args[left]), Int::from_type(&args[right])) {
                        (Some(Int::Int), _) | (_, Some(Int::Int)) => DslCondition::Unknown,
                        (Some(left), Some(right)) if left == right => DslCondition::True,
                        (Some(Int::Literal(_)), Some(Int::Literal(_))) => DslCondition::False,
                        _ => DslCondition::Unknown,
                    }
                }
            }
            TypeShapeDslConditionKind::FlagIntLessThanLiteral { parameter, literal } => {
                match (flag_int_literal(&args[parameter]), literal) {
                    (Some(left), Some(right)) if left < right => DslCondition::True,
                    (Some(_), Some(_)) => DslCondition::False,
                    _ => DslCondition::Unknown,
                }
            }
            TypeShapeDslConditionKind::LengthEqualLiteral { shape, literal } => {
                let Some(shape) = IntTuple::from_shape_arg_type(&args[shape]) else {
                    return DslCondition::Unknown;
                };
                let IntTupleView::Concrete(shape) = shape.view() else {
                    return DslCondition::Unknown;
                };
                match literal {
                    Some(literal) if i128::from(literal) == shape.len() as i128 => {
                        DslCondition::True
                    }
                    Some(_) => DslCondition::False,
                    None => DslCondition::Unknown,
                }
            }
        }
    }
}

fn evaluate_broadcast(left: &Type, right: &Type) -> Result<DslEvaluation, ShapeError> {
    let Some(left) = IntTuple::from_shape_arg_type(left).or_else(|| tuple_carrier_to_shape(left))
    else {
        return Ok(DslEvaluation::AutomaticFallback);
    };
    let Some(right) =
        IntTuple::from_shape_arg_type(right).or_else(|| tuple_carrier_to_shape(right))
    else {
        return Ok(DslEvaluation::AutomaticFallback);
    };
    broadcast_shapes(&left, &right)
        .map(DslValue::IntTuple)
        .map(DslEvaluation::Value)
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
            Self::DimensionTuple(_) => {
                unreachable!("intermediate DSL values cannot be returned directly")
            }
        }
    }
}
