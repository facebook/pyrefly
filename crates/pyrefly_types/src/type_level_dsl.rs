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
use ruff_python_ast::CmpOp;
use ruff_python_ast::Expr;
use ruff_python_ast::Parameters;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::StmtIf;
use ruff_python_ast::StmtReturn;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

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

impl TypeShapeDslDomain {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Int => "Int",
            Self::IntTuple => "IntTuple",
        }
    }
}

/// A syntax-validated DSL definition paired with its resolved parameter domains.
#[derive(Debug, Clone, PartialEq, Eq, TypeEq, PartialOrd, Ord, Hash)]
pub struct ResolvedTypeShapeDslFunction {
    definition: Arc<ValidatedTypeShapeDslFunction>,
    parameter_domains: Vec<TypeShapeDslDomain>,
}

impl Visit<Type> for TypeShapeDslDomain {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

impl VisitMut<Type> for TypeShapeDslDomain {
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
    Gradual(TypeShapeDslDomain),
}

/// What a validated DSL body returns. Resolving this depends on more than the AST, so it
/// participates in `ValidatedTypeShapeDslFunction` identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TypeShapeDslReturnKind {
    Parameter(usize),
    Gradual(TypeShapeDslDomain),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeShapeDslReturn {
    statement_range: TextRange,
    value_range: TextRange,
    kind: TypeShapeDslReturnKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeShapeDslEquality {
    range: TextRange,
    left: usize,
    right: usize,
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum DslStatementFlow {
    Continue,
    Return,
}

struct DslValidator<'a, F> {
    parameters: &'a Parameters,
    resolve_intrinsic: &'a F,
    returns: Vec<TypeShapeDslReturn>,
    conditions: Vec<TypeShapeDslEquality>,
}

impl<'a, F: Fn(&Expr) -> Option<TypeShapeDslIntrinsic>> DslValidator<'a, F> {
    fn new(parameters: &'a Parameters, resolve_intrinsic: &'a F) -> Self {
        Self {
            parameters,
            resolve_intrinsic,
            returns: Vec::new(),
            conditions: Vec::new(),
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
            Some(Expr::Call(call))
                if let Some(TypeShapeDslIntrinsic::Gradual(domain)) =
                    (self.resolve_intrinsic)(&call.func) =>
            {
                if !call.arguments.args.is_empty() || !call.arguments.keywords.is_empty() {
                    return Err(TypeShapeDslDefinitionError {
                        range: call.arguments.range,
                        message: "gradual return does not accept arguments",
                    });
                }
                TypeShapeDslReturnKind::Gradual(domain)
            }
            _ => {
                return Err(TypeShapeDslDefinitionError {
                    range: return_stmt.range,
                    message: "return value must be a bare parameter name or a gradual return",
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
        let Expr::Compare(compare) = &*if_stmt.test else {
            return Err(TypeShapeDslDefinitionError {
                range: if_stmt.test.range(),
                message: "condition must be exactly `<Int parameter> == <Int parameter>`",
            });
        };
        if compare.ops.len() != 1 || compare.ops[0] != CmpOp::Eq || compare.comparators.len() != 1 {
            return Err(TypeShapeDslDefinitionError {
                range: compare.range,
                message: "condition must be exactly `<Int parameter> == <Int parameter>`",
            });
        }
        let (Some(left), Some(right)) = (
            parameter_index(self.parameters, &compare.left),
            parameter_index(self.parameters, &compare.comparators[0]),
        ) else {
            return Err(TypeShapeDslDefinitionError {
                range: compare.range,
                message: "condition operands must name parameters",
            });
        };
        self.conditions.push(TypeShapeDslEquality {
            range: compare.range,
            left,
            right,
        });
        // Without an `else`, this statement may fall through even if its body returns.
        self.validate_block(&if_stmt.body)?;
        Ok(())
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
    conditions: Vec<TypeShapeDslEquality>,
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
                    .map(|x| (offsets(x.range), x.left, x.right))
                    .cmp(
                        other
                            .conditions
                            .iter()
                            .map(|x| (offsets(x.range), x.left, x.right)),
                    )
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

    pub fn conditions(&self) -> impl Iterator<Item = TypeShapeDslEquality> + '_ {
        self.conditions.iter().copied()
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

impl TypeShapeDslEquality {
    pub fn range(self) -> TextRange {
        self.range
    }

    pub fn parameters(self) -> (usize, usize) {
        (self.left, self.right)
    }
}

impl ResolvedTypeShapeDslFunction {
    pub fn try_new(
        definition: Arc<ValidatedTypeShapeDslFunction>,
        parameter_domains: Vec<TypeShapeDslDomain>,
    ) -> Option<Self> {
        if definition.parameter_count() != parameter_domains.len() {
            return None;
        }
        let resolved = Self {
            definition,
            parameter_domains,
        };
        let result_domain = resolved
            .definition
            .returns()
            .next()
            .map(|x| resolved.return_domain(x.kind()))
            .expect("validated type-level DSL function must return");
        assert!(
            resolved
                .definition
                .returns()
                .all(|x| resolved.return_domain(x.kind()) == result_domain),
            "resolved type-level DSL returns must have one domain"
        );
        Some(resolved)
    }

    pub fn name(&self) -> &Name {
        self.definition.name()
    }

    pub fn parameter_name(&self, index: usize) -> &Name {
        self.definition.parameter_name(index)
    }

    pub fn parameter_domains(&self) -> &[TypeShapeDslDomain] {
        &self.parameter_domains
    }

    pub fn result_domain(&self) -> TypeShapeDslDomain {
        let return_ = self
            .definition
            .returns()
            .next()
            .expect("validated type-level DSL function must return");
        self.return_domain(return_.kind())
    }

    fn return_domain(&self, kind: TypeShapeDslReturnKind) -> TypeShapeDslDomain {
        match kind {
            TypeShapeDslReturnKind::Parameter(index) => self.parameter_domains[index],
            TypeShapeDslReturnKind::Gradual(domain) => domain,
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
            TypeLevelDslFunction::UserDefined(function) => {
                Ok(match function.evaluate(&self.args) {
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
    fn evaluate(&self, args: &[Type]) -> DslEvaluation {
        assert_eq!(
            args.len(),
            self.parameter_domains.len(),
            "type-level DSL values must align with resolved parameters"
        );
        match self.evaluate_block(&self.definition.parsed.definition.body, args) {
            DslControlFlow::Return(result) => result,
            DslControlFlow::Continue => {
                unreachable!("validated type-level DSL function cannot fall through")
            }
        }
    }

    fn evaluate_block(&self, block: &[Stmt], args: &[Type]) -> DslControlFlow {
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
                    return DslControlFlow::Return(match kind {
                        TypeShapeDslReturnKind::Parameter(index) => {
                            DslValue::from_type(&args[index], self.parameter_domains[index])
                                .map_or(DslEvaluation::AutomaticFallback, DslEvaluation::Value)
                        }
                        TypeShapeDslReturnKind::Gradual(_) => DslEvaluation::ExplicitGradual,
                    });
                }
                Stmt::If(if_stmt) => {
                    let (left, right) = self
                        .definition
                        .conditions
                        .iter()
                        .find_map(|equality| {
                            (equality.range == if_stmt.test.range())
                                .then_some((equality.left, equality.right))
                        })
                        .expect("validated if condition must have validation metadata");
                    // Reflexive equality is true even when the parameter itself is gradual.
                    let condition = if left == right {
                        DslCondition::True
                    } else {
                        match (Int::from_type(&args[left]), Int::from_type(&args[right])) {
                            (Some(Int::Int), _) | (_, Some(Int::Int)) => DslCondition::Unknown,
                            (Some(left), Some(right)) if left == right => DslCondition::True,
                            (Some(Int::Literal(_)), Some(Int::Literal(_))) => DslCondition::False,
                            _ => DslCondition::Unknown,
                        }
                    };
                    match condition {
                        DslCondition::True => match self.evaluate_block(&if_stmt.body, args) {
                            DslControlFlow::Continue => {}
                            result @ DslControlFlow::Return(_) => return result,
                        },
                        DslCondition::False => {}
                        DslCondition::Unknown => {
                            return DslControlFlow::Return(DslEvaluation::AutomaticFallback);
                        }
                    }
                }
                _ => unreachable!("validated type-level DSL block contains only if and return"),
            }
        }
        DslControlFlow::Continue
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
