/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Type-level shape DSL definitions move through four phases. Binding retains the parsed function
//! AST; `DslValidator` checks the restricted syntax and records semantic metadata; the solver
//! validates that metadata against resolved parameter domains and links helper functions into a
//! bounded program; evaluation interprets that program to produce an `Int` or `IntTuple`.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use compact_str::CompactString;
use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::BoolOp;
use ruff_python_ast::CmpOp;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprGenerator;
use ruff_python_ast::ExprName;
use ruff_python_ast::ExprSlice;
use ruff_python_ast::ExprSubscript;
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
use crate::einsum::EinsumClassification;
use crate::einsum::evaluate_einsum;
use crate::einsum::parse_einsum_equation;
use crate::equality::TypeEq as TypeEqTrait;
use crate::equality::TypeEqCtx;
use crate::function::FuncDefId;
use crate::literal::Lit;
use crate::map_int_tuples::MapIntTuples;
use crate::quantified::Quantified;
use crate::shaped_array::IntTuple;
use crate::shaped_array::IntTupleView;
use crate::shaped_array::broadcast_shapes;
use crate::tuple::Tuple;
use crate::type_var::FlagDomain;
use crate::type_var::FlagMember;
use crate::types::Type;

#[derive(Debug, Clone, Copy, PartialEq, Eq, TypeEq, PartialOrd, Ord, Hash)]
pub enum TypeShapeDslDomain {
    Int,
    IntTuple,
    IntTuples,
}

impl TypeShapeDslDomain {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Int => "Int",
            Self::IntTuple => "IntTuple",
            Self::IntTuples => "IntTuples",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, TypeEq, PartialOrd, Ord, Hash)]
/// The type-system domain accepted by one DSL parameter.
///
/// `Value` represents a shape dimension or shape tuple. `Flag` represents literal-preserving
/// configuration values supplied by ordinary Python calls.
pub enum TypeShapeDslInputDomain {
    Value(TypeShapeDslDomain),
    /// The input-only domain spelled exactly `Int | None`.
    OptionalInt,
    Flag(FlagDomain),
}

impl TypeShapeDslInputDomain {
    /// Whether a parameter in this domain may be used as a value in `result` after narrowing.
    pub fn can_use_as(
        self,
        result: TypeShapeDslDomain,
        narrowing: TypeShapeDslParameterNarrowing,
    ) -> bool {
        self == Self::Value(result)
            || (self == Self::OptionalInt
                && result == TypeShapeDslDomain::Int
                && narrowing.proves_not_none())
    }

    /// Whether a value from this domain can be passed to a helper parameter in `expected`.
    fn can_forward_to(self, expected: Self) -> bool {
        match (self, expected) {
            (Self::Flag(actual), Self::Flag(expected)) => actual.is_subset_of(expected),
            (Self::Value(TypeShapeDslDomain::Int), Self::OptionalInt) => true,
            _ => self == expected,
        }
    }
}

/// A syntactically valid helper call retained until ordinary name resolution is available.
///
/// DSL validation records the callee AST and each argument's shape-domain source. The solver then
/// resolves imports and aliases through normal function identity before attaching the resulting
/// helper program at the narrow boundary between Pyrefly's function model and the shape DSL.
/// Helper calls are valid only as return values, so evaluation follows a bounded chain of calls.
#[derive(Debug, Clone)]
pub struct TypeShapeDslHelperCall {
    callee: Expr,
    arguments: Vec<TypeShapeDslHelperArgument>,
}

// The parsed function owns the retained AST, so a callee's source range identifies it within
// that function even though `Expr` itself does not implement `Eq` or `Hash`.
impl PartialEq for TypeShapeDslHelperCall {
    fn eq(&self, other: &Self) -> bool {
        self.callee.range() == other.callee.range() && self.arguments == other.arguments
    }
}

impl Eq for TypeShapeDslHelperCall {}

impl Hash for TypeShapeDslHelperCall {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.callee.range().hash(state);
        self.arguments.hash(state);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum TypeShapeDslHelperArgumentSource {
    Parameters(Box<[usize]>),
    ParametersWithRequiredDomain {
        parameters: Box<[usize]>,
        domain: TypeShapeDslInputDomain,
    },
    ValueSources {
        sources: DslValueSources,
        observed_domain: FlagDomain,
    },
    NoneLiteral,
    Exact(TypeShapeDslInputDomain),
    DeferredInteger {
        index: usize,
        parameter_uses: Box<[TypeShapeDslParameterUse]>,
        resolved_domain: Option<TypeShapeDslInputDomain>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct TypeShapeDslHelperArgument {
    slot: usize,
    source: TypeShapeDslHelperArgumentSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeShapeDslHelperArgumentError {
    Arity,
    IncompatibleDomain {
        argument: usize,
        actual: TypeShapeDslInputDomain,
        expected: TypeShapeDslInputDomain,
    },
}

impl TypeShapeDslHelperCall {
    pub fn callee(&self) -> &Expr {
        &self.callee
    }

    pub fn argument_domains(
        &self,
        caller_domains: &[TypeShapeDslInputDomain],
        expected_domains: &[TypeShapeDslInputDomain],
        deferred_domains: &mut HashMap<usize, TypeShapeDslInputDomain>,
    ) -> Result<Vec<TypeShapeDslInputDomain>, TypeShapeDslHelperArgumentError> {
        if self.arguments.len() != expected_domains.len() {
            return Err(TypeShapeDslHelperArgumentError::Arity);
        }
        let effective_parameter_domain =
            |use_: TypeShapeDslParameterUse, observed_flag_domain: FlagDomain| match caller_domains
                [use_.parameter]
            {
                TypeShapeDslInputDomain::Value(TypeShapeDslDomain::Int) => {
                    Some(TypeShapeDslInputDomain::Value(TypeShapeDslDomain::Int))
                }
                TypeShapeDslInputDomain::OptionalInt if use_.narrowing.proves_not_none() => {
                    Some(TypeShapeDslInputDomain::Value(TypeShapeDslDomain::Int))
                }
                TypeShapeDslInputDomain::Flag(domain)
                    if use_.narrowing != TypeShapeDslParameterNarrowing::Unnarrowed =>
                {
                    domain
                        .intersection(observed_flag_domain)
                        .map(TypeShapeDslInputDomain::Flag)
                }
                domain => Some(domain),
            };
        self.arguments
            .iter()
            .zip(expected_domains)
            .enumerate()
            .map(|(argument_index, (argument, expected))| {
                let incompatible = |actual| TypeShapeDslHelperArgumentError::IncompatibleDomain {
                    argument: argument_index,
                    actual,
                    expected: *expected,
                };
                match &argument.source {
                    TypeShapeDslHelperArgumentSource::Exact(domain) => domain
                        .can_forward_to(*expected)
                        .then_some(*expected)
                        .ok_or_else(|| incompatible(*domain)),
                    TypeShapeDslHelperArgumentSource::ParametersWithRequiredDomain {
                        parameters,
                        domain,
                    } => {
                        if let Some(actual) = parameters
                            .iter()
                            .map(|parameter| caller_domains[*parameter])
                            .find(|actual| *actual != *domain)
                        {
                            Err(incompatible(actual))
                        } else {
                            domain
                                .can_forward_to(*expected)
                                .then_some(*expected)
                                .ok_or_else(|| incompatible(*domain))
                        }
                    }
                    TypeShapeDslHelperArgumentSource::ValueSources {
                        sources,
                        observed_domain,
                    } => {
                        if let Some(actual) = sources.parameter_uses.iter().find_map(|use_| {
                            let actual = effective_parameter_domain(*use_, *observed_domain)
                                .unwrap_or(caller_domains[use_.parameter]);
                            (!actual.can_forward_to(*expected)).then_some(actual)
                        }) {
                            return Err(incompatible(actual));
                        }
                        if sources.non_parameter_kinds != 0 {
                            let Some(domain) = flag_domain_from_kinds(sources.non_parameter_kinds)
                            else {
                                unreachable!("nonempty non-parameter Flag kinds are representable")
                            };
                            let compatible = (*expected == TypeShapeDslInputDomain::OptionalInt
                                && domain == FlagDomain::of(FlagMember::NoneType))
                                || TypeShapeDslInputDomain::Flag(domain).can_forward_to(*expected);
                            if !compatible {
                                return Err(incompatible(TypeShapeDslInputDomain::Flag(domain)));
                            }
                        }
                        Ok(*expected)
                    }
                    TypeShapeDslHelperArgumentSource::NoneLiteral => match expected {
                        TypeShapeDslInputDomain::OptionalInt => Ok(*expected),
                        TypeShapeDslInputDomain::Flag(domain)
                            if domain.contains(FlagMember::NoneType) =>
                        {
                            Ok(*expected)
                        }
                        _ => Err(incompatible(TypeShapeDslInputDomain::Flag(FlagDomain::of(
                            FlagMember::NoneType,
                        )))),
                    },
                    TypeShapeDslHelperArgumentSource::Parameters(parameters) => {
                        if let Some(actual) = parameters
                            .iter()
                            .map(|parameter| caller_domains[*parameter])
                            .find(|actual| !actual.can_forward_to(*expected))
                        {
                            Err(incompatible(actual))
                        } else {
                            Ok(*expected)
                        }
                    }
                    TypeShapeDslHelperArgumentSource::DeferredInteger {
                        index,
                        parameter_uses,
                        resolved_domain,
                        ..
                    } => {
                        let selected_domain = match expected {
                            TypeShapeDslInputDomain::Value(TypeShapeDslDomain::Int)
                            | TypeShapeDslInputDomain::OptionalInt => {
                                if let Some(actual) = parameter_uses.iter().find_map(|use_| {
                                    let actual = effective_parameter_domain(
                                        *use_,
                                        FlagDomain::of(FlagMember::Int),
                                    )
                                    .unwrap_or(caller_domains[use_.parameter]);
                                    let is_integer = match actual {
                                        TypeShapeDslInputDomain::Value(TypeShapeDslDomain::Int) => {
                                            true
                                        }
                                        TypeShapeDslInputDomain::Flag(domain) => {
                                            domain.is_subset_of(FlagDomain::of(FlagMember::Int))
                                        }
                                        _ => false,
                                    };
                                    (!is_integer).then_some(actual)
                                }) {
                                    return Err(incompatible(actual));
                                }
                                TypeShapeDslInputDomain::Value(TypeShapeDslDomain::Int)
                            }
                            TypeShapeDslInputDomain::Flag(domain)
                                if FlagDomain::of(FlagMember::Int).is_subset_of(*domain) =>
                            {
                                if let Some(actual) = parameter_uses.iter().find_map(|use_| {
                                    let actual = effective_parameter_domain(
                                        *use_,
                                        FlagDomain::of(FlagMember::Int),
                                    )
                                    .unwrap_or(caller_domains[use_.parameter]);
                                    (!actual.can_forward_to(*expected)).then_some(actual)
                                }) {
                                    return Err(incompatible(actual));
                                }
                                TypeShapeDslInputDomain::Flag(FlagDomain::of(FlagMember::Int))
                            }
                            _ => {
                                let actual = parameter_uses
                                    .first()
                                    .and_then(|use_| {
                                        effective_parameter_domain(
                                            *use_,
                                            FlagDomain::of(FlagMember::Int),
                                        )
                                    })
                                    .unwrap_or(TypeShapeDslInputDomain::Value(
                                        TypeShapeDslDomain::Int,
                                    ));
                                return Err(incompatible(actual));
                            }
                        };
                        if let Some(domain) = *resolved_domain
                            && domain != selected_domain
                        {
                            return Err(incompatible(domain));
                        }
                        let previous = deferred_domains.entry(*index).or_insert(selected_domain);
                        (*previous == selected_domain)
                            .then_some(selected_domain)
                            .ok_or_else(|| incompatible(*previous))
                    }
                }
            })
            .collect()
    }
}

impl fmt::Display for TypeShapeDslInputDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Value(domain) => f.write_str(domain.as_str()),
            Self::OptionalInt => f.write_str("Int | None"),
            Self::Flag(domain) => write!(f, "Flag[{domain}]"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, TypeEq, PartialOrd, Ord, Hash)]
struct ResolvedTypeShapeDslNodeId(u32);

impl ResolvedTypeShapeDslNodeId {
    const ROOT: Self = Self(0);

    fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Clone, PartialEq, Eq, TypeEq, PartialOrd, Ord, Hash)]
struct ResolvedTypeShapeDslNode {
    id: Arc<FuncDefId>,
    definition: Arc<StructurallyValidatedTypeShapeDslFunction>,
    parameter_domains: Vec<TypeShapeDslInputDomain>,
    result_domain: TypeShapeDslDomain,
    /// Targets correspond positionally to `definition.helper_calls`.
    helper_targets: Box<[ResolvedTypeShapeDslNodeId]>,
}

impl ResolvedTypeShapeDslNode {
    fn parameter_domains(&self) -> &[TypeShapeDslInputDomain] {
        &self.parameter_domains
    }

    fn result_domain(&self) -> TypeShapeDslDomain {
        self.result_domain
    }
}

/// The complete evaluation program for one type-level shape DSL entry point.
///
/// A function with no helper calls has only its root node. Keeping nodes in a flat program lets
/// helper definitions be shared rather than recursively embedded once helper calls are resolved.
#[derive(Debug, Clone, PartialEq, Eq, TypeEq, PartialOrd, Ord, Hash)]
struct ResolvedTypeShapeDslProgram {
    nodes: Box<[ResolvedTypeShapeDslNode]>,
    edge_count: usize,
    max_depth: usize,
}

impl ResolvedTypeShapeDslProgram {
    fn root(&self) -> &ResolvedTypeShapeDslNode {
        self.node(ResolvedTypeShapeDslNodeId::ROOT)
    }

    fn node(&self, id: ResolvedTypeShapeDslNodeId) -> &ResolvedTypeShapeDslNode {
        &self.nodes[id.index()]
    }

    fn evaluate(
        &self,
        id: ResolvedTypeShapeDslNodeId,
        args: &[Type],
        budget: &mut DslEvaluationBudget,
    ) -> DslOutcome {
        let node = self.node(id);
        let parameter_count = node.definition.parsed.parameter_count();
        assert_eq!(
            parameter_count,
            node.parameter_domains().len(),
            "validated type-level DSL AST must align with its signature"
        );
        assert_eq!(
            args.len(),
            parameter_count,
            "type-level DSL values must align with validated parameters"
        );
        let slots = args
            .iter()
            .zip(node.parameter_domains())
            .map(|(argument, domain)| lower_parameter(argument, *domain))
            .collect::<Vec<_>>();
        self.evaluate_lowered(id, slots, budget)
    }
}

/// A validated DSL definition paired with its resolved program.
#[derive(Debug, Clone, PartialEq, Eq, TypeEq, PartialOrd, Ord, Hash)]
pub struct ResolvedTypeShapeDslFunction {
    program: Arc<ResolvedTypeShapeDslProgram>,
}

impl ResolvedTypeShapeDslFunction {
    /// Builds the resolved helper program after the caller has validated all retained metadata
    /// that depends on resolved parameter domains. Evaluation relies on that cross-crate boundary
    /// and on parameter lowering mapping unsupported runtime values to `DslValue::Unknown`.
    pub fn try_new(
        id: Arc<FuncDefId>,
        definition: Arc<StructurallyValidatedTypeShapeDslFunction>,
        parameter_domains: Vec<TypeShapeDslInputDomain>,
        result_domain: TypeShapeDslDomain,
        helpers: Vec<(Arc<FuncDefId>, Arc<Self>)>,
    ) -> Result<Self, TypeShapeDslProgramError> {
        if definition.parsed.parameter_count() != parameter_domains.len()
            || definition.helper_calls.len() != helpers.len()
        {
            return Err(TypeShapeDslProgramError::InconsistentDependency);
        }
        let root = ResolvedTypeShapeDslNode {
            id,
            definition,
            parameter_domains,
            result_domain,
            helper_targets: Box::new([]),
        };
        let mut builder = ResolvedTypeShapeDslProgramBuilder::new(root);
        let mut targets = Vec::with_capacity(helpers.len());
        for (id, helper) in helpers {
            if helper.root().id != id {
                return Err(TypeShapeDslProgramError::InconsistentDependency);
            }
            targets.push(builder.import_node(&helper.program, ResolvedTypeShapeDslNodeId::ROOT)?);
        }
        builder.set_targets(ResolvedTypeShapeDslNodeId::ROOT, targets)?;
        Ok(Self {
            program: Arc::new(builder.finish()?),
        })
    }

    fn root(&self) -> &ResolvedTypeShapeDslNode {
        self.program.root()
    }

    fn evaluate(&self, args: &[Type]) -> DslOutcome {
        let mut budget = DslEvaluationBudget::new();
        self.program
            .evaluate(ResolvedTypeShapeDslNodeId::ROOT, args, &mut budget)
    }

    pub fn name(&self) -> &Name {
        self.root().definition.name()
    }

    pub fn parameter_name(&self, index: usize) -> &Name {
        self.root().definition.parameter_name(index)
    }

    pub fn parameter_domains(&self) -> &[TypeShapeDslInputDomain] {
        &self.root().parameter_domains
    }

    pub fn result_domain(&self) -> TypeShapeDslDomain {
        self.root().result_domain
    }

    pub fn contains_function(&self, id: &FuncDefId) -> bool {
        self.program.nodes.iter().any(|node| node.id.as_ref() == id)
    }

    pub fn helper_graph_metrics(&self) -> (usize, usize, usize) {
        (
            self.program.nodes.len(),
            self.program.edge_count,
            self.program.max_depth,
        )
    }
}

pub const MAX_HELPER_CALL_DEPTH: usize = 32;
pub const MAX_HELPER_GRAPH_NODES: usize = 4096;
pub const MAX_HELPER_GRAPH_EDGES: usize = 16384;

#[derive(Debug, Clone, Copy)]
pub enum TypeShapeDslProgramError {
    Cycle,
    Depth,
    NodeBudget,
    EdgeBudget,
    InconsistentDependency,
}

impl TypeShapeDslProgramError {
    pub fn message(self) -> &'static str {
        match self {
            Self::Cycle => "recursive DSL helper calls are not supported",
            Self::Depth => "DSL helper call depth exceeds 32",
            Self::NodeBudget => "DSL helper graph exceeds the 4096-function budget",
            Self::EdgeBudget => "DSL helper graph exceeds the 16384-call-edge budget",
            Self::InconsistentDependency => {
                "DSL helper graph contains inconsistent definitions for one function"
            }
        }
    }
}

struct ResolvedTypeShapeDslProgramBuilderNode {
    node: ResolvedTypeShapeDslNode,
    targets: Option<Vec<ResolvedTypeShapeDslNodeId>>,
}

struct ResolvedTypeShapeDslProgramBuilder {
    nodes: Vec<ResolvedTypeShapeDslProgramBuilderNode>,
    visited: HashMap<Arc<FuncDefId>, ResolvedTypeShapeDslNodeId>,
    edge_count: usize,
}

impl ResolvedTypeShapeDslProgramBuilder {
    // `FuncDefId` contains mutable class data, but its `Eq` and `Hash` use immutable source identity.
    #[allow(clippy::mutable_key_type)]
    fn new(root: ResolvedTypeShapeDslNode) -> Self {
        let visited = HashMap::from([(root.id.clone(), ResolvedTypeShapeDslNodeId::ROOT)]);
        Self {
            nodes: vec![ResolvedTypeShapeDslProgramBuilderNode {
                node: root,
                targets: None,
            }],
            visited,
            edge_count: 0,
        }
    }

    fn import_node(
        &mut self,
        source: &ResolvedTypeShapeDslProgram,
        source_id: ResolvedTypeShapeDslNodeId,
    ) -> Result<ResolvedTypeShapeDslNodeId, TypeShapeDslProgramError> {
        // Imported programs are already bounded, while this builder enforces the aggregate node
        // and edge limits. `finish` separately rejects cycles and excessive combined depth.
        let source_node = source.node(source_id);
        if let Some(&target_id) = self.visited.get(&source_node.id) {
            let target = &self.nodes[target_id.index()];
            if target.node.definition != source_node.definition
                || target.node.parameter_domains != source_node.parameter_domains
                || target.node.result_domain != source_node.result_domain
            {
                return Err(TypeShapeDslProgramError::InconsistentDependency);
            }
            if let Some(targets) = &target.targets {
                let target_ids = targets
                    .iter()
                    .map(|target| self.nodes[target.index()].node.id.as_ref())
                    .collect::<Vec<_>>();
                let source_ids = source_node
                    .helper_targets
                    .iter()
                    .map(|target| source.node(*target).id.as_ref())
                    .collect::<Vec<_>>();
                if target_ids != source_ids {
                    return Err(TypeShapeDslProgramError::InconsistentDependency);
                }
            }
            return Ok(target_id);
        }
        if self.nodes.len() >= MAX_HELPER_GRAPH_NODES {
            return Err(TypeShapeDslProgramError::NodeBudget);
        }
        let target_id = ResolvedTypeShapeDslNodeId(
            u32::try_from(self.nodes.len()).expect("helper graph node budget fits in u32"),
        );
        self.visited.insert(source_node.id.clone(), target_id);
        self.nodes.push(ResolvedTypeShapeDslProgramBuilderNode {
            node: ResolvedTypeShapeDslNode {
                id: source_node.id.clone(),
                definition: source_node.definition.clone(),
                parameter_domains: source_node.parameter_domains.clone(),
                result_domain: source_node.result_domain,
                helper_targets: Box::new([]),
            },
            targets: None,
        });
        let targets = source_node
            .helper_targets
            .iter()
            .map(|target| self.import_node(source, *target))
            .collect::<Result<Vec<_>, _>>()?;
        self.set_targets(target_id, targets)?;
        Ok(target_id)
    }

    fn set_targets(
        &mut self,
        id: ResolvedTypeShapeDslNodeId,
        targets: Vec<ResolvedTypeShapeDslNodeId>,
    ) -> Result<(), TypeShapeDslProgramError> {
        if self.nodes[id.index()].node.definition.helper_calls.len() != targets.len() {
            return Err(TypeShapeDslProgramError::InconsistentDependency);
        }
        if self.edge_count + targets.len() > MAX_HELPER_GRAPH_EDGES {
            return Err(TypeShapeDslProgramError::EdgeBudget);
        }
        self.edge_count += targets.len();
        let old = self.nodes[id.index()].targets.replace(targets);
        assert!(old.is_none(), "DSL helper targets are assigned once");
        Ok(())
    }

    fn finish(self) -> Result<ResolvedTypeShapeDslProgram, TypeShapeDslProgramError> {
        let nodes = self
            .nodes
            .into_iter()
            .map(|node| ResolvedTypeShapeDslNode {
                helper_targets: node
                    .targets
                    .expect("every DSL helper node has resolved targets")
                    .into_boxed_slice(),
                ..node.node
            })
            .collect::<Vec<_>>();
        let mut state = vec![0u8; nodes.len()];
        let mut depths = vec![0usize; nodes.len()];
        fn depth(
            id: ResolvedTypeShapeDslNodeId,
            nodes: &[ResolvedTypeShapeDslNode],
            state: &mut [u8],
            depths: &mut [usize],
        ) -> Result<usize, TypeShapeDslProgramError> {
            match state[id.index()] {
                1 => return Err(TypeShapeDslProgramError::Cycle),
                2 => return Ok(depths[id.index()]),
                _ => {}
            }
            state[id.index()] = 1;
            let mut result = 0;
            for target in &nodes[id.index()].helper_targets {
                result = result.max(depth(*target, nodes, state, depths)? + 1);
            }
            if result > MAX_HELPER_CALL_DEPTH {
                return Err(TypeShapeDslProgramError::Depth);
            }
            state[id.index()] = 2;
            depths[id.index()] = result;
            Ok(result)
        }
        let max_depth = depth(
            ResolvedTypeShapeDslNodeId::ROOT,
            &nodes,
            &mut state,
            &mut depths,
        )?;
        Ok(ResolvedTypeShapeDslProgram {
            nodes: nodes.into_boxed_slice(),
            edge_count: self.edge_count,
            max_depth,
        })
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

trait SourceRangeKey {
    fn source_range_key(&self) -> TextRange;
}

/// Validation metadata keyed by the exact source range of the AST node it describes.
///
/// Entries have unique ranges and are sorted by source offset. This supports exact binary-search
/// lookup and gives the table deterministic equality and hashing independent of insertion order.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SourceRangeTable<T> {
    entries: Vec<T>,
}

impl<T: SourceRangeKey> SourceRangeTable<T> {
    fn new(table_name: &'static str, mut entries: Vec<T>) -> Self {
        entries.sort_unstable_by_key(|entry| {
            let range = entry.source_range_key();
            (range.start(), range.end())
        });
        assert!(
            entries
                .windows(2)
                .all(|pair| pair[0].source_range_key() != pair[1].source_range_key()),
            "validated type-level DSL {table_name} metadata ranges must be unique"
        );
        Self { entries }
    }

    fn get(&self, range: TextRange) -> Option<&T> {
        self.entries
            .binary_search_by_key(&(range.start(), range.end()), |entry| {
                let range = entry.source_range_key();
                (range.start(), range.end())
            })
            .ok()
            .map(|index| &self.entries[index])
    }

    fn iter(&self) -> impl Iterator<Item = &T> {
        self.entries.iter()
    }
}

#[cfg(test)]
mod source_range_table_tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hash;
    use std::hash::Hasher;

    use ruff_text_size::TextSize;

    use super::SourceRangeTable;
    use super::TextRange;
    use super::TypeShapeDslAssignment;

    fn range(start: u32, end: u32) -> TextRange {
        TextRange::new(TextSize::new(start), TextSize::new(end))
    }

    fn entry(range: TextRange, slot: usize) -> TypeShapeDslAssignment {
        TypeShapeDslAssignment { range, slot }
    }

    fn hashed(table: &SourceRangeTable<TypeShapeDslAssignment>) -> u64 {
        let mut hasher = DefaultHasher::new();
        table.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn source_range_table_sorts_and_finds_exact_ranges() {
        let first = range(0, 1);
        let inner = range(2, 4);
        let outer = range(2, 8);
        let last = range(10, 12);
        let missing = range(5, 6);
        let table = SourceRangeTable::new(
            "test",
            vec![
                entry(outer, 3),
                entry(last, 4),
                entry(inner, 2),
                entry(first, 1),
            ],
        );

        assert_eq!(
            table
                .iter()
                .map(|entry| (entry.range, entry.slot))
                .collect::<Vec<_>>(),
            vec![(first, 1), (inner, 2), (outer, 3), (last, 4),]
        );
        assert_eq!(table.get(first).map(|entry| entry.slot), Some(1));
        assert_eq!(table.get(inner).map(|entry| entry.slot), Some(2));
        assert_eq!(table.get(outer).map(|entry| entry.slot), Some(3));
        assert_eq!(table.get(last).map(|entry| entry.slot), Some(4));
        assert_eq!(table.get(missing), None);
    }

    #[test]
    fn source_range_table_identity_ignores_input_order() {
        let first = range(0, 1);
        let inner = range(2, 4);
        let outer = range(2, 8);
        let table = SourceRangeTable::new(
            "test",
            vec![entry(outer, 3), entry(first, 1), entry(inner, 2)],
        );
        let reordered = SourceRangeTable::new(
            "test",
            vec![entry(inner, 2), entry(outer, 3), entry(first, 1)],
        );

        assert_eq!(table, reordered);
        assert_eq!(hashed(&table), hashed(&reordered));
    }

    #[test]
    #[should_panic(expected = "validated type-level DSL test metadata ranges must be unique")]
    fn source_range_table_rejects_duplicate_ranges() {
        let duplicate = range(2, 4);
        let _ = SourceRangeTable::new("test", vec![entry(duplicate, 1), entry(duplicate, 2)]);
    }
}

/// A type-level shape DSL declaration whose envelope was validated during binding.
#[derive(Debug, Clone)]
pub struct ParsedTypeShapeDslFunction {
    definition: Arc<StmtFunctionDef>,
}

/// An owned function AST whose restricted declaration syntax and body have been structurally
/// validated.
/// Future evaluation may interpret the definition relying on these invariants.
///
/// Identity is derived from the parsed program's pointer identity plus the resolved metadata. The
/// latter is required because resolving an intrinsic depends on imports outside this AST, so an
/// unedited declaration whose gradual constructor now resolves to a different domain is unequal.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructurallyValidatedTypeShapeDslFunction {
    parsed: ParsedTypeShapeDslFunction,
    // These source-keyed facts are validation invariants for the retained AST, not a body IR.
    returns: SourceRangeTable<TypeShapeDslReturn>,
    conditions: SourceRangeTable<TypeShapeDslCondition>,
    expressions: SourceRangeTable<TypeShapeDslExpression>,
    assignments: SourceRangeTable<TypeShapeDslAssignment>,
    helper_calls: Vec<TypeShapeDslHelperCall>,
    /// The number of indexed storage entries the body needs: one per parameter, then one per local.
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

impl SourceRangeKey for TypeShapeDslReturn {
    fn source_range_key(&self) -> TextRange {
        self.statement_range
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

impl SourceRangeKey for TypeShapeDslCondition {
    fn source_range_key(&self) -> TextRange {
        self.range
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

impl SourceRangeKey for TypeShapeDslExpression {
    fn source_range_key(&self) -> TextRange {
        self.range
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TypeShapeDslAssignment {
    range: TextRange,
    slot: usize,
}

impl SourceRangeKey for TypeShapeDslAssignment {
    fn source_range_key(&self) -> TextRange {
        self.range
    }
}

/// The structurally validated source of a type-level shape DSL function's return value.
/// Resolving this depends on more than the AST, so it participates in
/// `StructurallyValidatedTypeShapeDslFunction` identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TypeShapeDslReturnKind {
    /// Return the value stored in a parameter or local slot.
    Slot {
        slot: usize,
        kind: TypeShapeDslSlotReturnKind,
    },
    /// Return the broadcast of two shape parameters.
    Broadcast {
        left_slot: usize,
        right_slot: usize,
        left_parameters: Box<[usize]>,
        right_parameters: Box<[usize]>,
    },
    /// Evaluate a structurally validated expression in the required result domain.
    Expression(TypeShapeDslDomain),
    /// Return an invalid shape computation with a source-provided message.
    Invalid,
    /// Return the gradual value for the function's declared result domain.
    Gradual(TypeShapeDslDomain),
    /// Evaluate a statically resolved user-defined DSL helper.
    HelperCall(usize),
}

/// Validation information for a returned parameter or local slot.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TypeShapeDslSlotReturnKind {
    /// The slot is the returned parameter itself.
    DirectParameter(TypeShapeDslParameterUse),
    /// The slot is a local whose domain comes from the listed parameters.
    ParameterAlias(Box<[TypeShapeDslParameterUse]>),
    /// The slot has a known domain, with optional contributing parameters.
    KnownDomain {
        domain: TypeShapeDslDomain,
        parameter_uses: Option<Box<[TypeShapeDslParameterUse]>>,
    },
}

/// What control flow has established about a parameter at a particular use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TypeShapeDslParameterNarrowing {
    /// The parameter is used without narrowing.
    Unnarrowed,
    /// Control flow has established that the value is `None`.
    IsNone,
    /// Control flow has ruled out `None` at this use.
    NonNone,
    /// Control flow has established that the value is an integer.
    Integer,
}

impl TypeShapeDslParameterNarrowing {
    /// Whether this use excludes `None`.
    pub fn proves_not_none(self) -> bool {
        matches!(self, Self::NonNone | Self::Integer)
    }

    fn proves(self, required: Self) -> bool {
        match required {
            Self::Unnarrowed => true,
            Self::IsNone => self == Self::IsNone,
            Self::NonNone => self.proves_not_none(),
            Self::Integer => self == Self::Integer,
        }
    }

    fn and(self, other: Self) -> Option<Self> {
        match (self, other) {
            (Self::Unnarrowed, narrowing) | (narrowing, Self::Unnarrowed) => Some(narrowing),
            (Self::IsNone, Self::IsNone) => Some(Self::IsNone),
            (Self::NonNone, Self::NonNone) => Some(Self::NonNone),
            (Self::Integer, Self::Integer) => Some(Self::Integer),
            (Self::NonNone, Self::Integer) | (Self::Integer, Self::NonNone) => Some(Self::Integer),
            (Self::IsNone, Self::NonNone | Self::Integer)
            | (Self::NonNone | Self::Integer, Self::IsNone) => None,
        }
    }

    fn or(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unnarrowed, _) | (_, Self::Unnarrowed) => Self::Unnarrowed,
            (Self::IsNone, Self::IsNone) => Self::IsNone,
            (Self::NonNone, Self::NonNone) => Self::NonNone,
            (Self::Integer, Self::Integer) => Self::Integer,
            (Self::NonNone, Self::Integer) | (Self::Integer, Self::NonNone) => Self::NonNone,
            (Self::IsNone, Self::NonNone | Self::Integer)
            | (Self::NonNone | Self::Integer, Self::IsNone) => Self::Unnarrowed,
        }
    }
}

/// A parameter that contributes to a DSL value, together with its narrowing at that use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TypeShapeDslParameterUse {
    parameter: usize,
    narrowing: TypeShapeDslParameterNarrowing,
}

impl TypeShapeDslParameterUse {
    pub fn parameter(self) -> usize {
        self.parameter
    }

    pub fn narrowing(self) -> TypeShapeDslParameterNarrowing {
        self.narrowing
    }
}

/// The arithmetic a structurally validated dimension or Flag expression applies. Reached through
/// `TypeShapeDslReturnKind` and `TypeShapeDslExpressionKind`, so it shares their identity
/// requirements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TypeShapeDslArithmeticOp {
    Add,
    Subtract,
    Multiply,
    FloorDivide,
    Modulo,
}

/// The comparison a structurally validated DSL condition applies. `CmpOp` has no total order, so
/// the DSL records its own closed operator set and keeps evaluator matching exhaustive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TypeShapeDslComparisonOp {
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

impl TypeShapeDslComparisonOp {
    fn from_cmp_op(op: CmpOp) -> Option<Self> {
        match op {
            CmpOp::Eq => Some(Self::Equal),
            CmpOp::NotEq => Some(Self::NotEqual),
            CmpOp::Lt => Some(Self::LessThan),
            CmpOp::LtE => Some(Self::LessThanOrEqual),
            CmpOp::Gt => Some(Self::GreaterThan),
            CmpOp::GtE => Some(Self::GreaterThanOrEqual),
            _ => None,
        }
    }

    fn apply(self, left: i64, right: i64) -> bool {
        match self {
            Self::Equal => left == right,
            Self::NotEqual => left != right,
            Self::LessThan => left < right,
            Self::LessThanOrEqual => left <= right,
            Self::GreaterThan => left > right,
            Self::GreaterThanOrEqual => left >= right,
        }
    }
}

/// A closed, canonical operation the DSL recognizes by callable identity rather than by spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeShapeDslIntrinsic {
    Any,
    Broadcast,
    Concat,
    Einsum,
    Gradual(TypeShapeDslDomain),
    IsConcreteInt,
    IsIntValue,
    IntTuple,
    IntTuples,
    Prod,
    Sum,
    Invalid,
    Len,
    Range,
    Tuple,
    Zip,
}

/// What a structurally validated DSL value expression computes. Like `TypeShapeDslReturnKind`,
/// this depends on intrinsic resolution, so it participates in
/// `StructurallyValidatedTypeShapeDslFunction` identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TypeShapeDslExpressionKind {
    IntTupleSlot {
        slot: usize,
        parameter_origins: Option<Box<[usize]>>,
    },
    IntTupleSlice,
    IntTupleConcat,
    Einsum {
        shapes: usize,
        parameter_origins: Option<Box<[usize]>>,
    },
    DimensionSlot {
        slot: usize,
        parameter_uses: Option<Box<[TypeShapeDslParameterUse]>>,
    },
    IntegerSlot {
        slot: usize,
        parameter_uses: Option<Box<[TypeShapeDslParameterUse]>>,
    },
    DimensionLiteral(Option<i64>),
    /// An explicit `Int.gradual()` used where a dimension is expected.
    Gradual,
    IntTupleIndex {
        shape: usize,
        parameter_origins: Option<Box<[usize]>>,
    },
    IntTuplesIndex {
        shapes: usize,
    },
    DimensionTuple,
    IntTupleConstructor,
    IntTuplesConstructor,
    IntTupleProduct,
    IntTupleSum,
    IntTupleLength {
        shape: usize,
        parameter_origins: Option<Box<[usize]>>,
        domain: DslIntegerDomain,
    },
    GeneratorSourceSlot {
        slot: usize,
        parameter_uses: Option<Box<[TypeShapeDslParameterUse]>>,
        allow_int_tuples: bool,
    },
    GeneratorElementAsDimension {
        slot: usize,
        source_parameter_uses: Option<Box<[TypeShapeDslParameterUse]>>,
    },
    GeneratorElementAsFlagInt {
        slot: usize,
        source_parameter_uses: Option<Box<[TypeShapeDslParameterUse]>>,
    },
    GeneratorElementAsIntTuple {
        slot: usize,
        source_parameter_uses: Box<[TypeShapeDslParameterUse]>,
    },
    GeneratorZip {
        sources: usize,
    },
    Slot(usize),
    FlagValueSlot {
        slot: usize,
        parameter_uses: Option<Box<[TypeShapeDslParameterUse]>>,
        required: TypeShapeDslFlagValueKind,
    },
    FlagIntLiteral(Option<i64>),
    FlagStringLiteral,
    FlagBool(bool),
    FlagNone,
    FlagTuple,
    FlagRange,
    FlagSequenceLength,
    FlagSequenceCount,
    FlagSequenceIndex,
    FlagIntArithmetic(TypeShapeDslArithmeticOp),
    DimensionArithmetic(TypeShapeDslArithmeticOp),
    Conditional,
    DimensionGenerator {
        binder: usize,
        binders: usize,
    },
    FlagGenerator {
        binder: usize,
        binders: usize,
    },
    IntTuplesGenerator {
        binder: usize,
        binders: usize,
    },
}

/// The Flag value domain a validated operation requires of its operand. Reached through
/// `TypeShapeDslExpressionKind`, so it shares that type's identity requirements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TypeShapeDslFlagValueKind {
    Int,
    String,
    Sequence,
}

/// The element kind bound by each lane of a validated generator source.
#[derive(Debug, Clone)]
enum TypeShapeDslGeneratorSource {
    Single(DslStaticKind),
    Zip(Vec<DslStaticKind>),
}

/// Information needed to validate one comparison operand after parameter domains are resolved.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TypeShapeDslComparisonOperand {
    pub parameter_uses: Option<Box<[TypeShapeDslParameterUse]>>,
    pub is_flag_operand: bool,
    pub non_parameter_flag_domain: Option<FlagDomain>,
}

/// What a structurally validated DSL condition tests. Like `TypeShapeDslReturnKind`, this depends
/// on intrinsic resolution, so it participates in `StructurallyValidatedTypeShapeDslFunction`
/// identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TypeShapeDslConditionKind {
    Any {
        binder: usize,
    },
    // The Flag marker distinguishes an operand with no non-parameter values from a non-Flag
    // operand. This keeps literals and literal-backed locals adaptable without losing their
    // representation.
    SlotCompare {
        left: usize,
        right: usize,
        left_operand: TypeShapeDslComparisonOperand,
        right_operand: TypeShapeDslComparisonOperand,
        op: TypeShapeDslComparisonOp,
    },
    IntegerCompare {
        left_operand: TypeShapeDslComparisonOperand,
        right_operand: TypeShapeDslComparisonOperand,
        op: TypeShapeDslComparisonOp,
    },
    DimensionEquality {
        negated: bool,
    },
    GeneratorElementSelfCompare(TypeShapeDslComparisonOp),
    IsConcreteInt {
        slot: usize,
        parameter_origins: Option<Box<[usize]>>,
    },
    IsIntValue {
        slot: usize,
        parameter_origins: Option<Box<[usize]>>,
    },
    IsNone {
        slot: usize,
        parameter_origins: Option<Box<[usize]>>,
        negated: bool,
    },
    FlagIntCompare(TypeShapeDslComparisonOp),
    BoolSlot {
        slot: usize,
        parameter_origins: Option<Box<[usize]>>,
    },
    StringEquality {
        negated: bool,
    },
    Membership {
        negated: bool,
    },
    LengthEqualLiteral {
        slot: usize,
        literal: i64,
    },
}

// These bits are flow facts recorded before parameter annotations are resolved. `FlagDomain`
// represents declared domains after solver integration; `flag_domain_from_kinds` is the boundary
// between the two representations.
const FLAG_INT: u8 = 1;
const FLAG_SEQUENCE: u8 = 2;
const FLAG_NONE: u8 = 4;
const FLAG_BOOL: u8 = 8;
const FLAG_STRING: u8 = 16;
// Integer/sequence narrowing predicates can distinguish these Flag values.
const FLAG_NARROWABLE: u8 = FLAG_INT | FLAG_SEQUENCE | FLAG_NONE;
// Every Flag value the validator and evaluator can represent.
const FLAG_REPRESENTABLE: u8 = FLAG_NARROWABLE | FLAG_BOOL | FLAG_STRING;
const FLAG_NOT_NONE: u8 = FLAG_REPRESENTABLE & !FLAG_NONE;
const MAX_GENERATOR_STEPS: usize = 4096;

/// Bounds the total generator iterations performed by one public DSL evaluation.
struct DslEvaluationBudget {
    remaining_generator_steps: usize,
}

impl DslEvaluationBudget {
    fn new() -> Self {
        Self {
            remaining_generator_steps: MAX_GENERATOR_STEPS,
        }
    }

    fn consume_generator_step(&mut self) -> bool {
        let Some(remaining) = self.remaining_generator_steps.checked_sub(1) else {
            return false;
        };
        self.remaining_generator_steps = remaining;
        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GeneratorResultKind {
    Dimensions,
    FlagValues,
    IntTuples,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GeneratorValidationKind {
    Condition,
    Dimension,
    FlagValue,
    IntTuple,
}

fn flag_domain_from_kinds(kinds: u8) -> Option<FlagDomain> {
    if kinds == 0 || kinds & !FLAG_REPRESENTABLE != 0 {
        return None;
    }
    let mut members = [
        (FLAG_INT, FlagMember::Int),
        (FLAG_SEQUENCE, FlagMember::IntTuple),
        (FLAG_NONE, FlagMember::NoneType),
        (FLAG_BOOL, FlagMember::Bool),
        (FLAG_STRING, FlagMember::Str),
    ]
    .into_iter()
    .filter_map(|(bit, member)| (kinds & bit != 0).then_some(member));
    let first = FlagDomain::of(members.next()?);
    Some(members.fold(first, |domain, member| domain.join(FlagDomain::of(member))))
}

/// Parameter and literal sources contributing to a finite set of possible runtime value kinds.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct DslValueSources {
    parameter_uses: Box<[TypeShapeDslParameterUse]>,
    non_parameter_kinds: u8,
}

impl DslValueSources {
    fn parameters(parameters: &[usize], narrowing: TypeShapeDslParameterNarrowing) -> Self {
        Self {
            parameter_uses: parameter_uses(parameters, narrowing),
            non_parameter_kinds: 0,
        }
    }

    fn non_parameter(kinds: u8) -> Self {
        Self {
            parameter_uses: Box::new([]),
            non_parameter_kinds: kinds,
        }
    }

    fn parameter_uses(&self) -> Option<Box<[TypeShapeDslParameterUse]>> {
        (!self.parameter_uses.is_empty()).then(|| self.parameter_uses.clone())
    }

    fn parameter_indices(&self) -> Option<Box<[usize]>> {
        (!self.parameter_uses.is_empty()).then(|| {
            self.parameter_uses
                .iter()
                .map(|use_| use_.parameter)
                .collect::<Vec<_>>()
                .into_boxed_slice()
        })
    }

    fn narrow(mut self, mask: u8) -> Self {
        let narrowing = if mask == FLAG_NONE {
            TypeShapeDslParameterNarrowing::IsNone
        } else if mask == FLAG_INT {
            TypeShapeDslParameterNarrowing::Integer
        } else if mask & FLAG_NONE == 0 {
            TypeShapeDslParameterNarrowing::NonNone
        } else {
            TypeShapeDslParameterNarrowing::Unnarrowed
        };
        self.parameter_uses = self
            .parameter_uses
            .into_vec()
            .into_iter()
            .filter_map(|mut use_| {
                let possible = match use_.narrowing {
                    TypeShapeDslParameterNarrowing::Unnarrowed => mask != 0,
                    TypeShapeDslParameterNarrowing::IsNone => mask & FLAG_NONE != 0,
                    TypeShapeDslParameterNarrowing::NonNone => mask & FLAG_NOT_NONE != 0,
                    TypeShapeDslParameterNarrowing::Integer => mask & FLAG_INT != 0,
                };
                if !possible {
                    return None;
                }
                use_.narrowing = use_.narrowing.and(narrowing)?;
                Some(use_)
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        self.non_parameter_kinds &= mask;
        self
    }

    fn all_prove(&self, narrowing: TypeShapeDslParameterNarrowing) -> bool {
        self.parameter_uses
            .iter()
            .all(|use_| use_.narrowing.proves(narrowing))
    }

    fn all_narrowed(&self) -> bool {
        self.parameter_uses
            .iter()
            .all(|use_| use_.narrowing != TypeShapeDslParameterNarrowing::Unnarrowed)
    }

    fn non_parameter_values_are(&self, kinds: u8) -> bool {
        self.non_parameter_kinds & !kinds == 0
    }

    fn has_non_parameter_values(&self) -> bool {
        self.non_parameter_kinds != 0
    }

    /// Branch merges retain the weakest fact established for each contributing parameter.
    fn merge(left: Self, right: Self) -> Self {
        let non_parameter_kinds = left.non_parameter_kinds | right.non_parameter_kinds;
        let mut uses = left.parameter_uses.into_vec();
        uses.extend(right.parameter_uses);
        uses.sort_unstable_by_key(|use_| use_.parameter);
        let mut merged: Vec<TypeShapeDslParameterUse> = Vec::with_capacity(uses.len());
        for use_ in uses {
            if let Some(previous) = merged.last_mut()
                && previous.parameter == use_.parameter
            {
                previous.narrowing = previous.narrowing.or(use_.narrowing);
            } else {
                merged.push(use_);
            }
        }
        Self {
            parameter_uses: merged.into_boxed_slice(),
            non_parameter_kinds,
        }
    }
}

fn merge_parameter_origins(
    left: Option<Box<[usize]>>,
    right: Option<Box<[usize]>>,
) -> Option<Box<[usize]>> {
    let mut parameters = left.into_iter().chain(right).flatten().collect::<Vec<_>>();
    if parameters.is_empty() {
        return None;
    }
    parameters.sort_unstable();
    parameters.dedup();
    Some(parameters.into_boxed_slice())
}

fn parameter_uses(
    parameters: &[usize],
    narrowing: TypeShapeDslParameterNarrowing,
) -> Box<[TypeShapeDslParameterUse]> {
    parameters
        .iter()
        .map(|parameter| TypeShapeDslParameterUse {
            parameter: *parameter,
            narrowing,
        })
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

fn record_parameter_use(
    uses: &mut Vec<TypeShapeDslParameterUse>,
    new_use: TypeShapeDslParameterUse,
) {
    if let Some(previous) = uses
        .iter_mut()
        .find(|use_| use_.parameter == new_use.parameter)
    {
        previous.narrowing = previous.narrowing.or(new_use.narrowing);
    } else {
        uses.push(new_use);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DslStaticKind {
    /// One or more parameter slots whose domains are resolved after syntax validation.
    UnknownParameters(Box<[usize]>),
    /// An integer local whose shape dimension versus `Flag[int]` domain is determined by use.
    DeferredInteger(usize),
    /// An `IntTuple` expression, plus any parameters whose domains must resolve to `IntTuple`.
    IntTuple {
        parameter_origins: Option<Box<[usize]>>,
    },
    /// An `IntTuples` expression, plus any parameters whose domains must resolve to `IntTuples`.
    IntTuples {
        parameter_origins: Option<Box<[usize]>>,
    },
    Dimension,
    GeneratorElement {
        source_parameter_uses: Option<Box<[TypeShapeDslParameterUse]>>,
    },
    /// A finite set of possible runtime value kinds, before parameter annotations determine
    /// whether parameter-backed values belong to a `Flag`, `Int`, or `Int | None` domain.
    ValueSet {
        sources: DslValueSources,
        kinds: u8,
    },
}

impl DslStaticKind {
    /// Parameter origins stored before control-flow narrowing adds per-use metadata.
    fn unnarrowed_parameter_origins(&self) -> Option<&[usize]> {
        match self {
            Self::UnknownParameters(parameters)
            | Self::IntTuple {
                parameter_origins: Some(parameters),
            }
            | Self::IntTuples {
                parameter_origins: Some(parameters),
            } => Some(parameters),
            _ => None,
        }
    }

    fn parameter_uses(&self) -> Option<Box<[TypeShapeDslParameterUse]>> {
        if let Some(parameters) = self.unnarrowed_parameter_origins() {
            return Some(parameter_uses(
                parameters,
                TypeShapeDslParameterNarrowing::Unnarrowed,
            ));
        }
        match self {
            Self::ValueSet { sources, .. } => sources.parameter_uses(),
            _ => None,
        }
    }

    fn parameter_alias_uses(&self) -> Option<Box<[TypeShapeDslParameterUse]>> {
        match self {
            Self::UnknownParameters(parameters) => Some(parameter_uses(
                parameters,
                TypeShapeDslParameterNarrowing::Unnarrowed,
            )),
            Self::ValueSet { sources, .. } => sources.parameter_uses(),
            _ => None,
        }
    }

    fn parameter_origins(&self) -> Option<Box<[usize]>> {
        if let Some(parameters) = self.unnarrowed_parameter_origins() {
            return Some(parameters.into());
        }
        match self {
            Self::ValueSet { sources, .. } => sources.parameter_indices(),
            Self::IntTuple { parameter_origins } | Self::IntTuples { parameter_origins } => {
                parameter_origins.clone()
            }
            _ => None,
        }
    }

    fn int_tuple_parameter_origins(&self) -> Option<Box<[usize]>> {
        match self {
            Self::UnknownParameters(parameters)
            | Self::IntTuple {
                parameter_origins: Some(parameters),
            } => Some(parameters.clone()),
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
            (
                Self::ValueSet {
                    sources: left_sources,
                    kinds: left,
                },
                Self::ValueSet {
                    sources: right_sources,
                    kinds: right,
                },
            ) => Some(Self::ValueSet {
                sources: DslValueSources::merge(left_sources, right_sources),
                kinds: left | right,
            }),
            (
                Self::IntTuple {
                    parameter_origins: left,
                },
                Self::IntTuple {
                    parameter_origins: right,
                },
            ) => Some(Self::IntTuple {
                parameter_origins: merge_parameter_origins(left, right),
            }),
            (
                Self::IntTuples {
                    parameter_origins: left,
                },
                Self::IntTuples {
                    parameter_origins: right,
                },
            ) => Some(Self::IntTuples {
                parameter_origins: merge_parameter_origins(left, right),
            }),
            (Self::UnknownParameters(parameters), Self::IntTuple { parameter_origins })
            | (Self::IntTuple { parameter_origins }, Self::UnknownParameters(parameters)) => {
                Some(Self::IntTuple {
                    parameter_origins: merge_parameter_origins(Some(parameters), parameter_origins),
                })
            }
            (Self::UnknownParameters(parameters), Self::IntTuples { parameter_origins })
            | (Self::IntTuples { parameter_origins }, Self::UnknownParameters(parameters)) => {
                Some(Self::IntTuples {
                    parameter_origins: merge_parameter_origins(Some(parameters), parameter_origins),
                })
            }
            (Self::UnknownParameters(parameters), Self::ValueSet { sources, kinds })
            | (Self::ValueSet { sources, kinds }, Self::UnknownParameters(parameters)) => {
                // The known branch determines the possible runtime kinds. Keeping the other
                // branch unnarrowed makes resolution require its exact domain.
                Some(Self::ValueSet {
                    sources: DslValueSources::merge(
                        DslValueSources::parameters(
                            &parameters,
                            TypeShapeDslParameterNarrowing::Unnarrowed,
                        ),
                        sources,
                    ),
                    kinds,
                })
            }
            _ => None,
        }
    }
}

enum IntegerLiteral {
    NotLiteral,
    Unrepresentable { negative: bool },
    Value(i64),
}

impl IntegerLiteral {
    fn into_value(self) -> Result<Option<i64>, ()> {
        match self {
            Self::NotLiteral => Err(()),
            Self::Unrepresentable { .. } => Ok(None),
            Self::Value(value) => Ok(Some(value)),
        }
    }
}

fn integer_literal(expr: &Expr) -> IntegerLiteral {
    match expr {
        Expr::NumberLiteral(number) => match &number.value {
            Number::Int(value) => value.as_i64().map_or(
                IntegerLiteral::Unrepresentable { negative: false },
                IntegerLiteral::Value,
            ),
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
                value.as_i64().map_or(
                    IntegerLiteral::Unrepresentable { negative: false },
                    IntegerLiteral::Value,
                )
            } else {
                value
                    .as_i64()
                    .and_then(i64::checked_neg)
                    .or_else(|| (value.as_u64() == Some(i64::MAX as u64 + 1)).then_some(i64::MIN))
                    .map_or(
                        IntegerLiteral::Unrepresentable { negative: true },
                        IntegerLiteral::Value,
                    )
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
    reachable: bool,
}

/// Which integer domain an integer-valued expression is evaluated in.
///
/// This is public because it is embedded in `TypeShapeDslExpressionKind`.
// `Copy` reflects that this is a small value enum; the comparison and hash traits are required by
// `TypeShapeDslExpressionKind`'s derives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DslIntegerDomain {
    Flag,
    Dimension,
}

impl DslIntegerDomain {
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Dimension, _) | (_, Self::Dimension) => Self::Dimension,
            (Self::Flag, Self::Flag) => Self::Flag,
        }
    }

    fn input_domain(self) -> TypeShapeDslInputDomain {
        match self {
            Self::Dimension => TypeShapeDslInputDomain::Value(TypeShapeDslDomain::Int),
            Self::Flag => TypeShapeDslInputDomain::Flag(FlagDomain::of(FlagMember::Int)),
        }
    }
}

/// What a well-formed `len(...)` is applied to. Classification is domain-neutral: the caller
/// decides whether the operand's cardinality is usable in the domain it needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DslLengthOperand<'a> {
    /// An IntTuple whose length is a rank, usable as either a Flag integer or a dimension.
    IntTuple {
        slot: usize,
        parameter_origins: Option<&'a [usize]>,
    },
    /// A generator element whose source must resolve to `IntTuples`.
    GeneratorIntTuple {
        slot: usize,
        source_parameter_uses: &'a [TypeShapeDslParameterUse],
    },
    /// A Flag sequence, whose cardinality only exists in the Flag domain.
    FlagSequence,
}

/// The source domain and result domain of a validated DSL subscript expression.
enum DslSubscriptKind<'a> {
    /// An unresolved integer-like source whose domain is selected by later uses.
    UnresolvedIndex {
        source: usize,
        parameter_origins: Box<[usize]>,
        index: &'a Expr,
    },
    /// Indexing an `IntTuple` produces an `Int`.
    IntTupleIndex { source: usize, index: &'a Expr },
    /// Slicing an `IntTuple` produces an `IntTuple`.
    IntTupleSlice {
        source: DslIntTupleSliceSource<'a>,
        slice: &'a ExprSlice,
    },
    /// Indexing `IntTuples` produces an `IntTuple`.
    IntTuplesIndex { source: usize, index: &'a Expr },
}

/// An `IntTuple` slice source that is either stored in a local slot or evaluated directly.
enum DslIntTupleSliceSource<'a> {
    /// A previously validated local slot.
    Slot(usize),
    /// An expression whose value is not stored in a local slot.
    Expression(&'a Expr),
}

#[derive(Clone)]
/// Validation-time state for an integer-valued local that may be either a symbolic `Int`
/// dimension or a Flag integer. Validation retains its expression until a later use selects the
/// domain, and redirects group related locals under one decision.
struct DeferredInteger {
    expression: Expr,
    flow: DslValidationFlow,
    state: DeferredIntegerState,
    validated: bool,
}

#[derive(Clone)]
enum DeferredIntegerState {
    /// A root before a use selects the group's evaluation domain.
    UnresolvedRoot {
        default_domain: DslIntegerDomain,
        dependencies: Vec<usize>,
    },
    /// A root whose expressions have a selected evaluation domain.
    ResolvedRoot(DslIntegerDomain),
    /// A non-root entry in the union-find structure.
    Redirect(usize),
}

struct DeferredIntegerClassification {
    default_domain: DslIntegerDomain,
    dependencies: Vec<usize>,
}

/// Sources that determine how a validated integer expression behaves in the DSL.
///
/// Parameter uses preserve narrowing until resolved annotations are available. `has_shape_source`
/// records syntax rooted in a shape-domain value. Validation separately checks whether the
/// expression actually produces an `Int`; for example, indexing `IntTuples` is routed here so it
/// can receive a specific invalid-source diagnostic.
#[derive(Default)]
struct DslIntegerExpressionSources {
    parameter_uses: Vec<TypeShapeDslParameterUse>,
    has_shape_source: bool,
}

// TODO(stroxler): Isolate deferred integer union-find state and AST revalidation behind a
// dedicated abstraction. They remain on `DslValidator` while evaluation consumes the retained AST;
// compiling to a typed IR should remove the need to revisit expressions after resolving domains.
struct DslValidator<'a, F> {
    parameters: &'a Parameters,
    parameter_domains: Option<&'a [TypeShapeDslInputDomain]>,
    intrinsic: &'a F,
    helper_argument_domains: Option<&'a [Vec<TypeShapeDslInputDomain>]>,
    returns: Vec<TypeShapeDslReturn>,
    conditions: Vec<TypeShapeDslCondition>,
    expressions: Vec<TypeShapeDslExpression>,
    assignments: Vec<TypeShapeDslAssignment>,
    helper_calls: Vec<TypeShapeDslHelperCall>,
    slots: HashMap<Name, usize>,
    declared_local_kinds: Vec<Option<DslStaticKind>>,
    deferred_integers: Vec<DeferredInteger>,
}

impl<'a, F: Fn(&Expr) -> Option<TypeShapeDslIntrinsic>> DslValidator<'a, F> {
    fn new(
        parameters: &'a Parameters,
        intrinsic: &'a F,
        parameter_domains: Option<&'a [TypeShapeDslInputDomain]>,
        helper_argument_domains: Option<&'a [Vec<TypeShapeDslInputDomain>]>,
    ) -> (Self, DslValidationFlow) {
        if let Some(parameter_domains) = parameter_domains {
            assert_eq!(
                parameters.args.len(),
                parameter_domains.len(),
                "DSL parameter domains must align with the retained AST"
            );
        }
        let mut slots = HashMap::new();
        let mut kinds = Vec::new();
        for (index, parameter) in parameters.args.iter().enumerate() {
            slots.insert(parameter.parameter.name.id.clone(), index);
            kinds.push(
                match parameter_domains.and_then(|domains| domains.get(index)) {
                    Some(TypeShapeDslInputDomain::Value(TypeShapeDslDomain::IntTuple)) => {
                        DslStaticKind::IntTuple {
                            parameter_origins: Some(Box::new([index])),
                        }
                    }
                    Some(TypeShapeDslInputDomain::Value(TypeShapeDslDomain::IntTuples)) => {
                        DslStaticKind::IntTuples {
                            parameter_origins: Some(Box::new([index])),
                        }
                    }
                    _ => DslStaticKind::UnknownParameters(Box::new([index])),
                },
            );
        }
        let assigned = vec![true; kinds.len()];
        let maybe_assigned = assigned.clone();
        (
            Self {
                parameters,
                parameter_domains,
                intrinsic,
                helper_argument_domains,
                returns: Vec::new(),
                conditions: Vec::new(),
                expressions: Vec::new(),
                assignments: Vec::new(),
                helper_calls: Vec::new(),
                slots,
                declared_local_kinds: vec![None; kinds.len()],
                deferred_integers: Vec::new(),
            },
            DslValidationFlow {
                assigned,
                maybe_assigned,
                kinds,
                reachable: true,
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

    fn defer_integer(
        &mut self,
        expression: &Expr,
        flow: &DslValidationFlow,
        classification: DeferredIntegerClassification,
    ) -> DslStaticKind {
        if classification.default_domain == DslIntegerDomain::Dimension {
            self.promote_deferred_integer_defaults(&classification.dependencies);
        }
        let index = self.deferred_integers.len();
        self.deferred_integers.push(DeferredInteger {
            expression: expression.clone(),
            flow: flow.clone(),
            state: DeferredIntegerState::UnresolvedRoot {
                default_domain: classification.default_domain,
                dependencies: classification.dependencies,
            },
            validated: false,
        });
        DslStaticKind::DeferredInteger(index)
    }

    fn promote_deferred_integer_defaults(&mut self, dependencies: &[usize]) {
        let mut pending = dependencies.to_vec();
        let mut promoted = HashSet::new();
        while let Some(index) = pending.pop() {
            let root = self.deferred_integer_root(index);
            if !promoted.insert(root) {
                continue;
            }
            let dependencies = match &mut self.deferred_integers[root].state {
                DeferredIntegerState::UnresolvedRoot {
                    default_domain,
                    dependencies,
                } => {
                    *default_domain = DslIntegerDomain::Dimension;
                    dependencies.clone()
                }
                // A resolved root has already validated its complete group, so dependencies can
                // no longer affect its domain.
                DeferredIntegerState::ResolvedRoot(_) => Vec::new(),
                DeferredIntegerState::Redirect(_) => {
                    unreachable!("a deferred integer root cannot be a redirect")
                }
            };
            pending.extend(dependencies);
        }
    }

    fn deferred_integer_root(&self, mut index: usize) -> usize {
        while let DeferredIntegerState::Redirect(parent) = self.deferred_integers[index].state {
            index = parent;
        }
        index
    }

    fn deferred_integer_domain(&self, index: usize) -> (usize, Option<DslIntegerDomain>) {
        let root = self.deferred_integer_root(index);
        let domain = match self.deferred_integers[root].state {
            DeferredIntegerState::UnresolvedRoot { .. } => None,
            DeferredIntegerState::ResolvedRoot(domain) => Some(domain),
            DeferredIntegerState::Redirect(_) => {
                unreachable!("a deferred integer root cannot be a redirect")
            }
        };
        (root, domain)
    }

    fn merge_deferred_integers(
        &mut self,
        left: usize,
        right: usize,
    ) -> Result<usize, TypeShapeDslDefinitionError> {
        let right_range = self.deferred_integers[right].expression.range();
        let (left, left_domain) = self.deferred_integer_domain(left);
        let (right, right_domain) = self.deferred_integer_domain(right);
        if left == right {
            return Ok(left);
        }
        if left_domain.is_some() && right_domain.is_some() && left_domain != right_domain {
            return Err(TypeShapeDslDefinitionError {
                range: right_range,
                message: "an integer local cannot be used as both a dimension and a Flag value",
            });
        }
        let merged_state = match (
            self.deferred_integers[left].state.clone(),
            self.deferred_integers[right].state.clone(),
        ) {
            (
                DeferredIntegerState::UnresolvedRoot {
                    default_domain: left_default,
                    mut dependencies,
                },
                DeferredIntegerState::UnresolvedRoot {
                    default_domain: right_default,
                    dependencies: right_dependencies,
                },
            ) => {
                dependencies.extend(right_dependencies);
                dependencies.sort_unstable();
                dependencies.dedup();
                DeferredIntegerState::UnresolvedRoot {
                    default_domain: left_default.join(right_default),
                    dependencies,
                }
            }
            (
                DeferredIntegerState::ResolvedRoot(domain),
                DeferredIntegerState::UnresolvedRoot { .. },
            )
            | (
                DeferredIntegerState::UnresolvedRoot { .. },
                DeferredIntegerState::ResolvedRoot(domain),
            )
            | (DeferredIntegerState::ResolvedRoot(domain), DeferredIntegerState::ResolvedRoot(_)) =>
            {
                // Resolution fixes the merged group's domain, so unresolved default dependencies
                // are no longer needed.
                DeferredIntegerState::ResolvedRoot(domain)
            }
            (DeferredIntegerState::Redirect(_), _) | (_, DeferredIntegerState::Redirect(_)) => {
                unreachable!("canonical deferred integer roots cannot be redirects")
            }
        };
        self.deferred_integers[right].state = DeferredIntegerState::Redirect(left);
        match merged_state {
            DeferredIntegerState::ResolvedRoot(domain) => {
                self.resolve_deferred_integer(left, domain)?;
            }
            unresolved @ DeferredIntegerState::UnresolvedRoot { .. } => {
                self.deferred_integers[left].state = unresolved;
                if let DeferredIntegerState::UnresolvedRoot {
                    default_domain: DslIntegerDomain::Dimension,
                    dependencies,
                } = &self.deferred_integers[left].state
                {
                    let dependencies = dependencies.clone();
                    self.promote_deferred_integer_defaults(&dependencies);
                }
            }
            DeferredIntegerState::Redirect(_) => {
                unreachable!("merged deferred integer roots cannot be redirects")
            }
        }
        Ok(left)
    }

    fn collect_integer_expression_sources(
        &self,
        expression: &Expr,
        flow: &DslValidationFlow,
        sources: &mut DslIntegerExpressionSources,
        expanded_deferred: &mut HashSet<usize>,
    ) -> bool {
        match expression {
            Expr::Name(name) => {
                let Some(&slot) = self.slots.get(&name.id) else {
                    // Deferred validation will report the unbound name with its normal diagnostic.
                    return true;
                };
                let kind = &flow.kinds[slot];
                if let Some(origins) = kind.unnarrowed_parameter_origins() {
                    for &parameter in origins {
                        record_parameter_use(
                            &mut sources.parameter_uses,
                            TypeShapeDslParameterUse {
                                parameter,
                                narrowing: TypeShapeDslParameterNarrowing::Unnarrowed,
                            },
                        );
                    }
                    true
                } else {
                    match kind {
                        DslStaticKind::ValueSet {
                            sources: value_sources,
                            kinds: FLAG_INT,
                        } => {
                            if !value_sources.non_parameter_values_are(FLAG_INT) {
                                return false;
                            }
                            for use_ in &value_sources.parameter_uses {
                                record_parameter_use(&mut sources.parameter_uses, *use_);
                            }
                            true
                        }
                        DslStaticKind::DeferredInteger(index) => {
                            let root = self.deferred_integer_root(*index);
                            let mut helper_compatible = true;
                            for (index, deferred) in self.deferred_integers.iter().enumerate() {
                                if self.deferred_integer_root(index) == root
                                    && expanded_deferred.insert(index)
                                {
                                    helper_compatible &= self.collect_integer_expression_sources(
                                        &deferred.expression,
                                        &deferred.flow,
                                        sources,
                                        expanded_deferred,
                                    );
                                }
                            }
                            helper_compatible
                        }
                        DslStaticKind::Dimension => {
                            sources.has_shape_source = true;
                            false
                        }
                        _ => false,
                    }
                }
            }
            Expr::BinOp(binop) => {
                let left = self.collect_integer_expression_sources(
                    &binop.left,
                    flow,
                    sources,
                    expanded_deferred,
                );
                let right = self.collect_integer_expression_sources(
                    &binop.right,
                    flow,
                    sources,
                    expanded_deferred,
                );
                left && right
            }
            Expr::If(if_expr) => {
                // Conditional expressions are valid integer values, but helpers retain their
                // existing narrower forwarding syntax.
                self.collect_integer_expression_sources(
                    &if_expr.body,
                    flow,
                    sources,
                    expanded_deferred,
                );
                self.collect_integer_expression_sources(
                    &if_expr.orelse,
                    flow,
                    sources,
                    expanded_deferred,
                );
                false
            }
            // `len(shape)` creates a new integer cardinality; the shape operand is not an
            // integer-value dependency of that result. Other `len` operands are not
            // domain-polymorphic deferred integers.
            Expr::Call(call) if self.intrinsic(&call.func) == Some(TypeShapeDslIntrinsic::Len) => {
                let has_int_tuple_operand = self.length_has_int_tuple_operand(call, flow);
                sources.has_shape_source |= has_int_tuple_operand;
                has_int_tuple_operand
            }
            Expr::Subscript(subscript) if !matches!(subscript.slice.as_ref(), Expr::Slice(_)) => {
                if matches!(
                    self.classify_subscript(subscript, flow),
                    Ok(DslSubscriptKind::UnresolvedIndex { .. }
                        | DslSubscriptKind::IntTupleIndex { .. }
                        | DslSubscriptKind::IntTuplesIndex { .. })
                ) {
                    sources.has_shape_source = true;
                }
                false
            }
            Expr::Call(call)
                if matches!(
                    self.intrinsic(&call.func),
                    Some(
                        TypeShapeDslIntrinsic::Prod
                            | TypeShapeDslIntrinsic::Sum
                            | TypeShapeDslIntrinsic::Gradual(TypeShapeDslDomain::Int)
                    )
                ) =>
            {
                sources.has_shape_source = true;
                false
            }
            _ => !matches!(integer_literal(expression), IntegerLiteral::NotLiteral),
        }
    }

    /// Whether an integer-syntax expression contains a shape-domain source.
    ///
    /// This is deliberately separate from deciding whether the expression is a valid integer:
    /// both `count: Flag[int]` and `size: Int` may be used in integer syntax, but only the latter
    /// carries symbolic shape dimensions. Resolved parameter domains supply that distinction,
    /// while the shared source traversal handles narrowed parameters and cyclic deferred locals.
    fn integer_expression_has_shape_source(
        &self,
        expression: &Expr,
        flow: &DslValidationFlow,
    ) -> bool {
        let Some(parameter_domains) = self.parameter_domains else {
            return false;
        };
        let mut sources = DslIntegerExpressionSources::default();
        self.collect_integer_expression_sources(
            expression,
            flow,
            &mut sources,
            &mut HashSet::new(),
        );
        sources.has_shape_source
            || sources.parameter_uses.iter().any(|use_| {
                parameter_domains[use_.parameter]
                    .can_use_as(TypeShapeDslDomain::Int, use_.narrowing)
            })
    }

    fn deferred_integer_parameter_uses(
        &self,
        index: usize,
    ) -> Result<Box<[TypeShapeDslParameterUse]>, TypeShapeDslDefinitionError> {
        let root = self.deferred_integer_root(index);
        let mut sources = DslIntegerExpressionSources::default();
        let mut expanded_deferred = HashSet::new();
        for (index, deferred) in self.deferred_integers.iter().enumerate() {
            if self.deferred_integer_root(index) == root
                && expanded_deferred.insert(index)
                && !self.collect_integer_expression_sources(
                    &deferred.expression,
                    &deferred.flow,
                    &mut sources,
                    &mut expanded_deferred,
                )
            {
                return Err(TypeShapeDslDefinitionError {
                    range: deferred.expression.range(),
                    message: "helper integer arguments must contain only parameters, Flag integers, and integer literals",
                });
            }
        }
        sources
            .parameter_uses
            .sort_unstable_by_key(|use_| use_.parameter);
        Ok(sources.parameter_uses.into_boxed_slice())
    }

    fn finalize_helper_deferred_domains(&mut self) -> Result<(), TypeShapeDslDefinitionError> {
        let domains = self
            .helper_calls
            .iter()
            .flat_map(|call| &call.arguments)
            .filter_map(|argument| match &argument.source {
                TypeShapeDslHelperArgumentSource::DeferredInteger { index, .. } => Some(*index),
                _ => None,
            })
            .map(|index| {
                let (root, domain) = self.deferred_integer_domain(index);
                Ok((
                    index,
                    (
                        root,
                        self.deferred_integer_parameter_uses(root)?,
                        domain.map(DslIntegerDomain::input_domain),
                    ),
                ))
            })
            .collect::<Result<HashMap<_, _>, TypeShapeDslDefinitionError>>()?;
        for argument in self
            .helper_calls
            .iter_mut()
            .flat_map(|call| &mut call.arguments)
        {
            let TypeShapeDslHelperArgumentSource::DeferredInteger {
                index,
                parameter_uses,
                resolved_domain,
            } = &mut argument.source
            else {
                continue;
            };
            let (root, final_parameter_uses, domain) = &domains[index];
            // Helper arguments may retain any member of a merged group; store the canonical root
            // so later resolution observes the shared domain.
            *index = *root;
            *parameter_uses = final_parameter_uses.clone();
            *resolved_domain = *domain;
        }
        Ok(())
    }

    fn resolve_deferred_integer(
        &mut self,
        index: usize,
        domain: DslIntegerDomain,
    ) -> Result<(), TypeShapeDslDefinitionError> {
        let (root, previous) = self.deferred_integer_domain(index);
        if previous.is_some() && previous != Some(domain) {
            return Err(TypeShapeDslDefinitionError {
                range: self.deferred_integers[index].expression.range(),
                message: "an integer local cannot be used as both a dimension and a Flag value",
            });
        }
        self.deferred_integers[root].state = DeferredIntegerState::ResolvedRoot(domain);
        let group = (0..self.deferred_integers.len())
            .filter(|index| self.deferred_integer_root(*index) == root)
            .collect::<Vec<_>>();
        for index in group {
            if self.deferred_integers[index].validated {
                continue;
            }
            self.deferred_integers[index].validated = true;
            let expression = self.deferred_integers[index].expression.clone();
            let flow = self.deferred_integers[index].flow.clone();
            match domain {
                DslIntegerDomain::Flag => self.validate_flag_int(&expression, &flow)?,
                DslIntegerDomain::Dimension => self.validate_dimension(&expression, &flow)?,
            }
        }
        Ok(())
    }

    fn resolve_unused_deferred_integers(&mut self) -> Result<(), TypeShapeDslDefinitionError> {
        for index in 0..self.deferred_integers.len() {
            let root = self.deferred_integer_root(index);
            let default_domain = match self.deferred_integers[root].state {
                DeferredIntegerState::UnresolvedRoot { default_domain, .. } => Some(default_domain),
                DeferredIntegerState::ResolvedRoot(_) => None,
                DeferredIntegerState::Redirect(_) => {
                    unreachable!("a deferred integer root cannot be a redirect")
                }
            };
            if let Some(default_domain) = default_domain {
                self.resolve_deferred_integer(root, default_domain)?;
            }
        }
        Ok(())
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
        if let Expr::If(if_expr) = expression {
            let (when_true, when_false) = self.validate_condition(&if_expr.test, flow)?;
            self.validate_dimension(&if_expr.body, &when_true)?;
            self.validate_dimension(&if_expr.orelse, &when_false)?;
            self.expressions.push(TypeShapeDslExpression {
                range: if_expr.range,
                kind: TypeShapeDslExpressionKind::Conditional,
            });
            return Ok(());
        }
        let kind = match expression {
            Expr::Name(_) => {
                let slot = self.slot(expression, flow)?;
                match &flow.kinds[slot] {
                    kind if kind.unnarrowed_parameter_origins().is_some() => {
                        TypeShapeDslExpressionKind::DimensionSlot {
                            slot,
                            parameter_uses: kind.parameter_uses(),
                        }
                    }
                    DslStaticKind::Dimension => TypeShapeDslExpressionKind::DimensionSlot {
                        slot,
                        parameter_uses: None,
                    },
                    DslStaticKind::DeferredInteger(index) => {
                        self.resolve_deferred_integer(*index, DslIntegerDomain::Dimension)?;
                        TypeShapeDslExpressionKind::DimensionSlot {
                            slot,
                            parameter_uses: None,
                        }
                    }
                    DslStaticKind::GeneratorElement {
                        source_parameter_uses,
                    } => TypeShapeDslExpressionKind::GeneratorElementAsDimension {
                        slot,
                        source_parameter_uses: source_parameter_uses.clone(),
                    },
                    DslStaticKind::ValueSet { sources, kinds }
                        if kinds & FLAG_NONE == 0 && sources.non_parameter_values_are(FLAG_INT) =>
                    {
                        TypeShapeDslExpressionKind::DimensionSlot {
                            slot,
                            parameter_uses: flow.kinds[slot].parameter_uses(),
                        }
                    }
                    _ => {
                        return Err(TypeShapeDslDefinitionError {
                            range: expression.range(),
                            message: "`IntTuple` elements must be dimension values",
                        });
                    }
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
            Expr::BinOp(binop) => {
                let op = match binop.op {
                    Operator::Add => TypeShapeDslArithmeticOp::Add,
                    Operator::Sub => TypeShapeDslArithmeticOp::Subtract,
                    Operator::Mult => TypeShapeDslArithmeticOp::Multiply,
                    Operator::FloorDiv => TypeShapeDslArithmeticOp::FloorDivide,
                    Operator::Mod => TypeShapeDslArithmeticOp::Modulo,
                    _ => {
                        return Err(TypeShapeDslDefinitionError {
                            range: binop.range,
                            message: "dimension arithmetic supports only `+`, `-`, `*`, `//`, and `%`",
                        });
                    }
                };
                self.validate_dimension_arithmetic_operand(&binop.left, flow)?;
                self.validate_dimension_arithmetic_operand(&binop.right, flow)?;
                TypeShapeDslExpressionKind::DimensionArithmetic(op)
            }
            Expr::Subscript(subscript) => {
                let shape = match self.classify_subscript(subscript, flow)? {
                    DslSubscriptKind::UnresolvedIndex { source, .. }
                    | DslSubscriptKind::IntTupleIndex { source, .. } => source,
                    DslSubscriptKind::IntTupleSlice { .. }
                    | DslSubscriptKind::IntTuplesIndex { .. } => {
                        return Err(TypeShapeDslDefinitionError {
                            range: subscript.value.range(),
                            message: "indexed dimension source must be an `IntTuple` value",
                        });
                    }
                };
                let kind = &flow.kinds[shape];
                let parameter_origins = if let Some(parameters) =
                    kind.unnarrowed_parameter_origins()
                {
                    Some(parameters.into())
                } else {
                    match kind {
                        DslStaticKind::IntTuple { parameter_origins } => parameter_origins.clone(),
                        DslStaticKind::GeneratorElement {
                            source_parameter_uses: Some(source_parameter_uses),
                        } => {
                            self.expressions.push(TypeShapeDslExpression {
                                range: subscript.value.range(),
                                kind: TypeShapeDslExpressionKind::GeneratorElementAsIntTuple {
                                    slot: shape,
                                    source_parameter_uses: source_parameter_uses.clone(),
                                },
                            });
                            None
                        }
                        _ => {
                            return Err(TypeShapeDslDefinitionError {
                                range: subscript.value.range(),
                                message: "indexed dimension source must be an `IntTuple` value",
                            });
                        }
                    }
                };
                self.validate_flag_int(&subscript.slice, flow)?;
                TypeShapeDslExpressionKind::IntTupleIndex {
                    shape,
                    parameter_origins,
                }
            }
            Expr::Call(call)
                if matches!(
                    self.intrinsic(&call.func),
                    Some(TypeShapeDslIntrinsic::Prod | TypeShapeDslIntrinsic::Sum)
                ) =>
            {
                self.validate_int_tuple_reduction(call, flow)?
            }
            Expr::Call(call)
                if self.intrinsic(&call.func)
                    == Some(TypeShapeDslIntrinsic::Gradual(TypeShapeDslDomain::Int)) =>
            {
                Self::validate_gradual_call(call)?;
                TypeShapeDslExpressionKind::Gradual
            }
            Expr::Call(call) if self.intrinsic(&call.func) == Some(TypeShapeDslIntrinsic::Len) => {
                self.validate_length(call, flow, DslIntegerDomain::Dimension)?
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

    fn validate_dimension_arithmetic_operand(
        &mut self,
        expression: &Expr,
        flow: &DslValidationFlow,
    ) -> Result<(), TypeShapeDslDefinitionError> {
        let Expr::Name(_) = expression else {
            return self.validate_dimension(expression, flow);
        };
        let slot = self.slot(expression, flow)?;
        let parameter_uses = match &flow.kinds[slot] {
            kind if kind.unnarrowed_parameter_origins().is_some() => kind.parameter_uses(),
            DslStaticKind::DeferredInteger(index) => {
                self.resolve_deferred_integer(*index, DslIntegerDomain::Dimension)?;
                None
            }
            DslStaticKind::Dimension | DslStaticKind::GeneratorElement { .. } => None,
            DslStaticKind::ValueSet { sources, kinds }
                if kinds & FLAG_NONE == 0 && sources.non_parameter_values_are(FLAG_INT) =>
            {
                sources.parameter_uses()
            }
            _ => {
                return Err(TypeShapeDslDefinitionError {
                    range: expression.range(),
                    message: "dimension arithmetic operands must be integer values; an `Int | None` operand must be narrowed to exclude `None`",
                });
            }
        };
        self.expressions.push(TypeShapeDslExpression {
            range: expression.range(),
            kind: TypeShapeDslExpressionKind::IntegerSlot {
                slot,
                parameter_uses,
            },
        });
        Ok(())
    }

    fn validate_flag_slot(
        &mut self,
        expression: &Expr,
        flow: &DslValidationFlow,
        required: TypeShapeDslFlagValueKind,
    ) -> Result<(), TypeShapeDslDefinitionError> {
        let slot = self.slot(expression, flow)?;
        let expected = match required {
            TypeShapeDslFlagValueKind::Int => FLAG_INT,
            TypeShapeDslFlagValueKind::String => FLAG_STRING,
            TypeShapeDslFlagValueKind::Sequence => FLAG_SEQUENCE,
        };
        if let DslStaticKind::DeferredInteger(index) = flow.kinds[slot] {
            if expected != FLAG_INT {
                return Err(TypeShapeDslDefinitionError {
                    range: expression.range(),
                    message: "integer local has the wrong domain for this Flag operation",
                });
            }
            self.resolve_deferred_integer(index, DslIntegerDomain::Flag)?;
        }
        let parameter_uses = match &flow.kinds[slot] {
            kind if kind.unnarrowed_parameter_origins().is_some() => kind.parameter_uses(),
            DslStaticKind::DeferredInteger(_) if expected == FLAG_INT => None,
            DslStaticKind::GeneratorElement {
                source_parameter_uses,
            } if expected == FLAG_INT => {
                self.expressions.push(TypeShapeDslExpression {
                    range: expression.range(),
                    kind: TypeShapeDslExpressionKind::GeneratorElementAsFlagInt {
                        slot,
                        source_parameter_uses: source_parameter_uses.clone(),
                    },
                });
                return Ok(());
            }
            DslStaticKind::ValueSet { sources, kinds }
                if *kinds != 0
                    && (kinds & !expected == 0
                        || (required == TypeShapeDslFlagValueKind::String
                            && kinds & FLAG_STRING != 0
                            && kinds & !(FLAG_STRING | FLAG_NONE) == 0)) =>
            {
                sources.parameter_uses()
            }
            DslStaticKind::ValueSet { sources, kinds }
                if required == TypeShapeDslFlagValueKind::String
                    && (*kinds == FLAG_NONE
                        || (*kinds == FLAG_NOT_NONE
                            && sources.non_parameter_values_are(FLAG_STRING))) =>
            {
                sources.parameter_uses()
            }
            _ => {
                return Err(TypeShapeDslDefinitionError {
                    range: expression.range(),
                    message: "Flag value has the wrong domain for this operation",
                });
            }
        };
        self.expressions.push(TypeShapeDslExpression {
            range: expression.range(),
            kind: TypeShapeDslExpressionKind::FlagValueSlot {
                slot,
                parameter_uses,
                required,
            },
        });
        Ok(())
    }

    fn validate_flag_string(
        &mut self,
        expression: &Expr,
        flow: &DslValidationFlow,
    ) -> Result<(), TypeShapeDslDefinitionError> {
        match expression {
            Expr::StringLiteral(literal) if Lit::from_string_literal(literal).is_some() => {
                self.expressions.push(TypeShapeDslExpression {
                    range: expression.range(),
                    kind: TypeShapeDslExpressionKind::FlagStringLiteral,
                });
                Ok(())
            }
            Expr::Name(_) => {
                self.validate_flag_slot(expression, flow, TypeShapeDslFlagValueKind::String)
            }
            _ => Err(TypeShapeDslDefinitionError {
                range: expression.range(),
                message: "Flag string expressions support only literals and immutable aliases",
            }),
        }
    }

    fn validate_flag_int(
        &mut self,
        expression: &Expr,
        flow: &DslValidationFlow,
    ) -> Result<(), TypeShapeDslDefinitionError> {
        if let Expr::If(if_expr) = expression {
            let (when_true, when_false) = self.validate_condition(&if_expr.test, flow)?;
            self.validate_flag_int(&if_expr.body, &when_true)?;
            self.validate_flag_int(&if_expr.orelse, &when_false)?;
            self.expressions.push(TypeShapeDslExpression {
                range: if_expr.range,
                kind: TypeShapeDslExpressionKind::Conditional,
            });
            return Ok(());
        }
        match integer_literal(expression) {
            IntegerLiteral::NotLiteral => {}
            IntegerLiteral::Unrepresentable { .. } => {
                self.expressions.push(TypeShapeDslExpression {
                    range: expression.range(),
                    kind: TypeShapeDslExpressionKind::FlagIntLiteral(None),
                });
                return Ok(());
            }
            IntegerLiteral::Value(value) => {
                self.expressions.push(TypeShapeDslExpression {
                    range: expression.range(),
                    kind: TypeShapeDslExpressionKind::FlagIntLiteral(Some(value)),
                });
                return Ok(());
            }
        }
        match expression {
            Expr::Name(_) => {
                self.validate_flag_slot(expression, flow, TypeShapeDslFlagValueKind::Int)
            }
            Expr::BinOp(binop) => {
                let op = match binop.op {
                    Operator::Add => TypeShapeDslArithmeticOp::Add,
                    Operator::Sub => TypeShapeDslArithmeticOp::Subtract,
                    Operator::Mult => TypeShapeDslArithmeticOp::Multiply,
                    Operator::FloorDiv => TypeShapeDslArithmeticOp::FloorDivide,
                    Operator::Mod => TypeShapeDslArithmeticOp::Modulo,
                    _ => {
                        return Err(TypeShapeDslDefinitionError {
                            range: binop.range,
                            message: "Flag integer arithmetic supports only `+`, `-`, `*`, `//`, and `%`",
                        });
                    }
                };
                self.validate_flag_int(&binop.left, flow)?;
                self.validate_flag_int(&binop.right, flow)?;
                self.expressions.push(TypeShapeDslExpression {
                    range: binop.range,
                    kind: TypeShapeDslExpressionKind::FlagIntArithmetic(op),
                });
                Ok(())
            }
            Expr::Call(call) if self.intrinsic(&call.func) == Some(TypeShapeDslIntrinsic::Len) => {
                let kind = self.validate_length(call, flow, DslIntegerDomain::Flag)?;
                self.expressions.push(TypeShapeDslExpression {
                    range: call.range(),
                    kind,
                });
                Ok(())
            }
            Expr::Call(call)
                if matches!(
                    &*call.func,
                    Expr::Attribute(attribute)
                        if matches!(attribute.attr.id.as_str(), "count" | "index")
                ) =>
            {
                let Expr::Attribute(attribute) = &*call.func else {
                    unreachable!("guarded sequence method call has an attribute callee")
                };
                let (kind, arity_message) = match attribute.attr.id.as_str() {
                    "count" => (
                        TypeShapeDslExpressionKind::FlagSequenceCount,
                        "Flag sequence `.count` requires exactly one positional argument",
                    ),
                    "index" => (
                        TypeShapeDslExpressionKind::FlagSequenceIndex,
                        "Flag sequence `.index` requires exactly one positional argument",
                    ),
                    _ => unreachable!("guarded sequence method is `count` or `index`"),
                };
                if call.arguments.args.len() != 1 || !call.arguments.keywords.is_empty() {
                    return Err(TypeShapeDslDefinitionError {
                        range: call.arguments.range,
                        message: arity_message,
                    });
                }
                self.validate_flag_sequence(&attribute.value, flow)?;
                self.validate_flag_int(&call.arguments.args[0], flow)?;
                self.expressions.push(TypeShapeDslExpression {
                    range: call.range(),
                    kind,
                });
                Ok(())
            }
            _ => Err(TypeShapeDslDefinitionError {
                range: expression.range(),
                message: "Flag integer expression is not supported",
            }),
        }
    }

    /// Classifies a `len` operand without recording anything, so that validation, deferred
    /// integer resolution, and traceability all agree on which `len` calls are well formed.
    fn length_operand<'b>(
        &self,
        call: &ExprCall,
        flow: &'b DslValidationFlow,
    ) -> Result<DslLengthOperand<'b>, TypeShapeDslDefinitionError> {
        if call.arguments.args.len() != 1 || !call.arguments.keywords.is_empty() {
            return Err(TypeShapeDslDefinitionError {
                range: call.arguments.range,
                message: "`len` requires exactly one positional argument",
            });
        }
        let argument = &call.arguments.args[0];
        let slot = self.slot(argument, flow)?;
        match &flow.kinds[slot] {
            DslStaticKind::UnknownParameters(parameters)
            | DslStaticKind::IntTuples {
                parameter_origins: Some(parameters),
            } => Ok(DslLengthOperand::IntTuple {
                slot,
                parameter_origins: Some(parameters),
            }),
            DslStaticKind::IntTuple { parameter_origins } => Ok(DslLengthOperand::IntTuple {
                slot,
                parameter_origins: parameter_origins.as_deref(),
            }),
            DslStaticKind::GeneratorElement {
                source_parameter_uses: Some(source_parameter_uses),
            } => Ok(DslLengthOperand::GeneratorIntTuple {
                slot,
                source_parameter_uses,
            }),
            DslStaticKind::ValueSet {
                kinds: FLAG_SEQUENCE,
                ..
            } => Ok(DslLengthOperand::FlagSequence),
            _ => Err(TypeShapeDslDefinitionError {
                range: argument.range(),
                message: "`len` requires an IntTuple or Flag sequence",
            }),
        }
    }

    fn length_has_int_tuple_operand(&self, call: &ExprCall, flow: &DslValidationFlow) -> bool {
        matches!(
            self.length_operand(call, flow),
            Ok(DslLengthOperand::IntTuple { .. } | DslLengthOperand::GeneratorIntTuple { .. })
        )
    }

    fn validate_length(
        &mut self,
        call: &ExprCall,
        flow: &DslValidationFlow,
        domain: DslIntegerDomain,
    ) -> Result<TypeShapeDslExpressionKind, TypeShapeDslDefinitionError> {
        let operand = self.length_operand(call, flow)?;
        // `length_operand` has already checked arity, so the sole argument exists.
        let argument = &call.arguments.args[0];
        match operand {
            DslLengthOperand::IntTuple {
                slot,
                parameter_origins,
            } => Ok(TypeShapeDslExpressionKind::IntTupleLength {
                shape: slot,
                parameter_origins: parameter_origins.map(Box::from),
                domain,
            }),
            DslLengthOperand::GeneratorIntTuple {
                slot,
                source_parameter_uses,
            } => {
                self.expressions.push(TypeShapeDslExpression {
                    range: argument.range(),
                    kind: TypeShapeDslExpressionKind::GeneratorElementAsIntTuple {
                        slot,
                        source_parameter_uses: Box::from(source_parameter_uses),
                    },
                });
                Ok(TypeShapeDslExpressionKind::IntTupleLength {
                    shape: slot,
                    parameter_origins: None,
                    domain,
                })
            }
            DslLengthOperand::FlagSequence if domain == DslIntegerDomain::Flag => {
                self.validate_flag_slot(argument, flow, TypeShapeDslFlagValueKind::Sequence)?;
                Ok(TypeShapeDslExpressionKind::FlagSequenceLength)
            }
            DslLengthOperand::FlagSequence => Err(TypeShapeDslDefinitionError {
                range: argument.range(),
                message: "Flag-sequence length cannot be used as a dimension",
            }),
        }
    }

    fn validate_flag_sequence(
        &mut self,
        expression: &Expr,
        flow: &DslValidationFlow,
    ) -> Result<(), TypeShapeDslDefinitionError> {
        match expression {
            Expr::Name(_) => {
                self.validate_flag_slot(expression, flow, TypeShapeDslFlagValueKind::Sequence)
            }
            Expr::Tuple(tuple) => {
                for element in &tuple.elts {
                    if matches!(element, Expr::Starred(_)) {
                        return Err(TypeShapeDslDefinitionError {
                            range: element.range(),
                            message: "Flag tuple expressions do not support starred elements",
                        });
                    }
                    self.validate_flag_int(element, flow)?;
                }
                self.expressions.push(TypeShapeDslExpression {
                    range: tuple.range,
                    kind: TypeShapeDslExpressionKind::FlagTuple,
                });
                Ok(())
            }
            Expr::Call(call)
                if self.intrinsic(&call.func) == Some(TypeShapeDslIntrinsic::Range) =>
            {
                if !(1..=3).contains(&call.arguments.args.len())
                    || !call.arguments.keywords.is_empty()
                {
                    return Err(TypeShapeDslDefinitionError {
                        range: call.arguments.range,
                        message: "`range` requires one to three positional arguments",
                    });
                }
                for argument in &call.arguments.args {
                    self.validate_flag_int(argument, flow)?;
                }
                self.expressions.push(TypeShapeDslExpression {
                    range: call.range(),
                    kind: TypeShapeDslExpressionKind::FlagRange,
                });
                Ok(())
            }
            Expr::Call(call)
                if self.intrinsic(&call.func) == Some(TypeShapeDslIntrinsic::Tuple) =>
            {
                if call.arguments.args.len() != 1 || !call.arguments.keywords.is_empty() {
                    return Err(TypeShapeDslDefinitionError {
                        range: call.arguments.range,
                        message: "`tuple` requires exactly one positional generator argument",
                    });
                }
                let Some(Expr::Generator(generator)) = call.arguments.args.first() else {
                    return Err(TypeShapeDslDefinitionError {
                        range: call.arguments.args[0].range(),
                        message: "`tuple` argument must be a bounded generator",
                    });
                };
                let (binder, binders) =
                    self.validate_generator(generator, flow, GeneratorValidationKind::FlagValue)?;
                self.expressions.push(TypeShapeDslExpression {
                    range: call.range(),
                    kind: TypeShapeDslExpressionKind::FlagGenerator { binder, binders },
                });
                Ok(())
            }
            _ => Err(TypeShapeDslDefinitionError {
                range: expression.range(),
                message: "Flag sequence must be a Flag value, tuple display, `tuple(...)`, or `range(...)`",
            }),
        }
    }

    fn validate_generator_source(
        &mut self,
        source: &Expr,
        flow: &DslValidationFlow,
        allow_zip: bool,
        allow_int_tuples: bool,
    ) -> Result<TypeShapeDslGeneratorSource, TypeShapeDslDefinitionError> {
        if let Expr::Name(_) = source {
            let slot = self.slot(source, flow)?;
            let kind = &flow.kinds[slot];
            let (parameter_uses, element_kind) = match kind {
                kind if kind.unnarrowed_parameter_origins().is_some() => {
                    let parameter_uses = kind.parameter_uses();
                    (
                        parameter_uses.clone(),
                        DslStaticKind::GeneratorElement {
                            source_parameter_uses: parameter_uses,
                        },
                    )
                }
                DslStaticKind::IntTuple {
                    parameter_origins: None,
                } => (
                    None,
                    DslStaticKind::GeneratorElement {
                        source_parameter_uses: None,
                    },
                ),
                DslStaticKind::IntTuples {
                    parameter_origins: None,
                } if allow_int_tuples => (
                    None,
                    DslStaticKind::IntTuple {
                        parameter_origins: None,
                    },
                ),
                DslStaticKind::ValueSet { sources, kinds }
                    if *kinds == FLAG_SEQUENCE
                        && sources.non_parameter_values_are(FLAG_SEQUENCE) =>
                {
                    (
                        sources.parameter_uses(),
                        DslStaticKind::GeneratorElement {
                            source_parameter_uses: None,
                        },
                    )
                }
                _ => {
                    return Err(TypeShapeDslDefinitionError {
                        range: source.range(),
                        message: if allow_int_tuples {
                            "generator source must be an IntTuple, IntTuples, or Flag sequence"
                        } else {
                            "generator source must be an IntTuple or Flag sequence"
                        },
                    });
                }
            };
            self.expressions.push(TypeShapeDslExpression {
                range: source.range(),
                kind: TypeShapeDslExpressionKind::GeneratorSourceSlot {
                    slot,
                    parameter_uses,
                    allow_int_tuples,
                },
            });
            return Ok(TypeShapeDslGeneratorSource::Single(element_kind));
        }
        match source {
            Expr::Call(call) if self.intrinsic(&call.func) == Some(TypeShapeDslIntrinsic::Zip) => {
                if !allow_zip {
                    return Err(TypeShapeDslDefinitionError {
                        range: source.range(),
                        message: "`zip` sources are only supported in constructor generators",
                    });
                }
                if !call.arguments.keywords.is_empty() {
                    return Err(TypeShapeDslDefinitionError {
                        range: call.arguments.range,
                        message: "generator `zip` does not support keyword arguments",
                    });
                }
                let element_kinds = call
                    .arguments
                    .args
                    .iter()
                    .map(|argument| {
                        let source = self.validate_generator_source(
                            argument,
                            flow,
                            false,
                            allow_int_tuples,
                        )?;
                        match source {
                            TypeShapeDslGeneratorSource::Single(kind) => Ok(kind),
                            TypeShapeDslGeneratorSource::Zip(_) => {
                                unreachable!("nested generator zip sources are rejected")
                            }
                        }
                    })
                    .collect::<Result<Vec<_>, TypeShapeDslDefinitionError>>()?;
                self.expressions.push(TypeShapeDslExpression {
                    range: call.range(),
                    kind: TypeShapeDslExpressionKind::GeneratorZip {
                        sources: call.arguments.args.len(),
                    },
                });
                Ok(TypeShapeDslGeneratorSource::Zip(element_kinds))
            }
            Expr::Call(call)
                if self.intrinsic(&call.func) == Some(TypeShapeDslIntrinsic::IntTuple) =>
            {
                if matches!(call.arguments.args.first(), Some(Expr::Generator(_))) {
                    return Err(TypeShapeDslDefinitionError {
                        range: source.range(),
                        message: "nested generators are not supported",
                    });
                }
                self.validate_int_tuple_constructor(call, flow)?;
                Ok(TypeShapeDslGeneratorSource::Single(
                    DslStaticKind::GeneratorElement {
                        source_parameter_uses: None,
                    },
                ))
            }
            Expr::Call(call)
                if allow_int_tuples
                    && self.intrinsic(&call.func) == Some(TypeShapeDslIntrinsic::IntTuples) =>
            {
                self.validate_int_tuples_constructor(call, flow)?;
                Ok(TypeShapeDslGeneratorSource::Single(
                    DslStaticKind::IntTuple {
                        parameter_origins: None,
                    },
                ))
            }
            Expr::Call(call)
                if self.intrinsic(&call.func) == Some(TypeShapeDslIntrinsic::Tuple)
                    && matches!(call.arguments.args.first(), Some(Expr::Generator(_))) =>
            {
                Err(TypeShapeDslDefinitionError {
                    range: source.range(),
                    message: "nested generators are not supported",
                })
            }
            Expr::Tuple(_) | Expr::Call(_) => {
                self.validate_flag_sequence(source, flow)?;
                Ok(TypeShapeDslGeneratorSource::Single(
                    DslStaticKind::GeneratorElement {
                        source_parameter_uses: None,
                    },
                ))
            }
            _ => Err(TypeShapeDslDefinitionError {
                range: source.range(),
                message: if allow_int_tuples {
                    "generator source must be an IntTuple, IntTuples, Flag sequence, or `zip(...)`"
                } else {
                    "generator source must be an IntTuple, Flag sequence, or `zip(...)`"
                },
            }),
        }
    }

    fn validate_generator(
        &mut self,
        generator: &ExprGenerator,
        flow: &DslValidationFlow,
        kind: GeneratorValidationKind,
    ) -> Result<(usize, usize), TypeShapeDslDefinitionError> {
        let [comprehension] = generator.generators.as_slice() else {
            return Err(TypeShapeDslDefinitionError {
                range: generator.range,
                message: match kind {
                    GeneratorValidationKind::Condition => {
                        "`any` generators require exactly one `for` clause"
                    }
                    GeneratorValidationKind::Dimension
                    | GeneratorValidationKind::FlagValue
                    | GeneratorValidationKind::IntTuple => {
                        "constructor generators require exactly one `for` clause"
                    }
                },
            });
        };
        if comprehension.is_async {
            return Err(TypeShapeDslDefinitionError {
                range: comprehension.range,
                message: "async generators are not supported",
            });
        }
        if comprehension.ifs.len() > 1 {
            return Err(TypeShapeDslDefinitionError {
                range: comprehension.range,
                message: match kind {
                    GeneratorValidationKind::Condition => {
                        "`any` generators support at most one `if` filter"
                    }
                    GeneratorValidationKind::Dimension
                    | GeneratorValidationKind::FlagValue
                    | GeneratorValidationKind::IntTuple => {
                        "constructor generators support at most one `if` filter"
                    }
                },
            });
        }
        let source_is_zip = matches!(
            &comprehension.iter,
            Expr::Call(call)
                if self.intrinsic(&call.func) == Some(TypeShapeDslIntrinsic::Zip)
        );
        if !source_is_zip && !matches!(&comprehension.target, Expr::Name(_)) {
            return Err(TypeShapeDslDefinitionError {
                range: comprehension.target.range(),
                message: "generator target must be exactly one bare name",
            });
        }
        let source = self.validate_generator_source(
            &comprehension.iter,
            flow,
            kind != GeneratorValidationKind::Condition,
            kind != GeneratorValidationKind::Condition,
        )?;
        let (targets, element_kinds) = match (source, &comprehension.target) {
            (TypeShapeDslGeneratorSource::Single(kind), Expr::Name(target)) => {
                (vec![target], vec![kind])
            }
            (TypeShapeDslGeneratorSource::Zip(kinds), Expr::Tuple(targets))
                if targets.elts.len() == kinds.len() =>
            {
                let mut names: Vec<&ExprName> = Vec::with_capacity(kinds.len());
                for target in &targets.elts {
                    let Expr::Name(target) = target else {
                        return Err(TypeShapeDslDefinitionError {
                            range: target.range(),
                            message: "generator tuple targets require one bare name per `zip` source",
                        });
                    };
                    if names.iter().any(|name| name.id == target.id) {
                        return Err(TypeShapeDslDefinitionError {
                            range: target.range,
                            message: "generator tuple target names must be distinct",
                        });
                    }
                    names.push(target);
                }
                (names, kinds)
            }
            (TypeShapeDslGeneratorSource::Zip(_), Expr::Tuple(_)) => {
                return Err(TypeShapeDslDefinitionError {
                    range: comprehension.target.range(),
                    message: "generator tuple target arity must match the number of `zip` sources",
                });
            }
            (TypeShapeDslGeneratorSource::Zip(_), _) => {
                return Err(TypeShapeDslDefinitionError {
                    range: comprehension.target.range(),
                    message: "generator `zip` sources require a fixed tuple target",
                });
            }
            (TypeShapeDslGeneratorSource::Single(_), _) => {
                return Err(TypeShapeDslDefinitionError {
                    range: comprehension.target.range(),
                    message: "generator target must be exactly one bare name",
                });
            }
        };
        let binder = self.declared_local_kinds.len();
        let mut previous = Vec::with_capacity(targets.len());
        for (target, element_kind) in targets.iter().zip(&element_kinds) {
            let slot = self.declared_local_kinds.len();
            self.declared_local_kinds.push(Some(element_kind.clone()));
            previous.push((
                target.id.clone(),
                self.slots.insert(target.id.clone(), slot),
            ));
        }
        let mut generator_flow = flow.clone();
        self.normalize_flow(&mut generator_flow);
        for (slot, element_kind) in (binder..binder + targets.len()).zip(&element_kinds) {
            generator_flow.assigned[slot] = true;
            generator_flow.maybe_assigned[slot] = true;
            generator_flow.kinds[slot] = element_kind.clone();
        }

        let validation = (|| {
            if let Some(filter) = comprehension.ifs.first() {
                let (when_true, _) = self.validate_condition(filter, &generator_flow)?;
                generator_flow = when_true;
            }
            match kind {
                GeneratorValidationKind::Condition => self
                    .validate_condition(&generator.elt, &generator_flow)
                    .map(|_| ()),
                GeneratorValidationKind::Dimension => {
                    self.validate_dimension(&generator.elt, &generator_flow)
                }
                GeneratorValidationKind::FlagValue => {
                    self.validate_flag_int(&generator.elt, &generator_flow)
                }
                GeneratorValidationKind::IntTuple => self
                    .validate_int_tuple_expression(&generator.elt, &generator_flow)
                    .map(|_| ()),
            }
        })();
        for (target, old) in previous {
            if let Some(old) = old {
                self.slots.insert(target, old);
            } else {
                self.slots.remove(&target);
            }
        }
        validation?;
        Ok((binder, targets.len()))
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
        if let Expr::Generator(generator) = &call.arguments.args[0] {
            let (binder, binders) =
                self.validate_generator(generator, flow, GeneratorValidationKind::Dimension)?;
            self.expressions.push(TypeShapeDslExpression {
                range: generator.range,
                kind: TypeShapeDslExpressionKind::DimensionGenerator { binder, binders },
            });
            self.expressions.push(TypeShapeDslExpression {
                range: call.range(),
                kind: TypeShapeDslExpressionKind::IntTupleConstructor,
            });
            return Ok(());
        }
        let Expr::Tuple(tuple) = &call.arguments.args[0] else {
            return Err(TypeShapeDslDefinitionError {
                range: call.arguments.args[0].range(),
                message: "`dsl.IntTuple` argument must be a fixed tuple or bounded generator",
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
            range: call.range(),
            kind: TypeShapeDslExpressionKind::IntTupleConstructor,
        });
        Ok(())
    }

    fn validate_int_tuples_constructor(
        &mut self,
        call: &ExprCall,
        flow: &DslValidationFlow,
    ) -> Result<(), TypeShapeDslDefinitionError> {
        if call.arguments.args.len() != 1 || !call.arguments.keywords.is_empty() {
            return Err(TypeShapeDslDefinitionError {
                range: call.arguments.range,
                message: "`dsl.IntTuples` requires exactly one positional argument",
            });
        }
        if let Expr::Generator(generator) = &call.arguments.args[0] {
            let (binder, binders) =
                self.validate_generator(generator, flow, GeneratorValidationKind::IntTuple)?;
            self.expressions.push(TypeShapeDslExpression {
                range: generator.range,
                kind: TypeShapeDslExpressionKind::IntTuplesGenerator { binder, binders },
            });
            self.expressions.push(TypeShapeDslExpression {
                range: call.range(),
                kind: TypeShapeDslExpressionKind::IntTuplesConstructor,
            });
            return Ok(());
        }
        let Expr::Tuple(tuple) = &call.arguments.args[0] else {
            return Err(TypeShapeDslDefinitionError {
                range: call.arguments.args[0].range(),
                message: "`dsl.IntTuples` argument must be a fixed tuple or generator expression",
            });
        };
        for element in &tuple.elts {
            self.validate_int_tuple_expression(element, flow)?;
        }
        self.expressions.push(TypeShapeDslExpression {
            range: call.range(),
            kind: TypeShapeDslExpressionKind::IntTuplesConstructor,
        });
        Ok(())
    }

    /// Validates a `dsl.prod` or `dsl.sum` call and returns its evaluation operation.
    fn validate_int_tuple_reduction(
        &mut self,
        call: &ExprCall,
        flow: &DslValidationFlow,
    ) -> Result<TypeShapeDslExpressionKind, TypeShapeDslDefinitionError> {
        let (message, kind) = match self.intrinsic(&call.func) {
            Some(TypeShapeDslIntrinsic::Prod) => (
                "`dsl.prod` requires exactly one positional IntTuple argument",
                TypeShapeDslExpressionKind::IntTupleProduct,
            ),
            Some(TypeShapeDslIntrinsic::Sum) => (
                "`dsl.sum` requires exactly one positional IntTuple argument",
                TypeShapeDslExpressionKind::IntTupleSum,
            ),
            _ => unreachable!("IntTuple reduction validation requires `dsl.prod` or `dsl.sum`"),
        };
        if call.arguments.args.len() != 1
            || !call.arguments.keywords.is_empty()
            || matches!(call.arguments.args.first(), Some(Expr::Starred(_)))
        {
            return Err(TypeShapeDslDefinitionError {
                range: call.arguments.range,
                message,
            });
        }
        self.validate_int_tuple_expression(&call.arguments.args[0], flow)?;
        Ok(kind)
    }

    fn validate_einsum(
        &mut self,
        call: &ExprCall,
        flow: &DslValidationFlow,
    ) -> Result<(), TypeShapeDslDefinitionError> {
        const SHAPES_ERROR: &str =
            "`dsl.einsum` shapes must be an `IntTuples` parameter or immutable alias";
        if call.arguments.args.len() != 2
            || !call.arguments.keywords.is_empty()
            || call
                .arguments
                .args
                .iter()
                .any(|argument| matches!(argument, Expr::Starred(_)))
        {
            return Err(TypeShapeDslDefinitionError {
                range: call.arguments.range,
                message: "`dsl.einsum` requires exactly two positional arguments",
            });
        }
        self.validate_flag_string(&call.arguments.args[0], flow)?;
        let shapes = &call.arguments.args[1];
        let Expr::Name(_) = shapes else {
            return Err(TypeShapeDslDefinitionError {
                range: shapes.range(),
                message: SHAPES_ERROR,
            });
        };
        let slot = self.slot(shapes, flow)?;
        let parameter_origins = match &flow.kinds[slot] {
            DslStaticKind::UnknownParameters(parameters) => Some(parameters.clone()),
            DslStaticKind::IntTuples { parameter_origins } => parameter_origins.clone(),
            _ => {
                return Err(TypeShapeDslDefinitionError {
                    range: shapes.range(),
                    message: SHAPES_ERROR,
                });
            }
        };
        self.expressions.push(TypeShapeDslExpression {
            range: call.range(),
            kind: TypeShapeDslExpressionKind::Einsum {
                shapes: slot,
                parameter_origins,
            },
        });
        Ok(())
    }

    fn validate_gradual_call(call: &ExprCall) -> Result<(), TypeShapeDslDefinitionError> {
        if call.arguments.args.is_empty() && call.arguments.keywords.is_empty() {
            Ok(())
        } else {
            Err(TypeShapeDslDefinitionError {
                range: call.arguments.range,
                message: "`gradual()` does not accept arguments",
            })
        }
    }

    fn classify_subscript<'b>(
        &self,
        subscript: &'b ExprSubscript,
        flow: &DslValidationFlow,
    ) -> Result<DslSubscriptKind<'b>, TypeShapeDslDefinitionError> {
        let slice = match subscript.slice.as_ref() {
            Expr::Slice(slice) => Some(slice),
            _ => None,
        };
        let Expr::Name(_) = subscript.value.as_ref() else {
            return slice.map_or_else(
                || {
                    Err(TypeShapeDslDefinitionError {
                        range: subscript.value.range(),
                        message: "value must be a bare parameter or local name",
                    })
                },
                |slice| {
                    Ok(DslSubscriptKind::IntTupleSlice {
                        source: DslIntTupleSliceSource::Expression(subscript.value.as_ref()),
                        slice,
                    })
                },
            );
        };
        let source = self.slot(&subscript.value, flow)?;
        match (&flow.kinds[source], slice) {
            (DslStaticKind::IntTuples { .. }, Some(_)) => Err(TypeShapeDslDefinitionError {
                range: subscript.range,
                message: "`IntTuples` does not support slicing",
            }),
            (DslStaticKind::IntTuples { .. }, None) => Ok(DslSubscriptKind::IntTuplesIndex {
                source,
                index: &subscript.slice,
            }),
            (DslStaticKind::UnknownParameters(parameter_origins), None) => {
                Ok(DslSubscriptKind::UnresolvedIndex {
                    source,
                    parameter_origins: parameter_origins.clone(),
                    index: &subscript.slice,
                })
            }
            (_, Some(slice)) => Ok(DslSubscriptKind::IntTupleSlice {
                source: DslIntTupleSliceSource::Slot(source),
                slice,
            }),
            (_, None) => Ok(DslSubscriptKind::IntTupleIndex {
                source,
                index: &subscript.slice,
            }),
        }
    }

    fn validate_int_tuple_slot(
        &mut self,
        expression: &Expr,
        slot: usize,
        flow: &DslValidationFlow,
    ) -> Result<Option<Box<[usize]>>, TypeShapeDslDefinitionError> {
        let kind = &flow.kinds[slot];
        let parameter_origins = if let Some(parameters) = kind.unnarrowed_parameter_origins() {
            Some(parameters.into())
        } else {
            match kind {
                DslStaticKind::IntTuple { parameter_origins } => parameter_origins.clone(),
                DslStaticKind::GeneratorElement {
                    source_parameter_uses: Some(source_parameter_uses),
                } => {
                    self.expressions.push(TypeShapeDslExpression {
                        range: expression.range(),
                        kind: TypeShapeDslExpressionKind::GeneratorElementAsIntTuple {
                            slot,
                            source_parameter_uses: source_parameter_uses.clone(),
                        },
                    });
                    return Ok(None);
                }
                _ => {
                    return Err(TypeShapeDslDefinitionError {
                        range: expression.range(),
                        message: "shape expression names must be `IntTuple` parameters or shape locals",
                    });
                }
            }
        };
        self.expressions.push(TypeShapeDslExpression {
            range: expression.range(),
            kind: TypeShapeDslExpressionKind::IntTupleSlot {
                slot,
                parameter_origins: parameter_origins.clone(),
            },
        });
        Ok(parameter_origins)
    }

    fn validate_int_tuple_expression(
        &mut self,
        expression: &Expr,
        flow: &DslValidationFlow,
    ) -> Result<Option<Box<[usize]>>, TypeShapeDslDefinitionError> {
        let parameter_origins = match expression {
            Expr::Name(_) => {
                let slot = self.slot(expression, flow)?;
                self.validate_int_tuple_slot(expression, slot, flow)?
            }
            Expr::Subscript(subscript) => match self.classify_subscript(subscript, flow)? {
                DslSubscriptKind::IntTuplesIndex { source, index } => {
                    self.validate_flag_int(index, flow)?;
                    self.expressions.push(TypeShapeDslExpression {
                        range: expression.range(),
                        kind: TypeShapeDslExpressionKind::IntTuplesIndex { shapes: source },
                    });
                    return Ok(None);
                }
                DslSubscriptKind::IntTupleSlice { source, slice } => {
                    let parameter_origins = match source {
                        DslIntTupleSliceSource::Slot(source) => {
                            self.validate_int_tuple_slot(&subscript.value, source, flow)?
                        }
                        DslIntTupleSliceSource::Expression(source) => {
                            self.validate_int_tuple_expression(source, flow)?
                        }
                    };
                    if slice.step.is_some() {
                        return Err(TypeShapeDslDefinitionError {
                            range: slice.range,
                            message: "IntTuple slices do not support steps",
                        });
                    }
                    if let Some(lower) = slice.lower.as_deref() {
                        self.validate_flag_int(lower, flow)?;
                    }
                    if let Some(upper) = slice.upper.as_deref() {
                        self.validate_flag_int(upper, flow)?;
                    }
                    self.expressions.push(TypeShapeDslExpression {
                        range: expression.range(),
                        kind: TypeShapeDslExpressionKind::IntTupleSlice,
                    });
                    return Ok(parameter_origins);
                }
                DslSubscriptKind::UnresolvedIndex { index, .. }
                | DslSubscriptKind::IntTupleIndex { index, .. } => {
                    return Err(TypeShapeDslDefinitionError {
                        range: index.range(),
                        message: "IntTuple shape expression subscripts must use slice syntax",
                    });
                }
            },
            Expr::Call(call)
                if self.intrinsic(&call.func) == Some(TypeShapeDslIntrinsic::IntTuple) =>
            {
                self.validate_int_tuple_constructor(call, flow)?;
                None
            }
            Expr::Call(call)
                if self.intrinsic(&call.func) == Some(TypeShapeDslIntrinsic::Concat) =>
            {
                if call.arguments.args.len() != 2 || !call.arguments.keywords.is_empty() {
                    return Err(TypeShapeDslDefinitionError {
                        range: call.arguments.range,
                        message: "`dsl.concat` requires exactly two positional arguments",
                    });
                }
                let left = self.validate_int_tuple_expression(&call.arguments.args[0], flow)?;
                let right = self.validate_int_tuple_expression(&call.arguments.args[1], flow)?;
                self.expressions.push(TypeShapeDslExpression {
                    range: call.range(),
                    kind: TypeShapeDslExpressionKind::IntTupleConcat,
                });
                merge_parameter_origins(left, right)
            }
            Expr::Call(call)
                if self.intrinsic(&call.func) == Some(TypeShapeDslIntrinsic::Einsum) =>
            {
                self.validate_einsum(call, flow)?;
                None
            }
            _ => {
                return Err(TypeShapeDslDefinitionError {
                    range: expression.range(),
                    message: "IntTuple shape expressions support parameters, immutable aliases, restricted slices, `dsl.IntTuple`, `dsl.concat`, and `dsl.einsum`",
                });
            }
        };
        Ok(parameter_origins)
    }

    fn validate_assignment_value(
        &mut self,
        expression: &Expr,
        flow: &DslValidationFlow,
    ) -> Result<DslStaticKind, TypeShapeDslDefinitionError> {
        if let Expr::If(if_expr) = expression {
            let (when_true, when_false) = self.validate_condition(&if_expr.test, flow)?;
            let when_true = self.validate_assignment_value(&if_expr.body, &when_true)?;
            let when_false = self.validate_assignment_value(&if_expr.orelse, &when_false)?;
            let Some(kind) = when_true.join(when_false) else {
                return Err(TypeShapeDslDefinitionError {
                    range: if_expr.range,
                    message: "conditional expression branches must have the same value domain",
                });
            };
            self.expressions.push(TypeShapeDslExpression {
                range: if_expr.range,
                kind: TypeShapeDslExpressionKind::Conditional,
            });
            return Ok(kind);
        }
        match expression {
            Expr::Name(_) => {
                let slot = self.slot(expression, flow)?;
                self.expressions.push(TypeShapeDslExpression {
                    range: expression.range(),
                    kind: TypeShapeDslExpressionKind::Slot(slot),
                });
                Ok(flow.kinds[slot].clone())
            }
            Expr::NoneLiteral(_) => {
                self.expressions.push(TypeShapeDslExpression {
                    range: expression.range(),
                    kind: TypeShapeDslExpressionKind::FlagNone,
                });
                Ok(DslStaticKind::ValueSet {
                    sources: DslValueSources::non_parameter(FLAG_NONE),
                    kinds: FLAG_NONE,
                })
            }
            Expr::BooleanLiteral(literal) => {
                self.expressions.push(TypeShapeDslExpression {
                    range: expression.range(),
                    kind: TypeShapeDslExpressionKind::FlagBool(literal.value),
                });
                Ok(DslStaticKind::ValueSet {
                    sources: DslValueSources::non_parameter(FLAG_BOOL),
                    kinds: FLAG_BOOL,
                })
            }
            Expr::StringLiteral(literal) if Lit::from_string_literal(literal).is_some() => {
                self.validate_flag_string(expression, flow)?;
                Ok(DslStaticKind::ValueSet {
                    sources: DslValueSources::non_parameter(FLAG_STRING),
                    kinds: FLAG_STRING,
                })
            }
            Expr::NumberLiteral(_) | Expr::UnaryOp(_)
                if !matches!(integer_literal(expression), IntegerLiteral::NotLiteral) =>
            {
                self.validate_flag_int(expression, flow)?;
                Ok(DslStaticKind::ValueSet {
                    sources: DslValueSources::non_parameter(FLAG_INT),
                    kinds: FLAG_INT,
                })
            }
            Expr::BinOp(_) => {
                if self.is_dimension_expression(expression, flow) {
                    self.validate_dimension(expression, flow)?;
                    Ok(DslStaticKind::Dimension)
                } else if let Some(classification) =
                    self.classify_deferred_integer_expression(expression, flow)
                {
                    if self.is_traceable_integer_expression(expression, flow) {
                        Ok(self.defer_integer(expression, flow, classification))
                    } else {
                        self.validate_dimension(expression, flow)?;
                        Ok(DslStaticKind::Dimension)
                    }
                } else {
                    self.validate_flag_int(expression, flow)?;
                    Ok(DslStaticKind::ValueSet {
                        sources: DslValueSources::non_parameter(FLAG_INT),
                        kinds: FLAG_INT,
                    })
                }
            }
            Expr::Subscript(subscript) => match self.classify_subscript(subscript, flow)? {
                DslSubscriptKind::UnresolvedIndex {
                    parameter_origins,
                    index,
                    ..
                } => {
                    self.validate_flag_int(index, flow)?;
                    Ok(DslStaticKind::UnknownParameters(parameter_origins))
                }
                DslSubscriptKind::IntTupleSlice { .. }
                | DslSubscriptKind::IntTuplesIndex { .. } => {
                    let parameter_origins = self.validate_int_tuple_expression(expression, flow)?;
                    Ok(DslStaticKind::IntTuple { parameter_origins })
                }
                DslSubscriptKind::IntTupleIndex { .. } => {
                    self.validate_dimension(expression, flow)?;
                    Ok(DslStaticKind::Dimension)
                }
            },
            Expr::Call(call)
                if matches!(
                    self.intrinsic(&call.func),
                    Some(
                        TypeShapeDslIntrinsic::IntTuple
                            | TypeShapeDslIntrinsic::Concat
                            | TypeShapeDslIntrinsic::Einsum
                    )
                ) =>
            {
                let parameter_origins = self.validate_int_tuple_expression(expression, flow)?;
                Ok(DslStaticKind::IntTuple { parameter_origins })
            }
            Expr::Call(call)
                if self.intrinsic(&call.func) == Some(TypeShapeDslIntrinsic::IntTuples) =>
            {
                self.validate_int_tuples_constructor(call, flow)?;
                Ok(DslStaticKind::IntTuples {
                    parameter_origins: None,
                })
            }
            Expr::Tuple(_) => {
                self.validate_flag_sequence(expression, flow)?;
                Ok(DslStaticKind::ValueSet {
                    sources: DslValueSources::non_parameter(FLAG_SEQUENCE),
                    kinds: FLAG_SEQUENCE,
                })
            }
            Expr::Call(call)
                if matches!(
                    self.intrinsic(&call.func),
                    Some(TypeShapeDslIntrinsic::Prod | TypeShapeDslIntrinsic::Sum)
                ) =>
            {
                let kind = self.validate_int_tuple_reduction(call, flow)?;
                self.expressions.push(TypeShapeDslExpression {
                    range: call.range(),
                    kind,
                });
                Ok(DslStaticKind::Dimension)
            }
            Expr::Call(call)
                if self.intrinsic(&call.func)
                    == Some(TypeShapeDslIntrinsic::Gradual(TypeShapeDslDomain::Int)) =>
            {
                Self::validate_gradual_call(call)?;
                self.expressions.push(TypeShapeDslExpression {
                    range: call.range(),
                    kind: TypeShapeDslExpressionKind::Gradual,
                });
                Ok(DslStaticKind::Dimension)
            }
            Expr::Call(call)
                if matches!(
                    self.intrinsic(&call.func),
                    Some(TypeShapeDslIntrinsic::Range | TypeShapeDslIntrinsic::Tuple)
                ) =>
            {
                self.validate_flag_sequence(expression, flow)?;
                Ok(DslStaticKind::ValueSet {
                    sources: DslValueSources::non_parameter(FLAG_SEQUENCE),
                    kinds: FLAG_SEQUENCE,
                })
            }
            Expr::Call(call) if self.intrinsic(&call.func) == Some(TypeShapeDslIntrinsic::Len) => {
                if let Some(classification) =
                    self.classify_deferred_integer_expression(expression, flow)
                {
                    Ok(self.defer_integer(expression, flow, classification))
                } else {
                    self.validate_flag_int(expression, flow)?;
                    Ok(DslStaticKind::ValueSet {
                        sources: DslValueSources::non_parameter(FLAG_INT),
                        kinds: FLAG_INT,
                    })
                }
            }
            Expr::Call(call)
                if matches!(
                    &*call.func,
                    Expr::Attribute(attribute)
                        if matches!(attribute.attr.id.as_str(), "count" | "index")
                ) =>
            {
                self.validate_flag_int(expression, flow)?;
                Ok(DslStaticKind::ValueSet {
                    sources: DslValueSources::non_parameter(FLAG_INT),
                    kinds: FLAG_INT,
                })
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

    fn narrow_value_set(flow: &mut DslValidationFlow, slot: usize, unknown_kinds: u8, mask: u8) {
        let current = flow.kinds[slot].clone();
        let kind = match current {
            DslStaticKind::UnknownParameters(parameters)
            | DslStaticKind::IntTuple {
                parameter_origins: Some(parameters),
            }
            | DslStaticKind::IntTuples {
                parameter_origins: Some(parameters),
            } => DslStaticKind::ValueSet {
                sources: DslValueSources::parameters(
                    &parameters,
                    TypeShapeDslParameterNarrowing::Unnarrowed,
                )
                .narrow(mask),
                kinds: unknown_kinds & mask,
            },
            DslStaticKind::ValueSet { sources, kinds } => DslStaticKind::ValueSet {
                sources: sources.narrow(mask),
                kinds: kinds & mask,
            },
            DslStaticKind::DeferredInteger(_) => DslStaticKind::ValueSet {
                sources: DslValueSources::non_parameter(FLAG_INT & mask),
                kinds: FLAG_INT & mask,
            },
            DslStaticKind::Dimension
            | DslStaticKind::IntTuple {
                parameter_origins: None,
            }
            | DslStaticKind::IntTuples {
                parameter_origins: None,
            } => {
                unreachable!(
                    "non-parameter dimension and shape locals cannot be narrowed as Flag values"
                )
            }
            DslStaticKind::GeneratorElement { .. } => {
                unreachable!("generator elements are not narrowed as Flag union values")
            }
        };
        if matches!(
            &kind,
            DslStaticKind::ValueSet { sources, kinds: 0 }
                if sources.parameter_uses.is_empty() && !sources.has_non_parameter_values()
        ) {
            flow.reachable = false;
        }
        flow.kinds[slot] = kind;
    }

    fn validate_value_set_narrowing_operand(
        &mut self,
        expression: &Expr,
        flow: &DslValidationFlow,
        allowed: u8,
        message: &'static str,
    ) -> Result<(usize, Option<Box<[usize]>>), TypeShapeDslDefinitionError> {
        let slot = self.slot(expression, flow)?;
        if let DslStaticKind::DeferredInteger(index) = flow.kinds[slot] {
            self.resolve_deferred_integer(index, DslIntegerDomain::Flag)?;
            return Ok((slot, None));
        }
        let kind = &flow.kinds[slot];
        let parameter_origins = if let Some(parameters) = kind.unnarrowed_parameter_origins() {
            Some(parameters.into())
        } else {
            match kind {
                DslStaticKind::ValueSet { sources, kinds }
                    if *kinds != 0 && kinds & !allowed == 0 =>
                {
                    sources.parameter_indices()
                }
                DslStaticKind::ValueSet { sources, .. }
                    if sources.all_narrowed() && sources.non_parameter_values_are(allowed) =>
                {
                    sources.parameter_indices()
                }
                _ => {
                    return Err(TypeShapeDslDefinitionError {
                        range: expression.range(),
                        message,
                    });
                }
            }
        };
        Ok((slot, parameter_origins))
    }

    /// Selects dimension validation when syntax and local flow identify a shape-domain value.
    ///
    /// This includes malformed shape subscripts so `validate_dimension` can emit its specific
    /// domain diagnostic. Bare parameters and literals do not establish a domain by themselves,
    /// though the other operand can select this path for them.
    fn is_dimension_expression(&self, expression: &Expr, flow: &DslValidationFlow) -> bool {
        let is_dimension = match expression {
            Expr::Name(name) => self.slots.get(&name.id).is_some_and(|slot| {
                matches!(flow.kinds.get(*slot), Some(DslStaticKind::Dimension))
            }),
            Expr::Subscript(subscript) => matches!(
                self.classify_subscript(subscript, flow),
                Ok(DslSubscriptKind::UnresolvedIndex { .. }
                    | DslSubscriptKind::IntTupleIndex { .. }
                    | DslSubscriptKind::IntTuplesIndex { .. })
            ),
            Expr::BinOp(binop) => {
                self.is_dimension_expression(&binop.left, flow)
                    || self.is_dimension_expression(&binop.right, flow)
            }
            Expr::Call(call) => matches!(
                self.intrinsic(&call.func),
                Some(
                    TypeShapeDslIntrinsic::Prod
                        | TypeShapeDslIntrinsic::Sum
                        | TypeShapeDslIntrinsic::Gradual(TypeShapeDslDomain::Int)
                )
            ),
            Expr::If(if_expr) => {
                self.is_dimension_expression(&if_expr.body, flow)
                    || self.is_dimension_expression(&if_expr.orelse, flow)
            }
            _ => false,
        };
        if is_dimension && self.parameter_domains.is_some() {
            debug_assert!(self.integer_expression_has_shape_source(expression, flow));
        }
        is_dimension
    }

    fn classify_deferred_integer_expression(
        &self,
        expression: &Expr,
        flow: &DslValidationFlow,
    ) -> Option<DeferredIntegerClassification> {
        match expression {
            Expr::Name(name) => {
                self.slots
                    .get(&name.id)
                    .and_then(|slot| match &flow.kinds[*slot] {
                        DslStaticKind::UnknownParameters(_)
                        | DslStaticKind::IntTuple {
                            parameter_origins: Some(_),
                        }
                        | DslStaticKind::IntTuples {
                            parameter_origins: Some(_),
                        } => Some(DeferredIntegerClassification {
                            default_domain: DslIntegerDomain::Dimension,
                            dependencies: Vec::new(),
                        }),
                        DslStaticKind::DeferredInteger(index) => {
                            let (root, _) = self.deferred_integer_domain(*index);
                            let default_domain = match &self.deferred_integers[root].state {
                                DeferredIntegerState::UnresolvedRoot { default_domain, .. } => {
                                    *default_domain
                                }
                                DeferredIntegerState::ResolvedRoot(domain) => *domain,
                                DeferredIntegerState::Redirect(_) => {
                                    unreachable!("a deferred integer root cannot be a redirect")
                                }
                            };
                            Some(DeferredIntegerClassification {
                                default_domain,
                                dependencies: vec![root],
                            })
                        }
                        DslStaticKind::ValueSet {
                            sources,
                            kinds: FLAG_INT,
                        } if sources.all_prove(TypeShapeDslParameterNarrowing::Integer)
                            && sources.non_parameter_values_are(FLAG_INT) =>
                        {
                            Some(DeferredIntegerClassification {
                                default_domain: DslIntegerDomain::Flag,
                                dependencies: Vec::new(),
                            })
                        }
                        DslStaticKind::ValueSet { sources, kinds }
                            if kinds & FLAG_NONE == 0
                                && sources.all_prove(TypeShapeDslParameterNarrowing::NonNone)
                                && sources.non_parameter_values_are(FLAG_INT) =>
                        {
                            Some(DeferredIntegerClassification {
                                default_domain: DslIntegerDomain::Dimension,
                                dependencies: Vec::new(),
                            })
                        }
                        _ => None,
                    })
            }
            Expr::BinOp(binop) => {
                let left = self.classify_deferred_integer_expression(&binop.left, flow);
                let right = self.classify_deferred_integer_expression(&binop.right, flow);
                match (left, right) {
                    (Some(mut left), Some(right)) => {
                        left.default_domain = left.default_domain.join(right.default_domain);
                        left.dependencies.extend(right.dependencies);
                        left.dependencies.sort_unstable();
                        left.dependencies.dedup();
                        Some(left)
                    }
                    (Some(classification), None) | (None, Some(classification)) => {
                        Some(classification)
                    }
                    (None, None) => None,
                }
            }
            // Only an IntTuple-backed length is domain-polymorphic; a Flag sequence has no
            // dimension reading, so it stays an ordinary Flag integer.
            Expr::Call(call) if self.intrinsic(&call.func) == Some(TypeShapeDslIntrinsic::Len) => {
                self.length_has_int_tuple_operand(call, flow).then_some(
                    DeferredIntegerClassification {
                        default_domain: DslIntegerDomain::Dimension,
                        dependencies: Vec::new(),
                    },
                )
            }
            _ => None,
        }
    }

    /// A deferred integer stays domain-polymorphic only while `collect_integer_parameters` can
    /// trace every operand back to a caller parameter, a Flag integer, or a literal, because
    /// that trace is what justifies passing it to a `Flag[int]` helper parameter. Any other
    /// operand may evaluate to a symbolic dimension, so the local must resolve as a dimension.
    /// Unbound names stay traceable so that deferred validation reports them.
    fn is_traceable_integer_expression(&self, expression: &Expr, flow: &DslValidationFlow) -> bool {
        match expression {
            Expr::Name(name) => self.slots.get(&name.id).is_none_or(|slot| {
                matches!(
                    flow.kinds.get(*slot),
                    Some(
                        DslStaticKind::UnknownParameters(_)
                            | DslStaticKind::IntTuple {
                                parameter_origins: Some(_),
                            }
                            | DslStaticKind::IntTuples {
                                parameter_origins: Some(_),
                            }
                            | DslStaticKind::DeferredInteger(_)
                            | DslStaticKind::ValueSet {
                                kinds: FLAG_INT,
                                ..
                            }
                    )
                )
            }),
            Expr::BinOp(binop) => {
                self.is_traceable_integer_expression(&binop.left, flow)
                    && self.is_traceable_integer_expression(&binop.right, flow)
            }
            // An IntTuple-backed length is a rank: concrete whenever the shape is, so it is as
            // traceable as a literal even though its operand is not an integer.
            Expr::Call(call) if self.intrinsic(&call.func) == Some(TypeShapeDslIntrinsic::Len) => {
                self.length_has_int_tuple_operand(call, flow)
            }
            _ => !matches!(integer_literal(expression), IntegerLiteral::NotLiteral),
        }
    }

    fn validate_condition(
        &mut self,
        condition: &Expr,
        flow: &DslValidationFlow,
    ) -> Result<(DslValidationFlow, DslValidationFlow), TypeShapeDslDefinitionError> {
        if let Expr::Call(call) = condition
            && self.intrinsic(&call.func) == Some(TypeShapeDslIntrinsic::Any)
        {
            if call.arguments.args.len() != 1 || !call.arguments.keywords.is_empty() {
                return Err(TypeShapeDslDefinitionError {
                    range: call.arguments.range,
                    message: "`any` requires exactly one positional boolean generator",
                });
            }
            let Expr::Generator(generator) = &call.arguments.args[0] else {
                return Err(TypeShapeDslDefinitionError {
                    range: call.arguments.args[0].range(),
                    message: "`any` argument must be a bounded boolean generator",
                });
            };
            let (binder, _) =
                self.validate_generator(generator, flow, GeneratorValidationKind::Condition)?;
            self.conditions.push(TypeShapeDslCondition {
                range: call.range(),
                kind: TypeShapeDslConditionKind::Any { binder },
            });
            return Ok((flow.clone(), flow.clone()));
        }
        if let Expr::BoolOp(bool_op) = condition {
            let mut sequential = flow.clone();
            match bool_op.op {
                BoolOp::And => {
                    for value in &bool_op.values {
                        if !sequential.reachable {
                            break;
                        }
                        let (when_true, _) = self.validate_condition(value, &sequential)?;
                        sequential = when_true;
                    }
                    return Ok((sequential, flow.clone()));
                }
                BoolOp::Or => {
                    for value in &bool_op.values {
                        if !sequential.reachable {
                            break;
                        }
                        let (_, when_false) = self.validate_condition(value, &sequential)?;
                        sequential = when_false;
                    }
                    return Ok((flow.clone(), sequential));
                }
            }
        }
        if let Expr::UnaryOp(unary) = condition
            && unary.op == UnaryOp::Not
        {
            let (when_true, when_false) = self.validate_condition(&unary.operand, flow)?;
            return Ok((when_false, when_true));
        }

        if matches!(condition, Expr::Name(_)) {
            let slot = self.slot(condition, flow)?;
            let kind = &flow.kinds[slot];
            let parameter_origins = if let Some(parameters) = kind.unnarrowed_parameter_origins() {
                Some(parameters.into())
            } else {
                match kind {
                    DslStaticKind::ValueSet {
                        sources,
                        kinds: FLAG_BOOL,
                        ..
                    } => sources.parameter_indices(),
                    DslStaticKind::ValueSet {
                        sources,
                        kinds: FLAG_NOT_NONE,
                    } if sources.all_prove(TypeShapeDslParameterNarrowing::NonNone)
                        && sources.non_parameter_values_are(FLAG_BOOL) =>
                    {
                        sources.parameter_indices()
                    }
                    _ => {
                        return Err(TypeShapeDslDefinitionError {
                            range: condition.range(),
                            message: "a name used directly as a condition requires a `Flag[bool]` value",
                        });
                    }
                }
            };
            self.conditions.push(TypeShapeDslCondition {
                range: condition.range(),
                kind: TypeShapeDslConditionKind::BoolSlot {
                    slot,
                    parameter_origins,
                },
            });
            return Ok((flow.clone(), flow.clone()));
        }

        if let Expr::Compare(compare) = condition
            && compare.ops.len() == 1
            && matches!(compare.ops[0], CmpOp::Is | CmpOp::IsNot)
            && compare.comparators.len() == 1
            && matches!(&compare.comparators[0], Expr::NoneLiteral(_))
        {
            let negated = compare.ops[0] == CmpOp::IsNot;
            let (slot, origins) = self.validate_value_set_narrowing_operand(
                &compare.left,
                flow,
                FLAG_REPRESENTABLE,
                if negated {
                    "`is not None` requires an `Int | None` or Flag value"
                } else {
                    "`is None` requires an `Int | None` or Flag value"
                },
            )?;
            let mut when_none = flow.clone();
            let mut when_not_none = flow.clone();
            Self::narrow_value_set(&mut when_none, slot, FLAG_REPRESENTABLE, FLAG_NONE);
            Self::narrow_value_set(&mut when_not_none, slot, FLAG_REPRESENTABLE, FLAG_NOT_NONE);
            self.conditions.push(TypeShapeDslCondition {
                range: compare.range,
                kind: TypeShapeDslConditionKind::IsNone {
                    slot,
                    parameter_origins: origins,
                    negated,
                },
            });
            return Ok(if negated {
                (when_not_none, when_none)
            } else {
                (when_none, when_not_none)
            });
        }

        if let Expr::Call(call) = condition
            && matches!(
                self.intrinsic(&call.func),
                Some(TypeShapeDslIntrinsic::IsConcreteInt | TypeShapeDslIntrinsic::IsIntValue)
            )
        {
            let intrinsic = self
                .intrinsic(&call.func)
                .expect("matched integer predicate intrinsic");
            if call.arguments.args.len() != 1
                || !call.arguments.keywords.is_empty()
                || matches!(call.arguments.args.first(), Some(Expr::Starred(_)))
            {
                return Err(TypeShapeDslDefinitionError {
                    range: call.arguments.range,
                    message: match intrinsic {
                        TypeShapeDslIntrinsic::IsConcreteInt => {
                            "`is_concrete_int` condition requires exactly one positional argument"
                        }
                        TypeShapeDslIntrinsic::IsIntValue => {
                            "`is_int_value` condition requires exactly one positional argument"
                        }
                        _ => unreachable!("validated integer predicate intrinsic"),
                    },
                });
            }
            let mut when_true = flow.clone();
            let mut when_false = flow.clone();
            let kind = if intrinsic == TypeShapeDslIntrinsic::IsIntValue {
                let (slot, parameter_origins) = self.validate_value_set_narrowing_operand(
                    &call.arguments.args[0],
                    flow,
                    FLAG_NARROWABLE,
                    "`is_int_value` requires a `Flag[int | tuple[int, ...] | None]` value",
                )?;
                Self::narrow_value_set(&mut when_true, slot, FLAG_NARROWABLE, FLAG_INT);
                Self::narrow_value_set(
                    &mut when_false,
                    slot,
                    FLAG_NARROWABLE,
                    FLAG_SEQUENCE | FLAG_NONE,
                );
                TypeShapeDslConditionKind::IsIntValue {
                    slot,
                    parameter_origins,
                }
            } else {
                let slot = self.slot(&call.arguments.args[0], flow)?;
                if let DslStaticKind::DeferredInteger(index) = flow.kinds[slot] {
                    self.resolve_deferred_integer(index, DslIntegerDomain::Dimension)?;
                }
                let parameter_origins = flow.kinds[slot].parameter_origins();
                let valid = match &flow.kinds[slot] {
                    DslStaticKind::UnknownParameters(_)
                    | DslStaticKind::DeferredInteger(_)
                    | DslStaticKind::Dimension
                    | DslStaticKind::GeneratorElement { .. } => true,
                    DslStaticKind::ValueSet { sources, .. } => {
                        sources.parameter_uses().is_some() && !sources.has_non_parameter_values()
                    }
                    DslStaticKind::IntTuple { .. } | DslStaticKind::IntTuples { .. } => false,
                };
                if !valid {
                    return Err(TypeShapeDslDefinitionError {
                        range: call.arguments.args[0].range(),
                        message: "`is_concrete_int` requires an `Int` or `Int | None` value",
                    });
                }
                // A true result proves both that an optional dimension is present and that it is
                // concrete. The false branch learns nothing because symbolic and gradual `Int`
                // values take it alongside `None`.
                if matches!(
                    flow.kinds[slot],
                    DslStaticKind::UnknownParameters(_) | DslStaticKind::ValueSet { .. }
                ) {
                    Self::narrow_value_set(&mut when_true, slot, FLAG_REPRESENTABLE, FLAG_INT);
                }
                TypeShapeDslConditionKind::IsConcreteInt {
                    slot,
                    parameter_origins,
                }
            };
            self.conditions.push(TypeShapeDslCondition {
                range: call.range(),
                kind,
            });
            return Ok((when_true, when_false));
        }

        let Expr::Compare(compare) = condition else {
            return Err(TypeShapeDslDefinitionError {
                range: condition.range(),
                message: "condition may use only boolean Flag values, `and`, `or`, `not`, `any(...)`, `is None`, `is_concrete_int(...)`, `is_int_value(...)`, integer or string comparisons, and Flag sequence membership",
            });
        };
        if compare.ops.len() != 1 || compare.comparators.len() != 1 {
            return Err(TypeShapeDslDefinitionError {
                range: compare.range,
                message: "comparison must be exactly one binary comparison",
            });
        }
        let op = compare.ops[0];
        let right = &compare.comparators[0];
        if matches!(op, CmpOp::In | CmpOp::NotIn) {
            self.validate_flag_int(&compare.left, flow)?;
            self.validate_flag_sequence(right, flow)?;
            self.conditions.push(TypeShapeDslCondition {
                range: compare.range,
                kind: TypeShapeDslConditionKind::Membership {
                    negated: op == CmpOp::NotIn,
                },
            });
            return Ok((flow.clone(), flow.clone()));
        }
        let Some(comparison_op) = TypeShapeDslComparisonOp::from_cmp_op(op) else {
            return Err(TypeShapeDslDefinitionError {
                range: compare.range,
                message: "comparison operator is not supported",
            });
        };

        let string_operand = |expression: &Expr| {
            matches!(expression, Expr::StringLiteral(literal) if Lit::from_string_literal(literal).is_some())
                || matches!(expression, Expr::Name(name)
                if self.slots.get(&name.id).is_some_and(|slot| matches!(
                    flow.kinds[*slot],
                    DslStaticKind::ValueSet { kinds, .. }
                        if kinds & FLAG_STRING != 0
                            && kinds & !(FLAG_STRING | FLAG_NONE) == 0
                )))
        };
        if string_operand(&compare.left) || string_operand(right) {
            if !matches!(op, CmpOp::Eq | CmpOp::NotEq) {
                return Err(TypeShapeDslDefinitionError {
                    range: compare.range,
                    message: "Flag strings support only `==` and `!=`",
                });
            }
            self.validate_flag_string(&compare.left, flow)?;
            self.validate_flag_string(right, flow)?;
            self.conditions.push(TypeShapeDslCondition {
                range: compare.range,
                kind: TypeShapeDslConditionKind::StringEquality {
                    negated: op == CmpOp::NotEq,
                },
            });
            return Ok((flow.clone(), flow.clone()));
        }

        let comparison_operand = |kind: &DslStaticKind| match kind {
            kind if kind.unnarrowed_parameter_origins().is_some() => {
                Some(TypeShapeDslComparisonOperand {
                    parameter_uses: kind.parameter_uses(),
                    is_flag_operand: false,
                    non_parameter_flag_domain: None,
                })
            }
            DslStaticKind::Dimension => Some(TypeShapeDslComparisonOperand {
                parameter_uses: None,
                is_flag_operand: false,
                non_parameter_flag_domain: None,
            }),
            DslStaticKind::ValueSet {
                sources,
                kinds: FLAG_INT,
            } if sources.all_prove(TypeShapeDslParameterNarrowing::Integer)
                && sources.non_parameter_values_are(FLAG_INT) =>
            {
                Some(TypeShapeDslComparisonOperand {
                    parameter_uses: Some(sources.parameter_uses.clone()),
                    is_flag_operand: true,
                    non_parameter_flag_domain: flag_domain_from_kinds(sources.non_parameter_kinds),
                })
            }
            DslStaticKind::ValueSet {
                sources,
                kinds: FLAG_NOT_NONE,
            } if sources.all_prove(TypeShapeDslParameterNarrowing::NonNone) => {
                Some(TypeShapeDslComparisonOperand {
                    parameter_uses: Some(sources.parameter_uses.clone()),
                    is_flag_operand: true,
                    non_parameter_flag_domain: flag_domain_from_kinds(sources.non_parameter_kinds),
                })
            }
            DslStaticKind::ValueSet {
                sources,
                kinds: FLAG_NONE,
            } if sources.all_prove(TypeShapeDslParameterNarrowing::IsNone) => {
                Some(TypeShapeDslComparisonOperand {
                    parameter_uses: Some(sources.parameter_uses.clone()),
                    is_flag_operand: true,
                    non_parameter_flag_domain: flag_domain_from_kinds(sources.non_parameter_kinds),
                })
            }
            _ => None,
        };
        let slot_comparison = match (&*compare.left, right) {
            (Expr::Name(_), Expr::Name(_)) => {
                let left = self.slot(&compare.left, flow)?;
                let right = self.slot(right, flow)?;
                if left == right
                    && matches!(flow.kinds[left], DslStaticKind::GeneratorElement { .. })
                {
                    Some(TypeShapeDslConditionKind::GeneratorElementSelfCompare(
                        comparison_op,
                    ))
                } else {
                    comparison_operand(&flow.kinds[left])
                        .zip(comparison_operand(&flow.kinds[right]))
                        .map(|(left_operand, right_operand)| {
                            TypeShapeDslConditionKind::SlotCompare {
                                left,
                                right,
                                left_operand,
                                right_operand,
                                op: comparison_op,
                            }
                        })
                }
            }
            _ => None,
        };
        let has_dimension_expression = self.is_dimension_expression(&compare.left, flow)
            || self.is_dimension_expression(right, flow);
        let right_literal = match integer_literal(right) {
            IntegerLiteral::Value(value) => Some(value),
            IntegerLiteral::NotLiteral | IntegerLiteral::Unrepresentable { .. } => None,
        };
        let left_literal = match integer_literal(&compare.left) {
            IntegerLiteral::Value(value) => Some(value),
            IntegerLiteral::NotLiteral | IntegerLiteral::Unrepresentable { .. } => None,
        };
        let is_integer_comparison_candidate = |expression: &Expr| {
            let Expr::Name(_) = expression else {
                return Ok(false);
            };
            let slot = self.slot(expression, flow)?;
            Ok(match &flow.kinds[slot] {
                DslStaticKind::ValueSet {
                    sources,
                    kinds: FLAG_INT,
                } if sources.all_prove(TypeShapeDslParameterNarrowing::Integer)
                    && sources.non_parameter_values_are(FLAG_INT) =>
                {
                    true
                }
                DslStaticKind::ValueSet {
                    sources,
                    kinds: FLAG_NOT_NONE,
                } if !sources.parameter_uses.is_empty()
                    && sources.all_prove(TypeShapeDslParameterNarrowing::NonNone) =>
                {
                    true
                }
                _ => false,
            })
        };
        let simple_operands = (matches!(&*compare.left, Expr::Name(_)) || left_literal.is_some())
            && (matches!(right, Expr::Name(_)) || right_literal.is_some());
        let integer_comparison = if simple_operands
            && (is_integer_comparison_candidate(&compare.left)?
                || is_integer_comparison_candidate(right)?)
        {
            let integer_literal_operand = || {
                Some(TypeShapeDslComparisonOperand {
                    parameter_uses: Some(Vec::<TypeShapeDslParameterUse>::new().into_boxed_slice()),
                    is_flag_operand: true,
                    non_parameter_flag_domain: flag_domain_from_kinds(FLAG_INT),
                })
            };
            let left_operand = match &*compare.left {
                Expr::Name(_) => {
                    let left = self.slot(&compare.left, flow)?;
                    comparison_operand(&flow.kinds[left])
                }
                _ if left_literal.is_some() => integer_literal_operand(),
                _ => None,
            };
            let right_operand = match right {
                Expr::Name(_) => {
                    let right = self.slot(right, flow)?;
                    comparison_operand(&flow.kinds[right])
                }
                _ if right_literal.is_some() => integer_literal_operand(),
                _ => None,
            };
            left_operand.zip(right_operand)
        } else {
            None
        };
        let kind = match (slot_comparison, integer_comparison) {
            (Some(kind), _) => kind,
            (None, Some((left_operand, right_operand))) => {
                self.validate_dimension_arithmetic_operand(&compare.left, flow)?;
                self.validate_dimension_arithmetic_operand(right, flow)?;
                TypeShapeDslConditionKind::IntegerCompare {
                    left_operand,
                    right_operand,
                    op: comparison_op,
                }
            }
            (None, None) if has_dimension_expression => {
                if !matches!(op, CmpOp::Eq | CmpOp::NotEq) {
                    return Err(TypeShapeDslDefinitionError {
                        range: compare.range,
                        message: "derived dimension comparisons support only `==` and `!=`",
                    });
                }
                self.validate_dimension(&compare.left, flow)?;
                self.validate_dimension(right, flow)?;
                TypeShapeDslConditionKind::DimensionEquality {
                    negated: op == CmpOp::NotEq,
                }
            }
            (None, None)
                if op == CmpOp::Eq
                    && right_literal.is_some()
                    && matches!(
                        &*compare.left,
                        Expr::Call(call)
                            if self.intrinsic(&call.func) == Some(TypeShapeDslIntrinsic::Len)
                                && call.arguments.args.len() == 1
                                && call.arguments.keywords.is_empty()
                    ) =>
            {
                self.validate_flag_int(&compare.left, flow)?;
                self.validate_flag_int(right, flow)?;
                let Expr::Call(call) = &*compare.left else {
                    unreachable!("guarded length equality has a call on the left")
                };
                let slot = self.slot(&call.arguments.args[0], flow)?;
                TypeShapeDslConditionKind::LengthEqualLiteral {
                    slot,
                    literal: right_literal
                        .expect("guarded length equality has a representable integer literal"),
                }
            }
            (None, None) => {
                self.validate_flag_int(&compare.left, flow)?;
                self.validate_flag_int(right, flow)?;
                TypeShapeDslConditionKind::FlagIntCompare(comparison_op)
            }
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
                    let kind = if slot < self.parameters.args.len() {
                        let uses = flow.kinds[slot]
                            .parameter_uses()
                            .expect("a parameter slot retains its source");
                        let [use_] = uses.as_ref() else {
                            unreachable!("a parameter slot has exactly one source")
                        };
                        TypeShapeDslSlotReturnKind::DirectParameter(*use_)
                    } else {
                        // A local that merely aliases a parameter returns the parameter itself, so
                        // the declared parameter domain — not the local's inferred domain — decides
                        // whether the return is legal.
                        if matches!(
                            &flow.kinds[slot],
                            DslStaticKind::ValueSet { sources, .. }
                                if sources.has_non_parameter_values()
                        ) {
                            return Err(TypeShapeDslDefinitionError {
                                range: returned.range(),
                                message: "Flag values are input-only and cannot be returned",
                            });
                        }
                        if let Some(parameter_uses) = flow.kinds[slot].parameter_alias_uses() {
                            TypeShapeDslSlotReturnKind::ParameterAlias(parameter_uses)
                        } else {
                            if matches!(flow.kinds[slot], DslStaticKind::ValueSet { .. }) {
                                return Err(TypeShapeDslDefinitionError {
                                    range: returned.range(),
                                    message: "Flag values are input-only and cannot be returned",
                                });
                            }
                            match &flow.kinds[slot] {
                                DslStaticKind::Dimension => {
                                    TypeShapeDslSlotReturnKind::KnownDomain {
                                        domain: TypeShapeDslDomain::Int,
                                        parameter_uses: None,
                                    }
                                }
                                DslStaticKind::DeferredInteger(index) => {
                                    self.resolve_deferred_integer(
                                        *index,
                                        DslIntegerDomain::Dimension,
                                    )?;
                                    TypeShapeDslSlotReturnKind::KnownDomain {
                                        domain: TypeShapeDslDomain::Int,
                                        parameter_uses: None,
                                    }
                                }
                                DslStaticKind::IntTuple { parameter_origins } => {
                                    TypeShapeDslSlotReturnKind::KnownDomain {
                                        domain: TypeShapeDslDomain::IntTuple,
                                        parameter_uses: parameter_origins.as_ref().map(
                                            |parameters| {
                                                parameter_uses(
                                                    parameters,
                                                    TypeShapeDslParameterNarrowing::Unnarrowed,
                                                )
                                            },
                                        ),
                                    }
                                }
                                DslStaticKind::IntTuples { parameter_origins } => {
                                    TypeShapeDslSlotReturnKind::KnownDomain {
                                        domain: TypeShapeDslDomain::IntTuples,
                                        parameter_uses: parameter_origins.as_ref().map(
                                            |parameters| {
                                                parameter_uses(
                                                    parameters,
                                                    TypeShapeDslParameterNarrowing::Unnarrowed,
                                                )
                                            },
                                        ),
                                    }
                                }
                                DslStaticKind::UnknownParameters(_) => unreachable!(
                                    "parameter-backed locals return with their parameter uses"
                                ),
                                DslStaticKind::ValueSet { .. } => {
                                    unreachable!("literal-backed Flag locals are rejected above")
                                }
                                DslStaticKind::GeneratorElement { .. } => {
                                    unreachable!("generator elements cannot escape their generator")
                                }
                            }
                        }
                    };
                    TypeShapeDslReturnKind::Slot { slot, kind }
                }
            }
            Some(returned @ Expr::Call(call)) => match self.intrinsic(&call.func) {
                Some(TypeShapeDslIntrinsic::Gradual(domain)) => {
                    Self::validate_gradual_call(call)?;
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
                        flow.kinds[left].int_tuple_parameter_origins(),
                        flow.kinds[right].int_tuple_parameter_origins(),
                    ) else {
                        return Err(TypeShapeDslDefinitionError {
                            range: call.arguments.range,
                            message: "`broadcast` arguments must be IntTuple parameters or immutable aliases of them",
                        });
                    };
                    TypeShapeDslReturnKind::Broadcast {
                        left_slot: left,
                        right_slot: right,
                        left_parameters,
                        right_parameters,
                    }
                }
                Some(
                    TypeShapeDslIntrinsic::IntTuple
                    | TypeShapeDslIntrinsic::Concat
                    | TypeShapeDslIntrinsic::Einsum,
                ) => {
                    self.validate_int_tuple_expression(returned, flow)?;
                    TypeShapeDslReturnKind::Expression(TypeShapeDslDomain::IntTuple)
                }
                Some(TypeShapeDslIntrinsic::IntTuples) => {
                    self.validate_int_tuples_constructor(call, flow)?;
                    TypeShapeDslReturnKind::Expression(TypeShapeDslDomain::IntTuples)
                }
                Some(TypeShapeDslIntrinsic::Prod | TypeShapeDslIntrinsic::Sum) => {
                    let kind = self.validate_int_tuple_reduction(call, flow)?;
                    self.expressions.push(TypeShapeDslExpression {
                        range: call.range(),
                        kind,
                    });
                    TypeShapeDslReturnKind::Expression(TypeShapeDslDomain::Int)
                }
                Some(TypeShapeDslIntrinsic::Len) => {
                    let kind = self.validate_length(call, flow, DslIntegerDomain::Dimension)?;
                    self.expressions.push(TypeShapeDslExpression {
                        range: call.range(),
                        kind,
                    });
                    TypeShapeDslReturnKind::Expression(TypeShapeDslDomain::Int)
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
                None => {
                    if !call.arguments.keywords.is_empty()
                        || call
                            .arguments
                            .args
                            .iter()
                            .any(|argument| matches!(argument, Expr::Starred(_)))
                    {
                        return Err(TypeShapeDslDefinitionError {
                            range: call.arguments.range,
                            message: "DSL helper calls accept only positional arguments",
                        });
                    }
                    let helper = self.helper_calls.len();
                    let selected_domains = self
                        .helper_argument_domains
                        .and_then(|domains| domains.get(helper));
                    if selected_domains
                        .is_some_and(|domains| domains.len() != call.arguments.args.len())
                    {
                        return Err(TypeShapeDslDefinitionError {
                            range: call.arguments.range,
                            message: "resolved DSL helper arguments must align with the call",
                        });
                    }
                    let mut arguments = Vec::with_capacity(call.arguments.args.len());
                    for (argument_index, argument) in call.arguments.args.iter().enumerate() {
                        let Ok(slot) = self.slot(argument, flow) else {
                            return Err(TypeShapeDslDefinitionError {
                                range: argument.range(),
                                message: "return value must be a bare parameter name or validated DSL helper call; helper arguments must be bare parameter or local names",
                            });
                        };
                        let source = match flow.kinds[slot].clone() {
                            DslStaticKind::UnknownParameters(parameters) => {
                                TypeShapeDslHelperArgumentSource::Parameters(parameters)
                            }
                            DslStaticKind::Dimension => TypeShapeDslHelperArgumentSource::Exact(
                                TypeShapeDslInputDomain::Value(TypeShapeDslDomain::Int),
                            ),
                            DslStaticKind::DeferredInteger(index) => {
                                if let Some(domains) = selected_domains {
                                    let domain = match domains[argument_index] {
                                        TypeShapeDslInputDomain::Value(TypeShapeDslDomain::Int) => {
                                            DslIntegerDomain::Dimension
                                        }
                                        TypeShapeDslInputDomain::Flag(domain)
                                            if domain == FlagDomain::of(FlagMember::Int) =>
                                        {
                                            DslIntegerDomain::Flag
                                        }
                                        _ => {
                                            return Err(TypeShapeDslDefinitionError {
                                                range: argument.range(),
                                                message: "deferred integer helper arguments require an Int or Flag[int] parameter",
                                            });
                                        }
                                    };
                                    self.resolve_deferred_integer(index, domain)?;
                                }
                                TypeShapeDslHelperArgumentSource::DeferredInteger {
                                    index,
                                    parameter_uses: Box::new([]),
                                    resolved_domain: None,
                                }
                            }
                            DslStaticKind::IntTuple {
                                parameter_origins: Some(parameters),
                            } => TypeShapeDslHelperArgumentSource::ParametersWithRequiredDomain {
                                parameters,
                                domain: TypeShapeDslInputDomain::Value(
                                    TypeShapeDslDomain::IntTuple,
                                ),
                            },
                            DslStaticKind::IntTuple {
                                parameter_origins: None,
                            } => TypeShapeDslHelperArgumentSource::Exact(
                                TypeShapeDslInputDomain::Value(TypeShapeDslDomain::IntTuple),
                            ),
                            DslStaticKind::IntTuples {
                                parameter_origins: Some(parameters),
                            } => TypeShapeDslHelperArgumentSource::ParametersWithRequiredDomain {
                                parameters,
                                domain: TypeShapeDslInputDomain::Value(
                                    TypeShapeDslDomain::IntTuples,
                                ),
                            },
                            DslStaticKind::IntTuples {
                                parameter_origins: None,
                            } => TypeShapeDslHelperArgumentSource::Exact(
                                TypeShapeDslInputDomain::Value(TypeShapeDslDomain::IntTuples),
                            ),
                            DslStaticKind::ValueSet { sources, kinds } => {
                                let Some(domain) = flag_domain_from_kinds(kinds) else {
                                    return Err(TypeShapeDslDefinitionError {
                                        range: argument.range(),
                                        message: "DSL helper arguments must have a nonempty supported Flag domain",
                                    });
                                };
                                if sources.parameter_uses.is_empty() {
                                    if domain == FlagDomain::of(FlagMember::NoneType) {
                                        TypeShapeDslHelperArgumentSource::NoneLiteral
                                    } else {
                                        TypeShapeDslHelperArgumentSource::Exact(
                                            TypeShapeDslInputDomain::Flag(domain),
                                        )
                                    }
                                } else {
                                    TypeShapeDslHelperArgumentSource::ValueSources {
                                        sources,
                                        observed_domain: domain,
                                    }
                                }
                            }
                            DslStaticKind::GeneratorElement { .. } => {
                                return Err(TypeShapeDslDefinitionError {
                                    range: argument.range(),
                                    message: "generator elements cannot escape their generator",
                                });
                            }
                        };
                        arguments.push(TypeShapeDslHelperArgument { slot, source });
                    }
                    self.helper_calls.push(TypeShapeDslHelperCall {
                        callee: (*call.func).clone(),
                        arguments,
                    });
                    TypeShapeDslReturnKind::HelperCall(helper)
                }
                Some(_) => {
                    return Err(TypeShapeDslDefinitionError {
                        range: return_stmt.range,
                        message: "return value must be a bare parameter name, gradual return, `broadcast(...)`, `dsl.Invalid(...)`, an Int/IntTuple/IntTuples expression, or a validated DSL helper call",
                    });
                }
            },
            Some(returned @ Expr::Subscript(subscript)) => {
                let domain = match self.classify_subscript(subscript, flow)? {
                    DslSubscriptKind::IntTupleSlice { .. }
                    | DslSubscriptKind::IntTuplesIndex { .. } => {
                        self.validate_int_tuple_expression(returned, flow)?;
                        TypeShapeDslDomain::IntTuple
                    }
                    DslSubscriptKind::UnresolvedIndex { .. }
                    | DslSubscriptKind::IntTupleIndex { .. } => {
                        self.validate_dimension(returned, flow)?;
                        TypeShapeDslDomain::Int
                    }
                };
                TypeShapeDslReturnKind::Expression(domain)
            }
            Some(Expr::BinOp(_)) => {
                self.validate_dimension(
                    return_stmt.value.as_deref().expect("matched return value"),
                    flow,
                )?;
                TypeShapeDslReturnKind::Expression(TypeShapeDslDomain::Int)
            }
            _ => {
                return Err(TypeShapeDslDefinitionError {
                    range: return_stmt.range,
                    message: "return value must be a bare parameter name, gradual return, `broadcast(...)`, `dsl.Invalid(...)`, an Int/IntTuple/IntTuples expression, or a validated DSL helper call",
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
        &mut self,
        flows: Vec<DslValidationFlow>,
        range: TextRange,
    ) -> Result<Option<DslValidationFlow>, TypeShapeDslDefinitionError> {
        let mut flows = flows.into_iter().filter(|flow| flow.reachable);
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
                    let left = result.kinds[slot].clone();
                    let right = flow.kinds[slot].clone();
                    let kind = match (left, right) {
                        (
                            DslStaticKind::DeferredInteger(left),
                            DslStaticKind::DeferredInteger(right),
                        ) => DslStaticKind::DeferredInteger(
                            self.merge_deferred_integers(left, right)?,
                        ),
                        (DslStaticKind::DeferredInteger(index), DslStaticKind::Dimension)
                        | (DslStaticKind::Dimension, DslStaticKind::DeferredInteger(index)) => {
                            self.resolve_deferred_integer(index, DslIntegerDomain::Dimension)?;
                            DslStaticKind::Dimension
                        }
                        (
                            DslStaticKind::DeferredInteger(index),
                            flag @ DslStaticKind::ValueSet {
                                kinds: FLAG_INT, ..
                            },
                        )
                        | (
                            flag @ DslStaticKind::ValueSet {
                                kinds: FLAG_INT, ..
                            },
                            DslStaticKind::DeferredInteger(index),
                        ) => {
                            self.resolve_deferred_integer(index, DslIntegerDomain::Flag)?;
                            flag
                        }
                        (left, right) => left.join(right).ok_or(TypeShapeDslDefinitionError {
                            range,
                            message: "all continuing branch assignments to a local must have the same value domain",
                        })?,
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
            if !when_false.reachable {
                break;
            }
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
        if !has_else && when_false.reachable {
            continuing.push(when_false);
        }
        self.merge_flows(continuing, if_stmt.range)
    }

    fn validate_suite(
        &mut self,
        suite: &[Stmt],
        mut flow: DslValidationFlow,
    ) -> Result<Option<DslValidationFlow>, TypeShapeDslDefinitionError> {
        if !flow.reachable {
            return Ok(None);
        }
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

// Like the pointer ordering on the parsed program, this is a process-local tie-breaker required by
// type nodes that derive `Ord`; it must not be used for stable output. The metadata tables order
// themselves canonically and `helper_calls` is ordered by its offsets because `TextRange` has no
// total order, so this stays consistent with the derived `Eq` above.
impl PartialOrd for StructurallyValidatedTypeShapeDslFunction {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for StructurallyValidatedTypeShapeDslFunction {
    fn cmp(&self, other: &Self) -> Ordering {
        fn offsets(range: TextRange) -> (TextSize, TextSize) {
            (range.start(), range.end())
        }
        self.parsed
            .cmp(&other.parsed)
            .then_with(|| {
                self.returns
                    .iter()
                    .map(|x| (offsets(x.statement_range), offsets(x.value_range), &x.kind))
                    .cmp(
                        other
                            .returns
                            .iter()
                            .map(|x| (offsets(x.statement_range), offsets(x.value_range), &x.kind)),
                    )
            })
            .then_with(|| {
                self.conditions
                    .iter()
                    .map(|x| (offsets(x.range), &x.kind))
                    .cmp(other.conditions.iter().map(|x| (offsets(x.range), &x.kind)))
            })
            .then_with(|| {
                self.expressions
                    .iter()
                    .map(|x| (offsets(x.range), &x.kind))
                    .cmp(
                        other
                            .expressions
                            .iter()
                            .map(|x| (offsets(x.range), &x.kind)),
                    )
            })
            .then_with(|| {
                self.assignments
                    .iter()
                    .map(|x| (offsets(x.range), x.slot))
                    .cmp(other.assignments.iter().map(|x| (offsets(x.range), x.slot)))
            })
            .then_with(|| {
                self.helper_calls
                    .iter()
                    .map(|call| (offsets(call.callee.range()), &call.arguments))
                    .cmp(
                        other
                            .helper_calls
                            .iter()
                            .map(|call| (offsets(call.callee.range()), &call.arguments)),
                    )
            })
            .then_with(|| self.slot_count.cmp(&other.slot_count))
    }
}

impl Visit<Type> for StructurallyValidatedTypeShapeDslFunction {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

impl VisitMut<Type> for StructurallyValidatedTypeShapeDslFunction {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}

impl Visit<Type> for Arc<StructurallyValidatedTypeShapeDslFunction> {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

impl VisitMut<Type> for Arc<StructurallyValidatedTypeShapeDslFunction> {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}

impl TypeEqTrait for StructurallyValidatedTypeShapeDslFunction {
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

    /// Validate the body: its indexed storage, control flow, and every resolved execution fact
    /// evaluation later replays against the retained AST.
    pub fn validate(
        &self,
        intrinsic: impl Fn(&Expr) -> Option<TypeShapeDslIntrinsic>,
    ) -> Result<StructurallyValidatedTypeShapeDslFunction, TypeShapeDslDefinitionError> {
        self.validate_with_helper_argument_domains(intrinsic, None, None)
    }

    fn validate_with_helper_argument_domains(
        &self,
        intrinsic: impl Fn(&Expr) -> Option<TypeShapeDslIntrinsic>,
        parameter_domains: Option<&[TypeShapeDslInputDomain]>,
        helper_argument_domains: Option<&[Vec<TypeShapeDslInputDomain>]>,
    ) -> Result<StructurallyValidatedTypeShapeDslFunction, TypeShapeDslDefinitionError> {
        let parameters = &self.definition.parameters;
        // `DslValidator::intrinsic` suppresses resolution for any name bound by a slot, which
        // seeds with every parameter, so shadowing needs no separate check here.
        let (mut validator, flow) = DslValidator::new(
            parameters,
            &intrinsic,
            parameter_domains,
            helper_argument_domains,
        );
        if validator
            .validate_suite(&self.definition.body, flow)?
            .is_some()
        {
            return Err(TypeShapeDslDefinitionError {
                range: self.definition.name.range(),
                message: "every control-flow path must return",
            });
        }
        // Helper signatures can determine the domain of otherwise-unconstrained integer locals,
        // so preserve their unresolved state before defaulting integers that have no such use.
        validator.finalize_helper_deferred_domains()?;
        validator.resolve_unused_deferred_integers()?;
        let DslValidator {
            returns,
            conditions,
            expressions,
            assignments,
            helper_calls,
            declared_local_kinds,
            ..
        } = validator;
        let slot_count = declared_local_kinds.len();
        Ok(StructurallyValidatedTypeShapeDslFunction {
            parsed: self.clone(),
            returns: SourceRangeTable::new("return", returns),
            conditions: SourceRangeTable::new("condition", conditions),
            expressions: SourceRangeTable::new("expression", expressions),
            assignments: SourceRangeTable::new("assignment", assignments),
            helper_calls,
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

impl StructurallyValidatedTypeShapeDslFunction {
    /// Revalidates the body using resolved parameter domains and the domains selected by helpers.
    pub fn validate_with_resolved_domains(
        &self,
        intrinsic: impl Fn(&Expr) -> Option<TypeShapeDslIntrinsic>,
        parameter_domains: &[TypeShapeDslInputDomain],
        helper_argument_domains: Option<&[Vec<TypeShapeDslInputDomain>]>,
    ) -> Result<Self, TypeShapeDslDefinitionError> {
        if helper_argument_domains.is_some_and(|domains| domains.len() != self.helper_calls.len()) {
            return Err(TypeShapeDslDefinitionError {
                range: self.parsed.definition.name.range(),
                message: "resolved DSL helpers must align with structurally validated calls",
            });
        }
        self.parsed.validate_with_helper_argument_domains(
            intrinsic,
            Some(parameter_domains),
            helper_argument_domains,
        )
    }

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

    pub fn helper_calls(&self) -> impl Iterator<Item = &TypeShapeDslHelperCall> {
        self.helper_calls.iter()
    }
}

/// A deferred type-level DSL invocation retained until the operation's result is consumed.
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
    /// A deferred `shape_extensions.MapIntTuples[<lambda>, <source>]` application.
    MapIntTuples(MapIntTuples),
}

#[derive(Debug, Clone)]
enum DslValue {
    Unknown,
    /// A runtime value known to be an `int` whose value is not statically known.
    GradualInt,
    Dimension(Int),
    Shape(IntTuple),
    IntTuples(DslIntTuples),
    FlagInt(i64),
    FlagBool(bool),
    FlagString(CompactString),
    FlagNone,
    FlagSequence(DslFlagSequence),
    DimensionTuple(Vec<Int>),
}

#[derive(Debug, Clone)]
/// The evaluator's value for the `IntTuples` domain.
///
/// For example, `dsl.IntTuples((IntTuple[2], IntTuple[3, 4]))` produces
/// `tuple[IntTuple[2], IntTuple[3, 4]]`; a gradual value produces `tuple[IntTuple, ...]`.
enum DslIntTuples {
    Fixed(Vec<IntTuple>),
    Unbounded(IntTuple),
}

#[derive(Debug, Clone)]
enum DslFlagSequence {
    Values(Vec<i64>),
    Range { start: i64, stop: i64, step: i64 },
}

#[derive(Debug, PartialEq, Eq)]
enum DslFlagSequenceIndex {
    Exact(i64),
    Gradual,
    NotFound,
}

#[derive(Clone)]
enum DslOutcome {
    Value(DslValue),
    ExplicitGradual,
    Invalid(ShapeError),
}

#[derive(Clone)]
struct DslEnvironment {
    parameter_count: usize,
    slots: Vec<DslValue>,
}

/// The result of evaluating a DSL condition.
///
/// `UnknownWithPossibleError` means some concrete instantiation can raise before the condition
/// reaches a decisive value. Unlike ordinary `Unknown`, a later short-circuiting operand cannot
/// erase it. Condition consumers eventually project either unknown state to a gradual value.
#[derive(Clone, Copy)]
enum DslCondition {
    True,
    False,
    Unknown,
    UnknownWithPossibleError,
}

enum EvaluatedGeneratorItems {
    Known {
        values: Vec<DslValue>,
        truncated: bool,
        common_member: DslValue,
    },
    Unknown {
        common_member: DslValue,
    },
}

impl EvaluatedGeneratorItems {
    fn common_member(&self) -> &DslValue {
        match self {
            Self::Known { common_member, .. } | Self::Unknown { common_member } => common_member,
        }
    }
}

enum DslControlFlow {
    Continue,
    Return(DslOutcome),
}

impl Visit<Type> for TypeLevelDslFunction {
    fn recurse<'a>(&'a self, f: &mut dyn FnMut(&'a Type)) {
        match self {
            Self::Broadcast | Self::UserDefined(_) => {}
            Self::MapIntTuples(map) => map.visit(f),
        }
    }
}

impl VisitMut<Type> for TypeLevelDslFunction {
    fn recurse_mut(&mut self, f: &mut dyn FnMut(&mut Type)) {
        match self {
            Self::Broadcast | Self::UserDefined(_) => {}
            Self::MapIntTuples(map) => map.visit_mut(f),
        }
    }
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
            TypeLevelDslFunction::MapIntTuples(_) => "MapIntTuples",
        }
    }

    /// Returns the shape-DSL result domain, or `None` for a map of arbitrary result types.
    pub fn result_domain(&self) -> Option<TypeShapeDslDomain> {
        match &self.function {
            TypeLevelDslFunction::Broadcast => Some(TypeShapeDslDomain::IntTuple),
            TypeLevelDslFunction::UserDefined(function) => Some(function.result_domain()),
            // TODO(stroxler): Track the mapper's result domain so a deferred map whose mapper
            // returns `IntTuple` can itself be used as an `IntTuples` source.
            TypeLevelDslFunction::MapIntTuples(_) => None,
        }
    }

    /// Returns the gradual result for a call whose precise value cannot be determined.
    pub fn fallback(&self) -> Type {
        match &self.function {
            TypeLevelDslFunction::Broadcast => IntTuple::shapeless().to_shape_arg_type(),
            TypeLevelDslFunction::UserDefined(function) => match function.result_domain() {
                TypeShapeDslDomain::Int => gradual_size(),
                TypeShapeDslDomain::IntTuple => IntTuple::shapeless().to_shape_arg_type(),
                TypeShapeDslDomain::IntTuples => Type::Tuple(Tuple::Unbounded(Box::new(
                    IntTuple::shapeless().to_shape_arg_type(),
                ))),
            },
            TypeLevelDslFunction::MapIntTuples(map) => map.fallback(),
        }
    }

    /// Recurses through the call while respecting the binder introduced by a map's lambda.
    pub fn subst_parts_mut(
        &mut self,
        shadowed: &mut Vec<Quantified>,
        f: &mut dyn FnMut(&mut Type, &mut Vec<Quantified>),
    ) {
        for arg in &mut self.args {
            f(arg, shadowed);
        }
        if let TypeLevelDslFunction::MapIntTuples(map) = &mut self.function {
            map.subst_parts_mut(shadowed, f);
        }
    }

    /// Projects this call to the type used for generic-bound checking.
    ///
    /// For example, a call that constructs `IntTuples((IntTuple[2],))` checks against a bound as
    /// `tuple[IntTuple[2]]`, while the generic argument remains the unevaluated call so later
    /// specialization can still use its call-site arguments. Invalid calls use their gradual
    /// fallback here; their evaluation error is reported when the public call result is forced.
    pub fn type_for_generic_bound_check(&self) -> Type {
        self.evaluate().unwrap_or_else(|_| self.fallback())
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
            TypeLevelDslFunction::UserDefined(function) => project(function.evaluate(&self.args)),
            TypeLevelDslFunction::MapIntTuples(map) => map.evaluate(),
        }
    }
}

impl ResolvedTypeShapeDslProgram {
    fn evaluate_lowered(
        &self,
        node_id: ResolvedTypeShapeDslNodeId,
        mut slots: Vec<DslValue>,
        budget: &mut DslEvaluationBudget,
    ) -> DslOutcome {
        let node = self.node(node_id);
        assert_eq!(
            slots.len(),
            node.parameter_domains().len(),
            "DSL helper arguments must align with validated parameters"
        );
        slots.resize(node.definition.slot_count, DslValue::Unknown);
        let mut environment = DslEnvironment {
            parameter_count: node.parameter_domains().len(),
            slots,
        };
        match self.evaluate_suite(
            node_id,
            &node.definition.parsed.definition.body,
            &mut environment,
            budget,
        ) {
            DslControlFlow::Return(result) => result,
            DslControlFlow::Continue => {
                unreachable!("validated type-level DSL function cannot fall through")
            }
        }
    }

    fn evaluate_suite(
        &self,
        node_id: ResolvedTypeShapeDslNodeId,
        suite: &[Stmt],
        environment: &mut DslEnvironment,
        budget: &mut DslEvaluationBudget,
    ) -> DslControlFlow {
        let node = self.node(node_id);
        let definition = &node.definition;
        for statement in suite {
            match statement {
                Stmt::Assign(assign) => {
                    // Validation recorded a slot under this statement's exact range, so the
                    // exact-range lookup cannot miss for a statement of the retained AST.
                    let slot = definition
                        .assignments
                        .get(assign.range)
                        .map(|assignment| assignment.slot)
                        .expect("validated assignment must have indexed-storage metadata");
                    match definition.evaluate_expression(&assign.value, environment, budget) {
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
                    let kind = definition
                        .returns
                        .get(return_stmt.range)
                        .map(|return_| return_.kind.clone())
                        .expect("validated return statement must have validation metadata");
                    return DslControlFlow::Return(match kind {
                        TypeShapeDslReturnKind::Slot { slot, kind } => {
                            match kind {
                                TypeShapeDslSlotReturnKind::DirectParameter(use_) => assert!(
                                    node.parameter_domains()[use_.parameter]
                                        .can_use_as(node.result_domain(), use_.narrowing),
                                    "validated parameter return must match or narrow to its result domain"
                                ),
                                TypeShapeDslSlotReturnKind::ParameterAlias(uses) => assert!(
                                    uses.iter()
                                        .all(|use_| node.parameter_domains()[use_.parameter]
                                            .can_use_as(node.result_domain(), use_.narrowing)),
                                    "validated alias return must match or narrow to its result domain"
                                ),
                                TypeShapeDslSlotReturnKind::KnownDomain {
                                    domain,
                                    parameter_uses,
                                } => {
                                    assert!(
                                        parameter_uses
                                            .as_deref()
                                            .is_none_or(|uses| uses.iter().all(|use_| node
                                                .parameter_domains()[use_.parameter]
                                                .can_use_as(domain, use_.narrowing))),
                                        "validated slot parameter uses must match or narrow to its return domain"
                                    );
                                    assert_eq!(
                                        domain,
                                        node.result_domain(),
                                        "validated slot return domain must match its result domain"
                                    );
                                }
                            }
                            DslOutcome::Value(environment.value(slot).clone())
                        }
                        TypeShapeDslReturnKind::Broadcast {
                            left_slot,
                            right_slot,
                            ..
                        } => evaluate_broadcast(
                            environment.value(left_slot),
                            environment.value(right_slot),
                        ),
                        TypeShapeDslReturnKind::Expression(_) => {
                            let expression = return_stmt
                                .value
                                .as_deref()
                                .expect("validated expression return has a value");
                            definition.evaluate_expression(expression, environment, budget)
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
                                node.result_domain(),
                                "validated explicit gradual DSL return domain must match its result domain"
                            );
                            DslOutcome::ExplicitGradual
                        }
                        TypeShapeDslReturnKind::HelperCall(helper_index) => {
                            let helper = &definition.helper_calls[helper_index];
                            let target = node.helper_targets[helper_index];
                            let arguments = helper
                                .arguments
                                .iter()
                                .map(|argument| environment.value(argument.slot).clone())
                                .collect();
                            self.evaluate_lowered(target, arguments, budget)
                        }
                    });
                }
                Stmt::If(if_stmt) => {
                    match self.evaluate_if(node_id, if_stmt, environment, budget) {
                        DslControlFlow::Continue => {}
                        result @ DslControlFlow::Return(_) => return result,
                    }
                }
                _ => unreachable!(
                    "validated type-level DSL suite contains only assignments, if, and return"
                ),
            }
        }
        DslControlFlow::Continue
    }

    fn evaluate_if(
        &self,
        node_id: ResolvedTypeShapeDslNodeId,
        if_stmt: &StmtIf,
        environment: &mut DslEnvironment,
        budget: &mut DslEvaluationBudget,
    ) -> DslControlFlow {
        let definition = &self.node(node_id).definition;
        match definition.evaluate_condition(&if_stmt.test, environment, budget) {
            Err(error) => return DslControlFlow::Return(DslOutcome::Invalid(error)),
            Ok(DslCondition::True) => {
                return self.evaluate_suite(node_id, &if_stmt.body, environment, budget);
            }
            Ok(DslCondition::False) => {}
            Ok(DslCondition::Unknown | DslCondition::UnknownWithPossibleError) => {
                return DslControlFlow::Return(DslOutcome::Value(DslValue::Unknown));
            }
        }
        for clause in &if_stmt.elif_else_clauses {
            match &clause.test {
                Some(test) => match definition.evaluate_condition(test, environment, budget) {
                    Err(error) => return DslControlFlow::Return(DslOutcome::Invalid(error)),
                    Ok(DslCondition::True) => {
                        return self.evaluate_suite(node_id, &clause.body, environment, budget);
                    }
                    Ok(DslCondition::False) => {}
                    Ok(DslCondition::Unknown | DslCondition::UnknownWithPossibleError) => {
                        return DslControlFlow::Return(DslOutcome::Value(DslValue::Unknown));
                    }
                },
                None => {
                    return self.evaluate_suite(node_id, &clause.body, environment, budget);
                }
            }
        }
        DslControlFlow::Continue
    }
}

impl StructurallyValidatedTypeShapeDslFunction {
    fn expression_kind(&self, expression: &Expr) -> TypeShapeDslExpressionKind {
        self.expressions
            .get(expression.range())
            .map(|metadata| metadata.kind.clone())
            .expect("validated DSL value expression must have validation metadata")
    }

    // TODO(stroxler): Compile validated expressions into a small IR so evaluation does not need
    // to look up source-range metadata and then re-read the AST structure.
    fn evaluate_generator_items(
        &self,
        source: &Expr,
        environment: &DslEnvironment,
        budget: &mut DslEvaluationBudget,
    ) -> Result<EvaluatedGeneratorItems, ShapeError> {
        let source = self.evaluate_expression(source, environment, budget);
        let item_limit = budget.remaining_generator_steps;
        Ok(match source {
            DslOutcome::Value(DslValue::Shape(shape)) => match shape.view() {
                IntTupleView::Concrete(dimensions) => {
                    let common_member = DslValue::Dimension(
                        dimensions
                            .first()
                            .filter(|first| dimensions.iter().all(|dimension| dimension == *first))
                            .cloned()
                            .unwrap_or(Int::Int),
                    );
                    EvaluatedGeneratorItems::Known {
                        values: dimensions
                            .iter()
                            .take(item_limit)
                            .cloned()
                            .map(DslValue::Dimension)
                            .collect(),
                        truncated: dimensions.len() > item_limit,
                        common_member,
                    }
                }
                IntTupleView::Gradual | IntTupleView::Unpacked { .. } => {
                    EvaluatedGeneratorItems::Unknown {
                        common_member: DslValue::Dimension(Int::Int),
                    }
                }
            },
            DslOutcome::Value(DslValue::IntTuples(DslIntTuples::Fixed(shapes))) => {
                EvaluatedGeneratorItems::Known {
                    values: shapes
                        .iter()
                        .take(item_limit)
                        .cloned()
                        .map(DslValue::Shape)
                        .collect(),
                    truncated: shapes.len() > item_limit,
                    common_member: DslValue::Shape(common_int_tuple_member(&shapes)),
                }
            }
            DslOutcome::Value(DslValue::IntTuples(DslIntTuples::Unbounded(member))) => {
                EvaluatedGeneratorItems::Unknown {
                    common_member: DslValue::Shape(member),
                }
            }
            DslOutcome::Value(DslValue::FlagSequence(sequence)) => {
                let common_member = match &sequence {
                    DslFlagSequence::Values(values) => values
                        .first()
                        .filter(|first| values.iter().all(|value| value == *first))
                        .copied()
                        .map_or(DslValue::Unknown, DslValue::FlagInt),
                    DslFlagSequence::Range { .. } => DslValue::Unknown,
                };
                let Some((values, truncated)) = sequence.bounded_values(item_limit) else {
                    return Ok(EvaluatedGeneratorItems::Unknown { common_member });
                };
                EvaluatedGeneratorItems::Known {
                    values: values.into_iter().map(DslValue::FlagInt).collect(),
                    truncated,
                    common_member,
                }
            }
            DslOutcome::Value(DslValue::Unknown) => EvaluatedGeneratorItems::Unknown {
                common_member: DslValue::Unknown,
            },
            DslOutcome::Invalid(error) => return Err(error),
            DslOutcome::ExplicitGradual => {
                unreachable!("validated generator source cannot return gradual")
            }
            DslOutcome::Value(_) => {
                unreachable!(
                    "validated generator source is an IntTuple, IntTuples, or Flag sequence"
                )
            }
        })
    }

    fn evaluate_generator(
        &self,
        generator: &ExprGenerator,
        binder: usize,
        binders: usize,
        environment: &DslEnvironment,
        result: GeneratorResultKind,
        budget: &mut DslEvaluationBudget,
    ) -> DslOutcome {
        let [comprehension] = generator.generators.as_slice() else {
            unreachable!("validated constructor generator has exactly one clause")
        };
        let source_expressions = match self.expression_kind(&comprehension.iter) {
            TypeShapeDslExpressionKind::GeneratorZip { sources } => {
                let Expr::Call(call) = &comprehension.iter else {
                    unreachable!("validated generator zip source is a call")
                };
                assert_eq!(
                    call.arguments.args.len(),
                    sources,
                    "validated generator zip metadata must align with its arguments"
                );
                &call.arguments.args
            }
            _ => std::slice::from_ref(&comprehension.iter),
        };
        assert_eq!(
            source_expressions.len(),
            binders,
            "validated generator targets must align with source lanes"
        );
        if source_expressions.is_empty() {
            return match result {
                GeneratorResultKind::Dimensions => {
                    DslOutcome::Value(DslValue::DimensionTuple(Vec::new()))
                }
                GeneratorResultKind::FlagValues => {
                    DslOutcome::Value(DslValue::FlagSequence(DslFlagSequence::Values(Vec::new())))
                }
                GeneratorResultKind::IntTuples => {
                    DslOutcome::Value(DslValue::IntTuples(DslIntTuples::Fixed(Vec::new())))
                }
            };
        }
        let mut sources = Vec::with_capacity(source_expressions.len());
        for source in source_expressions {
            match self.evaluate_generator_items(source, environment, budget) {
                Ok(items) => sources.push(items),
                Err(error) => return DslOutcome::Invalid(error),
            }
        }
        if sources
            .iter()
            .any(|source| matches!(source, EvaluatedGeneratorItems::Known { values, truncated: false, .. } if values.is_empty()))
        {
            return match result {
                GeneratorResultKind::Dimensions => {
                    DslOutcome::Value(DslValue::DimensionTuple(Vec::new()))
                }
                GeneratorResultKind::FlagValues => {
                    DslOutcome::Value(DslValue::FlagSequence(DslFlagSequence::Values(Vec::new())))
                }
                GeneratorResultKind::IntTuples => {
                    DslOutcome::Value(DslValue::IntTuples(DslIntTuples::Fixed(Vec::new())))
                }
            };
        }
        let source_members = sources
            .iter()
            .map(|source| source.common_member().clone())
            .collect::<Vec<_>>();
        if sources
            .iter()
            .any(|source| matches!(source, EvaluatedGeneratorItems::Unknown { .. }))
        {
            return self.indefinite_generator_output(
                generator,
                binder,
                &source_members,
                environment,
                result,
                budget,
            );
        }
        let sources = sources
            .into_iter()
            .map(|source| match source {
                EvaluatedGeneratorItems::Known {
                    values, truncated, ..
                } => (values, truncated),
                EvaluatedGeneratorItems::Unknown { .. } => {
                    unreachable!("unknown generator sources returned above")
                }
            })
            .collect::<Vec<_>>();
        let length = sources
            .iter()
            .map(|(values, _)| values.len())
            .min()
            .expect("a generator with no source lanes returned above");
        let truncated = sources
            .iter()
            .filter(|(values, _)| values.len() == length)
            .all(|(_, truncated)| *truncated);

        let mut dimensions = Vec::new();
        let mut flag_values = Vec::new();
        let mut shapes = Vec::new();
        let mut unknown_cardinality = false;
        let mut iteration = environment.clone();
        for index in 0..length {
            if !budget.consume_generator_step() {
                return match result {
                    GeneratorResultKind::IntTuples => DslOutcome::Value(DslValue::IntTuples(
                        DslIntTuples::Unbounded(IntTuple::shapeless()),
                    )),
                    GeneratorResultKind::Dimensions | GeneratorResultKind::FlagValues => {
                        DslOutcome::Value(DslValue::Unknown)
                    }
                };
            }
            for (lane, (values, _)) in sources.iter().enumerate() {
                iteration.assign(
                    binder + lane,
                    values
                        .get(index)
                        .expect("shortest source length guarantees an item in every lane")
                        .clone(),
                );
            }
            if let Some(filter) = comprehension.ifs.first() {
                match self.evaluate_condition(filter, &iteration, budget) {
                    Err(error) => return DslOutcome::Invalid(error),
                    Ok(DslCondition::False) => continue,
                    Ok(DslCondition::Unknown | DslCondition::UnknownWithPossibleError) => {
                        unknown_cardinality = true;
                        if result == GeneratorResultKind::IntTuples {
                            shapes.push(self.evaluate_uncertain_int_tuples_generator_member(
                                &generator.elt,
                                &iteration,
                                budget,
                            ));
                        }
                        continue;
                    }
                    Ok(DslCondition::True) => {}
                }
            }
            match (
                result,
                self.evaluate_expression(&generator.elt, &iteration, budget),
            ) {
                (_, invalid @ DslOutcome::Invalid(_)) => return invalid,
                (
                    GeneratorResultKind::Dimensions,
                    DslOutcome::Value(DslValue::GradualInt | DslValue::Unknown),
                ) => {
                    // The source cardinality and filter membership already determine the rank.
                    dimensions.push(Int::Int)
                }
                (
                    GeneratorResultKind::FlagValues,
                    DslOutcome::Value(DslValue::GradualInt | DslValue::Unknown),
                ) => {
                    // Flag sequences have no gradual element representation.
                    unknown_cardinality = true
                }
                (GeneratorResultKind::IntTuples, DslOutcome::Value(DslValue::Unknown)) => {
                    shapes.push(IntTuple::shapeless())
                }
                (
                    GeneratorResultKind::Dimensions,
                    DslOutcome::Value(DslValue::Dimension(value)),
                ) => dimensions.push(value),
                (GeneratorResultKind::FlagValues, DslOutcome::Value(DslValue::FlagInt(value))) => {
                    flag_values.push(value)
                }
                (GeneratorResultKind::IntTuples, DslOutcome::Value(DslValue::Shape(shape))) => {
                    shapes.push(shape)
                }
                (_, DslOutcome::ExplicitGradual) => {
                    unreachable!("validated generator element cannot return gradual")
                }
                _ => unreachable!("validated generator element has its constructor's domain"),
            }
        }
        if unknown_cardinality || truncated {
            match result {
                GeneratorResultKind::IntTuples => {
                    DslOutcome::Value(DslValue::IntTuples(DslIntTuples::Unbounded(if truncated {
                        // The evaluated prefix says nothing about members beyond the shared
                        // iteration budget.
                        IntTuple::shapeless()
                    } else {
                        common_int_tuple_member(&shapes)
                    })))
                }
                GeneratorResultKind::Dimensions | GeneratorResultKind::FlagValues => {
                    DslOutcome::Value(DslValue::Unknown)
                }
            }
        } else {
            match result {
                GeneratorResultKind::Dimensions => {
                    DslOutcome::Value(DslValue::DimensionTuple(dimensions))
                }
                GeneratorResultKind::FlagValues => {
                    DslOutcome::Value(DslValue::FlagSequence(DslFlagSequence::Values(flag_values)))
                }
                GeneratorResultKind::IntTuples => {
                    DslOutcome::Value(DslValue::IntTuples(DslIntTuples::Fixed(shapes)))
                }
            }
        }
    }

    fn indefinite_generator_output(
        &self,
        generator: &ExprGenerator,
        binder: usize,
        source_members: &[DslValue],
        environment: &DslEnvironment,
        result: GeneratorResultKind,
        budget: &mut DslEvaluationBudget,
    ) -> DslOutcome {
        if result != GeneratorResultKind::IntTuples {
            return DslOutcome::Value(DslValue::Unknown);
        }
        let mut iteration = environment.clone();
        for (lane, member) in source_members.iter().enumerate() {
            iteration.assign(binder + lane, member.clone());
        }
        let member =
            self.evaluate_uncertain_int_tuples_generator_member(&generator.elt, &iteration, budget);
        DslOutcome::Value(DslValue::IntTuples(DslIntTuples::Unbounded(member)))
    }

    /// Evaluates an `IntTuples` generator member that may not occur at runtime.
    ///
    /// Uncertain membership or cardinality means an unknown or invalid common-member result
    /// cannot justify another diagnostic. It establishes only that the member is a shapeless
    /// `IntTuple`.
    fn evaluate_uncertain_int_tuples_generator_member(
        &self,
        element: &Expr,
        environment: &DslEnvironment,
        budget: &mut DslEvaluationBudget,
    ) -> IntTuple {
        match self.evaluate_expression(element, environment, budget) {
            DslOutcome::Value(DslValue::Shape(shape)) => shape,
            DslOutcome::Value(DslValue::Unknown) | DslOutcome::Invalid(_) => IntTuple::shapeless(),
            DslOutcome::ExplicitGradual => {
                unreachable!("validated generator element cannot return gradual")
            }
            DslOutcome::Value(_) => {
                unreachable!("validated IntTuples generator element is a shape")
            }
        }
    }

    fn evaluate_expression(
        &self,
        expression: &Expr,
        environment: &DslEnvironment,
        budget: &mut DslEvaluationBudget,
    ) -> DslOutcome {
        match self.expression_kind(expression) {
            TypeShapeDslExpressionKind::DimensionSlot { slot, .. } => match environment.value(slot)
            {
                DslValue::FlagInt(value) => {
                    DslOutcome::Value(DslValue::Dimension(Int::Literal(*value)))
                }
                DslValue::GradualInt => DslOutcome::Value(DslValue::Dimension(Int::Int)),
                value @ (DslValue::Dimension(_) | DslValue::Unknown) => {
                    DslOutcome::Value(value.clone())
                }
                _ => unreachable!("validated dimension slot contains a dimension value"),
            },
            TypeShapeDslExpressionKind::GeneratorElementAsDimension { slot, .. } => {
                match environment.value(slot) {
                    DslValue::Dimension(value) => {
                        DslOutcome::Value(DslValue::Dimension(value.clone()))
                    }
                    DslValue::FlagInt(value) => {
                        DslOutcome::Value(DslValue::Dimension(Int::Literal(*value)))
                    }
                    DslValue::GradualInt => DslOutcome::Value(DslValue::Dimension(Int::Int)),
                    DslValue::Unknown => DslOutcome::Value(DslValue::Unknown),
                    _ => unreachable!("generator elements are integer values"),
                }
            }
            TypeShapeDslExpressionKind::GeneratorZip { .. } => {
                unreachable!("generator zip sources are evaluated by their enclosing generator")
            }
            TypeShapeDslExpressionKind::IntegerSlot { slot, .. } => match environment.value(slot) {
                DslValue::FlagInt(value) => {
                    DslOutcome::Value(DslValue::Dimension(Int::Literal(*value)))
                }
                DslValue::GradualInt => DslOutcome::Value(DslValue::Dimension(Int::Int)),
                value @ (DslValue::Dimension(_) | DslValue::Unknown) => {
                    DslOutcome::Value(value.clone())
                }
                // Validation admits only dimension values, exact integer Flags, and optional
                // integers narrowed to exclude `None`.
                _ => unreachable!("validated integer slot contains an integer value"),
            },
            TypeShapeDslExpressionKind::IntTupleSlot { slot, .. } => {
                DslOutcome::Value(environment.value(slot).clone())
            }
            TypeShapeDslExpressionKind::IntTupleSlice => {
                let Expr::Subscript(subscript) = expression else {
                    unreachable!("validated IntTuple slice is a subscript")
                };
                let Expr::Slice(slice) = subscript.slice.as_ref() else {
                    unreachable!("validated IntTuple slice has slice syntax")
                };
                let shape = self.evaluate_expression(&subscript.value, environment, budget);
                let mut evaluate_bound = |bound: Option<&Expr>| {
                    bound.map_or(DslOutcome::Value(DslValue::FlagNone), |bound| {
                        match integer_literal(bound) {
                            IntegerLiteral::Unrepresentable { negative } => {
                                DslOutcome::Value(DslValue::FlagInt(if negative {
                                    i64::MIN
                                } else {
                                    i64::MAX
                                }))
                            }
                            IntegerLiteral::NotLiteral | IntegerLiteral::Value(_) => {
                                self.evaluate_expression(bound, environment, budget)
                            }
                        }
                    })
                };
                let start = evaluate_bound(slice.lower.as_deref());
                let stop = evaluate_bound(slice.upper.as_deref());
                for outcome in [&shape, &start, &stop] {
                    if let DslOutcome::Invalid(error) = outcome {
                        return DslOutcome::Invalid(error.clone());
                    }
                }
                let value = |outcome| match outcome {
                    DslOutcome::Value(value) => value,
                    DslOutcome::Invalid(_) => {
                        unreachable!("invalid slice outcomes return before decoding values")
                    }
                    DslOutcome::ExplicitGradual => {
                        unreachable!("validated slice expressions cannot return explicit gradual")
                    }
                };
                let shape = match value(shape) {
                    DslValue::Shape(shape) => shape,
                    DslValue::Unknown => return DslOutcome::Value(DslValue::Unknown),
                    _ => unreachable!("validated IntTuple slice operand is a shape"),
                };
                let bound = |outcome| match value(outcome) {
                    DslValue::FlagNone => Ok(None),
                    DslValue::FlagInt(value) => Ok(Some(value)),
                    DslValue::GradualInt | DslValue::Unknown => {
                        Err(DslOutcome::Value(DslValue::Unknown))
                    }
                    _ => unreachable!("validated IntTuple slice bound is an optional Flag[int]"),
                };
                let start = match bound(start) {
                    Ok(start) => start,
                    Err(outcome) => return outcome,
                };
                let stop = match bound(stop) {
                    Ok(stop) => stop,
                    Err(outcome) => return outcome,
                };
                evaluate_int_tuple_slice(&shape, start, stop)
                    .map_or(DslOutcome::Value(DslValue::Unknown), |shape| {
                        DslOutcome::Value(DslValue::Shape(shape))
                    })
            }
            TypeShapeDslExpressionKind::IntTupleConcat => {
                let Expr::Call(call) = expression else {
                    unreachable!("validated IntTuple concat is a call")
                };
                let left = self.evaluate_expression(&call.arguments.args[0], environment, budget);
                let right = self.evaluate_expression(&call.arguments.args[1], environment, budget);
                match (left, right) {
                    (DslOutcome::Invalid(error), _) | (_, DslOutcome::Invalid(error)) => {
                        DslOutcome::Invalid(error)
                    }
                    (
                        DslOutcome::Value(DslValue::Shape(left)),
                        DslOutcome::Value(DslValue::Shape(right)),
                    ) => DslOutcome::Value(DslValue::Shape(left.concat(&right))),
                    (DslOutcome::Value(DslValue::Unknown), _)
                    | (_, DslOutcome::Value(DslValue::Unknown)) => {
                        DslOutcome::Value(DslValue::Unknown)
                    }
                    (DslOutcome::ExplicitGradual, _) | (_, DslOutcome::ExplicitGradual) => {
                        unreachable!("validated shape expression cannot return explicit gradual")
                    }
                    _ => unreachable!("validated concat operands are shapes"),
                }
            }
            TypeShapeDslExpressionKind::Einsum { shapes, .. } => {
                let Expr::Call(call) = expression else {
                    unreachable!("validated einsum expression is a call")
                };
                let spec =
                    match self.evaluate_expression(&call.arguments.args[0], environment, budget) {
                        DslOutcome::Value(DslValue::FlagString(spec)) => Some(spec),
                        DslOutcome::Value(DslValue::FlagNone | DslValue::Unknown) => None,
                        invalid @ DslOutcome::Invalid(_) => return invalid,
                        DslOutcome::ExplicitGradual => {
                            unreachable!("validated value expression cannot return gradual")
                        }
                        DslOutcome::Value(_) => {
                            unreachable!("validated einsum equation is a string Flag")
                        }
                    };
                let equation = match spec {
                    Some(spec) => match parse_einsum_equation(&spec) {
                        EinsumClassification::Supported(equation) => Some(equation),
                        EinsumClassification::Unsupported(_) => None,
                        EinsumClassification::Invalid(error) => {
                            return DslOutcome::Invalid(ShapeError::ShapeComputation {
                                message: error.message(),
                            });
                        }
                    },
                    None => None,
                };
                let operands = match environment.value(shapes) {
                    DslValue::IntTuples(DslIntTuples::Fixed(operands)) => Some(operands.as_slice()),
                    DslValue::IntTuples(DslIntTuples::Unbounded(_)) | DslValue::Unknown => None,
                    _ => unreachable!("validated einsum operands are an IntTuples value"),
                };
                let Some((equation, operands)) = equation.zip(operands) else {
                    return DslOutcome::Value(DslValue::Unknown);
                };
                match evaluate_einsum(&equation, operands) {
                    Ok(shape) => DslOutcome::Value(DslValue::Shape(shape)),
                    Err(error) => DslOutcome::Invalid(error),
                }
            }
            TypeShapeDslExpressionKind::DimensionLiteral(literal) => literal
                .map_or(DslOutcome::Value(DslValue::Unknown), |literal| {
                    DslOutcome::Value(DslValue::Dimension(Int::Literal(literal)))
                }),
            TypeShapeDslExpressionKind::Gradual => DslOutcome::Value(DslValue::Dimension(Int::Int)),
            TypeShapeDslExpressionKind::IntTupleIndex { shape, .. } => {
                let Expr::Subscript(subscript) = expression else {
                    unreachable!("validated IntTuple index expression is a subscript")
                };
                let index = match integer_literal(&subscript.slice) {
                    IntegerLiteral::Unrepresentable { negative } => {
                        if negative {
                            i64::MIN
                        } else {
                            i64::MAX
                        }
                    }
                    IntegerLiteral::NotLiteral | IntegerLiteral::Value(_) => {
                        match self.evaluate_expression(&subscript.slice, environment, budget) {
                            DslOutcome::Value(DslValue::FlagInt(index)) => index,
                            DslOutcome::Value(DslValue::Unknown) => {
                                return DslOutcome::Value(DslValue::Unknown);
                            }
                            DslOutcome::Value(DslValue::GradualInt) => {
                                return DslOutcome::Value(DslValue::Dimension(Int::Int));
                            }
                            DslOutcome::ExplicitGradual => {
                                unreachable!(
                                    "validated Flag integer index cannot be explicitly gradual"
                                )
                            }
                            invalid @ DslOutcome::Invalid(_) => return invalid,
                            DslOutcome::Value(_) => {
                                unreachable!("validated IntTuple index evaluates to a Flag integer")
                            }
                        }
                    }
                };
                let shape = match environment.value(shape) {
                    DslValue::Shape(shape) => shape,
                    DslValue::Unknown => return DslOutcome::Value(DslValue::Unknown),
                    _ => {
                        unreachable!("validated IntTuple index parameter evaluates to a shape")
                    }
                };
                let index = i128::from(index);
                match shape.view() {
                    IntTupleView::Concrete(shape) => {
                        let length = shape.len() as i128;
                        let index = if index < 0 { index + length } else { index };
                        if index < 0 || index >= length {
                            return DslOutcome::Invalid(ShapeError::ShapeComputation {
                                message: "IntTuple index out of bounds".to_owned(),
                            });
                        }
                        DslOutcome::Value(DslValue::Dimension(shape[index as usize].clone()))
                    }
                    IntTupleView::Unpacked { prefix, .. }
                        if index >= 0
                            && usize::try_from(index).is_ok_and(|index| index < prefix.len()) =>
                    {
                        let index =
                            usize::try_from(index).expect("validated prefix index fits in usize");
                        DslOutcome::Value(DslValue::Dimension(prefix[index].clone()))
                    }
                    // A symbolic-rank shape has no known total length, but indexes within its
                    // fixed prefix or counting back within its fixed suffix are still known.
                    IntTupleView::Unpacked { suffix, .. }
                        if index < 0
                            && usize::try_from(index.unsigned_abs())
                                .is_ok_and(|offset| offset <= suffix.len()) =>
                    {
                        let offset = usize::try_from(index.unsigned_abs())
                            .expect("validated suffix index fits in usize");
                        let offset = suffix.len() - offset;
                        DslOutcome::Value(DslValue::Dimension(suffix[offset].clone()))
                    }
                    IntTupleView::Unpacked { .. } | IntTupleView::Gradual => {
                        DslOutcome::Value(DslValue::Unknown)
                    }
                }
            }
            TypeShapeDslExpressionKind::IntTuplesIndex { shapes } => {
                let Expr::Subscript(subscript) = expression else {
                    unreachable!("validated IntTuples index expression is a subscript")
                };
                let index = match integer_literal(&subscript.slice) {
                    IntegerLiteral::Unrepresentable { negative } => {
                        if negative {
                            i64::MIN
                        } else {
                            i64::MAX
                        }
                    }
                    IntegerLiteral::NotLiteral | IntegerLiteral::Value(_) => {
                        match self.evaluate_expression(&subscript.slice, environment, budget) {
                            DslOutcome::Value(DslValue::FlagInt(index)) => index,
                            DslOutcome::Value(DslValue::Unknown | DslValue::GradualInt) => {
                                return DslOutcome::Value(DslValue::Unknown);
                            }
                            DslOutcome::ExplicitGradual => {
                                unreachable!(
                                    "validated Flag integer index cannot be explicitly gradual"
                                )
                            }
                            invalid @ DslOutcome::Invalid(_) => return invalid,
                            DslOutcome::Value(_) => {
                                unreachable!(
                                    "validated IntTuples index evaluates to a Flag integer"
                                )
                            }
                        }
                    }
                };
                match environment.value(shapes) {
                    DslValue::IntTuples(DslIntTuples::Fixed(shapes)) => {
                        let length = shapes.len() as i128;
                        let index = i128::from(index);
                        let index = if index < 0 { index + length } else { index };
                        if index < 0 || index >= length {
                            return DslOutcome::Invalid(ShapeError::ShapeComputation {
                                message: "`IntTuples` index out of bounds".to_owned(),
                            });
                        }
                        DslOutcome::Value(DslValue::Shape(shapes[index as usize].clone()))
                    }
                    DslValue::IntTuples(DslIntTuples::Unbounded(shape)) => {
                        DslOutcome::Value(DslValue::Shape(shape.clone()))
                    }
                    DslValue::Unknown => DslOutcome::Value(DslValue::Unknown),
                    _ => unreachable!("validated IntTuples index operand is an IntTuples value"),
                }
            }
            TypeShapeDslExpressionKind::DimensionTuple => {
                let Expr::Tuple(tuple) = expression else {
                    unreachable!("validated dimension tuple expression is a tuple")
                };
                let mut dimensions = Vec::with_capacity(tuple.elts.len());
                for element in &tuple.elts {
                    match self.evaluate_expression(element, environment, budget) {
                        DslOutcome::Value(DslValue::Dimension(dimension)) => {
                            dimensions.push(dimension)
                        }
                        DslOutcome::Value(DslValue::Unknown) => dimensions.push(Int::Int),
                        DslOutcome::ExplicitGradual => {
                            unreachable!("validated value expression cannot return gradual")
                        }
                        invalid @ DslOutcome::Invalid(_) => return invalid,
                        DslOutcome::Value(_) => {
                            unreachable!("validated IntTuple element produces an Int value")
                        }
                    }
                }
                DslOutcome::Value(DslValue::DimensionTuple(dimensions))
            }
            TypeShapeDslExpressionKind::IntTupleConstructor => {
                let Expr::Call(call) = expression else {
                    unreachable!("validated IntTuple constructor expression is a call")
                };
                match self.evaluate_expression(&call.arguments.args[0], environment, budget) {
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
            TypeShapeDslExpressionKind::IntTuplesConstructor => {
                let Expr::Call(call) = expression else {
                    unreachable!("validated IntTuples constructor expression is a call")
                };
                if matches!(&call.arguments.args[0], Expr::Generator(_)) {
                    return self.evaluate_expression(&call.arguments.args[0], environment, budget);
                }
                let Expr::Tuple(tuple) = &call.arguments.args[0] else {
                    unreachable!("validated IntTuples constructor receives a fixed tuple")
                };
                let mut shapes = Vec::with_capacity(tuple.elts.len());
                for element in &tuple.elts {
                    match self.evaluate_expression(element, environment, budget) {
                        DslOutcome::Value(DslValue::Shape(shape)) => shapes.push(shape),
                        DslOutcome::Value(DslValue::Unknown) => shapes.push(IntTuple::shapeless()),
                        DslOutcome::ExplicitGradual => {
                            unreachable!("validated value expression cannot return gradual")
                        }
                        invalid @ DslOutcome::Invalid(_) => return invalid,
                        DslOutcome::Value(_) => {
                            unreachable!("validated IntTuples constructor elements are shapes")
                        }
                    }
                }
                DslOutcome::Value(DslValue::IntTuples(DslIntTuples::Fixed(shapes)))
            }
            TypeShapeDslExpressionKind::IntTupleProduct => {
                let Expr::Call(call) = expression else {
                    unreachable!("validated IntTuple product expression is a call")
                };
                match self.evaluate_expression(&call.arguments.args[0], environment, budget) {
                    DslOutcome::Value(DslValue::Shape(shape)) => match shape.product() {
                        // TODO(stroxler): Preserve a gradual product as a gradual dimension once
                        // callers can distinguish an unknown equality check from a proven match.
                        Int::Int => DslOutcome::Value(DslValue::Unknown),
                        product => DslOutcome::Value(DslValue::Dimension(product)),
                    },
                    DslOutcome::Value(DslValue::Unknown) => DslOutcome::Value(DslValue::Unknown),
                    DslOutcome::ExplicitGradual => {
                        unreachable!("validated IntTuple product operand cannot return gradual")
                    }
                    invalid @ DslOutcome::Invalid(_) => invalid,
                    DslOutcome::Value(_) => {
                        unreachable!("validated IntTuple product receives a shape")
                    }
                }
            }
            TypeShapeDslExpressionKind::IntTupleSum => {
                let Expr::Call(call) = expression else {
                    unreachable!("validated IntTuple sum expression is a call")
                };
                match self.evaluate_expression(&call.arguments.args[0], environment, budget) {
                    DslOutcome::Value(DslValue::Shape(shape)) => match shape.sum() {
                        Int::Int => DslOutcome::Value(DslValue::Unknown),
                        sum => DslOutcome::Value(DslValue::Dimension(sum)),
                    },
                    DslOutcome::Value(DslValue::Unknown) => DslOutcome::Value(DslValue::Unknown),
                    DslOutcome::ExplicitGradual => {
                        unreachable!("validated IntTuple sum operand cannot return gradual")
                    }
                    invalid @ DslOutcome::Invalid(_) => invalid,
                    DslOutcome::Value(_) => {
                        unreachable!("validated IntTuple sum receives a shape")
                    }
                }
            }
            TypeShapeDslExpressionKind::IntTupleLength { shape, domain, .. } => {
                let length = match environment.value(shape) {
                    DslValue::Shape(shape) => match shape.view() {
                        IntTupleView::Concrete(shape) => Some(shape.len()),
                        IntTupleView::Unpacked { .. } | IntTupleView::Gradual => None,
                    },
                    DslValue::IntTuples(DslIntTuples::Fixed(shapes)) => Some(shapes.len()),
                    DslValue::IntTuples(DslIntTuples::Unbounded(_)) => None,
                    DslValue::Unknown => return DslOutcome::Value(DslValue::Unknown),
                    _ => {
                        unreachable!(
                            "validated length parameter evaluates to an IntTuple or IntTuples value"
                        )
                    }
                };
                let Some(length) = length else {
                    return DslOutcome::Value(DslValue::Unknown);
                };
                let length = i64::try_from(length)
                    .expect("concrete IntTuple or IntTuples length must fit in an i64");
                DslOutcome::Value(match domain {
                    DslIntegerDomain::Flag => DslValue::FlagInt(length),
                    DslIntegerDomain::Dimension => DslValue::Dimension(Int::Literal(length)),
                })
            }
            TypeShapeDslExpressionKind::GeneratorSourceSlot { slot, .. } => {
                DslOutcome::Value(environment.value(slot).clone())
            }
            TypeShapeDslExpressionKind::Slot(slot) => {
                DslOutcome::Value(environment.value(slot).clone())
            }
            TypeShapeDslExpressionKind::GeneratorElementAsFlagInt { slot, .. } => match environment
                .value(slot)
            {
                DslValue::FlagInt(value) => DslOutcome::Value(DslValue::FlagInt(*value)),
                DslValue::Dimension(Int::Literal(value)) => {
                    DslOutcome::Value(DslValue::FlagInt(*value))
                }
                DslValue::Dimension(_) | DslValue::Unknown => DslOutcome::Value(DslValue::Unknown),
                DslValue::GradualInt => unreachable!(
                    "generator sources normalize gradual integer elements to dimensions"
                ),
                _ => unreachable!("generator elements are integer values"),
            },
            TypeShapeDslExpressionKind::GeneratorElementAsIntTuple { slot, .. } => {
                match environment.value(slot) {
                    value @ (DslValue::Shape(_) | DslValue::Unknown) => {
                        DslOutcome::Value(value.clone())
                    }
                    _ => unreachable!("validated generator element is an IntTuple value"),
                }
            }
            TypeShapeDslExpressionKind::FlagValueSlot { slot, .. } => {
                DslOutcome::Value(environment.value(slot).clone())
            }
            TypeShapeDslExpressionKind::FlagIntLiteral(literal) => literal
                .map_or(DslOutcome::Value(DslValue::Unknown), |literal| {
                    DslOutcome::Value(DslValue::FlagInt(literal))
                }),
            TypeShapeDslExpressionKind::FlagStringLiteral => {
                let Expr::StringLiteral(literal) = expression else {
                    unreachable!("validated Flag string literal is a string literal")
                };
                let Lit::Str(value) = Lit::from_string_literal(literal)
                    .expect("validated Flag string literal fits the literal size limit")
                else {
                    unreachable!("string literal lowering produces a string value")
                };
                DslOutcome::Value(DslValue::FlagString(value))
            }
            TypeShapeDslExpressionKind::FlagBool(value) => {
                DslOutcome::Value(DslValue::FlagBool(value))
            }
            TypeShapeDslExpressionKind::FlagNone => DslOutcome::Value(DslValue::FlagNone),
            TypeShapeDslExpressionKind::FlagTuple => {
                let Expr::Tuple(tuple) = expression else {
                    unreachable!("validated Flag tuple expression is a tuple display")
                };
                let mut values = Vec::with_capacity(tuple.elts.len());
                let mut unknown = false;
                for element in &tuple.elts {
                    match self.evaluate_expression(element, environment, budget) {
                        DslOutcome::Value(DslValue::FlagInt(value)) => values.push(value),
                        DslOutcome::Value(DslValue::GradualInt | DslValue::Unknown) => {
                            unknown = true
                        }
                        DslOutcome::ExplicitGradual => {
                            unreachable!("validated value expression cannot return gradual")
                        }
                        invalid @ DslOutcome::Invalid(_) => return invalid,
                        DslOutcome::Value(_) => {
                            unreachable!("validated Flag tuple elements evaluate to Flag integers")
                        }
                    }
                }
                if unknown {
                    DslOutcome::Value(DslValue::Unknown)
                } else {
                    DslOutcome::Value(DslValue::FlagSequence(DslFlagSequence::Values(values)))
                }
            }
            TypeShapeDslExpressionKind::FlagRange => {
                let Expr::Call(call) = expression else {
                    unreachable!("validated range expression is a call")
                };
                let mut values = Vec::with_capacity(call.arguments.args.len());
                for argument in &call.arguments.args {
                    match self.evaluate_expression(argument, environment, budget) {
                        DslOutcome::Value(DslValue::FlagInt(value)) => values.push(Some(value)),
                        DslOutcome::Value(DslValue::GradualInt | DslValue::Unknown) => {
                            values.push(None)
                        }
                        DslOutcome::ExplicitGradual => {
                            unreachable!("validated value expression cannot return gradual")
                        }
                        invalid @ DslOutcome::Invalid(_) => return invalid,
                        DslOutcome::Value(_) => {
                            unreachable!("validated range arguments are Flag integers")
                        }
                    }
                }
                let (start, stop, step) = match values.as_slice() {
                    [stop] => (Some(0), *stop, Some(1)),
                    [start, stop] => (*start, *stop, Some(1)),
                    [start, stop, step] => (*start, *stop, *step),
                    _ => unreachable!("validated range has one to three arguments"),
                };
                if step == Some(0) {
                    return DslOutcome::Invalid(ShapeError::ShapeComputation {
                        message: "range() arg 3 must not be zero".to_owned(),
                    });
                }
                let (Some(start), Some(stop), Some(step)) = (start, stop, step) else {
                    return DslOutcome::Value(DslValue::Unknown);
                };
                DslOutcome::Value(DslValue::FlagSequence(DslFlagSequence::Range {
                    start,
                    stop,
                    step,
                }))
            }
            TypeShapeDslExpressionKind::FlagSequenceLength => {
                let Expr::Call(call) = expression else {
                    unreachable!("validated Flag sequence length expression is a call")
                };
                match self.evaluate_expression(&call.arguments.args[0], environment, budget) {
                    DslOutcome::Value(DslValue::FlagSequence(sequence)) => sequence
                        .len()
                        .map_or(DslOutcome::Value(DslValue::Unknown), |length| {
                            DslOutcome::Value(DslValue::FlagInt(length))
                        }),
                    DslOutcome::Value(DslValue::Unknown) => DslOutcome::Value(DslValue::Unknown),
                    DslOutcome::ExplicitGradual => {
                        unreachable!("validated value expression cannot return gradual")
                    }
                    invalid @ DslOutcome::Invalid(_) => invalid,
                    DslOutcome::Value(_) => {
                        unreachable!("validated Flag length operand is a sequence")
                    }
                }
            }
            TypeShapeDslExpressionKind::FlagSequenceCount => {
                let Expr::Call(call) = expression else {
                    unreachable!("validated count expression is a call")
                };
                let Expr::Attribute(attribute) = &*call.func else {
                    unreachable!("validated count expression has an attribute callee")
                };
                let sequence = self.evaluate_expression(&attribute.value, environment, budget);
                let item = self.evaluate_expression(&call.arguments.args[0], environment, budget);
                match (sequence, item) {
                    (DslOutcome::Invalid(error), _) | (_, DslOutcome::Invalid(error)) => {
                        DslOutcome::Invalid(error)
                    }
                    (
                        DslOutcome::Value(DslValue::FlagSequence(sequence)),
                        DslOutcome::Value(DslValue::FlagInt(item)),
                    ) => sequence
                        .count(item)
                        .map_or(DslOutcome::Value(DslValue::Unknown), |count| {
                            DslOutcome::Value(DslValue::FlagInt(count))
                        }),
                    (DslOutcome::Value(DslValue::GradualInt | DslValue::Unknown), _)
                    | (_, DslOutcome::Value(DslValue::GradualInt | DslValue::Unknown)) => {
                        DslOutcome::Value(DslValue::Unknown)
                    }
                    (DslOutcome::ExplicitGradual, _) | (_, DslOutcome::ExplicitGradual) => {
                        unreachable!("validated value expression cannot return gradual")
                    }
                    _ => unreachable!("validated count uses a Flag sequence and integer"),
                }
            }
            TypeShapeDslExpressionKind::FlagSequenceIndex => {
                let Expr::Call(call) = expression else {
                    unreachable!("validated index expression is a call")
                };
                let Expr::Attribute(attribute) = &*call.func else {
                    unreachable!("validated index expression has an attribute callee")
                };
                let sequence = self.evaluate_expression(&attribute.value, environment, budget);
                let item = self.evaluate_expression(&call.arguments.args[0], environment, budget);
                match (sequence, item) {
                    (DslOutcome::Invalid(error), _) | (_, DslOutcome::Invalid(error)) => {
                        DslOutcome::Invalid(error)
                    }
                    (
                        DslOutcome::Value(DslValue::FlagSequence(sequence)),
                        DslOutcome::Value(DslValue::FlagInt(item)),
                    ) => match sequence.index(item) {
                        DslFlagSequenceIndex::Exact(index) => {
                            DslOutcome::Value(DslValue::FlagInt(index))
                        }
                        DslFlagSequenceIndex::Gradual => DslOutcome::Value(DslValue::Unknown),
                        DslFlagSequenceIndex::NotFound => {
                            DslOutcome::Invalid(ShapeError::ShapeComputation {
                                message: format!(
                                    "Flag sequence `.index` value `{item}` was not found"
                                ),
                            })
                        }
                    },
                    (DslOutcome::Value(DslValue::GradualInt | DslValue::Unknown), _)
                    | (_, DslOutcome::Value(DslValue::GradualInt | DslValue::Unknown)) => {
                        DslOutcome::Value(DslValue::Unknown)
                    }
                    (DslOutcome::ExplicitGradual, _) | (_, DslOutcome::ExplicitGradual) => {
                        unreachable!("validated value expression cannot return gradual")
                    }
                    _ => unreachable!("validated index uses a Flag sequence and integer"),
                }
            }
            TypeShapeDslExpressionKind::FlagIntArithmetic(op) => {
                let Expr::BinOp(binop) = expression else {
                    unreachable!("validated Flag arithmetic expression is a binary operation")
                };
                let left = match self.evaluate_expression(&binop.left, environment, budget) {
                    DslOutcome::Invalid(error) => return DslOutcome::Invalid(error),
                    left => left,
                };
                let right = match self.evaluate_expression(&binop.right, environment, budget) {
                    DslOutcome::Invalid(error) => return DslOutcome::Invalid(error),
                    right => right,
                };
                if matches!(
                    op,
                    TypeShapeDslArithmeticOp::FloorDivide | TypeShapeDslArithmeticOp::Modulo
                ) && matches!(right, DslOutcome::Value(DslValue::FlagInt(0)))
                {
                    return evaluate_flag_int_arithmetic(0, op, 0);
                }
                match (left, right) {
                    (DslOutcome::Value(DslValue::Unknown), _)
                    | (_, DslOutcome::Value(DslValue::Unknown)) => {
                        DslOutcome::Value(DslValue::Unknown)
                    }
                    (DslOutcome::Value(DslValue::GradualInt), _)
                    | (_, DslOutcome::Value(DslValue::GradualInt)) => {
                        DslOutcome::Value(DslValue::GradualInt)
                    }
                    (
                        DslOutcome::Value(DslValue::FlagInt(left)),
                        DslOutcome::Value(DslValue::FlagInt(right)),
                    ) => evaluate_flag_int_arithmetic(left, op, right),
                    (DslOutcome::Invalid(_), _) | (_, DslOutcome::Invalid(_)) => {
                        unreachable!("invalid eager operands are propagated before arithmetic")
                    }
                    _ => unreachable!("validated Flag arithmetic operands are integers"),
                }
            }
            TypeShapeDslExpressionKind::DimensionArithmetic(op) => {
                let Expr::BinOp(binop) = expression else {
                    unreachable!("validated dimension arithmetic expression is a binary operation")
                };
                let left = match self.evaluate_expression(&binop.left, environment, budget) {
                    DslOutcome::Invalid(error) => return DslOutcome::Invalid(error),
                    left => left,
                };
                let right = match self.evaluate_expression(&binop.right, environment, budget) {
                    DslOutcome::Invalid(error) => return DslOutcome::Invalid(error),
                    right => right,
                };
                evaluate_dimension_arithmetic(left, op, right)
            }
            TypeShapeDslExpressionKind::Conditional => {
                let Expr::If(if_expr) = expression else {
                    unreachable!("validated conditional expression is an if-expression")
                };
                match self.evaluate_condition(&if_expr.test, environment, budget) {
                    Err(error) => DslOutcome::Invalid(error),
                    Ok(DslCondition::True) => {
                        self.evaluate_expression(&if_expr.body, environment, budget)
                    }
                    Ok(DslCondition::False) => {
                        self.evaluate_expression(&if_expr.orelse, environment, budget)
                    }
                    Ok(DslCondition::Unknown | DslCondition::UnknownWithPossibleError) => {
                        DslOutcome::Value(DslValue::Unknown)
                    }
                }
            }
            TypeShapeDslExpressionKind::DimensionGenerator { binder, binders } => {
                let Expr::Generator(generator) = expression else {
                    unreachable!("validated dimension generator retains its generator AST")
                };
                self.evaluate_generator(
                    generator,
                    binder,
                    binders,
                    environment,
                    GeneratorResultKind::Dimensions,
                    budget,
                )
            }
            TypeShapeDslExpressionKind::FlagGenerator { binder, binders } => {
                let Expr::Call(call) = expression else {
                    unreachable!("validated Flag generator expression is a tuple call")
                };
                let Some(Expr::Generator(generator)) = call.arguments.args.first() else {
                    unreachable!("validated tuple call contains a generator")
                };
                self.evaluate_generator(
                    generator,
                    binder,
                    binders,
                    environment,
                    GeneratorResultKind::FlagValues,
                    budget,
                )
            }
            TypeShapeDslExpressionKind::IntTuplesGenerator { binder, binders } => {
                let Expr::Generator(generator) = expression else {
                    unreachable!("validated IntTuples generator retains its generator AST")
                };
                self.evaluate_generator(
                    generator,
                    binder,
                    binders,
                    environment,
                    GeneratorResultKind::IntTuples,
                    budget,
                )
            }
        }
    }

    fn evaluate_any_generator(
        &self,
        generator: &ExprGenerator,
        binder: usize,
        environment: &DslEnvironment,
        budget: &mut DslEvaluationBudget,
    ) -> Result<DslCondition, ShapeError> {
        let [comprehension] = generator.generators.as_slice() else {
            unreachable!("validated `any` generator has exactly one clause")
        };
        let (values, truncated) =
            match self.evaluate_generator_items(&comprehension.iter, environment, budget)? {
                EvaluatedGeneratorItems::Known {
                    values, truncated, ..
                } => (values, truncated),
                EvaluatedGeneratorItems::Unknown { .. } => return Ok(DslCondition::Unknown),
            };

        let mut saw_unknown = false;
        let mut possible_error = false;
        let mut iteration = environment.clone();
        for value in values {
            if !budget.consume_generator_step() {
                return Ok(if possible_error {
                    DslCondition::UnknownWithPossibleError
                } else {
                    DslCondition::Unknown
                });
            }
            iteration.assign(binder, value);
            if let Some(filter) = comprehension.ifs.first() {
                let filter = match self.evaluate_condition(filter, &iteration, budget) {
                    Ok(filter) => filter,
                    Err(_) if saw_unknown || possible_error => {
                        return Ok(DslCondition::UnknownWithPossibleError);
                    }
                    Err(error) => return Err(error),
                };
                match filter {
                    DslCondition::False => continue,
                    DslCondition::Unknown => {
                        saw_unknown = true;
                        // A concrete instantiation may include this item, so a guarded error must
                        // prevent a later item from making the reduction precisely true.
                        match self.evaluate_condition(&generator.elt, &iteration, budget) {
                            Err(_) | Ok(DslCondition::UnknownWithPossibleError) => {
                                possible_error = true;
                            }
                            Ok(_) => {}
                        }
                        continue;
                    }
                    DslCondition::UnknownWithPossibleError => {
                        possible_error = true;
                        continue;
                    }
                    DslCondition::True => {}
                }
            }
            let condition = match self.evaluate_condition(&generator.elt, &iteration, budget) {
                Ok(condition) => condition,
                Err(_) if saw_unknown || possible_error => {
                    return Ok(DslCondition::UnknownWithPossibleError);
                }
                Err(error) => return Err(error),
            };
            match condition {
                DslCondition::True if possible_error => {
                    return Ok(DslCondition::UnknownWithPossibleError);
                }
                DslCondition::True => return Ok(DslCondition::True),
                DslCondition::False => {}
                DslCondition::Unknown => saw_unknown = true,
                DslCondition::UnknownWithPossibleError => possible_error = true,
            }
        }
        Ok(if possible_error {
            DslCondition::UnknownWithPossibleError
        } else if saw_unknown || truncated {
            DslCondition::Unknown
        } else {
            DslCondition::False
        })
    }

    fn dimension_equality(left: &Int, right: &Int) -> DslCondition {
        match (left, right) {
            (Int::Literal(left), Int::Literal(right)) => {
                if left == right {
                    DslCondition::True
                } else {
                    DslCondition::False
                }
            }
            (Int::Int, _) | (_, Int::Int) => DslCondition::Unknown,
            (left, right) if left == right => DslCondition::True,
            _ => DslCondition::Unknown,
        }
    }

    fn negate_condition(condition: DslCondition) -> DslCondition {
        match condition {
            DslCondition::True => DslCondition::False,
            DslCondition::False => DslCondition::True,
            DslCondition::Unknown => DslCondition::Unknown,
            DslCondition::UnknownWithPossibleError => DslCondition::UnknownWithPossibleError,
        }
    }

    fn compare_dimensions(left: &Int, right: &Int, op: TypeShapeDslComparisonOp) -> DslCondition {
        match op {
            TypeShapeDslComparisonOp::Equal => Self::dimension_equality(left, right),
            TypeShapeDslComparisonOp::NotEqual => {
                Self::negate_condition(Self::dimension_equality(left, right))
            }
            TypeShapeDslComparisonOp::LessThan => match (left, right) {
                (left, right) if left == right && !matches!(left, Int::Int) => DslCondition::False,
                (Int::Literal(left), Int::Literal(right)) if left < right => DslCondition::True,
                (Int::Literal(_), Int::Literal(_)) => DslCondition::False,
                _ => DslCondition::Unknown,
            },
            TypeShapeDslComparisonOp::LessThanOrEqual
            | TypeShapeDslComparisonOp::GreaterThan
            | TypeShapeDslComparisonOp::GreaterThanOrEqual => {
                unreachable!("validated Int comparison uses only `==`, `!=`, or `<`")
            }
        }
    }

    // TODO(stroxler): Compile validated conditions into the same IR rather than traversing their
    // boolean/comparison AST again during evaluation.
    fn evaluate_condition(
        &self,
        condition: &Expr,
        environment: &DslEnvironment,
        budget: &mut DslEvaluationBudget,
    ) -> Result<DslCondition, ShapeError> {
        if let Expr::BoolOp(bool_op) = condition {
            let mut saw_unknown = false;
            let mut possible_error = false;
            for value in &bool_op.values {
                let value = match self.evaluate_condition(value, environment, budget) {
                    Ok(value) => value,
                    // An unknown prefix may short-circuit before this operand for a concrete
                    // instantiation, so an error here is not deterministic.
                    Err(_) if saw_unknown || possible_error => {
                        return Ok(DslCondition::UnknownWithPossibleError);
                    }
                    Err(error) => return Err(error),
                };
                match (bool_op.op, value) {
                    (BoolOp::And, DslCondition::False) | (BoolOp::Or, DslCondition::True) => {
                        return Ok(if possible_error {
                            DslCondition::UnknownWithPossibleError
                        } else {
                            value
                        });
                    }
                    (_, DslCondition::Unknown) => saw_unknown = true,
                    (_, DslCondition::UnknownWithPossibleError) => possible_error = true,
                    _ => {}
                }
            }
            return Ok(if possible_error {
                DslCondition::UnknownWithPossibleError
            } else if saw_unknown {
                DslCondition::Unknown
            } else {
                match bool_op.op {
                    BoolOp::And => DslCondition::True,
                    BoolOp::Or => DslCondition::False,
                }
            });
        }
        if let Expr::UnaryOp(unary) = condition
            && unary.op == UnaryOp::Not
        {
            return Ok(Self::negate_condition(self.evaluate_condition(
                &unary.operand,
                environment,
                budget,
            )?));
        }

        let kind = self
            .conditions
            .get(condition.range())
            .map(|metadata| metadata.kind.clone())
            .expect("validated atomic condition must have validation metadata");
        Ok(match kind {
            TypeShapeDslConditionKind::Any { binder } => {
                let Expr::Call(call) = condition else {
                    unreachable!("validated `any` condition is a call")
                };
                let Expr::Generator(generator) = &call.arguments.args[0] else {
                    unreachable!("validated `any` condition retains its generator")
                };
                return self.evaluate_any_generator(generator, binder, environment, budget);
            }
            TypeShapeDslConditionKind::BoolSlot { slot, .. } => match environment.value(slot) {
                DslValue::FlagBool(true) => DslCondition::True,
                DslValue::FlagBool(false) | DslValue::FlagNone => DslCondition::False,
                DslValue::Unknown => DslCondition::Unknown,
                DslValue::GradualInt
                | DslValue::FlagInt(_)
                | DslValue::FlagString(_)
                | DslValue::FlagSequence(_)
                | DslValue::Dimension(_)
                | DslValue::Shape(_)
                | DslValue::IntTuples(_)
                | DslValue::DimensionTuple(_) => {
                    unreachable!("validated boolean condition contains a boolean Flag value")
                }
            },
            TypeShapeDslConditionKind::IsConcreteInt { slot, .. } => {
                match environment.value(slot) {
                    DslValue::Dimension(Int::Literal(_)) | DslValue::FlagInt(_) => {
                        DslCondition::True
                    }
                    // Symbolic and explicit-gradual `Int` values are definitively non-concrete.
                    DslValue::GradualInt | DslValue::Dimension(_) => DslCondition::False,
                    // An omitted optional dimension is definitively non-concrete.
                    DslValue::FlagNone => DslCondition::False,
                    // An admitted argument we cannot read as an `Int` is gradual, so it must fall
                    // back rather than take the false branch and produce a precise result.
                    DslValue::Unknown => DslCondition::Unknown,
                    DslValue::FlagBool(_)
                    | DslValue::FlagString(_)
                    | DslValue::FlagSequence(_)
                    | DslValue::Shape(_)
                    | DslValue::IntTuples(_)
                    | DslValue::DimensionTuple(_) => {
                        unreachable!(
                            "validated is_concrete_int operand is an Int dimension, integer Flag value, or None"
                        )
                    }
                }
            }
            TypeShapeDslConditionKind::IsIntValue { slot, .. } => match environment.value(slot) {
                DslValue::GradualInt | DslValue::FlagInt(_) => DslCondition::True,
                DslValue::FlagNone | DslValue::FlagSequence(_) => DslCondition::False,
                DslValue::Unknown => DslCondition::Unknown,
                DslValue::FlagBool(_)
                | DslValue::FlagString(_)
                | DslValue::Dimension(_)
                | DslValue::Shape(_)
                | DslValue::IntTuples(_)
                | DslValue::DimensionTuple(_) => {
                    unreachable!("validated is_int_value operand is a non-boolean Flag value")
                }
            },
            TypeShapeDslConditionKind::IsNone { slot, negated, .. } => {
                let is_none = match environment.value(slot) {
                    DslValue::FlagNone => DslCondition::True,
                    DslValue::FlagInt(_)
                    | DslValue::GradualInt
                    | DslValue::FlagBool(_)
                    | DslValue::FlagString(_)
                    | DslValue::FlagSequence(_)
                    | DslValue::Dimension(_) => DslCondition::False,
                    DslValue::Unknown => DslCondition::Unknown,
                    DslValue::Shape(_) | DslValue::IntTuples(_) | DslValue::DimensionTuple(_) => {
                        unreachable!("validated `None` identity operands cannot be shape values")
                    }
                };
                if negated {
                    Self::negate_condition(is_none)
                } else {
                    is_none
                }
            }
            TypeShapeDslConditionKind::StringEquality { negated } => {
                let Expr::Compare(compare) = condition else {
                    unreachable!("validated Flag string equality is a comparison")
                };
                let left = self.evaluate_expression(&compare.left, environment, budget);
                let right = self.evaluate_expression(&compare.comparators[0], environment, budget);
                let equality = match (left, right) {
                    (
                        DslOutcome::Value(DslValue::FlagString(left)),
                        DslOutcome::Value(DslValue::FlagString(right)),
                    ) => match left == right {
                        true => DslCondition::True,
                        false => DslCondition::False,
                    },
                    (
                        DslOutcome::Value(DslValue::FlagNone),
                        DslOutcome::Value(DslValue::FlagNone),
                    ) => DslCondition::True,
                    (
                        DslOutcome::Value(DslValue::FlagNone),
                        DslOutcome::Value(DslValue::FlagString(_)),
                    )
                    | (
                        DslOutcome::Value(DslValue::FlagString(_)),
                        DslOutcome::Value(DslValue::FlagNone),
                    ) => DslCondition::False,
                    (DslOutcome::Invalid(error), _) | (_, DslOutcome::Invalid(error)) => {
                        return Err(error);
                    }
                    (DslOutcome::Value(DslValue::Unknown), _)
                    | (_, DslOutcome::Value(DslValue::Unknown)) => DslCondition::Unknown,
                    _ => {
                        unreachable!("validated Flag string equality operands are strings or None")
                    }
                };
                if negated {
                    Self::negate_condition(equality)
                } else {
                    equality
                }
            }
            TypeShapeDslConditionKind::FlagIntCompare(op) => {
                let Expr::Compare(compare) = condition else {
                    unreachable!("validated Flag comparison is a comparison")
                };
                let left = self.evaluate_expression(&compare.left, environment, budget);
                let right = self.evaluate_expression(&compare.comparators[0], environment, budget);
                match (left, right) {
                    (
                        DslOutcome::Value(DslValue::FlagInt(left)),
                        DslOutcome::Value(DslValue::FlagInt(right)),
                    ) => {
                        if op.apply(left, right) {
                            DslCondition::True
                        } else {
                            DslCondition::False
                        }
                    }
                    (DslOutcome::Invalid(error), _) | (_, DslOutcome::Invalid(error)) => {
                        return Err(error);
                    }
                    (DslOutcome::Value(DslValue::GradualInt | DslValue::Unknown), _)
                    | (_, DslOutcome::Value(DslValue::GradualInt | DslValue::Unknown)) => {
                        DslCondition::Unknown
                    }
                    _ => unreachable!("validated Flag comparison operands are integers"),
                }
            }
            TypeShapeDslConditionKind::Membership { negated } => {
                let Expr::Compare(compare) = condition else {
                    unreachable!("validated membership condition is a comparison")
                };
                let item = self.evaluate_expression(&compare.left, environment, budget);
                let sequence =
                    self.evaluate_expression(&compare.comparators[0], environment, budget);
                match (item, sequence) {
                    (
                        DslOutcome::Value(DslValue::FlagInt(item)),
                        DslOutcome::Value(DslValue::FlagSequence(sequence)),
                    ) => {
                        let contains = sequence.contains(item);
                        if contains != negated {
                            DslCondition::True
                        } else {
                            DslCondition::False
                        }
                    }
                    (DslOutcome::Invalid(error), _) | (_, DslOutcome::Invalid(error)) => {
                        return Err(error);
                    }
                    (DslOutcome::Value(DslValue::GradualInt | DslValue::Unknown), _)
                    | (_, DslOutcome::Value(DslValue::GradualInt | DslValue::Unknown)) => {
                        DslCondition::Unknown
                    }
                    _ => unreachable!("validated membership uses an integer and Flag sequence"),
                }
            }
            TypeShapeDslConditionKind::DimensionEquality { negated } => {
                let Expr::Compare(compare) = condition else {
                    unreachable!("validated dimension comparison is a comparison")
                };
                let left = self.evaluate_expression(&compare.left, environment, budget);
                let right = self.evaluate_expression(&compare.comparators[0], environment, budget);
                match (left, right) {
                    (
                        DslOutcome::Value(DslValue::Dimension(left)),
                        DslOutcome::Value(DslValue::Dimension(right)),
                    ) => {
                        let equality = Self::dimension_equality(&left, &right);
                        if negated {
                            Self::negate_condition(equality)
                        } else {
                            equality
                        }
                    }
                    (DslOutcome::Invalid(error), _) | (_, DslOutcome::Invalid(error)) => {
                        return Err(error);
                    }
                    (DslOutcome::Value(DslValue::Unknown), _)
                    | (_, DslOutcome::Value(DslValue::Unknown)) => DslCondition::Unknown,
                    (DslOutcome::Value(DslValue::GradualInt), _)
                    | (_, DslOutcome::Value(DslValue::GradualInt)) => unreachable!(
                        "validated dimension expressions normalize gradual integers to dimensions"
                    ),
                    _ => unreachable!("validated dimension comparison operands are dimensions"),
                }
            }
            TypeShapeDslConditionKind::IntegerCompare { op, .. } => {
                let Expr::Compare(compare) = condition else {
                    unreachable!("validated integer comparison is a comparison")
                };
                let left = self.evaluate_expression(&compare.left, environment, budget);
                let right = self.evaluate_expression(&compare.comparators[0], environment, budget);
                match (left, right) {
                    (
                        DslOutcome::Value(DslValue::Dimension(Int::Literal(left))),
                        DslOutcome::Value(DslValue::Dimension(Int::Literal(right))),
                    ) => {
                        if op.apply(left, right) {
                            DslCondition::True
                        } else {
                            DslCondition::False
                        }
                    }
                    (
                        DslOutcome::Value(DslValue::Dimension(left)),
                        DslOutcome::Value(DslValue::Dimension(right)),
                    ) => Self::compare_dimensions(&left, &right, op),
                    (
                        DslOutcome::Value(DslValue::Dimension(left)),
                        DslOutcome::Value(DslValue::FlagInt(right)),
                    ) => Self::compare_dimensions(&left, &Int::Literal(right), op),
                    (
                        DslOutcome::Value(DslValue::FlagInt(left)),
                        DslOutcome::Value(DslValue::Dimension(right)),
                    ) => Self::compare_dimensions(&Int::Literal(left), &right, op),
                    (
                        DslOutcome::Value(DslValue::FlagInt(left)),
                        DslOutcome::Value(DslValue::FlagInt(right)),
                    ) => {
                        if op.apply(left, right) {
                            DslCondition::True
                        } else {
                            DslCondition::False
                        }
                    }
                    (DslOutcome::Invalid(error), _) | (_, DslOutcome::Invalid(error)) => {
                        return Err(error);
                    }
                    (DslOutcome::Value(DslValue::GradualInt | DslValue::Unknown), _)
                    | (_, DslOutcome::Value(DslValue::GradualInt | DslValue::Unknown)) => {
                        DslCondition::Unknown
                    }
                    _ => unreachable!("validated integer comparison operands are integers"),
                }
            }
            TypeShapeDslConditionKind::SlotCompare {
                left, right, op, ..
            } => {
                let equality_condition = |equal: bool| {
                    let result = match op {
                        TypeShapeDslComparisonOp::Equal => equal,
                        TypeShapeDslComparisonOp::NotEqual => !equal,
                        TypeShapeDslComparisonOp::LessThan
                        | TypeShapeDslComparisonOp::LessThanOrEqual
                        | TypeShapeDslComparisonOp::GreaterThan
                        | TypeShapeDslComparisonOp::GreaterThanOrEqual => {
                            unreachable!("validated Flag string comparisons use equality operators")
                        }
                    };
                    if result {
                        DslCondition::True
                    } else {
                        DslCondition::False
                    }
                };
                if left == right {
                    match op {
                        TypeShapeDslComparisonOp::Equal
                        | TypeShapeDslComparisonOp::LessThanOrEqual
                        | TypeShapeDslComparisonOp::GreaterThanOrEqual => DslCondition::True,
                        TypeShapeDslComparisonOp::NotEqual
                        | TypeShapeDslComparisonOp::LessThan
                        | TypeShapeDslComparisonOp::GreaterThan => DslCondition::False,
                    }
                } else {
                    match (environment.value(left), environment.value(right)) {
                        (DslValue::Dimension(left), DslValue::Dimension(right)) => {
                            Self::compare_dimensions(left, right, op)
                        }
                        (DslValue::FlagInt(left), DslValue::FlagInt(right)) => {
                            if op.apply(*left, *right) {
                                DslCondition::True
                            } else {
                                DslCondition::False
                            }
                        }
                        (DslValue::Dimension(left), DslValue::FlagInt(right)) => {
                            Self::compare_dimensions(left, &Int::Literal(*right), op)
                        }
                        (DslValue::FlagInt(left), DslValue::Dimension(right)) => {
                            Self::compare_dimensions(&Int::Literal(*left), right, op)
                        }
                        (DslValue::FlagString(left), DslValue::FlagString(right)) => {
                            equality_condition(left == right)
                        }
                        (DslValue::FlagNone, DslValue::FlagNone) => equality_condition(true),
                        (DslValue::FlagNone, DslValue::FlagString(_))
                        | (DslValue::FlagString(_), DslValue::FlagNone) => {
                            equality_condition(false)
                        }
                        (DslValue::GradualInt | DslValue::Unknown, _)
                        | (_, DslValue::GradualInt | DslValue::Unknown) => DslCondition::Unknown,
                        _ => unreachable!(
                            "validated slot comparison operands share the same value domain"
                        ),
                    }
                }
            }
            TypeShapeDslConditionKind::GeneratorElementSelfCompare(op) => match op {
                TypeShapeDslComparisonOp::Equal
                | TypeShapeDslComparisonOp::LessThanOrEqual
                | TypeShapeDslComparisonOp::GreaterThanOrEqual => DslCondition::True,
                TypeShapeDslComparisonOp::NotEqual
                | TypeShapeDslComparisonOp::LessThan
                | TypeShapeDslComparisonOp::GreaterThan => DslCondition::False,
            },
            TypeShapeDslConditionKind::LengthEqualLiteral { slot, literal } => {
                if literal < 0 {
                    return Ok(DslCondition::False);
                }
                match environment.value(slot) {
                    DslValue::Shape(shape) => match shape.view() {
                        IntTupleView::Concrete(shape) => {
                            if i64::try_from(shape.len())
                                .expect("concrete IntTuple length must fit in a control integer")
                                == literal
                            {
                                DslCondition::True
                            } else {
                                DslCondition::False
                            }
                        }
                        IntTupleView::Unpacked { prefix, suffix, .. } => {
                            let minimum_length = i64::try_from(prefix.len() + suffix.len()).expect(
                                "fixed IntTuple prefix and suffix length must fit in a control integer",
                            );
                            if literal < minimum_length {
                                DslCondition::False
                            } else {
                                DslCondition::Unknown
                            }
                        }
                        IntTupleView::Gradual => DslCondition::Unknown,
                    },
                    DslValue::FlagSequence(sequence) => match sequence.len() {
                        Some(length) if length == literal => DslCondition::True,
                        Some(_) => DslCondition::False,
                        None => DslCondition::Unknown,
                    },
                    DslValue::IntTuples(DslIntTuples::Fixed(shapes)) => {
                        if i64::try_from(shapes.len()).ok() == Some(literal) {
                            DslCondition::True
                        } else {
                            DslCondition::False
                        }
                    }
                    DslValue::IntTuples(DslIntTuples::Unbounded(_)) => DslCondition::Unknown,
                    DslValue::Unknown => DslCondition::Unknown,
                    _ => {
                        unreachable!(
                            "validated length equality evaluates an IntTuple, IntTuples, or Flag sequence"
                        )
                    }
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

#[derive(Clone, Copy)]
enum IntTupleSlicePosition {
    Prefix(usize),
    Suffix(usize),
}

/// Slice an immutable shape while retaining an unpacked symbolic middle when both bounds can be
/// located in its fixed prefix or suffix. Bounds that depend on the unknown middle return `None`.
fn evaluate_int_tuple_slice(
    shape: &IntTuple,
    start: Option<i64>,
    stop: Option<i64>,
) -> Option<IntTuple> {
    if stop == Some(0)
        || stop.is_some_and(|stop| {
            let start = start.unwrap_or(0);
            // Equal or reversed bounds with the same sign are empty independently of tuple length.
            ((start >= 0 && stop >= 0) || (start < 0 && stop < 0)) && start >= stop
        })
    {
        return Some(IntTuple::new(Vec::new()));
    }
    match shape.view() {
        IntTupleView::Concrete(dimensions) => {
            let length =
                i128::try_from(dimensions.len()).expect("concrete tuple length must fit in i128");
            let normalize = |bound: Option<i64>, default: i128| {
                let bound = bound.map_or(default, i128::from);
                let normalized = if bound < 0 {
                    length
                        .checked_add(bound)
                        .expect("negative i64 slice bound plus tuple length fits in i128")
                        .max(0)
                } else {
                    bound.min(length)
                };
                usize::try_from(normalized)
                    .expect("clamped concrete tuple slice bound must fit in usize")
            };
            let start = normalize(start, 0);
            let stop = normalize(stop, length);
            Some(IntTuple::new(dimensions[start..stop.max(start)].to_vec()))
        }
        IntTupleView::Gradual => None,
        IntTupleView::Unpacked {
            prefix,
            middle,
            suffix,
        } => {
            let locate = |bound: i64| {
                if bound >= 0 {
                    usize::try_from(bound)
                        .ok()
                        .filter(|bound| *bound <= prefix.len())
                        .map(IntTupleSlicePosition::Prefix)
                } else {
                    usize::try_from(bound.unsigned_abs())
                        .ok()
                        .filter(|removed| *removed <= suffix.len())
                        .map(|removed| IntTupleSlicePosition::Suffix(suffix.len() - removed))
                }
            };
            let start = start.map_or(Some(IntTupleSlicePosition::Prefix(0)), locate)?;
            let stop = stop.map_or(Some(IntTupleSlicePosition::Suffix(suffix.len())), locate)?;
            match (start, stop) {
                (IntTupleSlicePosition::Prefix(start), IntTupleSlicePosition::Prefix(stop)) => {
                    Some(IntTuple::new(prefix[start..stop.max(start)].to_vec()))
                }
                (IntTupleSlicePosition::Suffix(start), IntTupleSlicePosition::Suffix(stop)) => {
                    Some(IntTuple::new(suffix[start..stop.max(start)].to_vec()))
                }
                (IntTupleSlicePosition::Prefix(start), IntTupleSlicePosition::Suffix(stop)) => {
                    Some(IntTuple::unpacked(
                        prefix[start..].to_vec(),
                        middle.clone(),
                        suffix[..stop].to_vec(),
                    ))
                }
                (IntTupleSlicePosition::Suffix(_), IntTupleSlicePosition::Prefix(_)) => {
                    // A suffix position cannot precede a prefix position, regardless of middle length.
                    Some(IntTuple::new(Vec::new()))
                }
            }
        }
    }
}

fn lower_parameter(ty: &Type, domain: TypeShapeDslInputDomain) -> DslValue {
    match domain {
        TypeShapeDslInputDomain::Value(domain) => DslValue::from_type(ty, domain),
        TypeShapeDslInputDomain::OptionalInt => match ty {
            Type::None => DslValue::FlagNone,
            _ => DslValue::from_type(ty, TypeShapeDslDomain::Int),
        },
        TypeShapeDslInputDomain::Flag(domain) => {
            // Tuple expressions used as Flag defaults are represented as `Type::Type`; evaluation
            // consumes the value described by the expression rather than the class object wrapper.
            let ty = match ty {
                Type::Type(inner) => inner.as_ref(),
                _ => ty,
            };
            if !domain.accepts(ty) {
                return DslValue::Unknown;
            }
            match ty {
                Type::None => DslValue::FlagNone,
                Type::ClassType(cls) if cls.is_builtin("int") => DslValue::GradualInt,
                Type::Int(Int::Literal(value)) => DslValue::FlagInt(*value),
                // Symbolic shape integers satisfy `Flag[int]`, but DSL flag operations inspect
                // only concrete runtime values. Generic substitution does not re-evaluate a call
                // that already fell back.
                Type::Int(_) => DslValue::Unknown,
                Type::Literal(literal) => match &literal.value {
                    Lit::Int(value) => value.as_i64().map_or(DslValue::Unknown, DslValue::FlagInt),
                    Lit::Bool(value) => DslValue::FlagBool(*value),
                    Lit::Str(value) => DslValue::FlagString(value.clone()),
                    _ => DslValue::Unknown,
                },
                Type::Tuple(Tuple::Concrete(elements)) => {
                    let values = elements
                        .iter()
                        .map(|element| match element {
                            Type::Literal(literal) => match &literal.value {
                                Lit::Int(value) => value.as_i64(),
                                _ => None,
                            },
                            _ => None,
                        })
                        .collect::<Option<Vec<_>>>();
                    values.map_or(DslValue::Unknown, |values| {
                        DslValue::FlagSequence(DslFlagSequence::Values(values))
                    })
                }
                // Nonliteral Flags are gradual DSL inputs and intentionally propagate to the
                // annotated result fallback.
                _ => DslValue::Unknown,
            }
        }
    }
}

fn evaluate_dimension_arithmetic(
    left: DslOutcome,
    op: TypeShapeDslArithmeticOp,
    right: DslOutcome,
) -> DslOutcome {
    fn operand(outcome: DslOutcome) -> Option<Int> {
        match outcome {
            DslOutcome::Value(DslValue::Dimension(value)) => Some(value),
            DslOutcome::Value(DslValue::GradualInt) => Some(Int::Int),
            DslOutcome::Value(DslValue::Unknown) => None,
            DslOutcome::Invalid(_) => unreachable!(
                "invalid dimension arithmetic operands are propagated before evaluation"
            ),
            DslOutcome::ExplicitGradual => {
                unreachable!("dimension arithmetic operands cannot be explicit gradual returns")
            }
            DslOutcome::Value(_) => {
                unreachable!("validated dimension arithmetic operands are integer values")
            }
        }
    }

    let right = operand(right);
    if let Some(Int::Literal(0)) = right
        && matches!(
            op,
            TypeShapeDslArithmeticOp::FloorDivide | TypeShapeDslArithmeticOp::Modulo
        )
    {
        let operation = if op == TypeShapeDslArithmeticOp::FloorDivide {
            "division"
        } else {
            "modulo"
        };
        return DslOutcome::Invalid(ShapeError::ShapeComputation {
            message: format!("dimension integer {operation} by zero"),
        });
    }
    let left = operand(left);
    // A broad integer becomes `Int::Int` at a dimension boundary. `Unknown` instead means that
    // evaluation could not establish an integer value, so arithmetic must retain that distinction.
    let (Some(left), Some(right)) = (left, right) else {
        return DslOutcome::Value(DslValue::Unknown);
    };
    if let (Int::Literal(left), Int::Literal(right)) = (&left, &right) {
        return evaluate_i64_arithmetic(*left, op, *right).map_or_else(
            |operation| {
                DslOutcome::Invalid(ShapeError::ShapeComputation {
                    message: format!("dimension integer {operation} by zero"),
                })
            },
            |value| {
                value.map_or(DslOutcome::Value(DslValue::Dimension(Int::Int)), |value| {
                    DslOutcome::Value(DslValue::Dimension(Int::Literal(value)))
                })
            },
        );
    }
    if matches!(op, TypeShapeDslArithmeticOp::Modulo) {
        return DslOutcome::Value(DslValue::Dimension(Int::Int));
    }
    let result = match op {
        TypeShapeDslArithmeticOp::Add => Int::add(Type::Int(left), Type::Int(right)),
        TypeShapeDslArithmeticOp::Subtract => Int::sub(Type::Int(left), Type::Int(right)),
        TypeShapeDslArithmeticOp::Multiply => Int::mul(Type::Int(left), Type::Int(right)),
        TypeShapeDslArithmeticOp::FloorDivide => Int::floor_div(Type::Int(left), Type::Int(right)),
        TypeShapeDslArithmeticOp::Modulo => unreachable!("symbolic modulo is gradual"),
    };
    match canonicalize(Type::Int(result)) {
        Type::Int(result) => DslOutcome::Value(DslValue::Dimension(result)),
        _ => unreachable!("canonicalized dimension arithmetic must remain an Int"),
    }
}

fn evaluate_flag_int_arithmetic(left: i64, op: TypeShapeDslArithmeticOp, right: i64) -> DslOutcome {
    evaluate_i64_arithmetic(left, op, right).map_or_else(
        |operation| {
            DslOutcome::Invalid(ShapeError::ShapeComputation {
                message: format!("Flag integer {operation} by zero"),
            })
        },
        |value| {
            value.map_or(DslOutcome::Value(DslValue::Unknown), |value| {
                DslOutcome::Value(DslValue::FlagInt(value))
            })
        },
    )
}

fn evaluate_i64_arithmetic(
    left: i64,
    op: TypeShapeDslArithmeticOp,
    right: i64,
) -> Result<Option<i64>, &'static str> {
    let result = match op {
        TypeShapeDslArithmeticOp::Add => left.checked_add(right),
        TypeShapeDslArithmeticOp::Subtract => left.checked_sub(right),
        TypeShapeDslArithmeticOp::Multiply => left.checked_mul(right),
        TypeShapeDslArithmeticOp::FloorDivide => {
            if right == 0 {
                return Err("division");
            }
            // Python's `i64::MIN // -1` result is outside the DSL's `Flag[int]` domain,
            // so checked overflow intentionally becomes an automatic unknown.
            left.checked_div(right).and_then(|quotient| {
                let remainder = left.checked_rem(right)?;
                quotient.checked_sub(i64::from(remainder != 0 && (left < 0) != (right < 0)))
            })
        }
        TypeShapeDslArithmeticOp::Modulo => {
            if right == 0 {
                return Err("modulo");
            }
            if right == -1 {
                // Unlike division, modulo's result is representable for every i64 dividend.
                return Ok(Some(0));
            }
            left.checked_rem(right).and_then(|remainder| {
                if remainder != 0 && (remainder < 0) != (right < 0) {
                    remainder.checked_add(right)
                } else {
                    Some(remainder)
                }
            })
        }
    };
    Ok(result)
}

impl DslFlagSequence {
    fn bounded_values(&self, item_limit: usize) -> Option<(Vec<i64>, bool)> {
        let length = usize::try_from(self.len()?).ok()?;
        let take = length.min(item_limit);
        let values = match self {
            Self::Values(values) => values.iter().take(take).copied().collect(),
            Self::Range { start, step, .. } => (0..take)
                .map(|index| {
                    i128::from(*start)
                        .checked_add(i128::try_from(index).ok()?.checked_mul(i128::from(*step))?)
                        .and_then(|value| i64::try_from(value).ok())
                })
                .collect::<Option<Vec<_>>>()?,
        };
        Some((values, length > item_limit))
    }

    fn contains(&self, value: i64) -> bool {
        match self {
            Self::Values(values) => values.contains(&value),
            Self::Range { start, stop, step } => {
                let in_bounds = if *step > 0 {
                    *start <= value && value < *stop
                } else {
                    *stop < value && value <= *start
                };
                in_bounds && (i128::from(value) - i128::from(*start)) % i128::from(*step) == 0
            }
        }
    }

    fn count(&self, value: i64) -> Option<i64> {
        match self {
            Self::Values(values) => i64::try_from(
                values
                    .iter()
                    .filter(|candidate| **candidate == value)
                    .count(),
            )
            .ok(),
            Self::Range { .. } => Some(i64::from(self.contains(value))),
        }
    }

    /// Returns the position of the first occurrence of a value. Range membership makes offset
    /// division exact; positions outside the Flag integer domain yield `Gradual`.
    fn index(&self, value: i64) -> DslFlagSequenceIndex {
        match self {
            Self::Values(values) => match values.iter().position(|candidate| *candidate == value) {
                Some(index) => DslFlagSequenceIndex::Exact(
                    i64::try_from(index).expect("materialized sequence index fits in i64"),
                ),
                None => DslFlagSequenceIndex::NotFound,
            },
            Self::Range { start, step, .. } => {
                if !self.contains(value) {
                    return DslFlagSequenceIndex::NotFound;
                }
                i64::try_from((i128::from(value) - i128::from(*start)) / i128::from(*step))
                    .map_or(DslFlagSequenceIndex::Gradual, DslFlagSequenceIndex::Exact)
            }
        }
    }

    fn len(&self) -> Option<i64> {
        match self {
            Self::Values(values) => i64::try_from(values.len()).ok(),
            Self::Range { start, stop, step } => {
                let start = i128::from(*start);
                let stop = i128::from(*stop);
                let step = i128::from(*step);
                let length = if step > 0 {
                    if start >= stop {
                        0
                    } else {
                        (stop - start - 1) / step + 1
                    }
                } else if start <= stop {
                    0
                } else {
                    (start - stop - 1) / -step + 1
                };
                i64::try_from(length).ok()
            }
        }
    }
}

impl DslValue {
    fn from_type(ty: &Type, domain: TypeShapeDslDomain) -> Self {
        match domain {
            TypeShapeDslDomain::Int => match Int::from_type(ty) {
                Some(dimension) => match canonicalize(Type::Int(dimension)) {
                    Type::Int(dimension) => Self::Dimension(dimension),
                    canonical => unreachable!("canonicalizing a dimension yields `{canonical}`"),
                },
                None => Self::Unknown,
            },
            TypeShapeDslDomain::IntTuple => Self::from_shape_type(ty),
            TypeShapeDslDomain::IntTuples => match ty {
                Type::Tuple(Tuple::Concrete(elements)) => elements
                    .iter()
                    .map(IntTuple::from_shape_arg_or_tuple_carrier)
                    .collect::<Option<Vec<_>>>()
                    .map_or(Self::Unknown, |shapes| {
                        Self::IntTuples(DslIntTuples::Fixed(shapes))
                    }),
                Type::Tuple(Tuple::Unbounded(element)) => {
                    IntTuple::from_shape_arg_or_tuple_carrier(element)
                        .map_or(Self::Unknown, |shape| {
                            Self::IntTuples(DslIntTuples::Unbounded(shape))
                        })
                }
                _ => Self::Unknown,
            },
        }
    }

    fn from_shape_type(ty: &Type) -> Self {
        IntTuple::from_shape_arg_or_tuple_carrier(ty).map_or(Self::Unknown, Self::Shape)
    }

    fn into_type(self) -> Type {
        match self {
            Self::Dimension(value) => Type::Int(value),
            Self::GradualInt => Type::Int(Int::Int),
            Self::Shape(value) => value.to_shape_arg_type(),
            Self::IntTuples(DslIntTuples::Fixed(shapes)) => Type::Tuple(Tuple::Concrete(
                shapes
                    .into_iter()
                    .map(|shape| shape.to_shape_arg_type())
                    .collect(),
            )),
            Self::IntTuples(DslIntTuples::Unbounded(shape)) => {
                Type::Tuple(Tuple::Unbounded(Box::new(shape.to_shape_arg_type())))
            }
            Self::Unknown => unreachable!("unknown DSL values project through the fallback"),
            Self::FlagInt(_)
            | Self::FlagBool(_)
            | Self::FlagString(_)
            | Self::FlagNone
            | Self::FlagSequence(_)
            | Self::DimensionTuple(_) => {
                unreachable!("intermediate DSL values cannot be returned directly")
            }
        }
    }
}

impl DslEnvironment {
    fn value(&self, slot: usize) -> &DslValue {
        &self.slots[slot]
    }

    fn assign(&mut self, slot: usize, value: DslValue) {
        assert!(
            slot >= self.parameter_count,
            "validated assignment cannot target a parameter slot"
        );
        self.slots[slot] = value;
    }
}

fn common_int_tuple_member(shapes: &[IntTuple]) -> IntTuple {
    let Some(first) = shapes.first() else {
        return IntTuple::shapeless();
    };
    if shapes.iter().all(|shape| shape == first) {
        first.clone()
    } else {
        IntTuple::shapeless()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Var;

    #[test]
    fn optional_int_lowering_preserves_dimension_values() {
        for dimension in [
            Int::Literal(0),
            Int::Literal(3),
            Int::Int,
            Int::Symbolic(Box::new(Type::Var(Var::ZERO))),
        ] {
            assert!(matches!(
                lower_parameter(
                    &Type::Int(dimension.clone()),
                    TypeShapeDslInputDomain::OptionalInt,
                ),
                DslValue::Dimension(actual) if actual == dimension
            ));
        }
        assert!(matches!(
            lower_parameter(&Type::None, TypeShapeDslInputDomain::OptionalInt),
            DslValue::FlagNone
        ));
        assert!(matches!(
            lower_parameter(&Type::any_error(), TypeShapeDslInputDomain::OptionalInt),
            DslValue::Unknown
        ));
    }

    #[test]
    fn optional_int_requires_non_none_narrowing_before_int_use() {
        let optional = TypeShapeDslInputDomain::OptionalInt;
        assert!(optional.can_use_as(
            TypeShapeDslDomain::Int,
            TypeShapeDslParameterNarrowing::NonNone
        ));
        assert!(!optional.can_use_as(
            TypeShapeDslDomain::Int,
            TypeShapeDslParameterNarrowing::Unnarrowed
        ));
        assert!(!optional.can_use_as(
            TypeShapeDslDomain::IntTuple,
            TypeShapeDslParameterNarrowing::NonNone
        ));
        assert!(optional.can_use_as(
            TypeShapeDslDomain::Int,
            TypeShapeDslParameterNarrowing::Integer
        ));
        let int = TypeShapeDslInputDomain::Value(TypeShapeDslDomain::Int);
        for narrowing in [
            TypeShapeDslParameterNarrowing::Unnarrowed,
            TypeShapeDslParameterNarrowing::NonNone,
            TypeShapeDslParameterNarrowing::Integer,
        ] {
            assert!(int.can_use_as(TypeShapeDslDomain::Int, narrowing));
            assert!(!int.can_use_as(TypeShapeDslDomain::IntTuple, narrowing));
        }
    }

    #[test]
    fn helper_argument_domain_compatibility() {
        let int = TypeShapeDslInputDomain::Value(TypeShapeDslDomain::Int);
        let shape = TypeShapeDslInputDomain::Value(TypeShapeDslDomain::IntTuple);
        let optional_int = TypeShapeDslInputDomain::OptionalInt;
        let flag_int = TypeShapeDslInputDomain::Flag(FlagDomain::of(FlagMember::Int));
        let flag_optional_int = TypeShapeDslInputDomain::Flag(
            FlagDomain::of(FlagMember::Int).join(FlagDomain::of(FlagMember::NoneType)),
        );

        assert!(int.can_forward_to(int));
        assert!(int.can_forward_to(optional_int));
        assert!(optional_int.can_forward_to(optional_int));
        assert!(flag_int.can_forward_to(flag_optional_int));
        assert!(!optional_int.can_forward_to(int));
        assert!(!flag_optional_int.can_forward_to(flag_int));
        assert!(!flag_int.can_forward_to(int));
        assert!(!int.can_forward_to(flag_int));
        assert!(!shape.can_forward_to(optional_int));
    }

    #[test]
    fn deferred_integer_dimension_selection_rejects_non_integer_sources() {
        let helper_call = TypeShapeDslHelperCall {
            callee: Expr::Name(ExprName {
                node_index: ruff_python_ast::AtomicNodeIndex::default(),
                range: TextRange::default(),
                id: Name::new("helper"),
                ctx: ruff_python_ast::ExprContext::Load,
            }),
            arguments: vec![TypeShapeDslHelperArgument {
                slot: 0,
                source: TypeShapeDslHelperArgumentSource::DeferredInteger {
                    index: 0,
                    parameter_uses: vec![TypeShapeDslParameterUse {
                        parameter: 0,
                        narrowing: TypeShapeDslParameterNarrowing::Unnarrowed,
                    }]
                    .into_boxed_slice(),
                    resolved_domain: None,
                },
            }],
        };
        let int = TypeShapeDslInputDomain::Value(TypeShapeDslDomain::Int);
        let optional_int = TypeShapeDslInputDomain::OptionalInt;
        let broad_flag = TypeShapeDslInputDomain::Flag(
            FlagDomain::of(FlagMember::Int)
                .join(FlagDomain::of(FlagMember::IntTuple))
                .join(FlagDomain::of(FlagMember::NoneType)),
        );

        for actual in [optional_int, broad_flag] {
            assert_eq!(
                helper_call.argument_domains(&[actual], &[int], &mut HashMap::new()),
                Err(TypeShapeDslHelperArgumentError::IncompatibleDomain {
                    argument: 0,
                    actual,
                    expected: int,
                })
            );
        }
    }

    #[test]
    fn flag_sequence_index_finds_first_concrete_value() {
        let values = DslFlagSequence::Values(vec![3, 7, 3]);
        let singleton = DslFlagSequence::Values(vec![11]);
        assert_eq!(values.index(3), DslFlagSequenceIndex::Exact(0));
        assert_eq!(values.index(7), DslFlagSequenceIndex::Exact(1));
        assert_eq!(values.index(9), DslFlagSequenceIndex::NotFound);
        assert_eq!(singleton.index(11), DslFlagSequenceIndex::Exact(0));
    }

    #[test]
    fn flag_sequence_index_uses_range_arithmetic() {
        let positive = DslFlagSequence::Range {
            start: 0,
            stop: 10_000,
            step: 2,
        };
        let negative = DslFlagSequence::Range {
            start: 9,
            stop: -10,
            step: -3,
        };
        let singleton = DslFlagSequence::Range {
            start: 5,
            stop: 6,
            step: 1,
        };
        let empty = DslFlagSequence::Range {
            start: 5,
            stop: 5,
            step: 1,
        };
        assert_eq!(positive.index(9_998), DslFlagSequenceIndex::Exact(4_999));
        assert_eq!(negative.index(0), DslFlagSequenceIndex::Exact(3));
        assert_eq!(negative.index(1), DslFlagSequenceIndex::NotFound);
        assert_eq!(negative.index(-10), DslFlagSequenceIndex::NotFound);
        assert_eq!(singleton.index(5), DslFlagSequenceIndex::Exact(0));
        assert_eq!(empty.index(5), DslFlagSequenceIndex::NotFound);
    }

    #[test]
    fn flag_sequence_index_searches_materialized_values_without_a_budget() {
        let mut items = vec![0; MAX_GENERATOR_STEPS + 1];
        items[MAX_GENERATOR_STEPS] = 5;
        let values = DslFlagSequence::Values(items);
        assert_eq!(values.index(0), DslFlagSequenceIndex::Exact(0));
        assert_eq!(
            values.index(5),
            DslFlagSequenceIndex::Exact(
                i64::try_from(MAX_GENERATOR_STEPS).expect("test bound fits in i64")
            )
        );
        assert_eq!(values.index(9), DslFlagSequenceIndex::NotFound);
    }

    #[test]
    fn flag_sequence_index_is_gradual_when_range_offset_overflows() {
        let oversized_index = DslFlagSequence::Range {
            start: i64::MIN,
            stop: i64::MAX,
            step: 1,
        };
        let negative_endpoints = DslFlagSequence::Range {
            start: i64::MAX,
            stop: i64::MIN,
            step: i64::MIN,
        };
        assert_eq!(
            oversized_index.index(i64::MIN),
            DslFlagSequenceIndex::Exact(0)
        );
        assert_eq!(
            oversized_index.index(i64::MAX),
            DslFlagSequenceIndex::NotFound
        );
        assert_eq!(
            oversized_index.index(i64::MAX - 1),
            DslFlagSequenceIndex::Gradual
        );
        assert_eq!(
            negative_endpoints.index(i64::MAX),
            DslFlagSequenceIndex::Exact(0)
        );
        assert_eq!(negative_endpoints.index(-1), DslFlagSequenceIndex::Exact(1));
        assert_eq!(
            negative_endpoints.index(i64::MIN),
            DslFlagSequenceIndex::NotFound
        );
    }

    #[test]
    fn lower_bool_literal_for_union_flag_domain() {
        let domain = FlagDomain::of(FlagMember::Bool).join(FlagDomain::of(FlagMember::NoneType));

        assert!(matches!(
            lower_parameter(
                &Lit::Bool(true).to_implicit_type(),
                TypeShapeDslInputDomain::Flag(domain),
            ),
            DslValue::FlagBool(true)
        ));
    }
}
