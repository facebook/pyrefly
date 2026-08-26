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
use crate::function::FuncDefId;
use crate::literal::Lit;
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
/// The type-system domain accepted by one DSL parameter.
///
/// `Value` represents a shape dimension or shape tuple. `Flag` represents literal-preserving
/// configuration values supplied by ordinary Python calls.
pub enum TypeShapeDslInputDomain {
    Value(TypeShapeDslDomain),
    Flag(FlagDomain),
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
enum TypeShapeDslHelperArgumentProvenance {
    Parameters(Box<[usize]>),
    ParametersWithRequiredDomain {
        parameters: Box<[usize]>,
        domain: TypeShapeDslInputDomain,
    },
    Exact(TypeShapeDslInputDomain),
    DeferredInteger {
        index: usize,
        parameters: Box<[usize]>,
        resolved_domain: Option<TypeShapeDslInputDomain>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct TypeShapeDslHelperArgument {
    slot: usize,
    provenance: TypeShapeDslHelperArgumentProvenance,
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
    ) -> Option<Vec<TypeShapeDslInputDomain>> {
        if self.arguments.len() != expected_domains.len() {
            return None;
        }
        self.arguments
            .iter()
            .zip(expected_domains)
            .map(|(argument, expected)| match &argument.provenance {
                TypeShapeDslHelperArgumentProvenance::Exact(domain) => {
                    (domain == expected).then_some(*domain)
                }
                TypeShapeDslHelperArgumentProvenance::ParametersWithRequiredDomain {
                    parameters,
                    domain,
                } => parameters
                    .iter()
                    .all(|parameter| caller_domains[*parameter] == *domain)
                    .then_some(*domain)
                    .filter(|domain| domain == expected),
                TypeShapeDslHelperArgumentProvenance::Parameters(parameters) => {
                    let mut domains = parameters
                        .iter()
                        .map(|parameter| caller_domains[*parameter]);
                    let first = domains
                        .next()
                        .expect("validated helper argument provenance is nonempty");
                    (domains.all(|domain| domain == first) && first == *expected).then_some(first)
                }
                TypeShapeDslHelperArgumentProvenance::DeferredInteger {
                    index,
                    parameters,
                    resolved_domain,
                } => {
                    let accepts_expected = match expected {
                        TypeShapeDslInputDomain::Value(TypeShapeDslDomain::Int) => true,
                        TypeShapeDslInputDomain::Flag(domain)
                            if *domain == FlagDomain::of(FlagMember::Int) =>
                        {
                            parameters.iter().all(|parameter| {
                                caller_domains[*parameter]
                                    == TypeShapeDslInputDomain::Flag(FlagDomain::of(
                                        FlagMember::Int,
                                    ))
                            })
                        }
                        _ => false,
                    };
                    if !accepts_expected
                        || resolved_domain.is_some_and(|domain| domain != *expected)
                    {
                        return None;
                    }
                    let previous = deferred_domains.entry(*index).or_insert(*expected);
                    (*previous == *expected).then_some(*expected)
                }
            })
            .collect()
    }
}

impl fmt::Display for TypeShapeDslInputDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Value(domain) => f.write_str(domain.as_str()),
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
    definition: Arc<ValidatedTypeShapeDslFunction>,
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
        definition: Arc<ValidatedTypeShapeDslFunction>,
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

/// The validated source of a type-level shape DSL function's return value. Resolving this depends
/// on more than the AST, so it participates in `ValidatedTypeShapeDslFunction` identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TypeShapeDslReturnKind {
    /// Return the parameter at the given zero-based position.
    Parameter(usize),
    Local {
        slot: usize,
        domain: TypeShapeDslDomain,
        parameter_origins: Option<Box<[usize]>>,
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
    /// Evaluate a validated `IntTuple` expression from the retained AST.
    IntTupleExpression,
    /// Evaluate a validated `IntTuple` product from the retained AST.
    IntTupleProduct,
    /// Evaluate another validated expression in the declared result domain.
    Expression(TypeShapeDslDomain),
    /// Return an invalid shape computation with a source-provided message.
    Invalid,
    /// Return the gradual value for the function's declared result domain.
    Gradual(TypeShapeDslDomain),
    /// Evaluate a statically resolved user-defined DSL helper.
    HelperCall(usize),
}

/// The arithmetic a validated dimension or Flag expression applies. Reached through
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

/// The comparison a validated Flag condition applies. `CmpOp` has no total order, so the DSL
/// records its own closed operator set, which also keeps the evaluator's match exhaustive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TypeShapeDslFlagIntComparisonOp {
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

impl TypeShapeDslFlagIntComparisonOp {
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
    Gradual(TypeShapeDslDomain),
    IsConcreteInt,
    IsIntValue,
    IntTuple,
    Prod,
    Invalid,
    Len,
    Range,
    Tuple,
}

/// What a validated DSL value expression computes. Like `TypeShapeDslReturnKind` this depends on
/// intrinsic resolution, so it participates in `ValidatedTypeShapeDslFunction` identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TypeShapeDslExpressionKind {
    IntTupleSlot {
        slot: usize,
        parameter_origins: Option<Box<[usize]>>,
    },
    IntTupleSlice {
        stop: i64,
    },
    IntTupleConcat,
    DimensionSlot {
        slot: usize,
        parameter_origins: Option<Box<[usize]>>,
    },
    IntegerSlot {
        slot: usize,
        parameter_origins: Option<Box<[usize]>>,
        narrowed: bool,
    },
    DimensionLiteral(Option<i64>),
    IntTupleIndex {
        shape: usize,
        parameter_origins: Option<Box<[usize]>>,
    },
    DimensionTuple,
    IntTupleConstructor,
    IntTupleProduct,
    IntTupleLength {
        shape: usize,
        parameter_origins: Option<Box<[usize]>>,
    },
    GeneratorSourceSlot {
        slot: usize,
        parameter_origins: Option<Box<[usize]>>,
        narrowed: bool,
    },
    GeneratorElementAsDimension(usize),
    GeneratorElementAsFlagInt(usize),
    Slot(usize),
    FlagValueSlot {
        slot: usize,
        parameter_origins: Option<Box<[usize]>>,
        required: TypeShapeDslFlagValueKind,
        narrowed: bool,
    },
    FlagIntLiteral(Option<i64>),
    FlagBool(bool),
    FlagNone,
    FlagTuple,
    FlagRange,
    FlagSequenceLength,
    FlagSequenceCount,
    FlagIntArithmetic(TypeShapeDslArithmeticOp),
    DimensionArithmetic(TypeShapeDslArithmeticOp),
    Conditional,
    DimensionGenerator {
        binder: usize,
    },
    FlagGenerator {
        binder: usize,
    },
}

/// The Flag value domain a validated operation requires of its operand. Reached through
/// `TypeShapeDslExpressionKind`, so it shares that type's identity requirements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TypeShapeDslFlagValueKind {
    Int,
    Sequence,
}

/// What a validated DSL condition tests. Like `TypeShapeDslReturnKind` this depends on intrinsic
/// resolution, so it participates in `ValidatedTypeShapeDslFunction` identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TypeShapeDslConditionKind {
    Any {
        binder: usize,
    },
    SlotCompare {
        left: usize,
        right: usize,
        left_parameters: Option<Box<[usize]>>,
        right_parameters: Option<Box<[usize]>>,
        op: TypeShapeDslFlagIntComparisonOp,
    },
    DimensionEquality {
        negated: bool,
    },
    GeneratorElementSelfCompare(TypeShapeDslFlagIntComparisonOp),
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
    },
    FlagIntCompare(TypeShapeDslFlagIntComparisonOp),
    BoolSlot {
        slot: usize,
        parameter_origins: Option<Box<[usize]>>,
    },
    Membership {
        negated: bool,
    },
    LengthEqualLiteral {
        slot: usize,
        literal: i64,
    },
}

const FLAG_INT: u8 = 1;
const FLAG_SEQUENCE: u8 = 2;
const FLAG_NONE: u8 = 4;
const FLAG_BOOL: u8 = 8;
// Control-flow narrowing can distinguish these non-boolean Flag values.
const FLAG_NARROWABLE: u8 = FLAG_INT | FLAG_SEQUENCE | FLAG_NONE;
// Every Flag value the validator and evaluator can represent.
const FLAG_REPRESENTABLE: u8 = FLAG_NARROWABLE | FLAG_BOOL;
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GeneratorValidationKind {
    Condition,
    Dimension,
    FlagValue,
}

fn flag_domain_from_kinds(kinds: u8) -> Option<FlagDomain> {
    if kinds == 0 || kinds & !FLAG_REPRESENTABLE != 0 {
        return None;
    }
    let mut members = [
        (FLAG_INT, FlagMember::Int),
        (FLAG_SEQUENCE, FlagMember::Tuple),
        (FLAG_NONE, FlagMember::NoneType),
        (FLAG_BOOL, FlagMember::Bool),
    ]
    .into_iter()
    .filter_map(|(bit, member)| (kinds & bit != 0).then_some(member));
    let first = FlagDomain::of(members.next()?);
    Some(members.fold(first, |domain, member| domain.join(FlagDomain::of(member))))
}

/// Parameter provenance for a locally-computed Flag value.
///
/// `Inferred` means syntax suggests a Flag domain, but each parameter must still declare that
/// exact domain. `Narrowed` means control flow proved the domain for parameters whose declarations
/// may be broader.
#[derive(Debug, Clone, PartialEq, Eq)]
enum DslFlagOrigins {
    Inferred(Box<[usize]>),
    Narrowed(Box<[usize]>),
}

impl DslFlagOrigins {
    fn parameters(&self) -> &[usize] {
        match self {
            Self::Inferred(parameters) | Self::Narrowed(parameters) => parameters,
        }
    }

    fn into_parameters(self) -> Box<[usize]> {
        match self {
            Self::Inferred(parameters) | Self::Narrowed(parameters) => parameters,
        }
    }

    fn clone_parameters_with_narrowing(&self) -> (Box<[usize]>, bool) {
        match self {
            Self::Inferred(parameters) => (parameters.clone(), false),
            Self::Narrowed(parameters) => (parameters.clone(), true),
        }
    }

    /// Combines branch provenance. The result is narrowed only when every branch that carries
    /// parameter origins was independently narrowed.
    fn merge(left: Option<Self>, right: Option<Self>) -> Option<Self> {
        let all_narrowed = left
            .as_ref()
            .is_none_or(|origins| matches!(origins, Self::Narrowed(_)))
            && right
                .as_ref()
                .is_none_or(|origins| matches!(origins, Self::Narrowed(_)));
        let mut parameters = left
            .into_iter()
            .chain(right)
            .flat_map(Self::into_parameters)
            .collect::<Vec<_>>();
        if parameters.is_empty() {
            return None;
        }
        parameters.sort_unstable();
        parameters.dedup();
        let parameters = parameters.into_boxed_slice();
        Some(if all_narrowed {
            Self::Narrowed(parameters)
        } else {
            Self::Inferred(parameters)
        })
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
    Dimension,
    GeneratorElement,
    Flag {
        origins: Option<DslFlagOrigins>,
        kinds: u8,
    },
}

impl DslStaticKind {
    fn parameter_origins(&self) -> Option<&[usize]> {
        match self {
            Self::UnknownParameters(parameters) => Some(parameters),
            Self::IntTuple { parameter_origins } => parameter_origins.as_deref(),
            Self::Flag {
                origins: Some(origins),
                ..
            } => Some(origins.parameters()),
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
                Self::Flag {
                    origins: left_origins,
                    kinds: left,
                },
                Self::Flag {
                    origins: right_origins,
                    kinds: right,
                },
            ) => Some(Self::Flag {
                origins: DslFlagOrigins::merge(left_origins, right_origins),
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
            (Self::UnknownParameters(parameters), Self::IntTuple { parameter_origins })
            | (Self::IntTuple { parameter_origins }, Self::UnknownParameters(parameters)) => {
                Some(Self::IntTuple {
                    parameter_origins: merge_parameter_origins(Some(parameters), parameter_origins),
                })
            }
            (Self::UnknownParameters(parameters), Self::Flag { origins, kinds })
            | (Self::Flag { origins, kinds }, Self::UnknownParameters(parameters)) => {
                // The known branch determines the possible runtime kinds. Keeping the other
                // branch as inferred provenance makes resolution require its exact domain.
                Some(Self::Flag {
                    origins: DslFlagOrigins::merge(
                        Some(DslFlagOrigins::Inferred(parameters)),
                        origins,
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum DslIntegerDomain {
    Flag,
    Dimension,
}

#[derive(Clone)]
struct DeferredInteger {
    expression: Expr,
    flow: DslValidationFlow,
    domain: Option<DslIntegerDomain>,
    parent: usize,
    validated: bool,
}

struct DslValidator<'a, F> {
    parameters: &'a Parameters,
    intrinsic: &'a F,
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
                helper_calls: Vec::new(),
                slots,
                declared_local_kinds: vec![None; kinds.len()],
                deferred_integers: Vec::new(),
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

    fn defer_integer(&mut self, expression: &Expr, flow: &DslValidationFlow) -> DslStaticKind {
        let index = self.deferred_integers.len();
        self.deferred_integers.push(DeferredInteger {
            expression: expression.clone(),
            flow: flow.clone(),
            domain: None,
            parent: index,
            validated: false,
        });
        DslStaticKind::DeferredInteger(index)
    }

    fn deferred_integer_root(&self, mut index: usize) -> usize {
        while self.deferred_integers[index].parent != index {
            debug_assert!(
                self.deferred_integers[index].domain.is_none(),
                "only a deferred integer group root may carry its domain"
            );
            index = self.deferred_integers[index].parent;
        }
        index
    }

    fn deferred_integer_domain(&self, index: usize) -> (usize, Option<DslIntegerDomain>) {
        let root = self.deferred_integer_root(index);
        debug_assert!(
            self.deferred_integers
                .iter()
                .enumerate()
                .all(
                    |(index, deferred)| self.deferred_integer_root(index) != root
                        || index == root
                        || deferred.domain.is_none()
                ),
            "only a deferred integer group root may carry its domain"
        );
        (root, self.deferred_integers[root].domain)
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
        let domain = left_domain.or(right_domain);
        self.deferred_integers[right].domain = None;
        self.deferred_integers[right].parent = left;
        self.deferred_integers[left].domain = domain;
        if let Some(domain) = domain {
            self.resolve_deferred_integer(left, domain)?;
        }
        Ok(left)
    }

    fn collect_integer_parameters(
        &self,
        expression: &Expr,
        flow: &DslValidationFlow,
        parameters: &mut Vec<usize>,
        expanded_deferred: &mut HashSet<usize>,
    ) -> bool {
        match expression {
            Expr::Name(name) => {
                let Some(&slot) = self.slots.get(&name.id) else {
                    // Deferred validation will report the unbound name with its normal diagnostic.
                    return true;
                };
                match &flow.kinds[slot] {
                    DslStaticKind::UnknownParameters(origins) => {
                        for &parameter in origins.iter() {
                            if !parameters.contains(&parameter) {
                                parameters.push(parameter);
                            }
                        }
                        true
                    }
                    DslStaticKind::Flag {
                        origins: Some(origins),
                        kinds: FLAG_INT,
                    } => {
                        for &parameter in origins.parameters() {
                            if !parameters.contains(&parameter) {
                                parameters.push(parameter);
                            }
                        }
                        true
                    }
                    DslStaticKind::DeferredInteger(index) => {
                        let root = self.deferred_integer_root(*index);
                        for (index, deferred) in self.deferred_integers.iter().enumerate() {
                            if self.deferred_integer_root(index) == root
                                && expanded_deferred.insert(index)
                                && !self.collect_integer_parameters(
                                    &deferred.expression,
                                    &deferred.flow,
                                    parameters,
                                    expanded_deferred,
                                )
                            {
                                return false;
                            }
                        }
                        true
                    }
                    DslStaticKind::Flag {
                        origins: None,
                        kinds: FLAG_INT,
                    } => true,
                    _ => false,
                }
            }
            Expr::BinOp(binop) => {
                self.collect_integer_parameters(&binop.left, flow, parameters, expanded_deferred)
                    && self.collect_integer_parameters(
                        &binop.right,
                        flow,
                        parameters,
                        expanded_deferred,
                    )
            }
            _ => !matches!(integer_literal(expression), IntegerLiteral::NotLiteral),
        }
    }

    fn deferred_integer_parameters(
        &self,
        index: usize,
    ) -> Result<Box<[usize]>, TypeShapeDslDefinitionError> {
        let root = self.deferred_integer_root(index);
        let mut parameters = Vec::new();
        let mut expanded_deferred = HashSet::new();
        for (index, deferred) in self.deferred_integers.iter().enumerate() {
            if self.deferred_integer_root(index) == root
                && expanded_deferred.insert(index)
                && !self.collect_integer_parameters(
                    &deferred.expression,
                    &deferred.flow,
                    &mut parameters,
                    &mut expanded_deferred,
                )
            {
                return Err(TypeShapeDslDefinitionError {
                    range: deferred.expression.range(),
                    message: "helper integer arguments must contain only parameters, Flag integers, and integer literals",
                });
            }
        }
        parameters.sort_unstable();
        parameters.dedup();
        Ok(parameters.into_boxed_slice())
    }

    fn finalize_helper_deferred_domains(&mut self) {
        let domains = self
            .helper_calls
            .iter()
            .flat_map(|call| &call.arguments)
            .filter_map(|argument| match &argument.provenance {
                TypeShapeDslHelperArgumentProvenance::DeferredInteger { index, .. } => Some(*index),
                _ => None,
            })
            .map(|index| {
                let (root, domain) = self.deferred_integer_domain(index);
                (index, (root, domain))
            })
            .collect::<HashMap<_, _>>();
        for argument in self
            .helper_calls
            .iter_mut()
            .flat_map(|call| &mut call.arguments)
        {
            let TypeShapeDslHelperArgumentProvenance::DeferredInteger {
                index,
                resolved_domain,
                ..
            } = &mut argument.provenance
            else {
                continue;
            };
            let (root, domain) = domains[index];
            *index = root;
            *resolved_domain = domain.map(|domain| match domain {
                DslIntegerDomain::Dimension => {
                    TypeShapeDslInputDomain::Value(TypeShapeDslDomain::Int)
                }
                DslIntegerDomain::Flag => {
                    TypeShapeDslInputDomain::Flag(FlagDomain::of(FlagMember::Int))
                }
            });
        }
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
        self.deferred_integers[root].domain = Some(domain);
        let group = (0..self.deferred_integers.len())
            .filter(|index| self.deferred_integer_root(*index) == root)
            .collect::<Vec<_>>();
        for index in group {
            debug_assert!(
                index == root || self.deferred_integers[index].domain.is_none(),
                "only a deferred integer group root may carry its domain"
            );
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
            let (_, domain) = self.deferred_integer_domain(index);
            if domain.is_none() {
                self.resolve_deferred_integer(index, DslIntegerDomain::Dimension)?;
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
                    DslStaticKind::DeferredInteger(index) => {
                        self.resolve_deferred_integer(*index, DslIntegerDomain::Dimension)?;
                        TypeShapeDslExpressionKind::DimensionSlot {
                            slot,
                            parameter_origins: None,
                        }
                    }
                    DslStaticKind::GeneratorElement => {
                        TypeShapeDslExpressionKind::GeneratorElementAsDimension(slot)
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
                let shape = self.slot(&subscript.value, flow)?;
                let parameter_origins = match &flow.kinds[shape] {
                    DslStaticKind::UnknownParameters(parameters) => Some(parameters.clone()),
                    DslStaticKind::IntTuple { parameter_origins } => parameter_origins.clone(),
                    _ => {
                        return Err(TypeShapeDslDefinitionError {
                            range: subscript.value.range(),
                            message: "indexed dimension source must be an `IntTuple` value",
                        });
                    }
                };
                self.validate_flag_int(&subscript.slice, flow)?;
                TypeShapeDslExpressionKind::IntTupleIndex {
                    shape,
                    parameter_origins,
                }
            }
            Expr::Call(call) if self.intrinsic(&call.func) == Some(TypeShapeDslIntrinsic::Prod) => {
                self.validate_int_tuple_product(call, flow)?;
                TypeShapeDslExpressionKind::IntTupleProduct
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
        let (parameter_origins, narrowed) = match &flow.kinds[slot] {
            DslStaticKind::UnknownParameters(parameters) => (Some(parameters.clone()), false),
            DslStaticKind::DeferredInteger(index) => {
                self.resolve_deferred_integer(*index, DslIntegerDomain::Dimension)?;
                (None, false)
            }
            DslStaticKind::Dimension | DslStaticKind::GeneratorElement => (None, false),
            DslStaticKind::Flag {
                origins,
                kinds: FLAG_INT,
            } => origins.as_ref().map_or((None, false), |origins| {
                let (parameters, narrowed) = origins.clone_parameters_with_narrowing();
                (Some(parameters), narrowed)
            }),
            _ => {
                return Err(TypeShapeDslDefinitionError {
                    range: expression.range(),
                    message: "dimension arithmetic operands must be integer values",
                });
            }
        };
        self.expressions.push(TypeShapeDslExpression {
            range: expression.range(),
            kind: TypeShapeDslExpressionKind::IntegerSlot {
                slot,
                parameter_origins,
                narrowed,
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
        let (parameter_origins, narrowed) = match &flow.kinds[slot] {
            DslStaticKind::UnknownParameters(parameters) => (Some(parameters.clone()), false),
            DslStaticKind::DeferredInteger(_) if expected == FLAG_INT => (None, true),
            DslStaticKind::GeneratorElement if expected == FLAG_INT => {
                self.expressions.push(TypeShapeDslExpression {
                    range: expression.range(),
                    kind: TypeShapeDslExpressionKind::GeneratorElementAsFlagInt(slot),
                });
                return Ok(());
            }
            DslStaticKind::Flag { origins, kinds } if *kinds != 0 && kinds & !expected == 0 => {
                origins.as_ref().map_or((None, false), |origins| {
                    let (parameters, narrowed) = origins.clone_parameters_with_narrowing();
                    (Some(parameters), narrowed)
                })
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
                parameter_origins,
                required,
                narrowed,
            },
        });
        Ok(())
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
            IntegerLiteral::Unrepresentable => {
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
                if call.arguments.args.len() != 1 || !call.arguments.keywords.is_empty() {
                    return Err(TypeShapeDslDefinitionError {
                        range: call.arguments.range,
                        message: "`len` requires exactly one positional argument",
                    });
                }
                let argument = &call.arguments.args[0];
                let slot = self.slot(argument, flow)?;
                match &flow.kinds[slot] {
                    DslStaticKind::UnknownParameters(parameters) => {
                        self.expressions.push(TypeShapeDslExpression {
                            range: call.range,
                            kind: TypeShapeDslExpressionKind::IntTupleLength {
                                shape: slot,
                                parameter_origins: Some(parameters.clone()),
                            },
                        });
                    }
                    DslStaticKind::Flag { kinds, .. } if *kinds == FLAG_SEQUENCE => {
                        self.validate_flag_slot(
                            argument,
                            flow,
                            TypeShapeDslFlagValueKind::Sequence,
                        )?;
                        self.expressions.push(TypeShapeDslExpression {
                            range: call.range,
                            kind: TypeShapeDslExpressionKind::FlagSequenceLength,
                        });
                    }
                    _ => {
                        return Err(TypeShapeDslDefinitionError {
                            range: argument.range(),
                            message: "`len` requires an IntTuple or Flag sequence",
                        });
                    }
                }
                Ok(())
            }
            Expr::Call(call)
                if matches!(
                    &*call.func,
                    Expr::Attribute(attribute) if attribute.attr.id.as_str() == "count"
                ) =>
            {
                let Expr::Attribute(attribute) = &*call.func else {
                    unreachable!("guarded count call has an attribute callee")
                };
                if call.arguments.args.len() != 1 || !call.arguments.keywords.is_empty() {
                    return Err(TypeShapeDslDefinitionError {
                        range: call.arguments.range,
                        message: "Flag sequence `.count` requires exactly one positional argument",
                    });
                }
                self.validate_flag_sequence(&attribute.value, flow)?;
                self.validate_flag_int(&call.arguments.args[0], flow)?;
                self.expressions.push(TypeShapeDslExpression {
                    range: call.range,
                    kind: TypeShapeDslExpressionKind::FlagSequenceCount,
                });
                Ok(())
            }
            _ => Err(TypeShapeDslDefinitionError {
                range: expression.range(),
                message: "Flag integer expression is not supported",
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
                    range: call.range,
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
                let binder =
                    self.validate_generator(generator, flow, GeneratorValidationKind::FlagValue)?;
                self.expressions.push(TypeShapeDslExpression {
                    range: call.range,
                    kind: TypeShapeDslExpressionKind::FlagGenerator { binder },
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
    ) -> Result<(), TypeShapeDslDefinitionError> {
        if let Expr::Name(_) = source {
            let slot = self.slot(source, flow)?;
            let (parameter_origins, narrowed) = match &flow.kinds[slot] {
                DslStaticKind::UnknownParameters(parameters) => (Some(parameters.clone()), false),
                DslStaticKind::Flag { origins, kinds } if *kinds == FLAG_SEQUENCE => {
                    origins.as_ref().map_or((None, false), |origins| {
                        let (parameters, narrowed) = origins.clone_parameters_with_narrowing();
                        (Some(parameters), narrowed)
                    })
                }
                _ => {
                    return Err(TypeShapeDslDefinitionError {
                        range: source.range(),
                        message: "generator source must be an IntTuple or Flag sequence",
                    });
                }
            };
            self.expressions.push(TypeShapeDslExpression {
                range: source.range(),
                kind: TypeShapeDslExpressionKind::GeneratorSourceSlot {
                    slot,
                    parameter_origins,
                    narrowed,
                },
            });
            return Ok(());
        }
        if let Expr::Call(call) = source
            && self.intrinsic(&call.func) == Some(TypeShapeDslIntrinsic::Tuple)
            && matches!(call.arguments.args.first(), Some(Expr::Generator(_)))
        {
            return Err(TypeShapeDslDefinitionError {
                range: source.range(),
                message: "nested generators are not supported",
            });
        }
        match source {
            Expr::Tuple(_) | Expr::Call(_) => self.validate_flag_sequence(source, flow),
            _ => Err(TypeShapeDslDefinitionError {
                range: source.range(),
                message: "generator source must be an IntTuple, tuple display, or `range(...)`",
            }),
        }
    }

    fn validate_generator(
        &mut self,
        generator: &ExprGenerator,
        flow: &DslValidationFlow,
        kind: GeneratorValidationKind,
    ) -> Result<usize, TypeShapeDslDefinitionError> {
        let [comprehension] = generator.generators.as_slice() else {
            return Err(TypeShapeDslDefinitionError {
                range: generator.range,
                message: match kind {
                    GeneratorValidationKind::Condition => {
                        "`any` generators require exactly one `for` clause"
                    }
                    GeneratorValidationKind::Dimension | GeneratorValidationKind::FlagValue => {
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
        let Expr::Name(target) = &comprehension.target else {
            return Err(TypeShapeDslDefinitionError {
                range: comprehension.target.range(),
                message: "generator target must be exactly one bare name",
            });
        };
        if comprehension.ifs.len() > 1 {
            return Err(TypeShapeDslDefinitionError {
                range: comprehension.range,
                message: match kind {
                    GeneratorValidationKind::Condition => {
                        "`any` generators support at most one `if` filter"
                    }
                    GeneratorValidationKind::Dimension | GeneratorValidationKind::FlagValue => {
                        "constructor generators support at most one `if` filter"
                    }
                },
            });
        }

        self.validate_generator_source(&comprehension.iter, flow)?;
        let binder = self.declared_local_kinds.len();
        self.declared_local_kinds
            .push(Some(DslStaticKind::GeneratorElement));
        let previous = self.slots.insert(target.id.clone(), binder);
        let mut generator_flow = flow.clone();
        self.normalize_flow(&mut generator_flow);
        generator_flow.assigned[binder] = true;
        generator_flow.maybe_assigned[binder] = true;
        generator_flow.kinds[binder] = DslStaticKind::GeneratorElement;

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
            }
        })();
        if let Some(previous) = previous {
            self.slots.insert(target.id.clone(), previous);
        } else {
            self.slots.remove(&target.id);
        }
        validation?;
        Ok(binder)
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
            let binder =
                self.validate_generator(generator, flow, GeneratorValidationKind::Dimension)?;
            self.expressions.push(TypeShapeDslExpression {
                range: generator.range,
                kind: TypeShapeDslExpressionKind::DimensionGenerator { binder },
            });
            self.expressions.push(TypeShapeDslExpression {
                range: call.range,
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
            range: call.range,
            kind: TypeShapeDslExpressionKind::IntTupleConstructor,
        });
        Ok(())
    }

    fn validate_int_tuple_product(
        &mut self,
        call: &ExprCall,
        flow: &DslValidationFlow,
    ) -> Result<(), TypeShapeDslDefinitionError> {
        if call.arguments.args.len() != 1
            || !call.arguments.keywords.is_empty()
            || matches!(call.arguments.args.first(), Some(Expr::Starred(_)))
        {
            return Err(TypeShapeDslDefinitionError {
                range: call.arguments.range,
                message: "`dsl.prod` requires exactly one positional IntTuple argument",
            });
        }
        self.validate_int_tuple_expression(&call.arguments.args[0], flow)?;
        Ok(())
    }

    fn validate_int_tuple_expression(
        &mut self,
        expression: &Expr,
        flow: &DslValidationFlow,
    ) -> Result<Option<Box<[usize]>>, TypeShapeDslDefinitionError> {
        let parameter_origins = match expression {
            Expr::Name(_) => {
                let slot = self.slot(expression, flow)?;
                let parameter_origins = match &flow.kinds[slot] {
                    DslStaticKind::UnknownParameters(parameters) => Some(parameters.clone()),
                    DslStaticKind::IntTuple { parameter_origins } => parameter_origins.clone(),
                    _ => {
                        return Err(TypeShapeDslDefinitionError {
                            range: expression.range(),
                            message: "shape expression names must be `IntTuple` parameters or shape locals",
                        });
                    }
                };
                self.expressions.push(TypeShapeDslExpression {
                    range: expression.range(),
                    kind: TypeShapeDslExpressionKind::IntTupleSlot {
                        slot,
                        parameter_origins: parameter_origins.clone(),
                    },
                });
                parameter_origins
            }
            Expr::Subscript(subscript) => {
                let parameter_origins =
                    self.validate_int_tuple_expression(&subscript.value, flow)?;
                let Expr::Slice(slice) = subscript.slice.as_ref() else {
                    return Err(TypeShapeDslDefinitionError {
                        range: subscript.slice.range(),
                        message: "IntTuple shape expressions support only `shape[:literal_stop]`",
                    });
                };
                if slice.lower.is_some() || slice.step.is_some() {
                    return Err(TypeShapeDslDefinitionError {
                        range: slice.range,
                        message: "IntTuple slices require an omitted lower bound and step",
                    });
                }
                let Some(upper) = slice.upper.as_deref() else {
                    return Err(TypeShapeDslDefinitionError {
                        range: slice.range,
                        message: "IntTuple slices require a literal stop",
                    });
                };
                let IntegerLiteral::Value(stop) = integer_literal(upper) else {
                    return Err(TypeShapeDslDefinitionError {
                        range: upper.range(),
                        message: "IntTuple slice stop must be a representable signed integer literal",
                    });
                };
                self.expressions.push(TypeShapeDslExpression {
                    range: expression.range(),
                    kind: TypeShapeDslExpressionKind::IntTupleSlice { stop },
                });
                parameter_origins
            }
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
                    range: call.range,
                    kind: TypeShapeDslExpressionKind::IntTupleConcat,
                });
                merge_parameter_origins(left, right)
            }
            _ => {
                return Err(TypeShapeDslDefinitionError {
                    range: expression.range(),
                    message: "IntTuple shape expressions support parameters, immutable aliases, restricted slices, `dsl.IntTuple`, and `dsl.concat`",
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
                Ok(DslStaticKind::Flag {
                    origins: None,
                    kinds: FLAG_NONE,
                })
            }
            Expr::BooleanLiteral(literal) => {
                self.expressions.push(TypeShapeDslExpression {
                    range: expression.range(),
                    kind: TypeShapeDslExpressionKind::FlagBool(literal.value),
                });
                Ok(DslStaticKind::Flag {
                    origins: None,
                    kinds: FLAG_BOOL,
                })
            }
            Expr::NumberLiteral(_) | Expr::UnaryOp(_)
                if !matches!(integer_literal(expression), IntegerLiteral::NotLiteral) =>
            {
                self.validate_flag_int(expression, flow)?;
                Ok(DslStaticKind::Flag {
                    origins: None,
                    kinds: FLAG_INT,
                })
            }
            Expr::BinOp(_) => {
                if self.is_dimension_expression(expression, flow) {
                    self.validate_dimension(expression, flow)?;
                    Ok(DslStaticKind::Dimension)
                } else if self.is_deferred_integer_expression(expression, flow) {
                    if self.is_traceable_integer_expression(expression, flow) {
                        Ok(self.defer_integer(expression, flow))
                    } else {
                        self.validate_dimension(expression, flow)?;
                        Ok(DslStaticKind::Dimension)
                    }
                } else {
                    self.validate_flag_int(expression, flow)?;
                    Ok(DslStaticKind::Flag {
                        origins: None,
                        kinds: FLAG_INT,
                    })
                }
            }
            Expr::Subscript(_) => {
                if matches!(expression, Expr::Subscript(subscript) if matches!(subscript.slice.as_ref(), Expr::Slice(_)))
                {
                    let parameter_origins = self.validate_int_tuple_expression(expression, flow)?;
                    Ok(DslStaticKind::IntTuple { parameter_origins })
                } else {
                    self.validate_dimension(expression, flow)?;
                    Ok(DslStaticKind::Dimension)
                }
            }
            Expr::Call(call)
                if matches!(
                    self.intrinsic(&call.func),
                    Some(TypeShapeDslIntrinsic::IntTuple | TypeShapeDslIntrinsic::Concat)
                ) =>
            {
                let parameter_origins = self.validate_int_tuple_expression(expression, flow)?;
                Ok(DslStaticKind::IntTuple { parameter_origins })
            }
            Expr::Tuple(_) => {
                self.validate_flag_sequence(expression, flow)?;
                Ok(DslStaticKind::Flag {
                    origins: None,
                    kinds: FLAG_SEQUENCE,
                })
            }
            Expr::Call(call) if self.intrinsic(&call.func) == Some(TypeShapeDslIntrinsic::Prod) => {
                self.validate_int_tuple_product(call, flow)?;
                self.expressions.push(TypeShapeDslExpression {
                    range: call.range,
                    kind: TypeShapeDslExpressionKind::IntTupleProduct,
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
                Ok(DslStaticKind::Flag {
                    origins: None,
                    kinds: FLAG_SEQUENCE,
                })
            }
            Expr::Call(call) if self.intrinsic(&call.func) == Some(TypeShapeDslIntrinsic::Len) => {
                self.validate_flag_int(expression, flow)?;
                Ok(DslStaticKind::Flag {
                    origins: None,
                    kinds: FLAG_INT,
                })
            }
            Expr::Call(call)
                if matches!(
                    &*call.func,
                    Expr::Attribute(attribute) if attribute.attr.id.as_str() == "count"
                ) =>
            {
                self.validate_flag_int(expression, flow)?;
                Ok(DslStaticKind::Flag {
                    origins: None,
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

    fn narrow_flag(kind: DslStaticKind, mask: u8) -> DslStaticKind {
        match kind {
            DslStaticKind::UnknownParameters(parameters) => DslStaticKind::Flag {
                origins: Some(DslFlagOrigins::Narrowed(parameters)),
                kinds: FLAG_NARROWABLE & mask,
            },
            DslStaticKind::Flag { origins, kinds } => DslStaticKind::Flag {
                origins: origins
                    .map(DslFlagOrigins::into_parameters)
                    .map(DslFlagOrigins::Narrowed),
                kinds: kinds & mask,
            },
            DslStaticKind::DeferredInteger(_) => DslStaticKind::Flag {
                origins: None,
                kinds: FLAG_INT & mask,
            },
            DslStaticKind::Dimension | DslStaticKind::IntTuple { .. } => {
                unreachable!("control-flow narrowing requires a Flag value")
            }
            DslStaticKind::GeneratorElement => {
                unreachable!("generator elements are not narrowed as Flag union values")
            }
        }
    }

    fn validate_flag_narrowing_operand(
        &mut self,
        expression: &Expr,
        flow: &DslValidationFlow,
        message: &'static str,
    ) -> Result<(usize, Option<Box<[usize]>>), TypeShapeDslDefinitionError> {
        let slot = self.slot(expression, flow)?;
        if let DslStaticKind::DeferredInteger(index) = flow.kinds[slot] {
            self.resolve_deferred_integer(index, DslIntegerDomain::Flag)?;
            return Ok((slot, None));
        }
        let parameter_origins = match &flow.kinds[slot] {
            DslStaticKind::UnknownParameters(parameters) => Some(parameters.clone()),
            DslStaticKind::Flag { origins, kinds }
                if *kinds != 0 && kinds & !FLAG_NARROWABLE == 0 =>
            {
                origins
                    .as_ref()
                    .map(DslFlagOrigins::parameters)
                    .map(<[usize]>::to_vec)
                    .map(Vec::into_boxed_slice)
            }
            _ => {
                return Err(TypeShapeDslDefinitionError {
                    range: expression.range(),
                    message,
                });
            }
        };
        Ok((slot, parameter_origins))
    }

    /// Selects the expression comparison path when syntax and local flow identify a dimension.
    /// This is intentionally narrower than `validate_dimension`: bare parameters and literals do
    /// not establish a domain by themselves, though the other operand can select this path for
    /// them. This only routes the comparison; `validate_dimension` still checks assignment and
    /// records the expression's complete validation metadata.
    fn is_dimension_expression(&self, expression: &Expr, flow: &DslValidationFlow) -> bool {
        match expression {
            Expr::Name(name) => self.slots.get(&name.id).is_some_and(|slot| {
                matches!(flow.kinds.get(*slot), Some(DslStaticKind::Dimension))
            }),
            Expr::Subscript(subscript) if !matches!(subscript.slice.as_ref(), Expr::Slice(_)) => {
                let Expr::Name(name) = &*subscript.value else {
                    return false;
                };
                self.slots.get(&name.id).is_some_and(|slot| {
                    matches!(
                        flow.kinds.get(*slot),
                        Some(DslStaticKind::UnknownParameters(_) | DslStaticKind::IntTuple { .. })
                    )
                })
            }
            Expr::BinOp(binop) => {
                self.is_dimension_expression(&binop.left, flow)
                    || self.is_dimension_expression(&binop.right, flow)
            }
            Expr::Call(call) => self.intrinsic(&call.func) == Some(TypeShapeDslIntrinsic::Prod),
            Expr::If(if_expr) => {
                self.is_dimension_expression(&if_expr.body, flow)
                    || self.is_dimension_expression(&if_expr.orelse, flow)
            }
            _ => false,
        }
    }

    fn is_deferred_integer_expression(&self, expression: &Expr, flow: &DslValidationFlow) -> bool {
        match expression {
            Expr::Name(name) => self.slots.get(&name.id).is_some_and(|slot| {
                matches!(
                    flow.kinds.get(*slot),
                    Some(DslStaticKind::UnknownParameters(_) | DslStaticKind::DeferredInteger(_))
                )
            }),
            Expr::BinOp(binop) => {
                self.is_deferred_integer_expression(&binop.left, flow)
                    || self.is_deferred_integer_expression(&binop.right, flow)
            }
            _ => false,
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
                            | DslStaticKind::DeferredInteger(_)
                            | DslStaticKind::Flag {
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
            let binder =
                self.validate_generator(generator, flow, GeneratorValidationKind::Condition)?;
            self.conditions.push(TypeShapeDslCondition {
                range: call.range,
                kind: TypeShapeDslConditionKind::Any { binder },
            });
            return Ok((flow.clone(), flow.clone()));
        }
        if let Expr::BoolOp(bool_op) = condition {
            let mut sequential = flow.clone();
            match bool_op.op {
                BoolOp::And => {
                    for value in &bool_op.values {
                        let (when_true, _) = self.validate_condition(value, &sequential)?;
                        sequential = when_true;
                    }
                    return Ok((sequential, flow.clone()));
                }
                BoolOp::Or => {
                    for value in &bool_op.values {
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
            let parameter_origins = match &flow.kinds[slot] {
                DslStaticKind::UnknownParameters(parameters) => Some(parameters.clone()),
                DslStaticKind::Flag {
                    origins,
                    kinds: FLAG_BOOL,
                    ..
                } => origins
                    .as_ref()
                    .map(DslFlagOrigins::parameters)
                    .map(<[usize]>::to_vec)
                    .map(Vec::into_boxed_slice),
                _ => {
                    return Err(TypeShapeDslDefinitionError {
                        range: condition.range(),
                        message: "a name used directly as a condition requires a `Flag[bool]` value",
                    });
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
            && compare.ops[0] == CmpOp::Is
            && compare.comparators.len() == 1
            && matches!(&compare.comparators[0], Expr::NoneLiteral(_))
        {
            let (slot, origins) = self.validate_flag_narrowing_operand(
                &compare.left,
                flow,
                "`is None` requires a `Flag[int | tuple[int, ...] | None]` value",
            )?;
            let mut when_true = flow.clone();
            let mut when_false = flow.clone();
            when_true.kinds[slot] = Self::narrow_flag(flow.kinds[slot].clone(), FLAG_NONE);
            when_false.kinds[slot] =
                Self::narrow_flag(flow.kinds[slot].clone(), FLAG_INT | FLAG_SEQUENCE);
            self.conditions.push(TypeShapeDslCondition {
                range: compare.range,
                kind: TypeShapeDslConditionKind::IsNone {
                    slot,
                    parameter_origins: origins,
                },
            });
            return Ok((when_true, when_false));
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
                let (slot, parameter_origins) = self.validate_flag_narrowing_operand(
                    &call.arguments.args[0],
                    flow,
                    "`is_int_value` requires a `Flag[int | tuple[int, ...] | None]` value",
                )?;
                when_true.kinds[slot] = Self::narrow_flag(flow.kinds[slot].clone(), FLAG_INT);
                when_false.kinds[slot] =
                    Self::narrow_flag(flow.kinds[slot].clone(), FLAG_SEQUENCE | FLAG_NONE);
                TypeShapeDslConditionKind::IsIntValue {
                    slot,
                    parameter_origins,
                }
            } else {
                let slot = self.slot(&call.arguments.args[0], flow)?;
                if let DslStaticKind::DeferredInteger(index) = flow.kinds[slot] {
                    self.resolve_deferred_integer(index, DslIntegerDomain::Dimension)?;
                }
                let parameter_origins = flow.kinds[slot]
                    .parameter_origins()
                    .map(<[usize]>::to_vec)
                    .map(Vec::into_boxed_slice);
                if !matches!(
                    &flow.kinds[slot],
                    DslStaticKind::UnknownParameters(_)
                        | DslStaticKind::DeferredInteger(_)
                        | DslStaticKind::Dimension
                        | DslStaticKind::GeneratorElement
                ) {
                    return Err(TypeShapeDslDefinitionError {
                        range: call.arguments.args[0].range(),
                        message: "`is_concrete_int` requires an Int dimension value",
                    });
                }
                TypeShapeDslConditionKind::IsConcreteInt {
                    slot,
                    parameter_origins,
                }
            };
            self.conditions.push(TypeShapeDslCondition {
                range: call.range,
                kind,
            });
            return Ok((when_true, when_false));
        }

        let Expr::Compare(compare) = condition else {
            return Err(TypeShapeDslDefinitionError {
                range: condition.range(),
                message: "condition may use only boolean Flag values, `and`, `or`, `not`, `any(...)`, `is None`, `is_concrete_int(...)`, `is_int_value(...)`, integer comparisons, and Flag sequence membership",
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
        let Some(comparison_op) = TypeShapeDslFlagIntComparisonOp::from_cmp_op(op) else {
            return Err(TypeShapeDslDefinitionError {
                range: compare.range,
                message: "comparison operator is not supported",
            });
        };

        let slot_comparison = match (&*compare.left, right) {
            (Expr::Name(_), Expr::Name(_)) => {
                let left = self.slot(&compare.left, flow)?;
                let right = self.slot(right, flow)?;
                if left == right && matches!(flow.kinds[left], DslStaticKind::GeneratorElement) {
                    Some(TypeShapeDslConditionKind::GeneratorElementSelfCompare(
                        comparison_op,
                    ))
                } else {
                    let dimension_parameters = |kind: &DslStaticKind| match kind {
                        DslStaticKind::UnknownParameters(parameters) => {
                            Some(Some(parameters.clone()))
                        }
                        DslStaticKind::Dimension => Some(None),
                        _ => None,
                    };
                    dimension_parameters(&flow.kinds[left])
                        .zip(dimension_parameters(&flow.kinds[right]))
                        .map(|(left_parameters, right_parameters)| {
                            TypeShapeDslConditionKind::SlotCompare {
                                left,
                                right,
                                left_parameters,
                                right_parameters,
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
            IntegerLiteral::NotLiteral | IntegerLiteral::Unrepresentable => None,
        };
        let kind = match slot_comparison {
            Some(kind) => kind,
            None if has_dimension_expression => {
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
            None if op == CmpOp::Eq
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
            _ => {
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
                    if slot < self.parameters.args.len() {
                        TypeShapeDslReturnKind::Parameter(slot)
                    } else {
                        let (domain, parameter_origins) = match &flow.kinds[slot] {
                            DslStaticKind::Dimension => (TypeShapeDslDomain::Int, None),
                            DslStaticKind::DeferredInteger(index) => {
                                self.resolve_deferred_integer(*index, DslIntegerDomain::Dimension)?;
                                (TypeShapeDslDomain::Int, None)
                            }
                            DslStaticKind::IntTuple { parameter_origins } => {
                                (TypeShapeDslDomain::IntTuple, parameter_origins.clone())
                            }
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
                            _ => {
                                return Err(TypeShapeDslDefinitionError {
                                    range: returned.range(),
                                    message: "Flag values cannot be returned as shape values",
                                });
                            }
                        };
                        TypeShapeDslReturnKind::Local {
                            slot,
                            domain,
                            parameter_origins,
                        }
                    }
                }
            }
            Some(returned @ Expr::Call(call)) => match self.intrinsic(&call.func) {
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
                    let (
                        DslStaticKind::UnknownParameters(left_parameters),
                        DslStaticKind::UnknownParameters(right_parameters),
                    ) = (&flow.kinds[left], &flow.kinds[right])
                    else {
                        return Err(TypeShapeDslDefinitionError {
                            range: call.arguments.range,
                            message: "`broadcast` arguments must be IntTuple parameters or immutable aliases of them",
                        });
                    };
                    TypeShapeDslReturnKind::Broadcast {
                        left_slot: left,
                        right_slot: right,
                        left_parameters: left_parameters.clone(),
                        right_parameters: right_parameters.clone(),
                    }
                }
                Some(TypeShapeDslIntrinsic::IntTuple | TypeShapeDslIntrinsic::Concat) => {
                    self.validate_int_tuple_expression(returned, flow)?;
                    TypeShapeDslReturnKind::IntTupleExpression
                }
                Some(TypeShapeDslIntrinsic::Prod) => {
                    self.validate_int_tuple_product(call, flow)?;
                    self.expressions.push(TypeShapeDslExpression {
                        range: call.range,
                        kind: TypeShapeDslExpressionKind::IntTupleProduct,
                    });
                    TypeShapeDslReturnKind::IntTupleProduct
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
                    let mut arguments = Vec::with_capacity(call.arguments.args.len());
                    for argument in &call.arguments.args {
                        let Ok(slot) = self.slot(argument, flow) else {
                            return Err(TypeShapeDslDefinitionError {
                                range: argument.range(),
                                message: "return value must be a bare parameter name or validated DSL helper call; helper arguments must be bare parameter or local names",
                            });
                        };
                        let provenance = match &flow.kinds[slot] {
                            DslStaticKind::UnknownParameters(parameters) => {
                                TypeShapeDslHelperArgumentProvenance::Parameters(parameters.clone())
                            }
                            DslStaticKind::Dimension => {
                                TypeShapeDslHelperArgumentProvenance::Exact(
                                    TypeShapeDslInputDomain::Value(TypeShapeDslDomain::Int),
                                )
                            }
                            DslStaticKind::DeferredInteger(index) => {
                                TypeShapeDslHelperArgumentProvenance::DeferredInteger {
                                    index: *index,
                                    parameters: self.deferred_integer_parameters(*index)?,
                                    resolved_domain: None,
                                }
                            }
                            DslStaticKind::IntTuple {
                                parameter_origins: Some(parameters),
                            } => {
                                TypeShapeDslHelperArgumentProvenance::ParametersWithRequiredDomain {
                                    parameters: parameters.clone(),
                                    domain: TypeShapeDslInputDomain::Value(
                                        TypeShapeDslDomain::IntTuple,
                                    ),
                                }
                            }
                            DslStaticKind::IntTuple {
                                parameter_origins: None,
                            } => TypeShapeDslHelperArgumentProvenance::Exact(
                                TypeShapeDslInputDomain::Value(TypeShapeDslDomain::IntTuple),
                            ),
                            DslStaticKind::Flag { origins, kinds } => {
                                let Some(domain) = flag_domain_from_kinds(*kinds) else {
                                    return Err(TypeShapeDslDefinitionError {
                                        range: argument.range(),
                                        message: "DSL helper arguments must have a nonempty supported Flag domain",
                                    });
                                };
                                let domain = TypeShapeDslInputDomain::Flag(domain);
                                match origins {
                                    Some(DslFlagOrigins::Inferred(parameters)) => {
                                        let parameters = parameters.clone();
                                        TypeShapeDslHelperArgumentProvenance::ParametersWithRequiredDomain {
                                            parameters,
                                            domain,
                                        }
                                    }
                                    _ => TypeShapeDslHelperArgumentProvenance::Exact(domain),
                                }
                            }
                            DslStaticKind::GeneratorElement => {
                                return Err(TypeShapeDslDefinitionError {
                                    range: argument.range(),
                                    message: "generator elements cannot escape their generator",
                                });
                            }
                        };
                        arguments.push(TypeShapeDslHelperArgument { slot, provenance });
                    }
                    let helper = self.helper_calls.len();
                    self.helper_calls.push(TypeShapeDslHelperCall {
                        callee: (*call.func).clone(),
                        arguments,
                    });
                    TypeShapeDslReturnKind::HelperCall(helper)
                }
                Some(_) => {
                    return Err(TypeShapeDslDefinitionError {
                        range: return_stmt.range,
                        message: "return value must be a bare parameter name, gradual return, `broadcast(...)`, `dsl.Invalid(...)`, an Int/IntTuple expression, or a validated DSL helper call",
                    });
                }
            },
            Some(returned @ Expr::Subscript(subscript))
                if matches!(subscript.slice.as_ref(), Expr::Slice(_)) =>
            {
                self.validate_int_tuple_expression(returned, flow)?;
                TypeShapeDslReturnKind::IntTupleExpression
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
                    message: "return value must be a bare parameter name, gradual return, `broadcast(...)`, `dsl.Invalid(...)`, an Int/IntTuple expression, or a validated DSL helper call",
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
                            flag @ DslStaticKind::Flag {
                                kinds: FLAG_INT, ..
                            },
                        )
                        | (
                            flag @ DslStaticKind::Flag {
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

    /// Validate the body: its indexed storage, control flow, and every resolved execution fact
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
        // Helper signatures can determine the domain of otherwise-unconstrained integer locals,
        // so preserve their unresolved state before defaulting integers that have no such use.
        validator.finalize_helper_deferred_domains();
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
        Ok(ValidatedTypeShapeDslFunction {
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

    pub fn helper_calls(&self) -> impl Iterator<Item = &TypeShapeDslHelperCall> {
        self.helper_calls.iter()
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
    FlagBool(bool),
    FlagNone,
    FlagSequence(DslFlagSequence),
    DimensionTuple(Vec<Int>),
}

#[derive(Debug, Clone)]
enum DslFlagSequence {
    Values(Vec<i64>),
    Range { start: i64, stop: i64, step: i64 },
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
    },
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
            TypeLevelDslFunction::UserDefined(function) => project(function.evaluate(&self.args)),
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
                        TypeShapeDslReturnKind::Parameter(return_index) => {
                            assert_eq!(
                                node.parameter_domains()[return_index],
                                TypeShapeDslInputDomain::Value(node.result_domain()),
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
                        TypeShapeDslReturnKind::Broadcast {
                            left_slot,
                            right_slot,
                            ..
                        } => evaluate_broadcast(
                            environment.value(left_slot),
                            environment.value(right_slot),
                        ),
                        TypeShapeDslReturnKind::IntTupleExpression
                        | TypeShapeDslReturnKind::IntTupleProduct
                        | TypeShapeDslReturnKind::Expression(_) => {
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
                            let expected_domains = self.node(target).parameter_domains();
                            let arguments = helper
                                .arguments
                                .iter()
                                .zip(expected_domains)
                                .map(|(argument, expected)| {
                                    let value = environment.value(argument.slot);
                                    if matches!(
                                        argument.provenance,
                                        TypeShapeDslHelperArgumentProvenance::DeferredInteger { .. }
                                    ) && *expected
                                        == TypeShapeDslInputDomain::Flag(FlagDomain::of(
                                            FlagMember::Int,
                                        ))
                                    {
                                        match value {
                                            DslValue::Dimension(Int::Literal(value)) => {
                                                DslValue::FlagInt(*value)
                                            }
                                            DslValue::FlagInt(_) | DslValue::Unknown => {
                                                value.clone()
                                            }
                                            _ => unreachable!(
                                                "validated deferred Flag[int] helper argument is literal or gradual"
                                            ),
                                        }
                                    } else {
                                        value.clone()
                                    }
                                })
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

impl ValidatedTypeShapeDslFunction {
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
            DslOutcome::Value(DslValue::Shape(shape)) => {
                let IntTupleView::Concrete(shape) = shape.view() else {
                    return Ok(EvaluatedGeneratorItems::Unknown);
                };
                EvaluatedGeneratorItems::Known {
                    values: shape
                        .iter()
                        .take(item_limit)
                        .cloned()
                        .map(DslValue::Dimension)
                        .collect(),
                    truncated: shape.len() > item_limit,
                }
            }
            DslOutcome::Value(DslValue::FlagSequence(sequence)) => {
                let Some((values, truncated)) = sequence.bounded_values(item_limit) else {
                    return Ok(EvaluatedGeneratorItems::Unknown);
                };
                EvaluatedGeneratorItems::Known {
                    values: values.into_iter().map(DslValue::FlagInt).collect(),
                    truncated,
                }
            }
            DslOutcome::Value(DslValue::Unknown) => EvaluatedGeneratorItems::Unknown,
            DslOutcome::Invalid(error) => return Err(error),
            DslOutcome::ExplicitGradual => {
                unreachable!("validated generator source cannot return gradual")
            }
            DslOutcome::Value(_) => {
                unreachable!("validated generator source is an IntTuple or Flag sequence")
            }
        })
    }

    fn evaluate_generator(
        &self,
        generator: &ExprGenerator,
        binder: usize,
        environment: &DslEnvironment,
        result: GeneratorResultKind,
        budget: &mut DslEvaluationBudget,
    ) -> DslOutcome {
        let [comprehension] = generator.generators.as_slice() else {
            unreachable!("validated constructor generator has exactly one clause")
        };
        let (values, truncated) =
            match self.evaluate_generator_items(&comprehension.iter, environment, budget) {
                Ok(EvaluatedGeneratorItems::Known { values, truncated }) => (values, truncated),
                Ok(EvaluatedGeneratorItems::Unknown) => {
                    return DslOutcome::Value(DslValue::Unknown);
                }
                Err(error) => return DslOutcome::Invalid(error),
            };

        let mut dimensions = Vec::new();
        let mut flag_values = Vec::new();
        let mut unknown = false;
        let mut iteration = environment.clone();
        for value in values {
            if !budget.consume_generator_step() {
                return DslOutcome::Value(DslValue::Unknown);
            }
            iteration.assign(binder, value);
            if let Some(filter) = comprehension.ifs.first() {
                match self.evaluate_condition(filter, &iteration, budget) {
                    Err(error) => return DslOutcome::Invalid(error),
                    Ok(DslCondition::False) => continue,
                    Ok(DslCondition::Unknown | DslCondition::UnknownWithPossibleError) => {
                        unknown = true;
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
                (_, DslOutcome::Value(DslValue::Unknown)) => unknown = true,
                (
                    GeneratorResultKind::Dimensions,
                    DslOutcome::Value(DslValue::Dimension(value)),
                ) => dimensions.push(value),
                (GeneratorResultKind::FlagValues, DslOutcome::Value(DslValue::FlagInt(value))) => {
                    flag_values.push(value)
                }
                (_, DslOutcome::ExplicitGradual) => {
                    unreachable!("validated generator element cannot return gradual")
                }
                _ => unreachable!("validated generator element has its constructor's domain"),
            }
        }
        if unknown || truncated {
            DslOutcome::Value(DslValue::Unknown)
        } else {
            match result {
                GeneratorResultKind::Dimensions => {
                    DslOutcome::Value(DslValue::DimensionTuple(dimensions))
                }
                GeneratorResultKind::FlagValues => {
                    DslOutcome::Value(DslValue::FlagSequence(DslFlagSequence::Values(flag_values)))
                }
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
            TypeShapeDslExpressionKind::DimensionSlot { slot, .. } => {
                DslOutcome::Value(environment.value(slot).clone())
            }
            TypeShapeDslExpressionKind::GeneratorElementAsDimension(slot) => {
                match environment.value(slot) {
                    DslValue::Dimension(value) => {
                        DslOutcome::Value(DslValue::Dimension(value.clone()))
                    }
                    DslValue::FlagInt(value) => {
                        DslOutcome::Value(DslValue::Dimension(Int::Literal(*value)))
                    }
                    DslValue::Unknown => DslOutcome::Value(DslValue::Unknown),
                    _ => unreachable!("generator elements are integer values"),
                }
            }
            TypeShapeDslExpressionKind::IntegerSlot { slot, .. } => match environment.value(slot) {
                DslValue::FlagInt(value) => {
                    DslOutcome::Value(DslValue::Dimension(Int::Literal(*value)))
                }
                value @ (DslValue::Dimension(_) | DslValue::Unknown) => {
                    DslOutcome::Value(value.clone())
                }
                _ => unreachable!("validated integer slot contains an integer value"),
            },
            TypeShapeDslExpressionKind::IntTupleSlot { slot, .. } => {
                DslOutcome::Value(environment.value(slot).clone())
            }
            TypeShapeDslExpressionKind::IntTupleSlice { stop } => {
                let Expr::Subscript(subscript) = expression else {
                    unreachable!("validated IntTuple slice is a subscript")
                };
                let shape = match self.evaluate_expression(&subscript.value, environment, budget) {
                    DslOutcome::Value(DslValue::Shape(shape)) => shape,
                    DslOutcome::Value(DslValue::Unknown) => {
                        return DslOutcome::Value(DslValue::Unknown);
                    }
                    DslOutcome::ExplicitGradual => {
                        unreachable!("validated shape expression cannot return explicit gradual")
                    }
                    invalid @ DslOutcome::Invalid(_) => return invalid,
                    DslOutcome::Value(_) => unreachable!("validated slice operand is a shape"),
                };
                DslOutcome::Value(DslValue::Shape(shape.prefix_slice(stop)))
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
            TypeShapeDslExpressionKind::DimensionLiteral(literal) => literal
                .map_or(DslOutcome::Value(DslValue::Unknown), |literal| {
                    DslOutcome::Value(DslValue::Dimension(Int::Literal(literal)))
                }),
            TypeShapeDslExpressionKind::IntTupleIndex { shape, .. } => {
                let Expr::Subscript(subscript) = expression else {
                    unreachable!("validated IntTuple index expression is a subscript")
                };
                let index = match self.evaluate_expression(&subscript.slice, environment, budget) {
                    DslOutcome::Value(DslValue::FlagInt(index)) => index,
                    DslOutcome::Value(DslValue::Unknown) => {
                        return DslOutcome::Value(DslValue::Unknown);
                    }
                    DslOutcome::ExplicitGradual => {
                        unreachable!("validated Flag integer index cannot be explicitly gradual")
                    }
                    invalid @ DslOutcome::Invalid(_) => return invalid,
                    DslOutcome::Value(_) => {
                        unreachable!("validated IntTuple index evaluates to a Flag integer")
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
            TypeShapeDslExpressionKind::DimensionTuple => {
                let Expr::Tuple(tuple) = expression else {
                    unreachable!("validated dimension tuple expression is a tuple")
                };
                let mut dimensions = Vec::with_capacity(tuple.elts.len());
                let mut unknown = false;
                for element in &tuple.elts {
                    match self.evaluate_expression(element, environment, budget) {
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
            TypeShapeDslExpressionKind::IntTupleProduct => {
                let Expr::Call(call) = expression else {
                    unreachable!("validated IntTuple product expression is a call")
                };
                match self.evaluate_expression(&call.arguments.args[0], environment, budget) {
                    DslOutcome::Value(DslValue::Shape(shape)) => match shape.product() {
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
            TypeShapeDslExpressionKind::IntTupleLength { shape, .. } => {
                let shape = match environment.value(shape) {
                    DslValue::Shape(shape) => shape,
                    DslValue::Unknown => return DslOutcome::Value(DslValue::Unknown),
                    _ => {
                        unreachable!("validated IntTuple length parameter evaluates to a shape")
                    }
                };
                let IntTupleView::Concrete(shape) = shape.view() else {
                    return DslOutcome::Value(DslValue::Unknown);
                };
                let length = i64::try_from(shape.len())
                    .expect("concrete IntTuple length must fit in a Flag integer");
                DslOutcome::Value(DslValue::FlagInt(length))
            }
            TypeShapeDslExpressionKind::GeneratorSourceSlot { slot, .. } => {
                DslOutcome::Value(environment.value(slot).clone())
            }
            TypeShapeDslExpressionKind::Slot(slot) => {
                DslOutcome::Value(environment.value(slot).clone())
            }
            TypeShapeDslExpressionKind::GeneratorElementAsFlagInt(slot) => {
                match environment.value(slot) {
                    DslValue::FlagInt(value) => DslOutcome::Value(DslValue::FlagInt(*value)),
                    DslValue::Dimension(Int::Literal(value)) => {
                        DslOutcome::Value(DslValue::FlagInt(*value))
                    }
                    DslValue::Dimension(_) | DslValue::Unknown => {
                        DslOutcome::Value(DslValue::Unknown)
                    }
                    _ => unreachable!("generator elements are integer values"),
                }
            }
            TypeShapeDslExpressionKind::FlagValueSlot { slot, .. } => {
                DslOutcome::Value(environment.value(slot).clone())
            }
            TypeShapeDslExpressionKind::FlagIntLiteral(literal) => literal
                .map_or(DslOutcome::Value(DslValue::Unknown), |literal| {
                    DslOutcome::Value(DslValue::FlagInt(literal))
                }),
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
                        DslOutcome::Value(DslValue::Unknown) => unknown = true,
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
                        DslOutcome::Value(DslValue::Unknown) => values.push(None),
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
                    (DslOutcome::Value(DslValue::Unknown), _)
                    | (_, DslOutcome::Value(DslValue::Unknown)) => {
                        DslOutcome::Value(DslValue::Unknown)
                    }
                    (DslOutcome::ExplicitGradual, _) | (_, DslOutcome::ExplicitGradual) => {
                        unreachable!("validated value expression cannot return gradual")
                    }
                    _ => unreachable!("validated count uses a Flag sequence and integer"),
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
            TypeShapeDslExpressionKind::DimensionGenerator { binder } => {
                let Expr::Generator(generator) = expression else {
                    unreachable!("validated dimension generator retains its generator AST")
                };
                self.evaluate_generator(
                    generator,
                    binder,
                    environment,
                    GeneratorResultKind::Dimensions,
                    budget,
                )
            }
            TypeShapeDslExpressionKind::FlagGenerator { binder } => {
                let Expr::Call(call) = expression else {
                    unreachable!("validated Flag generator expression is a tuple call")
                };
                let Some(Expr::Generator(generator)) = call.arguments.args.first() else {
                    unreachable!("validated tuple call contains a generator")
                };
                self.evaluate_generator(
                    generator,
                    binder,
                    environment,
                    GeneratorResultKind::FlagValues,
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
                EvaluatedGeneratorItems::Known { values, truncated } => (values, truncated),
                EvaluatedGeneratorItems::Unknown => return Ok(DslCondition::Unknown),
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

    fn compare_dimensions(
        left: &Int,
        right: &Int,
        op: TypeShapeDslFlagIntComparisonOp,
    ) -> DslCondition {
        match op {
            TypeShapeDslFlagIntComparisonOp::Equal => Self::dimension_equality(left, right),
            TypeShapeDslFlagIntComparisonOp::NotEqual => {
                Self::negate_condition(Self::dimension_equality(left, right))
            }
            TypeShapeDslFlagIntComparisonOp::LessThan => match (left, right) {
                (left, right) if left == right && !matches!(left, Int::Int) => DslCondition::False,
                (Int::Literal(left), Int::Literal(right)) if left < right => DslCondition::True,
                (Int::Literal(_), Int::Literal(_)) => DslCondition::False,
                _ => DslCondition::Unknown,
            },
            TypeShapeDslFlagIntComparisonOp::LessThanOrEqual
            | TypeShapeDslFlagIntComparisonOp::GreaterThan
            | TypeShapeDslFlagIntComparisonOp::GreaterThanOrEqual => {
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
                DslValue::FlagBool(false) => DslCondition::False,
                DslValue::Unknown => DslCondition::Unknown,
                DslValue::FlagInt(_)
                | DslValue::FlagNone
                | DslValue::FlagSequence(_)
                | DslValue::Dimension(_)
                | DslValue::Shape(_)
                | DslValue::DimensionTuple(_) => {
                    unreachable!("validated boolean condition contains a boolean Flag value")
                }
            },
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
            TypeShapeDslConditionKind::IsIntValue { slot, .. } => match environment.value(slot) {
                DslValue::FlagInt(_) => DslCondition::True,
                DslValue::FlagNone | DslValue::FlagSequence(_) => DslCondition::False,
                DslValue::Unknown => DslCondition::Unknown,
                DslValue::FlagBool(_)
                | DslValue::Dimension(_)
                | DslValue::Shape(_)
                | DslValue::DimensionTuple(_) => {
                    unreachable!("validated is_int_value operand is a non-boolean Flag value")
                }
            },
            TypeShapeDslConditionKind::IsNone { slot, .. } => match environment.value(slot) {
                DslValue::FlagNone => DslCondition::True,
                DslValue::FlagInt(_) | DslValue::FlagSequence(_) => DslCondition::False,
                DslValue::Unknown => DslCondition::Unknown,
                DslValue::FlagBool(_)
                | DslValue::Dimension(_)
                | DslValue::Shape(_)
                | DslValue::DimensionTuple(_) => {
                    // Function-level domain validation rejects non-Flag parameter origins.
                    unreachable!("validated `is None` operand is a Flag value")
                }
            },
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
                    (DslOutcome::Value(DslValue::Unknown), _)
                    | (_, DslOutcome::Value(DslValue::Unknown)) => DslCondition::Unknown,
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
                    (DslOutcome::Value(DslValue::Unknown), _)
                    | (_, DslOutcome::Value(DslValue::Unknown)) => DslCondition::Unknown,
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
                    _ => unreachable!("validated dimension comparison operands are dimensions"),
                }
            }
            TypeShapeDslConditionKind::SlotCompare {
                left, right, op, ..
            } => {
                if left == right {
                    match op {
                        TypeShapeDslFlagIntComparisonOp::Equal
                        | TypeShapeDslFlagIntComparisonOp::LessThanOrEqual
                        | TypeShapeDslFlagIntComparisonOp::GreaterThanOrEqual => DslCondition::True,
                        TypeShapeDslFlagIntComparisonOp::NotEqual
                        | TypeShapeDslFlagIntComparisonOp::LessThan
                        | TypeShapeDslFlagIntComparisonOp::GreaterThan => DslCondition::False,
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
                        (DslValue::Unknown, _) | (_, DslValue::Unknown) => DslCondition::Unknown,
                        _ => unreachable!(
                            "validated slot comparison operands share the same value domain"
                        ),
                    }
                }
            }
            TypeShapeDslConditionKind::GeneratorElementSelfCompare(op) => match op {
                TypeShapeDslFlagIntComparisonOp::Equal
                | TypeShapeDslFlagIntComparisonOp::LessThanOrEqual
                | TypeShapeDslFlagIntComparisonOp::GreaterThanOrEqual => DslCondition::True,
                TypeShapeDslFlagIntComparisonOp::NotEqual
                | TypeShapeDslFlagIntComparisonOp::LessThan
                | TypeShapeDslFlagIntComparisonOp::GreaterThan => DslCondition::False,
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
                    DslValue::Unknown => DslCondition::Unknown,
                    _ => {
                        unreachable!(
                            "validated length equality evaluates an IntTuple or Flag sequence"
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

fn lower_parameter(ty: &Type, domain: TypeShapeDslInputDomain) -> DslValue {
    match domain {
        TypeShapeDslInputDomain::Value(domain) => DslValue::from_type(ty, domain),
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
                Type::Int(Int::Literal(value)) => DslValue::FlagInt(*value),
                // Symbolic shape integers satisfy `Flag[int]`, but DSL flag operations inspect
                // only concrete runtime values. Generic substitution does not re-evaluate a call
                // that already fell back.
                Type::Int(_) => DslValue::Unknown,
                Type::Literal(literal) => match &literal.value {
                    Lit::Int(value) => value.as_i64().map_or(DslValue::Unknown, DslValue::FlagInt),
                    Lit::Bool(value) => DslValue::FlagBool(*value),
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
                value.map_or(DslOutcome::Value(DslValue::Unknown), |value| {
                    DslOutcome::Value(DslValue::Dimension(Int::Literal(value)))
                })
            },
        );
    }
    if matches!(op, TypeShapeDslArithmeticOp::Modulo) {
        return DslOutcome::Value(DslValue::Unknown);
    }
    let result = match op {
        TypeShapeDslArithmeticOp::Add => Int::add(Type::Int(left), Type::Int(right)),
        TypeShapeDslArithmeticOp::Subtract => Int::sub(Type::Int(left), Type::Int(right)),
        TypeShapeDslArithmeticOp::Multiply => Int::mul(Type::Int(left), Type::Int(right)),
        TypeShapeDslArithmeticOp::FloorDivide => Int::floor_div(Type::Int(left), Type::Int(right)),
        TypeShapeDslArithmeticOp::Modulo => unreachable!("symbolic modulo is gradual"),
    };
    match canonicalize(Type::Int(result)) {
        Type::Int(Int::Int) => DslOutcome::Value(DslValue::Unknown),
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
            TypeShapeDslDomain::Int => Int::from_type(ty).map_or(Self::Unknown, Self::Dimension),
            TypeShapeDslDomain::IntTuple => Self::from_shape_type(ty),
        }
    }

    fn from_shape_type(ty: &Type) -> Self {
        IntTuple::from_shape_arg_or_tuple_carrier(ty).map_or(Self::Unknown, Self::Shape)
    }

    fn into_type(self) -> Type {
        match self {
            Self::Dimension(value) => Type::Int(value),
            Self::Shape(value) => value.to_shape_arg_type(),
            Self::Unknown => unreachable!("unknown DSL values project through the fallback"),
            Self::FlagInt(_)
            | Self::FlagBool(_)
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

#[cfg(test)]
mod tests {
    use super::*;

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
