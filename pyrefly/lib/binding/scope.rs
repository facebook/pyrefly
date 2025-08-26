/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt::Debug;
use std::mem;

use parse_display::Display;
use pyrefly_python::ast::Ast;
use pyrefly_python::dunder;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_python::sys_info::SysInfo;
use ruff_python_ast::AtomicNodeIndex;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprName;
use ruff_python_ast::ExprYield;
use ruff_python_ast::ExprYieldFrom;
use ruff_python_ast::Identifier;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtReturn;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use starlark_map::Hashed;
use starlark_map::small_map::Entry;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use vec1::Vec1;

use crate::binding::binding::ExprOrBinding;
use crate::binding::binding::Key;
use crate::binding::binding::KeyAnnotation;
use crate::binding::binding::KeyClass;
use crate::binding::binding::KeyClassBaseType;
use crate::binding::binding::KeyClassMetadata;
use crate::binding::binding::KeyClassMro;
use crate::binding::binding::KeyClassSynthesizedFields;
use crate::binding::binding::KeyConsistentOverrideCheck;
use crate::binding::binding::KeyDecoratedFunction;
use crate::binding::binding::KeyVariance;
use crate::binding::binding::KeyYield;
use crate::binding::binding::KeyYieldFrom;
use crate::binding::binding::MethodThatSetsAttr;
use crate::binding::bindings::BindingTable;
use crate::binding::bindings::CurrentIdx;
use crate::binding::function::SelfAssignments;
use crate::export::definitions::DefinitionStyle;
use crate::export::definitions::Definitions;
use crate::export::exports::LookupExport;
use crate::export::special::SpecialExport;
use crate::graph::index::Idx;
use crate::module::module_info::ModuleInfo;
use crate::types::class::ClassDefIndex;

/// Many names may map to the same TextRange (e.g. from foo import *).
/// But no other static will point at the same TextRange.
#[derive(Default, Clone, Debug)]
pub struct Static(pub SmallMap<Name, StaticInfo>);

#[derive(Clone, Debug)]
pub struct StaticInfo {
    pub loc: TextRange,
    /// The location of the first annotated name for this binding, if any.
    pub annot: Option<Idx<KeyAnnotation>>,
    /// How many times this will be redefined
    pub count: usize,
    pub style: DefinitionStyle,
    /// Is this just a name declaration (i.e. a global or a nonlocal)?
    pub is_declaration: bool,
}

impl StaticInfo {
    pub fn as_key(&self, name: &Name) -> Key {
        if self.count == 1 {
            match self.style {
                DefinitionStyle::ImportModule(_) => Key::Import(name.clone(), self.loc),
                DefinitionStyle::Global => Key::Global(name.clone()),
                _ => {
                    // We are constructing an identifier, but it must have been one that we saw earlier
                    assert_ne!(self.loc, TextRange::default());
                    let short_identifier = ShortIdentifier::new(&Identifier {
                        node_index: AtomicNodeIndex::dummy(),
                        id: name.clone(),
                        range: self.loc,
                    });
                    if self.is_declaration {
                        Key::Declaration(short_identifier)
                    } else {
                        Key::Definition(short_identifier)
                    }
                }
            }
        } else {
            Key::Anywhere(name.clone(), self.loc)
        }
    }
}

impl Static {
    fn add_with_count(
        &mut self,
        name: Hashed<Name>,
        loc: TextRange,
        symbol_kind: SymbolKind,
        annot: Option<Idx<KeyAnnotation>>,
        is_declaration: bool,
        count: usize,
    ) -> &mut StaticInfo {
        // Use whichever one we see first
        let res = self.0.entry_hashed(name).or_insert(StaticInfo {
            loc,
            annot,
            count: 0,
            style: DefinitionStyle::Local(symbol_kind),
            is_declaration,
        });
        res.count += count;
        res
    }

    pub fn add(
        &mut self,
        name: Name,
        range: TextRange,
        symbol_kind: SymbolKind,
        annot: Option<Idx<KeyAnnotation>>,
        is_declaration: bool,
    ) {
        self.add_with_count(
            Hashed::new(name),
            range,
            symbol_kind,
            annot,
            is_declaration,
            1,
        );
    }

    pub fn stmts(
        &mut self,
        x: &[Stmt],
        module_info: &ModuleInfo,
        top_level: bool,
        lookup: &dyn LookupExport,
        sys_info: &SysInfo,
        mut get_annotation_idx: impl FnMut(ShortIdentifier) -> Idx<KeyAnnotation>,
    ) {
        let mut d = Definitions::new(
            x,
            module_info.name(),
            module_info.path().is_init(),
            sys_info,
        );
        if top_level {
            if module_info.name() != ModuleName::builtins() {
                d.inject_builtins();
            }
            d.inject_globals();
        }

        let mut wildcards = Vec::with_capacity(d.import_all.len());
        for (m, range) in d.import_all {
            if let Ok(exports) = lookup.get(m) {
                wildcards.push((range, exports.wildcard(lookup)));
            }
        }

        // Try and avoid rehashing while we insert, with a little bit of spare space
        let capacity_guess =
            d.definitions.len() + wildcards.iter().map(|x| x.1.len()).sum::<usize>();
        self.0.reserve(((capacity_guess * 5) / 4) + 25);

        for (name, def) in d.definitions.into_iter_hashed() {
            let annot = def.annot.map(&mut get_annotation_idx);
            let info = self.add_with_count(
                name,
                def.range,
                SymbolKind::Variable,
                annot,
                false,
                def.count,
            );
            info.style = def.style;
        }
        for (range, wildcard) in wildcards {
            for name in wildcard.iter_hashed() {
                // TODO: semantics of import * and global var with same name
                self.add_with_count(name.cloned(), range, SymbolKind::Module, None, false, 1)
                    .style = DefinitionStyle::ImportModule(module_info.name());
            }
        }
    }

    pub fn expr_lvalue(&mut self, x: &Expr) {
        let mut add = |name: &ExprName| {
            self.add(
                name.id.clone(),
                name.range,
                SymbolKind::Variable,
                None,
                false,
            )
        };
        Ast::expr_lvalue(x, &mut add);
    }
}

/// Flow-sensitive information about a name.
#[derive(Default, Clone, Debug)]
pub struct Flow {
    pub info: SmallMap<Name, FlowInfo>,
    // Have we seen control flow terminate?
    //
    // We continue to analyze the rest of the code after a flow terminates, but
    // we don't include terminated flows when merging after loops and branches.
    pub has_terminated: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FlowStyle {
    /// Not one of the styles below.
    Other,
    /// I am a name defined by an Assign or AnnAssign in a class body.
    /// - If `initial_value` is `None`, then I am defined by an `AnnAssign`
    ///   with no value (in other words, I am an instance attribute annotation)
    /// - If `initial_value` is `Some(_)`, then I am defined by an assignment,
    ///   and the initial value may be needed later (if I turn out to be a dataclass
    ///   field, which requires inspecting the actual expression).
    ClassField { initial_value: Option<Expr> },
    /// Am I the result of an import (which needs merging).
    /// E.g. `import foo.bar` and `import foo.baz` need merging.
    /// The `ModuleName` will be the most recent entry.
    MergeableImport(ModuleName),
    /// Was I imported from somewhere (and if so, where)
    /// E.g. Both `from foo import bar` and
    /// `from foo import bar as baz` would get `(foo, bar)`.
    Import(ModuleName, Name),
    /// Am I an alias for a module import, `import foo.bar as baz`
    /// would get `foo.bar` here.
    ImportAs(ModuleName),
    /// Am I a function definition? Used to chain overload definitions.
    /// If so, does my return type have an explicit annotation?
    FunctionDef(Idx<KeyDecoratedFunction>, bool),
    /// The name is possibly uninitialized (perhaps due to merging branches)
    PossiblyUninitialized,
    /// The name was in an annotated declaration like `x: int` but not initialized
    Uninitialized,
}

impl FlowStyle {
    /// Produce an error message for an uninitialized or unbound variable.
    pub fn uninitialized_error_message(&self, name: &Identifier) -> Option<String> {
        match self {
            Self::Uninitialized => Some(format!("`{name}` is uninitialized")),
            Self::PossiblyUninitialized => Some(format!("`{name}` may be uninitialized")),
            _ => None,
        }
    }

    pub fn merged(styles: Vec<FlowStyle>) -> FlowStyle {
        let mut it = styles.into_iter();
        let mut merged = it.next().unwrap_or(FlowStyle::Other);
        for x in it {
            match (&merged, x) {
                // If they're identical, keep it
                (l, r) if l == &r => {}
                // Uninitialized and initialized branches merge into PossiblyUninitialized
                (FlowStyle::Uninitialized, _) => {
                    return FlowStyle::PossiblyUninitialized;
                }
                (_, FlowStyle::PossiblyUninitialized | FlowStyle::Uninitialized) => {
                    return FlowStyle::PossiblyUninitialized;
                }
                // Unclear how to merge, default to None
                _ => {
                    merged = FlowStyle::Other;
                }
            }
        }
        merged
    }
}

#[derive(Debug, Clone)]
pub struct FlowInfo {
    /// The key to use if you need the value of this name.
    pub key: Idx<Key>,
    /// The default value - used to create Default bindings inside loops.
    /// - Always set to `key` when a flow is first created.
    /// - Set to `key` whenever a flow is updated outside of loops, but not inside.
    pub default: Idx<Key>,
    /// The style of this binding.
    pub style: FlowStyle,
}

/// Represent what we know about a class field based on the scope information
/// at the end of the class body.
pub enum ClassFieldInBody {
    InitializedByAssign(Expr),
    InitializedWithoutAssign,
    Uninitialized,
}

impl FlowInfo {
    fn new(key: Idx<Key>, style: Option<FlowStyle>) -> Self {
        Self {
            key,
            default: key,
            style: style.unwrap_or(FlowStyle::Other),
        }
    }

    /// Create a new FlowInfo after an update.
    ///
    /// Also return the `Idx<Key>` of the default binding, if we are updating inside a loop
    /// (and `None` if we are not in a loop).
    fn updated(
        &self,
        key: Idx<Key>,
        style: Option<FlowStyle>,
        in_loop: bool,
    ) -> (Self, Option<Idx<Key>>) {
        let default = if in_loop { Some(self.default) } else { None };
        (
            Self {
                key,
                default: default.unwrap_or(key),
                style: style.unwrap_or_else(|| self.style.clone()),
            },
            default,
        )
    }

    pub fn as_initial_value(&self) -> ClassFieldInBody {
        match &self.style {
            FlowStyle::ClassField {
                initial_value: Some(e),
            } => ClassFieldInBody::InitializedByAssign(e.clone()),
            // This is only reachable via `AnnAssign` with no value.
            FlowStyle::ClassField {
                initial_value: None,
            } => ClassFieldInBody::Uninitialized,
            // All other styles (e.g. function def, import) indicate we do have
            // a value, but it is not coming from a simple style.
            _ => ClassFieldInBody::InitializedWithoutAssign,
        }
    }
}

/// Because of complications related both to recursion in the binding graph and to
/// the need for efficient representations, Pyrefly relies on multiple different integer
/// indexes used to refer to classes and retrieve different kinds of binding information.
///
/// This struct type captures the requirement that a class must always have all of these
/// indexes available, and provides a convenient way to pass them.
///
/// This is used in bindings code, but the solver depends on the invariant that all these
/// indexes, which get stored in various Binding nodes, must be valid.
#[derive(Debug, Clone)]
pub struct ClassIndices {
    pub def_index: ClassDefIndex,
    pub class_idx: Idx<KeyClass>,
    pub base_type_idx: Idx<KeyClassBaseType>,
    pub metadata_idx: Idx<KeyClassMetadata>,
    pub mro_idx: Idx<KeyClassMro>,
    pub synthesized_fields_idx: Idx<KeyClassSynthesizedFields>,
    pub variance_idx: Idx<KeyVariance>,
    pub consistent_override_check_idx: Idx<KeyConsistentOverrideCheck>,
}

#[derive(Clone, Debug)]
pub struct ScopeClass {
    pub name: Identifier,
    pub indices: ClassIndices,
    attributes_from_recognized_methods: SmallMap<Name, SmallMap<Name, InstanceAttribute>>,
    attributes_from_other_methods: SmallMap<Name, SmallMap<Name, InstanceAttribute>>,
}

impl ScopeClass {
    pub fn new(name: Identifier, indices: ClassIndices) -> Self {
        Self {
            name,
            indices,
            attributes_from_recognized_methods: SmallMap::new(),
            attributes_from_other_methods: SmallMap::new(),
        }
    }

    pub fn add_attributes_defined_by_method(
        &mut self,
        method_name: Name,
        attributes: SmallMap<Name, InstanceAttribute>,
    ) {
        if is_attribute_defining_method(&method_name, &self.name.id) {
            self.attributes_from_recognized_methods
                .insert(method_name, attributes);
        } else {
            self.attributes_from_other_methods
                .insert(method_name, attributes);
        }
    }

    /// Produces triples (hashed_attr_name, MethodThatSetsAttr, attribute) for all assignments
    /// to `self.<attr_name>` in methods.
    ///
    /// We iterate recognized methods first, which - assuming that the first result is the one
    /// used in our class logic, which is the case - ensures both that we don't produce
    /// unnecessary errors about attributes implicitly defined in unrecognized methods
    /// and that the types inferred from recognized methods take precedence.
    pub fn method_defined_attributes(
        self,
    ) -> impl Iterator<Item = (Hashed<Name>, MethodThatSetsAttr, InstanceAttribute)> {
        Self::iter_attributes(self.attributes_from_recognized_methods, true).chain(
            Self::iter_attributes(self.attributes_from_other_methods, false),
        )
    }

    fn iter_attributes(
        attrs: SmallMap<Name, SmallMap<Name, InstanceAttribute>>,
        recognized_attribute_defining_method: bool,
    ) -> impl Iterator<Item = (Hashed<Name>, MethodThatSetsAttr, InstanceAttribute)> {
        {
            attrs.into_iter().flat_map(move |(method_name, attrs)| {
                attrs.into_iter_hashed().map(move |(name, attr)| {
                    (
                        name,
                        MethodThatSetsAttr {
                            method_name: method_name.clone(),
                            recognized_attribute_defining_method,
                        },
                        attr,
                    )
                })
            })
        }
    }
}

fn is_attribute_defining_method(method_name: &Name, class_name: &Name) -> bool {
    if method_name == &dunder::INIT
        || method_name == &dunder::INIT_SUBCLASS
        || method_name == &dunder::NEW
        || method_name == &dunder::POST_INIT
    {
        true
    } else {
        (class_name.contains("Test") || class_name.contains("test"))
            && is_test_setup_method(method_name)
    }
}

fn is_test_setup_method(method_name: &Name) -> bool {
    match method_name.as_str() {
        "asyncSetUp" | "async_setUp" | "setUp" | "_setup" | "_async_setup"
        | "async_with_context" | "with_context" | "setUpClass" => true,
        _ => false,
    }
}

/// Things we collect from inside a function
#[derive(Default, Clone, Debug)]
pub struct YieldsAndReturns {
    pub returns: Vec<(Idx<Key>, StmtReturn)>,
    pub yields: Vec<(Idx<KeyYield>, ExprYield)>,
    pub yield_froms: Vec<(Idx<KeyYieldFrom>, ExprYieldFrom)>,
}

#[derive(Clone, Debug)]
pub struct InstanceAttribute(
    pub ExprOrBinding,
    pub Option<Idx<KeyAnnotation>>,
    pub TextRange,
);

#[derive(Clone, Debug)]
pub struct ScopeMethod {
    pub name: Identifier,
    pub self_name: Option<Identifier>,
    pub instance_attributes: SmallMap<Name, InstanceAttribute>,
    pub yields_and_returns: YieldsAndReturns,
    pub is_async: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ScopeFunction {
    pub yields_and_returns: YieldsAndReturns,
    pub is_async: bool,
}

#[derive(Clone, Debug)]
pub enum ScopeKind {
    Annotation,
    Class(ScopeClass),
    Comprehension,
    Function(ScopeFunction),
    Method(ScopeMethod),
    Module,
}

#[derive(Clone, Debug, Display, Copy)]
pub enum LoopExit {
    NeverRan,
    #[display("break")]
    Break,
    #[display("continue")]
    Continue,
}

/// Flow snapshots for all possible exitpoints from a loop.
#[derive(Clone, Debug)]
pub struct Loop(pub Vec<(LoopExit, Flow)>);

#[derive(Clone, Debug)]
pub struct Scope {
    pub range: TextRange,
    /// Things that are defined in this scope, statically, e.g. `x = 1` or `def f():`.
    /// Populated at the beginning before entering the scope.
    pub stat: Static,
    /// Things that are defined in this scope as they are reached.
    /// Initially starts out empty, but is populated as statements are encountered.
    /// Updated if there are multiple assignments. E.g. `x = 1; x = 2` would update the `x` binding twice.
    /// All flow bindings will have a static binding, _usually_ in this scope, but occasionally
    /// in a parent scope (e.g. for narrowing operations).
    pub flow: Flow,
    /// Are Flow types from containing scopes unreachable from this scope?
    ///
    /// Set when we enter a scope like a function body with deferred evaluation, where the
    /// values we might see from containing scopes may not match their current values.
    pub barrier: bool,
    /// What kind of scope is this? Used for a few purposes, including propagating
    /// information down from scopes (e.g. to figure out when we're in a class) and
    /// storing data from the current AST traversal for later analysis, especially
    /// self-attribute-assignments in methods.
    pub kind: ScopeKind,
    /// Stack of for/while loops we're in. Does not include comprehensions, which
    /// define a new scope.
    pub loops: Vec<Loop>,
}

impl Scope {
    fn new(range: TextRange, barrier: bool, kind: ScopeKind) -> Self {
        Self {
            range,
            stat: Default::default(),
            flow: Default::default(),
            barrier,
            kind,
            loops: Default::default(),
        }
    }

    pub fn annotation(range: TextRange) -> Self {
        Self::new(range, false, ScopeKind::Annotation)
    }

    pub fn class_body(range: TextRange, indices: ClassIndices, name: Identifier) -> Self {
        Self::new(
            range,
            false,
            ScopeKind::Class(ScopeClass::new(name, indices)),
        )
    }

    pub fn comprehension(range: TextRange) -> Self {
        Self::new(range, false, ScopeKind::Comprehension)
    }

    pub fn function(range: TextRange, is_async: bool) -> Self {
        Self::new(
            range,
            true,
            ScopeKind::Function(ScopeFunction {
                yields_and_returns: Default::default(),
                is_async,
            }),
        )
    }
    pub fn lambda(range: TextRange, is_async: bool) -> Self {
        Self::new(
            range,
            false,
            ScopeKind::Function(ScopeFunction {
                yields_and_returns: Default::default(),
                is_async,
            }),
        )
    }

    pub fn method(range: TextRange, name: Identifier, is_async: bool) -> Self {
        Self::new(
            range,
            true,
            ScopeKind::Method(ScopeMethod {
                name,
                self_name: None,
                instance_attributes: SmallMap::new(),
                yields_and_returns: Default::default(),
                is_async,
            }),
        )
    }

    fn module(range: TextRange) -> Self {
        Self::new(range, false, ScopeKind::Module)
    }
}

#[derive(Clone, Debug)]
struct ScopeTreeNode {
    scope: Scope,
    children: Vec<ScopeTreeNode>,
}

/// Determines if a range contains a position, inclusive on both ends.
fn contains_inclusive(range: TextRange, position: TextSize) -> bool {
    range.start() <= position && position <= range.end()
}

impl ScopeTreeNode {
    /// Return whether we hit a child scope with a barrier
    fn visit_available_definitions(
        &self,
        table: &BindingTable,
        position: TextSize,
        visitor: &mut impl FnMut(Idx<Key>),
    ) -> bool {
        if !contains_inclusive(self.scope.range, position) {
            return false;
        }
        let mut barrier = false;
        for node in &self.children {
            let hit_barrier = node.visit_available_definitions(table, position, visitor);
            barrier = barrier || hit_barrier
        }
        if !barrier {
            for info in self.scope.flow.info.values() {
                visitor(info.key);
            }
        }
        for (name, info) in &self.scope.stat.0 {
            if let Some(key) = table.types.0.key_to_idx(&info.as_key(name)) {
                visitor(key);
            }
        }
        barrier || self.scope.barrier
    }

    fn collect_available_definitions(
        &self,
        table: &BindingTable,
        position: TextSize,
        collector: &mut SmallSet<Idx<Key>>,
    ) {
        self.visit_available_definitions(table, position, &mut |key| {
            collector.insert(key);
        });
    }
}

/// Scopes keep track of the current stack of the scopes we are in.
#[derive(Clone, Debug)]
pub struct Scopes {
    scopes: Vec1<ScopeTreeNode>,
    /// When `keep_scope_tree` flag is on, the stack will maintain a tree of all the scopes
    /// throughout the program, even if the scope has already been popped. This is useful
    /// for autocomplete purposes.
    keep_scope_tree: bool,
}

impl Scopes {
    pub fn module(range: TextRange, keep_scope_tree: bool) -> Self {
        let module_scope = Scope::module(range);
        Self {
            scopes: Vec1::new(ScopeTreeNode {
                scope: module_scope,
                children: Vec::new(),
            }),
            keep_scope_tree,
        }
    }

    pub fn current(&self) -> &Scope {
        &self.scopes.last().scope
    }

    pub fn clone_current_flow(&self) -> Flow {
        self.current().flow.clone()
    }

    pub fn in_class_body(&self) -> bool {
        match self.current().kind {
            ScopeKind::Class(_) => true,
            _ => false,
        }
    }

    // Is this scope a class scope? If so, return the keys for the class and its metadata.
    pub fn get_class_and_metadata_keys(
        scope: &Scope,
    ) -> Option<(Idx<KeyClass>, Idx<KeyClassMetadata>)> {
        match &scope.kind {
            ScopeKind::Class(class_scope) => Some((
                class_scope.indices.class_idx,
                class_scope.indices.metadata_idx,
            )),
            _ => None,
        }
    }

    // Are we anywhere inside a class? If so, return the keys for the class and its metadata.
    // This function looks at enclosing scopes, unlike `current_class_and_metadata_keys`.
    pub fn enclosing_class_and_metadata_keys(
        &self,
    ) -> Option<(Idx<KeyClass>, Idx<KeyClassMetadata>)> {
        for scope in self.iter_rev() {
            if let Some(class_and_metadata) = Self::get_class_and_metadata_keys(scope) {
                return Some(class_and_metadata);
            }
        }
        None
    }

    // Are we inside an async function or method?
    pub fn is_in_async_def(&self) -> bool {
        for scope in self.iter_rev() {
            match &scope.kind {
                ScopeKind::Function(function_scope) => {
                    return function_scope.is_async;
                }
                ScopeKind::Method(method_scope) => {
                    return method_scope.is_async;
                }
                _ => {}
            }
        }
        false
    }

    pub fn function_predecessor_indices(
        &self,
        name: &Name,
    ) -> Option<(Idx<Key>, Idx<KeyDecoratedFunction>)> {
        if let Some(flow) = self.current().flow.info.get(name)
            && let FlowStyle::FunctionDef(fidx, _) = flow.style
        {
            return Some((flow.key, fidx));
        }
        None
    }

    pub fn current_mut(&mut self) -> &mut Scope {
        &mut self.current_mut_node().scope
    }

    fn current_mut_node(&mut self) -> &mut ScopeTreeNode {
        self.scopes.last_mut()
    }

    /// There is only one scope remaining, return it.
    pub fn finish(self) -> ScopeTrace {
        let (a, b) = self.scopes.split_off_last();
        assert_eq!(a.len(), 0);
        ScopeTrace(b)
    }

    pub fn push(&mut self, scope: Scope) {
        self.scopes.push(ScopeTreeNode {
            scope,
            children: Vec::new(),
        });
    }

    pub fn pop(&mut self) -> Scope {
        let ScopeTreeNode { scope, children } = self.scopes.pop().unwrap();
        if self.keep_scope_tree {
            self.current_mut_node().children.push(ScopeTreeNode {
                scope: scope.clone(),
                children,
            });
        }
        scope
    }

    pub fn push_function_scope(
        &mut self,
        range: TextRange,
        name: &Identifier,
        in_class: bool,
        is_async: bool,
    ) {
        if in_class {
            self.push(Scope::method(range, name.clone(), is_async));
        } else {
            self.push(Scope::function(range, is_async));
        }
    }

    pub fn pop_function_scope(&mut self) -> (YieldsAndReturns, Option<SelfAssignments>) {
        match self.pop().kind {
            ScopeKind::Method(method_scope) => (
                method_scope.yields_and_returns,
                Some(SelfAssignments {
                    method_name: method_scope.name.id,
                    instance_attributes: method_scope.instance_attributes,
                }),
            ),
            ScopeKind::Function(function_scope) => (function_scope.yields_and_returns, None),
            unexpected => unreachable!("Tried to pop a function scope, but got {unexpected:?}"),
        }
    }

    pub fn iter_rev(&self) -> impl ExactSizeIterator<Item = &Scope> {
        self.scopes.iter().map(|node| &node.scope).rev()
    }

    fn iter_rev_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Scope> {
        self.scopes.iter_mut().map(|node| &mut node.scope).rev()
    }

    /// In methods, we track assignments to `self` attribute targets so that we can
    /// be aware of class fields implicitly defined in methods.
    ///
    /// We currently apply this logic in all methods, although downstream code will
    /// often complain if an attribute is implicitly defined outside of methods
    /// (like constructors) that we recognize as always being called.
    ///
    /// Returns `true` if the attribute was a self attribute.
    pub fn record_self_attr_assign(
        &mut self,
        x: &ExprAttribute,
        value: ExprOrBinding,
        annotation: Option<Idx<KeyAnnotation>>,
    ) -> bool {
        for scope in self.iter_rev_mut() {
            if let ScopeKind::Method(method_scope) = &mut scope.kind
                && let Some(self_name) = &method_scope.self_name
                && matches!(&*x.value, Expr::Name(name) if name.id == self_name.id)
            {
                if !method_scope.instance_attributes.contains_key(&x.attr.id) {
                    method_scope.instance_attributes.insert(
                        x.attr.id.clone(),
                        InstanceAttribute(value, annotation, x.attr.range()),
                    );
                }
                return true;
            }
        }
        false
    }

    pub fn loop_depth(&self) -> usize {
        self.current().loops.len()
    }

    /// Set the flow info to bind `name` to `key`, maybe with `FlowStyle` `style`
    ///
    /// - If `style` is `None`, then:
    ///   - Preserve the existing style, when updating an existing name.
    ///   - Use `FlowStyle::Other`, when inserting a new name.
    /// - Maybe return the `Idx<Key>` of the default binding:
    ///   - Return it if this is an update of an existing name, inside a loop.
    ///   - For insert of a new name or if we are not in a loop, return `None`.
    ///
    /// A caller of this function promises to create a binding for `key`; the
    /// binding may not exist yet (it might depend on the returned default).
    ///
    /// TODO(grievejia): Properly separate out `FlowStyle` from the indices
    pub fn upsert_flow_info(
        &mut self,
        name: Hashed<&Name>,
        key: Idx<Key>,
        style: Option<FlowStyle>,
    ) -> Option<Idx<Key>> {
        let in_loop = self.loop_depth() != 0;
        match self.current_mut().flow.info.entry_hashed(name.cloned()) {
            Entry::Vacant(e) => {
                e.insert(FlowInfo::new(key, style));
                None
            }
            Entry::Occupied(mut e) => {
                let (updated, default) = e.get().updated(key, style, in_loop);
                *e.get_mut() = updated;
                default
            }
        }
    }

    fn get_flow_info(&self, name: &Name) -> Option<&FlowInfo> {
        let name = Hashed::new(name);
        for scope in self.iter_rev() {
            if let Some(flow) = scope.flow.info.get_hashed(name) {
                return Some(flow);
            }
        }
        None
    }

    fn get_static_info(&self, name: &Name, should_skip_current_scope: bool) -> Option<&StaticInfo> {
        let name = Hashed::new(name);
        let mut iter = self.iter_rev();
        if should_skip_current_scope {
            iter.next();
        }
        for scope in iter {
            if let Some(info) = scope.stat.0.get_hashed(name) {
                return Some(info);
            }
        }
        None
    }

    /// Get the flow style for `name`, depending on whether `name` is used in a
    /// static type.
    ///
    /// If we can find a flow info for `name`, return its style. Otherwise, we
    /// check the static type information to see if we have a uninitialized
    /// binding, in which case, `FlowStyle::Uninitialized` is returned.
    /// Otherwise we return `FlowStyle::Other` to indicate no information
    /// available.
    pub fn get_flow_style(&self, name: &Name, used_in_static_type: bool) -> &FlowStyle {
        match self.get_flow_info(name) {
            Some(flow) => &flow.style,
            None => {
                // If the name is used for static type information, we can look
                // at the current scope.
                // Otherwise, we should skip the current scope, because it may
                // permit a name to be used before it is defined.
                if self.get_static_info(name, !used_in_static_type).is_some() {
                    // If we have a static binding, then we are in a scope where
                    // the name is defined, so we can return Other.
                    &FlowStyle::Other
                } else {
                    // If we don't have a static binding, then we are in a scope
                    // where the name is not defined, so we return
                    // Uninitialized.
                    &FlowStyle::Uninitialized
                }
            }
        }
    }

    /// Look up either `name` or `base_name.name` in the current scope, assuming we are
    /// in the module with name `module_name`. If it is a `SpecialExport`, return it (otherwise None)
    pub fn as_special_export(
        &self,
        name: &Name,
        base_name: Option<&Name>,
        current_module: ModuleName,
    ) -> Option<SpecialExport> {
        if let Some(base_name) = base_name {
            // Check to see whether there's an imported module `base_name` such that `base_name.name`
            // is a special export.
            let special = SpecialExport::new(name)?;
            let flow = self.get_flow_info(base_name)?;
            match &flow.style {
                FlowStyle::MergeableImport(m) | FlowStyle::ImportAs(m) => {
                    if special.defined_in(*m) {
                        Some(special)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        } else {
            // Check to see whether `name` is a special export; either it must be
            // defined in the current module, or be an imported name from some other module.
            let flow = self.get_flow_info(name)?;
            match &flow.style {
                FlowStyle::Import(m, upstream_name) => {
                    let special = SpecialExport::new(upstream_name)?;
                    if special.defined_in(*m) {
                        Some(special)
                    } else {
                        None
                    }
                }
                _ => {
                    let special = SpecialExport::new(name)?;
                    if special.defined_in(current_module) {
                        Some(special)
                    } else {
                        None
                    }
                }
            }
        }
    }

    pub fn add_to_current_static(
        &mut self,
        name: Name,
        range: TextRange,
        symbol_kind: SymbolKind,
        ann: Option<Idx<KeyAnnotation>>,
        is_declaration: bool,
    ) {
        self.current_mut()
            .stat
            .add(name, range, symbol_kind, ann, is_declaration);
    }

    pub fn add_lvalue_to_current_static(&mut self, x: &Expr) {
        self.current_mut().stat.expr_lvalue(x);
    }

    /// Add a loop exit point to the current innermost loop with the current flow.
    ///
    /// Return a bool indicating whether we were in a loop (if we weren't, we do nothing).
    pub fn add_loop_exitpoint(&mut self, exit: LoopExit) -> bool {
        let scope = self.current_mut();
        let flow = scope.flow.clone();
        if let Some(innermost) = scope.loops.last_mut() {
            innermost.0.push((exit, flow));
            scope.flow.has_terminated = true;
            true
        } else {
            false
        }
    }

    pub fn swap_current_flow_with(&mut self, flow: &mut Flow) {
        mem::swap(&mut self.current_mut().flow, flow);
    }

    pub fn replace_current_flow(&mut self, mut flow: Flow) -> Flow {
        mem::swap(&mut self.current_mut().flow, &mut flow);
        flow
    }

    pub fn mark_flow_termination(&mut self) {
        self.current_mut().flow.has_terminated = true;
    }

    pub fn finish_current_loop(&mut self) -> Loop {
        assert!(self.loop_depth() > 0);
        self.current_mut().loops.pop().unwrap()
    }

    /// Whenever we enter the scope of a method *and* we see a matching
    /// parameter, we record the name of it so that we can detect `self` assignments
    /// that might define class fields.
    pub fn set_self_name_if_applicable(&mut self, self_name: Option<Identifier>) {
        if let Scope {
            kind: ScopeKind::Method(method_scope),
            ..
        } = self.current_mut()
        {
            method_scope.self_name = self_name;
        }
    }

    /// Whenever we exit a function definition scope that was a method where we accumulated
    /// assignments to `self`, we need to record those assignments on the parent class scope;
    /// they may later be used to define class fields.
    pub fn record_self_assignments_if_applicable(
        &mut self,
        self_assignments: Option<SelfAssignments>,
    ) {
        if let Some(self_assignments) = self_assignments
            && let ScopeKind::Class(class_scope) = &mut self.current_mut().kind
        {
            class_scope.add_attributes_defined_by_method(
                self_assignments.method_name,
                self_assignments.instance_attributes,
            );
        }
    }

    fn current_yields_and_returns_mut(&mut self) -> Option<&mut YieldsAndReturns> {
        for scope in self.iter_rev_mut() {
            match &mut scope.kind {
                ScopeKind::Function(scope) => return Some(&mut scope.yields_and_returns),
                ScopeKind::Method(scope) => return Some(&mut scope.yields_and_returns),
                _ => {}
            }
        }
        None
    }

    /// Record a return in the enclosing function body there is one.
    ///
    /// Return `None` if this succeeded and Some(rejected_return) if we are at the top-level
    pub fn record_or_reject_return(
        &mut self,
        ret: CurrentIdx,
        x: StmtReturn,
    ) -> Result<(), (CurrentIdx, StmtReturn)> {
        match self.current_yields_and_returns_mut() {
            Some(yields_and_returns) => {
                yields_and_returns.returns.push((ret.into_idx(), x));
                Ok(())
            }
            None => Err((ret, x)),
        }
    }

    /// Record a yield in the enclosing function body there is one.
    ///
    /// Return `None` if this succeeded and Some(rejected_yield) if we are at the top-level
    pub fn record_or_reject_yield(
        &mut self,
        idx: Idx<KeyYield>,
        x: ExprYield,
    ) -> Result<(), ExprYield> {
        match self.current_yields_and_returns_mut() {
            Some(yields_and_returns) => {
                yields_and_returns.yields.push((idx, x));
                Ok(())
            }
            None => Err(x),
        }
    }

    /// Record a yield in the enclosing function body there is one.
    ///
    /// Return `None` if this succeeded and Some(rejected_yield) if we are at the top-level
    pub fn record_or_reject_yield_from(
        &mut self,
        idx: Idx<KeyYieldFrom>,
        x: ExprYieldFrom,
    ) -> Result<(), ExprYieldFrom> {
        match self.current_yields_and_returns_mut() {
            Some(yields_and_returns) => {
                yields_and_returns.yield_froms.push((idx, x));
                Ok(())
            }
            None => Err(x),
        }
    }

    /// Insert an annotation pulled from some ancestor scope for a name
    /// defined by a `global` or `nonlocal` declaration.
    pub fn set_annotation_for_mutable_capture(
        &mut self,
        name: Hashed<&Name>,
        ann: Option<Idx<KeyAnnotation>>,
    ) {
        if ann.is_some()
            && let Some(current_scope_info) = self.current_mut().stat.0.get_mut_hashed(name)
        {
            current_scope_info.annot = ann;
        }
    }
}

#[derive(Clone, Debug)]
pub struct ScopeTrace(ScopeTreeNode);

impl ScopeTrace {
    pub fn toplevel_scope(&self) -> &Scope {
        &self.0.scope
    }

    pub fn available_definitions(
        &self,
        table: &BindingTable,
        position: TextSize,
    ) -> SmallSet<Idx<Key>> {
        let mut collector = SmallSet::new();
        self.0
            .collect_available_definitions(table, position, &mut collector);
        collector
    }

    pub fn definition_at_position<'a>(
        &self,
        table: &'a BindingTable,
        position: TextSize,
    ) -> Option<&'a Key> {
        let mut definition = None;
        self.0
            .visit_available_definitions(table, position, &mut |idx| {
                let key = table.types.0.idx_to_key(idx);
                match key {
                    Key::Definition(short_identifier)
                        if short_identifier.range().contains_inclusive(position) =>
                    {
                        definition = Some(key);
                    }
                    _ => {}
                }
            });
        definition
    }
}
