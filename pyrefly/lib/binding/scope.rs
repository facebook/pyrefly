/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::max;
use std::collections::HashSet;
use std::fmt::Debug;
use std::mem;

use itertools::Either;
use itertools::Itertools;
use parse_display::Display;
use pyrefly_graph::index::Idx;
use pyrefly_python::ast::Ast;
use pyrefly_python::dunder;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::nesting_context::NestingContext;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::suggest::best_suggestion;
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

use crate::binding::binding::Binding;
use crate::binding::binding::ClassFieldDefinition;
use crate::binding::binding::ExprOrBinding;
use crate::binding::binding::Key;
use crate::binding::binding::KeyAbstractClassCheck;
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
use crate::binding::binding::MethodSelfKind;
use crate::binding::binding::MethodThatSetsAttr;
use crate::binding::binding::NarrowUseLocation;
use crate::binding::bindings::BindingTable;
use crate::binding::bindings::BindingsBuilder;
use crate::binding::bindings::CurrentIdx;
use crate::binding::bindings::InitializedInFlow;
use crate::binding::expr::Usage;
use crate::binding::function::SelfAssignments;
use crate::binding::narrow::NarrowOps;
use crate::export::definitions::Definition;
use crate::export::definitions::DefinitionStyle;
use crate::export::definitions::Definitions;
use crate::export::definitions::MutableCaptureKind;
use crate::export::exports::ExportLocation;
use crate::export::exports::LookupExport;
use crate::export::special::SpecialExport;
use crate::module::module_info::ModuleInfo;
use crate::types::class::ClassDefIndex;
use crate::types::type_info::JoinStyle;

/// The result of looking up a name in the current scope stack for a read
/// operation.
#[derive(Debug)]
pub enum NameReadInfo {
    /// A normal key bound in the current flow. The key is always already in the bindings table.
    ///
    /// I may be "possibly uninitialized", meaning there is some upstream branching control
    /// flow such that I am not defined in at least one branch.
    Flow {
        idx: Idx<Key>,
        initialized: InitializedInFlow,
    },
    /// The name is an anywhere-style lookup. If it came from a non-barrier scope
    /// relative to the current one, this means it is uninitialized; otherwise we
    /// assume delayed evaluation (e.g. inside a function you may call functions defined
    /// below it) and treat the read as initialized.
    Anywhere {
        key: Key,
        initialized: InitializedInFlow,
    },
    /// No such name is defined in the current scope stack.
    NotFound,
}

/// The result of a successful lookup of a name for a write operation.
#[derive(Debug)]
pub struct NameWriteInfo {
    /// The annotation associated with this name in the current scope stack, if
    /// any. Used both for contextual typing and because write operations must
    /// have values assignable to the annotated type.
    pub annotation: Option<Idx<KeyAnnotation>>,
    /// If this name has multiple assignments - in which case we need to create an
    /// `Anywhere` binding and record each assignment in it's Phi binding - this is
    /// the text range used for the `Anywhere`.
    ///
    /// If this name only has one assignment, we will skip the `Anywhere` as
    /// an optimization, and this field will be `None`.
    pub anywhere_range: Option<TextRange>,
}

#[derive(Clone, Debug)]
pub enum MutableCaptureError {
    /// We can't find the name at all
    NotFound,
    /// We expected the name to be in an enclosing, non-global scope, but it's not
    NonlocalScope,
    /// This variable was assigned before the nonlocal declaration
    AssignedBeforeNonlocal,
    /// This variable was assigned before the global declaration
    AssignedBeforeGlobal,
}

impl MutableCaptureError {
    pub fn message(&self, name: &Identifier) -> String {
        match self {
            Self::NotFound => format!("Could not find name `{name}`"),
            Self::NonlocalScope => {
                format!("Found `{name}`, but it is coming from the global scope")
            }
            Self::AssignedBeforeNonlocal => {
                format!(
                    "`{name}` was assigned in the current scope before the nonlocal declaration"
                )
            }
            Self::AssignedBeforeGlobal => {
                format!("`{name}` was assigned in the current scope before the global declaration")
            }
        }
    }
}

/// A name defined in a module, which needs to be convertable to an export.
#[derive(Debug)]
pub enum Exportable {
    /// The typical case: this name has key `Key` in the flow at the end of
    /// the module, and may or may not be annotated.
    Initialized(Idx<Key>, Option<Idx<KeyAnnotation>>),
    /// This case occurs if a name is missing from the flow at the end of the
    /// module - for example it might be a name defined only in a branch that
    /// raises.
    ///
    /// We still need export behavior to be well-defined so we use an
    /// anywhere-style lookup for this case.
    Uninitialized(Key),
}

/// Many names may map to the same TextRange (e.g. from foo import *).
/// But no other static will point at the same TextRange.
#[derive(Default, Clone, Debug)]
struct Static(SmallMap<Name, StaticInfo>);

#[derive(Clone, Debug)]
struct StaticInfo {
    range: TextRange,
    style: StaticStyle,
}

#[derive(Clone, Debug)]
enum StaticStyle {
    /// I have multiple definitions, lookups should be anywhere-style.
    ///
    /// If I have annotations, this is the first one.
    Anywhere(Option<Idx<KeyAnnotation>>),
    /// I am a mutable capture of a name defined in some enclosing scope.
    MutableCapture(MutableCapture),
    /// I have a single definition, possibly annotated.
    SingleDef(Option<Idx<KeyAnnotation>>),
    /// I am an ImplicitGlobal definition.
    ImplicitGlobal,
    /// I am defined only by delete statements, with no other definitions.
    Delete,
    /// I am either a module import, like `import foo`, or a name defined by a wildcard import
    MergeableImport,
    /// I am a name that might be a scoped legacy type parameter.
    PossibleLegacyTParam,
}

/// Information about a mutable capture.
///
/// We track:
/// - The kind of the mutable capture
/// - The original definition, if any was found, otherwise an error from searching
///
/// TODO(stroxler): At the moment, if any actual assignments occur we will
/// get `Multiple` and the annotation will instead come from local code.
#[derive(Clone, Debug)]
struct MutableCapture {
    kind: MutableCaptureKind,
    original: Result<Box<StaticInfo>, MutableCaptureError>,
}

impl MutableCapture {
    fn annotation(&self) -> Option<Idx<KeyAnnotation>> {
        match &self.original {
            Result::Ok(static_info) => static_info.annotation(),
            Result::Err(_) => None,
        }
    }

    fn key_or_error(
        &self,
        name: &Name,
        kind: MutableCaptureKind,
    ) -> Result<Key, MutableCaptureError> {
        match &self.original {
            Result::Ok(static_info) => {
                if self.kind == kind {
                    Ok(static_info.as_key(name))
                } else {
                    // TODO(stroxler): this error isn't quite right but preserves existing behavior
                    Err(MutableCaptureError::AssignedBeforeNonlocal)
                }
            }
            Result::Err(e) => Err(e.clone()),
        }
    }
}

impl StaticStyle {
    fn annotation(&self) -> Option<Idx<KeyAnnotation>> {
        match self {
            Self::MutableCapture(capture) => capture.annotation(),
            Self::Anywhere(ann) | Self::SingleDef(ann) => *ann,
            Self::Delete
            | Self::ImplicitGlobal
            | Self::MergeableImport
            | Self::PossibleLegacyTParam => None,
        }
    }

    fn of_definition(
        name: Hashed<&Name>,
        definition: Definition,
        scopes: Option<&Scopes>,
        get_annotation_idx: &mut impl FnMut(ShortIdentifier) -> Idx<KeyAnnotation>,
    ) -> Self {
        if definition.needs_anywhere {
            Self::Anywhere(definition.annotation().map(get_annotation_idx))
        } else {
            match &definition.style {
                DefinitionStyle::Delete => Self::Delete,
                DefinitionStyle::MutableCapture(kind) => {
                    let original = scopes
                        .map_or(Result::Err(MutableCaptureError::NotFound), |scopes| {
                            scopes.look_up_name_for_mutable_capture(name, *kind)
                        });
                    Self::MutableCapture(MutableCapture {
                        kind: *kind,
                        original,
                    })
                }
                DefinitionStyle::Annotated(.., ann) => {
                    Self::SingleDef(Some(get_annotation_idx(*ann)))
                }
                DefinitionStyle::ImplicitGlobal => Self::ImplicitGlobal,
                DefinitionStyle::ImportModule(..) => Self::MergeableImport,
                DefinitionStyle::Unannotated(..)
                | DefinitionStyle::ImportAs(..)
                | DefinitionStyle::Import(..)
                | DefinitionStyle::ImportAsEq(..)
                | DefinitionStyle::ImportInvalidRelative => Self::SingleDef(None),
            }
        }
    }
}

impl StaticInfo {
    fn annotation(&self) -> Option<Idx<KeyAnnotation>> {
        self.style.annotation()
    }

    fn as_key(&self, name: &Name) -> Key {
        let short_identifier = || {
            ShortIdentifier::new(&Identifier {
                node_index: AtomicNodeIndex::default(),
                id: name.clone(),
                range: self.range,
            })
        };
        match self.style {
            StaticStyle::Anywhere(..) => Key::Anywhere(name.clone(), self.range),
            StaticStyle::Delete => Key::Delete(self.range),
            StaticStyle::MutableCapture(..) => Key::MutableCapture(short_identifier()),
            StaticStyle::MergeableImport => Key::Import(name.clone(), self.range),
            StaticStyle::ImplicitGlobal => Key::ImplicitGlobal(name.clone()),
            StaticStyle::SingleDef(..) => Key::Definition(short_identifier()),
            StaticStyle::PossibleLegacyTParam => Key::PossibleLegacyTParam(self.range),
        }
    }

    fn as_name_write_info(&self) -> NameWriteInfo {
        NameWriteInfo {
            annotation: self.annotation(),
            anywhere_range: if matches!(self.style, StaticStyle::Anywhere(..)) {
                Some(self.range)
            } else {
                None
            },
        }
    }
}

impl Static {
    fn upsert(&mut self, name: Hashed<Name>, range: TextRange, style: StaticStyle) {
        match self.0.entry_hashed(name) {
            Entry::Vacant(e) => {
                e.insert(StaticInfo { range, style });
            }
            Entry::Occupied(mut e) => {
                let found = e.get_mut();
                if matches!(style, StaticStyle::PossibleLegacyTParam) {
                    // This case is reachable when the same module has multiple attributes accessed
                    // on it, each of which produces a separate possible-legacy-tparam binding that
                    // narrows a different attribute.
                    //
                    // At the moment, this is a flaw in the design - we really should have all
                    // of the narrows, but that is currently not possible.
                    //
                    // For now, we'll let the last one win: this is arbitrary, but is probably more
                    // compatible with a future in which the `BindingsBuilder` tracks multiple attributes
                    // and combines them properly.
                    found.style = style;
                    found.range = range;
                } else {
                    let annotation = found.annotation().or_else(|| style.annotation());
                    // This logic is hit when a name is a parameter
                    //
                    // We try to handle parameters that are also bound by the body in the same way that `Definitions`
                    // would have handled an assignment that preceded all other definitions:
                    // - A parameter that only gets deleted is similar to a single-assingment name.
                    // - A mutable capture that is also a prameter is illegal, but for consistency
                    //   we treat it like a mutable capture.
                    match &style {
                        StaticStyle::Delete => {}
                        StaticStyle::MutableCapture(..) => {
                            found.style = style;
                            found.range = range;
                        }
                        _ => {
                            found.style = StaticStyle::Anywhere(annotation);
                        }
                    }
                }
            }
        }
    }

    fn stmts(
        &mut self,
        x: &[Stmt],
        module_info: &ModuleInfo,
        top_level: bool,
        lookup: &dyn LookupExport,
        sys_info: &SysInfo,
        get_annotation_idx: &mut impl FnMut(ShortIdentifier) -> Idx<KeyAnnotation>,
        scopes: Option<&Scopes>,
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
            d.inject_implicit_globals();
        }

        let mut wildcards = Vec::with_capacity(d.import_all.len());
        for (m, range) in d.import_all {
            if let Some(exports) = lookup.get(m).finding() {
                wildcards.push((range, exports.wildcard(lookup)));
            }
        }

        // Try and avoid rehashing while we insert, with a little bit of spare space
        let capacity_guess =
            d.definitions.len() + wildcards.iter().map(|x| x.1.len()).sum::<usize>();
        self.0.reserve(((capacity_guess * 5) / 4) + 25);

        for (name, definition) in d.definitions.into_iter_hashed() {
            // Note that this really is an upsert: there might already be a parameter of the
            // same name in this scope.
            let range = definition.range;
            let style =
                StaticStyle::of_definition(name.as_ref(), definition, scopes, get_annotation_idx);
            self.upsert(name, range, style);
        }
        for (range, wildcard) in wildcards {
            for name in wildcard.iter_hashed() {
                // TODO: semantics of import * and global var with same name
                self.upsert(name.cloned(), range, StaticStyle::MergeableImport)
            }
        }
    }

    fn expr_lvalue(&mut self, x: &Expr) {
        let mut add = |name: &ExprName| {
            self.upsert(
                Hashed::new(name.id.clone()),
                name.range,
                StaticStyle::SingleDef(None),
            )
        };
        Ast::expr_lvalue(x, &mut add);
    }
}

/// Flow-sensitive information about a name.
#[derive(Default, Clone, Debug)]
pub struct Flow {
    info: SmallMap<Name, FlowInfo>,
    // Have we seen control flow terminate?
    //
    // We continue to analyze the rest of the code after a flow terminates, but
    // we don't include terminated flows when merging after loops and branches.
    has_terminated: bool,
}

impl Flow {
    fn get_info(&self, name: &Name) -> Option<&FlowInfo> {
        self.info.get(name)
    }

    fn get_info_hashed(&self, name: Hashed<&Name>) -> Option<&FlowInfo> {
        self.info.get_hashed(name)
    }

    fn get_value(&self, name: &Name) -> Option<&FlowValue> {
        self.get_info(name)?.value()
    }

    fn get_value_hashed(&self, name: Hashed<&Name>) -> Option<&FlowValue> {
        self.get_info_hashed(name)?.value()
    }

    fn get_value_mut(&mut self, name: &Name) -> Option<&mut FlowValue> {
        self.info.get_mut(name)?.value_mut()
    }
}

/// Bound names can accumulate facet narrows from long assignment chains (e.g. huge
/// literal dictionaries). Limiting how many consecutive narrows we remember keeps
/// the flow graph shallow enough to avoid recursive explosions in the solver.
const MAX_FLOW_NARROW_DEPTH: usize = 512;

/// Flow information about a name. At least one of `narrow` and `value` will always
/// be non-None (although in some cases the value may have FlowStyle::Uninitialized,
/// meaning we track a type but are aware that the name is not bound at this point,
/// e.g. after a `del`)
#[derive(Debug, Clone)]
struct FlowInfo {
    /// The most recent value bound to this name, if any.
    value: Option<FlowValue>,
    /// The most recent narrow for this name, if any. Always set to `None` when
    /// `value` is re-bound.
    narrow: Option<FlowNarrow>,
    /// How many consecutive narrows have been recorded since the last value assignment.
    narrow_depth: usize,
    /// An idx used to wrap loop Phi with our guess at the type above the loop.
    /// - Always set to our current inferred type when a flow info is created
    /// - Updated whenever we update the inferred type outside of all loops, but not inside
    loop_prior: Idx<Key>,
}

/// The most recent value for a name. Used in several cases:
/// - Actual runtime assignments
/// - Certain cases where we track a type for unbound locals, such as after a bare
///   annotation like `x: int` or `del x` - these cases use `FlowStyle::Uninitialized`
/// - Loop recursion bindings in cases where a name was narrowed above a loop; we
///   don't know whether the name might be assigned in the loop so we have to assume
///   so; in that case we use `FlowStyle::LoopRecursion`
#[derive(Debug, Clone)]
struct FlowValue {
    idx: Idx<Key>,
    style: FlowStyle,
}

/// The most recent narrow for a name.
#[derive(Debug, Clone)]
struct FlowNarrow {
    idx: Idx<Key>,
}

impl FlowInfo {
    fn new_value(idx: Idx<Key>, style: FlowStyle) -> Self {
        Self {
            value: Some(FlowValue { idx, style }),
            narrow: None,
            narrow_depth: 0,
            loop_prior: idx,
        }
    }

    fn new_narrow(idx: Idx<Key>) -> Self {
        Self {
            value: None,
            narrow: Some(FlowNarrow { idx }),
            narrow_depth: 1,
            loop_prior: idx,
        }
    }

    fn updated_value(&self, idx: Idx<Key>, style: FlowStyle, in_loop: bool) -> Self {
        Self {
            value: Some(FlowValue { idx, style }),
            // Note that any existing narrow is wiped when a new value is bound.
            narrow: None,
            narrow_depth: 0,
            loop_prior: if in_loop { self.loop_prior } else { idx },
        }
    }

    fn updated_narrow(&self, idx: Idx<Key>, in_loop: bool) -> Self {
        Self {
            value: self.value.clone(),
            narrow: Some(FlowNarrow { idx }),
            narrow_depth: self.narrow_depth.saturating_add(1),
            loop_prior: if in_loop { self.loop_prior } else { idx },
        }
    }

    fn clear_narrow(&mut self) {
        self.narrow = None;
        self.narrow_depth = 0;
    }

    fn idx(&self) -> Idx<Key> {
        match (&self.narrow, &self.value) {
            (Some(FlowNarrow { idx, .. }), _) => *idx,
            (None, Some(FlowValue { idx, .. })) => *idx,
            (None, None) => unreachable!("A FlowInfo always has at least one of a narrow or value"),
        }
    }

    fn value(&self) -> Option<&FlowValue> {
        self.value.as_ref()
    }

    fn value_mut(&mut self) -> Option<&mut FlowValue> {
        self.value.as_mut()
    }

    fn initialized(&self) -> InitializedInFlow {
        self.value()
            .map_or(InitializedInFlow::Yes, |v| match v.style {
                FlowStyle::Uninitialized
                | FlowStyle::ClassField {
                    initial_value: None,
                } => InitializedInFlow::No,
                FlowStyle::PossiblyUninitialized => InitializedInFlow::Conditionally,
                _ => InitializedInFlow::Yes,
            })
    }
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
    /// Am I a class definition?
    ClassDef,
    /// The name is possibly uninitialized (perhaps due to merging branches)
    PossiblyUninitialized,
    /// The name was in an annotated declaration like `x: int` but not initialized
    Uninitialized,
    /// I'm a speculative binding for a name that was narrowed but not assigned above
    /// a loop. Because we don't yet know whether the name will be assigned, we have
    /// to assume it might be, so the loop recursion binding is treated as a `FlowValue`
    /// with this style.
    LoopRecursion,
}

impl FlowStyle {
    fn merged(
        always_defined: bool,
        mut styles: impl Iterator<Item = FlowStyle>,
        merge_style: MergeStyle,
    ) -> FlowStyle {
        let mut merged = styles.next().unwrap_or(FlowStyle::Other);
        for x in styles {
            match (&merged, x) {
                // If they're identical, keep it
                (l, r) if l == &r => {}
                // Uninitialized and initialized branches merge into PossiblyUninitialized
                (FlowStyle::Uninitialized | FlowStyle::PossiblyUninitialized, _)
                | (_, FlowStyle::Uninitialized | FlowStyle::PossiblyUninitialized) => {
                    return FlowStyle::PossiblyUninitialized;
                }
                // Unclear how to merge, default to None
                _ => {
                    merged = FlowStyle::Other;
                }
            }
        }
        if always_defined {
            merged
        } else {
            // If the name is missing in some flows, then it must be uninitialized in at
            // least some of them.
            match merged {
                FlowStyle::Uninitialized => FlowStyle::Uninitialized,
                _ => {
                    // A boolean expression like `(x := condition()) and (y := condition)`
                    // actually defines three downstream flows:
                    // - the normal downstream, where `y` is possibly uninitialized
                    // - the narrowed downstream, relevant if this is the test of an `if`,
                    //   where `y` is always defined.
                    // - the negated narrowed downstream (relevant if this were an `or`)
                    //
                    // We cannot currently model that in our bindings phase, and as a result
                    // we have to be lax about whether boolean ops define new names
                    match merge_style {
                        MergeStyle::BoolOp => FlowStyle::Other,
                        MergeStyle::Loop
                        | MergeStyle::LoopDefinitelyRuns
                        | MergeStyle::Exclusive
                        | MergeStyle::Inclusive => FlowStyle::PossiblyUninitialized,
                    }
                }
            }
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
    pub abstract_class_check_idx: Idx<KeyAbstractClassCheck>,
}

#[derive(Clone, Debug)]
struct ScopeClass {
    name: Identifier,
    indices: ClassIndices,
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
                            instance_or_class: attr.3,
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
    pub MethodSelfKind,
);

#[derive(Clone, Debug)]
struct ScopeMethod {
    name: Identifier,
    self_name: Option<Identifier>,
    instance_attributes: SmallMap<Name, InstanceAttribute>,
    parameters: SmallMap<Name, ParameterUsage>,
    yields_and_returns: YieldsAndReturns,
    is_async: bool,
    receiver_kind: MethodSelfKind,
}

#[derive(Clone, Debug)]
struct ScopeFunction {
    parameters: SmallMap<Name, ParameterUsage>,
    yields_and_returns: YieldsAndReturns,
    is_async: bool,
}

#[derive(Clone, Debug)]
struct ParameterUsage {
    range: TextRange,
    used: bool,
    allow_unused: bool,
}

#[derive(Clone, Debug)]
struct ImportUsage {
    range: TextRange,
    used: bool,
    /// Skip reporting this import as unused. This is true for star imports
    /// and __future__ imports, which have side effects even if not explicitly used.
    skip_unused_check: bool,
}

#[derive(Clone, Debug)]
struct VariableUsage {
    range: TextRange,
    used: bool,
}

#[derive(Clone, Debug)]
pub struct UnusedParameter {
    pub name: Name,
    pub range: TextRange,
}

#[derive(Clone, Debug)]
pub struct UnusedImport {
    pub name: Name,
    pub range: TextRange,
}

#[derive(Clone, Debug)]
pub struct UnusedVariable {
    pub name: Name,
    pub range: TextRange,
}

impl Default for ScopeFunction {
    fn default() -> Self {
        Self::new(false)
    }
}

impl ScopeFunction {
    fn new(is_async: bool) -> Self {
        Self {
            parameters: SmallMap::new(),
            yields_and_returns: Default::default(),
            is_async,
        }
    }
}

impl ScopeMethod {
    fn new(name: Identifier, is_async: bool) -> Self {
        Self {
            name,
            self_name: None,
            instance_attributes: SmallMap::new(),
            parameters: SmallMap::new(),
            yields_and_returns: Default::default(),
            is_async,
            receiver_kind: MethodSelfKind::Instance,
        }
    }
}

#[derive(Clone, Debug)]
enum ScopeKind {
    Annotation,
    Class(ScopeClass),
    Comprehension { is_generator: bool },
    Function(ScopeFunction),
    Method(ScopeMethod),
    Module,
    TypeAlias,
}

#[derive(Clone, Debug, Display, Copy)]
pub enum LoopExit {
    #[display("break")]
    Break,
    #[display("continue")]
    Continue,
}

/// Flow snapshots for all possible exitpoints from a loop.
#[derive(Clone, Debug)]
struct Loop {
    base: Flow,
    exits: Vec<(LoopExit, Flow)>,
    /// For PEP 765: The depth of finally-blocks that this loop was created in
    finally_depth: usize,
}

impl Loop {
    pub fn new(base: Flow, finally_depth: usize) -> Self {
        Self {
            base,
            exits: Default::default(),
            finally_depth,
        }
    }
}

/// Represents forks in control flow that contain branches. Used to
/// control how the final flow from merging branches behaves.
#[derive(Clone, Debug)]
pub struct Fork {
    /// The Flow that was live at the top of the fork
    base: Flow,
    /// The flow resulting from branches of the fork
    branches: Vec<Flow>,
    /// Fork operations involve non type-safe invariants around calling `start_branch` that are
    /// used to minimize flow clones.
    ///
    /// This bit allows us to panic instead of producing buggy analysis if a caller messes them up.
    branch_started: bool,
    /// A text range for the fork - used as part of the key construction when we merge the fork.
    range: TextRange,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum FlowBarrier {
    /// Allow flow information from containing scopes, and check for name initialization errors.
    AllowFlowChecked,
    /// Allow flow information from containing scopes, and skip checks for name initialization errors.
    AllowFlowUnchecked,
    BlockFlow,
}

#[derive(Clone, Debug)]
pub struct Scope {
    range: TextRange,
    /// Things that are defined in this scope, statically, e.g. `x = 1` or `def f():`.
    /// Populated at the beginning before entering the scope.
    stat: Static,
    /// Things that are defined in this scope as they are reached.
    /// Initially starts out empty, but is populated as statements are encountered.
    /// Updated if there are multiple assignments. E.g. `x = 1; x = 2` would update the `x` binding twice.
    /// All flow bindings will have a static binding, _usually_ in this scope, but occasionally
    /// in a parent scope (e.g. for narrowing operations).
    flow: Flow,
    /// Are Flow types from containing scopes unreachable from this scope?
    ///
    /// Set when we enter a scope like a function body with deferred evaluation, where the
    /// values we might see from containing scopes may not match their current values.
    flow_barrier: FlowBarrier,
    /// What kind of scope is this? Used for a few purposes, including propagating
    /// information down from scopes (e.g. to figure out when we're in a class) and
    /// storing data from the current AST traversal for later analysis, especially
    /// self-attribute-assignments in methods.
    kind: ScopeKind,
    /// Stack of for/while loops we're in. Does not include comprehensions, which
    /// define a new scope.
    loops: Vec<Loop>,
    /// Stack of branches we're in. Branches occur anywhere that we split and later
    /// merge flows, including boolean ops, ternary operators, if and match statements,
    /// and exception handlers
    forks: Vec<Fork>,
    /// Tracking imports in the current scope (module-level only)
    imports: SmallMap<Name, ImportUsage>,
    /// Tracking variables in the current scope (module, function, and method scopes)
    variables: SmallMap<Name, VariableUsage>,
    /// Depth of finally blocks we're in. Resets in new function scopes (PEP 765).
    finally_depth: usize,
}

impl Scope {
    fn new(range: TextRange, flow_barrier: FlowBarrier, kind: ScopeKind) -> Self {
        Self {
            range,
            stat: Default::default(),
            flow: Default::default(),
            flow_barrier,
            kind,
            loops: Default::default(),
            forks: Default::default(),
            imports: SmallMap::new(),
            variables: SmallMap::new(),
            finally_depth: 0,
        }
    }

    pub fn annotation(range: TextRange) -> Self {
        Self::new(range, FlowBarrier::AllowFlowChecked, ScopeKind::Annotation)
    }

    pub fn type_alias(range: TextRange) -> Self {
        Self::new(range, FlowBarrier::AllowFlowChecked, ScopeKind::TypeAlias)
    }

    pub fn class_body(range: TextRange, indices: ClassIndices, name: Identifier) -> Self {
        Self::new(
            range,
            FlowBarrier::AllowFlowChecked,
            ScopeKind::Class(ScopeClass::new(name, indices)),
        )
    }

    pub fn comprehension(range: TextRange, is_generator: bool) -> Self {
        Self::new(
            range,
            FlowBarrier::AllowFlowChecked,
            ScopeKind::Comprehension { is_generator },
        )
    }

    pub fn function(range: TextRange, is_async: bool) -> Self {
        Self::new(
            range,
            FlowBarrier::BlockFlow,
            ScopeKind::Function(ScopeFunction::new(is_async)),
        )
    }
    pub fn lambda(range: TextRange, is_async: bool) -> Self {
        Self::new(
            range,
            FlowBarrier::AllowFlowUnchecked,
            ScopeKind::Function(ScopeFunction::new(is_async)),
        )
    }

    pub fn method(range: TextRange, name: Identifier, is_async: bool) -> Self {
        Self::new(
            range,
            FlowBarrier::BlockFlow,
            ScopeKind::Method(ScopeMethod::new(name, is_async)),
        )
    }

    fn module(range: TextRange) -> Self {
        Self::new(range, FlowBarrier::AllowFlowChecked, ScopeKind::Module)
    }

    fn parameters_mut(&mut self) -> Option<&mut SmallMap<Name, ParameterUsage>> {
        match &mut self.kind {
            ScopeKind::Function(scope) => Some(&mut scope.parameters),
            ScopeKind::Method(scope) => Some(&mut scope.parameters),
            _ => None,
        }
    }

    fn class_and_metadata_keys(&self) -> Option<(Idx<KeyClass>, Idx<KeyClassMetadata>)> {
        match &self.kind {
            ScopeKind::Class(class_scope) => Some((
                class_scope.indices.class_idx,
                class_scope.indices.metadata_idx,
            )),
            _ => None,
        }
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
    /// Return any flow barrier we hit in a child scope
    fn visit_available_definitions(
        &self,
        table: &BindingTable,
        position: TextSize,
        visitor: &mut impl FnMut(Idx<Key>),
    ) -> FlowBarrier {
        if !contains_inclusive(self.scope.range, position) {
            return FlowBarrier::AllowFlowChecked;
        }
        let mut flow_barrier = FlowBarrier::AllowFlowChecked;
        for node in &self.children {
            let hit_barrier = node.visit_available_definitions(table, position, visitor);
            flow_barrier = max(flow_barrier, hit_barrier);
        }
        if flow_barrier < FlowBarrier::BlockFlow {
            for info in self.scope.flow.info.values() {
                if let Some(value) = info.value() {
                    visitor(value.idx);
                }
            }
        }
        for (name, info) in &self.scope.stat.0 {
            if let Some(key) = table.types.0.key_to_idx(&info.as_key(name)) {
                visitor(key);
            }
        }
        max(flow_barrier, self.scope.flow_barrier)
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

    fn current(&self) -> &Scope {
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

    pub fn in_function_scope(&self) -> bool {
        self.iter_rev()
            .any(|scope| matches!(scope.kind, ScopeKind::Function(_) | ScopeKind::Method(_)))
    }

    /// Enter a finally block (PEP 765).
    pub fn enter_finally(&mut self) {
        self.current_mut().finally_depth += 1;
    }

    /// Exit a finally block (PEP 765).
    pub fn exit_finally(&mut self) {
        self.current_mut().finally_depth -= 1;
    }

    /// Check if we're in a finally block at the current scope level.
    /// This resets when entering a new function scope, so nested functions are OK.
    pub fn in_finally(&self) -> bool {
        self.current().finally_depth > 0
    }

    pub fn finally_depth(&self) -> usize {
        self.current().finally_depth
    }

    /// Check if we're inside a loop that was started inside the inner-most finally block
    pub fn loop_protects_from_finally_exit(&self) -> bool {
        self.current()
            .loops
            .last()
            .is_some_and(|l| l.finally_depth == self.current().finally_depth)
    }

    /// Are we currently in a class body. If so, return the keys for the class and its metadata.
    pub fn current_class_and_metadata_keys(
        &self,
    ) -> Option<(Idx<KeyClass>, Idx<KeyClassMetadata>)> {
        self.current().class_and_metadata_keys()
    }

    /// Are we anywhere inside a class? If so, return the keys for the class and its metadata.
    /// This function looks at enclosing scopes, unlike `current_class_and_metadata_keys`.
    pub fn enclosing_class_and_metadata_keys(
        &self,
    ) -> Option<(Idx<KeyClass>, Idx<KeyClassMetadata>)> {
        for scope in self.iter_rev() {
            if let Some(class_and_metadata) = scope.class_and_metadata_keys() {
                return Some(class_and_metadata);
            }
        }
        None
    }

    /// Are we inside an async function or method?
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

    /// Check if a name is defined as a type parameter in any enclosing Annotation scope.
    pub fn name_shadows_enclosing_annotation_scope(&self, name: &Name) -> bool {
        // Skip the current scope, which we know isn't relevant to the check.
        for scope in self.iter_rev().skip(1) {
            if matches!(scope.kind, ScopeKind::Annotation) && scope.stat.0.get(name).is_some() {
                return true;
            }
        }
        false
    }

    pub fn function_predecessor_indices(
        &self,
        name: &Name,
    ) -> Option<(Idx<Key>, Idx<KeyDecoratedFunction>)> {
        if let Some(value) = self.current().flow.get_value(name)
            && let FlowStyle::FunctionDef(fidx, _) = value.style
        {
            return Some((value.idx, fidx));
        }
        None
    }

    fn current_mut(&mut self) -> &mut Scope {
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

    pub fn collect_module_unused_imports(&self) -> Vec<UnusedImport> {
        let module_scope = self.scopes.first();
        if !matches!(module_scope.scope.kind, ScopeKind::Module) {
            return Vec::new();
        }
        Self::collect_unused_imports(module_scope.scope.imports.clone())
    }

    pub fn init_current_static(
        &mut self,
        x: &[Stmt],
        module_info: &ModuleInfo,
        top_level: bool,
        lookup: &dyn LookupExport,
        sys_info: &SysInfo,
        get_annotation_idx: &mut impl FnMut(ShortIdentifier) -> Idx<KeyAnnotation>,
    ) {
        let mut initialize = |scope: &mut Scope, myself: Option<&Self>| {
            scope.stat.stmts(
                x,
                module_info,
                top_level,
                lookup,
                sys_info,
                get_annotation_idx,
                myself,
            );
            // Presize the flow, as its likely to need as much space as static
            scope.flow.info.reserve(scope.stat.0.capacity());
        };
        if top_level {
            // If we are in the top-level scope, all `global` / `nonlocal` directives fail, so we can
            // pass `None` to `initialize`
            let current = self.current_mut();
            initialize(current, None);
        } else {
            // If we are in any other scope, we want to pass `self` to `initialize`. To satisfy
            // the borrow checker, we pop the current scope first and then push it back after.
            let mut current = self.pop();
            initialize(&mut current, Some(self));
            self.push(current);
        }
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

    fn collect_unused_parameters(
        parameters: SmallMap<Name, ParameterUsage>,
    ) -> Vec<UnusedParameter> {
        parameters
            .into_iter()
            .filter_map(|(name, usage)| {
                if usage.used || usage.allow_unused {
                    None
                } else {
                    Some(UnusedParameter {
                        name,
                        range: usage.range,
                    })
                }
            })
            .collect()
    }

    fn collect_unused_imports(imports: SmallMap<Name, ImportUsage>) -> Vec<UnusedImport> {
        imports
            .into_iter()
            .filter_map(|(name, usage)| {
                if usage.used || usage.skip_unused_check {
                    None
                } else {
                    Some(UnusedImport {
                        name,
                        range: usage.range,
                    })
                }
            })
            .collect()
    }

    fn collect_unused_variables(variables: SmallMap<Name, VariableUsage>) -> Vec<UnusedVariable> {
        variables
            .into_iter()
            .filter_map(|(name, usage)| {
                if usage.used {
                    None
                } else {
                    Some(UnusedVariable {
                        name,
                        range: usage.range,
                    })
                }
            })
            .collect()
    }

    pub fn pop_function_scope(
        &mut self,
    ) -> (
        YieldsAndReturns,
        Option<SelfAssignments>,
        Vec<UnusedParameter>,
        Vec<UnusedVariable>,
    ) {
        let scope = self.pop();
        let unused_variables = Self::collect_unused_variables(scope.variables.clone());
        match scope.kind {
            ScopeKind::Method(method_scope) => (
                method_scope.yields_and_returns,
                Some(SelfAssignments {
                    method_name: method_scope.name.id,
                    instance_attributes: method_scope.instance_attributes,
                }),
                Self::collect_unused_parameters(method_scope.parameters),
                unused_variables,
            ),
            ScopeKind::Function(function_scope) => (
                function_scope.yields_and_returns,
                None,
                Self::collect_unused_parameters(function_scope.parameters),
                unused_variables,
            ),
            unexpected => unreachable!("Tried to pop a function scope, but got {unexpected:?}"),
        }
    }

    fn iter_rev(&self) -> impl ExactSizeIterator<Item = &Scope> {
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
                        InstanceAttribute(
                            value,
                            annotation,
                            x.attr.range(),
                            method_scope.receiver_kind,
                        ),
                    );
                }
                return true;
            }
        }
        false
    }

    pub fn method_that_sets_attr(&self, x: &ExprAttribute) -> Option<MethodThatSetsAttr> {
        let mut method_name: Option<Name> = None;
        let mut receiver_kind = MethodSelfKind::Instance;
        for scope in self.iter_rev() {
            match &scope.kind {
                ScopeKind::Method(method_scope) if method_name.is_none() => {
                    if let Some(self_name) = &method_scope.self_name
                        && matches!(&*x.value, Expr::Name(name) if name.id == self_name.id)
                    {
                        method_name = Some(method_scope.name.id.clone());
                        receiver_kind = method_scope.receiver_kind;
                    } else {
                        return None;
                    }
                }
                ScopeKind::Class(class_scope) => {
                    if let Some(method_name) = &method_name {
                        return Some(MethodThatSetsAttr {
                            method_name: method_name.clone(),
                            recognized_attribute_defining_method: is_attribute_defining_method(
                                method_name,
                                &class_scope.name.id,
                            ),
                            instance_or_class: receiver_kind,
                        });
                    }
                }
                _ => {}
            }
        }
        None
    }

    pub fn loop_depth(&self) -> usize {
        self.current().loops.len()
    }

    /// Check if a name is declared as global in the current scope.
    /// Returns the range of the global statement if found.
    pub fn get_global_declaration(&self, name: &str) -> Option<TextRange> {
        if let Some(static_info) = self.current().stat.0.get(&Name::new(name))
            && let StaticStyle::MutableCapture(MutableCapture {
                kind: MutableCaptureKind::Global,
                ..
            }) = &static_info.style
        {
            return Some(static_info.range);
        }
        None
    }

    /// Check if a name has a nonlocal binding in an enclosing scope.
    pub fn has_nonlocal_binding(&self, name: &str) -> bool {
        let name_obj = Name::new(name);
        // Skip the current scope and check enclosing scopes
        for scope in self.iter_rev().skip(1) {
            // Check if this name is defined in this scope (not as a mutable capture)
            if let Some(static_info) = scope.stat.0.get(&name_obj) {
                match &static_info.style {
                    // Don't count mutable captures as nonlocal bindings
                    StaticStyle::MutableCapture(..) => continue,
                    // Any other definition counts as a binding
                    _ => return true,
                }
            }
        }
        false
    }

    /// Check if we're currently in a comprehension scope (not a generator expression).
    pub fn in_comprehension(&self) -> bool {
        matches!(self.current().kind, ScopeKind::Comprehension { .. })
    }

    /// Check if we're in a synchronous comprehension.
    /// A comprehension is synchronous unless we're in an async function.
    pub fn in_sync_comprehension(&self) -> bool {
        if !self.in_comprehension() {
            return false;
        }
        // Check if any enclosing scope is an async function
        for scope in self.iter_rev().skip(1) {
            if let ScopeKind::Function(func_scope) = &scope.kind {
                return !func_scope.is_async;
            } else if let ScopeKind::Method(method_scope) = &scope.kind {
                return !method_scope.is_async;
            }
        }
        // If we didn't find a function, it's synchronous
        true
    }

    /// Check if we're in a generator expression scope.
    /// Generator expressions are created for `Expr::Generator` comprehensions.
    pub fn in_generator_expression(&self) -> bool {
        matches!(
            self.current().kind,
            ScopeKind::Comprehension { is_generator: true }
        )
    }

    /// Check if a name is a bound parameter in the current function scope.
    pub fn is_bound_parameter(&self, name: &str) -> bool {
        let name_obj = Name::new(name);
        // Check the current scope and enclosing scopes for a function with this parameter
        for scope in self.iter_rev() {
            match &scope.kind {
                ScopeKind::Function(func_scope) => {
                    return func_scope.parameters.contains_key(&name_obj);
                }
                ScopeKind::Method(method_scope) => {
                    return method_scope.parameters.contains_key(&name_obj);
                }
                // Don't look past module or class boundaries
                ScopeKind::Module | ScopeKind::Class(_) => return false,
                _ => {}
            }
        }
        false
    }

    /// Track a narrow for a name in the current flow. This should result from options
    /// that only narrow an existing value, not operations that assign a new value at runtime.
    ///
    /// A caller of this function promises to create a binding for `idx`.
    pub fn narrow_in_current_flow(&mut self, name: Hashed<&Name>, idx: Idx<Key>) {
        let in_loop = self.loop_depth() != 0;
        match self.current_mut().flow.info.entry_hashed(name.cloned()) {
            Entry::Vacant(e) => {
                e.insert(FlowInfo::new_narrow(idx));
            }
            Entry::Occupied(mut e) => {
                let mut info = e.get().clone();
                if info.narrow_depth >= MAX_FLOW_NARROW_DEPTH {
                    info.clear_narrow();
                }
                *e.get_mut() = info.updated_narrow(idx, in_loop);
            }
        }
    }

    /// Track the binding from assigning a name in the current flow. Here "define" means:
    /// - any operation that actually binds a value at runtime (e.g. `x = 5`,
    ///   `x := 5`, `for x in ...`)
    /// - annotated assignment `x: int` which we model in the flow (but we remember
    ///   that `x` is uninitialized)
    ///
    /// A caller of this function promises to create a binding for `idx`.
    ///
    /// Returns a `NameWriteInfo` with information that bindings code may need,
    /// e.g. to validate against annotations and/or keep track of `Anywhere` bindings.
    pub fn define_in_current_flow(
        &mut self,
        name: Hashed<&Name>,
        idx: Idx<Key>,
        style: FlowStyle,
    ) -> Option<NameWriteInfo> {
        let in_loop = self.loop_depth() != 0;
        match self.current_mut().flow.info.entry_hashed(name.cloned()) {
            Entry::Vacant(e) => {
                e.insert(FlowInfo::new_value(idx, style));
            }
            Entry::Occupied(mut e) => {
                *e.get_mut() = e.get().updated_value(idx, style, in_loop);
            }
        }
        let static_info = self.current().stat.0.get_hashed(name)?;
        Some(static_info.as_name_write_info())
    }

    /// Handle a delete operation by marking a name as uninitialized in this flow.
    ///
    /// Don't change the type if one is present - downstream we'll emit
    /// uninitialized local errors but keep using our best guess for the type.
    pub fn mark_as_deleted(&mut self, name: &Name) {
        if let Some(value) = self.current_mut().flow.get_value_mut(name) {
            value.style = FlowStyle::Uninitialized;
        }
    }

    fn get_flow_info(&self, name: &Name) -> Option<&FlowInfo> {
        let name = Hashed::new(name);
        for scope in self.iter_rev() {
            if let Some(flow) = scope.flow.get_info_hashed(name) {
                return Some(flow);
            }
        }
        None
    }

    /// Get the flow style for `name` in the current scope.
    ///
    /// Returns `None` if there is no current flow (which may mean the
    /// name is uninitialized in the current scope, or is not in scope at all).
    pub fn current_flow_style(&self, name: &Name) -> Option<FlowStyle> {
        Some(self.current().flow.get_info(name)?.value()?.style.clone())
    }

    /// Return the current binding index and flow style for `name`, if it exists
    /// in any enclosing scope.
    pub fn binding_idx_for_name(&self, name: &Name) -> Option<(Idx<Key>, FlowStyle)> {
        let info = self.get_flow_info(name)?;
        let value = info.value()?;
        Some((value.idx, value.style.clone()))
    }

    // This helper handles re-exported symbols during special export lookups
    fn lookup_special_export(
        &self,
        mut name: Name,
        mut module: ModuleName,
        lookup: &dyn LookupExport,
    ) -> Option<SpecialExport> {
        let mut seen = HashSet::new();
        let mut exports = lookup.get(module).finding()?.exports(lookup);
        loop {
            if let Some(special) = SpecialExport::new(&name)
                && special.defined_in(module)
            {
                return Some(special);
            }
            if !seen.insert(module) {
                break;
            }
            match exports.as_ref().get(&name)? {
                ExportLocation::ThisModule(export) => {
                    return export.special_export;
                }
                ExportLocation::OtherModule(other_module, original_name) => {
                    if let Some(original_name) = original_name {
                        name = original_name.clone();
                    }
                    module = *other_module;
                    exports = lookup.get(module).finding()?.exports(lookup);
                }
            }
        }
        None
    }

    /// Look up either `name` or `base_name.name` in the current scope, assuming we are
    /// in the module with name `module_name`. If it is a `SpecialExport`, return it (otherwise None)
    pub fn as_special_export(
        &self,
        name: &Name,
        base_name: Option<&Name>,
        current_module: ModuleName,
        lookup: &dyn LookupExport,
    ) -> Option<SpecialExport> {
        if let Some(base_name) = base_name {
            // Check to see whether there's an imported module `base_name` such that `base_name.name`
            // is a special export.
            let value = self.get_flow_info(base_name)?.value()?;
            match &value.style {
                FlowStyle::MergeableImport(m) | FlowStyle::ImportAs(m) => {
                    self.lookup_special_export(name.clone(), *m, lookup)
                }
                FlowStyle::Import(m, upstream_name) => {
                    self.lookup_special_export(upstream_name.clone(), *m, lookup)
                }
                _ => None,
            }
        } else {
            // Check to see whether `name` is a special export; either it must be
            // defined in the current module, or be an imported name from some other module.
            let value = self.get_flow_info(name)?.value()?;
            match &value.style {
                FlowStyle::MergeableImport(m) | FlowStyle::ImportAs(m) => {
                    self.lookup_special_export(name.clone(), *m, lookup)
                }
                FlowStyle::Import(m, upstream_name) => {
                    self.lookup_special_export(upstream_name.clone(), *m, lookup)
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

    /// Add a parameter to the current static.
    ///
    /// Callers must always define the name via a `Key::Definition` immediately
    /// afterward or downstream lookups may panic.
    pub fn add_parameter_to_current_static(
        &mut self,
        name: &Identifier,
        ann: Option<Idx<KeyAnnotation>>,
    ) {
        self.current_mut().stat.upsert(
            Hashed::new(name.id.clone()),
            name.range,
            StaticStyle::SingleDef(ann),
        )
    }

    pub fn register_parameter(&mut self, name: &Identifier, allow_unused: bool) {
        if let Some(parameters) = self.current_mut().parameters_mut() {
            parameters.insert(
                name.id.clone(),
                ParameterUsage {
                    range: name.range,
                    used: false,
                    allow_unused,
                },
            );
        }
    }

    pub fn mark_parameter_used(&mut self, name: &Name) {
        for scope in self.iter_rev_mut() {
            if let Some(parameters) = scope.parameters_mut()
                && let Some(info) = parameters.get_mut(name)
            {
                info.used = true;
                break;
            }
        }
    }

    pub fn register_import(&mut self, name: &Identifier) {
        self.register_import_internal(name, false);
    }

    pub fn register_import_with_star(&mut self, name: &Identifier) {
        self.register_import_internal(name, true);
    }

    pub fn register_future_import(&mut self, name: &Identifier) {
        self.register_import_internal(name, true);
    }

    fn register_import_internal(&mut self, name: &Identifier, skip_unused_check: bool) {
        if matches!(self.current().kind, ScopeKind::Module) {
            self.current_mut().imports.insert(
                name.id.clone(),
                ImportUsage {
                    range: name.range,
                    used: false,
                    skip_unused_check,
                },
            );
        }
    }

    pub fn mark_import_used(&mut self, name: &Name) {
        for scope in self.iter_rev_mut() {
            if let Some(info) = scope.imports.get_mut(name) {
                info.used = true;
                break;
            }
        }
    }

    pub fn register_variable(&mut self, name: &Identifier) {
        // Track variables in Module, Function, and Method scopes
        // Module-level variables won't be reported as unused since they can be imported
        // by other modules, but function/method-level variables will be reported
        if matches!(
            self.current().kind,
            ScopeKind::Module | ScopeKind::Function(_) | ScopeKind::Method(_)
        ) {
            // Preserve the `used` flag if the variable was already marked as used
            // This handles cases like `foo = foo + 1` in loops where the variable
            // is read before being reassigned
            let was_used = self
                .current()
                .variables
                .get(&name.id)
                .is_some_and(|usage| usage.used);
            self.current_mut().variables.insert(
                name.id.clone(),
                VariableUsage {
                    range: name.range,
                    used: was_used,
                },
            );
        }
    }

    pub fn mark_variable_used(&mut self, name: &Name) {
        for scope in self.iter_rev_mut() {
            if let Some(info) = scope.variables.get_mut(name) {
                info.used = true;
                break;
            }
        }
    }

    /// Add an intercepted possible legacy TParam - this is a name that's part
    /// of the scope, but only for static type lookups, and might potentially
    /// intercept the raw runtime value of a pre-PEP-695 legacy type variable
    /// to turn it into a quantified type parameter.
    pub fn add_possible_legacy_tparam(&mut self, name: &Identifier) {
        self.current_mut().stat.upsert(
            Hashed::new(name.id.clone()),
            name.range,
            StaticStyle::PossibleLegacyTParam,
        )
    }

    /// Add an adhoc name - if it does not already exist - to the current static
    /// scope. If the name already exists, nothing happens.
    ///
    /// Callers must always define the name via a `Key::Definition` immediately
    /// afterward or downstream lookups may panic.
    ///
    /// Used to bind names in comprehension and lambda scopes, where we
    /// don't have `Definitions` to work from so we discover the names during
    /// the main AST traversal in bindings.
    pub fn add_lvalue_to_current_static(&mut self, x: &Expr) {
        self.current_mut().stat.expr_lvalue(x);
    }

    /// Add a loop exit point to the current innermost loop with the current flow.
    ///
    /// Return a bool indicating whether we were in a loop (if we weren't, we do nothing).
    pub fn add_loop_exit(&mut self, exit: LoopExit) -> bool {
        let scope = self.current_mut();
        let flow = scope.flow.clone();
        if let Some(innermost) = scope.loops.last_mut() {
            innermost.exits.push((exit, flow));
            scope.flow.has_terminated = true;
            true
        } else {
            false
        }
    }

    fn finish_loop(&mut self) -> Loop {
        assert!(self.loop_depth() > 0);
        self.current_mut().loops.pop().unwrap()
    }

    pub fn swap_current_flow_with(&mut self, flow: &mut Flow) {
        mem::swap(&mut self.current_mut().flow, flow);
    }

    pub fn mark_flow_termination(&mut self) {
        self.current_mut().flow.has_terminated = true;
    }

    /// Whenever we enter the scope of a method *and* we see a matching
    /// parameter, we record the name of it so that we can detect `self` assignments
    /// that might define class fields.
    pub fn set_self_name_if_applicable(
        &mut self,
        self_name: Option<Identifier>,
        receiver_kind: MethodSelfKind,
    ) {
        if let Scope {
            kind: ScopeKind::Method(method_scope),
            ..
        } = self.current_mut()
        {
            method_scope.self_name = self_name;
            method_scope.receiver_kind = receiver_kind;
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

    /// Finish traversing a class body: pop both the class body scope and the annotation scope
    /// that wraps it, and extract the class field definitions.
    ///
    /// The resulting map of field definitions:
    /// - Includes both fields defined in the class body and implicit definitions
    ///   coming from self-assignment in methods. If both occur, only the class body
    ///   definition is tracked.
    /// - Panics if the current scope is not a class body.
    pub fn finish_class_and_get_field_definitions(
        &mut self,
    ) -> SmallMap<Name, (ClassFieldDefinition, TextRange)> {
        let mut field_definitions = SmallMap::new();
        let class_body = self.pop();
        let class_scope = {
            if let ScopeKind::Class(class_scope) = class_body.kind {
                class_scope
            } else {
                unreachable!("Expected class body scope, got {:?}", class_body.kind)
            }
        };
        self.pop(); // Also pop the annotation scope that wrapped the class body.
        class_body.stat.0.iter_hashed().for_each(
            |(name, static_info)| {
            if matches!(static_info.style, StaticStyle::MutableCapture(..)) {
                // Mutable captures are not actually owned by the class scope, and do not become attributes.
            } else if let Some(value) = class_body.flow.get_info_hashed(name).and_then(|flow| flow.value()) {
                let definition = match &value.style {
                    FlowStyle::FunctionDef(_, has_return_annotation) => ClassFieldDefinition::MethodLike {
                        definition: value.idx,
                        has_return_annotation: *has_return_annotation,
                    },
                    FlowStyle::ClassDef => ClassFieldDefinition::NestedClass {
                        definition: value.idx,
                    },
                    FlowStyle::ClassField {
                        initial_value: Some(e),
                    } => ClassFieldDefinition::AssignedInBody {
                        value: ExprOrBinding::Expr(e.clone()),
                        annotation: static_info.annotation(),
                    },
                    FlowStyle::ClassField {
                        initial_value: None,
                    } => ClassFieldDefinition::DeclaredByAnnotation {
                        annotation: static_info.annotation().unwrap_or_else(
                            || panic!("A class field known in the body but uninitialized always has an annotation.")
                        ),
                    },
                    _ => ClassFieldDefinition::DefinedWithoutAssign {
                        definition: value.idx,
                    },
                };
                field_definitions.insert_hashed(name.owned(), (definition, static_info.range));
            }
        });
        class_scope.method_defined_attributes().for_each(
            |(name, method, InstanceAttribute(value, annotation, range, _))| {
                if !field_definitions.contains_key_hashed(name.as_ref()) {
                    field_definitions.insert_hashed(
                        name,
                        (
                            ClassFieldDefinition::DefinedInMethod {
                                value,
                                annotation,
                                method,
                            },
                            range,
                        ),
                    );
                }
            },
        );
        field_definitions
    }

    /// Return a pair Some((method_name, class_key)) if we are currently in a method
    /// (if we are in nested classes, we'll get the innermost).
    ///
    /// Used to resolve `super()` behaviors.
    pub fn current_method_and_class(&self) -> Option<(Identifier, Idx<KeyClass>)> {
        let mut method_name = None;
        let mut class_key = None;
        for scope in self.iter_rev() {
            match &scope.kind {
                ScopeKind::Method(method_scope) => {
                    method_name = Some(method_scope.name.clone());
                }
                ScopeKind::Class(class_scope) if method_name.is_some() => {
                    class_key = Some(class_scope.indices.class_idx);
                    break;
                }
                _ => {}
            }
        }
        match (method_name, class_key) {
            (Some(method_name), Some(class_key)) => Some((method_name, class_key)),
            _ => None,
        }
    }

    /// Get the name of the (innermost) enclosing class, if any.
    pub fn enclosing_class_name(&self) -> Option<&Identifier> {
        for scope in self.iter_rev() {
            if let ScopeKind::Class(ScopeClass { name, .. }) = &scope.kind {
                return Some(name);
            }
        }
        None
    }

    pub fn in_module_or_class_top_level(&self) -> bool {
        matches!(self.current().kind, ScopeKind::Module | ScopeKind::Class(_))
    }

    /// Check whether the current flow has a module import at a given name.
    ///
    /// Used when binding imports, because the semantics of multiple imports from
    /// the same root (like `import foo.bar; import foo.baz`) are that the sub-modules
    /// will be added as attributes of `foo`.
    pub fn existing_module_import_at(&self, module_name: &Name) -> Option<Idx<Key>> {
        match self.current().flow.get_value(module_name) {
            Some(value) if matches!(value.style, FlowStyle::MergeableImport(..)) => Some(value.idx),
            _ => None,
        }
    }

    /// Helper for iterating over scopes in a way that respects class body visibility rules.
    fn visit_scopes<'a, T>(
        &'a self,
        mut visitor: impl FnMut(usize, &'a Scope, FlowBarrier) -> Option<T>,
    ) -> Option<T> {
        let mut flow_barrier = FlowBarrier::AllowFlowChecked;
        let is_current_scope_annotation = matches!(self.current().kind, ScopeKind::Annotation);
        for (lookup_depth, scope) in self.iter_rev().enumerate() {
            let is_class = matches!(scope.kind, ScopeKind::Class(_));
            // From https://docs.python.org/3/reference/executionmodel.html#resolution-of-names:
            //   The scope of names defined in a class block is limited to the
            //   class block; it does not extend to the code blocks of
            //   methods. This includes comprehensions and generator
            //   expressions, but it does not include annotation scopes, which
            //   have access to their enclosing class scopes.
            if is_class
                && !((lookup_depth == 0) || (is_current_scope_annotation && lookup_depth == 1))
            {
                // Note: class body scopes have `flow_barrier = AllowFlowChecked`, so skipping the flow_barrier update is okay.
                continue;
            }

            if let Some(result) = visitor(lookup_depth, scope, flow_barrier) {
                return Some(result);
            }

            flow_barrier = max(flow_barrier, scope.flow_barrier);
        }
        None
    }

    pub fn suggest_similar_name(&self, missing: &Name, position: TextSize) -> Option<Name> {
        let mut candidates: Vec<(&Name, usize)> = Vec::new();

        self.visit_scopes(|lookup_depth, scope, flow_barrier| {
            let is_class = matches!(scope.kind, ScopeKind::Class(_));

            if flow_barrier < FlowBarrier::BlockFlow {
                for candidate in scope.flow.info.keys() {
                    if let Some(static_info) = scope.stat.0.get(candidate)
                        && static_info.range.start() >= position
                    {
                        continue;
                    }
                    candidates.push((candidate, lookup_depth));
                }
            }

            if !is_class {
                for (candidate, static_info) in scope.stat.0.iter() {
                    if static_info.range.start() < position {
                        candidates.push((candidate, lookup_depth));
                    }
                }
            }
            None::<()>
        });

        best_suggestion(missing, candidates)
    }

    /// Look up the information needed to create a `Usage` binding for a read of a name
    /// in the current scope stack.
    pub fn look_up_name_for_read(&self, name: Hashed<&Name>) -> NameReadInfo {
        self.visit_scopes(|_, scope, flow_barrier| {
            let is_class = matches!(scope.kind, ScopeKind::Class(_));

            if let Some(flow_info) = scope.flow.get_info_hashed(name)
                && flow_barrier < FlowBarrier::BlockFlow
            {
                let initialized = if flow_barrier == FlowBarrier::AllowFlowUnchecked {
                    // Just assume the name is initialized without checking.
                    InitializedInFlow::Yes
                } else {
                    flow_info.initialized()
                };
                // Because class body scopes are dynamic, if we know that the the name is
                // definitely not initialized in the flow, we should skip it.
                if is_class && matches!(initialized, InitializedInFlow::No) {
                    return None;
                }
                return Some(NameReadInfo::Flow {
                    idx: flow_info.idx(),
                    initialized,
                });
            }
            // Class body scopes are dynamic, not static, so if we don't find a name in the
            // current flow we keep looking. In every other kind of scope, anything the Python
            // compiler has identified as local shadows enclosing scopes, so we should prefer
            // inner static lookups to outer flow lookups.
            if !is_class && let Some(static_info) = scope.stat.0.get_hashed(name) {
                let forward_ref_key = static_info.as_key(name.into_key());
                return Some(NameReadInfo::Anywhere {
                    key: forward_ref_key,
                    // If we look up static info from the a non-barrier scope because we didn't find
                    // flow, it is not initialized. PossibleLegacyTParam scope entries are an
                    // exception because they are synthesized scope entries that don't exist at all
                    // in the runtime; we treat them as always initialized to avoid false positives
                    // for uninitialized local checks in class bodies.
                    initialized: if flow_barrier > FlowBarrier::AllowFlowChecked
                        || matches!(static_info.style, StaticStyle::PossibleLegacyTParam)
                    {
                        InitializedInFlow::Yes
                    } else {
                        InitializedInFlow::No
                    },
                });
            }
            None
        })
        .unwrap_or(NameReadInfo::NotFound)
    }

    /// Look up a name for a mutable capture during initialization of static scope.
    ///
    /// Returns either a `StaticInfo` that we found, or an error indicating why we
    /// failed to find a match.
    fn look_up_name_for_mutable_capture(
        &self,
        name: Hashed<&Name>,
        kind: MutableCaptureKind,
    ) -> Result<Box<StaticInfo>, MutableCaptureError> {
        let found = match kind {
            MutableCaptureKind::Global => self
                .scopes
                .first()
                .scope
                .stat
                .0
                .get_hashed(name)
                .map(|static_info| Result::Ok(Box::new(static_info.clone()))),
            MutableCaptureKind::Nonlocal => self.iter_rev().find_map(|scope| {
                if matches!(scope.kind, ScopeKind::Class(..)) {
                    None
                } else {
                    scope
                        .stat
                        .0
                        .get_hashed(name)
                        .map(|static_info| match &static_info.style {
                            // If the enclosing name is a capture, look through it and also catch
                            // any mismatches between `nonlocal` and `global`.
                            StaticStyle::MutableCapture(MutableCapture {
                                kind, original, ..
                            }) => match kind {
                                MutableCaptureKind::Nonlocal => original.clone(),
                                MutableCaptureKind::Global => {
                                    Result::Err(MutableCaptureError::NonlocalScope)
                                }
                            },
                            // Otherwise, the enclosing name *is* the original, but we need
                            // to check whether we fell all the way back to the global scope.
                            _ => match scope.kind {
                                ScopeKind::Module => {
                                    Result::Err(MutableCaptureError::NonlocalScope)
                                }
                                _ => Result::Ok(Box::new(static_info.clone())),
                            },
                        })
                }
            }),
        };
        found.unwrap_or(Result::Err(MutableCaptureError::NotFound))
    }

    pub fn validate_mutable_capture_and_get_key(
        &self,
        name: Hashed<&Name>,
        kind: MutableCaptureKind,
    ) -> Result<Key, MutableCaptureError> {
        if self.current().flow.get_info_hashed(name).is_some() {
            return match kind {
                MutableCaptureKind::Global => Err(MutableCaptureError::AssignedBeforeGlobal),
                MutableCaptureKind::Nonlocal => Err(MutableCaptureError::AssignedBeforeNonlocal),
            };
        }
        match self.current().stat.0.get_hashed(name) {
            Some(StaticInfo {
                style: StaticStyle::MutableCapture(capture),
                ..
            }) => capture.key_or_error(name.into_key(), kind),
            Some(_) | None => Err(MutableCaptureError::NotFound),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ScopeTrace(ScopeTreeNode);

impl ScopeTrace {
    pub fn toplevel_scope(&self) -> &Scope {
        &self.0.scope
    }

    pub fn exportables(&self) -> SmallMap<Name, Exportable> {
        let mut exportables = SmallMap::new();
        let scope = self.toplevel_scope();
        for (name, static_info) in scope.stat.0.iter_hashed() {
            let exportable = match scope.flow.get_value_hashed(name) {
                Some(FlowValue { idx: key, .. }) => {
                    if let Some(ann) = static_info.annotation() {
                        Exportable::Initialized(*key, Some(ann))
                    } else {
                        Exportable::Initialized(*key, None)
                    }
                }
                None => Exportable::Uninitialized(static_info.as_key(name.into_key())),
            };
            exportables.insert_hashed(name.owned(), exportable);
        }
        exportables
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

/// What kind of merge are we performing? Most logic is shared across all merges,
/// but a few details diverge for different kinds of merges.
#[derive(Debug, Clone, Copy)]
enum MergeStyle {
    /// This is a loopback merge for the top of a loop; the base flow is part of the
    /// merge.
    Loop,
    /// This is a loopback merge for a loop that definitely runs at least once
    /// (e.g., `for _ in range(3)` or `for _ in [1, 2, 3]`). The base flow is NOT
    /// counted as a branch for the purpose of determining if names are always defined,
    /// since the loop body will always execute at least once.
    LoopDefinitelyRuns,
    /// This is a fork in which the current flow should be discarded - for example
    /// the end of an `if` statement with an `else` branch.
    ///
    /// The base flow is not part of the merge.
    Exclusive,
    /// This is a fork in which the current flow is part of the merge - for example
    /// after an `if` statement with no `else`, typically the current flow is
    /// the base flow after applying negated branch conditions.
    ///
    /// The base flow is not part of the merge.
    Inclusive,
    /// This is a merge of the flow from traversing an entire boolean op (`and`
    /// or `or`) with the flow when we exit early from the very first part. The
    /// base flow is not part of the merge.
    ///
    /// Distinct from [Branching] because we have to be more lax about
    /// uninitialized locals (see `FlowStyle::merge` for details).
    BoolOp,
}

impl MergeStyle {
    fn is_loop(self) -> bool {
        matches!(self, MergeStyle::Loop | MergeStyle::LoopDefinitelyRuns)
    }
}

struct MergeItem {
    base: Option<FlowInfo>,
    branches: Vec<FlowInfo>,
}

struct MergeItems(SmallMap<Name, MergeItem>);

impl MergeItems {
    pub fn new(presize_to: usize) -> Self {
        Self(SmallMap::with_capacity(presize_to))
    }

    pub fn add_base_flow_info(&mut self, name: Hashed<Name>, base: FlowInfo, n_branches: usize) {
        self.0.insert_hashed(
            name,
            MergeItem {
                base: Some(base),
                branches: Vec::with_capacity(n_branches),
            },
        );
    }

    pub fn add_branch_flow_info(
        &mut self,
        name: Hashed<Name>,
        branch: FlowInfo,
        n_branches: usize,
    ) {
        match self.0.entry_hashed(name) {
            Entry::Vacant(e) => {
                let mut branches = Vec::with_capacity(n_branches);
                branches.push(branch);
                e.insert(MergeItem {
                    base: None,
                    branches,
                });
            }
            Entry::Occupied(mut e) => e.get_mut().branches.push(branch),
        }
    }
}

impl<'a> BindingsBuilder<'a> {
    /// Create the idx of a merged type from the idxs of the branch types
    fn merge_idxs(
        &mut self,
        branch_idxs: SmallSet<Idx<Key>>,
        phi_idx: Idx<Key>,
        loop_prior: Option<Idx<Key>>,
        join_style: JoinStyle<Idx<Key>>,
    ) -> Idx<Key> {
        if branch_idxs.len() == 1 {
            // We hit this case if any of these are true:
            // - the name was defined in the base flow and no branch modified it
            // - we're in a loop and there were only narrows
            // - the name was defined in only one branch
            // In all three cases, we can avoid a Phi and just forward to the one idx.
            let idx = *branch_idxs.first().unwrap();
            self.insert_binding_idx(phi_idx, Binding::Forward(idx));
            idx
        } else if let Some(loop_prior) = loop_prior {
            self.insert_binding_idx(phi_idx, Binding::LoopPhi(loop_prior, branch_idxs));
            phi_idx
        } else {
            self.insert_binding_idx(phi_idx, Binding::Phi(join_style, branch_idxs));
            phi_idx
        }
    }

    /// Get the flow info for an item in the merged flow, which is a combination
    /// of the `phi_key` that will have the merged type information and the merged
    /// flow styles.
    ///
    /// The binding for the phi key is typically a Phi, but if this merge is from a loop
    /// we'll use a LoopPhi, and if all branches were the same we'll just use a
    /// Forward instead.
    ///
    /// The default value will depend on whether we are still in a loop after the
    /// current merge. If so, we preserve the existing default; if not, the
    /// merged phi is the new default used for downstream loops.
    fn merged_flow_info(
        &mut self,
        merge_item: MergeItem,
        phi_idx: Idx<Key>,
        merge_style: MergeStyle,
        n_branches: usize,
    ) -> FlowInfo {
        let base_idx = merge_item.base.as_ref().map(|base| base.idx());
        let mut flow_infos = merge_item.branches;
        // Track the number of branch values before adding base (for LoopDefinitelyRuns)
        let n_branch_flow_infos = flow_infos.len();
        // Track if base has a value for this name (for LoopDefinitelyRuns init check)
        let base_has_value = merge_item.base.as_ref().is_some_and(|b| b.value.is_some());
        // If this is a loop, we want to use the current default in any phis we produce,
        // and the base flow is part of the merge for type inference purposes.
        let loop_prior = if merge_style.is_loop()
            && let Some(base) = merge_item.base
        {
            let loop_prior = base.loop_prior;
            flow_infos.push(base);
            Some(loop_prior)
        } else {
            None
        };
        let merged_loop_prior = {
            let contained_in_loop = self.scopes.loop_depth() > 0;
            move |merged_idx| {
                if contained_in_loop && let Some(prior) = loop_prior {
                    prior
                } else {
                    merged_idx
                }
            }
        };
        // Collect the idxs.
        //
        // Skip over all branches whose value is the phi - this is only possible
        // in loops, and it benefits us by:
        // - Allowing us to skip over branches that either don't change the binding
        //   at all or only perform narrow operations. In many cases, this can
        //   allow us to avoid the loop recursion altogether.
        // - Ensuring that even if we cannot eliminate the Phi, it won't be directly
        //   recursive in itself (which just makes more work in the solver).
        //
        // Note that because the flow above the loop flows into the Phi, this
        // can never result in empty `branch_idxs`.
        //
        // We keep track separately of `value_idxs` and `branch_idxs` so that
        // we know whether to treat the Phi binding as a value or a narrow - it's
        // a narrow only when all the value idxs are the same.
        let mut value_idxs = SmallSet::with_capacity(flow_infos.len());
        let mut branch_idxs = SmallSet::with_capacity(flow_infos.len());
        let mut styles = Vec::with_capacity(flow_infos.len());
        let mut n_values = 0;
        for flow_info in flow_infos.into_iter() {
            let branch_idx = flow_info.idx();
            if let Some(v) = flow_info.value {
                n_values += 1;
                if v.idx == phi_idx {
                    continue;
                }
                if value_idxs.insert(v.idx) {
                    // An invariant in Pyrefly is that we only set style when we
                    // set a value, so duplicate value_idxs always have the same style.
                    styles.push(v.style);
                }
            }
            branch_idxs.insert(branch_idx);
        }
        // For LoopDefinitelyRuns, a name is always defined if:
        // - It was defined before the loop (base_has_value), OR
        // - It's defined in all loop body branches (since the loop definitely runs at least once)
        // For regular loops and other merges, a name is always defined if it's in all branches.
        let this_name_always_defined = match merge_style {
            MergeStyle::LoopDefinitelyRuns => base_has_value || n_branch_flow_infos == n_branches,
            _ => n_values == n_branches,
        };
        match value_idxs.len() {
            // If there are no values, then this name isn't assigned at all
            // and is only narrowed (it's most likely a capture, but could be
            // a local if the code we're analyzing is buggy)
            0 => {
                let merged_idx = self.merge_idxs(
                    branch_idxs,
                    phi_idx,
                    loop_prior,
                    base_idx.map_or(JoinStyle::SimpleMerge, JoinStyle::NarrowOf),
                );
                FlowInfo {
                    value: None,
                    narrow: Some(FlowNarrow { idx: merged_idx }),
                    narrow_depth: 1,
                    loop_prior: merged_loop_prior(merged_idx),
                }
            }
            // If there is exactly one value (after discarding the phi itself,
            // for a loop), then the phi should be treated as a narrow, not a
            // value, and the value should continue to point at upstream.
            1 => {
                let merged_idx = self.merge_idxs(
                    branch_idxs,
                    phi_idx,
                    loop_prior,
                    base_idx.map_or(JoinStyle::SimpleMerge, JoinStyle::NarrowOf),
                );
                FlowInfo {
                    value: Some(FlowValue {
                        idx: *value_idxs.first().unwrap(),
                        style: FlowStyle::merged(
                            this_name_always_defined,
                            styles.into_iter(),
                            merge_style,
                        ),
                    }),
                    narrow: Some(FlowNarrow { idx: merged_idx }),
                    narrow_depth: 1,
                    loop_prior: merged_loop_prior(merged_idx),
                }
            }
            // If there are multiple values, then the phi should be treated
            // as a value (it may still include narrowed type information,
            // but it is not reducible to just narrows).
            _ => {
                let merged_idx = self.merge_idxs(
                    branch_idxs,
                    phi_idx,
                    loop_prior,
                    base_idx.map_or(JoinStyle::SimpleMerge, JoinStyle::ReassignmentOf),
                );
                FlowInfo {
                    value: Some(FlowValue {
                        idx: merged_idx,
                        style: FlowStyle::merged(
                            this_name_always_defined,
                            styles.into_iter(),
                            merge_style,
                        ),
                    }),
                    narrow: None,
                    narrow_depth: 0,
                    loop_prior: merged_loop_prior(merged_idx),
                }
            }
        }
    }

    fn merge_flow(
        &mut self,
        base: Flow,
        mut branches: Vec<Flow>,
        range: TextRange,
        merge_style: MergeStyle,
    ) {
        // Include the current flow in the merge if the merge style calls for it.
        if merge_style.is_loop() || matches!(merge_style, MergeStyle::Inclusive) {
            branches.push(mem::take(&mut self.scopes.current_mut().flow));
        }

        // Short circuit when there is only one flow. Note that we can never short
        // circuit for loops, because (a) we need to merge with the base flow, and
        // (b) we have already promised the phi keys so we'll panic if we short-circuit.
        if !merge_style.is_loop() && branches.len() == 1 {
            self.scopes.current_mut().flow = branches.pop().unwrap();
            return;
        }

        // We normally only merge the live branches (where control flow is not
        // known to terminate), but if nothing is live we still need to fill in
        // the Phi keys and potentially analyze downstream code, so in that case
        // we'll use the terminated branches.
        let (terminated_branches, live_branches): (Vec<_>, Vec<_>) =
            branches.into_iter().partition(|flow| flow.has_terminated);
        let has_terminated = live_branches.is_empty() && !merge_style.is_loop();
        let flows = if has_terminated {
            terminated_branches
        } else {
            live_branches
        };

        // For a regular loop, we merge the base so there's one extra branch being merged.
        // For LoopDefinitelyRuns, we don't count the base as an extra branch because we
        // know the loop body will definitely execute at least once.
        let n_branches = flows.len()
            + if matches!(merge_style, MergeStyle::Loop) {
                1
            } else {
                0
            };

        // Collect all the branches into a `MergeItem` per name we need to merge
        let mut merge_items = MergeItems::new(flows.first().unwrap_or(&base).info.len());
        for (name, info) in base.info.into_iter_hashed() {
            merge_items.add_base_flow_info(name, info, n_branches)
        }
        for flow in flows {
            for (name, info) in flow.info.into_iter_hashed() {
                merge_items.add_branch_flow_info(name, info, n_branches)
            }
        }

        // For each name and merge item, produce the merged FlowInfo for our new Flow
        let mut merged_flow_infos = SmallMap::with_capacity(merge_items.0.len());
        for (name, merge_item) in merge_items.0.into_iter_hashed() {
            let phi_idx = self.idx_for_promise(Key::Phi(name.key().clone(), range));
            merged_flow_infos.insert_hashed(
                name,
                self.merged_flow_info(merge_item, phi_idx, merge_style, n_branches),
            );
        }

        // The resulting flow has terminated only if all branches had terminated.
        let flow = Flow {
            info: merged_flow_infos,
            has_terminated,
        };
        self.scopes.current_mut().flow = flow
    }

    /// Helper for loops, inserts a phi key for every name in the given flow.
    fn insert_phi_keys(
        &mut self,
        mut flow: Flow,
        range: TextRange,
        exclude_names: &SmallSet<Name>,
    ) -> Flow {
        for (name, info) in flow.info.iter_mut() {
            if exclude_names.contains(name) {
                continue;
            }
            // We are promising to insert a bidning for this key when we merge the flow
            let phi_idx = self.idx_for_promise(Key::Phi(name.clone(), range));
            match &mut info.value {
                Some(value) => {
                    value.idx = phi_idx;
                }
                None => {
                    // Because we don't yet know whether the name might be assigned, we have to
                    // treat the phi as a value rather than a narrow here.
                    info.value = Some(FlowValue {
                        idx: phi_idx,
                        style: FlowStyle::LoopRecursion,
                    });
                    info.narrow = None;
                }
            }
        }
        flow
    }

    /// Set up a loop: preserve the base flow and push the loop to the current
    /// scope's `loops`, set up loop phi keys, and bind any narrow ops from the
    /// loop header.
    ///
    /// Names in `loop_header_targets` will not get phi keys - this is used for loop
    /// variables that are unconditionally reassigned in `for` loop headers
    pub fn setup_loop(&mut self, range: TextRange, loop_header_targets: &SmallSet<Name>) {
        let finally_depth = self.scopes.finally_depth();
        let base = mem::take(&mut self.scopes.current_mut().flow);
        // To account for possible assignments to existing names in a loop, we
        // speculatively insert phi keys upfront.
        self.scopes.current_mut().flow =
            self.insert_phi_keys(base.clone(), range, loop_header_targets);
        self.scopes
            .current_mut()
            .loops
            .push(Loop::new(base, finally_depth));
    }

    pub fn teardown_loop(
        &mut self,
        range: TextRange,
        narrow_ops: &NarrowOps,
        orelse: Vec<Stmt>,
        parent: &NestingContext,
        is_while_true: bool,
        loop_definitely_runs: bool,
    ) {
        let finished_loop = self.scopes.finish_loop();
        let (breaks, other_exits): (Vec<Flow>, Vec<Flow>) = finished_loop
            .exits
            .into_iter()
            .partition_map(|(exit, flow)| match exit {
                LoopExit::Break => Either::Left(flow),
                LoopExit::Continue => Either::Right(flow),
            });
        let base_if_breaks = if breaks.is_empty() {
            None
        } else {
            Some(finished_loop.base.clone())
        };
        // We associate a range to the non-`break` exits from the loop; it doesn't matter much what
        // it is as long as it's different from the loop's range.
        let other_range = TextRange::new(range.start(), range.start());
        // Create the loopback merge, which is the flow at the top of the loop.
        // Use LoopDefinitelyRuns when we know the loop will execute at least once.
        let merge_style = if loop_definitely_runs {
            MergeStyle::LoopDefinitelyRuns
        } else {
            MergeStyle::Loop
        };
        self.merge_flow(finished_loop.base, other_exits, range, merge_style);
        // When control falls off the end of a loop (either the `while` test fails or the loop
        // finishes), we're at the loopback flow but the test (if there is one) is negated.
        self.bind_narrow_ops(
            &narrow_ops.negate(),
            NarrowUseLocation::Span(other_range),
            &Usage::Narrowing(None),
        );
        self.stmts(orelse, parent);
        // Exiting from a break skips past any `else`, so we merge them after, and the
        // test is not negated in flows coming from breaks.
        //
        // If this is a `while` loop with a statically true test like `while true`, then we
        // also know that breaks are the only way to exit, so we drop the current flow,
        // which is actually unreachable.
        //
        // TODO(stroxler): in the `is_while_true` case, empty breaks might have implications
        // for flow termination and/or `NoReturn` behaviors, we should investigate.
        if let Some(base) = base_if_breaks {
            if is_while_true {
                self.merge_flow(base, breaks, other_range, MergeStyle::Exclusive)
            } else {
                self.merge_flow(base, breaks, other_range, MergeStyle::Inclusive)
            }
        }
    }

    pub fn add_loop_exitpoint(&mut self, exit: LoopExit) {
        self.scopes.add_loop_exit(exit);
    }

    /// Start a new fork in control flow (e.g. an if/else, match statement, etc)
    ///
    /// The end state of this is involves an empty flow not initialized for
    /// analyzing a branch, callers must call `start_branch` before proceeding
    /// with analysis.
    pub fn start_fork(&mut self, range: TextRange) {
        let scope = self.scopes.current_mut();
        let mut base = Flow::default();
        mem::swap(&mut base, &mut scope.flow);
        scope.forks.push(Fork {
            base,
            branches: Default::default(),
            branch_started: false,
            range,
        })
    }

    /// Set the current flow to a copy of the current Fork's base so we can analyze a branch.
    /// Panics if no flow is active.
    pub fn start_branch(&mut self) {
        let scope = self.scopes.current_mut();
        let fork = scope.forks.last_mut().unwrap();
        fork.branch_started = true;
        scope.flow = fork.base.clone();
    }

    /// Abandon a branch we began without including it in the merge. Used for a few cases
    /// where we need to analyze a test, but we then determine statically that the branch
    /// is unreachable in a way that should not be analyzed (e.g. python version and platform
    /// gates).
    pub fn abandon_branch(&mut self) {
        let scope = self.scopes.current_mut();
        let fork = scope.forks.last_mut().unwrap();
        // Not needed but a ram optimization: frees the current flow which isn't needed.
        scope.flow = Flow::default();
        fork.branch_started = false;
    }

    /// Finish a branch in the current fork: save the branch, reset the flow to `base`.
    /// Panics if called when no fork is active.
    ///
    /// The end state of this is involves an empty flow not initialized for
    /// analyzing a branch, callers must call `start_branch` before proceeding
    /// with analysis.
    ///
    /// Panics if `start_branch` was not used to initialize the flow since
    /// `start_fork` / `finish_branch`.
    pub fn finish_branch(&mut self) {
        let scope = self.scopes.current_mut();
        let fork = scope.forks.last_mut().unwrap();
        assert!(
            fork.branch_started,
            "No branch started - did you forget to call `start_branch`?"
        );
        let mut flow = Flow::default();
        mem::swap(&mut scope.flow, &mut flow);
        fork.branches.push(flow);
        fork.branch_started = false;
    }

    fn finish_fork_impl(
        &mut self,
        negated_prev_ops_if_nonexhaustive: Option<&NarrowOps>,
        is_bool_op: bool,
    ) {
        let fork = self.scopes.current_mut().forks.pop().unwrap();
        assert!(
            !fork.branch_started,
            "A branch is started - did you forget to call `finish_branch`?"
        );
        let branches = fork.branches;
        if let Some(negated_prev_ops) = negated_prev_ops_if_nonexhaustive {
            self.scopes.current_mut().flow = fork.base.clone();
            self.bind_narrow_ops(
                negated_prev_ops,
                // Generate a range that is distinct from other use_ranges of the same narrow.
                NarrowUseLocation::End(fork.range),
                &Usage::Narrowing(None),
            );
            self.merge_flow(fork.base, branches, fork.range, MergeStyle::Inclusive);
        } else {
            self.merge_flow(
                fork.base,
                branches,
                fork.range,
                if is_bool_op {
                    MergeStyle::BoolOp
                } else {
                    MergeStyle::Exclusive
                },
            );
        }
    }

    /// Finish an exhaustive fork (one that does not include the base flow),
    /// popping it and setting flow to the merge result.
    ///
    /// Panics if called when no fork is active, or if a branch is started (which
    /// means the caller forgot to call `finish_branch` and is always a bug).
    pub fn finish_exhaustive_fork(&mut self) {
        self.finish_fork_impl(None, false)
    }

    /// Finish a non-exhaustive fork in which the base flow is part of the merge. It negates
    /// the branch-choosing narrows by applying `negated_prev_ops` to base before merging, which
    /// is important so that we can preserve any cases where a termanating branch has permanently
    /// narrowed the type (e.g. an early return when an optional variable is None).
    ///
    /// Panics if called when no fork is active, or if a branch is started (which
    /// means the caller forgot to call `finish_branch` and is always a bug).
    pub fn finish_non_exhaustive_fork(&mut self, negated_prev_ops: &NarrowOps) {
        self.finish_fork_impl(Some(negated_prev_ops), false)
    }

    /// Finish the fork for a boolean operation. This requires lax handling of
    /// possibly-uninitialized locals, see the inline comment in `FlowStyle::merge`.
    pub fn finish_bool_op_fork(&mut self) {
        self.finish_fork_impl(None, true)
    }

    /// Finish a `MatchOr`, which behaves like an exhaustive fork except that we know
    /// only some of the base flow cases will get here, which means we should preserve
    /// all narrows.
    pub fn finish_match_or_fork(&mut self) {
        // TODO(stroxler): At the moment these are the same, but once we start eliminating
        // narrows aggressively we will need to handle this case differently
        self.finish_exhaustive_fork();
    }

    pub fn start_fork_and_branch(&mut self, range: TextRange) {
        self.start_fork(range);
        self.start_branch();
    }

    pub fn next_branch(&mut self) {
        self.finish_branch();
        self.start_branch();
    }
}
