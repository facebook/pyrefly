/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;

use dupe::Dupe;
use itertools::Either;
use pyrefly_derive::TypeEq;
use pyrefly_derive::VisitMut;
use pyrefly_util::assert_bytes;
use pyrefly_util::assert_words;
use pyrefly_util::display::DisplayWith;
use pyrefly_util::display::DisplayWithCtx;
use pyrefly_util::display::commas_iter;
use pyrefly_util::uniques::Unique;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprSubscript;
use ruff_python_ast::ExprYield;
use ruff_python_ast::ExprYieldFrom;
use ruff_python_ast::Identifier;
use ruff_python_ast::StmtAugAssign;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::TypeParams;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::alt::class::class_field::ClassField;
use crate::alt::class::class_metadata::BaseClass;
use crate::alt::class::variance_inference::VarianceMap;
use crate::alt::solve::TypeFormContext;
use crate::alt::types::class_metadata::ClassMetadata;
use crate::alt::types::class_metadata::ClassSynthesizedFields;
use crate::alt::types::decorated_function::DecoratedFunction;
use crate::alt::types::legacy_lookup::LegacyTypeParameterLookup;
use crate::alt::types::yields::YieldFromResult;
use crate::alt::types::yields::YieldResult;
use crate::binding::bindings::Bindings;
use crate::binding::narrow::NarrowOp;
use crate::common::symbol_kind::SymbolKind;
use crate::dunder;
use crate::graph::index::Idx;
use crate::module::module_info::ModuleInfo;
use crate::module::module_name::ModuleName;
use crate::module::short_identifier::ShortIdentifier;
use crate::types::annotation::Annotation;
use crate::types::class::Class;
use crate::types::class::ClassDefIndex;
use crate::types::class::ClassFieldProperties;
use crate::types::equality::TypeEq;
use crate::types::quantified::QuantifiedKind;
use crate::types::stdlib::Stdlib;
use crate::types::tuple::Tuple;
use crate::types::type_info::TypeInfo;
use crate::types::types::Type;
use crate::types::types::Var;

assert_words!(Key, 5);
assert_words!(KeyExpect, 1);
assert_words!(KeyExport, 3);
assert_words!(KeyClass, 1);
assert_words!(KeyClassField, 4);
assert_bytes!(KeyClassSynthesizedFields, 4);
assert_bytes!(KeyAnnotation, 12);
assert_bytes!(KeyClassMetadata, 4);
assert_words!(KeyLegacyTypeParam, 1);
assert_words!(KeyYield, 1);
assert_words!(KeyYieldFrom, 1);
assert_words!(KeyFunction, 1);

assert_words!(Binding, 9);
assert_words!(BindingExpect, 8);
assert_words!(BindingAnnotation, 13);
assert_words!(BindingClass, 22);
assert_words!(BindingClassMetadata, 8);
assert_words!(BindingClassField, 26);
assert_bytes!(BindingClassSynthesizedFields, 4);
assert_bytes!(BindingLegacyTypeParam, 4);
assert_words!(BindingYield, 3);
assert_words!(BindingYieldFrom, 3);
assert_words!(BindingFunction, 21);

pub trait Keyed: Hash + Eq + Clone + DisplayWith<ModuleInfo> + Debug + Ranged + 'static {
    const EXPORTED: bool = false;
    type Value: Debug + DisplayWith<Bindings>;
    type Answer: Clone + Debug + Display + TypeEq + VisitMut<Type>;
}

/// Should be equivalent to Keyed<EXPORTED=true>.
/// Once `associated_const_equality` is stabilised, can switch to that.
pub trait Exported: Keyed {}

impl Keyed for Key {
    type Value = Binding;
    type Answer = TypeInfo;
}
impl Keyed for KeyExpect {
    type Value = BindingExpect;
    type Answer = EmptyAnswer;
}
impl Keyed for KeyClass {
    type Value = BindingClass;
    type Answer = NoneIfRecursive<Class>;
}
impl Keyed for KeyClassField {
    const EXPORTED: bool = true;
    type Value = BindingClassField;
    type Answer = ClassField;
}
impl Exported for KeyClassField {}
impl Keyed for KeyClassSynthesizedFields {
    const EXPORTED: bool = true;
    type Value = BindingClassSynthesizedFields;
    type Answer = ClassSynthesizedFields;
}
impl Exported for KeyVariance {}
impl Keyed for KeyVariance {
    const EXPORTED: bool = true;
    type Value = BindingVariance;
    type Answer = VarianceMap;
}

impl Exported for KeyClassSynthesizedFields {}
impl Keyed for KeyExport {
    const EXPORTED: bool = true;
    type Value = BindingExport;
    type Answer = Type;
}
impl Exported for KeyExport {}
impl Keyed for KeyFunction {
    type Value = BindingFunction;
    type Answer = DecoratedFunction;
}
impl Keyed for KeyAnnotation {
    type Value = BindingAnnotation;
    type Answer = AnnotationWithTarget;
}
impl Keyed for KeyClassMetadata {
    const EXPORTED: bool = true;
    type Value = BindingClassMetadata;
    type Answer = ClassMetadata;
}
impl Exported for KeyClassMetadata {}
impl Keyed for KeyLegacyTypeParam {
    type Value = BindingLegacyTypeParam;
    type Answer = LegacyTypeParameterLookup;
}
impl Keyed for KeyYield {
    type Value = BindingYield;
    type Answer = YieldResult;
}
impl Keyed for KeyYieldFrom {
    type Value = BindingYieldFrom;
    type Answer = YieldFromResult;
}

/// Keys that refer to a `Type`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Key {
    /// I am an `import` at this location with this name.
    /// Used for `import foo.x` (the `foo` might not be literally present with `.` modules),
    /// and `from foo import *` (the names are injected from the exports)
    Import(Name, TextRange),
    /// I am defined in this module at this location.
    Definition(ShortIdentifier),
    /// I am a name with possible attribute/subscript narrowing coming from an assignment at this location.
    PropertyAssign(ShortIdentifier),
    /// The type at a specific return point.
    ReturnExplicit(TextRange),
    /// The implicit return type of a function, either Type::None or Type::Never.
    ReturnImplicit(ShortIdentifier),
    /// The actual type of the return for a function.
    ReturnType(ShortIdentifier),
    /// I am a name in this module at this location, bound to the associated binding.
    BoundName(ShortIdentifier),
    /// I am an expression that does not have a simple name but needs its type inferred.
    Anon(TextRange),
    /// I am an expression that appears in a statement. The range for this key is the range of the expr itself, which is different than the range of the stmt expr.
    StmtExpr(TextRange),
    /// I am an expression that appears in a `with` context.
    ContextExpr(TextRange),
    /// I am the result of joining several branches.
    Phi(Name, TextRange),
    /// I am the result of narrowing a type. The two ranges are the range at which the operation is
    /// defined and the one at which it is used. For example, in:
    ///   if x is None:
    ///       pass
    ///   else:
    ///       pass
    /// The `x is None` operation is defined once in the `if` test but generates two key/binding
    /// pairs, when it is used to narrow `x` in the `if` and the `else`, respectively.
    Narrow(Name, TextRange, TextRange),
    /// The binding definition site, anywhere it occurs
    Anywhere(Name, TextRange),
    /// Result of a super() call
    SuperInstance(TextRange),
    /// The intermediate used in an unpacking assignment.
    Unpack(TextRange),
    /// A usage link - a placeholder used for first-usage type inference.
    UsageLink(TextRange),
}

impl Ranged for Key {
    fn range(&self) -> TextRange {
        match self {
            Self::Import(_, r) => *r,
            Self::Definition(x) => x.range(),
            Self::PropertyAssign(x) => x.range(),
            Self::ReturnExplicit(r) => *r,
            Self::ReturnImplicit(x) => x.range(),
            Self::ReturnType(x) => x.range(),
            Self::BoundName(x) => x.range(),
            Self::Anon(r) => *r,
            Self::StmtExpr(r) => *r,
            Self::ContextExpr(r) => *r,
            Self::Phi(_, r) => *r,
            Self::Narrow(_, r, _) => *r,
            Self::Anywhere(_, r) => *r,
            Self::SuperInstance(r) => *r,
            Self::Unpack(r) => *r,
            Self::UsageLink(r) => *r,
        }
    }
}

impl DisplayWith<ModuleInfo> for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, ctx: &ModuleInfo) -> fmt::Result {
        match self {
            Self::Import(n, r) => write!(f, "import {n} {r:?}"),
            Self::Definition(x) => write!(f, "{} {:?}", ctx.display(x), x.range()),
            Self::PropertyAssign(x) => {
                write!(f, "prop assign {}._ = _ {:?}", ctx.display(x), x.range())
            }
            Self::BoundName(x) => write!(f, "use {} {:?}", ctx.display(x), x.range()),
            Self::Anon(r) => write!(f, "anon {r:?}"),
            Self::StmtExpr(r) => write!(f, "stmt expr {r:?}"),
            Self::ContextExpr(r) => write!(f, "context expr {r:?}"),
            Self::Phi(n, r) => write!(f, "phi {n} {r:?}"),
            Self::Narrow(n, r1, r2) => write!(f, "narrow {n} {r1:?} {r2:?}"),
            Self::Anywhere(n, r) => write!(f, "anywhere {n} {r:?}"),
            Self::ReturnType(x) => write!(f, "return {} {:?}", ctx.display(x), x.range()),
            Self::ReturnExplicit(r) => write!(f, "return {r:?}"),
            Self::ReturnImplicit(x) => {
                write!(f, "return implicit {} {:?}", ctx.display(x), x.range())
            }
            Self::SuperInstance(r) => write!(f, "super {r:?}"),
            Self::Unpack(r) => write!(f, "unpack {r:?}"),
            Self::UsageLink(r) => write!(f, "usage link {r:?}"),
        }
    }
}

impl DisplayWith<Bindings> for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, ctx: &Bindings) -> fmt::Result {
        write!(f, "{}", ctx.module_info().display(self))
    }
}

/// An expectation to be checked. For example, that a sequence is of an expected length.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KeyExpect(pub TextRange);

impl Ranged for KeyExpect {
    fn range(&self) -> TextRange {
        self.0
    }
}

impl DisplayWith<ModuleInfo> for KeyExpect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, _: &ModuleInfo) -> fmt::Result {
        write!(f, "expect {:?}", self.0)
    }
}

#[derive(Clone, Debug)]
pub enum ExprOrBinding {
    Expr(Expr),
    Binding(Binding),
}

#[derive(Clone, Debug)]
pub enum BindingExpect {
    /// An expression where we need to check for type errors, but don't need the result type.
    TypeCheckExpr(Box<Expr>),
    /// The expected number of values in an unpacked iterable expression.
    UnpackedLength(Idx<Key>, TextRange, SizeExpectation),
    /// An exception and its cause from a raise statement.
    CheckRaisedException(RaisedException),
    /// If a name already has an existing definition and we encounter a new definition,
    /// make sure the annotations are equal, with an associated name for error messages.
    Redefinition {
        new: Idx<KeyAnnotation>,
        existing: Idx<KeyAnnotation>,
        name: Name,
    },
    /// `del` statement
    Delete(Box<Expr>),
    /// Expression used in a boolean context (`bool()`, `if`, or `while`)
    Bool(Box<Expr>, TextRange),
}

impl DisplayWith<Bindings> for BindingExpect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, ctx: &Bindings) -> fmt::Result {
        let m = ctx.module_info();
        match self {
            Self::TypeCheckExpr(x) => {
                write!(f, "type check expr {}", m.display(x))
            }
            Self::Bool(x, ..) => {
                write!(f, "check bool expr {}", m.display(x))
            }
            Self::Delete(x) => {
                write!(f, "del {}", m.display(x))
            }
            Self::UnpackedLength(x, range, expect) => {
                let expectation = match expect {
                    SizeExpectation::Eq(n) => n.to_string(),
                    SizeExpectation::Ge(n) => format!(">={n}"),
                };
                write!(
                    f,
                    "expect length {} for {} {:?}",
                    expectation,
                    ctx.display(*x),
                    range
                )
            }
            Self::CheckRaisedException(RaisedException::WithoutCause(exc)) => {
                write!(f, "raise {}", m.display(exc))
            }
            Self::CheckRaisedException(RaisedException::WithCause(box (exc, cause))) => {
                write!(f, "raise {} from {}", m.display(exc), m.display(cause))
            }
            Self::Redefinition {
                new,
                existing,
                name,
            } => write!(
                f,
                "{} == {} on {}",
                ctx.display(*new),
                ctx.display(*existing),
                name
            ),
        }
    }
}

#[derive(Debug, Clone, TypeEq, VisitMut, PartialEq, Eq)]
pub struct EmptyAnswer;

impl Display for EmptyAnswer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "()")
    }
}

#[derive(Debug, Clone, TypeEq, VisitMut, PartialEq, Eq)]
pub struct NoneIfRecursive<T>(pub Option<T>);

impl<T> Display for NoneIfRecursive<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.0 {
            Some(x) => x.fmt(f),
            None => write!(f, "recursive"),
        }
    }
}

/// The binding definition site, at the end of the module (used for export).
/// If it has an annotation, only the annotation will be returned.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KeyExport(pub Name);

impl Ranged for KeyExport {
    fn range(&self) -> TextRange {
        TextRange::default()
    }
}

impl DisplayWith<ModuleInfo> for KeyExport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, _: &ModuleInfo) -> fmt::Result {
        write!(f, "export {}", self.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KeyFunction(pub ShortIdentifier);

impl Ranged for KeyFunction {
    fn range(&self) -> TextRange {
        self.0.range()
    }
}

impl DisplayWith<ModuleInfo> for KeyFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, ctx: &ModuleInfo) -> fmt::Result {
        write!(f, "{} {:?}", ctx.display(&self.0), self.0.range())
    }
}

/// A reference to a class.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KeyClass(pub ShortIdentifier);

impl Ranged for KeyClass {
    fn range(&self) -> TextRange {
        self.0.range()
    }
}

impl DisplayWith<ModuleInfo> for KeyClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, ctx: &ModuleInfo) -> fmt::Result {
        write!(f, "class {} {:?}", ctx.display(&self.0), self.0.range())
    }
}

/// A reference to a field in a class.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KeyClassField(pub ClassDefIndex, pub Name);

impl Ranged for KeyClassField {
    fn range(&self) -> TextRange {
        TextRange::default()
    }
}

impl DisplayWith<ModuleInfo> for KeyClassField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, _ctx: &ModuleInfo) -> fmt::Result {
        write!(f, "field {} . {}", self.0, self.1)
    }
}

/// Keys that refer to fields synthesized by a class, such as a dataclass's `__init__` method. This
/// has to be its own key/binding type because of the dependencies between the various pieces of
/// information about a class: ClassDef -> ClassMetadata -> ClassField -> ClassSynthesizedFields.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KeyClassSynthesizedFields(pub ClassDefIndex);

impl Ranged for KeyClassSynthesizedFields {
    fn range(&self) -> TextRange {
        TextRange::default()
    }
}

impl DisplayWith<ModuleInfo> for KeyClassSynthesizedFields {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, _ctx: &ModuleInfo) -> fmt::Result {
        write!(f, "synthesized fields of {}", self.0)
    }
}

// A key that denotes the variance of a type parameter
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KeyVariance(pub ClassDefIndex);

impl Ranged for KeyVariance {
    fn range(&self) -> TextRange {
        TextRange::default()
    }
}

impl DisplayWith<ModuleInfo> for KeyVariance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, _ctx: &ModuleInfo) -> fmt::Result {
        write!(f, "variance of {}", self.0)
    }
}

/// Keys that refer to an `Annotation`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum KeyAnnotation {
    /// I am the annotation for this instance of a name.
    Annotation(ShortIdentifier),
    /// The return type annotation for a function.
    ReturnAnnotation(ShortIdentifier),
    /// I am the annotation for the attribute at this range.
    AttrAnnotation(TextRange),
}

impl Ranged for KeyAnnotation {
    fn range(&self) -> TextRange {
        match self {
            Self::Annotation(x) => x.range(),
            Self::ReturnAnnotation(x) => x.range(),
            Self::AttrAnnotation(r) => *r,
        }
    }
}

impl DisplayWith<ModuleInfo> for KeyAnnotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, ctx: &ModuleInfo) -> fmt::Result {
        match self {
            Self::Annotation(x) => write!(f, "annot {} {:?}", ctx.display(x), x.range()),
            Self::ReturnAnnotation(x) => write!(f, "return {} {:?}", ctx.display(x), x.range()),
            Self::AttrAnnotation(r) => write!(f, "attr {:?}", r),
        }
    }
}

/// Keys that refer to a class's `Mro` (which tracks its ancestors, in method
/// resolution order).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KeyClassMetadata(pub ClassDefIndex);

impl Ranged for KeyClassMetadata {
    fn range(&self) -> TextRange {
        TextRange::default()
    }
}

impl DisplayWith<ModuleInfo> for KeyClassMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, _ctx: &ModuleInfo) -> fmt::Result {
        write!(f, "mro {}", self.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KeyLegacyTypeParam(pub ShortIdentifier);

impl Ranged for KeyLegacyTypeParam {
    fn range(&self) -> TextRange {
        self.0.range()
    }
}

impl DisplayWith<ModuleInfo> for KeyLegacyTypeParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, ctx: &ModuleInfo) -> fmt::Result {
        write!(
            f,
            "legacy_type_param {} {:?}",
            ctx.display(&self.0),
            self.0.range()
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KeyYield(pub TextRange);

impl Ranged for KeyYield {
    fn range(&self) -> TextRange {
        self.0
    }
}

impl DisplayWith<ModuleInfo> for KeyYield {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, ctx: &ModuleInfo) -> fmt::Result {
        write!(f, "{} {:?}", ctx.display(&self.0), self.0.range())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KeyYieldFrom(pub TextRange);

impl Ranged for KeyYieldFrom {
    fn range(&self) -> TextRange {
        self.0
    }
}

impl DisplayWith<ModuleInfo> for KeyYieldFrom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, ctx: &ModuleInfo) -> fmt::Result {
        write!(f, "{} {:?}", ctx.display(&self.0), self.0.range())
    }
}

#[derive(Clone, Copy, Dupe, Debug)]
pub enum UnpackedPosition {
    /// Zero-based index
    Index(usize),
    /// A negative index, counting from the back
    ReverseIndex(usize),
    /// Slice represented as an index from the front to an index from the back.
    /// Note that even though the second index is conceptually negative, we can
    /// represent it as a usize because it is always negative.
    Slice(usize, usize),
}

#[derive(Clone, Debug)]
pub enum SizeExpectation {
    Eq(usize),
    Ge(usize),
}

#[derive(Clone, Debug)]
pub enum RaisedException {
    WithoutCause(Expr),
    WithCause(Box<(Expr, Expr)>),
}

#[derive(Clone, Dupe, Copy, Debug, Eq, PartialEq)]
pub enum IsAsync {
    Sync,
    Async,
}

impl IsAsync {
    pub fn new(is_async: bool) -> Self {
        if is_async { Self::Async } else { Self::Sync }
    }

    pub fn is_async(self) -> bool {
        matches!(self, Self::Async)
    }

    pub fn context_exit_dunder(self) -> Name {
        match self {
            Self::Sync => dunder::EXIT,
            Self::Async => dunder::AEXIT,
        }
    }
}

/// Is the body of this function stubbed out (contains nothing but `...`)?
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FunctionStubOrImpl {
    /// The function body is `...`.
    Stub,
    /// The function body is not `...`.
    Impl,
}

#[derive(Clone, Debug)]
pub struct BindingFunction {
    /// A function definition, but with the return/body stripped out.
    pub def: StmtFunctionDef,
    pub stub_or_impl: FunctionStubOrImpl,
    pub class_key: Option<Idx<KeyClass>>,
    pub decorators: Box<[Idx<Key>]>,
    pub legacy_tparams: Box<[Idx<KeyLegacyTypeParam>]>,
    pub successor: Option<Idx<KeyFunction>>,
}

impl DisplayWith<Bindings> for BindingFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, _ctx: &Bindings) -> fmt::Result {
        write!(f, "def {}", self.def.name.id)
    }
}

#[derive(Clone, Debug)]
pub struct ClassBinding {
    pub def: StmtClassDef,
    pub def_index: ClassDefIndex,
    pub fields: SmallMap<Name, ClassFieldProperties>,
    pub bases: Box<[Expr]>,
    pub legacy_tparams: Box<[Idx<KeyLegacyTypeParam>]>,
}

#[derive(Clone, Debug)]
pub struct ReturnExplicit {
    pub annot: Option<Idx<KeyAnnotation>>,
    pub expr: Option<Box<Expr>>,
    pub is_generator: bool,
    pub is_async: bool,
}

#[derive(Clone, Debug)]
pub struct ReturnType {
    /// The annotation for the return type.
    pub annot: Option<(TextRange, Idx<KeyAnnotation>)>,
    /// The returns from the function.
    pub returns: Box<[Idx<Key>]>,
    pub implicit_return: Idx<Key>,
    /// The yeilds and yield froms. If either of these are nonempty, this is a generator function.
    pub yields: Box<[Idx<KeyYield>]>,
    pub yield_froms: Box<[Idx<KeyYieldFrom>]>,
    pub is_async: bool,
    /// Used to ignore the implicit return type for stub functions (returning `...`). This is
    /// unsafe, but is convenient and matches Pyright's behavior.
    pub stub_or_impl: FunctionStubOrImpl,
    pub decorators: Box<[Idx<Key>]>,
}

#[derive(Clone, Dupe, Copy, Debug)]
pub enum LastStmt {
    /// The last statement is a expression
    Expr,
    /// The last statement is a `with`, with the following context,
    /// which might (if exit is true) catch an exception
    With(IsAsync),
}

#[derive(Clone, Debug)]
pub struct ReturnImplicit {
    /// Terminal statements in the function control flow, used to determine whether the
    /// function has an implicit `None` return.
    /// When `None`, the function always has an implicit `None` return. When `Some(xs)`,
    /// the function has an implicit `None` return if there exists a non-`Never` in this
    /// list.
    pub last_exprs: Option<Box<[(LastStmt, Idx<Key>)]>>,
}

#[derive(Clone, Debug)]
pub enum SuperStyle {
    /// A `super(cls, obj)` call. The keys are the arguments.
    ExplicitArgs(Idx<Key>, Idx<Key>),
    /// A no-argument `super()` call. The key is the `Self` type of the class we are in.
    /// The name is the method we are in.
    ImplicitArgs(Idx<KeyClass>, Identifier),
    /// `super(Any, Any)`. Useful when we encounter an error.
    Any,
}

#[derive(Clone, Debug, Copy, Dupe, PartialEq, Eq)]
pub enum AnnotationStyle {
    /// Annotated assignment, x: MyType = my_value
    Direct,
    /// Forwarded annotation, x: MyType; x = my_value
    Forwarded,
}

#[derive(Clone, Debug)]
pub struct TypeParameter {
    pub name: Name,
    pub unique: Unique,
    pub kind: QuantifiedKind,
    pub bound: Option<Expr>,
    pub default: Option<Expr>,
    pub constraints: Option<(Vec<Expr>, TextRange)>,
}

/// Represents an `Idx<K>` for some `K: Keyed` other than `Key`
/// that we want to track for first-usage type inference.
#[derive(Clone, Debug)]
pub enum LinkedKey {
    Yield(Idx<KeyYield>),
    YieldFrom(Idx<KeyYieldFrom>),
    Expect(Idx<KeyExpect>),
}

#[derive(Clone, Debug)]
pub enum Binding {
    /// An expression, optionally with a Key saying what the type must be.
    /// The Key must be a type of types, e.g. `Type::Type`.
    Expr(Option<Idx<KeyAnnotation>>, Expr),
    /// TypeVar, ParamSpec, or TypeVarTuple
    TypeVar(Option<Idx<KeyAnnotation>>, Identifier, Box<ExprCall>),
    ParamSpec(Option<Idx<KeyAnnotation>>, Identifier, Box<ExprCall>),
    TypeVarTuple(Option<Idx<KeyAnnotation>>, Identifier, Box<ExprCall>),
    /// An expression returned from a function.
    ReturnExplicit(ReturnExplicit),
    /// The implicit return from a function.
    ReturnImplicit(ReturnImplicit),
    /// The return type of a function.
    ReturnType(Box<ReturnType>),
    /// A value in an iterable expression, e.g. IterableValue(\[1\]) represents 1.
    /// The second argument is the expression being iterated.
    /// The third argument indicates whether iteration is async or not.
    IterableValue(Option<Idx<KeyAnnotation>>, Expr, IsAsync),
    /// A value produced by entering a context manager.
    /// The second argument is the expression of the context manager and its range.
    /// The fourth argument indicates whether the context manager is async or not.
    ContextValue(Option<Idx<KeyAnnotation>>, Idx<Key>, TextRange, IsAsync),
    /// A value at a specific position in an unpacked iterable expression.
    /// Example: UnpackedValue(('a', 'b')), 1) represents 'b'.
    UnpackedValue(
        Option<Idx<KeyAnnotation>>,
        Idx<Key>,
        TextRange,
        UnpackedPosition,
    ),
    /// A type where we have an annotation, but also a type we computed.
    /// If the annotation has a type inside it (e.g. `int` then use the annotation).
    /// If the annotation doesn't (e.g. it's `Final`), then use the binding.
    AnnotatedType(Idx<KeyAnnotation>, Box<Binding>),
    /// A record of an "augmented assignment" statement like `x -= _`
    /// or `a.b *= _`. These desugar to special method calls.
    AugAssign(Option<Idx<KeyAnnotation>>, StmtAugAssign),
    /// An explicit type.
    Type(Type),
    /// The str type.
    StrType,
    /// The bool type.
    BoolType,
    /// A type parameter.
    TypeParameter(Box<TypeParameter>),
    /// The type of a function. The fields are:
    /// - A reference to the KeyFunction that point to the def
    /// - An optional reference to any previous function in the same flow by the same name;
    ///   this is needed to fold `@overload` decorated defs into a single type.
    /// - An optional reference to class metadata, which will be non-None when the function
    ///   is defined within a class scope.
    Function(
        Idx<KeyFunction>,
        Option<Idx<Key>>,
        Option<Idx<KeyClassMetadata>>,
    ),
    /// An import statement, typically with Self::Import.
    Import(ModuleName, Name),
    /// A class definition, points to a BindingClass and any decorators.
    ClassDef(Idx<KeyClass>, Box<[Idx<Key>]>),
    /// A forward reference to another binding.
    Forward(Idx<Key>),
    /// A phi node, representing the union of several alternative keys.
    Phi(SmallSet<Idx<Key>>),
    /// Used if the binding ends up being recursive, instead of defaulting to `Any`, should
    /// default to the given type.
    Default(Idx<Key>, Box<Binding>),
    /// A narrowed type.
    Narrow(Idx<Key>, Box<NarrowOp>, TextRange),
    /// An import of a module.
    /// Also contains the path along the module to bind, and optionally a key
    /// with the previous import to this binding (in which case merge the modules).
    Module(ModuleName, Vec<Name>, Option<Idx<Key>>),
    /// A name that might be a legacy type parameter. Solving this gives the Quantified type if so.
    /// The TextRange is optional and should be set at most once per identifier
    /// to avoid duplicate type errors (this is not type safe, because we might
    /// produce multiple `CheckLegacyTypeParam` bindings for the same
    /// identifier).
    /// It controls whether to produce an error saying there are scoped type parameters for this
    /// function / class, and therefore the use of legacy type parameters is invalid.
    CheckLegacyTypeParam(Idx<KeyLegacyTypeParam>, Option<TextRange>),
    /// An assignment to a name.
    NameAssign(
        Name,
        Option<(AnnotationStyle, Idx<KeyAnnotation>)>,
        Box<Expr>,
    ),
    /// A type alias declared with the `type` soft keyword
    ScopedTypeAlias(Name, Option<TypeParams>, Box<Expr>),
    /// An entry in a MatchMapping. The Key looks up the value being matched, the Expr is the key we're extracting.
    PatternMatchMapping(Expr, Idx<Key>),
    /// An entry in a MatchClass. The Key looks up the value being matched, the Expr is the class name.
    /// Positional patterns index into __match_args__, and keyword patterns match an attribute name.
    PatternMatchClassPositional(Box<Expr>, usize, Idx<Key>, TextRange),
    PatternMatchClassKeyword(Box<Expr>, Identifier, Idx<Key>),
    /// Binding for an `except` (if the boolean flag is false) or `except*` (if the boolean flag is true) clause
    ExceptionHandler(Box<Expr>, bool),
    /// Binding for an `@decorator` decoration on a function or class
    Decorator(Expr),
    /// Binding for a lambda parameter.
    LambdaParameter(Var),
    /// Binding for a function parameter. We either have an annotation, or we will determine the
    /// parameter type when solving the function type. To ensure the parameter is solved before it
    /// can be observed as a Var, we include the function key and force it to be solved first.
    FunctionParameter(Either<Idx<KeyAnnotation>, (Var, Idx<KeyFunction>)>),
    /// The result of a `super()` call.
    SuperInstance(SuperStyle, TextRange),
    /// The result of assigning to an attribute. This operation cannot change the *type* of the
    /// name to which we are assigning, but it *can* change the live attribute narrows.
    AssignToAttribute(Box<(ExprAttribute, ExprOrBinding)>),
    /// The result of assigning to a subscript, used for narrowing.
    AssignToSubscript(Box<(ExprSubscript, ExprOrBinding)>),
    /// A placeholder binding, used to force the solving of some other `K::Value` (for
    /// example, forcing a `BindingExpect` to be solved) in the context of first-usage-based
    /// type inference.
    UsageLink(LinkedKey),
}

impl DisplayWith<Bindings> for Binding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, ctx: &Bindings) -> fmt::Result {
        let m = ctx.module_info();
        match self {
            Self::Expr(None, x) => write!(f, "expr {}", m.display(x)),
            Self::Expr(Some(k), x) => {
                write!(f, "expr {}: {}", ctx.display(*k), m.display(x))
            }
            Self::TypeVar(_, name, x) => {
                write!(f, "typevar {} = {}", name, m.display(x))
            }
            Self::ParamSpec(_, name, x) => {
                write!(f, "paramspec {} = {}", name, m.display(x))
            }
            Self::TypeVarTuple(_, name, x) => {
                write!(f, "typevartuple {} = {}", name, m.display(x))
            }
            Self::ReturnExplicit(x) => {
                if let Some(annot) = x.annot {
                    write!(f, "{} ", ctx.display(annot))?
                }
                if let Some(expr) = x.expr.as_ref() {
                    write!(f, "{}", m.display(expr))
                } else {
                    write!(f, "return")
                }
            }
            Self::ReturnImplicit(_) => write!(f, "implicit return"),
            Self::ReturnType(_) => write!(f, "return type"),
            Self::IterableValue(None, x, IsAsync::Async) => {
                write!(f, "async iter {}", m.display(x))
            }
            Self::IterableValue(Some(k), x, IsAsync::Async) => {
                write!(f, "async iter {}: {}", ctx.display(*k), m.display(x))
            }
            Self::IterableValue(None, x, IsAsync::Sync) => write!(f, "iter {}", m.display(x)),
            Self::IterableValue(Some(k), x, IsAsync::Sync) => {
                write!(f, "iter {}: {}", ctx.display(*k), m.display(x))
            }
            Self::ExceptionHandler(x, true) => write!(f, "except* {}", m.display(x)),
            Self::ExceptionHandler(x, false) => write!(f, "except {}", m.display(x)),
            Self::ContextValue(_ann, x, _, kind) => {
                let name = match kind {
                    IsAsync::Sync => "context",
                    IsAsync::Async => "async context",
                };
                write!(f, "{name} {}", ctx.display(*x))
            }
            Self::UnpackedValue(_ann, x, range, pos) => {
                let pos = match pos {
                    UnpackedPosition::Index(i) => i.to_string(),
                    UnpackedPosition::ReverseIndex(i) => format!("-{i}"),
                    UnpackedPosition::Slice(i, j) => {
                        let end = match j {
                            0 => "".to_owned(),
                            _ => format!("-{j}"),
                        };
                        format!("{}:{}", i, end)
                    }
                };
                write!(f, "unpack {} {:?} @ {}", ctx.display(*x), range, pos)
            }
            Self::Function(x, _pred, _class) => write!(f, "{}", ctx.display(*x)),
            Self::Import(m, n) => write!(f, "import {m}.{n}"),
            Self::ClassDef(x, _) => write!(f, "{}", ctx.display(*x)),
            Self::Forward(k) => write!(f, "{}", ctx.display(*k)),
            Self::AugAssign(_, s) => write!(f, "augmented_assign {:?}", s),
            Self::Type(t) => write!(f, "type {t}"),
            Self::StrType => write!(f, "strtype"),
            Self::BoolType => write!(f, "booltype"),
            Self::TypeParameter(box TypeParameter { unique, kind, .. }) => {
                write!(f, "type_parameter({unique}, {kind})")
            }
            Self::CheckLegacyTypeParam(k, _) => {
                write!(f, "check_legacy_type_param {}", ctx.display(*k))
            }
            Self::AnnotatedType(k1, k2) => {
                write!(f, "({}): {}", k2.display_with(ctx), ctx.display(*k1))
            }
            Self::Module(m, path, key) => {
                write!(
                    f,
                    "module {}({}){}",
                    path.join("."),
                    m,
                    match key {
                        None => String::new(),
                        Some(k) => format!("+ {}", ctx.display(*k)),
                    }
                )
            }
            Self::Phi(xs) => {
                write!(f, "phi(")?;
                for (i, x) in xs.iter().enumerate() {
                    if i != 0 {
                        write!(f, "; ")?;
                    }
                    write!(f, "{}", ctx.display(*x))?;
                }
                write!(f, ")")
            }
            Self::Default(k, x) => {
                write!(f, "default({}): {}", ctx.display(*k), x.display_with(ctx))
            }
            Self::Narrow(k, op, _) => {
                write!(f, "narrow({}, {op:?})", ctx.display(*k))
            }
            Self::NameAssign(name, None, expr) => {
                write!(f, "{} = {}", name, expr.display_with(ctx.module_info()))
            }
            Self::NameAssign(name, Some((_, annot)), expr) => {
                write!(
                    f,
                    "{}: {} = {}",
                    name,
                    ctx.display(*annot),
                    expr.display_with(ctx.module_info())
                )
            }
            Self::ScopedTypeAlias(name, None, expr) => {
                write!(
                    f,
                    "type {} = {}",
                    name,
                    expr.display_with(ctx.module_info())
                )
            }
            Self::ScopedTypeAlias(name, Some(params), expr) => {
                write!(
                    f,
                    "type {}[{}] = {}",
                    name,
                    commas_iter(|| params.iter().map(|p| format!("{}", p.name()))),
                    expr.display_with(ctx.module_info())
                )
            }
            Self::PatternMatchMapping(mapping_key, binding_key) => {
                write!(
                    f,
                    "PatternMatchMapping {} = {}",
                    m.display(mapping_key),
                    ctx.display(*binding_key),
                )
            }
            Self::PatternMatchClassPositional(class, idx, key, range) => {
                write!(
                    f,
                    "PatternMatchClassPositional {}[{}] = {} {:?}",
                    m.display(class),
                    idx,
                    ctx.display(*key),
                    range
                )
            }
            Self::PatternMatchClassKeyword(class, attr, key) => {
                write!(
                    f,
                    "PatternMatchClassKeyword {}.{} = {}",
                    m.display(class),
                    attr,
                    ctx.display(*key),
                )
            }
            Self::Decorator(e) => write!(f, "decorator {}", m.display(e)),
            Self::LambdaParameter(_) => write!(f, "lambda parameter"),
            Self::FunctionParameter(_) => write!(f, "function parameter"),
            Self::SuperInstance(SuperStyle::ExplicitArgs(cls, obj), _range) => {
                write!(f, "super({}, {})", ctx.display(*cls), ctx.display(*obj))
            }
            Self::SuperInstance(SuperStyle::ImplicitArgs(_, _), _range) => write!(f, "super()"),
            Self::SuperInstance(SuperStyle::Any, _range) => write!(f, "super(Any, Any)"),
            Self::AssignToAttribute(box (attr, ExprOrBinding::Expr(value))) => {
                write!(
                    f,
                    "check assign expr to attr {}.{} {}",
                    m.display(attr.value.as_ref()),
                    attr.attr,
                    m.display(value),
                )
            }
            Self::AssignToAttribute(box (attr, ExprOrBinding::Binding(binding))) => {
                write!(
                    f,
                    "check assign type to attr {}.{} ({})",
                    m.display(attr.value.as_ref()),
                    attr.attr,
                    binding.display_with(ctx)
                )
            }
            Self::AssignToSubscript(box (subscript, ExprOrBinding::Expr(value))) => {
                write!(
                    f,
                    "check assign expr to subscript {}[{}] ({})",
                    m.display(subscript.value.as_ref()),
                    m.display(subscript.slice.as_ref()),
                    m.display(value),
                )
            }
            Self::AssignToSubscript(box (subscript, ExprOrBinding::Binding(binding))) => {
                write!(
                    f,
                    "check assign type to subscript {}[{}] ({})",
                    m.display(subscript.value.as_ref()),
                    m.display(subscript.slice.as_ref()),
                    binding.display_with(ctx)
                )
            }
            Self::UsageLink(usage_key) => {
                write!(f, "usage link to ",)?;
                match usage_key {
                    LinkedKey::Yield(idx) => {
                        write!(f, "{}", m.display(ctx.idx_to_key(*idx)))
                    }
                    LinkedKey::YieldFrom(idx) => {
                        write!(f, "{}", m.display(ctx.idx_to_key(*idx)))
                    }
                    LinkedKey::Expect(idx) => {
                        write!(f, "{}", m.display(ctx.idx_to_key(*idx)))
                    }
                }
            }
        }
    }
}

impl Binding {
    pub fn symbol_kind(&self) -> Option<SymbolKind> {
        match self {
            Binding::TypeVar(_, _, _)
            | Binding::ParamSpec(_, _, _)
            | Binding::TypeVarTuple(_, _, _)
            | Binding::TypeParameter(_)
            | Binding::CheckLegacyTypeParam(_, _) => Some(SymbolKind::TypeParameter),
            Binding::StrType => Some(SymbolKind::Str),
            Binding::BoolType => Some(SymbolKind::Bool),
            Binding::Function(_, _, _) => Some(SymbolKind::Function),
            Binding::Import(_, _) => {
                // TODO: maybe we can resolve it to see its symbol kind
                Some(SymbolKind::Variable)
            }
            Binding::ClassDef(_, _) => Some(SymbolKind::Class),
            Binding::Module(_, _, _) => Some(SymbolKind::Module),
            Binding::ScopedTypeAlias(_, _, _) => Some(SymbolKind::TypeAlias),
            Binding::NameAssign(name, _, _) if name.as_str() == name.to_uppercase() => {
                Some(SymbolKind::Constant)
            }
            Binding::NameAssign(_, _, _) => Some(SymbolKind::Variable),
            Binding::LambdaParameter(_) | Binding::FunctionParameter(_) => {
                Some(SymbolKind::Parameter)
            }
            Binding::Expr(_, _)
            | Binding::ReturnExplicit(_)
            | Binding::ReturnImplicit(_)
            | Binding::ReturnType(_)
            | Binding::IterableValue(_, _, _)
            | Binding::ContextValue(_, _, _, _)
            | Binding::UnpackedValue(_, _, _, _)
            | Binding::AnnotatedType(_, _)
            | Binding::AugAssign(_, _)
            | Binding::Type(_)
            | Binding::Forward(_)
            | Binding::Phi(_)
            | Binding::Default(_, _)
            | Binding::Narrow(_, _, _)
            | Binding::PatternMatchMapping(_, _)
            | Binding::PatternMatchClassPositional(_, _, _, _)
            | Binding::PatternMatchClassKeyword(_, _, _)
            | Binding::Decorator(_)
            | Binding::ExceptionHandler(_, _)
            | Binding::SuperInstance(_, _)
            | Binding::AssignToAttribute(_)
            | Binding::UsageLink(_)
            | Binding::AssignToSubscript(_) => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BindingExport(pub Binding);

impl DisplayWith<Bindings> for BindingExport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, ctx: &Bindings) -> fmt::Result {
        DisplayWith::fmt(&self.0, f, ctx)
    }
}

#[derive(Debug, Clone, Copy, VisitMut, TypeEq, PartialEq, Eq)]
pub enum Initialized {
    Yes,
    No,
}

#[derive(Debug, Clone, VisitMut, TypeEq, PartialEq, Eq)]
pub struct AnnotationWithTarget {
    pub target: AnnotationTarget,
    pub annotation: Annotation,
}

impl AnnotationWithTarget {
    pub fn ty(&self, stdlib: &Stdlib) -> Option<Type> {
        let annotation_ty = self.annotation.ty.as_ref()?;
        match self.target {
            AnnotationTarget::ArgsParam(_) => {
                if let Type::Unpack(unpacked) = annotation_ty {
                    Some((**unpacked).clone())
                } else if matches!(annotation_ty, Type::Args(_) | Type::Unpack(_)) {
                    Some(annotation_ty.clone())
                } else {
                    Some(Type::Tuple(Tuple::Unbounded(Box::new(
                        annotation_ty.clone(),
                    ))))
                }
            }
            AnnotationTarget::KwargsParam(_) => {
                if let Type::Unpack(unpacked) = annotation_ty {
                    Some((**unpacked).clone())
                } else if matches!(annotation_ty, Type::Kwargs(_) | Type::Unpack(_)) {
                    Some(annotation_ty.clone())
                } else {
                    Some(
                        stdlib
                            .dict(stdlib.str().clone().to_type(), annotation_ty.clone())
                            .to_type(),
                    )
                }
            }
            _ => Some(annotation_ty.clone()),
        }
    }
}

impl Display for AnnotationWithTarget {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: {}", self.target, self.annotation)
    }
}

#[derive(Debug, Clone, VisitMut, TypeEq, PartialEq, Eq)]
pub enum AnnotationTarget {
    /// A function parameter with a type annotation
    Param(Name),
    ArgsParam(Name),
    KwargsParam(Name),
    /// A return type annotation on a function. The name is that of the function
    Return(Name),
    /// An annotated assignment. For attribute assignments, the name is the attribute name ("attr" in "x.attr")
    /// Does the annotated assignment have an initial value?
    Assign(Name, Initialized),
    /// A member of a class
    ClassMember(Name),
}

impl Display for AnnotationTarget {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Param(name) => write!(f, "param {name}"),
            Self::ArgsParam(name) => write!(f, "args {name}"),
            Self::KwargsParam(name) => write!(f, "kwargs {name}"),
            Self::Return(name) => write!(f, "{name} return"),
            Self::Assign(name, _initialized) => write!(f, "var {name}"),
            Self::ClassMember(name) => write!(f, "attr {name}"),
        }
    }
}

impl AnnotationTarget {
    pub fn type_form_context(&self) -> TypeFormContext {
        match self {
            Self::Param(_) => TypeFormContext::ParameterAnnotation,
            Self::ArgsParam(_) => TypeFormContext::ParameterArgsAnnotation,
            Self::KwargsParam(_) => TypeFormContext::ParameterKwargsAnnotation,
            Self::Return(_) => TypeFormContext::ReturnAnnotation,
            Self::Assign(_, is_initialized) => TypeFormContext::VarAnnotation(*is_initialized),
            Self::ClassMember(_) => TypeFormContext::ClassVarAnnotation,
        }
    }
}

/// Values that return an annotation.
#[derive(Clone, Debug)]
pub enum BindingAnnotation {
    /// The type is annotated to be this key, will have the outer type removed.
    /// Optionally occurring within a class, in which case Self refers to this class.
    AnnotateExpr(AnnotationTarget, Expr, Option<Idx<KeyClass>>),
    /// A literal type we know statically.
    Type(AnnotationTarget, Type),
}

impl DisplayWith<Bindings> for BindingAnnotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, ctx: &Bindings) -> fmt::Result {
        match self {
            Self::AnnotateExpr(_, x, class_key) => write!(
                f,
                "_: {}{}",
                ctx.module_info().display(x),
                match class_key {
                    None => String::new(),
                    Some(t) => format!(" (self {})", ctx.display(*t)),
                }
            ),
            Self::Type(_, t) => write!(f, "type {t}"),
        }
    }
}

/// Binding for a class.
#[derive(Clone, Debug)]
pub enum BindingClass {
    ClassDef(ClassBinding),
    FunctionalClassDef(
        ClassDefIndex,
        Identifier,
        SmallMap<Name, ClassFieldProperties>,
    ),
}

impl DisplayWith<Bindings> for BindingClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, _ctx: &Bindings) -> fmt::Result {
        match self {
            Self::ClassDef(c) => write!(f, "class {}", c.def.name),
            Self::FunctionalClassDef(_, id, _) => write!(f, "class {}", id),
        }
    }
}

/// Binding for a class field, which is any attribute (including methods) of a class defined in
/// either the class body or in method (like `__init__`) that we recognize as
/// defining instance attributes.
#[derive(Clone, Debug)]
pub struct BindingClassField {
    pub class_idx: Idx<KeyClass>,
    pub name: Name,
    pub value: ExprOrBinding,
    pub annotation: Option<Idx<KeyAnnotation>>,
    pub range: TextRange,
    pub initial_value: ClassFieldInitialValue,
    pub is_function_without_return_annotation: bool,
    pub implicit_def_method: Option<Name>,
}

impl DisplayWith<Bindings> for BindingClassField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, ctx: &Bindings) -> fmt::Result {
        write!(f, "class field ")?;
        match &self.value {
            ExprOrBinding::Expr(e) => write!(f, "{}", ctx.module_info().display(e)),
            ExprOrBinding::Binding(b) => write!(f, "{}", b.display_with(ctx)),
        }
    }
}

/// The value that the class field is initialized to.
#[derive(Clone, Debug)]
pub enum ClassFieldInitialValue {
    /// The field does not have an initial value. If a name is provided, it is the name of the
    /// method where the field was inferred (which will be None for fields declared but not
    /// initialized in the class body, or for instance-only fields of synthesized classes)
    Instance(Option<Name>),
    /// The field has an initial value.
    ///
    /// If the value is from an assignment, stores the expression that the field is assigned to,
    /// which is needed for some cases like dataclass fields. The `None` case is for fields that
    /// have values which don't come from assignment (e.g. function defs, imports in a class body)
    Class(Option<Expr>),
}

/// Bindings for fields synthesized by a class, such as a dataclass's `__init__` method. This
/// has to be its own key/binding type because of the dependencies between the various pieces of
/// information about a class: ClassDef -> ClassMetadata -> ClassField -> ClassSynthesizedFields.
#[derive(Clone, Debug)]
pub struct BindingClassSynthesizedFields(pub Idx<KeyClass>);

impl DisplayWith<Bindings> for BindingClassSynthesizedFields {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, ctx: &Bindings) -> fmt::Result {
        write!(f, "synthesized fields of {}", ctx.display(self.0))
    }
}

#[derive(Clone, Debug)]
pub struct BindingVariance {
    pub class_key: Idx<KeyClass>,
    pub base_classes: Box<[Expr]>,
    pub fields: SmallSet<Idx<KeyClassField>>,
}

impl DisplayWith<Bindings> for BindingVariance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, ctx: &Bindings) -> fmt::Result {
        write!(f, "Variance of {}", ctx.display(self.class_key))
    }
}

/// Binding for the class's metadata (type level information derived from the class header - this
/// includes the MRO, the class keywords, and the metaclass).
///
/// The `Key` points to the definition of the class.
/// The `[Expr]` contains the base classes from the class header.
/// The `[(Name, Expr)]` contains the class keywords from the class header.
/// The `[Idx<Key>]` points to the class's decorators.
#[derive(Clone, Debug)]
pub struct BindingClassMetadata {
    pub class_idx: Idx<KeyClass>,
    pub bases: Box<[Expr]>,
    pub keywords: Box<[(Name, Expr)]>,
    pub decorators: Box<[Idx<Key>]>,
    pub is_new_type: bool,
    pub special_base: Option<Box<BaseClass>>,
}

impl DisplayWith<Bindings> for BindingClassMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, ctx: &Bindings) -> fmt::Result {
        write!(f, "metadata {}", ctx.display(self.class_idx))
    }
}

#[derive(Clone, Debug)]
pub struct BindingLegacyTypeParam(pub Idx<Key>);

impl DisplayWith<Bindings> for BindingLegacyTypeParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, ctx: &Bindings) -> fmt::Result {
        write!(f, "legacy_type_param {}", ctx.display(self.0))
    }
}

#[derive(Clone, Debug)]
pub enum BindingYield {
    Yield(Option<Idx<KeyAnnotation>>, ExprYield),
    Invalid(ExprYield),
}

impl BindingYield {
    fn expr(&self) -> &ExprYield {
        match self {
            Self::Yield(_, x) => x,
            Self::Invalid(x) => x,
        }
    }
}

impl DisplayWith<Bindings> for BindingYield {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, ctx: &Bindings) -> fmt::Result {
        let m = ctx.module_info();
        write!(f, "{}", m.display(&self.expr()))
    }
}

#[derive(Clone, Debug)]
pub enum BindingYieldFrom {
    YieldFrom(Option<Idx<KeyAnnotation>>, ExprYieldFrom),
    Invalid(ExprYieldFrom),
}

impl BindingYieldFrom {
    fn expr(&self) -> &ExprYieldFrom {
        match self {
            Self::YieldFrom(_, x) => x,
            Self::Invalid(x) => x,
        }
    }
}

impl DisplayWith<Bindings> for BindingYieldFrom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, ctx: &Bindings) -> fmt::Result {
        let m = ctx.module_info();
        write!(f, "{}", m.display(&self.expr()))
    }
}
