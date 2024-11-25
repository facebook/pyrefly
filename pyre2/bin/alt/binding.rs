/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Display;

use ruff_python_ast::name::Name;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprSubscript;
use ruff_python_ast::Identifier;
use ruff_python_ast::StmtAugAssign;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::StmtFunctionDef;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::small_set::SmallSet;

use crate::alt::bindings::Bindings;
use crate::graph::index::Idx;
use crate::module::module_name::ModuleName;
use crate::types::types::AnyStyle;
use crate::types::types::Quantified;
use crate::types::types::Type;
use crate::util::display::DisplayWith;

/// A Key that can be accessed from another module.
pub trait Exported {}

impl Exported for KeyExported {}
impl Exported for KeyBaseClass {}
impl Exported for KeyMro {}
impl Exported for KeyTypeParams {}

/// Keys that refer to a `Type`.
///
/// Within a `Key`, `Identifier` MUST be a name in the original AST,
/// not something we've synthesized.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Key {
    /// I am an `import` at this location with this name.
    /// Used for `import foo.x` (the `foo` might not be literally present with `.` modules),
    /// and `from foo import *` (the names are injected from the exports)
    Import(Name, TextRange),
    /// I am defined in this module at this location.
    Definition(Identifier),
    /// I am the self type for a particular class.
    SelfType(Identifier),
    /// The type at a specific return point.
    ReturnExpression(Identifier, TextRange),
    /// The actual type of the return for a function.
    ReturnType(Identifier),
    /// I am a use in this module at this location.
    Usage(Identifier),
    /// I am not defining a name or using one, but record me for checking.
    Anon(TextRange),
    /// I am the result of joining several branches.
    Phi(Name, TextRange),
    /// The binding definition site, anywhere it occurs
    Anywhere(Name, TextRange),
    /// A 'keyword argument' appearing in a class header, e.g. `metaclass`.
    ClassKeyword(Identifier, Name),
}

/// Like `Key`, but used for things accessible in another module.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum KeyExported {
    /// The binding definition site, at the end of the module (used for export).
    /// If it has an annotation, only the annotation will be returned.
    Export(Name),
    /// A reference to a field in a class.
    /// The range is the range of the class name, not the field name.
    ClassField(Identifier, Name),
}

impl Ranged for Key {
    fn range(&self) -> TextRange {
        match self {
            Self::Import(_, r) => *r,
            Self::Definition(x) => x.range,
            Self::SelfType(x) => x.range,
            Self::ReturnExpression(_, r) => *r,
            Self::ReturnType(x) => x.range,
            Self::Usage(x) => x.range,
            Self::Anon(r) => *r,
            Self::Phi(_, r) => *r,
            Self::Anywhere(_, r) => *r,
            Self::ClassKeyword(c, _) => c.range,
        }
    }
}

impl Ranged for KeyExported {
    fn range(&self) -> TextRange {
        match self {
            Self::Export(_) => TextRange::default(),
            Self::ClassField(c, _) => c.range,
        }
    }
}

/// Keys that refer to an `Annotation`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum KeyAnnotation {
    /// I am the annotation for this instance of a name.
    Annotation(Identifier),
    /// The return type annotation for a function.
    ReturnAnnotation(Identifier),
    /// I am the annotation for the attribute at this range.
    AttrAnnotation(TextRange),
}

impl Ranged for KeyAnnotation {
    fn range(&self) -> TextRange {
        match self {
            Self::Annotation(x) => x.range,
            Self::ReturnAnnotation(x) => x.range,
            Self::AttrAnnotation(r) => *r,
        }
    }
}

/// Key that refers to a `BaseClass`
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KeyBaseClass(pub Identifier, pub usize);

impl Ranged for KeyBaseClass {
    fn range(&self) -> TextRange {
        self.0.range
    }
}

/// Keys that refer to a class's `Mro` (which tracks its ancestors, in method
/// resolution order).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KeyMro(pub Identifier);

impl Ranged for KeyMro {
    fn range(&self) -> TextRange {
        self.0.range
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KeyLegacyTypeParam(pub Identifier);

impl Ranged for KeyLegacyTypeParam {
    fn range(&self) -> TextRange {
        self.0.range
    }
}

/// Keys that refer to the `TypeParams` for a class or function.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KeyTypeParams(pub Identifier);

impl Ranged for KeyTypeParams {
    fn range(&self) -> TextRange {
        self.0.range
    }
}

#[derive(Clone, Debug)]
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
    WithCause(Expr, Expr),
}

#[derive(Clone, Copy, Debug)]
pub enum ContextManagerKind {
    Sync,
    Async,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FunctionKind {
    Stub,
    Impl,
}

#[derive(Clone, Debug)]
pub enum Binding {
    /// An expression, optionally with a Key saying what the type must be.
    /// The Key must be a type of types, e.g. `Type::Type`.
    Expr(Option<Idx<KeyAnnotation>>, Expr),
    /// A value in an iterable expression, e.g. IterableValue(\[1\]) represents 1.
    IterableValue(Option<Idx<KeyAnnotation>>, Expr),
    /// A value produced by entering a context manager.
    /// The second argument is the expression of the context manager. The third argument
    /// indicates whether the context manager is async or not.
    ContextValue(Option<Idx<KeyAnnotation>>, Expr, ContextManagerKind),
    /// A value at a specific position in an unpacked iterable expression.
    /// Example: UnpackedValue(('a', 'b')), 1) represents 'b'.
    UnpackedValue(Box<Binding>, TextRange, UnpackedPosition),
    /// The expected number of values in an unpacked iterable expression.
    UnpackedLength(Box<Binding>, TextRange, SizeExpectation),
    /// A subscript expression and the value assigned to it
    SubscriptValue(Box<Binding>, ExprSubscript),
    /// A type where we have an annotation, but also a type we computed.
    /// If the annotation has a type inside it (e.g. `int` then use the annotation).
    /// If the annotation doesn't (e.g. it's `Final`), then use the binding.
    AnnotatedType(Idx<KeyAnnotation>, Box<Binding>),
    AugAssign(StmtAugAssign),
    /// The Any type.
    AnyType(AnyStyle),
    /// The str type.
    StrType,
    /// A type parameter.
    TypeParameter(Quantified),
    /// A function definition, but with the return/body stripped out.
    Function(StmtFunctionDef, FunctionKind),
    /// An import statement, typically with Self::Import.
    Import(ModuleName, Name),
    /// A class definition, but with the body stripped out.
    /// The field names should be used to build `ClassField` keys for lookup.
    /// Track the number of base classes, use this to build `BaseClass` keys for lookup.
    Class(StmtClassDef, SmallSet<Name>, usize),
    /// A class header keyword argument (e.g. the `metaclass`).
    ClassKeyword(Expr),
    /// The Self type for a class, must point at a class.
    SelfType(Idx<Key>),
    /// A forward reference to another binding.
    Forward(Idx<Key>),
    /// A phi node, representing the union of several alternative keys.
    Phi(SmallSet<Idx<Key>>),
    /// An import of a module.
    /// Also contains the path along the module to bind, and optionally a key
    /// with the previous import to this binding (in which case merge the modules).
    /// FIXME: Once we fix on alt, we the Module type will be ModuleName+Vec<Name>,
    /// so people using the None option will be able to use Self::Type instead.
    Module(ModuleName, Vec<Name>, Option<Idx<Key>>),
    /// An exception and its cause from a raise statement.
    CheckRaisedException(RaisedException),
    /// A name that might be a legacy type parameter. Solving this gives the Quantified type if so.
    /// The TextRange is optional and should be set at most once per identifier
    /// to avoid duplicate type errors (this is not type safe, because we might
    /// produce multpiple `CheckLegacyTypeParam` bindings for the same
    /// identifier).
    /// It controls whether to produce an error saying there are scoped type parameters for this
    /// function / class, and therefore the use of legacy type parameters is invalid.
    CheckLegacyTypeParam(Idx<KeyLegacyTypeParam>, Option<TextRange>),
    /// An expectation that the types are identical, with an associated name for error messages.
    Eq(Idx<KeyAnnotation>, Idx<KeyAnnotation>, Name),
    /// An assignment to a name. The text range is the range of the RHS, and is used so that we
    /// can error on bad type forms in type aliases.
    NameAssign(Name, Option<Idx<KeyAnnotation>>, Box<Binding>, TextRange),
    /// A type alias declared with the `type` soft keyword
    ScopedTypeAlias(Name, Vec<Quantified>, Box<Binding>, TextRange),
}

impl Binding {
    /// Helper function that turns trivial Phi nodes into a forward.
    pub fn phi(xs: SmallSet<Idx<Key>>) -> Self {
        if xs.len() == 1 {
            Self::Forward(xs.into_iter().next().unwrap())
        } else {
            Self::Phi(xs)
        }
    }
}

/// Values that reutrn an annotation.
#[derive(Clone, Debug)]
pub enum BindingAnnotation {
    /// The type is annotated to be this key, will have the outer type removed.
    /// Optionally occuring within a class, in which case Self refers to this class.
    AnnotateExpr(Expr, Option<Idx<Key>>),
    /// A literal type we know statically.
    Type(Type),
    /// Type of an attribute.
    AttrType(ExprAttribute),
    /// A forward reference to another binding.
    Forward(Idx<Key>),
}

/// Binding used to compute a `BaseClass`.
///
/// The `Expr` is the base class expression, from the containing class header.
#[derive(Clone, Debug)]
pub struct BindingBaseClass(pub Expr);

/// Binding for the class `Mro`. The `Key` is the self type of the class.
#[derive(Clone, Debug)]
pub struct BindingMro(pub Idx<Key>);

/// Values that represent type parameters of either functions or classes.
#[derive(Clone, Debug)]
pub enum BindingTypeParams {
    /// The first argument is any scoped type parameters.
    /// The second argument tracks all names that appear in parameter and return annotations, which might
    /// indicate legacy type parameters if they point to variable declarations.
    Function(Vec<Quantified>, Vec<Idx<KeyLegacyTypeParam>>),
    /// The first argument is a lookup for the class definition.
    /// The second argument tracks all names that appear in bases, which might
    /// indicate legacy type parameters if they point to variable declarations.
    Class(Idx<Key>, Vec<Idx<KeyLegacyTypeParam>>),
}

#[derive(Clone, Debug)]
pub struct BindingLegacyTypeParam(pub Idx<Key>);

impl Display for KeyAnnotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Annotation(x) => write!(f, "annot {} {:?}", x.id, x.range),
            Self::ReturnAnnotation(x) => write!(f, "return {} {:?}", x.id, x.range),
            Self::AttrAnnotation(r) => write!(f, "attr {:?}", r),
        }
    }
}

impl Display for KeyBaseClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "base_class {} {:?} [{}]",
            self.0.id, self.0.range, self.1
        )
    }
}

impl Display for KeyMro {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mro {} {:?}", self.0.id, self.0.range)
    }
}

impl Display for KeyLegacyTypeParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "legacy_type_param {} {:?}", self.0.id, self.0.range)
    }
}

impl Display for KeyTypeParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "type_params {} {:?}", self.0.id, self.0.range)
    }
}

impl Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Import(n, r) => write!(f, "import {n} {r:?}"),
            Self::Definition(x) => write!(f, "{} {:?}", x.id, x.range),
            Self::SelfType(x) => write!(f, "self {} {:?}", x.id, x.range),
            Self::Usage(x) => write!(f, "use {} {:?}", x.id, x.range),
            Self::Anon(r) => write!(f, "anon {r:?}"),
            Self::Phi(n, r) => write!(f, "phi {n} {r:?}"),
            Self::Anywhere(n, r) => write!(f, "anywhere {n} {r:?}"),
            Self::ClassKeyword(x, n) => write!(f, "class_keyword {} {:?} . {}", x.id, x.range, n),
            Self::ReturnType(x) => write!(f, "return {} {:?}", x.id, x.range),
            Self::ReturnExpression(x, i) => write!(f, "return {} {:?} @ {i:?}", x.id, x.range),
        }
    }
}

impl Display for KeyExported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Export(n) => write!(f, "export {n}"),
            Self::ClassField(x, n) => write!(f, "field {} {:?} . {}", x.id, x.range, n),
        }
    }
}

impl DisplayWith<Bindings> for BindingAnnotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, ctx: &Bindings) -> fmt::Result {
        match self {
            Self::AnnotateExpr(x, self_type) => write!(
                f,
                "_: {}{}",
                ctx.module_info().display(x),
                match self_type {
                    None => String::new(),
                    Some(t) => format!(" (self {})", ctx.display(*t)),
                }
            ),
            Self::Forward(k) => write!(f, "{}", ctx.display(*k)),
            Self::Type(t) => write!(f, "type {t}"),
            Self::AttrType(attr) => write!(f, "type {attr:?}"),
        }
    }
}

impl DisplayWith<Bindings> for BindingBaseClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, ctx: &Bindings) -> fmt::Result {
        write!(f, "base_class {}", ctx.module_info().display(&self.0),)
    }
}

impl DisplayWith<Bindings> for BindingMro {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, ctx: &Bindings) -> fmt::Result {
        write!(f, "mro {}", ctx.display(self.0))
    }
}

impl DisplayWith<Bindings> for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, _: &Bindings) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl DisplayWith<Bindings> for BindingTypeParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, ctx: &Bindings) -> fmt::Result {
        match self {
            Self::Function(_, _) => write!(f, "function_type_params"),
            Self::Class(id, _) => write!(f, "class_type_params {}", ctx.display(*id)),
        }
    }
}

impl DisplayWith<Bindings> for BindingLegacyTypeParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, ctx: &Bindings) -> fmt::Result {
        write!(f, "legacy_type_param {}", ctx.display(self.0))
    }
}

impl DisplayWith<Bindings> for Binding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, ctx: &Bindings) -> fmt::Result {
        let m = ctx.module_info();
        match self {
            Self::Expr(None, x) => write!(f, "{}", m.display(x)),
            Self::Expr(Some(k), x) => write!(f, "{}: {}", ctx.display(*k), m.display(x)),
            Self::IterableValue(None, x) => write!(f, "iter {}", m.display(x)),
            Self::IterableValue(Some(k), x) => {
                write!(f, "iter {}: {}", ctx.display(*k), m.display(x))
            }
            Self::ContextValue(_ann, x, kind) => {
                let name = match kind {
                    ContextManagerKind::Sync => "context",
                    ContextManagerKind::Async => "async context",
                };
                write!(f, "{name} {}", m.display(x))
            }
            Self::SubscriptValue(x, subscript) => {
                write!(
                    f,
                    "subscript {}[{}] = {}",
                    m.display(&subscript.value),
                    m.display(&subscript.slice),
                    x.display_with(ctx),
                )
            }
            Self::UnpackedValue(x, range, pos) => {
                let pos = match pos {
                    UnpackedPosition::Index(i) => i.to_string(),
                    UnpackedPosition::ReverseIndex(i) => format!("-{i}"),
                    UnpackedPosition::Slice(i, j) => {
                        let end = match j {
                            0 => "".to_string(),
                            _ => format!("-{j}"),
                        };
                        format!("{}:{}", i, end)
                    }
                };
                write!(f, "unpack {} {:?} @ {}", x.display_with(ctx), range, pos)
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
                    x.display_with(ctx),
                    range
                )
            }
            Self::Function(x, _) => write!(f, "def {}", x.name.id),
            Self::Import(m, n) => write!(f, "import {m}.{n}"),
            Self::Class(c, _, _) => write!(f, "class {}", c.name.id),
            Self::ClassKeyword(x) => write!(f, "class_keyword {}", m.display(x)),
            Self::SelfType(k) => write!(f, "self {}", ctx.display(*k)),
            Self::Forward(k) => write!(f, "{}", ctx.display(*k)),
            Self::AugAssign(s) => write!(f, "augmented_assign {:?}", s),
            Self::AnyType(s) => write!(f, "anytype {s}"),
            Self::StrType => write!(f, "strtype"),
            Self::TypeParameter(q) => write!(f, "type_parameter {q}"),
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
            Self::CheckRaisedException(RaisedException::WithoutCause(exc)) => {
                write!(f, "raise {}", m.display(exc))
            }
            Self::CheckRaisedException(RaisedException::WithCause(exc, cause)) => {
                write!(f, "raise {} from {}", m.display(exc), m.display(cause))
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
            Self::Eq(k1, k2, name) => write!(
                f,
                "{} == {} on {}",
                ctx.display(*k1),
                ctx.display(*k2),
                name
            ),
            Self::NameAssign(name, None, binding, _) => {
                write!(f, "{} = {}", name, binding.display_with(ctx))
            }
            Self::NameAssign(name, Some(annot), binding, _) => {
                write!(
                    f,
                    "{}: {} = {}",
                    name,
                    ctx.display(*annot),
                    binding.display_with(ctx)
                )
            }
            Self::ScopedTypeAlias(name, qs, binding, _r) if qs.is_empty() => {
                write!(f, "type {} = {}", name, binding.display_with(ctx))
            }
            Self::ScopedTypeAlias(name, qs, binding, _r) => {
                write!(
                    f,
                    "type {}[{}] = {}",
                    name,
                    qs.iter()
                        .map(|q| format!("{q}"))
                        .collect::<Vec<_>>()
                        .join(", "),
                    binding.display_with(ctx)
                )
            }
        }
    }
}
