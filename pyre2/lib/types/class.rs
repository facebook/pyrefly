/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::Ord;
use std::cmp::Ordering;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use dupe::Dupe;
use parse_display::Display;
use pyrefly_derive::TypeEq;
use ruff_python_ast::name::Name;
use ruff_python_ast::Identifier;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;

use crate::module::module_info::ModuleInfo;
use crate::module::module_name::ModuleName;
use crate::types::callable::Param;
use crate::types::callable::Required;
use crate::types::equality::TypeEq;
use crate::types::qname::QName;
use crate::types::quantified::Quantified;
use crate::types::quantified::QuantifiedKind;
use crate::types::types::TParams;
use crate::types::types::Type;
use crate::util::display::commas_iter;
use crate::util::visit::Visit;
use crate::util::visit::VisitMut;

/// The name of a nominal type, e.g. `str`
#[derive(Debug, Clone, TypeEq, Display, Dupe)]
pub struct Class(Arc<ClassInner>);

impl Class {
    /// Key to use for equality purposes. If we have the same module and index,
    /// we must point at the same class underneath.
    fn key_eq(&self) -> (ClassIndex, ModuleName) {
        (self.0.index, self.0.qname.module_name())
    }

    /// Key to use for comparison purposes. Main used to sort identifiers in union,
    /// and then alphabetically sorting by the name gives a predictable answer.
    fn key_ord(&self) -> (&QName, ClassIndex) {
        (&self.0.qname, self.0.index)
    }
}

impl Hash for Class {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key_eq().hash(state)
    }
}

impl PartialEq for Class {
    fn eq(&self, other: &Self) -> bool {
        self.key_eq().eq(&other.key_eq())
    }
}

impl Eq for Class {}

impl Ord for Class {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key_ord().cmp(&other.key_ord())
    }
}

impl PartialOrd for Class {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// There are types stored inside Class, in the TParams, but we don't want to visit them.
// We typically visit types to get rid of Var, and promise these are not interesting in that sense.
impl VisitMut<Type> for Class {
    fn visit_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}
impl Visit<Type> for Class {
    fn visit<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

/// Simple properties of class fields that can be attached to the class definition. Note that this
/// does not include the type of a field, which needs to be computed lazily to avoid a recursive loop.
#[derive(Debug, Clone)]
pub struct ClassFieldProperties {
    is_annotated: bool,
    range: TextRange,
}

impl PartialEq for ClassFieldProperties {
    fn eq(&self, other: &Self) -> bool {
        self.is_annotated == other.is_annotated
    }
}

impl Eq for ClassFieldProperties {}
impl TypeEq for ClassFieldProperties {}

/// The index of a class within the file, used as a reference to data associated with the class.
#[derive(
    Debug, Clone, Dupe, Copy, TypeEq, Eq, PartialEq, Hash, PartialOrd, Ord, Display
)]
pub struct ClassIndex(pub u32);

impl ClassFieldProperties {
    pub fn new(is_annotated: bool, range: TextRange) -> Self {
        Self {
            is_annotated,
            range,
        }
    }
}

#[derive(TypeEq, Eq, PartialEq)]
struct ClassInner {
    index: ClassIndex,
    qname: QName,
    tparams: TParams,
    fields: SmallMap<Name, ClassFieldProperties>,
}

impl Debug for ClassInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClassInner")
            .field("index", &self.index)
            .field("qname", &self.qname)
            .field("tparams", &self.tparams)
            // We don't print `fields` because it's way too long.
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub enum ClassKind {
    StaticMethod,
    ClassMethod,
    Property,
    Class,
    EnumMember,
}

impl ClassKind {
    fn from_qname(qname: &QName) -> Self {
        match (qname.module_name().as_str(), qname.id().as_str()) {
            ("builtins", "staticmethod") => Self::StaticMethod,
            ("builtins", "classmethod") => Self::ClassMethod,
            ("builtins", "property") => Self::Property,
            ("enum", "member") => Self::EnumMember,
            _ => Self::Class,
        }
    }
}

impl Display for ClassInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "class {}", self.qname.id())?;
        if !self.tparams.is_empty() {
            write!(f, "[{}]", commas_iter(|| self.tparams.iter()))?;
        }
        writeln!(f, ": ...")
    }
}

// A note on terminology regarding attribute-related concepts:
// - "field" refers to something defined in a class body, with a raw type as written.
// - "member" refers to a name defined on a class, including inherited members whose
//   types should be expressed in terms of the current classe's type parameters.
// - "attribute" refers to a value actually accessed from an instance or class object,
//   which involves substituting type arguments for the class type parameters as
//   well as descriptor handling (including method binding).
impl Class {
    pub fn new(
        index: ClassIndex,
        name: Identifier,
        module_info: ModuleInfo,
        tparams: TParams,
        fields: SmallMap<Name, ClassFieldProperties>,
    ) -> Self {
        Self(Arc::new(ClassInner {
            index,
            qname: QName::new(name, module_info),
            tparams,
            fields,
        }))
    }

    pub fn contains(&self, name: &Name) -> bool {
        self.0.fields.contains_key(name)
    }

    pub fn range(&self) -> TextRange {
        self.0.qname.range()
    }

    pub fn name(&self) -> &Name {
        self.0.qname.id()
    }

    pub fn qname(&self) -> &QName {
        &self.0.qname
    }

    pub fn kind(&self) -> ClassKind {
        ClassKind::from_qname(self.qname())
    }

    pub fn tparams(&self) -> &TParams {
        &self.0.tparams
    }

    pub fn self_type(&self) -> Type {
        let tparams_as_targs =
            TArgs::new(self.tparams().quantified().map(|q| q.to_type()).collect());
        ClassType::new(self.clone(), tparams_as_targs).to_type()
    }

    pub fn self_param(&self) -> Param {
        Param::Pos(Name::new("self"), self.self_type(), Required::Required)
    }

    pub fn index(&self) -> ClassIndex {
        self.0.index
    }

    pub fn module_name(&self) -> ModuleName {
        self.0.qname.module_name()
    }

    pub fn module_info(&self) -> ModuleInfo {
        self.0.qname.module_info()
    }

    pub fn fields(&self) -> impl Iterator<Item = &Name> {
        self.0.fields.keys()
    }

    pub fn is_field_annotated(&self, name: &Name) -> bool {
        self.0
            .fields
            .get(name)
            .is_some_and(|prop| prop.is_annotated)
    }

    pub fn field_decl_range(&self, name: &Name) -> Option<TextRange> {
        Some(self.0.fields.get(name)?.range)
    }

    pub fn has_qname(&self, module: &str, name: &str) -> bool {
        self.0.qname.module_name().as_str() == module && self.0.qname.id() == name
    }
}

#[derive(Debug, Clone, TypeEq, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct TArgs(Box<[Type]>);

impl Visit<Type> for TArgs {
    fn visit<'a>(&'a self, f: &mut dyn FnMut(&'a Type)) {
        self.0.visit0(f);
    }
}

impl VisitMut<Type> for TArgs {
    fn visit_mut(&mut self, f: &mut dyn FnMut(&mut Type)) {
        self.0.visit0_mut(f);
    }
}

impl TArgs {
    pub fn new(targs: Vec<Type>) -> Self {
        Self(targs.into_boxed_slice())
    }

    pub fn as_slice(&self) -> &[Type] {
        &self.0
    }

    pub fn as_mut(&mut self) -> &mut [Type] {
        &mut self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Apply a substitution to type arguments.
    ///
    /// This is useful mainly to re-express ancestors (which, in the MRO, are in terms of class
    /// type parameters)
    ///
    /// This is mainly useful to take ancestors coming from the MRO (which are always in terms
    /// of the current class's type parameters) and re-express them in terms of the current
    /// class specialized with type arguments.
    pub fn substitute(&self, substitution: &Substitution) -> Self {
        Self::new(
            self.0
                .iter()
                .map(|ty| substitution.substitute(ty.clone()))
                .collect(),
        )
    }
}

pub struct Substitution(SmallMap<Quantified, Type>);

impl Substitution {
    pub fn substitute(&self, ty: Type) -> Type {
        ty.subst(&self.0)
    }

    pub fn new(mapping: SmallMap<Quantified, Type>) -> Self {
        Self(mapping)
    }
}

#[derive(Debug, Clone, TypeEq, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClassType(Class, TArgs);

impl Display for ClassType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", Type::ClassType(self.clone()))
    }
}

impl Visit<Type> for ClassType {
    fn visit<'a>(&'a self, f: &mut dyn FnMut(&'a Type)) {
        self.1.visit0(f)
    }
}

impl VisitMut<Type> for ClassType {
    fn visit_mut(&mut self, f: &mut dyn FnMut(&mut Type)) {
        self.1.visit0_mut(f)
    }
}

impl ClassType {
    fn new_impl(class: Class, targs: TArgs, extra_context: &str) -> Self {
        let tparams = class.tparams();
        if targs.0.len() != tparams.len()
            && !tparams
                .quantified()
                .any(|q| q.kind() == QuantifiedKind::TypeVarTuple)
        {
            // Invariant violation: we should always have valid type arguments when
            // constructing `ClassType`.
            assert_eq!(
                targs.0.len(),
                tparams.len(),
                "Encountered invalid type arguments in class {}.{}",
                class.name(),
                extra_context,
            )
        }
        Self(class, targs)
    }

    /// Create a class type.
    /// The `targs` must match the `tparams`, if this fails we will panic.
    pub fn new(class: Class, targs: TArgs) -> Self {
        Self::new_impl(class, targs, "")
    }

    pub fn new_for_stdlib(class: Class, targs: TArgs) -> Self {
        Self::new_impl(
            class,
            targs,
            " This is caused by typeshed not matching the type checker assumptions about stdlib.",
        )
    }

    pub fn class_object(&self) -> &Class {
        &self.0
    }

    pub fn tparams(&self) -> &TParams {
        self.0.tparams()
    }

    pub fn targs(&self) -> &TArgs {
        &self.1
    }

    pub fn targs_mut(&mut self) -> &mut TArgs {
        &mut self.1
    }

    /// Instantiate a type that is relative to the class type parameters
    /// by substituting in the type arguments.
    pub fn instantiate_member(&self, ty: Type) -> Type {
        self.substitution().substitute(ty)
    }

    /// Rewrite type arguments of some class relative to another.
    ///
    /// This is used to propagate instantiation of base class type parameters when computing
    /// the MRO.
    pub fn substitute(&self, substitution: &Substitution) -> Self {
        Self(self.0.dupe(), self.1.substitute(substitution))
    }

    pub fn substitution(&self) -> Substitution {
        let tparams = self.tparams();
        let targs = self.1.as_slice();
        Substitution(tparams.quantified().zip(targs.iter().cloned()).collect())
    }

    pub fn name(&self) -> &Name {
        self.0.name()
    }

    pub fn qname(&self) -> &QName {
        self.0.qname()
    }

    pub fn to_type(self) -> Type {
        Type::ClassType(self)
    }

    pub fn self_type(&self) -> Type {
        self.instantiate_member(self.class_object().self_type())
    }
}
