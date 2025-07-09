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
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::Identifier;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;

use crate::module::module_info::ModuleInfo;
use crate::types::equality::TypeEq;
use crate::types::qname::QName;
use crate::types::types::Substitution;
use crate::types::types::TArgs;
use crate::types::types::TParams;
use crate::types::types::Type;

/// The name of a nominal type, e.g. `str`
#[derive(Debug, Clone, TypeEq, Display, Dupe)]
pub struct Class(Arc<ClassInner>);

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
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}
impl Visit<Type> for Class {
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

/// Simple properties of class fields that can be attached to the class definition. Note that this
/// does not include the type of a field, which needs to be computed lazily to avoid a recursive loop.
#[derive(Debug, Clone)]
pub struct ClassFieldProperties {
    is_annotated: bool,
    // The field is initialized on the class (outside of a method)
    is_initialized_on_class: bool,
    range: TextRange,
}

impl PartialEq for ClassFieldProperties {
    fn eq(&self, other: &Self) -> bool {
        self.is_annotated == other.is_annotated
            && self.is_initialized_on_class == other.is_initialized_on_class
    }
}

impl Eq for ClassFieldProperties {}
impl TypeEq for ClassFieldProperties {}

/// The index of a class within the file, used as a reference to data associated with the class.
#[derive(Debug, Clone, Dupe, Copy, Eq, PartialEq, Hash, PartialOrd, Ord)]
#[derive(TypeEq, Display)]
pub struct ClassDefIndex(pub u32);

impl ClassFieldProperties {
    pub fn new(is_annotated: bool, has_default_value: bool, range: TextRange) -> Self {
        Self {
            is_annotated,
            is_initialized_on_class: has_default_value,
            range,
        }
    }

    pub fn is_initialized_on_class(&self) -> bool {
        self.is_initialized_on_class
    }
}

#[derive(TypeEq, Eq, PartialEq)]
struct ClassInner {
    def_index: ClassDefIndex,
    qname: QName,
    /// The precomputed tparams will be `Some(..)` if we were able to verify that there
    /// are no legacy type variables (at which point there's no chance of producing a cycle
    /// when computing the class tparams). Whenever it is `None`, there will be a corresponding
    /// `KeyTParams` / `BindingTParams` pair to compute the class tparams.
    precomputed_tparams: Option<Arc<TParams>>,
    fields: SmallMap<Name, ClassFieldProperties>,
}

impl Debug for ClassInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClassInner")
            .field("index", &self.def_index)
            .field("qname", &self.qname)
            .field("tparams", &self.precomputed_tparams)
            // We don't print `fields` because it's way too long.
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum ClassKind {
    StaticMethod,
    ClassMethod,
    Property,
    Class,
    EnumMember,
    DataclassField,
}

impl ClassKind {
    fn from_qname(qname: &QName) -> Self {
        match (qname.module_name().as_str(), qname.id().as_str()) {
            ("builtins", "staticmethod") => Self::StaticMethod,
            ("builtins", "classmethod") => Self::ClassMethod,
            ("builtins", "property") => Self::Property,
            ("functools", "cached_property") => Self::Property,
            ("cached_property", "cached_property") => Self::Property,
            ("cinder", "cached_property") => Self::Property,
            ("cinder", "async_cached_property") => Self::Property,
            ("enum", "member") => Self::EnumMember,
            ("dataclasses", "Field") => Self::DataclassField,
            _ => Self::Class,
        }
    }
}

impl Display for ClassInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "class {}", self.qname.id())?;
        writeln!(f, ": ...")
    }
}

// A note on terminology regarding attribute-related concepts:
// - "field" refers to something defined in a class body, with a raw type as written.
// - "member" refers to a name defined on a class, including inherited members whose
//   types should be expressed in terms of the type parameters of the current class
// - "attribute" refers to a value actually accessed from an instance or class object,
//   which involves substituting type arguments for the class type parameters as
//   well as descriptor handling (including method binding).
impl Class {
    pub fn new(
        def_index: ClassDefIndex,
        name: Identifier,
        module_info: ModuleInfo,
        precomputed_tparams: Option<Arc<TParams>>,
        fields: SmallMap<Name, ClassFieldProperties>,
    ) -> Self {
        Self(Arc::new(ClassInner {
            def_index,
            qname: QName::new(name, module_info),
            precomputed_tparams,
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

    pub fn precomputed_tparams(&self) -> &Option<Arc<TParams>> {
        &self.0.precomputed_tparams
    }

    pub fn index(&self) -> ClassDefIndex {
        self.0.def_index
    }

    pub fn module_name(&self) -> ModuleName {
        self.0.qname.module_name()
    }

    pub fn module_path(&self) -> &ModulePath {
        self.0.qname.module_path()
    }

    pub fn module_info(&self) -> &ModuleInfo {
        self.0.qname.module_info()
    }

    pub fn fields(&self) -> impl ExactSizeIterator<Item = &Name> {
        self.0.fields.keys()
    }

    pub fn is_field_annotated(&self, name: &Name) -> bool {
        self.0
            .fields
            .get(name)
            .is_some_and(|prop| prop.is_annotated)
    }

    pub fn is_field_initialized_on_class(&self, name: &Name) -> bool {
        self.0
            .fields
            .get(name)
            .is_some_and(|prop| prop.is_initialized_on_class)
    }

    pub fn field_decl_range(&self, name: &Name) -> Option<TextRange> {
        Some(self.0.fields.get(name)?.range)
    }

    pub fn has_qname(&self, module: &str, name: &str) -> bool {
        self.0.qname.module_name().as_str() == module && self.0.qname.id() == name
    }

    pub fn is_builtin(&self, name: &str) -> bool {
        self.has_qname("builtins", name)
    }

    /// Key to use for equality purposes. If we have the same module and index,
    /// we must point at the same class underneath.
    fn key_eq(&self) -> (ClassDefIndex, ModuleName, &ModulePath) {
        (
            self.0.def_index,
            self.0.qname.module_name(),
            self.0.qname.module_path(),
        )
    }

    /// Key to use for comparison purposes. Main used to sort identifiers in union,
    /// and then alphabetically sorting by the name gives a predictable answer.
    fn key_ord(&self) -> (&QName, ClassDefIndex) {
        (&self.0.qname, self.0.def_index)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct ClassType(Class, TArgs);

impl Display for ClassType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", Type::ClassType(self.clone()))
    }
}

impl ClassType {
    /// Create a class type.
    /// The `targs` must match the `tparams`, if this fails we will panic.
    pub fn new(class: Class, targs: TArgs) -> Self {
        Self(class, targs)
    }

    pub fn class_object(&self) -> &Class {
        &self.0
    }

    pub fn tparams(&self) -> &TParams {
        self.1.tparams()
    }

    pub fn targs(&self) -> &TArgs {
        &self.1
    }

    pub fn targs_mut(&mut self) -> &mut TArgs {
        &mut self.1
    }

    /// Rewrite type arguments of some class relative to another.
    ///
    /// This is used to propagate instantiation of base class type parameters when computing
    /// the MRO.
    pub fn substitute(&self, substitution: &Substitution) -> Self {
        Self(self.0.dupe(), self.1.apply_substitution(substitution))
    }

    pub fn substitution(&self) -> Substitution {
        self.targs().substitution()
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

    pub fn has_qname(&self, module: &str, name: &str) -> bool {
        self.0.has_qname(module, name)
    }

    pub fn is_builtin(&self, name: &str) -> bool {
        self.0.is_builtin(name)
    }
}
