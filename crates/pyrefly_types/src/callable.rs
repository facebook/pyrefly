/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::borrow::Cow;
use std::cmp::Ord;
use std::cmp::Ordering;
use std::fmt;
use std::fmt::Display;
use std::hash::Hash;
use std::hash::Hasher;

use dupe::Dupe;
use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_python::dunder;
use pyrefly_python::module::Module;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::display::commas_iter;
use pyrefly_util::prelude::VecExt;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::Keyword;
use ruff_python_ast::name::Name;

use crate::class::Class;
use crate::class::ClassType;
use crate::equality::TypeEq;
use crate::keywords::DataclassTransformKeywords;
use crate::types::Type;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct Callable {
    pub params: Params,
    pub ret: Type,
}

impl Display for Callable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_with_type(f, &|t| t)
    }
}

#[derive(Debug, Clone)]
pub struct ArgCount {
    pub min: usize,
    pub max: Option<usize>,
}

impl ArgCount {
    fn none_allowed() -> Self {
        Self {
            min: 0,
            max: Some(0),
        }
    }

    fn any_allowed() -> Self {
        Self { min: 0, max: None }
    }

    fn add_arg(&mut self, req: &Required) {
        if *req == Required::Required {
            self.min += 1;
        }
        if let Some(n) = self.max {
            self.max = Some(n + 1);
        }
    }
}

#[derive(Debug, Clone)]
pub struct ArgCounts {
    pub positional: ArgCount,
    pub keyword: ArgCount,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct ParamList(Vec<Param>);

impl ParamList {
    pub fn new(xs: Vec<Param>) -> Self {
        Self(xs)
    }

    /// Create a new ParamList from a list of types, as required position-only parameters.
    pub fn new_types(xs: Vec<Type>) -> Self {
        Self(xs.into_map(|t| Param::PosOnly(None, t, Required::Required)))
    }

    /// Prepend some required position-only parameters.
    pub fn prepend_types(&self, pre: &[Type]) -> Cow<'_, ParamList> {
        if pre.is_empty() {
            Cow::Borrowed(self)
        } else {
            Cow::Owned(ParamList(
                pre.iter()
                    .map(|t| Param::PosOnly(None, t.clone(), Required::Required))
                    .chain(self.0.iter().cloned())
                    .collect(),
            ))
        }
    }

    pub fn fmt_with_type<'a, D: Display + 'a>(
        &'a self,
        f: &mut fmt::Formatter<'_>,
        wrap: &'a impl Fn(&'a Type) -> D,
    ) -> fmt::Result {
        // Keep track of whether we encounter a posonly parameter with a name, so we can emit the
        // `/` posonly marker. For conciseness, we don't want to emit this marker for
        // `typing.Callable` and other situations where we only have anonymous posonly parameters.
        let mut named_posonly = false;
        let mut kwonly = false;
        for (i, param) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            if matches!(param, Param::PosOnly(Some(_), _, _)) {
                named_posonly = true;
            } else if named_posonly {
                named_posonly = false;
                write!(f, "/, ")?;
            }
            if !kwonly && matches!(param, Param::KwOnly(..)) {
                kwonly = true;
                write!(f, "*, ")?;
            }
            param.fmt_with_type(f, wrap)?;
        }
        if named_posonly {
            write!(f, ", /")?;
        }
        Ok(())
    }

    /// Format parameters each parameter on a new line
    pub fn fmt_with_type_with_newlines<'a, D: Display + 'a>(
        &'a self,
        f: &mut fmt::Formatter<'_>,
        wrap: &'a impl Fn(&'a Type) -> D,
    ) -> fmt::Result {
        let mut named_posonly = false;
        let mut kwonly = false;

        for (i, param) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ",\n    ")?;
            }

            if matches!(param, Param::PosOnly(Some(_), _, _)) {
                named_posonly = true;
            } else if named_posonly {
                named_posonly = false;
                write!(f, "/,\n    ")?;
            }

            if !kwonly && matches!(param, Param::KwOnly(..)) {
                kwonly = true;
                write!(f, "*,\n    ")?;
            }

            param.fmt_with_type(f, wrap)?;
        }

        if named_posonly {
            write!(f, ",\n    /")?;
        }

        Ok(())
    }

    pub fn items(&self) -> &[Param] {
        &self.0
    }

    pub fn into_items(self) -> Vec<Param> {
        self.0
    }

    pub fn items_mut(&mut self) -> &mut [Param] {
        &mut self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn split_first(&self) -> Option<(&Type, ParamList)> {
        self.0
            .split_first()
            .map(|(first, rest)| (first.as_type(), ParamList(rest.to_vec())))
    }

    /// Type signature that permits everything, namely `*args, **kwargs`.
    pub fn everything() -> ParamList {
        ParamList(vec![
            Param::VarArg(None, Type::any_implicit()),
            Param::Kwargs(None, Type::any_implicit()),
        ])
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum Params {
    List(ParamList),
    Ellipsis,
    /// All possible materializations of `...`. A subset check with Callable[Materialization, R]
    /// succeeds only if it would succeed with Materialization replaced with any parameter list.
    /// See the comment on Type::Materialization - the intuition here is similar.
    Materialization,
    /// Any arguments to Concatenate, followed by a ParamSpec.
    /// E.g. `Concatenate[int, str, P]` would be `ParamSpec([int, str], P)`,
    /// while `P` alone would be `ParamSpec([], P)`.
    /// `P` may resolve to `Type::ParamSpecValue`, `Type::Concatenate`, or `Type::Ellipsis`
    ParamSpec(Box<[Type]>, Type),
}

impl Params {
    fn arg_counts(&self) -> ArgCounts {
        match self {
            Self::List(params) => {
                let mut counts = ArgCounts {
                    positional: ArgCount::none_allowed(),
                    keyword: ArgCount::none_allowed(),
                };
                for param in params.items() {
                    match param {
                        Param::PosOnly(_, _, req) => {
                            counts.positional.add_arg(req);
                        }
                        Param::Pos(..) => {
                            counts.positional.add_arg(&Required::Optional(None));
                            counts.keyword.add_arg(&Required::Optional(None));
                        }
                        Param::KwOnly(_, _, req) => {
                            counts.keyword.add_arg(req);
                        }
                        Param::VarArg(..) => {
                            counts.positional.max = None;
                        }
                        Param::Kwargs(..) => {
                            counts.keyword.max = None;
                        }
                    }
                }
                counts
            }
            Self::Ellipsis | Self::Materialization => ArgCounts {
                positional: ArgCount::any_allowed(),
                keyword: ArgCount::any_allowed(),
            },
            Self::ParamSpec(prefix, _) => ArgCounts {
                positional: ArgCount {
                    min: prefix.len(),
                    max: None,
                },
                keyword: ArgCount::any_allowed(),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum Param {
    PosOnly(Option<Name>, Type, Required),
    Pos(Name, Type, Required),
    VarArg(Option<Name>, Type),
    KwOnly(Name, Type, Required),
    Kwargs(Option<Name>, Type),
}

/// Requiredness for a function parameter.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum Required {
    Required,
    /// The parameter is optional, with the `Type` being the type of its default value, if available.
    Optional(Option<Type>),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct Function {
    pub signature: Callable,
    pub metadata: FuncMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct FuncMetadata {
    pub kind: FunctionKind,
    pub flags: FuncFlags,
}

impl FuncMetadata {
    pub fn def(module: Module, cls: Class, func: Name) -> Self {
        Self {
            kind: FunctionKind::Def(Box::new(FuncId {
                module,
                cls: Some(cls),
                name: func,
            })),
            flags: FuncFlags::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct FuncFlags {
    pub is_overload: bool,
    pub is_staticmethod: bool,
    pub is_classmethod: bool,
    /// A function decorated with `@deprecated`
    pub is_deprecated: bool,
    /// A function decorated with `@property`
    pub is_property_getter: bool,
    /// A function decorated with `functools.cached_property` or equivalent.
    pub is_cached_property: bool,
    /// A `foo.setter` function, where `foo` is some `@property`-decorated function.
    /// When used to decorate a function, turns the decorated function into a property setter.
    pub is_property_setter_decorator: bool,
    /// If None, this is a function decorated with `@foo.setter`, where `foo` is
    /// a property (i.e. a function decoratoed with `@property`)
    ///
    /// The stored type is `foo` (the getter).
    pub is_property_setter_with_getter: Option<Type>,
    pub has_enum_member_decoration: bool,
    pub is_override: bool,
    pub has_final_decoration: bool,
    /// A function decorated with `@abc.abstractmethod`
    pub is_abstract_method: bool,
    /// A function decorated with `typing.dataclass_transform(...)`, turning it into a
    /// `dataclasses.dataclass`-like decorator. Stores the keyword values passed to the
    /// `dataclass_transform` call. See
    /// https://typing.python.org/en/latest/spec/dataclasses.html#specification.
    pub dataclass_transform_metadata: Option<DataclassTransformKeywords>,
}

#[derive(Debug, Clone)]
pub struct FuncId {
    pub module: Module,
    pub cls: Option<Class>,
    pub name: Name,
}

impl PartialEq for FuncId {
    fn eq(&self, other: &Self) -> bool {
        self.key_eq().eq(&other.key_eq())
    }
}

impl Eq for FuncId {}
impl TypeEq for FuncId {}

impl Ord for FuncId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key_ord().cmp(&other.key_ord())
    }
}

impl PartialOrd for FuncId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Hash for FuncId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key_eq().hash(state)
    }
}

impl VisitMut<Type> for FuncId {
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}
impl Visit<Type> for FuncId {
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

impl FuncId {
    fn key_eq(&self) -> (ModuleName, ModulePath, Option<Class>, &Name) {
        (
            self.module.name(),
            self.module.path().to_key_eq(),
            self.cls.clone(),
            &self.name,
        )
    }

    fn key_ord(&self) -> (ModuleName, ModulePath, Option<Class>, &Name) {
        self.key_eq()
    }

    fn format_impl(
        func_module: ModuleName,
        func_cls: Option<Class>,
        func_name: &Name,
        current_module: ModuleName,
    ) -> String {
        let module_prefix =
            if func_module == current_module || func_module == ModuleName::builtins() {
                "".to_owned()
            } else {
                format!("{}.", func_module)
            };
        let class_prefix = match &func_cls {
            Some(cls) => {
                format!("{}.", cls.name())
            }
            None => "".to_owned(),
        };
        format!("{module_prefix}{class_prefix}{}", func_name)
    }

    pub fn format(&self, current_module: ModuleName) -> String {
        Self::format_impl(
            self.module.name(),
            self.cls.clone(),
            &self.name,
            current_module,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum FunctionKind {
    IsInstance,
    IsSubclass,
    Dataclass,
    DataclassField,
    /// `typing.dataclass_transform`. Note that this is `dataclass_transform` itself, *not* the
    /// decorator created by a `dataclass_transform(...)` call. See
    /// https://typing.python.org/en/latest/spec/dataclasses.html#specification.
    DataclassTransform,
    ClassMethod,
    Overload,
    Override,
    Cast,
    AssertType,
    RevealType,
    Final,
    RuntimeCheckable,
    Def(Box<FuncId>),
    AbstractMethod,
    /// Instance of a protocol with a `__call__` method. The function has the `__call__` signature.
    CallbackProtocol(Box<ClassType>),
    TotalOrdering,
}

impl Callable {
    pub fn fmt_with_type<'a, D: Display + 'a>(
        &'a self,
        f: &mut fmt::Formatter<'_>,
        wrap: &'a impl Fn(&'a Type) -> D,
    ) -> fmt::Result {
        match &self.params {
            Params::List(params) => {
                write!(f, "(")?;
                params.fmt_with_type(f, wrap)?;
                write!(f, ") -> {}", wrap(&self.ret))
            }
            Params::Ellipsis => write!(f, "(...) -> {}", wrap(&self.ret)),
            Params::Materialization => write!(f, "(Materialization) -> {}", wrap(&self.ret)),
            Params::ParamSpec(args, pspec) => {
                write!(f, "({}", commas_iter(|| args.iter().map(wrap)))?;
                match pspec {
                    Type::ParamSpecValue(params) => {
                        if !args.is_empty() && !params.is_empty() {
                            write!(f, ", ")?;
                        }
                        params.fmt_with_type(f, wrap)?;
                    }
                    Type::Ellipsis => {
                        if !args.is_empty() {
                            write!(f, ", ")?;
                        }
                        write!(f, "...")?;
                    }
                    _ => {
                        if !args.is_empty() {
                            write!(f, ", ")?;
                        }
                        write!(f, "ParamSpec({})", wrap(pspec))?;
                    }
                }
                write!(f, ") -> {}", wrap(&self.ret))
            }
        }
    }

    /// Format the function type for use in a hover tooltip. This is similar to `fmt_with_type`, but
    /// it puts args on new lines if there is more than one argument
    pub fn fmt_with_type_with_newlines<'a, D: Display + 'a>(
        &'a self,
        f: &mut fmt::Formatter<'_>,
        wrap: &'a impl Fn(&'a Type) -> D,
    ) -> fmt::Result {
        match &self.params {
            Params::List(params) if params.len() > 1 => {
                // For multiple parameters, put each on a new line with indentation
                write!(f, "(\n    ")?;
                params.fmt_with_type_with_newlines(f, wrap)?;
                write!(f, "\n) -> {}", wrap(&self.ret))
            }
            Params::List(..)
            | Params::ParamSpec(..)
            | Params::Ellipsis
            | Params::Materialization => self.fmt_with_type(f, wrap),
        }
    }

    pub fn list(params: ParamList, ret: Type) -> Self {
        Self {
            params: Params::List(params),
            ret,
        }
    }

    pub fn ellipsis(ret: Type) -> Self {
        Self {
            params: Params::Ellipsis,
            ret,
        }
    }

    pub fn param_spec(p: Type, ret: Type) -> Self {
        Self {
            params: Params::ParamSpec(Box::default(), p),
            ret,
        }
    }

    pub fn concatenate(args: Box<[Type]>, param_spec: Type, ret: Type) -> Self {
        Self {
            params: Params::ParamSpec(args, param_spec),
            ret,
        }
    }

    pub fn split_first_param(&self) -> Option<(&Type, Self)> {
        match self {
            Self {
                params: Params::List(params),
                ret,
            } => {
                let (first, rest) = params.split_first()?;
                Some((first, Self::list(rest, ret.clone())))
            }
            Self {
                params: Params::ParamSpec(ts, p),
                ret,
            } => {
                let (first, rest) = ts.split_first()?;
                Some((
                    first,
                    Self::concatenate(rest.iter().cloned().collect(), p.clone(), ret.clone()),
                ))
            }
            _ => None,
        }
    }

    pub fn get_first_param(&self) -> Option<Type> {
        match self {
            Self {
                params: Params::List(params),
                ret: _,
            } if let Some(param) = params.items().first() => match param {
                Param::PosOnly(_, ty, _) | Param::Pos(_, ty, _) | Param::VarArg(_, ty) => {
                    Some(ty.clone())
                }
                _ => None,
            },
            Self {
                params: Params::ParamSpec(ts, _),
                ret: _,
            } => ts.first().cloned(),
            _ => None,
        }
    }

    pub fn is_typeguard(&self) -> bool {
        matches!(
            self,
            Self {
                params: _,
                ret: Type::TypeGuard(_)
            }
        )
    }

    pub fn is_typeis(&self) -> bool {
        matches!(
            self,
            Self {
                params: _,
                ret: Type::TypeIs(_),
            }
        )
    }

    pub fn subst_self_type_mut(&mut self, replacement: &Type) {
        self.visit_mut(&mut |t: &mut Type| t.subst_self_type_mut(replacement));
    }

    pub fn arg_counts(&self) -> ArgCounts {
        self.params.arg_counts()
    }
}

impl Param {
    fn fmt_default(&self, default: &Option<Type>) -> String {
        match default {
            Some(Type::Literal(lit)) => format!("{lit}"),
            Some(Type::None) => "None".to_owned(),
            _ => "...".to_owned(),
        }
    }

    pub fn fmt_with_type<'a, D: Display + 'a>(
        &'a self,
        f: &mut fmt::Formatter<'_>,
        wrap: impl Fn(&'a Type) -> D,
    ) -> fmt::Result {
        match self {
            Param::PosOnly(None, ty, Required::Required) => write!(f, "{}", wrap(ty)),
            Param::PosOnly(None, ty, Required::Optional(default)) => {
                write!(f, "_: {} = {}", wrap(ty), self.fmt_default(default))
            }
            Param::PosOnly(Some(name), ty, Required::Required)
            | Param::Pos(name, ty, Required::Required)
            | Param::KwOnly(name, ty, Required::Required) => {
                write!(f, "{}: {}", name, wrap(ty),)
            }
            Param::PosOnly(Some(name), ty, Required::Optional(default))
            | Param::Pos(name, ty, Required::Optional(default))
            | Param::KwOnly(name, ty, Required::Optional(default)) => {
                write!(f, "{}: {} = {}", name, wrap(ty), self.fmt_default(default))
            }
            Param::VarArg(Some(name), ty) => write!(f, "*{}: {}", name, wrap(ty)),
            Param::VarArg(None, ty) => write!(f, "*{}", wrap(ty)),
            Param::Kwargs(Some(name), ty) => write!(f, "**{}: {}", name, wrap(ty)),
            Param::Kwargs(None, ty) => write!(f, "**{}", wrap(ty)),
        }
    }

    pub fn as_type(&self) -> &Type {
        match self {
            Param::PosOnly(_, ty, _)
            | Param::Pos(_, ty, _)
            | Param::VarArg(_, ty)
            | Param::KwOnly(_, ty, _)
            | Param::Kwargs(_, ty) => ty,
        }
    }

    pub fn as_type_mut(&mut self) -> &mut Type {
        match self {
            Param::PosOnly(_, ty, _)
            | Param::Pos(_, ty, _)
            | Param::VarArg(_, ty)
            | Param::KwOnly(_, ty, _)
            | Param::Kwargs(_, ty) => ty,
        }
    }

    pub fn is_required(&self) -> bool {
        match self {
            Param::PosOnly(_, _, Required::Required)
            | Param::Pos(_, _, Required::Required)
            | Param::KwOnly(_, _, Required::Required) => true,
            _ => false,
        }
    }
}

impl Display for Param {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_with_type(f, |t| t)?;
        Ok(())
    }
}

impl FunctionKind {
    pub fn from_name(module: Module, cls: Option<Class>, func: &Name) -> Self {
        match (module.name().as_str(), cls.as_ref(), func.as_str()) {
            ("builtins", None, "isinstance") => Self::IsInstance,
            ("builtins", None, "issubclass") => Self::IsSubclass,
            ("builtins", None, "classmethod") => Self::ClassMethod,
            ("dataclasses", None, "dataclass") => Self::Dataclass,
            ("dataclasses", None, "field") => Self::DataclassField,
            ("typing", None, "overload") => Self::Overload,
            ("typing", None, "override") => Self::Override,
            ("typing", None, "cast") => Self::Cast,
            ("typing", None, "assert_type") => Self::AssertType,
            ("typing", None, "reveal_type") => Self::RevealType,
            ("typing", None, "final") => Self::Final,
            ("typing", None, "runtime_checkable") => Self::RuntimeCheckable,
            ("typing", None, "dataclass_transform") => Self::DataclassTransform,
            ("typing_extensions", None, "dataclass_transform") => Self::DataclassTransform,
            ("typing_extensions", None, "runtime_checkable") => Self::RuntimeCheckable,
            ("abc", None, "abstractmethod") => Self::AbstractMethod,
            ("functools", None, "total_ordering") => Self::TotalOrdering,
            _ => Self::Def(Box::new(FuncId {
                module,
                cls,
                name: func.clone(),
            })),
        }
    }

    pub fn module_name(&self) -> ModuleName {
        match self {
            Self::IsInstance => ModuleName::builtins(),
            Self::IsSubclass => ModuleName::builtins(),
            Self::ClassMethod => ModuleName::builtins(),
            Self::Dataclass => ModuleName::dataclasses(),
            Self::DataclassField => ModuleName::dataclasses(),
            Self::DataclassTransform => ModuleName::typing(),
            Self::Final => ModuleName::typing(),
            Self::Overload => ModuleName::typing(),
            Self::Override => ModuleName::typing(),
            Self::Cast => ModuleName::typing(),
            Self::AssertType => ModuleName::typing(),
            Self::RevealType => ModuleName::typing(),
            Self::RuntimeCheckable => ModuleName::typing(),
            Self::CallbackProtocol(cls) => cls.qname().module_name(),
            Self::AbstractMethod => ModuleName::abc(),
            Self::TotalOrdering => ModuleName::functools(),
            Self::Def(func_id) => func_id.module.name().dupe(),
        }
    }

    pub fn function_name(&self) -> Cow<'_, Name> {
        match self {
            Self::IsInstance => Cow::Owned(Name::new_static("isinstance")),
            Self::IsSubclass => Cow::Owned(Name::new_static("issubclass")),
            Self::ClassMethod => Cow::Owned(Name::new_static("classmethod")),
            Self::Dataclass => Cow::Owned(Name::new_static("dataclass")),
            Self::DataclassField => Cow::Owned(Name::new_static("field")),
            Self::DataclassTransform => Cow::Owned(Name::new_static("dataclass_transform")),
            Self::Final => Cow::Owned(Name::new_static("final")),
            Self::Overload => Cow::Owned(Name::new_static("overload")),
            Self::Override => Cow::Owned(Name::new_static("override")),
            Self::Cast => Cow::Owned(Name::new_static("cast")),
            Self::AssertType => Cow::Owned(Name::new_static("assert_type")),
            Self::RevealType => Cow::Owned(Name::new_static("reveal_type")),
            Self::RuntimeCheckable => Cow::Owned(Name::new_static("runtime_checkable")),
            Self::CallbackProtocol(_) => Cow::Owned(dunder::CALL),
            Self::AbstractMethod => Cow::Owned(Name::new_static("abstractmethod")),
            Self::TotalOrdering => Cow::Owned(Name::new_static("total_ordering")),
            Self::Def(func_id) => Cow::Borrowed(&func_id.name),
        }
    }

    pub fn class(&self) -> Option<Class> {
        match self {
            Self::IsInstance => None,
            Self::IsSubclass => None,
            Self::ClassMethod => None,
            Self::Dataclass => None,
            Self::DataclassField => None,
            Self::DataclassTransform => None,
            Self::Final => None,
            Self::Overload => None,
            Self::Override => None,
            Self::Cast => None,
            Self::AssertType => None,
            Self::RevealType => None,
            Self::RuntimeCheckable => None,
            Self::CallbackProtocol(cls) => Some(cls.class_object().dupe()),
            Self::AbstractMethod => None,
            Self::TotalOrdering => None,
            Self::Def(func_id) => func_id.cls.clone(),
        }
    }

    pub fn format(&self, current_module: ModuleName) -> String {
        FuncId::format_impl(
            self.module_name(),
            self.class(),
            self.function_name().as_ref(),
            current_module,
        )
    }
}

pub fn unexpected_keyword(error: &dyn Fn(String), func: &str, keyword: &Keyword) {
    let desc = if let Some(id) = &keyword.arg {
        format!(" `{id}`")
    } else {
        "".to_owned()
    };
    error(format!("`{func}` got an unexpected keyword argument{desc}"));
}

#[cfg(test)]
mod tests {
    use ruff_python_ast::name::Name;

    use crate::callable::Callable;
    use crate::callable::Param;
    use crate::callable::ParamList;
    use crate::callable::Required;
    use crate::types::Type;

    #[test]
    fn test_arg_counts_positional() {
        // (x: Any, /, y: Any = ...) -> None
        let callable = Callable::list(
            ParamList::new(vec![
                Param::PosOnly(
                    Some(Name::new("x")),
                    Type::any_implicit(),
                    Required::Required,
                ),
                Param::Pos(
                    Name::new("y"),
                    Type::any_implicit(),
                    Required::Optional(None),
                ),
            ]),
            Type::None,
        );
        let counts = callable.arg_counts();
        assert_eq!(counts.positional.min, 1);
        assert_eq!(counts.positional.max, Some(2));
        assert_eq!(counts.keyword.min, 0);
        assert_eq!(counts.keyword.max, Some(1));
    }

    #[test]
    fn test_arg_counts_keyword() {
        // (*, x: Any, y: Any = ...) -> None
        let callable = Callable::list(
            ParamList::new(vec![
                Param::KwOnly(Name::new("x"), Type::any_implicit(), Required::Required),
                Param::KwOnly(
                    Name::new("y"),
                    Type::any_implicit(),
                    Required::Optional(None),
                ),
            ]),
            Type::None,
        );
        let counts = callable.arg_counts();
        assert_eq!(counts.positional.min, 0);
        assert_eq!(counts.positional.max, Some(0));
        assert_eq!(counts.keyword.min, 1);
        assert_eq!(counts.keyword.max, Some(2));
    }

    #[test]
    fn test_arg_counts_varargs() {
        // (*args) -> None
        let callable = Callable::list(
            ParamList::new(vec![Param::VarArg(None, Type::any_implicit())]),
            Type::None,
        );
        let counts = callable.arg_counts();
        assert_eq!(counts.positional.min, 0);
        assert_eq!(counts.positional.max, None);
        assert_eq!(counts.keyword.min, 0);
        assert_eq!(counts.keyword.max, Some(0));
    }

    #[test]
    fn test_arg_counts_kwargs() {
        // (**kwargs) -> None
        let callable = Callable::list(
            ParamList::new(vec![Param::Kwargs(None, Type::any_implicit())]),
            Type::None,
        );
        let counts = callable.arg_counts();
        assert_eq!(counts.positional.min, 0);
        assert_eq!(counts.positional.max, Some(0));
        assert_eq!(counts.keyword.min, 0);
        assert_eq!(counts.keyword.max, None);
    }

    #[test]
    fn test_arg_counts_paramlist() {
        // (w, /, x, *args, y, z=...) -> None
        let callable = Callable::list(
            ParamList::new(vec![
                Param::PosOnly(
                    Some(Name::new("w")),
                    Type::any_implicit(),
                    Required::Required,
                ),
                Param::Pos(Name::new("x"), Type::any_implicit(), Required::Required),
                Param::VarArg(None, Type::any_implicit()),
                Param::KwOnly(Name::new("y"), Type::any_implicit(), Required::Required),
                Param::KwOnly(
                    Name::new("z"),
                    Type::any_implicit(),
                    Required::Optional(None),
                ),
            ]),
            Type::None,
        );
        let counts = callable.arg_counts();
        assert_eq!(counts.positional.min, 1);
        assert_eq!(counts.positional.max, None);
        assert_eq!(counts.keyword.min, 1);
        assert_eq!(counts.keyword.max, Some(3));
    }

    #[test]
    fn test_arg_counts_ellipsis() {
        let callable = Callable::ellipsis(Type::None);
        let counts = callable.arg_counts();
        assert_eq!(counts.positional.min, 0);
        assert_eq!(counts.positional.max, None);
        assert_eq!(counts.keyword.min, 0);
        assert_eq!(counts.keyword.max, None);
    }

    #[test]
    fn test_arg_counts_paramspec() {
        let callable = Callable::concatenate(
            vec![Type::None, Type::None].into_boxed_slice(),
            Type::any_implicit(),
            Type::None,
        );
        let counts = callable.arg_counts();
        assert_eq!(counts.positional.min, 2);
        assert_eq!(counts.positional.max, None);
        assert_eq!(counts.keyword.min, 0);
        assert_eq!(counts.keyword.max, None);
    }
}
