/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::borrow::Cow;
use std::fmt;
use std::fmt::Display;

use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_python::dunder;
use pyrefly_python::module_name::ModuleName;
use pyrefly_util::display::commas_iter;
use pyrefly_util::prelude::VecExt;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::Keyword;
use ruff_python_ast::name::Name;

use crate::class::ClassType;
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
    /// Any arguments to Concatenate, followed by a ParamSpec.
    /// E.g. `Concatenate[int, str, P]` would be `ParamSpec([int, str], P)`,
    /// while `P` alone would be `ParamSpec([], P)`.
    /// `P` may resolve to `Type::ParamSpecValue`, `Type::Concatenate`, or `Type::Ellipsis`
    ParamSpec(Box<[Type]>, Type),
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
    pub fn def(module: ModuleName, cls: Name, func: Name) -> Self {
        Self {
            kind: FunctionKind::Def(Box::new(FuncId {
                module,
                cls: Some(cls),
                func,
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct FuncId {
    pub module: ModuleName,
    pub cls: Option<Name>,
    pub func: Name,
}

impl FuncId {
    pub fn format(&self, current_module: ModuleName) -> String {
        let module_prefix =
            if self.module == current_module || self.module == ModuleName::builtins() {
                "".to_owned()
            } else {
                format!("{}.", self.module)
            };
        let class_prefix = match &self.cls {
            Some(cls) => {
                format!("{cls}.")
            }
            None => "".to_owned(),
        };
        format!("{module_prefix}{class_prefix}{}", self.func)
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
            Params::List(..) | Params::ParamSpec(..) | Params::Ellipsis => {
                self.fmt_with_type(f, wrap)
            }
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
    pub fn from_name(module: ModuleName, cls: Option<&Name>, func: &Name) -> Self {
        match (module.as_str(), cls, func.as_str()) {
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
            ("typing_extensions", None, "runtime_checkable") => Self::RuntimeCheckable,
            ("abc", None, "abstractmethod") => Self::AbstractMethod,
            ("functools", None, "total_ordering") => Self::TotalOrdering,
            _ => Self::Def(Box::new(FuncId {
                module,
                cls: cls.cloned(),
                func: func.clone(),
            })),
        }
    }

    pub fn as_func_id(&self) -> FuncId {
        match self {
            Self::IsInstance => FuncId {
                module: ModuleName::builtins(),
                cls: None,
                func: Name::new_static("isinstance"),
            },
            Self::IsSubclass => FuncId {
                module: ModuleName::builtins(),
                cls: None,
                func: Name::new_static("issubclass"),
            },
            Self::ClassMethod => FuncId {
                module: ModuleName::builtins(),
                cls: None,
                func: Name::new_static("classmethod"),
            },
            Self::Dataclass => FuncId {
                module: ModuleName::dataclasses(),
                cls: None,
                func: Name::new_static("dataclass"),
            },
            Self::DataclassField => FuncId {
                module: ModuleName::dataclasses(),
                cls: None,
                func: Name::new_static("field"),
            },
            Self::DataclassTransform => FuncId {
                module: ModuleName::typing(),
                cls: None,
                func: Name::new_static("dataclass_transform"),
            },
            Self::Final => FuncId {
                module: ModuleName::typing(),
                cls: None,
                func: Name::new_static("final"),
            },
            Self::Overload => FuncId {
                module: ModuleName::typing(),
                cls: None,
                func: Name::new_static("overload"),
            },
            Self::Override => FuncId {
                module: ModuleName::typing(),
                cls: None,
                func: Name::new_static("override"),
            },
            Self::Cast => FuncId {
                module: ModuleName::typing(),
                cls: None,
                func: Name::new_static("cast"),
            },
            Self::AssertType => FuncId {
                module: ModuleName::typing(),
                cls: None,
                func: Name::new_static("assert_type"),
            },
            Self::RevealType => FuncId {
                module: ModuleName::typing(),
                cls: None,
                func: Name::new_static("reveal_type"),
            },
            Self::RuntimeCheckable => FuncId {
                module: ModuleName::typing(),
                cls: None,
                func: Name::new_static("runtime_checkable"),
            },
            Self::CallbackProtocol(cls) => FuncId {
                module: cls.qname().module_name(),
                cls: Some(cls.name().clone()),
                func: dunder::CALL,
            },
            Self::AbstractMethod => FuncId {
                module: ModuleName::abc(),
                cls: None,
                func: Name::new_static("abstractmethod"),
            },
            Self::TotalOrdering => FuncId {
                module: ModuleName::functools(),
                cls: None,
                func: Name::new_static("total_ordering"),
            },
            Self::Def(func_id) => (**func_id).clone(),
        }
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
