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
use std::ops::Deref;

use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_util::display::Fmt;
use pyrefly_util::owner::Owner;
use pyrefly_util::prelude::VecExt;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::Keyword;
use ruff_python_ast::name::Name;
use starlark_map::small_set::SmallSet;

use crate::display::TypeDisplayContext;
use crate::equality::TypeEq;
use crate::equality::TypeEqCtx;
use crate::type_output::DisplayOutput;
use crate::type_output::TypeOutput;
use crate::types::AnyStyle;
use crate::types::Type;

/// A wrapper for auxiliary data whose identity should be completely ignored
/// in equality, hashing, ordering, and type-equality comparisons.
/// `IdentityIgnored<T>` always compares as equal, hashes as a no-op, and
/// orders as `Equal` — making it transparent to all identity checks.
///
/// This is useful for attaching auxiliary data (e.g. closure caches) to
/// types that derive `PartialEq`, `Hash`, `Ord`, etc. without affecting
/// their logical identity.
#[derive(Debug, Clone)]
pub struct IdentityIgnored<T>(pub T);

impl<T> PartialEq for IdentityIgnored<T> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl<T> Eq for IdentityIgnored<T> {}

impl<T> Hash for IdentityIgnored<T> {
    fn hash<H: Hasher>(&self, _state: &mut H) {}
}

impl<T> PartialOrd for IdentityIgnored<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for IdentityIgnored<T> {
    fn cmp(&self, _other: &Self) -> Ordering {
        Ordering::Equal
    }
}

impl<T> Visit<Type> for IdentityIgnored<T> {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

impl<T> VisitMut<Type> for IdentityIgnored<T> {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}

impl<T> TypeEq for IdentityIgnored<T> {
    fn type_eq(&self, _other: &Self, _ctx: &mut TypeEqCtx) -> bool {
        true
    }
}

impl<T> Deref for IdentityIgnored<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct Callable {
    pub params: Params,
    pub ret: Type,
}

impl Callable {
    /// Returns true if this callable has the `*args: Any, **kwargs: Any -> Any`
    /// signature (plus an optional unannotated self/cls at index 0).
    /// Used as a heuristic in decorator type resolution for union-typed
    /// decorators.
    pub fn is_args_kwargs_wrapper(&self) -> bool {
        if !matches!(&self.ret, Type::Any(AnyStyle::Implicit)) {
            return false;
        }
        match &self.params {
            Params::List(params) | Params::Partial(params) => {
                let items = params.items();
                items.iter().any(|p| matches!(p, Param::Varargs(..)))
                    && items.iter().any(|p| matches!(p, Param::Kwargs(..)))
                    && items.iter().enumerate().all(|(i, p)| match p {
                        Param::Varargs(..) | Param::Kwargs(..) => true,
                        Param::Pos(_, ty, _) | Param::PosOnly(Some(_), ty, _) if i == 0 => {
                            matches!(ty, Type::Any(AnyStyle::Implicit))
                        }
                        _ => false,
                    })
            }
            _ => false,
        }
    }

    pub fn contains_callable_residual(&self) -> bool {
        let check = |t: &Type| matches!(t, Type::CallableResidual(_));
        if self.ret.any(check) {
            return true;
        }
        match &self.params {
            Params::List(params) | Params::Partial(params) => {
                params.items().iter().any(|p| p.as_type().any(check))
            }
            Params::ParamSpec(prefix, p) => {
                prefix.iter().any(|pp| {
                    let ty = match pp {
                        PrefixParam::PosOnly(_, ty, _) | PrefixParam::Pos(_, ty, _) => ty,
                    };
                    ty.any(check)
                }) || p.any(check)
            }
            Params::Ellipsis | Params::Materialization => false,
        }
    }

    /// Returns true if this callable carries no real type information: all
    /// parameters and the return type are `Any(Implicit)` (i.e. Unknown).
    pub fn is_fully_unknown(&self) -> bool {
        if !matches!(&self.ret, Type::Any(AnyStyle::Implicit)) {
            return false;
        }
        match &self.params {
            Params::List(params) | Params::Partial(params) => params
                .items()
                .iter()
                .all(|p| matches!(p.as_type(), Type::Any(AnyStyle::Implicit))),
            Params::Ellipsis => true,
            _ => false,
        }
    }
}

impl Display for Callable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ty = Type::Callable(Box::new(self.clone()));
        let ctx = TypeDisplayContext::new(&[&ty]);
        let mut output = DisplayOutput::new(&ctx, f);
        self.fmt_with_type(&mut output, &|t, o| ctx.fmt_helper_generic(t, false, o))
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
    pub overall: ArgCount,
}

/// Controls which parameters are displayed by `ParamList::fmt_with_type`
#[derive(Debug, Clone)]
pub enum ParamOverlay {
    /// Display all parameters
    All,
    /// Display only this set of named parameters, plus all anonymous parameters
    Subset(SmallSet<Name>),
}

#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct ParamList(Vec<Param>);

impl ParamList {
    pub fn new(xs: Vec<Param>) -> Self {
        Self(xs)
    }

    /// Create a new ParamList from a list of types,
    /// converting each one into a positional-only param
    pub fn new_types(xs: Vec<PrefixParam>) -> Self {
        Self(xs.into_map(|p| p.into_posonly_param()))
    }

    /// Prepend some parameters for `Concatenate`
    pub fn prepend_types(&self, pre: &[PrefixParam]) -> Cow<'_, ParamList> {
        if pre.is_empty() {
            Cow::Borrowed(self)
        } else {
            // Use `to_subset_param` to preserve Pos v.s. PosOnly
            Cow::Owned(ParamList(
                pre.iter()
                    .map(|p| p.to_param_preserve_name())
                    .chain(self.0.iter().cloned())
                    .collect(),
            ))
        }
    }

    pub fn fmt_with_type<O: TypeOutput>(
        &self,
        output: &mut O,
        write_type: &impl Fn(&Type, &mut O) -> fmt::Result,
        overlay: &ParamOverlay,
    ) -> fmt::Result {
        let mut named_posonly = false;
        let mut kwonly = false;
        let mut skipped_prev = false;
        for (i, param) in self.0.iter().enumerate() {
            // `/` or `*` markers that should be printed before the param
            let mut marker_prefixes = Vec::new();
            if matches!(param, Param::PosOnly(Some(_), _, _)) {
                named_posonly = true;
            } else if named_posonly {
                named_posonly = false;
                marker_prefixes.push("/");
            }
            if !kwonly && matches!(param, Param::KwOnly(..)) {
                kwonly = true;
                marker_prefixes.push("*");
            }
            // Should we elide the param?
            let skip = matches!(overlay, ParamOverlay::Subset(names) if param.name().is_some_and(|name| !names.contains(name)));
            if i > 0 && (!skipped_prev || !marker_prefixes.is_empty() || !skip) {
                output.write_str(", ")?;
            }
            for marker_prefix in marker_prefixes.iter() {
                output.write_str(marker_prefix)?;
                output.write_str(", ")?;
            }
            if skip {
                if !skipped_prev || !marker_prefixes.is_empty() {
                    output.write_str("...")?;
                }
            } else {
                param.fmt_with_type(output, write_type)?;
            }
            skipped_prev = skip;
        }
        if named_posonly {
            output.write_str(", /")?;
        }
        Ok(())
    }

    /// Format parameters each parameter on a new line
    pub fn fmt_with_type_with_newlines<O: TypeOutput>(
        &self,
        output: &mut O,
        write_type: &impl Fn(&Type, &mut O, usize) -> fmt::Result,
        indent: usize,
    ) -> fmt::Result {
        let mut named_posonly = false;
        let mut kwonly = false;

        for (i, param) in self.0.iter().enumerate() {
            if i > 0 {
                output.write_str(",\n")?;
                write_indent(output, indent)?;
            }

            if matches!(param, Param::PosOnly(Some(_), _, _)) {
                named_posonly = true;
            } else if named_posonly {
                named_posonly = false;
                output.write_str("/,\n")?;
                write_indent(output, indent)?;
            }

            if !kwonly && matches!(param, Param::KwOnly(..)) {
                kwonly = true;
                output.write_str("*,\n")?;
                write_indent(output, indent)?;
            }

            param.fmt_with_type(output, &|t, o| write_type(t, o, indent))?;
        }

        if named_posonly {
            output.write_str(",\n")?;
            write_indent(output, indent)?;
            output.write_str("/")?;
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

    /// Type signature that permits everything, namely `*args, **kwargs`.
    pub fn everything() -> ParamList {
        ParamList(vec![
            Param::Varargs(None, Type::any_implicit()),
            Param::Kwargs(None, Type::any_implicit()),
        ])
    }
}

/// True if `params` has both a `*args` and a `**kwargs` parameter and both are typed `Any`
/// (explicitly, or implicitly because they are unannotated). Per the typing spec such a
/// signature is equivalent to `...`.
pub fn params_are_gradual_variadic(params: &[Param]) -> bool {
    let has_vararg_any = params
        .iter()
        .any(|p| matches!(p, Param::Varargs(_, Type::Any(_))));
    let has_kwargs_any = params
        .iter()
        .any(|p| matches!(p, Param::Kwargs(_, Type::Any(_))));
    has_vararg_any && has_kwargs_any
}

fn write_indent<O: TypeOutput>(output: &mut O, indent: usize) -> fmt::Result {
    for _ in 0..indent {
        output.write_str(" ")?;
    }
    Ok(())
}

/// Represents a prefix parameter in `Concatenate`.
/// Prefix params can be either positional-only or positional (named).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum PrefixParam {
    PosOnly(Option<Name>, Type, Required),
    Pos(Name, Type, Required),
}

impl PrefixParam {
    /// Create a positional-only prefix param (no name).
    pub fn new(ty: Type, required: Required) -> Self {
        Self::PosOnly(None, ty, required)
    }

    pub fn ty(&self) -> &Type {
        match self {
            Self::PosOnly(_, ty, _) | Self::Pos(_, ty, _) => ty,
        }
    }

    /// Convert to a positional-only `Param`. Per the typing spec, params in
    /// `Concatenate` are positional-only at the call site. This is also appropriate
    /// for ParamSpec forwarding where prefix params must be passed positionally.
    pub fn into_posonly_param(self) -> Param {
        match self {
            Self::PosOnly(name, ty, required) => Param::PosOnly(name, ty, required),
            Self::Pos(name, ty, required) => Param::PosOnly(Some(name), ty, required),
        }
    }

    /// Convert to a `Param` preserving the Pos vs PosOnly distinction.
    /// Used for subset/subtype checking where name matching matters,
    /// and for direct calls where prefix params should remain keyword-passable.
    pub fn to_param_preserve_name(&self) -> Param {
        match self {
            Self::PosOnly(name, ty, required) => {
                Param::PosOnly(name.clone(), ty.clone(), required.clone())
            }
            Self::Pos(name, ty, required) => Param::Pos(name.clone(), ty.clone(), required.clone()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum Params {
    List(ParamList),
    /// The residual parameter list of a `functools.partial(...)`: behaves like `List` for
    /// call-checking, but is additionally recognized as assignable to `functools.partial[ret]`
    /// (see the subtyping rule in `subset.rs`). Carries the parameters left after binding a prefix.
    Partial(ParamList),
    Ellipsis,
    /// All possible materializations of `...`. A subset check with Callable[Materialization, R]
    /// succeeds only if it would succeed with Materialization replaced with any parameter list.
    /// See the comment on Type::Materialization - the intuition here is similar.
    Materialization,
    /// Any arguments to Concatenate, followed by a ParamSpec.
    /// E.g. `Concatenate[int, str, P]` would be `ParamSpec([int, str], P)`,
    /// while `P` alone would be `ParamSpec([], P)`.
    /// `P` may resolve to `Type::ParamSpecValue`, `Type::Concatenate`, or `Type::Ellipsis`
    ParamSpec(Box<[PrefixParam]>, Type),
}

impl Params {
    fn arg_counts(&self) -> ArgCounts {
        match self {
            Self::List(params) | Self::Partial(params) => {
                let mut counts = ArgCounts {
                    positional: ArgCount::none_allowed(),
                    keyword: ArgCount::none_allowed(),
                    overall: ArgCount::none_allowed(),
                };
                for param in params.items() {
                    match param {
                        Param::PosOnly(_, _, req) => {
                            counts.positional.add_arg(req);
                            counts.overall.add_arg(req);
                        }
                        Param::Pos(_, _, req) => {
                            counts.positional.add_arg(&Required::Optional(None));
                            counts.keyword.add_arg(&Required::Optional(None));
                            counts.overall.add_arg(req);
                        }
                        Param::KwOnly(_, _, req) => {
                            counts.keyword.add_arg(req);
                            counts.overall.add_arg(req);
                        }
                        Param::Varargs(..) => {
                            counts.positional.max = None;
                            counts.overall.max = None;
                        }
                        Param::Kwargs(..) => {
                            counts.keyword.max = None;
                            counts.overall.max = None;
                        }
                    }
                }
                counts
            }
            Self::Ellipsis | Self::Materialization => ArgCounts {
                positional: ArgCount::any_allowed(),
                keyword: ArgCount::any_allowed(),
                overall: ArgCount::any_allowed(),
            },
            Self::ParamSpec(prefix, _) => ArgCounts {
                positional: ArgCount {
                    min: prefix.len(),
                    max: None,
                },
                keyword: ArgCount::any_allowed(),
                overall: ArgCount {
                    min: prefix.len(),
                    max: None,
                },
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum Param {
    PosOnly(Option<Name>, Type, Required),
    Pos(Name, Type, Required),
    Varargs(Option<Name>, Type),
    KwOnly(Name, Type, Required),
    Kwargs(Option<Name>, Type),
}

/// The default value of an optional parameter, containing its type and an optional
/// display string for values whose types don't preserve the literal value (e.g. floats).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DefaultValue {
    pub ty: Type,
    /// Display string for defaults that can't be recovered from the type alone,
    /// e.g. `"3.14"` for float literals whose type is just `float`.
    pub display: Option<String>,
}

/// Visit/VisitMut/TypeEq delegate to `ty` only; `display` is display-only metadata.
impl<To> Visit<To> for DefaultValue
where
    Type: Visit<To>,
{
    const RECURSE_CONTAINS: bool = true;
    fn recurse<'a>(&'a self, f: &mut dyn FnMut(&'a To)) {
        self.ty.visit(f);
    }
}

impl<To> VisitMut<To> for DefaultValue
where
    Type: VisitMut<To>,
{
    const RECURSE_CONTAINS: bool = true;
    fn recurse_mut(&mut self, f: &mut dyn FnMut(&mut To)) {
        self.ty.visit_mut(f);
    }
}

impl TypeEq for DefaultValue {
    fn type_eq(&self, other: &Self, ctx: &mut TypeEqCtx) -> bool {
        self.ty.type_eq(&other.ty, ctx)
    }
}

impl DefaultValue {
    pub fn new(ty: Type) -> Self {
        Self { ty, display: None }
    }

    pub fn with_display(ty: Type, display: String) -> Self {
        Self {
            ty,
            display: Some(display),
        }
    }
}

/// Requiredness for a function parameter.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum Required {
    Required,
    /// The parameter is optional, with the default value info if available.
    Optional(Option<DefaultValue>),
}

impl Callable {
    pub fn fmt_with_type<O: TypeOutput>(
        &self,
        output: &mut O,
        write_type: &impl Fn(&Type, &mut O) -> fmt::Result,
    ) -> fmt::Result {
        match &self.params {
            Params::List(params) | Params::Partial(params) => {
                output.write_str("(")?;
                params.fmt_with_type(output, write_type, &ParamOverlay::All)?;
                output.write_str(") -> ")?;
                write_type(&self.ret, output)
            }
            Params::Ellipsis => {
                output.write_str("(...) -> ")?;
                write_type(&self.ret, output)
            }
            Params::Materialization => {
                output.write_str("(Materialization) -> ")?;
                write_type(&self.ret, output)
            }
            Params::ParamSpec(args, pspec) => {
                output.write_str("(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        output.write_str(", ")?;
                    }
                    write_type(arg.ty(), output)?;
                }
                match pspec {
                    Type::ParamSpecValue(params) => {
                        if !args.is_empty() && !params.is_empty() {
                            output.write_str(", ")?;
                        }
                        params.fmt_with_type(output, write_type, &ParamOverlay::All)?;
                    }
                    Type::Ellipsis => {
                        if !args.is_empty() {
                            output.write_str(", ")?;
                        }
                        output.write_str("...")?;
                    }
                    _ => {
                        if !args.is_empty() {
                            output.write_str(", ")?;
                        }
                        output.write_str("ParamSpec(")?;
                        write_type(pspec, output)?;
                        output.write_str(")")?;
                    }
                }
                output.write_str(") -> ")?;
                write_type(&self.ret, output)
            }
        }
    }

    /// Format the function type for use in a hover tooltip. This is similar to `fmt_with_type`, but
    /// it puts args on new lines if there is more than one argument
    pub fn fmt_with_type_with_newlines<O: TypeOutput>(
        &self,
        output: &mut O,
        write_type: &impl Fn(&Type, &mut O, usize) -> fmt::Result,
        indent: usize,
    ) -> fmt::Result {
        match &self.params {
            Params::List(params) | Params::Partial(params) if params.len() > 1 => {
                // For multiple parameters, put each on a new line with indentation
                let param_indent = indent + 4;
                output.write_str("(\n")?;
                write_indent(output, param_indent)?;
                params.fmt_with_type_with_newlines(output, write_type, param_indent)?;
                output.write_str("\n")?;
                write_indent(output, indent)?;
                output.write_str(") -> ")?;
                write_type(&self.ret, output, indent)
            }
            Params::ParamSpec(args, _) if !args.is_empty() => {
                let param_indent = indent + 4;
                output.write_str("(\n")?;
                write_indent(output, param_indent)?;
                self.fmt_param_spec_with_newlines(output, write_type, param_indent)?;
                output.write_str("\n")?;
                write_indent(output, indent)?;
                output.write_str(") -> ")?;
                write_type(&self.ret, output, indent)
            }
            Params::List(..)
            | Params::Partial(..)
            | Params::ParamSpec(..)
            | Params::Ellipsis
            | Params::Materialization => {
                self.fmt_with_type(output, &|t, o| write_type(t, o, indent))
            }
        }
    }

    fn fmt_param_spec_with_newlines<O: TypeOutput>(
        &self,
        output: &mut O,
        write_type: &impl Fn(&Type, &mut O, usize) -> fmt::Result,
        indent: usize,
    ) -> fmt::Result {
        let Params::ParamSpec(args, pspec) = &self.params else {
            unreachable!("only ParamSpec callables can be formatted as ParamSpec")
        };
        for (i, arg) in args.iter().enumerate() {
            if i > 0 {
                output.write_str(",\n")?;
                write_indent(output, indent)?;
            }
            write_type(arg.ty(), output, indent)?;
        }
        match pspec {
            Type::ParamSpecValue(params) if !params.is_empty() => {
                if !args.is_empty() {
                    output.write_str(",\n")?;
                    write_indent(output, indent)?;
                }
                params.fmt_with_type_with_newlines(output, write_type, indent)
            }
            Type::ParamSpecValue(_) => Ok(()),
            Type::Ellipsis => {
                if !args.is_empty() {
                    output.write_str(",\n")?;
                    write_indent(output, indent)?;
                }
                output.write_str("...")
            }
            _ => {
                if !args.is_empty() {
                    output.write_str(",\n")?;
                    write_indent(output, indent)?;
                }
                output.write_str("ParamSpec(")?;
                write_type(pspec, output, indent)?;
                output.write_str(")")
            }
        }
    }

    pub fn list(params: ParamList, ret: Type) -> Self {
        Self {
            params: Params::List(params),
            ret,
        }
    }

    /// Build a `Callable` carrying a [`Params::Partial`] residual signature, returning `ret`.
    pub fn partial(params: ParamList, ret: Type) -> Self {
        Self {
            params: Params::Partial(params),
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

    pub fn concatenate(args: Box<[PrefixParam]>, param_spec: Type, ret: Type) -> Self {
        Self {
            params: Params::ParamSpec(args, param_spec),
            ret,
        }
    }

    /// Return a new Callable with the first parameter removed (the `self` param for bound methods).
    pub fn strip_first_param(&self) -> Option<Self> {
        self.split_first_param(&mut Owner::new())
            .map(|(_, rest)| rest)
    }

    pub fn split_first_param<'a>(&'a self, owner: &'a mut Owner<Type>) -> Option<(&'a Type, Self)> {
        match self {
            Self {
                params: Params::List(params),
                ret,
            } => {
                let (first, rest) = params.0.split_first()?;
                if let Param::Varargs(_, first) = first {
                    Some((first, self.clone()))
                } else {
                    Some((
                        first.as_type(),
                        Self::list(ParamList(rest.to_vec()), ret.clone()),
                    ))
                }
            }
            Self {
                params: Params::Partial(params),
                ret,
            } => {
                let (first, rest) = params.0.split_first()?;
                if let Param::Varargs(_, first) = first {
                    Some((first, self.clone()))
                } else {
                    Some((
                        first.as_type(),
                        Self::partial(ParamList(rest.to_vec()), ret.clone()),
                    ))
                }
            }
            Self {
                params: Params::ParamSpec(ts, p),
                ret,
            } => {
                let (first, rest) = ts.split_first()?;
                Some((
                    first.ty(),
                    Self::concatenate(rest.iter().cloned().collect(), p.clone(), ret.clone()),
                ))
            }
            Self {
                params: Params::Ellipsis,
                ret: _,
            } => Some((owner.push(Type::any_implicit()), self.clone())),
            _ => None,
        }
    }

    pub fn get_first_param(&self) -> Option<&Type> {
        match &self.params {
            Params::List(params) | Params::Partial(params) => {
                params.0.first().map(|param| param.as_type())
            }
            Params::ParamSpec(prefix, _) => prefix.first().map(|param| param.ty()),
            Params::Ellipsis => Some(&Type::Any(AnyStyle::Implicit)),
            Params::Materialization => None,
        }
    }

    /// Type of the parameter at `index`, but only when it is a positional
    /// parameter of a concrete parameter list. Returns `None` for a
    /// non-positional param, an out-of-range index, or a non-`List`/`Partial` signature.
    /// A bound `self`/`cls` counts as index 0.
    pub fn get_positional_param(&self, index: usize) -> Option<&Type> {
        match &self.params {
            Params::List(params) | Params::Partial(params) => match params.0.get(index) {
                Some(Param::Pos(_, t, _) | Param::PosOnly(_, t, _)) => Some(t),
                _ => None,
            },
            _ => None,
        }
    }

    /// Whether this signature can be called with a single positional argument, i.e. none of the
    /// parameters after the first is required (positional or keyword-only). `*args`/`**kwargs`
    /// don't prevent it.
    pub fn accepts_single_positional_arg(&self) -> bool {
        match &self.params {
            Params::List(params) | Params::Partial(params) => match params.0.split_first() {
                Some((_, rest)) => !rest.iter().any(|p| {
                    matches!(
                        p,
                        Param::PosOnly(_, _, Required::Required)
                            | Param::Pos(_, _, Required::Required)
                            | Param::KwOnly(_, _, Required::Required)
                    )
                }),
                None => false,
            },
            Params::Ellipsis | Params::Materialization => true,
            // A bare `P` or `Concatenate[A, P]` can resolve to anything, so stay permissive; only
            // `Concatenate[A, B, ...]` (2+ prepended args) cannot be called with one positional.
            Params::ParamSpec(prefix, _) => prefix.len() <= 1,
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
    pub(crate) fn fmt_default(&self, default: &Option<DefaultValue>) -> String {
        match default {
            Some(DefaultValue {
                display: Some(text),
                ..
            }) => text.clone(),
            Some(DefaultValue {
                ty: Type::Literal(lit),
                ..
            }) => format!("{}", lit.value),
            Some(DefaultValue { ty: Type::None, .. }) => "None".to_owned(),
            _ => "...".to_owned(),
        }
    }

    pub fn fmt_with_type<O: TypeOutput>(
        &self,
        output: &mut O,
        write_type: &impl Fn(&Type, &mut O) -> fmt::Result,
    ) -> fmt::Result {
        match self {
            Param::PosOnly(None, ty, Required::Required) => write_type(ty, output),
            Param::PosOnly(None, ty, Required::Optional(default)) => {
                output.write_str("_: ")?;
                write_type(ty, output)?;
                output.write_str(" = ")?;
                output.write_str(&self.fmt_default(default))
            }
            Param::PosOnly(Some(name), ty, Required::Required)
            | Param::Pos(name, ty, Required::Required)
            | Param::KwOnly(name, ty, Required::Required) => {
                output.write_str(name.as_str())?;
                output.write_str(": ")?;
                write_type(ty, output)
            }
            Param::PosOnly(Some(name), ty, Required::Optional(default))
            | Param::Pos(name, ty, Required::Optional(default))
            | Param::KwOnly(name, ty, Required::Optional(default)) => {
                output.write_str(name.as_str())?;
                output.write_str(": ")?;
                write_type(ty, output)?;
                output.write_str(" = ")?;
                output.write_str(&self.fmt_default(default))
            }
            Param::Varargs(Some(name), ty) => {
                output.write_str("*")?;
                output.write_str(name.as_str())?;
                output.write_str(": ")?;
                write_type(ty, output)
            }
            Param::Varargs(None, ty) => {
                output.write_str("*")?;
                write_type(ty, output)
            }
            Param::Kwargs(Some(name), ty) => {
                output.write_str("**")?;
                output.write_str(name.as_str())?;
                output.write_str(": ")?;
                write_type(ty, output)
            }
            Param::Kwargs(None, ty) => {
                output.write_str("**")?;
                write_type(ty, output)
            }
        }
    }

    pub fn name(&self) -> Option<&Name> {
        match self {
            Param::PosOnly(name, ..) | Param::Varargs(name, ..) | Param::Kwargs(name, ..) => {
                name.as_ref()
            }
            Param::Pos(name, ..) | Param::KwOnly(name, ..) => Some(name),
        }
    }

    pub fn as_type(&self) -> &Type {
        match self {
            Param::PosOnly(_, ty, _)
            | Param::Pos(_, ty, _)
            | Param::Varargs(_, ty)
            | Param::KwOnly(_, ty, _)
            | Param::Kwargs(_, ty) => ty,
        }
    }

    pub fn as_type_mut(&mut self) -> &mut Type {
        match self {
            Param::PosOnly(_, ty, _)
            | Param::Pos(_, ty, _)
            | Param::Varargs(_, ty)
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

    /// Format a parameter for display using the proper type display infrastructure.
    /// This ensures consistent formatting with default values, position-only markers, etc.
    ///
    /// This is similar to the `Display` impl, but allows passing in a `TypeDisplayContext`
    /// for context-aware formatting (e.g., disambiguating types with the same name).
    pub fn format_for_signature(&self, type_ctx: &TypeDisplayContext) -> String {
        format!(
            "{}",
            Fmt(|f| {
                let mut output = DisplayOutput::new(type_ctx, f);
                self.fmt_with_type(&mut output, &|ty, o| {
                    type_ctx.fmt_helper_generic(ty, false, o)
                })
            })
        )
    }
}

impl Display for Param {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ctx = TypeDisplayContext::new(&[self.as_type()]);
        let mut output = DisplayOutput::new(&ctx, f);
        self.fmt_with_type(&mut output, &|t, o| ctx.fmt_helper_generic(t, false, o))
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
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_util::owner::Owner;
    use pyrefly_util::visit::Visit;
    use pyrefly_util::visit::VisitMut;
    use ruff_python_ast::name::Name;
    use ruff_text_size::TextRange;

    use crate::callable::Callable;
    use crate::callable::DefaultValue;
    use crate::callable::Param;
    use crate::callable::ParamList;
    use crate::callable::Params;
    use crate::callable::PrefixParam;
    use crate::callable::Required;
    use crate::quantified::AnchorIndex;
    use crate::quantified::Quantified;
    use crate::quantified::QuantifiedIdentity;
    use crate::quantified::QuantifiedKind;
    use crate::quantified::QuantifiedOrigin;
    use crate::type_var::PreInferenceVariance;
    use crate::type_var::Restriction;
    use crate::types::Type;

    #[test]
    fn test_get_first_param_matches_split_first_param() {
        let callables = [
            Callable::list(ParamList::new(Vec::new()), Type::None),
            Callable::list(
                ParamList::new(vec![Param::PosOnly(None, Type::None, Required::Required)]),
                Type::None,
            ),
            Callable::partial(
                ParamList::new(vec![Param::Varargs(None, Type::any_implicit())]),
                Type::None,
            ),
            Callable::concatenate(
                vec![PrefixParam::new(Type::None, Required::Required)].into_boxed_slice(),
                Type::any_implicit(),
                Type::None,
            ),
            Callable::param_spec(Type::any_implicit(), Type::None),
            Callable::ellipsis(Type::None),
            Callable {
                params: Params::Materialization,
                ret: Type::None,
            },
        ];
        for callable in callables {
            let expected = callable
                .split_first_param(&mut Owner::new())
                .map(|(first, _)| first.clone());
            assert_eq!(callable.get_first_param(), expected.as_ref());
        }
    }

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
            ParamList::new(vec![Param::Varargs(None, Type::any_implicit())]),
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
                Param::Varargs(None, Type::any_implicit()),
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
            vec![
                PrefixParam::new(Type::None, Required::Required),
                PrefixParam::new(Type::None, Required::Required),
            ]
            .into_boxed_slice(),
            Type::any_implicit(),
            Type::None,
        );
        let counts = callable.arg_counts();
        assert_eq!(counts.positional.min, 2);
        assert_eq!(counts.positional.max, None);
        assert_eq!(counts.keyword.min, 0);
        assert_eq!(counts.keyword.max, None);
    }

    #[test]
    fn test_default_value_visit_delegates_to_ty() {
        let q = Quantified::new(
            QuantifiedIdentity::new(
                ModuleName::from_str("__test__"),
                AnchorIndex::first(TextRange::default()),
                QuantifiedOrigin::Pep695,
            ),
            Name::new("T"),
            QuantifiedKind::TypeVar,
            None,
            Restriction::Unrestricted,
            PreInferenceVariance::Invariant,
        );
        let quantified_ty = Type::Quantified(Box::new(q));
        let default = DefaultValue::with_display(quantified_ty.clone(), "default".to_owned());

        // Visit should yield the inner type from ty, not the display metadata.
        let mut visited = Vec::new();
        default.visit(&mut |ty: &Type| visited.push(ty.clone()));
        assert_eq!(visited, vec![quantified_ty]);
    }

    #[test]
    fn test_default_value_visit_mut_delegates_to_ty() {
        let q = Quantified::new(
            QuantifiedIdentity::new(
                ModuleName::from_str("__test__"),
                AnchorIndex::new(TextRange::default(), 1),
                QuantifiedOrigin::Pep695,
            ),
            Name::new("T"),
            QuantifiedKind::TypeVar,
            None,
            Restriction::Unrestricted,
            PreInferenceVariance::Invariant,
        );
        let mut default =
            DefaultValue::with_display(Type::Quantified(Box::new(q)), "default".to_owned());

        // VisitMut should be able to mutate the inner type.
        default.visit_mut(&mut |ty: &mut Type| {
            *ty = Type::None;
        });
        assert_eq!(default.ty, Type::None);
        // Display metadata should be unaffected.
        assert_eq!(default.display, Some("default".to_owned()));
    }
}
