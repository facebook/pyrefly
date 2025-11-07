/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Display a type. The complexity comes from if we have two classes with the same name,
//! we want to display disambiguating information (e.g. module name or location).
use std::fmt;
use std::fmt::Display;

use pyrefly_python::module_name::ModuleName;
use pyrefly_python::qname::QName;
use pyrefly_util::display::Fmt;
use pyrefly_util::display::append;
use pyrefly_util::display::commas_iter;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::small_map::Entry;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use starlark_map::smallmap;

use crate::callable::Function;
use crate::class::Class;
use crate::literal::Lit;
use crate::tuple::Tuple;
use crate::types::AnyStyle;
use crate::types::BoundMethod;
use crate::types::BoundMethodType;
use crate::types::Forall;
use crate::types::Forallable;
use crate::types::NeverStyle;
use crate::types::SuperObj;
use crate::types::TArgs;
use crate::types::TParam;
use crate::types::Type;

/// Information about the qnames we have seen.
/// Set to None to indicate we have seen different values, or Some if they are all the same.
#[derive(Clone, Debug)]
struct QNameInfo {
    /// For each module, record either the one unique range, or None if there are multiple.
    info: SmallMap<ModuleName, Option<TextRange>>,
}

impl QNameInfo {
    fn new(qname: &QName) -> Self {
        Self {
            info: smallmap! {qname.module_name() => Some(qname.range())},
        }
    }

    fn qualified() -> Self {
        Self {
            info: SmallMap::new(),
        }
    }

    fn update(&mut self, qname: &QName) {
        match self.info.entry(qname.module_name()) {
            Entry::Vacant(e) => {
                e.insert(Some(qname.range()));
            }
            Entry::Occupied(mut e) => {
                if e.get() != &Some(qname.range()) {
                    *e.get_mut() = None;
                }
            }
        }
    }

    fn fmt(&self, qname: &QName, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let module_name = qname.module_name();
        match self.info.get(&module_name) {
            Some(None) | None => qname.fmt_with_location(f),
            _ if self.info.len() > 1 => qname.fmt_with_module(f),
            _ => qname.fmt_name(f),
        }
    }
}

#[derive(Debug, Default)]
pub struct TypeDisplayContext<'a> {
    qnames: SmallMap<&'a Name, QNameInfo>,
    /// Should we display for IDE Hover? This makes type names more readable but less precise.
    hover: bool,
    always_display_module_name: bool,
}

impl<'a> TypeDisplayContext<'a> {
    pub fn new(xs: &[&'a Type]) -> Self {
        let mut res = Self::default();
        for x in xs {
            res.add(x);
        }
        res
    }

    fn add_qname(&mut self, qname: &'a QName) {
        match self.qnames.entry(qname.id()) {
            Entry::Vacant(e) => {
                e.insert(QNameInfo::new(qname));
            }
            Entry::Occupied(mut e) => e.get_mut().update(qname),
        }
    }

    pub fn add(&mut self, t: &'a Type) {
        t.universe(&mut |t| {
            if let Some(qname) = t.qname() {
                self.add_qname(qname);
            }
        })
    }

    /// Force that we always display at least the module name for qualified names.
    pub fn always_display_module_name(&mut self) {
        // We pretend that every qname is also in a fake module, and thus requires disambiguating.
        let fake_module = ModuleName::from_str("__pyrefly__type__display__context__");
        for c in self.qnames.values_mut() {
            c.info.insert(fake_module, None);
        }
        self.always_display_module_name = true;
    }

    /// Always display the module name, except for builtins.
    pub fn always_display_module_name_except_builtins(&mut self) {
        let builtins_module = ModuleName::from_str("builtins");
        let fake_module = ModuleName::from_str("__pyrefly__type__display__context__");
        for c in self.qnames.values_mut() {
            if c.info.len() > 1 {
                continue; // Multiple modules, so we need to keep the module name to disambiguate.
            }
            if let Some(value) = c.info.get_mut(&builtins_module) {
                // Name is a builtin, we set it a default location so we hit the fallback branch in `QNameInfo::fmt`.
                *value = Some(TextRange::default());
            } else {
                // Name is not a builtins, so we add a fake module to force the module name to be displayed.
                c.info.insert(fake_module, None);
            }
        }
        self.always_display_module_name = true;
    }

    /// Set the context to display for hover. This makes type names more readable but less precise.
    pub fn set_display_mode_to_hover(&mut self) {
        self.hover = true;
    }

    pub fn display(&'a self, t: &'a Type) -> impl Display + 'a {
        Fmt(|f| self.fmt(t, f))
    }

    // Private method for internal use
    fn display_internal(&'a self, t: &'a Type) -> impl Display + 'a {
        Fmt(|f| self.fmt_helper(t, f, false))
    }

    fn fmt_targ(&self, param: &TParam, arg: &Type, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if param.quantified.is_type_var_tuple()
            && let Type::Tuple(tuple) = arg
        {
            match tuple {
                Tuple::Concrete(elts) if !elts.is_empty() => write!(
                    f,
                    "{}",
                    commas_iter(|| elts.iter().map(|elt| self.display_internal(elt)))
                ),
                Tuple::Unpacked(box (prefix, middle, suffix)) => {
                    let unpacked_middle = Type::Unpack(Box::new(middle.clone()));
                    write!(
                        f,
                        "{}",
                        commas_iter(|| {
                            prefix
                                .iter()
                                .chain(std::iter::once(&unpacked_middle))
                                .chain(suffix.iter())
                                .map(|elt| self.display_internal(elt))
                        })
                    )
                }
                _ => {
                    write!(f, "*{}", self.display_internal(arg))
                }
            }
        } else {
            write!(f, "{}", self.display_internal(arg))
        }
    }

    fn fmt_targs(&self, targs: &TArgs, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !targs.is_empty() {
            write!(
                f,
                "[{}]",
                commas_iter(|| targs
                    .iter_paired()
                    .map(|(param, arg)| Fmt(|f| self.fmt_targ(param, arg, f))))
            )
        } else {
            Ok(())
        }
    }

    fn fmt_qname(&self, qname: &QName, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.qnames.get(&qname.id()) {
            Some(info) => info.fmt(qname, f),
            None => QNameInfo::qualified().fmt(qname, f), // we should not get here, if we do, be safe
        }
    }

    fn fmt_lit(&self, lit: &Lit, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match lit {
            Lit::Enum(e) => {
                self.fmt_qname(e.class.qname(), f)?;
                write!(f, ".{}", e.member)
            }
            _ => write!(f, "{lit}"),
        }
    }

    fn fmt<'b>(&self, t: &'b Type, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_helper(t, f, true)
    }

    fn maybe_fmt_with_module(
        &self,
        module: &str,
        name: &str,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        if self.always_display_module_name {
            write!(f, "{module}.{name}")
        } else {
            write!(f, "{name}")
        }
    }

    fn fmt_helper<'b>(
        &self,
        t: &'b Type,
        f: &mut fmt::Formatter<'_>,
        is_toplevel: bool,
    ) -> fmt::Result {
        match t {
            // Things that have QName's and need qualifying
            Type::ClassDef(cls) => {
                write!(f, "type[")?;
                self.fmt_qname(cls.qname(), f)?;
                write!(f, "]")
            }
            Type::ClassType(class_type)
                if class_type.qname().module_name().as_str() == "builtins"
                    && class_type.qname().id().as_str() == "tuple"
                    && class_type.targs().as_slice().len() == 1 =>
            {
                self.fmt_qname(class_type.qname(), f)?;
                write!(
                    f,
                    "[{}, ...]",
                    self.display_internal(&class_type.targs().as_slice()[0])
                )
            }
            Type::ClassType(class_type) => {
                self.fmt_qname(class_type.qname(), f)?;
                self.fmt_targs(class_type.targs(), f)
            }
            Type::TypedDict(typed_dict) => {
                write!(f, "TypedDict[")?;
                self.fmt_qname(typed_dict.qname(), f)?;
                self.fmt_targs(typed_dict.targs(), f)?;
                write!(f, "]")
            }
            Type::PartialTypedDict(typed_dict) => {
                write!(f, "Partial[")?;
                self.fmt_qname(typed_dict.qname(), f)?;
                self.fmt_targs(typed_dict.targs(), f)?;
                write!(f, "]")
            }
            Type::TypeVar(t) => {
                write!(f, "TypeVar[")?;
                self.fmt_qname(t.qname(), f)?;
                write!(f, "]")
            }
            Type::TypeVarTuple(t) => {
                write!(f, "TypeVarTuple[")?;
                self.fmt_qname(t.qname(), f)?;
                write!(f, "]")
            }
            Type::ParamSpec(t) => {
                write!(f, "ParamSpec[")?;
                self.fmt_qname(t.qname(), f)?;
                write!(f, "]")
            }
            Type::SelfType(cls) => {
                self.maybe_fmt_with_module("typing", "Self@", f)?;
                self.fmt_qname(cls.qname(), f)
            }

            // Other things
            Type::Literal(lit) => {
                self.maybe_fmt_with_module("typing", "Literal", f)?;
                write!(f, "[")?;
                self.fmt_lit(lit, f)?;
                write!(f, "]")
            }
            Type::LiteralString => self.maybe_fmt_with_module("typing", "LiteralString", f),
            Type::Callable(box c) => {
                if self.hover && is_toplevel {
                    c.fmt_with_type_with_newlines(f, &|t| self.display_internal(t))
                } else {
                    c.fmt_with_type(f, &|t| self.display_internal(t))
                }
            }
            Type::Function(box Function {
                signature,
                metadata,
            }) => {
                if self.hover && is_toplevel {
                    let func_name = metadata.kind.function_name();
                    write!(f, "def {func_name}")?;
                    signature.fmt_with_type_with_newlines(f, &|t| self.display_internal(t))?;
                    write!(f, ": ...")
                } else {
                    signature.fmt_with_type(f, &|t| self.display_internal(t))
                }
            }
            Type::Overload(overload) => {
                if self.hover && is_toplevel {
                    write!(
                        f,
                        "\n@overload\n{}",
                        self.display(&overload.signatures.first().as_type())
                    )?;
                    for sig in overload.signatures.iter().skip(1) {
                        write!(f, "\n{}", self.display(&sig.as_type()))?;
                    }
                    Ok(())
                } else {
                    write!(
                        f,
                        "Overload[{}",
                        self.display_internal(&overload.signatures.first().as_type())
                    )?;
                    for sig in overload.signatures.iter().skip(1) {
                        write!(f, ", {}", self.display_internal(&sig.as_type()))?;
                    }
                    write!(f, "]")
                }
            }
            Type::ParamSpecValue(x) => {
                write!(f, "[")?;
                x.fmt_with_type(f, &|t| self.display_internal(t))?;
                write!(f, "]")
            }
            Type::BoundMethod(box BoundMethod { obj, func }) => {
                if self.hover && is_toplevel {
                    match func {
                        BoundMethodType::Function(Function {
                            signature,
                            metadata,
                        }) => {
                            let func_name = metadata.kind.function_name();
                            write!(f, "def {func_name}")?;
                            signature
                                .fmt_with_type_with_newlines(f, &|t| self.display_internal(t))?;
                            write!(f, ": ...")
                        }
                        BoundMethodType::Forall(Forall {
                            tparams,
                            body:
                                Function {
                                    signature,
                                    metadata,
                                },
                        }) => {
                            let func_name = metadata.kind.function_name();
                            write!(f, "def {func_name}")?;
                            write!(f, "[{}]", commas_iter(|| tparams.iter()))?;
                            signature
                                .fmt_with_type_with_newlines(f, &|t| self.display_internal(t))?;
                            write!(f, ": ...")
                        }
                        BoundMethodType::Overload(_) => {
                            // Use display instead of display_internal to show overloads w/ top-level formatting
                            write!(f, "{}", self.display(&func.clone().as_type()))
                        }
                    }
                } else if self.hover {
                    write!(f, "{}", self.display_internal(&func.clone().as_type()))
                } else {
                    write!(
                        f,
                        "BoundMethod[{}, {}]",
                        self.display_internal(obj),
                        self.display_internal(&func.clone().as_type())
                    )
                }
            }
            Type::Never(NeverStyle::NoReturn) => {
                self.maybe_fmt_with_module("typing", "NoReturn", f)
            }
            Type::Never(NeverStyle::Never) => self.maybe_fmt_with_module("typing", "Never", f),
            Type::Union(types) if types.is_empty() => {
                self.maybe_fmt_with_module("typing", "Never", f)
            }
            Type::Union(types) => {
                // All Literals will be collected into a single Literal at the index of the first Literal.
                let mut literal_idx = None;
                let mut literals = Vec::new();
                let mut display_types = Vec::new();
                for (i, t) in types.iter().enumerate() {
                    match t {
                        Type::Literal(lit) => {
                            if literal_idx.is_none() {
                                literal_idx = Some(i);
                            }
                            literals.push(format!("{}", Fmt(|f| self.fmt_lit(lit, f))))
                        }
                        Type::Callable(_) | Type::Function(_) => {
                            display_types.push(format!("({})", self.display_internal(t)))
                        }
                        _ => display_types.push(format!("{}", self.display_internal(t))),
                    }
                }
                if let Some(i) = literal_idx {
                    if self.always_display_module_name {
                        display_types
                            .insert(i, format!("typing.Literal[{}]", commas_iter(|| &literals)));
                    } else {
                        display_types.insert(i, format!("Literal[{}]", commas_iter(|| &literals)));
                    }
                }
                // This is mainly to prettify types for functions with different names but the same signature
                let display_types_deduped = display_types
                    .into_iter()
                    .collect::<SmallSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>();
                write!(f, "{}", display_types_deduped.join(" | "))
            }
            Type::Intersect(types) => {
                write!(
                    f,
                    "Intersect[{}]",
                    commas_iter(|| types.iter().map(|t| self.display_internal(t)))
                )
            }
            Type::Tuple(t) => t.fmt_with_type(f, |t| self.display_internal(t)),
            Type::Forall(box Forall {
                tparams,
                body: body @ Forallable::Callable(c),
            }) => {
                if self.hover && is_toplevel {
                    write!(f, "[{}]", commas_iter(|| tparams.iter()))?;
                    c.fmt_with_type_with_newlines(f, &|t| self.display_internal(t))
                } else {
                    write!(
                        f,
                        "[{}]{}",
                        commas_iter(|| tparams.iter()),
                        self.display_internal(&body.clone().as_type()),
                    )
                }
            }
            Type::Forall(box Forall {
                tparams,
                body:
                    body @ Forallable::Function(Function {
                        signature,
                        metadata,
                        ..
                    }),
            }) => {
                if self.hover && is_toplevel {
                    let func_name = metadata.kind.function_name();
                    write!(f, "def {func_name}")?;
                    write!(f, "[{}]", commas_iter(|| tparams.iter()))?;
                    signature.fmt_with_type_with_newlines(f, &|t| self.display_internal(t))?;
                    write!(f, ": ...")
                } else {
                    write!(
                        f,
                        "[{}]{}",
                        commas_iter(|| tparams.iter()),
                        self.display_internal(&body.clone().as_type()),
                    )
                }
            }
            Type::Forall(box Forall {
                tparams,
                body: Forallable::TypeAlias(ta),
            }) => {
                if is_toplevel {
                    ta.fmt_with_type(f, &|t| self.display_internal(t), Some(tparams))
                } else {
                    write!(f, "{}", *ta.name)
                }
            }
            Type::Type(ty) => write!(f, "type[{}]", self.display_internal(ty)),
            Type::TypeGuard(ty) => {
                self.maybe_fmt_with_module("typing", "TypeGuard", f)?;
                write!(f, "[{}]", self.display_internal(ty))
            }
            Type::TypeIs(ty) => {
                self.maybe_fmt_with_module("typing", "TypeIs", f)?;
                write!(f, "[{}]", self.display_internal(ty))
            }
            Type::Unpack(box ty @ Type::TypedDict(_)) => {
                self.maybe_fmt_with_module("typing", "Unpack", f)?;
                write!(f, "[{}]", self.display_internal(ty))
            }
            Type::Unpack(ty) => write!(f, "*{}", self.display_internal(ty)),
            Type::Concatenate(args, pspec) => {
                self.maybe_fmt_with_module("typing", "Concatenate", f)?;
                write!(f, "[{}]", commas_iter(|| append(args.iter(), [pspec])))
            }
            Type::Module(m) => write!(f, "Module[{m}]"),
            Type::Var(var) => write!(f, "{var}"),
            Type::Quantified(var) => write!(f, "{var}"),
            Type::QuantifiedValue(var) => write!(f, "{var}"),
            Type::Args(q) => {
                write!(f, "Args[{q}]")
            }
            Type::Kwargs(q) => {
                write!(f, "Kwargs[{q}]")
            }
            Type::ArgsValue(q) => {
                write!(f, "ArgsValue[{q}]")
            }
            Type::KwargsValue(q) => {
                write!(f, "KwargsValue[{q}]")
            }
            Type::SpecialForm(x) => write!(f, "{x}"),
            Type::Ellipsis => write!(f, "Ellipsis"),
            Type::Any(style) => match style {
                AnyStyle::Explicit => self.maybe_fmt_with_module("typing", "Any", f),
                AnyStyle::Implicit | AnyStyle::Error => write!(f, "Unknown"),
            },
            Type::TypeAlias(ta) => {
                if is_toplevel {
                    ta.fmt_with_type(f, &|t| self.display_internal(t), None)
                } else {
                    write!(f, "{}", *ta.name)
                }
            }
            Type::SuperInstance(box (cls, obj)) => {
                write!(f, "super[")?;
                self.fmt_qname(cls.qname(), f)?;
                write!(f, ", ")?;
                match obj {
                    SuperObj::Instance(obj) => {
                        self.fmt_qname(obj.qname(), f)?;
                        self.fmt_targs(obj.targs(), f)?;
                    }
                    SuperObj::Class(cls) => {
                        self.fmt_qname(cls.qname(), f)?;
                    }
                }
                write!(f, "]")
            }
            Type::KwCall(call) => self.fmt_helper(&call.return_ty, f, false),
            Type::Materialization => write!(f, "Materialization"),
            Type::None => write!(f, "None"),
        }
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TypeDisplayContext::new(&[self]).fmt(self, f)
    }
}

impl Type {
    pub fn as_hover_string(&self) -> String {
        let mut c = TypeDisplayContext::new(&[self]);
        c.set_display_mode_to_hover();
        c.display(self).to_string()
    }
}

pub struct ClassDisplayContext<'a>(TypeDisplayContext<'a>);

impl<'a> ClassDisplayContext<'a> {
    pub fn new(classes: &[&'a Class]) -> Self {
        let mut ctx = TypeDisplayContext::new(&[]);
        for cls in classes {
            ctx.add_qname(cls.qname());
        }
        Self(ctx)
    }

    pub fn display(&'a self, cls: &'a Class) -> impl Display + 'a {
        Fmt(|f| self.0.fmt_qname(cls.qname(), f))
    }
}

#[cfg(test)]
pub mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use dupe::Dupe;
    use pyrefly_python::module::Module;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use pyrefly_python::nesting_context::NestingContext;
    use pyrefly_util::uniques::UniqueFactory;
    use ruff_python_ast::Identifier;
    use ruff_text_size::TextSize;
    use vec1::vec1;

    use super::*;
    use crate::callable::Callable;
    use crate::callable::FuncMetadata;
    use crate::callable::Function;
    use crate::callable::Param;
    use crate::callable::ParamList;
    use crate::callable::Required;
    use crate::class::Class;
    use crate::class::ClassDefIndex;
    use crate::class::ClassType;
    use crate::literal::Lit;
    use crate::literal::LitEnum;
    use crate::quantified::Quantified;
    use crate::quantified::QuantifiedKind;
    use crate::tuple::Tuple;
    use crate::type_var::PreInferenceVariance;
    use crate::type_var::Restriction;
    use crate::type_var::TypeVar;
    use crate::typed_dict::TypedDict;
    use crate::types::BoundMethodType;
    use crate::types::Overload;
    use crate::types::OverloadType;
    use crate::types::TParam;
    use crate::types::TParams;
    use crate::types::TypeAlias;
    use crate::types::TypeAliasStyle;

    pub fn fake_class(name: &str, module: &str, range: u32) -> Class {
        let mi = Module::new(
            ModuleName::from_str(module),
            ModulePath::filesystem(PathBuf::from(module)),
            Arc::new("1234567890".to_owned()),
        );

        Class::new(
            ClassDefIndex(0),
            Identifier::new(Name::new(name), TextRange::empty(TextSize::new(range))),
            NestingContext::toplevel(),
            mi,
            None,
            SmallMap::new(),
        )
    }

    pub fn fake_tparams(tparams: Vec<TParam>) -> Arc<TParams> {
        Arc::new(TParams::new(tparams))
    }

    fn fake_tparam(uniques: &UniqueFactory, name: &str, kind: QuantifiedKind) -> TParam {
        TParam {
            quantified: Quantified::new(
                uniques.fresh(),
                Name::new(name),
                kind,
                None,
                Restriction::Unrestricted,
            ),
            variance: PreInferenceVariance::PInvariant,
        }
    }

    fn fake_tyvar(name: &str, module: &str, range: u32) -> TypeVar {
        let mi = Module::new(
            ModuleName::from_str(module),
            ModulePath::filesystem(PathBuf::from(module)),
            Arc::new("1234567890".to_owned()),
        );
        TypeVar::new(
            Identifier::new(Name::new(name), TextRange::empty(TextSize::new(range))),
            mi,
            Restriction::Unrestricted,
            None,
            PreInferenceVariance::PInvariant,
        )
    }

    fn fake_bound_method(method_name: &str, class_name: &str, module_name_str: &str) -> Type {
        let class = fake_class(class_name, module_name_str, 10);
        let method = Callable::list(
            ParamList::new(vec![
                Param::Pos(
                    Name::new_static("self"),
                    Type::any_explicit(),
                    Required::Required,
                ),
                Param::Pos(
                    Name::new_static("x"),
                    Type::any_explicit(),
                    Required::Required,
                ),
                Param::Pos(
                    Name::new_static("y"),
                    Type::any_explicit(),
                    Required::Required,
                ),
            ]),
            Type::None,
        );
        Type::BoundMethod(Box::new(BoundMethod {
            obj: Type::ClassDef(class.dupe()),
            func: BoundMethodType::Function(Function {
                signature: method,
                metadata: FuncMetadata::def(
                    class.dupe().module().dupe(),
                    class.dupe(),
                    Name::new(method_name),
                ),
            }),
        }))
    }

    fn fake_generic_bound_method(
        method_name: &str,
        class_name: &str,
        module_name_str: &str,
        tparams: Arc<TParams>,
    ) -> Type {
        let class = fake_class(class_name, module_name_str, 10);
        let method = Callable::list(
            ParamList::new(vec![
                Param::Pos(
                    Name::new_static("self"),
                    Type::any_explicit(),
                    Required::Required,
                ),
                Param::Pos(
                    Name::new_static("x"),
                    Type::any_explicit(),
                    Required::Required,
                ),
                Param::Pos(
                    Name::new_static("y"),
                    Type::any_explicit(),
                    Required::Required,
                ),
            ]),
            Type::None,
        );
        Type::BoundMethod(Box::new(BoundMethod {
            obj: Type::ClassDef(class.dupe()),
            func: BoundMethodType::Forall(Forall {
                tparams,
                body: Function {
                    signature: method,
                    metadata: FuncMetadata::def(
                        class.dupe().module().dupe(),
                        class.dupe(),
                        Name::new(method_name),
                    ),
                },
            }),
        }))
    }

    #[test]
    fn test_display() {
        let uniques = UniqueFactory::new();
        let foo1 = fake_class("foo", "mod.ule", 5);
        let foo2 = fake_class("foo", "mod.ule", 8);
        let foo3 = fake_class("foo", "ule", 3);
        let bar = fake_class("bar", "mod.ule", 0);
        let bar_tparams = fake_tparams(vec![fake_tparam(&uniques, "T", QuantifiedKind::TypeVar)]);
        let tuple_param = fake_class("TupleParam", "mod.ule", 0);
        let tuple_param_tparams = fake_tparams(vec![fake_tparam(
            &uniques,
            "T",
            QuantifiedKind::TypeVarTuple,
        )]);
        fn class_type(class: &Class, targs: TArgs) -> Type {
            Type::ClassType(ClassType::new(class.dupe(), targs))
        }

        assert_eq!(
            class_type(
                &tuple_param,
                TArgs::new(
                    tuple_param_tparams.dupe(),
                    vec![Type::tuple(vec![
                        class_type(&foo1, TArgs::default()),
                        class_type(&foo1, TArgs::default())
                    ])]
                )
            )
            .to_string(),
            "TupleParam[foo, foo]"
        );
        assert_eq!(
            class_type(
                &tuple_param,
                TArgs::new(tuple_param_tparams.dupe(), vec![Type::tuple(Vec::new())])
            )
            .to_string(),
            "TupleParam[*tuple[()]]"
        );
        assert_eq!(
            class_type(
                &tuple_param,
                TArgs::new(
                    tuple_param_tparams.dupe(),
                    vec![Type::Tuple(Tuple::unbounded(class_type(
                        &foo1,
                        TArgs::default()
                    )))]
                )
            )
            .to_string(),
            "TupleParam[*tuple[foo, ...]]"
        );
        assert_eq!(
            class_type(
                &tuple_param,
                TArgs::new(
                    tuple_param_tparams.dupe(),
                    vec![Type::Tuple(Tuple::Unpacked(Box::new((
                        vec![class_type(&foo1, TArgs::default())],
                        Type::Tuple(Tuple::unbounded(class_type(&foo1, TArgs::default(),))),
                        vec![class_type(&foo1, TArgs::default())],
                    ))))]
                )
            )
            .to_string(),
            "TupleParam[foo, *tuple[foo, ...], foo]"
        );

        assert_eq!(
            Type::Tuple(Tuple::unbounded(class_type(&foo1, TArgs::default()))).to_string(),
            "tuple[foo, ...]"
        );
        assert_eq!(
            Type::Tuple(Tuple::concrete(vec![
                class_type(&foo1, TArgs::default()),
                class_type(
                    &bar,
                    TArgs::new(
                        bar_tparams.dupe(),
                        vec![class_type(&foo1, TArgs::default())]
                    )
                )
            ]))
            .to_string(),
            "tuple[foo, bar[foo]]"
        );
        assert_eq!(
            Type::Tuple(Tuple::concrete(vec![
                class_type(&foo1, TArgs::default()),
                class_type(
                    &bar,
                    TArgs::new(
                        bar_tparams.dupe(),
                        vec![class_type(&foo2, TArgs::default())]
                    )
                )
            ]))
            .to_string(),
            "tuple[mod.ule.foo@1:6, bar[mod.ule.foo@1:9]]"
        );
        assert_eq!(
            Type::Tuple(Tuple::concrete(vec![
                class_type(&foo1, TArgs::default()),
                class_type(&foo3, TArgs::default())
            ]))
            .to_string(),
            "tuple[mod.ule.foo, ule.foo]"
        );
        assert_eq!(
            Type::Tuple(Tuple::concrete(vec![])).to_string(),
            "tuple[()]"
        );

        let t1 = class_type(&foo1, TArgs::default());
        let t2 = class_type(&foo2, TArgs::default());
        let ctx = TypeDisplayContext::new(&[&t1, &t2]);
        assert_eq!(
            format!("{} <: {}", ctx.display(&t1), ctx.display(&t2)),
            "mod.ule.foo@1:6 <: mod.ule.foo@1:9"
        );
    }

    #[test]
    fn test_display_qualified() {
        let c = fake_class("foo", "mod.ule", 5);
        let t = Type::ClassType(ClassType::new(c, TArgs::default()));
        let mut ctx = TypeDisplayContext::new(&[&t]);
        assert_eq!(ctx.display(&t).to_string(), "foo");
        assert_eq!(
            ctx.display(&Type::LiteralString).to_string(),
            "LiteralString"
        );
        assert_eq!(ctx.display(&Type::any_explicit()).to_string(), "Any");
        assert_eq!(ctx.display(&Type::never()).to_string(), "Never");

        ctx.always_display_module_name();
        assert_eq!(ctx.display(&t).to_string(), "mod.ule.foo");
        assert_eq!(
            ctx.display(&Type::LiteralString).to_string(),
            "typing.LiteralString"
        );
        assert_eq!(ctx.display(&Type::any_explicit()).to_string(), "typing.Any");
        assert_eq!(ctx.display(&Type::never()).to_string(), "typing.Never");
    }

    #[test]
    fn test_display_qualified_except_builtins() {
        let foo_class = fake_class("foo", "test", 5);
        let foo_type = Type::ClassType(ClassType::new(foo_class, TArgs::default()));

        {
            let mut ctx = TypeDisplayContext::new(&[&foo_type]);
            ctx.always_display_module_name_except_builtins();
            assert_eq!(ctx.display(&foo_type).to_string(), "test.foo");
        }

        let int_class = fake_class("int", "builtins", 6);
        let int_type = Type::ClassType(ClassType::new(int_class, TArgs::default()));

        {
            let mut ctx = TypeDisplayContext::new(&[&int_type]);
            ctx.always_display_module_name_except_builtins();
            assert_eq!(ctx.display(&int_type).to_string(), "int");
        }

        let union_foo_int = Type::Union(vec![foo_type, int_type]);

        {
            let mut ctx = TypeDisplayContext::new(&[&union_foo_int]);
            ctx.always_display_module_name_except_builtins();
            assert_eq!(ctx.display(&union_foo_int).to_string(), "test.foo | int");
        }
    }

    #[test]
    fn test_display_typevar() {
        let t1 = fake_tyvar("foo", "bar", 1);
        let t2 = fake_tyvar("foo", "bar", 2);
        let t3 = fake_tyvar("qux", "bar", 2);

        assert_eq!(
            Type::Union(vec![t1.to_type(), t2.to_type()]).to_string(),
            "TypeVar[bar.foo@1:2] | TypeVar[bar.foo@1:3]"
        );
        assert_eq!(
            Type::Union(vec![t1.to_type(), t3.to_type()]).to_string(),
            "TypeVar[foo] | TypeVar[qux]"
        );
    }

    #[test]
    fn test_display_literal() {
        // Simple literals
        assert_eq!(Type::Literal(Lit::Bool(true)).to_string(), "Literal[True]");
        assert_eq!(
            Type::Literal(Lit::Bool(false)).to_string(),
            "Literal[False]"
        );

        // Enum literals (not all of these types make sense, we're only providing what's relevant)
        let my_enum = ClassType::new(fake_class("MyEnum", "mod.ule", 5), TArgs::default());
        let t = Type::Literal(Lit::Enum(Box::new(LitEnum {
            class: my_enum,
            member: Name::new_static("X"),
            ty: Type::any_implicit(),
        })));

        let mut ctx = TypeDisplayContext::new(&[&t]);
        assert_eq!(ctx.display(&t).to_string(), "Literal[MyEnum.X]");

        ctx.always_display_module_name();
        assert_eq!(
            ctx.display(&t).to_string(),
            "typing.Literal[mod.ule.MyEnum.X]"
        );
    }

    #[test]
    fn test_display_union() {
        let lit1 = Type::Literal(Lit::Bool(true));
        let lit2 = Type::Literal(Lit::Str("test".into()));
        let nonlit1 = Type::None;
        let nonlit2 = Type::LiteralString;

        assert_eq!(
            Type::Union(vec![nonlit1.clone(), nonlit2.clone()]).to_string(),
            "None | LiteralString"
        );
        assert_eq!(
            Type::Union(vec![nonlit1, lit1, nonlit2, lit2]).to_string(),
            "None | Literal[True, 'test'] | LiteralString"
        );
    }

    #[test]
    fn test_display_single_param_callable() {
        let param1 = Param::Pos(Name::new_static("hello"), Type::None, Required::Required);
        let callable = Callable::list(ParamList::new(vec![param1]), Type::None);
        let callable_type = Type::Callable(Box::new(callable));
        let mut ctx = TypeDisplayContext::new(&[&callable_type]);
        assert_eq!(
            ctx.display(&callable_type).to_string(),
            "(hello: None) -> None"
        );
        ctx.set_display_mode_to_hover();
        assert_eq!(
            ctx.display(&callable_type).to_string(),
            "(hello: None) -> None"
        );
    }

    #[test]
    fn test_display_callable() {
        let param1 = Param::Pos(Name::new_static("hello"), Type::None, Required::Required);
        let param2 = Param::KwOnly(Name::new_static("world"), Type::None, Required::Required);
        let callable = Callable::list(ParamList::new(vec![param1, param2]), Type::None);
        let callable_type = Type::Callable(Box::new(callable));
        let mut ctx = TypeDisplayContext::new(&[&callable_type]);
        assert_eq!(
            ctx.display(&callable_type).to_string(),
            "(hello: None, *, world: None) -> None"
        );
        ctx.set_display_mode_to_hover();
        assert_eq!(
            ctx.display(&callable_type).to_string(),
            r#"(
    hello: None,
    *,
    world: None
) -> None"#
        );
    }

    #[test]
    fn test_display_generic_callable() {
        let uniques = UniqueFactory::new();
        let param1 = Param::Pos(Name::new_static("hello"), Type::None, Required::Required);
        let param2 = Param::KwOnly(Name::new_static("world"), Type::None, Required::Required);
        let callable = Callable::list(ParamList::new(vec![param1, param2]), Type::None);
        let generic_callable_type = Type::Forall(Box::new(Forall {
            tparams: fake_tparams(vec![fake_tparam(&uniques, "T", QuantifiedKind::TypeVar)]),
            body: Forallable::Callable(callable),
        }));
        let mut ctx = TypeDisplayContext::new(&[&generic_callable_type]);
        assert_eq!(
            ctx.display(&generic_callable_type).to_string(),
            "[T](hello: None, *, world: None) -> None"
        );
        ctx.set_display_mode_to_hover();
        assert_eq!(
            ctx.display(&generic_callable_type).to_string(),
            r#"[T](
    hello: None,
    *,
    world: None
) -> None"#
        );
    }

    #[test]
    fn test_display_args_kwargs_callable() {
        let args = Param::VarArg(Some(Name::new_static("my_args")), Type::any_implicit());
        let kwargs = Param::Kwargs(Some(Name::new_static("my_kwargs")), Type::any_implicit());
        let callable = Callable::list(ParamList::new(vec![args, kwargs]), Type::None);
        let callable_type = Type::Callable(Box::new(callable));
        let mut ctx = TypeDisplayContext::new(&[&callable_type]);
        assert_eq!(
            ctx.display(&callable_type).to_string(),
            "(*my_args: Unknown, **my_kwargs: Unknown) -> None"
        );
        ctx.set_display_mode_to_hover();
        assert_eq!(
            ctx.display(&callable_type).to_string(),
            r#"(
    *my_args: Unknown,
    **my_kwargs: Unknown
) -> None"#
        );
    }

    #[test]
    fn test_display_callable_in_container() {
        let param1 = Param::Pos(Name::new_static("hello"), Type::None, Required::Required);
        let param2 = Param::KwOnly(Name::new_static("world"), Type::None, Required::Required);
        let callable = Callable::list(ParamList::new(vec![param1, param2]), Type::None);
        let callable_type = Type::Callable(Box::new(callable));
        let tuple = Type::Tuple(Tuple::concrete(vec![callable_type.clone()]));
        let mut ctx = TypeDisplayContext::new(&[&tuple]);
        assert_eq!(
            ctx.display(&tuple).to_string(),
            "tuple[(hello: None, *, world: None) -> None]"
        );
        ctx.set_display_mode_to_hover();
        assert_eq!(
            ctx.display(&tuple).to_string(),
            "tuple[(hello: None, *, world: None) -> None]"
        );
    }

    #[test]
    fn test_display_type_alias() {
        let alias = Type::TypeAlias(Box::new(TypeAlias::new(
            Name::new_static("MyAlias"),
            Type::None,
            TypeAliasStyle::LegacyImplicit,
            Vec::new(),
        )));
        let wrapped = Type::tuple(vec![alias.clone()]);
        let type_of = Type::type_form(alias.clone());
        let mut ctx = TypeDisplayContext::new(&[]);
        // regular display
        assert_eq!(ctx.display(&alias).to_string(), "None");
        assert_eq!(ctx.display(&wrapped).to_string(), "tuple[MyAlias]");
        assert_eq!(ctx.display(&type_of).to_string(), "type[MyAlias]");
        // hover display
        ctx.set_display_mode_to_hover();
        assert_eq!(ctx.display(&alias).to_string(), "None");
        assert_eq!(ctx.display(&wrapped).to_string(), "tuple[MyAlias]");
        assert_eq!(ctx.display(&type_of).to_string(), "type[MyAlias]");
    }

    #[test]
    fn test_display_optional_parameter() {
        let param1 = Param::PosOnly(
            Some(Name::new_static("x")),
            Type::any_explicit(),
            Required::Optional(None),
        );
        let param2 = Param::Pos(
            Name::new_static("y"),
            Type::any_explicit(),
            Required::Optional(Some(Type::Literal(Lit::Bool(true)))),
        );
        let param3 = Param::Pos(
            Name::new_static("z"),
            Type::any_explicit(),
            Required::Optional(Some(Type::None)),
        );
        let callable = Callable::list(ParamList::new(vec![param1, param2, param3]), Type::None);
        let callable_type = Type::Callable(Box::new(callable));
        let mut ctx = TypeDisplayContext::new(&[&callable_type]);
        assert_eq!(
            ctx.display(&callable_type).to_string(),
            "(x: Any = ..., /, y: Any = True, z: Any = None) -> None"
        );
        ctx.set_display_mode_to_hover();
        assert_eq!(
            ctx.display(&callable_type).to_string(),
            r#"(
    x: Any = ...,
    /,
    y: Any = True,
    z: Any = None
) -> None"#
        );
    }

    #[test]
    fn test_posonly_parameter_only() {
        let param = Param::PosOnly(
            Some(Name::new_static("x")),
            Type::any_explicit(),
            Required::Required,
        );
        let callable = Callable::list(ParamList::new(vec![param]), Type::None);
        let callable_type = Type::Callable(Box::new(callable));
        let mut ctx = TypeDisplayContext::new(&[&callable_type]);
        assert_eq!(
            ctx.display(&callable_type).to_string(),
            "(x: Any, /) -> None"
        );
        ctx.set_display_mode_to_hover();
        assert_eq!(
            ctx.display(&callable_type).to_string(),
            "(x: Any, /) -> None"
        );
    }

    #[test]
    fn test_anon_posonly_parameters() {
        let param1 = Param::PosOnly(None, Type::any_explicit(), Required::Required);
        let param2 = Param::PosOnly(None, Type::any_explicit(), Required::Optional(None));
        let callable = Callable::list(ParamList::new(vec![param1, param2]), Type::None);
        let callable_type = Type::Callable(Box::new(callable));
        let mut ctx = TypeDisplayContext::new(&[&callable_type]);
        assert_eq!(
            ctx.display(&callable_type).to_string(),
            "(Any, _: Any = ...) -> None"
        );
        ctx.set_display_mode_to_hover();
        assert_eq!(
            ctx.display(&callable_type).to_string(),
            r#"(
    Any,
    _: Any = ...
) -> None"#
        );
    }

    #[test]
    fn test_optional_kwonly_parameter() {
        let param = Param::KwOnly(
            Name::new_static("x"),
            Type::any_explicit(),
            Required::Optional(None),
        );
        let callable = Callable::list(ParamList::new(vec![param]), Type::None);
        let callable_type = Type::Callable(Box::new(callable));
        let ctx = TypeDisplayContext::new(&[&callable_type]);
        assert_eq!(
            ctx.display(&callable_type).to_string(),
            "(*, x: Any = ...) -> None"
        );
    }

    #[test]
    fn test_display_generic_typeddict() {
        let uniques = UniqueFactory::new();
        let cls = fake_class("C", "test", 0);
        let tparams = fake_tparams(vec![fake_tparam(&uniques, "T", QuantifiedKind::TypeVar)]);
        let t = Type::None;
        let targs = TArgs::new(tparams.dupe(), vec![t]);
        let td = TypedDict::new(cls, targs);
        assert_eq!(Type::TypedDict(td).to_string(), "TypedDict[C[None]]");
    }

    #[test]
    fn test_display_bound_method() {
        let bound_method = fake_bound_method("foo", "MyClass", "my.module");
        let mut ctx = TypeDisplayContext::new(&[&bound_method]);
        assert_eq!(
            ctx.display(&bound_method).to_string(),
            "BoundMethod[type[MyClass], (self: Any, x: Any, y: Any) -> None]"
        );
        ctx.set_display_mode_to_hover();
        assert_eq!(
            ctx.display(&bound_method).to_string(),
            r#"def foo(
    self: Any,
    x: Any,
    y: Any
) -> None: ..."#
        );
    }

    #[test]
    fn test_display_generic_bound_method() {
        let uniques = UniqueFactory::new();
        let bound_method = fake_generic_bound_method(
            "foo",
            "MyClass",
            "my.module",
            fake_tparams(vec![fake_tparam(&uniques, "T", QuantifiedKind::TypeVar)]),
        );
        let mut ctx = TypeDisplayContext::new(&[&bound_method]);
        assert_eq!(
            ctx.display(&bound_method).to_string(),
            "BoundMethod[type[MyClass], [T](self: Any, x: Any, y: Any) -> None]"
        );
        ctx.set_display_mode_to_hover();
        assert_eq!(
            ctx.display(&bound_method).to_string(),
            r#"def foo[T](
    self: Any,
    x: Any,
    y: Any
) -> None: ..."#
        );
    }

    #[test]
    fn test_display_overload() {
        let uniques = UniqueFactory::new();
        let class = fake_class("TestClass", "test", 0);
        let sig1 = Function {
            signature: Callable::list(
                ParamList::new(vec![Param::Pos(
                    Name::new_static("x"),
                    Type::any_explicit(),
                    Required::Required,
                )]),
                Type::None,
            ),
            metadata: FuncMetadata::def(
                class.dupe().module().dupe(),
                class.dupe(),
                Name::new_static("overloaded_func"),
            ),
        };

        let sig2 = Function {
            signature: Callable::list(
                ParamList::new(vec![
                    Param::Pos(
                        Name::new_static("x"),
                        Type::any_explicit(),
                        Required::Required,
                    ),
                    Param::Pos(
                        Name::new_static("y"),
                        Type::any_explicit(),
                        Required::Required,
                    ),
                ]),
                Type::None,
            ),
            metadata: FuncMetadata::def(
                class.dupe().module().dupe(),
                class.dupe(),
                Name::new_static("overloaded_func"),
            ),
        };

        let overload = Type::Overload(Overload {
            signatures: vec1![
                OverloadType::Function(sig1.clone()),
                OverloadType::Forall(Forall {
                    tparams: fake_tparams(vec![fake_tparam(
                        &uniques,
                        "T",
                        QuantifiedKind::TypeVar
                    )]),
                    body: sig2.clone()
                })
            ],
            metadata: Box::new(sig1.metadata.clone()),
        });

        // Test compact display mode (non-hover)
        let ctx = TypeDisplayContext::new(&[&overload]);
        assert_eq!(
            ctx.display(&overload).to_string(),
            "Overload[(x: Any) -> None, [T](x: Any, y: Any) -> None]"
        );

        // Test hover display mode (with @overload decorators)
        let mut hover_ctx = TypeDisplayContext::new(&[&overload]);
        hover_ctx.set_display_mode_to_hover();
        assert_eq!(
            hover_ctx.display(&overload).to_string(),
            r#"
@overload
def overloaded_func(x: Any) -> None: ...
def overloaded_func[T](
    x: Any,
    y: Any
) -> None: ..."#
        );

        let bound_method_overload = Type::BoundMethod(Box::new(BoundMethod {
            obj: Type::any_explicit(),
            func: BoundMethodType::Overload(Overload {
                signatures: vec1![
                    OverloadType::Function(sig1.clone()),
                    OverloadType::Forall(Forall {
                        tparams: fake_tparams(vec![fake_tparam(
                            &uniques,
                            "T",
                            QuantifiedKind::TypeVar
                        )]),
                        body: sig2
                    })
                ],
                metadata: Box::new(sig1.metadata),
            }),
        }));

        // Test compact display mode (non-hover)
        let ctx = TypeDisplayContext::new(&[&bound_method_overload]);
        assert_eq!(
            ctx.display(&bound_method_overload).to_string(),
            "BoundMethod[Any, Overload[(x: Any) -> None, [T](x: Any, y: Any) -> None]]"
        );

        // Test hover display mode (with @overload decorators)
        let mut hover_ctx = TypeDisplayContext::new(&[&bound_method_overload]);
        hover_ctx.set_display_mode_to_hover();
        assert_eq!(
            hover_ctx.display(&bound_method_overload).to_string(),
            r#"
@overload
def overloaded_func(x: Any) -> None: ...
def overloaded_func[T](
    x: Any,
    y: Any
) -> None: ..."#
        );
    }
}
