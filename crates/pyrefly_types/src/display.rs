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

use pyrefly_python::module::TextRangeWithModule;
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
use crate::type_output::DisplayOutput;
use crate::type_output::OutputWithLocations;
use crate::type_output::TypeOutput;
use crate::typed_dict::TypedDict;
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
use crate::types::Union;

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

/// Display mode for type formatting for certain LSP requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LspDisplayMode {
    /// Standard display mode
    #[default]
    Standard,
    /// Hover mode: Multi-line for readability
    Hover,
    /// Signature help mode: Single-line for LSP compatibility
    SignatureHelp,
}

#[derive(Debug, Default)]
pub struct TypeDisplayContext<'a> {
    qnames: SmallMap<&'a Name, QNameInfo>,
    /// Display mode for formatting
    lsp_display_mode: LspDisplayMode,
    always_display_module_name: bool,
    always_display_expanded_unions: bool,
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

    pub fn always_display_expanded_unions(&mut self) {
        self.always_display_expanded_unions = true;
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

    /// Set the context to display in LSP.
    pub fn set_lsp_display_mode(&mut self, display_mode: LspDisplayMode) {
        self.lsp_display_mode = display_mode;
    }

    pub fn display(&'a self, t: &'a Type) -> impl Display + 'a {
        Fmt(|f| self.fmt(t, f))
    }

    // Private method for internal use
    pub fn display_internal(&'a self, t: &'a Type) -> impl Display + 'a {
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

    pub(crate) fn fmt_targs(&self, targs: &TArgs, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

    pub(crate) fn fmt_qname(&self, qname: &QName, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.qnames.get(&qname.id()) {
            Some(info) => info.fmt(qname, f),
            None => QNameInfo::qualified().fmt(qname, f), // we should not get here, if we do, be safe
        }
    }

    pub(crate) fn fmt_lit(&self, lit: &Lit, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
        output: &mut impl TypeOutput,
    ) -> fmt::Result {
        if self.always_display_module_name {
            output.write_str(&format!("{}.{}", module, name))
        } else {
            output.write_str(name)
        }
    }

    /// Helper function to format a sequence of types with a separator.
    /// Used for unions, intersections, and other type sequences.
    fn fmt_type_sequence<'b>(
        &self,
        types: impl IntoIterator<Item = &'b Type>,
        separator: &str,
        wrap_callables_and_intersect: bool,
        output: &mut impl TypeOutput,
    ) -> fmt::Result {
        for (i, t) in types.into_iter().enumerate() {
            if i > 0 {
                output.write_str(separator)?;
            }

            let needs_parens = wrap_callables_and_intersect
                && matches!(
                    t,
                    Type::Callable(_) | Type::Function(_) | Type::Intersect(_)
                );
            if needs_parens {
                output.write_str("(")?;
            }
            self.fmt_helper_generic(t, false, output)?;
            if needs_parens {
                output.write_str(")")?;
            }
        }
        Ok(())
    }

    /// Core formatting logic for types that works with any `TypeOutput` implementation.
    ///
    /// The method uses the `TypeOutput` trait abstraction to write output in various ways.
    /// This allows it to work for various purposes. (e.g., `DisplayOutput` for plain text
    /// or `OutputWithLocations` for tracking source locations).
    ///
    /// Note that the formatted type is not actually returned from this function. The type will
    /// be collected in whatever `TypeOutput` is provided.
    ///
    /// # Arguments
    ///
    /// * `t` - The type to format
    /// * `is_toplevel` - Whether this type is at the top level of the display.
    ///   - When `true` and hover mode is enabled:
    ///     - Callables, functions, and overloads are formatted with newlines for readability
    ///     - Functions show `def func_name(...)` syntax instead of compact callable syntax
    ///     - Overloads are displayed with `@overload` decorators
    ///     - Type aliases are expanded to show their definition
    ///   - When `false`, these types use compact inline formatting.
    /// * `output` - The output writer implementing `TypeOutput`. This abstraction allows
    ///   the same formatting logic to be used for different purposes (plain formatting,
    ///   location tracking, etc.)
    pub fn fmt_helper_generic(
        &self,
        t: &Type,
        is_toplevel: bool,
        output: &mut impl TypeOutput,
    ) -> fmt::Result {
        match t {
            // Things that have QName's and need qualifying
            Type::ClassDef(cls) => {
                output.write_str("type[")?;
                output.write_qname(cls.qname())?;
                output.write_str("]")
            }
            Type::ClassType(class_type)
                if class_type.qname().module_name().as_str() == "builtins"
                    && class_type.qname().id().as_str() == "tuple"
                    && class_type.targs().as_slice().len() == 1 =>
            {
                output.write_qname(class_type.qname())?;
                output.write_str("[")?;
                output.write_type(&class_type.targs().as_slice()[0])?;
                output.write_str(", ...]")
            }
            Type::ClassType(class_type) => {
                output.write_qname(class_type.qname())?;
                output.write_targs(class_type.targs())
            }
            Type::TypedDict(typed_dict) => match typed_dict {
                TypedDict::TypedDict(inner) => {
                    output.write_qname(inner.qname())?;
                    output.write_targs(inner.targs())
                }
                TypedDict::Anonymous(inner) => {
                    output.write_str("dict[str, ")?;
                    self.fmt_helper_generic(&inner.value_type, false, output)?;
                    output.write_str("]")
                }
            },
            Type::PartialTypedDict(typed_dict) => match typed_dict {
                TypedDict::TypedDict(inner) => {
                    output.write_qname(inner.qname())?;
                    output.write_targs(inner.targs())
                }
                TypedDict::Anonymous(inner) => {
                    output.write_str("dict[str, ")?;
                    self.fmt_helper_generic(&inner.value_type, false, output)?;
                    output.write_str("]")
                }
            },
            Type::TypeVar(t) => {
                output.write_str("TypeVar[")?;
                output.write_qname(t.qname())?;
                output.write_str("]")
            }
            Type::TypeVarTuple(t) => {
                output.write_str("TypeVarTuple[")?;
                output.write_qname(t.qname())?;
                output.write_str("]")
            }
            Type::ParamSpec(t) => {
                output.write_str("ParamSpec[")?;
                output.write_qname(t.qname())?;
                output.write_str("]")
            }
            Type::SelfType(cls) => {
                self.maybe_fmt_with_module("typing", "Self@", output)?;
                output.write_qname(cls.qname())
            }

            // Other things
            Type::Literal(lit) => {
                self.maybe_fmt_with_module("typing", "Literal", output)?;
                output.write_str("[")?;
                output.write_lit(lit)?;
                output.write_str("]")
            }
            Type::LiteralString => self.maybe_fmt_with_module("typing", "LiteralString", output),
            Type::Callable(box c) => {
                if self.lsp_display_mode == LspDisplayMode::Hover && is_toplevel {
                    c.fmt_with_type_with_newlines(output, &|t, o| {
                        self.fmt_helper_generic(t, false, o)
                    })
                } else {
                    c.fmt_with_type(output, &|t, o| self.fmt_helper_generic(t, false, o))
                }
            }
            Type::Function(box Function {
                signature,
                metadata,
            }) => match self.lsp_display_mode {
                LspDisplayMode::Hover | LspDisplayMode::SignatureHelp if is_toplevel => {
                    let func_name = metadata.kind.function_name();
                    output.write_str("def ")?;
                    output.write_str(func_name.as_ref().as_str())?;
                    match self.lsp_display_mode {
                        LspDisplayMode::Hover => {
                            signature.fmt_with_type_with_newlines(output, &|t, o| {
                                self.fmt_helper_generic(t, false, o)
                            })?;
                        }
                        LspDisplayMode::SignatureHelp => {
                            signature.fmt_with_type(output, &|t, o| {
                                self.fmt_helper_generic(t, false, o)
                            })?;
                        }
                        _ => unreachable!(),
                    }
                    output.write_str(": ...")
                }
                _ => signature.fmt_with_type(output, &|t, o| self.fmt_helper_generic(t, false, o)),
            },
            Type::Overload(overload) => {
                if self.lsp_display_mode == LspDisplayMode::Hover && is_toplevel {
                    output.write_str("\n@overload\n")?;
                    self.fmt_helper_generic(&overload.signatures.first().as_type(), true, output)?;
                    for sig in overload.signatures.iter().skip(1) {
                        output.write_str("\n")?;
                        self.fmt_helper_generic(&sig.as_type(), true, output)?;
                    }
                    Ok(())
                } else {
                    output.write_str("Overload[")?;
                    self.fmt_helper_generic(&overload.signatures.first().as_type(), false, output)?;
                    for sig in overload.signatures.iter().skip(1) {
                        output.write_str(", ")?;
                        self.fmt_helper_generic(&sig.as_type(), false, output)?;
                    }
                    output.write_str("]")
                }
            }
            Type::ParamSpecValue(x) => {
                output.write_str("[")?;
                x.fmt_with_type(output, &|t, o| self.fmt_helper_generic(t, false, o))?;
                output.write_str("]")
            }
            Type::BoundMethod(box BoundMethod { obj, func }) => {
                match self.lsp_display_mode {
                    LspDisplayMode::Hover | LspDisplayMode::SignatureHelp if is_toplevel => {
                        match func {
                            BoundMethodType::Function(Function {
                                signature,
                                metadata,
                            }) => {
                                let func_name = metadata.kind.function_name();
                                output.write_str("def ")?;
                                output.write_str(func_name.as_ref().as_str())?;
                                match self.lsp_display_mode {
                                    LspDisplayMode::Hover => {
                                        signature
                                            .fmt_with_type_with_newlines(output, &|t, o| {
                                                self.fmt_helper_generic(t, false, o)
                                            })?;
                                    }
                                    LspDisplayMode::SignatureHelp => {
                                        signature.fmt_with_type(output, &|t, o| {
                                            self.fmt_helper_generic(t, false, o)
                                        })?;
                                    }
                                    LspDisplayMode::Standard => unreachable!(),
                                }
                                output.write_str(": ...")
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
                                output.write_str("def ")?;
                                output.write_str(func_name.as_ref().as_str())?;
                                output.write_str("[")?;
                                output.write_str(&format!("{}", commas_iter(|| tparams.iter())))?;
                                output.write_str("]")?;
                                match self.lsp_display_mode {
                                    LspDisplayMode::Hover => {
                                        signature
                                            .fmt_with_type_with_newlines(output, &|t, o| {
                                                self.fmt_helper_generic(t, false, o)
                                            })?;
                                    }
                                    LspDisplayMode::SignatureHelp => {
                                        signature.fmt_with_type(output, &|t, o| {
                                            self.fmt_helper_generic(t, false, o)
                                        })?;
                                    }
                                    LspDisplayMode::Standard => unreachable!(),
                                }
                                output.write_str(": ...")
                            }
                            BoundMethodType::Overload(_) => {
                                // Use display instead of display_internal to show overloads w/ top-level formatting
                                self.fmt_helper_generic(&func.clone().as_type(), true, output)
                            }
                        }
                    }
                    LspDisplayMode::Hover | LspDisplayMode::SignatureHelp => {
                        self.fmt_helper_generic(&func.clone().as_type(), false, output)
                    }
                    _ => {
                        output.write_str("BoundMethod[")?;
                        self.fmt_helper_generic(obj, false, output)?;
                        output.write_str(", ")?;
                        self.fmt_helper_generic(&func.clone().as_type(), false, output)?;
                        output.write_str("]")
                    }
                }
            }
            Type::Never(NeverStyle::NoReturn) => {
                self.maybe_fmt_with_module("typing", "NoReturn", output)
            }
            Type::Never(NeverStyle::Never) => self.maybe_fmt_with_module("typing", "Never", output),
            Type::Union(box Union { members: types, .. }) if types.is_empty() => {
                self.maybe_fmt_with_module("typing", "Never", output)
            }
            Type::Union(box Union {
                display_name: Some(name),
                ..
            }) if !(self.always_display_expanded_unions || is_toplevel) => output.write_str(name),
            Type::Union(box Union { members, .. }) => {
                let mut literal_idx = None;
                let mut literals = Vec::new();
                let mut union_members: Vec<&Type> = Vec::new();
                // Track seen types to deduplicate (mainly to prettify types for functions with different names but the same signature)
                let mut seen_types = SmallSet::new();

                for t in members.iter() {
                    match t {
                        Type::Literal(lit) => {
                            if literal_idx.is_none() {
                                // First literal encountered: save this position in union_members.
                                // All Literal types in the union will be combined into a single
                                // "Literal[a, b, c]" output at this position for readability.
                                // Example: int | Literal[1] | str | Literal[2] → int | Literal[1, 2] | str
                                literal_idx = Some(union_members.len());
                                // Insert a placeholder since we don't know all literals yet.
                                // When outputting (line 505), we check `if i == idx` to detect this
                                // placeholder position and output the combined literal instead.
                                union_members.push(&Type::None);
                            }
                            literals.push(lit)
                        }
                        Type::Callable(_) | Type::Function(_) | Type::Intersect(_) => {
                            // These types need parentheses in union context
                            let mut temp = String::new();
                            {
                                use std::fmt::Write;
                                let temp_formatter = Fmt(|f| {
                                    let mut temp_output = DisplayOutput::new(self, f);
                                    self.fmt_helper_generic(t, false, &mut temp_output)
                                });
                                write!(&mut temp, "({})", temp_formatter).ok();
                            }
                            // Only add if we haven't seen this type string before
                            if seen_types.insert(temp) {
                                union_members.push(t);
                            }
                        }
                        _ => {
                            // Format the type to a string for deduplication
                            let mut temp = String::new();
                            {
                                use std::fmt::Write;
                                let temp_formatter = Fmt(|f| {
                                    let mut temp_output = DisplayOutput::new(self, f);
                                    self.fmt_helper_generic(t, false, &mut temp_output)
                                });
                                write!(&mut temp, "{}", temp_formatter).ok();
                            }
                            // Only add if we haven't seen this type string before
                            if seen_types.insert(temp) {
                                union_members.push(t);
                            }
                        }
                    }
                }

                // If we found literals, create a combined Literal type and replace the placeholder
                if let Some(idx) = literal_idx {
                    // We need to format the combined Literal manually since it's not a real Type
                    // but a special formatting construct
                    for (i, t) in union_members.iter().enumerate() {
                        if i > 0 {
                            output.write_str(" | ")?;
                        }

                        if i == idx {
                            // This is where the combined Literal goes
                            self.maybe_fmt_with_module("typing", "Literal", output)?;
                            output.write_str("[")?;
                            for (j, lit) in literals.iter().enumerate() {
                                if j > 0 {
                                    output.write_str(", ")?;
                                }
                                output.write_lit(lit)?;
                            }
                            output.write_str("]")?;
                        } else {
                            // Regular union member - use helper for just this one
                            let needs_parens = matches!(
                                t,
                                Type::Callable(_) | Type::Function(_) | Type::Intersect(_)
                            );
                            if needs_parens {
                                output.write_str("(")?;
                            }
                            self.fmt_helper_generic(t, false, output)?;
                            if needs_parens {
                                output.write_str(")")?;
                            }
                        }
                    }
                    Ok(())
                } else {
                    // No literals, just use the helper directly
                    self.fmt_type_sequence(union_members, " | ", true, output)
                }
            }
            Type::Intersect(x) => self.fmt_type_sequence(x.0.iter(), " & ", true, output),
            Type::Tuple(t) => {
                t.fmt_with_type(output, &|ty, o| self.fmt_helper_generic(ty, false, o))
            }
            Type::Forall(box Forall {
                tparams,
                body: body @ Forallable::Callable(c),
            }) => {
                if self.lsp_display_mode == LspDisplayMode::Hover && is_toplevel {
                    output.write_str("[")?;
                    output.write_str(&format!("{}", commas_iter(|| tparams.iter())))?;
                    output.write_str("]")?;
                    c.fmt_with_type_with_newlines(output, &|t, o| {
                        self.fmt_helper_generic(t, false, o)
                    })
                } else {
                    output.write_str("[")?;
                    output.write_str(&format!("{}", commas_iter(|| tparams.iter())))?;
                    output.write_str("]")?;
                    self.fmt_helper_generic(&body.clone().as_type(), false, output)
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
            }) => match self.lsp_display_mode {
                LspDisplayMode::Hover | LspDisplayMode::SignatureHelp if is_toplevel => {
                    let func_name = metadata.kind.function_name();
                    output.write_str("def ")?;
                    output.write_str(func_name.as_ref().as_str())?;
                    output.write_str("[")?;
                    output.write_str(&format!("{}", commas_iter(|| tparams.iter())))?;
                    output.write_str("]")?;
                    match self.lsp_display_mode {
                        LspDisplayMode::Hover => {
                            signature.fmt_with_type_with_newlines(output, &|t, o| {
                                self.fmt_helper_generic(t, false, o)
                            })?;
                        }
                        LspDisplayMode::SignatureHelp => {
                            signature.fmt_with_type(output, &|t, o| {
                                self.fmt_helper_generic(t, false, o)
                            })?;
                        }
                        _ => unreachable!(),
                    }
                    output.write_str(": ...")
                }
                _ => {
                    output.write_str("[")?;
                    output.write_str(&format!("{}", commas_iter(|| tparams.iter())))?;
                    output.write_str("]")?;
                    self.fmt_helper_generic(&body.clone().as_type(), false, output)
                }
            },
            Type::Forall(box Forall {
                tparams,
                body: Forallable::TypeAlias(ta),
            }) => {
                if is_toplevel {
                    ta.fmt_with_type(
                        output,
                        &|t, o| self.fmt_helper_generic(t, false, o),
                        Some(tparams),
                    )
                } else {
                    output.write_str(ta.name.as_str())
                }
            }
            Type::Type(ty) => {
                output.write_str("type[")?;
                self.fmt_helper_generic(ty, false, output)?;
                output.write_str("]")
            }
            Type::TypeGuard(ty) => {
                self.maybe_fmt_with_module("typing", "TypeGuard", output)?;
                output.write_str("[")?;
                self.fmt_helper_generic(ty, false, output)?;
                output.write_str("]")
            }
            Type::TypeIs(ty) => {
                self.maybe_fmt_with_module("typing", "TypeIs", output)?;
                output.write_str("[")?;
                self.fmt_helper_generic(ty, false, output)?;
                output.write_str("]")
            }
            Type::Unpack(box ty @ Type::TypedDict(_)) => {
                self.maybe_fmt_with_module("typing", "Unpack", output)?;
                output.write_str("[")?;
                self.fmt_helper_generic(ty, false, output)?;
                output.write_str("]")
            }
            Type::Unpack(ty) => {
                output.write_str("*")?;
                self.fmt_helper_generic(ty, false, output)
            }
            Type::Concatenate(args, pspec) => {
                self.maybe_fmt_with_module("typing", "Concatenate", output)?;
                output.write_str("[")?;
                output.write_str(&format!(
                    "{}",
                    commas_iter(|| append(args.iter().map(|x| x.0.clone()), [pspec]))
                ))?;
                output.write_str("]")
            }
            Type::Module(m) => {
                output.write_str("Module[")?;
                output.write_str(&format!("{m}"))?;
                output.write_str("]")
            }
            Type::Var(var) => output.write_str(&format!("{var}")),
            Type::Quantified(var) => output.write_str(&format!("{var}")),
            Type::QuantifiedValue(var) => output.write_str(&format!("{var}")),
            Type::ElementOfTypeVarTuple(var) => output.write_str(&format!("ElementOf[{var}]")),
            Type::Args(q) => {
                output.write_str("Args[")?;
                output.write_str(&format!("{q}"))?;
                output.write_str("]")
            }
            Type::Kwargs(q) => {
                output.write_str("Kwargs[")?;
                output.write_str(&format!("{q}"))?;
                output.write_str("]")
            }
            Type::ArgsValue(q) => {
                output.write_str("ArgsValue[")?;
                output.write_str(&format!("{q}"))?;
                output.write_str("]")
            }
            Type::KwargsValue(q) => {
                output.write_str("KwargsValue[")?;
                output.write_str(&format!("{q}"))?;
                output.write_str("]")
            }
            Type::SpecialForm(x) => output.write_str(&format!("{x}")),
            Type::Ellipsis => output.write_str("Ellipsis"),
            Type::Any(style) => match style {
                AnyStyle::Explicit => self.maybe_fmt_with_module("typing", "Any", output),
                AnyStyle::Implicit | AnyStyle::Error => output.write_str("Unknown"),
            },
            Type::TypeAlias(ta) => {
                if is_toplevel {
                    ta.fmt_with_type(output, &|t, o| self.fmt_helper_generic(t, false, o), None)
                } else {
                    output.write_str(ta.name.as_str())
                }
            }
            Type::SuperInstance(box (cls, obj)) => {
                output.write_str("super[")?;
                output.write_qname(cls.qname())?;
                output.write_str(", ")?;
                match obj {
                    SuperObj::Instance(obj) => {
                        output.write_qname(obj.qname())?;
                        output.write_targs(obj.targs())?;
                    }
                    SuperObj::Class(cls) => {
                        output.write_qname(cls.qname())?;
                    }
                }
                output.write_str("]")
            }
            Type::KwCall(call) => self.fmt_helper_generic(&call.return_ty, false, output),
            Type::Materialization => output.write_str("Materialization"),
            Type::None => output.write_str("None"),
        }
    }

    /// Formats a type to a standard `fmt::Formatter` for display purposes.
    ///
    /// This is a convenience wrapper around [`fmt_helper_generic`](Self::fmt_helper_generic)
    /// that uses `DisplayOutput` to write plain text output. Use this when you need to
    /// implement the `Display` trait or format types to strings.
    ///
    /// See `fmt_helper_generic` for detailed formatting behavior.
    fn fmt_helper<'b>(
        &self,
        t: &'b Type,
        f: &mut fmt::Formatter<'_>,
        is_toplevel: bool,
    ) -> fmt::Result {
        let output = &mut DisplayOutput::new(self, f);
        self.fmt_helper_generic(t, is_toplevel, output)
    }

    /// This method wraps `fmt_helper_generic` with `OutputWithLocations` to track
    /// the source location (module and text range) of each type component in the output
    /// This is useful for IDE features like goto-type-definition
    /// where you need to map displayed type names back to their source locations.
    ///
    /// # Returns
    ///
    /// Unlike fmt_helper and fmt_helper_generic this function will not return a Result.
    /// Instead it will return an `OutputWithLocations` containing both the formatted string and location
    /// information for each part that has a source location.
    pub fn get_types_with_location<'b>(
        &self,
        t: &'b Type,
        is_toplevel: bool,
    ) -> OutputWithLocations<'_> {
        let mut output = OutputWithLocations::new(self);
        self.fmt_helper_generic(t, is_toplevel, &mut output)
            .unwrap();
        output
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TypeDisplayContext::new(&[self]).fmt(self, f)
    }
}

impl Type {
    pub fn as_lsp_string(&self, mode: LspDisplayMode) -> String {
        self.as_lsp_string_with_fallback_name(None, mode)
    }

    pub fn as_lsp_string_with_fallback_name(
        &self,
        fallback_name: Option<&str>,
        mode: LspDisplayMode,
    ) -> String {
        let mut c = TypeDisplayContext::new(&[self]);
        c.set_lsp_display_mode(mode);
        let rendered = c.display(self).to_string();
        if let Some(name) = fallback_name
            && self.is_toplevel_callable()
        {
            let trimmed = rendered.trim_start();
            if trimmed.starts_with('(') {
                return format!("def {}{}: ...", name, trimmed);
            }
        }
        rendered
    }

    pub fn get_types_with_locations(&self) -> Vec<(String, Option<TextRangeWithModule>)> {
        let ctx = TypeDisplayContext::new(&[self]);
        let mut output = OutputWithLocations::new(&ctx);
        ctx.fmt_helper_generic(self, false, &mut output).unwrap();
        output.parts().to_vec()
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

    use TypedDict;
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
    use crate::callable::Params;
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
            variance: PreInferenceVariance::Invariant,
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
            PreInferenceVariance::Invariant,
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
                    vec![Type::concrete_tuple(vec![
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
                TArgs::new(
                    tuple_param_tparams.dupe(),
                    vec![Type::concrete_tuple(Vec::new())]
                )
            )
            .to_string(),
            "TupleParam[*tuple[()]]"
        );
        assert_eq!(
            class_type(
                &tuple_param,
                TArgs::new(
                    tuple_param_tparams.dupe(),
                    vec![Type::unbounded_tuple(class_type(&foo1, TArgs::default()))]
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
                        Type::unbounded_tuple(class_type(&foo1, TArgs::default())),
                        vec![class_type(&foo1, TArgs::default())],
                    ))))]
                )
            )
            .to_string(),
            "TupleParam[foo, *tuple[foo, ...], foo]"
        );

        assert_eq!(
            Type::unbounded_tuple(class_type(&foo1, TArgs::default())).to_string(),
            "tuple[foo, ...]"
        );
        assert_eq!(
            Type::concrete_tuple(vec![
                class_type(&foo1, TArgs::default()),
                class_type(
                    &bar,
                    TArgs::new(
                        bar_tparams.dupe(),
                        vec![class_type(&foo1, TArgs::default())]
                    )
                )
            ])
            .to_string(),
            "tuple[foo, bar[foo]]"
        );
        assert_eq!(
            Type::concrete_tuple(vec![
                class_type(&foo1, TArgs::default()),
                class_type(
                    &bar,
                    TArgs::new(
                        bar_tparams.dupe(),
                        vec![class_type(&foo2, TArgs::default())]
                    )
                )
            ])
            .to_string(),
            "tuple[mod.ule.foo@1:6, bar[mod.ule.foo@1:9]]"
        );
        assert_eq!(
            Type::concrete_tuple(vec![
                class_type(&foo1, TArgs::default()),
                class_type(&foo3, TArgs::default())
            ])
            .to_string(),
            "tuple[mod.ule.foo, ule.foo]"
        );
        assert_eq!(Type::concrete_tuple(vec![]).to_string(), "tuple[()]");

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

        let union_foo_int = Type::union(vec![foo_type, int_type]);

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
            Type::union(vec![t1.to_type(), t2.to_type()]).to_string(),
            "TypeVar[bar.foo@1:2] | TypeVar[bar.foo@1:3]"
        );
        assert_eq!(
            Type::union(vec![t1.to_type(), t3.to_type()]).to_string(),
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
            Type::union(vec![nonlit1.clone(), nonlit2.clone()]).to_string(),
            "None | LiteralString"
        );
        assert_eq!(
            Type::union(vec![nonlit1.clone(), lit1, nonlit2.clone(), lit2]).to_string(),
            "None | Literal[True, 'test'] | LiteralString"
        );
        assert_eq!(
            Type::type_form(Type::Union(Box::new(Union {
                members: vec![nonlit1, nonlit2],
                display_name: Some("MyUnion".to_owned())
            })))
            .to_string(),
            "type[MyUnion]"
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
        ctx.set_lsp_display_mode(LspDisplayMode::Hover);
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
        ctx.set_lsp_display_mode(LspDisplayMode::Hover);
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
        ctx.set_lsp_display_mode(LspDisplayMode::Hover);
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
        ctx.set_lsp_display_mode(LspDisplayMode::Hover);
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
        let tuple = Type::concrete_tuple(vec![callable_type.clone()]);
        let mut ctx = TypeDisplayContext::new(&[&tuple]);
        assert_eq!(
            ctx.display(&tuple).to_string(),
            "tuple[(hello: None, *, world: None) -> None]"
        );
        ctx.set_lsp_display_mode(LspDisplayMode::Hover);
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
        let wrapped = Type::concrete_tuple(vec![alias.clone()]);
        let type_of = Type::type_form(alias.clone());
        let mut ctx = TypeDisplayContext::new(&[]);
        // regular display
        assert_eq!(ctx.display(&alias).to_string(), "None");
        assert_eq!(ctx.display(&wrapped).to_string(), "tuple[MyAlias]");
        assert_eq!(ctx.display(&type_of).to_string(), "type[MyAlias]");
        // hover display
        ctx.set_lsp_display_mode(LspDisplayMode::Hover);
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
        ctx.set_lsp_display_mode(LspDisplayMode::Hover);
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
        ctx.set_lsp_display_mode(LspDisplayMode::Hover);
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
        ctx.set_lsp_display_mode(LspDisplayMode::Hover);
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
        assert_eq!(Type::TypedDict(td).to_string(), "C[None]");
    }

    #[test]
    fn test_display_bound_method() {
        let bound_method = fake_bound_method("foo", "MyClass", "my.module");
        let mut ctx = TypeDisplayContext::new(&[&bound_method]);
        assert_eq!(
            ctx.display(&bound_method).to_string(),
            "BoundMethod[type[MyClass], (self: Any, x: Any, y: Any) -> None]"
        );
        ctx.set_lsp_display_mode(LspDisplayMode::Hover);
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
        ctx.set_lsp_display_mode(LspDisplayMode::Hover);
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
        hover_ctx.set_lsp_display_mode(LspDisplayMode::Hover);
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
        hover_ctx.set_lsp_display_mode(LspDisplayMode::Hover);
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

    #[test]
    fn test_intersection() {
        let x = Type::Intersect(Box::new((
            vec![Type::LiteralString, Type::None],
            Type::any_implicit(),
        )));
        let ctx = TypeDisplayContext::new(&[&x]);
        assert_eq!(ctx.display(&x).to_string(), "LiteralString & None");
    }

    #[test]
    fn test_union_of_intersection() {
        let x = Type::union(vec![
            Type::Intersect(Box::new((
                vec![Type::any_explicit(), Type::LiteralString],
                Type::any_implicit(),
            ))),
            Type::None,
        ]);
        let ctx = TypeDisplayContext::new(&[&x]);
        assert_eq!(ctx.display(&x).to_string(), "(Any & LiteralString) | None");
    }

    #[test]
    fn test_callable_in_intersection() {
        let x = Type::Intersect(Box::new((
            vec![
                Type::Callable(Box::new(Callable {
                    params: Params::Ellipsis,
                    ret: Type::None,
                })),
                Type::any_explicit(),
            ],
            Type::any_implicit(),
        )));
        let ctx = TypeDisplayContext::new(&[&x]);
        assert_eq!(ctx.display(&x).to_string(), "((...) -> None) & Any");
    }

    // Helper functions for testing get_types_with_location
    fn get_parts(t: &Type) -> Vec<(String, Option<TextRangeWithModule>)> {
        let ctx = TypeDisplayContext::new(&[t]);
        let output = ctx.get_types_with_location(t, false);
        output.parts().to_vec()
    }

    fn parts_to_string(parts: &[(String, Option<TextRangeWithModule>)]) -> String {
        parts.iter().map(|(s, _)| s.as_str()).collect::<String>()
    }

    fn assert_part_has_location(
        parts: &[(String, Option<TextRangeWithModule>)],
        name: &str,
        module: &str,
        position: u32,
    ) {
        let part = parts.iter().find(|(s, _)| s == name);
        assert!(part.is_some(), "Should have {} in parts", name);
        let (_, location) = part.unwrap();
        assert!(location.is_some(), "{} should have location", name);
        let loc = location.as_ref().unwrap();
        assert_eq!(loc.module.name().as_str(), module);
        assert_eq!(loc.range.start().to_u32(), position);
    }

    fn assert_output_contains(parts: &[(String, Option<TextRangeWithModule>)], needle: &str) {
        let full_str = parts_to_string(parts);
        assert!(
            full_str.contains(needle),
            "Output should contain '{}'",
            needle
        );
    }

    #[test]
    fn test_get_types_with_location_simple_class() {
        let foo = fake_class("Foo", "test.module", 10);
        let t = Type::ClassType(ClassType::new(foo, TArgs::default()));
        let parts = get_parts(&t);

        assert_part_has_location(&parts, "Foo", "test.module", 10);
    }

    #[test]
    fn test_get_types_with_location_class_with_targs() {
        let uniques = UniqueFactory::new();
        let foo = fake_class("Foo", "test.module", 10);
        let bar = fake_class("Bar", "test.module", 20);
        let tparams = fake_tparams(vec![fake_tparam(&uniques, "T", QuantifiedKind::TypeVar)]);

        let inner_type = Type::ClassType(ClassType::new(bar, TArgs::default()));
        let t = Type::ClassType(ClassType::new(foo, TArgs::new(tparams, vec![inner_type])));
        let parts = get_parts(&t);

        assert_eq!(parts[0].0, "Foo");
        assert_part_has_location(&parts, "Foo", "test.module", 10);
        assert!(parts.iter().any(|(s, _)| s == "Bar"), "Should have Bar");
    }

    #[test]
    fn test_get_types_with_location_typevar() {
        let tvar = fake_tyvar("T", "test.module", 15);
        let t = tvar.to_type();
        let parts = get_parts(&t);

        assert_eq!(parts[0].0, "TypeVar[");
        assert!(parts[0].1.is_none(), "TypeVar[ should not have location");
        assert_part_has_location(&parts, "T", "test.module", 15);
    }

    #[test]
    fn test_get_types_with_location_disambiguated() {
        let foo1 = fake_class("Foo", "mod.ule", 5);
        let foo2 = fake_class("Foo", "mod.ule", 8);
        let t1 = Type::ClassType(ClassType::new(foo1, TArgs::default()));
        let t2 = Type::ClassType(ClassType::new(foo2, TArgs::default()));
        let union = Type::union(vec![t1.clone(), t2.clone()]);
        let ctx = TypeDisplayContext::new(&[&union]);

        let parts1 = ctx.get_types_with_location(&t1, false).parts().to_vec();
        let parts2 = ctx.get_types_with_location(&t2, false).parts().to_vec();

        let loc1 = parts1.iter().find_map(|(_, loc)| loc.as_ref()).unwrap();
        let loc2 = parts2.iter().find_map(|(_, loc)| loc.as_ref()).unwrap();
        assert_ne!(
            loc1.range.start().to_u32(),
            loc2.range.start().to_u32(),
            "Different Foos should have different locations"
        );
    }

    #[test]
    fn test_get_types_with_location_literal() {
        let t = Type::Literal(Lit::Bool(true));
        let parts = get_parts(&t);

        assert_output_contains(&parts, "Literal");
        assert_output_contains(&parts, "True");
    }

    #[test]
    fn test_get_types_with_location_nested_types() {
        let uniques = UniqueFactory::new();
        let outer = fake_class("Outer", "test", 10);
        let inner = fake_class("Inner", "test", 20);
        let tparams = fake_tparams(vec![fake_tparam(&uniques, "T", QuantifiedKind::TypeVar)]);

        let inner_type = Type::ClassType(ClassType::new(inner, TArgs::default()));
        let outer_type =
            Type::ClassType(ClassType::new(outer, TArgs::new(tparams, vec![inner_type])));
        let parts = get_parts(&outer_type);

        assert_part_has_location(&parts, "Outer", "test", 10);
        assert_output_contains(&parts, "Inner");
    }

    #[test]
    fn test_get_types_with_location_type_without_location() {
        let t = Type::None;
        let parts = get_parts(&t);

        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].0, "None");
        assert!(parts[0].1.is_none(), "None should not have location");
    }

    #[test]
    fn test_get_types_with_location_tparams() {
        let uniques = UniqueFactory::new();
        let t_param = fake_tparam(&uniques, "T", QuantifiedKind::TypeVar);
        let u_param = fake_tparam(&uniques, "U", QuantifiedKind::TypeVar);
        let ts_param = fake_tparam(&uniques, "Ts", QuantifiedKind::TypeVarTuple);
        let tparams = fake_tparams(vec![t_param, u_param, ts_param]);

        let param1 = Param::Pos(
            Name::new_static("x"),
            Type::any_explicit(),
            Required::Required,
        );
        let callable = Callable::list(ParamList::new(vec![param1]), Type::None);
        let generic_callable = Type::Forall(Box::new(Forall {
            tparams,
            body: Forallable::Callable(callable),
        }));
        let parts = get_parts(&generic_callable);

        for param in &["T", "U", "Ts"] {
            assert_output_contains(&parts, param);
        }
        assert!(parts.iter().any(|(s, loc)| s == "[" && loc.is_none()));
        assert!(parts_to_string(&parts).starts_with('['));
        assert_output_contains(&parts, "](");
    }

    #[test]
    fn test_get_types_with_location_typed_dict() {
        let cls = fake_class("MyTypedDict", "mymodule", 25);
        let td = TypedDict::new(cls, TArgs::default());
        let t = Type::TypedDict(td);
        let parts = get_parts(&t);

        assert_part_has_location(&parts, "MyTypedDict", "mymodule", 25);
    }

    #[test]
    fn test_get_types_with_location_enum_literal() {
        let enum_class = fake_class("Color", "colors", 30);
        let class_type = ClassType::new(enum_class, TArgs::default());
        let enum_lit = Lit::Enum(Box::new(LitEnum {
            class: class_type,
            member: Name::new_static("RED"),
            ty: Type::any_implicit(),
        }));
        let t = Type::Literal(enum_lit);
        let parts = get_parts(&t);

        for expected in &["Literal", "Color", "RED"] {
            assert_output_contains(&parts, expected);
        }
        assert!(parts.iter().any(|(_, loc)| loc.is_some()));
    }

    #[test]
    fn test_get_types_with_location_self_type() {
        let cls = fake_class("MyClass", "mymodule", 40);
        let cls_type = ClassType::new(cls, TArgs::default());
        let t = Type::SelfType(cls_type);
        let parts = get_parts(&t);

        assert_output_contains(&parts, "Self");
        assert_part_has_location(&parts, "MyClass", "mymodule", 40);
    }

    #[test]
    fn test_get_types_with_location_class_def() {
        let cls = fake_class("MyClass", "mymodule", 45);
        let t = Type::ClassDef(cls);
        let parts = get_parts(&t);

        assert_output_contains(&parts, "type");
        assert_part_has_location(&parts, "MyClass", "mymodule", 45);
    }

    #[test]
    fn test_get_types_with_location_tuple() {
        let foo = fake_class("Foo", "test", 50);
        let bar = fake_class("Bar", "test", 55);
        let foo_type = Type::ClassType(ClassType::new(foo, TArgs::default()));
        let bar_type = Type::ClassType(ClassType::new(bar, TArgs::default()));

        // Test concrete tuple: tuple[Foo, Bar]
        let concrete_tuple = Type::Tuple(Tuple::Concrete(vec![foo_type.clone(), bar_type.clone()]));
        let parts = get_parts(&concrete_tuple);
        for expected in &["tuple", "Foo", "Bar"] {
            assert_output_contains(&parts, expected);
        }

        // Test unbounded tuple: tuple[Foo, ...]
        let unbounded_tuple = Type::Tuple(Tuple::Unbounded(Box::new(foo_type)));
        let parts2 = get_parts(&unbounded_tuple);
        for expected in &["tuple", "Foo", "..."] {
            assert_output_contains(&parts2, expected);
        }
    }

    #[test]
    fn test_get_types_with_location_callable() {
        let foo = fake_class("Foo", "test", 60);
        let bar = fake_class("Bar", "test", 65);
        let foo_type = Type::ClassType(ClassType::new(foo, TArgs::default()));
        let bar_type = Type::ClassType(ClassType::new(bar, TArgs::default()));

        let param = Param::Pos(Name::new_static("foo"), foo_type, Required::Required);
        let callable = Callable::list(ParamList::new(vec![param]), bar_type);
        let t = Type::Callable(Box::new(callable));
        let parts = get_parts(&t);

        for expected in &["foo", "Foo", "Bar", "->"] {
            assert_output_contains(&parts, expected);
        }
    }

    #[test]
    fn test_get_types_with_location_type_var_tuple() {
        let mi = Module::new(
            ModuleName::from_str("test.module"),
            ModulePath::filesystem(PathBuf::from("test.module")),
            Arc::new("1234567890".to_owned()),
        );
        let tv_tuple = crate::type_var_tuple::TypeVarTuple::new(
            Identifier::new(Name::new("Ts"), TextRange::empty(TextSize::new(70))),
            mi,
            None,
        );
        let t = Type::TypeVarTuple(tv_tuple);
        let parts = get_parts(&t);

        assert_output_contains(&parts, "TypeVarTuple");
        assert_part_has_location(&parts, "Ts", "test.module", 70);
    }

    #[test]
    fn test_get_types_with_location_param_spec() {
        let mi = Module::new(
            ModuleName::from_str("test.module"),
            ModulePath::filesystem(PathBuf::from("test.module")),
            Arc::new("1234567890".to_owned()),
        );
        let param_spec = crate::param_spec::ParamSpec::new(
            Identifier::new(Name::new("P"), TextRange::empty(TextSize::new(75))),
            mi,
            None,
        );
        let t = Type::ParamSpec(param_spec);
        let parts = get_parts(&t);

        assert_output_contains(&parts, "ParamSpec");
        assert_part_has_location(&parts, "P", "test.module", 75);
    }

    #[test]
    fn test_get_types_with_location_super_instance() {
        let base_class = fake_class("Base", "test", 80);
        let base_type = ClassType::new(base_class, TArgs::default());
        let derived_class = fake_class("Derived", "test", 85);
        let derived_type = ClassType::new(derived_class, TArgs::default());

        let t = Type::SuperInstance(Box::new((
            base_type,
            crate::types::SuperObj::Instance(derived_type),
        )));
        let parts = get_parts(&t);

        for expected in &["super", "Base", "Derived"] {
            assert_output_contains(&parts, expected);
        }
        assert_part_has_location(&parts, "Base", "test", 80);
        assert_part_has_location(&parts, "Derived", "test", 85);
    }

    #[test]
    fn test_get_types_with_location_partial_typed_dict() {
        let cls = fake_class("MyTypedDict", "mymodule", 90);
        let td = TypedDict::new(cls, TArgs::default());
        let t = Type::PartialTypedDict(td);
        let parts = get_parts(&t);

        assert_part_has_location(&parts, "MyTypedDict", "mymodule", 90);
    }
}
