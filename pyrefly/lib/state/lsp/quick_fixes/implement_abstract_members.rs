/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::docstring::dedent_block_preserving_layout;
use pyrefly_python::module::Module;
use pyrefly_python::module_name::ModuleName;
use pyrefly_types::callable::Callable;
use pyrefly_types::callable::DefaultValue;
use pyrefly_types::callable::Param;
use pyrefly_types::callable::ParamList;
use pyrefly_types::callable::Params;
use pyrefly_types::callable::Required;
use pyrefly_types::literal::Lit;
use pyrefly_types::stdlib::Stdlib;
use pyrefly_types::type_output::AnnotationPart;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::types::TParams;
use pyrefly_types::types::Type;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::helpers::is_docstring_stmt;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use super::extract_shared::line_indent_and_start;

const DEFAULT_INDENT: &str = "    ";

#[derive(Clone)]
struct AbstractMemberInfo {
    name: Name,
    ty: Type,
    defining_module: Module,
    docstring_range: Option<TextRange>,
    is_class_var: bool,
}

fn find_class_context<'a>(ast: &'a ModModule, selection: TextSize) -> Option<&'a StmtClassDef> {
    for stmt in &ast.body {
        if let Stmt::ClassDef(class_def) = stmt
            && class_def.range().contains(selection)
        {
            if let Some(inner) = find_class_in_body(&class_def.body, selection) {
                return Some(inner);
            }
            return Some(class_def);
        }
    }
    None
}

fn find_class_in_body<'a>(body: &'a [Stmt], selection: TextSize) -> Option<&'a StmtClassDef> {
    for stmt in body {
        if let Stmt::ClassDef(class_def) = stmt
            && class_def.range().contains(selection)
        {
            if let Some(inner) = find_class_in_body(&class_def.body, selection) {
                return Some(inner);
            }
            return Some(class_def);
        }
    }
    None
}

fn insertion_point(class_def: &StmtClassDef, source: &str) -> Option<(String, TextRange, String)> {
    let first = class_def.body.first()?;
    let (class_indent, _) = line_indent_and_start(source, class_def.range().start())?;
    let (_, body_line_start) = line_indent_and_start(source, first.start())?;
    if !source[body_line_start.to_usize()..first.start().to_usize()]
        .trim()
        .is_empty()
    {
        let start = source[..first.start().to_usize()].rfind(':')? + 1;
        let end = line_end_position(source, class_def.end());
        let indent = format!("{class_indent}{DEFAULT_INDENT}");
        let mut prefix = "\n".to_owned();
        if replaceable_pass_stmt(class_def).is_none() || class_def.body.len() > 1 {
            prefix.push_str(&indent);
            prefix.push_str(
                &source[first.start().to_usize()..end.to_usize()]
                    .trim_end_matches(['\r', '\n'])
                    .replace("\r\n", "\n"),
            );
            prefix.push('\n');
        }
        return Some((
            indent,
            TextRange::new(TextSize::try_from(start).ok()?, end),
            prefix,
        ));
    }
    if let Some(pass_stmt) = replaceable_pass_stmt(class_def) {
        let (indent, line_start) = line_indent_and_start(source, pass_stmt.range().start())?;
        if source[line_start.to_usize()..pass_stmt.start().to_usize()]
            .trim()
            .is_empty()
        {
            let line_end = line_end_position(source, pass_stmt.range().end());
            return Some((indent, TextRange::new(line_start, line_end), String::new()));
        }
    }
    for stmt in &class_def.body {
        if is_docstring_stmt(stmt) {
            continue;
        }
        let anchor = match stmt {
            Stmt::FunctionDef(function) => function
                .decorator_list
                .first()
                .map_or(stmt.start(), |d| d.start()),
            Stmt::ClassDef(class) => class
                .decorator_list
                .first()
                .map_or(stmt.start(), |d| d.start()),
            _ => stmt.start(),
        };
        let (indent, line_start) = line_indent_and_start(source, anchor)?;
        if !source[line_start.to_usize()..anchor.to_usize()]
            .trim()
            .is_empty()
        {
            continue;
        }
        return Some((indent, TextRange::empty(line_start), String::new()));
    }
    let (indent, _) = line_indent_and_start(source, first.start())?;
    let line_end = line_end_position(source, class_def.end());
    let prefix = if source[..line_end.to_usize()].ends_with('\n') {
        ""
    } else {
        "\n"
    };
    Some((indent, TextRange::empty(line_end), prefix.to_owned()))
}

fn replaceable_pass_stmt(class_def: &StmtClassDef) -> Option<&Stmt> {
    let mut non_docstring = class_def
        .body
        .iter()
        .filter(|stmt| !is_docstring_stmt(stmt));
    let only_stmt = non_docstring.next()?;
    if non_docstring.next().is_some() {
        return None;
    }
    match only_stmt {
        Stmt::Pass(_) => Some(only_stmt),
        _ => None,
    }
}

fn class_indent_unit(source: &str, class_def: &StmtClassDef, method_indent: &str) -> String {
    let (class_indent, _) = match line_indent_and_start(source, class_def.range().start()) {
        Some(value) => value,
        None => return DEFAULT_INDENT.to_owned(),
    };
    if let Some(unit) = method_indent.strip_prefix(&class_indent)
        && !unit.is_empty()
    {
        return unit.to_owned();
    }
    DEFAULT_INDENT.to_owned()
}

fn line_end_position(source: &str, position: TextSize) -> TextSize {
    let mut idx = position.to_usize().min(source.len());
    let bytes = source.as_bytes();
    while idx < bytes.len() && bytes[idx] != b'\n' {
        idx += 1;
    }
    if idx < bytes.len() {
        idx += 1;
    }
    TextSize::try_from(idx).expect("The source length fits in TextSize.")
}

/// Builds source text while retaining the module references used by annotations.
struct MemberRenderer<'a> {
    stdlib: &'a Stdlib,
    parts: Vec<AnnotationPart>,
}

impl MemberRenderer<'_> {
    fn text(&mut self, text: &str) {
        if let Some(AnnotationPart::Text(previous)) = self.parts.last_mut() {
            previous.push_str(text);
        } else {
            self.parts.push(AnnotationPart::Text(text.to_owned()));
        }
    }

    fn reference(&mut self, module: &str, name: &str) {
        self.parts.push(AnnotationPart::Reference {
            module: ModuleName::from_str(module),
            name: name.to_owned(),
        });
    }

    fn annotation(&mut self, ty: &Type) {
        self.parts
            .extend(ty.get_annotation_parts(Some(self.stdlib)));
    }

    fn member(
        &mut self,
        member: &AbstractMemberInfo,
        indent: &str,
        indent_unit: &str,
    ) -> Option<()> {
        let Some(metadata) = member
            .ty
            .toplevel_func_metadata()
            .filter(|_| !member.is_class_var)
        else {
            self.text(indent);
            self.text(member.name.as_str());
            self.text(": ");
            if member.is_class_var {
                self.reference("typing", "ClassVar");
                self.text("[");
            }
            self.annotation(&member.ty);
            if member.is_class_var {
                self.text("]");
            }
            return Some(());
        };
        let property = member.ty.property_metadata();
        let getter = property.map(|p| &p.getter);
        let setter = property.and_then(|p| p.setter.as_ref());
        let is_cached = metadata.flags.is_cached_property;
        let is_static = metadata.flags.is_staticmethod;
        let is_class = metadata.flags.is_classmethod;
        let is_async = metadata.flags.is_async;
        let overloaded = member.ty.toplevel_callable_signatures().count() > 1;
        for (index, (callable, tparams)) in getter
            .unwrap_or(&member.ty)
            .toplevel_callable_signatures()
            .enumerate()
        {
            if index > 0 {
                self.text("\n\n");
            }
            if overloaded {
                self.text(indent);
                self.text("@");
                self.reference("typing", "overload");
                self.text("\n");
            }
            if getter.is_some() || is_cached {
                self.text(indent);
                self.text("@");
                if is_cached {
                    self.reference("functools", "cached_property");
                } else {
                    self.text("property");
                }
                self.text("\n");
            } else if is_class || is_static {
                self.text(indent);
                self.text(if is_class {
                    "@classmethod\n"
                } else {
                    "@staticmethod\n"
                });
            }
            self.text(indent);
            self.text(if is_async { "async def " } else { "def " });
            self.text(member.name.as_str());
            self.signature(callable, tparams.map(|p| p.as_ref()), !is_static)?;
            self.text(":\n");
            if !overloaded {
                self.body(member, indent, indent_unit);
            } else {
                self.text(&format!("{indent}{indent_unit}..."));
            }
        }
        if overloaded {
            // The overloads define the contract. The implementation accepts every overload.
            self.text("\n");
            if is_class || is_static {
                self.text(&format!(
                    "\n\n{indent}@{}",
                    if is_class {
                        "classmethod"
                    } else {
                        "staticmethod"
                    }
                ));
            }
            self.text(&format!(
                "\n{indent}{}def {}(",
                if is_async { "async " } else { "" },
                member.name
            ));
            if !is_static {
                self.text(if is_class { "cls, " } else { "self, " });
            }
            self.text("*args, **kwargs) -> ");
            self.reference("typing", "Any");
            self.text(":\n");
            self.body(member, indent, indent_unit);
        }
        if let Some(setter) = setter {
            let (callable, tparams) = setter.toplevel_callable_signatures().next()?;
            self.text(&format!(
                "\n\n{indent}@{}.setter\n{indent}def {}",
                member.name, member.name
            ));
            self.signature(callable, tparams.map(|p| p.as_ref()), true)?;
            self.text(":\n");
            self.body(member, indent, indent_unit);
        }
        Some(())
    }

    fn body(&mut self, member: &AbstractMemberInfo, indent: &str, indent_unit: &str) {
        let body_indent = format!("{indent}{indent_unit}");
        if let Some(range) = member.docstring_range
            && let Some(docstring) =
                dedent_block_preserving_layout(member.defining_module.code_at(range))
        {
            for line in docstring.trim_end().lines() {
                self.text(&body_indent);
                self.text(line);
                self.text("\n");
            }
        }
        self.text(&body_indent);
        self.text("raise NotImplementedError");
    }

    fn signature(
        &mut self,
        callable: &Callable,
        tparams: Option<&TParams>,
        strip_receiver: bool,
    ) -> Option<()> {
        if let Some(tparams) = tparams.filter(|p| !p.is_empty()) {
            self.text("[");
            for (i, param) in tparams.iter().enumerate() {
                if i > 0 {
                    self.text(", ");
                }
                self.text(&param.display_name_with_prefix().to_string());
                match param.restriction() {
                    Restriction::Bound(ty) => {
                        self.text(": ");
                        self.annotation(ty);
                    }
                    Restriction::Constraints(types) => {
                        self.text(": (");
                        for (i, ty) in types.iter().enumerate() {
                            if i > 0 {
                                self.text(", ");
                            }
                            self.annotation(ty);
                        }
                        self.text(")");
                    }
                    Restriction::Unrestricted => {}
                    Restriction::ShapeExtension(_) => return None,
                }
                if let Some(default) = param.default() {
                    self.text(" = ");
                    self.annotation(default);
                }
            }
            self.text("]");
        }
        self.text("(");
        let Params::List(params) = &callable.params else {
            return None;
        };
        self.params(params, strip_receiver);
        self.text(") -> ");
        self.annotation(&callable.ret);
        Some(())
    }

    fn params(&mut self, params: &ParamList, strip_receiver: bool) {
        let mut posonly = false;
        let mut kwonly = false;
        for (i, param) in params.items().iter().enumerate() {
            if i > 0 {
                self.text(", ");
            }
            if matches!(param, Param::PosOnly(..)) {
                posonly = true;
            } else if posonly {
                self.text("/, ");
                posonly = false;
            }
            if matches!(param, Param::Varargs(..)) {
                kwonly = true;
                self.text("*");
            } else if matches!(param, Param::Kwargs(..)) {
                self.text("**");
            } else if matches!(param, Param::KwOnly(..)) && !kwonly {
                kwonly = true;
                self.text("*, ");
            }
            if let Some(name) = param.name() {
                self.text(name.as_str());
            } else {
                self.text(&format!("arg{i}"));
            }
            if !(strip_receiver && i == 0 && matches!(param, Param::PosOnly(..) | Param::Pos(..))) {
                self.text(": ");
                self.annotation(param.as_type());
            }
            if let Param::PosOnly(_, _, Required::Optional(default))
            | Param::Pos(_, _, Required::Optional(default))
            | Param::KwOnly(_, _, Required::Optional(default)) = param
            {
                self.text(" = ");
                match default.as_ref() {
                    Some(DefaultValue {
                        display: Some(text),
                        ..
                    }) => self.text(text),
                    Some(DefaultValue { ty: Type::None, .. }) => self.text("None"),
                    Some(DefaultValue {
                        ty: Type::Literal(literal),
                        ..
                    }) if !matches!(literal.value, Lit::Enum(_)) => {
                        self.text(&literal.value.to_string());
                    }
                    _ => {
                        // Arbitrary default expressions can depend on names in the defining module.
                        self.reference("typing", "cast");
                        self.text("(");
                        self.reference("typing", "Any");
                        self.text(", ...)");
                    }
                }
            }
        }
        if posonly {
            self.text(", /");
        }
    }
}
