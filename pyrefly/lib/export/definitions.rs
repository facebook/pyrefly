/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp;

use pyrefly_python::ast::Ast;
use pyrefly_python::docstring::Docstring;
use pyrefly_python::dunder;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModuleStyle;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::visit::Visit;
use ruff_python_ast::ExceptHandler;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprName;
use ruff_python_ast::Identifier;
use ruff_python_ast::Operator;
use ruff_python_ast::Pattern;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtExpr;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::small_map::Entry;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::types::globals::Global;

/// How a name is defined. If a name is defined outside of this
/// module, we additionally store the module we got it from
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DefinitionStyle {
    /// Defined in this module, e.g. `x = 1` or `def x(): ...`
    /// We also store what kind of symbol it is
    Local(SymbolKind),
    /// Defined as an implicit global like `__name__`.
    Global,
    /// Imported with an alias, e.g. `from x import y as z`
    ImportAs(ModuleName),
    /// Imported with an alias, where the alias is identical, e.g. `from x import y as y`
    ImportAsEq(ModuleName),
    /// Imported from another module, e.g. `from x import y`
    Import(ModuleName),
    /// Imported directly, e.g. `import x` or `import x.y` (both of which add `x`)
    ImportModule(ModuleName),
    /// A relative import that does not exist: we do `....` more than we have depth
    ImportInvalidRelative,
}

#[derive(Debug, Clone)]
pub struct Definition {
    /// A location where the definition is defined, there is no guarantee it is the first/last or otherwise.
    pub range: TextRange,
    /// If the definition occurs multiple times, the lowest `DefinitionStyle` is used (e.g. prefer `Local`).
    pub style: DefinitionStyle,
    /// The location of the first annotated name for this definition, if any.
    pub annot: Option<ShortIdentifier>,
    /// The number is the distinct times this variable was defined.
    pub count: usize,
    /// If the first statement in a definition (class, function) is a string literal, PEP 257 convention
    /// states that is is the docstring.
    pub docstring_range: Option<TextRange>,
}

/// Find the definitions available in a scope. Does not traverse inside classes/functions,
/// since they are separate scopes.
#[derive(Debug, Clone, Default)]
pub struct Definitions {
    /// All the things defined in this module.
    pub definitions: SmallMap<Name, Definition>,
    /// All the global names declared in this scope.
    /// TODO(grievejia): This field is currently unused, but may be useful in the future.
    pub globals: SmallSet<Name>,
    /// All the nonlocal names declared in this scope.
    /// TODO(grievejia): This field is currently unused, but may be useful in the future.
    pub nonlocals: SmallSet<Name>,
    /// All the modules that are imported with `from x import *`.
    pub import_all: SmallMap<ModuleName, TextRange>,
    /// The `__all__` variable contents.
    pub dunder_all: Vec<DunderAllEntry>,
    /// If the containing module `foo` is a __init__ file, then this is the set of submodules
    /// that are guaranteed to be imported under `foo` when `foo` is itself imported in downstream
    /// files.
    pub implicitly_imported_submodules: SmallSet<Name>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DunderAllEntry {
    Name(TextRange, Name),
    Module(TextRange, ModuleName),
    // We have to have this explicitly, as you might remove something in a Module
    Remove(TextRange, Name),
}

impl DunderAllEntry {
    fn is_all(x: &Expr) -> bool {
        matches!(x, Expr::Name(ExprName { id, .. }) if id == &dunder::ALL)
    }

    fn as_list(x: &Expr) -> Vec<Self> {
        match x {
            Expr::List(x) => x.elts.iter().filter_map(DunderAllEntry::as_item).collect(),
            Expr::Tuple(x) => x.elts.iter().filter_map(DunderAllEntry::as_item).collect(),
            Expr::Attribute(ExprAttribute { value, attr, .. })
                if let Expr::Name(name) = &**value
                    && attr.id == dunder::ALL =>
            {
                vec![DunderAllEntry::Module(
                    name.range,
                    ModuleName::from_name(&name.id),
                )]
            }
            _ => Vec::new(),
        }
    }

    fn as_item(x: &Expr) -> Option<Self> {
        match x {
            Expr::StringLiteral(x) => {
                Some(DunderAllEntry::Name(x.range, Name::new(x.value.to_str())))
            }
            _ => None,
        }
    }
}

struct DefinitionsBuilder<'a> {
    module_name: ModuleName,
    is_init: bool,
    sys_info: &'a SysInfo,
    inner: Definitions,
}

fn is_private_name(name: &Name) -> bool {
    name.starts_with('_')
}

fn implicitly_imported_submodule(
    importing_module_name: ModuleName,
    imported_module_name: ModuleName,
) -> Option<Name> {
    imported_module_name
        .components()
        .strip_prefix(importing_module_name.components().as_slice())
        .and_then(|components| components.first())
        .cloned()
}

impl Definitions {
    pub fn new(x: &[Stmt], module_name: ModuleName, is_init: bool, sys_info: &SysInfo) -> Self {
        let mut builder = DefinitionsBuilder {
            module_name,
            sys_info,
            is_init,
            inner: Definitions::default(),
        };
        builder.stmts(x);
        builder.inner
    }

    /// Add an implicit `from builtins import *` to the definitions.
    pub fn inject_builtins(&mut self) {
        self.import_all.entry(ModuleName::builtins()).or_default();
    }

    pub fn inject_globals(&mut self) {
        for global in Global::globals(false) {
            self.definitions.insert(
                global.name().clone(),
                Definition {
                    range: TextRange::default(),
                    style: DefinitionStyle::Global,
                    annot: None,
                    count: 1,
                    docstring_range: None,
                },
            );
        }
    }

    /// Ensure that `dunder_all` is populated, synthesising it if `__all__` isn't present.
    pub fn ensure_dunder_all(&mut self, style: ModuleStyle) {
        if self.definitions.contains_key(&dunder::ALL) {
            // Explicitly defined, so don't redefine it
            return;
        }
        if style == ModuleStyle::Executable {
            for (x, range) in self.import_all.iter() {
                self.dunder_all.push(DunderAllEntry::Module(*range, *x));
            }
        }
        for (name, def) in self.definitions.iter() {
            if !is_private_name(name)
                && (style == ModuleStyle::Executable
                    || matches!(
                        def.style,
                        DefinitionStyle::Local(_) | DefinitionStyle::ImportAsEq(_)
                    ))
            {
                self.dunder_all
                    .push(DunderAllEntry::Name(def.range, name.clone()));
            }
        }
    }

    /// Add these names to `duner_all`, if they are defined in the module.
    pub fn extend_dunder_all(&mut self, extra: &[Name]) {
        for name in extra {
            if let Some(def) = self.definitions.get(name) {
                self.dunder_all
                    .push(DunderAllEntry::Name(def.range, name.clone()))
            }
        }
    }
}

impl<'a> DefinitionsBuilder<'a> {
    fn stmts(&mut self, xs: &[Stmt]) {
        for x in xs {
            self.stmt(x);
        }
    }

    fn add_name_with_body(
        &mut self,
        x: &Name,
        range: TextRange,
        style: DefinitionStyle,
        annot: Option<ShortIdentifier>,
        body: Option<&[Stmt]>,
    ) {
        match self.inner.definitions.entry(x.clone()) {
            Entry::Occupied(mut e) => {
                e.get_mut().style = cmp::min(e.get().style, style);
                if e.get().annot.is_none() {
                    e.get_mut().annot = annot;
                }
                e.get_mut().count += 1;
            }
            Entry::Vacant(e) => {
                e.insert(Definition {
                    range,
                    style,
                    annot,
                    count: 1,
                    docstring_range: body.and_then(Docstring::range_from_stmts),
                });
            }
        }
    }

    fn add_name(
        &mut self,
        x: &Name,
        range: TextRange,
        style: DefinitionStyle,
        annot: Option<ShortIdentifier>,
    ) {
        self.add_name_with_body(x, range, style, annot, None)
    }

    fn add_identifier(&mut self, x: &Identifier, style: DefinitionStyle) {
        self.add_name(&x.id, x.range, style, None);
    }

    fn add_identifier_with_body(
        &mut self,
        x: &Identifier,
        style: DefinitionStyle,
        body: Option<&[Stmt]>,
    ) {
        self.add_name_with_body(&x.id, x.range, style, None, body);
    }

    fn expr_lvalue(&mut self, x: &Expr) {
        let mut add_name = |x: &ExprName| {
            self.add_name(
                &x.id,
                x.range,
                DefinitionStyle::Local(SymbolKind::Variable),
                None,
            )
        };
        Ast::expr_lvalue(x, &mut add_name);
        self.named_in_expr(x);
    }

    fn pattern(&mut self, x: &Pattern) {
        Ast::pattern_lvalue(x, &mut |x| {
            self.add_identifier(x, DefinitionStyle::Local(SymbolKind::Variable))
        });
    }

    fn stmt(&mut self, x: &Stmt) {
        match x {
            Stmt::Import(x) => {
                for a in &x.names {
                    let imported_module = ModuleName::from_name(&a.name.id);
                    if self.is_init
                        && let Some(submodule) =
                            implicitly_imported_submodule(self.module_name, imported_module)
                    {
                        self.inner.implicitly_imported_submodules.insert(submodule);
                    }
                    match &a.asname {
                        None => self.add_name(
                            &imported_module.first_component(),
                            a.name.range,
                            DefinitionStyle::ImportModule(imported_module),
                            None,
                        ),
                        Some(alias) => self.add_identifier(
                            alias,
                            if alias.id == a.name.id {
                                DefinitionStyle::ImportAsEq(imported_module)
                            } else {
                                DefinitionStyle::ImportAs(imported_module)
                            },
                        ),
                    };
                }
            }
            Stmt::ImportFrom(x) => {
                let name = self.module_name.new_maybe_relative(
                    self.is_init,
                    x.level,
                    x.module.as_ref().map(|x| &x.id),
                );
                if self.is_init
                    && let Some(imported_module) = name
                    && let Some(submodule) =
                        implicitly_imported_submodule(self.module_name, imported_module)
                {
                    self.inner.implicitly_imported_submodules.insert(submodule);
                }
                for a in &x.names {
                    if &a.name == "*" {
                        if let Some(module) = name {
                            self.inner.import_all.insert(module, a.name.range);
                        }
                    } else {
                        let style = match name {
                            None => DefinitionStyle::ImportInvalidRelative,
                            Some(name) => {
                                if a.asname.as_ref().map(|x| &x.id) == Some(&a.name.id) {
                                    DefinitionStyle::ImportAsEq(name)
                                } else if a.asname.is_some() {
                                    DefinitionStyle::ImportAs(name)
                                } else {
                                    DefinitionStyle::Import(name)
                                }
                            }
                        };
                        self.add_identifier(a.asname.as_ref().unwrap_or(&a.name), style);
                        if matches!(style, DefinitionStyle::ImportAsEq(_))
                            && a.name.id == dunder::ALL
                            && let Some(module) = name
                        {
                            self.inner.dunder_all = vec![DunderAllEntry::Module(x.range, module)]
                        }
                    }
                }
            }
            Stmt::ClassDef(x) => {
                self.add_identifier_with_body(
                    &x.name,
                    DefinitionStyle::Local(SymbolKind::Class),
                    Some(&x.body),
                );
                return; // These things are inside a scope
            }
            Stmt::Nonlocal(x) => {
                for name in &x.names {
                    self.inner.nonlocals.insert(name.id.clone());
                }
            }
            Stmt::Global(x) => {
                for name in &x.names {
                    self.inner.globals.insert(name.id.clone());
                }
            }
            Stmt::Assign(x) => {
                self.named_in_expr(&x.value);
                for t in &x.targets {
                    self.expr_lvalue(t);
                    if DunderAllEntry::is_all(t) {
                        self.inner.dunder_all = DunderAllEntry::as_list(&x.value);
                    }
                }
            }
            Stmt::AugAssign(x) => {
                self.named_in_expr(&x.value);
                if DunderAllEntry::is_all(&x.target) && x.op == Operator::Add {
                    self.inner
                        .dunder_all
                        .extend(DunderAllEntry::as_list(&x.value));
                } else if let Expr::Name(name) = &*x.target {
                    self.add_name(
                        &name.id,
                        name.range,
                        DefinitionStyle::Local(SymbolKind::Variable),
                        None,
                    )
                }
            }
            Stmt::Expr(StmtExpr { value, .. }) => {
                self.named_in_expr(value);
                if let Expr::Call(
                    ExprCall {
                        func, arguments, ..
                    },
                    ..,
                ) = &**value
                    && let Expr::Attribute(ExprAttribute { value, attr, .. }) = &**func
                    && DunderAllEntry::is_all(value)
                    && arguments.len() == 1
                    && arguments.keywords.is_empty()
                {
                    match attr.as_str() {
                        "extend" => self
                            .inner
                            .dunder_all
                            .extend(DunderAllEntry::as_list(&arguments.args[0])),
                        "append" => self
                            .inner
                            .dunder_all
                            .extend(DunderAllEntry::as_item(&arguments.args[0])),
                        "remove" => {
                            if let Some(DunderAllEntry::Name(range, remove)) =
                                DunderAllEntry::as_item(&arguments.args[0])
                            {
                                self.inner
                                    .dunder_all
                                    .push(DunderAllEntry::Remove(range, remove));
                            }
                        }
                        _ => {}
                    }
                }
            }
            Stmt::AnnAssign(x) => {
                if let Some(value) = &x.value {
                    self.named_in_expr(value);
                }
                match &*x.target {
                    Expr::Name(x) => {
                        self.add_name(
                            &x.id,
                            x.range,
                            DefinitionStyle::Local(SymbolKind::Variable),
                            Some(ShortIdentifier::expr_name(x)),
                        );
                    }
                    _ => self.expr_lvalue(&x.target),
                }
            }
            Stmt::TypeAlias(x) => {
                self.named_in_expr(&x.value);
                if matches!(&*x.name, Expr::Name(_)) {
                    self.expr_lvalue(&x.name)
                }
            }
            Stmt::FunctionDef(x) => {
                self.add_identifier_with_body(
                    &x.name,
                    DefinitionStyle::Local(SymbolKind::Function),
                    Some(&x.body),
                );
                return; // don't recurse because a separate scope
            }
            Stmt::For(x) => {
                self.named_in_expr(&x.iter);
                self.expr_lvalue(&x.target)
            }
            Stmt::With(x) => {
                for x in &x.items {
                    self.named_in_expr(&x.context_expr);
                    if let Some(target) = &x.optional_vars {
                        self.expr_lvalue(target);
                    }
                }
            }
            Stmt::Match(x) => {
                self.named_in_expr(&x.subject);
                for x in &x.cases {
                    self.pattern(&x.pattern);
                }
            }
            Stmt::Try(x) => {
                for x in &x.handlers {
                    match x {
                        ExceptHandler::ExceptHandler(x) => {
                            if let Some(name) = &x.name {
                                self.add_identifier(
                                    name,
                                    DefinitionStyle::Local(SymbolKind::Variable),
                                );
                            }
                        }
                    }
                }
            }
            Stmt::If(x) => {
                self.named_in_expr(&x.test);
                for (_, body) in self.sys_info.pruned_if_branches(x) {
                    self.stmts(body);
                }
                return; // We went through the relevant branches already
            }
            Stmt::While(x) => {
                self.named_in_expr(&x.test);
            }
            Stmt::Assert(x) => {
                self.named_in_expr(&x.test);
                if let Some(msg) = &x.msg {
                    self.named_in_expr(msg);
                }
            }
            Stmt::Raise(x) => {
                if let Some(exc) = &x.exc {
                    self.named_in_expr(exc);
                }
                if let Some(c) = &x.cause {
                    self.named_in_expr(c);
                }
            }
            Stmt::Return(..)
            | Stmt::Delete(..)
            | Stmt::Pass(..)
            | Stmt::Break(..)
            | Stmt::Continue(..)
            | Stmt::IpyEscapeCommand(..) => {}
        }
        x.recurse(&mut |xs| self.stmt(xs))
    }

    /// Accumulate names defined by walrus operators in an expression.
    fn named_in_expr(&mut self, x: &Expr) {
        match x {
            Expr::Named(expr_named) => {
                self.expr_lvalue(&expr_named.target);
            }
            Expr::Lambda(..) | Expr::SetComp(..) | Expr::DictComp(..) | Expr::ListComp(..) => {
                // These expressions define a scope, so walrus operators only define a name
                // within that scope, not in the surrounding statement's scope.
            }
            _ => x.recurse(&mut |x| self.named_in_expr(x)),
        }
    }
}

#[cfg(test)]
mod tests {
    use pyrefly_util::prelude::SliceExt;

    use super::*;

    #[test]
    fn test_implicitly_imported_submodule() {
        assert_eq!(
            implicitly_imported_submodule(
                ModuleName::from_str("foo"),
                ModuleName::from_str("foo.bar.baz")
            ),
            Some(Name::new_static("bar"))
        );

        assert_eq!(
            implicitly_imported_submodule(ModuleName::from_str("foo"), ModuleName::from_str("foo")),
            None
        );

        assert_eq!(
            implicitly_imported_submodule(
                ModuleName::from_str("foo.bar"),
                ModuleName::from_str("foo.bar.baz.qux")
            ),
            Some(Name::new_static("baz"))
        );

        assert_eq!(
            implicitly_imported_submodule(
                ModuleName::from_str("foo.bar"),
                ModuleName::from_str("baz.qux")
            ),
            None
        );
    }

    fn unrange(x: &mut DunderAllEntry) {
        match x {
            DunderAllEntry::Name(range, _)
            | DunderAllEntry::Module(range, _)
            | DunderAllEntry::Remove(range, _) => {
                *range = TextRange::default();
            }
        }
    }

    fn calculate_unranged_definitions(
        contents: &str,
        module_name: ModuleName,
        is_init: bool,
    ) -> Definitions {
        let mut res = Definitions::new(
            &Ast::parse(contents).0.body,
            module_name,
            is_init,
            &SysInfo::default(),
        );
        res.dunder_all.iter_mut().for_each(unrange);
        res
    }

    fn calculate_unranged_definitions_with_defaults(contents: &str) -> Definitions {
        calculate_unranged_definitions(contents, ModuleName::from_str("main"), false)
    }

    fn assert_import_all(defs: &Definitions, expected_import_all: &[&str]) {
        assert_eq!(
            expected_import_all,
            defs.import_all
                .keys()
                .map(|x| x.as_str())
                .collect::<Vec<_>>()
        );
    }

    fn assert_implicitly_imported_submodules(defs: &Definitions, expected: &[&str]) {
        assert_eq!(
            expected,
            defs.implicitly_imported_submodules
                .iter()
                .map(|x| x.as_str())
                .collect::<Vec<_>>()
        );
    }

    fn assert_definition_names(defs: &Definitions, expected_names: &[&str]) {
        assert_eq!(
            expected_names,
            defs.definitions
                .keys()
                .map(|x| x.as_str())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_definitions() {
        let defs = calculate_unranged_definitions_with_defaults(
            r#"
from foo import *
from bar import baz as qux
from bar import moo
import mod.ule
import mod.lue

def x():
    y = 1

for z, w in []:
    pass

no.thing = 8

n = True

r[p] = 1

type X = int
type Y[T] = list[T]

match x():
    case case0: pass
    case moo.Moo(case1): pass
"#,
        );
        assert_import_all(&defs, &["foo"]);
        assert_definition_names(
            &defs,
            &[
                "qux", "moo", "mod", "x", "z", "w", "n", "X", "Y", "case0", "case1",
            ],
        );
    }

    #[test]
    fn test_walrus() {
        let defs = calculate_unranged_definitions_with_defaults(
            r#"
# Most named expressions should appear in definitions.
y: int = (x0 := 42)
y = (x1 := 42)
y += (x2 := 42)
(x3 := 42)
with (x4 := 42) as y: pass
for y in (x5 := 42): pass
while (x6 := True): pass
match (x7 := 42):
    case int(): pass
(x8 := 42)[y] = 42
assert (x9 := 42), (x10 := "oops")
type y = (x11 := int)
# Named expressions inside expression-level scopes should not appear in definitions.
lambda x: (z := 42)
[z for x in [1, 2, 3] if z := x > 2]
"#,
        );
        assert_definition_names(
            &defs,
            &[
                "x0", "y", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11",
            ],
        );
    }

    #[test]
    fn test_overload() {
        let defs = calculate_unranged_definitions_with_defaults(
            r#"
from typing import overload

@overload
def foo(x: int) -> int: ...
@overload
def foo(x: str) -> str: ...
def foo(x: str | int) -> str | int:
    return x

def bar(x: int) -> int: ...
def bar(x: str) -> str: ...
            "#,
        );
        assert_import_all(&defs, &[]);
        assert_definition_names(&defs, &["overload", "foo", "bar"]);

        let foo = defs.definitions.get(&Name::new_static("foo")).unwrap();
        assert_eq!(foo.style, DefinitionStyle::Local(SymbolKind::Function));
        assert_eq!(foo.count, 3);

        let bar = defs.definitions.get(&Name::new_static("bar")).unwrap();
        assert_eq!(bar.style, DefinitionStyle::Local(SymbolKind::Function));
        assert_eq!(bar.count, 2);
    }

    #[test]
    fn test_all() {
        let defs = calculate_unranged_definitions_with_defaults(
            r#"
from foo import *
a = 1
b = 1

# Follow the spec at https://typing.readthedocs.io/en/latest/spec/distributing.html#library-interface-public-and-private-symbols
__all__ = ("a", "b")
__all__ += ["a", "b"]
__all__ += foo.__all__
__all__.extend(['a', 'b'])
__all__.extend(foo.__all__)
__all__.append('a')
__all__.remove('r')
        "#,
        );
        assert_import_all(&defs, &["foo"]);
        assert_definition_names(&defs, &["a", "b", "__all__"]);

        let loc = TextRange::default();
        let a = &DunderAllEntry::Name(loc, Name::new_static("a"));
        let b = &DunderAllEntry::Name(loc, Name::new_static("b"));
        let foo = &DunderAllEntry::Module(loc, ModuleName::from_str("foo"));
        let r = &DunderAllEntry::Remove(loc, Name::new_static("r"));
        assert_eq!(
            defs.dunder_all.map(|x| x),
            vec![a, b, a, b, foo, a, b, foo, a, r]
        );
    }

    #[test]
    fn test_all_reexport() {
        // Not in the spec, but see collections.abc which does this.
        let defs = calculate_unranged_definitions_with_defaults(
            r#"
from _collections_abc import *
from _collections_abc import __all__ as __all__
"#,
        );
        assert_import_all(&defs, &["_collections_abc"]);
        assert_definition_names(&defs, &["__all__"]);

        assert_eq!(
            defs.dunder_all,
            vec![DunderAllEntry::Module(
                TextRange::default(),
                ModuleName::from_str("_collections_abc")
            )]
        );
    }

    #[test]
    fn test_implicitly_imported_submodule_from_import_stmt() {
        let defs = calculate_unranged_definitions(
            r#"
from . import a
from .a import x
from .b.c import y
from ..derp.d import z
"#,
            ModuleName::from_str("derp"),
            true,
        );
        assert_implicitly_imported_submodules(&defs, &["a", "b", "d"]);
    }

    #[test]
    fn test_implicitly_imported_submodule_import_stmt() {
        let defs = calculate_unranged_definitions(
            r#"
import a
import derp.b
import derp.c.d
"#,
            ModuleName::from_str("derp"),
            true,
        );
        assert_implicitly_imported_submodules(&defs, &["b", "c"]);
    }
}
