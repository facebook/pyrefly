/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A flat, source-order table of definitions used to find nested workspace
//! symbols: functions, classes, methods, type aliases, and simple module/class
//! assignments, including attributes assigned through a method's receiver.
//! It is cached on `Exports` for first-party modules because the
//! export table itself contains only top-level names and therefore cannot
//! provide nested definitions.
//!
//! The table lists source declarations, not logical bindings, so a name that is
//! declared several times is recorded once per declaration and never merged.
//! A property's getter and setter, each member of an overload set, each branch
//! of a conditional definition, and same-name definitions of different kinds are
//! all distinct entries, because each is a place a user may want to jump to.
//! Collapsing them is a presentation decision for the consumer, which alone
//! knows what the user asked for; doing it here would destroy the information
//! irrecoverably.

use std::num::NonZeroU32;
use std::slice;

use pyrefly_python::ast::Ast;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Expr;
use ruff_python_ast::Stmt;
use ruff_python_ast::name::Name;

use crate::binding::scope::is_constant_name;

#[derive(Clone, Copy, Debug)]
struct FlatSymbolIndex(NonZeroU32);

impl FlatSymbolIndex {
    fn new(idx: usize) -> Self {
        let idx = u32::try_from(idx)
            .expect("flat symbol index should fit in u32")
            .checked_add(1)
            .expect("flat symbol count should fit in u32");
        Self(NonZeroU32::new(idx).expect("one-based index is nonzero"))
    }

    fn to_usize(self) -> usize {
        (self.0.get() - 1) as usize
    }
}

#[derive(Debug)]
pub struct FlatSymbol {
    /// The name's source range. Its text remains available from the module.
    pub name: ShortIdentifier,
    pub kind: SymbolKind,
    /// Index into the owning `FlatSymbols` of the enclosing definition, if any.
    /// Top-level definitions have no parent.
    parent: Option<FlatSymbolIndex>,
}

/// A compact, source-ordered table of declarations and their parent links.
#[derive(Debug)]
pub struct FlatSymbols(Box<[FlatSymbol]>);

impl FlatSymbols {
    /// Build the symbol table from a module body.
    pub fn new(body: &[Stmt]) -> Self {
        let mut out = Vec::new();
        build(
            body,
            Scope {
                parent: None,
                kind: ScopeKind::Module,
                receiver: None,
            },
            &mut out,
        );
        Self(out.into_boxed_slice())
    }

    /// Iterate in source order with each declaration's immediate parent.
    ///
    /// The parent is `None` only when no named enclosing definition exists.
    /// Parser recovery skips unnamed scopes and preserves the nearest named parent.
    pub fn iter(&self) -> impl Iterator<Item = (&FlatSymbol, Option<&FlatSymbol>)> {
        self.0
            .iter()
            .map(|symbol| (symbol, symbol.parent.map(|idx| &self.0[idx.to_usize()])))
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ScopeKind {
    Module,
    Class,
    Function,
}

/// Where a run of statements sits and which definition encloses it.
#[derive(Clone, Copy)]
struct Scope<'a> {
    /// Index of the enclosing definition; `None` at module level.
    parent: Option<FlatSymbolIndex>,
    kind: ScopeKind,
    /// A bound method's first positional parameter and its enclosing class.
    /// Nested functions and classes establish their own receiver scope.
    receiver: Option<(&'a Name, FlatSymbolIndex)>,
}

fn push_symbol(
    out: &mut Vec<FlatSymbol>,
    name: ShortIdentifier,
    kind: SymbolKind,
    parent: Option<FlatSymbolIndex>,
) -> FlatSymbolIndex {
    let idx = FlatSymbolIndex::new(out.len());
    out.push(FlatSymbol { name, kind, parent });
    idx
}

fn assignment_kind(name: &Name, scope: Scope<'_>) -> SymbolKind {
    if is_constant_name(name) {
        SymbolKind::Constant
    } else if scope.kind == ScopeKind::Class {
        SymbolKind::Attribute
    } else {
        SymbolKind::Variable
    }
}

fn push_assignment_targets(out: &mut Vec<FlatSymbol>, target: &Expr, scope: Scope<'_>) {
    match target {
        Expr::Name(name)
            if scope.kind != ScopeKind::Function && !Ast::is_synthesized_empty_name(name) =>
        {
            push_symbol(
                out,
                ShortIdentifier::expr_name(name),
                assignment_kind(&name.id, scope),
                scope.parent,
            );
        }
        Expr::Attribute(attr) => {
            if let Some((receiver, class)) = scope.receiver
                && let Expr::Name(value) = &*attr.value
                && &value.id == receiver
                && !Ast::is_synthesized_empty_identifier(&attr.attr)
            {
                push_symbol(
                    out,
                    ShortIdentifier::new(&attr.attr),
                    SymbolKind::Attribute,
                    Some(class),
                );
            }
        }
        Expr::Tuple(tuple) => {
            for target in &tuple.elts {
                push_assignment_targets(out, target, scope);
            }
        }
        Expr::List(list) => {
            for target in &list.elts {
                push_assignment_targets(out, target, scope);
            }
        }
        Expr::Starred(starred) => push_assignment_targets(out, &starred.value, scope),
        _ => {}
    }
}

/// Walk `stmts` appending symbols to `out`.
///
/// Functions and classes are recorded at any depth, including inside a function
/// body. Control-flow statements are descended into with the scope unchanged, so
/// a class attribute guarded by an `if` still attaches to its class.
fn build<'a>(stmts: &'a [Stmt], scope: Scope<'a>, out: &mut Vec<FlatSymbol>) {
    for stmt in stmts {
        match stmt {
            Stmt::FunctionDef(f) => {
                // Parser error recovery can leave the name empty. Such a def is
                // not a navigable symbol, but its body is still a function
                // scope, so it is descended into as one and anything inside it
                // attaches to the enclosing definition.
                let body_parent = if Ast::is_synthesized_empty_identifier(&f.name) {
                    scope.parent
                } else {
                    Some(push_symbol(
                        out,
                        ShortIdentifier::new(&f.name),
                        if scope.kind == ScopeKind::Class {
                            SymbolKind::Method
                        } else {
                            SymbolKind::Function
                        },
                        scope.parent,
                    ))
                };
                // The export pass is syntactic: recognize spelled-out staticmethod
                // decorators without resolving imports or evaluating decorators.
                let is_staticmethod =
                    f.decorator_list
                        .iter()
                        .any(|decorator| match &decorator.expression {
                            Expr::Name(name) => name.id == "staticmethod",
                            Expr::Attribute(attr) => attr.attr.id == "staticmethod",
                            _ => false,
                        });
                let receiver = if scope.kind == ScopeKind::Class && !is_staticmethod {
                    f.parameters
                        .posonlyargs
                        .first()
                        .or_else(|| f.parameters.args.first())
                        .zip(scope.parent)
                        .map(|(parameter, class)| (&parameter.parameter.name.id, class))
                } else {
                    None
                };
                build(
                    &f.body,
                    Scope {
                        parent: body_parent,
                        kind: ScopeKind::Function,
                        receiver,
                    },
                    out,
                );
            }
            Stmt::ClassDef(c) => {
                let body_parent = if Ast::is_synthesized_empty_identifier(&c.name) {
                    scope.parent
                } else {
                    Some(push_symbol(
                        out,
                        ShortIdentifier::new(&c.name),
                        SymbolKind::Class,
                        scope.parent,
                    ))
                };
                build(
                    &c.body,
                    Scope {
                        parent: body_parent,
                        kind: ScopeKind::Class,
                        receiver: None,
                    },
                    out,
                );
            }
            Stmt::Assign(a) => {
                for target in &a.targets {
                    push_assignment_targets(out, target, scope);
                }
            }
            Stmt::AnnAssign(a) => {
                push_assignment_targets(out, &a.target, scope);
            }
            Stmt::TypeAlias(t) if scope.kind != ScopeKind::Function => {
                if let Expr::Name(name) = &*t.name
                    && !Ast::is_synthesized_empty_name(name)
                {
                    push_symbol(
                        out,
                        ShortIdentifier::expr_name(name),
                        SymbolKind::TypeAlias,
                        scope.parent,
                    );
                }
            }
            _ => stmt.recurse(&mut |s: &Stmt| build(slice::from_ref(s), scope, out)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use pyrefly_python::ast::Ast;
    use ruff_python_ast::PySourceType;
    use ruff_text_size::Ranged;

    use super::*;

    /// Render each symbol as `kind name` prefixed by one `.` per level of
    /// nesting, so both the flat order and the parent links are covered.
    fn walk(source: &str) -> Vec<String> {
        let ast = Ast::parse(source, PySourceType::Python).0;
        let symbols = FlatSymbols::new(&ast.body);
        symbols
            .iter()
            .map(|(sym, mut parent)| {
                let mut depth = 0;
                while let Some(symbol) = parent {
                    depth += 1;
                    parent = symbol.parent.map(|idx| &symbols.0[idx.to_usize()]);
                }
                let range = sym.name.range();
                let name = &source[range.start().to_usize()..range.end().to_usize()];
                format!("{}{:?} {name}", ".".repeat(depth), sym.kind)
            })
            .collect()
    }

    #[test]
    fn test_compact_layout() {
        assert!(
            size_of::<FlatSymbol>() <= 16,
            "a flat symbol should fit its source range, kind, and parent index in 16 bytes"
        );
    }

    #[test]
    fn test_methods_and_class_variables() {
        assert_eq!(
            walk(
                "MAX_SIZE = 1\nx = 1\nclass C:\n  MAX_CLASS_SIZE = 2\n  y: int = 2\n  def m(self): pass\ndef f(): pass\n"
            ),
            vec![
                "Constant MAX_SIZE",
                "Variable x",
                "Class C",
                ".Constant MAX_CLASS_SIZE",
                ".Attribute y",
                ".Method m",
                "Function f",
            ]
        );
    }

    #[test]
    fn test_function_locals_are_suppressed_but_nested_defs_are_not() {
        assert_eq!(
            walk("def f():\n  local = 1\n  def g(): pass\n  class D:\n    z = 2\n"),
            vec!["Function f", ".Function g", ".Class D", "..Attribute z"]
        );
    }

    #[test]
    fn test_receiver_attribute_assignment_targets() {
        assert_eq!(
            walk(
                r#"
class Example:
    def __init__(this, /):
        this._private = this.public = 1
        this.annotated: int = 2
        this.first, [this.second, *this.rest] = values
        if condition:
            this.conditional = 3
        local = 4
        other.unrelated = 5
        this.child.unrelated = 6
        this.items[0] = 7
"#
            ),
            vec![
                "Class Example",
                ".Method __init__",
                ".Attribute _private",
                ".Attribute public",
                ".Attribute annotated",
                ".Attribute first",
                ".Attribute second",
                ".Attribute rest",
                ".Attribute conditional",
            ]
        );
    }

    #[test]
    fn test_receiver_scope_does_not_leak_into_nested_definitions() {
        assert_eq!(
            walk(
                r#"
def top_level(self):
    self.unrelated = 1
class Example:
    def method(self):
        def nested(self):
            self.unrelated = 2
        class Inner:
            self.unrelated = 3
            def method(this):
                this.inner = 4
        self.outer = 5
"#
            ),
            vec![
                "Function top_level",
                "Class Example",
                ".Method method",
                "..Function nested",
                "..Class Inner",
                "...Method method",
                "...Attribute inner",
                ".Attribute outer",
            ]
        );
    }

    #[test]
    fn test_static_methods_have_no_receiver() {
        assert_eq!(
            walk(
                r#"
class Example:
    @staticmethod
    def static(self):
        self.unrelated = 1
    @builtins.staticmethod
    def qualified_static(self):
        self.unrelated = 2
    def no_positional_parameter(*, self):
        self.unrelated = 3
    @classmethod
    def class_method(cls):
        cls.class_attribute = 4
"#
            ),
            vec![
                "Class Example",
                ".Method static",
                ".Method qualified_static",
                ".Method no_positional_parameter",
                ".Method class_method",
                ".Attribute class_attribute",
            ]
        );
    }

    #[test]
    fn test_control_flow_does_not_change_the_enclosing_scope() {
        assert_eq!(
            walk("class C:\n  if True:\n    a = 1\n    def m(self): pass\n  else:\n    b = 2\n"),
            vec!["Class C", ".Attribute a", ".Method m", ".Attribute b"]
        );
    }

    /// A PEP 695 alias is a definition worth jumping to, and it is gated like a
    /// variable: recorded at module and class level, suppressed in a function.
    #[test]
    fn test_type_aliases() {
        assert_eq!(
            walk("type A = int\nclass C:\n  type B = str\ndef f():\n  type D = bytes\n"),
            vec!["TypeAlias A", "Class C", ".TypeAlias B", "Function f"]
        );
    }

    #[test]
    fn test_assignment_targets() {
        assert_eq!(
            walk("a = b = 1\nclass C:\n  x, [y, *rest] = values\n  obj.attr = 1\n  items[0] = 2\n"),
            vec![
                "Variable a",
                "Variable b",
                "Class C",
                ".Attribute x",
                ".Attribute y",
                ".Attribute rest",
            ]
        );
    }

    /// `for`/`with` bindings are not definitions we surface.
    #[test]
    fn test_unsupported_binding_forms_are_skipped() {
        assert_eq!(
            walk("for i in []: pass\nwith ctx() as value: pass\n"),
            Vec::<String>::new()
        );
    }

    /// A property's accessors are separate declarations at separate lines, so
    /// both are listed. Which one to show is the consumer's decision.
    #[test]
    fn test_property_accessor_pair_is_not_merged() {
        assert_eq!(
            walk(
                "class C:\n  @property\n  def value(self): pass\n  @value.setter\n  def value(self, v): pass\n"
            ),
            vec!["Class C", ".Method value", ".Method value"]
        );
    }

    /// Every member of an overload set is listed, implementation included: each
    /// is a distinct signature a user may be looking for.
    #[test]
    fn test_overload_set_is_not_merged() {
        assert_eq!(
            walk("@overload\ndef f(x: int): ...\n@overload\ndef f(x: str): ...\ndef f(x): pass\n"),
            vec!["Function f", "Function f", "Function f"]
        );
    }

    /// Conditional definitions are distinct implementations at distinct
    /// locations, and control flow leaves the scope unchanged, so both branches
    /// are listed at the enclosing level.
    #[test]
    fn test_conditional_definitions_are_not_merged() {
        assert_eq!(
            walk("if TYPE_CHECKING:\n  def f(): pass\nelse:\n  def f(): pass\n"),
            vec!["Function f", "Function f"]
        );
    }

    /// The same name bound as a class, a function and a variable stays three
    /// entries, and each keeps its own kind.
    #[test]
    fn test_same_name_different_kinds_are_not_merged() {
        assert_eq!(
            walk("class X: pass\ndef X(): pass\nX = 1\n"),
            vec!["Class X", "Function X", "Constant X"]
        );
    }

    /// A def or class whose name the parser could not recover yields no symbol,
    /// but still establishes its scope: locals inside the unnamed function stay
    /// suppressed, and a def inside the unnamed class is still a `Method`.
    /// Anything it contains attaches to the enclosing definition.
    #[test]
    fn test_unnamed_definitions_still_establish_scope() {
        assert_eq!(
            walk("def ():\n  local = 1\n  def g(): pass\n"),
            vec!["Function g"]
        );
        assert_eq!(walk("class :\n  def m(self): pass\n"), vec!["Method m"]);
    }
}
