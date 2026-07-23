/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::iter;
use std::slice;

use pyrefly_util::visit::Visit;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::AtomicNodeIndex;
use ruff_python_ast::CmpOp;
use ruff_python_ast::DictItem;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprBinOp;
use ruff_python_ast::ExprBooleanLiteral;
use ruff_python_ast::ExprCompare;
use ruff_python_ast::ExprName;
use ruff_python_ast::ExprNoneLiteral;
use ruff_python_ast::ExprStringLiteral;
use ruff_python_ast::Identifier;
use ruff_python_ast::ModModule;
use ruff_python_ast::Number;
use ruff_python_ast::Operator;
use ruff_python_ast::Parameter;
use ruff_python_ast::ParameterWithDefault;
use ruff_python_ast::Parameters;
use ruff_python_ast::Pattern;
use ruff_python_ast::PatternMatchSingleton;
use ruff_python_ast::PySourceType;
use ruff_python_ast::PythonVersion as RuffPythonVersion;
use ruff_python_ast::Singleton;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtIf;
use ruff_python_ast::StringFlags;
use ruff_python_ast::StringLiteral;
use ruff_python_ast::StringLiteralFlags;
use ruff_python_ast::StringLiteralValue;
use ruff_python_ast::name::Name;
use ruff_python_ast::visitor::source_order::SourceOrderVisitor;
use ruff_python_ast::visitor::source_order::TraversalSignal;
use ruff_python_parser::ParseError;
use ruff_python_parser::ParseOptions;
use ruff_python_parser::Parsed;
use ruff_python_parser::UnsupportedSyntaxError;
use ruff_python_parser::parse_expression_range;
use ruff_python_parser::parse_unchecked;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use thin_vec::ThinVec;

use crate::sys_info::PythonVersion;

/// Just used for convenient namespacing - not a real type
pub struct Ast;

struct CoveringNodeVisitor<'a> {
    position: TextSize,
    level: usize,
    covering_nodes: Vec<AnyNodeRef<'a>>,
}

impl CoveringNodeVisitor<'_> {
    fn new(position: TextSize) -> Self {
        Self {
            position,
            level: 0,
            covering_nodes: Vec::new(),
        }
    }
}

impl<'a> SourceOrderVisitor<'a> for CoveringNodeVisitor<'a> {
    fn enter_node(&mut self, node: AnyNodeRef<'a>) -> TraversalSignal {
        self.level += 1;
        if self.level <= self.covering_nodes.len() {
            // This is to prevent the (extremely niche) case where multiple sibling nodes cover the same range.
            // If we have already found a covering node at the same level, we can stop looking.
            TraversalSignal::Skip
        } else if node.range().contains_inclusive(self.position) {
            // We can't stop looking because there could be child nodes that cover `self.range` more tightly.
            self.covering_nodes.push(node);
            TraversalSignal::Traverse
        } else {
            TraversalSignal::Skip
        }
    }

    fn leave_node(&mut self, _: AnyNodeRef<'a>) {
        self.level -= 1;
    }
}

impl Ast {
    pub fn parse(
        contents: &str,
        source_type: PySourceType,
    ) -> (ModModule, Vec<ParseError>, Vec<UnsupportedSyntaxError>) {
        let (parsed, parse_errors, unsupported_syntax_errors) =
            Ast::parse_with_version(contents, PythonVersion::default(), source_type);
        (
            parsed.into_syntax(),
            parse_errors,
            unsupported_syntax_errors,
        )
    }

    pub fn parse_with_version(
        contents: &str,
        version: PythonVersion,
        source_type: PySourceType,
    ) -> (
        Parsed<ModModule>,
        Vec<ParseError>,
        Vec<UnsupportedSyntaxError>,
    ) {
        // PySourceType of Python vs Stub doesn't actually change the parsing
        let options = ParseOptions::from(source_type).with_target_version(RuffPythonVersion {
            major: version.major as u8,
            minor: version.minor as u8,
        });
        let res = parse_unchecked(contents, options)
            .try_into_module()
            .unwrap();
        let parse_errors = res.errors().to_owned();
        let unsupported_syntax_errors = res.unsupported_syntax_errors().to_owned();
        (res, parse_errors, unsupported_syntax_errors)
    }

    pub fn parse_expr(contents: &str, pos: TextSize) -> anyhow::Result<Expr> {
        // I really want to use Parser::new_starts_at, but it's private.
        // Discussion in https://github.com/astral-sh/ruff/pull/13542.
        // Until then, fake it with a lot of spaces.
        let s = format!("{}{contents}", " ".repeat(pos.to_usize()));
        let end = pos
            .checked_add(TextSize::new(contents.len() as u32))
            .unwrap();
        Ok(*parse_expression_range(&s, TextRange::new(pos, end))?
            .into_syntax()
            .body)
    }

    pub fn parse_type_literal(x: &StringLiteral) -> anyhow::Result<Expr> {
        let mut s = &*x.value;
        let buffer;
        let mut add = x.flags.prefix().text_len() + TextSize::new(1);

        if x.flags.is_triple_quoted() {
            // Implicitly bracketed, so add them explicitly
            buffer = format!("({s})");
            s = &buffer;
            add += TextSize::new(1); // 3 for the quotes, minus 1 for the bracket, minus 1 for the raw quote
        }
        // Make sure the range is precise, so that we get the right UTF8 indices.
        // We might have a problem with \ escapes moving indices, but if necessary we can ban those.
        Ast::parse_expr(s, x.range.start() + add)
    }

    pub fn unpack_slice(x: &Expr) -> &[Expr] {
        match x {
            Expr::Tuple(x) => &x.elts,
            _ => slice::from_ref(x),
        }
    }

    pub fn is_literal(x: &Expr) -> bool {
        matches!(
            x,
            Expr::BooleanLiteral(_)
                | Expr::NumberLiteral(_)
                | Expr::StringLiteral(_)
                | Expr::BytesLiteral(_)
                | Expr::NoneLiteral(_)
                | Expr::EllipsisLiteral(_)
        )
    }

    /// Iterates over the branches of an if statement, returning the test and body.
    /// A test on `None` is an `else` branch that is always taken.
    pub fn if_branches(x: &StmtIf) -> impl Iterator<Item = (Option<&Expr>, &[Stmt])> {
        let first = iter::once((Some(&*x.test), x.body.as_slice()));
        let elses = x
            .elif_else_clauses
            .iter()
            .map(|x| (x.test.as_ref(), x.body.as_slice()));
        first.chain(elses)
    }

    /// Like `if_branches`, but returns owned values.
    pub fn if_branches_owned(
        x: StmtIf,
    ) -> impl Iterator<Item = (TextRange, Option<Expr>, ThinVec<Stmt>)> {
        let first = iter::once((x.range, Some(*x.test), x.body));
        let elses = x
            .elif_else_clauses
            .into_iter()
            .map(|x| (x.range, x.test, x.body));
        first.chain(elses)
    }

    pub fn is_main_guard(test: &Expr) -> bool {
        let Expr::Compare(ExprCompare {
            left,
            ops,
            comparators,
            ..
        }) = test
        else {
            return false;
        };

        if ops.len() != 1 || comparators.len() != 1 {
            return false;
        }

        let op = ops[0];
        if !matches!(op, CmpOp::Eq | CmpOp::Is) {
            return false;
        }

        let left = left.as_ref();
        let right = &comparators[0];
        (Self::is_name_dunder_name(left) && Self::is_main_string(right))
            || (Self::is_main_string(left) && Self::is_name_dunder_name(right))
    }

    fn is_name_dunder_name(expr: &Expr) -> bool {
        matches!(expr, Expr::Name(name) if name.id.as_str() == "__name__")
    }

    fn is_main_string(expr: &Expr) -> bool {
        matches!(
            expr,
            Expr::StringLiteral(ExprStringLiteral { value, .. }) if value.to_str() == "__main__"
        )
    }

    /// Iterates over parameters, returning the parameters and defaults
    pub fn parameters_iter_mut(
        x: &mut Parameters,
    ) -> impl Iterator<Item = (&mut Parameter, Option<&mut Option<Box<Expr>>>)> {
        fn param_default(
            x: &mut ParameterWithDefault,
        ) -> (&mut Parameter, Option<&mut Option<Box<Expr>>>) {
            (&mut x.parameter, Some(&mut x.default))
        }
        fn param(x: &mut Box<Parameter>) -> (&mut Parameter, Option<&mut Option<Box<Expr>>>) {
            (&mut *x, None)
        }

        x.posonlyargs
            .iter_mut()
            .map(param_default)
            .chain(x.args.iter_mut().map(param_default))
            .chain(x.vararg.iter_mut().map(param))
            .chain(x.kwonlyargs.iter_mut().map(param_default))
            .chain(x.kwarg.iter_mut().map(param))
    }

    /// We really want to avoid "making up" identifiers out of nowhere.
    /// But there, there isn't an identifier, but morally should be, so create the implicit one.
    pub fn expr_name_identifier(x: ExprName) -> Identifier {
        Identifier::new(x.id, x.range)
    }

    /// The trailing identifier of a decorator expression, looking through calls and
    /// attribute access: `foo`, `mod.foo`, `foo(...)`, and `mod.foo(...)` all yield
    /// `foo`. `None` for shapes with no trailing name (e.g. a subscript).
    pub fn decorator_trailing_name(decorator: &Expr) -> Option<&str> {
        match decorator {
            Expr::Name(name) => Some(name.id.as_str()),
            Expr::Attribute(attribute) => Some(attribute.attr.as_str()),
            Expr::Call(call) => Self::decorator_trailing_name(&call.func),
            _ => None,
        }
    }

    /// Returns true if this is a synthesized empty name from parser error recovery.
    ///
    /// The parser uses empty identifiers when recovering from syntax errors.
    /// Treat any empty identifier as synthesized, even if we still know the range, so
    /// downstream stages don't try to bind it.
    pub fn is_synthesized_empty_name(x: &ExprName) -> bool {
        x.id.as_str().is_empty()
    }

    /// Same as `is_synthesized_empty_name` but for `Identifier` instead of `ExprName`.
    pub fn is_synthesized_empty_identifier(x: &Identifier) -> bool {
        x.id.as_str().is_empty()
    }

    /// Calls a function on all of the names bound by this lvalue expression.
    pub fn expr_lvalue<'a>(x: &'a Expr, f: &mut impl FnMut(&'a ExprName)) {
        match x {
            Expr::Name(x) if !Self::is_synthesized_empty_name(x) => {
                f(x);
            }
            Expr::Tuple(x) => {
                for x in &x.elts {
                    Ast::expr_lvalue(x, f);
                }
            }

            Expr::List(x) => {
                for x in &x.elts {
                    Ast::expr_lvalue(x, f);
                }
            }
            Expr::Starred(x) => {
                Ast::expr_lvalue(&x.value, f);
            }
            Expr::Subscript(_) => { /* no-op */ }
            Expr::Attribute(_) => { /* no-op */ }
            _ => {
                // Should not occur in well-formed Python code, doesn't introduce bindings.
                // Will raise an error later.
            }
        }
    }

    /// The [`Pattern`] type contains lvalues as identifiers. Although some patterns like
    /// MatchValue contain [`Expr`], those do not contain lvalues and thus are ignored.
    pub fn pattern_lvalue<'a>(x: &'a Pattern, f: &mut impl FnMut(&'a Identifier)) {
        match x {
            Pattern::MatchStar(x) => {
                if let Some(x) = &x.name {
                    f(x);
                }
            }
            Pattern::MatchAs(x) => {
                if let Some(x) = &x.name {
                    f(x);
                }
            }
            Pattern::MatchMapping(x) => {
                if let Some(x) = &x.rest {
                    f(x);
                }
            }
            _ => {}
        }
        x.recurse(&mut |x| Ast::pattern_lvalue(x, f));
    }

    /// Pull all dictionary items up to the top level, so `{a: 1, **{b: 2}}`
    /// has the same items as `{a: 1, b: 2}`.
    pub fn flatten_dict_items<'b>(x: &'b [DictItem]) -> Vec<&'b DictItem> {
        fn f<'b>(xs: &'b [DictItem], res: &mut Vec<&'b DictItem>) {
            for x in xs {
                if x.key.is_none()
                    && let Expr::Dict(dict) = &x.value
                {
                    f(&dict.items, res);
                } else {
                    res.push(x);
                }
            }
        }
        let mut res = Vec::new();
        f(x, &mut res);
        res
    }

    pub fn pattern_match_singleton_to_expr(x: &PatternMatchSingleton) -> Expr {
        match x.value {
            Singleton::None => Expr::NoneLiteral(ExprNoneLiteral {
                node_index: AtomicNodeIndex::default(),
                range: x.range,
            }),
            Singleton::True | Singleton::False => Expr::BooleanLiteral(ExprBooleanLiteral {
                node_index: AtomicNodeIndex::default(),
                range: x.range,
                value: x.value == Singleton::True,
            }),
        }
    }

    /// Does the module have a docstring.
    pub fn has_docstring(x: &ModModule) -> bool {
        matches!(
            x.body.first(),
            Some(Stmt::Expr(x)) if x.value.is_string_literal_expr()
        )
    }

    /// Given a module and a position, find all AST nodes that "cover" the position.
    /// Return a vector of AST nodes sorted by the node's range, where the "innermost" node that covers the
    /// position comes first, and parent nodes of that covering node come later.
    pub fn locate_node<'a>(module: &'a ModModule, position: TextSize) -> Vec<AnyNodeRef<'a>> {
        let mut visitor = CoveringNodeVisitor::new(position);
        AnyNodeRef::from(module).visit_source_order(&mut visitor);
        let mut covering_nodes = visitor.covering_nodes;
        covering_nodes.reverse();
        covering_nodes
    }

    /// The tightest AST node that strictly contains `target` — i.e. the parent
    /// of the node whose range is `target`, or `None` at module top level.
    pub fn parent_node(module: &ModModule, target: TextRange) -> Option<AnyNodeRef<'_>> {
        Ast::locate_node(module, target.start())
            .into_iter()
            .find(|node| node.range() != target && node.range().contains_range(target))
    }

    /// Whether `node` must be wrapped in parentheses to preserve its meaning
    /// when spliced in as a direct child of `parent` (the node that will contain
    /// it, or `None` when the containing node is unknown).
    ///
    /// Self-delimiting expressions — names; string/bytes/bool/`None`/`...`
    /// literals; f-strings; calls; subscripts; attribute access; and bracketed
    /// containers — never need brackets. An integer literal is the only
    /// context-sensitive case, so it is the only one that consults `parent`.
    /// Everything else (tuples, boolean/binary/comparison/unary operations,
    /// conditionals, lambdas, walrus, generators, await/yield) binds loosely and
    /// is bracketed conservatively wherever it becomes a sub-expression,
    /// regardless of `parent`.
    pub fn needs_brackets(parent: Option<AnyNodeRef>, node: &Expr) -> bool {
        match node {
            Expr::Name(_)
            | Expr::StringLiteral(_)
            | Expr::BytesLiteral(_)
            | Expr::BooleanLiteral(_)
            | Expr::NoneLiteral(_)
            | Expr::EllipsisLiteral(_)
            | Expr::FString(_)
            | Expr::List(_)
            | Expr::Dict(_)
            | Expr::Set(_)
            | Expr::Call(_)
            | Expr::Subscript(_)
            | Expr::Attribute(_) => false,
            Expr::NumberLiteral(number) => {
                // `42.bit_length()` parses `42.` as a float, so an integer
                // literal needs brackets only directly before attribute access.
                matches!(number.value, Number::Int(_))
                    && matches!(parent, Some(AnyNodeRef::ExprAttribute(_)))
            }
            _ => true,
        }
    }

    pub fn str_expr(s: &str, range: TextRange) -> Expr {
        Expr::StringLiteral(ExprStringLiteral {
            node_index: AtomicNodeIndex::default(),
            range,
            value: StringLiteralValue::single(StringLiteral {
                node_index: AtomicNodeIndex::default(),
                range,
                value: s.into(),
                flags: StringLiteralFlags::empty(),
            }),
        })
    }

    pub fn contains_await(expr: &Expr) -> bool {
        let mut found = false;
        // Recursive function that checks this node and recurses to children
        fn check(expr: &Expr, found: &mut bool) {
            if matches!(expr, Expr::Await(_)) {
                *found = true;
            }
            expr.recurse(&mut |child: &Expr| check(child, found));
        }
        expr.visit(&mut |node: &Expr| check(node, &mut found));
        found
    }

    /// Check if a body of statements syntactically contains `yield` or `yield from`
    /// at the current function scope level. Does not descend into nested function
    /// definitions, class definitions, or lambdas, since yields in those scopes
    /// belong to those scopes, not the enclosing one.
    ///
    /// This is used to detect generators even when the yield is inside a
    /// statically-dead branch like `if False:`, where the binding phase skips
    /// traversal but Python still treats the function as a generator.
    pub fn body_contains_yield(stmts: &[Stmt]) -> bool {
        use ruff_python_ast::visitor;
        use ruff_python_ast::visitor::Visitor;

        struct YieldFinder {
            found: bool,
        }

        impl<'a> Visitor<'a> for YieldFinder {
            fn visit_stmt(&mut self, stmt: &'a Stmt) {
                if self.found {
                    return;
                }
                match stmt {
                    // Nested function/class definitions create new scopes.
                    Stmt::FunctionDef(_) | Stmt::ClassDef(_) => {}
                    _ => visitor::walk_stmt(self, stmt),
                }
            }

            fn visit_expr(&mut self, expr: &'a Expr) {
                if self.found {
                    return;
                }
                match expr {
                    Expr::Yield(_) | Expr::YieldFrom(_) => self.found = true,
                    // Lambda creates a new scope.
                    Expr::Lambda(_) => {}
                    _ => visitor::walk_expr(self, expr),
                }
            }
        }

        let mut finder = YieldFinder { found: false };
        for stmt in stmts {
            finder.visit_stmt(stmt);
            if finder.found {
                return true;
            }
        }
        false
    }

    pub fn is_mangled_attr(name: &Name) -> bool {
        name.starts_with("__") && !name.ends_with("__")
    }

    // Parameters and variables that are prefixed (but not suffixed) with a single underscore
    // are potentially unused, so we should skip some diagnostics/errors.
    // Examples: `_`, `_x`
    // Non-examples: `x`, `__x__`, `__x`, `_x_`
    pub fn is_intentionally_unused(name: &str) -> bool {
        name.starts_with('_')
            && !name.starts_with("__")
            && (name.len() == 1 || !name.ends_with('_'))
    }

    pub fn is_list_literal_or_comprehension(expr: &Expr) -> bool {
        matches!(expr, Expr::List(_) | Expr::ListComp(_))
    }

    /// Returns a description of the syntax problem if `x` is not valid
    /// annotation syntax, or `None` if valid. This only checks syntax;
    /// semantic validation (e.g. that `typing.Self` is in a class context)
    /// is handled elsewhere.
    /// See https://typing.readthedocs.io/en/latest/spec/annotations.html#type-and-annotation-expressions
    pub fn annotation_syntax_problem(x: &Expr) -> Option<&'static str> {
        match x {
            Expr::Name(..)
            | Expr::Named(..)
            | Expr::StringLiteral(..)
            | Expr::NoneLiteral(..)
            | Expr::Attribute(..)
            | Expr::Starred(..) => None,
            Expr::BinOp(ExprBinOp {
                op: Operator::BitOr,
                ..
            }) => None,
            Expr::Subscript(s) => match *s.value {
                Expr::Name(..)
                | Expr::BinOp(ExprBinOp {
                    op: Operator::BitOr,
                    ..
                })
                | Expr::Named(..)
                | Expr::StringLiteral(..)
                | Expr::NoneLiteral(..)
                | Expr::Attribute(..) => None,
                _ => Some("Invalid subscript expression"),
            },
            Expr::Call(..) => Some("Function call"),
            Expr::Lambda(..) => Some("Lambda definition"),
            Expr::List(..) => Some("List literal"),
            Expr::NumberLiteral(..) => Some("Number literal"),
            Expr::Tuple(..) => Some("Tuple literal"),
            Expr::Dict(..) => Some("Dict literal"),
            Expr::ListComp(..) => Some("List comprehension"),
            Expr::If(..) => Some("If expression"),
            Expr::BooleanLiteral(..) => Some("Bool literal"),
            Expr::BoolOp(..) => Some("Boolean operation"),
            Expr::FString(..) => Some("F-string"),
            Expr::TString(..) => Some("T-string"),
            Expr::UnaryOp(..) => Some("Unary operation"),
            // Intentionally omits the specific operator to keep `&'static str`.
            Expr::BinOp(..) => Some("Binary operation"),
            _ => Some("Expression"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_expr_stmt(code: &str) -> Expr {
        let (module, errors, _) = Ast::parse(code, PySourceType::Python);
        assert!(errors.is_empty(), "unexpected parse errors in {code:?}");
        match module.body.into_iter().next() {
            Some(Stmt::Expr(stmt)) => *stmt.value,
            other => panic!("expected an expression statement, got {other:?}"),
        }
    }

    #[test]
    fn needs_brackets_classifies_by_node() {
        // Self-delimiting expressions never need brackets.
        assert!(!Ast::needs_brackets(None, &parse_expr_stmt("name")));
        assert!(!Ast::needs_brackets(None, &parse_expr_stmt("f(x)")));
        assert!(!Ast::needs_brackets(None, &parse_expr_stmt("x[0]")));
        assert!(!Ast::needs_brackets(None, &parse_expr_stmt("[1, 2]")));
        assert!(!Ast::needs_brackets(None, &parse_expr_stmt("\"s\"")));
        assert!(!Ast::needs_brackets(None, &parse_expr_stmt("42")));

        // Loosely-binding expressions must be bracketed when nested. A bare
        // tuple is the case the previous per-feature heuristics disagreed on.
        assert!(Ast::needs_brackets(None, &parse_expr_stmt("1, 2")));
        assert!(Ast::needs_brackets(None, &parse_expr_stmt("a and b")));
        assert!(Ast::needs_brackets(None, &parse_expr_stmt("a + b")));
        assert!(Ast::needs_brackets(None, &parse_expr_stmt("lambda: 1")));
    }

    #[test]
    fn needs_brackets_wraps_int_literal_only_before_attribute_access() {
        // `42.bit_length()` would parse `42.` as a float, so the integer needs
        // brackets when it becomes the value of an attribute access.
        let expr = parse_expr_stmt("(42).bit_length()\n");
        let Expr::Call(call) = &expr else {
            panic!("expected a call expression");
        };
        let Expr::Attribute(attribute) = call.func.as_ref() else {
            panic!("expected an attribute access");
        };
        let int_literal = attribute.value.as_ref();
        assert!(Ast::needs_brackets(
            Some(AnyNodeRef::from(attribute)),
            int_literal
        ));
        // The same literal in any other position stands alone.
        assert!(!Ast::needs_brackets(None, int_literal));
    }
}
