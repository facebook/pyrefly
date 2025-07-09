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
use ruff_python_ast::DictItem;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprBooleanLiteral;
use ruff_python_ast::ExprName;
use ruff_python_ast::ExprNoneLiteral;
use ruff_python_ast::Identifier;
use ruff_python_ast::ModModule;
use ruff_python_ast::Parameter;
use ruff_python_ast::ParameterWithDefault;
use ruff_python_ast::Parameters;
use ruff_python_ast::Pattern;
use ruff_python_ast::PatternMatchSingleton;
use ruff_python_ast::PySourceType;
use ruff_python_ast::Singleton;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtIf;
use ruff_python_ast::StringFlags;
use ruff_python_ast::StringLiteral;
use ruff_python_ast::visitor::source_order::SourceOrderVisitor;
use ruff_python_ast::visitor::source_order::TraversalSignal;
use ruff_python_parser::ParseError;
use ruff_python_parser::parse_expression_range;
use ruff_python_parser::parse_unchecked_source;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

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
    pub fn parse(contents: &str) -> (ModModule, Vec<ParseError>) {
        // PySourceType of Python vs Stub doesn't actually change the parsing
        let res = parse_unchecked_source(contents, PySourceType::Python);
        let errors = res.errors().to_owned();
        (res.into_syntax(), errors)
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
        // Make sure the range is precise, so that we get the right UTF8 indicies.
        // We might have a problem with \ escapes moving indicies, but if necessary we can ban those.
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
    ) -> impl Iterator<Item = (TextRange, Option<Expr>, Vec<Stmt>)> {
        let first = iter::once((x.range, Some(*x.test), x.body));
        let elses = x
            .elif_else_clauses
            .into_iter()
            .map(|x| (x.range, x.test, x.body));
        first.chain(elses)
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

    /// Calls a function on all of the names bound by this lvalue expression.
    pub fn expr_lvalue<'a>(x: &'a Expr, f: &mut impl FnMut(&'a ExprName)) {
        match x {
            Expr::Name(x) => {
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
                node_index: AtomicNodeIndex::dummy(),
                range: x.range,
            }),
            Singleton::True | Singleton::False => Expr::BooleanLiteral(ExprBooleanLiteral {
                node_index: AtomicNodeIndex::dummy(),
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
}
