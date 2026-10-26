/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Shape annotations written in comments, so that other type checkers see only the
//! annotations they already understand.
//!
//! A module opts in with a `# enable-shape-type-comments` comment before its first
//! statement other than a docstring. Its comments may then declare shape variables
//! and annotate types in the language of `Shaped` strings:
//!
//! ```python
//! # shapes: M, N, *Batch
//! def f(
//!     x: Tensor,  # shaped: [*Batch, M, N]
//!     n: int,  # shaped: N
//! ) -> tuple[int, int]: ...  # shaped: = [M, N]
//! ```
//!
//! - `# shapes: <names>`, on its own line above a function or class definition or
//!   between its decorators, declares names as `@shape_vars("<names>")` does.
//! - `# shaped: <shape>`, after a parameter, a return annotation, or an annotated
//!   assignment, applies the shape as `Shaped[<annotation>, "<shape>"]` does.
//! - `# shaped: = <payload>` replaces the attached type component, and
//!   `# shaped: .0.1 = <payload>` replaces the written type argument
//!   that the path selects. `type =` and `.args[i]` remain accepted aliases.
//!
//! This module only finds the comments and the annotations they belong to, and
//! leaves their meaning to the binder. It works on the syntax of the whole module,
//! so code the binder skips as unreachable is covered too. Comments can belong to
//! whole annotations or complete components within a multiline annotation. An
//! enclosing type ending on the same line takes precedence over its children.

use pyrefly_python::ast::Ast;
use pyrefly_util::lined_buffer::LineNumber;
use ruff_python_ast::Decorator;
use ruff_python_ast::Expr;
use ruff_python_ast::Identifier;
use ruff_python_ast::ModModule;
use ruff_python_ast::Operator;
use ruff_python_ast::Stmt;
use ruff_python_ast::statement_visitor::StatementVisitor;
use ruff_python_ast::statement_visitor::walk_stmt;
use ruff_python_parser::parse_expression_range;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use starlark_map::small_map::SmallMap;

use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::module::module_info::ModuleInfo;
use crate::module::parse::expression_depth_error;

const HEADER: &str = "enable-shape-type-comments";

/// Which part of an annotation a `# shaped:` comment applies to.
#[derive(Clone, Debug)]
pub enum ShapeCommentTarget {
    /// The payload is the shape string of `Shaped[<annotation>, "<payload>"]`.
    Append,
    /// The payload replaces the written type that the path of `.args[i]` indices
    /// selects. An empty path selects the whole annotation.
    Replace(Box<[usize]>),
}

/// A `# shapes:` comment.
#[derive(Clone, Debug)]
pub struct ShapeCommentDeclaration {
    /// The range of `text` in the source.
    pub range: TextRange,
    /// The declared names, in the syntax of a `@shape_vars` string.
    pub text: Box<str>,
}

/// A path through the written annotation to a complete type component.
#[derive(Clone, Copy, Debug)]
pub enum TypeComponentStep {
    Argument(usize),
    UnionLeft,
    UnionRight,
}

/// A `# shaped:` comment.
#[derive(Clone, Debug)]
pub struct ShapeCommentAnnotation {
    /// The range of the whole comment.
    pub range: TextRange,
    /// The component within the enclosing annotation that owns this comment.
    pub component: Box<[TypeComponentStep]>,
    pub target: ShapeCommentTarget,
    /// The payload, with ranges inside the comment.
    pub payload: Expr,
}

/// The shape comments of a module, keyed by what they belong to.
#[derive(Default, Debug)]
pub struct ShapeComments {
    /// Keyed by the range of the declaring function or class name.
    pub declarations: SmallMap<TextRange, ShapeCommentDeclaration>,
    /// Keyed by the whole written annotation, before string annotations are parsed.
    pub annotations: SmallMap<TextRange, Vec<ShapeCommentAnnotation>>,
}

enum Directive {
    Declaration(ShapeCommentDeclaration),
    Annotation(ShapeCommentAnnotation),
}

/// What could own the comment on a line.
enum Owner {
    /// A function or class definition, by the range of its name.
    Definition(TextRange),
    Annotation {
        annotation: TextRange,
        component: Box<[TypeComponentStep]>,
    },
    UnannotatedParameter,
    /// The header of a function with no return annotation.
    MissingReturn,
}

impl ShapeComments {
    pub fn new(module: &ModModule, info: &ModuleInfo, errors: &ErrorCollector) -> Self {
        let mut comments = Self::default();
        let contents = info.contents().as_str();
        let header_end = module
            .body
            .get(usize::from(Ast::has_docstring(module)))
            .map_or(TextSize::of(contents), |stmt| stmt.start());
        let enabled = info
            .ignore()
            .comment_ranges()
            .take_while(|range| range.start() < header_end)
            .any(|range| info.code_at(range)[1..].trim() == HEADER);
        if !enabled {
            return comments;
        }

        let mut claims = Claims {
            info,
            directives: SmallMap::new(),
            owners: SmallMap::new(),
        };
        for range in info.ignore().comment_ranges() {
            if let Some(directive) = parse_directive(range, contents, errors) {
                let line = claims.line(range.start());
                claims.directives.insert(line, (range, directive));
            }
        }
        if claims.directives.is_empty() {
            return comments;
        }
        claims.visit_body(&module.body);

        for (line, (range, directive)) in claims.directives {
            let owners = claims.owners.get(&line).map_or(&[][..], Vec::as_slice);
            let message = match (directive, owners) {
                (_, [_, _, ..]) => "This shape comment could belong to several annotations. \
                     Put each type component on its own line"
                    .to_owned(),
                (Directive::Declaration(declaration), [Owner::Definition(name)]) => {
                    if comments.declarations.contains_key(name) {
                        "Duplicate `# shapes:` comment".to_owned()
                    } else {
                        comments.declarations.insert(*name, declaration);
                        continue;
                    }
                }
                (Directive::Declaration(_), _) => "A `# shapes:` comment must be on its own line \
                     directly above a function or class definition or one of its decorators"
                    .to_owned(),
                (
                    Directive::Annotation(mut comment),
                    [
                        Owner::Annotation {
                            annotation,
                            component,
                        },
                    ],
                ) => {
                    comment.component = component.clone();
                    comments
                        .annotations
                        .entry(*annotation)
                        .or_default()
                        .push(comment);
                    continue;
                }
                (Directive::Annotation(_), [Owner::MissingReturn]) => {
                    "A `# shaped:` comment after a function header needs a return annotation"
                        .to_owned()
                }
                (Directive::Annotation(_), [Owner::UnannotatedParameter]) => {
                    "A `# shaped:` comment needs a parameter annotation to apply to".to_owned()
                }
                (Directive::Annotation(_), [] | [Owner::Definition(_)]) => {
                    "A `# shaped:` comment must end the line of a parameter, a return \
                     annotation, an annotated assignment, or a complete type component"
                        .to_owned()
                }
            };
            errors
                .error_builder(range, ErrorKind::InvalidAnnotation, message)
                .emit();
        }
        comments
    }
}

/// Parse the comment at `range`, if it is a `# shapes:` or `# shaped:` comment.
fn parse_directive(range: TextRange, contents: &str, errors: &ErrorCollector) -> Option<Directive> {
    let body = contents[range].strip_prefix('#')?.trim_start();
    // Each suffix of the comment ends where the comment does, which locates it.
    let start_of = |suffix: &str| range.end() - TextSize::of(suffix);

    if let Some(rest) = body.strip_prefix("shapes:") {
        // Names cannot contain `#`, so anything after one is a separate comment, such
        // as a suppression.
        let names = rest
            .split('#')
            .next()
            .expect("`split` yields at least one piece");
        let leading = &names[..names.len() - names.trim_start().len()];
        let text = names.trim();
        return Some(Directive::Declaration(ShapeCommentDeclaration {
            range: TextRange::at(start_of(rest) + TextSize::of(leading), TextSize::of(text)),
            text: text.into(),
        }));
    }

    let rest = body.strip_prefix("shaped:")?.trim_start();
    // An explicit form is recognized only by how the comment starts, because a payload
    // may itself contain `=`, as in the keyword arguments of a shape function call.
    let (target, payload) = if let Some(payload) =
        assignment(rest).or_else(|| rest.strip_prefix("type").and_then(assignment))
    {
        (ShapeCommentTarget::Replace(Box::new([])), payload)
    } else if rest.starts_with('.') {
        let mut path = Vec::new();
        let mut selector = rest;
        while let Some((index, after)) = selector_index(selector) {
            path.push(index);
            selector = after;
        }
        let Some(payload) = assignment(selector).filter(|_| !path.is_empty()) else {
            errors
                .error_builder(
                    range,
                    ErrorKind::InvalidAnnotation,
                    "Invalid `# shaped:` selector. Write `.<index>`, possibly repeated, \
                     followed by `= <payload>`"
                        .to_owned(),
                )
                .emit();
            return None;
        };
        (
            ShapeCommentTarget::Replace(path.into_boxed_slice()),
            payload,
        )
    } else {
        (ShapeCommentTarget::Append, rest)
    };

    // The payload runs to the end of the comment. The parser reads any later `#` that
    // is not inside a string literal as the start of a comment, such as a suppression.
    let payload = payload.trim_start();
    match parse_expression_range(contents, TextRange::new(start_of(payload), range.end())) {
        Ok(parsed) => {
            let payload = *parsed.into_syntax().body;
            if let Some(location) = expression_depth_error(&payload) {
                errors
                    .error_builder(
                        location,
                        ErrorKind::InvalidAnnotation,
                        "Shape comment payload is too deeply nested".to_owned(),
                    )
                    .emit();
                return None;
            }
            Some(Directive::Annotation(ShapeCommentAnnotation {
                range,
                component: Box::new([]),
                target,
                payload,
            }))
        }
        Err(error) => {
            errors
                .error_builder(
                    error.location,
                    ErrorKind::InvalidAnnotation,
                    format!("Could not parse `# shaped:` payload: {}", error.error),
                )
                .emit();
            None
        }
    }
}

/// Parse one zero-based type-argument selector; the long form remains an alias.
fn selector_index(text: &str) -> Option<(usize, &str)> {
    let after = text.strip_prefix('.')?;
    let (index, rest) = if let Some(after) = after.strip_prefix("args[") {
        after.split_once(']')?
    } else {
        let digits = after.bytes().take_while(u8::is_ascii_digit).count();
        after.split_at(digits)
    };
    if index.is_empty() || !index.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    Some((index.parse().ok()?, rest))
}

/// The text after a leading `=`, if `text` starts with one that is not `==`.
fn assignment(text: &str) -> Option<&str> {
    let payload = text.trim_start().strip_prefix('=')?;
    (!payload.starts_with('=')).then_some(payload)
}

/// Records, for each line with a shape comment, what could own that comment.
struct Claims<'a> {
    info: &'a ModuleInfo,
    /// Shape comments in source order. The lexer emits at most one comment per line.
    directives: SmallMap<LineNumber, (TextRange, Directive)>,
    owners: SmallMap<LineNumber, Vec<Owner>>,
}

impl Claims<'_> {
    fn line(&self, offset: TextSize) -> LineNumber {
        LineNumber::from_one_indexed(self.info.lined_buffer().line_index().line_index(offset))
    }

    fn claim(&mut self, line: LineNumber, owner: Owner) {
        if self.directives.contains_key(&line) {
            self.owners.entry(line).or_default().push(owner);
        }
    }

    /// Register only complete type components, not the head of a generic or attribute.
    /// An enclosing component ending on the same line takes precedence over its children.
    fn claim_components(
        &mut self,
        annotation: TextRange,
        expression: &Expr,
        path: &mut Vec<TypeComponentStep>,
        parent_line: Option<LineNumber>,
    ) {
        let line = self.line(expression.end());
        if Some(line) != parent_line
            && matches!(
                self.directives.get(&line),
                Some((_, Directive::Annotation(_)))
            )
        {
            self.claim(
                line,
                Owner::Annotation {
                    annotation,
                    component: path.clone().into_boxed_slice(),
                },
            );
        }
        match expression {
            Expr::Subscript(subscript) => {
                for (index, argument) in Ast::unpack_slice(&subscript.slice).iter().enumerate() {
                    path.push(TypeComponentStep::Argument(index));
                    self.claim_components(annotation, argument, path, Some(line));
                    path.pop();
                }
            }
            Expr::BinOp(binary) if binary.op == Operator::BitOr => {
                path.push(TypeComponentStep::UnionLeft);
                self.claim_components(annotation, &binary.left, path, Some(line));
                path.pop();
                path.push(TypeComponentStep::UnionRight);
                self.claim_components(annotation, &binary.right, path, Some(line));
                path.pop();
            }
            _ => {}
        }
    }

    /// A `# shapes:` comment belongs to the definition when it is on one of the
    /// comment-only lines directly above the definition's keyword or one of its
    /// decorators. Decorators and the keyword each start a line.
    fn claim_declaration(&mut self, decorators: &[Decorator], name: &Identifier) {
        for anchor in decorators.iter().map(Ranged::start).chain([name.start()]) {
            let mut line = self.line(anchor);
            while let Some(above) = line.decrement()
                && let Some(comment) = self.info.ignore().comment_range(above)
                && self.info.contents().as_str()
                    [TextRange::new(self.info.lined_buffer().line_start(above), comment.start())]
                .trim()
                .is_empty()
            {
                if matches!(
                    self.directives.get(&above),
                    Some((_, Directive::Declaration(_)))
                ) {
                    self.claim(above, Owner::Definition(name.range));
                }
                line = above;
            }
        }
    }
}

impl<'a> StatementVisitor<'a> for Claims<'_> {
    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        match stmt {
            Stmt::FunctionDef(x) => {
                self.claim_declaration(&x.decorator_list, &x.name);
                let (header_end, header) = match &x.returns {
                    Some(returns) => (
                        returns.end(),
                        Owner::Annotation {
                            annotation: returns.range(),
                            component: Box::new([]),
                        },
                    ),
                    None => (x.parameters.range.end(), Owner::MissingReturn),
                };
                let header_line = self.line(header_end);
                self.claim(header_line, header);
                if let Some(returns) = &x.returns {
                    self.claim_components(
                        returns.range(),
                        returns,
                        &mut Vec::new(),
                        Some(header_line),
                    );
                }
                for parameter in x.parameters.iter() {
                    let line = self.line(parameter.end());
                    if line != header_line {
                        let owner = match parameter.annotation() {
                            Some(annotation) => Owner::Annotation {
                                annotation: annotation.range(),
                                component: Box::new([]),
                            },
                            None => Owner::UnannotatedParameter,
                        };
                        self.claim(line, owner);
                    }
                    if let Some(annotation) = parameter.annotation() {
                        // The final parameter/default line preserves the original whole-annotation form.
                        self.claim_components(
                            annotation.range(),
                            annotation,
                            &mut Vec::new(),
                            Some(line),
                        );
                    }
                }
            }
            Stmt::ClassDef(x) => self.claim_declaration(&x.decorator_list, &x.name),
            Stmt::AnnAssign(x) => {
                let line = self.line(x.end());
                self.claim(
                    line,
                    Owner::Annotation {
                        annotation: x.annotation.range(),
                        component: Box::new([]),
                    },
                );
                self.claim_components(
                    x.annotation.range(),
                    &x.annotation,
                    &mut Vec::new(),
                    Some(line),
                );
            }
            _ => {}
        }
        walk_stmt(self, stmt);
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use pyrefly_python::ast::Ast;
    use pyrefly_python::module::Module;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use ruff_python_ast::PySourceType;
    use ruff_text_size::Ranged;

    use super::ShapeCommentTarget;
    use super::ShapeComments;
    use super::TypeComponentStep;
    use super::selector_index;
    use crate::error::collector::ErrorCollector;
    use crate::error::style::ErrorStyle;

    fn comments(source: &str) -> ShapeComments {
        let info = Module::new(
            ModuleName::from_str("comments"),
            ModulePath::filesystem(PathBuf::from("comments.pyi")),
            Arc::new(source.to_owned()),
        );
        let (module, parse_errors, _) = Ast::parse(source, PySourceType::Stub);
        assert!(
            parse_errors.is_empty(),
            "test source should parse as Python"
        );
        let errors = ErrorCollector::new(info.clone(), ErrorStyle::Never);
        ShapeComments::new(&module, &info, &errors)
    }

    #[test]
    fn test_shaped_comments_attach_declaration_and_header_tail() {
        let comments = comments(
            "# enable-shape-type-comments\n# shapes: M, *Batch\n@overload\ndef f(x: Array) -> Array: ...  # shaped: .args[0] = [M]\n",
        );
        assert_eq!(comments.declarations.len(), 1);
        assert_eq!(comments.annotations.len(), 1);
        let annotation = comments
            .annotations
            .iter()
            .next()
            .expect("return comment")
            .1
            .first()
            .expect("one return comment");
        assert!(
            matches!(&annotation.target, ShapeCommentTarget::Replace(path) if path.as_ref() == [0])
        );
    }

    #[test]
    fn test_shaped_comments_parse_keyword_payload_without_selector() {
        let comments = comments(
            "# enable-shape-type-comments\ndef f(\n    x: Array,  # shaped: helper(dims=[1], tag=\"#\")\n) -> None: ...\n",
        );
        let annotation = comments
            .annotations
            .iter()
            .next()
            .expect("parameter comment")
            .1
            .first()
            .expect("one parameter comment");
        assert!(matches!(annotation.target, ShapeCommentTarget::Append));
        let call = annotation
            .payload
            .as_call_expr()
            .expect("the whole payload is a call");
        assert_eq!(call.arguments.keywords.len(), 2);
    }

    #[test]
    fn test_shaped_comments_header_is_required() {
        let comments = comments(
            "marker = \"# enable-shape-type-comments\"\n# shapes: M\ndef f(x: int) -> int: ...  # shaped: not valid Python\n",
        );
        assert!(comments.declarations.is_empty());
        assert!(comments.annotations.is_empty());
    }

    #[test]
    fn test_shaped_comments_selectors_are_unsigned_decimal() {
        let comments = comments(
            "# enable-shape-type-comments\ndef f(x: Array) -> Array: ...  # shaped: .args[+1] = [3]\n",
        );
        assert!(comments.annotations.is_empty());
    }

    #[test]
    fn test_shaped_comments_multiple_components_share_one_annotation() {
        let comments = comments(
            "# enable-shape-type-comments\ndef f() -> tuple[\n    int,  # shaped: M\n    Array,  # shaped: [M]\n]: ...\n",
        );
        let attached = comments
            .annotations
            .iter()
            .next()
            .expect("return annotation")
            .1;
        assert_eq!(attached.len(), 2);
        assert!(matches!(
            attached[0].component.as_ref(),
            [TypeComponentStep::Argument(0)]
        ));
        assert!(matches!(
            attached[1].component.as_ref(),
            [TypeComponentStep::Argument(1)]
        ));
    }

    #[test]
    fn test_shaped_comments_compact_selector_indices() {
        assert_eq!(selector_index(".0.12 = [M]"), Some((0, ".12 = [M]")));
        assert_eq!(selector_index(".12 = [M]"), Some((12, " = [M]")));
        assert_eq!(selector_index(".args[12] = [M]"), Some((12, " = [M]")));
        for text in [
            ".-1 = M",
            ".+1 = M",
            ". = M",
            ".args[+1] = M",
            ".999999999999999999999999999999999999 = M",
        ] {
            assert_eq!(selector_index(text), None, "invalid selector: {text}");
        }
    }

    #[test]
    fn test_shaped_comments_compact_replacements_preserve_payload_positions() {
        for spelling in ["=", "type =", ".0.12 =", ".args[0].args[12] ="] {
            let source = format!(
                "# enable-shape-type-comments\ndef f(x: Array) -> Array: ...  # shaped: {spelling} helper(dims=[1], tag=\"#\")  # pyrefly: ignore\n"
            );
            let parsed = comments(&source);
            let attached = parsed
                .annotations
                .iter()
                .next()
                .expect("return annotation")
                .1;
            let comment = &attached[0];
            let expected = if spelling.starts_with('.') {
                &[0, 12][..]
            } else {
                &[][..]
            };
            assert!(
                matches!(&comment.target, ShapeCommentTarget::Replace(path) if path.as_ref() == expected)
            );
            assert_eq!(
                &source[comment.payload.range()],
                "helper(dims=[1], tag=\"#\")"
            );
        }
    }

    #[test]
    fn test_shaped_comments_compact_selectors_require_replacement() {
        for spelling in [".0 == M", ".0. = M", ".0 M", ".0[1] = M", ".0 ="] {
            let source = format!(
                "# enable-shape-type-comments\ndef f(x: Array) -> Array: ...  # shaped: {spelling}\n"
            );
            assert!(
                comments(&source).annotations.is_empty(),
                "invalid directive: {spelling}"
            );
        }
    }

    #[test]
    fn test_shaped_comments_outermost_completed_component_wins() {
        let comments = comments(
            "# enable-shape-type-comments\ndef f(\n    x: tuple[\n        Array[tuple[int, int], DType],  # shaped: .args[0] = [M, N]\n    ],\n) -> None: ...\n",
        );
        let attached = comments
            .annotations
            .iter()
            .next()
            .expect("parameter annotation")
            .1;
        assert_eq!(attached.len(), 1);
        assert!(matches!(
            attached[0].component.as_ref(),
            [TypeComponentStep::Argument(0)]
        ));
    }
}
