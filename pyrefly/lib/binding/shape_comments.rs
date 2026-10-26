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
//! ) -> tuple[int, int]: ...  # shaped: type = [M, N]
//! ```
//!
//! - `# shapes: <names>`, on its own line above a function or class definition or
//!   between its decorators, declares names as `@shape_vars("<names>")` does.
//! - `# shaped: <shape>`, after a parameter, a return annotation, or an annotated
//!   assignment, applies the shape as `Shaped[<annotation>, "<shape>"]` does.
//! - `# shaped: type = <payload>` replaces the whole annotation, and
//!   `# shaped: .args[i].args[j] = <payload>` replaces the written type argument
//!   that the path selects.
//!
//! This module only finds the comments and the annotations they belong to, and
//! leaves their meaning to the binder. It works on the syntax of the whole module,
//! so code the binder skips as unreachable is covered too. A comment on a
//! parameter's line belongs to that parameter, unless the line also ends the
//! function's return annotation, in which case it belongs to the return
//! annotation.

use pyrefly_python::ast::Ast;
use pyrefly_util::lined_buffer::LineNumber;
use ruff_python_ast::Decorator;
use ruff_python_ast::Expr;
use ruff_python_ast::Identifier;
use ruff_python_ast::ModModule;
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

/// A `# shaped:` comment.
#[derive(Clone, Debug)]
pub struct ShapeCommentAnnotation {
    /// The range of the whole comment.
    pub range: TextRange,
    pub target: ShapeCommentTarget,
    /// The payload, with ranges inside the comment.
    pub payload: Expr,
}

/// The shape comments of a module, keyed by what they belong to.
#[derive(Default, Debug)]
pub struct ShapeComments {
    /// Keyed by the range of the declaring function or class name.
    pub declarations: SmallMap<TextRange, ShapeCommentDeclaration>,
    /// Keyed by the range of the whole written annotation, before any string
    /// annotation is parsed.
    pub annotations: SmallMap<TextRange, ShapeCommentAnnotation>,
}

enum Directive {
    Declaration(ShapeCommentDeclaration),
    Annotation(ShapeCommentAnnotation),
}

/// What could own the comment on a line.
enum Owner {
    /// A function or class definition, by the range of its name.
    Definition(TextRange),
    Annotation(TextRange),
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
                     Put each parameter or statement on its own line"
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
                (Directive::Annotation(annotation), [Owner::Annotation(target)]) => {
                    comments.annotations.insert(*target, annotation);
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
                     annotation, or an annotated assignment"
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
    let (target, payload) = if let Some(payload) = rest.strip_prefix("type").and_then(assignment) {
        (ShapeCommentTarget::Replace(Box::new([])), payload)
    } else if rest.starts_with('.') {
        let mut path = Vec::new();
        let mut selector = rest;
        while let Some(after) = selector.strip_prefix(".args[")
            && let Some((index, after)) = after.split_once(']')
            && !index.is_empty()
            && index.bytes().all(|byte| byte.is_ascii_digit())
            && let Ok(index) = index.parse()
        {
            path.push(index);
            selector = after;
        }
        let Some(payload) = assignment(selector).filter(|_| !path.is_empty()) else {
            errors
                .error_builder(
                    range,
                    ErrorKind::InvalidAnnotation,
                    "Invalid `# shaped:` selector. Write `.args[<index>]`, possibly repeated, \
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
                    Some(returns) => (returns.end(), Owner::Annotation(returns.range())),
                    None => (x.parameters.range.end(), Owner::MissingReturn),
                };
                let header_line = self.line(header_end);
                self.claim(header_line, header);
                for parameter in x.parameters.iter() {
                    let line = self.line(parameter.end());
                    if line != header_line {
                        let owner = match parameter.annotation() {
                            Some(annotation) => Owner::Annotation(annotation.range()),
                            None => Owner::UnannotatedParameter,
                        };
                        self.claim(line, owner);
                    }
                }
            }
            Stmt::ClassDef(x) => self.claim_declaration(&x.decorator_list, &x.name),
            Stmt::AnnAssign(x) => {
                let line = self.line(x.end());
                self.claim(line, Owner::Annotation(x.annotation.range()));
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

    use super::ShapeCommentTarget;
    use super::ShapeComments;
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
            .1;
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
            .1;
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
}
