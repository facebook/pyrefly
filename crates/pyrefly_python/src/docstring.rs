/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::min;
use std::collections::HashMap;

use ruff_python_ast::Expr;
use ruff_python_ast::Stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::module::Module;

const STRING_LITERAL_PATTERNS: [(&str, &str); 8] = [
    ("\"\"\"", "\"\"\""),  // Multiline double quotes
    ("\'\'\'", "\'\'\'"),  // Multiline single quotes
    ("r\"\"\"", "\"\"\""), // Raw multiline double quotes
    ("r\'\'\'", "\'\'\'"), // Raw multiline single quotes
    ("\'", "\'"),          // Single quotes
    ("r\'", "\'"),         // Raw single quotes
    ("\"", "\""),          // Double quotes
    ("r\"", "\""),         // Raw double quotes
];

fn strip_literal_quotes<'a>(mut text: &'a str) -> &'a str {
    for (prefix, suffix) in STRING_LITERAL_PATTERNS {
        if let Some(x) = text.strip_prefix(prefix)
            && let Some(x) = x.strip_suffix(suffix)
        {
            text = x;
            break;
        }
    }
    text
}

#[derive(Debug, Clone)]
pub struct Docstring(pub TextRange, pub Module);

impl Docstring {
    pub fn range_from_stmts(xs: &[Stmt]) -> Option<TextRange> {
        if let Some(stmt) = xs.first()
            && let Stmt::Expr(expr_stmt) = stmt
            && let Expr::StringLiteral(_) = &*expr_stmt.value
        {
            return Some(stmt.range());
        }
        None
    }

    /// Clean a string literal ("""...""") and turn it into a docstring.
    pub fn clean(docstring: &str) -> String {
        let result = normalize_literal(docstring);

        // Remove the shortest amount of whitespace from the beginning of each line
        let min_indent = minimal_indentation(result.lines().skip(1));

        result
            .lines()
            .enumerate()
            .map(|(i, line)| {
                if i == 0 {
                    line.to_owned()
                } else {
                    let trimmed = &line[min(min_indent, line.len())..];
                    let mut without_blockquote = trimmed;
                    while let Some(rest) = without_blockquote.strip_prefix('>') {
                        without_blockquote = rest.strip_prefix(' ').unwrap_or(rest);
                    }
                    // Replace remaining leading spaces with &nbsp; or they might be ignored in markdown parsers
                    let leading_spaces = without_blockquote
                        .bytes()
                        .take_while(|&c| c == b' ')
                        .count();
                    if leading_spaces > 0 {
                        format!(
                            "{}{}",
                            "&nbsp;".repeat(leading_spaces),
                            &without_blockquote[leading_spaces..]
                        )
                    } else {
                        without_blockquote.to_owned()
                    }
                }
            })
            .collect::<Vec<_>>()
            // Note: markdown doesn't break on just `\n`
            .join("  \n")
    }

    /// Resolve the docstring to a string. This involves parsing the file to get the contents of the docstring and then cleaning it.
    pub fn resolve(&self) -> String {
        Self::clean(self.1.code_at(self.0))
    }
}

fn normalize_literal(docstring: &str) -> String {
    let normalized = docstring.replace("\r", "").replace("\t", "    ");
    let stripped = strip_literal_quotes(&normalized);
    stripped.replace("\r", "").replace("\t", "    ")
}

fn dedented_lines_for_parsing(docstring: &str) -> Vec<String> {
    let stripped = normalize_literal(docstring);
    let stripped = stripped.trim_matches('\n');
    if stripped.is_empty() {
        return Vec::new();
    }

    let lines: Vec<&str> = stripped.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let min_indent = minimal_indentation(lines.iter().copied());

    lines
        .into_iter()
        .map(|line| {
            if line.trim_end().is_empty() {
                String::new()
            } else {
                let start = min_indent.min(line.len());
                line[start..].to_owned()
            }
        })
        .collect()
}

/// Dedent a block of text while preserving blank lines, similar to how we handle docstrings.
pub fn dedent_block_preserving_layout(text: &str) -> Option<String> {
    if text.trim().is_empty() {
        return None;
    }

    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return None;
    }

    let min_indent = minimal_indentation(lines.iter().copied());
    let mut dedented = String::new();
    for line in lines {
        if line.trim().is_empty() {
            dedented.push('\n');
            continue;
        }
        let start = min_indent.min(line.len());
        dedented.push_str(&line[start..]);
        dedented.push('\n');
    }
    if !text.ends_with('\n') {
        dedented.push('\n');
    }
    Some(dedented)
}

fn leading_space_count(line: &str) -> usize {
    line.as_bytes().iter().take_while(|c| **c == b' ').count()
}

fn minimal_indentation<'a, I>(lines: I) -> usize
where
    I: Iterator<Item = &'a str>,
{
    lines
        .filter_map(|line| {
            let trimmed = line.trim_end();
            if trimmed.is_empty() {
                None
            } else {
                Some(leading_space_count(line))
            }
        })
        .min()
        .unwrap_or(0)
}

/// Persist the documentation collected so far for the current parameter.
fn commit_parameter_doc(
    current_param: &mut Option<String>,
    current_lines: &mut Vec<String>,
    docs: &mut HashMap<String, String>,
) {
    if let Some(name) = current_param.take() {
        let content = current_lines.join("\n").trim().to_owned();
        current_lines.clear();
        if !content.is_empty() {
            docs.entry(name).or_insert(content);
        }
    }
}

/// Parse [`Sphinx`](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
/// style `:param foo: description` blocks into a map of parameter docs.
fn parse_sphinx_params(lines: &[String], docs: &mut HashMap<String, String>) {
    let mut current_param = None;
    let mut current_lines = Vec::new();
    let mut base_indent = 0usize;

    for line in lines {
        let trimmed = line.trim_start();
        if trimmed.starts_with(":param") {
            commit_parameter_doc(&mut current_param, &mut current_lines, docs);

            let rest = trimmed.trim_start_matches(":param").trim_start();
            let (name_part, desc_part) = match rest.split_once(':') {
                Some(parts) => parts,
                None => continue,
            };
            let name = name_part
                .split_whitespace()
                .last()
                .unwrap_or("")
                .trim_matches(',')
                .trim_end_matches(':')
                .trim_start_matches('*')
                .trim_start_matches('*')
                .to_owned();
            if name.is_empty() {
                continue;
            }
            base_indent = leading_space_count(line);
            current_param = Some(name);
            current_lines.clear();
            let desc = desc_part.trim_start();
            if !desc.is_empty() {
                current_lines.push(desc.to_owned());
            }
            continue;
        }

        if trimmed.starts_with(':') {
            commit_parameter_doc(&mut current_param, &mut current_lines, docs);
            continue;
        }

        if current_param.is_some() {
            if trimmed.is_empty() {
                commit_parameter_doc(&mut current_param, &mut current_lines, docs);
                continue;
            }
            let indent = leading_space_count(line);
            if indent > base_indent {
                current_lines.push(line.trim_start().to_owned());
            } else {
                commit_parameter_doc(&mut current_param, &mut current_lines, docs);
            }
        }
    }

    commit_parameter_doc(&mut current_param, &mut current_lines, docs);
}

fn is_google_section(header: &str) -> bool {
    matches!(
        header,
        "Args" | "Arguments" | "Parameters" | "Keyword Args" | "Keyword Arguments"
    )
}

fn extract_google_param_name(header: &str) -> Option<String> {
    let token = header
        .split_whitespace()
        .next()
        .unwrap_or("")
        .split('(')
        .next()
        .unwrap_or("")
        .trim();
    if token.is_empty() {
        None
    } else {
        Some(token.to_owned())
    }
}

/// Parse Google-style `Args:`/`Arguments:` sections of the form:
///
/// ```text
/// Args:
///     foo (int): description
///     bar: another description
/// ```
///
/// See <https://google.github.io/styleguide/pyguide.html#383-functions-and-methods>.
fn parse_google_params(lines: &[String], docs: &mut HashMap<String, String>) {
    let mut in_section = false;
    let mut section_indent = 0usize;
    let mut current_param = None;
    let mut current_lines = Vec::new();
    let mut param_indent = 0usize;

    for line in lines {
        let indent = leading_space_count(line);
        let trimmed = line.trim();

        if trimmed.is_empty() {
            commit_parameter_doc(&mut current_param, &mut current_lines, docs);
            continue;
        }

        if trimmed.ends_with(':') {
            let header = trimmed.trim_end_matches(':');
            if is_google_section(header) {
                commit_parameter_doc(&mut current_param, &mut current_lines, docs);
                in_section = true;
                section_indent = indent;
                continue;
            }
            if in_section && indent <= section_indent {
                commit_parameter_doc(&mut current_param, &mut current_lines, docs);
                in_section = false;
            }
        } else if in_section && indent <= section_indent {
            commit_parameter_doc(&mut current_param, &mut current_lines, docs);
            in_section = false;
        }

        if !in_section {
            continue;
        }

        if indent <= section_indent {
            commit_parameter_doc(&mut current_param, &mut current_lines, docs);
            in_section = false;
            continue;
        }

        let content = line[section_indent.min(line.len())..].trim_start();
        if let Some((header, rest)) = content.split_once(':')
            && let Some(name) = extract_google_param_name(header.trim())
        {
            commit_parameter_doc(&mut current_param, &mut current_lines, docs);
            current_param = Some(name);
            current_lines.clear();
            param_indent = indent;
            let desc = rest.trim_start();
            if !desc.is_empty() {
                current_lines.push(desc.to_owned());
            }
            continue;
        }

        if current_param.is_some() {
            if indent > param_indent {
                current_lines.push(content.trim_start().to_owned());
            } else {
                commit_parameter_doc(&mut current_param, &mut current_lines, docs);
            }
        }
    }

    commit_parameter_doc(&mut current_param, &mut current_lines, docs);
}

/// Extract a map of `parameter -> markdown` documentation snippets from the
/// supplied docstring, supporting both Sphinx (`:param foo:`) and
/// Google-style (`Args:`) formats.
pub fn parse_parameter_documentation(docstring: &str) -> HashMap<String, String> {
    let lines = dedented_lines_for_parsing(docstring);
    let mut docs = HashMap::new();
    if lines.is_empty() {
        return docs;
    }
    parse_sphinx_params(&lines, &mut docs);
    parse_google_params(&lines, &mut docs);
    docs
}

#[cfg(test)]
mod tests {
    use crate::docstring::Docstring;
    use crate::docstring::parse_parameter_documentation;

    #[test]
    fn test_clean_removes_double_multiline_double_quotes() {
        assert_eq!(
            Docstring::clean("\"\"\"test docstring\"\"\"").as_str(),
            "test docstring"
        );
    }
    #[test]
    fn test_clean_removes_multiline_single_quotes() {
        assert_eq!(
            Docstring::clean("\"\"\"test docstring\"\"\"").as_str(),
            "test docstring"
        );
    }

    #[test]
    fn test_clean_removes_single_quotes() {
        assert_eq!(
            Docstring::clean("\'test docstring\'").as_str(),
            "test docstring"
        );
    }

    #[test]
    fn test_clean_removes_raw_multiline_double_quotes() {
        assert_eq!(
            Docstring::clean("r\"\"\"test docstring\"\"\"").as_str(),
            "test docstring"
        );
    }

    #[test]
    fn test_clean_removes_raw_multiline_single_quotes() {
        assert_eq!(
            Docstring::clean("r\"\"\"test docstring\"\"\"").as_str(),
            "test docstring"
        );
    }

    #[test]
    fn test_clean_removes_double_quotes() {
        assert_eq!(
            Docstring::clean("\"test docstring\"").as_str(),
            "test docstring"
        );
    }

    #[test]
    fn test_clean_removes_carriage_returns() {
        assert_eq!(Docstring::clean("hello\rworld").as_str(), "helloworld");
    }

    #[test]
    fn test_clean_replaces_tabs_with_spaces() {
        assert_eq!(Docstring::clean("hello\tworld").as_str(), "hello    world");
    }

    #[test]
    fn test_clean_trims_shortest_whitespace_and_replaces_space_with_nbsp() {
        assert_eq!(
            Docstring::clean("\n  hello\n    world\n  test").as_str(),
            "  \nhello  \n&nbsp;&nbsp;world  \ntest"
        );
    }

    #[test]
    fn test_docstring_panic() {
        Docstring::clean(" F\n\u{85}");
    }

    #[test]
    fn test_docstring_multiline_starts_at_first() {
        assert_eq!(
            Docstring::clean("\"\"\"hello\n  world\n  test\"\"\"").as_str(),
            "hello  \nworld  \ntest"
        );
    }
    #[test]
    fn test_parse_sphinx_param_docs() {
        let doc = r#"
:param foo: first line
    second line
:param str bar: another
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "first line\nsecond line");
        assert_eq!(docs.get("bar").unwrap(), "another");
    }

    #[test]
    fn test_parse_google_param_docs() {
        let doc = r#"
Args:
    foo (int): first line
        second line
    bar: final
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "first line\nsecond line");
        assert_eq!(docs.get("bar").unwrap(), "final");
    }

    #[test]
    fn test_parse_sphinx_empty_param() {
        let doc = r#"
:param foo:
:param bar: has description
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo"), None); // Empty params should not be included
        assert_eq!(docs.get("bar").unwrap(), "has description");
    }

    #[test]
    fn test_parse_sphinx_with_type_annotations() {
        let doc = r#"
:param int foo: an integer
:param str bar: a string
:param Optional[Dict[str, int]] baz: complex type
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "an integer");
        assert_eq!(docs.get("bar").unwrap(), "a string");
        assert_eq!(docs.get("baz").unwrap(), "complex type");
    }

    #[test]
    fn test_parse_sphinx_multiple_continuation_lines() {
        let doc = r#"
:param foo: line one
    line two
    line three
    line four
:param bar: single line
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(
            docs.get("foo").unwrap(),
            "line one\nline two\nline three\nline four"
        );
        assert_eq!(docs.get("bar").unwrap(), "single line");
    }

    #[test]
    fn test_parse_sphinx_with_other_directives() {
        let doc = r#"
:param foo: the foo parameter
:param bar: the bar parameter
:return: the return value
:raises ValueError: when invalid
:type foo: int
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "the foo parameter");
        assert_eq!(docs.get("bar").unwrap(), "the bar parameter");
        // Other directives should not be included as parameters
        assert_eq!(docs.get("return"), None);
        assert_eq!(docs.get("raises"), None);
    }

    #[test]
    fn test_parse_sphinx_with_varargs() {
        let doc = r#"
:param *args: positional arguments
:param **kwargs: keyword arguments
:param regular: regular param
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("args").unwrap(), "positional arguments");
        assert_eq!(docs.get("kwargs").unwrap(), "keyword arguments");
        assert_eq!(docs.get("regular").unwrap(), "regular param");
    }

    #[test]
    fn test_parse_sphinx_indented_in_docstring() {
        let doc = r#"""
    Summary of function.

    :param foo: first param
        with continuation
    :param bar: second param
    """#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "first param\nwith continuation");
        assert_eq!(docs.get("bar").unwrap(), "second param");
    }

    #[test]
    fn test_parse_google_different_headers() {
        let doc = r#"
Arguments:
    foo: using Arguments header
    bar: second arg

def another_func():
    """
    Keyword Arguments:
        baz: keyword arg
    """
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "using Arguments header");
        assert_eq!(docs.get("bar").unwrap(), "second arg");
    }

    #[test]
    fn test_parse_google_parameters_header() {
        let doc = r#"
Parameters:
    foo (int): first param
    bar (str): second param
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "first param");
        assert_eq!(docs.get("bar").unwrap(), "second param");
    }

    #[test]
    fn test_parse_google_keyword_args_header() {
        let doc = r#"
Keyword Args:
    foo: keyword arg one
    bar: keyword arg two
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "keyword arg one");
        assert_eq!(docs.get("bar").unwrap(), "keyword arg two");
    }

    #[test]
    fn test_parse_google_deeply_indented_continuation() {
        let doc = r#"
Args:
    foo: first line
        second line
            third line deeply indented
        fourth line
    bar: simple
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(
            docs.get("foo").unwrap(),
            "first line\nsecond line\nthird line deeply indented\nfourth line"
        );
        assert_eq!(docs.get("bar").unwrap(), "simple");
    }

    #[test]
    fn test_parse_google_no_type_annotation() {
        let doc = r#"
Args:
    foo: no type annotation
    bar (int): with type annotation
    baz: also no type
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "no type annotation");
        assert_eq!(docs.get("bar").unwrap(), "with type annotation");
        assert_eq!(docs.get("baz").unwrap(), "also no type");
    }

    #[test]
    fn test_parse_google_complex_type_annotations() {
        let doc = r#"
Args:
    foo (Optional[List[Dict[str, int]]]): complex type
    bar (Callable[[int, str], bool]): callable type
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "complex type");
        assert_eq!(docs.get("bar").unwrap(), "callable type");
    }

    #[test]
    fn test_parse_google_section_ends_with_other_section() {
        let doc = r#"
Args:
    foo: the foo param
    bar: the bar param

Returns:
    int: the return value

Raises:
    ValueError: when things go wrong
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "the foo param");
        assert_eq!(docs.get("bar").unwrap(), "the bar param");
        // Returns and Raises should not be parsed as params
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn test_parse_google_empty_parameter_description() {
        let doc = r#"
Args:
    foo:
    bar: has description
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo"), None); // Empty description should not be included
        assert_eq!(docs.get("bar").unwrap(), "has description");
    }

    #[test]
    fn test_parse_mixed_sphinx_and_google() {
        let doc = r#"
:param sphinx_param: using Sphinx style
    with continuation

Args:
    google_param: using Google style
    another_google: second Google param
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(
            docs.get("sphinx_param").unwrap(),
            "using Sphinx style\nwith continuation"
        );
        assert_eq!(docs.get("google_param").unwrap(), "using Google style");
        assert_eq!(docs.get("another_google").unwrap(), "second Google param");
    }

    #[test]
    fn test_parse_sphinx_param_with_comma_in_type() {
        let doc = r#"
:param Dict[str, int] foo: dict param
:param Tuple[int, str, bool] bar: tuple param
"#;
        let docs = parse_parameter_documentation(doc);
        assert_eq!(docs.get("foo").unwrap(), "dict param");
        assert_eq!(docs.get("bar").unwrap(), "tuple param");
    }
}
