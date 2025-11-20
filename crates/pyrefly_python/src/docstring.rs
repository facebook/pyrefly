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
        let normalized = docstring.replace("\r", "").replace("\t", "    ");
        let stripped = strip_literal_quotes(&normalized);
        let result = stripped.replace("\r", "").replace("\t", "    ");

        // Remove the shortest amount of whitespace from the beginning of each line
        let min_indent = result
            .lines()
            .skip(1)
            .flat_map(|line| {
                let spaces = line.bytes().take_while(|&c| c == b' ').count();
                if spaces == line.len() {
                    None
                } else {
                    Some(spaces)
                }
            })
            .min()
            .unwrap_or(0);

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

fn dedented_lines_for_parsing(docstring: &str) -> Vec<String> {
    let normalized = docstring.replace("\r", "").replace("\t", "    ");
    let stripped = strip_literal_quotes(&normalized);
    let stripped = stripped.trim_matches('\n');
    if stripped.is_empty() {
        return Vec::new();
    }

    let lines: Vec<&str> = stripped.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let min_indent = lines
        .iter()
        .filter_map(|line| {
            let trimmed = line.trim_end();
            if trimmed.is_empty() {
                None
            } else {
                Some(leading_space_count(line))
            }
        })
        .min()
        .unwrap_or(0);

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

fn leading_space_count(line: &str) -> usize {
    line.as_bytes().iter().take_while(|c| **c == b' ').count()
}

fn flush_parameter_doc(
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

fn parse_sphinx_params(lines: &[String], docs: &mut HashMap<String, String>) {
    let mut current_param = None;
    let mut current_lines = Vec::new();
    let mut base_indent = 0usize;

    for line in lines {
        let trimmed = line.trim_start();
        if trimmed.starts_with(":param") {
            flush_parameter_doc(&mut current_param, &mut current_lines, docs);

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
            flush_parameter_doc(&mut current_param, &mut current_lines, docs);
            continue;
        }

        if current_param.is_some() {
            if trimmed.is_empty() {
                flush_parameter_doc(&mut current_param, &mut current_lines, docs);
                continue;
            }
            let indent = leading_space_count(line);
            if indent > base_indent {
                current_lines.push(line.trim_start().to_owned());
            } else {
                flush_parameter_doc(&mut current_param, &mut current_lines, docs);
            }
        }
    }

    flush_parameter_doc(&mut current_param, &mut current_lines, docs);
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
            flush_parameter_doc(&mut current_param, &mut current_lines, docs);
            continue;
        }

        if trimmed.ends_with(':') {
            let header = trimmed.trim_end_matches(':');
            if is_google_section(header) {
                flush_parameter_doc(&mut current_param, &mut current_lines, docs);
                in_section = true;
                section_indent = indent;
                continue;
            }
            if in_section && indent <= section_indent {
                flush_parameter_doc(&mut current_param, &mut current_lines, docs);
                in_section = false;
            }
        } else if in_section && indent <= section_indent {
            flush_parameter_doc(&mut current_param, &mut current_lines, docs);
            in_section = false;
        }

        if !in_section {
            continue;
        }

        if indent <= section_indent {
            flush_parameter_doc(&mut current_param, &mut current_lines, docs);
            in_section = false;
            continue;
        }

        let content = line[section_indent.min(line.len())..].trim_start();
        if let Some((header, rest)) = content.split_once(':') {
            if let Some(name) = extract_google_param_name(header.trim()) {
                flush_parameter_doc(&mut current_param, &mut current_lines, docs);
                current_param = Some(name);
                current_lines.clear();
                param_indent = indent;
                let desc = rest.trim_start();
                if !desc.is_empty() {
                    current_lines.push(desc.to_owned());
                }
                continue;
            }
        }

        if current_param.is_some() {
            if indent > param_indent {
                current_lines.push(content.trim_start().to_owned());
            } else {
                flush_parameter_doc(&mut current_param, &mut current_lines, docs);
            }
        }
    }

    flush_parameter_doc(&mut current_param, &mut current_lines, docs);
}

pub fn parse_parameter_documentation(docstring: &str) -> HashMap<String, String> {
    let lines = dedented_lines_for_parsing(docstring);
    if lines.is_empty() {
        return HashMap::new();
    }

    let mut docs = HashMap::new();
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
}
