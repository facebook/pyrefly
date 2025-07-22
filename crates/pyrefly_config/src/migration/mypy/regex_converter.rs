/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use regex_syntax::hir;
use regex_syntax::hir::Hir;
use regex_syntax::hir::HirKind;
use regex_syntax::hir::Visitor;

/// Ir is the intermediate representation that the visitor uses to track the regex components it has processed.
#[derive(Debug)]
enum Ir {
    /// Represents a literal string.
    Part(String),
    /// Represents a sequence of strings that will be concatenated into one (or more) final strings.
    Concat(Vec<Ir>),
    /// Represents a choice of strings.
    Alter(Vec<Ir>),
}

impl Ir {
    /// Consumes the IR to create the list of strings.
    fn to_strings(self) -> Vec<String> {
        match self {
            // Part is easy: its string is itself.
            Self::Part(s) => vec![s],
            // For concat, the components are glued together.
            // An Alter in this Concat produces multiple strings:
            // `a(b|c)` -> Concat(a, Alter(b, c)) -> ["ab", "ac"]
            // To handle this, each Ir in the Concat is glued to each of the strings that is being built.
            Self::Concat(parts) => parts
                .into_iter()
                .map(Ir::to_strings)
                .reduce(|acc, ps| {
                    acc.iter()
                        .flat_map(|a| ps.iter().map(|p| format!("{a}{p}")).collect::<Vec<_>>())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default(),
            // Alter is also easy: each Ir is handled independently.
            Self::Alter(parts) => parts
                .into_iter()
                .flat_map(Ir::to_strings)
                .collect::<Vec<_>>(),
        }
    }
}

/// RegexConverter is a regex_syntax Visitor for turning (simple) mypy exclude regexes into pyrefly globs.
// Conversion follows simple rules:
// - directory (ends in `/`) => prepend `**/`.
//    - e.g. `foo/` -> `**/foo/` which matches `foo/`, `some/foo/`, etc.
// - file (ends in `.py`) => prepend `**/`.
//    - e.g. `foo.py` -> `**/foo.py` which matches `foo.py`, `some/foo.py`, etc.
// - ambiguous name (neither file nor directory) => prepend with **/ and append *.
//    - e.g. `foo` -> `**/foo*` which matches `some/foo.py`, `some/foo/bar`, etc.
// - all repetitions (e.g. `.*`) are turned into wildcards (`*`).
// - all other `.`s are treated as literal `.` instead of wildcards.
//    - this is to handle cases like `foo.py` where the `.` is clearly part of the file extension.
// - alternations (`|`) are split into individual items.
// Conversion proceeds by turning each component of the regex into a literal, a sequence of components, or an
// alternation of components. This results in a tree of Strings and Vec<String>s.
#[derive(Debug)]
struct RegexConverter {
    stack: Vec<Ir>,
    dot: hir::Hir,
    repetition: Option<Hir>,
}

impl RegexConverter {
    fn new() -> Self {
        let dot = hir::Hir::dot(hir::Dot::AnyChar);
        Self {
            stack: vec![],
            dot,
            repetition: None,
        }
    }

    fn push_ir(&mut self, ir: Ir) {
        match &mut self.stack.last_mut() {
            Some(Ir::Concat(v)) => v.push(ir),
            Some(Ir::Alter(v)) => v.push(ir),
            _ => self.stack.push(ir),
        }
    }
}

impl Visitor for RegexConverter {
    type Output = Vec<String>;
    type Err = anyhow::Error;

    fn finish(mut self) -> Result<Self::Output, Self::Err> {
        if self.stack.len() != 1 {
            return Err(anyhow::anyhow!(
                "Expected to find exactly one element on the stack, but found {}",
                self.stack.len()
            ));
        }
        let curr = self.stack.pop().unwrap();
        let curr = curr.to_strings();
        let globs = curr
            .iter()
            .map(|g| {
                let g = g.strip_prefix('/').unwrap_or(g);
                let mut g = format!("**/{g}");
                if !(g.ends_with('/') || g.ends_with(".py")) {
                    g.push('*');
                }
                g
            })
            .collect();
        Ok(globs)
    }

    fn visit_pre(&mut self, hir: &Hir) -> Result<(), Self::Err> {
        if self.repetition.is_some() {
            return Ok(());
        }

        match hir.kind() {
            HirKind::Empty => {}
            HirKind::Literal(lit) => {
                let slice = lit.0.as_ref();
                self.push_ir(Ir::Part(String::from_utf8_lossy(slice).to_string()));
            }
            HirKind::Look(_) => {}
            // It's common to see `.` as in `.py`. Treat it as just a `.`.
            // If it was inside a repetition, where it would actually matter, then it would have been skipped.
            HirKind::Class(_) if *hir == self.dot => self.push_ir(Ir::Part(".".to_owned())),
            // Otherwise, we're ignoring Classes for now, since they're uncommon.
            HirKind::Class(_) => {}
            // Treat all repetitions as .* => *
            HirKind::Repetition(_) => {
                self.repetition = Some(hir.clone());
                self.push_ir(Ir::Part("*".to_owned()));
            }
            // Ignore captures. We'll just process their subexpression normally.
            HirKind::Capture(_) => {}
            HirKind::Concat(_) => self.stack.push(Ir::Concat(vec![])),
            HirKind::Alternation(_) => self.stack.push(Ir::Alter(vec![])),
        }
        Ok(())
    }

    fn visit_post(&mut self, _hir: &Hir) -> Result<(), Self::Err> {
        if self.repetition.as_ref() == Some(_hir) {
            self.repetition = None;
        }
        if matches!(_hir.kind(), HirKind::Concat(_) | HirKind::Alternation(_)) {
            let top = self.stack.pop().unwrap();
            self.push_ir(top);
        }
        Ok(())
    }
}

pub fn convert(regex: &str) -> anyhow::Result<Vec<String>> {
    let mut parser = regex_syntax::ParserBuilder::new()
        // This enables `.` to match all characters, which is a small simplification that makes Hir easier to reason about.
        .dot_matches_new_line(true)
        .build();
    let h = parser.parse(regex)?;
    hir::visit(&h, RegexConverter::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Case {
        input: &'static str,
        want: Vec<&'static str>,
    }

    #[test]
    fn test_simple_regex() -> anyhow::Result<()> {
        let cases = vec![
            Case {
                input: "ambiguous",
                want: vec!["**/ambiguous*"],
            },
            Case {
                input: "dir/",
                want: vec!["**/dir/"],
            },
            Case {
                input: "unambiguous.py",
                want: vec!["**/unambiguous.py"],
            },
        ];
        for Case { input, want } in cases {
            let got = convert(input)?;
            assert_eq!(got, want, "input: {input:?}");
        }
        Ok(())
    }

    #[test]
    fn test_special_chars() -> anyhow::Result<()> {
        let cases = vec![
            Case {
                input: r"src/foo/bar\.py",
                want: vec!["**/src/foo/bar.py"],
            },
            Case {
                input: r"dev/.*\.py",
                want: vec!["**/dev/*.py"],
            },
            Case {
                input: r"^bar/",
                want: vec!["**/bar/"],
            },
            Case {
                input: r"some/file.py$",
                want: vec!["**/some/file.py"],
            },
            Case {
                input: r"^try/both.py$",
                want: vec!["**/try/both.py"],
            },
        ];
        for Case { input, want } in cases {
            let got = convert(input)?;
            assert_eq!(got, want, "input: {input:?}");
        }
        Ok(())
    }

    #[test]
    fn test_alternation() -> anyhow::Result<()> {
        let cases = vec![
            Case {
                input: "foo|bar",
                want: vec!["**/foo*", "**/bar*"],
            },
            Case {
                input: r"(src/foo/bar\.py|dev/.*\.py)",
                want: vec!["**/src/foo/bar.py", "**/dev/*.py"],
            },
        ];
        for Case { input, want } in cases {
            let got = convert(input)?;
            assert_eq!(got, want, "input: {input:?}");
        }
        Ok(())
    }

    #[test]
    fn test_x_mode() -> anyhow::Result<()> {
        let input = r"(?x)(
            alpha/
            | beta/
            | gamma.py
        )";
        let want = vec!["**/alpha/", "**/beta/", "**/gamma.py"];
        let got = convert(input)?;
        assert_eq!(got, want);
        Ok(())
    }
}
