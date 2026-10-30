/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A syntactic summary that associates test clients and route decorators with the module-scope
//! application constructor they refer to, without retaining the module's AST.
//!
//! The summary only describes source shapes. It makes no claim about types: callers must verify
//! that the referenced callees and receivers really are test clients and applications.

use std::collections::HashMap;
use std::collections::HashSet;
use std::slice;

use pyrefly_python::ast::Ast;
use ruff_python_ast::Arguments;
use ruff_python_ast::ExceptHandler;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprContext;
use ruff_python_ast::MatchCase;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtImport;
use ruff_python_ast::StmtImportFrom;
use ruff_python_ast::name::Name;
use ruff_python_ast::visitor::Visitor;
use ruff_python_ast::visitor::walk_except_handler;
use ruff_python_ast::visitor::walk_expr;
use ruff_python_ast::visitor::walk_match_case;
use ruff_python_ast::visitor::walk_stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

/// The value bound by a simple assignment or `with ... as` target. Every retained expression is a
/// reference: a name or a chain of attribute accesses on a name.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Declaration {
    /// A call of a reference. `application` is the single positional or `app=` argument when it is
    /// a reference and no other argument can change the client's effective URL scope.
    Call {
        callee: Expr,
        application: Option<Expr>,
    },
    Alias(Expr),
}

impl Declaration {
    pub(super) fn from_value(value: &Expr) -> Option<Self> {
        match value {
            Expr::Call(call) if is_reference(&call.func) => Some(Self::Call {
                callee: (*call.func).clone(),
                application: application(&call.arguments).cloned(),
            }),
            _ if is_reference(value) => Some(Self::Alias(value.clone())),
            _ => None,
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct SourceBindings {
    /// Declarations in every lexical scope, keyed by the range of the target identifier.
    pub declarations: HashMap<TextRange, Declaration>,
    /// Module-scope call and alias targets, mapped to the target of the module-scope call that
    /// created the object. A call target maps to itself.
    pub application_origins: HashMap<TextRange, TextRange>,
    /// Module names whose final value has an unambiguous object origin.
    pub exported_origins: HashMap<Name, TextRange>,
    /// Decorator expressions of module-scope functions whose receiver is a name with a known origin,
    /// mapped to that origin.
    pub route_owners: HashMap<TextRange, TextRange>,
}

impl SourceBindings {
    pub(super) fn from_ast(ast: &ModModule) -> Self {
        let mut collector = Collector::default();
        collector.visit_body(&ast.body);
        let mut summary = Self {
            declarations: collector.declarations,
            ..Self::default()
        };
        // Module-scope names whose current binding is known to hold the object created at the
        // mapped origin. Every statement first removes the names it may write, so a name is only
        // present while every execution path agrees on its binding.
        let mut writes = Writes::default();
        for stmt in &ast.body {
            let (targets, value) = match stmt {
                Stmt::Assign(assign) => (assign.targets.as_slice(), Some(&*assign.value)),
                Stmt::AnnAssign(assign) => {
                    (slice::from_ref(&*assign.target), assign.value.as_deref())
                }
                _ => (&[][..], None),
            };
            // The value is evaluated before any target is written.
            let origin = match value {
                Some(Expr::Call(call)) if is_reference(&call.func) => targets
                    .iter()
                    .find_map(Expr::as_name_expr)
                    .map(|name| name.range()),
                Some(Expr::Name(name)) => writes.active.get(&name.id).copied(),
                _ => None,
            };
            writes.visit_stmt(stmt);
            if let Some(origin) = origin {
                for name in targets.iter().filter_map(Expr::as_name_expr) {
                    // A function declaring the name `global` may rebind it at any time.
                    if !collector.globals.contains(&name.id) {
                        writes.active.insert(name.id.clone(), origin);
                        summary.application_origins.insert(name.range(), origin);
                    }
                }
            }
            // Writes made while evaluating the decorators and defaults have already been removed,
            // so the remaining receivers are bound as they were before the definition.
            if let Stmt::FunctionDef(function) = stmt {
                for decorator in &function.decorator_list {
                    if let Expr::Call(call) = &decorator.expression
                        && let Expr::Attribute(attribute) = &*call.func
                        && let Expr::Name(receiver) = &*attribute.value
                        && let Some(origin) = writes.active.get(&receiver.id)
                    {
                        summary
                            .route_owners
                            .insert(decorator.expression.range(), *origin);
                    }
                }
            }
        }
        summary.exported_origins = writes.active;
        summary
    }
}

/// Whether `expr` is a name or a chain of attribute accesses on a name.
fn is_reference(expr: &Expr) -> bool {
    match expr {
        Expr::Name(_) => true,
        Expr::Attribute(attribute) => is_reference(&attribute.value),
        _ => false,
    }
}

/// The application passed to a test client constructor. Calls with `**` arguments, URL scope
/// overrides, or more than one application argument have no statically known application.
fn application(arguments: &Arguments) -> Option<&Expr> {
    let mut keyword = None;
    for argument in &arguments.keywords {
        match argument.arg.as_ref().map(|arg| arg.id.as_str()) {
            None | Some("base_url" | "root_path") => return None,
            Some("app") => keyword = Some(&argument.value),
            Some(_) => {}
        }
    }
    let app = match (arguments.args.as_slice(), keyword) {
        ([app], None) | ([], Some(app)) => app,
        _ => return None,
    };
    is_reference(app).then_some(app)
}

/// Collects declarations in every scope, and the names any scope declares `global`.
#[derive(Default)]
struct Collector {
    declarations: HashMap<TextRange, Declaration>,
    globals: HashSet<Name>,
}

impl Collector {
    fn record(&mut self, target: &Expr, value: &Expr) {
        let Expr::Name(name) = target else {
            return;
        };
        let Some(declaration) = Declaration::from_value(value) else {
            return;
        };
        self.declarations.insert(name.range(), declaration);
    }
}

impl<'a> Visitor<'a> for Collector {
    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        match stmt {
            Stmt::Assign(assign) => {
                for target in &assign.targets {
                    self.record(target, &assign.value);
                }
            }
            Stmt::AnnAssign(assign) => {
                if let Some(value) = &assign.value {
                    self.record(&assign.target, value);
                }
            }
            Stmt::With(with) => {
                for item in &with.items {
                    if let Some(target) = &item.optional_vars {
                        self.record(target, &item.context_expr);
                    }
                }
            }
            Stmt::Global(global) => {
                self.globals
                    .extend(global.names.iter().map(|name| name.id.clone()));
            }
            _ => {}
        }
        walk_stmt(self, stmt);
    }
}

/// Removes every module-scope name that a module-scope statement may write, in any branch.
/// Comprehension and lambda targets are removed too, which is conservative.
#[derive(Default)]
struct Writes {
    active: HashMap<Name, TextRange>,
}

impl<'a> Visitor<'a> for Writes {
    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        match stmt {
            // Bodies run later in their own scope; only the definition-time expressions run here.
            Stmt::FunctionDef(function) => {
                self.active.remove(&function.name.id);
                for decorator in &function.decorator_list {
                    self.visit_decorator(decorator);
                }
                self.visit_parameters(&function.parameters);
                if let Some(returns) = &function.returns {
                    self.visit_annotation(returns);
                }
            }
            Stmt::ClassDef(class) => {
                self.active.remove(&class.name.id);
                for decorator in &class.decorator_list {
                    self.visit_decorator(decorator);
                }
                if let Some(arguments) = &class.arguments {
                    self.visit_arguments(arguments);
                }
            }
            Stmt::Import(StmtImport { names, .. })
            | Stmt::ImportFrom(StmtImportFrom { names, .. }) => {
                for alias in names {
                    match &alias.asname {
                        Some(asname) => {
                            self.active.remove(&asname.id);
                        }
                        None if alias.name.id == "*" => self.active.clear(),
                        None => {
                            // `import a.b` binds `a`.
                            let name = alias.name.id.as_str();
                            let bound = name.split_once('.').map_or(name, |(first, _)| first);
                            self.active.remove(&Name::new(bound));
                        }
                    }
                }
            }
            _ => walk_stmt(self, stmt),
        }
    }

    fn visit_expr(&mut self, expr: &'a Expr) {
        if let Expr::Name(name) = expr
            && name.ctx != ExprContext::Load
        {
            self.active.remove(&name.id);
        }
        walk_expr(self, expr);
    }

    fn visit_except_handler(&mut self, except_handler: &'a ExceptHandler) {
        let ExceptHandler::ExceptHandler(handler) = except_handler;
        if let Some(name) = &handler.name {
            self.active.remove(&name.id);
        }
        walk_except_handler(self, except_handler);
    }

    fn visit_match_case(&mut self, match_case: &'a MatchCase) {
        let active = &mut self.active;
        Ast::pattern_lvalue(&match_case.pattern, &mut |name| {
            active.remove(&name.id);
        });
        walk_match_case(self, match_case);
    }
}

#[cfg(test)]
mod tests {
    use pyrefly_python::ast::Ast;
    use ruff_python_ast::PySourceType;
    use ruff_text_size::Ranged;
    use ruff_text_size::TextRange;
    use ruff_text_size::TextSize;

    use super::Declaration;
    use super::SourceBindings;

    fn summary(source: &str) -> SourceBindings {
        SourceBindings::from_ast(&Ast::parse(source, PySourceType::Python).0)
    }

    /// The range of the `nth` occurrence of `needle`.
    fn range(source: &str, needle: &str, nth: usize) -> TextRange {
        let (start, _) = source
            .match_indices(needle)
            .nth(nth)
            .unwrap_or_else(|| panic!("`{needle}` should occur {} times", nth + 1));
        TextRange::at(TextSize::try_from(start).unwrap(), TextSize::of(needle))
    }

    /// The range of the identifier that starts the `nth` occurrence of `needle`.
    fn ident(source: &str, needle: &str, nth: usize) -> TextRange {
        let len = needle
            .find(|c: char| !(c.is_alphanumeric() || c == '_'))
            .unwrap_or(needle.len());
        let start = range(source, needle, nth).start();
        TextRange::at(start, TextSize::try_from(len).unwrap())
    }

    fn render(source: &str, declaration: Option<&Declaration>) -> Option<String> {
        Some(match declaration? {
            Declaration::Call {
                callee,
                application,
            } => format!(
                "{}({})",
                &source[callee.range()],
                application.as_ref().map_or("?", |app| &source[app.range()])
            ),
            Declaration::Alias(alias) => format!("={}", &source[alias.range()]),
        })
    }

    #[test]
    fn separate_applications_and_root_aliases() {
        let source = r#"
from fastapi import FastAPI
main = FastAPI()
admin = FastAPI()
api = main
first = second = FastAPI()
@api.get("/a")
def a(): ...
@admin.get("/b")
async def b(): ...
@second.get("/c")
def c(): ...
"#;
        let bindings = summary(source);
        let main = ident(source, "main = FastAPI()", 0);
        let admin = ident(source, "admin = FastAPI()", 0);
        let first = ident(source, "first = ", 0);
        assert_eq!(bindings.application_origins.get(&main), Some(&main));
        assert_eq!(
            bindings
                .application_origins
                .get(&ident(source, "api = main", 0)),
            Some(&main),
            "an alias copies the origin of its source"
        );
        assert_eq!(
            bindings
                .application_origins
                .get(&ident(source, "second = ", 0)),
            Some(&first),
            "chained targets share the origin of the first target"
        );
        let owner = |needle| bindings.route_owners.get(&range(source, needle, 0));
        assert_eq!(owner(r#"api.get("/a")"#), Some(&main));
        assert_eq!(owner(r#"admin.get("/b")"#), Some(&admin));
        assert_eq!(owner(r#"second.get("/c")"#), Some(&first));
    }

    #[test]
    fn rebinding_keeps_earlier_routes_and_isolates_later_routes() {
        let source = r#"
app = FastAPI()
alias = app
@app.get("/a")
def a(): ...
app = FastAPI()
@app.get("/b")
def b(): ...
@alias.get("/c")
def c(): ...
app = settings.app
@app.get("/d")
def d(): ...
"#;
        let bindings = summary(source);
        let first = ident(source, "app = FastAPI()", 0);
        let second = ident(source, "app = FastAPI()", 1);
        let owner = |needle| bindings.route_owners.get(&range(source, needle, 0));
        assert_eq!(owner(r#"app.get("/a")"#), Some(&first));
        assert_eq!(owner(r#"app.get("/b")"#), Some(&second));
        assert_eq!(
            owner(r#"alias.get("/c")"#),
            Some(&first),
            "an alias keeps the object it was bound to"
        );
        assert_eq!(
            owner(r#"app.get("/d")"#),
            None,
            "an attribute alias has no known origin"
        );
    }

    #[test]
    fn local_scopes_do_not_leak() {
        let source = r#"
def make():
    app = FastAPI()
    return app
@app.get("/a")
def a(): ...
main = FastAPI()
def outer():
    @main.get("/inner")
    def inner(): ...
"#;
        let bindings = summary(source);
        assert!(bindings.route_owners.is_empty());
        assert_eq!(
            bindings.application_origins.keys().collect::<Vec<_>>(),
            vec![&ident(source, "main = FastAPI()", 0)]
        );
    }

    #[test]
    fn potential_writes_invalidate_the_root_binding() {
        let invalidating = [
            "if flag:\n    app = other",
            "for app in apps:\n    pass",
            "while flag:\n    app += 1",
            "try:\n    pass\nexcept Exception as app:\n    pass",
            "match x:\n    case [*app]:\n        pass",
            "with open(p) as app:\n    pass",
            "del app",
            "import app",
            "from m import *",
            "def app(): ...",
            "class app: ...",
            "def f(x=(app := other)): ...",
            "@register(app := other)\ndef f(): ...",
            "print(app := other)",
            "app: FastAPI",
            "def reset():\n    global app\n    app = FastAPI()",
        ];
        let preserving = [
            "app = app",
            "print(app)",
            "if flag:\n    app.add_middleware(M)",
            "def f():\n    app = other",
            "class C:\n    app = other",
            "import fastapi.app",
            "from m import other as application",
        ];
        for (statement, owned) in invalidating
            .iter()
            .map(|s| (s, false))
            .chain(preserving.iter().map(|s| (s, true)))
        {
            let source = format!("app = FastAPI()\n{statement}\n@app.get('/x')\ndef x(): ...\n");
            let owner = summary(&source)
                .route_owners
                .get(&range(&source, "app.get('/x')", 0))
                .copied();
            assert_eq!(
                owner,
                owned.then(|| ident(&source, "app = FastAPI()", 0)),
                "after `{statement}`"
            );
        }
    }

    #[test]
    fn test_client_declarations() {
        let source = r#"
def test():
    client = TestClient(app)
    with testclient.TestClient(app=main.app) as managed:
        pass
    alias = client
    options = TestClient(app, raise_server_exceptions=False)
    positional = TestClient(app, "http://x")
    base = TestClient(app, base_url="http://x")
    root = TestClient(app, root_path="/api")
    kwargs = TestClient(**config)
    both = TestClient(app, app=app)
    built = TestClient(make_app())
    starred = TestClient(*apps)
    factory = make()(app)
    pair, other = TestClient(app), None
"#;
        let bindings = summary(source);
        let declaration =
            |name: &str| render(source, bindings.declarations.get(&ident(source, name, 0)));
        let expected = [
            ("client =", Some("TestClient(app)")),
            ("managed:", Some("testclient.TestClient(main.app)")),
            ("alias =", Some("=client")),
            ("options =", Some("TestClient(app)")),
            ("positional =", Some("TestClient(?)")),
            ("base =", Some("TestClient(?)")),
            ("root =", Some("TestClient(?)")),
            ("kwargs =", Some("TestClient(?)")),
            ("both =", Some("TestClient(?)")),
            ("built =", Some("TestClient(?)")),
            ("starred =", Some("TestClient(?)")),
            ("factory =", None),
            ("pair,", None),
        ];
        for (name, rendered) in expected {
            assert_eq!(
                declaration(name).as_deref(),
                rendered,
                "declaration of `{name}`"
            );
        }
        assert!(
            bindings.application_origins.is_empty(),
            "function-local calls are not module-scope origins"
        );
    }
}
