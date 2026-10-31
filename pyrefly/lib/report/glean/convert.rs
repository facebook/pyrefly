/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::collections::HashSet;
use std::env::current_dir;
use std::iter;
use std::ops::Sub;
use std::path::Path;
use std::slice;
use std::sync::Arc;

use dupe::Dupe;
use num_traits::ToPrimitive;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::docstring::Docstring;
use pyrefly_python::dunder;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_util::visit::Visit;
use regex::RegexBuilder;
use ruff_python_ast::Alias;
use ruff_python_ast::Comprehension;
use ruff_python_ast::Decorator;
use ruff_python_ast::ExceptHandler;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprName;
use ruff_python_ast::ExprNamed;
use ruff_python_ast::ExprStringLiteral;
use ruff_python_ast::Identifier;
use ruff_python_ast::Parameter;
use ruff_python_ast::ParameterWithDefault;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::StmtImport;
use ruff_python_ast::StmtImportFrom;
use ruff_python_ast::name::Name;
use ruff_python_ast::visitor::source_order::SourceOrderVisitor;
use ruff_python_ast::visitor::source_order::walk_comprehension;
use ruff_python_ast::visitor::source_order::walk_expr;
use ruff_python_ast::visitor::source_order::walk_stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use starlark_map::small_set::SmallSet;
use thin_vec::ThinVec;
use vec1::Vec1;

use crate::module::module_info::ModuleInfo;
use crate::report::glean::facts::*;
use crate::report::glean::schema::*;
use crate::state::lsp::FindDefinitionItemWithDocstring;
use crate::state::lsp::FindPreference;
use crate::state::state::Transaction;
use crate::types::types::Type;

const TYPE_SEPARATORS: [char; 12] = [',', '|', '[', ']', '{', '}', '(', ')', '=', ':', '\'', '"'];

#[derive(Clone, Debug)]
pub struct DefinitionLocation {
    pub name: String,
    pub file: Option<src::File>,
}

fn hash(x: &[u8]) -> String {
    // Glean uses blake3
    blake3::hash(x).to_string()
}

pub(crate) fn join_names(base_name: &str, name: &str) -> String {
    if base_name.is_empty() {
        name.to_owned()
    } else if name.is_empty() {
        base_name.to_owned()
    } else {
        base_name.to_owned() + "." + name
    }
}

/// Compute the scope prefix for a Glean qualified name.
///
/// This is the single source of truth for how Glean scopes are determined
/// from the container hierarchy. The full qualified name is
/// `join_names(&scope, declaration_name)`.
pub(crate) fn compute_scope(
    container_fq_name: &str,
    is_function_container: bool,
    scope_type: &ScopeType,
    module_name: &str,
) -> String {
    match scope_type {
        ScopeType::Global => module_name.to_owned(),
        ScopeType::Nonlocal => {
            let mut parts: Vec<&str> = container_fq_name.split('.').collect();
            parts.pop();
            parts.join(".")
        }
        ScopeType::Local => {
            if is_function_container {
                container_fq_name.to_owned() + ".<locals>"
            } else {
                container_fq_name.to_owned()
            }
        }
    }
}

fn find_preference_glean() -> FindPreference {
    FindPreference {
        prefer_pyi: false, // Similar to Pyrefly behavior, we prefer py over pyi files
        ..Default::default()
    }
}

fn all_modules_with_range(
    module_name: ModuleName,
    position: TextSize,
) -> impl Iterator<Item = (String, TextRange)> {
    module_name.components().into_iter().scan(
        ("".to_owned(), position),
        |(module, start), component| {
            let name = component.as_str();
            let range = TextRange::at(*start, TextSize::try_from(name.len()).unwrap());
            *module = join_names(module, name);
            *start = range.end() + TextSize::from(1);
            Some((module.to_owned(), range))
        },
    )
}

fn range_without_decorators(range: TextRange, decorators: &[Decorator]) -> TextRange {
    let Some(last) = decorators.last() else {
        return range;
    };
    let new_start = (last.range().end() + TextSize::from(1)).min(range.end());
    TextRange::new(new_start, range.end())
}

fn to_span(range: TextRange) -> src::ByteSpan {
    src::ByteSpan {
        start: range.start().to_u32().into(),
        length: range.len().to_u32().into(),
    }
}

/// Create a Glean file fact from module info, using forward slashes for
/// cross-platform consistency regardless of the OS path separator.
/// Symlinks are resolved so the same physical file always gets the same key.
fn file_fact(module_info: &ModuleInfo) -> src::File {
    let file_path = module_info.path().as_path();
    let cwd = current_dir().unwrap_or_default();
    let (resolved_path, resolved_cwd) = match (
        std::fs::canonicalize(file_path),
        std::fs::canonicalize(&cwd),
    ) {
        (Ok(p), Ok(c)) => (p, c),
        _ => (file_path.to_path_buf(), cwd),
    };
    let relative_path = resolved_path
        .strip_prefix(&resolved_cwd)
        .unwrap_or(&resolved_path)
        .to_str()
        .unwrap();

    // Normalize to forward slashes so Glean keys are consistent across platforms.
    src::File::new(relative_path.replace('\\', "/"))
}

/// Inputs beyond Pyrefly's own import resolution that decide the names and files the Glean
/// report records.
#[derive(Default)]
pub struct GleanNaming {
    /// Module names by repository-relative file path (`--glean-module-names`).
    pub module_names: HashMap<String, String>,
    /// Repository-relative directory holding the stdlib sources that Pyrefly's bundled typeshed
    /// stdlib stubs describe (`--glean-stdlib-root`), e.g. `third-party/python/3.12/Lib`.
    pub stdlib_root: Option<String>,
}

/// The file fact of the module that defines a cross-module reference target, or `None` if that
/// module is not a source file of the repository. Glean only indexes source files, so such a file
/// would never have facts of its own, and code navigation matches the target by name alone. The
/// target name is kept either way. A module is not a source file when it is:
/// - a Buck build output under `buck-out/` (e.g. a generated Thrift module) or a file outside
///   the working directory;
/// - a namespace package, whose path is a directory;
/// - a stub bundled with Pyrefly. A bundled stdlib stub instead takes the path of the stdlib
///   source it describes under `stdlib_root`, e.g. `logging/__init__.pyi` becomes
///   `<stdlib_root>/logging/__init__.py`, where the stdlib's own declarations are indexed.
fn definition_file_fact(module_info: &ModuleInfo, stdlib_root: Option<&str>) -> Option<src::File> {
    match module_info.path().details() {
        ModulePathDetails::FileSystem(_) | ModulePathDetails::Memory(_) => {
            let file = file_fact(module_info);
            let is_source_file =
                !file.key.starts_with("buck-out/") && !Path::new(&*file.key).is_absolute();
            is_source_file.then_some(file)
        }
        ModulePathDetails::BundledTypeshed(relative_path) => {
            stdlib_root.map(|root| stdlib_source_file(root, relative_path))
        }
        ModulePathDetails::Namespace(_)
        | ModulePathDetails::BundledTypeshedThirdParty(_)
        | ModulePathDetails::BundledThirdParty(_) => None,
    }
}

/// The stdlib source file under `root` that the bundled typeshed stub at `stub_path` describes.
/// Typeshed and CPython sometimes lay a module out differently: typeshed has `os/__init__.pyi`
/// for CPython's `os.py`, and `re.pyi` for `re/__init__.py`. So both layouts are tried, the
/// stub's own first, and the first that exists in the working directory is used.
fn stdlib_source_file(root: &str, stub_path: &Path) -> src::File {
    let module_path = stub_path.with_extension("");
    let candidates = if module_path
        .file_name()
        .is_some_and(|name| name == "__init__")
    {
        let package = module_path.parent().unwrap_or(Path::new(""));
        [
            module_path.with_extension("py"),
            package.with_extension("py"),
        ]
    } else {
        [
            module_path.with_extension("py"),
            module_path.join("__init__.py"),
        ]
    };
    let source = candidates
        .iter()
        .find(|candidate| Path::new(root).join(candidate).exists())
        .unwrap_or(&candidates[0]);
    src::File::new(format!(
        "{root}/{}",
        source.to_string_lossy().replace('\\', "/")
    ))
}

fn gather_nonlocal_variables(body: &[Stmt]) -> (Arc<SmallSet<Name>>, Arc<SmallSet<Name>>) {
    let mut globals = SmallSet::new();
    let mut nonlocals = SmallSet::new();
    for stmt in body {
        match stmt {
            Stmt::Global(stmt_global) => {
                globals.extend(stmt_global.names.iter().map(|name| name.id.clone()))
            }
            Stmt::Nonlocal(stmt_nonlocal) => {
                nonlocals.extend(stmt_nonlocal.names.iter().map(|name| name.id.clone()))
            }
            _ => {}
        }
    }

    (Arc::new(globals), Arc::new(nonlocals))
}

/// The expressions in a statement that bind names: walrus (`:=`) expressions, and the `for`
/// clauses of comprehensions and generator expressions.
#[derive(Default)]
struct ExprBindings<'a> {
    named_exprs: Vec<&'a ExprNamed>,
    comprehensions: Vec<&'a Comprehension>,
}

impl<'a> SourceOrderVisitor<'a> for ExprBindings<'a> {
    fn visit_stmt(&mut self, _: &'a Stmt) {}

    fn visit_expr(&mut self, expr: &'a Expr) {
        if let Expr::Named(named) = expr {
            self.named_exprs.push(named);
        }
        walk_expr(self, expr);
    }

    fn visit_comprehension(&mut self, comprehension: &'a Comprehension) {
        self.comprehensions.push(comprehension);
        walk_comprehension(self, comprehension);
    }
}

/// The name-binding expressions that `stmt` evaluates itself. Those in nested statement
/// bodies are excluded, because they belong to the nested statements, which are processed
/// with their own context.
fn expr_bindings(stmt: &Stmt) -> ExprBindings<'_> {
    let mut bindings = ExprBindings::default();
    walk_stmt(&mut bindings, stmt);
    bindings
}

/// The names that assigning to `target` binds, e.g. `a` and `b` in `a, *b = ...`. Assigning
/// to an attribute or a subscript binds no name.
fn bound_names(target: &Expr) -> Vec<&ExprName> {
    match target {
        Expr::Name(name) => vec![name],
        Expr::Tuple(tuple) => tuple.elts.iter().flat_map(bound_names).collect(),
        Expr::List(list) => list.elts.iter().flat_map(bound_names).collect(),
        Expr::Starred(starred) => bound_names(&starred.value),
        _ => vec![],
    }
}

fn create_sname(name: &str) -> python::SName {
    let parts = name.split(".");
    let mut parent = None;

    for local_name in parts {
        let local_name_fact = python::Name::new(local_name.to_owned());
        let sname = python::SName::new(local_name_fact, parent);
        parent = Some(sname);
    }

    parent.unwrap()
}
pub(crate) enum ScopeType {
    Global,
    Nonlocal,
    Local,
}
struct DeclarationInfo {
    declaration: python::Declaration,
    decl_span: src::ByteSpan,
    definition: Option<python::Definition>,
    def_span: Option<src::ByteSpan>,
    top_level_decl: python::Declaration,
    docstring_range: Option<TextRange>,
}

/// A name that an import statement binds.
struct ImportBinding {
    /// The fully qualified name of what is imported, e.g. `os.path` in `import os.path as p`
    /// or `os.sep` in `from os import sep`, at the range of the imported name in the statement.
    from_name: Identifier,
    /// The name bound in the statement's scope, e.g. `p` or `sep`.
    as_name: Identifier,
    resolved: DefinitionLocation,
    /// The file of the module imported from, for `from ... import` statements.
    from_file: Option<src::File>,
}

struct Facts {
    file: src::File,
    module: python::Module,
    modules: Vec<python::Module>,
    decl_locations: Vec<python::DeclarationLocation>,
    def_locations: Vec<python::DefinitionLocation>,
    import_star_locations: Vec<python::ImportStarLocation>,
    file_calls: Vec<python::FileCall>,
    callee_to_callers: Vec<python::CalleeToCaller>,
    containing_top_level_declarations: Vec<python::ContainingTopLevelDeclaration>,
    xrefs_via_name: Vec<python::XRefViaName>,
    xrefs_via_name_by_target: HashMap<python::Name, Vec<src::ByteSpan>>,
    xrefs: Vec<python_xrefs::XRef>,
    declaration_docstrings: Vec<python::DeclarationDocstring>,
    name_to_sname: Vec<python::NameToSName>,
}

#[derive(Clone)]
struct NodeContext {
    container: Arc<python::DeclarationContainer>,
    top_level_decl: Arc<python::Declaration>,
    globals: Arc<SmallSet<Name>>,
    nonlocals: Arc<SmallSet<Name>>,
}

struct GleanState<'a> {
    transaction: &'a Transaction<'a>,
    handle: &'a Handle,
    module: ModuleInfo,
    /// The module name written to Glean for this file: its entry in `naming.module_names` if any,
    /// otherwise `module.name()`. Pyrefly itself keeps using `module.name()` to resolve
    /// imports, so the two differ for files in Buck targets with a `base_module`.
    emitted_module_name: ModuleName,
    naming: &'a GleanNaming,
    facts: Facts,
    names: HashSet<Arc<String>>,
    locations_fqnames: HashMap<TextSize, Arc<String>>,
    import_names: HashMap<
        Arc<String>,
        (
            Arc<String>,
            Option<Arc<String>>,
            Option<src::File>,
            Option<src::File>,
        ),
    >,
    /// The local scope of the statement being processed, e.g. `m.f.<locals>` for a statement
    /// in the body of function `m.f`, `m.C` in the body of class `m.C`, or `m` at the top
    /// level of module `m`. Imports in the statement bind names in this scope.
    statement_scope: String,
}

struct AssignInfo<'a> {
    range: TextRange,
    type_info: Option<python::TypeInfo>,
    value: Option<&'a Expr>,
}

impl Facts {
    fn new(file: src::File, module: python::Module) -> Facts {
        Facts {
            file,
            module,
            modules: vec![],
            decl_locations: vec![],
            def_locations: vec![],
            import_star_locations: vec![],
            file_calls: vec![],
            callee_to_callers: vec![],
            containing_top_level_declarations: vec![],
            xrefs_via_name: vec![],
            xrefs: vec![],
            xrefs_via_name_by_target: HashMap::new(),
            declaration_docstrings: vec![],
            name_to_sname: vec![],
        }
    }
}

impl GleanState<'_> {
    fn new<'a>(
        transaction: &'a Transaction<'a>,
        handle: &'a Handle,
        naming: &'a GleanNaming,
    ) -> GleanState<'a> {
        let module_info = &transaction.get_module_info(handle).unwrap();
        let file = file_fact(module_info);
        let module_name = naming
            .module_names
            .get(file.key.as_str())
            .map_or(module_info.name(), |name| ModuleName::from_str(name));
        GleanState {
            transaction,
            handle,
            module: module_info.clone(),
            emitted_module_name: module_name,
            naming,
            facts: Facts::new(
                file,
                python::Module::new(python::Name::new(module_name.to_string())),
            ),
            names: HashSet::new(),
            locations_fqnames: HashMap::new(),
            import_names: HashMap::new(),
            statement_scope: module_info.name().to_string(),
        }
    }

    fn module_fact(&self) -> python::Module {
        self.facts.module.clone()
    }

    fn file_fact(&self) -> src::File {
        self.facts.file.clone()
    }

    fn digest_fact(&self) -> digest::FileDigest {
        let digest = digest::Digest {
            hash: hash(self.module.contents().as_bytes()),
            size: self.module.contents().len() as u64,
        };
        digest::FileDigest::new(self.file_fact(), digest)
    }

    fn gencode_fact(&mut self) -> Option<gencode::GenCode> {
        let generated_pattern = RegexBuilder::new(
            r"^.*@(?P<tag>(partially-)?generated)( (?:SignedSource<<(?P<sign>[0-9a-f]+)>>|<<SignedSource::[^>]+>>))?$",
        )
        .multi_line(true)
        .build()
        .unwrap();

        let codegen_pattern = RegexBuilder::new(
            r"^.*@codegen-(?P<key>(command|class|source))\s*.*?\s+(?P<value>[A-Za-z0-9_\/\.\-\\\\]+)\n?$",
        )
        .multi_line(true)
        .build()
        .unwrap();

        let contents = self.module.contents();
        let generated_tag_match = generated_pattern.captures(contents);
        generated_tag_match.map(|tag_match| {
            let codegen_details = codegen_pattern
                .captures_iter(contents)
                .filter_map(|codegen_match| {
                    codegen_match.name("key").and_then(|key| {
                        codegen_match
                            .name("value")
                            .map(|value| (key.as_str().to_owned(), value.as_str().to_owned()))
                    })
                })
                .collect::<HashMap<String, String>>();

            let tag = tag_match.name("tag").map(|x| x.as_str());

            let variant = if Some(&"generated") == tag.as_ref() {
                gencode::GenCodeVariant::Full
            } else {
                gencode::GenCodeVariant::Partial
            };

            let signature = tag_match
                .name("sign")
                .map(|x| gencode::GenCodeSignature::new(x.as_str().to_owned()));

            let source = codegen_details
                .get("source")
                .map(|x| src::File::new(x.to_owned()));

            let class_ = codegen_details
                .get("class")
                .map(|x| gencode::GenCodeClass::new(x.to_owned()));

            let command = codegen_details
                .get("command")
                .map(|x| gencode::GenCodeCommand::new(x.to_owned()));

            gencode::GenCode::new(
                self.file_fact(),
                variant,
                source,
                command,
                class_,
                signature,
            )
        })
    }

    fn module_facts(&mut self, range: TextRange) {
        let module_docstring_range = self.transaction.get_module_docstring_range(self.handle);

        for (name, _) in all_modules_with_range(self.emitted_module_name, TextSize::ZERO) {
            self.record_name(name.clone());
            self.facts
                .modules
                .push(python::Module::new(python::Name::new(name)));
        }

        let mod_decl_info = DeclarationInfo {
            declaration: python::Declaration::module(self.module_fact()),
            decl_span: to_span(range),
            definition: Some(python::Definition::module(python::ModuleDefinition::new(
                self.module_fact(),
            ))),
            def_span: Some(to_span(range)),
            top_level_decl: python::Declaration::module(self.module_fact()),
            docstring_range: module_docstring_range,
        };

        self.declaration_facts(mod_decl_info);
    }

    fn file_lines_fact(&self) -> src::FileLines {
        let lined_buffer = self.module.lined_buffer();
        let lens: Vec<u64> = lined_buffer
            .lines()
            .map(|x| x.len().to_u64().unwrap() + 1)
            .collect();
        let ends_in_new_line = lens.len() < lined_buffer.line_count();
        src::FileLines::new(
            self.facts.file.clone(),
            lens,
            ends_in_new_line,
            !lined_buffer.contents().is_ascii() || lined_buffer.contents().contains('\t'),
        )
    }

    fn declaration_facts(&mut self, decl_info: DeclarationInfo) {
        self.facts.containing_top_level_declarations.push(
            python::ContainingTopLevelDeclaration::new(
                decl_info.declaration.clone(),
                decl_info.top_level_decl,
            ),
        );

        self.facts
            .decl_locations
            .push(python::DeclarationLocation::new(
                decl_info.declaration.clone(),
                self.facts.file.clone(),
                decl_info.decl_span,
            ));
        if let Some(def_info) = decl_info.definition {
            self.facts
                .def_locations
                .push(python::DefinitionLocation::new(
                    def_info,
                    self.facts.file.clone(),
                    decl_info.def_span.unwrap(),
                ));
        }

        if let Some(docstring_range) = decl_info.docstring_range {
            let docstring = Docstring::clean(self.module.code_at(docstring_range));
            self.facts
                .declaration_docstrings
                .push(python::DeclarationDocstring::new(
                    decl_info.declaration.clone(),
                    to_span(docstring_range),
                    docstring.trim().to_owned(),
                ));
        }
    }

    fn record_name(&mut self, name: String) -> Arc<String> {
        let arc_name = Arc::new(name.clone());
        if self.names.insert(arc_name.dupe()) {
            self.facts.name_to_sname.push(python::NameToSName::new(
                python::Name::new(name.clone()),
                create_sname(&name),
            ));
        }
        arc_name
    }

    /// Record `name` as declared in `scope`, so that its definition position maps to its fully
    /// qualified name.
    fn declare(&mut self, name: &Identifier, scope: String) {
        let fqname = self.record_name(join_names(&scope, name));
        self.record_name(scope);
        self.locations_fqnames.insert(name.range.start(), fqname);
    }

    /// The fully qualified name of the declaration at `name`, which the first pass recorded.
    fn declared_name(&self, name: impl Ranged) -> python::Name {
        let fqname = self
            .locations_fqnames
            .get(&name.start())
            .expect("every declaration should be recorded by the first pass");
        python::Name::new((**fqname).clone())
    }

    /// Record the name that `binding` binds in the current statement's scope.
    fn record_import(&mut self, binding: ImportBinding) {
        self.record_name(self.statement_scope.clone());
        let arc_name = self.record_name(join_names(&self.statement_scope, &binding.as_name));
        let ImportBinding {
            from_name,
            resolved,
            from_file,
            ..
        } = binding;
        let arc_resolved_name = self.record_name(resolved.name.clone());
        let (arc_from_name, from_file) = if from_name.id().as_str() != resolved.name {
            (
                Some(self.record_name(from_name.id().to_string())),
                from_file,
            )
        } else {
            (None, None)
        };
        self.import_names.insert(
            arc_name,
            (arc_resolved_name, arc_from_name, resolved.file, from_file),
        );
    }

    fn scope_in(&self, container: &python::DeclarationContainer, scope_type: &ScopeType) -> String {
        let (container_fq_name, is_function_container) = match container {
            python::DeclarationContainer::module(module) => (module.key.name.key.as_str(), false),
            python::DeclarationContainer::cls(cls) => (cls.key.name.key.as_str(), false),
            python::DeclarationContainer::func(func) => (func.key.name.key.as_str(), true),
        };
        compute_scope(
            container_fq_name,
            is_function_container,
            scope_type,
            &self.emitted_module_name.to_string(),
        )
    }

    /// The fully qualified name of `local_name` as used in the current statement. If an
    /// enclosing scope visible from the statement imports `local_name`, or is a function that
    /// declares it, this is the name in the innermost such scope, e.g. `m.f.<locals>.os` for a
    /// use of `os` in function `m.f` that runs `import os`, or `m.f.<locals>.x` for a variable
    /// `x` that `m.f` assigns. Otherwise it is the module-level name, e.g. `m.os`.
    ///
    /// A class body's own declarations are not matched: in `x = x` in a class body, the right
    /// side refers to the enclosing scope's `x`, but the class attribute is already recorded.
    ///
    /// The visible scopes are the statement's own scope, then those of the enclosing functions,
    /// innermost first, then the module, as in Python's name resolution. The scope of an
    /// enclosing function is the prefix of the statement's scope that ends in `.<locals>`.
    fn qualified_use_name(&self, local_name: &str) -> String {
        iter::successors(Some(self.statement_scope.as_str()), |scope| {
            scope
                .rfind(".<locals>.")
                .map(|end| &scope[..end + ".<locals>".len()])
        })
        .map(|scope| (scope.ends_with(".<locals>"), join_names(scope, local_name)))
        .find(|(is_function_scope, name)| {
            self.import_names.contains_key(name)
                || (*is_function_scope && self.names.contains(name))
        })
        .map(|(_, name)| name)
        .unwrap_or_else(|| join_names(self.emitted_module_name.as_str(), local_name))
    }

    fn get_definition_location(
        &self,
        def_range: TextRange,
        module: &ModuleInfo,
        base_type: Option<&Type>,
        additional_definitions: Vec<DefinitionLocation>,
    ) -> Vec<DefinitionLocation> {
        // `module_name` is the definition's module as Pyrefly resolves it, and is compared
        // with `self.module.name()`, this file's own resolved name. `emitted_name` is the
        // definition module's name written to Glean, so that cross-references match the
        // declarations its own file emits.
        let module_name = module.name();
        let file = definition_file_fact(module, self.naming.stdlib_root.as_deref());
        let emitted_name = file
            .as_ref()
            .and_then(|file| self.naming.module_names.get(file.key.as_str()))
            .map_or(module_name.as_str(), String::as_str);
        let type_name = base_type.and_then(|ty| ty.qname());
        let local_name = module.code_at(def_range);

        if module_name == self.module.name() {
            let fqname_type =
                type_name.and_then(|qname| self.locations_fqnames.get(&qname.range().start()));

            let fqname = if let Some(ty) = fqname_type {
                Some(join_names(ty, local_name))
            } else {
                self.locations_fqnames
                    .get(&def_range.start())
                    .map(|name| (**name).clone())
            };

            if let Some(name) = fqname {
                vec![DefinitionLocation { name, file }]
            } else {
                additional_definitions
            }
        } else {
            let fqname_type = type_name.map(|name| name.id().as_str()).unwrap_or_default();
            let fqname = join_names(fqname_type, local_name);
            if module_name == ModuleName::builtins() {
                vec![DefinitionLocation { name: fqname, file }]
            } else {
                let name = join_names(emitted_name, &fqname);
                let mut definitions = vec![DefinitionLocation {
                    name: name.clone(),
                    file,
                }];

                definitions.extend(
                    additional_definitions
                        .into_iter()
                        .filter(|def| def.name != name),
                );

                definitions
            }
        }
    }

    fn find_definition_for_expr(
        &self,
        expr: &Expr,
        only_resolved_name: bool,
    ) -> Vec<DefinitionLocation> {
        match expr {
            Expr::Subscript(expr_subscript) => {
                self.find_definition_for_expr(&expr_subscript.value, only_resolved_name)
            }
            Expr::Attribute(attr) => self.find_definition_for_expr_attribute(attr),
            Expr::Name(name) => self.find_definition_for_expr_name(name, only_resolved_name),
            _ => vec![],
        }
    }

    fn find_definition_for_expr_name(
        &self,
        expr_name: &ExprName,
        only_resolved_name: bool,
    ) -> Vec<DefinitionLocation> {
        let identifier = Ast::expr_name_identifier(expr_name.clone());
        self.find_definition_for_name_use(identifier, only_resolved_name)
    }

    fn get_additional_definitions(
        &self,
        range: TextRange,
        only_resolved_name: bool,
    ) -> Vec<DefinitionLocation> {
        let as_name = self.qualified_use_name(self.module.code_at(range));

        let mut definitions = vec![];
        if let Some((resolved_name, from_name, resolved_file, from_file)) =
            self.import_names.get(&as_name)
        {
            definitions.push(DefinitionLocation {
                name: resolved_name.to_string(),
                file: resolved_file.clone(),
            });
            if !only_resolved_name && let Some(name) = from_name.as_ref() {
                definitions.push(DefinitionLocation {
                    name: name.to_string(),
                    file: from_file.clone(),
                });
            }
        }
        if !only_resolved_name || definitions.is_empty() {
            definitions.push(DefinitionLocation {
                name: as_name,
                file: Some(self.file_fact()),
            });
        }
        definitions
    }

    fn resolve_name_use(
        &self,
        identifier: &Identifier,
        only_resolved_name: bool,
        definition: Option<FindDefinitionItemWithDocstring>,
    ) -> Vec<DefinitionLocation> {
        let additional_definitions =
            self.get_additional_definitions(identifier.range(), only_resolved_name);

        match definition {
            Some(def) => self.get_definition_location(
                def.definition_range,
                &def.module,
                None,
                additional_definitions,
            ),
            None => additional_definitions,
        }
    }

    fn find_definition_for_name_use(
        &self,
        identifier: Identifier,
        only_resolved_name: bool,
    ) -> Vec<DefinitionLocation> {
        let definition = self
            .transaction
            .find_definition_for_name_use(self.handle, &identifier, find_preference_glean())
            .unwrap_or(None);
        self.resolve_name_use(&identifier, only_resolved_name, definition)
    }

    fn find_definition_for_str_literal(
        &self,
        range: TextRange,
        base_range: Option<TextRange>,
    ) -> Vec<DefinitionLocation> {
        let name = self.module.code_at(range);
        if let Some(base_range) = base_range {
            self.find_definition_for_typed_attribute(base_range, &Name::new(name))
        } else {
            let as_name = self.qualified_use_name(name);
            let identifier = Identifier::new(name, range);
            let definition = self
                .transaction
                .find_definition_for_name_use(self.handle, &identifier, find_preference_glean())
                .unwrap_or(None);
            if definition.is_some() || self.names.contains(&as_name) {
                self.resolve_name_use(&identifier, false, definition)
            } else {
                vec![]
            }
        }
    }

    fn find_definition_for_literal_symbol(&self, name: &str) -> DefinitionLocation {
        DefinitionLocation {
            name: name.to_owned(),
            file: None,
        }
    }

    fn get_xrefs_for_str_lit(
        &self,
        expr: &ExprStringLiteral,
    ) -> Vec<(DefinitionLocation, TextRange)> {
        let separators: Vec<(usize, usize)> = self
            .module
            .code_at(expr.range())
            .match_indices(|x: char| x.is_whitespace() || TYPE_SEPARATORS.contains(&x))
            .map(|(idx, matched)| (idx, matched.len()))
            .collect();

        let ranges = (1..separators.len())
            .map(|i| {
                let (prev_idx, prev_len) = separators[i - 1];
                let (curr_idx, _) = separators[i];
                let start = TextSize::try_from(prev_idx + prev_len).ok().unwrap();
                let end = TextSize::try_from(curr_idx).ok().unwrap();
                TextRange::new(start, end) + expr.range().start()
            })
            .filter(|range| !range.is_empty());

        ranges
            .flat_map(|range| {
                let token = self.module.code_at(range);
                if token == "None" {
                    vec![(self.find_definition_for_literal_symbol(token), range)]
                } else {
                    token
                        .split('.')
                        .scan(
                            (range.start(), None::<TextRange>),
                            |(pos, base_range), name| {
                                let name_start = *pos;
                                let name_end = name_start + TextSize::try_from(name.len()).unwrap();
                                let name_range = TextRange::new(name_start, name_end);

                                let defs =
                                    self.find_definition_for_str_literal(name_range, *base_range);

                                *base_range = Some(TextRange::new(range.start(), name_end));
                                *pos = name_end + TextSize::new(1);

                                Some(defs.into_iter().map(move |def| (def, name_range)))
                            },
                        )
                        .flatten()
                        .collect()
                }
            })
            .collect()
    }

    /// Type-driven attribute definition lookup. Returns empty when no type trace
    /// exists for `base_range`; caller is responsible for fallback.
    fn find_definition_for_typed_attribute(
        &self,
        base_range: TextRange,
        attr_name: &Name,
    ) -> Vec<DefinitionLocation> {
        let definitions_with_type: Vec<_> = if let Some(answers) =
            self.transaction.get_answers(self.handle)
            && let Some(base_type) = answers.get_type_trace(base_range)
        {
            self.transaction
                .ad_hoc_solve(self.handle, "glean_attribute_definition", |solver| {
                    let completions = |ty| solver.completions(ty, Some(attr_name), false, false);
                    // The class that defines `attr_name`, as found by the solver's own member
                    // lookup, so that an inherited member is named after its defining class.
                    let defining_type = |ty: Type| match &ty {
                        Type::ClassType(cls) | Type::SelfType(cls) => solver
                            .get_class_member_with_defining_class(cls.class_object(), attr_name)
                            .map_or(ty.clone(), |member| Type::ClassDef(member.defining_class)),
                        _ => ty,
                    };

                    let tys = match base_type.clone() {
                        Type::Union(u) => u.members,
                        Type::Intersect(i) => i.0,
                        ty => vec![ty],
                    };

                    tys.into_iter()
                        .filter_map(|ty| {
                            self.transaction
                                .find_definition_for_base_type(
                                    self.handle,
                                    find_preference_glean(),
                                    completions(ty.clone()),
                                    attr_name,
                                )
                                .map(|def| (defining_type(ty), def))
                        })
                        .collect()
                })
                .unwrap_or_default()
        } else {
            return vec![];
        };

        if definitions_with_type.is_empty() {
            return vec![];
        }

        let base_expr_name = self.qualified_use_name(self.module.code_at(base_range));
        // A local variable's attributes are never declared under the variable's name, so
        // `m.f.<locals>.x.attr` would not resolve; imports and module-level names can be.
        let is_local_variable = base_expr_name.contains(".<locals>.")
            && !self.import_names.contains_key(&base_expr_name);
        let additional_definitions = if self.names.contains(&base_expr_name) && !is_local_variable {
            self.get_additional_definitions(base_range, false)
                .into_iter()
                .map(|ty| DefinitionLocation {
                    name: join_names(&ty.name, attr_name),
                    file: ty.file,
                })
                .collect()
        } else {
            vec![]
        };

        definitions_with_type
            .into_iter()
            .flat_map(|(ty, def)| {
                self.get_definition_location(
                    def.definition_range,
                    &def.module,
                    Some(&ty),
                    additional_definitions.clone(),
                )
            })
            .collect()
    }

    fn find_definition_for_expr_attribute(
        &self,
        expr_attr: &ExprAttribute,
    ) -> Vec<DefinitionLocation> {
        let attr_name = &expr_attr.attr;
        let base_expr = expr_attr.value.as_ref();

        let definitions =
            self.find_definition_for_typed_attribute(base_expr.range(), attr_name.id());

        if !definitions.is_empty() {
            definitions
        } else {
            self.find_definition_for_expr(base_expr, false)
                .into_iter()
                .map(|base_def| DefinitionLocation {
                    name: join_names(&base_def.name, attr_name),
                    file: base_def.file,
                })
                .collect()
        }
    }

    fn make_decorators(&self, decorators: &[Decorator]) -> Option<Vec<String>> {
        let glean_decorators: Vec<String> = decorators
            .iter()
            .map(|x| self.module.code_at(x.range()).to_owned())
            .collect();

        if glean_decorators.is_empty() {
            None
        } else {
            Some(glean_decorators)
        }
    }

    fn class_facts(
        &mut self,
        cls: &StmtClassDef,
        cls_declaration: python::ClassDeclaration,
        context: &NodeContext,
    ) -> DeclarationInfo {
        let bases = if let Some(arguments) = &cls.arguments {
            arguments
                .args
                .iter()
                .filter_map(|expr| {
                    self.find_definition_for_expr(expr, true)
                        .into_iter()
                        .next()
                        .map(|def| python::ClassDeclaration::new(python::Name::new(def.name), None))
                })
                .collect()
        } else {
            vec![]
        };

        let cls_definition = python::ClassDefinition::new(
            cls_declaration.clone(),
            Some(bases),
            None,
            self.make_decorators(&cls.decorator_list),
            Some((*context.container).clone()),
        );

        DeclarationInfo {
            declaration: python::Declaration::cls(cls_declaration),
            decl_span: to_span(range_without_decorators(cls.range, &cls.decorator_list)),
            definition: Some(python::Definition::cls(cls_definition)),
            def_span: Some(to_span(cls.range)),
            top_level_decl: (*context.top_level_decl).clone(),
            docstring_range: Docstring::range_from_stmts(&cls.body),
        }
    }

    fn make_xrefs(
        &self,
        expr: &Expr,
        include_str_lit_xrefs: bool,
    ) -> Vec<(DefinitionLocation, TextRange)> {
        match expr {
            Expr::Attribute(attr) if attr.ctx.is_load() => self
                .find_definition_for_expr_attribute(attr)
                .into_iter()
                .map(|name| (name, attr.attr.range()))
                .collect(),
            Expr::Name(name) if name.ctx.is_load() => self
                .find_definition_for_expr_name(name, false)
                .into_iter()
                .map(|x| (x, name.range()))
                .collect(),
            Expr::StringLiteral(str_lit) if include_str_lit_xrefs => {
                self.get_xrefs_for_str_lit(str_lit)
            }
            Expr::BooleanLiteral(_) | Expr::NoneLiteral(_) => {
                let range = expr.range();
                let name = self.module.code_at(range);
                vec![(self.find_definition_for_literal_symbol(name), range)]
            }
            _ => {
                vec![]
            }
        }
    }

    fn add_xref(&mut self, definition_location: DefinitionLocation, range: TextRange) {
        let source = to_span(range);
        let target_name = python::Name::new(definition_location.name);
        let target = python_xrefs::XRefDefinitionLocation {
            name: target_name.clone(),
            file: definition_location.file,
        };
        if let Some(spans) = self.facts.xrefs_via_name_by_target.get_mut(&target_name) {
            spans.push(source.clone());
        } else {
            self.facts
                .xrefs_via_name_by_target
                .insert(target_name.clone(), vec![source.clone()]);
        }
        self.facts.xrefs_via_name.push(python::XRefViaName {
            target: target_name,
            source: source.clone(),
        });

        self.facts.xrefs.push(python_xrefs::XRef { target, source });
    }

    fn generate_expr_facts_and_xrefs(
        &mut self,
        expr: &Expr,
        container: &python::DeclarationContainer,
        include_str_lit_xrefs: bool,
    ) -> Vec<(DefinitionLocation, TextRange)> {
        if let Some(call) = expr.as_call_expr() {
            self.file_call_facts(call);
            if let python::DeclarationContainer::func(caller) = container {
                self.callee_to_caller_facts(call, caller);
            }
        }
        self.make_xrefs(expr, include_str_lit_xrefs)
    }

    fn visit_annotation_expr(
        &mut self,
        expr: &Expr,
        xrefs: &mut Vec<python::XRefViaName>,
        offset: TextSize,
        container: &python::DeclarationContainer,
    ) {
        for (def, abs_range) in self.generate_expr_facts_and_xrefs(expr, container, true) {
            let rel_range = abs_range.sub(offset);
            xrefs.push(python::XRefViaName {
                target: python::Name::new(def.name.clone()),
                source: to_span(rel_range),
            });
            self.add_xref(def, abs_range);
        }

        expr.recurse(&mut |x| self.visit_annotation_expr(x, xrefs, offset, container));
    }

    fn display_type_info(&self, range: TextRange) -> python::Type {
        let parts: Vec<&str> = self
            .module
            .code_at(range)
            .split_whitespace()
            .flat_map(|x| x.split_inclusive(TYPE_SEPARATORS))
            .flat_map(|x| {
                if x.ends_with(TYPE_SEPARATORS) {
                    let (name, sep) = x.split_at(x.len() - 1);
                    vec![name, sep].into_iter()
                } else {
                    vec![x].into_iter()
                }
            })
            .filter(|x| !x.is_empty())
            .map(|x| {
                if x == "await" {
                    "await "
                } else if x == ":" {
                    ": "
                } else if x == "|" {
                    " | "
                } else {
                    x
                }
            })
            .collect();

        let mut display = "".to_owned();
        for i in 0..parts.len() {
            let part = parts[i];
            if part == "," {
                let next = parts.get(i + 1);
                if next.is_some_and(|x| !["]", ")", "}"].contains(x)) {
                    display.push_str(", ");
                }
            } else {
                display.push_str(part);
            }
        }
        python::Type::new(display)
    }

    fn visit_annotation_exprs(
        &mut self,
        annotation: Option<&Expr>,
        container: &python::DeclarationContainer,
    ) -> Option<python::TypeInfo> {
        annotation.map(|type_annotation| {
            let mut xrefs = vec![];
            let range = type_annotation.range();
            type_annotation.visit(&mut |expr| {
                self.visit_annotation_expr(expr, &mut xrefs, range.start(), container)
            });
            python::TypeInfo {
                displayType: self.display_type_info(range),
                xrefs,
            }
        })
    }

    fn variable_info(
        &self,
        name: python::Name,
        range: TextRange,
        type_info: Option<python::TypeInfo>,
        docstring_range: Option<TextRange>,
        ctx: &NodeContext,
    ) -> DeclarationInfo {
        let variable_declaration = python::VariableDeclaration::new(name);
        let variable_definition = python::VariableDefinition::new(
            variable_declaration.clone(),
            type_info,
            Some((*ctx.container).clone()),
        );

        DeclarationInfo {
            declaration: python::Declaration::variable(variable_declaration),
            decl_span: to_span(range),
            definition: Some(python::Definition::variable(variable_definition)),
            def_span: Some(to_span(range)),
            top_level_decl: (*ctx.top_level_decl).clone(),
            docstring_range,
        }
    }

    fn parameter_info(
        &mut self,
        param: &Parameter,
        value: Option<String>,
        context: &NodeContext,
        decl_infos: &mut Vec<DeclarationInfo>,
        container: &python::DeclarationContainer,
    ) -> python::Parameter {
        let type_info: Option<python::TypeInfo> =
            self.visit_annotation_exprs(param.annotation(), container);
        decl_infos.push(self.variable_info(
            self.declared_name(&param.name),
            param.name.range(),
            type_info.clone(),
            None,
            context,
        ));
        python::Parameter {
            name: python::Name::new(param.name().to_string()),
            typeInfo: type_info,
            value,
        }
    }

    fn parameter_with_default_info(
        &mut self,
        parameter_with_default: &ParameterWithDefault,
        context: &NodeContext,
        decl_infos: &mut Vec<DeclarationInfo>,
        container: &python::DeclarationContainer,
    ) -> python::Parameter {
        let value: Option<String> = parameter_with_default
            .default
            .as_ref()
            .map(|x| self.module.code_at(x.range()).to_owned());
        self.visit_exprs(&parameter_with_default.default, container);
        self.parameter_info(
            &parameter_with_default.parameter,
            value,
            context,
            decl_infos,
            container,
        )
    }

    fn function_facts(
        &mut self,
        func: &StmtFunctionDef,
        func_declaration: python::FunctionDeclaration,
        parent_ctx: &NodeContext,
        func_ctx: &NodeContext,
    ) -> Vec<DeclarationInfo> {
        let params = &func.parameters;

        let mut decl_infos = vec![];
        let container = &*parent_ctx.container;
        let args = params
            .args
            .iter()
            .map(|x| self.parameter_with_default_info(x, func_ctx, &mut decl_infos, container))
            .collect();

        let pos_only_args = params
            .posonlyargs
            .iter()
            .map(|x| self.parameter_with_default_info(x, func_ctx, &mut decl_infos, container))
            .collect();

        let kwonly_args = params
            .kwonlyargs
            .iter()
            .map(|x| self.parameter_with_default_info(x, func_ctx, &mut decl_infos, container))
            .collect();

        let star_arg = params
            .vararg
            .as_ref()
            .map(|x| self.parameter_info(x.as_ref(), None, func_ctx, &mut decl_infos, container));

        let star_kwarg = params
            .kwarg
            .as_ref()
            .map(|x| self.parameter_info(x.as_ref(), None, func_ctx, &mut decl_infos, container));

        let func_definition = python::FunctionDefinition::new(
            func_declaration.clone(),
            func.is_async,
            self.visit_annotation_exprs(func.returns.as_ref().map(|x| x.as_ref()), container),
            args,
            Some(pos_only_args),
            Some(kwonly_args),
            star_arg,
            star_kwarg,
            self.make_decorators(&func.decorator_list),
            Some((*parent_ctx.container).clone()),
        );

        decl_infos.push(DeclarationInfo {
            declaration: python::Declaration::func(func_declaration),
            decl_span: to_span(range_without_decorators(func.range, &func.decorator_list)),
            definition: Some(python::Definition::func(func_definition)),
            def_span: Some(to_span(func.range)),
            top_level_decl: (*parent_ctx.top_level_decl).clone(),
            docstring_range: Docstring::range_from_stmts(&func.body),
        });

        decl_infos
    }

    fn variable_facts(
        &mut self,
        target: &Expr,
        info: &AssignInfo,
        ctx: &NodeContext,
        next: Option<&Stmt>,
        def_infos: &mut Vec<DeclarationInfo>,
    ) {
        for name in bound_names(target) {
            let docstring_range =
                next.and_then(|stmt| Docstring::range_from_stmts(slice::from_ref(stmt)));
            def_infos.push(self.variable_info(
                self.declared_name(name),
                info.range,
                info.type_info.clone(),
                docstring_range,
                ctx,
            ));

            if name.id == dunder::ALL
                && let Some(Expr::List(list_expr)) = info.value
            {
                for str_lit in list_expr
                    .elts
                    .iter()
                    .filter_map(|e| e.as_string_literal_expr())
                {
                    for (definition, range) in self.get_xrefs_for_str_lit(str_lit) {
                        self.add_xref(definition, range);
                    }
                }
            }
        }
    }

    fn find_definition(&self, position: TextSize) -> Vec<DefinitionLocation> {
        let definitions = self
            .transaction
            .find_definition(self.handle, position, find_preference_glean())
            .map(Vec1::into_vec)
            .unwrap_or_default();

        definitions
            .into_iter()
            .flat_map(|def| {
                self.get_definition_location(def.definition_range, &def.module, None, vec![])
            })
            .collect()
    }

    fn find_definition_for_imported_module(&self, module: ModuleName) -> DefinitionLocation {
        let definition = self
            .transaction
            .find_definition_for_imported_module(self.handle, module, find_preference_glean())
            .unwrap_or(None);

        let file = definition
            .and_then(|def| definition_file_fact(&def.module, self.naming.stdlib_root.as_deref()));
        let name = file
            .as_ref()
            .and_then(|file| self.naming.module_names.get(file.key.as_str()))
            .cloned()
            .unwrap_or_else(|| module.to_string());

        DefinitionLocation { name, file }
    }

    fn import_fact(
        &self,
        binding: &ImportBinding,
        top_level_declaration: &python::Declaration,
    ) -> DeclarationInfo {
        let import_fact = python::ImportStatement::new(
            python::Name::new(binding.from_name.id().to_string()),
            python::Name::new(join_names(&self.statement_scope, &binding.as_name)),
        );

        DeclarationInfo {
            declaration: python::Declaration::imp(import_fact),
            decl_span: to_span(binding.from_name.range),
            definition: None,
            def_span: None,
            top_level_decl: top_level_declaration.clone(),
            docstring_range: None,
        }
    }

    fn add_xrefs_for_module(&mut self, module_name: ModuleName, position: TextSize, prefix: &str) {
        for (module, range) in all_modules_with_range(module_name, position) {
            let resolved_name = join_names(prefix, &module);
            let def =
                self.find_definition_for_imported_module(ModuleName::from_str(&resolved_name));

            self.record_name(def.name.clone());
            self.add_xref(def, range);
        }
    }

    /// The names that `import` binds in an `import` statement: `c` for `import a.b as c`, and
    /// both `a` and `a.b` for `import a.b`.
    fn import_bindings(&self, import: &Alias) -> Vec<ImportBinding> {
        let from_name = &import.name;
        let module_name = ModuleName::from_name(from_name.id());
        if let Some(as_name) = &import.asname {
            vec![ImportBinding {
                from_name: from_name.clone(),
                as_name: as_name.clone(),
                resolved: self.find_definition_for_imported_module(module_name),
                from_file: None,
            }]
        } else {
            let position = from_name.range.start();
            all_modules_with_range(module_name, position)
                .map(|(module, range)| {
                    let mod_range = TextRange::new(position, range.end());
                    let mod_id = Identifier::new(Name::new(module.clone()), mod_range);
                    ImportBinding {
                        from_name: mod_id.clone(),
                        as_name: mod_id,
                        resolved: self
                            .find_definition_for_imported_module(ModuleName::from_str(&module)),
                        from_file: None,
                    }
                })
                .collect()
        }
    }

    fn import_facts(
        &mut self,
        import: &StmtImport,
        top_level_declaration: &python::Declaration,
    ) -> Vec<DeclarationInfo> {
        import
            .names
            .iter()
            .flat_map(|import| {
                let from_name = &import.name;
                let module_name = ModuleName::from_name(from_name.id());
                self.add_xrefs_for_module(module_name, from_name.range.start(), "");
                self.import_bindings(import)
                    .iter()
                    .map(|binding| self.import_fact(binding, top_level_declaration))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// The package that the dots of a relative `from ... import` statement refer to, e.g. `a.b`
    /// for `from .. import x` in module `a.b.c`.
    fn relative_import_base(&self, import_from: &StmtImportFrom) -> Option<ModuleName> {
        if import_from.level > 0 {
            self.module.name().new_maybe_relative(
                self.module.path().is_init(),
                import_from.level,
                None,
            )
        } else {
            None
        }
    }

    fn add_xrefs_for_from_module(&mut self, import_from: &StmtImportFrom) {
        let resolved_module_prefix = self.relative_import_base(import_from);
        let dots_range = self
            .module
            .code_at(import_from.range())
            .match_indices(['.'])
            .next()
            .and_then(|(s, _)| {
                let len = import_from.level;
                if len > 0 {
                    let offset = TextSize::try_from(s).unwrap();
                    let len = TextSize::from(len);
                    Some(TextRange::at(import_from.range().start() + offset, len))
                } else {
                    None
                }
            });

        let module_prefix_str = resolved_module_prefix
            .as_ref()
            .map_or("", |module| module.as_str());

        if let Some(range) = dots_range
            && let Some(module_prefix) = resolved_module_prefix
            && !module_prefix_str.is_empty()
        {
            let def = self.find_definition_for_imported_module(module_prefix);
            self.add_xref(def, range);
        }

        if let Some(module_id) = &import_from.module {
            let position = module_id.range.start();
            self.add_xrefs_for_module(ModuleName::from_str(module_id), position, module_prefix_str);
        }
    }

    fn from_module(&self, import_from: &StmtImportFrom) -> DefinitionLocation {
        let module_prefix = self.relative_import_base(import_from);
        let module_prefix_str = module_prefix.as_ref().map_or("", |module| module.as_str());
        let module_str = import_from
            .module
            .as_ref()
            .map_or("", |module| module.as_str());
        let fqname_module = join_names(module_prefix_str, module_str);
        self.find_definition_for_imported_module(ModuleName::from_str(&fqname_module))
    }

    /// The name that `import` binds in a `from from_module import ...` statement, or `None`
    /// for `*`.
    fn import_from_binding(
        &self,
        import: &Alias,
        from_module: &DefinitionLocation,
    ) -> Option<ImportBinding> {
        let from_name = &import.name;
        if from_name.id.as_str() == "*" {
            return None;
        }
        let from_name_string = join_names(&from_module.name, from_name.id());
        let resolved = self
            .find_definition(from_name.range.start())
            .into_iter()
            .next()
            .unwrap_or_else(|| DefinitionLocation {
                name: from_name_string.clone(),
                file: from_module.file.clone(),
            });
        Some(ImportBinding {
            from_name: Identifier::new(Name::new(from_name_string), from_name.range),
            as_name: import.asname.clone().unwrap_or_else(|| from_name.clone()),
            resolved,
            from_file: from_module.file.clone(),
        })
    }

    fn import_from_facts(
        &mut self,
        import_from: &StmtImportFrom,
        top_level_declaration: &python::Declaration,
    ) -> Vec<DeclarationInfo> {
        self.add_xrefs_for_from_module(import_from);
        let from_module = self.from_module(import_from);

        let mut decl_infos = vec![];
        for import in &import_from.names {
            if let Some(binding) = self.import_from_binding(import, &from_module) {
                let range = binding.from_name.range;
                let from_name_definition = DefinitionLocation {
                    name: binding.from_name.id().to_string(),
                    file: binding.from_file.clone(),
                };
                if binding.resolved.name != from_name_definition.name {
                    self.add_xref(from_name_definition, range)
                }
                self.add_xref(binding.resolved.clone(), range);
                decl_infos.push(self.import_fact(&binding, top_level_declaration));
            } else {
                let import_star = python::ImportStarStatement::new(
                    python::Name::new(from_module.name.clone()),
                    self.facts.module.clone(),
                );
                self.facts
                    .import_star_locations
                    .push(python::ImportStarLocation::new(
                        import_star,
                        self.facts.file.clone(),
                        to_span(import_from.range),
                    ));
            }
        }

        decl_infos
    }

    fn arg_string_lit(&self, argument: &Expr) -> Option<python::Argument> {
        let string_literal = match argument {
            Expr::StringLiteral(expr) => Some(expr.value.to_string()),
            Expr::BytesLiteral(expr) => {
                let bytes_lit: Vec<u8> = expr.value.bytes().collect();
                str::from_utf8(&bytes_lit).ok().map(|x| x.to_owned())
            }
            _ => None,
        };

        string_literal.map(|lit| python::Argument::lit(python::StringLiteral::new(lit)))
    }

    fn file_call_facts(&mut self, call: &ExprCall) {
        let callee_span = to_span(call.range());
        let mut call_args: Vec<python::CallArgument> = call
            .arguments
            .args
            .iter()
            .map(|arg| python::CallArgument {
                label: None,
                span: to_span(arg.range()),
                argument: self.arg_string_lit(arg),
            })
            .collect();

        let keyword_args = call
            .arguments
            .keywords
            .iter()
            .map(|keyword| python::CallArgument {
                label: keyword
                    .arg
                    .as_ref()
                    .map(|id| python::Name::new(id.id().to_string())),
                span: to_span(keyword.range()),
                argument: self.arg_string_lit(&keyword.value),
            });

        call_args.extend(keyword_args);

        self.facts.file_calls.push(python::FileCall::new(
            self.facts.file.clone(),
            callee_span,
            call_args,
        ));
    }

    fn callee_to_caller_facts(&mut self, call: &ExprCall, caller: &python::FunctionDeclaration) {
        let caller_fact = &caller.key.name;
        let callee_names: Vec<String> = self
            .find_definition_for_expr(call.func.as_ref(), false)
            .into_iter()
            .map(|definition| definition.name)
            .collect();
        for callee_name in callee_names {
            self.facts
                .callee_to_callers
                .push(python::CalleeToCaller::new(
                    python::Name::new(callee_name),
                    caller_fact.clone(),
                ));
        }
    }

    fn visit_expr(&mut self, expr: &Expr, container: &python::DeclarationContainer) {
        for (definition, range) in self.generate_expr_facts_and_xrefs(expr, container, false) {
            self.add_xref(definition, range);
        }
        expr.recurse(&mut |s| self.visit_expr(s, container));
    }

    fn visit_exprs(&mut self, node: &impl Visit<Expr>, container: &python::DeclarationContainer) {
        node.visit(&mut |expr| self.visit_expr(expr, container));
    }

    /// Emit the facts of the module's statements in two passes. The first pass records the
    /// fully qualified name of every declaration, and the second emits the declaration facts
    /// and the cross-references. Python resolves a name to the binding in its scope wherever
    /// that binding is in the scope, so a use can come before its declaration, e.g. in a
    /// function that uses a class defined later in the module, or in a `return x` where `x` is
    /// assigned in nested blocks. Every lookup of the second pass thus finds its declaration.
    fn generate_facts(&mut self, ast: &ThinVec<Stmt>, range: TextRange) {
        self.module_facts(range);

        let root_context = NodeContext {
            container: Arc::new(python::DeclarationContainer::module(self.module_fact())),
            top_level_decl: Arc::new(python::Declaration::module(self.module_fact())),
            globals: Arc::new(SmallSet::new()),
            nonlocals: Arc::new(SmallSet::new()),
        };
        let mut top_level_statements = vec![];
        ast.visit(&mut |x| top_level_statements.push(x));
        let mut statements = vec![];
        self.declare_statements(&top_level_statements, &root_context, &mut statements);
        for (statement, next, context) in statements {
            self.process_statement(statement, next, &context);
        }
    }

    /// Record the declarations of `statements` and of their nested statements. Each statement
    /// is appended to `visited` in source order, with its following sibling, which is needed
    /// to compute the docstring range of variables, and the context it is processed in.
    fn declare_statements<'a>(
        &mut self,
        statements: &[&'a Stmt],
        context: &NodeContext,
        visited: &mut Vec<(&'a Stmt, Option<&'a Stmt>, NodeContext)>,
    ) {
        for (i, statement) in statements.iter().enumerate() {
            visited.push((statement, statements.get(i + 1).copied(), context.clone()));
            let children_context = self.declare_statement(statement, context);
            let mut children = vec![];
            statement.recurse(&mut |x| children.push(x));
            self.declare_statements(&children, &children_context, visited);
        }
    }

    /// Record the declarations of `stmt` itself, and return the context of its nested
    /// statements.
    ///
    /// The variables of all comprehensions in a scope share the `<comprehension>` scope inside
    /// it, e.g. `x` in `[x for x in xs]` is `m.f.<locals>.<comprehension>.x` in function `m.f`.
    /// Uses resolve to it through the definition position, which is distinct for each
    /// comprehension. A walrus target in a comprehension binds the enclosing scope, as in Python.
    fn declare_statement(&mut self, stmt: &Stmt, context: &NodeContext) -> NodeContext {
        self.statement_scope = self.scope_in(&context.container, &ScopeType::Local);
        let bindings = expr_bindings(stmt);
        for named in bindings.named_exprs {
            self.declare_targets(&named.target, context);
        }
        let comprehension_scope = join_names(&self.statement_scope, "<comprehension>");
        for comprehension in bindings.comprehensions {
            for name in bound_names(&comprehension.target) {
                let identifier = Ast::expr_name_identifier(name.clone());
                self.declare(&identifier, comprehension_scope.clone());
            }
        }

        match stmt {
            Stmt::ClassDef(cls) => self.declare(&cls.name, self.statement_scope.clone()),
            Stmt::FunctionDef(func) => self.declare(&func.name, self.statement_scope.clone()),
            Stmt::Assign(assign) => {
                for target in &assign.targets {
                    self.declare_targets(target, context);
                }
            }
            Stmt::AnnAssign(assign) => self.declare_targets(&assign.target, context),
            Stmt::AugAssign(assign) => self.declare_targets(&assign.target, context),
            Stmt::For(stmt_for) => self.declare_targets(&stmt_for.target, context),
            Stmt::With(stmt_with) => {
                for item in &stmt_with.items {
                    if let Some(target) = &item.optional_vars {
                        self.declare_targets(target, context);
                    }
                }
            }
            Stmt::Try(stmt_try) => {
                for ExceptHandler::ExceptHandler(handler) in &stmt_try.handlers {
                    if let Some(name) = &handler.name {
                        self.declare(name, self.statement_scope.clone());
                    }
                }
            }
            Stmt::Import(import) => {
                let bindings: Vec<_> = import
                    .names
                    .iter()
                    .flat_map(|import| self.import_bindings(import))
                    .collect();
                bindings
                    .into_iter()
                    .for_each(|binding| self.record_import(binding));
            }
            Stmt::ImportFrom(import_from) => {
                let from_module = self.from_module(import_from);
                let bindings: Vec<_> = import_from
                    .names
                    .iter()
                    .filter_map(|import| self.import_from_binding(import, &from_module))
                    .collect();
                bindings
                    .into_iter()
                    .for_each(|binding| self.record_import(binding));
            }
            _ => {}
        }

        let children_context = self.child_context(stmt, context);
        if let Stmt::FunctionDef(func) = stmt {
            let function_scope = self.scope_in(&children_context.container, &ScopeType::Local);
            for parameter in func.parameters.iter() {
                self.declare(parameter.name(), function_scope.clone());
            }
        }
        children_context
    }

    /// Record the names that assigning to `target` binds, in the scope selected by the
    /// `global` and `nonlocal` statements of `context`.
    fn declare_targets(&mut self, target: &Expr, context: &NodeContext) {
        for name in bound_names(target) {
            let scope_type = if context.globals.contains(&name.id) {
                ScopeType::Global
            } else if context.nonlocals.contains(&name.id) {
                ScopeType::Nonlocal
            } else {
                ScopeType::Local
            };
            let scope = self.scope_in(&context.container, &scope_type);
            self.declare(&Ast::expr_name_identifier(name.clone()), scope);
        }
    }

    /// The context of the statements nested in `stmt`. A class or function body has the class
    /// or function as its container, and as its top-level declaration if `stmt` is at the top
    /// level. Other nested statements share `context`.
    fn child_context(&self, stmt: &Stmt, context: &NodeContext) -> NodeContext {
        let (declaration, container, body) = match stmt {
            Stmt::ClassDef(cls) => {
                let cls_declaration =
                    python::ClassDeclaration::new(self.declared_name(&cls.name), None);
                (
                    python::Declaration::cls(cls_declaration.clone()),
                    python::DeclarationContainer::cls(cls_declaration),
                    &cls.body,
                )
            }
            Stmt::FunctionDef(func) => {
                let func_declaration =
                    python::FunctionDeclaration::new(self.declared_name(&func.name));
                (
                    python::Declaration::func(func_declaration.clone()),
                    python::DeclarationContainer::func(func_declaration),
                    &func.body,
                )
            }
            _ => return context.clone(),
        };
        let top_level_decl = match *context.top_level_decl {
            python::Declaration::module(_) => Arc::new(declaration),
            _ => context.top_level_decl.dupe(),
        };
        let (globals, nonlocals) = gather_nonlocal_variables(body);
        NodeContext {
            container: Arc::new(container),
            top_level_decl,
            globals,
            nonlocals,
        }
    }

    fn process_statement(&mut self, stmt: &Stmt, next: Option<&Stmt>, context: &NodeContext) {
        let container = &context.container;
        let top_level_decl = &*context.top_level_decl;
        self.statement_scope = self.scope_in(container, &ScopeType::Local);

        let mut decl_infos = vec![];
        let bindings = expr_bindings(stmt);
        for named in bindings.named_exprs {
            let info = AssignInfo {
                range: named.range(),
                type_info: None,
                value: Some(named.value.as_ref()),
            };
            self.variable_facts(&named.target, &info, context, None, &mut decl_infos);
        }
        for comprehension in bindings.comprehensions {
            let info = AssignInfo {
                range: TextRange::new(
                    comprehension.range.start(),
                    comprehension.iter.range().end(),
                ),
                type_info: None,
                value: None,
            };
            self.variable_facts(&comprehension.target, &info, context, None, &mut decl_infos);
        }
        match stmt {
            Stmt::ClassDef(cls) => {
                let cls_declaration =
                    python::ClassDeclaration::new(self.declared_name(&cls.name), None);
                decl_infos.push(self.class_facts(cls, cls_declaration, context));
                self.visit_exprs(&cls.decorator_list, container);
                self.visit_exprs(&cls.type_params, container);
                self.visit_exprs(&cls.arguments, container);
            }
            Stmt::FunctionDef(func) => {
                let func_declaration =
                    python::FunctionDeclaration::new(self.declared_name(&func.name));
                let func_ctx = self.child_context(stmt, context);
                decl_infos.extend(self.function_facts(func, func_declaration, context, &func_ctx));
                self.visit_exprs(&func.decorator_list, container);
                self.visit_exprs(&func.type_params, container);
            }
            Stmt::Assign(assign) => {
                let info = AssignInfo {
                    range: assign.range(),
                    type_info: None,
                    value: Some(assign.value.as_ref()),
                };
                assign.targets.visit(&mut |target| {
                    self.variable_facts(target, &info, context, next, &mut decl_infos)
                });
                self.visit_exprs(&assign.targets, container);
                self.visit_exprs(&assign.value, container);
            }
            Stmt::AnnAssign(assign) => {
                let type_info = self.visit_annotation_exprs(Some(&assign.annotation), container);
                let info = AssignInfo {
                    range: assign.range(),
                    type_info,
                    value: assign.value.as_ref().map(|v| v.as_ref()),
                };
                self.variable_facts(&assign.target, &info, context, next, &mut decl_infos);
                self.visit_exprs(&assign.target, container);
                self.visit_exprs(&assign.value, container);
            }
            Stmt::AugAssign(assign) => {
                let info = AssignInfo {
                    range: assign.range(),
                    type_info: None,
                    value: Some(assign.value.as_ref()),
                };
                self.variable_facts(&assign.target, &info, context, next, &mut decl_infos);
                self.visit_exprs(&assign.target, container);
                self.visit_exprs(&assign.value, container);
            }
            Stmt::Import(import) => {
                let mut imp_decl_infos = self.import_facts(import, top_level_decl);
                decl_infos.append(&mut imp_decl_infos);
            }
            Stmt::ImportFrom(import) => {
                let mut imp_decl_infos = self.import_from_facts(import, top_level_decl);
                decl_infos.append(&mut imp_decl_infos);
            }
            Stmt::For(stmt_for) => {
                let range = TextRange::new(stmt_for.range().start(), stmt_for.iter.range().end());
                let info = AssignInfo {
                    range,
                    type_info: None,
                    value: None,
                };
                stmt_for.target.visit(&mut |target| {
                    self.variable_facts(target, &info, context, next, &mut decl_infos)
                });
                self.visit_exprs(&stmt_for.iter, container);
            }
            Stmt::While(stmt_while) => self.visit_exprs(&stmt_while.test, container),
            Stmt::If(stmt_if) => {
                self.visit_exprs(&stmt_if.test, container);
                for x in &stmt_if.elif_else_clauses {
                    self.visit_exprs(&x.test, container);
                }
            }
            Stmt::With(stmt_with) => {
                for item in &stmt_with.items {
                    self.visit_exprs(&item.context_expr, container);
                    item.optional_vars.visit(&mut |target| {
                        let info = AssignInfo {
                            range: target.range(),
                            type_info: None,
                            value: None,
                        };
                        self.variable_facts(target, &info, context, next, &mut decl_infos)
                    });
                }
            }
            Stmt::Match(stmt_match) => {
                self.visit_exprs(&stmt_match.subject, container);
                for x in &stmt_match.cases {
                    self.visit_exprs(&x.guard, container);
                    self.visit_exprs(&x.pattern, container);
                }
            }
            Stmt::Try(stmt_try) => {
                stmt_try.handlers.iter().for_each(|x| match x {
                    ExceptHandler::ExceptHandler(x) => {
                        self.visit_exprs(&x.type_, container);
                        if let Some(name) = &x.name {
                            decl_infos.push(self.variable_info(
                                self.declared_name(name),
                                name.range(),
                                None,
                                None,
                                context,
                            ));
                        }
                    }
                });
            }
            _ => self.visit_exprs(stmt, container),
        }
        for decl_info in decl_infos {
            self.declaration_facts(decl_info);
        }
    }
}

/// Emit Glean facts, with names and files adjusted by `naming` and Pyrefly's import resolution
/// unchanged.
pub fn glean(
    transaction: &Transaction,
    handle: &Handle,
    ownership: bool,
    naming: &GleanNaming,
) -> Glean {
    let ast = &*transaction.get_ast(handle).unwrap();
    let mut glean_state = GleanState::new(transaction, handle, naming);

    glean_state.record_name("".to_owned());
    let file_language_fact = src::FileLanguage::new(glean_state.file_fact(), src::Language::Python);
    let digest_fact = glean_state.digest_fact();
    let file_lines = glean_state.file_lines_fact();
    glean_state.generate_facts(&ast.body, ast.range());

    let file_fact = glean_state.file_fact();
    let gencode_fact = glean_state.gencode_fact();

    let facts = glean_state.facts;

    let xrefs_via_name_by_file_fact =
        python::XRefsViaNameByFile::new(file_fact.clone(), facts.xrefs_via_name.to_owned());

    let xrefs_via_name_by_target: Vec<python::XRefsViaNameByTarget> = facts
        .xrefs_via_name_by_target
        .into_iter()
        .map(|(target, spans)| {
            python::XRefsViaNameByTarget::new(
                target.to_owned(),
                file_fact.clone(),
                spans.to_owned(),
            )
        })
        .collect();

    let xrefs_by_file = python_xrefs::XRefsByFile::new(file_fact.clone(), facts.xrefs);

    let mut entries = vec![
        GleanEntry::SchemaId {
            schema_id: builtin::SCHEMA_ID.to_owned(),
        },
        python::Name::new("".to_owned()).glean_entry(),
        facts.modules.glean_entry(),
        file_language_fact.glean_entry(),
        file_lines.glean_entry(),
        digest_fact.glean_entry(),
        facts.decl_locations.glean_entry(),
        facts.def_locations.glean_entry(),
        facts.import_star_locations.glean_entry(),
        facts.file_calls.glean_entry(),
        facts.callee_to_callers.glean_entry(),
        facts.containing_top_level_declarations.glean_entry(),
        xrefs_via_name_by_file_fact.glean_entry(),
        xrefs_via_name_by_target.glean_entry(),
        xrefs_by_file.glean_entry(),
        facts.declaration_docstrings.glean_entry(),
        facts.name_to_sname.glean_entry(),
        gencode_fact.glean_entry(),
    ];
    if ownership {
        for entry in &mut entries {
            if let GleanEntry::Predicate { unit, .. } = entry {
                *unit = Some(file_fact.key.to_string());
            }
        }
    }
    Glean { entries }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use pyrefly_python::module_path::ModulePath;

    use super::*;

    fn module_at(path: &str) -> ModuleInfo {
        ModuleInfo::new(
            ModuleName::from_str("m"),
            ModulePath::filesystem(PathBuf::from(path)),
            Arc::new(String::new()),
        )
    }

    #[test]
    fn test_definition_file_fact_source_file() {
        let file = definition_file_fact(&module_at("fbcode/foo/bar.py"), None);
        assert_eq!(file.map(|f| *f.key), Some("fbcode/foo/bar.py".to_owned()));
    }

    #[test]
    fn test_definition_file_fact_buck_out() {
        let path = "buck-out/.pyrelsp/art/fbcode/foo/__bar-python-bar.thrift__/out/gen-python/foo";
        assert!(
            definition_file_fact(&module_at(path), None).is_none(),
            "Buck outputs have no file fact"
        );
    }

    #[test]
    fn test_definition_file_fact_outside_working_directory() {
        // Windows only treats a path with a drive as absolute, so build the path from the
        // root of the working directory, which is absolute on every platform.
        let root = current_dir().unwrap();
        let path = root
            .ancestors()
            .last()
            .unwrap()
            .join("nonexistent/site-packages/foo.py");
        let file = definition_file_fact(&module_at(path.to_str().unwrap()), None);
        assert!(
            file.is_none(),
            "files outside the working directory have no file fact"
        );
    }

    fn module_with_path(path: ModulePath) -> ModuleInfo {
        ModuleInfo::new(ModuleName::from_str("m"), path, Arc::new(String::new()))
    }

    #[test]
    fn test_definition_file_fact_bundled_stdlib_stub() {
        let stub = module_with_path(ModulePath::bundled_typeshed(PathBuf::from(
            "logging/__init__.pyi",
        )));
        assert_eq!(
            definition_file_fact(&stub, Some("third-party/python/3.12/Lib")).map(|f| *f.key),
            Some("third-party/python/3.12/Lib/logging/__init__.py".to_owned()),
            "a bundled stdlib stub takes the path of the stdlib source it describes"
        );
        assert!(
            definition_file_fact(&stub, None).is_none(),
            "without a stdlib root, a bundled stub is not a repository file"
        );
    }

    #[test]
    fn test_stdlib_source_file_follows_cpython_layout() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("os.py"), "").unwrap();
        fs::create_dir(root.path().join("re")).unwrap();
        fs::write(root.path().join("re/__init__.py"), "").unwrap();
        let root_str = root.path().to_str().unwrap();
        let source = |stub: &str| *stdlib_source_file(root_str, Path::new(stub)).key;
        assert_eq!(
            source("os/__init__.pyi"),
            format!("{root_str}/os.py"),
            "typeshed's os package is CPython's os.py"
        );
        assert_eq!(
            source("re.pyi"),
            format!("{root_str}/re/__init__.py"),
            "typeshed's re module is CPython's re package"
        );
        assert_eq!(
            source("json/__init__.pyi"),
            format!("{root_str}/json/__init__.py"),
            "without a source on disk, the stub's own layout is used"
        );
    }

    #[test]
    fn test_definition_file_fact_namespace_and_bundled_third_party() {
        for path in [
            ModulePath::namespace(PathBuf::from("fbcode/torch")),
            ModulePath::bundled_typeshed_third_party(PathBuf::from("requests/__init__.pyi")),
        ] {
            assert!(
                definition_file_fact(&module_with_path(path), Some("third-party/python/3.12/Lib"))
                    .is_none(),
                "namespace package directories and bundled third-party stubs have no file fact"
            );
        }
    }

    #[test]
    fn test_compute_scope_module() {
        assert_eq!(
            compute_scope("mymod", false, &ScopeType::Local, "mymod"),
            "mymod"
        );
    }

    #[test]
    fn test_compute_scope_class() {
        assert_eq!(
            compute_scope("mymod.Foo", false, &ScopeType::Local, "mymod"),
            "mymod.Foo"
        );
    }

    #[test]
    fn test_compute_scope_function() {
        assert_eq!(
            compute_scope("mymod.foo", true, &ScopeType::Local, "mymod"),
            "mymod.foo.<locals>"
        );
    }

    #[test]
    fn test_compute_scope_global() {
        assert_eq!(
            compute_scope("mymod.foo", true, &ScopeType::Global, "mymod"),
            "mymod"
        );
    }

    #[test]
    fn test_compute_scope_nonlocal() {
        assert_eq!(
            compute_scope("mymod.foo.bar", true, &ScopeType::Nonlocal, "mymod"),
            "mymod.foo"
        );
    }
}
