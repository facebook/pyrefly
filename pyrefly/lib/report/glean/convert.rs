/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::env::current_dir;
use std::slice;
use std::sync::Arc;

use dupe::Dupe;
use num_traits::ToPrimitive;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::docstring::Docstring;
use pyrefly_python::dunder;
use pyrefly_python::module_name::ModuleName;
use pyrefly_types::types::Union;
use pyrefly_util::visit::Visit;
use regex::RegexBuilder;
use ruff_python_ast::Decorator;
use ruff_python_ast::ExceptHandler;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprList;
use ruff_python_ast::ExprName;
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
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use starlark_map::small_set::SmallSet;

use crate::module::module_info::ModuleInfo;
use crate::report::glean::facts::*;
use crate::report::glean::schema::*;
use crate::state::lsp::FindPreference;
use crate::state::state::Transaction;
use crate::types::types::Type;

trait UnwrapOrDefaultAndLog<T: Default> {
    fn unwrap_or_default_and_log(self) -> T;
}

impl<T: Default> UnwrapOrDefaultAndLog<T> for Option<T> {
    #[track_caller]
    fn unwrap_or_default_and_log(self) -> T {
        match self {
            Some(x) => x,
            None => {
                tracing::warn!("unexpected None");
                Default::default()
            }
        }
    }
}

const TYPE_SEPARATORS: [char; 12] = [',', '|', '[', ']', '{', '}', '(', ')', '=', ':', '\'', '"'];

fn hash(x: &[u8]) -> String {
    // Glean uses blake3
    blake3::hash(x).to_string()
}

fn join_names(base_name: &str, name: &str) -> String {
    if base_name.is_empty() {
        name.to_owned()
    } else {
        base_name.to_owned() + "." + name
    }
}

fn all_modules(module_name: ModuleName) -> impl Iterator<Item = String> {
    module_name
        .components()
        .into_iter()
        .scan("".to_owned(), |module, component| {
            *module = join_names(module, component.as_str());
            Some(module.to_owned())
        })
}

fn all_modules_with_range(module_id: &Identifier) -> impl Iterator<Item = (String, TextRange)> {
    let offset = module_id.range().start();
    all_modules(ModuleName::from_name(module_id.id())).map(move |name| {
        let range = TextRange::at(offset, TextSize::try_from(name.len()).unwrap());
        (name, range)
    })
}

fn range_without_decorators(range: TextRange, decorators: &[Decorator]) -> TextRange {
    let decorators_range = decorators
        .first()
        .map(|first| first.range().cover(decorators.last().unwrap().range()));

    decorators_range.map_or(range, |x| range.add_start(x.len() + TextSize::from(1)))
}

fn to_span(range: TextRange) -> src::ByteSpan {
    src::ByteSpan {
        start: range.start().to_u32().into(),
        length: range.len().to_u32().into(),
    }
}

fn to_span_with_offset(range: TextRange, offset: Option<TextSize>) -> src::ByteSpan {
    let start = range.start();
    src::ByteSpan {
        start: offset.map_or(start, |x| start - x).to_u32().into(),
        length: range.len().to_u32().into(),
    }
}

fn file_fact(module_info: &ModuleInfo) -> src::File {
    let file_path = module_info.path().as_path();
    let relative_path = file_path
        .strip_prefix(current_dir().unwrap_or_default())
        .unwrap_or(file_path)
        .to_str()
        .unwrap();

    src::File::new(relative_path.to_owned())
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
enum ScopeType {
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
    xrefs_by_target: HashMap<python::Name, Vec<src::ByteSpan>>,
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
    module_name: ModuleName,
    facts: Facts,
    names: HashSet<Arc<String>>,
    locations_fqnames: HashMap<TextSize, Arc<String>>,
    import_names: HashMap<Arc<String>, Arc<String>>,
}

struct AssignInfo<'a> {
    range: TextRange,
    annotation: Option<&'a Expr>,
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
            xrefs_by_target: HashMap::new(),
            declaration_docstrings: vec![],
            name_to_sname: vec![],
        }
    }
}

impl GleanState<'_> {
    fn new<'a>(transaction: &'a Transaction<'a>, handle: &'a Handle) -> GleanState<'a> {
        let module_info = &transaction.get_module_info(handle).unwrap();
        GleanState {
            transaction,
            handle,
            module: module_info.clone(),
            module_name: module_info.name(),
            facts: Facts::new(
                file_fact(module_info),
                python::Module::new(python::Name::new(module_info.name().to_string())),
            ),
            names: HashSet::new(),
            locations_fqnames: HashMap::new(),
            import_names: HashMap::new(),
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
            r"^.*@(?P<tag>(partially-)?generated)( SignedSource<<(?P<sign>[0-9a-f]+)>>)?$",
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

        for name in all_modules(self.module_name) {
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
            !lined_buffer.is_ascii() || lined_buffer.contents().contains('\t'),
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

    fn record_name_with_position(&mut self, name: String, position: TextSize) -> python::Name {
        let arc_name = self.record_name(name.clone());
        self.locations_fqnames.insert(position, arc_name.dupe());
        python::Name::new(name)
    }

    fn record_name_for_import(&mut self, name: String, import_name: &str) {
        let arc_name = self.record_name(name);
        let arc_import_name = Arc::new(import_name.to_owned());
        self.import_names.insert(arc_name.dupe(), arc_import_name);
    }

    fn make_fq_name_for_declaration(
        &mut self,
        name: &Identifier,
        container: &python::DeclarationContainer,
        scope_type: ScopeType,
    ) -> python::Name {
        let container_name = match container {
            python::DeclarationContainer::module(module) => &module.key.name,
            python::DeclarationContainer::cls(cls) => &cls.key.name,
            python::DeclarationContainer::func(func) => &func.key.name,
        };
        let container_str = container_name.key.as_str();
        let scope = match scope_type {
            ScopeType::Global => self.module_name.to_string(),
            ScopeType::Nonlocal => {
                let mut parts: Vec<&str> = container_str.split(".").collect();
                parts.pop();
                parts.join(".")
            }
            ScopeType::Local => {
                if let python::DeclarationContainer::func(_) = container {
                    container_str.to_owned() + ".<locals>"
                } else {
                    container_str.to_owned()
                }
            }
        };
        if !self.names.contains(&scope) {
            self.record_name(scope.clone());
        }
        self.record_name_with_position(join_names(&scope, name), name.range.start())
    }

    fn fq_name_for_xref_definition(
        &self,
        def_range: TextRange,
        module: &ModuleInfo,
    ) -> Option<String> {
        let local_name = module.code_at(def_range);
        let module_name = module.name();
        if module_name == ModuleName::builtins() {
            Some(local_name.to_owned())
        } else if module_name == self.module_name {
            self.locations_fqnames
                .get(&def_range.start())
                .map(|x| (**x).clone())
        } else {
            let local_name = module.code_at(def_range);
            let fq_name = if local_name.is_empty() {
                module_name.to_string()
            } else {
                join_names(module_name.as_str(), local_name)
            };
            Some(fq_name)
        }
    }

    fn fq_names_for_expr(&self, expr: &Expr) -> Vec<String> {
        match expr {
            Expr::Subscript(expr_subscript) => self.fq_names_for_expr(&expr_subscript.value),
            Expr::Attribute(attr) => self.fq_names_for_attribute(attr),
            Expr::Name(name) => self.fq_name_for_expr_name(name).map_or(vec![], |x| vec![x]),
            _ => vec![],
        }
    }

    fn fq_name_for_expr_name(&self, expr_name: &ExprName) -> Option<String> {
        let identifier = Ast::expr_name_identifier(expr_name.clone());
        self.fq_name_for_name_use(identifier)
    }

    fn fq_name_for_name_use(&self, identifier: Identifier) -> Option<String> {
        let definition = self.transaction.find_definition_for_name_use(
            self.handle,
            &identifier,
            FindPreference::default(),
        );

        definition
            .and_then(|def| self.fq_name_for_xref_definition(def.definition_range, &def.module))
    }

    fn fq_name_for_type(&self, ty: Type, range: TextRange) -> Option<String> {
        match ty {
            Type::Module(module) => Some(module.parts().join(".")),
            Type::None => Some("None".to_owned()),
            Type::Type(inner_ty) => self.fq_name_for_type(*inner_ty, range),
            Type::SpecialForm(x) => {
                let identifier = Identifier::new(x.to_string(), range);
                self.fq_name_for_name_use(identifier)
            }
            _ => ty
                .qname()
                .and_then(|qname| self.fq_name_for_xref_definition(qname.range(), qname.module())),
        }
    }

    fn get_xrefs_types_for_str_lit(&self, expr: &ExprStringLiteral) -> Vec<(String, TextRange)> {
        let sep_indexes: Vec<usize> = self
            .module
            .code_at(expr.range())
            .match_indices(|x: char| x.is_whitespace() || TYPE_SEPARATORS.contains(&x))
            .map(|m| m.0)
            .collect();

        let ranges = (1..sep_indexes.len())
            .map(|i| {
                let start = TextSize::try_from(sep_indexes[i - 1] + 1).ok().unwrap();
                let end = TextSize::try_from(sep_indexes[i]).ok().unwrap();
                TextRange::new(start, end) + expr.range().start()
            })
            .filter(|range| !range.is_empty());

        if let Some(answers) = self.transaction.get_answers(self.handle) {
            ranges
                .filter_map(|range| {
                    if let Some(ty) = answers.get_type_trace(range)
                        && let Some(name) = self.fq_name_for_type(ty, range)
                    {
                        Some((name, range))
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            vec![]
        }
    }

    fn fq_names_for_attribute(&self, expr_attr: &ExprAttribute) -> Vec<String> {
        let name = &expr_attr.attr;
        let base_expr = expr_attr.value.as_ref();

        let base_types = if let Some(answers) = self.transaction.get_answers(self.handle)
            && let Some(base_type) = answers.get_type_trace(base_expr.range())
        {
            self.transaction
                .ad_hoc_solve(self.handle, |solver| match base_type {
                    Type::Union(box Union { members: tys, .. }) | Type::Intersect(box (tys, _)) => {
                        tys.into_iter()
                            .filter(|ty: &Type| {
                                solver
                                    .completions(ty.clone(), Some(name.id()), false)
                                    .into_iter()
                                    .any(|attr| &attr.name == name.id())
                            })
                            .collect()
                    }
                    ty => vec![ty],
                })
                .unwrap_or_default()
        } else {
            vec![]
        };

        if base_types.is_empty() {
            let base_fq_names = self.fq_names_for_expr(base_expr);
            base_fq_names
                .into_iter()
                .map(|base| join_names(&base, name))
                .collect()
        } else {
            base_types
                .into_iter()
                .filter_map(|ty| {
                    self.fq_name_for_type(ty, base_expr.range())
                        .as_deref()
                        .map(|base| join_names(base, name))
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
                .flat_map(|expr| {
                    self.fq_names_for_expr(expr)
                        .into_iter()
                        .map(|name| python::ClassDeclaration::new(python::Name::new(name), None))
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

    fn make_xrefs(&self, expr: &Expr, offset: Option<TextSize>) -> Vec<python::XRefViaName> {
        let xrefs = match expr {
            Expr::Attribute(attr) => {
                if attr.ctx.is_load() {
                    self.fq_names_for_attribute(attr)
                        .into_iter()
                        .map(|name| (name, attr.attr.range()))
                        .collect()
                } else {
                    vec![]
                }
            }
            Expr::Name(name) => {
                if name.ctx.is_load() {
                    self.fq_name_for_expr_name(name)
                        .map_or(vec![], |x| vec![(x, name.range())])
                } else {
                    vec![]
                }
            }
            Expr::StringLiteral(str_lit) => self.get_xrefs_types_for_str_lit(str_lit),
            Expr::BooleanLiteral(bool_lit) => {
                let name = if bool_lit.value { "True" } else { "False" };
                vec![(name.to_owned(), bool_lit.range())]
            }
            Expr::NoneLiteral(none) => vec![("None".to_owned(), none.range())],
            _ => {
                vec![]
            }
        };

        xrefs
            .into_iter()
            .map(|(name, range)| python::XRefViaName {
                target: python::Name::new(name),
                source: to_span_with_offset(range, offset),
            })
            .collect()
    }

    fn add_xref(&mut self, xref: python::XRefViaName) {
        if let Some(spans) = self.facts.xrefs_by_target.get_mut(&xref.target) {
            spans.push(xref.source.clone());
        } else {
            self.facts
                .xrefs_by_target
                .insert(xref.target.clone(), vec![xref.source.clone()]);
        }
        self.facts.xrefs_via_name.push(xref);
    }

    fn xrefs_for_type_info(
        &self,
        expr: &Expr,
        xrefs: &mut Vec<python::XRefViaName>,
        offset: TextSize,
    ) {
        xrefs.extend(self.make_xrefs(expr, Some(offset)));

        expr.recurse(&mut |x| self.xrefs_for_type_info(x, xrefs, offset));
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

    fn type_info(&self, annotation: Option<&Expr>) -> Option<python::TypeInfo> {
        annotation.map(|type_annotation| {
            let mut xrefs = vec![];
            let range = type_annotation.range();
            type_annotation
                .visit(&mut |expr| self.xrefs_for_type_info(expr, &mut xrefs, range.start()));
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
    ) -> python::Parameter {
        let type_info: Option<python::TypeInfo> = self.type_info(param.annotation());
        let fqname =
            self.make_fq_name_for_declaration(&param.name, &context.container, ScopeType::Local);
        decl_infos.push(self.variable_info(
            fqname,
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
    ) -> python::Parameter {
        let value: Option<String> = parameter_with_default
            .default
            .as_ref()
            .map(|x| self.module.code_at(x.range()).to_owned());
        self.parameter_info(
            &parameter_with_default.parameter,
            value,
            context,
            decl_infos,
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
        let args = params
            .args
            .iter()
            .map(|x| self.parameter_with_default_info(x, func_ctx, &mut decl_infos))
            .collect();

        let pos_only_args = params
            .posonlyargs
            .iter()
            .map(|x| self.parameter_with_default_info(x, func_ctx, &mut decl_infos))
            .collect();

        let kwonly_args = params
            .kwonlyargs
            .iter()
            .map(|x| self.parameter_with_default_info(x, func_ctx, &mut decl_infos))
            .collect();

        let star_arg = params
            .vararg
            .as_ref()
            .map(|x| self.parameter_info(x.as_ref(), None, func_ctx, &mut decl_infos));

        let star_kwarg = params
            .kwarg
            .as_ref()
            .map(|x| self.parameter_info(x.as_ref(), None, func_ctx, &mut decl_infos));

        let func_definition = python::FunctionDefinition::new(
            func_declaration.clone(),
            func.is_async,
            self.type_info(func.returns.as_ref().map(|x| x.as_ref())),
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

    fn add_xrefs_for_explicit_exports(&mut self, exprs_list: &[Expr]) {
        let exports = exprs_list
            .iter()
            .filter_map(|e| e.as_string_literal_expr())
            .map(|str_lit| (str_lit.value.to_str(), str_lit.range()));

        for (local_name, source) in exports {
            let name = join_names(self.module_name.as_str(), local_name);
            if self.names.contains(&name) {
                let fqname = self
                    .import_names
                    .get(&name)
                    .map(|n| (**n).clone())
                    .unwrap_or(name);

                self.add_xref(python::XRefViaName {
                    target: python::Name::new(fqname),
                    source: to_span(source),
                });
            }
        }
    }

    fn variable_facts(
        &mut self,
        expr: &Expr,
        info: &AssignInfo,
        ctx: &NodeContext,
        next: Option<&Stmt>,
        def_infos: &mut Vec<DeclarationInfo>,
    ) {
        if let Some(name) = expr.as_name_expr() {
            let scope_type = if ctx.globals.contains(&name.id) {
                ScopeType::Global
            } else if ctx.nonlocals.contains(&name.id) {
                ScopeType::Nonlocal
            } else {
                ScopeType::Local
            };

            let name_id = Ast::expr_name_identifier(name.clone());
            let fqname = self.make_fq_name_for_declaration(&name_id, &ctx.container, scope_type);
            let docstring_range =
                next.and_then(|stmt| Docstring::range_from_stmts(slice::from_ref(stmt)));
            def_infos.push(self.variable_info(
                fqname,
                info.range,
                self.type_info(info.annotation),
                docstring_range,
                ctx,
            ));

            if name.id() == &dunder::ALL
                && let Some(value) = info.value
                && let Expr::List(ExprList { elts, .. }) = value
            {
                self.add_xrefs_for_explicit_exports(elts);
            }
        }
        expr.recurse(&mut |expr| self.variable_facts(expr, info, ctx, next, def_infos));
    }

    fn find_fqname_definition_at_position(&self, position: TextSize) -> Vec<String> {
        let definitions =
            self.transaction
                .find_definition(self.handle, position, FindPreference::default());

        definitions
            .into_iter()
            .filter_map(|def| self.fq_name_for_xref_definition(def.definition_range, &def.module))
            .collect()
    }

    fn make_import_fact(
        &mut self,
        from_name: &str,
        from_name_range: TextRange,
        as_name: &str,
        as_name_range: TextRange,
        top_level_declaration: &python::Declaration,
        resolve_original_name: bool,
    ) -> DeclarationInfo {
        let as_name_fqname = join_names(self.module_name.as_str(), as_name);
        let from_name_fact = python::Name::new(from_name.to_owned());
        let as_name_fact = python::Name::new(as_name_fqname.clone());
        let original_name = if resolve_original_name
            && let Some(name) = self
                .find_fqname_definition_at_position(from_name_range.start())
                .first()
        {
            name.clone()
        } else {
            from_name.to_owned()
        };

        self.record_name_with_position(from_name.to_owned(), as_name_range.start());
        self.record_name_for_import(as_name_fqname, &original_name);
        self.add_xref(python::XRefViaName {
            target: python::Name::new(original_name),
            source: to_span(from_name_range),
        });
        let import_fact = python::ImportStatement::new(from_name_fact, as_name_fact);

        DeclarationInfo {
            declaration: python::Declaration::imp(import_fact),
            decl_span: to_span(from_name_range),
            definition: None,
            def_span: None,
            top_level_decl: top_level_declaration.clone(),
            docstring_range: None,
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
                if let Some(as_name) = &import.asname {
                    let mut it = all_modules_with_range(from_name).peekable();
                    while let Some((module, range)) = it.next() {
                        if it.peek().is_some() {
                            self.record_name(module.clone());
                            self.add_xref(python::XRefViaName {
                                target: python::Name::new(module),
                                source: to_span(range),
                            });
                        }
                    }
                    vec![self.make_import_fact(
                        from_name.as_str(),
                        from_name.range(),
                        as_name.as_str(),
                        as_name.range(),
                        top_level_declaration,
                        true,
                    )]
                } else {
                    all_modules_with_range(from_name)
                        .map(|(module, range)| {
                            self.make_import_fact(
                                &module,
                                range,
                                &module,
                                range,
                                top_level_declaration,
                                false,
                            )
                        })
                        .collect()
                }
            })
            .collect()
    }

    fn get_from_module(&mut self, import_from: &StmtImportFrom) -> Option<String> {
        let from_module_name = import_from.module.as_ref().map(|x| x.id());

        let (from_module, dots_range) = if import_from.level > 0 {
            let module = self
                .module_name
                .new_maybe_relative(
                    self.module.path().is_init(),
                    import_from.level,
                    from_module_name,
                )
                .map(|x| x.to_string());

            let range = self
                .module
                .code_at(import_from.range())
                .match_indices(['.'])
                .next()
                .map(|(s, _)| {
                    let offset = TextSize::try_from(s).unwrap();
                    let len = TextSize::from(import_from.level);
                    TextRange::at(import_from.range().start() + offset, len)
                });
            (module, range)
        } else {
            (from_module_name.map(|x| x.to_string()), None)
        };

        let from_module_range = import_from.module.as_ref().map(|id| id.range());
        let range = from_module_range
            .and_then(|x| dots_range.map(|y| x.cover(y)))
            .or(from_module_range)
            .or(dots_range);

        self.add_xref(python::XRefViaName {
            target: python::Name::new(from_module.clone().unwrap_or_default_and_log()),
            source: to_span(range.unwrap_or_default_and_log()),
        });
        from_module
    }

    fn import_from_facts(
        &mut self,
        import_from: &StmtImportFrom,
        top_level_declaration: &python::Declaration,
    ) -> Vec<DeclarationInfo> {
        let from_module = self.get_from_module(import_from);

        let mut decl_infos = vec![];
        for import in &import_from.names {
            let from_name = &import.name;
            let star_import = "*";

            if *from_name.id.as_str() == *star_import {
                let import_star = python::ImportStarStatement::new(
                    python::Name::new(from_module.clone().unwrap_or_default()),
                    self.facts.module.clone(),
                );
                self.facts
                    .import_star_locations
                    .push(python::ImportStarLocation::new(
                        import_star,
                        self.facts.file.clone(),
                        to_span(import_from.range),
                    ));
            } else {
                let from_name_string = from_module
                    .as_deref()
                    .map_or(from_name.id().to_string(), |x| {
                        join_names(x, from_name.id())
                    });
                let as_name = import.asname.as_ref().unwrap_or(from_name);
                decl_infos.push(self.make_import_fact(
                    &from_name_string,
                    from_name.range,
                    as_name.as_str(),
                    as_name.range,
                    top_level_declaration,
                    true,
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
        let callee_names = self.fq_names_for_expr(call.func.as_ref());
        for callee_name in callee_names {
            self.facts
                .callee_to_callers
                .push(python::CalleeToCaller::new(
                    python::Name::new(callee_name),
                    caller_fact.clone(),
                ));
        }
    }

    fn generate_facts_from_exprs(&mut self, expr: &Expr, container: &python::DeclarationContainer) {
        if let Some(call) = expr.as_call_expr() {
            self.file_call_facts(call);
            if let python::DeclarationContainer::func(caller) = container {
                self.callee_to_caller_facts(call, caller);
            }
        };
        for xref in self.make_xrefs(expr, None) {
            self.add_xref(xref);
        }
        expr.recurse(&mut |s| self.generate_facts_from_exprs(s, container));
    }

    fn visit_exprs(&mut self, node: &impl Visit<Expr>, container: &python::DeclarationContainer) {
        node.visit(&mut |expr| self.generate_facts_from_exprs(expr, container));
    }

    fn generate_facts(&mut self, ast: &Vec<Stmt>, range: TextRange) {
        self.module_facts(range);
        let mut nodes = VecDeque::new();

        let root_context = NodeContext {
            container: Arc::new(python::DeclarationContainer::module(self.module_fact())),
            top_level_decl: Arc::new(python::Declaration::module(self.module_fact())),
            globals: Arc::new(SmallSet::new()),
            nonlocals: Arc::new(SmallSet::new()),
        };
        ast.visit(&mut |x| nodes.push_back((x, root_context.clone())));

        while let Some((node, node_context)) = nodes.pop_front() {
            // Get next node if in same level. Needed to compute docstring range for variables
            let next = nodes
                .front()
                .filter(|(_, ctx)| ctx.container == node_context.container)
                .map(|(x, _)| *x);
            let children_context = self.process_statement(node, next, &node_context);
            node.recurse(&mut |x| nodes.push_back((x, children_context.clone())));
        }
    }

    fn process_statement(
        &mut self,
        stmt: &Stmt,
        next: Option<&Stmt>,
        context: &NodeContext,
    ) -> NodeContext {
        let container = &context.container;
        let top_level_decl = &*context.top_level_decl;

        let mut this_ctx = context.clone();

        let mut decl_infos = vec![];
        match stmt {
            Stmt::ClassDef(cls) => {
                let cls_fq_name =
                    self.make_fq_name_for_declaration(&cls.name, container, ScopeType::Local);
                let cls_declaration = python::ClassDeclaration::new(cls_fq_name, None);
                let decl_info = self.class_facts(cls, cls_declaration.clone(), context);
                self.visit_exprs(&cls.decorator_list, container);
                self.visit_exprs(&cls.type_params, container);
                self.visit_exprs(&cls.arguments, container);
                if let python::Declaration::module(_) = top_level_decl {
                    this_ctx.top_level_decl = Arc::new(decl_info.declaration.clone());
                }
                this_ctx.container = Arc::new(python::DeclarationContainer::cls(cls_declaration));
                (this_ctx.globals, this_ctx.nonlocals) = gather_nonlocal_variables(&cls.body);

                decl_infos.push(decl_info);
            }
            Stmt::FunctionDef(func) => {
                let func_fq_name =
                    self.make_fq_name_for_declaration(&func.name, container, ScopeType::Local);
                let func_declaration = python::FunctionDeclaration::new(func_fq_name);
                if let python::Declaration::module(_) = top_level_decl {
                    this_ctx.top_level_decl =
                        Arc::new(python::Declaration::func(func_declaration.clone()));
                }
                this_ctx.container =
                    Arc::new(python::DeclarationContainer::func(func_declaration.clone()));
                (this_ctx.globals, this_ctx.nonlocals) = gather_nonlocal_variables(&func.body);
                let mut func_decl_infos =
                    self.function_facts(func, func_declaration, context, &this_ctx);

                self.visit_exprs(&func.decorator_list, container);
                self.visit_exprs(&func.type_params, container);
                self.visit_exprs(&func.parameters, container);
                self.visit_exprs(&func.returns, container);

                decl_infos.append(&mut func_decl_infos);
            }
            Stmt::Assign(assign) => {
                let info = AssignInfo {
                    range: assign.range(),
                    annotation: None,
                    value: Some(assign.value.as_ref()),
                };
                assign.targets.visit(&mut |target| {
                    self.variable_facts(target, &info, context, next, &mut decl_infos)
                });
                self.visit_exprs(&assign.value, container);
            }
            Stmt::AnnAssign(assign) => {
                let info = AssignInfo {
                    range: assign.range(),
                    annotation: Some(&assign.annotation),
                    value: assign.value.as_ref().map(|v| v.as_ref()),
                };
                self.variable_facts(&assign.target, &info, context, next, &mut decl_infos);
                self.visit_exprs(&assign.annotation, container);
                self.visit_exprs(&assign.value, container);
            }
            Stmt::AugAssign(assign) => {
                let info = AssignInfo {
                    range: assign.range(),
                    annotation: None,
                    value: Some(assign.value.as_ref()),
                };
                self.variable_facts(&assign.target, &info, context, next, &mut decl_infos);
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
                stmt_for.target.visit(&mut |target| {
                    let info = AssignInfo {
                        range: target.range(),
                        annotation: None,
                        value: None,
                    };
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
                            annotation: None,
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
                        if let Some(name) = &x.name {
                            let fq_name = self.make_fq_name_for_declaration(
                                name,
                                container,
                                ScopeType::Local,
                            );
                            decl_infos.push(self.variable_info(
                                fq_name,
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

        this_ctx
    }
}

impl Glean {
    pub fn new(transaction: &Transaction, handle: &Handle) -> Self {
        let ast = &*transaction.get_ast(handle).unwrap();
        let mut glean_state = GleanState::new(transaction, handle);

        glean_state.record_name("".to_owned());
        let file_language_fact =
            src::FileLanguage::new(glean_state.file_fact(), src::Language::Python);
        let digest_fact = glean_state.digest_fact();
        let file_lines = glean_state.file_lines_fact();
        glean_state.generate_facts(&ast.body, ast.range());

        let file_fact = glean_state.file_fact();
        let gencode_fact = glean_state.gencode_fact();

        let facts = glean_state.facts;

        let xrefs_via_name_by_file_fact =
            python::XRefsViaNameByFile::new(file_fact.clone(), facts.xrefs_via_name.to_owned());

        let xrefs_by_target: Vec<python::XRefsViaNameByTarget> = facts
            .xrefs_by_target
            .into_iter()
            .map(|(target, spans)| {
                python::XRefsViaNameByTarget::new(
                    target.to_owned(),
                    file_fact.clone(),
                    spans.to_owned(),
                )
            })
            .collect();

        let entries = vec![
            GleanEntry::SchemaId {
                schema_id: builtin::SCHEMA_ID.to_owned(),
            },
            GleanEntry::Predicate {
                predicate: python::Name::GLEAN_name(),
                facts: vec![json(python::Name::new("".to_owned()))],
            },
            GleanEntry::Predicate {
                predicate: python::Module::GLEAN_name(),
                facts: facts.modules.into_iter().map(json).collect(),
            },
            GleanEntry::Predicate {
                predicate: src::FileLanguage::GLEAN_name(),
                facts: vec![json(file_language_fact)],
            },
            GleanEntry::Predicate {
                predicate: src::FileLines::GLEAN_name(),
                facts: vec![json(file_lines)],
            },
            GleanEntry::Predicate {
                predicate: digest::FileDigest::GLEAN_name(),
                facts: vec![json(digest_fact)],
            },
            GleanEntry::Predicate {
                predicate: python::DeclarationLocation::GLEAN_name(),
                facts: facts.decl_locations.into_iter().map(json).collect(),
            },
            GleanEntry::Predicate {
                predicate: python::DefinitionLocation::GLEAN_name(),
                facts: facts.def_locations.into_iter().map(json).collect(),
            },
            GleanEntry::Predicate {
                predicate: python::ImportStarLocation::GLEAN_name(),
                facts: facts.import_star_locations.into_iter().map(json).collect(),
            },
            GleanEntry::Predicate {
                predicate: python::FileCall::GLEAN_name(),
                facts: facts.file_calls.into_iter().map(json).collect(),
            },
            GleanEntry::Predicate {
                predicate: python::CalleeToCaller::GLEAN_name(),
                facts: facts.callee_to_callers.into_iter().map(json).collect(),
            },
            GleanEntry::Predicate {
                predicate: python::ContainingTopLevelDeclaration::GLEAN_name(),
                facts: facts
                    .containing_top_level_declarations
                    .into_iter()
                    .map(json)
                    .collect(),
            },
            GleanEntry::Predicate {
                predicate: python::XRefsViaNameByFile::GLEAN_name(),
                facts: vec![json(xrefs_via_name_by_file_fact)],
            },
            GleanEntry::Predicate {
                predicate: python::XRefsViaNameByTarget::GLEAN_name(),
                facts: xrefs_by_target.into_iter().map(json).collect(),
            },
            GleanEntry::Predicate {
                predicate: python::DeclarationDocstring::GLEAN_name(),
                facts: facts.declaration_docstrings.into_iter().map(json).collect(),
            },
            GleanEntry::Predicate {
                predicate: python::NameToSName::GLEAN_name(),
                facts: facts.name_to_sname.into_iter().map(json).collect(),
            },
            GleanEntry::Predicate {
                predicate: gencode::GenCode::GLEAN_name(),
                facts: gencode_fact.map_or(vec![], |f| vec![json(f)]),
            },
        ];
        Glean { entries }
    }
}
