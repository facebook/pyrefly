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
use pyrefly_python::module_name::ModuleName;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Decorator;
use ruff_python_ast::ExceptHandler;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprName;
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

fn hash(x: &[u8]) -> String {
    // Glean uses blake3
    blake3::hash(x).to_string()
}

fn join_names(base_name: &str, name: &str) -> String {
    base_name.to_owned() + "." + name
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

    fn all_modules(&mut self, module_name: ModuleName) -> Vec<ModuleName> {
        let mut module_names = vec![];
        let components = module_name.components();
        let mut module = None;
        for component in components.into_iter() {
            let name = module.map_or(ModuleName::from_name(&component), |x: ModuleName| {
                x.append(&component)
            });
            self.record_name(name.to_string(), None);
            module = Some(name);
            module_names.push(name);
        }

        module_names
    }

    fn module_facts(&mut self, range: TextRange) {
        let module_docstring_range = self.transaction.get_module_docstring_range(self.handle);
        let components = self.module_name.components();
        let mut module = None;

        for component in components.into_iter() {
            let name = module.map_or(ModuleName::from_name(&component), |x: ModuleName| {
                x.append(&component)
            });
            self.record_name(name.to_string(), None);
            self.facts
                .modules
                .push(python::Module::new(python::Name::new(name.to_string())));
            module = Some(name);
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

    fn record_name(&mut self, name: String, position: Option<TextSize>) -> python::Name {
        let arc_name = Arc::new(name.clone());
        if self.names.insert(arc_name.dupe()) {
            self.facts.name_to_sname.push(python::NameToSName::new(
                python::Name::new(name.clone()),
                create_sname(&name),
            ));
        }
        position.map(|x| self.locations_fqnames.insert(x, arc_name.dupe()));

        python::Name::new(name)
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
            self.record_name(scope.clone(), None);
        }
        self.record_name(join_names(&scope, name), Some(name.range.start()))
    }

    fn make_fq_names_for_expr(&self, expr: &Expr) -> Vec<python::Name> {
        self.fq_names_for_name_or_attr(expr)
            .into_iter()
            .map(python::Name::new)
            .collect()
    }

    fn fq_name_for_xref_definition(
        &self,
        name: &Name,
        def_range: TextRange,
        module: &ModuleInfo,
    ) -> Option<String> {
        let module_name = module.name();
        if module_name == ModuleName::builtins() {
            Some(name.to_string())
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

    fn fq_names_for_name_or_attr(&self, expr: &Expr) -> Vec<String> {
        match expr {
            Expr::Attribute(attr) => self.fq_names_for_attribute(attr),
            Expr::Name(name) => self.fq_name_for_name_use(name).map_or(vec![], |x| vec![x]),
            _ => vec![],
        }
    }

    fn fq_name_for_name_use(&self, expr_name: &ExprName) -> Option<String> {
        let name = expr_name.id();
        let identifier = Ast::expr_name_identifier(expr_name.clone());

        let definition = self.transaction.find_definition_for_name_use(
            self.handle,
            &identifier,
            &FindPreference {
                jump_through_renamed_import: false,
                ..Default::default()
            },
        );

        definition.and_then(|def| {
            self.fq_name_for_xref_definition(name, def.definition_range, &def.module)
        })
    }

    fn fq_name_for_type(&self, ty: Type) -> Option<String> {
        if let Some(module) = ty.as_module() {
            Some(module.parts().join("."))
        } else {
            ty.qname().and_then(|qname| {
                self.fq_name_for_xref_definition(qname.id(), qname.range(), qname.module())
            })
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
                    Type::Union(tys) | Type::Intersect(tys) => tys
                        .into_iter()
                        .filter(|ty: &Type| {
                            solver
                                .completions(ty.clone(), Some(name.id()), false)
                                .into_iter()
                                .any(|attr| &attr.name == name.id())
                        })
                        .collect(),
                    ty => vec![ty],
                })
                .unwrap_or_default()
        } else {
            vec![]
        };

        if base_types.is_empty() {
            let base_fq_names = self.fq_names_for_name_or_attr(base_expr);
            base_fq_names.into_iter().map(|base| base + name).collect()
        } else {
            base_types
                .into_iter()
                .filter_map(|ty| {
                    self.fq_name_for_type(ty)
                        .as_deref()
                        .map(|base| join_names(base, name))
                })
                .collect()
        }
    }

    fn make_decorators(&self, decorators: &[Decorator]) -> Option<Vec<String>> {
        let lined_buffer = self.module.lined_buffer();
        let glean_decorators: Vec<String> = decorators
            .iter()
            .map(|x| lined_buffer.code_at(x.range()).to_owned())
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
                .flat_map(|expr| self.fq_names_for_name_or_attr(expr))
                .map(|name| python::ClassDeclaration::new(python::Name::new(name), None))
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
        let (names, range) = match expr {
            Expr::Attribute(attr) => {
                let fq_names = if attr.ctx.is_load() {
                    self.fq_names_for_attribute(attr)
                } else {
                    vec![]
                };
                (fq_names, attr.attr.range())
            }
            Expr::Name(name) => {
                let fq_names = if name.ctx.is_load() {
                    self.fq_name_for_name_use(name).map_or(vec![], |x| vec![x])
                } else {
                    vec![]
                };
                (fq_names, name.range())
            }
            Expr::NoneLiteral(none) => (vec!["None".to_owned()], none.range()),
            _ => (vec![], expr.range()),
        };

        names
            .into_iter()
            .map(|name| python::XRefViaName {
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
        let lined_buffer = self.module.lined_buffer();
        let separators = [',', '|', '[', ']', '{', '}', '(', ')', '=', ':'];
        let parts: Vec<&str> = lined_buffer
            .code_at(range)
            .split_whitespace()
            .flat_map(|x| x.split_inclusive(separators))
            .flat_map(|x| {
                if x.ends_with(separators) {
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
        let lined_buffer = self.module.lined_buffer();
        let value: Option<String> = parameter_with_default
            .default
            .as_ref()
            .map(|x| lined_buffer.code_at(x.range()).to_owned());
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

    fn variable_facts(
        &mut self,
        expr: &Expr,
        range: TextRange,
        annotation: Option<&Expr>,
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
                range,
                self.type_info(annotation),
                docstring_range,
                ctx,
            ));
        }
        expr.recurse(&mut |expr| {
            self.variable_facts(expr, range, annotation, ctx, next, def_infos)
        });
    }

    fn make_import_fact(
        &mut self,
        from_name: &str,
        from_name_range: TextRange,
        as_name: &str,
        as_name_range: TextRange,
        top_level_declaration: &python::Declaration,
    ) -> DeclarationInfo {
        let as_name_fqname = join_names(self.module_name.as_str(), as_name);
        self.record_name(from_name.to_owned(), Some(as_name_range.start()));
        self.record_name(as_name_fqname.clone(), None);

        let from_name_fact = python::Name::new(from_name.to_owned());
        let as_name_fact = python::Name::new(as_name_fqname);
        self.add_xref(python::XRefViaName {
            target: from_name_fact.clone(),
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

    fn make_import_fact_with_alias(
        &mut self,
        from_name: &str,
        from_name_range: TextRange,
        as_name: &Identifier,
        top_level_declaration: &python::Declaration,
    ) -> DeclarationInfo {
        self.make_import_fact(
            from_name,
            from_name_range,
            as_name.as_str(),
            as_name.range(),
            top_level_declaration,
        )
    }

    fn make_import_facts_for_module(
        &mut self,
        import_module: &Identifier,
        top_level_declaration: &python::Declaration,
    ) -> Vec<DeclarationInfo> {
        let parts = import_module.id().split(".");

        self.all_modules(ModuleName::from_parts(parts))
            .into_iter()
            .map(|module| {
                let range = TextRange::empty(import_module.range().start())
                    .add_end(TextSize::from(module.as_str().len().to_u32().unwrap()));

                self.make_import_fact(
                    module.as_str(),
                    range,
                    module.as_str(),
                    range,
                    top_level_declaration,
                )
            })
            .collect()
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
                    vec![self.make_import_fact_with_alias(
                        from_name.as_str(),
                        from_name.range,
                        as_name,
                        top_level_declaration,
                    )]
                } else {
                    self.make_import_facts_for_module(&import.name, top_level_declaration)
                }
            })
            .collect()
    }

    fn import_from_facts(
        &mut self,
        import_from: &StmtImportFrom,
        top_level_declaration: &python::Declaration,
    ) -> Vec<DeclarationInfo> {
        let from_module_name = import_from.module.as_ref().map(|x| x.id());

        let from_module = if import_from.level > 0 {
            self.module_name
                .new_maybe_relative(
                    self.module.path().is_init(),
                    import_from.level,
                    from_module_name,
                )
                .map(|x| x.to_string())
        } else {
            from_module_name.map(|x| x.to_string())
        };

        let from_module_fact = python::Name::new(from_module.clone().unwrap_or_default());
        if let Some(module) = &import_from.module {
            self.add_xref(python::XRefViaName {
                target: from_module_fact.clone(),
                source: to_span(module.range()),
            });
        }

        let mut decl_infos = vec![];
        for import in &import_from.names {
            let from_name = &import.name;
            let star_import = "*";

            if *from_name.id.as_str() == *star_import {
                let import_star = python::ImportStarStatement::new(
                    from_module_fact.clone(),
                    self.facts.module.clone(),
                );
                self.facts
                    .import_star_locations
                    .push(python::ImportStarLocation::new(
                        import_star,
                        self.facts.file.clone(),
                        to_span(import.range),
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
        let callee_names = self.make_fq_names_for_expr(call.func.as_ref());
        for callee_fact in callee_names {
            self.facts
                .callee_to_callers
                .push(python::CalleeToCaller::new(
                    callee_fact,
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
                assign.targets.visit(&mut |target| {
                    self.variable_facts(
                        target,
                        assign.range(),
                        None,
                        context,
                        next,
                        &mut decl_infos,
                    )
                });
                self.visit_exprs(&assign.value, container);
            }
            Stmt::AnnAssign(assign) => {
                self.variable_facts(
                    &assign.target,
                    assign.range(),
                    Some(&assign.annotation),
                    context,
                    next,
                    &mut decl_infos,
                );
                self.visit_exprs(&assign.annotation, container);
                self.visit_exprs(&assign.value, container);
            }
            Stmt::AugAssign(assign) => {
                self.variable_facts(
                    &assign.target,
                    assign.range(),
                    None,
                    context,
                    next,
                    &mut decl_infos,
                );
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
                    self.variable_facts(
                        target,
                        target.range(),
                        None,
                        context,
                        next,
                        &mut decl_infos,
                    )
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
                        self.variable_facts(
                            target,
                            target.range(),
                            None,
                            context,
                            next,
                            &mut decl_infos,
                        )
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

        glean_state.record_name("".to_owned(), None);
        let file_language_fact =
            src::FileLanguage::new(glean_state.file_fact(), src::Language::Python);
        let digest_fact = glean_state.digest_fact();
        let file_lines = glean_state.file_lines_fact();
        glean_state.generate_facts(&ast.body, ast.range());

        let file_fact = glean_state.file_fact();
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
        ];
        Glean { entries }
    }
}
