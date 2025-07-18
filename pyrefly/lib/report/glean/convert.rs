/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use num_traits::ToPrimitive;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Alias;
use ruff_python_ast::ExceptHandler;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::Identifier;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::alt::answers::Answers;
use crate::binding::bindings::Bindings;
use crate::module::module_info::ModuleInfo;
use crate::report::glean::facts::*;
use crate::report::glean::schema::*;

fn hash(x: &[u8]) -> String {
    // Glean uses blake3
    blake3::hash(x).to_string()
}

struct Facts {
    file: src::File,
    module: python::Module,
    module_name: String,
    decl_locations: Vec<python::DeclarationLocation>,
    def_locations: Vec<python::DefinitionLocation>,
    import_star_locations: Vec<python::ImportStarLocation>,
    file_calls: Vec<python::FileCall>,
    callee_to_callers: Vec<python::CalleeToCaller>,
    containing_top_level_declarations: Vec<python::ContainingTopLevelDeclaration>,
}

fn to_span(range: TextRange) -> src::ByteSpan {
    src::ByteSpan {
        start: range.start().to_u32().into(),
        length: range.len().to_u32().into(),
    }
}

impl Facts {
    fn new(file: src::File, module: python::Module, module_name: String) -> Facts {
        Facts {
            file,
            module,
            module_name,
            decl_locations: vec![],
            def_locations: vec![],
            import_star_locations: vec![],
            file_calls: vec![],
            callee_to_callers: vec![],
            containing_top_level_declarations: vec![],
        }
    }

    fn file_lines_fact(&self, module_info: &ModuleInfo) -> src::FileLines {
        let lined_buffer = module_info.lined_buffer();
        let lens: Vec<u64> = lined_buffer
            .lines()
            .map(|x| x.len().to_u64().unwrap() + 1)
            .collect();
        let ends_in_new_line = lens.len() < lined_buffer.line_count();
        src::FileLines::new(
            self.file.clone(),
            lens,
            ends_in_new_line,
            !lined_buffer.is_ascii() || lined_buffer.contents().contains('\t'),
        )
    }

    fn decl_location_fact(
        &self,
        declaration: python::Declaration,
        range: TextRange,
    ) -> python::DeclarationLocation {
        python::DeclarationLocation::new(declaration.to_owned(), self.file.clone(), to_span(range))
    }

    fn def_location_fact(
        &self,
        definition: python::Definition,
        range: TextRange,
    ) -> python::DefinitionLocation {
        python::DefinitionLocation::new(definition.to_owned(), self.file.clone(), to_span(range))
    }

    fn make_fq_name(&self, name: &Name, module_name: Option<&str>) -> python::Name {
        // TODO(@rubmary) create fully qualified name
        let fq_name = if let Some(module) = module_name {
            module.to_owned() + "." + name
        } else {
            name.to_string()
        };
        python::Name::new(fq_name)
    }

    fn module_facts(
        &mut self,
        module: &python::Module,
        range: TextRange,
        top_level_decl: python::Declaration,
    ) {
        let declaration = python::Declaration::module(module.clone());
        let containing_top_level_decl =
            python::ContainingTopLevelDeclaration::new(declaration.clone(), top_level_decl);

        self.decl_locations
            .push(self.decl_location_fact(declaration, range));

        self.def_locations.push(self.def_location_fact(
            python::Definition::module(python::ModuleDefinition::new(module.clone())),
            range,
        ));

        self.containing_top_level_declarations
            .push(containing_top_level_decl);
    }

    fn class_facts(
        &mut self,
        cls: &StmtClassDef,
        cls_declaration: python::ClassDeclaration,
        container: python::DeclarationContainer,
        top_level_decl: python::Declaration,
    ) {
        let bases = if let Some(arguments) = &cls.arguments {
            arguments
                .args
                .iter()
                .filter_map(|expr| expr.as_name_expr())
                .map(|expr_name| {
                    python::ClassDeclaration::new(self.make_fq_name(&expr_name.id, None), None)
                })
                .collect()
        } else {
            vec![]
        };

        let cls_definition = python::ClassDefinition::new(
            cls_declaration.clone(),
            Some(bases),
            None,
            //TODO(@rubmary) Generate decorators and container for classes
            None,
            Some(container),
        );

        let declaration = python::Declaration::cls(cls_declaration);
        let containing_top_level_decl =
            python::ContainingTopLevelDeclaration::new(declaration.clone(), top_level_decl);

        self.decl_locations
            .push(self.decl_location_fact(declaration, cls.range));
        self.def_locations
            .push(self.def_location_fact(python::Definition::cls(cls_definition), cls.range));
        self.containing_top_level_declarations
            .push(containing_top_level_decl);
    }

    fn function_facts(
        &mut self,
        func: &StmtFunctionDef,
        func_declaration: python::FunctionDeclaration,
        container: python::DeclarationContainer,
        top_level_decl: python::Declaration,
    ) {
        let func_definition = python::FunctionDefinition::new(
            func_declaration.clone(),
            func.is_async,
            // TODO(@rubmary) generate additional fields
            None,
            vec![],
            None,
            None,
            None,
            None,
            None,
            Some(container),
        );

        let declaration = python::Declaration::func(func_declaration);
        let containing_top_level_decl =
            python::ContainingTopLevelDeclaration::new(declaration.clone(), top_level_decl);

        self.decl_locations
            .push(self.decl_location_fact(declaration, func.range));

        self.def_locations
            .push(self.def_location_fact(python::Definition::func(func_definition), func.range));

        self.containing_top_level_declarations
            .push(containing_top_level_decl);
    }

    fn variable_facts(&mut self, expr: &Expr, _type_info: Option<&Expr>) {
        // TODO(@rubmary) add type_info
        if let Some(name) = expr.as_name_expr() {
            let fqname = self.make_fq_name(&name.id, None);
            let variable_declaration = python::VariableDeclaration::new(fqname);
            let variable_definition =
                python::VariableDefinition::new(variable_declaration.clone(), None, None);
            self.decl_locations.push(self.decl_location_fact(
                python::Declaration::variable(variable_declaration),
                name.range,
            ));
            self.def_locations.push(self.def_location_fact(
                python::Definition::variable(variable_definition),
                name.range,
            ))
        }
        expr.recurse(&mut |expr| self.variable_facts(expr, _type_info));
    }

    fn import_facts(
        &mut self,
        imports: &Vec<Alias>,
        from_module_id: &Option<Identifier>,
        top_level_decl: &python::Declaration,
    ) {
        //TODO(@rubmary) Handle level for imports. Ex from ..a import A
        for import in imports {
            let from_module = from_module_id.as_ref().map(|module| module.as_str());
            let from_name = &import.name.id;
            let star_import = "*";

            if *from_name.as_str() == *star_import {
                let import_star = python::ImportStarStatement::new(
                    python::Name::new(from_module.unwrap_or_default().to_owned()),
                    self.module.clone(),
                );
                self.import_star_locations
                    .push(python::ImportStarLocation::new(
                        import_star,
                        self.file.clone(),
                        to_span(import.range()),
                    ));
            } else {
                let as_name = import.asname.as_ref().map_or(from_name, |x| &x.id);

                let import_fact = python::ImportStatement::new(
                    self.make_fq_name(from_name, from_module),
                    self.make_fq_name(as_name, Some(&self.module_name)),
                );

                let declaration = python::Declaration::imp(import_fact);
                let containing_top_level_decl = python::ContainingTopLevelDeclaration::new(
                    declaration.clone(),
                    top_level_decl.clone(),
                );

                self.decl_locations
                    .push(self.decl_location_fact(declaration, import.range()));
                self.containing_top_level_declarations
                    .push(containing_top_level_decl);
            }
        }
    }

    fn file_call_facts(&mut self, call: &ExprCall) {
        let callee_span = to_span(call.func.range());
        // TODO(@rubmary) Generate `argument` value for CallArgument predicate
        let mut call_args: Vec<python::CallArgument> = call
            .arguments
            .args
            .iter()
            .map(|arg| python::CallArgument {
                label: None,
                span: to_span(arg.range()),
                argument: None,
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
                    .map(|id| self.make_fq_name(id.id(), None)),
                span: to_span(keyword.range()),
                argument: None,
            });

        call_args.extend(keyword_args);

        self.file_calls.push(python::FileCall::new(
            self.file.clone(),
            callee_span,
            call_args,
        ));
    }

    fn callee_to_caller_facts(&mut self, call: &ExprCall, caller: &python::FunctionDeclaration) {
        let caller_fact = caller.key.name.clone();
        let callee_name = match call.func.as_ref() {
            Expr::Attribute(attr) => Some(attr.attr.id()),
            Expr::Name(expr_name) => Some(expr_name.id()),
            _ => None,
        };
        if let Some(name) = callee_name {
            let callee_fact = self.make_fq_name(name, None);
            self.callee_to_callers
                .push(python::CalleeToCaller::new(callee_fact, caller_fact));
        }
    }

    fn generate_facts_from_exprs(&mut self, expr: &Expr, container: &python::DeclarationContainer) {
        if let Some(call) = expr.as_call_expr() {
            self.file_call_facts(call);
            if let python::DeclarationContainer::func(caller) = container {
                self.callee_to_caller_facts(call, caller);
            }
        }
        expr.recurse(&mut |s| self.generate_facts_from_exprs(s, container));
    }

    fn visit_exprs(&mut self, node: &impl Visit<Expr>, container: &python::DeclarationContainer) {
        node.visit(&mut |expr| self.generate_facts_from_exprs(expr, container));
    }

    fn generate_facts(
        &mut self,
        stmt: &Stmt,
        container: &python::DeclarationContainer,
        top_level_decl: &python::Declaration,
    ) {
        let mut new_container = None;
        let mut new_top_level_decl = None;
        match stmt {
            Stmt::ClassDef(cls) => {
                let cls_declaration =
                    python::ClassDeclaration::new(self.make_fq_name(&cls.name.id, None), None);
                self.class_facts(
                    cls,
                    cls_declaration.clone(),
                    container.clone(),
                    top_level_decl.clone(),
                );
                self.visit_exprs(&cls.decorator_list, container);
                self.visit_exprs(&cls.type_params, container);
                self.visit_exprs(&cls.arguments, container);
                if let python::Declaration::module(_) = top_level_decl {
                    new_top_level_decl = Some(python::Declaration::cls(cls_declaration.clone()));
                }
                new_container = Some(python::DeclarationContainer::cls(cls_declaration));
            }
            Stmt::FunctionDef(func) => {
                let func_declaration =
                    python::FunctionDeclaration::new(self.make_fq_name(&func.name.id, None));
                self.function_facts(
                    func,
                    func_declaration.clone(),
                    container.clone(),
                    top_level_decl.clone(),
                );
                self.visit_exprs(&func.decorator_list, container);
                self.visit_exprs(&func.type_params, container);
                self.visit_exprs(&func.parameters, container);
                self.visit_exprs(&func.returns, container);
                if let python::Declaration::module(_) = top_level_decl {
                    new_top_level_decl = Some(python::Declaration::func(func_declaration.clone()));
                }
                new_container = Some(python::DeclarationContainer::func(func_declaration));
            }
            Stmt::Assign(assign) => {
                assign
                    .targets
                    .visit(&mut |target| self.variable_facts(target, None));
                self.visit_exprs(&assign.value, container);
            }
            Stmt::AnnAssign(assign) => {
                self.variable_facts(&assign.target, Some(&assign.annotation));
                self.visit_exprs(&assign.annotation, container);
                self.visit_exprs(&assign.value, container);
            }
            Stmt::AugAssign(assign) => {
                self.variable_facts(&assign.target, None);
                self.visit_exprs(&assign.value, container);
            }
            Stmt::Import(import) => self.import_facts(&import.names, &None, top_level_decl),
            Stmt::ImportFrom(import) => {
                self.import_facts(&import.names, &import.module, top_level_decl)
            }
            Stmt::For(stmt_for) => {
                stmt_for
                    .target
                    .visit(&mut |target| self.variable_facts(target, None));
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
                    item.optional_vars
                        .visit(&mut |target| self.variable_facts(target, None));
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
                    ExceptHandler::ExceptHandler(x) => self.visit_exprs(&x.type_, container),
                });
            }
            _ => self.visit_exprs(stmt, container),
        }
        stmt.recurse(&mut |x| {
            self.generate_facts(
                x,
                new_container.as_ref().unwrap_or(container),
                new_top_level_decl.as_ref().unwrap_or(top_level_decl),
            )
        });
    }
}

impl Glean {
    #[allow(unused_variables)]
    pub fn new(
        module_info: &ModuleInfo,
        ast: &ModModule,
        bindings: &Bindings,
        answers: &Answers,
    ) -> Self {
        let module_name = python::Name::new(module_info.name().as_str().to_owned());
        let module_fact = python::Module::new(module_name);
        let file_fact = src::File::new(module_info.path().to_string());
        let container = python::DeclarationContainer::module(module_fact.clone());
        let top_level_decl = python::Declaration::module(module_fact.clone());
        let file_language_fact = src::FileLanguage::new(file_fact.clone(), src::Language::Python);

        let mut facts = Facts::new(
            file_fact.clone(),
            module_fact.clone(),
            module_info.name().to_string(),
        );

        let file_lines = facts.file_lines_fact(module_info);

        facts.module_facts(&module_fact, ast.range, top_level_decl.clone());

        ast.body
            .visit(&mut |stmt: &Stmt| facts.generate_facts(stmt, &container, &top_level_decl));

        let digest = digest::Digest {
            hash: hash(module_info.contents().as_bytes()),
            size: module_info.contents().len() as u64,
        };
        let digest_fact = digest::FileDigest::new(file_fact, digest);

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
                facts: vec![json(module_fact)],
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
        ];

        // TODO: Add many more predicates here.

        Glean { entries }
    }
}
