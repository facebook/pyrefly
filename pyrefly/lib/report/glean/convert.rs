/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_util::visit::Visit;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
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
    decl_locations: Vec<python::DeclarationLocation>,
    def_locations: Vec<python::DefinitionLocation>,
}

fn to_span(range: TextRange) -> src::ByteSpan {
    src::ByteSpan {
        start: range.start().to_u32().into(),
        length: range.len().to_u32().into(),
    }
}

impl Facts {
    fn new(file: src::File) -> Facts {
        Facts {
            file,
            decl_locations: vec![],
            def_locations: vec![],
        }
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

    fn make_fq_name(&self, name: String) -> String {
        name // TODO(@rubmary) create fully qualified name
    }

    fn module_facts(&mut self, module: &python::Module, range: TextRange) {
        self.decl_locations
            .push(self.decl_location_fact(python::Declaration::module(module.clone()), range));

        self.def_locations.push(self.def_location_fact(
            python::Definition::module(python::ModuleDefinition::new(module.clone())),
            range,
        ))
    }

    fn class_facts(&mut self, cls: &StmtClassDef) {
        let fqname = python::Name::new(cls.name.to_string());
        let cls_declaration = python::ClassDeclaration::new(fqname, None);

        let bases = if let Some(arguments) = &cls.arguments {
            arguments
                .args
                .iter()
                .filter_map(|expr| expr.as_name_expr())
                .map(|expr_name| {
                    python::ClassDeclaration::new(
                        python::Name::new(self.make_fq_name(expr_name.id.to_string())),
                        None,
                    )
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
            None,
        );

        self.decl_locations
            .push(self.decl_location_fact(python::Declaration::cls(cls_declaration), cls.range));
        self.def_locations
            .push(self.def_location_fact(python::Definition::cls(cls_definition), cls.range));
    }

    fn generate_facts(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::ClassDef(cls) => self.class_facts(cls),
            Stmt::FunctionDef(func) => {
                let fqname = python::Name::new(self.make_fq_name(func.name.to_string()));
                let func_declaration = python::FunctionDeclaration::new(fqname);
                self.decl_locations.push(
                    self.decl_location_fact(
                        python::Declaration::func(func_declaration),
                        func.range,
                    ),
                );
            }
            _ => {}
        }
        stmt.recurse(&mut |x| self.generate_facts(x));
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

        let mut facts = Facts::new(file_fact.clone());

        facts.module_facts(&module_fact, ast.range);

        ast.body
            .visit(&mut |stmt: &Stmt| facts.generate_facts(stmt));

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
        ];

        // TODO: Add many more predicates here.

        Glean { entries }
    }
}
