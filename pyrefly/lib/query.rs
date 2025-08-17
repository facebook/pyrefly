/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Query interface for pyrefly. Just experimenting for the moment - not intended for external use.

use std::io::Cursor;
use std::iter;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use dupe::Dupe;
use itertools::Itertools;
use pyrefly_python::ast::Ast;
use pyrefly_python::dunder;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_types::callable::FuncMetadata;
use pyrefly_types::callable::Function;
use pyrefly_types::callable::FunctionKind;
use pyrefly_types::class::Class;
use pyrefly_types::qname::QName;
use pyrefly_types::types::BoundMethodType;
use pyrefly_types::types::Forallable;
use pyrefly_types::types::Type;
use pyrefly_util::events::CategorizedEvents;
use pyrefly_util::lined_buffer::DisplayRange;
use pyrefly_util::lock::Mutex;
use pyrefly_util::prelude::SliceExt;
use pyrefly_util::prelude::VecExt;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprName;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use starlark_map::small_set::SmallSet;

use crate::alt::answers::Answers;
use crate::alt::answers_solver::AnswersSolver;
use crate::binding::binding::Key;
use crate::binding::bindings::Bindings;
use crate::config::finder::ConfigFinder;
use crate::module::module_info::ModuleInfo;
use crate::state::handle::Handle;
use crate::state::lsp::DefinitionMetadata;
use crate::state::lsp::FindPreference;
use crate::state::require::Require;
use crate::state::state::State;
use crate::state::state::Transaction;
use crate::state::state::TransactionHandle;
use crate::types::display::TypeDisplayContext;

pub struct Query {
    /// The state that we use.
    state: State,
    /// The SysInfo, the same for all handles.
    sys_info: SysInfo,
    /// The files that have been used with `add_files`, used when files change.
    files: Mutex<SmallSet<(ModuleName, ModulePath)>>,
}

const CALLEE_KIND_FUNCTION: &str = "function";
const CALLEE_KIND_METHOD: &str = "method";
const CALLEE_KIND_CLASSMETHOD: &str = "classmethod";
const CALLEE_KIND_STATICMETHOD: &str = "staticmethod";

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct Callee {
    pub kind: String,
    pub target: String,
    pub class_name: Option<String>,
}

pub struct Attribute {
    pub name: String,
    pub kind: Option<String>,
}

impl Query {
    pub fn new(config_finder: ConfigFinder) -> Self {
        let state = State::new(config_finder);
        Self {
            state,
            sys_info: SysInfo::default(),
            files: Mutex::new(SmallSet::new()),
        }
    }

    fn make_handle(&self, name: ModuleName, path: ModulePath) -> Handle {
        Handle::new(name, path, self.sys_info.dupe())
    }

    pub fn change_files(&self, events: &CategorizedEvents) {
        let mut transaction = self
            .state
            .new_committable_transaction(Require::Everything, None);
        let new_transaction_mut = transaction.as_mut();
        new_transaction_mut.invalidate_events(events);
        new_transaction_mut.run(&[]);
        self.state.commit_transaction(transaction);
        let all_files = self.files.lock().iter().cloned().collect::<Vec<_>>();
        self.add_files(all_files);
    }

    pub fn change(&self, files: Vec<(ModuleName, std::path::PathBuf)>) {
        let modified_paths = files.into_iter().map(|(_, path)| path).collect();
        let events = CategorizedEvents {
            created: Vec::new(),
            modified: modified_paths,
            removed: Vec::new(),
            unknown: Vec::new(),
        };
        self.change_files(&events);
    }

    /// Load the given files and return any errors associated with them
    pub fn add_files(&self, files: Vec<(ModuleName, ModulePath)>) -> Vec<String> {
        self.files.lock().extend(files.clone());
        let mut transaction = self
            .state
            .new_committable_transaction(Require::Everything, None);
        let handles =
            files.into_map(|(name, file)| (self.make_handle(name, file), Require::Everything));
        transaction.as_mut().run(&handles);
        let errors = transaction
            .as_mut()
            .get_errors(handles.iter().map(|(h, _)| h));
        self.state.commit_transaction(transaction);
        errors.collect_errors().shown.map(|e| {
            // We deliberately don't have a Display for `Error`, to encourage doing the right thing.
            // But we just hack something up as this code is experimental.
            let mut s = Cursor::new(Vec::new());
            e.write_line(&mut s, false).unwrap();
            String::from_utf8_lossy(&s.into_inner()).into_owned()
        })
    }

    pub fn get_attributes(
        &self,
        name: ModuleName,
        path: ModulePath,
        class_name: &str,
    ) -> Option<Vec<Attribute>> {
        let transaction = self.state.transaction();
        let handle = self.make_handle(name, path);
        let ast = transaction.get_ast(&handle)?;
        // find last declaration of class with specified name in file
        let cls = ast
            .body
            .iter()
            .filter_map(|e| {
                if let Stmt::ClassDef(cls) = e {
                    if cls.name.id.as_str() == class_name {
                        Some(cls)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .last()?;
        let class_ty = transaction.get_type_at(&handle, cls.name.start());
        if let Some(Type::ClassDef(cd)) = &class_ty {
            let res = cd
                .fields()
                .filter_map(|n| {
                    let range = cd.field_decl_range(n)?;
                    let field_ty = transaction.get_type_at(&handle, range.start())?;
                    let kind = if let Type::Function(f) = &field_ty
                        && f.metadata.flags.is_property_getter
                    {
                        Some(String::from("property"))
                    } else {
                        None
                    };
                    Some(Attribute {
                        name: n.to_string(),
                        kind,
                    })
                })
                .collect_vec();
            Some(res)
        } else {
            None
        }
    }

    // fetches information about callees of a callable in a module
    pub fn get_callees_with_location(
        &self,
        name: ModuleName,
        path: ModulePath,
    ) -> Option<Vec<(DisplayRange, Callee)>> {
        let transaction = self.state.transaction();
        let handle = self.make_handle(name, path);
        let module_info = transaction.get_module_info(&handle)?;
        let ast = transaction.get_ast(&handle)?;
        let answers = transaction.get_answers(&handle)?;

        fn qname_to_string(n: &QName) -> String {
            format!("{}.{}", n.module_name(), n.id())
        }
        fn class_name_from_def_kind(kind: &FunctionKind) -> String {
            if let FunctionKind::Def(f) = kind
                && let Some(class_name) = &f.cls
            {
                format!("{}.{}", f.module, class_name)
            } else {
                panic!("class_name_from_def_kind - unsupported function kind: {kind:?}");
            }
        }
        fn target_from_def_kind(kind: &FunctionKind) -> String {
            match kind {
                FunctionKind::Def(f) => match &f.cls {
                    Some(class_name) => {
                        format!("{}.{}.{}", f.module, class_name, f.func)
                    }
                    None => {
                        format!("{}.{}", f.module, f.func)
                    }
                },
                FunctionKind::IsInstance => String::from("isinstance"),
                FunctionKind::IsSubclass => String::from("issubclass"),
                // should never see this in expression context
                FunctionKind::Dataclass => String::from("dataclasses.dataclass"),
                FunctionKind::DataclassField => String::from("dataclasses.field"),
                FunctionKind::CallbackProtocol(cls) => {
                    format!("{}.__call__", qname_to_string(cls.qname()))
                }
                _ => panic!("target_from_def_kind - unsupported function kind: {kind:?}"),
            }
        }

        fn target_from_function(f: &Function) -> String {
            target_from_def_kind(&f.metadata.kind)
        }
        fn callee_from_function(f: &Function) -> Callee {
            if f.metadata.flags.is_staticmethod {
                Callee {
                    kind: String::from(CALLEE_KIND_STATICMETHOD),
                    target: target_from_function(f),
                    class_name: Some(class_name_from_def_kind(&f.metadata.kind)),
                }
            } else if f.metadata.flags.is_classmethod {
                Callee {
                    kind: String::from(CALLEE_KIND_CLASSMETHOD),
                    target: target_from_function(f),
                    // TODO: use type of receiver
                    class_name: Some(class_name_from_def_kind(&f.metadata.kind)),
                }
            } else {
                Callee {
                    kind: String::from(CALLEE_KIND_FUNCTION),
                    target: target_from_function(f),
                    class_name: None,
                }
            }
        }

        fn target_from_bound_method_type(m: &BoundMethodType) -> String {
            match m {
                BoundMethodType::Function(f) => target_from_function(f),
                BoundMethodType::Forall(f) => target_from_function(&f.body),
                BoundMethodType::Overload(f) => target_from_def_kind(&f.metadata.kind),
            }
        }
        fn callee_method_kind_from_function_metadata(m: &FuncMetadata) -> String {
            if m.flags.is_staticmethod {
                String::from(CALLEE_KIND_STATICMETHOD)
            } else if m.flags.is_classmethod {
                String::from(CALLEE_KIND_CLASSMETHOD)
            } else {
                String::from(CALLEE_KIND_METHOD)
            }
        }

        fn callee_method_kind_from_bound_method_type(m: &BoundMethodType) -> String {
            match m {
                BoundMethodType::Function(f) => {
                    callee_method_kind_from_function_metadata(&f.metadata)
                }
                BoundMethodType::Forall(f) => {
                    callee_method_kind_from_function_metadata(&f.body.metadata)
                }
                BoundMethodType::Overload(f) => {
                    callee_method_kind_from_function_metadata(&f.metadata)
                }
            }
        }
        fn type_to_string(ty: &Type) -> String {
            match ty {
                Type::ClassType(c) => qname_to_string(c.qname()),
                Type::ClassDef(c) => qname_to_string(c.qname()),
                _ => panic!("unexpected type: {ty:?}"),
            }
        }
        fn callee_in_mro<F: Fn(&AnswersSolver<TransactionHandle>, &Class) -> Option<String>>(
            c: &Class,
            transaction: &Transaction<'_>,
            handle: &Handle,
            name: &str,
            f: F,
        ) -> Vec<Callee> {
            let call_target = transaction.ad_hoc_solve(handle, |solver| {
                let mro = solver.get_mro_for_class(c);
                iter::once(c)
                    .chain(mro.ancestors(solver.stdlib).map(|x| x.class_object()))
                    .find_map(|c| f(&solver, c))
            });
            let class_name = qname_to_string(c.qname());
            let target = if let Some(Some(t)) = call_target {
                t
            } else {
                format!("{class_name}.{name}")
            };
            vec![Callee {
                kind: String::from(CALLEE_KIND_METHOD),
                target,
                class_name: Some(class_name),
            }]
        }
        fn callee_for_type(
            ty: &Type,
            callee_range: TextRange,
            module_info: &ModuleInfo,
            transaction: &Transaction<'_>,
            handle: &Handle,
        ) -> Vec<Callee> {
            match ty {
                Type::Union(tys) => {
                    // get callee for each type
                    tys.iter()
                        .flat_map(|t| {
                            callee_for_type(t, callee_range, module_info, transaction, handle)
                        })
                        .unique()
                        // return sorted by target
                        .sorted_by(|a, b| a.target.cmp(&b.target))
                        .collect_vec()
                }
                Type::BoundMethod(m) => vec![Callee {
                    kind: callee_method_kind_from_bound_method_type(&m.func),
                    target: target_from_bound_method_type(&m.func),
                    class_name: Some(type_to_string(&m.obj)),
                }],
                Type::Function(f) => vec![callee_from_function(f)],
                Type::Overload(f) => vec![Callee {
                    // assuming that overload represents function and method overloads
                    // are handled by BoundMethod case
                    kind: String::from(CALLEE_KIND_FUNCTION),
                    target: target_from_def_kind(&f.metadata.kind),
                    class_name: None,
                }],
                Type::Callable(_) => {
                    // a bit unfortunate that we have to rely on LSP functionality to get the target
                    let defs = transaction.find_definition(
                        handle,
                        // take location of last included character in range (which should work for identifiers and attributes)
                        callee_range.end().checked_sub(TextSize::from(1)).unwrap(),
                        &FindPreference::default(),
                    );
                    if defs.len() == 1 {
                        // TODO: decide what do to with multiple definitions
                        if let DefinitionMetadata::Variable(_) = defs[0].metadata {
                            let name = module_info.code_at(defs[0].definition_range);
                            vec![Callee {
                                kind: String::from(CALLEE_KIND_FUNCTION),
                                target: format!("$parameter${name}"),
                                class_name: None,
                            }]
                        } else {
                            panic!(
                                "callable ty - unexpected metadata kind, {:?}",
                                defs[0].metadata
                            )
                        }
                    } else {
                        panic!("callable ty not supported yet, {defs:?}")
                    }
                }
                Type::ClassDef(cls) => {
                    callee_in_mro(cls, transaction, handle, "__init__", |solver, c| {
                        // find first class that has __init__ or __new__
                        let class_metadata = solver.get_metadata_for_class(c);
                        if c.contains(&dunder::INIT)
                            || class_metadata.dataclass_metadata().is_some()
                        {
                            // treat dataclasses as always having __init__
                            Some(format!("{}.{}.__init__", c.module_name(), c.name()))
                        } else if c.contains(&dunder::NEW) {
                            Some(format!("{}.{}.__new__", c.module_name(), c.name()))
                        } else {
                            None
                        }
                    })
                }
                Type::Forall(v) => {
                    if let Forallable::Function(func) = &v.body {
                        vec![callee_from_function(func)]
                    } else {
                        panic!("unsupported forallable type")
                    }
                }
                Type::ClassType(c) => callee_in_mro(
                    c.class_object(),
                    transaction,
                    handle,
                    "__call__",
                    |_solver, c| {
                        if c.contains(&dunder::CALL) {
                            Some(format!("{}.{}.__call__", c.module_name(), c.name()))
                        } else {
                            None
                        }
                    },
                ),
                Type::Any(_) => vec![],
                _ => panic!("unexpected type: {ty:?}"),
            }
        }

        let mut res = Vec::new();
        fn f<'a>(
            x: &Expr,
            module_info: &ModuleInfo,
            answers: &Answers,
            transaction: &Transaction<'a>,
            handle: &Handle,
            res: &mut Vec<(DisplayRange, Callee)>,
        ) {
            let (callee_ty, callee_range) = if let Expr::Attribute(attr) = x {
                (answers.try_get_getter_for_range(attr.range()), attr.range())
            } else if let Expr::Call(call) = x {
                (answers.get_type_trace(call.func.range()), call.func.range())
            } else {
                (None, x.range())
            };
            if let Some(func_ty) = callee_ty {
                callee_for_type(&func_ty, callee_range, module_info, transaction, handle)
                    .into_iter()
                    .for_each(|callee| {
                        res.push((module_info.display_range(callee_range), callee));
                    });
            }

            x.recurse(&mut |x| f(x, module_info, answers, transaction, handle, res));
        }

        ast.visit(&mut |x| f(x, &module_info, &answers, &transaction, &handle, &mut res));
        Some(res)
    }

    pub fn get_types_in_file(
        &self,
        name: ModuleName,
        path: ModulePath,
    ) -> Option<Vec<(DisplayRange, String)>> {
        let handle = self.make_handle(name, path);

        let transaction = self.state.transaction();
        let ast = transaction.get_ast(&handle)?;
        let module_info = transaction.get_module_info(&handle)?;
        let answers = transaction.get_answers(&handle)?;
        let bindings = transaction.get_bindings(&handle)?;

        let mut res = Vec::new();

        fn is_static_method(ty: &Type) -> bool {
            match ty {
                Type::Union(tys) => tys.iter().all(is_static_method),
                Type::BoundMethod(m) => m.func.metadata().flags.is_staticmethod,
                Type::Function(f) => f.metadata.flags.is_staticmethod,
                Type::Forall(f) => {
                    if let Forallable::Function(func) = &f.body {
                        func.metadata.flags.is_staticmethod
                    } else {
                        false
                    }
                }
                _ => false,
            }
        }

        fn add_type(
            ty: &Type,
            range: TextRange,
            module_info: &ModuleInfo,
            res: &mut Vec<(DisplayRange, String)>,
        ) {
            let mut ctx = TypeDisplayContext::new(&[ty]);
            ctx.always_display_module_name();
            let text = ctx.display(ty).to_string();
            let text = if is_static_method(ty) {
                format!("typing.StaticMethod[{text}]")
            } else {
                text
            };
            res.push((module_info.display_range(range), text));
        }
        fn try_find_key_for_name(name: &ExprName, bindings: &Bindings) -> Option<Key> {
            let key = Key::BoundName(ShortIdentifier::expr_name(name));
            if bindings.is_valid_key(&key) {
                Some(key)
            } else if let key = Key::Definition(ShortIdentifier::expr_name(name))
                && bindings.is_valid_key(&key)
            {
                Some(key)
            } else {
                None
            }
        }
        fn f(
            x: &Expr,
            module_info: &ModuleInfo,
            answers: &Answers,
            bindings: &Bindings,
            res: &mut Vec<(DisplayRange, String)>,
        ) {
            let range = x.range();
            if let Expr::Name(name) = x
                && let Some(key) = try_find_key_for_name(name, bindings)
                && let Some(idx) = answers.get_idx(bindings.key_to_idx(&key))
            {
                let ty = answers.for_display(idx.arc_clone_ty());
                add_type(&ty, range, module_info, res);
            } else if let Some(ty) = answers.get_type_trace(range) {
                add_type(&ty, range, module_info, res);
            }
            x.recurse(&mut |x| f(x, module_info, answers, bindings, res));
        }

        ast.visit(&mut |x| f(x, &module_info, &answers, &bindings, &mut res));
        Some(res)
    }

    /// Given an expression, which contains qualified types, guess which imports to add.
    ///
    /// For example `foo.bar.baz` will return `[foo.bar]`.
    fn find_imports(x: &ModModule) -> Vec<String> {
        fn g(x: &ExprAttribute) -> Option<Vec<&Name>> {
            match &*x.value {
                Expr::Attribute(x) => {
                    let mut res = g(x)?;
                    res.push(&x.attr.id);
                    Some(res)
                }
                Expr::Name(x) => Some(vec![&x.id]),
                _ => None,
            }
        }

        fn f(x: &Expr, res: &mut SmallSet<String>) {
            if let Expr::Attribute(x) = x {
                if let Some(module) = g(x) {
                    res.insert(module.map(|x| x.as_str()).join("."));
                }
            } else {
                x.recurse(&mut |x| f(x, res));
            }
        }
        let mut res = SmallSet::new();
        x.visit(&mut |x| f(x, &mut res));
        res.into_iter().collect()
    }

    /// Return `Err` if you can't resolve them to types, otherwise return `lt <: gt`.
    pub fn is_subtype(
        &self,
        name: ModuleName,
        path: PathBuf,
        lt: &str,
        gt: &str,
    ) -> Result<bool, String> {
        fn do_check(
            t: &mut Transaction<'_>,
            h: Handle,
            path: &Path,
            lt: &str,
            gt: &str,
            types: String,
            check: &'static str,
        ) -> Result<bool, String> {
            let imported = Query::find_imports(&Ast::parse(&types).0);
            let imports = imported.map(|x| format!("import {x}\n")).join("");

            let before = format!("{imports}\n{types}\n");
            let after = format!("{imports}\n{types}\n{check}");

            t.set_memory(vec![(path.to_owned(), Some(Arc::new(before.clone())))]);
            t.run(&[(h.dupe(), Require::Everything)]);
            let errors = t.get_errors([&h]).collect_errors();
            if !errors.shown.is_empty() {
                let mut res = Vec::new();
                for e in errors.shown {
                    e.write_line(&mut Cursor::new(&mut res), true).unwrap();
                }
                return Err(format!(
                    "Errors from is_subtype `{lt}` <: `{gt}`\n{}\n\nSource code:\n{before}",
                    str::from_utf8(&res).unwrap_or("UTF8 error")
                ));
            }
            t.set_memory(vec![(path.to_owned(), Some(Arc::new(after)))]);
            t.run(&[(h.dupe(), Require::Everything)]);
            let errors = t.get_errors([&h]).collect_errors();
            Ok(errors.shown.is_empty())
        }

        let mut t = self.state.transaction();
        let h = self.make_handle(name, ModulePath::memory(path.clone()));

        // py2hack uses pyre1 fake type order for typed dicts
        if gt == "TypedDictionary" || gt == "NonTotalTypedDictionary" {
            let types = format!("type pyrefly_lt = ({lt})");
            // check if type has attributes specific to TypedDict
            let check =
                "pyrefly_lt.__required_keys__, pyrefly_lt.__optional_keys__, pyrefly_lt.__total__";

            do_check(&mut t, h, &path, lt, gt, types, check)
        } else {
            let types = format!("type pyrefly_lt = ({lt})\ntype pyrefly_gt = ({gt})\n");
            let check = "def pyrefly_func(x: pyrefly_lt) -> pyrefly_gt:\n    return x";

            do_check(&mut t, h, &path, lt, gt, types, check)
        }
    }
}
