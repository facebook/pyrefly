/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Query interface for pyrefly. Just experimenting for the moment - not intended for external use.

use std::io::Cursor;
use std::iter;
use std::path::PathBuf;
use std::sync::Arc;

use dupe::Dupe;
use itertools::Itertools;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::dunder;
use pyrefly_python::module::Module;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::qname::QName;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_types::callable::FuncMetadata;
use pyrefly_types::callable::Function;
use pyrefly_types::callable::FunctionKind;
use pyrefly_types::class::Class;
use pyrefly_types::literal::Lit;
use pyrefly_types::quantified::Quantified;
use pyrefly_types::quantified::QuantifiedKind;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::types::BoundMethodType;
use pyrefly_types::types::Forallable;
use pyrefly_types::types::Type;
use pyrefly_util::events::CategorizedEvents;
use pyrefly_util::lined_buffer::LineNumber;
use pyrefly_util::lock::Mutex;
use pyrefly_util::prelude::SliceExt;
use pyrefly_util::prelude::VecExt;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Arguments;
use ruff_python_ast::Decorator;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprName;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::name::Name;
use ruff_source_file::OneIndexed;
use ruff_source_file::PositionEncoding;
use ruff_source_file::SourceLocation;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use serde::Serialize;
use starlark_map::small_set::SmallSet;

use crate::alt::answers::Answers;
use crate::alt::answers_solver::AnswersSolver;
use crate::binding::binding::Key;
use crate::binding::binding::KeyClassSynthesizedFields;
use crate::binding::bindings::Bindings;
use crate::config::finder::ConfigFinder;
use crate::module::module_info::ModuleInfo;
use crate::state::lsp::DefinitionMetadata;
use crate::state::lsp::FindPreference;
use crate::state::require::Require;
use crate::state::state::State;
use crate::state::state::Transaction;
use crate::state::state::TransactionHandle;
use crate::types::display::TypeDisplayContext;

const REPR: Name = Name::new_static("__repr__");

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
    /// What kind of callable is this? Distinguishes various kinds of methods from normal functions.
    pub kind: String,
    /// What's the qualified name of the callable. The name `target` is for Pyre compatibility and originates in Pysa vocabulary.
    pub target: String,
    /// If this is a method, what class is it defined on?
    pub class_name: Option<String>,
}

pub struct Attribute {
    pub name: String,
    pub kind: Option<String>,
    pub annotation: String,
}

#[derive(Debug, Clone)]
pub struct PythonASTRange {
    pub start_line: LineNumber,
    pub start_col: u32,
    pub end_line: LineNumber,
    pub end_col: u32,
}

impl Serialize for PythonASTRange {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("PythonASTRange", 4)?;
        state.serialize_field("start_line", &self.start_line.get())?;
        state.serialize_field("start_col", &self.start_col)?;
        state.serialize_field("end_line", &self.end_line.get())?;
        state.serialize_field("end_col", &self.end_col)?;
        state.end()
    }
}

fn python_ast_range_for_expr(
    module_info: &ModuleInfo,
    original_range: TextRange,
    expr: &Expr,
    parent_expr: Option<&Expr>,
) -> PythonASTRange {
    let expression_range = if let Expr::Generator(e) = expr {
        // python AST module reports locations of all generator expressions as if they are parenthesized
        // i.e any(any(a.b is not None for a in [l2]) for l2 in l1)
        //        ^-will be col_offset for generator expression over l1
        // ruff properly distinguishes between parenthesized and non-parenthesized expressions
        // and points to the first character of the expression
        // since queries are done based on Python AST for generator expression we will
        // need to adjust start/end column offsets
        if e.parenthesized {
            original_range
        } else if let Some(Expr::Call(p)) = parent_expr
            && p.arguments.len() == 1
            && p.arguments.inner_range().contains_range(original_range)
        {
            TextRange::new(
                p.arguments.l_paren_range().start(),
                p.arguments.r_paren_range().end(),
            )
        } else {
            original_range
                .sub_start(TextSize::new(1))
                .add_end(TextSize::new(1))
        }
    } else {
        original_range
    };

    let start_location = module_info.lined_buffer().line_index().source_location(
        expression_range.start(),
        module_info.lined_buffer().contents(),
        ruff_source_file::PositionEncoding::Utf8,
    );
    let end_location = module_info.lined_buffer().line_index().source_location(
        expression_range.end(),
        module_info.lined_buffer().contents(),
        ruff_source_file::PositionEncoding::Utf8,
    );

    PythonASTRange {
        start_line: LineNumber::new(start_location.line.get() as u32).unwrap(),
        start_col: start_location.character_offset.to_zero_indexed() as u32,
        end_line: LineNumber::new(end_location.line.get() as u32).unwrap(),
        end_col: end_location.character_offset.to_zero_indexed() as u32,
    }
}

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

fn bound_of_type_var(ty: &Type) -> Option<&Type> {
    match ty {
        Type::Quantified(box Quantified {
            kind: QuantifiedKind::TypeVar,
            restriction: Restriction::Bound(bound),
            ..
        })
        | Type::QuantifiedValue(box Quantified {
            kind: QuantifiedKind::TypeVar,
            restriction: Restriction::Bound(bound),
            ..
        }) => Some(bound),
        _ => None,
    }
}

fn bound_of_type_var_decl(ty: &Type) -> Option<(&QName, &Type)> {
    if let Type::TypeVar(tv) = ty
        && let Restriction::Bound(bound) = tv.restriction()
    {
        Some((tv.qname(), bound))
    } else {
        None
    }
}

fn type_to_string(ty: &Type) -> String {
    let mut ctx = TypeDisplayContext::new(&[ty]);
    ctx.always_display_module_name();
    if is_static_method(ty) {
        format!("typing.StaticMethod[{}]", ctx.display(ty))
    } else if let Some(bound) = bound_of_type_var(ty) {
        // pyre1 compatibility: return bound for type variable
        format!(
            "Variable[{} (bound to {})]",
            ctx.display(ty),
            type_to_string(bound)
        )
    } else if let Some((qname, bound)) = bound_of_type_var_decl(ty) {
        // pyre1 compatibility: return bound for type variable
        format!(
            "Variable[{} (bound to {})]",
            qname.id(),
            type_to_string(bound)
        )
    } else {
        ctx.display(ty).to_string()
    }
}

struct CalleesWithLocation<'a> {
    transaction: Transaction<'a>,
    handle: Handle,
    module_info: Module,
    ast: Arc<ModModule>,
    answers: Arc<Answers>,
}

impl<'a> CalleesWithLocation<'a> {
    pub fn new(transaction: Transaction<'a>, handle: Handle) -> Option<CalleesWithLocation<'a>> {
        let module_info = transaction.get_module_info(&handle)?;
        let answers = transaction.get_answers(&handle)?;
        let ast: Arc<ModModule> = transaction.get_ast(&handle)?;
        Some(Self {
            transaction,
            handle,
            module_info,
            ast,
            answers,
        })
    }
    fn process_expr(&self, x: &Expr, res: &mut Vec<(PythonASTRange, Callee)>) {
        let (callees, callee_range) = match x {
            Expr::Attribute(attr) => {
                let callees =
                    if let Some(func_ty) = self.answers.try_get_getter_for_range(attr.range()) {
                        self.callee_from_type(&func_ty, None, attr.range(), None)
                    } else {
                        vec![]
                    };
                (callees, attr.range())
            }
            Expr::Call(ExprCall {
                func: box Expr::Name(name),
                ..
            }) if name.id() == "prod_assert" => {
                // pyrefly has special treatment for prod_assert but for our purposes we still want to see this call
                let callees = vec![Callee {
                    kind: String::from(CALLEE_KIND_FUNCTION),
                    target: String::from("util.prod_assert"),
                    class_name: None,
                }];
                (callees, name.range())
            }
            Expr::Call(call) => {
                let callees = if let Some(func_ty) = self.answers.get_type_trace(call.func.range())
                {
                    self.callee_from_type(
                        &func_ty,
                        Some(&*call.func),
                        call.func.range(),
                        Some(&call.arguments),
                    )
                } else {
                    vec![]
                };
                (callees, call.func.range())
            }
            _ => (vec![], x.range()),
        };
        for callee in callees {
            res.push((
                python_ast_range_for_expr(&self.module_info, callee_range, x, None),
                callee,
            ));
        }

        x.recurse(&mut |x| self.process_expr(x, res));
    }

    fn add_decorators(&self, decorators: &[Decorator], res: &mut Vec<(PythonASTRange, Callee)>) {
        for dec in decorators {
            if matches!(dec.expression, Expr::Name(_) | Expr::Attribute(_))
                && let Some(call_ty) = self.answers.get_type_trace(dec.expression.range())
            {
                self.callee_from_type(&call_ty, None, dec.expression.range(), None)
                    .into_iter()
                    .for_each(|callee| {
                        res.push((
                            python_ast_range_for_expr(
                                &self.module_info,
                                dec.expression.range(),
                                &dec.expression,
                                None,
                            ),
                            callee,
                        ));
                    });
            }
        }
    }

    pub fn process(&self, location: Option<PythonASTRange>) -> Vec<(PythonASTRange, Callee)> {
        let mut res = Vec::new();

        if let Some(target_location) = location {
            // Helper function to convert line/column to TextSize
            fn line_col_to_text_size(
                module_info: &ModuleInfo,
                line: LineNumber,
                col: u32,
            ) -> TextSize {
                module_info.lined_buffer().line_index().offset(
                    SourceLocation {
                        line: OneIndexed::new(line.get() as usize).unwrap(),
                        character_offset: OneIndexed::from_zero_indexed(col as usize),
                    },
                    module_info.lined_buffer().contents(),
                    PositionEncoding::Utf8,
                )
            }

            // Convert PythonASTRange to TextRange using SourceLocation directly
            let start_pos = line_col_to_text_size(
                &self.module_info,
                target_location.start_line,
                target_location.start_col,
            );
            let end_pos = line_col_to_text_size(
                &self.module_info,
                target_location.end_line,
                target_location.end_col,
            );
            let target_range = TextRange::new(start_pos, end_pos);

            // Query the type information directly at the target location
            if let Some(func_ty) = self.answers.get_type_trace(target_range) {
                let callees = self.callee_from_type(&func_ty, None, target_range, None);

                for callee in callees {
                    res.push((target_location.clone(), callee));
                }
            }
        } else {
            for stmt in &self.ast.body {
                match &stmt {
                    Stmt::ClassDef(StmtClassDef {
                        decorator_list: d, ..
                    })
                    | Stmt::FunctionDef(StmtFunctionDef {
                        decorator_list: d, ..
                    }) => {
                        self.add_decorators(d, &mut res);
                    }
                    _ => {}
                }
                stmt.visit(&mut |x| self.process_expr(x, &mut res));
            }
        };
        res
    }

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
    fn target_from_def_kind(kind: &FunctionKind, module_name_override: Option<&str>) -> String {
        match kind {
            FunctionKind::Def(f) => {
                if let Some(module_name_override) = module_name_override {
                    format!("{module_name_override}.{}", f.func)
                } else {
                    match &f.cls {
                        Some(class_name) => {
                            format!("{}.{}.{}", f.module, class_name, f.func)
                        }
                        None => {
                            format!("{}.{}", f.module, f.func)
                        }
                    }
                }
            }
            FunctionKind::CallbackProtocol(cls) => {
                format!("{}.__call__", Self::qname_to_string(cls.qname()))
            }

            x => x.as_func_id().format(ModuleName::builtins()),
        }
    }
    fn repr_from_arguments(&self, arguments: &Arguments) -> Option<Callee> {
        // Use the type of the first argument to find the callee.
        if let Some(arg_type) = self.answers.get_type_trace(arguments.args[0].range())
            && let Type::ClassType(class) = &arg_type
        {
            let repr_callees =
                self.callee_from_mro(class.class_object(), "__repr__", |_solver, c| {
                    if c.contains(&REPR) {
                        Some(format!("{}.{}.__repr__", c.module_name(), c.name()))
                    } else {
                        None
                    }
                });
            if !repr_callees.is_empty() {
                return Some(repr_callees[0].clone());
            }
        }
        None
    }
    fn callee_from_function(
        &self,
        f: &Function,
        call_target: Option<&Expr>,
        call_arguments: Option<&Arguments>,
    ) -> Callee {
        if f.metadata.flags.is_staticmethod {
            Callee {
                kind: String::from(CALLEE_KIND_STATICMETHOD),
                target: Self::target_from_def_kind(&f.metadata.kind, None),
                class_name: Some(Self::class_name_from_def_kind(&f.metadata.kind)),
            }
        } else if f.metadata.flags.is_classmethod {
            Callee {
                kind: String::from(CALLEE_KIND_CLASSMETHOD),
                target: Self::target_from_def_kind(&f.metadata.kind, None),
                // TODO: use type of receiver
                class_name: Some(Self::class_name_from_def_kind(&f.metadata.kind)),
            }
        } else {
            // Check if this is a builtins function that needs special casing.
            if let FunctionKind::Def(def) = &f.metadata.kind
                && def.module.as_str() == "builtins"
                && def.func == "repr"
                && let Some(args) = call_arguments
                && let Some(callee) = self.repr_from_arguments(args)
            {
                return callee;
            }

            let class_name = self.class_name_from_call_target(call_target);
            let kind = if class_name.is_some() {
                String::from(CALLEE_KIND_METHOD)
            } else {
                String::from(CALLEE_KIND_FUNCTION)
            };

            Callee {
                kind,
                target: Self::target_from_def_kind(&f.metadata.kind, None),
                class_name,
            }
        }
    }
    fn target_from_bound_method_type(m: &BoundMethodType, method_of_typed_dict: bool) -> String {
        let module_name_override = if method_of_typed_dict {
            Some("TypedDictionary")
        } else {
            None
        };
        match m {
            BoundMethodType::Function(f) => {
                Self::target_from_def_kind(&f.metadata.kind, module_name_override)
            }
            BoundMethodType::Forall(f) => {
                Self::target_from_def_kind(&f.body.metadata.kind, module_name_override)
            }
            BoundMethodType::Overload(f) => {
                Self::target_from_def_kind(&f.metadata.kind, module_name_override)
            }
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
                Self::callee_method_kind_from_function_metadata(&f.metadata)
            }
            BoundMethodType::Forall(f) => {
                Self::callee_method_kind_from_function_metadata(&f.body.metadata)
            }
            BoundMethodType::Overload(f) => {
                Self::callee_method_kind_from_function_metadata(&f.metadata)
            }
        }
    }
    fn class_info_for_qname(qname: &QName, is_typed_dict: bool) -> Vec<(String, bool)> {
        vec![(Self::qname_to_string(qname), is_typed_dict)]
    }
    fn class_info_from_bound_obj(ty: &Type) -> Vec<(String, bool)> {
        match ty {
            Type::SelfType(c) => Self::class_info_for_qname(c.qname(), false),
            // TODO: wrap in 'type'
            Type::Type(t) => Self::class_info_from_bound_obj(t),
            Type::ClassType(c) => Self::class_info_for_qname(c.qname(), false),
            Type::ClassDef(c) => Self::class_info_for_qname(c.qname(), false),
            Type::TypedDict(d) => Self::class_info_for_qname(d.qname(), true),
            Type::Literal(Lit::Str(_)) | Type::LiteralString => {
                vec![(String::from("builtins.str"), false)]
            }
            Type::Literal(Lit::Int(_)) => vec![(String::from("builtins.int"), false)],
            Type::Literal(Lit::Bool(_)) => vec![(String::from("builtins.bool"), false)],
            Type::Quantified(q) => match &q.restriction {
                // for explicit bound - use name of the type used as bound
                Restriction::Bound(b) => Self::class_info_from_bound_obj(b),
                // no bound - use name of the type variable (not very useful but not worse than status quo)
                Restriction::Unrestricted => vec![(q.name().to_string(), false)],
                Restriction::Constraints(_) => {
                    panic!("unexpected restriction: {q:?}")
                }
            },
            Type::Union(tys) => tys
                .iter()
                .flat_map(Self::class_info_from_bound_obj)
                .collect_vec(),
            _ => panic!("unexpected type: {ty:?}"),
        }
    }
    fn callee_from_mro<F: Fn(&AnswersSolver<TransactionHandle>, &Class) -> Option<String>>(
        &self,
        c: &Class,
        fallback_name: &str,
        callee_from_ancestor: F,
    ) -> Vec<Callee> {
        let call_target = self.transaction.ad_hoc_solve(&self.handle, |solver| {
            let mro = solver.get_mro_for_class(c);
            iter::once(c)
                .chain(mro.ancestors(solver.stdlib).map(|x| x.class_object()))
                .find_map(|c| callee_from_ancestor(&solver, c))
        });
        let class_name = Self::qname_to_string(c.qname());
        let target = if let Some(Some(t)) = call_target {
            t
        } else {
            format!("{class_name}.{fallback_name}")
        };
        vec![Callee {
            kind: String::from(CALLEE_KIND_METHOD),
            target,
            class_name: Some(class_name),
        }]
    }
    fn for_callable(&self, callee_range: TextRange) -> Vec<Callee> {
        // a bit unfortunate that we have to rely on LSP functionality to get the target
        let defs = self
            .transaction
            .find_definition(
                &self.handle,
                // take location of last included character in range (which should work for identifiers and attributes)
                callee_range.end().checked_sub(TextSize::from(1)).unwrap(),
                &FindPreference::default(),
            )
            .into_iter()
            // filter out attributes since we don't know how to handle them
            .filter(|d| !matches!(d.metadata, DefinitionMetadata::Attribute(_)))
            .collect_vec();
        if defs.is_empty() {
            vec![]
        } else if defs.len() == 1 {
            // TODO: decide what do to with multiple definitions
            let def0 = &defs[0];
            if def0.module.name() == self.handle.module() {
                match &def0.metadata {
                    DefinitionMetadata::Variable(_) => {
                        let name = &self.module_info.code_at(defs[0].definition_range);
                        vec![Callee {
                            kind: String::from(CALLEE_KIND_FUNCTION),
                            target: format!("$parameter${name}"),
                            class_name: None,
                        }]
                    }
                    x => panic!("callable ty - unexpected metadata kind, {x:?}"),
                }
            } else {
                vec![]
            }
        } else {
            panic!(
                "callable ty at [{}] not supported yet, {defs:?}",
                self.module_info.display_range(callee_range)
            )
        }
    }
    fn class_name_from_call_target(&self, call_target: Option<&Expr>) -> Option<String> {
        if let Some(Expr::Attribute(attr)) = call_target
            && let Some(ty) = self.answers.get_type_trace(attr.value.range())
            && !matches!(ty, Type::Module(_))
        {
            // treat calls where targets are attribute access a.b and a is not a module
            // as method calls
            Some(type_to_string(&ty))
        } else {
            None
        }
    }

    fn find_init_or_new(&self, cls: &Class) -> Vec<Callee> {
        self.callee_from_mro(cls, "__init__", |solver, c| {
            // find first class that has __init__ or __new__
            let has_init = c.contains(&dunder::INIT)
                || solver
                    .get_from_class(c, &KeyClassSynthesizedFields(c.index()))
                    .is_some_and(|f| f.get(&dunder::INIT).is_some());
            if has_init {
                Some(format!("{}.{}.__init__", c.module_name(), c.name()))
            } else if c.contains(&dunder::NEW) {
                Some(format!("{}.{}.__new__", c.module_name(), c.name()))
            } else {
                None
            }
        })
    }
    fn init_or_new_from_type(&self, ty: &Type, callee_range: TextRange) -> Vec<Callee> {
        match ty {
            Type::SelfType(c) | Type::ClassType(c) => self.find_init_or_new(c.class_object()),
            Type::Quantified(box q) => match &q.restriction {
                Restriction::Bound(Type::ClassType(c)) => self.find_init_or_new(c.class_object()),
                x => panic!(
                    "unexpected restriction {}: {x:?}",
                    self.module_info.display_range(callee_range)
                ),
            },
            Type::Union(tys) => {
                // get callee for each type
                tys.iter()
                    .flat_map(|t| self.init_or_new_from_type(t, callee_range))
                    .unique()
                    // return sorted by target
                    .sorted_by(|a, b| a.target.cmp(&b.target))
                    .collect_vec()
            }
            x => {
                panic!(
                    "unexpected type at [{}]: {x:?}",
                    self.module_info.display_range(callee_range)
                );
            }
        }
    }
    pub fn callee_from_type(
        &self,
        ty: &Type,
        call_target: Option<&Expr>,
        callee_range: TextRange,
        call_arguments: Option<&Arguments>,
    ) -> Vec<Callee> {
        match ty {
            Type::Quantified(q) => match &q.restriction {
                Restriction::Bound(b) => {
                    self.callee_from_type(b, call_target, callee_range, call_arguments)
                }
                x => panic!(
                    "unexpected restriction {}: {x:?}",
                    self.module_info.display_range(callee_range)
                ),
            },
            Type::Never(_) => vec![],
            Type::Union(tys) => {
                // get callee for each type
                tys.iter()
                    .flat_map(|t| {
                        self.callee_from_type(t, call_target, callee_range, call_arguments)
                    })
                    .unique()
                    // return sorted by target
                    .sorted_by(|a, b| a.target.cmp(&b.target))
                    .collect_vec()
            }
            Type::BoundMethod(m) => Self::class_info_from_bound_obj(&m.obj)
                .into_iter()
                .map(|(class_name, class_is_typed_dict)| Callee {
                    kind: Self::callee_method_kind_from_bound_method_type(&m.func),
                    target: Self::target_from_bound_method_type(&m.func, class_is_typed_dict),
                    class_name: Some(class_name),
                })
                .unique()
                // return sorted by target
                .sorted_by(|a, b| a.target.cmp(&b.target))
                .collect_vec(),

            Type::Function(f) => {
                vec![self.callee_from_function(f, call_target, call_arguments)]
            }
            Type::Overload(f) => {
                let class_name = self.class_name_from_call_target(call_target);
                let kind = if class_name.is_some() {
                    String::from(CALLEE_KIND_METHOD)
                } else {
                    String::from(CALLEE_KIND_FUNCTION)
                };
                // assuming that overload represents function and method overloads
                // are handled by BoundMethod case
                vec![Callee {
                    kind,
                    target: Self::target_from_def_kind(&f.metadata.kind, None),
                    class_name,
                }]
            }
            Type::Callable(..) => self.for_callable(callee_range),
            Type::Type(box ty) => self.init_or_new_from_type(ty, callee_range),

            Type::ClassDef(cls) => self.find_init_or_new(cls),
            Type::Forall(v) => match &v.body {
                Forallable::Function(func) => {
                    vec![self.callee_from_function(func, call_target, call_arguments)]
                }
                Forallable::Callable(_) => self.for_callable(callee_range),
                Forallable::TypeAlias(t) => {
                    self.callee_from_type(&t.as_type(), call_target, callee_range, call_arguments)
                }
            },
            Type::SelfType(c) | Type::ClassType(c) => {
                self.callee_from_mro(c.class_object(), "__call__", |_solver, c| {
                    if c.contains(&dunder::CALL) {
                        Some(format!("{}.{}.__call__", c.module_name(), c.name()))
                    } else {
                        None
                    }
                })
            }
            Type::Any(_) => vec![],
            Type::Literal(_) => vec![],
            Type::TypeAlias(t) => {
                self.callee_from_type(&t.as_type(), call_target, callee_range, call_arguments)
            }
            _ => panic!(
                "unexpected type at [{}]: {ty:?}",
                self.module_info.display_range(callee_range)
            ),
        }
    }
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
        let config = self.state.config_finder().python_file(name.dupe(), &path);
        if config.source_db.is_some() {
            panic!("Pyrefly doesn't support sourcedb-powered queries yet");
        }
        // TODO(connernilsen): make this work with build systems
        Handle::new(name, path, self.sys_info.dupe())
    }

    pub fn change_files(&self, events: &CategorizedEvents) {
        let mut transaction = self
            .state
            .new_committable_transaction(Require::Everything, None);
        let new_transaction_mut = transaction.as_mut();
        new_transaction_mut.invalidate_events(events);
        new_transaction_mut.run(&[], Require::Everything);
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
        self.files.lock().extend(files.iter().cloned());
        let mut transaction = self
            .state
            .new_committable_transaction(Require::Exports, None);
        let handles = files.into_map(|(name, file)| self.make_handle(name, file));
        transaction.as_mut().run(&handles, Require::Everything);
        let errors = transaction.as_mut().get_errors(&handles);
        self.state.commit_transaction(transaction);
        let project_root = PathBuf::new();
        errors.collect_errors().shown.map(|e| {
            // We deliberately don't have a Display for `Error`, to encourage doing the right thing.
            // But we just hack something up as this code is experimental.
            let mut s = Cursor::new(Vec::new());
            e.write_line(&mut s, project_root.as_path(), false).unwrap();
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
        fn get_kind_and_field_type(ty: &Type) -> (Option<String>, &Type) {
            match ty {
                Type::Function(f) if f.metadata.flags.is_property_getter => {
                    (Some(String::from("property")), ty)
                }
                Type::ClassType(c)
                    if c.name() == "classproperty" || c.name() == "cached_classproperty" =>
                {
                    let result_ty = c.targs().as_slice().get(1).unwrap();
                    (Some(String::from("property")), result_ty)
                }
                _ => (None, ty),
            }
        }
        if let Some(Type::ClassDef(cd)) = &class_ty {
            let res = cd
                .fields()
                .filter_map(|n| {
                    let range = cd.field_decl_range(n)?;
                    let field_ty = transaction.get_type_at(&handle, range.start())?;
                    let (kind, field_ty) = get_kind_and_field_type(&field_ty);
                    Some(Attribute {
                        name: n.to_string(),
                        kind,
                        annotation: type_to_string(field_ty),
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
        location: Option<PythonASTRange>,
    ) -> Option<Vec<(PythonASTRange, Callee)>> {
        let transaction = self.state.transaction();
        let handle = self.make_handle(name, path);
        let find_callees = CalleesWithLocation::new(transaction, handle)?;
        Some(find_callees.process(location))
    }

    pub fn get_types_in_file(
        &self,
        name: ModuleName,
        path: ModulePath,
    ) -> Option<Vec<(PythonASTRange, String)>> {
        let handle = self.make_handle(name, path);

        let transaction = self.state.transaction();
        let ast = transaction.get_ast(&handle)?;
        let module_info = transaction.get_module_info(&handle)?;
        let answers = transaction.get_answers(&handle)?;
        let bindings = transaction.get_bindings(&handle)?;

        let mut res = Vec::new();

        fn add_type(
            ty: &Type,
            e: &Expr,
            parent: Option<&Expr>,
            range: TextRange,
            module_info: &ModuleInfo,
            res: &mut Vec<(PythonASTRange, String)>,
        ) {
            res.push((
                python_ast_range_for_expr(module_info, range, e, parent),
                type_to_string(ty),
            ));
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
            parent: Option<&Expr>,
            module_info: &ModuleInfo,
            answers: &Answers,
            bindings: &Bindings,
            res: &mut Vec<(PythonASTRange, String)>,
        ) {
            let range = x.range();
            if let Expr::Name(name) = x
                && let Some(key) = try_find_key_for_name(name, bindings)
                && let Some(ty) = answers.get_type_at(bindings.key_to_idx(&key))
            {
                add_type(&ty, x, parent, range, module_info, res);
            } else if let Some(ty) = answers.get_type_trace(range) {
                add_type(&ty, x, parent, range, module_info, res);
            }
            x.recurse(&mut |c| f(c, Some(x), module_info, answers, bindings, res));
        }

        ast.visit(&mut |x| f(x, None, &module_info, &answers, &bindings, &mut res));
        Some(res)
    }

    /// Given an expression, which contains qualified types, guess which imports to add.
    ///
    /// For example `foo.bar.baz` will return `[foo.bar]`.
    ///
    /// The expression comes in as a module because we are parsing it from a raw string
    /// input; we expect it to actually be a type expression.
    fn find_imports(module: &ModModule, t: &Transaction, h: &Handle) -> Vec<String> {
        fn compute_prefix(attr: &ExprAttribute) -> Option<Vec<&Name>> {
            match &*attr.value {
                Expr::Attribute(base) => {
                    let mut res = compute_prefix(base)?;
                    res.push(&base.attr.id);
                    Some(res)
                }
                Expr::Name(base) => Some(vec![&base.id]),
                _ => None,
            }
        }

        fn collect_attribute_prefixes(
            x: &Expr,
            res: &mut SmallSet<String>,
            t: &Transaction,
            h: &Handle,
        ) {
            if let Expr::Attribute(attr) = x {
                // `attr` is a qname of a type. Get its prefix, which is likely the
                // module where it is defined.
                if let Some(mut names) = compute_prefix(attr) {
                    // The initial prefix may not be an actual module, if the type in question
                    // is a nested class. Search recursively for the longest part of the prefix
                    // that is a module, and assume that is where the type is defined.
                    //
                    // Note: in messy codebases that include name collisions between submodules
                    // and attributes of `__init__.py` modules, this rule can fail (in this
                    // scenario it's also possible for the qname to be ambiguous, as in two
                    // distinct types have the same qname). We do not support such codebases.
                    loop {
                        if !names.is_empty() {
                            let module_name = names.map(|name| name.as_str()).join(".");
                            if t.import_handle(
                                h,
                                ModuleName::from_string(module_name.clone()),
                                None,
                            )
                            .is_ok()
                            {
                                // We found the longest matching prefix, assume this is the import.
                                res.insert(names.map(|name| name.as_str()).join("."));
                                break;
                            } else {
                                // No module at this prefix, keep looking.
                                names.pop();
                            }
                        } else {
                            // If we get here, either the name is undefined or it is is defined in `builtins`;
                            // either way we can skip it.
                            break;
                        }
                    }
                }
            } else {
                x.recurse(&mut |x| collect_attribute_prefixes(x, res, t, h));
            }
        }
        let mut res = SmallSet::new();
        module.visit(&mut |x| collect_attribute_prefixes(x, &mut res, t, h));
        res.into_iter().collect()
    }

    /// Check a snippet of code; used as part of performing is_subtype checks via the query API.
    fn check_code_snippet(
        &self,
        name: ModuleName,
        path: PathBuf,
        lt: &str,
        gt: &str,
        types: String,
        check: &'static str,
    ) -> Result<bool, String> {
        let mut t = self.state.transaction();
        let h = self.make_handle(name, ModulePath::memory(path.clone()));
        let imported = Query::find_imports(&Ast::parse(&types).0, &t, &h);
        let imports = imported.map(|x| format!("import {x}\n")).join("");

        // First, make sure that the types are well-formed and importable, return `Err` if not
        let before = format!("{imports}\n{types}\n");
        t.set_memory(vec![(path.clone(), Some(Arc::new(before.clone())))]);
        t.run(&[h.dupe()], Require::Everything);
        let errors = t.get_errors([&h]).collect_errors();
        if !errors.shown.is_empty() {
            let mut res = Vec::new();
            let project_root = PathBuf::new();
            for e in errors.shown {
                e.write_line(&mut Cursor::new(&mut res), project_root.as_path(), true)
                    .unwrap();
            }
            return Err(format!(
                "Errors from is_subtype `{lt}` <: `{gt}`\n{}\n\nSource code:\n{before}",
                str::from_utf8(&res).unwrap_or("UTF8 error")
            ));
        }

        // Now that we know the types are valid, check a snippet to do the actual subtype test.
        let after = format!("{imports}\n{types}\n{check}");
        t.set_memory(vec![(path, Some(Arc::new(after)))]);
        t.run(&[h.dupe()], Require::Everything);
        let errors = t.get_errors([&h]).collect_errors();
        Ok(errors.shown.is_empty())
    }

    /// Return `Err` if you can't resolve them to types, otherwise return `lt <: gt`.
    pub fn is_subtype(
        &self,
        name: ModuleName,
        path: PathBuf,
        lt: &str,
        gt: &str,
    ) -> Result<bool, String> {
        println!("At is_subtype top level");
        if gt == "TypedDictionary" || gt == "NonTotalTypedDictionary" {
            // For backward compatibility with Pyre, we allow `is_subset` comparison for checking if something
            // is a TypedDict. That isn't actually a valid subtype relationship, so we look for magic
            // attributes that in practice only exist on typed dicts.
            let types = format!("pyrefly_lt = ({lt})");
            let check =
                "pyrefly_lt.__required_keys__, pyrefly_lt.__optional_keys__, pyrefly_lt.__total__";
            self.check_code_snippet(name, path, lt, gt, types, check)
        } else {
            // In the normal case, synthesize a function whose return is a type error if the subset fails.
            let types = format!("type pyrefly_lt = ({lt})\ntype pyrefly_gt = ({gt})\n");
            let check = "def pyrefly_func(x: pyrefly_lt) -> pyrefly_gt:\n    return x";
            self.check_code_snippet(name, path, lt, gt, types, check)
        }
    }
}
