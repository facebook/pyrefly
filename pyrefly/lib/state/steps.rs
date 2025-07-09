/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use dupe::Dupe;
use enum_iterator::Sequence;
use parse_display::Display;
use paste::paste;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::uniques::UniqueFactory;
use ruff_python_ast::ModModule;

use crate::alt::answers::Answers;
use crate::alt::answers::LookupAnswer;
use crate::alt::answers::Solutions;
use crate::binding::bindings::Bindings;
use crate::config::base::UntypedDefBehavior;
use crate::error::style::ErrorStyle;
use crate::export::exports::Exports;
use crate::export::exports::LookupExport;
use crate::module::parse::module_parse;
use crate::solver::solver::Solver;
use crate::state::load::Load;
use crate::state::memory::MemoryFilesLookup;
use crate::state::require::Require;
use crate::types::stdlib::Stdlib;

pub struct Context<'a, Lookup> {
    pub require: Require,
    pub module: ModuleName,
    pub path: &'a ModulePath,
    pub sys_info: &'a SysInfo,
    pub memory: &'a MemoryFilesLookup<'a>,
    pub uniques: &'a UniqueFactory,
    pub stdlib: &'a Stdlib,
    pub lookup: &'a Lookup,
    pub untyped_def_behavior: UntypedDefBehavior,
}

#[derive(Debug, Default, Dupe, Clone)]
pub struct Steps {
    /// The last step that was computed.
    /// None means no steps have been computed yet.
    pub last_step: Option<Step>,
    pub load: Option<Arc<Load>>,
    pub ast: Option<Arc<ModModule>>,
    pub exports: Option<Exports>,
    pub answers: Option<Arc<(Bindings, Arc<Answers>)>>,
    pub solutions: Option<Arc<Solutions>>,
}

impl Steps {
    // The next step to compute, if any.
    pub fn next_step(&self) -> Option<Step> {
        match self.last_step {
            None => Some(Step::first()),
            Some(last) => last.next(),
        }
    }

    pub fn line_count(&self) -> usize {
        self.load
            .as_ref()
            .map_or(0, |load| load.module_info.line_count())
    }
}

#[derive(Debug, Clone, Copy, Dupe, Eq, PartialEq, PartialOrd, Ord)]
#[derive(Display, Sequence)]
pub enum Step {
    Load,
    Ast,
    Exports,
    Answers,
    Solutions,
}

pub struct ComputeStep<Lookup: LookupExport + LookupAnswer>(
    /// First you get given the `ModuleSteps`, from which you should grab what you need (cloning it).
    /// Second you get given the configs, from which you should compute the result.
    /// Third you get given the `ModuleSteps` to update.
    pub Box<dyn Fn(&Steps) -> Box<dyn FnOnce(&Context<Lookup>) -> Box<dyn FnOnce(&mut Steps)>>>,
);

macro_rules! compute_step {
    (<$ty:ty> $output:ident = $($input:ident),* $(,)? $(#$old_input:ident),*) => {
        ComputeStep(Box::new(|steps: &Steps| {
            let _ = steps; // Not used if $input is empty.
            $(let $input = steps.$input.dupe().unwrap();)*
            $(let $old_input = steps.$old_input.dupe();)*
            Box::new(move |ctx: &Context<$ty>| {
                let res = paste! { Step::[<step_ $output>] }(ctx, $($input,)* $($old_input,)* );
                Box::new(move |steps: &mut Steps| {
                    steps.$output = Some(res);
                    steps.last_step = Some(paste! { Step::[<$output:camel>] });
                })
            })
        }))
    };
}

// The steps within this module are all marked `inline(never)` and given
// globally unique names, so they are much easier to find in the profile.
impl Step {
    pub fn first() -> Self {
        Sequence::first().unwrap()
    }

    pub fn last() -> Self {
        Sequence::last().unwrap()
    }

    pub fn compute<Lookup: LookupExport + LookupAnswer>(self) -> ComputeStep<Lookup> {
        match self {
            Step::Load => compute_step!(<Lookup> load =),
            Step::Ast => compute_step!(<Lookup> ast = load),
            Step::Exports => compute_step!(<Lookup> exports = load, ast),
            Step::Answers => compute_step!(<Lookup> answers = load, ast, exports),
            Step::Solutions => compute_step!(<Lookup> solutions = load, answers),
        }
    }

    #[inline(never)]
    fn step_load<Lookup>(ctx: &Context<Lookup>) -> Arc<Load> {
        let error_style = if ctx.require.compute_errors() {
            ErrorStyle::Delayed
        } else {
            ErrorStyle::Never
        };
        let (code, self_error) = Load::load_from_path(ctx.path, ctx.memory);
        Arc::new(Load::load_from_data(
            ctx.module,
            ctx.path.dupe(),
            error_style,
            code,
            self_error,
        ))
    }

    #[inline(never)]
    fn step_ast<Lookup>(ctx: &Context<Lookup>, load: Arc<Load>) -> Arc<ModModule> {
        Arc::new(module_parse(
            load.module_info.contents(),
            ctx.sys_info.version(),
            &load.errors,
        ))
    }

    #[inline(never)]
    fn step_exports<Lookup>(
        ctx: &Context<Lookup>,
        load: Arc<Load>,
        ast: Arc<ModModule>,
    ) -> Exports {
        Exports::new(&ast.body, &load.module_info, ctx.sys_info)
    }

    #[inline(never)]
    fn step_answers<Lookup: LookupExport>(
        ctx: &Context<Lookup>,
        load: Arc<Load>,
        ast: Arc<ModModule>,
        exports: Exports,
    ) -> Arc<(Bindings, Arc<Answers>)> {
        let solver = Solver::new();
        let enable_index = ctx.require.keep_index();
        let enable_trace = ctx.require.keep_answers_trace();
        let bindings = Bindings::new(
            Arc::unwrap_or_clone(ast),
            load.module_info.dupe(),
            exports,
            &solver,
            ctx.lookup,
            ctx.sys_info,
            &load.errors,
            ctx.uniques,
            enable_trace,
            ctx.untyped_def_behavior,
        );
        let answers = Answers::new(&bindings, solver, enable_index, enable_trace);
        Arc::new((bindings, Arc::new(answers)))
    }

    #[inline(never)]
    fn step_solutions<Lookup: LookupExport + LookupAnswer>(
        ctx: &Context<Lookup>,
        load: Arc<Load>,
        answers: Arc<(Bindings, Arc<Answers>)>,
    ) -> Arc<Solutions> {
        let solutions = answers.1.solve(
            ctx.lookup,
            ctx.lookup,
            &answers.0,
            &load.errors,
            ctx.stdlib,
            ctx.uniques,
            ctx.require.compute_errors()
                || ctx.require.keep_answers_trace()
                || ctx.require.keep_answers(),
        );
        Arc::new(solutions)
    }
}
