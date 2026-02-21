/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering;

use arc_swap::ArcSwapOption;
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
use crate::config::base::RecursionLimitConfig;
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
    pub infer_with_first_use: bool,
    pub tensor_shapes: bool,
    pub recursion_limit_config: Option<RecursionLimitConfig>,
}

#[derive(Debug, Default, Dupe, Clone)]
pub struct Steps {
    /// The last step that was computed.
    /// None means no steps have been computed yet.
    pub last_step: Option<Step>,
    pub load: Option<Arc<Load>>,
    pub ast: Option<Arc<ModModule>>,
    pub exports: Option<Arc<Exports>>,
    pub answers: Option<Arc<(Bindings, Arc<Answers>)>>,
    pub solutions: Option<Arc<Solutions>>,
}

impl Steps {
    pub fn line_count(&self) -> usize {
        self.load
            .as_ref()
            .map_or(0, |load| load.module_info.line_count())
    }
}

const STEP_LOAD: u8 = 0;
const STEP_AST: u8 = 1;
const STEP_EXPORTS: u8 = 2;
const STEP_ANSWERS: u8 = 3;
const STEP_SOLUTIONS: u8 = 4;

/// Sentinel value representing no step computed.
const STEP_NONE: u8 = 0xFF;

#[derive(Debug, Clone, Copy, Dupe, Eq, PartialEq, PartialOrd, Ord)]
#[derive(Display, Sequence)]
pub enum Step {
    Load = STEP_LOAD as isize,
    Ast = STEP_AST as isize,
    Exports = STEP_EXPORTS as isize,
    Answers = STEP_ANSWERS as isize,
    Solutions = STEP_SOLUTIONS as isize,
}

impl Step {
    /// Encode a step as a u8 for atomic storage.
    fn to_u8(self) -> u8 {
        self as u8
    }

    /// Decode a u8 back to a Step. Panics on invalid values.
    fn from_u8(v: u8) -> Self {
        match v {
            STEP_LOAD => Step::Load,
            STEP_AST => Step::Ast,
            STEP_EXPORTS => Step::Exports,
            STEP_ANSWERS => Step::Answers,
            STEP_SOLUTIONS => Step::Solutions,
            _ => panic!("Invalid Step encoding: {v}"),
        }
    }
}

/// Atomic storage for `Option<Step>`, using `AtomicU8` with a sentinel
/// for `None`. Encapsulates the `Step` <-> `u8` encoding.
#[derive(Debug)]
pub struct AtomicStep(AtomicU8);

impl AtomicStep {
    pub fn new(step: Option<Step>) -> Self {
        Self(AtomicU8::new(Self::encode(step)))
    }

    /// Acquire-load the current step.
    pub fn load(&self) -> Option<Step> {
        Self::decode(self.0.load(Ordering::Acquire))
    }

    /// Store a step with the given ordering.
    pub fn store(&self, step: Option<Step>, order: Ordering) {
        self.0.store(Self::encode(step), order);
    }

    /// Store a specific completed step with release ordering.
    /// This is the synchronization point: readers seeing this value
    /// are guaranteed to see the step data stored before this call.
    pub fn store_completed(&self, step: Step) {
        self.0.store(step.to_u8(), Ordering::Release);
    }

    fn encode(step: Option<Step>) -> u8 {
        match step {
            None => STEP_NONE,
            Some(s) => s.to_u8(),
        }
    }

    fn decode(v: u8) -> Option<Step> {
        if v == STEP_NONE {
            None
        } else {
            Some(Step::from_u8(v))
        }
    }
}

// ---------------------------------------------------------------------------
// StepsMut — lock-free step data storage
// ---------------------------------------------------------------------------

/// For each step:
///   1. Gets inputs from `StepsMut` fields via `load_full().unwrap()`
///   2. Saves old data from the output slot into `$old`
///   3. Calls `Step::step_$output(ctx, inputs...)`
///   4. Stores the result via ArcSwap
macro_rules! compute_step {
    ($steps:ident, $old:ident, $ctx:ident, $output:ident = $($input:ident),* $(,)?) => {{
        $(let $input = $steps.$input.load_full().unwrap();)*
        $old.$output = $steps.$output.load_full();
        let res = paste! { Step::[<step_ $output>] }($ctx, $($input,)*);
        $steps.$output.store(Some(res));
    }};
}

/// Lock-free storage for step computation results.
///
/// Each slot is an `ArcSwapOption`, allowing concurrent readers to atomically
/// load `Arc` references while writers store new values. `current_step` is the
/// synchronization point between writers and readers: a reader seeing
/// `current_step >= X` is guaranteed that the data for step X has been stored.
///
/// Also usable standalone (outside `ModuleStateMut`) for isolated step
/// computation, e.g. in `report_timings`.
#[derive(Debug)]
pub struct StepsMut {
    pub current_step: AtomicStep,
    pub load: ArcSwapOption<Load>,
    pub ast: ArcSwapOption<ModModule>,
    pub exports: ArcSwapOption<Exports>,
    pub answers: ArcSwapOption<(Bindings, Arc<Answers>)>,
    pub solutions: ArcSwapOption<Solutions>,
}

impl StepsMut {
    /// Create from frozen `Steps`.
    pub fn from_frozen(steps: &Steps) -> Self {
        Self {
            current_step: AtomicStep::new(steps.last_step),
            load: ArcSwapOption::new(steps.load.dupe()),
            ast: ArcSwapOption::new(steps.ast.dupe()),
            exports: ArcSwapOption::new(steps.exports.dupe()),
            answers: ArcSwapOption::new(steps.answers.dupe()),
            solutions: ArcSwapOption::new(steps.solutions.dupe()),
        }
    }

    /// Create an empty `StepsMut` with no steps computed.
    pub fn new() -> Self {
        Self {
            current_step: AtomicStep::new(None),
            load: ArcSwapOption::empty(),
            ast: ArcSwapOption::empty(),
            exports: ArcSwapOption::empty(),
            answers: ArcSwapOption::empty(),
            solutions: ArcSwapOption::empty(),
        }
    }

    /// Create a `StepsMut` with pre-computed load data, marking the Load
    /// step as completed. Used by `report_timings` to re-run subsequent
    /// steps without re-doing I/O.
    pub fn new_loaded(load: Arc<Load>) -> Self {
        Self {
            current_step: AtomicStep::new(Some(Step::Load)),
            load: ArcSwapOption::from(Some(load)),
            ast: ArcSwapOption::empty(),
            exports: ArcSwapOption::empty(),
            answers: ArcSwapOption::empty(),
            solutions: ArcSwapOption::empty(),
        }
    }

    /// The next step to compute, if any.
    pub fn next_step(&self) -> Option<Step> {
        match self.current_step.load() {
            None => Some(Step::first()),
            Some(last) => last.next(),
        }
    }

    pub fn line_count(&self) -> usize {
        self.load
            .load_full()
            .as_ref()
            .map_or(0, |load| load.module_info.line_count())
    }

    /// Compute a step, writing the old data for the computed step into `old`.
    ///
    /// This method:
    /// 1. Reads inputs from slots (via ArcSwap)
    /// 2. Saves the old value of the output slot into `old`
    /// 3. Calls the appropriate `Step::step_*` function
    /// 4. Stores the result via ArcSwap
    /// 5. Release-stores `current_step`
    pub fn compute<Lookup: LookupExport + LookupAnswer>(
        &self,
        step: Step,
        old: &mut Steps,
        ctx: &Context<Lookup>,
    ) {
        match step {
            Step::Load => compute_step!(self, old, ctx, load =),
            Step::Ast => compute_step!(self, old, ctx, ast = load),
            Step::Exports => compute_step!(self, old, ctx, exports = load, ast),
            Step::Answers => compute_step!(self, old, ctx, answers = load, ast, exports),
            Step::Solutions => compute_step!(self, old, ctx, solutions = load, answers),
        }
        // Release-store current_step: readers seeing this value are guaranteed
        // to see the step data stored above.
        self.current_step.store_completed(step);
    }

    /// Reset steps for recomputation. Optionally clears AST, always clears answers.
    /// Uses relaxed ordering — caller is responsible for a subsequent release-store
    /// on another variable (e.g. `checked` epoch) to make these writes visible.
    pub fn reset_for_rebuild(&self, clear_ast: bool) {
        if clear_ast {
            self.ast.store(None);
        }

        // Determine the new last_step value based on what data remains.
        // This must be computed AFTER clearing/storing data above.
        let new_last_step = if clear_ast || self.ast.load_full().is_none() {
            if self.load.load_full().is_some() {
                Some(Step::Load)
            } else {
                None
            }
        } else {
            Some(Step::Ast)
        };

        // Do not clear solutions or exports, since we use them for equality comparison.
        self.answers.store(None);

        // Relaxed is fine here because the caller will release-store on `checked`,
        // which synchronizes all these writes with readers.
        self.current_step.store(new_last_step, Ordering::Relaxed);
    }

    /// Drain all step data into a frozen `Steps`. The `StepsMut` should not be
    /// reused after this call (it becomes empty).
    pub fn take_and_freeze(&self) -> Steps {
        let last_step = self.current_step.load();
        let load = self.load.swap(None);
        let ast = self.ast.swap(None);
        let exports_arc = self.exports.swap(None);
        let answers = self.answers.swap(None);
        let solutions = self.solutions.swap(None);
        Steps {
            last_step,
            load,
            ast,
            exports: exports_arc,
            answers,
            solutions,
        }
    }
}

// ---------------------------------------------------------------------------
// Step computation functions
// ---------------------------------------------------------------------------

// The steps within this module are all marked `inline(never)` and given
// globally unique names, so they are much easier to find in the profile.
impl Step {
    pub fn first() -> Self {
        Sequence::first().unwrap()
    }

    pub fn last() -> Self {
        Sequence::last().unwrap()
    }

    #[inline(never)]
    fn step_load<Lookup>(ctx: &Context<Lookup>) -> Arc<Load> {
        let error_style = if ctx.require.compute_errors() {
            ErrorStyle::Delayed
        } else {
            ErrorStyle::Never
        };
        let (file_contents, self_error) = Load::load_from_path(ctx.path, ctx.memory);
        Arc::new(Load::load_from_data(
            ctx.module,
            ctx.path.dupe(),
            error_style,
            file_contents,
            self_error,
        ))
    }

    #[inline(never)]
    fn step_ast<Lookup>(ctx: &Context<Lookup>, load: Arc<Load>) -> Arc<ModModule> {
        Arc::new(module_parse(
            load.module_info.contents(),
            ctx.sys_info.version(),
            load.module_info.source_type(),
            &load.errors,
        ))
    }

    #[inline(never)]
    fn step_exports<Lookup>(
        ctx: &Context<Lookup>,
        load: Arc<Load>,
        ast: Arc<ModModule>,
    ) -> Arc<Exports> {
        Arc::new(Exports::new(&ast.body, &load.module_info, ctx.sys_info))
    }

    #[inline(never)]
    fn step_answers<Lookup: LookupExport>(
        ctx: &Context<Lookup>,
        load: Arc<Load>,
        ast: Arc<ModModule>,
        exports: Arc<Exports>,
    ) -> Arc<(Bindings, Arc<Answers>)> {
        let solver = Solver::new(ctx.infer_with_first_use, ctx.tensor_shapes);
        let enable_index = ctx.require.keep_index();
        let enable_trace = ctx.require.keep_answers_trace();
        let bindings = Bindings::new(
            Arc::unwrap_or_clone(ast),
            load.module_info.dupe(),
            &exports,
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
            ctx.recursion_limit_config,
        );
        Arc::new(solutions)
    }
}
