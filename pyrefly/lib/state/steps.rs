/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ops::Deref;
use std::sync::Arc;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering;

use dupe::Dupe;
use enum_iterator::Sequence;
use parse_display::Display;
use paste::paste;
use pyrefly_build::handle::Handle;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::uniques::UniqueFactory;
use ruff_python_ast::ModModule;
use ruff_python_ast::token::Tokens;

use crate::alt::answers::Answers;
use crate::alt::answers::LookupAnswer;
use crate::alt::answers::Solutions;
use crate::binding::bindings::Bindings;
use crate::config::base::InferReturnTypes;
use crate::config::base::RecursionLimitConfig;
use crate::error::style::ErrorStyle;
use crate::export::exports::Exports;
use crate::export::exports::LookupExport;
use crate::module::parse::module_parse;
use crate::solver::solver::Solver;
use crate::solver::solver::SolverConfig;
use crate::state::load::Load;
use crate::state::memory::MemoryFilesLookup;
use crate::state::require::Require;
use crate::state::state::OldData;
use crate::state::state::TransactionTimingCounters;
use crate::state::step_slot::StepSlot;
use crate::types::stdlib::Stdlib;

/// Context for pysa data extraction during the Solutions step.
pub struct PysaContext<'a> {
    pub handle: &'a Handle,
    pub module_ids: &'a crate::report::pysa::module::ModuleIds,
    pub stdlib: Arc<Stdlib>,
}

pub struct Context<'a, Lookup> {
    pub require: Require,
    pub module: ModuleName,
    pub path: &'a ModulePath,
    pub sys_info: &'a SysInfo,
    pub memory: &'a MemoryFilesLookup<'a>,
    pub uniques: &'a UniqueFactory,
    pub stdlib: &'a Stdlib,
    pub lookup: &'a Lookup,
    pub check_unannotated_defs: bool,
    pub infer_return_types: InferReturnTypes,
    pub infer_with_first_use: bool,
    pub tensor_shapes: bool,
    pub strict_callable_subtyping: bool,
    pub strict_partial_subtyping: bool,
    pub spec_compliant_overloads: bool,
    pub legacy_overload_expansion: bool,
    pub treat_all_caps_as_final: bool,
    pub recursion_limit_config: Option<RecursionLimitConfig>,
    /// Pysa context for building PysaSolutions during the Solutions step.
    pub pysa_context: Option<PysaContext<'a>>,
    /// Build compact CinderX solutions during the Solutions step.
    pub cinderx_enabled: bool,
    /// Timing counters for filesystem stat/read latency tracking.
    pub timing: Option<&'a TransactionTimingCounters>,
}

/// AST and lexer tokens produced by the same parser invocation.
/// Tokens are only retained when the require level is `Everything`
/// (i.e. open files in the LSP), since they are only needed for
/// semantic token highlighting.
#[derive(Debug, Dupe, Clone)]
pub struct ParsedModule {
    module: Arc<ModModule>,
    tokens: Option<Arc<Tokens>>,
}

impl ParsedModule {
    pub fn new(module: ModModule, tokens: Option<Tokens>) -> Self {
        Self {
            module: Arc::new(module),
            tokens: tokens.map(Arc::new),
        }
    }

    pub fn module(&self) -> Arc<ModModule> {
        self.module.dupe()
    }

    pub fn tokens(&self) -> Option<Arc<Tokens>> {
        self.tokens.dupe()
    }
}

impl Deref for ParsedModule {
    type Target = ModModule;

    fn deref(&self) -> &Self::Target {
        self.module.as_ref()
    }
}

#[derive(Debug, Default, Dupe, Clone)]
pub struct Steps {
    /// The last step that was computed.
    /// None means no steps have been computed yet.
    pub last_step: Option<Step>,
    pub load: Option<Arc<Load>>,
    pub ast: Option<Arc<ParsedModule>>,
    pub exports: Option<Arc<Exports>>,
    pub answers: Option<Arc<Answers>>,
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
    /// Variant name as a static string. Used by demand-tree consumers
    /// (`pyrefly check --report-demand-tree` and the laziness test
    /// framework) so the label survives across crates without forcing
    /// callers to format through `Display`.
    pub fn label(self) -> &'static str {
        match self {
            Step::Load => "Load",
            Step::Ast => "Ast",
            Step::Exports => "Exports",
            Step::Answers => "Answers",
            Step::Solutions => "Solutions",
        }
    }

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
///   1. Gets inputs from `StepsMut` fields via `clone_arc().unwrap()`
///      (or `clone_arc()` for inputs suffixed with `?`, yielding `Option`)
///   2. Calls `Step::step_$output(ctx, inputs...)`
///   3. Stores the result via ArcSwap
macro_rules! compute_step {
    // Entry point: parse comma-separated inputs, then delegate to @exec.
    ($steps:ident, $ctx:ident, $output:ident = $($rest:tt)*) => {{
        compute_step!(@exec $steps, $ctx, $output, [] $($rest)*);
    }};
    // Base case: all inputs consumed, emit the step call.
    (@exec $steps:ident, $ctx:ident, $output:ident, [$($input:ident)*]) => {{
        let res = paste! { Step::[<step_ $output>] }($ctx, $($input,)*);
        $steps.$output.store(Some(res));
    }};
    // Optional input (name?): load as Option (no unwrap).
    (@exec $steps:ident, $ctx:ident, $output:ident, [$($acc:ident)*] $input:ident ? $(, $($rest:tt)*)?) => {{
        let $input = $steps.$input.clone_arc();
        compute_step!(@exec $steps, $ctx, $output, [$($acc)* $input] $($($rest)*)?);
    }};
    // Required input (name): load and unwrap.
    (@exec $steps:ident, $ctx:ident, $output:ident, [$($acc:ident)*] $input:ident $(, $($rest:tt)*)?) => {{
        let $input = $steps.$input.clone_arc().unwrap();
        compute_step!(@exec $steps, $ctx, $output, [$($acc)* $input] $($($rest)*)?);
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
///
/// The slots are private so every borrowed read goes through [`StepSlot::with`],
/// which holds the debt-bearing ArcSwap guard only while its callback runs, so
/// no guard outlives a borrow of the slot. This invariant lets [`StepSlot::into_inner`] skip the debt
/// handoff. Do not make these fields public.
#[derive(Debug)]
pub struct StepsMut {
    current_step: AtomicStep,
    load: StepSlot<Load>,
    ast: StepSlot<ParsedModule>,
    exports: StepSlot<Exports>,
    answers: StepSlot<Answers>,
    solutions: StepSlot<Solutions>,
}

impl StepsMut {
    /// Create from frozen `Steps`.
    pub fn from_frozen(steps: &Steps) -> Self {
        Self {
            current_step: AtomicStep::new(steps.last_step),
            load: StepSlot::new(steps.load.dupe()),
            ast: StepSlot::new(steps.ast.dupe()),
            exports: StepSlot::new(steps.exports.dupe()),
            answers: StepSlot::new(steps.answers.dupe()),
            solutions: StepSlot::new(steps.solutions.dupe()),
        }
    }

    /// Create an empty `StepsMut` with no steps computed.
    pub fn new() -> Self {
        Self {
            current_step: AtomicStep::new(None),
            load: StepSlot::new(None),
            ast: StepSlot::new(None),
            exports: StepSlot::new(None),
            answers: StepSlot::new(None),
            solutions: StepSlot::new(None),
        }
    }

    /// Create a `StepsMut` with pre-computed load data, marking the Load
    /// step as completed. Used by `report_timings` to re-run subsequent
    /// steps without re-doing I/O.
    pub fn new_loaded(load: Arc<Load>) -> Self {
        Self {
            current_step: AtomicStep::new(Some(Step::Load)),
            load: StepSlot::new(Some(load)),
            ast: StepSlot::new(None),
            exports: StepSlot::new(None),
            answers: StepSlot::new(None),
            solutions: StepSlot::new(None),
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
            .clone_arc()
            .as_ref()
            .map_or(0, |load| load.module_info.line_count())
    }

    /// Compute a step.
    ///
    /// This method:
    /// 1. Reads inputs from slots (via ArcSwap)
    /// 2. Calls the appropriate `Step::step_*` function
    /// 3. Stores the result via ArcSwap
    /// 4. Release-stores `current_step`
    ///
    /// Old data for diffing is stored in `old_*` fields by `reset_for_rebuild()`,
    /// not captured here.
    pub fn compute<Lookup: LookupExport + LookupAnswer>(&self, step: Step, ctx: &Context<Lookup>) {
        match step {
            Step::Load => compute_step!(self, ctx, load =),
            Step::Ast => compute_step!(self, ctx, ast = load),
            Step::Exports => compute_step!(self, ctx, exports = load, ast),
            Step::Answers => compute_step!(self, ctx, answers = load, ast, exports),
            Step::Solutions => compute_step!(self, ctx, solutions = load, ast?, answers),
        }
        // Release-store current_step: readers seeing this value are guaranteed
        // to see the step data stored above.
        self.current_step.store_completed(step);
    }

    /// Reset steps for recomputation. Optionally clears AST, always clears
    /// exports/answers/solutions (returning them as `OldData` for later diffing).
    /// Uses relaxed ordering — caller is responsible for a subsequent release-store
    /// on another variable (e.g. `checked` epoch) to make these writes visible.
    pub(crate) fn reset_for_rebuild(&self, clear_ast: bool, old: &mut OldData) {
        if clear_ast {
            self.ast.store(None);
        }

        // Determine the new last_step value based on what data remains.
        // This must be computed AFTER clearing/storing data above.
        let new_last_step = if clear_ast || self.ast.clone_arc().is_none() {
            if self.load.clone_arc().is_some() {
                Some(Step::Load)
            } else {
                None
            }
        } else {
            Some(Step::Ast)
        };

        // Take and clear exports/answers/solutions, saving for diffing at Solutions step.
        old.exports = self.exports.swap(None);
        old.answers = self.answers.swap(None);
        old.solutions = self.solutions.swap(None);

        // Relaxed is fine here because the caller will release-store on `checked`,
        // which synchronizes all these writes with readers.
        self.current_step.store(new_last_step, Ordering::Relaxed);
    }

    /// Consume and produce frozen `Steps` without ArcSwap's debt handoff.
    ///
    /// Every borrowed read is callback-scoped by [`StepSlot::with`], so all
    /// debt-bearing guards have been dropped before `self` can be consumed.
    pub fn take_and_freeze(self) -> Steps {
        Steps {
            last_step: self.current_step.load(),
            load: self.load.into_inner(),
            ast: self.ast.into_inner(),
            exports: self.exports.into_inner(),
            answers: self.answers.into_inner(),
            solutions: self.solutions.into_inner(),
        }
    }

    /// The last step that completed, if any.
    pub fn last_step(&self) -> Option<Step> {
        self.current_step.load()
    }

    pub fn get_load(&self) -> Option<Arc<Load>> {
        self.load.clone_arc()
    }

    pub fn get_ast(&self) -> Option<Arc<ParsedModule>> {
        self.ast.clone_arc()
    }

    pub fn get_exports(&self) -> Option<Arc<Exports>> {
        self.exports.clone_arc()
    }

    pub fn get_answers(&self) -> Option<Arc<Answers>> {
        self.answers.clone_arc()
    }

    pub fn get_solutions(&self) -> Option<Arc<Solutions>> {
        self.solutions.clone_arc()
    }

    pub fn store_load(&self, load: Option<Arc<Load>>) {
        self.load.store(load);
    }

    pub fn clear_ast(&self) {
        self.ast.store(None);
    }

    pub fn clear_answers(&self) {
        self.answers.store(None);
    }

    pub fn with_answers<R>(&self, f: impl for<'a> FnOnce(Option<&'a Answers>) -> R) -> R {
        self.answers.with(f)
    }

    pub fn with_solutions<R>(&self, f: impl for<'a> FnOnce(Option<&'a Solutions>) -> R) -> R {
        self.solutions.with(f)
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
        let (file_contents, self_error) = Load::load_from_path(ctx.path, ctx.memory, ctx.timing);
        Arc::new(Load::load_from_data(
            ctx.module,
            ctx.path.dupe(),
            error_style,
            file_contents,
            self_error,
        ))
    }

    #[inline(never)]
    fn step_ast<Lookup>(ctx: &Context<Lookup>, load: Arc<Load>) -> Arc<ParsedModule> {
        let (module, tokens, ignore) = module_parse(
            load.module_info.contents(),
            ctx.sys_info.version(),
            load.module_info.source_type(),
            &load.errors,
            ctx.require.keep_ast(),
        );
        load.module_info.initialize_ignore(ignore);
        Arc::new(ParsedModule::new(module, tokens))
    }

    #[inline(never)]
    fn step_exports<Lookup>(
        ctx: &Context<Lookup>,
        load: Arc<Load>,
        ast: Arc<ParsedModule>,
    ) -> Arc<Exports> {
        let build_symbols =
            ctx.require.keep_index() && load.module_info.path().is_first_party_for_indexing();
        Arc::new(Exports::new(
            &ast.body,
            &load.module_info,
            *ctx.sys_info,
            build_symbols,
        ))
    }

    #[inline(never)]
    fn step_answers<Lookup: LookupExport>(
        ctx: &Context<Lookup>,
        load: Arc<Load>,
        ast: Arc<ParsedModule>,
        exports: Arc<Exports>,
    ) -> Arc<Answers> {
        let solver = Solver::new(SolverConfig {
            infer_with_first_use: ctx.infer_with_first_use,
            tensor_shapes: ctx.tensor_shapes,
            strict_callable_subtyping: ctx.strict_callable_subtyping,
            strict_partial_subtyping: ctx.strict_partial_subtyping,
            spec_compliant_overloads: ctx.spec_compliant_overloads,
            legacy_overload_expansion: ctx.legacy_overload_expansion,
        });
        let ast = ast.module();
        let enable_index = ctx.require.keep_index();
        let enable_trace =
            ctx.require.keep_answers_trace() || ctx.pysa_context.is_some() || ctx.cinderx_enabled;
        let bindings = Bindings::new(
            Arc::unwrap_or_clone(ast),
            load.module_info.dupe(),
            &exports,
            &solver,
            ctx.lookup,
            *ctx.sys_info,
            &load.errors,
            enable_trace,
            ctx.check_unannotated_defs,
            ctx.require.keep_index(),
            ctx.infer_return_types,
            ctx.treat_all_caps_as_final,
        );
        Arc::new(Answers::new(bindings, solver, enable_index, enable_trace))
    }

    #[inline(never)]
    fn step_solutions<Lookup: LookupExport + LookupAnswer>(
        ctx: &Context<Lookup>,
        load: Arc<Load>,
        ast: Option<Arc<ParsedModule>>,
        answers: Arc<Answers>,
    ) -> Arc<Solutions> {
        let pysa_context = ctx.pysa_context.as_ref().map(|pysa_context| {
            crate::report::pysa::context::ModuleAnswersContext {
                handle: pysa_context.handle.dupe(),
                module_id: pysa_context.module_ids.get_from_handle(pysa_context.handle),
                module_info: load.module_info.dupe(),
                stdlib: pysa_context.stdlib.dupe(),
                ast: ast
                    .expect("AST must be available when pysa is enabled")
                    .module(),
                answers: answers.dupe(),
            }
        });

        let solutions = answers.solve(
            ctx.lookup,
            ctx.lookup,
            &load.errors,
            ctx.stdlib,
            ctx.uniques,
            ctx.require.compute_errors()
                || ctx.require.keep_answers_trace()
                || ctx.require.keep_answers()
                || ctx.pysa_context.is_some()
                || ctx.cinderx_enabled,
            ctx.recursion_limit_config,
            pysa_context.as_ref(),
            ctx.cinderx_enabled,
        );

        Arc::new(solutions)
    }
}
