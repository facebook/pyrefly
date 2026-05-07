# Laziness Opportunities

Observations from the laziness test demand trees that represent
opportunities to reduce unnecessary work.

## Methodology for the percentages below

Numbers come from running `pyrefly check --report-demand-tree <out>.json <file>` on a sample of real-world files and aggregating the resulting JSONs. Each `Exports` edge has `{from, target, kind.reason}`; each `Answer` edge has `{from, target, kind.key}`. Edges are recursive — an Answer span has `children` of nested edges, so aggregation walks every edge in the tree.

Statistics:

- **"X% of demands by reason"** — recursively walk every edge; count edges where `kind.reason == X`, divide by the relevant total (all edges, or all `Exports` edges).
- **Module steps**: aggregated from each report's `module_steps` array — count modules whose `last_step` is `Exports` / `Answers` / `Solutions`.
- **"X% of dep modules at Exports only with reason Y"** — among modules whose `last_step` is `Exports`, take the set of `Exports` reasons appearing on edges targeting them; count modules whose set equals (or contains) `{Y}`.

These numbers are from a single sampled run, not a continuous benchmark. Re-run on your own files to verify — the relative magnitudes are the load-bearing claim, not the precise percentages.

## Impact ranking (from `--report-demand-tree` on real-world files)

Aggregated over 25 hot files from one large codebase. Answer-side
demands account for the majority of cross-module work; most touched
dependency modules end at `Step::Exports` because `is_special_export`
forced their export sets during binding.

1. **MRO walk without early exit** (~64% of all demands across class
   keys) — `KeyClassSynthesizedFields` is ~24% of all demands (~36% of
   Answer), `KeyClassMetadata` ~23% (~35% of Answer), `KeyClassMro` ~9%
   (~13% of Answer), `KeyClassField` ~8% (~12% of Answer). Worst for
   files with deep class hierarchies. (See "MRO computed even when
   attribute is on the class itself" and "Multiple inheritance check
   resolves unique parent fields" below.)
2. **`is_special_export` during binding** (~23% of all demands; ~87%
   of `Exports` demands) — the dominant `LookupExport` reason by a wide
   margin. Fires for type-var classification (`T = TypeVar(...)`,
   `P = ParamSpec(...)`, etc.) and other special-form recognition
   during binding.
3. **Full function signature on import** — resolves 11 keys per imported
   function regardless of usage.
4. **Class metadata for annotation-only usage** — `is_typed_dict()` check
   on every class-as-annotation.
5. **Eagerly resolved builtins** — 150 KeyExport + cascading demands per
   module. Fixed cost but adds up.
6. **Multiple inheritance check resolves unique fields** — low volume but
   real waste for wide hierarchies.

Of the ~46k dependency modules touched across the 25 sampled files,
~77% are forced to `Step::Exports` only; <0.1% stop at `Step::Load`,
~23% reach `Step::Answers`, and ~0% reach `Step::Solutions`. The few
remaining `Step::Load` modules are submodules touched by
`solve_import`'s submodule-fallback or `attr.rs`'s module-attribute
cascade — both are inherently solve-time.

## Eagerly resolved builtins

Every module resolves ~150 `KeyExport` entries from `builtins`
(int, str, bool, list, dict, OverflowError, ValueError, ...) even
if the code never references them. This is because `inject_builtins`
creates `Binding::Import` for every name returned by
`get_wildcard(builtins)`, and the solver resolves all bindings.

**Visible in:** Every test — `(N builtin demands hidden)` line.

**Ideal behavior:** Only resolve builtin names that are actually
referenced in the module's code. Unreferenced builtins should not
trigger cross-module key solving.

**Root cause:** `inject_builtins` in `bindings.rs` imports all
builtin names via wildcard. The solver resolves every binding
including unused ones.

## Full function signature resolved on import

When `from b import helper` is written, pyrefly resolves the full
function signature even if `helper` is never called. The demand tree
for both `test_import_function_unused` and `test_import_function_called`
shows identical work: `a -> b::KeyExport("helper")` triggers resolving
11 keys in `b` including return annotation, decorated/undecorated
function, and legacy type param checks.

**Visible in:** `test_import_function_unused.md` and
`test_import_function_called.md` — identical demand trees despite
different usage.

**Ideal behavior:** `KeyExport(helper)` should return a lightweight
handle (function name + scope ID). The signature should only be
resolved when the function is actually called or its type is inspected.

**Root cause:** `KeyExport` for a function forwards to the function's
`Key::Definition`, which triggers the full solve chain including
`KeyDecoratedFunction` → `KeyUndecoratedFunction` → return annotation
→ legacy type param checks.

## Class metadata resolved for annotation-only usage

When a class is used only as a type annotation (`x: Foo`), pyrefly
resolves `KeyClassMetadata` (1 cross-module demand). This is much
better than instantiation (6 metadata demands) but still unnecessary.

**Visible in:** `test_import_class_as_annotation.md` — 2 demands:
`KeyExport("Foo")` + `KeyClassMetadata(0)`.

**Why it happens:** `type_of_instance` in targs.rs:358 calls
`get_metadata_for_class` to check `is_typed_dict()`. This is needed
because `Type::ClassType` and `Type::TypedDict` are different enum
variants — the code must decide which to construct when promoting a
class reference to a type form. The `is_typed_dict` check requires
resolving the class's base classes to see if any ancestor is
`TypedDict`, which can cascade up the MRO.

**Ideal behavior:** Unify `ClassType` and `TypedDict` into a single
type variant so the TypedDict distinction can be deferred until
TypedDict-specific features are actually used (field lookup,
structural matching, etc.). This would eliminate the metadata demand
for annotation-only class usage.

## LookupExport during binding forces transitive exports

Every `LookupExport` method (`module_exists`, `export_exists`,
`get_wildcard`, `is_special_export`, `is_final`, `get_deprecated`)
calls `demand(Step::Exports)` on the target module. These calls happen
during `Bindings::new`, meaning that when module B is being bound, ALL
of B's imports have their exports eagerly computed — even if the caller
(module A) never uses anything from those transitive dependencies.

**Visible in:** 5 tests show `c: Exports` with no demand tree entries
pointing to C:
- `test_import_star_forces_exports` — `from c import *` triggers `get_wildcard(c)`
- `test_export_exists_forces_exports` — `from c import Foo` triggers `export_exists(c, "Foo")`
- `test_deprecated_forces_exports` — `from c import old_func` triggers `get_deprecated(c, "old_func")`
- `test_is_final_forces_exports` — `X = 2` (reassigning import) triggers `is_final(c, "X")`
- `test_special_export_forces_exports` — `T = MyTypeVar("T")` triggers `is_special_export(c, "MyTypeVar")`

`test_bare_import_forces_exports` shows `c: Nothing` — the bare
`import c` no longer demands `c` at bind time, and nothing else
forces it. Also visible in `test_unused_import_from_same_module` and
`test_transitive_import_annotated` where `c: Exports` appears (other
LookupExport calls fire on `c` even though no name from `c` is used).

**Real-world impact:** `Exports(is_special_export)` accounts for ~23%
of all cross-module demands and ~87% of `Exports` demands — by far
the dominant remaining `LookupExport` reason. ~77% of dependency
modules end at `Step::Exports` only; almost none stop at `Step::Load`
(see Methodology above).

**Ideal behavior:** Module C should not be computed at all (`c: Nothing`)
when A doesn't need it. B's bindings should be constructible without
demanding exports from B's own imports.

**Root cause:** Bind-time call sites in `bindings.rs`, `stmt.rs`, and
`scope.rs` use `self.lookup.*`, which goes through
`TransactionHandle::with_exports` → `lookup_export` →
`demand(Step::Exports)`. Each remaining call forces the target module's
exports during binding.

**Where calls still fire at bind time:**

- `is_special_export` for `TypeVar` / `ParamSpec` / `TypeVarTuple` /
  `NewType` / `NamedTuple` / `Final` / `ClassVar` / etc. — controls
  which `Binding` variant is created. Could use syntactic name
  matching for the common case with solve-time validation for
  re-exports.
- `is_final` reassignment check — fires when a name imported from
  another module is reassigned to a different binding, to confirm
  the imported name is not `Final`. Duplicate re-imports of the
  same `(module, name)` short-circuit before the lookup, so files
  with many `if TYPE_CHECKING:` / method-local re-imports of the
  same names don't pay this cost.
- `get_wildcard` for `from X import *` — unavoidable; binder needs
  the set of names to create binding table entries.
- `module_exists` for builtins — unavoidable; fixed per-module cost.

**Where calls already fire only at solve time** (so they don't force
exports or load during binding):

- `export_exists` for cross-module `from X import Y` — `Binding::Import`
  carries an `ImportFallback`; solver does the
  `export → submodule → __getattr__ → error` cascade.
- `get_deprecated` for explicit `from X import Y` — deprecation warning
  fires at solve time only when the imported name actually exports.
- `is_special_export` for `typing.Self` — solve-time `untype_opt`
  recognizes `Type::Type[SpecialForm(SelfType)]`; the bind side records
  one `class_scopes` entry per class instead of one per use.
- `module_exists` for `import X` and `from X import …` — `Binding::Module`
  carries an optional error range; the solver runs the
  missing-module check (still demanding `Step::Load` so incremental
  re-check picks up edits) only for bindings that are actually solved,
  letting unused imports in transitive deps stay at `Step::Nothing`.

## Note on repeated demands to the same key

Multiple code paths may demand the same key (e.g., `KeyClassMetadata`)
during a single operation like class instantiation. This is NOT a
concern — the Calculation cell caches the result, so repeated lookups
are just hash lookups + Arc clones with negligible cost. The demand
trees in the tests show these repeated lookups but they are not
optimization targets. Only demands for truly UNNECESSARY keys matter.

## MRO computed even when attribute is on the class itself

When accessing `c.child_attr` where `child_attr` is defined on `Child`
(not inherited), pyrefly still walks the full MRO, resolving parent
class `Base` from module `c`.

**Visible in:** `test_attribute_on_class_itself.md` — `c` (Base module)
has 7 solved keys despite the attribute being on `Child`.

**Why it happens:** `get_class_member_impl` in class_field.rs:3538
calls `get_field_from_mro` which walks ALL ancestors to collect
candidate attributes, even if the attribute is found on the first
class. The code in attr.rs:651 does `for ancestor in mro.ancestors_no_object()`
without early exit.

**Real-world impact:** The class hierarchy keys
(`KeyClassSynthesizedFields` + `KeyClassMetadata` + `KeyClassMro` +
`KeyClassField`) collectively account for ~64% of all cross-module
demands — the largest cluster of work in a typical check.
`KeyClassSynthesizedFields` alone is ~24% of all demands (~36% of
`Answer` demands), and `KeyClassMro` is ~9% (~13% of `Answer`). Both
are consequences of the full MRO walk.

**Ideal behavior:** Check the class itself first. Only walk the MRO
if the attribute is not found on the class itself.

## Abstract class check on every instantiation

Every `Foo()` call triggers `KeyAbstractClassCheck` which enumerates
ALL abstract methods across ALL parent classes. For a class with no
abstract parents, this is pure overhead.

**Visible in:** `test_import_class_instantiated.md` — `KeyAbstractClassCheck`
appears. In `test_attribute_inherited.md` it cascades into `b -> c::KeyClassMetadata`.

**Why it happens:** `construct_class` in call.rs calls
`get_abstract_members_for_class` which recursively checks all parent
classes for unimplemented abstract methods, even when there are none.

**Ideal behavior:** Defer abstract class checking to error reporting.
Or: cache abstract status on the class metadata so it's a flag check
rather than a full hierarchy walk. If no parent in the MRO extends
`ABC` or has metaclass `ABCMeta`, skip the check entirely.

## Multiple inheritance check resolves unique parent fields

`check_consistent_multiple_inheritance` iterates each parent's fields
and calls `get_class_member` to resolve the field type for ALL of them.
But only fields that appear in MULTIPLE parents need type resolution
(line 3382 checks `len() > 1`). Fields unique to a single parent are
resolved then discarded.

**Visible in:** `test_multiple_inheritance_solves_unique_fields.md` —
`KeyClassField(B1, "p1")` and `KeyClassField(B2, "p2")` are demanded
cross-module but never compared (unique to one parent).

**Ideal behavior:** Collect field names from all parents first. Only
resolve `KeyClassField` for names that appear in multiple parents.
This is a two-pass approach: pass 1 collects names (cheap metadata),
pass 2 resolves types (expensive) only for shared names.

## Annotated return types DO break cascades (working correctly)

`test_annotated_return_breaks_cascade.md` shows that when
`get_config() -> int` has a return annotation, module `c` (containing
`Config` used in the body) has 0 solved keys. The demand tree shows
only `a -> b::KeyExport("get_config")`.

This demonstrates that pyrefly already implements the "annotation as
cascade breaker" pattern — callers trust the annotation without
inferring the function body.

## Transitive annotated exports break cascades

`test_transitive_import_annotated.md` shows that when `b` has
`value: int = 42`, module `c` has no keys solved — the annotation
`int` is resolved locally in `b` without cascading to `c`. `c` reaches
`Step::Load` only: the binder reads its file contents to discover the
module exists, but the export set is never materialized. Ideally `c`
would be `Nothing`.

## Unused imports' transitive deps are not checked

`test_unused_import_from_same_module.md` shows that `c` (Heavy's
module) has 0 solved keys when only `light()` is used from `b`.
The solver only resolves the `light` function and doesn't cascade
into `Heavy`'s module. `c` reaches `Step::Load` only — same gap as
above.
