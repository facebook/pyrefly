# Laziness Opportunities

Observations from the laziness test demand trees that represent
opportunities to reduce unnecessary work.

## Impact ranking (from `--report-demand-tree` on 5 real-world files)

1. **LookupExport during binding** (32-84% of demands) — `is_special_export`
   alone is 32-84%. 96.5% of dependency modules exist only for this.
2. **MRO walk without early exit** (10-40% of demands) — drives
   `KeyClassSynthesizedFields` (31%), `KeyClassMro` (10%), and part of
   `KeyClassMetadata` (19%). Worst for files with deep class hierarchies.
3. **Full function signature on import** — resolves 11 keys per imported
   function regardless of usage.
4. **Class metadata for annotation-only usage** — `is_typed_dict()` check
   on every class-as-annotation.
5. **Eagerly resolved builtins** — 150 KeyExport + cascading demands per
   module. Fixed cost but adds up.
6. **Multiple inheritance check resolves unique fields** — low volume but
   real waste for wide hierarchies.

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

**Visible in:** 6 tests all show `c: Exports` with no demand tree
entries pointing to C:
- `test_bare_import_forces_exports` — `import c` triggers `module_exists(c)`
- `test_import_star_forces_exports` — `from c import *` triggers `get_wildcard(c)`
- `test_export_exists_forces_exports` — `from c import Foo` triggers `export_exists(c, "Foo")`
- `test_deprecated_forces_exports` — `from c import old_func` triggers `get_deprecated(c, "old_func")`
- `test_is_final_forces_exports` — `X = 2` (reassigning import) triggers `is_final(c, "X")`
- `test_special_export_forces_exports` — `T = MyTypeVar("T")` triggers `is_special_export(c, "MyTypeVar")`

Also visible in `test_unused_import_from_same_module` and
`test_transitive_import_annotated` where `c: Exports` appears.

**Real-world impact:** `--report-demand-tree` across 5 real-world files
shows `Exports(is_special_export)` accounts for 32-84% of ALL cross-module
demands. In one file with ~16,000 dependency modules, a single imported
module is checked 5,334x for special export status during binding. Across
all dependency modules, 96.5% are at Exports only — computed solely
because of LookupExport binding calls.

**Ideal behavior:** Module C should not be computed at all (`c: Nothing`)
when A doesn't need it. B's bindings should be constructible without
demanding exports from B's own imports.

**Root cause:** 18 call sites in `bindings.rs`, `stmt.rs`, and `scope.rs`
use `self.lookup.*` which goes through `TransactionHandle::with_exports`
→ `lookup_export` → `demand(Step::Exports)`.

**Deferral strategies by call site:**
- `module_exists` / `export_exists` — create optimistic `Binding::Import`,
  validate at solve time (solver already does cascading check:
  export → submodule → `__getattr__` → error)
- `is_final` — move to solve time (emit error when solving the assignment)
- `get_deprecated` — move to solve time (emit warning when solving the import)
- `is_special_export` — hardest to defer; controls which `Binding` variant
  is created (e.g., `Binding::TypeVar` vs `Binding::NameAssign`). Could use
  syntactic name matching for the common case, with solve-time validation
  for re-exports
- `get_wildcard` — unavoidable at bind time (binder needs the set of names
  to create binding table entries). But only affects `from X import *`
- `module_exists` for builtins — fixed cost, unavoidable

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

**Real-world impact:** `--report-demand-tree` on a file with deep class
hierarchies (~16 interface classes) shows 90,161 `KeyClassSynthesizedFields`
demands (30.8% of all demands) and 28,497 `KeyClassMro` demands (9.7%) —
both consequences of the full MRO walk.

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

## Transitive annotated exports break cascades (partially working)

`test_transitive_import_annotated.md` shows that when `b` has
`value: int = 42`, module `c` has no keys solved — the annotation
`int` is resolved locally in `b` without cascading to `c`. However,
`c` is still computed to Exports due to LookupExport calls during
`b`'s binding (see "LookupExport during binding forces transitive
exports" above). Ideally `c` would be `Nothing`.

## Unused imports' transitive deps are not checked (partially working)

`test_unused_import_from_same_module.md` shows that `c` (Heavy's
module) has 0 solved keys when only `light()` is used from `b`.
The solver only resolves the `light` function and doesn't cascade
into `Heavy`'s module. However, `c` is still computed to Exports
due to LookupExport calls during `b`'s binding (see "LookupExport
during binding forces transitive exports" above). Ideally `c` would
be `Nothing`.
