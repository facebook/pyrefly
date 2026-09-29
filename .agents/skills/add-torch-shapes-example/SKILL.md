---
name: add-torch-shapes-example
description: Use when adding a new PyTorch model to Pyrefly's shape-tracking example corpus under tensor-shapes/pyrefly-torch-stubs/examples — i.e. importing a model as a tested, corpus-quality reference port. This is maintainer-facing fbsource work. For porting your own model elsewhere, use the porting skill directly; for fixing a wrong/missing shape rule, use modify-shaped-array-dsl.
---

You are importing a PyTorch model into Pyrefly's example corpus at
`tensor-shapes/pyrefly-torch-stubs/examples/`. This is the **contribution case** the porting
skill describes: these ports are tested reference material that others read to
learn the patterns, so produce its fuller deliverable — paste every artifact
(audit table, per-local `reveal_type` dumps, typed-interface receipts, exhaustive
`assert_type` coverage, completion report) in full, not just the annotated model.

**Why these ports matter.** They demonstrate what happens when you write a real
PyTorch model with tensor shape types. Record the upstream repository and
revision, exact files or dependency closure, concrete configuration, entry
points, and train/eval/cache/export modes included. "Complete" means complete
inside that declared boundary; list any omitted wrapper or mode rather than
calling a representative core the full upstream model.

**Start from evidence, not a blank page.** Before editing, skim two or three
existing examples with the closest architecture and mine the upstream source
for shape comments, docstrings, reshape/einsum equations, runtime assertions,
and tests. Treat that evidence as a hypothesis to verify with Pyrefly, not text
to copy. The existing ports demonstrate that substantial real models normally
reach useful shape coverage after a few checker-guided iterations.

**Improving the stubs is the point, not a side quest.** First distinguish a
true stub gap from an unavailable overlay symbol, `Any`, a declared gradual
return, third-party code, or unrepresentable dynamic construction. Fix genuine
general stub gaps in the corpus case rather than hiding them in the model. A
corpus port may retain a narrow precise cast, typed interface, or gradual
boundary for heterogeneous containers, dynamic factories, mutable caches, or
untyped external backends. Preserve every known public dimension and document
the boundary. Propose, but do not perform, a runtime rewrite unless the user
separately requests it.

## 1. Run the port

Do the actual porting by reading and following the `add-shape-types-to-torch-model`
skill's `SKILL.md` (in `tensor-shapes/skills/add-shape-types-to-torch-model/`) end to
end — its gated workflow (pre-flight gates → per-module loop → verification) is the
algorithm.

The general skill has two setup choices; for corpus work both are already
resolved, so do not stop to ask: use the Buck check below, and treat stub
improvements as in scope. Produce all of the corpus artifacts it requests.

## 2. Place the file

Write the port at `tensor-shapes/pyrefly-torch-stubs/examples/<model>.py`.
Every class, function, method, entry point, configuration, and mode inside the
declared upstream boundary belongs in the port. Do not silently shrink that
boundary when a difficult construct appears.

## 3. Verify (the fbsource commands)

The porting skill's verification phase tells you to run `verify_port.sh` and the
actual Pyrefly check. Run these commands from the `fbcode/pyrefly` checkout root.
First ensure the shared tensor-shapes virtual environment exists; add
`--fwdproxy` when the host needs it:

```bash
python3 tensor-shapes/bootstrap_venv.py
buck build fbcode//pyrefly/tensor-shapes:torch-stubs-search-path
SEARCH_ROOT="$(buck targets --show-output fbcode//pyrefly/tensor-shapes:torch-stubs-search-path | awk '{print $2}')"
VENV="${TENSOR_SHAPES_VENV:-$HOME/.tensor-shapes-venv}"
SITE="$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
buck run fbcode//pyrefly:pyrefly -- check --config /dev/null \
  --python-version 3.13 --search-path "$SEARCH_ROOT" \
  --site-package-path "$SITE" \
  tensor-shapes/pyrefly-torch-stubs/examples/<model>.py
```

If the model imports einops, also pass
`--search-path tensor-shapes/pyrefly-einops-stubs`; otherwise an einops call can
silently appear to preserve its input shape. The result must be `0 errors`, with
no leftover `reveal_type`.

Then run the corpus test target so the new example is covered by CI and checked
with the real Torch fallback modules:

```bash
python3 tensor-shapes/pyrefly-torch-stubs/run_pyrefly.py --buck --suite torch-examples
```

## If you hit a wrong or missing shape

When shape precision is missing, first distinguish among an unavailable symbol
in the partial overlay, `Any`, a declared gradual `Tensor`, a third-party
boundary, and a true stub-signature gap. Add or refine a general stub when that
is the right fix. A corpus port may retain a documented boundary for
unrepresentable dynamic construction; it should not hide an easily fixable stub
gap.

A *wrong* shape (Pyrefly computes a concrete shape that's incorrect) or a missing
shape that can't be expressed by a stub signature alone is a shape-DSL change: see
the `modify-shaped-array-dsl` skill. That skill insists on unit-testing the DSL
logic, not just relying on this example to exercise it. Don't reach for the DSL
for shapes a stub signature could express.
