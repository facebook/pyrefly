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
PyTorch model with tensor shape types, proving real-world utility. If you exclude
features or simplify the model, you prove nothing — the hard parts are exactly
where the value needs to be demonstrated. So port the model faithfully and in
full (see step 2).

**Improving the stubs is the point, not a side quest.** In the general porting
skill, changing stubs is optional and a missing shape can just be documented as a
gap. Here it is the opposite: a corpus example exists to *exercise and harden the
stubs*. When an op falls back to bare `Tensor`, treat it as a stub deficiency to
fix, not a gap to record — refine the stub signature (or add a shape DSL rule) so
the shape is recovered, then make that the general truth about the op, not a
special case for this model. A port that leaves easily-fixable bare `Tensor`s
behind is not done. Only genuinely data-dependent shapes (e.g. data-dependent
token counts) should remain bare, with a comment saying why.

## 1. Run the port

Do the actual porting by reading and following the `add-shape-types-to-torch-model`
skill's `SKILL.md` (in `tensor-shapes/skills/add-shape-types-to-torch-model/`) end to
end — its gated workflow (pre-flight gates → per-module loop → verification) is the
algorithm.

That skill opens with two questions for the user; for corpus work you already have
the answers, so don't stop to ask: the check command is the buck invocation in
step 3 below, and stub changes are in scope (corpus ports should track shapes as
fully as possible, so refine stub signatures when that recovers real shapes).
Produce **all** of its output artifacts; for the corpus they are required.

## 2. Place the file

Write the port at `tensor-shapes/pyrefly-torch-stubs/examples/<model>.py`. Every class,
function, and method from the original belongs in the port — the corpus values
completeness.

## 3. Verify (the fbsource commands)

The porting skill's verification phase tells you to run `verify_port.sh` and then "the
actual Pyrefly check." In fbsource that check is a buck invocation against the
shape-aware stubs:

```bash
buck build fbcode//pyrefly/tensor-shapes:torch-stubs-search-path
buck run fbcode//pyrefly:pyrefly -- check --config /dev/null --python-version 3.13 --tensor-shapes true --search-path "$(buck targets --show-output fbcode//pyrefly/tensor-shapes:torch-stubs-search-path | awk '{print $2}')" tensor-shapes/pyrefly-torch-stubs/examples/<model>.py
```

The result must be `0 errors`, with no leftover `reveal_type`.

Then run the corpus test target so the new example is covered by CI:

```bash
buck test fbcode//pyrefly/tensor-shapes/pyrefly-torch-stubs/examples:torch_examples_test
```

## If you hit a wrong or missing shape

A *missing* shape (op falls back to bare `Tensor`) is usually a loose or absent
stub signature — fix it in the stubs so the shape is recovered (see "Improving
the stubs is the point" above), rather than documenting it as a gap.

A *wrong* shape (Pyrefly computes a concrete shape that's incorrect) or a missing
shape that can't be expressed by a stub signature alone is a shape-DSL change: see
the `modify-shaped-array-dsl` skill. That skill insists on unit-testing the DSL
logic, not just relying on this example to exercise it. Don't reach for the DSL
for shapes a stub signature could express.
