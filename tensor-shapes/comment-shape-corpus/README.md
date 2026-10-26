# Comment-only shape annotation corpus

Small, independent API slices for evaluating `# shapes:` / `# shaped:` syntax.
These are examples to read, edit, and share, not installable replacement stubs.
Each project has its own directory, provenance, scope notes, and checked callers.

Start with [the SciPy demo](scipy/README.md):

- [FFT helpers](scipy/fft_helper.pyi): concise scalar dimensions, arithmetic,
  whole-shape variables, and dtype preservation.
- [Cholesky routines](scipy/cholesky.pyi): square matrices, batch dimensions,
  overloads, and selectors inside tuple parameters and returns.
- [Spatial distances](scipy/distance.pyi): relationships between multiple arrays,
  condensed-distance lengths, and explicit whole-type replacement.

[The NumPy demo](numpy/README.md) adds three more independent module slices:

- [Numeric operations](numpy/numeric.pyi): outer products, contractions, and
  whole-shape preservation.
- [2-D helpers](numpy/twodim_base.pyi): rank-changing overloads and batched
  triangular operations with dtype preservation.
- [Shape manipulation](numpy/shape_base.pyi): repetition tuples, symbolic
  multiplication, and an existing DSL-backed return transformation.

## Run

Use a Pyrefly build containing shape-comment support. Only NumPy's canonical
stubs and the repository's `pyrefly-numpy-stubs` / `shape_extensions` are needed.
**SciPy, scipy-stubs, optype, and numpy-typing-compat are not dependencies.**

From the Pyrefly repository root, with the usual NumPy virtualenv:

```sh
python3 tensor-shapes/comment-shape-corpus/run_pyrefly.py --buck --project scipy
```

Pass `--python /path/to/venv/bin/python` for a different NumPy environment, or
`--pyrefly /path/to/pyrefly` for an already-built experimental binary. A standalone
NumPy source/stub tree can be used without installing any numerical packages:

```sh
python3 tensor-shapes/comment-shape-corpus/run_pyrefly.py \
  --pyrefly /path/to/pyrefly --numpy-root /path/containing/numpy --project scipy
```

In fbsource, the runner is also a Buck target:

```sh
buck2 run fbcode//pyrefly/tensor-shapes/comment-shape-corpus:run_pyrefly -- \
  --buck --numpy-root /path/containing/numpy --project scipy
```

The runner checks every `.pyi` module before every `examples/*.py` caller and
matches expected rejections against `# E:` comments. Missing modules or callers
are errors, not empty successful suites. The examples are **type-checking
fixtures**, not programs to execute against SciPy.

## Add another project

Create an immediate project directory containing:

1. `README.md` with pinned upstream links, adaptation notes, and supported scope;
2. one or more self-contained `.pyi` module slices;
3. `examples/*.py` with inferred-shape assertions and representative rejections;
4. upstream license notices where applicable.

The runner discovers documented project directories automatically, so no central
project list needs updating. Avoid importing the rest of a project merely to make
its demo compile. Keep ordinary annotations readable to other checkers and put
shape relationships in comments.
