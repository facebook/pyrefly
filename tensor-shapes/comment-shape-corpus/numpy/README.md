# NumPy syntax demo

Three independent NumPy API excerpts expressed with ordinary NumPy annotations
and comment-only shape relationships. These local fixture modules are not an
installable `numpy-stubs` package and do not replace the repository's full
shape-aware NumPy stubs.

## Source and adaptations

Reference: [NumPy v2.2.3](https://github.com/numpy/numpy/tree/v2.2.3).
The upstream notice is retained in [LICENSE.numpy](LICENSE.numpy).

- [numeric.pyi](numeric.pyi) adapts
  [`numpy/_core/numeric.pyi`](https://github.com/numpy/numpy/blob/v2.2.3/numpy/_core/numeric.pyi):
  `outer`, `tensordot`, and `roll`.
- [twodim_base.pyi](twodim_base.pyi) adapts
  [`numpy/lib/_twodim_base_impl.pyi`](https://github.com/numpy/numpy/blob/v2.2.3/numpy/lib/_twodim_base_impl.pyi):
  `diag`, `tril`, and `triu`.
- [shape_base.pyi](shape_base.pyi) adapts
  [`numpy/lib/_shape_base_impl.pyi`](https://github.com/numpy/numpy/blob/v2.2.3/numpy/lib/_shape_base_impl.pyi):
  `expand_dims`, `tile`, and `kron`.

The upstream array-like protocols and dtype-promotion overloads are reduced to
NumPy-array cases. Types are spelled directly with `np.ndarray` and `np.dtype`;
shape information is added only by comments. The ordinary dtype type variable
is retained, so shape-preserving operations still preserve concrete dtypes.

## Selected cases

- **Numeric:** `outer` takes float64 vectors and an optional exactly shaped output
  buffer. NumPy also flattens higher-rank inputs; that form is omitted.
  `tensordot` takes float64 matrices and **requires an explicit `axes=1`**,
  contracting their shared dimension. NumPy's default `axes=2` and arbitrary
  axis lists are outside this excerpt. `roll` preserves the whole shape and
  dtype; invalid axes and incompatible shift/axis tuple lengths are not modeled.
- **2-D helpers:** `diag` constructs a square matrix from a vector or extracts
  the main diagonal of a square matrix (`k=0`). Rectangular extraction and offset
  diagonals are omitted. `tril` and `triu` preserve dtype and shape for arrays of
  rank at least two, including batch dimensions. NumPy's special 1-D triangular
  behavior is outside this excerpt.
- **Shape manipulation:** `expand_dims` inserts the leading singleton dimension
  with `axis=0`, using the existing shape DSL helper. Arbitrary axes are omitted.
  `tile` takes a matrix and a two-element repetition tuple, including zero
  repetitions. Scalar/1-D repetitions and rank promotion are omitted. `kron`
  multiplies the corresponding dimensions of two float64 matrices; mixed dtypes
  and inputs of other ranks are omitted.

These are deliberately narrow demos, not claims of full NumPy API coverage. The
negative fixtures distinguish unsupported excerpt cases from shape mismatches;
not every rejection indicates that NumPy itself would raise an exception.

## Syntax observations

- `roll` demonstrates one whole-shape variable and unchanged dtype typing.
- `diag` uses independent shape declarations on each overload.
- Matrix contractions connect the dimensions of separate arguments.
- Comments directly on tuple carriers use `# shaped: = payload` and keep dtype
  arguments readable. A whole-array comment can use `.0 = payload` instead.
- `tile` turns an ordinary `tuple[int, int]` argument into an `IntTuple` and
  multiplies symbolic dimensions without changing the visible parameter type.
- `expand_dims` demonstrates a DSL call inside the return carrier comment. Its
  `numpy._shapes.expand_dims_shape` import is an experimental helper supplied by
  `pyrefly-numpy-stubs`, **not a NumPy runtime API**. The import used only by
  comments has an external-linter `noqa: F401` marker. Other type checkers using
  stock NumPy will report this helper import as unresolved; `noqa` suppresses
  only the unused-import lint. A stable upstream helper/declaration mechanism
  is still a separate design question.

The [callers](examples) are static type-checking fixtures, not runtime tests.
Run from the Pyrefly root:

```sh
python3 tensor-shapes/comment-shape-corpus/run_pyrefly.py --buck --project numpy
```

Omit `--project` to check both NumPy and SciPy. No runner registration change is
needed: documented project directories are discovered automatically.
