# Jaxtyping error tests

These tests verify that the type checker correctly catches jaxtyping annotation errors.

## Mixed native and jaxtyping tensor syntax (test_mixed_syntax)

```scrut
$ $PYREFLY check "$JAXTYPING_TEST_ROOT/jaxtyping/negative_tests/test_mixed_syntax.py"
ERROR Cannot mix native tensor syntax (Tensor[N, M]) and jaxtyping syntax (Float[Tensor, "N M"]) in the same function [invalid-annotation]
  --> *test_mixed_syntax.py:15:5 (glob)
   |
15 | def mixed_syntax(
   |     ^^^^^^^^^^^^
   |
ERROR Returned type `Shaped[Tensor, "batch 3"]` is not assignable to declared return type `Tensor[3]` [bad-return]
  --> *test_mixed_syntax.py:18:12 (glob)
   |
17 | ) -> Tensor[3]:
   |      --------- declared return type
18 |     return x
   |            ^
   |
  Tensor rank mismatch: expected 1 dimensions, got 2 dimensions
[1]
```

## Matmul return type mismatch (test_matmul_mismatch)

```scrut
$ $PYREFLY check "$JAXTYPING_TEST_ROOT/jaxtyping/negative_tests/test_matmul_mismatch.py"
ERROR Returned type `Tensor[batch, 3, 5]` is not assignable to declared return type `Shaped[Tensor, "batch 3 99"]` [bad-return]
  --> *test_matmul_mismatch.py:22:12 (glob)
   |
20 | ) -> Shaped[Tensor, "batch 3 99"]:
   |      ---------------------------- declared return type
21 |     """Matmul produces batch×3×5, but return says batch×3×99."""
22 |     return torch.matmul(a, b)
   |            ^^^^^^^^^^^^^^^^^^
   |
  Size mismatch: expected 99, got 5
[1]
```

## Multiple variadic specifiers (test_multiple_variadics)

```scrut
$ $PYREFLY check "$JAXTYPING_TEST_ROOT/jaxtyping/negative_tests/test_multiple_variadics.py"
ERROR Tensor shape can have at most one variadic dimension [invalid-annotation]
  --> *test_multiple_variadics.py:16:23 (glob)
   |
16 |     x: Shaped[Tensor, "*batch ... 3"],
   |                       ^^^^^^^^^^^^^^
   |
[1]
```

## Direct shape mismatch on return (test_shape_mismatch)

```scrut
$ $PYREFLY check "$JAXTYPING_TEST_ROOT/jaxtyping/negative_tests/test_shape_mismatch.py"
ERROR Returned type `Shaped[Tensor, "batch 3"]` is not assignable to declared return type `Shaped[Tensor, "batch 4"]` [bad-return]
  --> *test_shape_mismatch.py:14:12 (glob)
   |
12 | def wrong_return_size(x: Shaped[Tensor, "batch 3"]) -> Shaped[Tensor, "batch 4"]:
   |                                                        ------------------------- declared return type
13 |     """Return type has size 4 but input has size 3 in last dim."""
14 |     return x
   |            ^
   |
  Size mismatch: expected 4, got 3
[1]
```

## Rank mismatch on return (test_rank_mismatch)

```scrut
$ $PYREFLY check "$JAXTYPING_TEST_ROOT/jaxtyping/negative_tests/test_rank_mismatch.py"
ERROR Returned type `Shaped[Tensor, "batch 3"]` is not assignable to declared return type `Shaped[Tensor, "batch 3 4"]` [bad-return]
  --> *test_rank_mismatch.py:14:12 (glob)
   |
12 | def wrong_rank(x: Shaped[Tensor, "batch 3"]) -> Shaped[Tensor, "batch 3 4"]:
   |                                                 --------------------------- declared return type
13 |     """Return type has rank 3 but input has rank 2."""
14 |     return x
   |            ^
   |
  Tensor rank mismatch: expected 3 dimensions, got 2 dimensions
[1]
```

## Malformed jaxtyping annotations (test_bad_annotation)

```scrut
$ $PYREFLY check "$JAXTYPING_TEST_ROOT/jaxtyping/negative_tests/test_bad_annotation.py"
ERROR jaxtyping annotations require exactly 2 arguments (array type and shape string), got 1 [invalid-annotation]
  --> *test_bad_annotation.py:12:21 (glob)
   |
12 | def too_few_args(x: Shaped[Tensor]) -> None:
   |                     ^^^^^^^^^^^^^^
   |
ERROR jaxtyping annotations require exactly 2 arguments (array type and shape string), got 3 [invalid-annotation]
  --> *test_bad_annotation.py:17:22 (glob)
   |
17 | def too_many_args(x: Shaped[Tensor, "3", "extra"]) -> None:  # noqa: F821
   |                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   |
ERROR Could not find name `extra` [unknown-name]
  --> *test_bad_annotation.py:17:43 (glob)
   |
17 | def too_many_args(x: Shaped[Tensor, "3", "extra"]) -> None:  # noqa: F821
   |                                           ^^^^^
   |
ERROR Second argument to jaxtyping annotation must be a string literal [invalid-annotation]
  --> *test_bad_annotation.py:22:45 (glob)
   |
22 | def non_string_second_arg(x: Shaped[Tensor, 42]) -> None:
   |                                             ^^
   |
[1]
```
