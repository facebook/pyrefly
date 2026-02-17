# Dim type checking error tests

These tests verify that the type checker correctly catches Dim type errors.

## Dim type variable unification (test_symint_unification)

```scrut
$ $PYREFLY check "$TENSOR_TEST_ROOT/negative_tests/test_symint_unification.py"
 INFO revealed type: Dim[(A * B)] [reveal-type]
  --> *test_symint_unification.py:23:16 (glob)
   |
23 |     reveal_type(expr)  # Should be Dim[A * B]
   |                ------
   |
 INFO revealed type: Dim[(A * B)] [reveal-type]
  --> *test_symint_unification.py:25:16 (glob)
   |
25 |     reveal_type(result)  # Should be Dim[A * B] if X is unified
   |                --------
   |
ERROR Argument `Dim[((A * B) // 2)]` is not assignable to parameter `x` with type `Dim[(@_ // 2)]` in function `half_symint` [bad-argument-type]
  --> *test_symint_unification.py:40:17 (glob)
   |
40 |     half_symint(expr)
   |                 ^^^^
   |
  Type variable cannot be inferred from a nested position
 INFO revealed type: Dim[N] [reveal-type]
  --> *test_symint_unification.py:53:16 (glob)
   |
53 |     reveal_type(result)  # Should be Dim[N]
   |                --------
   |
 INFO revealed type: Dim[(A + A)] [reveal-type]
  --> *test_symint_unification.py:64:16 (glob)
   |
64 |     reveal_type(result)  # Should be Dim[A + A]
   |                --------
   |
[1]
```

## Dim with bare type annotation (test_symint_any)

```scrut
$ $PYREFLY check "$TENSOR_TEST_ROOT/negative_tests/test_symint_any.py"
 INFO revealed type: Dim [reveal-type]
  --> *test_symint_any.py:18:12 (glob)
   |
18 | reveal_type(symint_implicit_any)  # Dim
   |            ---------------------
   |
 INFO revealed type: Dim[Any] [reveal-type]
  --> *test_symint_any.py:20:12 (glob)
   |
20 | reveal_type(symint_explicit_any)  # Dim[Any]
   |            ---------------------
   |
 INFO revealed type: Dim [reveal-type]
  --> *test_symint_any.py:32:16 (glob)
   |
32 |     reveal_type(s_n)  # Dim
   |                -----
   |
 INFO revealed type: Dim [reveal-type]
  --> *test_symint_any.py:34:16 (glob)
   |
34 |     reveal_type(s_implicit_any)  # Dim
   |                ----------------
   |
 INFO revealed type: Dim[Any] [reveal-type]
  --> *test_symint_any.py:36:16 (glob)
   |
36 |     reveal_type(s_explicit_any)  # Dim[Any]
   |                ----------------
   |
[0]
```

## Tensor subtyping rules (test_tensor_subtyping)

```scrut
$ $PYREFLY check "$TENSOR_TEST_ROOT/negative_tests/test_tensor_subtyping.py"
ERROR Returned type `Tensor[2, 3]` is not assignable to declared return type `Tensor[4, 3]` [bad-return]
  --> *test_tensor_subtyping.py:33:12 (glob)
   |
33 |     return x  # ERROR: Tensor[2, 3] not assignable to Tensor[4, 3]
   |            ^
   |
  Size mismatch: expected 4, got 2
ERROR Returned type `Tensor[2, 3]` is not assignable to declared return type `Tensor[2, 5]` [bad-return]
  --> *test_tensor_subtyping.py:38:12 (glob)
   |
38 |     return x  # ERROR: Tensor[2, 3] not assignable to Tensor[2, 5]
   |            ^
   |
  Size mismatch: expected 5, got 3
ERROR Returned type `Tensor[2, 3]` is not assignable to declared return type `Tensor[2, 3, 4]` [bad-return]
  --> *test_tensor_subtyping.py:43:12 (glob)
   |
43 |     return x  # ERROR: Tensor[2, 3] not assignable to Tensor[2, 3, 4]
   |            ^
   |
  Tensor rank mismatch: expected 3 dimensions, got 2 dimensions
ERROR Returned type `Tensor[N, M]` is not assignable to declared return type `Tensor[M, N]` [bad-return]
  --> *test_tensor_subtyping.py:68:12 (glob)
   |
68 |     return x  # ERROR: Tensor[N, M] not assignable to Tensor[M, N]
   |            ^
   |
ERROR Returned type `Tensor[N, 3]` is not assignable to declared return type `Tensor[N, 5]` [bad-return]
  --> *test_tensor_subtyping.py:78:12 (glob)
   |
78 |     return x  # ERROR: Tensor[N, 3] not assignable to Tensor[N, 5]
   |            ^
   |
  Size mismatch: expected 5, got 3
ERROR Returned type `Tensor[N, M]` is not assignable to declared return type `Tensor[(N + M)]` [bad-return]
  --> *test_tensor_subtyping.py:88:12 (glob)
   |
88 |     return x  # ERROR: Tensor[N, M] not assignable to Tensor[N + M]
   |            ^
   |
  Tensor rank mismatch: expected 1 dimensions, got 2 dimensions
ERROR Returned type `Tensor[(N + 1), 3]` is not assignable to declared return type `Tensor[(N + 2), 3]` [bad-return]
  --> *test_tensor_subtyping.py:98:12 (glob)
   |
98 |     return x  # ERROR: N + 1 not equal to N + 2
   |            ^
   |
  Size mismatch: expected (2 + N), got (1 + N)
ERROR Returned type `Tensor[(N + M), 3]` is not assignable to declared return type `Tensor[(N * M), 3]` [bad-return]
   --> *test_tensor_subtyping.py:108:12 (glob)
    |
108 |     return x  # ERROR: N + M not equal to N * M
    |            ^
    |
  Size mismatch: expected (M * N), got (M + N)
ERROR Argument `Tensor[2, 3]` is not assignable to parameter `x` with type `Tensor[4, 3]` in function `tensor_generic_identity` [bad-argument-type]
   --> *test_tensor_subtyping.py:123:36 (glob)
    |
123 |     return tensor_generic_identity(x)  # ERROR
    |                                    ^
    |
  Size mismatch: expected 4, got 2
[1]
```

## Tensor indexing operations (test_tensor_indexing)

```scrut
$ $PYREFLY check "$TENSOR_TEST_ROOT/negative_tests/test_tensor_indexing.py"
ERROR Returned type `Tensor[20]` is not assignable to declared return type `Tensor[10, 20]` [bad-return]
   --> *test_tensor_indexing.py:256:12 (glob)
    |
256 |     return x[0]  # ERROR: Tensor[20] not assignable to Tensor[10, 20]
    |            ^^^^
    |
  Tensor rank mismatch: expected 2 dimensions, got 1 dimensions
ERROR Returned type `Tensor[5, 20]` is not assignable to declared return type `Tensor[3, 20]` [bad-return]
   --> *test_tensor_indexing.py:261:12 (glob)
    |
261 |     return x[:5]  # ERROR: Tensor[5, 20] not assignable to Tensor[3, 20]
    |            ^^^^^
    |
  Size mismatch: expected 3, got 5
[1]
```

## Tensor arithmetic operations (test_tensor_arithmetic)

```scrut
$ $PYREFLY check "$TENSOR_TEST_ROOT/negative_tests/test_tensor_arithmetic.py"
ERROR Returned type `Tensor[2, 3]` is not assignable to declared return type `Tensor[4, 5]` [bad-return]
   --> *test_tensor_arithmetic.py:145:12 (glob)
    |
145 |     return x + 1.0  # ERROR: Tensor[2, 3] not assignable to Tensor[4, 5]
    |            ^^^^^^^
    |
  Size mismatch: expected 4, got 2
ERROR Returned type `Tensor[2, 3]` is not assignable to declared return type `Tensor[2, 3, 4]` [bad-return]
   --> *test_tensor_arithmetic.py:150:12 (glob)
    |
150 |     return x * 2.0  # ERROR: Tensor[2, 3] not assignable to Tensor[2, 3, 4]
    |            ^^^^^^^
    |
  Tensor rank mismatch: expected 3 dimensions, got 2 dimensions
ERROR Returned type `Tensor[2, 3]` is not assignable to declared return type `Tensor[1, 3]` [bad-return]
   --> *test_tensor_arithmetic.py:160:12 (glob)
    |
160 |     return x + y  # ERROR: Tensor[2, 3] not assignable to Tensor[1, 3]
    |            ^^^^^
    |
  Size mismatch: expected 1, got 2
ERROR Cannot broadcast tensor shapes: Invalid dimension value 0: Cannot broadcast dimension 3 with dimension 5 at position 1 [unsupported-operation]
   --> *test_tensor_arithmetic.py:165:12 (glob)
    |
165 |     return x + y  # ERROR: Cannot broadcast shapes [2, 3] and [4, 5]
    |            ^^^^^
    |
ERROR Cannot broadcast tensor shapes: Invalid dimension value 0: Cannot broadcast dimension N with dimension M at position 0 [unsupported-operation]
   --> *test_tensor_arithmetic.py:207:12 (glob)
    |
207 |     return x + y  # ERROR: Cannot broadcast dimension N with dimension M
    |            ^^^^^
    |
ERROR Cannot broadcast tensor shapes: Invalid dimension value 0: Cannot broadcast concrete dims with variadic shape: alignment is ambiguous [unsupported-operation]
   --> *test_tensor_arithmetic.py:244:12 (glob)
    |
244 |     return x + y  # ERROR: Cannot broadcast concrete dims with variadic shape
    |            ^^^^^
    |
ERROR Cannot broadcast tensor shapes: Invalid dimension value 0: Cannot broadcast variadic shapes: incompatible middles *Ts vs *Us [unsupported-operation]
   --> *test_tensor_arithmetic.py:278:12 (glob)
    |
278 |     return x + y  # ERROR: incompatible middles
    |            ^^^^^
    |
[1]
```

## Tensor generic expression substitution (test_tensor_generic_exprs)

```scrut
$ $PYREFLY check "$TENSOR_TEST_ROOT/negative_tests/test_tensor_generic_exprs.py"
ERROR Returned type `Tensor[(2 + 3)]` is not assignable to declared return type `Tensor[6]` [bad-return]
   --> *test_tensor_generic_exprs.py:122:12 (glob)
    |
122 |     return sum_dims(x)  # ERROR
    |            ^^^^^^^^^^^
    |
  Size mismatch: expected 6, got 5
ERROR Returned type `Tensor[(2 * 3)]` is not assignable to declared return type `Tensor[5]` [bad-return]
   --> *test_tensor_generic_exprs.py:127:12 (glob)
    |
127 |     return product_dims(x)  # ERROR
    |            ^^^^^^^^^^^^^^^
    |
  Size mismatch: expected 5, got 6
ERROR Returned type `Tensor[(4 * 2), 5]` is not assignable to declared return type `Tensor[4, 5]` [bad-return]
   --> *test_tensor_generic_exprs.py:132:12 (glob)
    |
132 |     return double_first(x)  # ERROR
    |            ^^^^^^^^^^^^^^^
    |
  Size mismatch: expected 4, got 8
[1]
```

## Tensor expression equivalence (test_tensor_expr_equiv)

```scrut
$ $PYREFLY check "$TENSOR_TEST_ROOT/negative_tests/test_tensor_expr_equiv.py"
ERROR Returned type `Tensor[(N + M)]` is not assignable to declared return type `Tensor[(N * M)]` [bad-return]
  --> *test_tensor_expr_equiv.py:85:12 (glob)
   |
85 |     return x
   |            ^
   |
  Size mismatch: expected (M * N), got (M + N)
ERROR Returned type `Tensor[(N + 1)]` is not assignable to declared return type `Tensor[(N + 2)]` [bad-return]
  --> *test_tensor_expr_equiv.py:90:12 (glob)
   |
90 |     return x
   |            ^
   |
  Size mismatch: expected (2 + N), got (1 + N)
ERROR Returned type `Tensor[5, 4]` is not assignable to declared return type `Tensor[6, 4]` [bad-return]
  --> *test_tensor_expr_equiv.py:95:12 (glob)
   |
95 |     return x
   |            ^
   |
  Size mismatch: expected 6, got 5
[1]
```

## Tensor variadic shape patterns (test_tensor_variadic)

```scrut
$ $PYREFLY check "$TENSOR_TEST_ROOT/negative_tests/test_tensor_variadic.py"
ERROR Argument `Tensor[10, 20]` is not assignable to parameter `x` with type `Tensor[10, 30]` in function `variadic_identity` [bad-argument-type]
  --> *test_tensor_variadic.py:97:30 (glob)
   |
97 |     return variadic_identity(x)
   |                              ^
   |
  Size mismatch: expected 30, got 20
ERROR Argument `Tensor[1, 2, 3, 4]` is not assignable to parameter `x` with type `Tensor[1, 2, 3]` in function `split_first_rest` [bad-argument-type]
   --> *test_tensor_variadic.py:104:29 (glob)
    |
104 |     return split_first_rest(x)
    |                             ^
    |
[1]
```
