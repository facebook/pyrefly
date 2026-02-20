# Tensor type checking error tests

These tests verify that the type checker correctly catches shape mismatches and other errors.

## Generic tensor substitution with wrong shape (test_compare_generic_tensor)

```scrut
$ $PYREFLY check "$TENSOR_TEST_ROOT/negative_tests/test_compare_generic_tensor.py"
 INFO revealed type: int [reveal-type]
  --> *test_compare_generic_tensor.py:20:12 (glob)
   |
20 | reveal_type(result1)  # Returns: int ✅
   |            ---------
   |
 INFO revealed type: Tensor[2, 3] [reveal-type]
  --> *test_compare_generic_tensor.py:32:12 (glob)
   |
32 | reveal_type(result2)  # Returns: Tensor[N, 3] or Tensor[2, 3] ??
   |            ---------
   |
ERROR `Tensor[2, 3]` is not assignable to `Tensor[100, 3]` [bad-assignment]
  --> *test_compare_generic_tensor.py:36:36 (glob)
   |
36 | wrong_assignment: Tensor[100, 3] = result2  # Should ERROR if result2 is Tensor[2, 3]
   |                                    ^^^^^^^
   |
  Size mismatch: expected 100, got 2
[1]
```

## Symbolic dimension binding with wrong expected type (test_check_symbolic_binding)

```scrut
$ $PYREFLY check "$TENSOR_TEST_ROOT/negative_tests/test_check_symbolic_binding.py"
 INFO revealed type: Tensor[2, 3] [reveal-type]
  --> *test_check_symbolic_binding.py:26:16 (glob)
   |
26 |     reveal_type(result)
   |                --------
   |
 INFO revealed type: Tensor[2, 3] [reveal-type]
  --> *test_check_symbolic_binding.py:34:16 (glob)
   |
34 |     reveal_type(result)
   |                --------
   |
ERROR Returned type `Tensor[2, 3]` is not assignable to declared return type `Tensor[4, 3]` [bad-return]
  --> *test_check_symbolic_binding.py:35:12 (glob)
   |
35 |     return result  # Should ERROR: Tensor[2, 3] not assignable to Tensor[4, 3]
   |            ^^^^^^
   |
  Size mismatch: expected 4, got 2
 INFO revealed type: Dim[(M * N)] [reveal-type]
  --> *test_check_symbolic_binding.py:40:16 (glob)
   |
40 |     reveal_type(s)
   |                ---
   |
ERROR Returned type `Dim[(M * N)]` is not assignable to declared return type `Dim[(M + N)]` [bad-return]
  --> *test_check_symbolic_binding.py:41:12 (glob)
   |
41 |     return s
   |            ^
   |
  Size mismatch: expected (M + N), got (M * N)
 INFO revealed type: Tensor[(M * N)] [reveal-type]
  --> *test_check_symbolic_binding.py:46:16 (glob)
   |
46 |     reveal_type(v)
   |                ---
   |
ERROR Returned type `Tensor[(M * N)]` is not assignable to declared return type `Tensor[(M + N)]` [bad-return]
  --> *test_check_symbolic_binding.py:47:12 (glob)
   |
47 |     return v
   |            ^
   |
  Size mismatch: expected (M + N), got (M * N)
 INFO revealed type: Dim[(M * N)] [reveal-type]
  --> *test_check_symbolic_binding.py:52:16 (glob)
   |
52 |     reveal_type(s)
   |                ---
   |
ERROR Returned type `Dim[(M * N)]` is not assignable to declared return type `Dim[K]` [bad-return]
  --> *test_check_symbolic_binding.py:53:12 (glob)
   |
53 |     return s
   |            ^
   |
  Size mismatch: expected K, got (M * N)
 INFO revealed type: Tensor[(M * N)] [reveal-type]
  --> *test_check_symbolic_binding.py:58:16 (glob)
   |
58 |     reveal_type(v)
   |                ---
   |
ERROR Returned type `Tensor[(M * N)]` is not assignable to declared return type `Tensor[K]` [bad-return]
  --> *test_check_symbolic_binding.py:59:12 (glob)
   |
59 |     return v
   |            ^
   |
  Size mismatch: expected K, got (M * N)
 INFO revealed type: Dim[@_] [reveal-type]
  --> *test_check_symbolic_binding.py:64:16 (glob)
   |
64 |     reveal_type(n)
   |                ---
   |
 INFO revealed type: Tensor[@_] [reveal-type]
  --> *test_check_symbolic_binding.py:71:16 (glob)
   |
71 |     reveal_type(t)
   |                ---
   |
[1]
```

## Literal shape mismatch (test_literal_shape_check)

```scrut
$ $PYREFLY check "$TENSOR_TEST_ROOT/negative_tests/test_literal_shape_check.py"
 INFO revealed type: Tensor[2, 3] [reveal-type]
  --> *test_literal_shape_check.py:19:16 (glob)
   |
19 |     reveal_type(x)
   |                ---
   |
ERROR Returned type `Tensor[2, 3]` is not assignable to declared return type `Tensor[4, 3]` [bad-return]
  --> *test_literal_shape_check.py:22:12 (glob)
   |
22 |     return x  # Tensor[2, 3] not assignable to Tensor[4, 3]
   |            ^
   |
  Size mismatch: expected 4, got 2
[1]
```

## TypeVar substitution with multiple mismatches (test_typevar_substitution_bug)

```scrut
$ $PYREFLY check "$TENSOR_TEST_ROOT/negative_tests/test_typevar_substitution_bug.py"
 INFO revealed type: Tensor[2, 3] [reveal-type]
  --> *test_typevar_substitution_bug.py:25:16 (glob)
   |
25 |     reveal_type(x_concrete)  # Should be Tensor[2, 3]
   |                ------------
   |
 INFO revealed type: Tensor[2, 3] [reveal-type]
  --> *test_typevar_substitution_bug.py:29:16 (glob)
   |
29 |       reveal_type(
   |  ________________-
30 | |         result
31 | |     )  # Should be Tensor[2, 3] (N substituted with 2), but is Tensor[N, 3]
   | |_____-
   |
ERROR `Tensor[2, 3]` is not assignable to `Tensor[4, 3]` [bad-assignment]
  --> *test_typevar_substitution_bug.py:35:27 (glob)
   |
35 |     case2: Tensor[4, 3] = result  # Should ERROR but doesn't (N=4)
   |                           ^^^^^^
   |
  Size mismatch: expected 4, got 2
ERROR `Tensor[2, 3]` is not assignable to `Tensor[100, 3]` [bad-assignment]
  --> *test_typevar_substitution_bug.py:36:29 (glob)
   |
36 |     case3: Tensor[100, 3] = result  # Should ERROR but probably doesn't (N=100)
   |                             ^^^^^^
   |
  Size mismatch: expected 100, got 2
[1]
```

## Flatten with wrong expected shape (test_concat_flatten_types)

```scrut
$ $PYREFLY check "$TENSOR_TEST_ROOT/negative_tests/test_concat_flatten_types.py"
 INFO revealed type: Tensor[N, 3] [reveal-type]
  --> *test_concat_flatten_types.py:18:16 (glob)
   |
18 |     reveal_type(x)
   |                ---
   |
 INFO revealed type: Tensor[M, 3] [reveal-type]
  --> *test_concat_flatten_types.py:19:16 (glob)
   |
19 |     reveal_type(y)
   |                ---
   |
 INFO revealed type: Tensor[(M + N), 3] [reveal-type]
  --> *test_concat_flatten_types.py:21:16 (glob)
   |
21 |     reveal_type(z)
   |                ---
   |
 INFO revealed type: Tensor[B, N, M] [reveal-type]
  --> *test_concat_flatten_types.py:27:16 (glob)
   |
27 |     reveal_type(x)
   |                ---
   |
 INFO revealed type: Tensor[(5 + 2), 3] [reveal-type]
  --> *test_concat_flatten_types.py:36:16 (glob)
   |
36 |     reveal_type(z)  # Expected: Tensor[7, 3], but might be Tensor[N + M, 3]?
   |                ---
   |
ERROR Returned type `Tensor[(5 + 2), 3]` is not assignable to declared return type `Tensor[100, 3]` [bad-return]
  --> *test_concat_flatten_types.py:39:12 (glob)
   |
39 |     return z  # Should ERROR if z is Tensor[7, 3]
   |            ^
   |
  Size mismatch: expected 100, got 7
 INFO revealed type: Tensor[((2 * 4) * 3)] [reveal-type]
  --> *test_concat_flatten_types.py:46:16 (glob)
   |
46 |     reveal_type(y)  # Expected: Tensor[24], but might be Tensor[B * N * M]?
   |                ---
   |
ERROR Returned type `Tensor[((2 * 4) * 3)]` is not assignable to declared return type `Tensor[999]` [bad-return]
  --> *test_concat_flatten_types.py:49:12 (glob)
   |
49 |     return y  # Should ERROR if y is Tensor[24]
   |            ^
   |
  Size mismatch: expected 999, got 24
[1]
```

## Dim type variable in non-Tensor context (test_dim_in_non_tensor)

```scrut
$ $PYREFLY check "$TENSOR_TEST_ROOT/negative_tests/test_dim_in_non_tensor.py"
 INFO revealed type: MyContainer[5] [reveal-type]
  --> *test_dim_in_non_tensor.py:23:12 (glob)
   |
23 | reveal_type(result)  # Should be: MyContainer[5]
   |            --------
   |
 INFO revealed type: int [reveal-type]
  --> *test_dim_in_non_tensor.py:33:12 (glob)
   |
33 | reveal_type(result2)  # Should be: int
   |            ---------
   |
[0]
```

## View/reshape validation errors (test_view_errors)

```scrut
$ $PYREFLY check "$TENSOR_TEST_ROOT/negative_tests/test_view_errors.py"
ERROR Invalid dimension value -1: can only specify one unknown dimension as -1 [invalid-argument]
  --> *test_view_errors.py:19:15 (glob)
   |
19 |     y = x.view(-1, -1)  # ERROR: can only specify one unknown dimension as -1
   |               ^^^^^^^^
   |
ERROR Invalid dimension value -1: shape 3, -1 is not compatible with input size 200 [invalid-argument]
  --> *test_view_errors.py:26:15 (glob)
   |
26 |     y = x.view(3, -1)  # ERROR: 200 / 3 is not an integer
   |               ^^^^^^^
   |
ERROR Invalid dimension value -2: cannot specify -2 as a reshape dimension [invalid-argument]
  --> *test_view_errors.py:33:15 (glob)
   |
33 |     y = x.view(-2, 10)  # ERROR: invalid dimension value
   |               ^^^^^^^^
   |
ERROR Invalid dimension value 0: reshape dimensions cannot contain 0 [invalid-argument]
  --> *test_view_errors.py:40:15 (glob)
   |
40 |     y = x.view(0, -1)  # ERROR: reshape dimensions cannot contain 0
   |               ^^^^^^^
   |
[1]
```

## Item validation on non-scalar tensors (test_item_error)

```scrut
$ $PYREFLY check "$TENSOR_TEST_ROOT/negative_tests/test_item_error.py"
ERROR Invalid dimension value 1: item() only works on 0-dimensional tensors, got 1D tensor [invalid-argument]
  --> *test_item_error.py:20:11 (glob)
   |
20 |     x.item()
   |           ^^
   |
ERROR Invalid dimension value 2: item() only works on 0-dimensional tensors, got 2D tensor [invalid-argument]
  --> *test_item_error.py:28:11 (glob)
   |
28 |     x.item()
   |           ^^
   |
[1]
```

## View with zero dimension (test_view_simple)

```scrut
$ $PYREFLY check "$TENSOR_TEST_ROOT/negative_tests/test_view_simple.py"
ERROR Invalid dimension value 0: reshape dimensions cannot contain 0 [invalid-argument]
  --> *test_view_simple.py:26:15 (glob)
   |
26 |     y = x.view(0, -1)  # Should ERROR but still return Tensor
   |               ^^^^^^^
   |
 INFO revealed type: Tensor [reveal-type]
  --> *test_view_simple.py:27:16 (glob)
   |
27 |     reveal_type(y)
   |                ---
   |
[1]
```

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
 INFO revealed type: Dim[(2 * A)] [reveal-type]
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

## Tensor subtyping errors (test_tensor_subtyping)

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
ERROR Returned type `Tensor[N, M]` is not assignable to declared return type `Tensor[(M + N)]` [bad-return]
  --> *test_tensor_subtyping.py:88:12 (glob)
   |
88 |     return x  # ERROR: Tensor[N, M] not assignable to Tensor[N + M]
   |            ^
   |
  Tensor rank mismatch: expected 1 dimensions, got 2 dimensions
ERROR Returned type `Tensor[(1 + N), 3]` is not assignable to declared return type `Tensor[(2 + N), 3]` [bad-return]
  --> *test_tensor_subtyping.py:98:12 (glob)
   |
98 |     return x  # ERROR: N + 1 not equal to N + 2
   |            ^
   |
  Size mismatch: expected (2 + N), got (1 + N)
ERROR Returned type `Tensor[(M + N), 3]` is not assignable to declared return type `Tensor[(M * N), 3]` [bad-return]
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

## Tensor indexing errors (test_tensor_indexing)

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

## Tensor arithmetic errors (test_tensor_arithmetic)

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

## Generic function substitution with expressions (test_tensor_generic_exprs)

```scrut
$ $PYREFLY check "$TENSOR_TEST_ROOT/negative_tests/test_tensor_generic_exprs.py"
ERROR Returned type `Tensor[(3 + 2)]` is not assignable to declared return type `Tensor[6]` [bad-return]
   --> *test_tensor_generic_exprs.py:122:12 (glob)
    |
122 |     return sum_dims(x)  # ERROR
    |            ^^^^^^^^^^^
    |
  Size mismatch: expected 6, got 5
ERROR Returned type `Tensor[(3 * 2)]` is not assignable to declared return type `Tensor[5]` [bad-return]
   --> *test_tensor_generic_exprs.py:127:12 (glob)
    |
127 |     return product_dims(x)  # ERROR
    |            ^^^^^^^^^^^^^^^
    |
  Size mismatch: expected 5, got 6
ERROR Returned type `Tensor[(2 * 4), 5]` is not assignable to declared return type `Tensor[4, 5]` [bad-return]
   --> *test_tensor_generic_exprs.py:132:12 (glob)
    |
132 |     return double_first(x)  # ERROR
    |            ^^^^^^^^^^^^^^^
    |
  Size mismatch: expected 4, got 8
[1]
```

## Shape expression equivalence (test_tensor_expr_equiv)

```scrut
$ $PYREFLY check "$TENSOR_TEST_ROOT/negative_tests/test_tensor_expr_equiv.py"
ERROR Returned type `Tensor[(M + N)]` is not assignable to declared return type `Tensor[(M * N)]` [bad-return]
  --> *test_tensor_expr_equiv.py:85:12 (glob)
   |
85 |     return x
   |            ^
   |
  Size mismatch: expected (M * N), got (M + N)
ERROR Returned type `Tensor[(1 + N)]` is not assignable to declared return type `Tensor[(2 + N)]` [bad-return]
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

## Variadic shape patterns (test_tensor_variadic)

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
