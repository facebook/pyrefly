# Dim type checking error tests

These tests verify that the type checker correctly catches Dim type errors.

## Dim type variable unification (test_symint_unification)

```scrut
$ $PYREFLY check "$TEST_ROOT/negative_tests/test_symint_unification.py"
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
$ $PYREFLY check "$TEST_ROOT/negative_tests/test_symint_any.py"
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
