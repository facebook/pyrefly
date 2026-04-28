# Transitive import without annotation

`a` imports `value` from `b`, where `value = compute()` has no type
annotation. `compute()` is defined in `c` with `-> int`.

`b -> c::KeyExport("compute")` resolves the function to determine
`value`'s type. Its children show `c` resolving `int` from builtins.
`a -> b::KeyExport("value")` then gets the resolved type.

All demands are necessary — without an annotation on `value`, the
type must be inferred by resolving the call chain. No superfluous
demands.

## Files

`a.py`:
```python
from b import value
x = value + 1
```

`b.py`:
```python
from c import compute
value = compute()
```

`c.py`:
```python
def compute() -> int: return 42
```

## Check `a.py`

```expected
a: Solutions
b: Answers
c: Answers

(168 builtin demands hidden)
a -> b::Exports(module_exists)
a -> b::Exports(export_exists)
a -> b::Exports(get_deprecated)
a -> b::Exports(is_special_export)
a -> b::KeyExport(Name("value"))
  b -> c::Exports(module_exists)
  b -> c::Exports(export_exists)
  b -> c::Exports(get_deprecated)
  b -> c::Exports(is_special_export)
  b -> c::Exports(is_special_export)
  b -> c::Exports(is_special_export)
  b -> c::Exports(is_special_export)
  b -> c::Exports(is_special_export)
  b -> c::Exports(is_special_export)
  b -> c::Exports(is_special_export)
  b -> c::Exports(is_special_export)
  b -> c::KeyExport(Name("compute"))
```
