# Transitive import with annotation

`a` imports `value` from `b`, which has annotation `value: int = 42`.
`b` imports `Inner` from `c`, but `value`'s type is determined by its
annotation, not by inference.

Only `a -> b::KeyExport("value")` appears. Module `c` has 0 solved
keys — the annotation `int` is resolved locally in `b` without
cascading. No superfluous demands.

## Files

`a.py`:
```python
from b import value
x = value + 1
```

`b.py`:
```python
from c import Inner
value: int = 42
```

`c.py`:
```python
class Inner:
    x: int = 1
```

## Check `a.py`

```expected
a: Solutions
b: Answers
c: Exports

(164 builtin demands hidden)
a -> b::Exports(module_exists)
a -> b::Exports(export_exists)
a -> b::Exports(get_deprecated)
a -> b::Exports(is_special_export)
a -> b::KeyExport(Name("value"))
  b -> c::Exports(module_exists)
  b -> c::Exports(export_exists)
  b -> c::Exports(get_deprecated)
```
