# Transitive import with annotation

`a` imports `value` from `b`, which has annotation `value: int = 42`.
`b` imports `Inner` from `c`, but `value`'s type is determined by its
annotation, not by inference.

Module `c` ends at `Step::Nothing` — `b`'s bind phase no longer
forces `c` via `module_exists`, and `value`'s annotation `int` is
resolved locally in `b` without cascading into `c`'s export set.

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
c: Nothing

(164 builtin demands hidden)
a -> b::Load(module_exists)
a -> b::Exports(export_exists)
a -> b::Exports(get_deprecated)
a -> b::KeyExport(Name("value"))
```
