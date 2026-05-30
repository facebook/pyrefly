# finality check forces exports on transitive dep

`a` imports `value` from `b`. `b` imports `X` from `c` and then
reassigns `X = 2`. `a` only uses `value`, not `X`.

**Superfluous:** `c` being computed to Exports. During `b`'s binding,
the reassignment `X = 2` triggers `export_origin(c, "X")` to check if `X`
was declared as `Final` in `c`. This demands `Step::Exports` on `c`,
even though `a` never uses `X`.

## Files

`a.py`:
```python
from b import value
x = value
```

`b.py`:
```python
from c import X
X = 2
def value() -> int: return 42
```

`c.py`:
```python
X: int = 1
```

## Check `a.py`

```expected
a: Solutions
b: Answers
c: Exports

(160 builtin demands hidden)
a -> b::Load(module_exists)
a -> b::Exports(export_exists)
a -> b::Exports(get_deprecated)
a -> b::KeyExport(Name("value"))
  b -> c::Exports(export_origin)
```
