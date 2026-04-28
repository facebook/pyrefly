# Bare import forces exports on transitive dep

`a` imports `light` from `b`. `b` has `import c` (bare import, not
`from c import ...`). `a` only uses `light()` which doesn't involve `c`.

**Superfluous:** `c` being computed to Exports. `b`'s binding calls
`module_exists(c)` which demands `Step::Exports` on `c`, even though
`b` never actually uses `c` in a way that `a` would need.

## Files

`a.py`:
```python
from b import light
x = light()
```

`b.py`:
```python
import c
def light() -> int: return 1
```

`c.py`:
```python
class Heavy:
    x: int = 1
```

## Check `a.py`

```expected
a: Solutions
b: Answers
c: Exports

(160 builtin demands hidden)
a -> b::Exports(module_exists)
a -> b::Exports(export_exists)
a -> b::Exports(get_deprecated)
a -> b::Exports(is_special_export)
a -> b::KeyExport(Name("light"))
  b -> c::Exports(module_exists)
```
