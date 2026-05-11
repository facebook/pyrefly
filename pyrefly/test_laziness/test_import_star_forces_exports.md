# Wildcard import forces exports on transitive dep

`a` imports `light` from `b`. `b` does `from c import *`, pulling in all
of `c`'s exports — but `a` only uses `light()`, which is defined locally
in `b` and doesn't involve anything from `c`.

**Superfluous:** `c` being computed to Exports at all. `b`'s binding calls
`get_wildcard(c)` which demands `Step::Exports` on `c`. This is harder to
avoid than other LookupExport calls because the binder needs the set of
wildcard names to create binding entries, but it means every transitive
`import *` forces the target module's exports to be computed.

## Files

`a.py`:
```python
from b import light
x = light()
```

`b.py`:
```python
from c import *
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
a -> b::Exports(is_special_export)
b -> c::Exports(get_wildcard)
a -> b::Load(module_exists)
a -> b::Exports(export_exists)
a -> b::Exports(get_deprecated)
a -> b::KeyExport(Name("light"))
  b -> c::Exports(get_wildcard)
  b -> c::Exports(get_wildcard)
  b -> c::Exports(export_exists)
```
