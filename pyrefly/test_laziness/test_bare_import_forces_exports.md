# Bare import does not force the target

`a` imports `light` from `b`. `b` has `import c` (bare import, not
`from c import ...`). `a` only uses `light()` which doesn't involve `c`.

`c` ends at `Step::Nothing`: `b`'s bind phase doesn't demand anything
from `c`, and the solver only checks `c`'s findability when
`Binding::Module` for the bare import is actually consumed. Since
`a` never reaches `b`'s `c` reference, that solve never runs.

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
c: Nothing

(160 builtin demands hidden)
a -> b::Exports(is_special_export)
a -> b::Load(module_exists)
a -> b::Exports(export_exists)
a -> b::Exports(get_deprecated)
a -> b::KeyExport(Name("light"))
```
