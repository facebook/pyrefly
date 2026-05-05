# Unused import from same module

`a` imports only `light` from `b`, which also exports `heavy`.
`heavy`'s return type references `Heavy` from `c`.

Module `c` ends at `Step::Nothing` — `b`'s bind phase no longer
forces `c` via `module_exists`, and `heavy`'s signature is never
resolved because nobody demands it: the only Answer-level demand
into `b` is `KeyExport("light")`, and `light`'s return is
annotated `int`, so the chain stops there.

## Files

`a.py`:
```python
from b import light
x = light()
```

`b.py`:
```python
from c import Heavy
def light() -> int: return 1
def heavy() -> Heavy: ...
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
