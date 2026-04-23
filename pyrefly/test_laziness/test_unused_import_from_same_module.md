# Unused import from same module

`a` imports only `light` from `b`, which also exports `heavy`.
`heavy`'s return type references `Heavy` from `c`.

Only `a -> b::KeyExport("light")` appears. Module `c` has 0 solved
keys — `heavy`'s signature is not resolved because it's not demanded.

No superfluous demands in the demand tree. However, `b`'s solved keys
show that `heavy`'s function chain IS partially resolved internally
(KeyDecoratedFunction, KeyUndecoratedFunction for `heavy`) even though
no one imports it. This happens because step_solutions for `b` resolves
all exported keys, not just the demanded one.

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
c: Exports

(160 builtin demands hidden)
a -> b::Exports(module_exists)
a -> b::Exports(export_exists)
a -> b::Exports(get_deprecated)
a -> b::Exports(is_special_export)
a -> b::KeyExport(Name("light"))
  b -> c::Exports(module_exists)
  b -> c::Exports(export_exists)
  b -> c::Exports(get_deprecated)
  b -> c::Exports(is_special_export)
```
