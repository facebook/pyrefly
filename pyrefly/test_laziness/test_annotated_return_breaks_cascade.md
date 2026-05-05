# Annotated return breaks cascade

Calling `get_config()` which has return annotation `-> int`. The
function body uses `Config` from module `c`, but callers should not
need to resolve `c` since the return type is explicitly annotated.

Module `c` has 0 solved keys — the annotation breaks the Answer-level
cascade as intended. `c` still reaches `Step::Exports` because
`b`'s binding still demands `is_special_export(c, "Config")` to
classify the name; that's a separate opportunity covered by
`test_special_export_forces_exports`.

## Files

`a.py`:
```python
from b import get_config
x = get_config()
```

`b.py`:
```python
from c import Config
def get_config() -> int:
    c = Config()
    return c.debug
```

`c.py`:
```python
class Config:
    debug: bool = False
```

## Check `a.py`

```expected
a: Solutions
b: Answers
c: Exports

(160 builtin demands hidden)
a -> b::Exports(is_special_export)
a -> b::Load(module_exists)
a -> b::Exports(export_exists)
a -> b::Exports(get_deprecated)
a -> b::KeyExport(Name("get_config"))
  b -> c::Exports(is_special_export)
  b -> c::Exports(is_special_export)
```
