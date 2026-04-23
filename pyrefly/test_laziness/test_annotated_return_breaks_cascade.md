# Annotated return breaks cascade

Calling `get_config()` which has return annotation `-> int`. The
function body uses `Config` from module `c`, but callers should not
need to resolve `c` since the return type is explicitly annotated.

Only `a -> b::KeyExport("get_config")` appears — this is correct.
Module `c` has 0 solved keys, confirming that the annotation breaks
the cascade. No superfluous demands.

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
a -> b::Exports(module_exists)
a -> b::Exports(export_exists)
a -> b::Exports(get_deprecated)
a -> b::Exports(is_special_export)
a -> b::KeyExport(Name("get_config"))
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
```
