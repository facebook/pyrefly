# Annotated recursive signatures do not force bodies

Resolving an annotated function signature must not solve its implicit-return
validation. Otherwise, fallthrough calls make mutually recursive bodies force
each other even though callers only need the annotations.

## Files

`a.py`:
```python
from b import f
x = f()
```

`b.py`:
```python
from c import g

def f() -> int:
    g()
```

`c.py`:
```python
from b import f

def g() -> int:
    f()
```

## Check `a.py`

```expected
a: Solutions
b: Answers
c: Exports

(36 builtin demands hidden)
a -> b::Exports(is_special_export)
a -> b::Load(module_exists)
a -> b::Exports(export_exists)
a -> b::Exports(is_implicit_reexport)
a -> b::Exports(get_deprecated)
a -> b::KeyExport(Name("f"))
  b -> c::Exports(is_special_export)
  b -> c::Exports(is_special_export)
```
