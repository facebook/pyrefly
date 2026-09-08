# Import PEP-695 generic functions with a bound and with a default

`f` has a bound and no default; `g` has a default and no bound. Both name a
class from `c`, so the two show up as separate `KeyExport` edges and can be
told apart in the snapshot. Neither is called, so neither the bound nor the
default is needed.

This separates the two pieces of type parameter metadata. Reading a bound and
reading a default are different questions, and a design that resolves them
independently should be visible here as one edge appearing without the other.

## Files

`a.py`:
```python
from b import f, g
```

`b.py`:
```python
from c import Bound, Default

def f[T: Bound](x: T) -> T:
    return x

def g[U = Default](x: U) -> U:
    return x
```

`c.py`:
```python
class Bound:
    x: int = 1

class Default:
    y: int = 2
```

## Check `a.py`

```expected
a: Solutions
b: Answers
c: Answers

(53 builtin demands hidden)
a -> b::Load(module_exists)
a -> b::Exports(export_exists)
a -> b::Exports(is_implicit_reexport)
a -> b::Exports(get_deprecated)
a -> b::KeyExport(Name("f"))
  b -> c::Exports(is_special_export)
  b -> c::Exports(is_special_export)
  b -> c::Exports(export_exists)
  b -> c::Exports(is_implicit_reexport)
  b -> c::Exports(get_deprecated)
  b -> c::KeyExport(Name("Bound"))
  b -> c::KeyClassMetadata(ClassDefIndex(0))
  b -> c::KeyClassMetadata(ClassDefIndex(0))
  b -> c::KeyClassMetadata(ClassDefIndex(0))
  b -> c::KeyClassMetadata(ClassDefIndex(0))
a -> b::KeyExport(Name("g"))
  b -> c::Exports(export_exists)
  b -> c::Exports(is_implicit_reexport)
  b -> c::Exports(get_deprecated)
  b -> c::KeyExport(Name("Default"))
  b -> c::KeyClassMetadata(ClassDefIndex(1))
  b -> c::KeyClassMetadata(ClassDefIndex(1))
  b -> c::KeyClassMetadata(ClassDefIndex(1))
  b -> c::KeyClassMetadata(ClassDefIndex(1))
```
