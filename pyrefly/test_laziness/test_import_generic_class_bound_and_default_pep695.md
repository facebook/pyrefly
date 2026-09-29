# Import PEP-695 generic classes with a bound and with a default

A class reaches its type parameters by a different route than a function does, so it
gets its own case. Importing is enough to force the bound: annotating with `G`,
constructing it, subclassing it and passing it to `isinstance` all produce the same
demand, because resolving the export is what forces the bound. Plain import is the
version of that with the least unrelated noise.

`G` has a bound and no default; `H` has a default and no bound. Neither is used, so
neither the bound nor the default is needed.

## Files

`a.py`:
```python
from b import G, H
```

`b.py`:
```python
from c import Bound, Default

class G[T: Bound]:
    pass

class H[U = Default]:
    pass
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

(51 builtin demands hidden)
a -> b::Load(module_exists)
a -> b::Exports(export_exists)
a -> b::Exports(is_implicit_reexport)
a -> b::Exports(get_deprecated)
a -> b::KeyExport(Name("G"))
  b -> c::Exports(is_special_export)
  b -> c::Exports(is_special_export)
  b -> c::Exports(export_exists)
  b -> c::Exports(is_implicit_reexport)
  b -> c::Exports(get_deprecated)
  b -> c::KeyExport(Name("Bound"))
  b -> c::KeyClassMetadata(ClassDefIndex(0))
  b -> c::KeyClassMetadata(ClassDefIndex(0))
a -> b::KeyExport(Name("H"))
  b -> c::Exports(export_exists)
  b -> c::Exports(is_implicit_reexport)
  b -> c::Exports(get_deprecated)
  b -> c::KeyExport(Name("Default"))
  b -> c::KeyClassMetadata(ClassDefIndex(1))
  b -> c::KeyClassMetadata(ClassDefIndex(1))
```
