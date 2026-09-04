# Import legacy generic functions with a bound and with a default

The pre-PEP-695 spelling of `test_import_generic_function_bound_and_default_pep695`. The two
are kept in step because the bound lives in a different place in each: on the
`TypeVar` call here, on the type parameter list there. They are stored differently
too, so they can start behaving differently.

`f` has a bound and no default; `g` has a default and no bound. Both name a class
from `c`, so the two show up as separate `KeyExport` edges. Neither is called, so
neither the bound nor the default is needed.

## Files

`a.py`:
```python
from b import f, g
```

`b.py`:
```python
from typing_extensions import TypeVar

from c import Bound, Default

T = TypeVar("T", bound=Bound)
U = TypeVar("U", default=Default)

def f(x: T) -> T:
    return x

def g(x: U) -> U:
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

(56 builtin demands hidden)
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
a -> b::KeyExport(Name("g"))
  b -> c::Exports(export_exists)
  b -> c::Exports(is_implicit_reexport)
  b -> c::Exports(get_deprecated)
  b -> c::KeyExport(Name("Default"))
  b -> c::KeyClassMetadata(ClassDefIndex(1))
  b -> c::KeyClassMetadata(ClassDefIndex(1))
```
