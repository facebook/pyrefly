# Import legacy generic classes with a bound and with a default

The pre-PEP-695 spelling of `test_import_generic_class_bound_and_default_pep695`. The
two are kept in step because the bound lives in a different place in each: on the
`TypeVar` call here, on the type parameter list there. They are stored differently
too, so they can start behaving differently.

`G` has a bound and no default; `H` has a default and no bound. Neither is used, so
neither the bound nor the default is needed.

## Files

`a.py`:
```python
from b import G, H
```

`b.py`:
```python
from typing import Generic

from typing_extensions import TypeVar

from c import Bound, Default

T = TypeVar("T", bound=Bound)
U = TypeVar("U", default=Default)

class G(Generic[T]):
    pass

class H(Generic[U]):
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
c: Exports

(38 builtin demands hidden)
a -> b::Load(module_exists)
a -> b::Exports(export_exists)
a -> b::Exports(is_implicit_reexport)
a -> b::Exports(get_deprecated)
a -> b::KeyExport(Name("G"))
  b -> c::Exports(is_special_export)
  b -> c::Exports(is_special_export)
a -> b::KeyExport(Name("H"))
```
