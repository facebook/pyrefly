# is_special_export forces exports on transitive dep

`a` imports `value` from `b`. `b` imports `MyTypeVar` from `c` (which
re-exports `TypeVar` from `typing`) and uses it to define a type
variable. `a` only uses `value`, not the type variable.

**Superfluous:** `c` being computed to Exports. During `b`'s binding,
`x = MyTypeVar("x")` triggers `is_special_export(c, "MyTypeVar")` which
demands `Step::Exports` on `c` to check if it's a re-export of a known
special name (TypeVar, ParamSpec, etc.). This cascades even though `a`
doesn't use the type variable.

## Files

`a.py`:
```python
from b import value
x = value
```

`b.py`:
```python
from c import MyTypeVar
T = MyTypeVar("T")
value: int = 42
```

`c.py`:
```python
from typing import TypeVar as MyTypeVar
```

## Check `a.py`

```expected
a: Solutions
b: Answers
c: Exports

(159 builtin demands hidden)
a -> b::Load(module_exists)
a -> b::Exports(export_exists)
a -> b::Exports(get_deprecated)
a -> b::KeyExport(Name("value"))
  b -> c::Exports(is_special_export)
```
