# Import function unused

Importing a function without calling it.

The single demand `a -> b::KeyExport("helper")` is necessary to resolve
the import. However, this triggers resolving the full function type in `b`
(11 solved keys), including the return annotation, decorated/undecorated
function chain, and legacy type param checks. Since `helper` is never
called, all of this work is superfluous — a lightweight handle would suffice.

## Files

`a.py`:
```python
from b import helper
```

`b.py`:
```python
def helper() -> int: return 1
```

## Check `a.py`

```expected
a: Solutions
b: Answers

(160 builtin demands hidden)
a -> b::Exports(module_exists)
a -> b::Exports(export_exists)
a -> b::Exports(get_deprecated)
a -> b::KeyExport(Name("helper"))
```
