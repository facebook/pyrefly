# Import function called

Importing a function and calling it. The demand tree is identical to
the unused case — `a -> b::KeyExport("helper")` — because `KeyExport`
eagerly resolves the full function type regardless of usage.

All demands are justified here since the function is called and its
return type is needed.

## Files

`a.py`:
```python
from b import helper
x = helper()
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
a -> b::Exports(is_special_export)
a -> b::Load(module_exists)
a -> b::Exports(export_exists)
a -> b::Exports(get_deprecated)
a -> b::KeyExport(Name("helper"))
```
