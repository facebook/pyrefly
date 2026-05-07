# Duplicate `from c import X` skips the `is_final` check

`a` imports `value` from `b`. `b` imports `X` from `c` twice -- a
common pattern when the same name is brought in inside multiple
`if TYPE_CHECKING:` or method-local blocks. Each duplicate goes
through `check_for_imported_final_reassignment`, but the
duplicate-import early exit fires *before* the cross-module
`is_final` lookup, so `c` is never forced to `Step::Exports`.

Compare with `test_is_final_forces_exports`, which has a real
reassignment (`X = 2`) and does force `c: Exports`.

## Files

`a.py`:
```python
from b import value
x = value
```

`b.py`:
```python
from c import X
from c import X
def value() -> int: return 42
```

`c.py`:
```python
X: int = 1
```

## Check `a.py`

```expected
a: Solutions
b: Answers
c: Nothing

(160 builtin demands hidden)
a -> b::Load(module_exists)
a -> b::Exports(export_exists)
a -> b::Exports(get_deprecated)
a -> b::KeyExport(Name("value"))
```
