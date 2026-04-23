# get_deprecated forces exports on transitive dep

`a` imports `value` from `b`. `b` imports `old_func` from `c` with a
deprecation marker. `a` only uses `value`, not `old_func`.

**Superfluous:** `c` being computed to Exports. During `b`'s binding,
`from c import old_func` triggers `get_deprecated(c, "old_func")` which
demands `Step::Exports` on `c`, even though `a` never uses `old_func`.

Note: This test may be hard to trigger since deprecation requires `c` to
actually mark `old_func` as deprecated via `warnings.deprecated` or
similar. The `get_deprecated` call happens regardless of whether the name
is actually deprecated — the check itself forces exports.

## Files

`a.py`:
```python
from b import value
x = value
```

`b.py`:
```python
from c import old_func
value: int = 42
```

`c.py`:
```python
def old_func() -> None: ...
```

## Check `a.py`

```expected
a: Solutions
b: Answers
c: Exports

(159 builtin demands hidden)
a -> b::Exports(module_exists)
a -> b::Exports(export_exists)
a -> b::Exports(get_deprecated)
a -> b::Exports(is_special_export)
a -> b::KeyExport(Name("value"))
  b -> c::Exports(module_exists)
  b -> c::Exports(export_exists)
  b -> c::Exports(get_deprecated)
```
