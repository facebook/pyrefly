# export_exists check forces exports on transitive dep

`a` imports `value` from `b`. `b` does `from c import Foo` — during
binding, this triggers `export_exists(c, "Foo")` and `module_exists(c)`
to verify the import. `a` only uses `value`, not `Foo`.

**Superfluous:** `c` being computed to Exports. The `export_exists` and
`module_exists` checks during `b`'s binding demand `Step::Exports` on
`c`, even though `a` never uses `Foo`. Ideally, `from c import Foo`
would create an optimistic `Binding::Import` without demanding `c`'s
exports, deferring validation to solve time.

This is the same pattern as `test_unused_import_from_same_module` but
simplified: `b` imports from `c` but only exports an unrelated value.

## Files

`a.py`:
```python
from b import value
x = value
```

`b.py`:
```python
from c import Foo
value: int = 42
```

`c.py`:
```python
class Foo:
    x: int = 1
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
