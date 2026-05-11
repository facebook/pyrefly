# Multiple inheritance check solves unique parent fields

`a.py` (check target) defines `C(B1, B2)` with no fields of its own.
`B1` (in `b.py`) defines `p1` and `shared`. `B2` (in `c.py`) defines
`p2` and `shared`. Only `shared` appears in both parents and needs a
consistency check.

`check_consistent_multiple_inheritance` iterates each parent's fields
and calls `get_class_member` on each, resolving `KeyClassField` for
ALL parent fields: `p1`, `shared` (from B1) and `p2`, `shared` (from
B2). But only `shared` appears in multiple bases (line 3382 checks
`len() > 1`). Resolving `p1` and `p2` is wasted work — they're unique
to one parent and can't have consistency issues.

**Superfluous:** `KeyClassField` demands for `p1` and `p2` — fields
unique to a single parent don't need type resolution for the multiple
inheritance check. The check could collect field names first, then only
resolve fields that appear in multiple bases.

## Files

`a.py`:
```python
from b import B1
from c import B2
class C(B1, B2):
    pass
```

`b.py`:
```python
class B1:
    p1: int = 1
    shared: int = 10
```

`c.py`:
```python
class B2:
    p2: int = 2
    shared: int = 20
```

## Check `a.py`

```expected
a: Solutions
b: Answers
c: Answers

(190 builtin demands hidden)
a -> b::Exports(is_special_export)
a -> c::Exports(is_special_export)
a -> b::Load(module_exists)
a -> b::Exports(export_exists)
a -> b::Exports(get_deprecated)
a -> b::KeyExport(Name("B1"))
a -> c::Load(module_exists)
a -> c::Exports(export_exists)
a -> c::Exports(get_deprecated)
a -> c::KeyExport(Name("B2"))
a -> b::KeyClassMetadata(ClassDefIndex(0))
a -> c::KeyClassMetadata(ClassDefIndex(0))
a -> b::KeyClassBaseType(ClassDefIndex(0))
a -> c::KeyClassBaseType(ClassDefIndex(0))
a -> b::KeyClassMro(ClassDefIndex(0))
a -> c::KeyClassMro(ClassDefIndex(0))
a -> b::KeyClassField(ClassDefIndex(0), Name("p1"))
a -> b::KeyClassField(ClassDefIndex(0), Name("shared"))
a -> c::KeyClassField(ClassDefIndex(0), Name("p2"))
a -> c::KeyClassField(ClassDefIndex(0), Name("shared"))
a -> b::KeyClassField(ClassDefIndex(0), Name("shared"))
```
