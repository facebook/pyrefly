# Attribute on class itself

Accessing `c.child_attr` where `child_attr` is defined on `Child`, not
inherited from `Base`.

`a -> b::KeyExport("Child")` is necessary.

The demand tree is nearly identical to `test_attribute_inherited` — all
the same class metadata, MRO, abstract check, and synthesized fields
demands appear. This is because the attribute lookup code in attr.rs
walks the FULL MRO for every attribute access, even when the attribute
is found on the first class checked.

**Superfluous demands:**
- ALL demands involving `c` (Base's module) are superfluous — `child_attr`
  is on `Child`, so `Base` is never needed.
- `a -> b::KeyAbstractClassCheck(0)` is superfluous for attribute access
  (only needed for instantiation).
- `a -> b::KeyClassMro(0)` and the cascading `b -> c::*` demands are
  superfluous — the MRO iterator trick would avoid materializing the
  MRO when the attribute is found on the class itself.

## Files

`a.py`:
```python
from b import Child
c = Child()
x = c.child_attr
```

`b.py`:
```python
from c import Base
class Child(Base):
    child_attr: str = 'hello'
```

`c.py`:
```python
class Base:
    base_attr: int = 1
```

## Check `a.py`

```expected
a: Solutions
b: Answers
c: Answers

(181 builtin demands hidden)
a -> b::Exports(is_special_export)
a -> b::Load(module_exists)
a -> b::Exports(export_exists)
a -> b::Exports(get_deprecated)
a -> b::KeyExport(Name("Child"))
  b -> c::Exports(is_special_export)
a -> b::KeyClassMetadata(ClassDefIndex(0))
  b -> c::Exports(export_exists)
  b -> c::Exports(get_deprecated)
  b -> c::KeyExport(Name("Base"))
  b -> c::KeyClassMetadata(ClassDefIndex(0))
  b -> c::KeyClassMetadata(ClassDefIndex(0))
a -> b::KeyClassMetadata(ClassDefIndex(0))
a -> b::KeyAbstractClassCheck(ClassDefIndex(0))
  b -> c::KeyClassMetadata(ClassDefIndex(0))
a -> b::KeyClassSynthesizedFields(ClassDefIndex(0))
a -> b::KeyClassMro(ClassDefIndex(0))
  b -> c::KeyClassMetadata(ClassDefIndex(0))
  b -> c::KeyClassBaseType(ClassDefIndex(0))
  b -> c::KeyClassMro(ClassDefIndex(0))
a -> c::KeyClassSynthesizedFields(ClassDefIndex(0))
a -> b::KeyClassMro(ClassDefIndex(0))
a -> b::KeyClassField(ClassDefIndex(0), Name("child_attr"))
```
