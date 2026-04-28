# Attribute inherited from parent

Accessing `c.base_attr` where `base_attr` is defined on `Base` (in
module `c`), inherited by `Child` (in module `b`).

All necessary demands:
- `a -> b::KeyExport("Child")` — resolve the import
- `a -> b::KeyClassMetadata(0)` — needed to know Child's bases for MRO
- `b -> c::KeyExport("Base")` and `b -> c::KeyClassMetadata(0)` —
  resolving Child's MRO requires knowing Base
- `a -> b::KeyClassMro(0)` — compute MRO to walk ancestors
- `a -> c::KeyClassField(0, "base_attr")` — the actual attribute
  resolution; children (`c -> builtins::*`) resolve `int`
- `a -> b::KeyClassSynthesizedFields(0)` and `a -> c::KeyClassSynthesizedFields(0)`
  — MRO walk checks synthesized fields on each ancestor

**Superfluous:**
- `a -> b::KeyAbstractClassCheck(0)` — abstract checking is only
  relevant for instantiation, not for attribute access.

## Files

`a.py`:
```python
from b import Child
c = Child()
x = c.base_attr
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

(170 builtin demands hidden)
a -> b::Exports(module_exists)
a -> b::Exports(export_exists)
a -> b::Exports(get_deprecated)
a -> b::Exports(is_special_export)
a -> b::KeyExport(Name("Child"))
  b -> c::Exports(module_exists)
  b -> c::Exports(export_exists)
  b -> c::Exports(get_deprecated)
  b -> c::Exports(is_special_export)
  b -> c::Exports(is_special_export)
a -> b::KeyClassMetadata(ClassDefIndex(0))
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
a -> c::KeyClassField(ClassDefIndex(0), Name("base_attr"))
```
