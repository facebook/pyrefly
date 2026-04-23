# Import class instantiated

Importing a class and calling its constructor (`f = Foo()`).

All demands are necessary for constructor resolution:
- `KeyExport("Foo")` — resolve the import
- `KeyClassMetadata(0)` — needed for metaclass check, `__init__`/`__new__`
  lookup, and field synthesis
- `KeyClassSynthesizedFields(0)` — resolve synthesized `__init__`
- `KeyClassMro(0)` — find `__init__`/`__new__` in parent classes

**Superfluous:**
- `KeyAbstractClassCheck(0)` — checks for unimplemented abstract methods.
  For a simple class with no abstract parents, this could be skipped
  when metadata shows no ABC lineage.

## Files

`a.py`:
```python
from b import Foo
f = Foo()
```

`b.py`:
```python
class Foo:
    x: int = 1
```

## Check `a.py`

```expected
a: Solutions
b: Answers

(159 builtin demands hidden)
a -> b::Exports(module_exists)
a -> b::Exports(export_exists)
a -> b::Exports(get_deprecated)
a -> b::Exports(is_special_export)
a -> b::KeyExport(Name("Foo"))
a -> b::KeyClassMetadata(ClassDefIndex(0))
a -> b::KeyAbstractClassCheck(ClassDefIndex(0))
a -> b::KeyClassSynthesizedFields(ClassDefIndex(0))
a -> b::KeyClassMro(ClassDefIndex(0))
```
