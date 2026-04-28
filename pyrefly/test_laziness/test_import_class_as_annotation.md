# Import class as annotation

Importing a class and using it only as a type annotation (`x: Foo`).

`a -> b::KeyExport("Foo")` is necessary to resolve the import.

`a -> b::KeyClassMetadata(0)` is superfluous for annotation-only
usage. It's triggered by `type_of_instance` in targs.rs which calls
`get_metadata_for_class` to check `is_typed_dict()` — because
`Type::ClassType` and `Type::TypedDict` are different type variants,
the code must decide which to create. The `is_typed_dict` check
requires resolving the class's bases to see if any ancestor is
`TypedDict`, which recursively demands metadata up the MRO.

Fixing this requires unifying `ClassType` and `TypedDict` into a
single type variant, deferring the TypedDict distinction until
TypedDict-specific features are actually used.

## Files

`a.py`:
```python
from b import Foo
x: Foo
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

(157 builtin demands hidden)
a -> b::Exports(module_exists)
a -> b::Exports(export_exists)
a -> b::Exports(get_deprecated)
a -> b::Exports(is_special_export)
a -> b::KeyExport(Name("Foo"))
a -> b::KeyClassMetadata(ClassDefIndex(0))
```
