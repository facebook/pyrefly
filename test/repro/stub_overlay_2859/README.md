# Pyrefly Issue #2859 - Minimal Reproduction

This directory contains a minimal reproduction for issue #2859: **Generated typestubs are not found for untyped modules**.

**Issue:** https://github.com/facebook/pyrefly/issues/2859

## The Bug

When a Python module has no type hints (common for compiled extensions like `.so`/`.pyd` files), Pyrefly should look for companion `.pyi` stub files and use them for type checking. Currently, it doesn't.

## Directory Structure

```
stub_overlay_2859/
├── mypackage/
│   ├── __init__.py      # Re-exports add(), defines typed sub()
│   └── _core.py         # NO type hints (simulates compiled extension)
├── stubs/
│   └── mypackage/
│       └── _core.pyi    # Type stub: add(a: int, b: int) -> int
├── test_script.py       # Test that calls add("hello", "world")
├── pyrefly.toml         # Config with search-path = ["stubs"]
└── README.md
```

## How to Reproduce

```bash
cd test/repro/stub_overlay_2859
pyrefly check test_script.py
```

### Expected Output (when fixed): 4 errors
```
ERROR ... add("hello", "world") ... [bad-argument-type]
ERROR ... add("hello", "world") ... [bad-argument-type]
ERROR ... sub("foo", "bar") ... [bad-argument-type]
ERROR ... sub("foo", "bar") ... [bad-argument-type]
INFO 4 errors
```

### Actual Output (buggy): 2 errors
```
ERROR ... sub("foo", "bar") ... [bad-argument-type]
ERROR ... sub("foo", "bar") ... [bad-argument-type]
INFO 2 errors
```

The `add("hello", "world")` call does NOT produce an error because Pyrefly doesn't find `stubs/mypackage/_core.pyi`.
