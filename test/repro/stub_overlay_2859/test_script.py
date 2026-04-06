from mypackage import add, sub

# =============================================================================
# BUG REPRODUCTION: Issue #2859
# https://github.com/facebook/pyrefly/issues/2859
# =============================================================================

# This call SHOULD error because stubs/mypackage/_core.pyi defines:
#   add(a: int, b: int) -> int
# But Pyrefly doesn't find the stub, so NO ERROR is reported.
result1 = add("hello", "world")  # BUG: No error! Should be "expected int, got str"

# This call DOES error correctly because sub() has inline type hints
# in mypackage/__init__.py
result2 = sub("foo", "bar")  # ERROR: Argument `Literal['foo']` is not assignable...

print(f"result1 = {result1}")
print(f"result2 = {result2}")
