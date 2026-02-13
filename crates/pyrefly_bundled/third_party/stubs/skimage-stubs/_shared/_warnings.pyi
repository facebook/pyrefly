from contextlib import contextmanager

__all__ = ["all_warnings", "expected_warnings", "warn"]

# A version of `warnings.warn` with a default stacklevel of 2.
# functool is used so as not to increase the call stack accidentally
warn = ...

@contextmanager
def all_warnings(): ...
@contextmanager
def expected_warnings(matching): ...
