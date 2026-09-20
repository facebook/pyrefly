Tests directly in this directory run both statically and against Torch at runtime.
Static suites enable expectation checking, so every `# E:` marker must match a
Pyrefly error. Tests under `old/` remain static-only until they are migrated here.

Each test function outside a `TYPE_CHECKING` block is executed against the installed
Torch package. In particular, `assert_raises(ExpectedError)` is a runtime assertion:
the call must raise that exception class (or a subclass). An adjacent `# E:` marker
independently asserts that Pyrefly reports a static error for the same call.
