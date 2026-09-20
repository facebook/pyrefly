Tests directly in this directory run both statically and against Torch at runtime.
Static suites enable expectation checking, so every `# E:` marker must match a
Pyrefly error. Tests under `old/` remain static-only until they are migrated here.
