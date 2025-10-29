# File that will be opened in tests (should always show diagnostics)

def greet(name: str) -> str:
    return f"Hello, {name}"

# No errors in this file
message = greet("World")
