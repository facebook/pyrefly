# File that will be opened in tests (should always show diagnostics)

def greet(name: str) -> str:
    return f"Hello, {name}"

# Type error: passing int to str parameter
message = greet(123)
