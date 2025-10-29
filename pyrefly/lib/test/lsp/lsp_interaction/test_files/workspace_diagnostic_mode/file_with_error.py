# File with intentional type error for testing workspace diagnostic mode

def add_numbers(x: int, y: int) -> int:
    return x + y

# This should cause a type error: passing string to int parameter
result = add_numbers("hello", "world")
