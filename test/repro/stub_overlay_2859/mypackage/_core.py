# Simulates a compiled extension module (like a .so/.pyd file)
# This module has NO type hints - the types only exist in _core.pyi

def add(a, b):
    """Add two values. No type hints here!"""
    return a + b
