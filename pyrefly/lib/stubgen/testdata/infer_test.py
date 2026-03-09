"""Test that the solver fills in types for unannotated items."""


def add(x, y):
    return x + y


def greet(name):
    return f"Hello, {name}!"


class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1

    def get_count(self):
        return self.count


def make_counter():
    return Counter()


add(1, 2)
greet("world")
