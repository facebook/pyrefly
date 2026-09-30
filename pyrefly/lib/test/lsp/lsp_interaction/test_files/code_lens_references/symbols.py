class Greeter:
    def greet(self) -> str:
        return "hi"

def run(g: Greeter) -> str:
    return g.greet()
