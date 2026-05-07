# @generated
class Empty:
    ...


class Animal:
    name: str
    sound: str

    def __init__(self, name: str, sound: str) -> None: ...

    def speak(self) -> str: ...


class Dog(Animal):
    breed: str

    def fetch(self, item: str) -> str: ...
