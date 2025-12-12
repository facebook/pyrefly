from conans.client.generators.virtualrunenv import VirtualRunEnvGenerator as VirtualRunEnvGenerator

class VirtualEnvPythonGenerator(VirtualRunEnvGenerator):
    suffix: str
    venv_name: str
    def __init__(self, conanfile) -> None: ...
