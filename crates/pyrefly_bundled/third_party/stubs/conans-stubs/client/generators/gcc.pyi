from conans.client.generators.compiler_args import CompilerArgsGenerator as CompilerArgsGenerator
from conans.paths import BUILD_INFO_GCC as BUILD_INFO_GCC

class GCCGenerator(CompilerArgsGenerator):
    @property
    def filename(self): ...
    @property
    def compiler(self): ...
