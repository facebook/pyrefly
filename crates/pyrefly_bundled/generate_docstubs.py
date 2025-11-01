# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "docify",
# ]
# ///

from __future__ import annotations

from pathlib import Path
from shutil import copytree, rmtree

from docify import main as docify  # type: ignore


def main(*, overwrite: bool):
    """
    adds docstrings to the stubs located in `../stubdocs` for any modules that are compiled,
    because otherwise pyright would have no way to see the docstrings because there's no source
    code.

    note that it only generates stubs for modules and objects that exist on your current python
    version and OS.

    :param overwrite:
        whether to overwrite existing generated docstubs if they already exist. should be `True`
        when running locally to avoid running with a potentially outdated version of typeshed, but
        `False` in CI when generating them for all platforms because it needs to run multiple times
        on the previously generated output
    """
    stubs_path = Path(__file__).parent / "third_party/typeshed"
    stubs_with_docs_path = Path(__file__).parent / "third_party/docstubs"
    if overwrite:
        if stubs_with_docs_path.exists():
            rmtree(stubs_with_docs_path)
        copytree(stubs_path, stubs_with_docs_path, dirs_exist_ok=False)
    elif not stubs_with_docs_path.exists():
        copytree(stubs_path, stubs_with_docs_path, dirs_exist_ok=True)
    docify([str(stubs_with_docs_path / "stdlib"), "--if-needed", "--in-place"])

if __name__ == "__main__":
    main(overwrite=False)
