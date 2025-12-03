# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# /// script
# requires-python = "==3.12"
# dependencies = [
#   "docify",
# ]
# ///

# pyre-strict

"""
Tools for pulling an upstream typeshed archive from github, cleaning out
irrelevant data, and producing artifacts that can be bundled into Pyrefly executable.
"""

import argparse
import dataclasses
import datetime
import hashlib
import io
import json
import logging
import shutil
import tarfile
import urllib.request
from pathlib import Path

from docify import run as run_docify

LOG: logging.Logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class TypeshedEntry:
    path: Path
    data: bytes


@dataclasses.dataclass(frozen=True)
class FetchMetadata:
    url: str
    sha256: str
    timestamp: float


def get_default_typeshed_url() -> str:
    commit_hash = json.loads(
        urllib.request.urlopen(
            "https://api.github.com/repos/python/typeshed/commits/main"
        )
        .read()
        .decode("utf-8")
    )["sha"]
    LOG.info(f"Found typeshed main at commit {commit_hash}")
    return f"https://api.github.com/repos/python/typeshed/tarball/{commit_hash}"


def get_typeshed_url(specified_url: str | None) -> str:
    if specified_url is not None:
        return specified_url
    LOG.info("Typeshed URL not specified. Trying to auto-determine it...")
    try:
        return get_default_typeshed_url()
    except Exception as e:
        raise RuntimeError(
            "Cannot determine the default typeshed URL. "
            + "Please manually specify one with `--url` argument. "
            + "If the download still fails, please check network connectivity."
        ) from e


def get_output_dir(specified_output: Path | None) -> Path:
    if specified_output is not None:
        return specified_output
    possible = Path(__file__).parent.joinpath("third_party")
    if possible.joinpath("typeshed_metadata.json").exists():
        return possible
    raise RuntimeError(
        "Cannot determine the output directory. "
        + "Please manually specify one with `--output` argument."
    )


def fetch_as_tarfile(
    url: str,
) -> tuple[tarfile.TarFile, FetchMetadata]:
    LOG.info(f"Fetch typeshed tarball from URL `{url}`...")
    data = urllib.request.urlopen(url).read()
    metadata = FetchMetadata(
        url=url,
        sha256=hashlib.sha256(data).hexdigest(),
        timestamp=datetime.datetime.now().timestamp(),
    )
    return (
        tarfile.open(
            # Buffer the url data stream in memory so seeking in tarfile is possible
            fileobj=io.BytesIO(data),
            mode="r:gz",
        ),
        metadata,
    )


def should_include_member(info: tarfile.TarInfo) -> bool:
    if info.isdir():
        return False

    path = Path(info.name)
    parts = path.parts

    # There are quite a few config files and some test directories in
    # typeshed that we don't need - only pull in the directories that
    # actually contain stubs.
    if len(parts) < 2:
        return False

    # The typeshed repository contains two stub roots, which are different:
    # - The "stdlib" directory contains stubs for the standard library, and
    #   for stdlib typeshed is the correct way to get them.
    # - The "stubs" directory contains stubs for third-party libraries, which
    #   are mirrorored into various `xyz-stubs` packages on PyPI.
    # We include both the stdlib stubs as well as third party stubs.
    if parts[1] not in ("stdlib", "stubs"):
        return False

    # Only Python source files are interesting to us
    if path.suffix not in (".py", ".pyi"):
        return False

    # Skip Python 2-only stubs; Pyre no longer supports Python 2.
    if "@python2" in parts:
        return False

    # Skip stub tests
    if "@tests" in parts:
        return False

    return True


def relative_path(info: tarfile.TarInfo) -> Path:
    """
    Convert a filename within a typeshed tarball into a path relative to the
    top of typeshed/.
    """
    return Path(*Path(info.name).parts[1:])


def trim_typeshed(input_tar: tarfile.TarFile) -> list[TypeshedEntry]:
    LOG.info("Trimming down typeshed tarfile...")
    entries = (
        (
            relative_path(member),
            input_tar.extractfile(member),
        )
        for member in input_tar.getmembers()
        if should_include_member(member)
    )
    return [
        TypeshedEntry(path, reader.read())
        for path, reader in entries
        if reader is not None
    ]


def write_typeshed(output_path: Path, entries: list[TypeshedEntry]) -> None:
    LOG.info("Clearing output directory...")
    shutil.rmtree(output_path, ignore_errors=True)
    LOG.info("Writing trimmed typeshed to disk...")
    for entry in entries:
        file_path = output_path / entry.path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(entry.data)


def add_docstrings(stubs_dir: Path) -> None:
    LOG.info("Running `docify` for enriched docstrings...")
    run_docify(
        input_dirs=[str(stubs_dir)],
        if_needed=True,
        in_place=True,
    )
    pass


def write_metadata(output_path: Path, metadata: FetchMetadata) -> None:
    output_path.write_text(
        json.dumps(
            {
                "url": metadata.url,
                "sha256": metadata.sha256,
                "update_time": datetime.datetime.fromtimestamp(
                    metadata.timestamp
                ).isoformat(sep=" ", timespec="seconds"),
            },
            indent=2,
        )
    )


def run(url: str, output_dir: Path, enable_docify: bool) -> None:
    if not output_dir.exists():
        raise RuntimeError(f"Output path `{output_dir}` does not exist")
    if not output_dir.is_dir():
        raise RuntimeError(f"Output path `{output_dir}` is not a directory")
    typeshed_tarfile, metadata = fetch_as_tarfile(url)
    entries = trim_typeshed(typeshed_tarfile)
    stubs_dir = output_dir / "typeshed"
    write_typeshed(stubs_dir, entries)
    if enable_docify:
        # TODO(grievejia): Only docify stdlib for now
        add_docstrings(stubs_dir / "stdlib")
    write_metadata(output_dir / "typeshed_metadata.json", metadata)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch typeshed from a given URL and trim out files that are not needed for Pyrefly bundling."
    )
    parser.add_argument(
        "-u",
        "--url",
        help="The URL to download from. If not specified, the trunk files on the typeshed main branch will be used.",
    )
    parser.add_argument(
        "--docify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run `docify` on the downloaded typeshed, which enriches the stub files with more docstrings. This flag is enabled by default.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="The directory to write the downloaded typeshed to.",
    )
    args = parser.parse_args()
    # @lint-ignore FIXIT1: OSS scripts cannot take on fb-internal dependency
    logging.basicConfig(
        format="[%(levelname)-7s] [%(asctime)s] [%(name)s]: %(message)s",
        level=logging.INFO,
    )
    run(get_typeshed_url(args.url), get_output_dir(args.output), args.docify)


if __name__ == "__main__":
    main()
