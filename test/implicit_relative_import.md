# Tests for the `implicit-relative-import` diagnostic

These tests exercise the diagnostic end-to-end through `pyrefly check`. The
diagnostic fires when an unqualified `import` resolves *only* via the
directory-upward fallback search path, not through any configured absolute
root. It defaults to `ignore`, so each surfacing test enables it explicitly.

To force the fallback tier to be the resolver, each tree uses two sibling
subdirectories (`a/` and `b/`). That makes pyrefly infer the project root
(their common parent) as the import root, so a module living in `a/` is NOT
covered by the inferred root and resolves only via the directory walk from
the importing file in `a/`.

Note on streams: with `--output-format=min-text`, diagnostic lines are written
to **stdout** while the ` INFO N errors` summary is written to **stderr**.
So the "fires" cases capture stdout (the diagnostic), and the clean cases
capture stderr (the summary).

## Fires when an import resolves only via the fallback path

```scrut
$ mkdir -p $TMPDIR/on_fallback/a $TMPDIR/on_fallback/b && \
> printf 'enable-fallback-search-path = true\n[errors]\nimplicit-relative-import = "error"\n' > $TMPDIR/on_fallback/pyrefly.toml && \
> echo "x: int = 1" > $TMPDIR/on_fallback/a/sibling.py && \
> echo "import sibling" > $TMPDIR/on_fallback/a/main.py && \
> echo "y: int = 2" > $TMPDIR/on_fallback/b/other.py && \
> $PYREFLY check -c $TMPDIR/on_fallback/pyrefly.toml --output-format=min-text $TMPDIR/on_fallback/a/main.py
*/a/main.py:1:8-15: Module `sibling` was imported using an implicit relative import. Prefer an explicit relative import (`from . import sibling`) or add the module's root to the configured search path. [implicit-relative-import] (glob)
[1]
```

## Does NOT fire when the module is on the configured search path

When `sibling` is discoverable through a configured absolute root, the
fallback tier never runs, so no caveat is attached (no false positive).

```scrut {output_stream: stderr}
$ mkdir -p $TMPDIR/on_search_path/a $TMPDIR/on_search_path/b && \
> printf 'enable-fallback-search-path = true\nsearch_path = ["%s/on_search_path/a"]\n[errors]\nimplicit-relative-import = "error"\n' "$TMPDIR" > $TMPDIR/on_search_path/pyrefly.toml && \
> echo "x: int = 1" > $TMPDIR/on_search_path/a/sibling.py && \
> echo "import sibling" > $TMPDIR/on_search_path/a/main.py && \
> echo "y: int = 2" > $TMPDIR/on_search_path/b/other.py && \
> $PYREFLY check -c $TMPDIR/on_search_path/pyrefly.toml --output-format=min-text $TMPDIR/on_search_path/a/main.py
 INFO 0 errors
[0]
```

## Default severity is ignore (no output unless explicitly enabled)

```scrut {output_stream: stderr}
$ mkdir -p $TMPDIR/default_ignore/a $TMPDIR/default_ignore/b && \
> printf 'enable-fallback-search-path = true\n' > $TMPDIR/default_ignore/pyrefly.toml && \
> echo "x: int = 1" > $TMPDIR/default_ignore/a/sibling.py && \
> echo "import sibling" > $TMPDIR/default_ignore/a/main.py && \
> echo "y: int = 2" > $TMPDIR/default_ignore/b/other.py && \
> $PYREFLY check -c $TMPDIR/default_ignore/pyrefly.toml --output-format=min-text $TMPDIR/default_ignore/a/main.py
 INFO 0 errors
[0]
```

## `[missing-import]` suppression does not silence `implicit-relative-import`

The two kinds are independent (no `parent_kind`), so ignoring
`missing-import` has no effect on `implicit-relative-import`.

```scrut
$ mkdir -p $TMPDIR/independent/a $TMPDIR/independent/b && \
> printf 'enable-fallback-search-path = true\n[errors]\nimplicit-relative-import = "error"\n' > $TMPDIR/independent/pyrefly.toml && \
> echo "x: int = 1" > $TMPDIR/independent/a/sibling.py && \
> printf '# pyrefly: ignore[missing-import]\nimport sibling\n' > $TMPDIR/independent/a/main.py && \
> echo "y: int = 2" > $TMPDIR/independent/b/other.py && \
> $PYREFLY check -c $TMPDIR/independent/pyrefly.toml --output-format=min-text $TMPDIR/independent/a/main.py
*/a/main.py:2:8-15: Module `sibling` was imported using an implicit relative import. Prefer an explicit relative import (`from . import sibling`) or add the module's root to the configured search path. [implicit-relative-import] (glob)
[1]
```

## `[implicit-relative-import]` suppression does silence it

```scrut {output_stream: stderr}
$ mkdir -p $TMPDIR/suppressed/a $TMPDIR/suppressed/b && \
> printf 'enable-fallback-search-path = true\n[errors]\nimplicit-relative-import = "error"\n' > $TMPDIR/suppressed/pyrefly.toml && \
> echo "x: int = 1" > $TMPDIR/suppressed/a/sibling.py && \
> printf '# pyrefly: ignore[implicit-relative-import]\nimport sibling\n' > $TMPDIR/suppressed/a/main.py && \
> echo "y: int = 2" > $TMPDIR/suppressed/b/other.py && \
> $PYREFLY check -c $TMPDIR/suppressed/pyrefly.toml --output-format=min-text $TMPDIR/suppressed/a/main.py
 INFO 0 errors (1 suppressed)
[0]
```
