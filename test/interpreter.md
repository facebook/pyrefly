# Tests for interpreter selection

## Active Conda environment takes priority over an ancestor venv

The active environment uses the Pixi layout without a `pyvenv.cfg`. Its interpreter
must be found through `CONDA_PREFIX`, even when it is not on `PATH`.

```scrut {output_stream: stdout}
$ mkdir -p "$TMPDIR/conda-priority/project/.pixi/envs/default/bin" "$TMPDIR/conda-priority/venv/bin" && \
> touch "$TMPDIR/conda-priority/project/pyrefly.toml" "$TMPDIR/conda-priority/project/test.py" && \
> touch "$TMPDIR/conda-priority/venv/pyvenv.cfg" "$TMPDIR/conda-priority/venv/bin/python" && \
> ln -s "$(python3 -c 'import sys; print(sys.executable)')" "$TMPDIR/conda-priority/project/.pixi/envs/default/bin/python" && \
> env -u VIRTUAL_ENV CONDA_PREFIX="$TMPDIR/conda-priority/project/.pixi/envs/default" \
> "$PYREFLY" dump-config -c "$TMPDIR/conda-priority/project/pyrefly.toml"
Configuration at * (glob)
  Using interpreter: */conda-priority/project/.pixi/envs/default/bin/python (glob)
* (glob+)
[0]
```

## Interpreter priority takes CLI interpreter

```scrut {output_stream: stdout}
$ mkdir $TMPDIR/interpreters && touch $TMPDIR/interpreters/test.py \
> touch $TMPDIR/test-interpreter && \
> echo 'python-interpreter = "$TMPDIR/test-interpreter"' > $TMPDIR/interpreters/pyrefly.toml && \
> mkdir -p $TMPDIR/interpreters/venv/bin && touch $TMPDIR/interpreters/venv/bin/python && \
> touch $TMPDIR/interpreters/venv/pyvenv.cfg && \
> mkdir -p $TMPDIR/alternative-venv/bin && touch $TMPDIR/alternative-venv/bin/python && \
> touch $TMPDIR/alternative-venv/pyvenv.cfg && \
> VIRTUAL_ENV=$TMPDIR/alternative-venv $PYREFLY dump-config -c $TMPDIR/interpreters/pyrefly.toml \
> --python-interpreter-path "cli-interpreter"
Configuration at * (glob)
  Using interpreter: cli-interpreter
* (glob+)
[0]
```

## Interpreter priority takes config-file interpreter

<!-- Reusing interpreters dir set up in "Interpreter priority takes CLI interpreter" -->

```scrut {output_stream: stdout}
$ VIRTUAL_ENV=$TMPDIR/alternative-venv $PYREFLY dump-config -c $TMPDIR/interpreters/pyrefly.toml
Configuration at * (glob)
  Using interpreter: */test-interpreter (glob)
* (glob+)
[0]
```

## Interpreter priority takes activated interpreter

<!-- Reusing interpreters dir set up in "Interpreter priority takes CLI interpreter" -->

```scrut {output_stream: stdout}
$ echo "" > $TMPDIR/interpreters/pyrefly.toml && \
> VIRTUAL_ENV=$TMPDIR/alternative-venv $PYREFLY dump-config -c $TMPDIR/interpreters/pyrefly.toml
Configuration at * (glob)
  Using interpreter: */alternative-venv/bin/python (glob)
* (glob+)
[0]
```

## Interpreter priority takes venv interpreter

<!-- Reusing interpreters dir set up in "Interpreter priority takes CLI interpreter" -->

```scrut {output_stream: stdout}
$ echo "" > $TMPDIR/interpreters/pyrefly.toml && \
> $PYREFLY dump-config -c $TMPDIR/interpreters/pyrefly.toml
Configuration at * (glob)
  Using interpreter: */interpreters/venv/bin/python (glob)
* (glob+)
[0]
```

## Interpreter priority takes system interpreter last

<!-- Reusing interpreters dir set up in "Interpreter priority takes CLI interpreter" -->

```scrut {output_stream: stdout}
$ rm -rf $TMPDIR/interpreters/venv && \
> $PYREFLY dump-config -c $TMPDIR/interpreters/pyrefly.toml
Configuration at * (glob)
  Using interpreter: /*/python3 (glob)
* (glob+)
[0]
```

## We can find a venv interpreter, even when not sourced

```scrut {output_stream: stderr}
$ touch $TMPDIR/pyrefly.toml && \
> python3 -m venv $TMPDIR/venv 2>$TMPDIR/venv_stderr || \
> { cat $TMPDIR/venv_stderr >&2; false; } && \
> echo "import third_party.test2" > $TMPDIR/test.py && \
> export site_packages=$($TMPDIR/venv/bin/python -c "import site; print(site.getsitepackages()[0])") && \
> mkdir $site_packages/third_party && \
> echo "x = 1" > $site_packages/third_party/test2.py && \
> $PYREFLY check $TMPDIR/test.py
 INFO 0 errors* (glob)
[0]
```

## We find a project venv without a config, including for imported files

```scrut {output_stream: stderr}
$ VENV_PROJECT=$(mktemp -d -p /tmp venv.XXXXXX) && \
> python3 -m venv $VENV_PROJECT/venv 2>$TMPDIR/venv_stderr || \
> { cat $TMPDIR/venv_stderr >&2; false; } && \
> site_packages=$($VENV_PROJECT/venv/bin/python -c "import site; print(site.getsitepackages()[0])") && \
> mkdir $site_packages/third_party && \
> printf 'def f() -> int:\n    return 1\n' > $site_packages/third_party/test2.py && \
> mkdir $VENV_PROJECT/pkg && touch $VENV_PROJECT/pkg/__init__.py && \
> echo "from third_party.test2 import f; g = f" > $VENV_PROJECT/pkg/mod.py && \
> echo "from pkg.mod import g; from typing import assert_type; assert_type(g(), int)" > $VENV_PROJECT/main.py && \
> $PYREFLY check $VENV_PROJECT/main.py; STATUS=$?; rm -rf $VENV_PROJECT; exit $STATUS
 INFO 0 errors* (glob)
No `pyrefly.toml` found — using preset `basic`.
Run `pyrefly init` to continue setting up Pyrefly.
Docs: https://pyrefly.org/en/docs/installation/
[0]
```

## We find an interpreter from an active Conda prefix

```scrut {output_stream: stderr}
$ CONDA_PROJECT=$(mktemp -d -p /tmp conda.XXXXXX) && \
> python3 -m venv $CONDA_PROJECT/conda 2>$TMPDIR/venv_stderr || \
> { cat $TMPDIR/venv_stderr >&2; false; } && \
> site_packages=$($CONDA_PROJECT/conda/bin/python -c "import site; print(site.getsitepackages()[0])") && \
> mkdir $site_packages/third_party && \
> echo "x = 1" > $site_packages/third_party/test2.py && \
> touch $CONDA_PROJECT/pyrefly.toml && \
> echo "import third_party.test2" > $CONDA_PROJECT/test.py && \
> env -u VIRTUAL_ENV CONDA_PREFIX=$CONDA_PROJECT/conda $PYREFLY check $CONDA_PROJECT/test.py; \
> STATUS=$?; rm -rf $CONDA_PROJECT; exit $STATUS
 INFO 0 errors* (glob)
[0]
```
