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
