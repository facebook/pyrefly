# Tests for Jupyter Notebooks

## Notebook Invalid

```scrut
$ echo -e "x: int = 1" > $TMPDIR/notebook.ipynb && \
> $PYREFLY check $TMPDIR/notebook.ipynb
ERROR * Expected a Jupyter Notebook, which must be internally stored as JSON, but this file isn't valid JSON* (glob)
--> */notebook.ipynb:1:1 (glob)
 |
 |
[1]
```

## Notebook Invalid JSON

```scrut
$ echo -e "{}" > $TMPDIR/notebook.ipynb && \
> $PYREFLY check $TMPDIR/notebook.ipynb
ERROR * This file does not match the schema expected of Jupyter Notebooks: missing field `cells`* (glob)
--> */notebook.ipynb:1:1 (glob)
 |
 |
[1]
```

## Notebook Top Level Await

```scrut
$ echo -e '{"cells":[{"cell_type":"code","execution_count":null,"metadata":{},"outputs":[],"source":["import asyncio\\nawait asyncio.sleep(1)"]}],"metadata":{"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":4}' > $TMPDIR/notebook.ipynb && \
> $PYREFLY check $TMPDIR/notebook.ipynb
[0]
```

## Notebook Error

```scrut
$ echo -e '{"cells":[{"cell_type":"code","execution_count":null,"metadata":{},"outputs":[],"source":["x: bool = 5"]}],"metadata":{"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":4}' > $TMPDIR/notebook.ipynb && \
> $PYREFLY check $TMPDIR/notebook.ipynb
ERROR `Literal[5]` is not assignable to `bool` [bad-assignment]
 --> */notebook.ipynb#1:1:11 (glob)
  |
1 | x: bool = 5
  |           ^
  |
[1]
```

## Notebook Error Second Cell

```scrut
$ echo -e '{"cells":[{"cell_type":"code","execution_count":null,"metadata":{},"outputs":[],"source":["x: bool = True"]},{"cell_type":"code","execution_count":null,"metadata":{},"outputs":[],"source":["x: bool = 5"]}],"metadata":{"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":4}' > $TMPDIR/notebook.ipynb && \
> $PYREFLY check $TMPDIR/notebook.ipynb
ERROR `Literal[5]` is not assignable to `bool` [bad-assignment]
 --> */notebook.ipynb#2:1:11 (glob)
  |
1 | x: bool = 5
  |           ^
  |
[1]
```

## Notebook Suppressed Error

```scrut
$ echo -e '{"cells":[{"cell_type":"code","execution_count":null,"metadata":{},"outputs":[],"source":["# type: ignore\\nx: bool = 5"]}],"metadata":{"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":4}' > $TMPDIR/notebook.ipynb && \
> $PYREFLY check $TMPDIR/notebook.ipynb
[0]
```
