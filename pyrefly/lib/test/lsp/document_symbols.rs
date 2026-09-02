/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pretty_assertions::assert_eq;
use pyrefly_build::handle::Handle;

use crate::lsp::non_wasm::document_symbols::flatten_to_symbol_information;
use crate::state::state::State;
use crate::test::util::get_batched_lsp_operations_report_no_cursor;
use crate::test::util::get_batched_lsp_operations_report_no_cursor_allow_error;

fn get_combined_report(state: &State, handle: &Handle) -> String {
    let hierarchical = get_hierarchical_symbol_report(state, handle);
    let flat = get_flat_symbol_report(state, handle);
    format!("## Hierarchical\n{hierarchical}\n## Flat\n{flat}")
}

fn get_hierarchical_symbol_report(state: &State, handle: &Handle) -> String {
    let transaction = state.transaction();
    if let Some(symbols) = transaction.symbols(handle, None) {
        serde_json::to_string_pretty(&symbols).unwrap()
    } else {
        "No document symbols found".to_owned()
    }
}

fn get_flat_symbol_report(state: &State, handle: &Handle) -> String {
    let transactions = state.transaction();
    let uri = lsp_types::Url::parse("file:///main.py").unwrap();
    if let Some(symbols) = transactions.symbols(handle, None) {
        let flat = flatten_to_symbol_information(symbols, &uri);
        serde_json::to_string_pretty(&flat).unwrap()
    } else {
        "No document symbols found".to_owned()
    }
}

fn extract_section<'a>(report: &'a str, section: &str) -> &'a str {
    let marker = format!("## {section}");
    let start = report
        .find(&marker)
        .expect("missing section marker in combined symbol report")
        + marker.len();
    let rest = &report[start..];
    let end = rest.find("\n## ").map(|i| i + 1).unwrap_or(rest.len());
    rest[..end].trim()
}

#[test]
fn function_test() {
    let code = r#"
def function1():
    """Test docstring"""
    x = 1
    return x

def function2(param1, param2):
    y = param1 + param2
    return y
"#;
    let report =
        get_batched_lsp_operations_report_no_cursor(&[("main", code)], get_combined_report);
    assert_eq!(
        r#"# main.py

## Hierarchical
[
  {
    "name": "function1",
    "kind": 12,
    "range": {
      "start": {
        "line": 1,
        "character": 0
      },
      "end": {
        "line": 4,
        "character": 12
      }
    },
    "selectionRange": {
      "start": {
        "line": 1,
        "character": 4
      },
      "end": {
        "line": 1,
        "character": 13
      }
    },
    "children": [
      {
        "name": "x",
        "kind": 13,
        "range": {
          "start": {
            "line": 3,
            "character": 4
          },
          "end": {
            "line": 3,
            "character": 9
          }
        },
        "selectionRange": {
          "start": {
            "line": 3,
            "character": 4
          },
          "end": {
            "line": 3,
            "character": 5
          }
        }
      }
    ]
  },
  {
    "name": "function2",
    "kind": 12,
    "range": {
      "start": {
        "line": 6,
        "character": 0
      },
      "end": {
        "line": 8,
        "character": 12
      }
    },
    "selectionRange": {
      "start": {
        "line": 6,
        "character": 4
      },
      "end": {
        "line": 6,
        "character": 13
      }
    },
    "children": [
      {
        "name": "y",
        "kind": 13,
        "range": {
          "start": {
            "line": 7,
            "character": 4
          },
          "end": {
            "line": 7,
            "character": 23
          }
        },
        "selectionRange": {
          "start": {
            "line": 7,
            "character": 4
          },
          "end": {
            "line": 7,
            "character": 5
          }
        }
      }
    ]
  }
]
## Flat
[
  {
    "name": "function1",
    "kind": 12,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 1,
          "character": 0
        },
        "end": {
          "line": 4,
          "character": 12
        }
      }
    }
  },
  {
    "name": "x",
    "kind": 13,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 3,
          "character": 4
        },
        "end": {
          "line": 3,
          "character": 9
        }
      }
    },
    "containerName": "function1"
  },
  {
    "name": "function2",
    "kind": 12,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 6,
          "character": 0
        },
        "end": {
          "line": 8,
          "character": 12
        }
      }
    }
  },
  {
    "name": "y",
    "kind": 13,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 7,
          "character": 4
        },
        "end": {
          "line": 7,
          "character": 23
        }
      }
    },
    "containerName": "function2"
  }
]"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn class_test() {
    let code = r#"
class MyClass:
    """Class docstring"""

    def __init__(self):
        self.x = 1

    def method1(self):
        return self.x

    def method2(self, y):
        return self.x + y
"#;
    let report =
        get_batched_lsp_operations_report_no_cursor(&[("main", code)], get_combined_report);
    assert_eq!(
        r#"# main.py

## Hierarchical
[
  {
    "name": "MyClass",
    "kind": 5,
    "range": {
      "start": {
        "line": 1,
        "character": 0
      },
      "end": {
        "line": 11,
        "character": 25
      }
    },
    "selectionRange": {
      "start": {
        "line": 1,
        "character": 6
      },
      "end": {
        "line": 1,
        "character": 13
      }
    },
    "children": [
      {
        "name": "__init__",
        "kind": 6,
        "range": {
          "start": {
            "line": 4,
            "character": 4
          },
          "end": {
            "line": 5,
            "character": 18
          }
        },
        "selectionRange": {
          "start": {
            "line": 4,
            "character": 8
          },
          "end": {
            "line": 4,
            "character": 16
          }
        },
        "children": []
      },
      {
        "name": "method1",
        "kind": 6,
        "range": {
          "start": {
            "line": 7,
            "character": 4
          },
          "end": {
            "line": 8,
            "character": 21
          }
        },
        "selectionRange": {
          "start": {
            "line": 7,
            "character": 8
          },
          "end": {
            "line": 7,
            "character": 15
          }
        },
        "children": []
      },
      {
        "name": "method2",
        "kind": 6,
        "range": {
          "start": {
            "line": 10,
            "character": 4
          },
          "end": {
            "line": 11,
            "character": 25
          }
        },
        "selectionRange": {
          "start": {
            "line": 10,
            "character": 8
          },
          "end": {
            "line": 10,
            "character": 15
          }
        },
        "children": []
      }
    ]
  }
]
## Flat
[
  {
    "name": "MyClass",
    "kind": 5,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 1,
          "character": 0
        },
        "end": {
          "line": 11,
          "character": 25
        }
      }
    }
  },
  {
    "name": "__init__",
    "kind": 6,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 4,
          "character": 4
        },
        "end": {
          "line": 5,
          "character": 18
        }
      }
    },
    "containerName": "MyClass"
  },
  {
    "name": "method1",
    "kind": 6,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 7,
          "character": 4
        },
        "end": {
          "line": 8,
          "character": 21
        }
      }
    },
    "containerName": "MyClass"
  },
  {
    "name": "method2",
    "kind": 6,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 10,
          "character": 4
        },
        "end": {
          "line": 11,
          "character": 25
        }
      }
    },
    "containerName": "MyClass"
  }
]"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn nested_class_test() {
    let code = r#"
while True:
    class Foo: pass
    print(str(Foo))
"#;
    let report =
        get_batched_lsp_operations_report_no_cursor(&[("main", code)], get_combined_report);
    assert_eq!(
        r#"
# main.py

## Hierarchical
[
  {
    "name": "Foo",
    "kind": 5,
    "range": {
      "start": {
        "line": 2,
        "character": 4
      },
      "end": {
        "line": 2,
        "character": 19
      }
    },
    "selectionRange": {
      "start": {
        "line": 2,
        "character": 10
      },
      "end": {
        "line": 2,
        "character": 13
      }
    },
    "children": []
  }
]
## Flat
[
  {
    "name": "Foo",
    "kind": 5,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 2,
          "character": 4
        },
        "end": {
          "line": 2,
          "character": 19
        }
      }
    }
  }
]"#
        .trim(),
        report.trim(),
    );
}

/// Syntax errors like `x = = None` cause the parser to produce assignment
/// targets with empty names. We must skip these to avoid returning
/// `DocumentSymbol { name: "" }`, which violates the LSP spec and causes
/// "name must not be falsy" errors in VS Code.
#[test]
fn test_syntax_error_empty_name_assign() {
    let code = r#"
def foo():
    x = = None
"#;
    let report = get_batched_lsp_operations_report_no_cursor_allow_error(
        &[("main", code)],
        get_combined_report,
    );

    // --- Hierarchical ---
    let hierarchical_json = extract_section(&report, "Hierarchical");
    let symbols: Vec<lsp_types::DocumentSymbol> = serde_json::from_str(hierarchical_json).unwrap();

    assert_eq!(symbols.len(), 1);
    assert_eq!(symbols[0].name, "foo");

    let children = symbols[0].children.as_ref().unwrap();
    // Only the valid "x" assignment should appear; the empty-name target
    // from the parser's error recovery must be filtered out.
    assert_eq!(children.len(), 1);
    assert_eq!(children[0].name, "x");

    // ---- Flat ----
    let flat_json = extract_section(&report, "Flat");
    let flat_symbols: Vec<lsp_types::SymbolInformation> = serde_json::from_str(flat_json).unwrap();

    assert_eq!(flat_symbols.len(), 2);
    assert_eq!(flat_symbols[0].name, "foo");
    assert_eq!(flat_symbols[0].container_name, None);
    assert_eq!(flat_symbols[1].name, "x");
    assert_eq!(flat_symbols[1].container_name, Some("foo".to_owned()));
}

// TODO(kylei): list comprehension document symbol
#[test]
fn list_comprehension_test() {
    let code = r#"
[x for x in list()]
"#;
    let report =
        get_batched_lsp_operations_report_no_cursor(&[("main", code)], get_combined_report);
    assert_eq!(
        r#"
# main.py

## Hierarchical
[]
## Flat
[]
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn test_does_include_local_variables_as_symbols() {
    let code = r#"
import os
from typing import List

x = 1

def helper_function():
    return 42

class MyClass:
    class_var = "hello"

    def method(self):
        local_var = helper_function()
        return local_var

y = MyClass()
result = y.method()
 "#;
    let report =
        get_batched_lsp_operations_report_no_cursor(&[("main", code)], get_combined_report);
    assert_eq!(
        r#"
# main.py

## Hierarchical
[
  {
    "name": "x",
    "kind": 13,
    "range": {
      "start": {
        "line": 4,
        "character": 0
      },
      "end": {
        "line": 4,
        "character": 5
      }
    },
    "selectionRange": {
      "start": {
        "line": 4,
        "character": 0
      },
      "end": {
        "line": 4,
        "character": 1
      }
    }
  },
  {
    "name": "helper_function",
    "kind": 12,
    "range": {
      "start": {
        "line": 6,
        "character": 0
      },
      "end": {
        "line": 7,
        "character": 13
      }
    },
    "selectionRange": {
      "start": {
        "line": 6,
        "character": 4
      },
      "end": {
        "line": 6,
        "character": 19
      }
    },
    "children": []
  },
  {
    "name": "MyClass",
    "kind": 5,
    "range": {
      "start": {
        "line": 9,
        "character": 0
      },
      "end": {
        "line": 14,
        "character": 24
      }
    },
    "selectionRange": {
      "start": {
        "line": 9,
        "character": 6
      },
      "end": {
        "line": 9,
        "character": 13
      }
    },
    "children": [
      {
        "name": "class_var",
        "kind": 13,
        "range": {
          "start": {
            "line": 10,
            "character": 4
          },
          "end": {
            "line": 10,
            "character": 23
          }
        },
        "selectionRange": {
          "start": {
            "line": 10,
            "character": 4
          },
          "end": {
            "line": 10,
            "character": 13
          }
        }
      },
      {
        "name": "method",
        "kind": 6,
        "range": {
          "start": {
            "line": 12,
            "character": 4
          },
          "end": {
            "line": 14,
            "character": 24
          }
        },
        "selectionRange": {
          "start": {
            "line": 12,
            "character": 8
          },
          "end": {
            "line": 12,
            "character": 14
          }
        },
        "children": [
          {
            "name": "local_var",
            "kind": 13,
            "range": {
              "start": {
                "line": 13,
                "character": 8
              },
              "end": {
                "line": 13,
                "character": 37
              }
            },
            "selectionRange": {
              "start": {
                "line": 13,
                "character": 8
              },
              "end": {
                "line": 13,
                "character": 17
              }
            }
          }
        ]
      }
    ]
  },
  {
    "name": "y",
    "kind": 13,
    "range": {
      "start": {
        "line": 16,
        "character": 0
      },
      "end": {
        "line": 16,
        "character": 13
      }
    },
    "selectionRange": {
      "start": {
        "line": 16,
        "character": 0
      },
      "end": {
        "line": 16,
        "character": 1
      }
    }
  },
  {
    "name": "result",
    "kind": 13,
    "range": {
      "start": {
        "line": 17,
        "character": 0
      },
      "end": {
        "line": 17,
        "character": 19
      }
    },
    "selectionRange": {
      "start": {
        "line": 17,
        "character": 0
      },
      "end": {
        "line": 17,
        "character": 6
      }
    }
  }
]
## Flat
[
  {
    "name": "x",
    "kind": 13,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 4,
          "character": 0
        },
        "end": {
          "line": 4,
          "character": 5
        }
      }
    }
  },
  {
    "name": "helper_function",
    "kind": 12,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 6,
          "character": 0
        },
        "end": {
          "line": 7,
          "character": 13
        }
      }
    }
  },
  {
    "name": "MyClass",
    "kind": 5,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 9,
          "character": 0
        },
        "end": {
          "line": 14,
          "character": 24
        }
      }
    }
  },
  {
    "name": "class_var",
    "kind": 13,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 10,
          "character": 4
        },
        "end": {
          "line": 10,
          "character": 23
        }
      }
    },
    "containerName": "MyClass"
  },
  {
    "name": "method",
    "kind": 6,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 12,
          "character": 4
        },
        "end": {
          "line": 14,
          "character": 24
        }
      }
    },
    "containerName": "MyClass"
  },
  {
    "name": "local_var",
    "kind": 13,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 13,
          "character": 8
        },
        "end": {
          "line": 13,
          "character": 37
        }
      }
    },
    "containerName": "MyClass.method"
  },
  {
    "name": "y",
    "kind": 13,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 16,
          "character": 0
        },
        "end": {
          "line": 16,
          "character": 13
        }
      }
    }
  },
  {
    "name": "result",
    "kind": 13,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 17,
          "character": 0
        },
        "end": {
          "line": 17,
          "character": 19
        }
      }
    }
  }
]"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn test_does_include_annotated_local_variables_as_symbols() {
    let code = r#"
import os
from typing import List

x: int = 1
name: str = "test"

def helper_function() -> int:
    return 42

class MyClass:
    class_var: str = "hello"
    counter: int = 0

    def method(self) -> int:
        local_var: int = helper_function()
        message: str = "done"
        return local_var

y: MyClass = MyClass()
result: int = y.method()
items: List[str] = ["a", "b", "c"]
 "#;
    let report =
        get_batched_lsp_operations_report_no_cursor(&[("main", code)], get_combined_report);
    assert_eq!(
        r#"
# main.py

## Hierarchical
[
  {
    "name": "x",
    "detail": "int",
    "kind": 13,
    "range": {
      "start": {
        "line": 4,
        "character": 0
      },
      "end": {
        "line": 4,
        "character": 10
      }
    },
    "selectionRange": {
      "start": {
        "line": 4,
        "character": 0
      },
      "end": {
        "line": 4,
        "character": 1
      }
    }
  },
  {
    "name": "name",
    "detail": "str",
    "kind": 13,
    "range": {
      "start": {
        "line": 5,
        "character": 0
      },
      "end": {
        "line": 5,
        "character": 18
      }
    },
    "selectionRange": {
      "start": {
        "line": 5,
        "character": 0
      },
      "end": {
        "line": 5,
        "character": 4
      }
    }
  },
  {
    "name": "helper_function",
    "kind": 12,
    "range": {
      "start": {
        "line": 7,
        "character": 0
      },
      "end": {
        "line": 8,
        "character": 13
      }
    },
    "selectionRange": {
      "start": {
        "line": 7,
        "character": 4
      },
      "end": {
        "line": 7,
        "character": 19
      }
    },
    "children": []
  },
  {
    "name": "MyClass",
    "kind": 5,
    "range": {
      "start": {
        "line": 10,
        "character": 0
      },
      "end": {
        "line": 17,
        "character": 24
      }
    },
    "selectionRange": {
      "start": {
        "line": 10,
        "character": 6
      },
      "end": {
        "line": 10,
        "character": 13
      }
    },
    "children": [
      {
        "name": "class_var",
        "detail": "str",
        "kind": 13,
        "range": {
          "start": {
            "line": 11,
            "character": 4
          },
          "end": {
            "line": 11,
            "character": 28
          }
        },
        "selectionRange": {
          "start": {
            "line": 11,
            "character": 4
          },
          "end": {
            "line": 11,
            "character": 13
          }
        }
      },
      {
        "name": "counter",
        "detail": "int",
        "kind": 13,
        "range": {
          "start": {
            "line": 12,
            "character": 4
          },
          "end": {
            "line": 12,
            "character": 20
          }
        },
        "selectionRange": {
          "start": {
            "line": 12,
            "character": 4
          },
          "end": {
            "line": 12,
            "character": 11
          }
        }
      },
      {
        "name": "method",
        "kind": 6,
        "range": {
          "start": {
            "line": 14,
            "character": 4
          },
          "end": {
            "line": 17,
            "character": 24
          }
        },
        "selectionRange": {
          "start": {
            "line": 14,
            "character": 8
          },
          "end": {
            "line": 14,
            "character": 14
          }
        },
        "children": [
          {
            "name": "local_var",
            "detail": "int",
            "kind": 13,
            "range": {
              "start": {
                "line": 15,
                "character": 8
              },
              "end": {
                "line": 15,
                "character": 42
              }
            },
            "selectionRange": {
              "start": {
                "line": 15,
                "character": 8
              },
              "end": {
                "line": 15,
                "character": 17
              }
            }
          },
          {
            "name": "message",
            "detail": "str",
            "kind": 13,
            "range": {
              "start": {
                "line": 16,
                "character": 8
              },
              "end": {
                "line": 16,
                "character": 29
              }
            },
            "selectionRange": {
              "start": {
                "line": 16,
                "character": 8
              },
              "end": {
                "line": 16,
                "character": 15
              }
            }
          }
        ]
      }
    ]
  },
  {
    "name": "y",
    "detail": "MyClass",
    "kind": 13,
    "range": {
      "start": {
        "line": 19,
        "character": 0
      },
      "end": {
        "line": 19,
        "character": 22
      }
    },
    "selectionRange": {
      "start": {
        "line": 19,
        "character": 0
      },
      "end": {
        "line": 19,
        "character": 1
      }
    }
  },
  {
    "name": "result",
    "detail": "int",
    "kind": 13,
    "range": {
      "start": {
        "line": 20,
        "character": 0
      },
      "end": {
        "line": 20,
        "character": 24
      }
    },
    "selectionRange": {
      "start": {
        "line": 20,
        "character": 0
      },
      "end": {
        "line": 20,
        "character": 6
      }
    }
  },
  {
    "name": "items",
    "detail": "List[str]",
    "kind": 13,
    "range": {
      "start": {
        "line": 21,
        "character": 0
      },
      "end": {
        "line": 21,
        "character": 34
      }
    },
    "selectionRange": {
      "start": {
        "line": 21,
        "character": 0
      },
      "end": {
        "line": 21,
        "character": 5
      }
    }
  }
]
## Flat
[
  {
    "name": "x",
    "kind": 13,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 4,
          "character": 0
        },
        "end": {
          "line": 4,
          "character": 10
        }
      }
    }
  },
  {
    "name": "name",
    "kind": 13,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 5,
          "character": 0
        },
        "end": {
          "line": 5,
          "character": 18
        }
      }
    }
  },
  {
    "name": "helper_function",
    "kind": 12,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 7,
          "character": 0
        },
        "end": {
          "line": 8,
          "character": 13
        }
      }
    }
  },
  {
    "name": "MyClass",
    "kind": 5,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 10,
          "character": 0
        },
        "end": {
          "line": 17,
          "character": 24
        }
      }
    }
  },
  {
    "name": "class_var",
    "kind": 13,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 11,
          "character": 4
        },
        "end": {
          "line": 11,
          "character": 28
        }
      }
    },
    "containerName": "MyClass"
  },
  {
    "name": "counter",
    "kind": 13,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 12,
          "character": 4
        },
        "end": {
          "line": 12,
          "character": 20
        }
      }
    },
    "containerName": "MyClass"
  },
  {
    "name": "method",
    "kind": 6,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 14,
          "character": 4
        },
        "end": {
          "line": 17,
          "character": 24
        }
      }
    },
    "containerName": "MyClass"
  },
  {
    "name": "local_var",
    "kind": 13,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 15,
          "character": 8
        },
        "end": {
          "line": 15,
          "character": 42
        }
      }
    },
    "containerName": "MyClass.method"
  },
  {
    "name": "message",
    "kind": 13,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 16,
          "character": 8
        },
        "end": {
          "line": 16,
          "character": 29
        }
      }
    },
    "containerName": "MyClass.method"
  },
  {
    "name": "y",
    "kind": 13,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 19,
          "character": 0
        },
        "end": {
          "line": 19,
          "character": 22
        }
      }
    }
  },
  {
    "name": "result",
    "kind": 13,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 20,
          "character": 0
        },
        "end": {
          "line": 20,
          "character": 24
        }
      }
    }
  },
  {
    "name": "items",
    "kind": 13,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 21,
          "character": 0
        },
        "end": {
          "line": 21,
          "character": 34
        }
      }
    }
  }
]"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn test_comment_sections_in_symbols() {
    let code = r#"
# Section 1 ----

x = 1

## Section 1.1 ----

def foo():
    pass

# Section 2 ----

class MyClass:
    pass
"#;
    let report =
        get_batched_lsp_operations_report_no_cursor(&[("main", code)], get_combined_report);
    assert_eq!(
        r#"
# main.py

## Hierarchical
[
  {
    "name": "Section 1",
    "kind": 15,
    "range": {
      "start": {
        "line": 1,
        "character": 0
      },
      "end": {
        "line": 1,
        "character": 16
      }
    },
    "selectionRange": {
      "start": {
        "line": 1,
        "character": 0
      },
      "end": {
        "line": 1,
        "character": 16
      }
    },
    "children": [
      {
        "name": "x",
        "kind": 13,
        "range": {
          "start": {
            "line": 3,
            "character": 0
          },
          "end": {
            "line": 3,
            "character": 5
          }
        },
        "selectionRange": {
          "start": {
            "line": 3,
            "character": 0
          },
          "end": {
            "line": 3,
            "character": 1
          }
        }
      },
      {
        "name": "Section 1.1",
        "kind": 15,
        "range": {
          "start": {
            "line": 5,
            "character": 0
          },
          "end": {
            "line": 5,
            "character": 19
          }
        },
        "selectionRange": {
          "start": {
            "line": 5,
            "character": 0
          },
          "end": {
            "line": 5,
            "character": 19
          }
        },
        "children": [
          {
            "name": "foo",
            "kind": 12,
            "range": {
              "start": {
                "line": 7,
                "character": 0
              },
              "end": {
                "line": 8,
                "character": 8
              }
            },
            "selectionRange": {
              "start": {
                "line": 7,
                "character": 4
              },
              "end": {
                "line": 7,
                "character": 7
              }
            },
            "children": []
          }
        ]
      }
    ]
  },
  {
    "name": "Section 2",
    "kind": 15,
    "range": {
      "start": {
        "line": 10,
        "character": 0
      },
      "end": {
        "line": 10,
        "character": 16
      }
    },
    "selectionRange": {
      "start": {
        "line": 10,
        "character": 0
      },
      "end": {
        "line": 10,
        "character": 16
      }
    },
    "children": [
      {
        "name": "MyClass",
        "kind": 5,
        "range": {
          "start": {
            "line": 12,
            "character": 0
          },
          "end": {
            "line": 13,
            "character": 8
          }
        },
        "selectionRange": {
          "start": {
            "line": 12,
            "character": 6
          },
          "end": {
            "line": 12,
            "character": 13
          }
        },
        "children": []
      }
    ]
  }
]
## Flat
[
  {
    "name": "Section 1",
    "kind": 15,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 1,
          "character": 0
        },
        "end": {
          "line": 1,
          "character": 16
        }
      }
    }
  },
  {
    "name": "x",
    "kind": 13,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 3,
          "character": 0
        },
        "end": {
          "line": 3,
          "character": 5
        }
      }
    },
    "containerName": "Section 1"
  },
  {
    "name": "Section 1.1",
    "kind": 15,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 5,
          "character": 0
        },
        "end": {
          "line": 5,
          "character": 19
        }
      }
    },
    "containerName": "Section 1"
  },
  {
    "name": "foo",
    "kind": 12,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 7,
          "character": 0
        },
        "end": {
          "line": 8,
          "character": 8
        }
      }
    },
    "containerName": "Section 1.Section 1.1"
  },
  {
    "name": "Section 2",
    "kind": 15,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 10,
          "character": 0
        },
        "end": {
          "line": 10,
          "character": 16
        }
      }
    }
  },
  {
    "name": "MyClass",
    "kind": 5,
    "location": {
      "uri": "file:///main.py",
      "range": {
        "start": {
          "line": 12,
          "character": 0
        },
        "end": {
          "line": 13,
          "character": 8
        }
      }
    },
    "containerName": "Section 2"
  }
]"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn test_comment_sections_with_ast_children() {
    let code = r#"
# Imports ----
import os

## Standard Library ----
from pathlib import Path
a = 1

def greeting():
    print("hello, world")

## Another second level ----

# Configuration ====
DEBUG = True

b = 2
"#;
    let report =
        get_batched_lsp_operations_report_no_cursor(&[("main", code)], get_combined_report);

    // Verify the structure:
    // - "Imports" should contain:
    //   - "Standard Library" (subsection)
    //     - a (variable)
    //     - greeting (function)
    //   - "Another second level" (subsection)
    // - "Configuration" should contain:
    //   - DEBUG (variable)
    //   - b (variable)

    // --- Hierarchical ---
    let hierarchical_json = extract_section(&report, "Hierarchical");
    let symbols: Vec<lsp_types::DocumentSymbol> = serde_json::from_str(hierarchical_json).unwrap();

    // Check top-level sections
    assert_eq!(symbols.len(), 2);
    assert_eq!(symbols[0].name, "Imports");
    assert_eq!(symbols[1].name, "Configuration");

    // Check "Imports" children
    let imports_children = symbols[0].children.as_ref().unwrap();
    assert_eq!(imports_children.len(), 2);
    assert_eq!(imports_children[0].name, "Standard Library");
    assert_eq!(imports_children[1].name, "Another second level");

    // Check "Standard Library" children
    let std_lib_children = imports_children[0].children.as_ref().unwrap();
    assert_eq!(std_lib_children.len(), 2);
    assert_eq!(std_lib_children[0].name, "a");
    assert_eq!(std_lib_children[1].name, "greeting");

    // Check "Configuration" children
    let config_children = symbols[1].children.as_ref().unwrap();
    assert_eq!(config_children.len(), 2);
    assert_eq!(config_children[0].name, "DEBUG");
    assert_eq!(config_children[1].name, "b");

    // --- Flat ---
    let flat_json = extract_section(&report, "Flat");
    let flat_symbols: Vec<lsp_types::SymbolInformation> = serde_json::from_str(flat_json).unwrap();

    assert_eq!(flat_symbols.len(), 8);
    assert_eq!(flat_symbols[0].name, "Imports");
    assert_eq!(flat_symbols[0].container_name, None);

    assert_eq!(flat_symbols[1].name, "Standard Library");
    assert_eq!(flat_symbols[1].container_name, Some("Imports".to_owned()));

    assert_eq!(flat_symbols[2].name, "a");
    assert_eq!(
        flat_symbols[2].container_name,
        Some("Imports.Standard Library".to_owned())
    );

    assert_eq!(flat_symbols[3].name, "greeting");
    assert_eq!(
        flat_symbols[3].container_name,
        Some("Imports.Standard Library".to_owned())
    );

    assert_eq!(flat_symbols[4].name, "Another second level");
    assert_eq!(flat_symbols[4].container_name, Some("Imports".to_owned()));

    assert_eq!(flat_symbols[5].name, "Configuration");
    assert_eq!(flat_symbols[5].container_name, None);

    assert_eq!(flat_symbols[6].name, "DEBUG");
    assert_eq!(
        flat_symbols[6].container_name,
        Some("Configuration".to_owned())
    );

    assert_eq!(flat_symbols[7].name, "b");
    assert_eq!(
        flat_symbols[7].container_name,
        Some("Configuration".to_owned())
    );
}
