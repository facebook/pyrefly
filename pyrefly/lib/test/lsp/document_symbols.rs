/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pretty_assertions::assert_eq;
use pyrefly_build::handle::Handle;

use crate::state::state::State;
use crate::test::util::get_batched_lsp_operations_report_no_cursor;

fn get_test_report(state: &State, handle: &Handle) -> String {
    let transaction = state.transaction();
    if let Some(symbols) = transaction.symbols(handle) {
        serde_json::to_string_pretty(&symbols).unwrap()
    } else {
        "No document symbols found".to_owned()
    }
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
    let report = get_batched_lsp_operations_report_no_cursor(&[("main", code)], get_test_report);
    assert_eq!(
        r#"# main.py

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
    let report = get_batched_lsp_operations_report_no_cursor(&[("main", code)], get_test_report);
    assert_eq!(
        r#"# main.py

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
        "kind": 12,
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
        "kind": 12,
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
        "kind": 12,
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
]"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn mixed_content_test() {
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
    let report = get_batched_lsp_operations_report_no_cursor(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py

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
        "kind": 12,
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
"#
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
    let report = get_batched_lsp_operations_report_no_cursor(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py

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
]"#
        .trim(),
        report.trim(),
    );
}

// TODO(kylei): list comprehension document symbol
#[test]
fn list_comprehension_test() {
    let code = r#"
[x for x in list()]
"#;
    let report = get_batched_lsp_operations_report_no_cursor(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py

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
    let report = get_batched_lsp_operations_report_no_cursor(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py

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
        "kind": 12,
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
"#
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
    let report = get_batched_lsp_operations_report_no_cursor(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py

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
        "kind": 12,
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
"#
        .trim(),
        report.trim(),
    );
}
