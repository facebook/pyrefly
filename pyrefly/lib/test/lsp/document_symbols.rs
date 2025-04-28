/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pretty_assertions::assert_eq;

use crate::state::handle::Handle;
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
    "children": []
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
    "children": []
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
        "children": []
      }
    ]
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
