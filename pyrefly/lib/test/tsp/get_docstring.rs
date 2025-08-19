/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Unit tests for get_docstring functionality
//!
//! These tests focus on testing the standalone get_docstring_at_position function
//! independently of the full TSP protocol. This allows for more targeted testing
//! of the core docstring extraction logic.

// Use protocol Position/Range instead of lsp_types

use tsp_types::Node;
use tsp_types::Position;
use tsp_types::Range;

use crate::test::util::mk_multi_file_state_assert_no_errors;
use crate::tsp::requests::get_docstring::get_docstring_at_position;

#[test]
fn test_get_docstring_at_position_with_function_docstring() {
    let (handles, state) = mk_multi_file_state_assert_no_errors(&[(
        "test.py",
        r#"def calculate_area(radius: float) -> float:
    """Calculate the area of a circle.
    
    Args:
        radius: The radius of the circle in units.
        
    Returns:
        The area of the circle in square units.
    """
    return 3.14159 * radius * radius
"#,
    )]);

    let transaction = state.transaction();
    let handle = handles.get("test.py").unwrap();

    // Create a node that points to the function name
    let node = Node {
        uri: "file:///test.py".to_string(),
        range: Range {
            start: Position {
                line: 0,
                character: 4,
            }, // Points to 'calculate_area'
            end: Position {
                line: 0,
                character: 18,
            },
        },
    };

    let result = get_docstring_at_position(&transaction, handle, &node);

    assert!(result.is_some());
    let docstring = result.unwrap();
    assert!(docstring.contains("Calculate the area of a circle."));
    assert!(docstring.contains("Args:"));
    assert!(docstring.contains("radius: The radius of the circle in units."));
    assert!(docstring.contains("Returns:"));
    assert!(docstring.contains("The area of the circle in square units."));
}

#[test]
fn test_get_docstring_at_position_with_class_docstring() {
    let (handles, state) = mk_multi_file_state_assert_no_errors(&[(
        "test.py",
        r#"class DataProcessor:
    """A class for processing and analyzing data.
    
    This class provides methods for loading, cleaning, and analyzing
    various types of data structures.
    
    Attributes:
        data: The raw data to be processed.
        cleaned_data: The processed and cleaned data.
    """
    
    def __init__(self):
        self.data = None
        self.cleaned_data = None
"#,
    )]);

    let transaction = state.transaction();
    let handle = handles.get("test.py").unwrap();

    // Create a node that points to the class name
    let node = Node {
        uri: "file:///test.py".to_string(),
        range: Range {
            start: Position {
                line: 0,
                character: 6,
            }, // Points to 'DataProcessor'
            end: Position {
                line: 0,
                character: 19,
            },
        },
    };

    let result = get_docstring_at_position(&transaction, handle, &node);

    assert!(result.is_some());
    let docstring = result.unwrap();
    assert!(docstring.contains("A class for processing and analyzing data."));
    assert!(docstring.contains("This class provides methods"));
    assert!(docstring.contains("Attributes:"));
    assert!(docstring.contains("data: The raw data to be processed."));
}

#[test]
fn test_get_docstring_at_position_no_docstring() {
    let (handles, state) = mk_multi_file_state_assert_no_errors(&[(
        "test.py",
        r#"def simple_function(x):
    return x * 2

class SimpleClass:
    def method(self):
        pass
"#,
    )]);

    let transaction = state.transaction();
    let handle = handles.get("test.py").unwrap();

    // Create a node that points to a function without docstring
    let node = Node {
        uri: "file:///test.py".to_string(),
        range: Range {
            start: Position {
                line: 0,
                character: 4,
            }, // Points to 'simple_function'
            end: Position {
                line: 0,
                character: 19,
            },
        },
    };

    let result = get_docstring_at_position(&transaction, handle, &node);

    // Should return None when no docstring is present
    assert!(result.is_none());
}

#[test]
fn test_get_docstring_at_position_method_docstring() {
    let (handles, state) = mk_multi_file_state_assert_no_errors(&[(
        "test.py",
        r#"class Calculator:
    def add(self, x: int, y: int) -> int:
        """Add two integers together.
        
        Args:
            x: First integer to add.
            y: Second integer to add.
            
        Returns:
            The sum of x and y.
        """
        return x + y
        
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b
"#,
    )]);

    let transaction = state.transaction();
    let handle = handles.get("test.py").unwrap();

    // Create a node that points to the 'add' method
    let node = Node {
        uri: "file:///test.py".to_string(),
        range: Range {
            start: Position {
                line: 1,
                character: 8,
            }, // Points to 'add' method
            end: Position {
                line: 1,
                character: 11,
            },
        },
    };

    let result = get_docstring_at_position(&transaction, handle, &node);

    assert!(result.is_some());
    let docstring = result.unwrap();
    assert!(docstring.contains("Add two integers together."));
    assert!(docstring.contains("Args:"));
    assert!(docstring.contains("x: First integer to add."));
    assert!(docstring.contains("y: Second integer to add."));
    assert!(docstring.contains("Returns:"));
    assert!(docstring.contains("The sum of x and y."));
}
