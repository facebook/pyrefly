# TSP Dependencies on Missing Pyrefly Functionality

This document tracks TSP functionality that is not complete or may be waiting on other parts of pyrefly to support specific features. These limitations represent areas where TSP capabilities are constrained.

## Core Type System

### 1. Argument Analysis & Type Inference
**Affected TSP Requests**: `getMatchingOverloads`
- **Current State**: Returns all overloads instead of filtering by argument types
- **Missing Feature**: Sophisticated argument matching at call sites. Not sure if this something base Pyrefly would do or if it should be implemented in the TSP logic
- **Impact**: Cannot provide precise overload suggestions based on actual arguments
- **Details**: Need ability to:
  - Parse call arguments from AST at call sites
  - Infer types of each argument
  - Match argument types against overload parameter signatures
- **File**: `get_matching_overloads.rs:66` - TODO comment for argument matching

### 2. Decorator Support
**Affected TSP Requests**: `getOverloads`, `getFunctionParts`, `getType`
- **Current State**: decorator support not fully implemented
- **Missing Feature**: Complete decorator analysis
- **Impact**: Cannot properly detect or analyze decorated functions/methods
- **Details**: 
  - Function overloads not properly detected
  - Method overloads not properly detected  
  - Overloaded function parts extraction not implemented
  - Pytest fixture decorator type not computable
- **Files**: 
  - `get_overloads.rs` tests marked as ignored
  - `get_function_parts.rs:169` - overloaded functions not implemented

## Symbol Resolution

### 3. Enhanced Symbol Definition Resolution
**Affected TSP Requests**: `getSymbol`, `resolveImportDeclaration`
- **Current State**: `find_definition` doesn't work for named parameters
- **Missing Feature**: Enhanced symbol resolution for:
  - Named function parameters
  - Import symbol resolution
  - Variable declarations in imports
- **Impact**: Cannot navigate to definitions of named parameters or properly resolve imports
- **Files**:
  - `get_symbol.rs:280` - TODO for named parameter definitions
  - `resolve_import_declaration.rs` - multiple TODOs for proper symbol resolution

## Advanced Features

### 4. Module Introspection & Export Analysis
**Affected TSP Requests**: `getTypeAttributes`
- **Current State**: Module attribute extraction not implemented
- **Missing Feature**: Module export symbol enumeration
- **Impact**: Cannot list exported symbols from module types
- **File**: `get_type_attributes.rs:282` - module attribute extraction TODO

### 5. Type Information Synthesis
**Affected TSP Requests**: `getSymbolsForFile`
- **Current State**: `synthesized_types` field is empty
- **Missing Feature**: Enhanced type information synthesis
- **Impact**: Missing inferred type information for symbols
- **Details**: Could include inferred type information for symbols
- **File**: `get_symbols_for_file.rs:123` - TODO for type information

## Last Updated
August 14, 2025
