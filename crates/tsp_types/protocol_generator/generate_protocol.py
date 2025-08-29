# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "generator @ git+https://github.com/microsoft/lsprotocol.git",
# ]
# ///

"""
Type Server Protocol Generator

This script generates a Rust protocol.rs file from the Type Server Protocol
JSON definitions using the lsprotocol library's generator infrastructure.
"""

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, Iterable, Optional

# Import the lsprotocol generator modules

import generator.model as model
from generator.plugins.rust import generate as rust_generate


def load_json_schema(file_path: str) -> Dict[str, Any]:
    """Load and parse a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_json_to_model(tsp_json: Dict[str, Any]) -> model.LSPModel:
    """
    Convert our TSP JSON format to the lsprotocol internal model format.

    This function maps our JSON structure to the Python dataclasses used
    by the lsprotocol generator.
    """

    # Extract metadata

    metadata = tsp_json["metaData"]

    # Convert enumerations

    enumerations = []
    for enum_def in tsp_json.get("enumerations", []):
        values = []
        for value_def in enum_def["values"]:
            values.append(
                model.EnumItem(
                    name=value_def["name"],
                    value=value_def["value"],
                    documentation=value_def.get("documentation"),
                )
            )
        # Determine enumeration base type

        first_value = enum_def["values"][0]["value"] if enum_def["values"] else ""
        if isinstance(first_value, int):
            enum_type = model.EnumValueType(kind="base", name="integer")
        else:
            enum_type = model.EnumValueType(kind="base", name="string")
        enumerations.append(
            model.Enum(
                name=enum_def["name"],
                type=enum_type,
                values=values,
                documentation=enum_def.get("documentation"),
                supportsCustomValues=enum_def.get("supportsCustomValues", False),
            )
        )
    # Convert structures

    structures = []
    for struct_def in tsp_json.get("structures", []):
        properties = []
        for prop_def in struct_def["properties"]:
            prop_type = convert_type_reference(prop_def["type"])
            properties.append(
                model.Property(
                    name=prop_def["name"],
                    type=prop_type,
                    optional=prop_def.get("optional", False),
                    documentation=prop_def.get("documentation"),
                )
            )
        structures.append(
            model.Structure(
                name=struct_def["name"],
                properties=properties,
                documentation=struct_def.get("documentation"),
            )
        )
    # Convert type aliases

    type_aliases = []
    for alias_def in tsp_json.get("typeAliases", []):
        alias_type = convert_type_reference(alias_def["type"])
        type_aliases.append(
            model.TypeAlias(
                name=alias_def["name"],
                type=alias_type,
                documentation=alias_def.get("documentation"),
            )
        )
    # Convert requests

    requests = []
    for req_def in tsp_json.get("requests", []):
        # Convert parameters

        params = None
        if req_def.get("params"):
            params = convert_type_reference(req_def["params"])
        # Convert result type - make sure it exists

        if "result" not in req_def:
            print(
                f"Warning: Request {req_def.get('method')} missing result field, skipping"
            )
            continue
        result = convert_type_reference(req_def["result"])

        requests.append(
            model.Request(
                method=req_def["method"],
                params=params,
                result=result,
                messageDirection=req_def["messageDirection"],
                documentation=req_def.get("documentation"),
                typeName=req_def.get("typeName"),
            )
        )
    # Convert notifications

    notifications = []
    for notif_def in tsp_json.get("notifications", []):
        # Convert parameters

        params = None
        if notif_def.get("params"):
            params = convert_type_reference(notif_def["params"])
        notifications.append(
            model.Notification(
                method=notif_def["method"],
                params=params,
                messageDirection=notif_def["messageDirection"],
                documentation=notif_def.get("documentation"),
                typeName=notif_def.get("typeName"),
            )
        )
    return model.LSPModel(
        metaData=metadata,
        enumerations=enumerations,
        structures=structures,
        typeAliases=type_aliases,
        requests=requests,
        notifications=notifications,
    )


def convert_type_reference(type_def: Dict[str, Any]) -> model.LSP_TYPE_SPEC:
    """Convert a type definition from JSON to the model format."""
    kind = type_def["kind"]

    if kind == "base":
        return model.BaseType(kind="base", name=type_def["name"])
    elif kind == "reference":
        return model.ReferenceType(kind="reference", name=type_def["name"])
    elif kind == "array":
        element_type = convert_type_reference(type_def["element"])
        return model.ArrayType(kind="array", element=element_type)
    elif kind == "or":
        items = [convert_type_reference(item) for item in type_def["items"]]
        return model.OrType(kind="or", items=items)
    elif kind == "and":
        items = [convert_type_reference(item) for item in type_def["items"]]
        return model.AndType(kind="and", items=items)
    elif kind == "literal":
        # Handle structure literals

        properties = []
        for prop_def in type_def["value"]["properties"]:
            prop_type = convert_type_reference(prop_def["type"])
            properties.append(
                model.Property(
                    name=prop_def["name"],
                    type=prop_type,
                    optional=prop_def.get("optional", False),
                    documentation=prop_def.get("documentation"),
                )
            )
        literal_value = model.LiteralValue(properties=properties)
        return model.LiteralType(kind="literal", value=literal_value)
    elif kind == "stringLiteral":
        return model.StringLiteralType(kind="stringLiteral", value=type_def["value"])
    elif kind == "integerLiteral":
        return model.IntegerLiteralType(kind="integerLiteral", value=type_def["value"])
    elif kind == "booleanLiteral":
        return model.BooleanLiteralType(kind="booleanLiteral", value=type_def["value"])
    else:
        raise ValueError(f"Unsupported type kind: {kind}")


def camel_to_upper_snake(name: str) -> str:
    """Convert CamelCase / mixedCase to UPPER_SNAKE."""
    s1 = re.sub("(.)([A-Z][a-z0-9]+)", r"\1_\2", name)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.upper()


def camel_to_snake(name: str) -> str:
    return camel_to_upper_snake(name).lower()


def generate_constants_rust(tsp_json: Dict[str, Any]) -> str:
    """Generate idiomatic Rust constant definitions (UPPER_SNAKE) from tsp.json."""
    constants = tsp_json.get("constants", [])
    if not constants:
        return ""
    rust_code = "// Type Server Protocol Constants (idiomatic Rust)\n\n"
    for const_def in constants:
        raw_name = const_def["name"]
        name = camel_to_upper_snake(raw_name)
        const_type = const_def["type"]
        value = const_def["value"]
        doc = const_def.get("documentation", "")
        if doc:
            rust_code += f"/// {doc}\n"
        if const_type["kind"] == "base":
            if const_type["name"] == "string":
                rust_type = "&str"
                rust_value = f'"{value}"'
            elif const_type["name"] == "integer":
                rust_type = "i32"
                rust_value = str(value)
            else:
                rust_type = const_type["name"]
                rust_value = str(value)
        else:
            rust_type = "/* unsupported type */"
            rust_value = str(value)
        rust_code += f"pub const {name}: {rust_type} = {rust_value};\n"
    return rust_code + "\n"


def generate_request_enum(content: str, requests: list[model.Request]) -> str:
    # Put a new enum right after the LSPRequestMethods

    offset = content.find("pub enum LSPRequestMethods")
    if offset == -1:
        return content
    end_offset = content.find("}", offset)
    if end_offset == -1:
        return content
    end_offset = content.find("\n", end_offset)

    new_str = "#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]\n"
    new_str += '#[serde(tag = "method")]'
    new_str += "pub enum TSPRequests {\n"
    for request in requests:
        new_str += f'    #[serde(rename = "{request.method}")]'
        request_params = (
            ""
            if request.params is None
            else f"\n        params: {request.params.name},"
        )
        new_str += f"    {request.typeName}{{\n        id: serde_json::Value,{request_params}\n    }},\n"
    new_str += "}\n"
    return content[:end_offset] + "\n\n" + new_str + content[end_offset:]


def fixup_request_response_in_content(content: str, request: model.Request) -> str:
    # Find the location in the content where the request response is defined

    response_name = request.typeName.replace("Request", "Response")
    offset = content.find(f"pub struct {response_name}")

    # Above this should be two # macros. Find those too

    if offset != -1:
        offset = content.rfind("#", None, offset)
        offset = content.rfind("#", None, offset)
        end_offset = content.find("}", offset)

        if end_offset == -1 or offset == -1:
            print(
                f"Warning: Could not find end of struct for {request.typeName}, skipping"
            )
            return content
        if offset > end_offset:
            print(
                f"Warning: Offset {offset} is after end_offset {end_offset} for {request.typeName}, skipping"
            )
            return content
        # The result type should be in the response struct

        result_offset = content.find("pub result: ", offset)
        if result_offset == -1 or result_offset > end_offset:
            print(f"Warning: Could not find result field in {response_name}, skipping")
            return content
        # Extract the result type

        result_end = content.find(",", result_offset)

        if result_end == -1 or result_end > end_offset:
            print(
                f"Warning: Could not find end of result field in {response_name}, skipping"
            )
        # Rewrite the response struct to match the expected format

        result_type = content[result_offset + len("pub result: ") : result_end].strip()

        # Remove the Option<> wrapper if it exists

        if result_type.startswith("Option<") and result_type.endswith(">"):
            result_type = result_type[7:-1].strip()
        # Create an alias for the result type that matches the response name

        new_str = f"pub type {response_name} = {result_type};\n\n"

        # Return the new content

        return content[:offset] + new_str + content[end_offset + 1 :]


def fixup_request_in_content(content: str, request: model.Request) -> str:
    # Find the location in the content where the request is defined

    offset = content.find(f"pub struct {request.typeName}")

    if offset != -1:
        end_offset = content.find("}", offset)

        if end_offset == -1 or offset == -1:
            print(
                f"Warning: Could not find end of struct for {request.typeName}, skipping"
            )
            return content
        if offset > end_offset:
            print(
                f"Warning: Offset {offset} is after end_offset {end_offset} for {request.typeName}, skipping"
            )
            return content
        # Remove the jsonrpc entry. In our handling in pyrefly we
        # have this already removed by the time we want to match

        request_text = content[offset : end_offset + 1]
        request_text = request_text.replace(
            "    /// The version of the JSON RPC protocol.\n    pub jsonrpc: String,\n",
            "",
        )

        return content[:offset] + request_text + content[end_offset + 1 :]


# Helper to replace enum flags with newtype bitflag style


def replace_flag_enum(content: str, name: str, mapping: Dict[str, int]) -> str:
    def find_block_end(idx: int) -> Optional[int]:
        if idx < 0:
            return None
        depth = 0
        i = idx
        while i < len(content):
            c = content[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        return None

    enum_marker = f"pub enum {name} "
    enum_start = content.find(enum_marker)
    if enum_start == -1:
        print(
            f"Warning: Enum {name} not found in content, skipping flag enum replacement"
        )
        return content
    enum_brace = content.find("{", enum_start)
    enum_end = find_block_end(enum_brace)
    if enum_end is None:
        print(
            f"Warning: Could not find end of enum {name}, skipping flag enum replacement"
        )
        return content
    ser_marker = f"impl Serialize for {name}"
    ser_start = content.find(ser_marker, enum_end)
    if ser_start == -1:
        print(
            f"Warning: Could not find Serialize impl for {name}, skipping flag enum replacement"
        )
        return content
    ser_brace = content.find("{", ser_start)
    ser_end = find_block_end(ser_brace)
    if ser_end is None:
        print(
            f"Warning: Could not find end of Serialize impl for {name}, skipping flag enum replacement"
        )
        return content
    de_marker = f"impl<'de> Deserialize<'de> for {name}"
    de_start = content.find(de_marker, ser_end)
    if de_start == -1:
        print(
            f"Warning: Could not find Deserialize impl for {name}, skipping flag enum replacement"
        )
        return content
    de_brace = content.find("{", de_start)
    de_end = find_block_end(de_brace)
    if de_end is None:
        print(
            f"Warning: Could not find end of Deserialize impl for {name}, skipping flag enum replacement"
        )
        return content
    # Capture doc comments preceding enum

    doc_start = enum_start
    line_start = content.rfind("\n", 0, enum_start) + 1
    while line_start >= 0:
        line = content[line_start:enum_start]
        stripped = line.strip()
        if (
            stripped.startswith("///")
            or stripped.startswith("#[derive")
            or stripped == ""
        ):
            doc_start = line_start
            if line_start == 0:
                break
            prev_line_end = content.rfind("\n", 0, line_start - 1)
            if prev_line_end == -1:
                line_start = 0
            else:
                line_start = prev_line_end + 1
        else:
            break
    doc_lines = [
        l
        for l in content[doc_start:enum_start].splitlines()
        if l.strip().startswith("///")
    ]

    lines: list[str] = []
    if doc_lines:
        lines.extend(doc_lines)
    lines.append(f"#[derive(PartialEq, Eq, Clone, Copy, Debug)]")
    lines.append(f"pub struct {name}(pub i32);")
    lines.append(f"impl {name} {{")
    for const_name, val in mapping.items():
        upper = camel_to_upper_snake(const_name)
        lines.append(f"    pub const {upper}: {name} = {name}({val});")
    lines.append("    #[inline] pub fn new() -> Self { Self::NONE }")
    for const_name, val in mapping.items():
        if val == 0:
            continue
        snake = camel_to_snake(const_name)
        upper = camel_to_upper_snake(const_name)
        lines.append(
            f"    #[inline] pub fn with_{snake}(self) -> Self {{ {name}(self.0 | {name}::{upper}.0) }}"
        )
    lines.append(
        "    #[inline] pub fn contains(self, other: Self) -> bool { (self.0 & other.0) == other.0 }"
    )
    lines.append("}")
    lines.append(
        f"impl Serialize for {name} {{ fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error> where S: serde::Serializer {{ s.serialize_i32(self.0) }} }}"
    )
    lines.append(
        f"impl<'de> Deserialize<'de> for {name} {{ fn deserialize<D>(d: D) -> Result<Self, D::Error> where D: serde::Deserializer<'de> {{ let v = i32::deserialize(d)?; Ok({name}(v)) }} }}"
    )
    lines.append(
        f"impl std::ops::BitOr for {name} {{ type Output = {name}; fn bitor(self, rhs: {name}) -> {name} {{ {name}(self.0 | rhs.0) }} }}"
    )
    lines.append(
        f"impl std::ops::BitOrAssign for {name} {{ fn bitor_assign(&mut self, rhs: {name}) {{ self.0 |= rhs.0; }} }}"
    )
    lines.append(
        f"impl std::ops::BitAnd for {name} {{ type Output = {name}; fn bitand(self, rhs: {name}) -> {name} {{ {name}(self.0 & rhs.0) }} }}"
    )
    replacement = "\n" + "\n".join(lines) + "\n"
    return content[:doc_start] + replacement + content[de_end:]


def extract_flag_enums_from_tsp(tsp_json: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """
    Extract flag enums from TSP JSON file.

    A flag enum is identified by:
    1. Having integer values
    2. Having "Flags" in the name, OR
    3. Having values that are powers of 2 (or 0), suggesting bitflag usage
    """
    flag_enums = {}

    for enum_def in tsp_json.get("enumerations", []):
        enum_name = enum_def["name"]

        # Check if this enum has integer values

        if enum_def.get("type", {}).get("name") != "integer":
            continue
        # Extract value mappings

        mapping = {}
        power_of_two_count = 0
        zero_count = 0

        for value_def in enum_def["values"]:
            name = value_def["name"]
            value = value_def["value"]
            mapping[name] = value

            if value == 0:
                zero_count += 1
            elif value > 0 and (value & (value - 1)) == 0:
                # This is a power of 2

                power_of_two_count += 1
        total_values = len(mapping)

        # Determine if this is likely a flag enum:

        is_flag_enum = False

        # Strong indicator: name contains "Flags"

        if "Flags" in enum_name:
            is_flag_enum = True
        # Special case: all values are powers of 2 or 0
        elif total_values > 1 and (power_of_two_count + zero_count) == total_values:
            is_flag_enum = True
        if is_flag_enum:
            flag_enums[enum_name] = mapping
    return flag_enums


def generate_rust_protocol(tsp_json_path: str, output_dir: str) -> None:
    """Generate the Rust protocol.rs file from TSP JSON."""

    # Load the TSP JSON

    print(f"Loading TSP definition from: {tsp_json_path}")
    tsp_json = load_json_schema(tsp_json_path)

    # Convert to internal model

    print("Converting TSP JSON to internal model...")
    lsp_model = convert_json_to_model(tsp_json)

    # Generate Rust code

    print(f"Generating Rust code to: {output_dir}")

    # Create output directory

    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate the Rust code using lsprotocol's generator
    # Since we don't have a test directory, we'll pass an empty string

    rust_generate(lsp_model, str(output_path), "")

    # The generator creates a 'lsprotocol' subdirectory, let's copy the lib.rs
    # to protocol.rs in our src directory

    generated_lib = output_path / "lsprotocol" / "src" / "lib.rs"
    target_protocol = output_path / "src" / "protocol.rs"

    if generated_lib.exists():
        print(f"Fixing up generated protocol.rs...")
        content = generated_lib.read_text(encoding="utf-8")

        # Add the requests enum

        content = generate_request_enum(content, lsp_model.requests)

        # For each of the request methods, change to match how lsp requests are defined. Meaning dont use
        # the structure generated by lsprotocol, but rather the enum style

        for request in lsp_model.requests:
            content = fixup_request_response_in_content(content, request)
            content = fixup_request_in_content(content, request)
        # Update the header comment and other identifiers

        content = (
            content.replace(
                "Language Server Protocol types for Rust generated from LSP specification.",
                "Type Server Protocol types for Rust generated from TSP specification.",
            )
            .replace(
                "// Steps to generate:\n// 1. Checkout https://github.com/microsoft/lsprotocol\n// 2. Install nox: `python -m pip install nox`\n// 3. Run command: `python -m nox --session build_lsp`",
                "// Steps to generate:\n// 1. Create tsp.json and tsp.schema.json from typeServerProtocol.ts\n// 2. Install lsprotocol generator: `pip install git+https://github.com/microsoft/lsprotocol.git`\n// 3. Run: `python generate_protocol.py`",
            )
            .replace("LSPRequestMethods", "TSPRequestMethods")
        )
        content = content.replace("LSPNotificationMethods", "TSPNotificationMethods")
        content = content.replace("LSPAny", "serde_json::Value")
        content = content.replace(
            "use std::collections::HashMap;\nuse url::Url;\nuse rust_decimal::Decimal;",
            "",
        )
        content = content.replace("GetDocString", "GetDocstring")

        # Cleanup the default output

        shutil.rmtree(output_path / "lsprotocol")

        # Append constants if present

        constants_rust = generate_constants_rust(tsp_json)
        if constants_rust:
            content += "\n\n" + constants_rust
        # Add crate-level allows

        content = "#![allow(clippy::all)]\n#![allow(dead_code)]\n\n" + content

        # Fixup flag enums - automatically detect flag enums from TSP JSON

        flag_enums = extract_flag_enums_from_tsp(tsp_json)
        for enum_name, mapping in flag_enums.items():
            content = replace_flag_enum(content, enum_name, mapping)
        target_protocol.write_text(content, encoding="utf-8")
        print(f"Successfully generated: {target_protocol}")
        subprocess.run(["cargo", "fmt", "--", str(target_protocol)], check=False)
    else:
        print(f"Warning: Generated lib.rs not found at {generated_lib}")


def main():
    """Main entry point."""
    if len(sys.argv) != 1:
        print("Usage: python generate_protocol.py")
        print("Example: python generate_protocol.py")
        sys.exit(1)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tsp_json_path = os.path.join(script_dir, "tsp.json")
    output_dir = os.path.abspath(os.path.join(script_dir, "../"))

    try:
        generate_rust_protocol(tsp_json_path, output_dir)
        print("✅ Protocol generation completed successfully!")
    except Exception as e:
        print(f"❌ Error generating protocol: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
