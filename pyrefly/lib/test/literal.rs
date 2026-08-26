/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::testcase;

testcase!(
    test_fstring_literal,
    r#"
from typing import assert_type, Literal, LiteralString
x0 = f"abc"
assert_type(x0, Literal["abc"])

x1 = f"abc{x0}"
assert_type(x1, LiteralString)

x2 = f"abc" "def"
assert_type(x2, Literal["abcdef"])

x3 = f"abc" f"def"
assert_type(x3, Literal["abcdef"])

x4 = "abc" f"def"
assert_type(x4, Literal["abcdef"])

x5 = "abc" f"def{x0}g" "hij" f"klm"
assert_type(x5, LiteralString)
"#,
);

testcase!(
    test_invalid_literal,
    r#"
from typing import Literal
x = 1
y: Literal[x]  # E: Expected a type form
"#,
);

testcase!(
    test_large_int_literal,
    r#"
from typing import assert_type, Literal
x = 1
y = 0xFFFFFFFFFFFFFFFFFF
assert_type(x, Literal[1])
assert_type(y, Literal[4722366482869645213695])
"#,
);

testcase!(
    test_large_int_type,
    r#"
from typing import Literal
x: Literal[0xFFFFFFFFFFFFFFFFFF]
"#,
);

testcase!(
    test_giant_bytes_literal,
    r#"
from typing import assert_type, Literal
# Small bytes literals stay precise.
assert_type(b"abc", Literal[b"abc"])
# A bytes literal larger than the cap is widened to `bytes`.
assert_type(b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", bytes)
"#,
);

testcase!(
    test_generic_create_literal,
    r#"
from typing import assert_type, Literal

class Foo[T]:
    def __init__(self, x: T) -> None: ...

x: Literal[42] = 42
assert_type(Foo(x), Foo[Literal[42]])
"#,
);

testcase!(
    test_generic_get_literal,
    r#"
from typing import assert_type, Literal

class Foo[T]:
    def get(self) -> T: ...

def test(x: Foo[Literal[42]]) -> None:
    assert_type(x.get(), Literal[42])
"#,
);

testcase!(
    test_literal_string_after_if,
    r#"
from typing import Literal

if True:
    pass

x: Literal["little", "big"] = "big"
"#,
);

testcase!(
    test_literal_none,
    r#"
from typing import Literal
Literal[None]
    "#,
);

testcase!(
    test_literal_alias,
    r#"
from typing import Literal as L
x: L["foo"] = "foo"
"#,
);

testcase!(
    test_literal_string_infer,
    r#"
from typing import LiteralString, assert_type
def f(x: LiteralString):
    assert_type(["foo"], list[str])
    assert_type([x], list[LiteralString])
    xs: list[str] = [x]
"#,
);

testcase!(
    test_index_literal,
    r#"
from typing import Literal, assert_type

def foo(x):
    assert_type("Magic"[0], Literal['M'])
    assert_type("Magic"[3:4], Literal['i'])
"#,
);

testcase!(
    test_index_bool,
    r#"
from typing import assert_type, Literal
t = ("a", "b")
assert_type(t[False], Literal["a"])
assert_type(t[True], Literal["b"])
"#,
);

testcase!(
    test_literal_nesting,
    r#"
from typing import Literal, assert_type

X = Literal["foo", "bar"]
Y = Literal["baz", None]
Z = Literal[X, Y]

def f(x: Z) -> None:
    assert_type(x, Literal["foo", "bar", "baz", None])
"#,
);

testcase!(
    test_literal_direct_nesting,
    r#"
from typing import Literal

good: Literal[Literal[Literal[1, 2, 3], "foo"], 5, None] = "foo"
bad: Literal[Literal, 3]  # E: Expected a type argument for `Literal`
"#,
);

testcase!(
    test_literal_brackets,
    r#"
from typing import Literal
bad6: Literal[(1, "foo", "bar")]  # E: `Literal` arguments cannot be parenthesized
"#,
);

testcase!(
    test_literal_with_nothing,
    r#"
from typing import Literal
bad1: Literal # E: Expected a type argument for `Literal`
bad2: list[Literal]  # E: Expected a type argument for `Literal`
"#,
);

testcase!(
    test_literal_with_byte,
    r#"
from typing import assert_type, Literal
x = b"far"

assert_type(x[0], Literal[102])
x[3.14]  # E: Cannot index into `Literal[b'far']`
y: Literal[0] = 0
assert_type(x[y], Literal[102])

# Negative index case
assert_type(x[-1], Literal[114])
x[-6.28]  # E: Cannot index into `Literal[b'far']`

# The `bytes` type is correct, but ideally we would understand
# literal slices and be able to give back the literal bytes.
assert_type(x[0:1], Literal[b"f"])  # E: assert_type(bytes, Literal[b'f'])

# Non-literal integers give back an `int` (one byte)
i: int = 42
assert_type(x[i], int)
"#,
);

testcase!(
    test_bad_literal,
    r#"
# This used to crash, see https://github.com/facebook/pyrefly/issues/453
0x_fffffffffffffffff
1_23
"#,
);

testcase!(
    test_promote_literal,
    r#"
from typing import assert_type, LiteralString

x = list("abcdefg")
assert_type(x, list[LiteralString])
"#,
);

testcase!(
    test_literal_string_format,
    r#"
from typing import assert_type, LiteralString

# Basic format with literal strings
sep: LiteralString = "{} {}"
x: LiteralString = "foo"
y: LiteralString = "bar"
result = sep.format(x, y)
assert_type(result, LiteralString)

# With keyword arguments
result2 = "{a} {b}".format(a=x, b=y)
assert_type(result2, LiteralString)

# Non-literal positional arg should return str
z: str = "baz"
result3 = sep.format(x, z)
assert_type(result3, str)

# Non-literal keyword arg should return str
result4 = "{a}".format(a=z)
assert_type(result4, str)

# Test starred arguments
args = (x, y)
result5 = sep.format(*args)
assert_type(result5, LiteralString)

args2: tuple[str, ...] = (x, y)
result6 = sep.format(*args2)
assert_type(result6, str)
"#,
);

testcase!(
    test_literal_string_join,
    r#"
from typing import assert_type, LiteralString

sep: LiteralString = ","
items: list[LiteralString] = ["a", "b", "c"]
result = sep.join(items)
assert_type(result, LiteralString)

# Tuple of literals
result2 = sep.join(("x", "y", "z"))
assert_type(result2, LiteralString)

# Non-literal items should return str
non_lit: list[str] = ["x", "y"]
result3 = sep.join(non_lit)
assert_type(result3, str)

# Union with non-literal should return str
mixed: list[LiteralString | str] = []
result4 = sep.join(mixed)
assert_type(result4, str)
"#,
);

testcase!(
    test_literal_string_replace,
    r#"
from typing import assert_type, LiteralString

x: LiteralString = "hello world"
old: LiteralString = "world"
new: LiteralString = "universe"

# Basic replace
result = x.replace(old, new)
assert_type(result, LiteralString)

# With count argument (should still return LiteralString)
result2 = x.replace(old, new, 1)
assert_type(result2, LiteralString)

# With count keyword
result3 = x.replace(old, new, count=1)
assert_type(result3, LiteralString)

# Non-literal old should return str
non_lit: str = "foo"
result4 = x.replace(non_lit, new)
assert_type(result4, str)

# Non-literal new should return str
result5 = x.replace(old, non_lit)
assert_type(result5, str)
"#,
);

testcase!(
    test_literal_string_as_collection,
    r#"
from collections.abc import Container, Collection, Sequence

a: Container[str] = ""
b: Collection[str] = ""
c: Sequence[str] = ""
"#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/2068
testcase!(
    test_literal_string_join_loop_inference,
    r#"
def test():
    items = ["a", "b"]
    lines = []
    for k in items:
        lines.append(f"*{k}")
    return "\n".join(lines)
"#,
);

testcase!(
    test_literal_int_sum_loop_inference,
    r#"
from typing import assert_type
def f(x: int):
    y = []
    for i in range(10):
        y.append(x)
    assert_type(y, list[int])
    sum(y, 0)
    "#,
);

testcase!(
    test_giant_literal_union,
    r#"
from typing import assert_type, Literal
def test(x: Literal[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
    140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
    160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
    180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199,
    200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
    220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
    240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255
], cond: bool) -> None:
    # if a non-enum literal union has >256 of the same kind of literal we promote it
    y = x if cond else 256
    assert_type(y, int)
    "#,
);

testcase!(
    test_giant_string_literal_union,
    r#"
from typing import assert_type, Literal
# Below the cap, the union stays a precise literal.
def small(x: str, cond: bool) -> None:
    y = "a" if cond else "b"
    assert_type(y, Literal["a", "b"])
# Above the cap (>256 distinct str literals), the union is promoted to `str`.
def big(x: Literal[
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
    "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
    "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
    "40", "41", "42", "43", "44", "45", "46", "47", "48", "49",
    "50", "51", "52", "53", "54", "55", "56", "57", "58", "59",
    "60", "61", "62", "63", "64", "65", "66", "67", "68", "69",
    "70", "71", "72", "73", "74", "75", "76", "77", "78", "79",
    "80", "81", "82", "83", "84", "85", "86", "87", "88", "89",
    "90", "91", "92", "93", "94", "95", "96", "97", "98", "99",
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "110", "111", "112", "113", "114", "115", "116", "117", "118", "119",
    "120", "121", "122", "123", "124", "125", "126", "127", "128", "129",
    "130", "131", "132", "133", "134", "135", "136", "137", "138", "139",
    "140", "141", "142", "143", "144", "145", "146", "147", "148", "149",
    "150", "151", "152", "153", "154", "155", "156", "157", "158", "159",
    "160", "161", "162", "163", "164", "165", "166", "167", "168", "169",
    "170", "171", "172", "173", "174", "175", "176", "177", "178", "179",
    "180", "181", "182", "183", "184", "185", "186", "187", "188", "189",
    "190", "191", "192", "193", "194", "195", "196", "197", "198", "199",
    "200", "201", "202", "203", "204", "205", "206", "207", "208", "209",
    "210", "211", "212", "213", "214", "215", "216", "217", "218", "219",
    "220", "221", "222", "223", "224", "225", "226", "227", "228", "229",
    "230", "231", "232", "233", "234", "235", "236", "237", "238", "239",
    "240", "241", "242", "243", "244", "245", "246", "247", "248", "249",
    "250", "251", "252", "253", "254", "255"
], cond: bool) -> None:
    y = x if cond else "256"
    assert_type(y, str)
    "#,
);

testcase!(
    test_giant_bytes_literal_union,
    r#"
from typing import assert_type, Literal
# Above the cap (>256 distinct bytes literals), the union is promoted to `bytes`.
def big(x: Literal[
    b"0", b"1", b"2", b"3", b"4", b"5", b"6", b"7", b"8", b"9",
    b"10", b"11", b"12", b"13", b"14", b"15", b"16", b"17", b"18", b"19",
    b"20", b"21", b"22", b"23", b"24", b"25", b"26", b"27", b"28", b"29",
    b"30", b"31", b"32", b"33", b"34", b"35", b"36", b"37", b"38", b"39",
    b"40", b"41", b"42", b"43", b"44", b"45", b"46", b"47", b"48", b"49",
    b"50", b"51", b"52", b"53", b"54", b"55", b"56", b"57", b"58", b"59",
    b"60", b"61", b"62", b"63", b"64", b"65", b"66", b"67", b"68", b"69",
    b"70", b"71", b"72", b"73", b"74", b"75", b"76", b"77", b"78", b"79",
    b"80", b"81", b"82", b"83", b"84", b"85", b"86", b"87", b"88", b"89",
    b"90", b"91", b"92", b"93", b"94", b"95", b"96", b"97", b"98", b"99",
    b"100", b"101", b"102", b"103", b"104", b"105", b"106", b"107", b"108", b"109",
    b"110", b"111", b"112", b"113", b"114", b"115", b"116", b"117", b"118", b"119",
    b"120", b"121", b"122", b"123", b"124", b"125", b"126", b"127", b"128", b"129",
    b"130", b"131", b"132", b"133", b"134", b"135", b"136", b"137", b"138", b"139",
    b"140", b"141", b"142", b"143", b"144", b"145", b"146", b"147", b"148", b"149",
    b"150", b"151", b"152", b"153", b"154", b"155", b"156", b"157", b"158", b"159",
    b"160", b"161", b"162", b"163", b"164", b"165", b"166", b"167", b"168", b"169",
    b"170", b"171", b"172", b"173", b"174", b"175", b"176", b"177", b"178", b"179",
    b"180", b"181", b"182", b"183", b"184", b"185", b"186", b"187", b"188", b"189",
    b"190", b"191", b"192", b"193", b"194", b"195", b"196", b"197", b"198", b"199",
    b"200", b"201", b"202", b"203", b"204", b"205", b"206", b"207", b"208", b"209",
    b"210", b"211", b"212", b"213", b"214", b"215", b"216", b"217", b"218", b"219",
    b"220", b"221", b"222", b"223", b"224", b"225", b"226", b"227", b"228", b"229",
    b"230", b"231", b"232", b"233", b"234", b"235", b"236", b"237", b"238", b"239",
    b"240", b"241", b"242", b"243", b"244", b"245", b"246", b"247", b"248", b"249",
    b"250", b"251", b"252", b"253", b"254", b"255"
], cond: bool) -> None:
    y = x if cond else b"256"
    assert_type(y, bytes)
    "#,
);

testcase!(
    bug = "Bad interaction between overload resolution and partial type inference",
    test_partial_inference_literalstring_join,
    r#"
from typing import assert_type, LiteralString, reveal_type

def f(x1: list[str], x2: list[LiteralString]):
    x3 = []
    assert_type(", ".join(x1), str)
    assert_type(", ".join(x2), LiteralString)
    # This is wrong: we should not assume `join`'s `LiteralString` overload is matched.
    reveal_type(", ".join(x3))  # E: revealed type: LiteralString
    "#,
);

testcase!(
    test_str_join_boolop_narrowing,
    r#"
from typing import assert_type

def format_types(types: set[type | None]) -> str:
    values = sorted((e and e.__name__) or "None" for e in types)
    assert_type(values, list[str])
    return ", ".join(values)
    "#,
);

testcase!(
    test_giant_literal_string,
    r#"
from typing import assert_type, LiteralString

x = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
"""
assert_type(x, LiteralString)
    "#,
);

testcase!(
    test_literal_try_except_import,
    r#"
from typing import assert_type

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

def fun(param: Literal["test"] = "test"):
    assert_type(param, Literal["test"])

x: Literal["a", "b"] = "a"
assert_type(x, Literal["a", "b"])
"#,
);

// Regression test for https://github.com/facebook/pyrefly/issues/2633
testcase!(
    test_literal_union_annotated,
    r#"
from typing import Annotated, Literal, TypeAlias

One: TypeAlias = Literal[1]
Two: TypeAlias = Annotated[Literal[2], "irrelevant"]
OneOrTwo: TypeAlias = One | Two

Spam: TypeAlias = Literal[OneOrTwo]
"#,
);

testcase!(
    test_enumerate_preserves_literal_type,
    r#"
from typing import Literal, assert_type

def test(x: Literal["a", "b"]) -> None:
    pass

c = ("a", "b")

# Direct iteration preserves Literal types
for i in c:
    test(i)

for i, j in enumerate(c):
    assert_type(i, int)
    test(j)

for i, j in enumerate(c, 1):
    test(j)

for i, j in enumerate(c, start=1):
    test(j)

for i, j in enumerate(iterable=c, start=1):
    test(j)
    "#,
);

testcase!(
    test_enumerate_start_type_checked,
    r#"
enumerate([1, 2], start="oops")  # E: Argument `Literal['oops']` is not assignable to parameter `start` with type `int`
    "#,
);

testcase!(
    test_promote_module_level_literal_in_function,
    r#"
from typing import Literal, assert_type

timeout = 100
MY_CONST = 42

def foo():
    assert_type(timeout, int)
    assert_type(MY_CONST, Literal[42])
    "#,
);

testcase!(
    test_promote_branchy_literal_in_function,
    r#"
from typing import assert_type

def cond() -> bool: ...
if cond():
    x = 1
else:
    x = 2

def foo():
    assert_type(x, int)
    "#,
);

testcase!(
    test_promote_module_level_enum_literal_in_function,
    r#"
from typing import assert_type
from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2

x = Color.RED

def foo():
    assert_type(x, Color)
    "#,
);
