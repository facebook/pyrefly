---
title: "Right Types, Wrong Code: Surprising Bugs A Type Checker Catches"
description: Pyrefly catches more than type mismatches. Here are five real Python bugs — from forgotten awaits to renamed parameters — that your type checker can find before your users do.
slug: surprising-errors
authors: [rebeccachen]
tags: [typechecking]
hide_table_of_contents: false
---

A type checker, as its name suggests, catches type mismatches: things like passing a `str` to a function that expects an `int`. But to understand your code's types, a type checker also has to understand its structure: control flow, scoping, class hierarchies, and more. This lets it detect a surprisingly wide range of issues that have nothing to do with `int` vs. `str`.

Here are five real categories of bugs that Pyrefly catches, none of which are straightforward type mismatches.

<!-- truncate -->

## 1. The Silent Coroutine

This code runs without error but silently fails to send a notification.

```python
async def send_notification(user_id: int, message: str) -> None: ...

async def handle_request(user_id: int) -> None:
    send_notification(user_id, "Request received")  # bug!
    process_request(user_id)
```

(See the code in the [sandbox](https://pyrefly.org/sandbox/?project=N4IgZglgNgpgziAXKOBDAdgEwEYHsAeAdAA4CeSIqcp6AxgASYxj1wxYD66uALhJLVR9c6ABQBXNgCcOETInoR0PADT0AtvDQBzGArg8pASnoBaAHz0AciL31CDgDpTH6Z66YtiU3LS0cpGABHcXgeCWlZeUVlEwtrWwUHQnc3F3QqGgZPegALDExYAODQgwiYGTkFJR44yxt0PVT6FtZ2TC5efghBYTFJCqi1RxAAJRKw%2BkC-CAA3GEwRk3oAYnpscW0AQmbW719-QJCw8srMIxAVEDJAsChSQh5cdSgKNYAFUlv71gK8fHotBEkG04ikQggIhS6DWAGUYDA8jweMQ4IgAPTom7Me6EXBSbTo9jozC%2BODooHoEFgiEidH0MD4%2BioWaoaCobCwQHAiCg8F9ei4Yh9ODQsg8XIiUzzKRwSHoegAXnoIwAzIQAIyqkauADaFR8soAuq5xOgIOpiPieAtTJgINM%2BPMlfQAOS87iBV2ubg8UxHcQO20AaxgpFMqFofjgcBdroA7qgpOhvegQABfK6Rp0wABi0BgFDQWH%2BJHI6aAA).)

Spot the bug? `send_notification` is an async function, and calling it without `await` creates a coroutine object that is immediately discarded. The notification never sends. Python won't raise an exception — you'll just get a `RuntimeWarning` that's easy to miss in logs.

Pyrefly flags this as an [**`unused-coroutine`**](https://pyrefly.org/en/docs/error-kinds/#unused-coroutine): the result of an async function call was neither awaited nor assigned to a variable. The fix is simple:

```python
await send_notification(user_id, "Request received")
```

## 2. The Forgotten Call

This code attempts to check whether a user is authorized before performing an action.

```python
def is_authorized(user: User) -> bool:
    return user.role in ADMIN_ROLES

def handle_admin_request(user: User) -> Response:
    if is_authorized:  # bug!
        return perform_action()
    return Response(403)
```

([sandbox](https://pyrefly.org/sandbox/?project=N4IgZglgNgpgziAXKOBDAdgEwEYHsAeAdAA4CeSIAggCICyAkgHID6ASgPIAyAogMqIACKBDgAXANpiATgF0BAXgHiZAHSkr0ajQGMoqOHAEBVODCmIt6AdYFTcsQdMuXd%2Bw63jFc6UxfVWbTBgwAWZmCHQIUTCAClMoMAAaAW1cIMEI0QBKAQBaAD4BRm8YQUJy538gkJFmVABXUQALXCkIAC8YTBj603NjPpyCgTx7Pw0bWxhReqkrXrNCO1gBCIEaBhYOHl5KjWqBYjMwVoBbOu1RCG8YocKPOC8fUstJqWnZqwen0xiAJgADACsnssMEBE0MJhYHVMKcIsx3gBHerwUQ9PqCExmO4Cb7eXyvGwQGpwOqNFptTqYQQCADEI3qAHMAIREybWd4zOaHY5nC5XG4g-xvD48-HPGIAFgBAGYsiBEiAyO8wFBSIRRLhTlAKAyAAqkVXqgRoLB4fApbyQJmzVCC9CEDQM3gwGAQ0SiYhwRAAel9KuC6sIrSZvpg6F9mFw2jgvtS6BtdodvoEJykAlQADdUNBUNgVgmk1J7dcrLhiA64E70GRmt5clmzHAywoBCoQLLCABGWUdjTiMx2KRwVToeqRU5eKSiLq5TAQd6XCBNtsAcggTPQrRga4029EuWR9UXc4A1jBSLlUNptPBDIo1wB3VBzPfoEAAXyVN6uTYAYtAMAUGaOAECQ5CfkAA))

However, the action is always performed regardless of authorization. `is_authorized` (without parentheses) is a reference to the function object, which is always truthy. What was meant was `is_authorized(user)`.

Pyrefly catches this as a [**`redundant-condition`**](https://pyrefly.org/en/docs/error-kinds/#redundant-condition): it knows that `is_authorized` is a function, and a function is always truthy, so the condition is redundant.

## 3. The Breaking Rename

```python
class BaseCache:
    def get(self, key: str, default: object = None) -> object:
        ...

class RedisCache(BaseCache):
    def get(self, key: str, fallback: object = None) -> object: ...  # bug!
```

([sandbox](https://pyrefly.org/sandbox/?project=N4IgZglgNgpgziAXKOBDAdgEwEYHsAeAdAA4CeSIAxlKnHAAQBCtMAwqpQBYyIA6ATr3T0R9TDDD0A5jAAuACjgwoYADT0A1jFKJ6cWf3XiwqAK5RZu3NgBWMSrPoBeegDlc6GAEp6AWgB89NZ2DnyCwqKRhNECQrHo1LQMAEowmBBw7Fww8sxKWdxeYUKRxtJyispqmtq6%2Bob0JlBQ2BwaVrb2ji7unj4BQZ2h9NGEIgDE9NimUgCEIKogZPwSUKSEsrgAtlAUkwAKpCtga3oYOAT0lB6QUqb8qLIQHoRCkwDKMDD0nLKyxHBEAB6IHLVbrXD8KRAmDoIGYXCUOBA67oW73R7POGNSH0VAAN1Q0FQ2FgVxuEDuDyeHiCxBp6Dgr3QZFknA8vnxMH4cCxznovBAAGZCABGIWCoQAbW5-EhcAAukJTOgIFtiJDZGlfOkVg4IFz%2BQBySnoSEwI1CM2yXwrACOpggK0wvi0pF8HEo8AYLiNAHdUPx0Jb0CAAL6LDhPLkAMWgMAoaCweCIZHDQA))

This looks harmless. `RedisCache.get` has the same types as `BaseCache.get`, just a different parameter name. But any caller using `cache.get("x", default=None)` will break when the cache is a `RedisCache`, because `RedisCache.get` doesn't have a parameter named `default`.

Pyrefly reports this as [**`bad-override-param-name`**](https://pyrefly.org/en/docs/error-kinds/#bad-override-param-name): a subclass renamed a parameter that callers might be passing by keyword. This is a violation of the [Liskov Substitution Principle](https://en.wikipedia.org/wiki/Liskov_substitution_principle) because you can't safely substitute a `RedisCache` where a `BaseCache` is expected.

It's easy to dismiss this as unlikely, but it happens a lot in practice. Someone inherits from a base class, doesn't look at the parameter names carefully, and chooses a name that makes more sense for their implementation. The bug only surfaces when someone passes the argument by name, which might be rare enough that tests don't cover it.

## 4. The Missing Case

```python
from enum import Enum

class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"

def to_hex(color: Color):
    match color:
        case Color.RED:
            return "#ff0000"
        case Color.GREEN:
            return "#00ff00"
        # forgot BLUE!
```

([sandbox](https://pyrefly.org/sandbox/?project=N4IgZglgNgpgziAXKOBDAdgEwEYHsAeAdAA4CeS4ATrgLYAEM6ArvRDcbpQC50CizNADqVB6YaIDGUVHDh0AwriicAFPxYBKROPR09dAEq8AInQC8dQSEoxMVnfroBxI7wBy5yyADmNxvZFdfQAhABkAVV5PK2woJhgA0R1MGDA6LlwAfQALGHwVCSVORAUiyi0HfRpULglsukLlSm1Ax0cJGRhSpsIjYxbRNqG6Gy4mSl0rAGIwMAAGBbnEoKGOuC7FHpdedwGV4b1R8cmQKYXZheWDqbowTm9cHjDIgEIQABoQMhswKFJCDI0KAUG4ABVIPz%2BdDQWDw%2BAauHQkG84xqEERhFENwAyjAutkuFxiHBEAB6UnfVJ-Qj3UmMUmYXASOCkwpIiAoyhoxGk26cOioABuqGgqFiXTZyNRXHRulwxBliLgmPQZC42URAFpBTBKHBZdEQABmQgARiN9nQAG1ddQ9QBdURMdBsDjcWyazAQGwSGU6zwAcg56E4MADohDXE1NgAjkxvR6ANYwUia1ASCTwOQWAMAd1QE3D6BAAF9Pum-TAAGLQGAUGE4AgkcgloA))

Pyrefly warns about this with [**`non-exhaustive-match`**](https://pyrefly.org/en/docs/error-kinds/#non-exhaustive-match): the `match` statement doesn't cover all members of `Color` and has no wildcard default. If someone passes `Color.BLUE`, Python falls through the match without entering any case, and the function implicitly returns `None`, which may then cause a confusing error somewhere downstream.

## 5. The Misleading Comparison

This code attempts to check whether a user is an admin.

```python
class User: ...
class Admin(User): ...

def check(user: User) -> None:
    if user is Admin:  # bug!
        print("Welcome, admin!")
```

([sandbox](https://pyrefly.org/sandbox/?project=N4IgZglgNgpgziAXKOBDAdgEwEYHsAeAdAA4CeSIAxlKnHAAQCqcMATovYVwDqvfrVaDAIKYAthHQAKZmwCUHLoV78VWGGHqUAFjEoBrKQFcW7Jqbn0AtAD56AOVzoYiNfXf0Imk208jxkhz0AMT02EYA5gCEbh5xxKySAC5S3CAA6jBQlLhiMAA09KgB6DEgciD5IGSsGlCkhEm5UBShAAqktWD19GhYePhaTpARRqyoSRBOyuihAMowMPTaSUnEcIgA9Js1dQ24rBGbMOibmLiUcJs56CNjE1On9GAHRQBuqNCo2LBDtxCjcaTJz0XDEYHoOAzMhJbROKxvNhwR70AC89DSAGZCABGTFpfgAbTYrAOcAAuvwjOgIGJiAckjBMFZMBBapRJoi0fQAOQA9AHGA8-gCpJWWoARyMbKZVn0MFIVlQlEo8AY6J5AHdUKx0ML0CAAL5VZWcmAAMWgMAofRwBBI5ENQA))

This checks whether `user` is *the same object* as the `Admin` class, not whether `user` is an instance of `Admin`. What was almost certainly meant was `isinstance(user, Admin)`. Pyrefly flags this as an [**`unnecessary-comparison`**](https://pyrefly.org/en/docs/error-kinds/#unnecessary-comparison): using `is` to compare a value against a type is almost always a mistake.

## Why a Type Checker?

You might wonder: shouldn't a linter catch these, rather than a type checker? Without an understanding of the types flowing through your program, a linter can see that you wrote `if f:`, but it might not be able to tell that `f` is a function, especially if `f` is imported. Pyrefly's type analysis allows it to report diagnostics that non-type-aware linters cannot detect with confidence.

To see the full list of error kinds Pyrefly supports and their severity levels, check out our [error kinds documentation](https://pyrefly.org/en/docs/error-kinds/). And if there's a bug pattern you wish we'd catch, let us know on [GitHub](https://github.com/facebook/pyrefly/issues) or [Discord](https://discord.gg/Cf7mFQtW7W).
