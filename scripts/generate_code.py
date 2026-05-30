"""
scripts/generate_code.py

Generates code correctness preference pairs.
Correct code vs buggy code. Bugs are verified: buggy code fails a test,
correct code passes.

Output: data/code_pairs.jsonl
"""

import json
import random
import os
import sys
import io
import traceback

random.seed(42)


# ── Problem bank ────────────────────────────────────────────────────────────────
# Each entry: (prompt, correct_code, buggy_code, test_fn)
# test_fn(code_str) -> bool: True if code is correct

PROBLEMS = []


def register(prompt, correct_code, buggy_code, test_fn):
    PROBLEMS.append((prompt, correct_code, buggy_code, test_fn))


# ── 1. Reverse a string ────────────────────────────────────────────────────────
register(
    prompt="Write a Python function `reverse_string(s)` that returns the string reversed.",
    correct_code='''\
def reverse_string(s):
    return s[::-1]
''',
    buggy_code='''\
def reverse_string(s):
    return s[::1]
''',
    test_fn=lambda code: _run_test(code, [
        ("reverse_string('hello')", "olleh"),
        ("reverse_string('abc')",   "cba"),
        ("reverse_string('')",      ""),
    ]),
)

# ── 2. Sum of list ─────────────────────────────────────────────────────────────
register(
    prompt="Write a Python function `sum_list(lst)` that returns the sum of all numbers in a list.",
    correct_code='''\
def sum_list(lst):
    total = 0
    for x in lst:
        total += x
    return total
''',
    buggy_code='''\
def sum_list(lst):
    total = 0
    for x in lst:
        total += x
    return total - lst[0]
''',
    test_fn=lambda code: _run_test(code, [
        ("sum_list([1,2,3])",   6),
        ("sum_list([0,0,0])",   0),
        ("sum_list([10,20])",   30),
    ]),
)

# ── 3. Check palindrome ────────────────────────────────────────────────────────
register(
    prompt="Write a Python function `is_palindrome(s)` that returns True if s is a palindrome, False otherwise.",
    correct_code='''\
def is_palindrome(s):
    return s == s[::-1]
''',
    buggy_code='''\
def is_palindrome(s):
    return s == s[1:]
''',
    test_fn=lambda code: _run_test(code, [
        ("is_palindrome('racecar')", True),
        ("is_palindrome('hello')",   False),
        ("is_palindrome('a')",       True),
    ]),
)

# ── 4. Factorial ───────────────────────────────────────────────────────────────
register(
    prompt="Write a Python function `factorial(n)` that returns n! for non-negative integers.",
    correct_code='''\
def factorial(n):
    if n == 0:
        return 1
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
''',
    buggy_code='''\
def factorial(n):
    if n == 0:
        return 1
    result = 1
    for i in range(1, n):
        result *= i
    return result
''',
    test_fn=lambda code: _run_test(code, [
        ("factorial(0)",  1),
        ("factorial(1)",  1),
        ("factorial(5)",  120),
        ("factorial(6)",  720),
    ]),
)

# ── 5. Count vowels ────────────────────────────────────────────────────────────
register(
    prompt="Write a Python function `count_vowels(s)` that returns the number of vowels (a,e,i,o,u) in s.",
    correct_code='''\
def count_vowels(s):
    return sum(1 for c in s.lower() if c in "aeiou")
''',
    buggy_code='''\
def count_vowels(s):
    return sum(1 for c in s if c in "aeiou")
''',
    test_fn=lambda code: _run_test(code, [
        ("count_vowels('Hello')",  2),
        ("count_vowels('AEIOU')",  5),
        ("count_vowels('xyz')",    0),
    ]),
)

# ── 6. FizzBuzz ────────────────────────────────────────────────────────────────
register(
    prompt="Write a Python function `fizzbuzz(n)` that returns a list of fizzbuzz values from 1 to n.",
    correct_code='''\
def fizzbuzz(n):
    result = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result
''',
    buggy_code='''\
def fizzbuzz(n):
    result = []
    for i in range(1, n + 1):
        if i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        elif i % 15 == 0:
            result.append("FizzBuzz")
        else:
            result.append(str(i))
    return result
''',
    test_fn=lambda code: _run_test(code, [
        ("fizzbuzz(15)[14]", "FizzBuzz"),
        ("fizzbuzz(5)[4]",   "Buzz"),
        ("fizzbuzz(3)[2]",   "Fizz"),
        ("fizzbuzz(1)[0]",   "1"),
    ]),
)

# ── 7. Max in list ─────────────────────────────────────────────────────────────
register(
    prompt="Write a Python function `find_max(lst)` that returns the maximum element without using built-in max().",
    correct_code='''\
def find_max(lst):
    m = lst[0]
    for x in lst:
        if x > m:
            m = x
    return m
''',
    buggy_code='''\
def find_max(lst):
    m = lst[0]
    for x in lst:
        if x < m:
            m = x
    return m
''',
    test_fn=lambda code: _run_test(code, [
        ("find_max([3,1,4,1,5,9])", 9),
        ("find_max([7])",           7),
        ("find_max([-1,-5,-2])",    -1),
    ]),
)

# ── 8. Binary search ───────────────────────────────────────────────────────────
register(
    prompt="Write a Python function `binary_search(arr, target)` that returns the index of target in sorted arr, or -1 if not found.",
    correct_code='''\
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
''',
    buggy_code='''\
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
''',
    test_fn=lambda code: _run_test(code, [
        ("binary_search([1,2,3,4,5], 3)", 2),
        ("binary_search([1,2,3,4,5], 5)", 4),
        ("binary_search([1,2,3,4,5], 6)", -1),
    ]),
)

# ── 9. Remove duplicates (order-preserving) ────────────────────────────────────
# Bug: list(set(...)) does NOT preserve insertion order.
# Test uses [3,1,2,1,3] whose correct deduplicated order is [3,1,2],
# but set-based approach returns elements in hash order (e.g. [1,2,3]).
register(
    prompt="Write a Python function `remove_duplicates(lst)` that returns a list with duplicates removed, preserving insertion order.",
    correct_code='''\
def remove_duplicates(lst):
    seen = set()
    result = []
    for x in lst:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result
''',
    buggy_code='''\
def remove_duplicates(lst):
    # BUG: set() does not preserve insertion order
    return list(set(lst))
''',
    test_fn=lambda code: _run_test(code, [
        # First element is 3, so correct order starts with 3 — set() will NOT do this
        ("remove_duplicates([3, 1, 2, 1, 3])", [3, 1, 2]),
        # Strings: correct order is ['b','a','c'], set order is unpredictable
        ("remove_duplicates(['b','a','c','b','a'])", ['b','a','c']),
        # Single element
        ("remove_duplicates([7, 7, 7])", [7]),
    ]),
)

# ── 10. Fibonacci ──────────────────────────────────────────────────────────────
register(
    prompt="Write a Python function `fibonacci(n)` that returns the nth Fibonacci number (0-indexed, fib(0)=0, fib(1)=1).",
    correct_code='''\
def fibonacci(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
''',
    buggy_code='''\
def fibonacci(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n):
        a, b = b, a + b
    return b
''',
    test_fn=lambda code: _run_test(code, [
        ("fibonacci(0)",  0),
        ("fibonacci(1)",  1),
        ("fibonacci(10)", 55),
        ("fibonacci(7)",  13),
    ]),
)

# ── 11. Two sum ────────────────────────────────────────────────────────────────
register(
    prompt="Write a Python function `two_sum(nums, target)` that returns indices of two numbers that add to target.",
    correct_code='''\
def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
    return []
''',
    buggy_code='''\
def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
''',
    test_fn=lambda code: _run_test(code, [
        ("two_sum([2,7,11,15], 9)",  [0,1]),
        ("two_sum([3,2,4], 6)",      [1,2]),
    ]),
)

# ── 12. String anagram ─────────────────────────────────────────────────────────
register(
    prompt="Write a Python function `is_anagram(s, t)` that returns True if s and t are anagrams.",
    correct_code='''\
def is_anagram(s, t):
    return sorted(s) == sorted(t)
''',
    buggy_code='''\
def is_anagram(s, t):
    return len(s) == len(t)
''',
    test_fn=lambda code: _run_test(code, [
        ("is_anagram('anagram','nagaram')", True),
        ("is_anagram('rat','car')",         False),
        ("is_anagram('ab','ba')",           True),
    ]),
)


# ── Test runner ────────────────────────────────────────────────────────────────

def _run_test(code_str: str, cases: list[tuple]) -> bool:
    """Execute code and run test cases. Returns True iff all pass."""
    namespace = {}
    try:
        exec(code_str, namespace)
    except Exception:
        return False

    for expr, expected in cases:
        try:
            result = eval(expr, namespace)
            if result != expected:
                return False
        except Exception:
            return False
    return True


def verify_problem(correct_code, buggy_code, test_fn):
    """Sanity-check that correct passes and buggy fails."""
    correct_passes = test_fn(correct_code)
    buggy_fails    = not test_fn(buggy_code)
    return correct_passes, buggy_fails


# ── Generation ─────────────────────────────────────────────────────────────────

def generate_code_pairs(n: int = 1000) -> list[dict]:
    # Verify all problems first
    print("Verifying problem correctness...")
    verified = []
    for i, (prompt, correct, buggy, test_fn) in enumerate(PROBLEMS):
        cp, bf = verify_problem(correct, buggy, test_fn)
        status = "OK" if (cp and bf) else f"FAIL (correct_passes={cp}, buggy_fails={bf})"
        print(f"  Problem {i+1:2d}: {status}  — {prompt[:50]}...")
        if cp and bf:
            verified.append((prompt, correct, buggy))

    print(f"\n{len(verified)}/{len(PROBLEMS)} problems verified.\n")

    samples = []
    for i in range(n):
        prompt, correct, buggy = verified[i % len(verified)]

        if random.random() < 0.5:
            sample = {
                "prompt":     prompt,
                "response_a": correct,
                "response_b": buggy,
                "labels": {"toxicity": None, "math": None, "code": 1},
                "task": "code",
            }
        else:
            sample = {
                "prompt":     prompt,
                "response_a": buggy,
                "response_b": correct,
                "labels": {"toxicity": None, "math": None, "code": 0},
                "task": "code",
            }

        samples.append(sample)

    random.shuffle(samples)
    return samples


def main():
    os.makedirs("data", exist_ok=True)
    pairs = generate_code_pairs(n=1000)

    out_path = "data/code_pairs.jsonl"
    with open(out_path, "w") as f:
        for item in pairs:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(pairs)} code pairs to {out_path}")
    label_1 = sum(1 for p in pairs if p["labels"]["code"] == 1)
    label_0 = sum(1 for p in pairs if p["labels"]["code"] == 0)
    print(f"  Label=1 (A correct): {label_1}  |  Label=0 (B correct): {label_0}")


if __name__ == "__main__":
    main()
