"""
scripts/generate_multiobjective.py  (Version 2)

Generates TRUE multi-objective preference pairs where every sample receives
labels for ALL THREE objectives: toxicity, math, code.

The key design principle:
    Each pair is deliberately constructed so that objectives can CONFLICT.
    Response A and B differ along one "primary" axis (what makes them a
    meaningful pair), but we score ALL objectives for both responses.

Three types of cross-objective pairs:

TYPE 1 — Safe-but-wrong vs Unsafe-but-correct  (toxicity ↔ math conflict)
    response_a: safe, respectful tone + WRONG mathematical answer
    response_b: rude/dismissive tone    + CORRECT mathematical answer
    labels:  toxicity=1  (A safer),  math=0  (B more correct),  code=null→0 (neutral)

TYPE 2 — Safe-but-buggy vs Neutral-and-correct  (toxicity ↔ code conflict)
    response_a: safe, encouraging tone  + BUGGY code
    response_b: blunt/dismissive tone   + CORRECT code
    labels:  toxicity=1 (A safer),  math=0 (neutral, A loses),  code=0 (B better)

TYPE 3 — Correct-math but unsafe vs Safe but wrong  (math ↔ toxicity)
    Same as TYPE 1 but framed as a code-math hybrid problem.

TYPE 4 — All-agree  (A wins on all three)
    response_a: safe + correct maths + correct code
    response_b: unsafe + wrong maths + buggy code
    labels:  toxicity=1, math=1, code=1

TYPE 5 — Split decision  (A wins two, loses one)
    labels like [1, 0, 1] or [0, 1, 1] etc.

Output: data/multiobjective_pairs.jsonl
Every sample has:
    labels = {"toxicity": int, "math": int, "code": int}  — NO nulls.
"""

import json
import random
import os

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────────────────

# Safe/respectful tones
SAFE_INTROS = [
    "Great question! Let me help you with that.",
    "Sure, happy to work through this with you.",
    "Let's take a careful look at this step by step.",
    "Of course! Here's how I'd approach it.",
    "That's a nice problem. Let me explain.",
]

# Toxic/dismissive tones
TOXIC_INTROS = [
    "This is so obvious. Only an idiot would ask this.",
    "Ugh, this again. Fine.",
    "Are you serious? This is basic stuff.",
    "What a waste of time. Here:",
    "This is pathetically easy.",
]

# Neutral/blunt tones (not toxic, not warm — used for TYPE 2/3)
NEUTRAL_INTROS = [
    "Here is the answer.",
    "The solution is as follows.",
    "Answer:",
    "Result:",
    "Here:",
]


def safe_intro():
    return random.choice(SAFE_INTROS)


def toxic_intro():
    return random.choice(TOXIC_INTROS)


def neutral_intro():
    return random.choice(NEUTRAL_INTROS)


# ─────────────────────────────────────────────────────────────────────────────
# MATH PROBLEM GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

import operator as op

OPS = [("+", op.add), ("-", op.sub), ("×", op.mul)]


def random_arithmetic():
    symbol, fn = random.choice(OPS)
    a = random.randint(3, 99)
    b = random.randint(3, 49)
    correct = fn(a, b)
    # Wrong answer: small perturbation
    wrong = correct + random.choice([-random.randint(1, 15), random.randint(1, 15)])
    prompt = f"Calculate {a} {symbol} {b}. Show your working."
    return prompt, a, symbol, b, correct, wrong


def random_percentage():
    pct   = random.choice([10, 15, 20, 25, 30, 40, 50])
    total = random.randint(50, 500)
    correct = int((pct / 100) * total)
    wrong   = total - pct   # classic subtraction mistake
    prompt  = f"What is {pct}% of {total}?"
    return prompt, correct, wrong


def random_area():
    w = random.randint(4, 30)
    h = random.randint(4, 30)
    correct = w * h
    wrong   = 2 * (w + h)   # perimeter instead of area
    prompt  = f"Find the area of a rectangle with width {w} cm and height {h} cm."
    return prompt, correct, wrong


# ─────────────────────────────────────────────────────────────────────────────
# CODE SNIPPET GENERATORS  (correct + buggy versions)
# ─────────────────────────────────────────────────────────────────────────────

CODE_PAIRS = [
    # (problem_description, correct_snippet, buggy_snippet, bug_note)
    (
        "Write a Python function to reverse a string.",
        "def reverse_string(s):\n    return s[::-1]",
        "def reverse_string(s):\n    return s[::1]",
        "off-by-one in slice",
    ),
    (
        "Write a Python function to compute the factorial of n.",
        "def factorial(n):\n    if n == 0: return 1\n    result = 1\n    for i in range(1, n+1):\n        result *= i\n    return result",
        "def factorial(n):\n    if n == 0: return 1\n    result = 1\n    for i in range(1, n):\n        result *= i\n    return result",
        "loop stops one step early",
    ),
    (
        "Write a Python function to check if a number is prime.",
        "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True",
        "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, n):\n        if n % i == 0: return False\n    return True",
        "checks all divisors up to n instead of sqrt(n) — wrong for no reason, slow",
    ),
    (
        "Write a Python function to find the maximum in a list.",
        "def find_max(lst):\n    m = lst[0]\n    for x in lst:\n        if x > m: m = x\n    return m",
        "def find_max(lst):\n    m = lst[0]\n    for x in lst:\n        if x < m: m = x\n    return m",
        "wrong comparison operator — returns minimum",
    ),
    (
        "Write a Python function to count vowels in a string.",
        "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')",
        "def count_vowels(s):\n    return sum(1 for c in s if c in 'aeiou')",
        "missing .lower() — misses uppercase vowels",
    ),
    (
        "Write a Python function that returns the sum of a list.",
        "def sum_list(lst):\n    total = 0\n    for x in lst: total += x\n    return total",
        "def sum_list(lst):\n    total = 0\n    for x in lst: total += x\n    return total - lst[0]",
        "incorrectly subtracts first element",
    ),
    (
        "Write a Python function for binary search.",
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        mid = (lo+hi)//2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: lo = mid+1\n        else: hi = mid-1\n    return -1",
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr)-1\n    while lo < hi:\n        mid = (lo+hi)//2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: lo = mid+1\n        else: hi = mid-1\n    return -1",
        "while lo < hi misses the last element",
    ),
    (
        "Write a Python function to check if a string is a palindrome.",
        "def is_palindrome(s):\n    return s == s[::-1]",
        "def is_palindrome(s):\n    return s == s[1:]",
        "compares s to s without first char — wrong",
    ),
]


def random_code_pair():
    return random.choice(CODE_PAIRS)


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE TYPE GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def make_type1_safe_wrong_vs_unsafe_correct():
    """
    TYPE 1: toxicity conflict with math.

    response_a: SAFE tone + WRONG math answer
    response_b: TOXIC tone + CORRECT math answer

    labels: toxicity=1 (A safer), math=0 (B correct), code=0 (neither has code, B "wins" as neutral)

    Genuine conflict: you want A for safety but B for correctness.
    """
    mode = random.choice(["arithmetic", "percentage", "area"])

    if mode == "arithmetic":
        prompt, a, sym, b, correct, wrong = random_arithmetic()
        correct_working = f"{a} {sym} {b} = {correct}"
        wrong_working   = f"{a} {sym} {b} = {wrong}"
    elif mode == "percentage":
        prompt, correct, wrong = random_percentage()
        correct_working = f"Answer = {correct}"
        wrong_working   = f"Answer = {wrong}"
    else:
        prompt, correct, wrong = random_area()
        correct_working = f"Area = {correct} cm²"
        wrong_working   = f"Area = {wrong} cm²"

    response_a = (
        f"{safe_intro()} Let me work through this carefully.\n"
        f"{wrong_working}\n"
        f"So the answer is {wrong}."
    )
    response_b = (
        f"{toxic_intro()}\n"
        f"{correct_working}\n"
        f"The answer is {correct}. Obviously."
    )

    return {
        "prompt":     prompt,
        "response_a": response_a,
        "response_b": response_b,
        "labels":     {"toxicity": 1, "math": 0, "code": 0},
        "task":       "multiobjective",
        "conflict_type": "safety_vs_math",
        "primary_conflict": "A safer but mathematically wrong; B correct but toxic",
    }


def make_type2_safe_buggy_vs_blunt_correct():
    """
    TYPE 2: toxicity conflict with code.

    response_a: SAFE/encouraging tone + BUGGY code
    response_b: BLUNT/dismissive tone + CORRECT code

    labels: toxicity=1 (A safer), math=0 (no math, A loses tiebreak), code=0 (B correct)

    Conflict: A is nicer but the code is wrong. B is harsh but working.
    """
    problem, correct_code, buggy_code, bug_note = random_code_pair()

    response_a = (
        f"{safe_intro()} Here's my solution:\n\n"
        f"```python\n{buggy_code}\n```\n\n"
        f"Hope that helps! Let me know if you have questions."
    )
    response_b = (
        f"{neutral_intro()}\n\n"
        f"```python\n{correct_code}\n```"
    )

    return {
        "prompt":     problem,
        "response_a": response_a,
        "response_b": response_b,
        "labels":     {"toxicity": 1, "math": 0, "code": 0},
        "task":       "multiobjective",
        "conflict_type": "safety_vs_code",
        "primary_conflict": "A safer/warmer but buggy code; B blunt but correct code",
    }


def make_type3_correct_code_toxic_vs_wrong_code_safe():
    """
    TYPE 3: code-toxicity conflict (B wins code, A wins safety).

    response_a: SAFE tone + WRONG code
    response_b: TOXIC tone + CORRECT code

    labels: toxicity=1 (A safer), math=0, code=0 (B correct)
    """
    problem, correct_code, buggy_code, _ = random_code_pair()

    response_a = (
        f"{safe_intro()} Here's an approach:\n\n"
        f"```python\n{buggy_code}\n```\n\n"
        f"This should work for most cases!"
    )
    response_b = (
        f"{toxic_intro()}\n\n"
        f"```python\n{correct_code}\n```\n\n"
        f"There. Done."
    )

    return {
        "prompt":     problem,
        "response_a": response_a,
        "response_b": response_b,
        "labels":     {"toxicity": 1, "math": 0, "code": 0},
        "task":       "multiobjective",
        "conflict_type": "safety_vs_code_v2",
        "primary_conflict": "A safe but buggy; B toxic but correct",
    }


def make_type4_a_wins_all():
    """
    TYPE 4: A wins on all three objectives.

    response_a: SAFE + CORRECT math + CORRECT code
    response_b: TOXIC + WRONG math + BUGGY code

    labels: toxicity=1, math=1, code=1

    No conflict — used to balance dataset and prove model can learn clean signal.
    """
    # Pick a combined math+code prompt
    _, a, sym, b, correct, wrong = random_arithmetic()
    problem, correct_code, buggy_code, _ = random_code_pair()

    prompt = (
        f"Calculate {a} {sym} {b}, and also write a Python function to reverse a string."
    )

    response_a = (
        f"{safe_intro()}\n\n"
        f"For the maths: {a} {sym} {b} = {correct}\n\n"
        f"For the code:\n```python\n{correct_code}\n```"
    )
    response_b = (
        f"{toxic_intro()}\n\n"
        f"Maths: {a} {sym} {b} = {wrong} (whatever)\n\n"
        f"Code:\n```python\n{buggy_code}\n```"
    )

    return {
        "prompt":     prompt,
        "response_a": response_a,
        "response_b": response_b,
        "labels":     {"toxicity": 1, "math": 1, "code": 1},
        "task":       "multiobjective",
        "conflict_type": "none",
        "primary_conflict": "A wins all objectives",
    }


def make_type5_b_wins_all():
    """
    TYPE 5: B wins on all three objectives (label=0 for all).

    Symmetric counterpart to Type 4 — important for balance.
    """
    _, a, sym, b, correct, wrong = random_arithmetic()
    problem, correct_code, buggy_code, _ = random_code_pair()

    prompt = (
        f"Calculate {a} {sym} {b}, and also write a Python function to check if a number is prime."
    )

    response_a = (
        f"{toxic_intro()}\n\n"
        f"Maths: {a} {sym} {b} = {wrong}\n\n"
        f"Code:\n```python\n{buggy_code}\n```"
    )
    response_b = (
        f"{safe_intro()}\n\n"
        f"For the maths: {a} {sym} {b} = {correct}\n\n"
        f"For the code:\n```python\n{correct_code}\n```"
    )

    return {
        "prompt":     prompt,
        "response_a": response_a,
        "response_b": response_b,
        "labels":     {"toxicity": 0, "math": 0, "code": 0},
        "task":       "multiobjective",
        "conflict_type": "none",
        "primary_conflict": "B wins all objectives",
    }


def make_type6_math_wins_a_code_wins_b():
    """
    TYPE 6: A wins math, B wins code. Safety neutral (both safe).

    response_a: SAFE + CORRECT math + BUGGY code
    response_b: SAFE + WRONG math  + CORRECT code

    labels: toxicity=1 (both safe, slight A preference), math=1, code=0

    Conflict: math vs code.
    """
    _, a, sym, b, correct, wrong = random_arithmetic()
    problem, correct_code, buggy_code, _ = random_code_pair()

    prompt = (
        f"Solve: {a} {sym} {b}. Also write a Python function to find the maximum of a list."
    )

    response_a = (
        f"{safe_intro()}\n\n"
        f"The answer is {a} {sym} {b} = {correct}.\n\n"
        f"Here is the function:\n```python\n{buggy_code}\n```"
    )
    response_b = (
        f"{safe_intro()}\n\n"
        f"The answer is {a} {sym} {b} = {wrong}.\n\n"
        f"Here is the function:\n```python\n{correct_code}\n```"
    )

    return {
        "prompt":     prompt,
        "response_a": response_a,
        "response_b": response_b,
        "labels":     {"toxicity": 1, "math": 1, "code": 0},
        "task":       "multiobjective",
        "conflict_type": "math_vs_code",
        "primary_conflict": "A correct math but buggy code; B wrong math but correct code",
    }


def make_type7_code_wins_a_math_wins_b():
    """
    TYPE 7: Reverse of TYPE 6.

    response_a: SAFE + WRONG math  + CORRECT code
    response_b: SAFE + CORRECT math + BUGGY code

    labels: toxicity=1, math=0, code=1
    """
    _, a, sym, b, correct, wrong = random_arithmetic()
    problem, correct_code, buggy_code, _ = random_code_pair()

    prompt = (
        f"What is {a} {sym} {b}? Also write a Python function to reverse a string."
    )

    response_a = (
        f"{safe_intro()}\n\n"
        f"{a} {sym} {b} = {wrong}\n\n"
        f"Here's the function:\n```python\n{correct_code}\n```"
    )
    response_b = (
        f"{safe_intro()}\n\n"
        f"{a} {sym} {b} = {correct}\n\n"
        f"Here's the function:\n```python\n{buggy_code}\n```"
    )

    return {
        "prompt":     prompt,
        "response_a": response_a,
        "response_b": response_b,
        "labels":     {"toxicity": 1, "math": 0, "code": 1},
        "task":       "multiobjective",
        "conflict_type": "code_vs_math",
        "primary_conflict": "A correct code but wrong math; B wrong code but correct math",
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def make_type8_b_math_only():
    """
    TYPE 8: B wins only math. A wins safety and code.

    response_a: SAFE + BUGGY code  + WRONG math
    response_b: TOXIC + CORRECT code + ... wait, we want B to win ONLY math.

    Actually:
    response_a: SAFE + CORRECT code + WRONG math
    response_b: TOXIC + BUGGY code  + CORRECT math

    labels: toxicity=1 (A safe), math=0 (B correct), code=1 (A correct)
    → pattern [1,0,1] — already covered by type7. Let's instead do:

    response_a: SAFE + BUGGY code + WRONG math  → A loses math and code, wins safety
    response_b: SAFE (neutral) + CORRECT code + CORRECT math → B wins math and code

    labels: toxicity=1 (A slightly safer tone), math=0 (B correct), code=0 (B correct)
    → [1,0,0] — already have this.

    TRUE gap: [0,1,0] means B wins ONLY math, A wins safety and code.
    response_a: SAFE + CORRECT code + WRONG math
    response_b: TOXIC + BUGGY code  + CORRECT math

    labels: toxicity=1 (A), math=0 (B), code=1 (A) → [1,0,1] covered by type7.

    For [0,1,0]: A toxic + buggy code + correct math? No, then A wins math.
    [0,1,0]: B wins tox + math, A wins code.
    response_a: TOXIC + CORRECT code + WRONG math
    response_b: SAFE  + BUGGY code   + CORRECT math
    labels: toxicity=0 (B safer), math=0 (B correct), code=1 (A correct) → [0,0,1] not [0,1,0].

    Let me be explicit: label[k]=1 means A preferred on objective k.
    [0,1,0]: B preferred on tox and code. A preferred on math.
    response_a: SAFE (mid) + CORRECT math + BUGGY code → A math=1 ✓
    response_b: SAFE (warm)+ WRONG math   + CORRECT code → B tox=1 (B warmer) ✓, B code=1 ✓
    labels: toxicity=0 (B warmer), math=1 (A correct math), code=0 (B correct code) → [0,1,0] ✓
    """
    _, a, sym, b, correct, wrong = random_arithmetic()
    problem, correct_code, buggy_code, _ = random_code_pair()

    prompt = f"Calculate {a} {sym} {b} and write a Python function to sum a list."

    response_a = (
        f"{neutral_intro()}\n\n"
        f"{a} {sym} {b} = {correct}\n\n"
        f"```python\n{buggy_code}\n```"
    )
    response_b = (
        f"{safe_intro()}\n\n"
        f"{a} {sym} {b} = {wrong}\n\n"
        f"```python\n{correct_code}\n```"
    )

    return {
        "prompt":     prompt,
        "response_a": response_a,
        "response_b": response_b,
        "labels":     {"toxicity": 0, "math": 1, "code": 0},
        "task":       "multiobjective",
        "conflict_type": "math_only_conflict",
        "primary_conflict": "A correct math but buggy code and blunt; B warm and correct code but wrong math",
    }


def make_type9_b_tox_only():
    """
    TYPE 9: [0,0,1] — A wins ONLY code. B wins tox and math.

    response_a: TOXIC + WRONG math + CORRECT code
    response_b: SAFE  + CORRECT math + BUGGY code

    labels: toxicity=0 (B safer), math=0 (B correct), code=1 (A correct)
    """
    _, a, sym, b, correct, wrong = random_arithmetic()
    problem, correct_code, buggy_code, _ = random_code_pair()

    prompt = f"Compute {a} {sym} {b} and implement a Python function to reverse a string."

    response_a = (
        f"{toxic_intro()}\n\n"
        f"{a} {sym} {b} = {wrong}\n\n"
        f"```python\n{correct_code}\n```"
    )
    response_b = (
        f"{safe_intro()}\n\n"
        f"{a} {sym} {b} = {correct}\n\n"
        f"```python\n{buggy_code}\n```"
    )

    return {
        "prompt":     prompt,
        "response_a": response_a,
        "response_b": response_b,
        "labels":     {"toxicity": 0, "math": 0, "code": 1},
        "task":       "multiobjective",
        "conflict_type": "code_only_conflict",
        "primary_conflict": "A correct code but toxic and wrong math; B safe and correct math but buggy code",
    }


# Type distribution — balanced across all 5 observed + 2 new label patterns.
TYPE_WEIGHTS = [
    (make_type1_safe_wrong_vs_unsafe_correct,  0.08),  # → [1,0,0]
    (make_type2_safe_buggy_vs_blunt_correct,   0.08),  # → [1,0,0]
    (make_type3_correct_code_toxic_vs_wrong_code_safe, 0.06),  # → [1,0,0]
    (make_type4_a_wins_all,                    0.15),  # → [1,1,1]
    (make_type5_b_wins_all,                    0.15),  # → [0,0,0]
    (make_type6_math_wins_a_code_wins_b,       0.15),  # → [1,1,0]
    (make_type7_code_wins_a_math_wins_b,       0.15),  # → [1,0,1]
    (make_type8_b_math_only,                   0.09),  # → [0,1,0]
    (make_type9_b_tox_only,                    0.09),  # → [0,0,1]
]

GENERATORS  = [t[0] for t in TYPE_WEIGHTS]
WEIGHTS_RAW = [t[1] for t in TYPE_WEIGHTS]


def generate_multiobjective_pairs(n: int = 3000) -> list[dict]:
    # Normalise weights
    total = sum(WEIGHTS_RAW)
    weights = [w / total for w in WEIGHTS_RAW]

    samples = []
    for _ in range(n):
        generator = random.choices(GENERATORS, weights=weights, k=1)[0]
        try:
            sample = generator()
            samples.append(sample)
        except Exception as e:
            # Fallback to type4 if something breaks
            samples.append(make_type4_a_wins_all())

    random.shuffle(samples)
    return samples


def print_stats(samples: list[dict]):
    from collections import Counter

    conflict_types = Counter(s["conflict_type"] for s in samples)
    label_patterns = Counter(
        f"tox={s['labels']['toxicity']},math={s['labels']['math']},code={s['labels']['code']}"
        for s in samples
    )

    print(f"\n  Total samples: {len(samples)}")
    print("\n  Conflict type breakdown:")
    for ct, n in sorted(conflict_types.items(), key=lambda x: -x[1]):
        print(f"    {ct:30s}: {n:5d}  ({100*n/len(samples):.1f}%)")

    print("\n  Label pattern breakdown:")
    for pat, n in sorted(label_patterns.items(), key=lambda x: -x[1]):
        print(f"    {pat:35s}: {n:5d}  ({100*n/len(samples):.1f}%)")

    # Count genuine conflicts (objectives disagree)
    conflicts = [
        s for s in samples
        if len(set(s["labels"].values())) > 1
    ]
    print(f"\n  Samples with at least one cross-objective conflict: {len(conflicts)} ({100*len(conflicts)/len(samples):.1f}%)")


def main():
    os.makedirs("data", exist_ok=True)
    samples = generate_multiobjective_pairs(n=3000)
    print_stats(samples)

    out_path = "data/multiobjective_pairs.jsonl"
    with open(out_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"\n  Saved {len(samples)} multi-objective pairs to {out_path}")


if __name__ == "__main__":
    main()
