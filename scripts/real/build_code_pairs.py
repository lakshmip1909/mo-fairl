"""
scripts/real/build_code_pairs.py

Builds code correctness preference pairs from HumanEval.

HumanEval has:
    - prompt      (docstring + function signature)
    - canonical_solution  (correct implementation)
    - test        (unit tests as a string)
    - entry_point (function name)

Strategy:
    For each problem, take the canonical solution as response_a (correct).
    Inject one of N deterministic bug types to create response_b (buggy).
    Verify using the actual unit tests:
        correct solution  → must pass all tests
        buggy solution    → must fail at least one test
    Discard any pair where verification fails.

Bug types (deterministic, not LLM):
    1. off_by_one      : change range(n) to range(n-1) or range(n+1)
    2. wrong_operator  : swap + with -, * with //, > with >=, etc.
    3. wrong_return    : return a constant instead of the variable
    4. missing_base    : remove base case in recursive/iterative logic
    5. wrong_init      : initialise accumulator to wrong value (1 instead of 0)
    6. swap_comparison : flip < to > or == to !=

Output: data/real/code_pairs.jsonl
"""

import json
import random
import re
import os
import sys
import ast
import copy
import textwrap
from pathlib import Path

random.seed(42)


# ── Bug injection via AST ──────────────────────────────────────────────────────

class OffByOneMutator(ast.NodeTransformer):
    """Change the first range() call's stop argument by ±1."""
    def __init__(self):
        self.mutated = False

    def visit_Call(self, node):
        if (not self.mutated
                and isinstance(node.func, ast.Name)
                and node.func.id == "range"
                and len(node.args) >= 1):
            # Modify the stop argument (last positional arg)
            stop_arg = node.args[-1]
            if isinstance(stop_arg, ast.Constant) and isinstance(stop_arg.value, int):
                node.args[-1] = ast.Constant(value=stop_arg.value - 1)
                self.mutated = True
            elif isinstance(stop_arg, ast.BinOp):
                # e.g. range(n+1) → range(n)
                node.args[-1] = stop_arg.left
                self.mutated = True
        self.generic_visit(node)
        return node


class WrongOperatorMutator(ast.NodeTransformer):
    """Swap the first binary operator found."""
    SWAPS = {
        ast.Add:    ast.Sub,
        ast.Sub:    ast.Add,
        ast.Mult:   ast.FloorDiv,
        ast.FloorDiv: ast.Mult,
        ast.Mod:    ast.Add,
    }

    def __init__(self):
        self.mutated = False

    def visit_BinOp(self, node):
        if not self.mutated:
            new_op = self.SWAPS.get(type(node.op))
            if new_op is not None:
                node.op = new_op()
                self.mutated = True
        self.generic_visit(node)
        return node


class WrongComparisonMutator(ast.NodeTransformer):
    """Flip the first comparison operator."""
    SWAPS = {
        ast.Lt:    ast.Gt,
        ast.Gt:    ast.Lt,
        ast.LtE:   ast.GtE,
        ast.GtE:   ast.LtE,
        ast.Eq:    ast.NotEq,
        ast.NotEq: ast.Eq,
    }

    def __init__(self):
        self.mutated = False

    def visit_Compare(self, node):
        if not self.mutated and node.ops:
            new_op = self.SWAPS.get(type(node.ops[0]))
            if new_op is not None:
                node.ops[0] = new_op()
                self.mutated = True
        self.generic_visit(node)
        return node


class WrongReturnMutator(ast.NodeTransformer):
    """Replace the first non-trivial return value with a constant."""
    def __init__(self):
        self.mutated = False

    def visit_Return(self, node):
        if (not self.mutated
                and node.value is not None
                and not isinstance(node.value, ast.Constant)):
            node.value = ast.Constant(value=0)
            self.mutated = True
        return node


class WrongInitMutator(ast.NodeTransformer):
    """Change first assignment of 0 to 1, or [] to None."""
    def __init__(self):
        self.mutated = False

    def visit_Assign(self, node):
        if not self.mutated and len(node.targets) == 1:
            val = node.value
            if isinstance(val, ast.Constant):
                if val.value == 0:
                    node.value = ast.Constant(value=1)
                    self.mutated = True
                elif val.value == []:
                    node.value = ast.Constant(value=None)
                    self.mutated = True
        self.generic_visit(node)
        return node


MUTATORS = [
    OffByOneMutator,
    WrongOperatorMutator,
    WrongComparisonMutator,
    WrongReturnMutator,
    WrongInitMutator,
]


def inject_bug(source_code: str) -> tuple[str | None, str]:
    """
    Try each mutator in random order until one successfully mutates the code.
    Returns (mutated_source, bug_type) or (None, '') if nothing worked.
    """
    mutators = list(MUTATORS)
    random.shuffle(mutators)

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return None, ""

    for MutatorClass in mutators:
        tree_copy = copy.deepcopy(tree)
        mutator = MutatorClass()
        new_tree = mutator.visit(tree_copy)
        if mutator.mutated:
            try:
                ast.fix_missing_locations(new_tree)
                mutated_code = ast.unparse(new_tree)
                if mutated_code != source_code:
                    return mutated_code, MutatorClass.__name__.replace("Mutator", "")
            except Exception:
                continue

    return None, ""


# ── Test runner ────────────────────────────────────────────────────────────────

def run_tests(code: str, test_code: str, entry_point: str, timeout: float = 5.0) -> bool:
    """
    Execute code + test_code in a restricted namespace.
    Returns True if all tests pass.
    """
    import signal

    def _handler(signum, frame):
        raise TimeoutError()

    namespace = {}
    try:
        exec(code, namespace)
    except Exception:
        return False

    try:
        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(int(timeout))
        exec(test_code, namespace)
        signal.alarm(0)
        # Call the check function if present
        check_fn = namespace.get("check")
        if check_fn is not None:
            check_fn(namespace.get(entry_point))
        return True
    except (AssertionError, Exception):
        signal.alarm(0)
        return False
    finally:
        signal.alarm(0)


# ── Main builder ───────────────────────────────────────────────────────────────

def build_code_pairs(n: int = 1000, out_path: str = "data/real/code_pairs.jsonl"):
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: run `pip install datasets` first.")
        return

    print("Loading HumanEval dataset...")
    ds = load_dataset("openai_humaneval", split="test", trust_remote_code=True)
    print(f"  Loaded {len(ds)} HumanEval problems")

    verified_pairs = []
    stats = {"ok": 0, "no_mutation": 0, "correct_fails": 0, "buggy_passes": 0}

    for ex in ds:
        prompt_text    = ex["prompt"]
        canonical      = ex["canonical_solution"]
        test_code      = ex["test"]
        entry_point    = ex["entry_point"]

        full_correct = prompt_text + canonical

        # Verify canonical solution passes
        if not run_tests(full_correct, test_code, entry_point):
            stats["correct_fails"] += 1
            continue

        # Inject bug
        buggy_impl, bug_type = inject_bug(prompt_text + canonical)
        if buggy_impl is None:
            stats["no_mutation"] += 1
            continue

        # Verify buggy solution fails
        if run_tests(buggy_impl, test_code, entry_point):
            stats["buggy_passes"] += 1
            continue

        verified_pairs.append({
            "prompt":     prompt_text,
            "correct":    prompt_text + canonical,
            "buggy":      buggy_impl,
            "bug_type":   bug_type,
            "entry_point": entry_point,
        })
        stats["ok"] += 1

    print(f"\n  Verified pairs  : {stats['ok']}")
    print(f"  No mutation     : {stats['no_mutation']}")
    print(f"  Correct failed  : {stats['correct_fails']}")
    print(f"  Buggy passed    : {stats['buggy_passes']}")

    if not verified_pairs:
        print("ERROR: No verified pairs produced. Check HumanEval loading.")
        return

    # Repeat to reach target n
    samples = []
    for i in range(n):
        pair = verified_pairs[i % len(verified_pairs)]

        if random.random() < 0.5:
            sample = {
                "prompt":     pair["prompt"],
                "response_a": pair["correct"],
                "response_b": pair["buggy"],
                "labels":     {"toxicity": None, "math": None, "code": 1},
                "task":       "code",
                "source":     "humaneval",
                "meta":       {"bug_type": pair["bug_type"], "entry_point": pair["entry_point"]},
            }
        else:
            sample = {
                "prompt":     pair["prompt"],
                "response_a": pair["buggy"],
                "response_b": pair["correct"],
                "labels":     {"toxicity": None, "math": None, "code": 0},
                "task":       "code",
                "source":     "humaneval",
                "meta":       {"bug_type": pair["bug_type"], "entry_point": pair["entry_point"]},
            }
        samples.append(sample)

    random.shuffle(samples)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    label_1 = sum(1 for s in samples if s["labels"]["code"] == 1)
    label_0 = sum(1 for s in samples if s["labels"]["code"] == 0)
    print(f"\nSaved {len(samples)} pairs to {out_path}")
    print(f"  Label=1 (A correct): {label_1}  |  Label=0 (B correct): {label_0}")
    print(f"  Bug types used: { {p['bug_type'] for p in verified_pairs} }")


if __name__ == "__main__":
    build_code_pairs(n=1000)
