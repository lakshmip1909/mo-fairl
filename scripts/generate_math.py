"""
scripts/generate_math.py

Generates math preference pairs.
Correct solution vs wrong solution (perturbed answer or reasoning mistake).
Label: math = 1 always when response_a is correct (swapped 50% of the time).

Output: data/math_pairs.jsonl
"""

import json
import random
import os
import operator

random.seed(42)


# ── Problem templates ──────────────────────────────────────────────────────────

def make_arithmetic_pair():
    """Simple arithmetic: a op b"""
    ops = [
        ("+",  operator.add,  "add"),
        ("-",  operator.sub,  "subtract"),
        ("×",  operator.mul,  "multiply"),
    ]
    symbol, fn, _ = random.choice(ops)
    a = random.randint(2, 99)
    b = random.randint(2, 99)
    correct = fn(a, b)

    # Wrong answer: perturb by small amount
    perturbation = random.choice([-random.randint(1, 10), random.randint(1, 10)])
    wrong = correct + perturbation

    prompt = f"Calculate: {a} {symbol} {b}"
    correct_resp = (
        f"To calculate {a} {symbol} {b}:\n"
        f"{a} {symbol} {b} = {correct}\n"
        f"The answer is {correct}."
    )
    wrong_resp = (
        f"To calculate {a} {symbol} {b}:\n"
        f"{a} {symbol} {b} = {wrong}\n"
        f"The answer is {wrong}."
    )
    return prompt, correct_resp, wrong_resp


def make_word_problem_pair():
    """Simple word problems"""
    templates = [
        lambda: _apples_problem(),
        lambda: _speed_problem(),
        lambda: _percentage_problem(),
        lambda: _ratio_problem(),
        lambda: _area_problem(),
    ]
    return random.choice(templates)()


def _apples_problem():
    n_start  = random.randint(5, 50)
    n_buy    = random.randint(1, 20)
    n_give   = random.randint(1, n_start)
    correct  = n_start + n_buy - n_give
    wrong    = correct + random.choice([-n_give, n_buy, -n_buy, n_give])  # common mistake

    prompt = (
        f"Alice has {n_start} apples. She buys {n_buy} more and then gives "
        f"{n_give} to her friend. How many apples does Alice have now?"
    )
    correct_resp = (
        f"Starting apples: {n_start}\n"
        f"After buying {n_buy}: {n_start} + {n_buy} = {n_start+n_buy}\n"
        f"After giving away {n_give}: {n_start+n_buy} - {n_give} = {correct}\n"
        f"Alice has {correct} apples."
    )
    wrong_resp = (
        f"Starting apples: {n_start}\n"
        f"After buying {n_buy}: {n_start} + {n_buy} = {n_start+n_buy}\n"
        f"Alice has {wrong} apples."  # skips or miscalculates the subtraction
    )
    return prompt, correct_resp, wrong_resp


def _speed_problem():
    speed    = random.randint(40, 120)
    time     = random.randint(1, 8)
    correct  = speed * time
    wrong    = speed + time  # classic wrong operation

    prompt   = f"A car travels at {speed} km/h for {time} hours. How far does it travel?"
    correct_resp = (
        f"Distance = speed × time\n"
        f"Distance = {speed} × {time} = {correct} km"
    )
    wrong_resp = (
        f"Distance = speed + time\n"
        f"Distance = {speed} + {time} = {wrong} km"
    )
    return prompt, correct_resp, wrong_resp


def _percentage_problem():
    total    = random.randint(100, 1000)
    pct      = random.choice([10, 15, 20, 25, 30, 50])
    correct  = (pct / 100) * total
    wrong    = total - pct  # wrong: subtraction instead of percentage

    prompt   = f"What is {pct}% of {total}?"
    correct_resp = (
        f"{pct}% of {total} = ({pct}/100) × {total} = {correct}"
    )
    wrong_resp = (
        f"{pct}% of {total} = {total} - {pct} = {wrong}"
    )
    return prompt, correct_resp, wrong_resp


def _ratio_problem():
    a = random.randint(2, 10)
    b = random.randint(2, 10)
    total = (a + b) * random.randint(2, 10)
    share_a = (a / (a + b)) * total
    wrong   = total / a  # wrong: divides by one part only

    prompt = (
        f"Divide {total} in the ratio {a}:{b}. What is the larger share?"
    )
    larger = max(share_a, (b / (a + b)) * total)
    correct_resp = (
        f"Total parts = {a} + {b} = {a+b}\n"
        f"Each part = {total} / {a+b} = {total/(a+b):.2f}\n"
        f"Larger share = {max(a,b)} × {total/(a+b):.2f} = {larger:.2f}"
    )
    wrong_resp = (
        f"Larger share = {total} / {a} = {wrong:.2f}"
    )
    return prompt, correct_resp, wrong_resp


def _area_problem():
    w = random.randint(3, 30)
    h = random.randint(3, 30)
    correct = w * h
    wrong   = 2 * (w + h)  # perimeter instead of area

    prompt = f"What is the area of a rectangle with width {w} and height {h}?"
    correct_resp = f"Area = width × height = {w} × {h} = {correct} square units"
    wrong_resp   = f"Area = 2 × (width + height) = 2 × ({w} + {h}) = {wrong} square units"
    return prompt, correct_resp, wrong_resp


# ── Reasoning / logic problems ─────────────────────────────────────────────────

REASONING_PAIRS = [
    (
        "If all cats are mammals and all mammals are animals, are all cats animals?",
        "Yes. Since all cats are mammals (premise 1) and all mammals are animals (premise 2), it follows by transitivity that all cats are animals. This is a valid syllogism.",
        "Not necessarily. Cats are cats, mammals are mammals, and animals are animals. They are different categories so you cannot be sure.",
    ),
    (
        "A bat and ball cost £1.10 together. The bat costs £1 more than the ball. How much does the ball cost?",
        "Let the ball cost x pence. The bat costs x + 100 pence. Together: x + (x + 100) = 110. So 2x = 10, x = 5. The ball costs 5p (£0.05).",
        "The ball costs 10p. Because £1.10 - £1.00 = £0.10.",
    ),
    (
        "There are 3 boxes: one has apples, one has oranges, one has both. All labels are wrong. You pick one fruit from the box labelled 'both'. How do you identify all boxes?",
        "Since all labels are wrong, the box labelled 'both' contains only apples or only oranges. Pick one fruit: if it's an apple, that box is the apples box. The box labelled 'oranges' (wrong label) must be 'both', and the box labelled 'apples' must be 'oranges'.",
        "You cannot determine anything from one pick. You need to open all boxes to know.",
    ),
    (
        "How many 9s appear in numbers from 1 to 100?",
        "9, 19, 29, 39, 49, 59, 69, 79, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99. Count: 9 appears as units digit 10 times (9,19,...,99) and as tens digit 10 times (90-99). But 99 has two 9s. Total = 20 nines.",
        "There are 10 nines: 9, 19, 29, 39, 49, 59, 69, 79, 89, 99.",
    ),
    (
        "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?",
        "Each machine makes 1 widget in 5 minutes. So 100 machines each make 1 widget in 5 minutes, producing 100 widgets total in 5 minutes.",
        "It takes 100 minutes because you have 100 machines making 100 widgets.",
    ),
]


def generate_math_pairs(n: int = 2000) -> list[dict]:
    samples = []

    for i in range(n):
        if i % 3 == 0 and i // 3 < len(REASONING_PAIRS):
            prompt, correct, wrong = REASONING_PAIRS[i // 3 % len(REASONING_PAIRS)]
        elif i % 5 == 0:
            prompt, correct, wrong = make_word_problem_pair()
        else:
            prompt, correct, wrong = make_arithmetic_pair()

        if random.random() < 0.5:
            sample = {
                "prompt":     prompt,
                "response_a": correct,
                "response_b": wrong,
                "labels": {"toxicity": None, "math": 1, "code": None},
                "task": "math",
            }
        else:
            sample = {
                "prompt":     prompt,
                "response_a": wrong,
                "response_b": correct,
                "labels": {"toxicity": None, "math": 0, "code": None},
                "task": "math",
            }

        samples.append(sample)

    random.shuffle(samples)
    return samples


def main():
    os.makedirs("data", exist_ok=True)
    pairs = generate_math_pairs(n=2000)

    out_path = "data/math_pairs.jsonl"
    with open(out_path, "w") as f:
        for item in pairs:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(pairs)} math pairs to {out_path}")
    label_1 = sum(1 for p in pairs if p["labels"]["math"] == 1)
    label_0 = sum(1 for p in pairs if p["labels"]["math"] == 0)
    print(f"  Label=1 (A correct): {label_1}  |  Label=0 (B correct): {label_0}")


if __name__ == "__main__":
    main()
