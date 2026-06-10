"""
scripts/real/build_toxicity_pairs.py

Builds toxicity preference pairs from the civil_comments dataset.

Strategy:
    - Load civil_comments from HuggingFace datasets
    - Score each comment by its existing toxicity label (0.0–1.0)
    - Threshold:
        toxic    : toxicity >= 0.5
        non-toxic: toxicity <= 0.1
    - Pair a non-toxic comment (response_a = safe) with a toxic comment
      (response_b = unsafe) under the same prompt template
    - Label: toxicity=1 (A safer), math=null, code=null

Output: data/real/toxicity_pairs.jsonl
"""

import json
import random
import os
from pathlib import Path

random.seed(42)

PROMPT_TEMPLATES = [
    "Please respond to the following statement: \"{text}\"",
    "A user wrote: \"{text}\". How would you respond?",
    "Comment: \"{text}\". What do you think?",
    "Respond to this message: \"{text}\"",
    "Someone posted: \"{text}\". Write a reply.",
]

def make_prompt(text: str) -> str:
    template = random.choice(PROMPT_TEMPLATES)
    # Truncate very long texts for the prompt
    short = text[:120] + "..." if len(text) > 120 else text
    return template.format(text=short)


def build_toxicity_pairs(n: int = 2000, out_path: str = "data/real/toxicity_pairs.jsonl"):
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: run `pip install datasets` first.")
        return

    print("Loading civil_comments dataset...")
    # civil_comments: each row has 'text' and 'toxicity' (float 0-1)
    ds = load_dataset("civil_comments", split="train", trust_remote_code=True)
    print(f"  Loaded {len(ds)} comments")

    # Separate into toxic / non-toxic
    toxic     = [r for r in ds if r["toxicity"] >= 0.5]
    non_toxic = [r for r in ds if r["toxicity"] <= 0.1]

    print(f"  Toxic (≥0.5)    : {len(toxic):,}")
    print(f"  Non-toxic (≤0.1): {len(non_toxic):,}")

    random.shuffle(toxic)
    random.shuffle(non_toxic)

    # Cap to what we need
    n_pairs = min(n, len(toxic), len(non_toxic))

    samples = []
    for i in range(n_pairs):
        safe_text  = non_toxic[i]["text"].strip()
        toxic_text = toxic[i]["text"].strip()

        # Prompt: use safe text as the original message being replied to
        prompt = make_prompt(safe_text[:100])

        if random.random() < 0.5:
            # A = safe response, B = toxic response
            sample = {
                "prompt":     prompt,
                "response_a": safe_text,
                "response_b": toxic_text,
                "labels":     {"toxicity": 1, "math": None, "code": None},
                "task":       "toxicity",
                "source":     "civil_comments",
                "meta": {
                    "toxicity_a": non_toxic[i]["toxicity"],
                    "toxicity_b": toxic[i]["toxicity"],
                },
            }
        else:
            # A = toxic, B = safe
            sample = {
                "prompt":     prompt,
                "response_a": toxic_text,
                "response_b": safe_text,
                "labels":     {"toxicity": 0, "math": None, "code": None},
                "task":       "toxicity",
                "source":     "civil_comments",
                "meta": {
                    "toxicity_a": toxic[i]["toxicity"],
                    "toxicity_b": non_toxic[i]["toxicity"],
                },
            }
        samples.append(sample)

    random.shuffle(samples)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    label_1 = sum(1 for s in samples if s["labels"]["toxicity"] == 1)
    label_0 = sum(1 for s in samples if s["labels"]["toxicity"] == 0)
    print(f"\nSaved {len(samples)} pairs to {out_path}")
    print(f"  Label=1 (A safer): {label_1}  |  Label=0 (B safer): {label_0}")


if __name__ == "__main__":
    build_toxicity_pairs(n=2000)
