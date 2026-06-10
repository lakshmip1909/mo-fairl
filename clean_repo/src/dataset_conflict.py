"""
src/dataset_conflict.py

Dataset class for conflict pairs.

Extends PythiaPreferenceDataset to also return:
    is_conflict : bool   — whether this pair has objectives disagreeing
    conflict_type: str   — e.g. "safety_vs_math", "none"

Both fields are used by evaluate_conflict.py to split metrics
by conflict vs non-conflict.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer

OBJECTIVES = ["toxicity", "math", "code"]
K          = len(OBJECTIVES)


class ConflictPreferenceDataset(Dataset):
    """
    Each item returns:
        input_ids_a, attention_mask_a  : tokenised response A
        input_ids_b, attention_mask_b  : tokenised response B
        rho        : FloatTensor [K]   {-1.0, +1.0}  (0.0 where null)
        mask       : BoolTensor  [K]   True where label is not null
        task       : str
        is_conflict: bool
        conflict_type: str
    """

    def __init__(
        self,
        jsonl_path:     str,
        tokenizer_name: str = "EleutherAI/pythia-410m",
        max_length:     int = 512,
    ):
        self.max_length = max_length
        self.tokenizer  = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.samples: list[dict] = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        n_conflict = sum(1 for s in self.samples if s.get("is_conflict", False))
        print(f"[Dataset] Loaded {len(self.samples)} samples from {jsonl_path}")
        print(f"[Dataset] Conflict pairs: {n_conflict}  ({100*n_conflict/max(len(self.samples),1):.1f}%)")

    @staticmethod
    def _make_text(prompt: str, response: str) -> str:
        return f"Prompt: {prompt}\n\nResponse: {response}"

    @staticmethod
    def _parse_rho(raw_labels: dict) -> tuple[torch.Tensor, torch.Tensor]:
        rho, mask = [], []
        for obj in OBJECTIVES:
            val = raw_labels.get(obj, None)
            if val is None:
                rho.append(0.0)
                mask.append(False)
            else:
                rho.append(1.0 if val == 1 else -1.0)
                mask.append(True)
        return torch.tensor(rho, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)

    def _tokenise(self, text: str) -> dict:
        enc = self.tokenizer(
            text,
            padding        = "max_length",
            truncation     = True,
            max_length     = self.max_length,
            return_tensors = "pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        tok_a = self._tokenise(self._make_text(s["prompt"], s["response_a"]))
        tok_b = self._tokenise(self._make_text(s["prompt"], s["response_b"]))
        rho, mask = self._parse_rho(s["labels"])

        return {
            "input_ids_a":      tok_a["input_ids"],
            "attention_mask_a": tok_a["attention_mask"],
            "input_ids_b":      tok_b["input_ids"],
            "attention_mask_b": tok_b["attention_mask"],
            "rho":              rho,
            "mask":             mask,
            "task":             s.get("task", "conflict"),
            "is_conflict":      s.get("is_conflict", False),
            "conflict_type":    s.get("conflict_type", "unknown"),
            "global_label":     s.get("global_label", None),
        }


def conflict_collate_fn(batch: list[dict]) -> dict:
    return {
        "input_ids_a":      torch.stack([b["input_ids_a"]      for b in batch]),
        "attention_mask_a": torch.stack([b["attention_mask_a"] for b in batch]),
        "input_ids_b":      torch.stack([b["input_ids_b"]      for b in batch]),
        "attention_mask_b": torch.stack([b["attention_mask_b"] for b in batch]),
        "rho":              torch.stack([b["rho"]              for b in batch]),
        "mask":             torch.stack([b["mask"]             for b in batch]),
        "task":             [b["task"]          for b in batch],
        "is_conflict":      [b["is_conflict"]   for b in batch],
        "conflict_type":    [b["conflict_type"] for b in batch],
        "global_label":     [b["global_label"] for b in batch],
    }


def build_conflict_dataloaders(
    jsonl_path:     str,
    tokenizer_name: str,
    batch_size:     int,
    train_split:    float = 0.8,
    val_split:      float = 0.1,
    seed:           int   = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:

    dataset = ConflictPreferenceDataset(jsonl_path, tokenizer_name=tokenizer_name)
    n       = len(dataset)
    n_train = int(n * train_split)
    n_val   = int(n * val_split)
    n_test  = n - n_train - n_val

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=gen)

    kwargs = dict(collate_fn=conflict_collate_fn, num_workers=2, pin_memory=True)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kwargs)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kwargs)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kwargs)

    print(f"[DataLoader] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    return train_dl, val_dl, test_dl
