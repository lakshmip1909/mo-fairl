"""
src/dataset.py

PreferenceDataset: loads JSONL preference pairs, encodes prompt+response,
handles null labels (masked in loss).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sentence_transformers import SentenceTransformer

# Objective index mapping — must match reward_model.py
OBJECTIVES = ["toxicity", "math", "code"]
OBJ2IDX    = {o: i for i, o in enumerate(OBJECTIVES)}
K          = len(OBJECTIVES)


class PreferenceDataset(Dataset):
    """
    Each item returns:
        enc_a   : FloatTensor [hidden_dim]  — encoded (prompt + response_a)
        enc_b   : FloatTensor [hidden_dim]  — encoded (prompt + response_b)
        labels  : FloatTensor [K]           — {0, 1, NaN} per objective
        mask    : BoolTensor  [K]           — True where label is not null
        task    : str
    """

    def __init__(
        self,
        jsonl_path: str,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_length: int = 512,
        cache_encodings: bool = True,
    ):
        self.path    = Path(jsonl_path)
        self.encoder = SentenceTransformer(encoder_name)
        self.encoder.max_seq_length = max_length
        self.cache   = cache_encodings

        self.samples: list[dict] = []
        self._enc_cache: dict[str, torch.Tensor] = {}

        self._load()
        if cache_encodings:
            self._precompute_encodings()

    # ── Loading ──────────────────────────────────────────────────────────────

    def _load(self):
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        print(f"[Dataset] Loaded {len(self.samples)} samples from {self.path}")

    # ── Encoding ─────────────────────────────────────────────────────────────

    def _encode_text(self, text: str) -> torch.Tensor:
        if self.cache and text in self._enc_cache:
            return self._enc_cache[text]
        vec = self.encoder.encode(text, convert_to_tensor=True, show_progress_bar=False)
        if self.cache:
            self._enc_cache[text] = vec
        return vec

    def _make_text(self, prompt: str, response: str) -> str:
        return f"Prompt: {prompt}\nResponse: {response}"

    def _precompute_encodings(self):
        """Encode all unique texts in one batched pass — much faster."""
        unique_texts = set()
        for s in self.samples:
            unique_texts.add(self._make_text(s["prompt"], s["response_a"]))
            unique_texts.add(self._make_text(s["prompt"], s["response_b"]))
        texts = list(unique_texts)
        print(f"[Dataset] Encoding {len(texts)} unique texts...")
        vecs = self.encoder.encode(texts, batch_size=128, convert_to_tensor=True,
                                   show_progress_bar=True)
        for text, vec in zip(texts, vecs):
            self._enc_cache[text] = vec
        print("[Dataset] Encoding complete.")

    # ── Label parsing ─────────────────────────────────────────────────────────

    @staticmethod
    def _parse_labels(raw_labels: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            labels: FloatTensor [K] — 0.0 or 1.0 (NaN where null)
            mask  : BoolTensor  [K] — True where label is not null
        """
        labels = []
        mask   = []
        for obj in OBJECTIVES:
            val = raw_labels.get(obj, None)
            if val is None:
                labels.append(float("nan"))
                mask.append(False)
            else:
                labels.append(float(val))
                mask.append(True)
        return torch.tensor(labels, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        text_a = self._make_text(s["prompt"], s["response_a"])
        text_b = self._make_text(s["prompt"], s["response_b"])

        enc_a = self._encode_text(text_a)
        enc_b = self._encode_text(text_b)

        labels, mask = self._parse_labels(s["labels"])

        return {
            "enc_a":  enc_a,
            "enc_b":  enc_b,
            "labels": labels,   # [K]
            "mask":   mask,     # [K]
            "task":   s.get("task", "unknown"),
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        "enc_a":  torch.stack([b["enc_a"]  for b in batch]),
        "enc_b":  torch.stack([b["enc_b"]  for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "mask":   torch.stack([b["mask"]   for b in batch]),
        "task":   [b["task"] for b in batch],
    }


def build_dataloaders(
    jsonl_path: str,
    encoder_name: str,
    batch_size: int,
    train_split: float = 0.8,
    val_split:   float = 0.1,
    seed:        int   = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns train, val, test DataLoaders.
    """
    dataset = PreferenceDataset(jsonl_path, encoder_name=encoder_name)
    n       = len(dataset)
    n_train = int(n * train_split)
    n_val   = int(n * val_split)
    n_test  = n - n_train - n_val

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=gen)

    kwargs = dict(collate_fn=collate_fn, num_workers=0, pin_memory=False)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kwargs)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kwargs)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kwargs)

    print(f"[DataLoader] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    return train_dl, val_dl, test_dl
