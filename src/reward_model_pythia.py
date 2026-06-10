"""
src/reward_model_pythia.py

Multi-objective reward model using Pythia-410M as the encoder.

Architecture:
    [prompt + response] → Pythia-410M (frozen) → last-token hidden state
                        → shared projection MLP
                        → [head_tox, head_math, head_code]  (scalar each)

Why Pythia-410M:
    - Better language understanding than sentence-transformers
    - Smaller than 1B but significantly more capable than GPT-2
    - EleutherAI's Pythia suite is standard in IRL/reward-model research
    - Supports fp16 for HPC training
    - Hidden dim: 1024

Why freeze the base model:
    - We only have ~5k preference pairs — not enough to fine-tune 410M params
    - The projection + reward heads are enough to learn preference structure
    - Keeps training fast and avoids catastrophic forgetting
    - Later extension: unfreeze top N layers for more expressive reward signal
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

OBJECTIVES = ["toxicity", "math", "code"]
K          = len(OBJECTIVES)
MODEL_NAME = "EleutherAI/pythia-410m"


class RewardHead(nn.Module):
    """Single-objective reward head: hidden → scalar."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, input_dim] → [batch] scalar."""
        return self.net(x).squeeze(-1)


class PythiaRewardModel(nn.Module):
    """
    Pythia-410M encoder (frozen) + shared projection + K reward heads.

    The encoder maps a (prompt, response) string to a fixed-dim vector
    by taking the hidden state of the last non-padding token.

    The projection and heads are trained on preference pairs.
    """

    def __init__(
        self,
        model_name:     str   = MODEL_NAME,
        hidden_dim:     int   = 256,
        dropout:        float = 0.1,
        num_objectives: int   = K,
        freeze_encoder: bool  = True,
        max_length:     int   = 512,
    ):
        super().__init__()
        self.max_length = max_length

        print(f"[Model] Loading tokenizer and model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.encoder = AutoModel.from_pretrained(model_name)
        encoder_dim  = self.encoder.config.hidden_size   # 1024 for pythia-410m

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print(f"[Model] Encoder frozen ({sum(p.numel() for p in self.encoder.parameters()):,} params)")

        # Shared projection: encoder_dim → hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # K reward heads
        self.heads = nn.ModuleList([
            RewardHead(hidden_dim, hidden_dim, dropout)
            for _ in range(num_objectives)
        ])

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Trainable params: {trainable:,}")
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Run Pythia and extract the last non-padding token's hidden state.

        Args:
            input_ids      : [batch, seq_len]
            attention_mask : [batch, seq_len]
        Returns:
            [batch, encoder_dim]
        """
        with torch.no_grad() if not any(p.requires_grad for p in self.encoder.parameters()) else torch.enable_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        hidden = outputs.last_hidden_state   # [batch, seq_len, encoder_dim]

        # Last non-padding token position
        seq_lengths = attention_mask.sum(dim=1) - 1   # [batch]
        batch_size  = hidden.size(0)
        last_token  = hidden[torch.arange(batch_size), seq_lengths]   # [batch, encoder_dim]
        return last_token

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward_encoded(self, enc: torch.Tensor) -> torch.Tensor:
        """
        enc: [batch, encoder_dim] (pre-computed)
        Returns: [batch, K]
        """
        h = self.projection(enc)
        return torch.stack([head(h) for head in self.heads], dim=-1)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        input_ids, attention_mask: [batch, seq_len]
        Returns: [batch, K]
        """
        enc = self.encode(input_ids, attention_mask)
        return self.forward_encoded(enc)

    # ── Reward gap ────────────────────────────────────────────────────────────

    def get_reward_gap(
        self,
        input_ids_a: torch.Tensor, attention_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor, attention_mask_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Δᵢᵏ = rₖ(A) - rₖ(B)
        Returns: [batch, K]
        """
        r_a = self.forward(input_ids_a, attention_mask_a)
        r_b = self.forward(input_ids_b, attention_mask_b)
        return r_a - r_b

    def preference_prob(self, *args) -> torch.Tensor:
        """p̂ᵢᵏ = σ(Δᵢᵏ), returns [batch, K]"""
        return torch.sigmoid(self.get_reward_gap(*args))

    def combined_reward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        weights:        torch.Tensor,
    ) -> torch.Tensor:
        """R_w(x,y) = Σₖ wₖ rₖ(x,y), returns [batch]"""
        rewards = self.forward(input_ids, attention_mask)
        return (rewards * weights.unsqueeze(0)).sum(dim=-1)

    # ── Tokenise helper ───────────────────────────────────────────────────────

    def tokenise(self, texts: list[str], device: torch.device) -> dict:
        """Tokenise a list of strings and return tensors on device."""
        enc = self.tokenizer(
            texts,
            padding     = True,
            truncation  = True,
            max_length  = self.max_length,
            return_tensors = "pt",
        )
        return {k: v.to(device) for k, v in enc.items()}


def build_pythia_model(config: dict) -> PythiaRewardModel:
    return PythiaRewardModel(
        model_name     = config["model"].get("encoder", MODEL_NAME),
        hidden_dim     = config["model"]["hidden_dim"],
        dropout        = config["model"]["dropout"],
        num_objectives = config["model"]["num_objectives"],
        freeze_encoder = config["model"].get("freeze_encoder", True),
        max_length     = config["model"].get("max_length", 512),
    )
