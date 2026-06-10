"""
src/reward_model_pythia_conflict.py

Extends PythiaRewardModel with learnable objective weights w_phi.

This implements the missing piece from the notes:

    P(y+ > y- | x)  =  sigma( w^T · Delta )

where:
    Delta  = [Delta_tox, Delta_math, Delta_code]   reward gap vector  [batch, K]
    w      = softmax(w_phi)                         learnable weights  [K]

Previously w was fixed at [1/3, 1/3, 1/3] and never updated.
Now w_phi is a trainable nn.Parameter optimised jointly with the reward heads.

This is the core missing piece identified in the notes (RQ1):
    "recover the latent weights w* from preferences"

Architecture:
    [prompt + response]
          ↓
    Pythia-410M (frozen)
          ↓  last-token hidden state  [batch, 1024]
    Shared projection MLP  →  [batch, 256]
          ↓
    Head_tox  →  r_tox(x,y)   scalar  ┐
    Head_math →  r_math(x,y)  scalar  ├→  r(x,y)  [batch, K]
    Head_code →  r_code(x,y)  scalar  ┘
          ↓
    Delta = r(A) - r(B)               [batch, K]
          ↓
    w = softmax(w_phi)                [K]     ← NEW: learned
          ↓
    global_score = w^T · Delta        [batch] ← NEW: global preference
          ↓
    P(A > B) = sigma(global_score)    [batch] ← NEW: global probability
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

OBJECTIVES = ["toxicity", "math", "code"]
K          = len(OBJECTIVES)
MODEL_NAME = "EleutherAI/pythia-410m"


class RewardHead(nn.Module):
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
        return self.net(x).squeeze(-1)


class PythiaConflictRewardModel(nn.Module):
    """
    Pythia-410M + shared projection + K reward heads + learnable w_phi.

    Two modes of preference prediction:

    1. Per-objective (same as before):
       p_hat_i^k = sigma(Delta_i^k)
       Used for: per-objective loss, per-objective accuracy

    2. Global (NEW — implements the notes):
       score_i = w^T · Delta_i  =  sum_k  w_k * Delta_i^k
       P_i     = sigma(score_i)
       Used for: global preference loss, latent weight recovery
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
        self.max_length    = max_length
        self.num_objectives = num_objectives

        print(f"[Model] Loading: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.encoder    = AutoModel.from_pretrained(model_name)
        encoder_dim     = self.encoder.config.hidden_size   # 1024

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            frozen = sum(p.numel() for p in self.encoder.parameters())
            print(f"[Model] Encoder frozen: {frozen:,} params")

        # Shared projection
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

        # ── Learnable objective weights w_phi ──────────────────────────────
        # Initialise uniform: softmax([0,0,0]) = [1/3, 1/3, 1/3]
        # w_phi is unconstrained; we apply softmax to get valid weights.
        # This means w is always on the probability simplex: w_k > 0, sum=1.
        self.w_phi = nn.Parameter(torch.zeros(num_objectives))

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Trainable params: {trainable:,}  (includes w_phi: {num_objectives})")

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ── w property ────────────────────────────────────────────────────────────

    @property
    def w(self) -> torch.Tensor:
        """
        Returns normalised weights w = softmax(w_phi).
        Always sums to 1, always positive.
        Shape: [K]
        """
        return F.softmax(self.w_phi, dim=0)

    def get_weights(self) -> dict:
        """Returns current weights as a readable dict."""
        w = self.w.detach().cpu()
        return {obj: round(float(w[i]), 4) for i, obj in enumerate(OBJECTIVES)}

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Last non-padding token hidden state. Returns [batch, encoder_dim]."""
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden      = out.last_hidden_state                          # [B, T, D]
        seq_lengths = attention_mask.sum(dim=1) - 1                  # [B]
        last_token  = hidden[torch.arange(hidden.size(0)), seq_lengths]  # [B, D]
        return last_token

    # ── Reward computation ────────────────────────────────────────────────────

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns per-objective rewards. Shape: [batch, K]"""
        enc = self.encode(input_ids, attention_mask)    # [B, encoder_dim]
        h   = self.projection(enc)                      # [B, hidden_dim]
        return torch.stack([head(h) for head in self.heads], dim=-1)  # [B, K]

    def get_reward_gap(
        self,
        input_ids_a: torch.Tensor, attention_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor, attention_mask_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Delta_i^k = r_k(A) - r_k(B)
        Returns: [batch, K]
        """
        r_a = self.forward(input_ids_a, attention_mask_a)   # [B, K]
        r_b = self.forward(input_ids_b, attention_mask_b)   # [B, K]
        return r_a - r_b                                     # [B, K]

    # ── Global preference (THE NEW PIECE) ────────────────────────────────────

    def global_score(
        self,
        input_ids_a: torch.Tensor, attention_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor, attention_mask_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Implements: score_i = w^T · Delta_i

        where w = softmax(w_phi)  and  Delta_i = [Delta_tox, Delta_math, Delta_code]

        This is the core equation from the notes:
            P(y+ > y- | x) = sigma(w^T · Delta)

        Returns: [batch]  scalar score per sample
        """
        delta = self.get_reward_gap(
            input_ids_a, attention_mask_a,
            input_ids_b, attention_mask_b,
        )                                      # [B, K]
        w     = self.w                         # [K]  normalised weights
        score = (w * delta).sum(dim=-1)        # [B]  w^T · Delta per sample
        return score

    def global_preference_prob(
        self,
        input_ids_a: torch.Tensor, attention_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor, attention_mask_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        P(A > B | x) = sigma(w^T · Delta)
        Returns: [batch]
        """
        return torch.sigmoid(
            self.global_score(input_ids_a, attention_mask_a,
                              input_ids_b, attention_mask_b)
        )

    # ── Tokenise helper ───────────────────────────────────────────────────────

    def tokenise(self, texts: list[str], device: torch.device) -> dict:
        enc = self.tokenizer(
            texts,
            padding        = True,
            truncation     = True,
            max_length     = self.max_length,
            return_tensors = "pt",
        )
        return {k: v.to(device) for k, v in enc.items()}


def build_conflict_model(config: dict) -> PythiaConflictRewardModel:
    return PythiaConflictRewardModel(
        model_name     = config["model"].get("encoder", MODEL_NAME),
        hidden_dim     = config["model"]["hidden_dim"],
        dropout        = config["model"]["dropout"],
        num_objectives = config["model"]["num_objectives"],
        freeze_encoder = config["model"].get("freeze_encoder", True),
        max_length     = config["model"].get("max_length", 512),
    )
