"""
src/reward_model.py

Shared encoder + K independent reward heads.

Architecture:
    [prompt + response] → (frozen) sentence-transformer → embedding
                        → MLP projection → [head_tox, head_math, head_code]
                                                ↓            ↓           ↓
                                             r_tox        r_math      r_code  (scalars)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

OBJECTIVES = ["toxicity", "math", "code"]
K          = len(OBJECTIVES)


class RewardHead(nn.Module):
    """Single-objective reward head: linear → scalar."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, input_dim] → [batch] scalar reward."""
        return self.net(x).squeeze(-1)


class MultiObjectiveRewardModel(nn.Module):
    """
    Shared projection + K independent reward heads.

    The sentence-transformer encoder is NOT fine-tuned (frozen).
    The projection MLP and reward heads are trainable.

    Forward call expects pre-encoded embeddings (done in Dataset for speed).
    """

    def __init__(
        self,
        encoder_dim: int  = 384,   # all-MiniLM-L6-v2 output dim
        hidden_dim:  int  = 256,
        dropout:     float = 0.1,
        num_objectives: int = K,
    ):
        super().__init__()

        # Shared projection: encoder_dim → hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # K objective-specific heads
        self.heads = nn.ModuleList([
            RewardHead(hidden_dim, hidden_dim // 2, dropout)
            for _ in range(num_objectives)
        ])

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, enc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enc: [batch, encoder_dim]  — pre-computed sentence embedding
        Returns:
            rewards: [batch, K]        — one scalar reward per objective
        """
        h = self.projection(enc)                           # [batch, hidden_dim]
        rewards = torch.stack([head(h) for head in self.heads], dim=-1)  # [batch, K]
        return rewards

    def get_reward_gap(
        self,
        enc_a: torch.Tensor,
        enc_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Δᵢᵏ = rₖ(A) - rₖ(B)

        Returns: [batch, K]
        """
        r_a = self.forward(enc_a)   # [batch, K]
        r_b = self.forward(enc_b)   # [batch, K]
        return r_a - r_b

    def preference_prob(
        self,
        enc_a: torch.Tensor,
        enc_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        p̂ᵢᵏ = σ(rₖ(A) - rₖ(B))

        Returns: [batch, K]
        """
        return torch.sigmoid(self.get_reward_gap(enc_a, enc_b))

    def combined_reward(
        self,
        enc: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        R_w(x,y) = Σₖ wₖ · rₖ(x,y)

        Args:
            enc:     [batch, encoder_dim]
            weights: [K]
        Returns:
            [batch]  scalar combined reward
        """
        rewards = self.forward(enc)            # [batch, K]
        return (rewards * weights.unsqueeze(0)).sum(dim=-1)  # [batch]


def build_model(config: dict) -> MultiObjectiveRewardModel:
    """Build model from config dict."""
    # Infer encoder dim from a dummy forward pass
    dummy_encoder = SentenceTransformer(config["model"]["encoder"])
    encoder_dim   = dummy_encoder.get_sentence_embedding_dimension()
    del dummy_encoder

    model = MultiObjectiveRewardModel(
        encoder_dim    = encoder_dim,
        hidden_dim     = config["model"]["hidden_dim"],
        dropout        = config["model"]["dropout"],
        num_objectives = config["model"]["num_objectives"],
    )
    return model
