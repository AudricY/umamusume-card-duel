from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
from torch import nn

from .features import ACTION_DIM, STATE_DIM


@dataclass(frozen=True)
class ModelConfig:
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    hidden_dim: int = 128
    depth: int = 3
    dropout: float = 0.05

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, int | float] | None) -> "ModelConfig":
        if not raw:
            return cls()
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in raw.items() if key in allowed})


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CandidatePolicyNet(nn.Module):
    """Candidate-conditioned policy/value network for variable legal-action sets."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        hidden = self.config.hidden_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(self.config.state_dim, hidden),
            nn.GELU(),
            ResidualBlock(hidden, self.config.dropout),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(self.config.action_dim, hidden),
            nn.GELU(),
            ResidualBlock(hidden, self.config.dropout),
        )
        self.joint_projection = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
        )
        self.joint_blocks = nn.Sequential(*[ResidualBlock(hidden, self.config.dropout) for _ in range(self.config.depth)])
        self.policy_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        state_features: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_encoded = self.state_encoder(state_features)
        action_encoded = self.action_encoder(action_features)
        state_for_actions = state_encoded.unsqueeze(1).expand(-1, action_encoded.size(1), -1)
        joint = torch.cat([state_for_actions, action_encoded, state_for_actions * action_encoded], dim=-1)
        joint = self.joint_blocks(self.joint_projection(joint))
        logits = self.policy_head(joint).squeeze(-1)
        logits = logits.masked_fill(~action_mask.bool(), -1.0e9)
        value = self.value_head(state_encoded).squeeze(-1)
        return logits, value
