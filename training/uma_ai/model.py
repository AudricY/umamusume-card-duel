from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
from torch import nn

from .features import ACTION_DIM, STATE_DIM, ZONE_ORDER

# R7.b.2 Phase 2: card-embedding vocab. `vocabSize` (107) + 1 for the
# pad/unknown slot at index 0 — `shared/src/cardVocab.json:3-4` reserves
# index 0 as `unknownIndex`, and `nn.Embedding(padding_idx=0)` doubles
# that as the gradient-free pad. Total table = 108 × 32 = 3456 params.
CARD_EMBED_DIM = 32
CARD_VOCAB_TABLE_SIZE = 108  # 107 cards + reserved 0 for unknown/pad
NUM_ZONES = len(ZONE_ORDER)
ACTION_PAIR_FANOUT = 2  # source + target idx per action


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
        # R7.b.2 Phase 2: shared per-card embedding table consumed by BOTH
        # the per-zone state branch (sum-pool over zone-packed ids → linear
        # to `hidden`, ADDED to `state_encoded` as an additive residual) and
        # the action branch (source + target embed concatenated to the
        # 48-d action vector before joint projection). Single table keeps
        # the policy/identity geometry consistent: the network reasons
        # about "what is card X" identically whether X is on the bench or
        # the target of an attack.
        #
        # Additive-residual rationale (vs replacing the 16 hash-identity
        # slots in the 110-d state vector now): scoping § 11 Phase 2 calls
        # for an additive lift so the Phase 2 ablation is apples-to-apples
        # vs Phase 1 hygiene. Phase 6 gate eval will decide whether the
        # hash slots can be retired in a later schema bump.
        self.card_embed = nn.Embedding(CARD_VOCAB_TABLE_SIZE, CARD_EMBED_DIM, padding_idx=0)
        # bias=False so an all-pad (all-zero ids) input produces an exact
        # zero residual; combined with `padding_idx=0` this guarantees the
        # additive-residual is structurally null when no cards are present
        # (matches the "zero-id forward == omitted-arg forward" sanity
        # contract in the Phase 2 smoke).
        self.zone_projection = nn.Linear(NUM_ZONES * CARD_EMBED_DIM, hidden, bias=False)
        # Joint projection input grows from `3 * hidden` to
        # `3 * hidden + ACTION_PAIR_FANOUT * CARD_EMBED_DIM` because the
        # action branch concatenates the source + target embedding before
        # the state×action joint trunk consumes the row.
        joint_input_dim = hidden * 3 + ACTION_PAIR_FANOUT * CARD_EMBED_DIM
        self.joint_projection = nn.Sequential(
            nn.Linear(joint_input_dim, hidden),
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
        card_ids_by_zone: torch.Tensor | None = None,
        action_card_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Candidate-conditioned forward.

        R7.b.2 Phase 2: `card_ids_by_zone` and `action_card_idx` are
        optional (default zero tensors). When omitted the embedding
        contributions collapse to `embed(0) = 0` because `padding_idx=0`
        ensures the row at index 0 is zero AND gradient-free; the zone
        sum-pool then sees zero in every slot, and the joint projection
        sees zero in the concatenated action-pair embedding. This keeps
        existing positional callers (`train_ppo.py`, `train_dpo.py`,
        `train_bc.py:325` ONNX roundtrip, `r13_value_retrain.py`,
        `r14_value_crossover_probe.py`, `calibrate_value.py`) and the
        110-d-only ONNX export (`export_onnx.py`) working unchanged until
        Phase 3 plumbs the new tensors through them. Phase 2/3 decoupling.

        Shapes:
            state_features:    `[B, state_dim]`
            action_features:   `[B, A, action_dim]`
            action_mask:       `[B, A]` (bool/0-1)
            card_ids_by_zone:  `[B, NUM_ZONES, max_cards_per_zone]` (int64);
                               positions are zero-padded per `CARD_ID_SHAPES`.
                               Max-cards-per-zone is the batch-wise max so
                               smaller zones see padding slots beyond their
                               cap, which read as 0 (pad).
            action_card_idx:   `[B, A, 2]` (int64; col 0 = source, col 1 = target).
        """

        batch_size = state_features.shape[0]
        num_actions = action_features.shape[1]
        device = state_features.device

        state_encoded = self.state_encoder(state_features)
        # R7.b.2 Phase 2: additive per-zone embedding pool. We sum-pool the
        # embedded ids within each zone (mask out padding to keep the pool
        # invariant to # of slots actually filled), concat across the 8
        # zones, project to `hidden`, and ADD to `state_encoded`. The
        # `padding_idx=0` semantics of `nn.Embedding` mean ids==0 produce a
        # zero row in the gather, so `(ids != 0)`-masking before sum is
        # equivalent to letting the pad rows pass — we keep the explicit
        # mask anyway because it documents the intent and survives any
        # future padding_idx change.
        if card_ids_by_zone is not None:
            # `embedded`: [B, NUM_ZONES, max_cards, embed]
            embedded = self.card_embed(card_ids_by_zone)
            mask = (card_ids_by_zone != 0).unsqueeze(-1).to(embedded.dtype)
            # Sum over the per-zone cards axis (`max_cards`), giving
            # `[B, NUM_ZONES, embed]`, then flatten zones into a single
            # feature axis for the projection.
            pooled = (embedded * mask).sum(dim=-2)
            zone_features = pooled.reshape(batch_size, NUM_ZONES * CARD_EMBED_DIM)
            state_encoded = state_encoded + self.zone_projection(zone_features)

        action_encoded = self.action_encoder(action_features)
        # R7.b.2 Phase 2: source + target embedding for each candidate.
        # When `action_card_idx` is omitted, fall back to a zero-tensor
        # (`embed(0)` is zero due to `padding_idx=0`) so the joint
        # projection sees zero contribution from the action-pair branch.
        if action_card_idx is not None:
            # `action_pair_embed`: [B, A, 2, embed] → flatten to
            # [B, A, 2 * embed] for concat alongside `action_encoded`.
            action_pair_embed = self.card_embed(action_card_idx)
            action_pair_flat = action_pair_embed.reshape(batch_size, num_actions, ACTION_PAIR_FANOUT * CARD_EMBED_DIM)
        else:
            action_pair_flat = torch.zeros(
                (batch_size, num_actions, ACTION_PAIR_FANOUT * CARD_EMBED_DIM),
                dtype=action_encoded.dtype,
                device=device,
            )

        state_for_actions = state_encoded.unsqueeze(1).expand(-1, num_actions, -1)
        joint = torch.cat(
            [state_for_actions, action_encoded, state_for_actions * action_encoded, action_pair_flat],
            dim=-1,
        )
        joint = self.joint_blocks(self.joint_projection(joint))
        logits = self.policy_head(joint).squeeze(-1)
        logits = logits.masked_fill(~action_mask.bool(), -1.0e9)
        value = self.value_head(state_encoded).squeeze(-1)
        return logits, value
