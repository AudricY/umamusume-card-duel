from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
from torch import nn

from .features import (
    ACTION_DIM,
    CARD_ID_SHAPES,
    STATE_DIM,
    UMA_SLOT_COUNT,
    UMA_SLOT_FEATURE_DIM,
    ZONE_ORDER,
)

# R7.b.2 Phase 2: card-embedding vocab. `vocabSize` (107) + 1 for the
# pad/unknown slot at index 0 — `shared/src/cardVocab.json:3-4` reserves
# index 0 as `unknownIndex`, and `nn.Embedding(padding_idx=0)` doubles
# that as the gradient-free pad. Total table = 108 × 32 = 3456 params.
CARD_EMBED_DIM = 32
CARD_VOCAB_TABLE_SIZE = 108  # 107 cards + reserved 0 for unknown/pad
NUM_ZONES = len(ZONE_ORDER)
ACTION_PAIR_FANOUT = 2  # source + target idx per action

# R7.b.3 set-attention probe: pre-trunk encoder hyperparameters. Frozen for
# the Slice 1/2 path; widening or going to >1 layer fires only if Slice 2 is
# marginal per the scoping doc. d_model is INTENTIONALLY equal to the
# hidden_dim=64 default so the encoder output drops directly into the existing
# joint trunk with no projection. n_heads=4 gives head_dim=16 which keeps the
# attention well-conditioned with the small d_model. ffn_dim=128 = 2 *
# d_model (the canonical 2x expansion factor).
SET_ATTN_D_MODEL = 64
SET_ATTN_N_HEADS = 4
SET_ATTN_FFN_DIM = 128
SET_ATTN_N_LAYERS = 1
# Maximum tokens that flow into the attention encoder. Order is FROZEN as:
#   [CLS] + per-zone card tokens (NUM_ZONES * MAX_CARDS_PER_ZONE) + uma slot
#   tokens (UMA_SLOT_COUNT). This matches the per-zone packed v3.2 tensors
#   so a single index arithmetic step ((zone_idx, pos_idx) -> token_idx)
#   maps a card_ids_by_zone entry into the sequence. `MAX_CARDS_PER_ZONE`
#   is shared with `training/export_onnx.py` (= 30, the cap on
#   own/oppDiscard); other zones contribute padding for the unused slots
#   (their `card_id == 0` so the pad mask zeroes them out at attention).
SET_ATTN_MAX_CARDS_PER_ZONE = max(CARD_ID_SHAPES.values())
SET_ATTN_NUM_CARD_TOKENS = NUM_ZONES * SET_ATTN_MAX_CARDS_PER_ZONE
SET_ATTN_NUM_TOKENS = 1 + SET_ATTN_NUM_CARD_TOKENS + UMA_SLOT_COUNT  # 1 + 240 + 10 = 251

_VALID_MODEL_VARIANTS = ("mlp", "set_attention")

# R16-P2 C2: board-zone lane indices into `card_ids_by_zone` (axis=1). When the
# per-Uma slot-token branch is active (`uses_uma_slot_tokens=True`), the
# `uma_slot_encoder` already encodes board identities at per-slot granularity,
# so we MUST zero these 4 lanes before `zone_projection` consumes them to
# avoid double-counting board identities (chunk plan § "Avoid double-counting
# board identities"). The 4 non-board zones (ownHand, ownDiscard, oppDiscard,
# stadium) pass through unchanged. The order MUST match the leading 4 entries
# of `ZONE_ORDER` in `features.py` (assertion below). The slot-token branch is
# a no-op when its kwargs are absent, so v3.0/v3.1 forwards stay byte-stable.
BOARD_ZONE_NAMES: tuple[str, ...] = ("ownActive", "oppActive", "ownBench", "oppBench")
BOARD_ZONE_LANE_INDICES: tuple[int, ...] = tuple(
    ZONE_ORDER.index(name) for name in BOARD_ZONE_NAMES
)
assert BOARD_ZONE_LANE_INDICES == (0, 1, 2, 3), (
    f"BOARD_ZONE_LANE_INDICES {BOARD_ZONE_LANE_INDICES} does not match the "
    f"frozen ZONE_ORDER head (0..3 = ownActive/oppActive/ownBench/oppBench); "
    f"slot-token board-zone masking depends on this layout."
)


@dataclass(frozen=True)
class ModelConfig:
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    hidden_dim: int = 128
    depth: int = 3
    dropout: float = 0.05
    # R16-P2 C2: when True, build the `uma_slot_encoder` branch in
    # `CandidatePolicyNet` AND zero the 4 board-zone lanes of
    # `card_ids_by_zone` before `zone_projection` to avoid double-counting
    # board identities. Defaults to False so v3.0/v3.1 forwards (and any
    # caller that does not opt in) are byte-identical to pre-C2 behavior.
    # The encoder's final Linear is zero-initialized at construction time so
    # the slot-residual contribution to `state_encoded` is structurally null
    # at init (the v3.1 `delta=0.0` parity trick analog — load-bearing for
    # C7's `make_v32_slot_token_init.py` warm-start from a v3.0 checkpoint).
    uses_uma_slot_tokens: bool = False
    # R7.b.3 set-attention probe: selects the pre-trunk encoder. "mlp"
    # (default) is byte-identical to the legacy state_encoder + zone_projection
    # + uma_slot_encoder sum-pool trunk that v3.0/v3.1/v3.2 ride. "set_attention"
    # adds a 1-layer self-attention residual over `[CLS, card_token × per-zone ×
    # max_cards_per_zone, uma_slot_token × UMA_SLOT_COUNT]`. The encoder output
    # projection is ZERO-initialized so at construction its contribution to
    # `state_encoded` is structurally null (matches the v3.1 / C7 parity trick
    # and lets a future additive warm-start from a v3.2 ckpt stay bit-stable).
    # The new value consumes the SAME v3.2
    # tensors (`card_ids_by_zone`, `uma_slot_card_ids`, `uma_slot_features`)
    # — the ONNX graph signature is identical to v3.2 so the existing 7-input
    # Rust + serve_onnx dispatch handles the attention model unchanged.
    model_variant: str = "mlp"

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


class SetAttentionEncoder(nn.Module):
    """R7.b.3 set-attention pre-trunk encoder.

    Adds a self-attention transformer residual over a per-card / per-slot
    token sequence. CLS-token output is projected to `state_encoded` via a
    ZERO-initialized output Linear so the attention contribution at init is
    structurally null (the v3.1 / C7 parity trick analog).

    Inputs (mirroring the v3.2 ONNX tensors):
      - `state_features: [B, state_dim]` — embedded via a small MLP and
        added to the CLS token at sequence position 0. This keeps the
        legacy 110-d scalar features in the trunk's input distribution
        (they otherwise would be ignored entirely by a pure set-attention
        over per-card tokens). Symmetric to the existing `state_encoder`
        path: the attention adds a relational lens on top of, not instead
        of, the global state vector.
      - `card_ids_by_zone: [B, NUM_ZONES, MAX_CARDS_PER_ZONE]` — flattened
        into `[B, NUM_ZONES * MAX_CARDS_PER_ZONE]` card tokens. Each token
        embedding is `card_embed(card_id) + zone_pos_embed(zone_idx) +
        slot_pos_embed(slot_idx)`. The pad mask (`card_id == 0`) suppresses
        these positions at attention.
      - `uma_slot_card_ids: [B, UMA_SLOT_COUNT]` and `uma_slot_features:
        [B, UMA_SLOT_COUNT, UMA_SLOT_FEATURE_DIM]` — UMA_SLOT_COUNT slot
        tokens. Each = `card_embed(slot_card_id) +
        slot_feature_proj(uma_slot_features) + slot_idx_embed`. Pad mask
        (`uma_slot_card_ids == 0`) suppresses absent slots.

    Token layout (FROZEN):
        position 0:                              CLS
        position 1 .. 1+NUM_CARD_TOKENS-1:       card tokens (8 zones * 30 slots)
        position 1+NUM_CARD_TOKENS .. END-1:     uma slot tokens (10)

    `pad_mask: [B, S] bool` is True at padding positions (passed to
    `MultiheadAttention.key_padding_mask`). CLS is ALWAYS unmasked.
    """

    def __init__(
        self,
        *,
        d_model: int = SET_ATTN_D_MODEL,
        n_heads: int = SET_ATTN_N_HEADS,
        ffn_dim: int = SET_ATTN_FFN_DIM,
        n_layers: int = SET_ATTN_N_LAYERS,
        dropout: float = 0.05,
        state_dim: int = STATE_DIM,
        card_embed: nn.Embedding | None = None,
    ) -> None:
        super().__init__()
        if card_embed is None:
            raise ValueError("SetAttentionEncoder requires the shared card_embed Embedding")
        self.d_model = d_model
        self.n_layers = n_layers
        self.card_embed = card_embed  # shared with the action branch

        # Project the existing 110-d state vector into d_model so it can be
        # added to the CLS token at position 0. Pre-LN downstream so we
        # don't need an explicit normalization here.
        self.state_proj = nn.Linear(state_dim, d_model)

        # CLS token (learned). Initialized small so it doesn't dominate the
        # attention at iter-0.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Position embeddings.
        #   - zone_pos_embed indexes 0..NUM_ZONES-1 (per-zone polarity /
        #     identity signal: 'this token is in own_active' etc.).
        #   - card_slot_pos_embed indexes 0..MAX_CARDS_PER_ZONE-1 (which
        #     position within a zone — preserves the ordering signal that
        #     the sum-pool baseline collapses).
        #   - uma_slot_pos_embed indexes 0..UMA_SLOT_COUNT-1 (per-slot
        #     identity: own_active vs own_bench_0 vs opp_active etc.).
        self.zone_pos_embed = nn.Embedding(NUM_ZONES, d_model)
        self.card_slot_pos_embed = nn.Embedding(SET_ATTN_MAX_CARDS_PER_ZONE, d_model)
        self.uma_slot_pos_embed = nn.Embedding(UMA_SLOT_COUNT, d_model)
        # Initialize position embeddings small so they don't dominate the
        # card_embed signal at iter-0.
        nn.init.trunc_normal_(self.zone_pos_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.card_slot_pos_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.uma_slot_pos_embed.weight, std=0.02)

        # Project the card embed (CARD_EMBED_DIM=32) up to d_model=64. The
        # action branch keeps using CARD_EMBED_DIM directly (32-d concat) —
        # the trunk-side projection lives here so the attention block sees
        # tokens at d_model uniformly.
        self.card_embed_proj = nn.Linear(CARD_EMBED_DIM, d_model)
        # Project the per-slot scalar features (UMA_SLOT_FEATURE_DIM=23) up
        # to d_model so the uma_slot_token = card_embed_proj(embed) +
        # slot_feature_proj(features) + uma_slot_pos_embed.
        self.slot_feature_proj = nn.Linear(UMA_SLOT_FEATURE_DIM, d_model)

        # Transformer encoder: pre-LN MHA + FFN, n_layers stacked. We use
        # PyTorch's `TransformerEncoderLayer(norm_first=True)` to get the
        # standard MHA+FFN block with the pre-LN ordering scoping § sketch
        # specifies. `batch_first=True` so the [B, S, D] axis convention
        # matches the rest of the model.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection from CLS to `state_encoded`. ZERO-INIT (weight
        # and bias) so the encoder's contribution to state_encoded is
        # structurally null at construction. This mirrors the C7
        # `make_v32_slot_token_init.py` zero-Linear parity trick (and the
        # uma_slot_encoder's zero-init final Linear at model.py:176).
        self.output_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        state_features: torch.Tensor,
        card_ids_by_zone: torch.Tensor | None,
        uma_slot_card_ids: torch.Tensor | None,
        uma_slot_features: torch.Tensor | None,
    ) -> torch.Tensor:
        """Emit `state_encoded: [B, d_model]` from the CLS-token output.

        At init the output projection is zero so the returned tensor is all
        zeros regardless of inputs — the joint trunk then receives a zero
        state_encoded contribution from the attention path. Training breaks
        this null over the first few gradient steps.
        """

        batch_size = state_features.shape[0]
        device = state_features.device
        dtype = state_features.dtype

        # CLS token: broadcast the learned vector across the batch and add
        # the projected scalar state vector. Position 0 is the CLS.
        cls = self.cls_token.expand(batch_size, -1, -1).to(dtype=dtype)
        cls = cls + self.state_proj(state_features).unsqueeze(1)

        # Card tokens: card_embed(id) projected up to d_model + zone-pos +
        # slot-pos. We always materialize the full [NUM_ZONES,
        # MAX_CARDS_PER_ZONE] grid so the ONNX graph has a fixed sequence
        # length (matches the v3.2 input contract). Absent cards (id == 0)
        # contribute via the pad mask only — their card_embed row is the
        # padding_idx=0 zero row, so the per-card token reduces to position
        # embeddings (which the pad mask suppresses at attention anyway).
        if card_ids_by_zone is None:
            card_ids_by_zone = torch.zeros(
                (batch_size, NUM_ZONES, SET_ATTN_MAX_CARDS_PER_ZONE),
                dtype=torch.int64,
                device=device,
            )
        # `card_embedded`: [B, NUM_ZONES, MAX_CARDS, CARD_EMBED_DIM]
        card_embedded = self.card_embed(card_ids_by_zone)
        # Project to d_model: [B, NUM_ZONES, MAX_CARDS, d_model]
        card_tokens = self.card_embed_proj(card_embedded)
        # Add zone position embedding (broadcast across the MAX_CARDS axis).
        # zone_pos: [NUM_ZONES, d_model] -> [1, NUM_ZONES, 1, d_model].
        zone_pos = self.zone_pos_embed.weight.unsqueeze(0).unsqueeze(2)
        card_tokens = card_tokens + zone_pos
        # Add per-zone-slot position embedding (broadcast across NUM_ZONES).
        # slot_pos: [MAX_CARDS, d_model] -> [1, 1, MAX_CARDS, d_model].
        slot_pos = self.card_slot_pos_embed.weight.unsqueeze(0).unsqueeze(0)
        card_tokens = card_tokens + slot_pos
        # Flatten the per-zone grid into the sequence dim:
        #   [B, NUM_ZONES, MAX_CARDS, d_model] -> [B, NUM_ZONES * MAX_CARDS, d_model]
        card_tokens = card_tokens.reshape(batch_size, SET_ATTN_NUM_CARD_TOKENS, self.d_model)
        # Pad mask: True at positions where card_id == 0 (padding). Same
        # reshape order as the tokens.
        card_pad = (card_ids_by_zone == 0).reshape(batch_size, SET_ATTN_NUM_CARD_TOKENS)

        # Uma slot tokens: card_embed_proj(slot_card_id) +
        # slot_feature_proj(uma_slot_features) + uma_slot_pos_embed.
        if uma_slot_card_ids is None or uma_slot_features is None:
            uma_slot_card_ids = torch.zeros(
                (batch_size, UMA_SLOT_COUNT), dtype=torch.int64, device=device,
            )
            uma_slot_features = torch.zeros(
                (batch_size, UMA_SLOT_COUNT, UMA_SLOT_FEATURE_DIM),
                dtype=dtype, device=device,
            )
        slot_embedded = self.card_embed(uma_slot_card_ids)  # [B, S, CARD_EMBED]
        slot_tokens = self.card_embed_proj(slot_embedded)
        slot_tokens = slot_tokens + self.slot_feature_proj(uma_slot_features)
        # Add slot-pos embedding (broadcast across batch): [UMA_SLOT_COUNT, d_model]
        slot_pos_vec = self.uma_slot_pos_embed.weight.unsqueeze(0)
        slot_tokens = slot_tokens + slot_pos_vec
        # Pad mask: True at absent slots.
        slot_pad = (uma_slot_card_ids == 0)

        # Assemble [CLS, card_tokens, slot_tokens] along the sequence axis.
        tokens = torch.cat([cls, card_tokens, slot_tokens], dim=1)
        # CLS is always unmasked: False at position 0.
        cls_pad = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        pad_mask = torch.cat([cls_pad, card_pad, slot_pad], dim=1)

        # Self-attention encoder (key_padding_mask: True = ignore).
        encoded = self.encoder(tokens, src_key_padding_mask=pad_mask)
        # CLS-token output -> projected to `state_encoded`.
        cls_out = encoded[:, 0, :]
        state_encoded = self.output_proj(cls_out)
        return state_encoded


class CandidatePolicyNet(nn.Module):
    """Candidate-conditioned policy/value network for variable legal-action sets."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        if self.config.model_variant not in _VALID_MODEL_VARIANTS:
            raise ValueError(
                f"unknown model_variant {self.config.model_variant!r}; "
                f"expected one of {_VALID_MODEL_VARIANTS}"
            )
        if self.config.model_variant == "set_attention" and not self.config.uses_uma_slot_tokens:
            raise ValueError(
                "model_variant='set_attention' requires uses_uma_slot_tokens=True; "
                "the attention encoder consumes the v3.2 slot-token tensors."
            )
        hidden = self.config.hidden_dim
        # R7.b.3 set-attention probe: the "mlp" variant is the legacy
        # additive state_encoder / zone_projection / uma_slot_encoder trunk.
        # The "set_attention" variant keeps that trunk live and adds a
        # zero-initialized attention residual. Keeping the base path live is
        # load-bearing for v3.2 warm-start parity: a checkpoint copied into
        # the set-attention schema has identical outputs until training
        # moves `set_attention_encoder.output_proj` away from zero.
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

        # R16-P2 C2: optional per-Uma slot-token encoder. Construction is
        # gated on `uses_uma_slot_tokens` so default v3.0/v3.1 callers (which
        # leave the flag False) get IDENTICAL parameter layouts to pre-C2 —
        # no extra params, no init-order shifts, no state_dict key drift. The
        # encoder shape mirrors the existing `zone_projection` pattern but
        # operates per-slot instead of per-zone-sum: GELU+Linear-Linear with
        # a ZERO-initialized FINAL Linear so `uma_slot_encoder(any) → 0` at
        # init — the slot residual added to `state_encoded` is structurally
        # null regardless of the slot tensors fed in. This is the load-
        # bearing parity claim C7's `make_v32_slot_token_init.py` relies on
        # (copy v3.0 weights verbatim + leave the slot encoder at its
        # zero-init → identical logits/value to the source v3.0 checkpoint).
        #
        # The FIRST Linear stays at default (Kaiming/Xavier) — its output
        # flows through GELU and then the zero Linear, so the structural
        # null at the encoder's output does not depend on the first layer's
        # init. The training dynamics get a useful starting Jacobian from
        # the non-zero first layer once the zero output Linear begins to
        # accumulate signal.
        if self.config.uses_uma_slot_tokens:
            self.uma_slot_encoder = nn.Sequential(
                nn.Linear(CARD_EMBED_DIM + UMA_SLOT_FEATURE_DIM, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden, bias=False),
            )
            nn.init.zeros_(self.uma_slot_encoder[-1].weight)
        else:
            self.uma_slot_encoder = None

        # R7.b.3 set-attention encoder. Built only when `model_variant ==
        # "set_attention"`; the default "mlp" variant leaves this None so
        # legacy v3.0/v3.1/v3.2 forwards are byte-identical to pre-R7.b.3.
        # `d_model` is pinned to `hidden_dim` (SET_ATTN_D_MODEL=64) so the
        # encoder residual drops directly into `state_encoded` without an
        # extra projection.
        if self.config.model_variant == "set_attention":
            # Hard contract: set_attention requires hidden_dim=64 so the
            # encoder's d_model matches the trunk hidden width. The
            # alternative (a projection from d_model -> hidden) would
            # complicate the zero-init parity claim, and the scoping doc
            # locks d_model=64 anyway.
            if hidden != SET_ATTN_D_MODEL:
                raise ValueError(
                    f"model_variant='set_attention' requires hidden_dim="
                    f"{SET_ATTN_D_MODEL} (the encoder's d_model); got "
                    f"hidden_dim={hidden}"
                )
            self.set_attention_encoder = SetAttentionEncoder(
                d_model=hidden,
                n_heads=SET_ATTN_N_HEADS,
                ffn_dim=SET_ATTN_FFN_DIM,
                n_layers=SET_ATTN_N_LAYERS,
                dropout=self.config.dropout,
                state_dim=self.config.state_dim,
                card_embed=self.card_embed,
            )
        else:
            self.set_attention_encoder = None

    def forward(
        self,
        state_features: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
        card_ids_by_zone: torch.Tensor | None = None,
        action_card_idx: torch.Tensor | None = None,
        uma_slot_card_ids: torch.Tensor | None = None,
        uma_slot_features: torch.Tensor | None = None,
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
            # R16-P2 C2 board-zone double-counting fix: when the per-Uma
            # slot-token branch is active, the 4 board-zone lanes
            # (ownActive=0, oppActive=1, ownBench=2, oppBench=3) are already
            # encoded at per-slot granularity by `uma_slot_encoder`. Zero
            # those lanes' card ids here (replace with the pad index 0) so
            # `zone_projection` only sees the 4 non-board zones (ownHand,
            # ownDiscard, oppDiscard, stadium). The 4 board lanes still feed
            # the projection but contribute embed(0)=0 per `padding_idx=0` —
            # the existing zero-pad contract carries the structural-zero
            # for us, no shape change to `zone_features`. Gated on the flag
            # so v3.0/v3.1 forwards stay byte-identical when False.
            if self.config.uses_uma_slot_tokens:
                card_ids_by_zone = card_ids_by_zone.clone()
                # Index axis 1 (NUM_ZONES axis) at the 4 board-zone lanes;
                # we keep a contiguous 0..3 slice rather than scatter so
                # the operation is a single-stride view-and-fill on CUDA.
                card_ids_by_zone[:, : len(BOARD_ZONE_LANE_INDICES), :] = 0
            # `embedded`: [B, NUM_ZONES, max_cards, embed]
            embedded = self.card_embed(card_ids_by_zone)
            mask = (card_ids_by_zone != 0).unsqueeze(-1).to(embedded.dtype)
            # Sum over the per-zone cards axis (`max_cards`), giving
            # `[B, NUM_ZONES, embed]`, then flatten zones into a single
            # feature axis for the projection.
            pooled = (embedded * mask).sum(dim=-2)
            zone_features = pooled.reshape(batch_size, NUM_ZONES * CARD_EMBED_DIM)
            state_encoded = state_encoded + self.zone_projection(zone_features)

        # R16-P2 C2: per-Uma slot-token branch. Active only when the model
        # was constructed with `uses_uma_slot_tokens=True` AND both slot
        # tensors are provided. When inactive (flag False, encoder None, or
        # kwargs absent) this is a strict no-op — state_encoded passes
        # through untouched, byte-identical to v3.0/v3.1.
        #
        # Mask source: we use `(uma_slot_card_ids != 0)` rather than
        # `uma_slot_features[..., _UMA_SLOT_F_PRESENT]` (slot col 3). Both
        # are equivalent under the C1 builder contract (absent slot → both
        # card_id=0 AND present_mask=0; present slot → card_id != 0 AND
        # present_mask=1.0). The card-id mask is preferred because:
        #   (a) it mirrors the existing `card_ids_by_zone` mask convention
        #       in `zone_projection` (single contract across both
        #       embedding branches);
        #   (b) it is a bit-exact predicate (int64 != 0) instead of a
        #       float-equality compare, so it never picks up FP noise from
        #       an upstream ablation that zeroes the features but leaves
        #       card_ids intact (no such ablation exists today, but the
        #       contract is cleaner);
        #   (c) C7's init builder mirrors this convention — when the slot
        #       tensors are all zero (the init-parity test), card_ids are
        #       all zero so the mask is all zero, sum-pool is zero, and
        #       the residual is structurally null regardless of the
        #       zero-Linear init. Double-guard.
        if (
            self.config.uses_uma_slot_tokens
            and self.uma_slot_encoder is not None
            and uma_slot_card_ids is not None
            and uma_slot_features is not None
        ):
            # Embed slot card ids using the SHARED `card_embed` table — no
            # new vocab, no new embedding parameters. `slot_embed`:
            # [B, UMA_SLOT_COUNT, CARD_EMBED_DIM].
            slot_embed = self.card_embed(uma_slot_card_ids)
            # Concat embedding + per-slot scalar features along the feature
            # axis: [B, UMA_SLOT_COUNT, CARD_EMBED_DIM + UMA_SLOT_FEATURE_DIM].
            slot_concat = torch.cat([slot_embed, uma_slot_features], dim=-1)
            # Per-slot encode: [B, UMA_SLOT_COUNT, hidden]. At init the
            # final Linear's weight is all zeros, so this output is
            # structurally zero regardless of input.
            per_slot_encoded = self.uma_slot_encoder(slot_concat)
            # Sum-pool over the UMA_SLOT_COUNT (=10) slots with the absent-
            # slot mask. Absent slot → card_id 0 → mask 0 → contributes 0.
            slot_mask = (uma_slot_card_ids != 0).to(per_slot_encoded.dtype).unsqueeze(-1)
            slot_pooled = (per_slot_encoded * slot_mask).sum(dim=1)
            state_encoded = state_encoded + slot_pooled

        # R7.b.3 set-attention probe: add the attention residual after the
        # legacy v3.2 additive trunk. `output_proj` is zero-initialized, so
        # this is a strict no-op at init and preserves v3.2 warm-start parity;
        # training can then learn relational corrections through this path.
        if self.set_attention_encoder is not None:
            state_encoded = state_encoded + self.set_attention_encoder(
                state_features,
                card_ids_by_zone,
                uma_slot_card_ids,
                uma_slot_features,
            )

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
        # dtype-safe mask fill: `-1.0e9` overflows at::Half (fp16) under
        # train_bc.py --amp on CUDA, silently producing inf/NaN logits.
        # `torch.finfo(logits.dtype).min` is the most-negative finite value
        # for the actual logits dtype (fp32 or fp16) — masks illegal actions
        # to ~-inf before softmax without overflowing.
        logits = logits.masked_fill(~action_mask.bool(), torch.finfo(logits.dtype).min)
        value = self.value_head(state_encoded).squeeze(-1)
        return logits, value
