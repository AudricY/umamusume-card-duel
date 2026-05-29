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
    UMA_SLOT_ORDER,
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
BELIEF_FEATURE_DIM = 16

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

_VALID_MODEL_VARIANTS = ("mlp", "set_attention", "relational")

# v6 relational scheme: default encoder hyperparameters. d_model is the trunk
# `hidden_dim` (so the cross-attention policy head and CLS value head share
# the trunk width); `relational_layers`/`relational_heads`/`relational_ffn_mult`
# size the pre-LN transformer. Unlike the set-attention probe these are NOT
# pinned to 64 — the v6 scheme is a clean break with no warm-start-parity
# constraint, so the trunk is free to be wide and deep.
RELATIONAL_DEFAULT_LAYERS = 4
RELATIONAL_DEFAULT_HEADS = 8
RELATIONAL_DEFAULT_FFN_MULT = 2
_VALID_VALUE_ADAPTERS = ("none", "mlp")

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
    # Stage-2 value-head program: optional action-value head trained against
    # per-action rootMeanQ targets from rollout-leaf MCTS. When enabled the
    # exported scalar `value` is derived from legal action Q values according
    # to `q_value_scalar`, so existing value-head-leaf MCTS can consume it
    # without an ONNX signature change. Default False keeps all legacy
    # checkpoints and graphs byte-compatible.
    uses_q_value_head: bool = False
    # Q-head scalarization used only when `uses_q_value_head=True`:
    # "max" preserves the first Stage-2 behavior, "mean" reduces max-Q
    # overestimation by averaging legal Qs, "top2_mean"/"top3_mean" average
    # the best legal Q values, and "policy_mean" uses the model's masked
    # policy distribution as action weights.
    q_value_scalar: str = "max"
    # Optional affine calibration applied to the exported Q-derived scalar,
    # then clamped back to the value range consumed by MCTS. Defaults preserve
    # existing Q-head graph behavior.
    q_value_scalar_scale: float = 1.0
    q_value_scalar_bias: float = 0.0
    # Pure-leaf value-capacity probe: optional value-only residual adapter
    # applied before `value_head` but NOT before the policy/action trunk. The
    # adapter's final Linear is zero-initialized, so enabling it preserves
    # checkpoint outputs at construction/load time while giving value retrain
    # a policy-preserving capacity path.
    value_adapter: str = "none"
    # ReBeL E2E: optional fixed-width public-belief summary vector. Defaults
    # are inert so existing checkpoints keep the same graph and parameter set.
    uses_belief_features: bool = False
    belief_feature_dim: int = BELIEF_FEATURE_DIM
    # v6 relational scheme: transformer-encoder hyperparameters consumed only
    # when `model_variant == "relational"`. d_model is `hidden_dim`. Inert for
    # mlp/set_attention (filtered defaults keep their configs byte-identical).
    relational_layers: int = RELATIONAL_DEFAULT_LAYERS
    relational_heads: int = RELATIONAL_DEFAULT_HEADS
    relational_ffn_mult: int = RELATIONAL_DEFAULT_FFN_MULT

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


class RelationalTrunk(nn.Module):
    """v6 relational / attention-native trunk (``model_variant="relational"``).

    The v6 scheme is a clean break from the sum-pool MLP. Where R7.b.3's
    ``SetAttentionEncoder`` is a *1-layer zero-init residual* bolted on top of
    the frozen v3.2 sum-pool trunk (so a v3.2 warm-start stays bit-stable),
    this trunk is the **primary representation**: a multi-layer pre-LN
    transformer that fully replaces the sum-pool path. Backward compatibility
    is explicitly out of scope, so nothing is zero-initialized for parity.

    Hypothesis: the strength bottleneck is the trunk's inability to express
    relations between board entities (the sum-pool collapses card/slot
    structure), NOT the scalar feature tail. The additive-tail schema ladder
    (v3.3-v3.8 / action v5) saturated near the same band, and those verdicts
    were taken on MCTS runs that later proved to carry training-loop
    correctness bugs (the corrected ReBeL R20 line supersedes R18/R19). v6
    therefore re-baselines the *architecture* axis under the corrected loop.

    It consumes the SAME v3.2 (slot) / belief input tensors as the sum-pool
    trunk and emits ``(logits, value)`` positionally, so a relational
    checkpoint exports to the identical 7-input / 8-input ONNX signature and
    is served by the existing ``serve_onnx`` + Rust dispatch with NO schema
    changes (dispatch keys on state_dim + input-name set, not model_variant).

    Token sequence (FROZEN order):
        position 0:                              CLS + projected 110-d state
        position 1 .. 1+NUM_CARD_TOKENS-1:       card tokens (8 zones * 30)
        next UMA_SLOT_COUNT:                     per-Uma slot tokens
        last (iff uses_belief_features):         public-belief token

    Every token also carries a learned polarity embedding (neutral / own /
    opp). This is the relational analog of the per-side-asymmetry probe: the
    two sides share weights and are distinguished by an explicit polarity
    signal rather than by separate sum-pool lanes, so the trunk reasons about
    "my board vs their board" symmetrically.
    """

    def __init__(self, config: "ModelConfig", *, card_embed: nn.Embedding) -> None:
        super().__init__()
        d = config.hidden_dim
        n_heads = config.relational_heads
        if d % n_heads != 0:
            raise ValueError(
                f"model_variant='relational' requires hidden_dim ({d}) divisible "
                f"by relational_heads ({n_heads})."
            )
        self.d_model = d
        self.uses_belief = config.uses_belief_features

        # Shared identity table (also used by the action source/target pair).
        self.card_embed = card_embed
        self.card_embed_proj = nn.Linear(CARD_EMBED_DIM, d)
        self.slot_feature_proj = nn.Linear(UMA_SLOT_FEATURE_DIM, d)
        # Project the 110-d global scalar vector into the CLS token so the
        # globals (points / turn / phase / energy budgets) the card tokens do
        # not carry stay in the trunk's input distribution.
        self.state_proj = nn.Linear(config.state_dim, d)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional / type embeddings. Small init so they do not dominate the
        # card identity signal early in training.
        self.zone_pos_embed = nn.Embedding(NUM_ZONES, d)
        self.card_slot_pos_embed = nn.Embedding(SET_ATTN_MAX_CARDS_PER_ZONE, d)
        self.uma_slot_pos_embed = nn.Embedding(UMA_SLOT_COUNT, d)
        # Polarity: 0 = neutral (e.g. stadium), 1 = own, 2 = opp.
        self.polarity_embed = nn.Embedding(3, d)
        for emb in (
            self.zone_pos_embed,
            self.card_slot_pos_embed,
            self.uma_slot_pos_embed,
            self.polarity_embed,
        ):
            nn.init.trunc_normal_(emb.weight, std=0.02)

        # Per-position polarity indices, derived from the FROZEN zone/slot
        # name orders so the mapping cannot drift silently. Registered as
        # buffers (move with .to(device), saved in the state_dict as int64).
        zone_polarity = [_name_polarity(name) for name in ZONE_ORDER]
        slot_polarity = [_name_polarity(name) for name in UMA_SLOT_ORDER]
        self.register_buffer(
            "zone_polarity", torch.tensor(zone_polarity, dtype=torch.int64), persistent=True
        )
        self.register_buffer(
            "slot_polarity", torch.tensor(slot_polarity, dtype=torch.int64), persistent=True
        )

        if self.uses_belief:
            self.belief_proj = nn.Linear(config.belief_feature_dim, d)
            # A learned "this is the belief token" type marker.
            self.belief_type = nn.Parameter(torch.zeros(1, 1, d))
            nn.init.trunc_normal_(self.belief_type, std=0.02)
        else:
            self.belief_proj = None
            self.belief_type = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=config.relational_ffn_mult * d,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.relational_layers)

        # Candidate-conditioned policy head: each action is a query token that
        # cross-attends over the encoded board, so every candidate gets a
        # board summary filtered through its own lens (the relational payoff
        # over a single shared pooled context).
        self.action_in_proj = nn.Linear(
            config.action_dim + ACTION_PAIR_FANOUT * CARD_EMBED_DIM, d
        )
        self.action_query_norm = nn.LayerNorm(d)
        self.policy_cross_attn = nn.MultiheadAttention(
            d, n_heads, dropout=config.dropout, batch_first=True
        )
        self.policy_out = nn.Sequential(
            nn.LayerNorm(3 * d),
            nn.Linear(3 * d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )
        # Value head reads the CLS token (whole public-state value); for a
        # belief graph the CLS has already attended to the belief token.
        self.value_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        state_features: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
        card_ids_by_zone: torch.Tensor | None,
        action_card_idx: torch.Tensor | None,
        uma_slot_card_ids: torch.Tensor | None,
        uma_slot_features: torch.Tensor | None,
        belief_features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = state_features.shape[0]
        num_actions = action_features.shape[1]
        device = state_features.device
        dtype = state_features.dtype
        d = self.d_model

        # Defensive defaults mirror the legacy forward: absent tensors read as
        # all-pad (card_id 0 → embed(0)=0 via padding_idx, pad mask True).
        if card_ids_by_zone is None:
            card_ids_by_zone = torch.zeros(
                (batch_size, NUM_ZONES, SET_ATTN_MAX_CARDS_PER_ZONE),
                dtype=torch.int64,
                device=device,
            )
        if uma_slot_card_ids is None or uma_slot_features is None:
            uma_slot_card_ids = torch.zeros(
                (batch_size, UMA_SLOT_COUNT), dtype=torch.int64, device=device
            )
            uma_slot_features = torch.zeros(
                (batch_size, UMA_SLOT_COUNT, UMA_SLOT_FEATURE_DIM),
                dtype=dtype,
                device=device,
            )

        # CLS token = learned vector + projected global state vector.
        cls = self.cls_token.expand(batch_size, -1, -1).to(dtype=dtype)
        cls = cls + self.state_proj(state_features).unsqueeze(1)

        # Card tokens: embed → project to d, + zone-type + within-zone-pos +
        # zone polarity. Pad where card_id == 0.
        card_tokens = self.card_embed_proj(self.card_embed(card_ids_by_zone))
        card_tokens = card_tokens + self.zone_pos_embed.weight.unsqueeze(0).unsqueeze(2)
        card_tokens = card_tokens + self.card_slot_pos_embed.weight.unsqueeze(0).unsqueeze(0)
        zone_pol = self.polarity_embed(self.zone_polarity)  # [NUM_ZONES, d]
        card_tokens = card_tokens + zone_pol.unsqueeze(0).unsqueeze(2)
        card_tokens = card_tokens.reshape(batch_size, SET_ATTN_NUM_CARD_TOKENS, d)
        card_pad = (card_ids_by_zone == 0).reshape(batch_size, SET_ATTN_NUM_CARD_TOKENS)

        # Uma slot tokens: embed + per-slot scalar features + slot-pos +
        # slot polarity. Pad where the slot is absent (card_id == 0).
        slot_tokens = self.card_embed_proj(self.card_embed(uma_slot_card_ids))
        slot_tokens = slot_tokens + self.slot_feature_proj(uma_slot_features)
        slot_tokens = slot_tokens + self.uma_slot_pos_embed.weight.unsqueeze(0)
        slot_pol = self.polarity_embed(self.slot_polarity)  # [UMA_SLOT_COUNT, d]
        slot_tokens = slot_tokens + slot_pol.unsqueeze(0)
        slot_pad = uma_slot_card_ids == 0

        token_list = [cls, card_tokens, slot_tokens]
        cls_pad = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        pad_list = [cls_pad, card_pad, slot_pad]

        if self.uses_belief:
            if belief_features is None:
                belief_features = torch.zeros(
                    (batch_size, self.belief_proj.in_features), dtype=dtype, device=device
                )
            belief_token = self.belief_proj(belief_features).unsqueeze(1) + self.belief_type
            token_list.append(belief_token.to(dtype=dtype))
            pad_list.append(torch.zeros((batch_size, 1), dtype=torch.bool, device=device))

        tokens = torch.cat(token_list, dim=1)
        pad_mask = torch.cat(pad_list, dim=1)

        # CLS is always unmasked, so no attention row is fully masked → no NaN.
        encoded = self.encoder(tokens, src_key_padding_mask=pad_mask)
        cls_out = encoded[:, 0, :]

        # Candidate queries: action features ⊕ source/target card embeds.
        if action_card_idx is not None:
            pair_embed = self.card_embed(action_card_idx).reshape(
                batch_size, num_actions, ACTION_PAIR_FANOUT * CARD_EMBED_DIM
            )
        else:
            pair_embed = torch.zeros(
                (batch_size, num_actions, ACTION_PAIR_FANOUT * CARD_EMBED_DIM),
                dtype=action_features.dtype,
                device=device,
            )
        query = self.action_in_proj(torch.cat([action_features, pair_embed], dim=-1))
        query = self.action_query_norm(query)
        attended, _ = self.policy_cross_attn(
            query, encoded, encoded, key_padding_mask=pad_mask, need_weights=False
        )
        joint = torch.cat([attended, query, attended * query], dim=-1)
        logits = self.policy_out(joint).squeeze(-1)
        logits = logits.masked_fill(~action_mask.bool(), torch.finfo(logits.dtype).min)

        value = self.value_head(cls_out).squeeze(-1)
        return logits, value


def _name_polarity(name: str) -> int:
    """Map a frozen zone / slot name to a polarity index (0 neutral, 1 own,
    2 opp). Used to build the relational trunk's polarity buffers."""

    if name.startswith("own"):
        return 1
    if name.startswith("opp"):
        return 2
    return 0


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
        if self.config.value_adapter not in _VALID_VALUE_ADAPTERS:
            raise ValueError(
                f"unknown value_adapter {self.config.value_adapter!r}; "
                f"expected one of {_VALID_VALUE_ADAPTERS}"
            )
        if self.config.belief_feature_dim <= 0:
            raise ValueError("belief_feature_dim must be positive")
        is_relational = self.config.model_variant == "relational"
        if is_relational:
            # v6 consumes the v3.2 slot-token tensors (so it exports to the
            # 7-input / 8-input signature the existing dispatch already
            # serves). The Q-head / value-adapter are MCTS-era sum-pool
            # add-ons outside the v6 scope.
            if not self.config.uses_uma_slot_tokens:
                raise ValueError(
                    "model_variant='relational' requires uses_uma_slot_tokens=True; "
                    "the v6 relational trunk consumes the v3.2 slot-token tensors."
                )
            if self.config.uses_q_value_head or self.config.value_adapter != "none":
                raise ValueError(
                    "model_variant='relational' does not support uses_q_value_head "
                    "or value_adapter (outside v6 scope); leave both at defaults."
                )
        hidden = self.config.hidden_dim
        if is_relational:
            # v6 relational scheme: a multi-layer attention trunk REPLACES the
            # sum-pool MLP. Build only the shared card-embed table + the
            # relational stack, None out every legacy submodule (so a
            # relational checkpoint's state_dict carries no dead sum-pool
            # parameters), and return — the legacy construction below is
            # skipped entirely. The shared `card_embed` is constructed here
            # rather than reusing the legacy position so the relational path
            # has no init-order coupling to the mlp/set_attention path.
            self.card_embed = nn.Embedding(CARD_VOCAB_TABLE_SIZE, CARD_EMBED_DIM, padding_idx=0)
            self.state_encoder = None
            self.action_encoder = None
            self.belief_encoder = None
            self.zone_projection = None
            self.joint_projection = None
            self.joint_blocks = None
            self.policy_head = None
            self.value_head = None
            self.value_adapter = None
            self.q_value_head = None
            self.uma_slot_encoder = None
            self.set_attention_encoder = None
            self.relational = RelationalTrunk(self.config, card_embed=self.card_embed)
            return
        self.relational = None
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
        if self.config.uses_belief_features:
            self.belief_encoder = nn.Sequential(
                nn.Linear(self.config.belief_feature_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden, bias=False),
            )
            nn.init.zeros_(self.belief_encoder[-1].weight)
        else:
            self.belief_encoder = None
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
        if self.config.value_adapter == "mlp":
            self.value_adapter = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden * 2),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(hidden * 2, hidden, bias=False),
            )
            nn.init.zeros_(self.value_adapter[-1].weight)
        else:
            self.value_adapter = None
        if self.config.uses_q_value_head:
            self.q_value_head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 1),
                nn.Tanh(),
            )
        else:
            self.q_value_head = None

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
        belief_features: torch.Tensor | None = None,
        return_q_values: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
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

        # v6 relational scheme: a fully separate attention trunk that emits
        # the same (logits, value) — dispatch here and skip the sum-pool path.
        if self.relational is not None:
            logits, value = self.relational(
                state_features,
                action_features,
                action_mask,
                card_ids_by_zone,
                action_card_idx,
                uma_slot_card_ids,
                uma_slot_features,
                belief_features,
            )
            if return_q_values:
                return logits, value, None
            return logits, value

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

        if (
            self.config.uses_belief_features
            and self.belief_encoder is not None
            and belief_features is not None
        ):
            state_encoded = state_encoded + self.belief_encoder(belief_features)

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
        q_values: torch.Tensor | None = None
        if self.q_value_head is not None:
            q_values = self.q_value_head(joint).squeeze(-1)
            masked_q = q_values.masked_fill(~action_mask.bool(), torch.finfo(q_values.dtype).min)
            scalar_mode = self.config.q_value_scalar
            if scalar_mode == "max":
                value = masked_q.max(dim=1).values
            elif scalar_mode == "mean":
                legal = action_mask.bool()
                legal_count = legal.sum(dim=1).clamp_min(1).to(q_values.dtype)
                value = q_values.masked_fill(~legal, 0.0).sum(dim=1) / legal_count
            elif scalar_mode in {"top2_mean", "top3_mean"}:
                k = 2 if scalar_mode == "top2_mean" else 3
                legal = action_mask.bool()
                sorted_q = masked_q.sort(dim=1, descending=True).values
                ranks = torch.arange(sorted_q.shape[1], device=sorted_q.device).unsqueeze(0)
                legal_count_long = legal.sum(dim=1, keepdim=True).clamp_min(1)
                top_count = legal_count_long.clamp_max(k)
                top_mask = ranks < top_count
                denom = top_count.squeeze(1).to(q_values.dtype)
                value = sorted_q.masked_fill(~top_mask, 0.0).sum(dim=1) / denom
            elif scalar_mode == "policy_mean":
                weights = torch.softmax(logits, dim=1).to(q_values.dtype)
                value = (weights * q_values).sum(dim=1)
            else:
                raise ValueError(f"unknown q_value_scalar={scalar_mode!r}")
            if self.config.q_value_scalar_scale != 1.0 or self.config.q_value_scalar_bias != 0.0:
                value = (
                    value * float(self.config.q_value_scalar_scale)
                    + float(self.config.q_value_scalar_bias)
                ).clamp(-1.0, 1.0)
        else:
            value_state_encoded = state_encoded
            if self.value_adapter is not None:
                value_state_encoded = value_state_encoded + self.value_adapter(value_state_encoded)
            value = self.value_head(value_state_encoded).squeeze(-1)
        if return_q_values:
            return logits, value, q_values
        return logits, value
