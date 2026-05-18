"""R16-P1: 164-d mcts-distill trainer wiring + additive-init equivalence.

Two load-bearing guarantees for the v3.1-vs-v3.0 ablation:

  A. Additive-init equivalence — the 164-d additive-tail-zero init
     checkpoint, fed an observation's v3.1 features, produces logits/value
     numerically identical (<=1e-6) to the 110-d source checkpoint fed the
     SAME observation's v3.0 features. Proves the 54 new tail features
     contribute exactly zero at init, so the ablation isolates the schema's
     learned lift. If this cannot hold, the ablation is not interpretable.

  B. 164-d mcts-distill gradient — one mcts-distill batch built at
     state_dim=164 runs forward/backward through `MctsSelfPlayDataset` +
     `collate_mcts_selfplay_batch` + the real `train_bc.run_epoch` loss,
     producing `[B,164]` state batches AND nonzero gradient on BOTH the
     new input-projection columns [110:164] and the existing [0:110].

Deliberately lean: a wiring + invariant check, not a strength test.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uma_ai.features import (  # noqa: E402
    STATE_DIM_V3,
    STATE_DIM_V3_1,
    observation_to_features,
    observation_to_features_v3_1,
)
from uma_ai.model import CandidatePolicyNet, ModelConfig  # noqa: E402
from uma_ai.selfplay_dataset import (  # noqa: E402
    MctsSelfPlayDataset,
    collate_mcts_selfplay_batch,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_CKPT = _REPO_ROOT / "runs/R7b2-card-embed/iter-000/model/checkpoint.pt"
INIT_164 = _REPO_ROOT / "runs/R16-P1-v31-ablation/init-164/checkpoint.pt"


def fail(msg: str) -> None:
    print(f"[r16-v31-distill-init-smoke] FAIL — {msg}", file=sys.stderr)
    sys.exit(1)


def _observation() -> dict:
    # Mirror the v3.1 feature smoke fixture so the v3.0 head + v3.1 tail
    # are both exercised on a realistic observation.
    from r16_temporal_v31_smoke import _observation as base_obs

    return base_obs()


def _legal_actions() -> list[dict]:
    return [
        {"id": "a0", "features": [0.0] * 48, "actionSourceCardIdx": 3, "actionTargetCardIdx": 0},
        {"id": "a1", "features": [0.1] * 48, "actionSourceCardIdx": None, "actionTargetCardIdx": None},
        {"id": "a2", "features": [0.2] * 48, "actionSourceCardIdx": 2, "actionTargetCardIdx": 1},
    ]


def _load_model(ckpt_path: Path) -> tuple[CandidatePolicyNet, ModelConfig]:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ModelConfig.from_dict(payload.get("model_config"))
    model = CandidatePolicyNet(config)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config


def check_additive_equivalence() -> None:
    if not SOURCE_CKPT.exists():
        fail(f"source checkpoint missing (env gap): {SOURCE_CKPT}")
    if not INIT_164.exists():
        fail(f"164-d init missing — run make_v31_additive_init.py: {INIT_164}")

    src_model, src_cfg = _load_model(SOURCE_CKPT)
    new_model, new_cfg = _load_model(INIT_164)
    if src_cfg.state_dim != STATE_DIM_V3:
        fail(f"source config state_dim {src_cfg.state_dim} != {STATE_DIM_V3}")
    if new_cfg.state_dim != STATE_DIM_V3_1:
        fail(f"init-164 config state_dim {new_cfg.state_dim} != {STATE_DIM_V3_1}")

    obs = _observation()
    legal = _legal_actions()
    f110 = observation_to_features(obs)
    f164 = observation_to_features_v3_1(obs)
    if not np.array_equal(f164[:STATE_DIM_V3], f110):
        fail("v3.1 head [0:110] != v3.0 features (frozen-head invariant broke)")

    from uma_ai.features import (
        action_card_idx_pair,
        legal_actions_to_features,
        observation_to_card_ids,
    )
    from uma_ai.model import NUM_ZONES
    from uma_ai.features import CARD_ID_SHAPES, ZONE_ORDER

    af = legal_actions_to_features(legal)
    czi = observation_to_card_ids(obs)
    aci = np.stack([action_card_idx_pair(a) for a in legal], axis=0)
    max_cards = max(CARD_ID_SHAPES.values())
    czi_t = torch.zeros((1, NUM_ZONES, max_cards), dtype=torch.int64)
    for zi, zone in enumerate(ZONE_ORDER):
        arr = czi[zone]
        czi_t[0, zi, : arr.shape[0]] = torch.from_numpy(arr)
    aci_t = torch.from_numpy(aci).unsqueeze(0)
    af_t = torch.from_numpy(af).unsqueeze(0)
    mask_t = torch.ones((1, af.shape[0]), dtype=torch.bool)

    with torch.no_grad():
        src_logits, src_value = src_model(
            torch.from_numpy(f110).unsqueeze(0), af_t, mask_t, czi_t, aci_t
        )
        new_logits, new_value = new_model(
            torch.from_numpy(f164).unsqueeze(0), af_t, mask_t, czi_t, aci_t
        )
    dl = float((src_logits - new_logits).abs().max())
    dv = float((src_value - new_value).abs().max())
    if dl > 1e-6 or dv > 1e-6:
        fail(
            f"additive-init NOT equivalent at init: max|Δlogits|={dl:.3e} "
            f"max|Δvalue|={dv:.3e} (>1e-6). New tail is not numerically null "
            f"— the ablation would be uninterpretable. STOP."
        )
    print(
        f"  PASS  additive-init equivalence: 164-d init == 110-d source "
        f"(max|Δlogits|={dl:.2e}, max|Δvalue|={dv:.2e})"
    )


def _write_mcts_row(obs: dict, legal: list[dict]) -> dict:
    return {
        "kind": "mcts-selfplay",
        "schemaVersion": 1,
        "observation": obs,
        "legalActions": legal,
        "visitDistribution": [10.0, 5.0, 1.0],
        "valueTarget": 0.25,
        "sampleWeight": 1.0,
    }


def check_164_distill_gradient() -> None:
    import train_bc

    obs = _observation()
    legal = _legal_actions()
    row = _write_mcts_row(obs, legal)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "mcts.jsonl"
        p.write_text(
            "\n".join(json.dumps(row) for _ in range(6)) + "\n", encoding="utf8"
        )
        ds = MctsSelfPlayDataset(p, min_actions=2, state_dim=STATE_DIM_V3_1)
        if len(ds) != 6:
            fail(f"expected 6 mcts samples, got {len(ds)}")
        for s in ds:
            if s.state_features.shape != (STATE_DIM_V3_1,):
                fail(f"sample state width {s.state_features.shape} != (164,)")
        batch = collate_mcts_selfplay_batch(list(ds))
        sf = batch["state_features"]
        if tuple(sf.shape) != (6, STATE_DIM_V3_1):
            fail(f"collated state batch {tuple(sf.shape)} != (6, 164)")
        if "card_ids_by_zone" not in batch or "action_card_idx" not in batch:
            fail("164-d distill batch must carry the embedding tensors (v3.0 head)")

        model = CandidatePolicyNet(ModelConfig(state_dim=STATE_DIM_V3_1, hidden_dim=32, depth=2))
        model.train()
        # Use the REAL train_bc soft-CE distill loss path so the wiring
        # under test is the production loss, not a re-implementation. We
        # drive one forward/backward directly (rather than run_epoch, which
        # zero_grad(set_to_none)s at the end and would discard the grads we
        # must inspect).
        logits, values = model(
            batch["state_features"],
            batch["action_features"],
            batch["action_mask"],
            card_ids_by_zone=batch.get("card_ids_by_zone"),
            action_card_idx=batch.get("action_card_idx"),
        )
        weights = train_bc.normalized_weights(batch["sample_weights"])
        log_probs = train_bc.masked_log_softmax_logits(logits, batch["action_mask"])
        per_row = -(batch["policy_targets"] * log_probs).sum(dim=1)
        policy_loss = train_bc.weighted_mean(per_row, weights)
        value_loss = train_bc.weighted_mean(
            torch.nn.functional.mse_loss(values, batch["value_targets"], reduction="none"),
            weights,
        )
        (policy_loss + 0.1 * value_loss).backward()

        w = model.state_encoder[0].weight
        if w.grad is None:
            fail("no gradient on state_encoder.0.weight after distill backward")
        g_new = float(w.grad[:, STATE_DIM_V3:STATE_DIM_V3_1].abs().sum())
        g_old = float(w.grad[:, :STATE_DIM_V3].abs().sum())
        if not (g_new > 0.0):
            fail(
                f"zero gradient on the NEW input-projection columns "
                f"[110:164] (sum|grad|={g_new}); v3.1 tail is not training"
            )
        if not (g_old > 0.0):
            fail(f"zero gradient on existing columns [0:110] (sum|grad|={g_old})")
    print(
        f"  PASS  164-d mcts-distill fwd/bwd: [6,164] batch, grad on new "
        f"[110:164] (Σ={g_new:.3e}) AND existing [0:110] (Σ={g_old:.3e})"
    )


def main() -> None:
    check_additive_equivalence()
    check_164_distill_gradient()
    print("r16-v31-distill-init smoke: ALL PASS")


if __name__ == "__main__":
    main()
