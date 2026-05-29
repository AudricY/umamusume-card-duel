"""Smoke the v6 relational model scheme (``model_variant="relational"``).

The v6 scheme is a clean break from the sum-pool MLP: a multi-layer
attention trunk that consumes the SAME v3.2 (slot) / belief input tensors and
emits the SAME ``(logits, value)`` outputs, so a relational checkpoint rides
the existing ``serve_onnx`` + Rust dispatch with NO new graph signature.

This smoke does NOT need a `runs/` checkpoint (v6 trains from scratch — there
is no warm-start-parity contract). It constructs fresh relational models and
asserts:

  1. forward(): logits [B, A], value [B], legal logits finite, illegal logits
     masked to finfo.min, value within tanh [-1, 1] — for hidden=128/256 and
     belief on/off.
  2. The relational trunk carries NO dead sum-pool parameters: every legacy
     submodule (state_encoder, joint_blocks, value_head, uma_slot_encoder,
     set_attention_encoder, ...) is None on a relational net.
  3. ONNX export roundtrip (PyTorch eager vs onnxruntime) max_abs_diff over
     legal logits and value is below 1e-3 — for the 7-input (slot) and
     8-input (slot + belief) graphs.
  4. Graph signature parity: the 7-input graph's input-name set is EXACTLY
     the v3.2 contract, and the 8-input graph appends only `belief_features`.
     This is what lets the existing dispatch serve v6 unchanged.
  5. Regression: mlp + set_attention variants still construct and report
     relational=None (their forward path is untouched).
  6. Invariants: relational requires uses_uma_slot_tokens and rejects
     hidden_dim indivisible by relational_heads.

Run: ``training/.venv/bin/python training/v6_relational_smoke.py``
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uma_ai.features import (  # noqa: E402
    ACTION_DIM,
    CARD_ID_SHAPES,
    STATE_DIM,
    UMA_SLOT_COUNT,
    UMA_SLOT_FEATURE_DIM,
    ZONE_ORDER,
)
from uma_ai.model import CandidatePolicyNet, ModelConfig  # noqa: E402

NUM_ZONES = len(ZONE_ORDER)
MAX_CARDS = max(CARD_ID_SHAPES.values())

# The frozen v3.2 7-input ONNX contract (order matters for the Rust /
# serve_onnx set-equality + positional binding).
V32_INPUTS = [
    "state_features",
    "action_features",
    "action_mask",
    "card_ids_by_zone",
    "action_card_idx",
    "uma_slot_card_ids",
    "uma_slot_features",
]

PASS_THRESHOLD = 1e-3


def _make_batch(B: int, A: int, belief_dim: int | None, seed: int):
    g = torch.Generator().manual_seed(seed)
    state = torch.randn(B, STATE_DIM, generator=g)
    actions = torch.randn(B, A, ACTION_DIM, generator=g)
    # At least one legal action per row, plus some random extras, plus one
    # guaranteed-illegal column to exercise the mask fill.
    mask = torch.zeros(B, A, dtype=torch.bool)
    mask[:, 0] = True
    mask = mask | (torch.rand(B, A, generator=g) > 0.4)
    mask[:, A - 1] = False
    czb = torch.randint(0, 60, (B, NUM_ZONES, MAX_CARDS), generator=g)
    aci = torch.randint(0, 60, (B, A, 2), generator=g)
    sids = torch.randint(0, 60, (B, UMA_SLOT_COUNT), generator=g)
    sfeat = torch.randn(B, UMA_SLOT_COUNT, UMA_SLOT_FEATURE_DIM, generator=g)
    bel = torch.randn(B, belief_dim, generator=g) if belief_dim else None
    return state, actions, mask, czb, aci, sids, sfeat, bel


class _SlotWrap(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = w

    def forward(self, st, ac, mk, zx, ai, si, sf):  # noqa: ANN001
        return self.w(
            st, ac, mk, card_ids_by_zone=zx, action_card_idx=ai,
            uma_slot_card_ids=si, uma_slot_features=sf,
        )


class _SlotBeliefWrap(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = w

    def forward(self, st, ac, mk, zx, ai, si, sf, bx):  # noqa: ANN001
        return self.w(
            st, ac, mk, card_ids_by_zone=zx, action_card_idx=ai,
            uma_slot_card_ids=si, uma_slot_features=sf, belief_features=bx,
        )


def _forward_checks(results: list[tuple[str, bool, str]]) -> None:
    # (hidden, belief, card_features, contextual_actions)
    cases = [
        (128, False, False, True),
        (256, False, True, True),
        (128, True, True, True),
        (128, True, False, False),  # ablation: no catalog, no contextual gather
    ]
    for hidden, belief, card_features, contextual in cases:
        cfg = ModelConfig(
            state_dim=STATE_DIM, hidden_dim=hidden, depth=3,
            uses_uma_slot_tokens=True, model_variant="relational",
            uses_belief_features=belief, uses_card_features=card_features,
            relational_contextual_actions=contextual,
        )
        m = CandidatePolicyNet(cfg).eval()
        B, A = 3, 7
        st, ac, mk, czb, aci, sids, sfeat, bel = _make_batch(
            B, A, cfg.belief_feature_dim if belief else None, seed=hidden + int(belief)
        )
        with torch.no_grad():
            logits, value = m(
                st, ac, mk, card_ids_by_zone=czb, action_card_idx=aci,
                uma_slot_card_ids=sids, uma_slot_features=sfeat, belief_features=bel,
            )
        ok = True
        detail = ""
        if logits.shape != (B, A) or value.shape != (B,):
            ok, detail = False, f"shapes {tuple(logits.shape)} {tuple(value.shape)}"
        elif not torch.isfinite(logits[mk]).all() or not torch.isfinite(value).all():
            ok, detail = False, "non-finite legal logits / value"
        elif not (logits[~mk] <= torch.finfo(logits.dtype).min / 2).all():
            ok, detail = False, "illegal logits not masked"
        elif not (value.abs() <= 1.0 + 1e-6).all():
            ok, detail = False, "value outside tanh range"
        else:
            nparams = sum(p.numel() for p in m.parameters())
            detail = f"params={nparams:,}"
        results.append((f"forward hidden={hidden} belief={belief}", ok, detail))


def _no_dead_params_check(results: list[tuple[str, bool, str]]) -> None:
    cfg = ModelConfig(
        state_dim=STATE_DIM, hidden_dim=128, depth=3,
        uses_uma_slot_tokens=True, model_variant="relational",
    )
    m = CandidatePolicyNet(cfg)
    legacy_attrs = [
        "state_encoder", "action_encoder", "belief_encoder", "zone_projection",
        "joint_projection", "joint_blocks", "policy_head", "value_head",
        "value_adapter", "q_value_head", "uma_slot_encoder", "set_attention_encoder",
    ]
    leaked = [a for a in legacy_attrs if getattr(m, a) is not None]
    ok = m.relational is not None and not leaked
    detail = "relational stack only" if ok else f"leaked legacy modules: {leaked}"
    results.append(("no dead sum-pool params on relational net", ok, detail))


def _roundtrip_check(
    belief: bool, results: list[tuple[str, bool, str]], card_features: bool = False
) -> list[str]:
    import onnxruntime as ort

    cfg = ModelConfig(
        state_dim=STATE_DIM, hidden_dim=128, depth=3,
        uses_uma_slot_tokens=True, model_variant="relational",
        uses_belief_features=belief, uses_card_features=card_features,
    )
    m = CandidatePolicyNet(cfg).eval()
    B, A = 2, 6
    st, ac, mk, czb, aci, sids, sfeat, bel = _make_batch(
        B, A, cfg.belief_feature_dim if belief else None, seed=99 + int(belief)
    )
    names = list(V32_INPUTS)
    inputs = [st, ac, mk, czb, aci, sids, sfeat]
    if belief:
        names.append("belief_features")
        inputs.append(bel)
        export_model = _SlotBeliefWrap(m)
    else:
        export_model = _SlotWrap(m)
    export_model.eval()
    with torch.no_grad():
        tl, tv = m(
            st, ac, mk, card_ids_by_zone=czb, action_card_idx=aci,
            uma_slot_card_ids=sids, uma_slot_features=sfeat,
            belief_features=(bel if belief else None),
        )
    dax = {n: {0: "batch"} for n in names}
    for n in ("action_features", "action_mask", "action_card_idx"):
        dax[n][1] = "actions"
    dax["logits"] = {0: "batch", 1: "actions"}
    dax["value"] = {0: "batch"}
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "v6.onnx")
        torch.onnx.export(
            export_model, tuple(inputs), path, input_names=names,
            output_names=["logits", "value"], dynamic_axes=dax, opset_version=17,
        )
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        graph_in = [i.name for i in sess.get_inputs()]
        feed = {}
        for n, inp in zip(names, inputs):
            if inp.dtype == torch.int64:
                feed[n] = inp.numpy().astype(np.int64)
            elif inp.dtype == torch.bool:
                feed[n] = inp.numpy().astype(np.bool_)
            else:
                feed[n] = inp.numpy().astype(np.float32)
        ol, ov = sess.run(["logits", "value"], feed)
    legal = mk.numpy()
    dl = float(np.abs(tl.numpy()[legal] - ol[legal]).max())
    dv = float(np.abs(tv.numpy() - ov.reshape(-1)).max())
    ok = dl < PASS_THRESHOLD and dv < PASS_THRESHOLD
    results.append((
        f"onnx roundtrip belief={belief} card_features={card_features}", ok,
        f"max_abs_diff logits={dl:.2e} value={dv:.2e}",
    ))
    return graph_in


def _signature_checks(gi7: list[str], gi8: list[str], results: list[tuple[str, bool, str]]) -> None:
    ok7 = gi7 == V32_INPUTS
    results.append((
        "7-input graph signature == v3.2 contract", ok7,
        "match" if ok7 else f"got {gi7}",
    ))
    ok8 = gi8 == V32_INPUTS + ["belief_features"]
    results.append((
        "8-input graph signature == v3.2 + belief_features", ok8,
        "match" if ok8 else f"got {gi8}",
    ))


def _regression_checks(results: list[tuple[str, bool, str]]) -> None:
    m_mlp = CandidatePolicyNet(ModelConfig(uses_uma_slot_tokens=True))
    results.append((
        "mlp variant still constructs, relational=None",
        m_mlp.relational is None and m_mlp.state_encoder is not None, "",
    ))
    m_sa = CandidatePolicyNet(
        ModelConfig(hidden_dim=64, uses_uma_slot_tokens=True, model_variant="set_attention")
    )
    results.append((
        "set_attention still constructs, relational=None",
        m_sa.relational is None and m_sa.set_attention_encoder is not None, "",
    ))


def _invariant_checks(results: list[tuple[str, bool, str]]) -> None:
    # relational requires uma slot tokens.
    try:
        CandidatePolicyNet(ModelConfig(model_variant="relational", uses_uma_slot_tokens=False))
        ok = False
    except ValueError:
        ok = True
    results.append(("relational rejects uses_uma_slot_tokens=False", ok, ""))
    # hidden_dim must be divisible by relational_heads.
    try:
        CandidatePolicyNet(ModelConfig(
            hidden_dim=130, model_variant="relational", uses_uma_slot_tokens=True,
            relational_heads=8,
        ))
        ok = False
    except ValueError:
        ok = True
    results.append(("relational rejects hidden_dim % heads != 0", ok, ""))


def main() -> int:
    results: list[tuple[str, bool, str]] = []
    _forward_checks(results)
    _no_dead_params_check(results)
    gi7 = _roundtrip_check(False, results, card_features=True)
    gi8 = _roundtrip_check(True, results, card_features=True)
    _signature_checks(gi7, gi8, results)
    _regression_checks(results)
    _invariant_checks(results)

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    for name, ok, detail in results:
        tag = "PASS" if ok else "FAIL"
        suffix = f"  [{detail}]" if detail else ""
        print(f"  [{tag}] {name}{suffix}")
    print(f"{passed}/{total} cases PASS")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
