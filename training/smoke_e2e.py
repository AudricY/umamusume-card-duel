from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from uma_ai.dataset import load_policy_samples
from uma_ai.features import (
    CARD_ID_SHAPES,
    ZONE_ORDER,
    action_card_idx_pair,
    card_vocab_metadata,
    legal_actions_to_features,
    observation_to_card_ids,
    observation_to_features,
)
from uma_ai.node_bridge import export_training_examples


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = repo_root / "training" / "runs" / "smoke"
    model_dir = run_dir / "model"
    examples_path = run_dir / "examples.jsonl"
    onnx_path = model_dir / "policy.onnx"
    run_dir.mkdir(parents=True, exist_ok=True)

    export_training_examples(repo_root, examples_path, seed_start=7000, games=14, max_steps=360)
    train = [
        sys.executable,
        str(repo_root / "training" / "train_bc.py"),
        "--data",
        str(examples_path),
        "--out-dir",
        str(model_dir),
        "--epochs",
        "16",
        "--batch-size",
        "32",
        "--hidden-dim",
        "96",
        "--depth",
        "2",
    ]
    subprocess.run(train, cwd=repo_root, check=True)
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf8"))
    assert_grouped_split(manifest)
    assert_card_vocab_recorded(manifest)
    subprocess.run([
        sys.executable,
        str(repo_root / "training" / "export_onnx.py"),
        "--checkpoint",
        str(model_dir / "checkpoint.pt"),
        "--out",
        str(onnx_path),
    ], cwd=repo_root, check=True)

    sample = next(iter(load_policy_samples(examples_path)))
    direct_prediction = run_onnx_prediction(onnx_path, sample.example)
    port = free_port()
    server = subprocess.Popen([
        sys.executable,
        str(repo_root / "training" / "serve_onnx.py"),
        "--model",
        str(onnx_path),
        "--port",
        str(port),
    ], cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        wait_for_health(port)
        served_prediction = post_json(
            f"http://127.0.0.1:{port}/predict",
            {
                "observation": sample.example["observation"],
                "legalActions": sample.example["legalActions"],
            },
        )
        selected = int(served_prediction["selectedIndex"][0])
        if selected < 0 or selected >= len(sample.example["legalActions"]):
            raise AssertionError(f"Server selected invalid index {selected}")
        # R7.b.2 Phase 3: served-prediction (via `request_to_arrays`)
        # must match direct-ORT (with manually-built tensors). This
        # catches `request_to_arrays` packing bugs distinct from any
        # ONNX-export issue — if the served logits diverge from the
        # direct ones, `request_to_arrays` is the culprit, because both
        # paths feed the same ONNX session.
        served_logits = served_prediction.get("logits")
        if served_logits is None:
            raise AssertionError("/predict response missing logits")
        direct_logits = direct_prediction.get("logits")
        if direct_logits is None:
            raise AssertionError("direct ORT prediction missing logits")
        served_arr = np.asarray(served_logits, dtype=np.float64)
        direct_arr = np.asarray(direct_logits, dtype=np.float64)
        if served_arr.shape != direct_arr.shape:
            raise AssertionError(
                f"R7.b.2 Phase 3: served/direct logits shape mismatch — "
                f"served {served_arr.shape} vs direct {direct_arr.shape}"
            )
        # Only compare legal positions; ORT padding fills are bumped to
        # -1e9 by `masked_log_softmax` only on the server side, so the
        # raw logits there can differ on padded positions.
        legal_count_first = len(sample.example["legalActions"])
        served_legal = served_arr[0, :legal_count_first]
        direct_legal = direct_arr[0, :legal_count_first]
        max_legal_diff = float(np.abs(served_legal - direct_legal).max())
        if max_legal_diff > 1e-3:
            raise AssertionError(
                f"R7.b.2 Phase 3: served vs direct ORT logits diverge "
                f"({max_legal_diff:.6f} > 1e-3); request_to_arrays packing "
                f"likely differs from run_onnx_prediction packing"
            )
        # Item 18: behavior-policy logging is in the serving path so the
        # warm-start checkpoint's PPO rollouts can recover importance ratios
        # without a serve-side change. /predict must return per-action
        # log-probs and the chosen-action log-prob.
        legal_count = len(sample.example["legalActions"])
        action_log_probs = served_prediction.get("actionLogProbs")
        if not action_log_probs or len(action_log_probs[0]) < legal_count:
            raise AssertionError(
                f"actionLogProbs missing or wrong length: {action_log_probs}"
            )
        selected_log_prob = served_prediction.get("selectedLogProb")
        if not selected_log_prob:
            raise AssertionError("selectedLogProb missing from /predict response")
        # The behavior distribution is softmax(logits), regardless of the
        # greedy/argmax selection rule. Chosen-action log-prob therefore lies
        # in (-inf, 0] and is the maximum over legal positions because the
        # selection is the argmax. Both invariants are checked here.
        chosen_lp = float(selected_log_prob[0])
        if chosen_lp > 1e-6:
            raise AssertionError(f"selectedLogProb must be <=0; got {chosen_lp}")
        max_legal_lp = max(action_log_probs[0][:legal_count])
        if chosen_lp < max_legal_lp - 1e-3:
            raise AssertionError(
                f"selectedLogProb {chosen_lp} should be argmax legal log-prob {max_legal_lp}"
            )
        # Sum of legal-action probs must be ~1.0; masked positions must
        # contribute zero.
        action_probs = served_prediction.get("actionProbs")
        if not action_probs:
            raise AssertionError("actionProbs missing from /predict response")
        legal_prob_sum = sum(action_probs[0][:legal_count])
        if not (0.999 <= legal_prob_sum <= 1.001):
            raise AssertionError(
                f"Sum of legal-action probs must be ~1.0; got {legal_prob_sum}"
            )
        behavior = served_prediction.get("behaviorPolicy")
        if not behavior or behavior.get("kind") != "greedy":
            raise AssertionError(f"behaviorPolicy.kind must be greedy; got {behavior}")

        # F1/PPO: stochastic sampling mode with Gumbel-max. Same request
        # body but with sampling=stochastic must return behaviorPolicy
        # {kind: 'stochastic', temperature: T}. Same samplingSeed produces
        # the same selection; legal-prob mass stays at 1.0. Across many
        # different seeds at T=1 the selected index must take at least
        # two distinct values when there are >= 2 legal actions, otherwise
        # the sampler is not stochastic.
        stoch_request_body = {
            "observation": sample.example["observation"],
            "legalActions": sample.example["legalActions"],
            "sampling": "stochastic",
            "temperature": 1.0,
        }
        stoch_a = post_json(
            f"http://127.0.0.1:{port}/predict",
            {**stoch_request_body, "samplingSeed": 42},
        )
        stoch_b = post_json(
            f"http://127.0.0.1:{port}/predict",
            {**stoch_request_body, "samplingSeed": 42},
        )
        if stoch_a["selectedIndex"] != stoch_b["selectedIndex"]:
            raise AssertionError(
                f"stochastic sampling with the same samplingSeed must be deterministic; "
                f"got {stoch_a['selectedIndex']} vs {stoch_b['selectedIndex']}"
            )
        if (stoch_a.get("behaviorPolicy") or {}).get("kind") != "stochastic":
            raise AssertionError(
                f"behaviorPolicy.kind must be stochastic in stochastic mode; got {stoch_a.get('behaviorPolicy')}"
            )
        if abs(float((stoch_a["behaviorPolicy"]).get("temperature", -1)) - 1.0) > 1e-6:
            raise AssertionError(
                f"behaviorPolicy.temperature must be 1.0; got {stoch_a['behaviorPolicy']}"
            )
        stoch_legal_prob_sum = sum(stoch_a["actionProbs"][0][:legal_count])
        if not (0.999 <= stoch_legal_prob_sum <= 1.001):
            raise AssertionError(
                f"stochastic actionProbs legal-sum must be ~1.0; got {stoch_legal_prob_sum}"
            )
        if legal_count >= 2:
            seen = set()
            for seed in range(1000, 1080):
                s = post_json(
                    f"http://127.0.0.1:{port}/predict",
                    {**stoch_request_body, "samplingSeed": seed},
                )
                seen.add(int(s["selectedIndex"][0]))
                if len(seen) >= 2:
                    break
            if len(seen) < 2:
                raise AssertionError(
                    f"stochastic sampling across 80 seeds must produce at least 2 distinct selections "
                    f"with {legal_count} legal actions; got {seen}"
                )
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()

    assert_export_rejects_vocab_mismatch(repo_root, model_dir, run_dir)
    assert_dataset_rejects_bad_schema(repo_root, run_dir, examples_path)
    assert_resume_continues_training(repo_root, run_dir, examples_path)
    assert_kl_anchor_smoke(repo_root, run_dir, examples_path, anchor_checkpoint=model_dir / "checkpoint.pt")
    assert_multi_teacher_dataset_loader(repo_root, run_dir, examples_path)
    assert_card_embedding_forward(repo_root, run_dir, examples_path)
    assert_dpo_smoke(repo_root, run_dir, reference_checkpoint=model_dir / "checkpoint.pt")

    print(json.dumps({
        "status": "PASS",
        "examples": str(examples_path),
        "checkpoint": str(model_dir / "checkpoint.pt"),
        "onnx": str(onnx_path),
        "directSelectedIndex": direct_prediction["selectedIndex"],
        "servedSelectedIndex": served_prediction["selectedIndex"][0],
        "servedSelectedActionId": served_prediction.get("selectedActionId", [None])[0],
    }, indent=2))


def assert_multi_teacher_dataset_loader(repo_root: Path, run_dir: Path, baseline_jsonl: Path) -> None:
    """R7 step 3: end-to-end smoke for the multi-teacher dataset loader.

    Generates a multi-teacher decision trace via the TS evaluator, relabels
    it (producing rows with a `policyTargets` array per scoping doc § 3),
    loads it through `JsonlPolicyDataset`, and asserts that:

    - Every loaded `PolicySample.policy_target` is a length-`numActions`
      ndarray that sums to 1 with mass on multiple teacher choices on at
      least one row (i.e. the mixture is genuinely soft, not collapsed).
    - `collate_policy_batch` emits a `policy_targets` tensor of shape
      `(B, max_actions)`, padded parallel to `action_features` /
      `action_mask`, with masked positions zeroed.
    - The trainer's soft-CE branch (`train_bc.py:361-369`) will fire,
      since `batch.get("policy_targets") is not None`.
    - REGRESSION GUARD: legacy rows without `policyTargets` still produce a
      batch that omits `policy_targets`, so the trainer's hard-CE branch
      fires for them. We use the pre-existing `baseline_jsonl` corpus
      (produced by `sim:export-training`, which does not write
      `policyTargets`) as the negative control.
    """

    import numpy as np
    import torch

    from uma_ai.dataset import JsonlPolicyDataset, collate_policy_batch
    from uma_ai.node_bridge import relabel_decision_trace, run_evaluator

    out_dir = run_dir / "r7-multi-teacher"
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "trace.multi.jsonl"
    relabeled_path = out_dir / "relabeled.multi.jsonl"

    run_evaluator(
        repo_root,
        selection="baseline",
        games=1,
        seed_start=8200,
        model_side="player",
        max_steps=120,
        rollout_steps=20,
        decision_trace_out=trace_path,
        trace_teacher="rollout,search,planner",
        extra=("--planner-max-sequences", "8", "--planner-max-depth", "4"),
    )
    relabel_decision_trace(
        repo_root,
        trace_in=trace_path,
        relabeled_out=relabeled_path,
        source="model-visited-multi-teacher",
        label_source="multi-teacher",
    )

    dataset = JsonlPolicyDataset(relabeled_path, min_actions=2)
    # Every relabeled row carries `policyTargets` per the TS contract
    # (relabelDecisionTrace.ts:148). The loader must surface that on every
    # sample as a finite ndarray summing to 1.
    soft_count = 0
    multi_mass_rows = 0
    for sample in dataset.samples:
        if sample.policy_target is None:
            raise AssertionError(
                f"R7: relabeled multi-teacher row must carry policy_target, got None on sample with "
                f"selectedActionIndex={sample.target_index}"
            )
        soft_count += 1
        arr = sample.policy_target
        if arr.shape != (sample.action_features.shape[0],):
            raise AssertionError(
                f"R7: policy_target shape {arr.shape} must match num_actions {sample.action_features.shape[0]}"
            )
        total = float(arr.sum())
        if abs(total - 1.0) > 1e-5:
            raise AssertionError(f"R7: policy_target must sum to 1.0 ± 1e-5; got {total}")
        # "Genuinely soft" means at least two teachers disagreed on this
        # state. A mixture where all three teachers agree is a one-hot and
        # is structurally identical to hard-CE — fine, but it doesn't
        # exercise the soft path. We need at least one row with mass on
        # 2+ action indices to claim the multi-teacher path is wired.
        if (arr > 0).sum() >= 2:
            multi_mass_rows += 1
    if soft_count == 0:
        raise AssertionError("R7: relabeled corpus had zero rows; cannot validate loader")
    if multi_mass_rows == 0:
        raise AssertionError(
            f"R7: every multi-teacher row collapsed to one-hot — soft target not exercised "
            f"(soft_count={soft_count}, multi_mass_rows=0). Increase trace breadth or check that "
            f"all three teachers ran."
        )

    # Collate a batch and verify the trainer's soft-CE branch will fire.
    batch = collate_policy_batch(list(dataset.samples))
    if "policy_targets" not in batch:
        raise AssertionError(
            "R7: collate_policy_batch must emit `policy_targets` when every sample carries one; "
            "trainer's soft-CE branch at train_bc.py:361-369 keys off `batch.get(\"policy_targets\")`"
        )
    pt = batch["policy_targets"]
    bsz = len(dataset.samples)
    max_actions = int(batch["action_features"].shape[1])
    if pt.shape != (bsz, max_actions):
        raise AssertionError(
            f"R7: policy_targets must be shape ({bsz}, {max_actions}), got {tuple(pt.shape)}"
        )
    # Each row's policy_targets must sum to ~1 (mass on legal positions only)
    # and masked positions must be zero (padding parallels action_mask).
    row_sums = pt.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
        raise AssertionError(f"R7: per-row policy_targets must sum to 1; got {row_sums.tolist()}")
    masked_positions = ~batch["action_mask"]
    if pt[masked_positions].abs().max().item() > 0.0:
        raise AssertionError(
            "R7: policy_targets at masked positions must be exactly zero; padding does not parallel action_mask"
        )

    # Regression guard: the legacy baseline corpus has no `policyTargets`
    # field — `policy_target` must be None on every sample, and
    # `collate_policy_batch` must omit `policy_targets` from the batch so
    # the trainer's hard-CE branch fires unchanged.
    baseline_dataset = JsonlPolicyDataset(baseline_jsonl, min_actions=2)
    if any(s.policy_target is not None for s in baseline_dataset.samples):
        raise AssertionError(
            "R7 regression: baseline `sim:export-training` corpus must not produce policy_target; "
            "got at least one sample with a soft target — the parser is misreading legacy rows"
        )
    baseline_batch = collate_policy_batch(list(baseline_dataset.samples[: min(8, len(baseline_dataset.samples))]))
    if "policy_targets" in baseline_batch:
        raise AssertionError(
            "R7 regression: legacy batch (no `policyTargets`) must omit `policy_targets` so the "
            "trainer takes the hard-CE branch on `targets`. The mixed-batch fallback in "
            "collate_policy_batch is broken."
        )
    # Also confirm a mixed batch (one soft + one hard sample) falls back to
    # the hard path (no `policy_targets` emitted). This is the `mix-sources`
    # contract: a batch with any legacy row uses hard-CE.
    mixed_samples = [dataset.samples[0], baseline_dataset.samples[0]]
    mixed_batch = collate_policy_batch(mixed_samples)
    if "policy_targets" in mixed_batch:
        raise AssertionError(
            "R7 regression: a mixed soft+hard batch must omit `policy_targets` (hard-CE for the "
            "whole batch). Synthesising a one-hot for the legacy row would silently change loss."
        )

    # Numerical equivalence: a soft target that happens to be one-hot must
    # produce a soft-CE loss numerically equal to hard-CE on the same
    # selectedActionIndex. We construct a synthetic one-hot soft target on
    # the legacy baseline samples and verify the loss matches the hard-CE
    # value to within float32 noise.
    from uma_ai.dataset import PolicySample
    from train_bc import masked_log_softmax_logits  # type: ignore[import-not-found]

    sample = baseline_dataset.samples[0]
    one_hot = np.zeros((sample.action_features.shape[0],), dtype=np.float32)
    one_hot[sample.target_index] = 1.0
    synthetic = PolicySample(
        state_features=sample.state_features,
        action_features=sample.action_features,
        target_index=sample.target_index,
        value_target=sample.value_target,
        sample_weight=sample.sample_weight,
        example=sample.example,
        policy_target=one_hot,
    )
    soft_batch = collate_policy_batch([synthetic])
    hard_batch = collate_policy_batch([sample])
    if "policy_targets" not in soft_batch or "policy_targets" in hard_batch:
        raise AssertionError("R7: synthetic one-hot soft-vs-hard batches did not partition as expected")
    # Loss numbers — we don't have a model handy, but we can simulate the
    # log-softmax over uniform logits (all zeros): hard-CE = -log p[i],
    # soft-CE with one-hot at i = -log p[i]. Equality is guaranteed by
    # construction; this assert just exercises the kernel and the masked
    # log_softmax helper to make sure the wiring is real.
    logits = torch.zeros_like(soft_batch["action_features"][:, :, 0])
    log_probs = masked_log_softmax_logits(logits, soft_batch["action_mask"])
    soft_loss = -(soft_batch["policy_targets"] * log_probs).sum(dim=1)
    hard_loss = torch.nn.functional.cross_entropy(
        logits, hard_batch["targets"], reduction="none"
    )
    if not torch.allclose(soft_loss, hard_loss, atol=1e-5):
        raise AssertionError(
            f"R7 regression: one-hot soft-CE must equal hard-CE under uniform logits; got "
            f"soft={soft_loss.tolist()} hard={hard_loss.tolist()}"
        )


def assert_card_embedding_forward(repo_root: Path, run_dir: Path, baseline_jsonl: Path) -> None:
    """R7.b.2 Phase 2: end-to-end smoke for the card-embedding pass.

    Covers four contracts from the brief:

      (1) `observation_to_card_ids` produces non-empty arrays for a fixture
          observation drawn from the live baseline corpus (which Phase 1
          ensured emits `cardIdsByZone`). Per-zone shapes match
          `CARD_ID_SHAPES`.
      (2) `load_policy_samples` populates `card_ids_by_zone` /
          `action_card_idx` on every sample; `collate_policy_batch` emits
          packed `LongTensor[B, NUM_ZONES, max_cards_per_zone]` and
          `LongTensor[B, max_actions, 2]` in the batch dict with correct
          shapes and dtypes.
      (3) `CandidatePolicyNet.forward` with the new tensors produces
          gradients on `card_embed.weight` AND `zone_projection.weight` —
          confirming the embedding pass is on the autograd graph.
      (4) Padding semantics: a batch with all-zero card ids must produce
          the SAME forward output as omitting the new tensors entirely.
          This validates that `padding_idx=0` doesn't leak signal and the
          additive-residual choice is structurally null when no cards are
          present.
    """

    import numpy as np
    import torch

    from uma_ai.dataset import JsonlPolicyDataset, collate_policy_batch
    from uma_ai.features import (
        CARD_ID_SHAPES,
        ZONE_ORDER,
        observation_to_card_ids,
    )
    from uma_ai.model import (
        ACTION_PAIR_FANOUT,
        CARD_EMBED_DIM,
        CARD_VOCAB_TABLE_SIZE,
        CandidatePolicyNet,
        ModelConfig,
        NUM_ZONES,
    )

    # (1) Fixture observation: pull from the live baseline corpus.
    dataset = JsonlPolicyDataset(baseline_jsonl, min_actions=2)
    sample = dataset.samples[0]
    observation = sample.example.get("observation", {})
    if "cardIdsByZone" not in observation:
        raise AssertionError(
            "R7.b.2 Phase 2 smoke: baseline corpus observation missing 'cardIdsByZone'. "
            "Phase 1 TS schema bump did not propagate to sim:export-training; rebuild backend."
        )
    card_ids = observation_to_card_ids(observation)
    if set(card_ids.keys()) != set(CARD_ID_SHAPES.keys()):
        raise AssertionError(
            f"R7.b.2 Phase 2 smoke: observation_to_card_ids keys {set(card_ids.keys())} "
            f"must match CARD_ID_SHAPES {set(CARD_ID_SHAPES.keys())}"
        )
    for zone, width in CARD_ID_SHAPES.items():
        if card_ids[zone].shape != (width,):
            raise AssertionError(
                f"R7.b.2 Phase 2 smoke: zone {zone!r} shape {card_ids[zone].shape} != ({width},)"
            )
        if card_ids[zone].dtype != np.int64:
            raise AssertionError(
                f"R7.b.2 Phase 2 smoke: zone {zone!r} dtype {card_ids[zone].dtype} != int64"
            )
    # At least one zone should have a non-zero entry (active is always
    # populated when the side has an active uma); else the corpus is
    # producing empty observations.
    if all(int(arr.sum()) == 0 for arr in card_ids.values()):
        raise AssertionError(
            "R7.b.2 Phase 2 smoke: every zone empty on the fixture observation; "
            "expected at least one card across own/opp active+bench+hand."
        )

    # Fail-loud guard: a fabricated observation without `cardIdsByZone`
    # must raise. This is the substantive replacement for the row-level
    # schemaVersion bump (which Phase 2 deferred to Phase 4).
    bare_obs = {"sideToAct": "player", "phase": "stadiumOrEnd"}
    raised = False
    try:
        observation_to_card_ids(bare_obs)
    except ValueError:
        raised = True
    if not raised:
        raise AssertionError(
            "R7.b.2 Phase 2 smoke: observation_to_card_ids must raise on missing cardIdsByZone"
        )

    # (2) Loader + collator shapes.
    if any(s.card_ids_by_zone is None or s.action_card_idx is None for s in dataset.samples):
        raise AssertionError(
            "R7.b.2 Phase 2 smoke: load_policy_samples did not populate card_ids_by_zone / "
            "action_card_idx on every sample"
        )
    batch = collate_policy_batch(list(dataset.samples[: min(8, len(dataset.samples))]))
    if "card_ids_by_zone" not in batch or "action_card_idx" not in batch:
        raise AssertionError(
            f"R7.b.2 Phase 2 smoke: collate_policy_batch missing new keys; got {sorted(batch.keys())}"
        )
    czi = batch["card_ids_by_zone"]
    aci = batch["action_card_idx"]
    bsz = int(batch["action_features"].shape[0])
    max_actions = int(batch["action_features"].shape[1])
    expected_max_cards = max(CARD_ID_SHAPES.values())
    if czi.shape != (bsz, NUM_ZONES, expected_max_cards):
        raise AssertionError(
            f"R7.b.2 Phase 2 smoke: card_ids_by_zone shape {tuple(czi.shape)} != "
            f"({bsz}, {NUM_ZONES}, {expected_max_cards})"
        )
    if czi.dtype != torch.int64:
        raise AssertionError(f"R7.b.2 Phase 2 smoke: card_ids_by_zone dtype {czi.dtype} != int64")
    if aci.shape != (bsz, max_actions, 2):
        raise AssertionError(
            f"R7.b.2 Phase 2 smoke: action_card_idx shape {tuple(aci.shape)} != ({bsz}, {max_actions}, 2)"
        )
    if aci.dtype != torch.int64:
        raise AssertionError(f"R7.b.2 Phase 2 smoke: action_card_idx dtype {aci.dtype} != int64")
    # Smaller-cap zones (active, stadium) must have zeros beyond their cap.
    for zone_index, zone in enumerate(ZONE_ORDER):
        cap = CARD_ID_SHAPES[zone]
        if cap < expected_max_cards:
            beyond = czi[:, zone_index, cap:expected_max_cards]
            if int(beyond.abs().sum().item()) != 0:
                raise AssertionError(
                    f"R7.b.2 Phase 2 smoke: zone {zone!r} has non-zero ids beyond its cap {cap}; "
                    f"collator padding broken (collator wrote {int(beyond.abs().sum().item())} non-zero entries)"
                )

    # (3) Forward + backward produces gradients on the embedding params.
    config = ModelConfig(hidden_dim=32, depth=1, dropout=0.0)
    model = CandidatePolicyNet(config)
    model.train()
    # Confirm the embedding table is the expected shape (108 × 32).
    if tuple(model.card_embed.weight.shape) != (CARD_VOCAB_TABLE_SIZE, CARD_EMBED_DIM):
        raise AssertionError(
            f"R7.b.2 Phase 2 smoke: card_embed.weight shape {tuple(model.card_embed.weight.shape)} != "
            f"({CARD_VOCAB_TABLE_SIZE}, {CARD_EMBED_DIM})"
        )
    if model.zone_projection.in_features != NUM_ZONES * CARD_EMBED_DIM:
        raise AssertionError(
            f"R7.b.2 Phase 2 smoke: zone_projection.in_features {model.zone_projection.in_features} != "
            f"{NUM_ZONES * CARD_EMBED_DIM}"
        )
    # Confirm joint_projection input dim grew by ACTION_PAIR_FANOUT * embed.
    expected_joint_in = 3 * config.hidden_dim + ACTION_PAIR_FANOUT * CARD_EMBED_DIM
    actual_joint_in = model.joint_projection[0].in_features
    if actual_joint_in != expected_joint_in:
        raise AssertionError(
            f"R7.b.2 Phase 2 smoke: joint_projection.in_features {actual_joint_in} != {expected_joint_in}"
        )

    logits, values = model(
        batch["state_features"],
        batch["action_features"],
        batch["action_mask"],
        card_ids_by_zone=czi,
        action_card_idx=aci,
    )
    # Build a sham loss that propagates through both the policy head and
    # the value head so both branches of the embedding pass receive grad.
    loss = logits.sum() + values.sum()
    loss.backward()
    if model.card_embed.weight.grad is None or model.card_embed.weight.grad.abs().sum().item() == 0.0:
        raise AssertionError(
            "R7.b.2 Phase 2 smoke: card_embed.weight received no gradient — the embedding "
            "pass is detached from the loss graph"
        )
    if model.zone_projection.weight.grad is None or model.zone_projection.weight.grad.abs().sum().item() == 0.0:
        raise AssertionError(
            "R7.b.2 Phase 2 smoke: zone_projection.weight received no gradient"
        )
    # padding_idx=0's row must remain zero-grad (pad rows don't accumulate).
    if model.card_embed.weight.grad[0].abs().sum().item() != 0.0:
        raise AssertionError(
            "R7.b.2 Phase 2 smoke: card_embed pad row (idx 0) received gradient; "
            "padding_idx semantics violated"
        )

    # (4) All-zero card ids produces same output as omitting the new args.
    # This is the additive-residual sanity check: with no cards, the
    # embedding pool is zero and the joint projection sees the same input
    # in both branches. We compare two no_grad forwards under eval mode
    # so dropout doesn't introduce stochastic noise.
    model.eval()
    with torch.no_grad():
        zero_czi = torch.zeros_like(czi)
        zero_aci = torch.zeros_like(aci)
        logits_zero, values_zero = model(
            batch["state_features"],
            batch["action_features"],
            batch["action_mask"],
            card_ids_by_zone=zero_czi,
            action_card_idx=zero_aci,
        )
        logits_none, values_none = model(
            batch["state_features"],
            batch["action_features"],
            batch["action_mask"],
        )
    max_logit_diff = float((logits_zero - logits_none).abs().max().item())
    max_value_diff = float((values_zero - values_none).abs().max().item())
    if max_logit_diff > 1e-6 or max_value_diff > 1e-6:
        raise AssertionError(
            f"R7.b.2 Phase 2 smoke: zero-id forward != omitted-arg forward; logits diff "
            f"{max_logit_diff}, values diff {max_value_diff}. padding_idx=0 leaks signal "
            "or the default-zero branch is wired differently from the embed(0) path."
        )


def assert_dpo_smoke(repo_root: Path, run_dir: Path, *, reference_checkpoint: Path) -> None:
    """R8 step 1: DPO trainer scaffold smoke.

    Synthetic 5-row preference batch (no JSONL needed — the BC-trained
    `reference_checkpoint` doubles as both the frozen reference policy
    AND the warm-start for the trainable model, matching the v1 setup
    in `docs/ai-research/scoping/r8-dpo.md` § 3.5).

    Asserts the four contracts from the brief:
      (a) loss is finite
      (b) gradients flow on policy params
      (c) reference params have NO grad
      (d) loss strictly decreases over 10 SGD steps on the same batch

    Imports happen inside the function so the smoke does not perturb the
    earlier ONNX server / serve_onnx tests if a DPO-side import is
    accidentally heavy.
    """

    # Inline imports keep the DPO-only deps out of the top-level smoke
    # module's import graph; everything below is already on sys.path
    # because smoke_e2e.py is invoked from the training/ directory.
    from train_dpo import dpo_loss_components, load_reference_policy
    from uma_ai.features import ACTION_DIM, STATE_DIM
    from uma_ai.model import CandidatePolicyNet, ModelConfig
    from train_bc import load_init_from_checkpoint

    torch.manual_seed(1234)
    payload = torch.load(
        reference_checkpoint, map_location="cpu", weights_only=False
    )
    config = ModelConfig.from_dict(payload.get("model_config"))
    device = torch.device("cpu")

    # Build the trainable policy (warm-started from the same checkpoint
    # as the reference, per scoping § 3.5) and a frozen reference.
    # Disable dropout so the strict-decrease assertion in (d) is a clean
    # signal about the optimizer/loss wiring rather than dropout noise on
    # a tiny synthetic batch. The dropout path is exercised by the BC
    # smoke earlier in this file.
    model = CandidatePolicyNet(config).to(device)
    load_init_from_checkpoint(reference_checkpoint, model)
    model.eval()
    reference = load_reference_policy(str(reference_checkpoint), config, device)

    # Synthetic 5-row preference batch with 4 actions per row. Random
    # state/action features keep the batch independent of the BC corpus
    # so this smoke is a pure DPO contract test, not a coupled regression.
    batch_size = 5
    num_actions = 4
    rng = torch.Generator().manual_seed(99)
    state_features = torch.randn(batch_size, STATE_DIM, generator=rng)
    action_features = torch.randn(
        batch_size, num_actions, ACTION_DIM, generator=rng
    )
    action_mask = torch.ones(batch_size, num_actions, dtype=torch.bool)
    # y_w/y_l: deterministic non-equal indices per row.
    y_w = torch.tensor([0, 1, 2, 3, 0], dtype=torch.int64)
    y_l = torch.tensor([1, 2, 3, 0, 2], dtype=torch.int64)
    sample_weights = torch.ones(batch_size, dtype=torch.float32)
    batch = {
        "state_features": state_features,
        "action_features": action_features,
        "action_mask": action_mask,
        "y_w_index": y_w,
        "y_l_index": y_l,
        "sample_weights": sample_weights,
    }

    # (a) loss finite + (c) reference has no grad.
    components = dpo_loss_components(model, reference, batch, beta=0.1)
    loss0 = float(components["loss"].item())
    if not np.isfinite(loss0):
        raise AssertionError(f"DPO smoke (a): initial loss not finite: {loss0}")
    if any(p.requires_grad for p in reference.parameters()):
        raise AssertionError(
            "DPO smoke (c): reference policy has parameters with requires_grad=True; "
            "load_reference_policy must freeze every parameter (§ 3.5)."
        )

    # (b) gradients flow on policy params: do one backward and check
    # that at least one parameter has a non-zero gradient.
    components["loss"].backward()
    grad_norm = sum(
        float(p.grad.detach().norm().item())
        for p in model.parameters()
        if p.grad is not None
    )
    has_any_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0.0
        for p in model.parameters()
    )
    if not has_any_grad:
        raise AssertionError(
            "DPO smoke (b): no policy parameter received a non-zero gradient; "
            "check that the loss reaches the policy via masked_log_softmax."
        )
    # Also verify that NO reference parameter received a grad — even a
    # nonzero grad attribute would mean the reference snuck into the
    # autograd graph (e.g. forgot the `with torch.no_grad():` block).
    for name, p in reference.named_parameters():
        if p.grad is not None and p.grad.abs().sum().item() > 0.0:
            raise AssertionError(
                f"DPO smoke (c): reference param {name!r} accumulated grad — "
                "reference forward must be wrapped in torch.no_grad()."
            )

    # (d) loss strictly decreases over 10 SGD steps on the *same* batch.
    # Re-init the optimizer with a moderate LR; the loss should not be
    # numerically constrained to monotone-decreasing for general DPO, but
    # on a single fixed 5-row batch with no regularization it should drop
    # at every step within float noise. We assert strict monotonicity to
    # surface optimizer-wiring bugs (e.g. zero_grad placement).
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    losses: list[float] = []
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        components = dpo_loss_components(model, reference, batch, beta=0.1)
        loss = components["loss"]
        if not np.isfinite(float(loss.item())):
            raise AssertionError(
                f"DPO smoke (a)/SGD: loss became non-finite mid-train: "
                f"history={losses + [float(loss.item())]}"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        losses.append(float(loss.item()))
    for i in range(1, len(losses)):
        if losses[i] >= losses[i - 1]:
            raise AssertionError(
                f"DPO smoke (d): loss did not strictly decrease at step {i}: "
                f"prev={losses[i - 1]} curr={losses[i]} history={losses}"
            )

    # Sanity: 10 steps should drop loss meaningfully on a fixed batch.
    if losses[-1] >= losses[0]:
        raise AssertionError(
            f"DPO smoke (d): loss did not decrease overall: "
            f"start={losses[0]} end={losses[-1]}"
        )


def assert_kl_anchor_smoke(repo_root: Path, run_dir: Path, source_jsonl: Path, *, anchor_checkpoint: Path) -> None:
    """Item 11/17: KL-anchor anti-forgetting plumbing smoke.

    Trains 2 epochs against a frozen anchor at weight=0.5; asserts the
    manifest's training_kwargs records the anchor checkpoint path and the
    weight, and that the per-epoch history carries a kl_loss field.
    """

    kl_dir = run_dir / "kl-anchor"
    cmd = [
        sys.executable,
        str(repo_root / "training" / "train_bc.py"),
        "--data",
        str(source_jsonl),
        "--out-dir",
        str(kl_dir),
        "--epochs",
        "2",
        "--batch-size",
        "16",
        "--hidden-dim",
        "32",
        "--depth",
        "1",
        "--kl-anchor-checkpoint",
        str(anchor_checkpoint),
        "--kl-anchor-weight",
        "0.5",
    ]
    subprocess.run(cmd, cwd=repo_root, check=True, capture_output=True, text=True)
    manifest = json.loads((kl_dir / "manifest.json").read_text(encoding="utf8"))
    kwargs = manifest.get("training_kwargs", {})
    if kwargs.get("kl_anchor_checkpoint") != str(anchor_checkpoint):
        raise AssertionError(f"manifest must record kl_anchor_checkpoint, got {kwargs}")
    if abs(float(kwargs.get("kl_anchor_weight", 0)) - 0.5) > 1e-6:
        raise AssertionError(f"manifest must record kl_anchor_weight=0.5, got {kwargs}")
    checkpoint = torch.load(kl_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    history = checkpoint.get("history", [])
    if not history or "kl_loss" not in history[0].get("train", {}):
        raise AssertionError(
            f"per-epoch history must record kl_loss when --kl-anchor-weight > 0; got {history[:1]}"
        )
    if float(history[0]["train"]["kl_loss"]) <= 0.0:
        # KL with anchor==current model would be 0; we resumed-from-scratch
        # against a *trained* anchor so the freshly-initialized model should
        # diverge from it. A 0 here means the anchor isn't actually being
        # consulted.
        raise AssertionError(
            f"kl_loss must be >0 when training a fresh model against a trained anchor; got {history[0]['train']['kl_loss']}"
        )


def assert_resume_continues_training(repo_root: Path, run_dir: Path, source_jsonl: Path) -> None:
    seed_dir = run_dir / "resume" / "seed"
    resume_dir = run_dir / "resume" / "resumed"
    base_args = [
        sys.executable,
        str(repo_root / "training" / "train_bc.py"),
        "--data",
        str(source_jsonl),
        "--epochs",
        "4",
        "--batch-size",
        "16",
        "--hidden-dim",
        "32",
        "--depth",
        "1",
        "--lr-schedule",
        "cosine",
        "--lr-warmup-steps",
        "2",
        "--grad-accum",
        "2",
    ]
    seed_args = base_args + ["--out-dir", str(seed_dir)]
    subprocess.run(seed_args, cwd=repo_root, check=True, capture_output=True, text=True)
    seed_manifest = json.loads((seed_dir / "manifest.json").read_text(encoding="utf8"))
    if seed_manifest.get("onnx_roundtrip_smoke", {}).get("status") != "PASS":
        raise AssertionError(f"Seed run did not pass ONNX roundtrip smoke: {seed_manifest.get('onnx_roundtrip_smoke')}")

    resume_args = base_args + [
        "--out-dir",
        str(resume_dir),
        "--epochs",
        "8",
        "--resume",
        str(seed_dir / "checkpoint.pt"),
    ]
    subprocess.run(resume_args, cwd=repo_root, check=True, capture_output=True, text=True)
    resume_manifest = json.loads((resume_dir / "manifest.json").read_text(encoding="utf8"))
    if resume_manifest.get("training_kwargs", {}).get("resume_from") != str(seed_dir / "checkpoint.pt"):
        raise AssertionError(f"Resume manifest did not record resume_from path: {resume_manifest.get('training_kwargs')}")
    if resume_manifest.get("onnx_roundtrip_smoke", {}).get("status") != "PASS":
        raise AssertionError(f"Resume run did not pass ONNX roundtrip smoke: {resume_manifest.get('onnx_roundtrip_smoke')}")


def assert_dataset_rejects_bad_schema(repo_root: Path, run_dir: Path, source_jsonl: Path) -> None:
    from uma_ai.dataset import JsonlPolicyDataset, RowSchemaError

    rows = source_jsonl.read_text(encoding="utf8").strip().splitlines()
    if not rows:
        raise AssertionError(f"Source JSONL {source_jsonl} unexpectedly empty")

    missing = run_dir / "missing_schema.jsonl"
    bumped = run_dir / "bumped_schema.jsonl"
    missing_lines = []
    bumped_lines = []
    for raw in rows:
        payload = json.loads(raw)
        no_version = {key: value for key, value in payload.items() if key != "schemaVersion"}
        missing_lines.append(json.dumps(no_version))
        bumped_payload = dict(payload)
        bumped_payload["schemaVersion"] = 99
        bumped_lines.append(json.dumps(bumped_payload))
    missing.write_text("\n".join(missing_lines) + "\n", encoding="utf8")
    bumped.write_text("\n".join(bumped_lines) + "\n", encoding="utf8")

    try:
        JsonlPolicyDataset(missing)
    except RowSchemaError as exc:
        if "Missing schemaVersion" not in str(exc):
            raise AssertionError(f"Unexpected RowSchemaError: {exc}")
    else:
        raise AssertionError("Dataset must reject rows with no schemaVersion")

    try:
        JsonlPolicyDataset(bumped)
    except RowSchemaError as exc:
        if "Incompatible schemaVersion" not in str(exc):
            raise AssertionError(f"Unexpected RowSchemaError: {exc}")
    else:
        raise AssertionError("Dataset must reject rows with bumped schemaVersion")

    JsonlPolicyDataset(missing, strict_schema_version=False)


def assert_export_rejects_vocab_mismatch(repo_root: Path, model_dir: Path, run_dir: Path) -> None:
    import torch

    checkpoint_path = model_dir / "checkpoint.pt"
    tampered_path = run_dir / "tampered.pt"
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    schema = dict(raw.get("feature_schema", {}))
    vocab_meta = dict(schema.get("card_vocab", {}))
    vocab_meta["hash"] = "tampered-hash-deadbeef"
    schema["card_vocab"] = vocab_meta
    raw["feature_schema"] = schema
    torch.save(raw, tampered_path)

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "training" / "export_onnx.py"),
            "--checkpoint",
            str(tampered_path),
            "--out",
            str(run_dir / "tampered.onnx"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        raise AssertionError(
            f"export_onnx must reject mismatched card vocab hash, got returncode 0 stdout={result.stdout!r}"
        )
    if "Card vocab hash mismatch" not in result.stderr:
        raise AssertionError(
            f"Expected 'Card vocab hash mismatch' in stderr; got {result.stderr!r}"
        )


def assert_grouped_split(manifest: dict) -> None:
    split = manifest.get("split", {})
    if split.get("split_by") != "episode":
        raise AssertionError(f"Expected episode split, got {split}")
    train_groups = set(split.get("train_groups", []))
    val_groups = set(split.get("val_groups", []))
    if not val_groups:
        raise AssertionError(f"Expected validation groups, got {split}")
    leaked = train_groups.intersection(val_groups)
    if leaked:
        raise AssertionError(f"Train/val group leakage: {sorted(leaked)}")


def assert_card_vocab_recorded(manifest: dict) -> None:
    schema = manifest.get("feature_schema", {})
    vocab = schema.get("card_vocab")
    if not vocab:
        raise AssertionError(f"Expected card_vocab metadata in manifest feature_schema, got {schema}")
    runtime = card_vocab_metadata()
    if vocab.get("hash") != runtime.get("hash"):
        raise AssertionError(
            f"Manifest vocab hash {vocab.get('hash')} != runtime {runtime.get('hash')}"
        )
    if vocab.get("vocabSize", 0) <= 0:
        raise AssertionError(f"Card vocab has unexpected size {vocab.get('vocabSize')}")


def run_onnx_prediction(model_path: Path, example: dict) -> dict[str, int]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    observation = example["observation"]
    legal_actions = example["legalActions"]
    state = observation_to_features(observation)[None, :]
    actions = legal_actions_to_features(legal_actions)[None, :, :]
    mask = np.ones(actions.shape[:2], dtype=np.bool_)
    # R7.b.2 Phase 3: build the embedding-pass tensors with the same
    # packing convention `request_to_arrays` uses. This is the direct-ORT
    # path that the served-prediction smoke is compared against — they
    # must agree byte-for-byte (same packing + same ONNX session).
    max_cards_per_zone = max(CARD_ID_SHAPES.values())
    num_zones = len(ZONE_ORDER)
    card_id_zones = observation_to_card_ids(observation)
    card_ids_by_zone = np.zeros((1, num_zones, max_cards_per_zone), dtype=np.int64)
    for zone_index, zone in enumerate(ZONE_ORDER):
        zone_arr = card_id_zones[zone]
        card_ids_by_zone[0, zone_index, : zone_arr.shape[0]] = zone_arr
    action_card_idx = np.zeros((1, len(legal_actions), 2), dtype=np.int64)
    for action_index, action in enumerate(legal_actions):
        action_card_idx[0, action_index, :] = action_card_idx_pair(action)
    logits, _value = session.run(None, {
        "state_features": state.astype(np.float32),
        "action_features": actions.astype(np.float32),
        "action_mask": mask,
        "card_ids_by_zone": card_ids_by_zone,
        "action_card_idx": action_card_idx,
    })
    selected = int(logits.argmax(axis=1)[0])
    if selected < 0 or selected >= len(legal_actions):
        raise AssertionError(f"ONNX selected invalid index {selected}")
    return {"selectedIndex": selected, "logits": logits.tolist()}


def wait_for_health(port: int) -> None:
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            post_json(f"http://127.0.0.1:{port}/health", None, method="GET")
            return
        except Exception:
            time.sleep(0.2)
    raise TimeoutError("ONNX server did not become healthy")


def post_json(url: str, payload: dict | None, *, method: str = "POST") -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf8")
    request = urllib.request.Request(url, data=data, method=method)
    if payload is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf8"))


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    main()
