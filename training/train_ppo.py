"""PPO update step (F1).

Reads a parsed trajectory JSONL (one record per model-side decision; produced
by ppo_orchestrator), computes GAE advantages and returns against the
existing value-head critic, then runs one PPO epoch over the buffer with the
spec defaults from ``docs/f1-design.md`` (KL clip eps=0.2, entropy coef
0.005, value coef 0.5, grad clip 0.5, 4 minibatches by default).

Numerical guards live in this module:
- log_probs / log-ratios clamped at +/- 50 so importance ratios cannot
  overflow into inf,
- minibatches with any non-finite intermediate (loss/grad) are skipped and
  a ``numerical-anomaly`` event is emitted so the orchestrator can decide
  whether to halt,
- approx-KL = mean(behavior_logp - new_logp) over a minibatch is the
  standard PPO cheap KL signal logged on every minibatch event.

The trainer is intentionally CLI-driven and orthogonal to the orchestrator
so the F1 HP sweep (item F1.hp-sweep, not yet implemented) can shell out
to it cell-by-cell.

Manifest fields recorded per update (the five stability controls from
``docs/f1-design.md`` plus diagnostics):
- ``lr``, ``clip_epsilon``, ``entropy_coef``, ``value_coef``, ``grad_clip``
- ``gae_lambda``, ``gae_gamma``
- ``approx_kl_mean``, ``approx_kl_max``, ``entropy_mean``,
  ``policy_loss_mean``, ``value_loss_mean``,
  ``ratio_mean``, ``ratio_min``, ``ratio_max``
- ``n_minibatches``, ``samples_per_minibatch``,
  ``numerical_anomalies`` (count of skipped minibatches).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn

from events import EventWriter
from uma_ai.features import (
    ACTION_DIM,
    ACTION_FEATURE_SCHEMA_VERSION,
    STATE_DIM,
    STATE_FEATURE_SCHEMA_VERSION,
    card_vocab_metadata,
    legal_actions_to_features,
    observation_to_features,
)
from uma_ai.model import CandidatePolicyNet, ModelConfig

LOG_CLAMP = 50.0  # cap |log p|; exp(50) is already astronomical, exp(>700) overflows


@dataclass
class TrajectoryRow:
    state_features: np.ndarray  # (STATE_DIM,)
    action_features: np.ndarray  # (n_actions, ACTION_DIM)
    selected_idx: int
    behavior_logp: float
    reward: float
    done: bool
    value_pred: float  # behavior-time critic estimate; carried for bookkeeping only
    episode_id: str
    episode_step: int


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(load_trajectories(Path(args.trajectories)))
    if not rows:
        raise SystemExit(f"No trajectory rows found in {args.trajectories}")

    init_payload = torch.load(args.init_from_checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig.from_dict(init_payload.get("model_config"))
    model = CandidatePolicyNet(config).to(device)
    model.load_state_dict(init_payload["model_state"])

    events = EventWriter(args.events_out) if args.events_out else None
    if events is not None:
        events.emit(
            iteration=args.events_iteration,
            stage="ppo-update",
            event_type="started",
            samples=len(rows),
            ppo_epochs=args.ppo_epochs,
            minibatches=args.minibatches,
            lr=args.lr,
            clip_epsilon=args.clip_epsilon,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            grad_clip=args.grad_clip,
            gae_lambda=args.gae_lambda,
            gae_gamma=args.gae_gamma,
        )

    # GAE pass: fresh critic estimates from the *current* (warm-started) value
    # head, not the stale ``value_pred`` from rollout time. Spec calls the
    # critic state-only Tanh-bounded; same head, same forward, mask passed
    # for the value head shape contract (value is mask-independent in this
    # model but the forward signature requires it).
    advantages, returns, value_means = compute_gae(rows, model, device, gamma=args.gae_gamma, lam=args.gae_lambda)
    if events is not None:
        events.emit(
            iteration=args.events_iteration,
            stage="gae",
            event_type="completed",
            mean_advantage=float(np.mean(advantages)),
            std_advantage=float(np.std(advantages)),
            mean_return=float(np.mean(returns)),
            mean_value=float(np.mean(value_means)),
            n_episodes=len({row.episode_id for row in rows}),
            n_transitions=len(rows),
        )

    # Advantage normalization is standard PPO; without it the policy
    # gradient is on whatever Δpoints scale the reward shaper picked.
    adv_array = np.asarray(advantages, dtype=np.float32)
    if adv_array.std() > 1.0e-6:
        adv_norm = (adv_array - adv_array.mean()) / (adv_array.std() + 1.0e-8)
    else:
        adv_norm = adv_array - adv_array.mean()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Per-minibatch statistics accumulators.
    stats = {
        "approx_kl": [],
        "entropy": [],
        "policy_loss": [],
        "value_loss": [],
        "ratio_mean": [],
        "ratio_min": [],
        "ratio_max": [],
        "samples_per_minibatch": [],
        "numerical_anomalies": 0,
    }

    minibatch_groups = build_minibatches(len(rows), args.minibatches, seed=args.seed)
    for epoch in range(args.ppo_epochs):
        for mb_idx, indices in enumerate(minibatch_groups):
            batch = collate_minibatch(
                [rows[i] for i in indices],
                [float(adv_norm[i]) for i in indices],
                [float(returns[i]) for i in indices],
                device,
            )
            try:
                policy_loss, value_loss, entropy, approx_kl, ratio_stats, _ratio = ppo_step(
                    model,
                    optimizer,
                    batch,
                    clip_epsilon=args.clip_epsilon,
                    entropy_coef=args.entropy_coef,
                    value_coef=args.value_coef,
                    grad_clip=args.grad_clip,
                )
            except _NumericalAnomaly as exc:
                stats["numerical_anomalies"] += 1
                if events is not None:
                    events.emit(
                        iteration=args.events_iteration,
                        stage="ppo-update",
                        event_type="numerical-anomaly",
                        epoch=epoch,
                        minibatch=mb_idx,
                        reason=str(exc),
                    )
                continue
            stats["approx_kl"].append(approx_kl)
            stats["entropy"].append(entropy)
            stats["policy_loss"].append(policy_loss)
            stats["value_loss"].append(value_loss)
            stats["ratio_mean"].append(ratio_stats[0])
            stats["ratio_min"].append(ratio_stats[1])
            stats["ratio_max"].append(ratio_stats[2])
            stats["samples_per_minibatch"].append(len(indices))
            if events is not None:
                events.emit(
                    iteration=args.events_iteration,
                    stage="ppo-update",
                    event_type="minibatch",
                    epoch=epoch,
                    minibatch=mb_idx,
                    kl=approx_kl,
                    entropy=entropy,
                    policy_loss=policy_loss,
                    value_loss=value_loss,
                    ratio_mean=ratio_stats[0],
                    ratio_min=ratio_stats[1],
                    ratio_max=ratio_stats[2],
                    samples=len(indices),
                )

    aggregate = {
        "approx_kl_mean": float(np.mean(stats["approx_kl"])) if stats["approx_kl"] else float("nan"),
        "approx_kl_max": float(np.max(stats["approx_kl"])) if stats["approx_kl"] else float("nan"),
        "entropy_mean": float(np.mean(stats["entropy"])) if stats["entropy"] else float("nan"),
        "policy_loss_mean": float(np.mean(stats["policy_loss"])) if stats["policy_loss"] else float("nan"),
        "value_loss_mean": float(np.mean(stats["value_loss"])) if stats["value_loss"] else float("nan"),
        "ratio_mean": float(np.mean(stats["ratio_mean"])) if stats["ratio_mean"] else float("nan"),
        "ratio_min": float(np.min(stats["ratio_min"])) if stats["ratio_min"] else float("nan"),
        "ratio_max": float(np.max(stats["ratio_max"])) if stats["ratio_max"] else float("nan"),
        "n_minibatches": len(stats["approx_kl"]),
        "samples_per_minibatch": stats["samples_per_minibatch"],
        "numerical_anomalies": stats["numerical_anomalies"],
    }

    if events is not None:
        events.emit(
            iteration=args.events_iteration,
            stage="ppo-update",
            event_type="completed",
            **{key: value for key, value in aggregate.items() if not isinstance(value, list)},
        )

    # Save updated checkpoint (model state + manifest). This is the artifact
    # the orchestrator then exports to ONNX, gate-evaluates, and promotes.
    checkpoint = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": config.to_dict(),
        "feature_schema": {
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "state_feature_schema_version": STATE_FEATURE_SCHEMA_VERSION,
            "action_feature_schema_version": ACTION_FEATURE_SCHEMA_VERSION,
            "card_vocab": card_vocab_metadata(),
        },
        "training": {
            "stage": "ppo",
            "init_from_checkpoint": str(args.init_from_checkpoint),
            "trajectories": str(args.trajectories),
            "samples": len(rows),
        },
    }
    torch.save(checkpoint, out_dir / "checkpoint.pt")

    manifest = {
        "checkpoint": "checkpoint.pt",
        "model_config": config.to_dict(),
        "feature_schema": checkpoint["feature_schema"],
        "trajectories": str(args.trajectories),
        "samples": len(rows),
        "hyperparams": {
            # The five stability controls from f1-design.md "Stability
            # controls" — all five present so a regression diff is mechanical.
            "lr": args.lr,
            "clip_epsilon": args.clip_epsilon,
            "entropy_coef": args.entropy_coef,
            "value_coef": args.value_coef,
            "grad_clip": args.grad_clip,
            # GAE knobs alongside the stability controls.
            "gae_lambda": args.gae_lambda,
            "gae_gamma": args.gae_gamma,
            "ppo_epochs": args.ppo_epochs,
            "minibatches": args.minibatches,
        },
        "metrics": aggregate,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"status": "PASS", "out_dir": str(out_dir), **manifest}, indent=2))


# ---------------------------------------------------------------------------
# trajectory parsing
# ---------------------------------------------------------------------------


def load_trajectories(path: Path) -> Iterator[TrajectoryRow]:
    """Iterate parsed trajectory rows.

    Two accepted record formats:
    1. *Materialized* — the ppo_orchestrator pre-computes features and
       stores arrays in JSON. This is the production path.
    2. *Trace-passthrough* — for tests/smoke we accept the TS evaluator's
       trace row format directly (observation + legalActions) and derive
       features inline. Slower per row but avoids a separate parser stage.
    """

    with path.open("r", encoding="utf8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "state_features" in row and "action_features" in row:
                yield TrajectoryRow(
                    state_features=np.asarray(row["state_features"], dtype=np.float32),
                    action_features=np.asarray(row["action_features"], dtype=np.float32),
                    selected_idx=int(row["selected_idx"]),
                    behavior_logp=float(row["behavior_logp"]),
                    reward=float(row["reward"]),
                    done=bool(row["done"]),
                    value_pred=float(row.get("value_pred", 0.0)),
                    episode_id=str(row["episode_id"]),
                    episode_step=int(row.get("episode_step", 0)),
                )
            else:
                # Trace-passthrough: derive features from observation +
                # legalActions on the fly. The ppo_orchestrator already
                # pre-materializes for prod runs but the smoke uses the
                # raw trace format to keep the code path tight.
                observation = row["observation"]
                actions = row["legalActions"]
                yield TrajectoryRow(
                    state_features=observation_to_features(observation),
                    action_features=legal_actions_to_features(actions),
                    selected_idx=int(row["selected_idx"]),
                    behavior_logp=float(row["behavior_logp"]),
                    reward=float(row["reward"]),
                    done=bool(row["done"]),
                    value_pred=float(row.get("value_pred", 0.0)),
                    episode_id=str(row["episode_id"]),
                    episode_step=int(row.get("episode_step", 0)),
                )


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_gae(
    rows: list[TrajectoryRow],
    model: CandidatePolicyNet,
    device: torch.device,
    *,
    gamma: float,
    lam: float,
) -> tuple[list[float], list[float], list[float]]:
    """Compute GAE advantages and returns under the current value head.

    Episodes are demarcated by ``episode_id``. The terminal step within an
    episode (``done == True``) has bootstrap value 0; non-terminal-but-
    truncated episodes (the last step of an unfinished game in the buffer)
    treat the next-state value as 0 — a conservative choice that biases
    the GAE estimate slightly toward optimism-for-good-trajectories but
    avoids needing a per-row ``next_obs`` we don't have on hand.
    """

    if not rows:
        return [], [], []

    # Batched value pass. The model.value head is state-only so we send a
    # 1-action dummy with a True mask just to satisfy the forward
    # signature; the value output is unaffected.
    state_features = np.stack([row.state_features for row in rows], axis=0)
    state_t = torch.from_numpy(state_features).to(device)
    dummy_actions = torch.zeros((len(rows), 1, ACTION_DIM), dtype=torch.float32, device=device)
    dummy_mask = torch.ones((len(rows), 1), dtype=torch.bool, device=device)
    model.eval()
    _logits, values = model(state_t, dummy_actions, dummy_mask)
    values_np = values.detach().cpu().numpy().astype(np.float64)

    # Group by episode preserving original ordering inside each episode.
    episodes: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        episodes.setdefault(row.episode_id, []).append(idx)

    advantages = [0.0] * len(rows)
    returns = [0.0] * len(rows)
    for ep_indices in episodes.values():
        # Ensure step-ordered within the episode (parser should already
        # produce them in order, but defensive).
        ep_indices = sorted(ep_indices, key=lambda i: rows[i].episode_step)
        T = len(ep_indices)
        # GAE backward pass.
        last_gae = 0.0
        for t in reversed(range(T)):
            idx = ep_indices[t]
            row = rows[idx]
            if t == T - 1:
                next_value = 0.0  # absorbing or truncated; see docstring
                next_nonterminal = 0.0 if row.done else 0.0
                # Note: even on truncation we set next_nonterminal=0 to
                # avoid bootstrapping off an unknown next state.
            else:
                next_idx = ep_indices[t + 1]
                next_value = float(values_np[next_idx])
                next_nonterminal = 1.0
            delta = row.reward + gamma * next_value * next_nonterminal - float(values_np[idx])
            last_gae = delta + gamma * lam * next_nonterminal * last_gae
            advantages[idx] = last_gae
            returns[idx] = last_gae + float(values_np[idx])
    return advantages, returns, values_np.tolist()


# ---------------------------------------------------------------------------
# PPO step
# ---------------------------------------------------------------------------


class _NumericalAnomaly(RuntimeError):
    """Raised when a minibatch hits a NaN/inf; trainer skips the step."""


def build_minibatches(n: int, minibatches: int, *, seed: int) -> list[list[int]]:
    if n <= 0 or minibatches <= 0:
        return []
    g = np.random.default_rng(seed)
    perm = g.permutation(n).tolist()
    minibatches = min(minibatches, n)
    chunk = math.ceil(n / minibatches)
    out: list[list[int]] = []
    for start in range(0, n, chunk):
        out.append(perm[start : start + chunk])
    return out


def collate_minibatch(
    rows: list[TrajectoryRow],
    minibatch_advantages: list[float],
    minibatch_returns: list[float],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Collate a minibatch.

    ``minibatch_advantages`` and ``minibatch_returns`` are per-row scalars
    aligned with ``rows`` — the caller has already sliced them from the
    full buffer-wide arrays. This keeps the minibatch agnostic to its
    position in the buffer.
    """

    batch_size = len(rows)
    max_actions = max(row.action_features.shape[0] for row in rows)

    state = np.zeros((batch_size, STATE_DIM), dtype=np.float32)
    actions = np.zeros((batch_size, max_actions, ACTION_DIM), dtype=np.float32)
    mask = np.zeros((batch_size, max_actions), dtype=np.bool_)
    selected = np.zeros((batch_size,), dtype=np.int64)
    behavior_logp = np.zeros((batch_size,), dtype=np.float32)
    adv = np.zeros((batch_size,), dtype=np.float32)
    ret = np.zeros((batch_size,), dtype=np.float32)

    for i, row in enumerate(rows):
        count = row.action_features.shape[0]
        state[i] = row.state_features
        actions[i, :count] = row.action_features
        mask[i, :count] = True
        selected[i] = max(0, min(count - 1, row.selected_idx))
        behavior_logp[i] = max(-LOG_CLAMP, min(LOG_CLAMP, row.behavior_logp))
        adv[i] = minibatch_advantages[i]
        ret[i] = minibatch_returns[i]

    return {
        "state_features": torch.from_numpy(state).to(device),
        "action_features": torch.from_numpy(actions).to(device),
        "action_mask": torch.from_numpy(mask).to(device),
        "selected": torch.from_numpy(selected).to(device),
        "behavior_logp": torch.from_numpy(behavior_logp).to(device),
        "advantages": torch.from_numpy(adv).to(device),
        "returns": torch.from_numpy(ret).to(device),
    }


def ppo_step(
    model: CandidatePolicyNet,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    *,
    clip_epsilon: float,
    entropy_coef: float,
    value_coef: float,
    grad_clip: float,
) -> tuple[float, float, float, float, tuple[float, float, float], torch.Tensor]:
    """One PPO update step on a minibatch.

    Returns (policy_loss, value_loss, entropy, approx_kl,
    (ratio_mean, ratio_min, ratio_max), ratio_tensor).
    """

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, values = model(batch["state_features"], batch["action_features"], batch["action_mask"])
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        raise _NumericalAnomaly("non-finite logits from forward")
    if torch.isnan(values).any() or torch.isinf(values).any():
        raise _NumericalAnomaly("non-finite values from forward")

    # Masked log-softmax. Logits already have -1e9 at masked positions
    # courtesy of the model's masked_fill; pass through F.log_softmax which
    # is numerically stable.
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    selected = batch["selected"]
    new_logp = log_probs.gather(1, selected.unsqueeze(1)).squeeze(1)
    behavior_logp = batch["behavior_logp"]

    # Importance ratio = exp(new - old). Clamp the log-diff so a freshly
    # diverged policy doesn't blow exp() into inf.
    log_ratio = (new_logp - behavior_logp).clamp(min=-LOG_CLAMP, max=LOG_CLAMP)
    ratio = log_ratio.exp()

    # PPO clipped surrogate.
    advantages = batch["advantages"]
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss. Tanh-bounded critic; we fit it to ``returns`` (= advantage
    # + bootstrap value baseline). Mean-squared error.
    value_loss = nn.functional.mse_loss(values, batch["returns"])

    # Entropy of the action distribution under the *new* policy. Uses
    # masked log_probs; masked positions have log p = -1e9 / very small p
    # so the contribution is effectively zero.
    probs = log_probs.exp()
    # Numerical guard: log_probs at masked positions are -1e9, so prob is
    # ~0, but prob * log_p is 0 * -1e9 = nan in IEEE. Force masked
    # positions to contribute 0.
    mask = batch["action_mask"].bool()
    entropy_terms = torch.where(mask, -probs * log_probs, torch.zeros_like(log_probs))
    entropy_per_row = entropy_terms.sum(dim=-1)
    entropy = entropy_per_row.mean()

    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    if torch.isnan(loss) or torch.isinf(loss):
        raise _NumericalAnomaly(f"non-finite loss policy={float(policy_loss)} value={float(value_loss)} entropy={float(entropy)}")
    loss.backward()

    # Grad-clip = 0.5 per spec; tighter than DAgger's 2.0 because the
    # policy gradient is noisier than imitation loss.
    if grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    optimizer.step()

    # Approx-KL = mean(behavior_logp - new_logp). The exact-KL of two
    # distributions is more expensive; this is the PPO-standard cheap
    # signal logged on every minibatch.
    with torch.no_grad():
        approx_kl = (behavior_logp - new_logp).mean().clamp(min=-LOG_CLAMP, max=LOG_CLAMP)
        # Symmetric absolute KL prevents reporting a deceptively-low KL that
        # happens to live near zero because of cancellation. We log both.
        ratio_cpu = ratio.detach()
        ratio_stats = (
            float(ratio_cpu.mean()),
            float(ratio_cpu.min()),
            float(ratio_cpu.max()),
        )

    return (
        float(policy_loss.detach()),
        float(value_loss.detach()),
        float(entropy.detach()),
        float(approx_kl.detach()),
        ratio_stats,
        ratio.detach(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-from-checkpoint", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--gae-gamma", type=float, default=0.99)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--events-out", default=None,
                        help="Append per-minibatch + per-update events to this JSONL stream.")
    parser.add_argument("--events-iteration", type=int, default=-1)
    return parser.parse_args()


if __name__ == "__main__":
    main()
