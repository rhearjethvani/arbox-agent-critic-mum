"""
RLHF-style PPO experiment: reward model drift and adaptive clipping.

Compares fixed reward model PPO, vanilla periodically-updated RM PPO, and
adaptive-clip PPO when the learned reward signal shifts during training.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from gridworld import GridWorld, Trajectory
from networks import ActorCritic, RewardModel
from ppo import PPOTrainer


# ---------------------------------------------------------------------------
# Preference data and reward-model training
# ---------------------------------------------------------------------------


@dataclass
class PreferencePair:
    preferred: Trajectory
    rejected: Trajectory


@dataclass
class PreferenceDataset:
    pairs: List[PreferencePair] = field(default_factory=list)

    def add_pairs_from_trajectories(
        self, trajs: List[Trajectory], max_new_pairs: int = 64, rng: Optional[random.Random] = None
    ) -> int:
        """Label pairs by true return: preferred if true_return(A) > true_return(B)."""
        if rng is None:
            rng = random.Random()
        if len(trajs) < 2:
            return 0
        added = 0
        indices = list(range(len(trajs)))
        attempts = 0
        while added < max_new_pairs and attempts < max_new_pairs * 10:
            attempts += 1
            i, j = rng.sample(indices, 2)
            a, b = trajs[i], trajs[j]
            if abs(a.true_return - b.true_return) < 1e-6:
                continue
            if a.true_return > b.true_return:
                self.pairs.append(PreferencePair(preferred=a, rejected=b))
            else:
                self.pairs.append(PreferencePair(preferred=b, rejected=a))
            added += 1
        return added

    def __len__(self) -> int:
        return len(self.pairs)


def train_reward_model(
    model: RewardModel,
    dataset: PreferenceDataset,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 0,
) -> List[float]:
    """Train reward model with Bradley-Terry preference loss."""
    if len(dataset) == 0:
        return []
    torch.manual_seed(seed)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses: List[float] = []
    pairs = dataset.pairs

    for _ in range(epochs):
        perm = torch.randperm(len(pairs)).tolist()
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(pairs), batch_size):
            batch_idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            loss = torch.stack(
                [model.preference_loss(pairs[i].preferred, pairs[i].rejected) for i in batch_idx]
            ).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))
    return losses


def compute_drift(
    old_rm: RewardModel, new_rm: RewardModel, validation_trajs: List[Trajectory]
) -> float:
    """Mean absolute change in predicted trajectory returns on fixed validation set."""
    if not validation_trajs:
        return 0.0
    diffs = []
    for traj in validation_trajs:
        old_r = old_rm.trajectory_return(traj)
        new_r = new_rm.trajectory_return(traj)
        diffs.append(abs(new_r - old_r))
    return float(np.mean(diffs))


def preference_accuracy(
    model: RewardModel, pairs: List[PreferencePair]
) -> float:
    if not pairs:
        return float("nan")
    correct = 0
    for pair in pairs:
        r_pref = model.trajectory_return(pair.preferred)
        r_rej = model.trajectory_return(pair.rejected)
        if r_pref > r_rej:
            correct += 1
    return correct / len(pairs)


def clip_eps_from_drift(drift: float) -> float:
    if drift < 0.05:
        return 0.25
    if drift < 0.15:
        return 0.15
    return 0.05


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def policy_rollout_true_return(env: GridWorld, policy: ActorCritic, n_episodes: int = 20) -> float:
    returns = []
    for _ in range(n_episodes):
        states, actions = [], []
        total = 0.0
        obs = env.reset()
        done = False
        while not done:
            action, _, _ = policy.act(obs)
            states.append(obs.copy())
            actions.append(action)
            result = env.step(action)
            total += result.reward
            obs = result.obs
            done = result.done
        returns.append(total)
    return float(np.mean(returns))


def policy_rollout_learned_return(
    policy: ActorCritic, reward_model: RewardModel, env: GridWorld, n_episodes: int = 20
) -> float:
    returns = []
    for _ in range(n_episodes):
        states = []
        obs = env.reset()
        done = False
        while not done:
            action, _, _ = policy.act(obs)
            states.append(obs.copy())
            result = env.step(action)
            obs = result.obs
            done = result.done
        traj = Trajectory(states=states, actions=[], true_return=0.0)
        returns.append(reward_model.trajectory_return(traj))
    return float(np.mean(returns))


def collect_random_trajectories(env: GridWorld, n: int, seed: int) -> List[Trajectory]:
    rng = random.Random(seed)
    trajs = []
    for ep in range(n):
        states, actions = [], []
        total = 0.0
        obs = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = rng.randint(0, 3)
            states.append(obs.copy())
            actions.append(action)
            result = env.step(action)
            total += result.reward
            obs = result.obs
            done = result.done
        trajs.append(Trajectory(states=states, actions=actions, true_return=total))
    return trajs


def collect_policy_trajectories(
    env: GridWorld, policy: ActorCritic, n: int
) -> List[Trajectory]:
    trajs = []
    for _ in range(n):
        states, actions = [], []
        total = 0.0
        obs = env.reset()
        done = False
        while not done:
            action, _, _ = policy.act(obs)
            states.append(obs.copy())
            actions.append(action)
            result = env.step(action)
            total += result.reward
            obs = result.obs
            done = result.done
        trajs.append(Trajectory(states=states, actions=actions, true_return=total))
    return trajs


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    method: str
    num_outer_iters: int = 25
    rm_update_interval: int = 5
    seed: int = 0
    ppo_updates_per_outer: int = 2
    initial_pref_pairs: int = 128
    pairs_per_update: int = 48
    eval_episodes: int = 15


def run_experiment(cfg: RunConfig) -> List[Dict]:
    """
    Outer loop:
      collect trajectories -> (maybe) update RM -> PPO updates -> evaluate
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    env = GridWorld(obs_type="onehot", seed=cfg.seed)
    obs_dim = env.obs_dim

    policy = ActorCritic(obs_dim, num_actions=4)
    reward_model = RewardModel(obs_dim)
    old_rm: Optional[RewardModel] = None

    # Fixed validation trajectories for drift measurement (random behavior).
    validation_trajs = collect_random_trajectories(env, n=40, seed=cfg.seed + 1000)
    heldout_pairs_ds = PreferenceDataset()
    heldout_pairs_ds.add_pairs_from_trajectories(
        collect_random_trajectories(env, n=30, seed=cfg.seed + 2000),
        max_new_pairs=60,
        rng=random.Random(cfg.seed + 3000),
    )

    # Initial preference data from random rollouts.
    pref_ds = PreferenceDataset()
    init_trajs = collect_random_trajectories(env, n=80, seed=cfg.seed + 10)
    pref_ds.add_pairs_from_trajectories(
        init_trajs, max_new_pairs=cfg.initial_pref_pairs, rng=random.Random(cfg.seed + 11)
    )
    train_reward_model(reward_model, pref_ds, epochs=40, seed=cfg.seed)

    clip_eps = 0.2
    last_drift = 0.0
    pref_acc = preference_accuracy(reward_model, heldout_pairs_ds.pairs)

    ppo = PPOTrainer(
        env=env,
        policy=policy,
        reward_model=reward_model,
        clip_eps=clip_eps,
        rollout_steps=192,
        ppo_epochs=3,
    )

    logs: List[Dict] = []

    for outer_iter in range(cfg.num_outer_iters):
        recent_trajs = collect_policy_trajectories(env, policy, n=24)

        rm_updated = False
        if cfg.method in ("vanilla_updated_rm", "adaptive_clip"):
            if outer_iter % cfg.rm_update_interval == 0 and outer_iter > 0:
                pref_ds.add_pairs_from_trajectories(
                    recent_trajs,
                    max_new_pairs=cfg.pairs_per_update,
                    rng=random.Random(cfg.seed + outer_iter),
                )
                old_rm = copy.deepcopy(reward_model)
                train_reward_model(
                    reward_model, pref_ds, epochs=20, seed=cfg.seed + outer_iter
                )
                last_drift = compute_drift(old_rm, reward_model, validation_trajs)
                pref_acc = preference_accuracy(reward_model, heldout_pairs_ds.pairs)
                rm_updated = True
                if cfg.method == "adaptive_clip":
                    clip_eps = clip_eps_from_drift(last_drift)
                    ppo.set_clip_eps(clip_eps)

        # PPO training on learned rewards (not true env reward).
        kl_values = []
        for _ in range(cfg.ppo_updates_per_outer):
            batch, _ = ppo.collect_rollout()
            metrics = ppo.update(batch)
            kl_values.append(metrics["approx_policy_kl"])

        true_ret = policy_rollout_true_return(env, policy, cfg.eval_episodes)
        learned_ret = policy_rollout_learned_return(policy, reward_model, env, cfg.eval_episodes)

        logs.append(
            {
                "method": cfg.method,
                "seed": cfg.seed,
                "outer_iter": outer_iter,
                "true_eval_return": true_ret,
                "learned_eval_return": learned_ret,
                "reward_model_drift": last_drift,
                "reward_model_pref_accuracy": pref_acc,
                "clip_epsilon": clip_eps,
                "approx_policy_kl": float(np.mean(kl_values)),
                "preference_dataset_size": len(pref_ds),
                "rm_updated": int(rm_updated),
            }
        )

    return logs


# ---------------------------------------------------------------------------
# Multi-seed aggregation, CSV, plots
# ---------------------------------------------------------------------------


METHODS = ["fixed_rm", "vanilla_updated_rm", "adaptive_clip"]
METHOD_LABELS = {
    "fixed_rm": "Fixed RM",
    "vanilla_updated_rm": "Vanilla updated RM",
    "adaptive_clip": "Adaptive clip",
}


def save_logs_csv(all_logs: List[Dict], path: str) -> None:
    if not all_logs:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = list(all_logs[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_logs)


def aggregate_by_method(
    all_logs: List[Dict], metric: str
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return per-method (xs, mean, std) over seeds."""
    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for method in METHODS:
        method_logs = [r for r in all_logs if r["method"] == method]
        seeds = sorted(set(r["seed"] for r in method_logs))
        if not seeds:
            continue
        # Align by outer_iter
        iters = sorted(set(r["outer_iter"] for r in method_logs))
        xs = np.array(iters)
        curves = []
        for seed in seeds:
            seed_rows = {r["outer_iter"]: r[metric] for r in method_logs if r["seed"] == seed}
            curves.append([seed_rows[i] for i in iters])
        arr = np.array(curves, dtype=np.float64)
        out[method] = (xs, arr.mean(axis=0), arr.std(axis=0))
    return out


def plot_metric(
    all_logs: List[Dict],
    metric: str,
    ylabel: str,
    title: str,
    out_path: str,
) -> None:
    agg = aggregate_by_method(all_logs, metric)
    plt.figure(figsize=(7, 4.5))
    for method, (xs, mean, std) in agg.items():
        plt.plot(xs, mean, label=METHOD_LABELS[method], linewidth=2)
        plt.fill_between(xs, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Outer iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def print_summary(all_logs: List[Dict]) -> None:
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY: Reward Model Drift & PPO Stability")
    print("=" * 60)
    for method in METHODS:
        rows = [r for r in all_logs if r["method"] == method]
        if not rows:
            continue
        final_by_seed: Dict[int, float] = {}
        kl_by_seed: Dict[int, List[float]] = {}
        for r in rows:
            kl_by_seed.setdefault(r["seed"], []).append(r["approx_policy_kl"])
            if r["outer_iter"] == max(x["outer_iter"] for x in rows if x["seed"] == r["seed"]):
                final_by_seed[r["seed"]] = r["true_eval_return"]
        finals = list(final_by_seed.values())
        kls = [np.mean(v) for v in kl_by_seed.values()]
        print(f"\n{METHOD_LABELS[method]}:")
        print(f"  Final true return: {np.mean(finals):.3f} ± {np.std(finals):.3f}")
        print(f"  Mean policy KL:    {np.mean(kls):.4f} ± {np.std(kls):.4f}")
    print("\nInterpretation:")
    print("  - Higher final true return = better alignment with real goal/trap rewards.")
    print("  - Large policy KL spikes after RM updates suggest PPO instability.")
    print("  - Adaptive clip should reduce KL spikes when drift is high.")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RLHF-style PPO experiment studying reward model drift."
    )
    parser.add_argument("--num_outer_iters", type=int, default=25)
    parser.add_argument("--rm_update_interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0, help="Starting seed when num_seeds=1")
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--ppo_updates_per_outer", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    seeds = [args.seed + i for i in range(args.num_seeds)]

    all_logs: List[Dict] = []
    for method in METHODS:
        for seed in seeds:
            print(f"Running {method} seed={seed} ...")
            cfg = RunConfig(
                method=method,
                num_outer_iters=args.num_outer_iters,
                rm_update_interval=args.rm_update_interval,
                seed=seed,
                ppo_updates_per_outer=args.ppo_updates_per_outer,
            )
            logs = run_experiment(cfg)
            all_logs.extend(logs)

    csv_path = os.path.join(args.results_dir, "experiment_logs.csv")
    save_logs_csv(all_logs, csv_path)
    print(f"Saved logs to {csv_path}")

    plots = [
        ("true_eval_return", "True eval return", "True environment return vs training"),
        ("learned_eval_return", "Learned eval return", "Reward model return vs training"),
        ("reward_model_drift", "RM drift", "Reward model drift on validation trajectories"),
        ("clip_epsilon", "PPO clip ε", "PPO clip epsilon vs training"),
        ("approx_policy_kl", "Approx policy KL", "Approximate policy KL vs training"),
    ]
    for metric, ylabel, title in plots:
        out = os.path.join(args.results_dir, f"{metric}.png")
        plot_metric(all_logs, metric, ylabel, title, out)
        print(f"Saved plot {out}")

    print_summary(all_logs)


if __name__ == "__main__":
    main()
