"""
RLHF-style PPO experiment: reward model drift and adaptive clipping.

Compares fixed reward model PPO, vanilla periodically-updated RM PPO, and
critic-informed drift-adaptive PPO when the learned reward signal shifts
during training.
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

from gridworld import GridWorld, Trajectory, default_max_steps
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


def normalize_drift_for_grid(drift: float, grid_size: int) -> float:
    """Per-step drift: trajectory-return drift scales with episode length (6×N)."""
    return drift / default_max_steps(grid_size)


def drift_thresholds_for_grid(
    grid_size: int, ref_grid: int = 5, base_low: float = 0.05, base_mid: float = 0.15
) -> Tuple[float, float]:
    """Reference thresholds at ref_grid; scale linearly with N for raw drift plots."""
    scale = grid_size / ref_grid
    return base_low * scale, base_mid * scale


def clip_eps_from_drift(
    drift: float,
    eps_max: float = 0.25,
    eps_min: float = 0.05,
    drift_low: float = 0.05,
    drift_mid: float = 0.15,
    grid_size: Optional[int] = None,
) -> float:
    if grid_size is not None:
        drift = normalize_drift_for_grid(drift, grid_size)
    eps_mid = (eps_max + eps_min) / 2.0
    if drift < drift_low:
        return eps_max
    if drift < drift_mid:
        return eps_mid
    return eps_min


def clip_eps_from_drift_and_critic(
    drift: float,
    critic_error: float,
    eps_max: float = 0.25,
    eps_min: float = 0.05,
    drift_low: float = 0.05,
    drift_mid: float = 0.15,
    critic_error_ref: float = 10.0,
    grid_size: Optional[int] = None,
) -> Tuple[float, float]:
    """Drift sets base ε; critic mismatch tightens toward eps_min."""
    base_eps = clip_eps_from_drift(
        drift,
        eps_max=eps_max,
        eps_min=eps_min,
        drift_low=drift_low,
        drift_mid=drift_mid,
        grid_size=grid_size,
    )
    stress = min(critic_error / critic_error_ref, 1.0)
    final_eps = base_eps - stress * (base_eps - eps_min)
    return final_eps, base_eps


def update_kl_coef(
    beta: float,
    measured_kl: float,
    target_kl: float,
    adapt_coef: float,
    beta_min: float,
    beta_max: float,
    mode: str = "step",
) -> float:
    """Target-KL adaptive β: increase when KL above target, decrease when below."""
    if mode == "smooth":
        if target_kl > 1e-8:
            beta *= 1.0 + 0.5 * (measured_kl / target_kl - 1.0)
    else:
        if measured_kl > target_kl:
            beta *= adapt_coef
        else:
            beta /= adapt_coef
    return float(np.clip(beta, beta_min, beta_max))


def critic_error_ref_for_grid(grid_size: int, base: float = 10.0, ref_grid: int = 5) -> float:
    return base * (grid_size / ref_grid)


def rollout_steps_for_grid(grid_size: int, base: int = 192, ref_grid: int = 5) -> int:
    return int(base * (grid_size / ref_grid))


def hidden_dim_for_grid(grid_size: int) -> int:
    if grid_size <= 5:
        return 64
    if grid_size <= 10:
        return 96
    return 128


RM_VARIANTS = ["full_rm", "broken_rm"]
RM_VARIANT_DIRS = {"full_rm": "full_rm", "broken_rm": "broken_rm"}

CONSTRAINT_SUITES = ["clip", "kl"]
SUITE_DIRS = {"clip": "clip", "kl": "kl"}

CLIP_METHODS = ["fixed_rm", "vanilla_updated_rm", "fixed_rm_critic_clip", "adaptive_clip"]
CLIP_METHOD_LABELS = {
    "fixed_rm": "Fixed RM, fixed ε",
    "vanilla_updated_rm": "Updated RM, fixed ε",
    "fixed_rm_critic_clip": "Fixed RM, critic ε",
    "adaptive_clip": "Updated RM, critic ε",
}

KL_METHODS = [
    "fixed_rm_static_kl",
    "updated_rm_static_kl",
    "fixed_rm_adaptive_kl",
    "updated_rm_adaptive_kl",
]
KL_METHOD_LABELS = {
    "fixed_rm_static_kl": "Fixed RM, static β",
    "updated_rm_static_kl": "Updated RM, static β",
    "fixed_rm_adaptive_kl": "Fixed RM, adaptive β",
    "updated_rm_adaptive_kl": "Updated RM, adaptive β",
}

UPDATED_RM_METHODS = {
    "vanilla_updated_rm",
    "adaptive_clip",
    "updated_rm_static_kl",
    "updated_rm_adaptive_kl",
}
ADAPTIVE_KL_METHODS = {"fixed_rm_adaptive_kl", "updated_rm_adaptive_kl"}
CRITIC_CLIP_METHODS = {"fixed_rm_critic_clip", "adaptive_clip"}


def methods_for_suite(constraint_suite: str) -> List[str]:
    if constraint_suite == "clip":
        return CLIP_METHODS
    if constraint_suite == "kl":
        return KL_METHODS
    raise ValueError(f"Unknown constraint_suite: {constraint_suite}")


def method_labels_for_suite(constraint_suite: str) -> Dict[str, str]:
    if constraint_suite == "clip":
        return CLIP_METHOD_LABELS
    if constraint_suite == "kl":
        return KL_METHOD_LABELS
    raise ValueError(f"Unknown constraint_suite: {constraint_suite}")


def method_updates_rm(method: str) -> bool:
    return method in UPDATED_RM_METHODS


def method_uses_critic_clip(method: str) -> bool:
    return method in CRITIC_CLIP_METHODS


def method_uses_adaptive_kl(method: str) -> bool:
    return method in ADAPTIVE_KL_METHODS


def method_uses_drift_for_clip(method: str) -> bool:
    return method == "adaptive_clip"


@dataclass
class RewardModelSpec:
    hidden: int
    layers: int
    input_mode: str


def reward_model_spec_for_variant(variant: str, grid_size: int) -> RewardModelSpec:
    if variant == "full_rm":
        return RewardModelSpec(
            hidden=hidden_dim_for_grid(grid_size), layers=2, input_mode="onehot"
        )
    if variant == "broken_rm":
        return RewardModelSpec(hidden=2, layers=1, input_mode="coords")
    raise ValueError(f"Unknown rm_variant: {variant}")


def reward_model_param_count(
    obs_dim: int,
    spec: RewardModelSpec,
    grid_size: Optional[int] = None,
) -> int:
    model = RewardModel(
        obs_dim,
        hidden=spec.hidden,
        layers=spec.layers,
        input_mode=spec.input_mode,
        grid_size=grid_size if spec.input_mode == "coords" else None,
    )
    return sum(p.numel() for p in model.parameters())


def normalised_drop_height(
    true_return: float,
    peak_true: float,
    initial_true: float,
) -> Tuple[float, float]:
    """
    Normalised drop height on true reward R_0 (Skalse et al. 2023, arXiv:2310.09144).

    Definition 5: NDH = J_R0(π_1) - max_{λ∈[0,1]} J_R0(π_λ), i.e. loss of true
    reward along the optimisation path. Here outer_iter proxies λ and true_return
    is J_R0(π). peak_true is the running max of J_R0 over training.

    ndh_norm divides by peak gain J_R0(π*) - J_R0(π_0) for scale-free comparison.
    ndh <= 0; more negative means larger drop from best true return so far.
    """
    ndh = true_return - peak_true
    gain = peak_true - initial_true
    ndh_norm = ndh / gain if abs(gain) > 1e-8 else 0.0
    return ndh, ndh_norm


def area_under_true_reward_curve(true_history: List[float]) -> Tuple[float, float]:
    """
    Area under the true-reward curve (AUTC).

    AUTC = ∫_0^1 J_R_true(π_λ) dλ, where outer_iter proxies optimisation pressure λ.
    Rewards methods that maintain high true reward over the whole path, not just at one point.

    autc_norm integrates returns normalized by peak gain (J* − J_0) for scale-free comparison.
    """
    n = len(true_history)
    if n < 2:
        return 0.0, 0.0
    lambdas = np.linspace(0.0, 1.0, n)
    autc = float(np.trapz(true_history, lambdas))
    initial = true_history[0]
    peak = max(true_history)
    gain = peak - initial
    if abs(gain) < 1e-8:
        return autc, 0.0
    g = [(val - initial) / gain for val in true_history]
    autc_norm = float(np.trapz(g, lambdas))
    return autc, autc_norm


def add_true_autc_to_logs(all_logs: List[Dict]) -> None:
    """Populate true_autc / true_autc_norm from true_eval_return (in-place)."""
    groups: Dict[Tuple[str, int], List[Dict]] = {}
    for row in all_logs:
        key = (row["method"], row["seed"])
        groups.setdefault(key, []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: r["outer_iter"])
        history: List[float] = []
        for row in rows:
            history.append(float(row["true_eval_return"]))
            autc, autc_norm = area_under_true_reward_curve(history)
            row["true_autc"] = autc
            row["true_autc_norm"] = autc_norm


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
    constraint_suite: str = "clip"
    num_outer_iters: int = 25
    rm_update_interval: int = 5
    seed: int = 0
    ppo_updates_per_outer: int = 2
    initial_pref_pairs: int = 128
    pairs_per_update: int = 48
    eval_episodes: int = 15
    grid_size: int = 10
    eps_max: float = 0.25
    eps_min: float = 0.05
    rm_variant: str = "full_rm"
    kl_beta_static: float = 0.1
    kl_target: float = 0.02
    kl_adapt_coef: float = 1.5
    kl_beta_min: float = 0.01
    kl_beta_max: float = 1.0
    kl_adapt_mode: str = "step"

    @property
    def rm_spec(self) -> RewardModelSpec:
        return reward_model_spec_for_variant(self.rm_variant, self.grid_size)

    @property
    def rm_hidden(self) -> int:
        return self.rm_spec.hidden

    @property
    def rm_layers(self) -> int:
        return self.rm_spec.layers

    @property
    def rm_input_mode(self) -> str:
        return self.rm_spec.input_mode

    @property
    def critic_error_ref(self) -> float:
        return critic_error_ref_for_grid(self.grid_size)

    @property
    def rollout_steps(self) -> int:
        return rollout_steps_for_grid(self.grid_size)

    @property
    def hidden_dim(self) -> int:
        return hidden_dim_for_grid(self.grid_size)


def run_experiment(cfg: RunConfig) -> List[Dict]:
    """
    Outer loop:
      collect trajectories -> (maybe) update RM -> PPO updates -> evaluate
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    env = GridWorld(grid_size=cfg.grid_size, obs_type="onehot", seed=cfg.seed)
    obs_dim = env.obs_dim
    hidden = cfg.hidden_dim

    policy = ActorCritic(obs_dim, num_actions=4, hidden=hidden)
    reward_model = RewardModel(
        obs_dim,
        hidden=cfg.rm_hidden,
        layers=cfg.rm_layers,
        input_mode=cfg.rm_input_mode,
        grid_size=cfg.grid_size if cfg.rm_input_mode == "coords" else None,
    )
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
    kl_coef = cfg.kl_beta_static
    last_drift = 0.0
    pref_acc = preference_accuracy(reward_model, heldout_pairs_ds.pairs)

    use_clipping = cfg.constraint_suite == "clip"
    ppo = PPOTrainer(
        env=env,
        policy=policy,
        reward_model=reward_model,
        clip_eps=clip_eps,
        use_clipping=use_clipping,
        kl_coef=kl_coef,
        rollout_steps=cfg.rollout_steps,
        ppo_epochs=3,
    )

    logs: List[Dict] = []
    initial_true: Optional[float] = None
    peak_true = float("-inf")
    true_history: List[float] = []

    for outer_iter in range(cfg.num_outer_iters):
        recent_trajs = collect_policy_trajectories(env, policy, n=24)

        rm_updated = False
        if method_updates_rm(cfg.method):
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

        ref_policy_kl_values: List[float] = []
        if not use_clipping:
            ref_policy = copy.deepcopy(policy)
            ppo.set_ref_policy(ref_policy)
            if method_uses_adaptive_kl(cfg.method):
                ppo.set_kl_coef(kl_coef)
            else:
                kl_coef = cfg.kl_beta_static
                ppo.set_kl_coef(kl_coef)

        # PPO training on learned rewards (not true env reward).
        kl_values = []
        critic_error = float("nan")
        clip_eps_base = float("nan")
        for ppo_i in range(cfg.ppo_updates_per_outer):
            batch, _ = ppo.collect_rollout()
            if use_clipping:
                clip_eps_base = clip_eps
                if method_uses_critic_clip(cfg.method) and ppo_i == 0:
                    critic_error = float(np.mean(np.abs(batch.returns - batch.values)))
                    drift = last_drift if method_uses_drift_for_clip(cfg.method) else 0.0
                    clip_eps, clip_eps_base = clip_eps_from_drift_and_critic(
                        drift,
                        critic_error,
                        eps_max=cfg.eps_max,
                        eps_min=cfg.eps_min,
                        critic_error_ref=cfg.critic_error_ref,
                        grid_size=cfg.grid_size,
                    )
                    ppo.set_clip_eps(clip_eps)
            metrics = ppo.update(batch)
            kl_values.append(metrics["approx_policy_kl"])
            if not use_clipping:
                ref_policy_kl_values.append(metrics.get("ref_policy_kl", float("nan")))
                if method_uses_adaptive_kl(cfg.method):
                    kl_coef = update_kl_coef(
                        kl_coef,
                        metrics["approx_policy_kl"],
                        cfg.kl_target,
                        cfg.kl_adapt_coef,
                        cfg.kl_beta_min,
                        cfg.kl_beta_max,
                        mode=cfg.kl_adapt_mode,
                    )
                    ppo.set_kl_coef(kl_coef)

        true_ret = policy_rollout_true_return(env, policy, cfg.eval_episodes)
        learned_ret = policy_rollout_learned_return(policy, reward_model, env, cfg.eval_episodes)

        if initial_true is None:
            initial_true = true_ret
        peak_true = max(peak_true, true_ret)
        true_ndh, true_ndh_norm = normalised_drop_height(true_ret, peak_true, initial_true)

        true_history.append(true_ret)
        true_autc, true_autc_norm = area_under_true_reward_curve(true_history)

        log_row: Dict = {
            "constraint_suite": cfg.constraint_suite,
            "method": cfg.method,
            "seed": cfg.seed,
            "grid_size": cfg.grid_size,
            "rm_variant": cfg.rm_variant,
            "rm_input_mode": cfg.rm_input_mode,
            "rm_hidden": cfg.rm_hidden,
            "rm_layers": cfg.rm_layers,
            "outer_iter": outer_iter,
            "true_eval_return": true_ret,
            "learned_eval_return": learned_ret,
            "true_ndh": true_ndh,
            "true_ndh_norm": true_ndh_norm,
            "true_autc": true_autc,
            "true_autc_norm": true_autc_norm,
            "reward_model_drift": last_drift,
            "reward_model_pref_accuracy": pref_acc,
            "approx_policy_kl": float(np.mean(kl_values)),
            "preference_dataset_size": len(pref_ds),
            "rm_updated": int(rm_updated),
        }
        if use_clipping:
            log_row["clip_epsilon"] = clip_eps
            log_row["clip_eps_base"] = clip_eps_base
            log_row["critic_error"] = critic_error
            log_row["kl_coef"] = float("nan")
            log_row["kl_target"] = float("nan")
            log_row["ref_policy_kl"] = float("nan")
        else:
            log_row["clip_epsilon"] = float("nan")
            log_row["clip_eps_base"] = float("nan")
            log_row["critic_error"] = float("nan")
            log_row["kl_coef"] = kl_coef
            log_row["kl_target"] = cfg.kl_target
            log_row["ref_policy_kl"] = (
                float(np.mean(ref_policy_kl_values)) if ref_policy_kl_values else float("nan")
            )
        logs.append(log_row)

    return logs


# ---------------------------------------------------------------------------
# Multi-seed aggregation, CSV, plots
# ---------------------------------------------------------------------------


METHODS = CLIP_METHODS
METHOD_LABELS = CLIP_METHOD_LABELS


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
    all_logs: List[Dict],
    metric: str,
    methods: Optional[List[str]] = None,
    method_labels: Optional[Dict[str, str]] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return per-method (xs, mean, std) over seeds."""
    if methods is None:
        methods = sorted(set(r["method"] for r in all_logs))
    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for method in methods:
        method_logs = [r for r in all_logs if r["method"] == method]
        seeds = sorted(set(r["seed"] for r in method_logs))
        if not seeds:
            continue
        iters = sorted(set(r["outer_iter"] for r in method_logs))
        xs = np.array(iters)
        curves = []
        for seed in seeds:
            seed_rows = {r["outer_iter"]: r[metric] for r in method_logs if r["seed"] == seed}
            curves.append([seed_rows[i] for i in iters])
        arr = np.array(curves, dtype=np.float64)
        out[method] = (xs, np.nanmean(arr, axis=0), np.nanstd(arr, axis=0))
    return out


def plot_metric(
    all_logs: List[Dict],
    metric: str,
    ylabel: str,
    title: str,
    out_path: str,
    methods: Optional[List[str]] = None,
    method_labels: Optional[Dict[str, str]] = None,
) -> None:
    labels = method_labels or METHOD_LABELS
    agg = aggregate_by_method(all_logs, metric, methods=methods)
    plt.figure(figsize=(7, 4.5))
    for method, (xs, mean, std) in agg.items():
        if np.all(np.isnan(mean)):
            continue
        plt.plot(xs, mean, label=labels.get(method, method), linewidth=2)
        plt.fill_between(xs, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Outer iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def print_summary(
    all_logs: List[Dict],
    constraint_suite: str,
    rm_variant: Optional[str] = None,
) -> None:
    methods = methods_for_suite(constraint_suite)
    labels = method_labels_for_suite(constraint_suite)
    print("\n" + "=" * 60)
    title = f"EXPERIMENT SUMMARY ({constraint_suite}): Reward Model Drift & PPO Stability"
    if rm_variant is not None:
        title += f" ({rm_variant})"
    print(title)
    print("=" * 60)
    for method in methods:
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
        ndh_norm_by_seed: Dict[int, float] = {}
        autc_norm_by_seed: Dict[int, float] = {}
        for r in rows:
            if r["outer_iter"] == max(x["outer_iter"] for x in rows if x["seed"] == r["seed"]):
                ndh_norm_by_seed[r["seed"]] = float(r["true_ndh_norm"])
                autc_norm_by_seed[r["seed"]] = float(r["true_autc_norm"])
        ndh_norms = list(ndh_norm_by_seed.values())
        autc_norms = list(autc_norm_by_seed.values())
        print(f"\n{labels[method]}:")
        print(f"  Final true return: {np.mean(finals):.3f} ± {np.std(finals):.3f}")
        print(f"  Mean policy KL:    {np.mean(kls):.4f} ± {np.std(kls):.4f}")
        print(f"  Final true NDH (norm): {np.mean(ndh_norms):.3f} ± {np.std(ndh_norms):.3f}")
        print(f"  Final true AUTC (norm): {np.mean(autc_norms):.3f} ± {np.std(autc_norms):.3f}")
    print("\nInterpretation:")
    print("  - Higher final true return = better alignment with real goal/trap rewards.")
    print("  - Large policy KL spikes after RM updates suggest PPO instability.")
    if constraint_suite == "clip":
        print("  - Critic-informed methods recompute ε every outer iter from critic mismatch.")
        print("  - adaptive_clip also uses last RM drift for base ε (updated when RM retrains).")
    else:
        print("  - KL suite: J(π)=E[R̂]−β·KL(π∥π_ref); adaptive β from measured KL vs target.")
        print("  - Static β methods hold β fixed; adaptive methods update β each PPO update.")
    print("  - True NDH (norm): J_R0(π) − max J_R0 vs peak gain (Skalse et al. 2023); ≤ 0.")
    print("  - True AUTC (norm): ∫ J_R0 along training path; higher = sustained true reward.")
    print("=" * 60 + "\n")


def plots_for_suite(constraint_suite: str) -> List[Tuple[str, str, str]]:
    shared = [
        ("true_eval_return", "True eval return", "True environment return vs training"),
        ("learned_eval_return", "Learned eval return", "Reward model return vs training"),
        ("true_ndh_norm", "True NDH (normalised)", "Normalised drop height on true reward R₀"),
        (
            "true_autc_norm",
            "True AUTC (normalised)",
            "Area under the true-reward curve (overoptimization)",
        ),
        ("reward_model_drift", "RM drift", "Reward model drift on validation trajectories"),
        ("approx_policy_kl", "Approx policy KL", "Approximate policy KL vs training"),
    ]
    if constraint_suite == "clip":
        return shared + [
            ("clip_epsilon", "PPO clip ε", "PPO clip epsilon vs training (drift ÷ 6×N)"),
            ("critic_error", "Critic error", "Mean |returns − values| when ε is set"),
        ]
    return shared + [
        ("kl_coef", "KL coefficient β", "KL penalty coefficient vs training"),
        ("ref_policy_kl", "Ref-policy KL", "KL(π ∥ π_ref) vs training"),
    ]


def run_variant_experiments(
    rm_variant: str,
    seeds: List[int],
    args: argparse.Namespace,
    constraint_suite: str,
) -> List[Dict]:
    suite_dir = os.path.join(args.results_dir, SUITE_DIRS[constraint_suite])
    variant_dir = os.path.join(suite_dir, RM_VARIANT_DIRS[rm_variant])
    os.makedirs(variant_dir, exist_ok=True)

    methods = methods_for_suite(constraint_suite)
    labels = method_labels_for_suite(constraint_suite)
    all_logs: List[Dict] = []
    for method in methods:
        for seed in seeds:
            print(f"Running [{constraint_suite}/{rm_variant}] {method} seed={seed} ...")
            cfg = RunConfig(
                method=method,
                constraint_suite=constraint_suite,
                num_outer_iters=args.num_outer_iters,
                rm_update_interval=args.rm_update_interval,
                seed=seed,
                ppo_updates_per_outer=args.ppo_updates_per_outer,
                grid_size=args.grid_size,
                rm_variant=rm_variant,
                kl_beta_static=args.kl_beta_static,
                kl_target=args.kl_target,
                kl_adapt_coef=args.kl_adapt_coef,
                kl_beta_min=args.kl_beta_min,
                kl_beta_max=args.kl_beta_max,
                kl_adapt_mode=args.kl_adapt_mode,
            )
            logs = run_experiment(cfg)
            all_logs.extend(logs)

    csv_path = os.path.join(variant_dir, "experiment_logs.csv")
    add_true_autc_to_logs(all_logs)
    save_logs_csv(all_logs, csv_path)
    print(f"Saved logs to {csv_path}")

    for metric, ylabel, title in plots_for_suite(constraint_suite):
        out = os.path.join(variant_dir, f"{metric}.png")
        plot_metric(all_logs, metric, ylabel, title, out, methods=methods, method_labels=labels)
        print(f"Saved plot {out}")

    print_summary(all_logs, constraint_suite=constraint_suite, rm_variant=rm_variant)
    return all_logs


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
    parser.add_argument("--grid_size", type=int, default=10, help="NxN gridworld size")
    parser.add_argument(
        "--rm_mode",
        type=str,
        default="both",
        choices=["both", "full_rm", "broken_rm"],
        help="Run with full RM, broken RM (coords linear), or both",
    )
    parser.add_argument(
        "--constraint_suite",
        type=str,
        default="both",
        choices=["both", "clip", "kl"],
        help="PPO constraint suite: clip epsilon, KL penalty, or both",
    )
    parser.add_argument("--kl_beta_static", type=float, default=0.1)
    parser.add_argument("--kl_target", type=float, default=0.02)
    parser.add_argument("--kl_adapt_coef", type=float, default=1.5)
    parser.add_argument("--kl_beta_min", type=float, default=0.01)
    parser.add_argument("--kl_beta_max", type=float, default=1.0)
    parser.add_argument(
        "--kl_adapt_mode",
        type=str,
        default="step",
        choices=["step", "smooth"],
        help="Adaptive β update rule",
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    seeds = [args.seed + i for i in range(args.num_seeds)]
    obs_dim = args.grid_size * args.grid_size

    print(f"Grid size: {args.grid_size}x{args.grid_size}")
    print(f"  max_steps={6 * args.grid_size}, rollout_steps={rollout_steps_for_grid(args.grid_size)}")
    print(f"  critic_error_ref={critic_error_ref_for_grid(args.grid_size):.2f}")
    print(f"  policy hidden={hidden_dim_for_grid(args.grid_size)}")
    for variant in RM_VARIANTS:
        spec = reward_model_spec_for_variant(variant, args.grid_size)
        n_params = reward_model_param_count(obs_dim, spec, grid_size=args.grid_size)
        print(
            f"  RM [{variant}]: input={spec.input_mode}, hidden={spec.hidden}, "
            f"layers={spec.layers}, params={n_params}"
        )
    if args.constraint_suite in ("both", "kl"):
        print(
            f"  KL penalty: β_static={args.kl_beta_static}, target={args.kl_target}, "
            f"adapt_coef={args.kl_adapt_coef}, mode={args.kl_adapt_mode}"
        )
    print()

    if args.rm_mode == "both":
        variants = RM_VARIANTS
    else:
        variants = [args.rm_mode]

    if args.constraint_suite == "both":
        suites = CONSTRAINT_SUITES
    else:
        suites = [args.constraint_suite]

    for constraint_suite in suites:
        for rm_variant in variants:
            run_variant_experiments(rm_variant, seeds, args, constraint_suite=constraint_suite)


if __name__ == "__main__":
    main()
