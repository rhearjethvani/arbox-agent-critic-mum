# Reward Model Drift & Adaptive PPO Constraints

A toy RLHF-style experiment asking: **when the reward model changes during PPO training, does PPO become unstable—and can adaptive trust-region constraints improve robustness?**

Two **constraint suites** are run in parallel (default: both):

| Suite | Mechanism | Output |
|-------|-----------|--------|
| **clip** | PPO clipped surrogate + critic/drift-informed ε | `results/clip/{full_rm,broken_rm}/` |
| **kl** | Unclipped surrogate + β·KL(π∥π_ref) penalty | `results/kl/{full_rm,broken_rm}/` |

## Research question

In RLHF pipelines, the policy is trained on a learned reward model (RM) that is periodically updated from new human or synthetic preferences. Each RM update shifts the reward signal the policy optimizes. This experiment studies whether that **reward model drift** causes PPO instability (large policy updates / high KL), and whether **adaptive clipping** or **adaptive KL penalties** mitigate it.

## Environment

A configurable **N×N gridworld** (default **10×10**):

- Random non-terminal start
- Goal cell (top-right): true reward **+1**
- Trap cell (bottom-left): true reward **-1**
- Step penalty: **-0.01**
- Episode ends at goal, trap, or **6×N** steps (60 on 10×10)
- Actions: up, down, left, right

Use `--grid_size 5` to reproduce the original 5×5 setup for comparison.

**Important:** PPO never trains on true rewards. True rewards are used only for evaluation and for labeling preference pairs.

## Reward model

A small neural network predicts per-state reward. Trajectory return is the sum of predicted rewards. The model is trained on **Bradley–Terry preference loss** from pairs labeled by true return:

```
loss = -log σ(RM_return_preferred - RM_return_rejected)
```

**Drift** is measured on a fixed validation set of trajectories as the mean absolute change in predicted returns before vs after an RM update.

### Full vs broken RM (dual-run)

By default each suite runs the **same** 4-method experiment **twice** with different RM capacity (policy/actor-critic unchanged):

| Variant | Architecture | Role |
|---------|--------------|------|
| **full_rm** | 2-layer MLP on **one-hot** obs, `hidden = hidden_dim_for_grid(N)` | Baseline (~9.8k params on 10×10) |
| **broken_rm** | **Linear(2→1)** on normalized **(row, col)** only | Weak proxy (~3 params); policy still uses one-hot |

Use `--rm_mode full_rm`, `--rm_mode broken_rm`, or `--rm_mode both` (default).

CSV rows include `constraint_suite`, `rm_variant`, `rm_input_mode`, `rm_hidden`, and `rm_layers`.

## Clip suite — four methods

| Method | Reward model | PPO clip ε |
|--------|--------------|------------|
| **fixed_rm** | Trained once, then frozen | Fixed 0.2 |
| **vanilla_updated_rm** | Retrained every K outer iterations | Fixed 0.2 |
| **fixed_rm_critic_clip** | Same as fixed_rm | Critic-informed ε every outer iteration (drift = 0) |
| **adaptive_clip** | Same as vanilla | Last RM drift sets base ε; critic tightens every outer iteration |

Critic-informed methods recompute ε on the **first PPO rollout** of each outer iteration from critic mismatch (and drift for `adaptive_clip`). See clip ε formula in code (`clip_eps_from_drift_and_critic`).

## KL suite — four methods

Pure KL-penalty PPO (no clipping):

```
J(π) = E_π[R̂] − β · D_KL(π ∥ π_ref)
```

- **R̂** = learned reward model
- **π_ref** = policy snapshot at start of each outer iteration (frozen for that iter's PPO updates)
- **β** = KL coefficient

| Method | Reward model | β schedule |
|--------|--------------|------------|
| **fixed_rm_static_kl** | Trained once, then frozen | Fixed `β_static` |
| **updated_rm_static_kl** | Retrained every K outer iterations | Fixed `β_static` |
| **fixed_rm_adaptive_kl** | Same as fixed_rm | Target-KL adaptive β |
| **updated_rm_adaptive_kl** | Same as updated | Target-KL adaptive β |

**Adaptive β** (not critic-based):

```
β_{t+1} = β_t · c   if KL > target
β_{t+1} = β_t / c   if KL < target
```

β is clamped to `[kl_beta_min, kl_beta_max]`. Defaults: `β_static=0.1`, `target=0.02`, `c=1.5`.

## Training loop

For each outer iteration:

1. Collect trajectories with the current policy
2. If the method updates the RM and `outer_iter % rm_update_interval == 0`: add preference pairs, retrain RM, compute drift
3. (KL suite) Snapshot π_ref; set β
4. Run PPO updates on **learned** RM rewards
5. (Clip suite) Critic-informed methods recompute ε on first PPO rollout
6. (KL suite) Adaptive methods update β after each PPO update from measured KL
7. Evaluate with true environment reward
8. Log metrics

## Metrics & plots

Saved under `results/clip/` and `results/kl/` (each with `full_rm/` and `broken_rm/` when `--rm_mode both`):

| Plot | Meaning |
|------|---------|
| `true_eval_return.png` | Policy quality under the **true** reward (main success metric) |
| `learned_eval_return.png` | Return under the **current** reward model (what PPO optimizes) |
| `true_ndh_norm.png` | Goodharting on true reward R₀ ([Skalse et al. 2023](https://arxiv.org/abs/2310.09144)) |
| `true_autc_norm.png` | Area under the true-reward curve — rewards sustained true performance |
| `reward_model_drift.png` | How much RM outputs shift on validation trajectories after updates |
| `approx_policy_kl.png` | Proxy for policy change magnitude |
| `clip_epsilon.png` | *(clip suite only)* PPO clip ε vs training |
| `critic_error.png` | *(clip suite only)* Critic mismatch when ε is set |
| `kl_coef.png` | *(KL suite only)* β vs training |
| `ref_policy_kl.png` | *(KL suite only)* KL(π ∥ π_ref) vs training |

### True-reward over-optimization metrics

**NDH** (Skalse et al. 2023): `true_ndh_norm` — drop from peak true return, normalized by peak gain. Values near 0 are good; negative means Goodharting.

**AUTC**: `true_autc_norm` — ∫ J_R0 along training path. Higher = sustained true reward over the whole optimisation path.

## How to run

```bash
pip install -r requirements.txt
python main.py
```

### Key arguments

```bash
python main.py \
  --grid_size 10 \
  --num_outer_iters 25 \
  --rm_update_interval 5 \
  --rm_mode both \
  --constraint_suite both \
  --seed 0 \
  --num_seeds 3 \
  --results_dir results
```

KL-only run:

```bash
python main.py --constraint_suite kl --rm_mode broken_rm
```

Clip-only run:

```bash
python main.py --constraint_suite clip --rm_mode full_rm
```

KL hyperparameters:

```bash
python main.py --constraint_suite kl \
  --kl_beta_static 0.1 \
  --kl_target 0.02 \
  --kl_adapt_coef 1.5 \
  --kl_beta_min 0.01 \
  --kl_beta_max 1.0 \
  --kl_adapt_mode step
```

Quick smoke test:

```bash
python main.py --constraint_suite kl --rm_mode broken_rm --num_seeds 1 --num_outer_iters 3 --grid_size 5
```

Runtime scales with grid size and number of suites (~14–20 min on CPU for clip-only 10×10; ~2× with both suites).

## Expected takeaways

- **Vanilla updated RM** may show KL spikes and noisier true returns after RM updates.
- **Fixed RM** is stable but may plateau if the initial RM is weak.
- **Clip suite:** critic-informed ε and adaptive_clip dampen updates when critic mismatch (and drift) are high.
- **KL suite:** adaptive β tracks measured KL toward target; static β isolates RM update effects.
- **Broken RM** should show lower preference accuracy and worse true returns, sharpening method contrasts.

Results are stochastic; run multiple seeds (`--num_seeds 3`) and compare aggregated curves (mean ± std shaded).
