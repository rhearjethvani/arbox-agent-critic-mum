# Reward Model Drift & Adaptive PPO Clipping

A toy RLHF-style experiment asking: **when the reward model changes during PPO training, does PPO become unstable—and can shrinking the clip range when drift is high improve robustness?**

## Research question

In RLHF pipelines, the policy is trained on a learned reward model (RM) that is periodically updated from new human or synthetic preferences. Each RM update shifts the reward signal the policy optimizes. This experiment studies whether that **reward model drift** causes PPO instability (large policy updates / high KL), and whether **adaptive clipping** mitigates it.

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

By default each invocation runs the **same** 4-method experiment **twice** with different RM capacity (policy/actor-critic unchanged):

| Variant | Architecture | Role |
|---------|--------------|------|
| **full_rm** | 2-layer MLP, `hidden = hidden_dim_for_grid(N)` | Current baseline (~9.8k params on 10×10) |
| **broken_rm** | **1-layer linear**, `hidden = 8` | Weak proxy (~808 params on 10×10) |

The broken RM has far fewer parameters and should fit preferences poorly, exaggerating proxy misalignment and method differences.

Use `--rm_mode full_rm`, `--rm_mode broken_rm`, or `--rm_mode both` (default).

Results are written to separate subdirs:

- `results/full_rm/` — CSV + plots
- `results/broken_rm/` — CSV + plots

CSV rows include `rm_variant`, `rm_hidden`, and `rm_layers`.

## Four methods

| Method | Reward model | PPO clip ε |
|--------|--------------|------------|
| **fixed_rm** | Trained once, then frozen | Fixed 0.2 |
| **vanilla_updated_rm** | Retrained every K outer iterations | Fixed 0.2 |
| **fixed_rm_critic_clip** | Same as fixed_rm | Critic-informed ε every outer iteration (drift = 0) |
| **adaptive_clip** | Same as vanilla | Last RM drift sets base ε; critic tightens every outer iteration |

### What affects PPO clip ε

| Method | When ε changes | What sets ε |
|--------|----------------|-------------|
| **fixed_rm** | Never (after init) | Fixed **0.2** |
| **vanilla_updated_rm** | Never (after init) | Fixed **0.2** |
| **fixed_rm_critic_clip** | Every outer iteration | Critic mismatch only (`drift = 0`) |
| **adaptive_clip** | Every outer iteration | Last RM **drift** (base) + critic mismatch |

**When** (critic-informed methods): on the **first PPO rollout** of each outer iteration, before that batch is used for a PPO update. The second PPO rollout in the same outer iter reuses the ε just set.

**How** (two-step formula):

1. **Base ε from drift** (`clip_eps_from_drift`) — `adaptive_clip` only; uses `last_drift` divided by **6×N** max steps (trajectory-return drift scales with episode length) before thresholding:

| Normalized drift | Base ε |
|------------------|--------|
| < 0.05 | `eps_max` = 0.25 |
| < 0.15 | 0.15 (midpoint) |
| ≥ 0.15 | `eps_min` = 0.05 |

`fixed_rm_critic_clip` skips this step (drift treated as 0 → base ε = 0.25).

2. **Critic tightening** (both critic-informed methods): collect on-policy rollout, then

```
critic_error = mean(|returns − values|)
stress = min(critic_error / critic_error_ref, 1.0)
final_ε = base_ε − stress × (base_ε − eps_min)
```

- `returns`: discounted sums of **learned RM** rewards on the rollout
- `values`: value-head predictions at collection time
- `critic_error_ref`: `10 × grid_size / 5` (e.g. 20 on 10×10)

Higher critic error → higher stress → smaller final ε (down to `eps_min`).

**What does *not* affect ε:** true environment reward, approximate policy KL (logged only), preference accuracy.

### Critic-informed adaptive clipping

`fixed_rm_critic_clip` and `adaptive_clip` both recompute ε on the **first PPO rollout of every outer iteration**:

1. (`adaptive_clip` only) Look up **last RM drift** → base ε via `clip_eps_from_drift` (refreshed when the RM retrains)
2. Collect an on-policy rollout
3. Measure **critic error** = mean(|returns − values|)
4. Tighten ε toward `eps_min`:

```
final_eps = base_eps − stress × (base_eps − eps_min)
stress = min(critic_error / critic_error_ref, 1.0)
```

`critic_error_ref` scales with grid size (`10 × grid_size/5`).

## Training loop

For each outer iteration:

1. Collect trajectories with the current policy
2. If the method updates the RM and `outer_iter % rm_update_interval == 0`: add preference pairs, retrain RM, compute drift
3. Run PPO updates using **learned** RM rewards (`fixed_rm_critic_clip` / `adaptive_clip` recompute ε on the first PPO rollout each outer iter)
4. Evaluate with true environment reward
5. Log metrics

Hyperparameters scale lightly with grid size: `rollout_steps` and network hidden dim increase for larger grids.

## Metrics & plots

Saved under `results/full_rm/` and `results/broken_rm/` (when `--rm_mode both`):

| Plot | Meaning |
|------|---------|
| `true_eval_return.png` | Policy quality under the **true** reward (main success metric) |
| `learned_eval_return.png` | Return under the **current** reward model (what PPO optimizes) |
| `true_ndh_norm.png` | Goodharting on true reward R₀ ([Skalse et al. 2023](https://arxiv.org/abs/2310.09144)) |
| `reward_model_drift.png` | How much RM outputs shift on validation trajectories after updates |
| `clip_epsilon.png` | Final PPO clip ε vs training (drift normalized by 6×N before thresholding) |
| `critic_error.png` | Critic mismatch each time ε is recomputed (critic-informed methods) |
| `approx_policy_kl.png` | Proxy for policy change magnitude—spikes may indicate instability |

Raw logs: `results/{full_rm,broken_rm}/experiment_logs.csv` (includes `rm_variant`, `rm_hidden`, `rm_layers`, `critic_error`, `clip_eps_base`, `true_ndh`, `true_ndh_norm`)

### True-reward over-optimization metric (NDH)

Skalse et al. (2023) Definition 5: optimise a proxy R₁ but measure **true** reward R₀:

```
NDH = J_R0(π_now) − max_{λ} J_R0(π_λ)
```

In this experiment R₀ is the **true gridworld reward** (`true_eval_return`); PPO trains on the learned RM (proxy). Each outer iteration:

- `true_ndh` = current true return − running peak true return (≤ 0; more negative = larger Goodhart drop)
- `true_ndh_norm` = `true_ndh` / (peak true return − initial true return)

Values near 0 mean true return is still at its training peak. Negative values mean the policy has fallen below the best true performance seen while optimising the proxy.

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
  --seed 0 \
  --num_seeds 3 \
  --results_dir results
```

Quick smoke test (both RM variants):

```bash
python main.py --rm_mode both --num_seeds 1 --num_outer_iters 3 --grid_size 5
```

Compare with the original small env:

```bash
python main.py --grid_size 5
```

Runtime scales with grid size (~14–20 min on CPU for 10×10, 4 methods × 3 seeds × 2 RM variants).

## Expected takeaways

- **Vanilla updated RM** may show KL spikes and noisier true returns after RM updates.
- **Fixed RM** is stable but may plateau if the initial RM is weak.
- **Fixed RM + critic ε** isolates whether critic-informed clipping helps even without RM drift.
- **Adaptive clip** should dampen policy updates when both RM drift and critic mismatch are high, potentially improving stability without fully freezing the RM.

- **Broken RM** (`broken_rm`) should show lower preference accuracy and worse true returns, making method contrasts sharper.
- **Full RM** (`full_rm`) is the capacity-matched baseline.

Results are stochastic; run multiple seeds (`--num_seeds 3`) and compare aggregated curves (mean ± std shaded).
