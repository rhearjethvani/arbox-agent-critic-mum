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

## Three methods

| Method | Reward model | PPO clip ε |
|--------|--------------|------------|
| **fixed_rm** | Trained once, then frozen | Fixed 0.2 |
| **vanilla_updated_rm** | Retrained every K outer iterations | Fixed 0.2 |
| **adaptive_clip** | Same as vanilla | Drift-based base ε (0.25 / 0.15 / 0.05), tightened by critic mismatch after RM updates |

### Critic-informed adaptive clipping

On each RM update iteration, `adaptive_clip`:

1. Computes **drift** → discrete base ε via `clip_eps_from_drift`
2. Collects an on-policy rollout under the **new** RM
3. Measures **critic error** = mean(|returns − values|) — how misaligned the value head is with the new reward signal
4. Tightens ε toward `eps_min` proportional to critic stress:

```
final_eps = base_eps − stress × (base_eps − eps_min)
stress = min(critic_error / critic_error_ref, 1.0)
```

`critic_error_ref` scales with grid size (`10 × grid_size/5`).

## Training loop

For each outer iteration:

1. Collect trajectories with the current policy
2. If the method updates the RM and `outer_iter % rm_update_interval == 0`: add preference pairs, retrain RM, compute drift
3. Run PPO updates using **learned** RM rewards (adaptive_clip sets ε from drift + critic on the first PPO step after an RM update)
4. Evaluate with true environment reward
5. Log metrics

Hyperparameters scale lightly with grid size: `rollout_steps` and network hidden dim increase for larger grids.

## Metrics & plots

Saved under `results/`:

| Plot | Meaning |
|------|---------|
| `true_eval_return.png` | Policy quality under the **true** reward (main success metric) |
| `learned_eval_return.png` | Return under the **current** reward model (what PPO optimizes) |
| `reward_model_drift.png` | How much RM outputs shift on validation trajectories after updates |
| `clip_epsilon.png` | Final PPO clip ε vs training |
| `clip_eps_base.png` | Drift-only ε before critic tightening (adaptive_clip) |
| `critic_error.png` | Critic mismatch after RM updates (adaptive_clip) |
| `approx_policy_kl.png` | Proxy for policy change magnitude—spikes may indicate instability |

Raw logs: `results/experiment_logs.csv` (includes `grid_size`, `critic_error`, `clip_eps_base`)

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
  --seed 0 \
  --num_seeds 3 \
  --results_dir results
```

Compare with the original small env:

```bash
python main.py --grid_size 5
```

Runtime scales with grid size (~5–8 min on CPU for 10×10, 3 methods × 3 seeds).

## Expected takeaways

- **Vanilla updated RM** may show KL spikes and noisier true returns after RM updates.
- **Fixed RM** is stable but may plateau if the initial RM is weak.
- **Adaptive clip** should dampen policy updates when both RM drift and critic mismatch are high, potentially improving stability without fully freezing the RM.

If methods still look too similar on 10×10, try `--grid_size 15`, lower `--initial_pref_pairs`, or shorter `--rm_update_interval`.

Results are stochastic; run multiple seeds (`--num_seeds 3`) and compare aggregated curves (mean ± std shaded).
