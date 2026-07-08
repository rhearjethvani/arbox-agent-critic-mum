# Reward Model Drift & Adaptive PPO Clipping

A toy RLHF-style experiment asking: **when the reward model changes during PPO training, does PPO become unstable—and can shrinking the clip range when drift is high improve robustness?**

## Research question

In RLHF pipelines, the policy is trained on a learned reward model (RM) that is periodically updated from new human or synthetic preferences. Each RM update shifts the reward signal the policy optimizes. This experiment studies whether that **reward model drift** causes PPO instability (large policy updates / high KL), and whether **adaptive clipping** mitigates it.

## Environment

A small **5×5 gridworld**:

- Random non-terminal start
- Goal cell: true reward **+1**
- Trap cell: true reward **-1**
- Step penalty: **-0.01**
- Episode ends at goal, trap, or **30** steps
- Actions: up, down, left, right

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
| **adaptive_clip** | Same as vanilla | Adaptive from drift: 0.25 (low), 0.15 (medium), 0.05 (high) |

## Training loop

For each outer iteration:

1. Collect trajectories with the current policy
2. If the method updates the RM and `outer_iter % rm_update_interval == 0`: add preference pairs, retrain RM, compute drift (and update clip ε for adaptive_clip)
3. Run PPO updates using **learned** RM rewards
4. Evaluate with true environment reward
5. Log metrics

## Metrics & plots

Saved under `results/`:

| Plot | Meaning |
|------|---------|
| `true_eval_return.png` | Policy quality under the **true** reward (main success metric) |
| `learned_eval_return.png` | Return under the **current** reward model (what PPO optimizes) |
| `reward_model_drift.png` | How much RM outputs shift on validation trajectories after updates |
| `clip_epsilon.png` | PPO clip range over time (adaptive for method 3) |
| `approx_policy_kl.png` | Proxy for policy change magnitude—spikes may indicate instability |

Raw logs: `results/experiment_logs.csv`

## How to run

```bash
pip install -r requirements.txt
python main.py
```

### Key arguments

```bash
python main.py \
  --num_outer_iters 25 \
  --rm_update_interval 5 \
  --seed 0 \
  --num_seeds 3 \
  --results_dir results
```

Runtime is intended to stay under a few minutes on CPU.

## Expected takeaways

- **Vanilla updated RM** may show KL spikes and noisier true returns after RM updates.
- **Fixed RM** is stable but may plateau if the initial RM is weak.
- **Adaptive clip** should dampen KL spikes when drift is high, potentially improving stability without fully freezing the RM.

Results are stochastic; run multiple seeds (`--num_seeds 3`) and compare aggregated curves (mean ± std shaded).
