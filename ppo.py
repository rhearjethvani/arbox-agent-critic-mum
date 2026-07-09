"""Minimal PPO implementation using learned reward-model signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from gridworld import GridWorld, Trajectory
from networks import ActorCritic, RewardModel


@dataclass
class PPOBatch:
    obs: np.ndarray
    actions: np.ndarray
    logp_old: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    values: np.ndarray


class PPOTrainer:
    def __init__(
        self,
        env: GridWorld,
        policy: ActorCritic,
        reward_model: RewardModel,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        use_clipping: bool = True,
        kl_coef: float = 0.0,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        ppo_epochs: int = 4,
        minibatch_size: int = 64,
        rollout_steps: int = 256,
        device: str = "cpu",
    ):
        self.env = env
        self.policy = policy
        self.reward_model = reward_model
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.use_clipping = use_clipping
        self.kl_coef = kl_coef
        self.ref_policy: Optional[ActorCritic] = None
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.rollout_steps = rollout_steps
        self.device = device
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def set_clip_eps(self, clip_eps: float) -> None:
        self.clip_eps = clip_eps

    def set_kl_coef(self, kl_coef: float) -> None:
        self.kl_coef = kl_coef

    def set_ref_policy(self, ref_policy: Optional[ActorCritic]) -> None:
        self.ref_policy = ref_policy

    def collect_rollout(self) -> Tuple[PPOBatch, List[Trajectory]]:
        """Collect on-policy rollout; rewards come from the frozen reward model."""
        obs_list, act_list, logp_list, val_list, rew_list, done_list = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        trajs: List[Trajectory] = []
        states_traj: List[np.ndarray] = []
        actions_traj: List[int] = []

        obs = self.env.reset()
        for _ in range(self.rollout_steps):
            action, logp, value = self.policy.act(obs)
            states_traj.append(obs.copy())
            actions_traj.append(action)

            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                rm_reward = float(self.reward_model(obs_t).item())

            result = self.env.step(action)
            obs_list.append(obs)
            act_list.append(action)
            logp_list.append(logp)
            val_list.append(value)
            rew_list.append(rm_reward)
            done_list.append(float(result.done))

            obs = result.obs
            if result.done:
                trajs.append(
                    Trajectory(
                        states=states_traj.copy(),
                        actions=actions_traj.copy(),
                        true_return=0.0,  # filled by caller if needed
                    )
                )
                states_traj, actions_traj = [], []
                obs = self.env.reset()

        if actions_traj:
            trajs.append(
                Trajectory(states=states_traj, actions=actions_traj, true_return=0.0)
            )

        rewards = np.array(rew_list, dtype=np.float32)
        values = np.array(val_list, dtype=np.float32)
        dones = np.array(done_list, dtype=np.float32)

        # Simple discounted returns as advantages baseline.
        returns = self._discounted_returns(rewards, dones, values[-1])
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch = PPOBatch(
            obs=np.array(obs_list, dtype=np.float32),
            actions=np.array(act_list, dtype=np.int64),
            logp_old=np.array(logp_list, dtype=np.float32),
            returns=returns.astype(np.float32),
            advantages=advantages.astype(np.float32),
            values=values.astype(np.float32),
        )
        return batch, trajs

    def _discounted_returns(
        self, rewards: np.ndarray, dones: np.ndarray, last_value: float
    ) -> np.ndarray:
        out = np.zeros_like(rewards)
        g = last_value
        for t in reversed(range(len(rewards))):
            g = rewards[t] + self.gamma * g * (1.0 - dones[t])
            out[t] = g
        return out

    def _ref_policy_kl(self, logp: torch.Tensor, mb_obs: torch.Tensor, mb_actions: torch.Tensor) -> torch.Tensor:
        if self.ref_policy is None:
            return torch.zeros(1, device=logp.device)
        with torch.no_grad():
            ref_logp, _, _ = self.ref_policy.evaluate_actions(mb_obs, mb_actions)
        return (torch.exp(ref_logp - logp) - (ref_logp - logp) - 1).mean()

    def update(self, batch: PPOBatch) -> Dict[str, float]:
        obs = torch.as_tensor(batch.obs, device=self.device)
        actions = torch.as_tensor(batch.actions, device=self.device)
        logp_old = torch.as_tensor(batch.logp_old, device=self.device)
        returns = torch.as_tensor(batch.returns, device=self.device)
        advantages = torch.as_tensor(batch.advantages, device=self.device)

        n = len(obs)
        approx_kls: List[float] = []
        ref_kls: List[float] = []
        policy_losses: List[float] = []
        value_losses: List[float] = []

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(n)
            for start in range(0, n, self.minibatch_size):
                idx = perm[start : start + self.minibatch_size]
                mb_obs = obs[idx]
                mb_actions = actions[idx]
                mb_logp_old = logp_old[idx]
                mb_returns = returns[idx]
                mb_adv = advantages[idx]

                logp, values, entropy = self.policy.evaluate_actions(mb_obs, mb_actions)
                ratio = torch.exp(logp - mb_logp_old)

                if self.use_clipping:
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                else:
                    policy_loss = -(ratio * mb_adv).mean()

                value_loss = F.mse_loss(values, mb_returns)
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                )

                if not self.use_clipping and self.kl_coef > 0:
                    ref_kl = self._ref_policy_kl(logp, mb_obs, mb_actions)
                    loss = loss + self.kl_coef * ref_kl

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    ratio = torch.exp(logp - mb_logp_old)
                    approx_kl = (ratio - 1 - torch.log(ratio)).mean().item()
                    ref_kl_val = self._ref_policy_kl(logp, mb_obs, mb_actions).item()
                approx_kls.append(approx_kl)
                ref_kls.append(ref_kl_val)
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

        metrics = {
            "approx_policy_kl": float(np.mean(approx_kls)),
            "ref_policy_kl": float(np.mean(ref_kls)),
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
        }
        if self.use_clipping:
            metrics["clip_epsilon"] = self.clip_eps
        else:
            metrics["kl_coef"] = self.kl_coef
        return metrics
