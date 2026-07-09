"""Neural networks for policy, value, and reward modeling."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gridworld import Trajectory


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    mods: List[nn.Module] = []
    d = in_dim
    for _ in range(layers - 1):
        mods += [nn.Linear(d, hidden), nn.Tanh()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


def onehot_to_coords_batch(obs: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Decode batch of one-hot observations to normalized (row, col) coords."""
    idx = obs.argmax(dim=-1)
    r = idx // grid_size
    c = idx % grid_size
    denom = max(grid_size - 1, 1)
    return torch.stack([r.float() / denom, c.float() / denom], dim=-1)


class ActorCritic(nn.Module):
    """Shared-body actor-critic for categorical actions."""

    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 64):
        super().__init__()
        self.body = mlp(obs_dim, hidden, hidden, layers=2)
        self.policy_head = nn.Linear(hidden, num_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.body(obs)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def act(self, obs: np.ndarray) -> Tuple[int, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.forward(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
        return int(action.item()), float(logp.item()), float(value.item())

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(actions)
        entropy = dist.entropy()
        return logp, values, entropy


class RewardModel(nn.Module):
    """Predicts scalar reward per state (state-action extension is trivial)."""

    def __init__(
        self,
        obs_dim: int,
        hidden: int = 64,
        layers: int = 2,
        input_mode: str = "onehot",
        grid_size: Optional[int] = None,
    ):
        super().__init__()
        if input_mode not in ("onehot", "coords"):
            raise ValueError("input_mode must be 'onehot' or 'coords'")
        if input_mode == "coords" and grid_size is None:
            raise ValueError("grid_size required when input_mode='coords'")
        self.input_mode = input_mode
        self.grid_size = grid_size
        rm_in_dim = 2 if input_mode == "coords" else obs_dim
        self.net = mlp(rm_in_dim, hidden, 1, layers=layers)

    def _encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self.input_mode == "coords":
            return onehot_to_coords_batch(obs, self.grid_size)
        return obs

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.net(self._encode_obs(obs)).squeeze(-1)

    def trajectory_return(self, traj: Trajectory) -> float:
        if len(traj.states) == 0:
            return 0.0
        obs = torch.as_tensor(np.stack(traj.states), dtype=torch.float32)
        with torch.no_grad():
            rewards = self.forward(obs)
        return float(rewards.sum().item())

    def batch_trajectory_returns(self, trajs: List[Trajectory]) -> torch.Tensor:
        returns = [self.trajectory_return(t) for t in trajs]
        return torch.tensor(returns, dtype=torch.float32)

    def preference_loss(
        self, preferred: Trajectory, rejected: Trajectory
    ) -> torch.Tensor:
        r_pref = self._sum_rewards(preferred)
        r_rej = self._sum_rewards(rejected)
        return -F.logsigmoid(r_pref - r_rej).mean()

    def _sum_rewards(self, traj: Trajectory) -> torch.Tensor:
        if len(traj.states) == 0:
            return torch.tensor(0.0)
        obs = torch.as_tensor(np.stack(traj.states), dtype=torch.float32)
        return self.forward(obs).sum()
