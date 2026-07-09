"""Configurable NxN gridworld for RLHF-style PPO experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# Actions: up, down, left, right
ACTIONS = ["up", "down", "left", "right"]
NUM_ACTIONS = 4

DEFAULT_GRID_SIZE = 10
STEP_PENALTY = -0.01
GOAL_REWARD = 1.0
TRAP_REWARD = -1.0

# Legacy module-level aliases (default 10x10).
GRID_SIZE = DEFAULT_GRID_SIZE
MAX_STEPS = 6 * DEFAULT_GRID_SIZE


def default_max_steps(grid_size: int) -> int:
    return 6 * grid_size


def goal_pos(grid_size: int) -> Tuple[int, int]:
    return (0, grid_size - 1)


def trap_pos(grid_size: int) -> Tuple[int, int]:
    return (grid_size - 1, 0)


def onehot_to_coords(obs: np.ndarray, grid_size: int) -> np.ndarray:
    """Decode one-hot grid cell to normalized (row, col) coords."""
    idx = int(np.argmax(obs))
    r, c = divmod(idx, grid_size)
    denom = max(grid_size - 1, 1)
    return np.array([r / denom, c / denom], dtype=np.float32)


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    info: dict


@dataclass
class Trajectory:
    """A rollout stored for preference labeling and reward-model training."""

    states: List[np.ndarray]
    actions: List[int]
    true_return: float

    def __len__(self) -> int:
        return len(self.actions)


class GridWorld:
    """
    NxN gridworld with a goal (+1), trap (-1), and small step penalty.

    Goal at top-right, trap at bottom-left. The policy is trained on learned
    reward-model outputs; true rewards are used only for evaluation and
    preference-pair labeling.
    """

    def __init__(
        self,
        grid_size: int = DEFAULT_GRID_SIZE,
        obs_type: str = "onehot",
        seed: Optional[int] = None,
    ):
        if grid_size < 3:
            raise ValueError("grid_size must be at least 3")
        if obs_type not in ("onehot", "coords"):
            raise ValueError("obs_type must be 'onehot' or 'coords'")
        self.grid_size = grid_size
        self.max_steps = default_max_steps(grid_size)
        self.goal_pos = goal_pos(grid_size)
        self.trap_pos = trap_pos(grid_size)
        self.obs_type = obs_type
        self.rng = random.Random(seed)
        self.pos: Tuple[int, int] = (0, 0)
        self.steps = 0

    @property
    def obs_dim(self) -> int:
        return self.grid_size * self.grid_size if self.obs_type == "onehot" else 2

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = random.Random(seed)
        terminals = (self.goal_pos, self.trap_pos)
        non_terminal = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in terminals
        ]
        self.pos = self.rng.choice(non_terminal)
        self.steps = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        r, c = self.pos
        if self.obs_type == "onehot":
            vec = np.zeros(self.grid_size * self.grid_size, dtype=np.float32)
            vec[r * self.grid_size + c] = 1.0
            return vec
        denom = max(self.grid_size - 1, 1)
        return np.array([r / denom, c / denom], dtype=np.float32)

    def step(self, action: int) -> StepResult:
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        r = min(max(self.pos[0] + dr, 0), self.grid_size - 1)
        c = min(max(self.pos[1] + dc, 0), self.grid_size - 1)
        self.pos = (r, c)
        self.steps += 1

        if self.pos == self.goal_pos:
            reward, done = GOAL_REWARD, True
        elif self.pos == self.trap_pos:
            reward, done = TRAP_REWARD, True
        elif self.steps >= self.max_steps:
            reward, done = STEP_PENALTY, True
        else:
            reward, done = STEP_PENALTY, False

        return StepResult(self._obs(), reward, done, {"pos": self.pos})

    def rollout(self, policy_fn, max_steps: Optional[int] = None) -> Trajectory:
        """Collect a trajectory using policy_fn(obs) -> action."""
        if max_steps is None:
            max_steps = self.max_steps
        states, actions = [], []
        total_reward = 0.0
        obs = self.reset()
        done = False
        while not done and len(actions) < max_steps:
            states.append(obs.copy())
            action = policy_fn(obs)
            actions.append(action)
            result = self.step(action)
            total_reward += result.reward
            obs = result.obs
            done = result.done
        return Trajectory(states=states, actions=actions, true_return=total_reward)
