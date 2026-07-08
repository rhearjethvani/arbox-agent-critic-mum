"""Small 5x5 gridworld for RLHF-style PPO experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# Actions: up, down, left, right
ACTIONS = ["up", "down", "left", "right"]
NUM_ACTIONS = 4

# Fixed layout: goal top-right, trap bottom-left (non-overlapping terminals).
GOAL_POS = (0, 4)
TRAP_POS = (4, 0)
GRID_SIZE = 5
MAX_STEPS = 30
STEP_PENALTY = -0.01
GOAL_REWARD = 1.0
TRAP_REWARD = -1.0


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
    5x5 gridworld with a goal (+1), trap (-1), and small step penalty.

    The policy is trained on learned reward-model outputs; true rewards are used
  only for evaluation and preference-pair labeling.
    """

    def __init__(self, obs_type: str = "onehot", seed: Optional[int] = None):
        if obs_type not in ("onehot", "coords"):
            raise ValueError("obs_type must be 'onehot' or 'coords'")
        self.obs_type = obs_type
        self.rng = random.Random(seed)
        self.pos: Tuple[int, int] = (0, 0)
        self.steps = 0

    @property
    def obs_dim(self) -> int:
        return GRID_SIZE * GRID_SIZE if self.obs_type == "onehot" else 2

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = random.Random(seed)
        non_terminal = [
            (r, c)
            for r in range(GRID_SIZE)
            for c in range(GRID_SIZE)
            if (r, c) not in (GOAL_POS, TRAP_POS)
        ]
        self.pos = self.rng.choice(non_terminal)
        self.steps = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        r, c = self.pos
        if self.obs_type == "onehot":
            vec = np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.float32)
            vec[r * GRID_SIZE + c] = 1.0
            return vec
        return np.array([r / (GRID_SIZE - 1), c / (GRID_SIZE - 1)], dtype=np.float32)

    def step(self, action: int) -> StepResult:
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        r = min(max(self.pos[0] + dr, 0), GRID_SIZE - 1)
        c = min(max(self.pos[1] + dc, 0), GRID_SIZE - 1)
        self.pos = (r, c)
        self.steps += 1

        if self.pos == GOAL_POS:
            reward, done = GOAL_REWARD, True
        elif self.pos == TRAP_POS:
            reward, done = TRAP_REWARD, True
        elif self.steps >= MAX_STEPS:
            reward, done = STEP_PENALTY, True
        else:
            reward, done = STEP_PENALTY, False

        return StepResult(self._obs(), reward, done, {"pos": self.pos})

    def rollout(self, policy_fn, max_steps: int = MAX_STEPS) -> Trajectory:
        """Collect a trajectory using policy_fn(obs) -> action."""
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
