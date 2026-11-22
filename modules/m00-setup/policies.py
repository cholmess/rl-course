"""Module 0: hand-written CartPole policies.

Observation: [x, x_dot, theta, theta_dot]. Actions: 0 = push left, 1 = push right.
"""

import gymnasium as gym
import numpy as np


def random_policy(obs: np.ndarray, rng: np.random.Generator) -> int:
    """Ignores the observation and picks left or right at random. Baseline."""
    return int(rng.integers(2))


def angle_policy(obs: np.ndarray, rng: np.random.Generator) -> int:
    """Pushes the cart toward the side the pole leans to. Uses the angle only."""
    theta = obs[2]
    return 1 if theta > 0 else 0


def angle_velocity_policy(obs: np.ndarray, rng: np.random.Generator) -> int:
    """Your policy: use both the pole angle and its angular velocity.

    Return 0 (push left) or 1 (push right).
    """
    # TODO: write your rule. See Hint 2 in README.md if you're stuck.
    raise NotImplementedError


def run_episode(env: gym.Env, policy, seed: int) -> float:
    """Run one episode of `policy` in `env` and return the total reward.

    - Reset the environment with `seed` so runs are reproducible.
    - Create the policy's random generator with `np.random.default_rng(seed)`.
    - Step until the episode is terminated or truncated.
    """
    # TODO: implement the agent-environment loop. See Hint 1 in README.md.
    raise NotImplementedError
