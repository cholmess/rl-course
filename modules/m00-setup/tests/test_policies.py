import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policies import angle_velocity_policy, random_policy, run_episode  # noqa: E402

RNG = np.random.default_rng(0)


@pytest.fixture
def env():
    e = gym.make("CartPole-v1")
    yield e
    e.close()


def test_run_episode_returns_a_number_in_range(env):
    total = run_episode(env, random_policy, seed=0)
    assert isinstance(total, (int, float, np.floating))
    assert 1 <= total <= 500


def test_run_episode_is_reproducible(env):
    assert run_episode(env, random_policy, seed=3) == run_episode(env, random_policy, seed=3)


def test_run_episode_stops_at_truncation(env):
    always_balanced = lambda obs, rng: 1 if obs[2] + 0.5 * obs[3] > 0 else 0  # noqa: E731
    assert run_episode(env, always_balanced, seed=0) <= 500


def test_policy_returns_valid_actions():
    for _ in range(100):
        obs = RNG.uniform(-0.2, 0.2, size=4)
        assert angle_velocity_policy(obs, RNG) in (0, 1)


def test_policy_pushes_toward_a_falling_pole():
    falling_right = np.array([0.0, 0.0, 0.05, 0.5])
    falling_left = np.array([0.0, 0.0, -0.05, -0.5])
    assert angle_velocity_policy(falling_right, RNG) == 1
    assert angle_velocity_policy(falling_left, RNG) == 0


def test_policy_balances_well(env):
    returns = [run_episode(env, angle_velocity_policy, seed) for seed in range(20)]
    assert np.mean(returns) >= 195, f"mean return {np.mean(returns):.1f}; aim for at least 195"
