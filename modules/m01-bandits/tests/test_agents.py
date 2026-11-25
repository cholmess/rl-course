import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import UCB, EpsilonGreedy  # noqa: E402
from bandits import run  # noqa: E402


def make(eps=0.0, k=5, seed=0, **kw):
    return EpsilonGreedy(k, eps, np.random.default_rng(seed), **kw)


def test_update_counts_and_incremental_sample_average():
    agent = make()
    for r in (2.0, 6.0, 8.0):
        agent.update(1, r)
    assert agent.N[1] == 3 and agent.N.sum() == 3
    assert agent.Q[1] == pytest.approx(16 / 3)
    assert agent.Q[0] == 0.0


def test_update_with_constant_step_size():
    agent = make(alpha=0.5)
    agent.update(0, 4.0)
    agent.update(0, 0.0)
    assert agent.Q[0] == pytest.approx(1.0)  # 0 -> 2 -> 1


def test_update_respects_initial_value():
    agent = make(q_init=5.0, alpha=0.1)
    agent.update(2, 0.0)
    assert agent.Q[2] == pytest.approx(4.5)
    assert agent.Q[0] == 5.0


def test_greedy_picks_the_best_arm():
    agent = make(eps=0.0)
    agent.Q = np.array([0.1, 0.9, 0.3, 0.2, 0.0])
    assert {agent.select_action() for _ in range(200)} == {1}


def test_greedy_breaks_ties_at_random():
    agent = make(eps=0.0, k=5)
    picks = {agent.select_action() for _ in range(300)}
    assert len(picks) == 5, f"all estimates are equal, but only arms {sorted(picks)} were ever picked"


def test_epsilon_one_is_uniform():
    agent = make(eps=1.0, k=4)
    agent.Q = np.array([9.0, 0.0, 0.0, 0.0])
    counts = np.bincount([agent.select_action() for _ in range(4000)], minlength=4)
    assert all(800 < c < 1200 for c in counts), counts


def test_epsilon_greedy_exploration_rate():
    k, eps = 10, 0.1
    agent = make(eps=eps, k=k)
    agent.Q = np.zeros(k)
    agent.Q[3] = 1.0
    greedy_fraction = np.mean([agent.select_action() == 3 for _ in range(10000)])
    expected = 1 - eps + eps / k
    assert greedy_fraction == pytest.approx(expected, abs=0.02), f"{greedy_fraction:.3f} vs expected {expected:.3f}"


def test_ucb_tries_every_arm_first():
    agent = UCB(5, 2.0, np.random.default_rng(0))
    picked = []
    for _ in range(5):
        a = agent.select_action()
        picked.append(a)
        agent.update(a, 0.0)
    assert sorted(picked) == [0, 1, 2, 3, 4]


def test_ucb_bonus_beats_a_slightly_better_estimate():
    agent = UCB(2, 2.0, np.random.default_rng(0))
    agent.Q = np.array([1.0, 0.0])
    agent.N = np.array([100, 1])
    assert agent.select_action() == 1  # arm 1 is barely tried, so its bonus is large


def test_ucb_with_c_zero_is_greedy():
    agent = UCB(3, 0.0, np.random.default_rng(0))
    agent.Q = np.array([0.0, 2.0, 1.0])
    agent.N = np.array([5, 5, 5])
    assert agent.select_action() == 1


def test_ucb_bonus_shrinks_with_counts():
    agent = UCB(2, 1.0, np.random.default_rng(0))
    agent.Q = np.array([1.0, 0.8])
    agent.N = np.array([1000, 1000])
    assert agent.select_action() == 0  # equal counts: the higher estimate wins


def test_learning_beats_random_play():
    mk = lambda eps: (lambda k, rng: EpsilonGreedy(k, eps, rng))  # noqa: E731
    learner, _ = run(mk(0.1), n_runs=60, n_steps=600)
    random_play, _ = run(mk(1.0), n_runs=60, n_steps=600)
    assert learner[:, -200:].mean() > random_play[:, -200:].mean() + 0.6
