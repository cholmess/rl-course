"""The 10-armed testbed and a runner. Provided: you don't need to edit this file."""

import numpy as np


class Bandit:
    """A k-armed bandit. Arm a pays a reward drawn from Normal(q_star[a], 1).

    The true values q_star are drawn from Normal(mean, 1). With the default
    mean=0 this is the 10-armed testbed from Sutton & Barto, section 2.3.
    """

    def __init__(self, k: int, rng: np.random.Generator, mean: float = 0.0):
        self.k = k
        self.rng = rng
        self.q_star = rng.normal(mean, 1.0, size=k)
        self.optimal_action = int(np.argmax(self.q_star))

    def pull(self, action: int) -> float:
        return float(self.rng.normal(self.q_star[action], 1.0))


def run(agent_factory, n_runs: int, n_steps: int, seed: int = 0, k: int = 10, mean: float = 0.0):
    """Run an agent on `n_runs` independent bandit problems for `n_steps` steps each.

    `agent_factory(k, rng)` must return a fresh agent with `select_action()` and
    `update(action, reward)`. Run i uses the same bandit for every agent (same
    seed), so comparisons between agents are paired.

    Returns two arrays of shape (n_runs, n_steps): the rewards received, and
    whether the optimal arm was chosen (1.0 or 0.0).
    """
    rewards = np.zeros((n_runs, n_steps))
    optimal = np.zeros((n_runs, n_steps))
    for i in range(n_runs):
        rng = np.random.default_rng([seed, i])
        bandit = Bandit(k, rng, mean)
        agent = agent_factory(k, rng)
        for t in range(n_steps):
            a = agent.select_action()
            r = bandit.pull(a)
            agent.update(a, r)
            rewards[i, t] = r
            optimal[i, t] = a == bandit.optimal_action
    return rewards, optimal
