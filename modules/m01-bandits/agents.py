"""Module 1: bandit agents.

You write two methods of EpsilonGreedy and one method of UCB.
GradientBandit is complete: read it, you'll need it for the write-up.
"""

import numpy as np


class EpsilonGreedy:
    """Keeps an estimate Q[a] of every arm's value and mostly picks the best one.

    - Q[a]: current estimate of arm a's value (starts at `q_init`).
    - N[a]: how many times arm a has been pulled.
    - epsilon: probability of exploring (picking an arm uniformly at random,
      which can by chance be the greedy arm too).
    - alpha: if None, use the sample average (step size 1/N[a]); otherwise use
      this constant step size.
    """

    def __init__(self, k: int, epsilon: float, rng: np.random.Generator, q_init: float = 0.0, alpha: float | None = None):
        self.k = k
        self.epsilon = epsilon
        self.rng = rng
        self.alpha = alpha
        self.Q = np.full(k, float(q_init))
        self.N = np.zeros(k, dtype=int)

    def select_action(self) -> int:
        """Pick an arm: explore with probability epsilon, otherwise the greedy arm.

        Break ties between equally good arms at random. If you always take the
        first one, an agent whose estimates are all equal would pull arm 0 forever.
        """
        # TODO
        raise NotImplementedError

    def update(self, action: int, reward: float) -> None:
        """Update N and Q after pulling `action` and receiving `reward`.

        Use the incremental form  Q <- Q + step * (reward - Q),  with
        step = 1 / N[action] (after incrementing N) or `self.alpha` if it's set.
        Don't store the list of past rewards.
        """
        # TODO
        raise NotImplementedError


class UCB(EpsilonGreedy):
    """Upper-confidence-bound selection: pick the arm with the best optimistic estimate

        Q[a] + c * sqrt(ln(t) / N[a])

    where t is the number of the step being chosen (1 on the first step). The update
    is inherited from EpsilonGreedy, so yours must be correct first.
    """

    def __init__(self, k: int, c: float, rng: np.random.Generator, q_init: float = 0.0, alpha: float | None = None):
        super().__init__(k, 0.0, rng, q_init, alpha)
        self.c = c

    def select_action(self) -> int:
        """Pick argmax of Q[a] + c * sqrt(ln(t) / N[a]).

        An arm with N[a] == 0 is treated as maximizing: pull untried arms first
        (ties between them at random). t is `self.N.sum() + 1`.
        """
        # TODO
        raise NotImplementedError


class GradientBandit:
    """Learns preferences H[a] instead of values; picks arms with probability softmax(H).

        pi(a) = exp(H[a]) / sum_b exp(H[b])
        H[a] <- H[a] + alpha * (reward - baseline) * (1{a taken} - pi(a))   for every arm a

    The baseline is the average of all rewards so far (or 0 if use_baseline=False).
    Provided complete: read it and match it to section 2.8 of the book.
    """

    def __init__(self, k: int, alpha: float, rng: np.random.Generator, use_baseline: bool = True):
        self.k = k
        self.alpha = alpha
        self.rng = rng
        self.use_baseline = use_baseline
        self.H = np.zeros(k)
        self.avg_reward = 0.0
        self.t = 0

    def policy(self) -> np.ndarray:
        z = np.exp(self.H - self.H.max())  # subtracting the max avoids overflow
        return z / z.sum()

    def select_action(self) -> int:
        return int(self.rng.choice(self.k, p=self.policy()))

    def update(self, action: int, reward: float) -> None:
        self.t += 1
        self.avg_reward += (reward - self.avg_reward) / self.t
        baseline = self.avg_reward if self.use_baseline else 0.0
        pi = self.policy()
        one_hot = np.zeros(self.k)
        one_hot[action] = 1.0
        self.H += self.alpha * (reward - baseline) * (one_hot - pi)
