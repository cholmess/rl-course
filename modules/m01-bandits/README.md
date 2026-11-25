# Module 1: Bandits and exploration

**Week 2 · about 7–8 hours**

A bandit is RL with the hard parts removed: your actions don't change what situation you're in next, you only choose which arm to pull and see a reward. What's left is the central tension of the whole field: **explore** to learn which arm is good, or **exploit** what you already believe.

## 1. Read (about 1 to 2 hours)

The quickest way is the **reading guide on the course page** ("Readings", written to be read on a phone). It covers sections 2.1 to 2.9 in about 40 minutes. If you want more depth, the original is:

Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., free at [incompleteideas.net/book/RLbook2020.pdf](http://incompleteideas.net/book/RLbook2020.pdf)), **chapter 2, sections 2.1 to 2.8**. Skim 2.9 (associative search, also called contextual bandits): you'll come back to it in phase 5 for your own work problem.

Skip the exercises, but work through the equations by hand, especially the incremental update in 2.4 and the gradient bandit in 2.8.

## 2. Quiz (about 1 hour)

Take the module 1 quiz on the course page. It has 9 questions on the reading and 4 review questions that repeat your open gaps from module 0.

## 3. Experiment: how should an agent explore? (about 3–4 hours)

### The setup

The **10-armed testbed**: 10 slot machines whose true average payouts are drawn from a normal distribution, with noisy rewards on top. An agent has 1,000 pulls to earn as much as it can. Results are averaged over hundreds of independent bandit problems, because one run says almost nothing.

Everything is provided (`bandits.py`, `run_experiment.py`, the gradient bandit) except the parts that make the three core ideas work:

| In `agents.py` | You write |
|---|---|
| `EpsilonGreedy.select_action` | explore with probability ε, otherwise pick the best arm. **Break ties at random.** |
| `EpsilonGreedy.update` | the incremental update `Q ← Q + step · (reward − Q)`, with step 1/N or a constant α |
| `UCB.select_action` | pick the arm with the highest `Q + c·sqrt(ln t / N)`, trying untried arms first |

### Steps

1. **Predict.** Fill in the predictions table in `WRITEUP.md` *before* you write or run any code.
2. **Implement** the three `TODO`s. Run `pytest modules/m01-bandits` as you go: the tests tell you which part is wrong.
3. **Run** `python modules/m01-bandits/run_experiment.py` (about 3 minutes; use `--runs 100` while debugging). It prints three tables and saves three plots:
   - `results_epsilon.png`: greedy vs ε = 0.01 vs ε = 0.1, over 3,000 steps so the long-run effect shows up.
   - `results_compare.png`: ε-greedy vs UCB vs optimistic initial values vs the gradient bandit.
   - `results_baseline.png`: **the break-it-on-purpose experiment**, the gradient bandit with and without a baseline when all true values are shifted up by 4.
4. **Write up** `WRITEUP.md` and commit the plots with it. Answer in English or Spanish.

### Hints (read only if stuck)

<details><summary>Hint 1: tie-breaking</summary>

`np.flatnonzero(self.Q == self.Q.max())` gives every arm that is tied for best. Pick one of them with `self.rng.choice`.
</details>

<details><summary>Hint 2: the incremental update</summary>

Increment `N[action]` first, then compute the step: `1 / N[action]` for a sample average. The new estimate is the old estimate plus the step times the error `(reward − old estimate)`.
</details>

<details><summary>Hint 3: UCB</summary>

If any arm has `N == 0`, choose among those arms. Otherwise compute the score for all arms at once with NumPy and pick the largest, again breaking ties at random.
</details>

## Grading

Quiz 40%, experiment 40% (tests pass, plots look right, code is clean), write-up 20% (predictions, explanations, the break-it analysis). You advance at A− (85+).
