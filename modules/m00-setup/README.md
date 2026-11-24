# Module 0: Setup and diagnostic

**Week 1 · about 6–8 hours**

Week 1 has three parts. None of it counts toward advancing, but do it properly: it sets your starting point and gets you used to the module routine.

## 1. Set up your Mac (about 1 hour)

Follow [SETUP.md](../../SETUP.md) and run `python modules/m00-setup/check_env.py`.

## 2. Diagnostic quiz (about 1.5 hours)

Take the diagnostic on the course page. It covers probability, linear algebra, calculus, Python/NumPy and PyTorch. No notes and no searching: the point is to find what to refresh first, so guessing only hides gaps. Answering "don't know" is fine.

## 3. Experiment: can a simple rule balance a pole? (about 3–4 hours)

### Background

[CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) is the "hello world" of RL. A pole stands on a cart, and every step you push the cart left (action `0`) or right (action `1`). You get +1 reward for every step the pole stays up, up to a maximum of 500. The episode ends early if the pole tilts past about 12° or the cart leaves the track.

Each step you observe four numbers:

| Index | Meaning |
|---|---|
| 0 | cart position `x` |
| 1 | cart velocity `x_dot` |
| 2 | pole angle `theta` (radians, positive = leaning right) |
| 3 | pole angular velocity `theta_dot` |

A **policy** is a rule that maps an observation to an action. In this module you write policies by hand. From module 1 onward the agent learns them.

Read the [Gymnasium basic usage page](https://gymnasium.farama.org/introduction/basic_usage/) first (15 minutes). It explains `reset`, `step`, `terminated` and `truncated`, which you'll use in every module.

### Steps

1. **Predict.** Open `WRITEUP.md` and fill in the predictions section *before* you write or run any code.
2. **Implement.** In `policies.py`, fill in the two `TODO` functions:
   - `run_episode`: the standard agent-environment loop.
   - `angle_velocity_policy`: your own hand-written rule.
3. **Test.** `pytest modules/m00-setup` must pass.
4. **Run.** `python modules/m00-setup/run_experiment.py`. It runs each policy for 100 episodes with fixed seeds and saves `results.png`.
5. **Ablate.** Compare `angle_policy` (uses the angle only, already written for you) with your `angle_velocity_policy`. Explain the gap.
6. **Write up.** Finish `WRITEUP.md` and commit `results.png` with it.

## 4. Book chapter (about 30 minutes)

Write `book/c00-preparacion.md`: one page in your own words on what CartPole is, what a policy is, and why a hand-written rule doesn't scale. See [book/README.md](../../book/README.md).

### Hints (read only if stuck)

<details><summary>Hint 1: run_episode</summary>

Call `env.reset(seed=seed)` once. Then loop: pick an action, call `env.step(action)`, add the reward, and stop when `terminated or truncated` is true.
</details>

<details><summary>Hint 2: the policy</summary>

The pole's angle tells you where it *is*. Its angular velocity tells you where it's *going*. Try pushing in the direction of `theta + k * theta_dot` for some k, and experiment with k.
</details>
