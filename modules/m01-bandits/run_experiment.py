"""Run the three bandit experiments, print summary tables and save the plots.

    python modules/m01-bandits/run_experiment.py            # 500 runs (about 2 minutes)
    python modules/m01-bandits/run_experiment.py --runs 100 # quick look while debugging
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from agents import UCB, EpsilonGreedy, GradientBandit
from bandits import run

N_STEPS = 1000
HERE = Path(__file__).parent


def summarize(name, rewards, optimal):
    last = slice(-100, None)
    print(f"{name:<34}{rewards.mean():9.3f}{rewards[:, last].mean():11.3f}{optimal[:, last].mean() * 100:11.1f}%")


def plot(results, title, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for name, (rewards, optimal) in results.items():
        ax1.plot(rewards.mean(axis=0), label=name, linewidth=1.2)
        ax2.plot(optimal.mean(axis=0) * 100, label=name, linewidth=1.2)
    ax1.set_xlabel("step")
    ax1.set_ylabel("average reward")
    ax2.set_xlabel("step")
    ax2.set_ylabel("% optimal action")
    ax2.set_ylim(0, 100)
    ax1.legend(loc="lower right", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(HERE / filename, dpi=120)
    plt.close(fig)
    print(f"Saved {HERE / filename}\n")


def experiment(title, configs, filename, n_runs, mean=0.0):
    print(title)
    print(f"{'agent':<34}{'avg reward':>9}{'last 100':>11}{'% optimal':>12}")
    results = {name: run(factory, n_runs, N_STEPS, mean=mean) for name, factory in configs.items()}
    for name, (r, o) in results.items():
        summarize(name, r, o)
    plot(results, title, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=500)
    n_runs = parser.parse_args().runs
    print(f"{n_runs} runs x {N_STEPS} steps per agent\n")

    experiment(
        "1. How much to explore? (epsilon-greedy)",
        {
            "greedy (eps=0)": lambda k, rng: EpsilonGreedy(k, 0.0, rng),
            "eps=0.01": lambda k, rng: EpsilonGreedy(k, 0.01, rng),
            "eps=0.1": lambda k, rng: EpsilonGreedy(k, 0.1, rng),
        },
        "results_epsilon.png",
        n_runs,
    )
    experiment(
        "2. Smarter exploration",
        {
            "eps-greedy eps=0.1": lambda k, rng: EpsilonGreedy(k, 0.1, rng),
            "UCB c=2": lambda k, rng: UCB(k, 2.0, rng),
            "optimistic greedy Q0=5, alpha=0.1": lambda k, rng: EpsilonGreedy(k, 0.0, rng, q_init=5.0, alpha=0.1),
            "gradient bandit alpha=0.1": lambda k, rng: GradientBandit(k, 0.1, rng),
        },
        "results_compare.png",
        n_runs,
    )
    experiment(
        "3. Break it: gradient bandit with and without a baseline (true values shifted by +4)",
        {
            "alpha=0.1 with baseline": lambda k, rng: GradientBandit(k, 0.1, rng, True),
            "alpha=0.1 no baseline": lambda k, rng: GradientBandit(k, 0.1, rng, False),
            "alpha=0.4 with baseline": lambda k, rng: GradientBandit(k, 0.4, rng, True),
            "alpha=0.4 no baseline": lambda k, rng: GradientBandit(k, 0.4, rng, False),
        },
        "results_baseline.png",
        n_runs,
        mean=4.0,
    )


if __name__ == "__main__":
    main()
