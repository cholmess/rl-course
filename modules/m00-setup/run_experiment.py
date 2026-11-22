"""Run each policy for 100 seeded episodes, print a summary and save results.png."""

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from policies import angle_policy, angle_velocity_policy, random_policy, run_episode

N_EPISODES = 100
POLICIES = {
    "random": random_policy,
    "angle only": angle_policy,
    "angle + velocity": angle_velocity_policy,
}


def evaluate(policy, n_episodes=N_EPISODES):
    env = gym.make("CartPole-v1")
    returns = np.array([run_episode(env, policy, seed) for seed in range(n_episodes)])
    env.close()
    return returns


def main():
    results = {name: evaluate(policy) for name, policy in POLICIES.items()}

    print(f"{'policy':<18}{'mean':>8}{'std':>8}{'min':>6}{'max':>6}{'% at 500':>10}")
    for name, r in results.items():
        print(f"{name:<18}{r.mean():8.1f}{r.std():8.1f}{r.min():6.0f}{r.max():6.0f}{(r == 500).mean() * 100:9.0f}%")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(list(results.values()), tick_labels=list(results.keys()), showmeans=True)
    ax.axhline(500, color="grey", linestyle="--", linewidth=1, label="max return (500)")
    ax.set_ylabel(f"episode return ({N_EPISODES} seeds)")
    ax.set_title("CartPole-v1: hand-written policies")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = Path(__file__).parent / "results.png"
    fig.savefig(out, dpi=120)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
