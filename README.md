# RL Course

A personal 28-week reinforcement learning course, from bandits to fine-tuning language models with RLHF, DPO and GRPO, and then to RL on real company data.

The full roadmap, quizzes and grades live on the [course page](https://claude.ai/artifact/6MDKxVGYBNuCoDhpLwLiPf) (private to you). This repo holds the code: one folder per module, each with a starter file, tests and a write-up.

## How each module works

1. **Read** the assigned chapter or paper.
2. **Quiz** on the course page. You get feedback on every missed answer.
3. **Experiment** in `modules/<module>/`:
   - Write your predictions in `WRITEUP.md` *before* running anything.
   - Fill in the `TODO`s in the starter code.
   - Run the tests: `pytest modules/<module>`.
   - Run the experiment and look at the plots.
   - Finish `WRITEUP.md`: what happened, was your prediction right, and why.
4. **Write your book chapter** in `book/` (see [book/README.md](book/README.md)): one or two pages in your own words, from `book/TEMPLATE.md`. Not graded, but Claude gives you feedback.
5. **Submit**: commit and push, then tell Claude in the project thread that the module is ready for grading.

You move on at A- (85+). See [RUBRIC.md](RUBRIC.md). Your grades are logged in [PROGRESS.md](PROGRESS.md).

## Getting started

Follow [SETUP.md](SETUP.md) to set up your Mac, then start with [module 0](modules/m00-setup/README.md).

## Modules

| Phase | Module | Weeks |
|---|---|---|
| 1 · Setup | M0 Course setup and diagnostic | 1 |
| 2 · Foundations | M1 Bandits · M2 MDPs · M3 Dynamic programming · M4 Monte Carlo · M5 TD learning · M6 Function approximation | 2–7 |
| 3 · Deep RL | M7 PyTorch for RL · M8 DQN · M9 Policy gradients · M10 Actor-critic and GAE · M11 PPO | 8–15 |
| 4 · RL for LLMs | M12 Transformers · M13 SFT and reward models · M14 RLHF with PPO · M15 DPO and GRPO · M16 Capstone | 16–24 |
| 5 · RL at work | M17 Contextual bandits and off-policy evaluation · M18 Offline RL and your work problem | 25–28 |
