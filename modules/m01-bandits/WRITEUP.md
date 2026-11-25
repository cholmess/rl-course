# Module 1 write-up

## Predictions (fill in before writing or running any code)

Average reward over the **last 100 steps** (the best possible average is about 1.5). Experiment 1 runs 3,000 steps, experiments 2 and 3 run 1,000:

| Agent | My prediction | Why |
|---|---|---|
| greedy (ε = 0) | | |
| ε = 0.01 | | |
| ε = 0.1 | | |
| UCB (c = 2) | | |
| Which of ε = 0.01 and ε = 0.1 is ahead at step 100? At step 1,000? At step 3,000? | | |

## Results

Paste the three summary tables from `run_experiment.py` and include the three plots.

![epsilon](results_epsilon.png)
![compare](results_compare.png)
![baseline](results_baseline.png)

## Questions

1. How close were your predictions? Where were you furthest off, and what did you misjudge?
2. **The greedy agent.** Its % optimal action levels off far below 100%. Why? What does it get stuck on, and why can't it recover? (Check: how does that change if you remove the random tie-breaking from your code?)
3. **ε = 0.1 vs ε = 0.01.** Which is better early and which later? Roughly at which step do the curves cross, and why do they cross? Also: even if the agent knew every arm's true value, ε = 0.1 could never exceed a certain % optimal action. Derive that number for k = 10, and say how close the curve gets.
4. **UCB.** Look at its first ~20 steps in `results_compare.png`. Describe the shape and explain what causes it.
5. **Optimistic initial values.** Why does an initial estimate of +5 make a purely greedy agent explore? Why does the effect fade after the first stretch of steps, and why would this trick be useless if the true values changed over time?
6. **Break it: the baseline.** In `results_baseline.png`, what happens to the gradient bandit without a baseline, and why does shifting all rewards up by 4 matter? Why does the version with a baseline not care about the shift?
7. **Your own problem.** Name a decision at your company where actions don't change the next situation and rewards arrive quickly (for example choosing which message, ranking or offer to show). What would the arms and the reward be, and what is the cost of exploring with real customers? No confidential details needed.
