# Progress log

| Module | Started | Quiz | Experiment | Write-up | Total | Grade | Notes |
|---|---|---|---|---|---|---|---|
| M0 Setup and diagnostic | 2026-11-22 | 60% (diagnostic) | 40/40 (code) | pending | – | – | Diagnostic 12.5/21. Review set 6/8 (2026-11-24). CartPole code pushed 2026-11-24: tests pass, 100/100 episodes at 500. WRITEUP.md still blank, so module not closed |

## Weak spots to review

From the M0 diagnostic and review set (2026-11-24). Use these for review questions in upcoming quizzes.

**Still open after the review set (put both in the M1 quiz):**
- E[X²] vs (E[X])²: answered both 0 for X = ±2. Averaging and squaring don't commute.
- NumPy axes: still thinks the named axis is kept (said (7,) for sum over axis=1 of shape (4, 7)).

**Fixed in the review set:** Bernoulli variance, conditional probability, discounted geometric series, one gradient descent step, PyTorch training-loop order.

Original diagnostic findings:

- **Probability (2.5/6):** variance of a Bernoulli (gave p² instead of p(1−p)); conditional probability (treated "at least one head" as independent); geometric series / discounted return (said infinite instead of 1/(1−γ)); E[f(X)] vs f(E[X]) needs a concrete example.
- **Gradient descent:** confused one update step with jumping to the minimum; thinks the gradient gives a shortest path to the optimum.
- **NumPy axes:** thinks `axis=0` is the axis kept rather than collapsed.
- **PyTorch loop (0/2):** thinks `backward()` updates weights and `zero_grad()` freezes them. Revisit properly in M7.
- **RL framing:** knows reward vs labels; missing exploration and delayed credit assignment.

Strong: linear algebra (4/4), calculus derivatives, basic NumPy indexing.
