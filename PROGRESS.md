# Progress log

| Module | Started | Quiz | Experiment | Write-up | Total | Grade | Notes |
|---|---|---|---|---|---|---|---|
| M0 Setup and diagnostic | 2026-11-22 | 60% (diagnostic) | pending | pending | – | – | Diagnostic 12.5/21. Review set assigned on the course page |

## Weak spots to review

From the M0 diagnostic (2026-11-24). Use these for review questions in upcoming quizzes.

- **Probability (2.5/6):** variance of a Bernoulli (gave p² instead of p(1−p)); conditional probability (treated "at least one head" as independent); geometric series / discounted return (said infinite instead of 1/(1−γ)); E[f(X)] vs f(E[X]) needs a concrete example.
- **Gradient descent:** confused one update step with jumping to the minimum; thinks the gradient gives a shortest path to the optimum.
- **NumPy axes:** thinks `axis=0` is the axis kept rather than collapsed.
- **PyTorch loop (0/2):** thinks `backward()` updates weights and `zero_grad()` freezes them. Revisit properly in M7.
- **RL framing:** knows reward vs labels; missing exploration and delayed credit assignment.

Strong: linear algebra (4/4), calculus derivatives, basic NumPy indexing.
