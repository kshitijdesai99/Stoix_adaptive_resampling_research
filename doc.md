# How to Interpret the Training Logs

## `MISC`
General run information.

- **`Timestep`**
  - Total environment steps completed so far.

## `ACTOR`
Metrics from data collection in the training environment.

This tells you how the current policy behaves while generating rollouts for learning.

- **`Steps per second`**
  - Rollout collection speed.

- **`Episode length mean/std/min/max`**
  - How long training episodes are.

- **`Episode return mean/std/min/max`**
  - Reward obtained during rollout collection.

Use this section to answer:

- **Is the agent behaving better during training rollouts?**
- **Is data collection fast enough?**

## `TRAINER`
Metrics from the parameter update step.

This tells you what happens when the model is trained on collected rollouts.

- **`Loss policy`**
  - Policy optimization loss.

- **`Value loss`**
  - Critic/value prediction loss.

- **`Entropy depth:*`**
  - Action or search uncertainty at each depth.

- **`Ess fraction depth:*`**
  - Effective sample size quality across particles.

- **`Resample depth:*`**
  - Whether resampling happened at that depth.

- **`Particles alive depth:*`**
  - Fraction of particles still active at each depth.

- **`Adaptive temperature` / `Loss temperature`**
  - Temperature tuning diagnostics.

- **`Alpha` / `Loss alpha`**
  - KL penalty coefficient and its dual loss.

- **`Loss dual`**
  - Combined dual loss (`loss_alpha + loss_temperature`).

- **`Loss kl penalty`**
  - KL penalty term added to the total loss.

- **`Kl div`**
  - Parametric KL divergence between online and target policy.

- **`Kl nonparametric`**
  - Non-parametric KL estimate from normalized advantage weights.

- **`Kl nonparametric relative`**
  - `Kl nonparametric` divided by the epsilon constraint target.

- **`Mean td weights depth:*`**
  - Mean TD weights used for resampling decisions at each depth.

- **`Value pred mean` / `Value pred std`**
  - Mean and standard deviation of critic value predictions.

- **`Steps per second`**
  - Optimizer update steps per second during training.

Use this section to answer:

- **Is learning stable?**
- **Is search behaving as expected?**
- **Is resampling happening at the intended depths?**

## `EVALUATOR`
Metrics from periodic evaluation.

This is the clean performance check during training, using the current parameters.

- **`Episode length mean/std/min/max`**
  - Episode lengths during evaluation.

- **`Episode return mean/std/min/max`**
  - Evaluation performance.

- **`Steps per second`**
  - Evaluation speed.

Use this section to answer:

- **How good is the current checkpoint?**
- **Is performance improving over time?**

## `ABSOLUTE`
Metrics from the final post-training evaluation.

This is usually run after all training timesteps are finished.

In this codebase, it evaluates the best parameters found during training, not just the final update.

Use this section to answer:

- **What is the final reported score of the run?**
- **How well did the best checkpoint perform?**

## Recommended Reading Order

- **`MISC`**
  - Check where you are in training.

- **`ACTOR`**
  - Check training rollout behavior.

- **`TRAINER`**
  - Check whether optimization and search are healthy.

- **`EVALUATOR`**
  - Check actual validation performance.

- **`ABSOLUTE`**
  - Check the final best-model result.

## Quick Interpretation of Your Example

- **`ACTOR` near `0.948` return**
  - Training rollouts are already strong.

- **`EVALUATOR` near `0.949` return**
  - Evaluation agrees with training performance.

- **`ABSOLUTE` near `0.948` return**
  - Final best-checkpoint performance is about the same as periodic evaluation.

- **`Resample depth:2,4,6,8 = 1.000`**
  - Resampling is happening exactly on the configured periodic schedule.

- **`Particles alive` decreases with depth**
  - Expected as search goes deeper.
