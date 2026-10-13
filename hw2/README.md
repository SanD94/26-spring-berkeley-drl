# Homework 2: Policy Gradients

## Setup

For general setup and Modal instructions, see Homework 1's README.

## Examples

Here are some example commands. Run them in the `hw2` directory.

* To run on a local machine:
  ```bash
  uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole
  ```


* To run on Modal:
  ```bash
  uv run modal run src/scripts/modal_run.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole
  ```
  * Note that Modal is likely not necessary for this assignment.
In testing, training was much faster on a local laptop CPU than on Modal.
However, you may still use Modal if you wish.
  * You may request a different GPU type, CPU count, and memory size by changing variables in `src/scripts/modal_run.py`
  * Use `modal run --detach` to keep your job running in the background.

## Section 3: Vanilla policy gradients

Implemented full-trajectory discounted returns, discounted reward-to-go, optional
batch-wide advantage normalization, categorical/Gaussian policies, rollout
collection, and the policy-gradient update. The neural baseline is implemented
in Section 4 below, and GAE in Section 5.

The policy loss sums over timesteps and divides by the number of trajectories,
matching the PDF's Equations (8), (11), and (14). Advantage normalization still
uses all transitions in the batch, and the critic loss is still mean-squared
error per transition. Runs collected before this change used the earlier
transition-mean policy loss; their logged actor losses use a different scale.

Run the algorithm checks with:

```bash
uv run python -m unittest discover -s tests -v
```

The eight assignment configurations are below. All use 100 iterations and the
default seed 1, discount 1.0, learning rate 0.005, two 64-unit hidden layers,
400 evaluation steps per iteration, and no video logging. `--no_gpu` explicitly
selects the CPU recommended by the assignment.

```bash
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 --no_gpu --exp_name cartpole
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 --no_gpu -rtg --exp_name cartpole_rtg
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 --no_gpu -na --exp_name cartpole_na
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 --no_gpu -rtg -na --exp_name cartpole_rtg_na
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 --no_gpu --exp_name cartpole_lb
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 --no_gpu -rtg --exp_name cartpole_lb_rtg
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 --no_gpu -na --exp_name cartpole_lb_na
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 --no_gpu -rtg -na --exp_name cartpole_lb_rtg_na
```

For concurrent CPU runs, set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` to
avoid thread oversubscription. Runs log online to the `cs285_hw2` W&B project;
return metrics use `Train_EnvstepsSoFar` as their default x-axis. In W&B, compare
`Train_AverageReturn` in two panels: the four `cartpole` runs without `_lb`, and
the four `cartpole_lb` runs. Use environment steps rather than iteration number.
After each run finishes, its logs, flags, and checkpoint are saved under
`exp/CartPole-v0_<exp_name>_sd1_<timestamp>/`.

## Section 4: Neural network baseline

The value critic predicts one scalar per observation and is trained against the
Monte Carlo Q-values with mean-squared error. Policy advantages use the critic's
predictions before it is updated, without backpropagating through the critic.
Each policy update is followed by `baseline_gradient_steps` critic updates;
`Baseline Loss` logs the loss of the last critic step.

The Chapter 4 experiments use the assignment's settings, seed 1, two 64-unit
hidden layers, CPU execution, and no advantage normalization or video logging:

```bash
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg --discount 0.95 -lr 0.01 --no_gpu --exp_name cheetah
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --no_gpu --exp_name cheetah_baseline
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 1 --no_gpu --exp_name cheetah_baseline_bgs1
```

The third run isolates reduced baseline training by changing only the number of
critic steps from 5 to 1. Compare `Baseline Loss` and `Eval_AverageReturn` in W&B
against `Train_EnvstepsSoFar`. Logs and checkpoints are saved under
`exp/HalfCheetah-v4_<exp_name>_sd1_<timestamp>/`.

The five earlier HalfCheetah experiments were deleted and the three commands
above restarted using trajectory-averaged policy loss. The training harness
seeds the environment as well as NumPy and PyTorch.

## Section 5: Generalized advantage estimation

GAE uses the critic's pre-update predictions to compute TD residuals and sums
them backward with discount `gamma * gae_lambda`. Both bootstrapping and the
recursive advantage stop at each trajectory boundary, including the rollout
length limit. The critic still trains against Monte Carlo reward-to-go targets.
Lambda 0 uses one-step TD advantages; lambda 1 recovers discounted reward-to-go
minus the baseline. Optional advantage normalization is applied after GAE.

The five Chapter 5 experiments preserve all assignment hyperparameters, using
seed 1, CPU execution, and no advantage normalization or video logging:

```bash
for lambda in 0 0.95 0.98 0.99 1; do
  uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda "$lambda" --no_gpu --exp_name "lunar_lander_lambda$lambda"
done
```

Each run uses the default baseline learning rate 0.005 and 5 baseline gradient
steps. Compare all five `Eval_AverageReturn` curves against
`Train_EnvstepsSoFar` in the `cs285_hw2` W&B project. Logs and checkpoints are
saved under `exp/LunarLander-v2_lunar_lander_lambda<lambda>_sd1_<timestamp>/`.
Use `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` when running concurrently.

## Section 6: Hyperparameters and sample efficiency

Chapter 6 uses the existing policy-gradient implementation; no new algorithm is
required. The experiments run on the local NVIDIA RTX 4070 with PyTorch CUDA.
Do not pass `--no_gpu`; the training script selects GPU 0 automatically. MuJoCo
simulation remains on the CPU, while policy inference and optimization run on
the GPU. Run the commands below from `hw2`, with `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1` when running concurrently.

The assignment's exact default configuration is:

```bash
uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 --exp_name pendulum
```

The selected configuration uses reward-to-go, normalized advantages, discount
0.99, learning rate 0.01, and 1000-step training batches. It retains the default
two 64-unit hidden layers, seed 1, full 1000-step episode horizon, and no baseline
or GAE:

```bash
uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 38 -b 1000 -eb 1000 -rtg -na --discount 0.99 -lr 0.01 --exp_name pendulum_best
```

The additional tuning commands are:

```bash
uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 50 -b 1000 -eb 1000 -rtg -na --discount 0.99 -lr 0.005 --exp_name pendulum_rtg_na
uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 50 -b 1000 -eb 1000 -rtg -na --discount 0.99 -lr 0.01 --exp_name pendulum_rtg_na_lr01
uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 50 -b 1000 -eb 1000 -na --discount 0.99 -lr 0.005 --exp_name pendulum_na
uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 50 -b 1000 -eb 1000 -rtg -na --discount 0.99 -lr 0.001 --use_baseline --gae_lambda 0.95 --exp_name pendulum_gae
for lr in 0.005 0.02 0.05; do
  uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 70 -b 1000 -eb 1000 -rtg -na --discount 1.0 -lr "$lr" --exp_name "pendulum_gamma1_lr$lr"
done
```

All runs log to the `cs285_hw2` W&B project and save `log.csv`, `log.pkl`,
`flags.json`, and `agent.pt` under `exp/InvertedPendulum-v4_pendulum*_sd1_*/`.
The batch size is a minimum: complete episodes can overshoot it. Success must
therefore be checked against the actual `Train_EnvstepsSoFar`, not `iteration *
batch_size`. Evaluation interactions are not included in that training counter.
With `-eb 1000`, a return of 1000 can represent a single evaluation episode;
this assignment's best-run criterion is not a multi-seed robustness guarantee.

### Results and tuning discussion

The selected run reaches evaluation return **1000 at 39,775 training steps**
(iteration 36, zero-based). Both training and evaluation averages are **1000 at
40,775 steps** (iteration 37). Its saved checkpoint comes from the latter point.
The 38-iteration rerun exactly reproduces the original candidate's first 38 rows
of training steps, returns, and actor losses.

| Experiment (all seed 1) | Best eval return within 100K steps | First eval 1000 at training steps |
| --- | ---: | ---: |
| `pendulum` (assignment default) | 61.06 | Not reached within 100K |
| `pendulum_rtg_na` (gamma 0.99, lr 0.005) | 215.50 | Not reached |
| `pendulum_na` (no reward-to-go) | 556.00 | Not reached |
| `pendulum_rtg_na_lr01` (gamma 0.99, lr 0.01) | 1000.00 | 39,775 |
| `pendulum_gae` (gamma 0.99, lr 0.001, lambda 0.95) | 110.90 | Not reached |
| `pendulum_gamma1_lr0.005` | 747.00 | Not reached |
| `pendulum_gamma1_lr0.02` | 232.40 | Not reached |
| `pendulum_gamma1_lr0.05` | 235.80 | Not reached |
| `pendulum_best` (38-iteration repeat of lr 0.01) | 1000.00 | 39,775 |

The default completed all 100 iterations, collecting 503,374 training steps,
with final evaluation return 129.625 and no evaluation average of 1000. The
sample-efficiency target is per run, not the total search cost: all nine runs
together collected 973,705 training steps, excluding evaluation interactions.

Learning rate mattered most in the controlled gamma-0.99/reward-to-go comparison:
changing only 0.005 to 0.01 raised the best evaluation return from 215.5 to 1000.
However, this did **not** produce stable convergence: the 50-iteration lr-0.01
run subsequently collapsed to return 3. The 38-iteration stopping point was
selected after observing this sweep, rather than specified in advance. The
full tuning curves retain the collapse; the selected run preserves the solved
checkpoint rather than its later degraded state.

At lr 0.005, removing reward-to-go improved the observed best return from 215.5
to 556, so reward-to-go was not uniformly better. Raising gamma from 0.99 to 1
at that same learning rate gave a peak evaluation return of 747; this run also
had a training-only average of 1000 at 65,782 steps, but later collapsed.
Gamma-1 learning rates 0.02 and 0.05 were less effective. The baseline/GAE
candidate did not reach the target, but its learning rate also differs, so this
is not an isolated test of GAE's effect. Smaller batches allow more updates per
training-step budget; their individual effect versus the default is not isolated
because other settings changed as well. Network size was held fixed throughout.

An independent subagent loaded `pendulum_best/agent.pt` on GPU 0 and evaluated
the stochastic policy on ten full-horizon episodes with environment, NumPy, and
PyTorch seeds 123. Nine episodes returned 1000 and one returned 233: mean 923.3,
minimum 233, maximum 1000, across 9233 evaluation steps. This confirms a useful
saved policy but **not** a robust mean of 1000 or multi-training-seed reliability.

Online logs: [selected run](https://wandb.ai/machinegunner-safa/cs285_hw2/runs/11bh16x4),
[default](https://wandb.ai/machinegunner-safa/cs285_hw2/runs/gjeow83r), and
[longer lr-0.01 run showing collapse](https://wandb.ai/machinegunner-safa/cs285_hw2/runs/kl2oas8w).

Reproduce the raw learning curves and per-run summaries with:

```bash
uv run src/scripts/plot_pendulum.py exp/InvertedPendulum-v4_pendulum_sd1_* exp/InvertedPendulum-v4_pendulum_best_sd1_*
uv run src/scripts/plot_pendulum.py exp/InvertedPendulum-v4_pendulum*_sd1_* --output_dir exp/chapter6/tuning
```

Each command creates full-range and 100K-step PNG/PDF comparisons plus
`summary.json`. Training returns are measured before the policy update;
evaluation returns are measured after it. Both use actual training environment
steps on the x-axis, with no smoothing that could obscure the success point.

[Default versus selected: full learning curves](exp/chapter6/pendulum_full.png)
and [first 100K steps](exp/chapter6/pendulum_within_100k.png).
[All tuning runs, including unstable later behavior](exp/chapter6/tuning/pendulum_within_100k.png).

## Troubleshooting

* If you see an error about `swig` when installing `box2d-py`, you may need to install `swig` and `cmake` on your machine.
If you are using a Mac and have Homebrew installed, you can run `brew install swig cmake`.
On Modal, it should already be installed.
