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
collection, and the policy-gradient update. Neural baselines and GAE are left for
Sections 4 and 5.

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

## Troubleshooting

* If you see an error about `swig` when installing `box2d-py`, you may need to install `swig` and `cmake` on your machine.
If you are using a Mac and have Homebrew installed, you can run `brew install swig cmake`.
On Modal, it should already be installed.
