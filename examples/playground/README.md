# MuJoCo Playground Example

This example registers a MuJoCo Playground environment into `plugrl-env-client`
the same way `examples/pusht` registers PushT: the example script performs the
registration and then hands off to the main CLI.

## Why a separate `.venv`

MuJoCo Playground depends on JAX and tends to be the first place where CUDA /
`jaxlib` conflicts show up. To avoid polluting the main project environment,
this example keeps its own `.venv` under `examples/playground/.venv`.

That environment installs:

- the local `plugrl-env-client` package in editable mode
- MuJoCo Playground
- a pinned JAX version (`0.4.38`, CUDA 12 on Linux)

It does not install the root project's dev dependencies.

## Setup

Run these commands from the repository root:

```bash
python -m venv examples/playground/.venv
examples/playground/.venv/bin/python -m pip install --upgrade pip
examples/playground/.venv/bin/python -m pip install -r examples/playground/requirements.txt
```

If you want to inspect the example environment directly:

```bash
examples/playground/.venv/bin/python examples/playground/smoke_test.py
```

## Run the example

```bash
examples/playground/.venv/bin/python examples/playground/playground_env.py playground-v1 \
  --env.name CheetahRun \
  --num-envs 8 \
  --num-episodes 1
```

`--num-envs` maps to MuJoCo Playground's batched JAX environment execution.
You can also combine it with `--num-procs` if you want multiple worker
processes, each with its own batched env shard.

## Notes

- On first load, MuJoCo Playground may download Menagerie assets for some tasks.
- This example defaults `MUJOCO_GL=egl` unless you explicitly override it
  before launch.
- On Linux, the example also prepends the wheel-shipped CUDA `ptxas` to `PATH`
  and sets `XLA_FLAGS=--xla_gpu_cuda_data_dir=...` before importing JAX.
- On Linux GPU machines, if JAX falls back to CPU, check
  `JAX_DEFAULT_MATMUL_PRECISION=highest` and your CUDA/JAX installation.
- The example exposes the MuJoCo Playground state tensors as `states/*` and
  always includes a rendered RGB frame under `images/env`.
