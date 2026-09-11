# 🚀 plugrl-env-client

[![CI](https://github.com/PlugRL/plugrl-env-client/actions/workflows/ci.yml/badge.svg)](https://github.com/PlugRL/plugrl-env-client/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**plugrl-env-client** runs Gymnasium environments and talks to a centralized PlugRL training server over WebSocket. It carries no deep learning dependencies, so an environment stack and a training stack never have to share a Python environment.

## ✨ Features

  * **No deep learning dependencies**: The policy stays on the server. This side needs only `gymnasium`, `websockets` and `msgpack`, so environments pinned to old `mujoco-py` or `cython<3` can be used with a modern training stack.
  * **Distributed communication**: `websockets` plus `msgpack` for asynchronous transfer between the server and any number of env clients, across machines.
  * **Modular design**: Separates the environment-side runtime (`plugrl-env-client`) from the shared protocol layer (`plugrl-protocol`).
  * **Command-line interface**: A `tyro`-powered CLI for starting and managing env clients.
  * **Gymnasium integration**: Works with standard `Gymnasium` environments.

-----

## 🛠️ Installation

The easiest way to get started is by cloning the repository and using `uv` to manage the environment.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/PlugRL/plugrl-env-client.git
    cd plugrl-env-client
    ```

2.  **Install dependencies:**

    **Option A: Use uv (recommended)**

    ```bash
    uv sync
    ```

    **Option B: Editable install with pip**

    ```bash
    pip install -e .
    ```

3. **Install Optional Environment Dependencies**

    Most people want `mujoco` and nothing else - it is the environment the
    quickstart uses, and the only one here that is dense-reward continuous
    control needing no assets, no display and no GPU:

    ```bash
    uv sync --extra mujoco
    ```

    Everything, with `uv`:

    ```bash
    uv sync --extra mujoco --extra robomimic --extra atari --extra classic --extra libero
    ```

    ```
    pip install -e ".[mujoco, robomimic, atari, classic, libero]"
    ```

    Note: `robomimic` and `libero` pull in `egl-probe`, whose legacy CMake build
    needs `CMAKE_POLICY_VERSION_MINIMUM=3.5` when using CMake 4+. `uv` is
    configured in this repository to apply that automatically. If you install
    with `pip`, set the variable manually, e.g.

    ```bash
    CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install -e ".[libero]"
    ```

## 🚀 Usage

The `plugrl-run-env-client` tool launches one or more env client processes for specific environments.

> Note: `plugrl-run-worker` is kept as a backwards-compatible alias.

### Basic Syntax

```bash
uv run plugrl-run-env-client <ENVIRONMENT_TYPE> [OPTIONS]
```

### Available Environments

| Type | Extra required | Description |
| :--- | :--- | :--- |
| **dummy-v1** | — | Dummy environment for protocol and connectivity tests |
| **mujoco-v1** | `mujoco` | Gymnasium MuJoCo control, default `HalfCheetah-v5` |
| **classic-v1** | `classic` | Classic control environments (e.g. CartPole) |
| **atari-v1** | `atari` | Atari games via ALE |
| **d4rl-v1** | `d4rl` | D4RL locomotion tasks |
| **robomimic-v1** | `robomimic` | RoboMimic robotic manipulation |
| **libero-v1** | `libero` | LIBERO manipulation benchmark |

An environment whose extra is not installed reports which extra it needs.

**`mujoco-v1` is the one to reach for first.** It is dense-reward continuous
control that needs no assets, no display and no GPU, and its default task
`HalfCheetah-v5` has a 17-dimensional observation and a 6-dimensional action
- exactly `plugrl-server`'s `fpo-policy` defaults, so the pair runs with no
configuration. It renders only with `--env.render`; a state-only policy never
looks at the frames, and producing them costs more per step than the physics
does.

```bash
uv sync --extra mujoco
uv run plugrl-run-env-client mujoco-v1 --num-envs 1 --num-episodes 600 \
    --runner.replan-steps 1 --runner.seed 0
```

### Examples

Run the `dummy-v1` environment with custom parameters:

```bash
# Run 100 episodes with 'debug' logging level
uv run plugrl-run-env-client dummy-v1 --num-episodes 100 --log-level debug

# Run with custom environment settings (64x64 image, 4-dim action space)
uv run plugrl-run-env-client dummy-v1 --env.img-width 64 --env.img-height 64 --env.action-dim 4
```

### Get More Help

To see all available options for a specific environment:

```bash
uv run plugrl-run-env-client dummy-v1 --help
```
