# 🚀 plugrl-env-client

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
    git clone git@github.com:PlugRL/plugrl-env-client.git
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

    With `uv`:

    ```bash
    uv sync --extra robomimic --extra atari --extra classic --extra libero
    ```

    ```
    pip install -e ".[robomimic, atari, classic, libero]"
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
| **classic-v1** | `classic` | Classic control environments (e.g. CartPole) |
| **atari-v1** | `atari` | Atari games via ALE |
| **d4rl-v1** | `d4rl` | D4RL locomotion tasks |
| **robomimic-v1** | `robomimic` | RoboMimic robotic manipulation |
| **libero-v1** | `libero` | LIBERO manipulation benchmark |

An environment whose extra is not installed reports which extra it needs.

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
